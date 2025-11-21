import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import networkx as nx


class GraphBasedCarinaRemoval:
    """
    Removes trachea by detecting carina on the topological graph.
    The carina is identified as a major bifurcation point in the upper part
    of the airway tree, where the graph splits into two main branches.
    """
    
    def __init__(self, analyzer):
        """
        Args:
            analyzer: AirwayGraphAnalyzer object with completed graph analysis
        """
        self.analyzer = analyzer
        self.graph = analyzer.graph
        self.skeleton = analyzer.skeleton
        self.sitk_image = analyzer.sitk_image
        self.spacing = analyzer.spacing
        
        # Results
        self.carina_node = None
        self.carina_location = None
        self.trachea_branches = []
        self.bronchi_branches = []
        self.cropped_skeleton = None
        self.cropped_mask = None
        
    def detect_carina_from_graph(self, visualize=True):
        """
        Detects carina as the first major bifurcation in the superior part
        of the airway tree where it splits into two main branches (left and right bronchi)
        """
        print("\n" + "="*80)
        print("CARINA DETECTION FROM GRAPH")
        print("="*80)
        
        if self.graph is None or len(self.graph.nodes()) == 0:
            raise ValueError("Graph is empty or not built. Run graph analysis first.")
        
        # Get all nodes with their positions
        node_positions = []
        for node in self.graph.nodes():
            pos = self.graph.nodes[node]['pos']
            degree = self.graph.degree(node)
            node_positions.append({
                'node_id': node,
                'z': pos[0],
                'y': pos[1],
                'x': pos[2],
                'degree': degree
            })
        
        nodes_df = pd.DataFrame(node_positions)
        
        # Find bifurcations (degree >= 3)
        bifurcations = nodes_df[nodes_df['degree'] >= 3].copy()
        
        if len(bifurcations) == 0:
            print("WARNING: No bifurcations found in graph!")
            return None
        
        print(f"\nFound {len(bifurcations)} bifurcation nodes")
        
        # Sort bifurcations by z coordinate (superior = lower z index)
        bifurcations = bifurcations.sort_values('z')
        
        # Strategy: Find the first major bifurcation that splits into 2 main branches
        # This is likely the carina
        carina_candidates = []
        
        for idx, bif in bifurcations.head(10).iterrows():  # Check top 10 most superior bifurcations
            node_id = bif['node_id']
            
            # Get neighbors
            neighbors = list(self.graph.neighbors(node_id))
            
            if len(neighbors) >= 2:
                # Analyze the subtrees from this bifurcation
                # Remove this node temporarily to see how the tree splits
                G_temp = self.graph.copy()
                G_temp.remove_node(node_id)
                
                # Count connected components after removing this node
                components = list(nx.connected_components(G_temp))
                num_components = len(components)
                
                # If it splits into 2-3 major components, this is likely the carina
                if num_components >= 2:
                    component_sizes = [len(comp) for comp in components]
                    component_sizes_sorted = sorted(component_sizes, reverse=True)
                    
                    # Check if we have 2 large components (left and right bronchi)
                    if len(component_sizes_sorted) >= 2:
                        largest = component_sizes_sorted[0]
                        second_largest = component_sizes_sorted[1]
                        
                        # Both should be significant (at least 10% of largest)
                        if second_largest >= 0.1 * largest:
                            carina_candidates.append({
                                'node_id': node_id,
                                'z': bif['z'],
                                'y': bif['y'],
                                'x': bif['x'],
                                'degree': bif['degree'],
                                'num_components': num_components,
                                'largest_component': largest,
                                'second_largest': second_largest,
                                'balance_ratio': second_largest / largest
                            })
        
        if len(carina_candidates) == 0:
            print("WARNING: No clear carina bifurcation found. Using first bifurcation as approximation.")
            carina = bifurcations.iloc[0]
            self.carina_node = carina['node_id']
            self.carina_location = {
                'node_id': int(carina['node_id']),
                'z_slice': int(carina['z']),
                'y_slice': int(carina['y']),
                'x_slice': int(carina['x']),
                'z_mm': carina['z'] * self.spacing[2],
                'degree': int(carina['degree'])
            }
        else:
            # Select the best candidate: most superior with good balance
            candidates_df = pd.DataFrame(carina_candidates)
            
            # Prefer candidates with balance ratio close to 0.5-1.0 (symmetric split)
            candidates_df['score'] = candidates_df['balance_ratio'] - np.abs(candidates_df['balance_ratio'] - 0.7)
            candidates_df = candidates_df.sort_values(['z', 'score'], ascending=[True, False])
            
            best = candidates_df.iloc[0]
            
            self.carina_node = best['node_id']
            self.carina_location = {
                'node_id': int(best['node_id']),
                'z_slice': int(best['z']),
                'y_slice': int(best['y']),
                'x_slice': int(best['x']),
                'z_mm': best['z'] * self.spacing[2],
                'degree': int(best['degree']),
                'num_components': int(best['num_components']),
                'balance_ratio': float(best['balance_ratio'])
            }
            
            print(f"\n✓ Carina detected at node {self.carina_node}")
            print(f"  Position: z={self.carina_location['z_slice']}, "
                  f"y={self.carina_location['y_slice']}, "
                  f"x={self.carina_location['x_slice']}")
            print(f"  Physical: z={self.carina_location['z_mm']:.1f} mm")
            print(f"  Degree: {self.carina_location['degree']}")
            print(f"  Components after split: {self.carina_location['num_components']}")
            print(f"  Balance ratio: {self.carina_location['balance_ratio']:.2f}")
        
        if visualize:
            self._visualize_carina_detection()
        
        return self.carina_location
    
    def identify_trachea_branches(self):
        """
        Identifies which branches belong to the trachea (superior to carina)
        vs bronchi (inferior to carina)
        """
        print("\n" + "="*80)
        print("IDENTIFYING TRACHEA VS BRONCHI BRANCHES")
        print("="*80)
        
        if self.carina_node is None:
            raise ValueError("Detect carina first with detect_carina_from_graph()")
        
        # Find all branches/edges in the graph
        all_branches = []
        
        if hasattr(self.analyzer, 'branch_metrics_df'):
            branch_data = self.analyzer.branch_metrics_df
        elif hasattr(self.analyzer, 'branch_data'):
            branch_data = self.analyzer.branch_data
        else:
            raise ValueError("No branch data available")
        
        # For each branch, determine if it's superior (trachea) or inferior (bronchi) to carina
        carina_z = self.carina_location['z_slice']
        
        for idx, row in branch_data.iterrows():
            branch_id = row['branch_id']
            
            try:
                # Get branch coordinates
                coords = self.analyzer.skeleton_obj.path_coordinates(int(branch_id))
                
                if len(coords) > 0:
                    # Calculate mean z position of branch
                    mean_z = np.mean(coords[:, 0])
                    min_z = np.min(coords[:, 0])
                    max_z = np.max(coords[:, 0])
                    
                    # Classification strategy:
                    # - If mean_z < carina_z: mostly superior (trachea)
                    # - If mean_z > carina_z: mostly inferior (bronchi)
                    # - If branch crosses carina: check which side has more voxels
                    
                    if max_z < carina_z:
                        # Entirely superior to carina
                        branch_type = 'trachea'
                        self.trachea_branches.append(branch_id)
                    elif min_z > carina_z:
                        # Entirely inferior to carina
                        branch_type = 'bronchi'
                        self.bronchi_branches.append(branch_id)
                    else:
                        # Branch crosses carina - classify by mean position
                        if mean_z < carina_z:
                            branch_type = 'trachea'
                            self.trachea_branches.append(branch_id)
                        else:
                            branch_type = 'bronchi'
                            self.bronchi_branches.append(branch_id)
                    
                    all_branches.append({
                        'branch_id': branch_id,
                        'mean_z': mean_z,
                        'min_z': min_z,
                        'max_z': max_z,
                        'branch_type': branch_type
                    })
            
            except Exception as e:
                print(f"Warning: Could not process branch {branch_id}: {e}")
                continue
        
        self.branch_classification_df = pd.DataFrame(all_branches)
        
        print(f"\n✓ Branch classification completed:")
        print(f"  Trachea branches: {len(self.trachea_branches)}")
        print(f"  Bronchi branches: {len(self.bronchi_branches)}")
        print(f"  Total branches: {len(all_branches)}")
        
        return self.branch_classification_df
    
    def remove_trachea_from_skeleton(self):
        """
        Removes trachea branches from skeleton, keeping only bronchi
        """
        print("\n" + "="*80)
        print("REMOVING TRACHEA FROM SKELETON")
        print("="*80)
        
        if len(self.trachea_branches) == 0:
            print("No trachea branches identified. Keeping entire skeleton.")
            self.cropped_skeleton = self.skeleton.copy()
            return self.cropped_skeleton
        
        # Start with copy of skeleton
        self.cropped_skeleton = self.skeleton.copy()
        
        # Remove each trachea branch
        removed_voxels = 0
        for branch_id in self.trachea_branches:
            try:
                coords = self.analyzer.skeleton_obj.path_coordinates(int(branch_id))
                
                for coord in coords:
                    z, y, x = coord
                    if (0 <= z < self.cropped_skeleton.shape[0] and
                        0 <= y < self.cropped_skeleton.shape[1] and
                        0 <= x < self.cropped_skeleton.shape[2]):
                        if self.cropped_skeleton[z, y, x]:
                            self.cropped_skeleton[z, y, x] = 0
                            removed_voxels += 1
            
            except Exception as e:
                print(f"Warning: Could not remove branch {branch_id}: {e}")
                continue
        
        original_voxels = np.sum(self.skeleton > 0)
        remaining_voxels = np.sum(self.cropped_skeleton > 0)
        
        print(f"\n✓ Trachea removal completed:")
        print(f"  Original skeleton voxels: {original_voxels:,}")
        print(f"  Removed voxels: {removed_voxels:,}")
        print(f"  Remaining voxels: {remaining_voxels:,}")
        print(f"  Percentage removed: {removed_voxels/original_voxels*100:.1f}%")
        
        return self.cropped_skeleton
    
    def remove_trachea_from_mask(self):
        """
        Removes trachea region from original mask based on z-coordinate cutoff
        """
        print("\n" + "="*80)
        print("REMOVING TRACHEA FROM ORIGINAL MASK")
        print("="*80)
        
        if self.carina_location is None:
            raise ValueError("Detect carina first")
        
        # Use carina z-position as cutoff
        carina_z = self.carina_location['z_slice']
        margin = 5  # Keep a few slices above carina
        cutoff_z = max(0, carina_z - margin)
        
        # Get original mask
        original_mask = self.analyzer.mask
        
        # Create cropped mask
        self.cropped_mask = original_mask.copy()
        self.cropped_mask[:cutoff_z, :, :] = 0
        
        original_voxels = np.sum(original_mask > 0)
        remaining_voxels = np.sum(self.cropped_mask > 0)
        removed_voxels = original_voxels - remaining_voxels
        
        print(f"\n✓ Mask cropping completed:")
        print(f"  Cutoff at z-slice: {cutoff_z} (carina at {carina_z}, margin={margin})")
        print(f"  Original mask voxels: {original_voxels:,}")
        print(f"  Removed voxels: {removed_voxels:,}")
        print(f"  Remaining voxels: {remaining_voxels:,}")
        print(f"  Percentage removed: {removed_voxels/original_voxels*100:.1f}%")
        
        return self.cropped_mask
    
    def _visualize_carina_detection(self):
        """Visualizes carina detection on graph"""
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: Graph nodes colored by z-position
        ax1 = fig.add_subplot(131, projection='3d')
        
        node_coords = []
        node_colors = []
        for node in self.graph.nodes():
            pos = self.graph.nodes[node]['pos']
            node_coords.append(pos)
            node_colors.append(pos[0])  # Color by z
        
        node_coords = np.array(node_coords)
        
        scatter = ax1.scatter(node_coords[:, 2], node_coords[:, 1], node_coords[:, 0],
                            c=node_colors, cmap='viridis', s=20, alpha=0.6)
        
        # Highlight carina
        if self.carina_node is not None:
            carina_pos = self.graph.nodes[self.carina_node]['pos']
            ax1.scatter([carina_pos[2]], [carina_pos[1]], [carina_pos[0]],
                       c='red', s=300, marker='*', edgecolors='black', linewidths=2,
                       label='Carina', zorder=10)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Graph Nodes (colored by Z position)')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Z coordinate', shrink=0.8)
        
        # Plot 2: Bifurcations distribution
        ax2 = fig.add_subplot(132)
        
        node_positions = []
        for node in self.graph.nodes():
            pos = self.graph.nodes[node]['pos']
            degree = self.graph.degree(node)
            node_positions.append({'z': pos[0], 'degree': degree})
        
        nodes_df = pd.DataFrame(node_positions)
        bifurcations = nodes_df[nodes_df['degree'] >= 3]
        
        ax2.scatter(nodes_df['z'], nodes_df['degree'], alpha=0.5, s=30, label='All nodes')
        ax2.scatter(bifurcations['z'], bifurcations['degree'], 
                   c='orange', s=50, alpha=0.7, label='Bifurcations')
        
        if self.carina_location is not None:
            ax2.axvline(self.carina_location['z_slice'], color='red', 
                       linestyle='--', linewidth=2, label='Carina')
            ax2.scatter([self.carina_location['z_slice']], 
                       [self.carina_location['degree']],
                       c='red', s=300, marker='*', edgecolors='black', 
                       linewidths=2, zorder=10)
        
        ax2.set_xlabel('Z coordinate (superior → inferior)')
        ax2.set_ylabel('Node degree')
        ax2.set_title('Bifurcations along Z-axis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Axial slice at carina
        ax3 = fig.add_subplot(133)
        
        if self.carina_location is not None:
            z_slice = self.carina_location['z_slice']
            ax3.imshow(self.analyzer.mask[z_slice, :, :], cmap='gray')
            ax3.plot(self.carina_location['x_slice'], 
                    self.carina_location['y_slice'],
                    'r*', markersize=20, label='Carina')
            ax3.set_title(f'Axial slice at Carina (z={z_slice})')
            ax3.legend()
            ax3.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_trachea_removal(self, save_path=None):
        """Visualizes before/after trachea removal"""
        if self.cropped_skeleton is None:
            raise ValueError("Remove trachea first")
        
        fig = plt.figure(figsize=(18, 6))
        
        # Original skeleton
        ax1 = fig.add_subplot(131, projection='3d')
        original_coords = np.argwhere(self.skeleton > 0)
        subsample = max(1, len(original_coords) // 5000)
        original_coords = original_coords[::subsample]
        ax1.scatter(original_coords[:, 2], original_coords[:, 1], original_coords[:, 0],
                   c='blue', s=1, alpha=0.5)
        
        if self.carina_location is not None:
            ax1.scatter([self.carina_location['x_slice']], 
                       [self.carina_location['y_slice']], 
                       [self.carina_location['z_slice']],
                       c='red', s=200, marker='*', label='Carina')
        
        ax1.set_title(f'Original Skeleton\n({np.sum(self.skeleton > 0):,} voxels)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # Cropped skeleton (bronchi only)
        ax2 = fig.add_subplot(132, projection='3d')
        cropped_coords = np.argwhere(self.cropped_skeleton > 0)
        subsample = max(1, len(cropped_coords) // 5000)
        cropped_coords = cropped_coords[::subsample]
        ax2.scatter(cropped_coords[:, 2], cropped_coords[:, 1], cropped_coords[:, 0],
                   c='green', s=1, alpha=0.5)
        ax2.set_title(f'After Trachea Removal\n({np.sum(self.cropped_skeleton > 0):,} voxels)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Removed parts (trachea)
        ax3 = fig.add_subplot(133, projection='3d')
        removed = (self.skeleton > 0) & (self.cropped_skeleton == 0)
        removed_coords = np.argwhere(removed)
        if len(removed_coords) > 0:
            subsample = max(1, len(removed_coords) // 5000)
            removed_coords = removed_coords[::subsample]
            ax3.scatter(removed_coords[:, 2], removed_coords[:, 1], removed_coords[:, 0],
                       c='red', s=1, alpha=0.5)
        ax3.set_title(f'Removed (Trachea)\n({np.sum(removed):,} voxels)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir):
        """Saves cropped skeleton and mask"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cropped skeleton
        if self.cropped_skeleton is not None:
            skeleton_sitk = sitk.GetImageFromArray(self.cropped_skeleton.astype(np.uint8))
            skeleton_sitk.CopyInformation(self.sitk_image)
            skeleton_path = os.path.join(output_dir, "skeleton_without_trachea.nii.gz")
            sitk.WriteImage(skeleton_sitk, skeleton_path)
            print(f"Cropped skeleton saved: {skeleton_path}")
        
        # Save cropped mask
        if self.cropped_mask is not None:
            mask_sitk = sitk.GetImageFromArray(self.cropped_mask.astype(np.uint8))
            mask_sitk.CopyInformation(self.sitk_image)
            mask_path = os.path.join(output_dir, "mask_without_trachea.nii.gz")
            sitk.WriteImage(mask_sitk, mask_path)
            print(f"Cropped mask saved: {mask_path}")
        
        # Save branch classification
        if hasattr(self, 'branch_classification_df'):
            csv_path = os.path.join(output_dir, "branch_classification_trachea_vs_bronchi.csv")
            self.branch_classification_df.to_csv(csv_path, index=False)
            print(f"Branch classification saved: {csv_path}")
        
        # Save report
        report_path = os.path.join(output_dir, "carina_removal_report.txt")
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRAPH-BASED CARINA DETECTION AND TRACHEA REMOVAL REPORT\n")
            f.write("="*80 + "\n\n")
            
            if self.carina_location:
                f.write("CARINA LOCATION:\n")
                for key, value in self.carina_location.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            f.write("BRANCH CLASSIFICATION:\n")
            f.write(f"  Trachea branches: {len(self.trachea_branches)}\n")
            f.write(f"  Bronchi branches: {len(self.bronchi_branches)}\n")
            f.write(f"  Total branches: {len(self.trachea_branches) + len(self.bronchi_branches)}\n\n")
            
            if self.skeleton is not None and self.cropped_skeleton is not None:
                f.write("SKELETON VOXELS:\n")
                f.write(f"  Original: {np.sum(self.skeleton > 0):,}\n")
                f.write(f"  After removal: {np.sum(self.cropped_skeleton > 0):,}\n")
                removed = np.sum(self.skeleton > 0) - np.sum(self.cropped_skeleton > 0)
                f.write(f"  Removed: {removed:,} ({removed/np.sum(self.skeleton > 0)*100:.1f}%)\n")
        
        print(f"Report saved: {report_path}")
    
    def run_complete_removal(self, output_dir, visualize=True):
        """
        Complete pipeline for graph-based carina detection and trachea removal
        """
        print("\n" + "="*80)
        print("GRAPH-BASED TRACHEA REMOVAL PIPELINE")
        print("="*80)
        
        # Detect carina
        self.detect_carina_from_graph(visualize=visualize)
        
        # Classify branches
        self.identify_trachea_branches()
        
        # Remove trachea from skeleton
        self.remove_trachea_from_skeleton()
        
        # Remove trachea from mask
        self.remove_trachea_from_mask()
        
        # Visualize
        if visualize:
            self.visualize_trachea_removal()
        
        # Save results
        self.save_results(output_dir)
        
        print("\n" + "="*80)
        print("TRACHEA REMOVAL COMPLETED!")
        print("="*80)
        
        return self.cropped_skeleton, self.cropped_mask