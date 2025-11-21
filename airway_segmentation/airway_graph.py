import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize, ball, binary_dilation
from scipy.ndimage import distance_transform_edt, label
from skan import Skeleton, summarize
import networkx as nx
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy import ndimage


class AirwayGraphAnalyzer:
    """
    Analyzes the 3D structure of airways:
    - Generates 3D skeleton
    - Builds topological graph
    - Calculates diameters and lengths along branches
    - Identifies and analyzes bifurcations
    - Analyzes and manages connected components of the skeleton
    """
    
    def __init__(self, airway_mask_path, spacing=None):
        """
        Args:
            airway_mask_path: Path to .nii.gz airway mask file
            spacing: Tuple (x,y,z) of spacing in mm. If None, reads from image
        """
        self.mask_path = airway_mask_path
        
        # Read image
        print(f"Loading mask from: {airway_mask_path}")
        self.sitk_image = sitk.ReadImage(airway_mask_path)
        self.mask = sitk.GetArrayFromImage(self.sitk_image)
        
        # Get spacing (z, y, x) -> convert to (x, y, z)
        self.spacing = spacing if spacing else self.sitk_image.GetSpacing()
        print(f"Spacing (x,y,z): {self.spacing} mm")
        print(f"Shape (z,y,x): {self.mask.shape}")
        print(f"Positive voxels: {np.sum(self.mask > 0)}")
        
        # Results
        self.skeleton = None
        self.skeleton_obj = None
        self.graph = None
        self.branch_data = None
        self.distance_transform = None
        self.connected_components = None
        self.component_stats = None
        
    def compute_skeleton(self):
        """Computes the 3D skeleton of the mask"""
        print("\n=== 3D Skeletonization ===")
        
        # Binarize the mask
        binary_mask = (self.mask > 0).astype(np.uint8)
        
        # Apply skeletonize (works for both 2D and 3D)
        print("Computing 3D skeleton (may take a few minutes)...")
        self.skeleton = skeletonize(binary_mask)
        
        self.skeleton_voxels = np.sum(self.skeleton > 0)  # SAVE AS ATTRIBUTE
        print(f"Skeleton computed: {self.skeleton_voxels} voxels")
        
        # Compute distance transform for diameters
        print("Computing distance transform for diameters...")
        self.distance_transform = distance_transform_edt(binary_mask, sampling=self.spacing)
        
        return self.skeleton
    
    def analyze_connected_components(self):
        """Analyzes connected components of the skeleton"""
        print("\n=== SKELETON CONNECTED COMPONENTS ANALYSIS ===")
        
        if self.skeleton is None:
            raise ValueError("Compute skeleton first with compute_skeleton()")
        
        # Label connected components (26-connectivity for 3D)
        labeled_array, num_features = label(self.skeleton, structure=np.ones((3,3,3)))
        
        print(f"Total number of connected components in skeleton: {num_features}")
        
        # Compute statistics for each component
        component_stats = []
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            voxel_count = np.sum(component_mask)
            
            # Compute volume in mm³
            volume_mm3 = voxel_count * self.spacing[0] * self.spacing[1] * self.spacing[2]
            
            # Find component coordinates
            coordinates = np.argwhere(component_mask)
            
            if len(coordinates) > 0:
                # Compute centroid
                centroid = np.mean(coordinates, axis=0)
                
                # Compute bounding box
                z_min, y_min, x_min = np.min(coordinates, axis=0)
                z_max, y_max, x_max = np.max(coordinates, axis=0)
                
                component_stats.append({
                    'component_id': i,
                    'voxel_count': voxel_count,
                    'volume_mm3': volume_mm3,
                    'centroid_z': centroid[0],
                    'centroid_y': centroid[1],
                    'centroid_x': centroid[2],
                    'bbox_min_z': z_min,
                    'bbox_min_y': y_min,
                    'bbox_min_x': x_min,
                    'bbox_max_z': z_max,
                    'bbox_max_y': y_max,
                    'bbox_max_x': x_max
                })
        
        self.connected_components = labeled_array
        self.component_stats = pd.DataFrame(component_stats)
        
        # Sort by size (largest to smallest)
        self.component_stats = self.component_stats.sort_values('voxel_count', ascending=False)
        
        print(f"\nConnected components statistics:")
        print(f"  Largest component: {self.component_stats['voxel_count'].iloc[0]} voxels "
              f"({self.component_stats['volume_mm3'].iloc[0]:.2f} mm³)")
        print(f"  Smallest component: {self.component_stats['voxel_count'].iloc[-1]} voxels "
              f"({self.component_stats['volume_mm3'].iloc[-1]:.2f} mm³)")
        print(f"  Average size: {self.component_stats['voxel_count'].mean():.1f} voxels "
              f"({self.component_stats['volume_mm3'].mean():.2f} mm³)")
        
        # Analyze size distribution
        large_components = self.component_stats[self.component_stats['voxel_count'] >= 10]
        small_components = self.component_stats[self.component_stats['voxel_count'] < 10]
        
        print(f"\nSize distribution:")
        print(f"  Components with ≥10 voxels: {len(large_components)}")
        print(f"  Components with <10 voxels: {len(small_components)}")
        
        if hasattr(self, 'skeleton_voxels') and self.skeleton_voxels > 0:
            print(f"  Percentage of voxels in main component: "
                  f"{self.component_stats['voxel_count'].iloc[0] / self.skeleton_voxels * 100:.1f}%")
        else:
            skeleton_voxels_current = np.sum(self.skeleton > 0)
            print(f"  Percentage of voxels in main component: "
                  f"{self.component_stats['voxel_count'].iloc[0] / skeleton_voxels_current * 100:.1f}%")
        
        return self.connected_components, self.component_stats
    
    def smart_component_management(self, max_reconnect_distance_mm=15.0, min_voxels_for_reconnect=5, 
                                 max_voxels_for_keep=100, remove_tiny_components=True):
        """
        Intelligent component management:
        - Attempts to reconnect nearby significant components
        - Removes components not worth reconnecting
        """
        print(f"\n=== INTELLIGENT COMPONENT MANAGEMENT ===")
        print(f"Max reconnection distance: {max_reconnect_distance_mm} mm")
        print(f"Min voxels for reconnection: {min_voxels_for_reconnect}")
        print(f"Max voxels to keep isolated: {max_voxels_for_keep}")
        
        if self.connected_components is None:
            raise ValueError("Analyze connected components first with analyze_connected_components()")
        
        if len(self.component_stats) <= 1:
            print("Only one component found, nothing to manage")
            return self.skeleton
        
        # Identify main component
        main_component_id = self.component_stats.iloc[0]['component_id']
        main_component_mask = (self.connected_components == main_component_id)
        main_component_coords = np.argwhere(main_component_mask)
        
        if len(main_component_coords) == 0:
            print("Main component is empty, cannot proceed")
            return self.skeleton
        
        components_reconnected = 0
        components_kept_isolated = 0
        components_removed = 0
        voxels_reconnected = 0
        voxels_kept = 0
        voxels_removed = 0
        
        print(f"\nAnalyzing isolated components:")
        
        # Create a copy of skeleton for modifications
        new_skeleton = self.skeleton.copy()
        
        for i in range(1, len(self.component_stats)):
            comp_row = self.component_stats.iloc[i]
            comp_id = comp_row['component_id']
            voxel_count = comp_row['voxel_count']
            centroid = np.array([comp_row['centroid_z'], comp_row['centroid_y'], comp_row['centroid_x']])
            
            # Find coordinates of isolated component
            comp_mask = (self.connected_components == comp_id)
            comp_coords = np.argwhere(comp_mask)
            
            if len(comp_coords) == 0:
                continue
            
            # Calculate minimum distance from main component
            min_distance = float('inf')
            best_comp_point = None
            best_main_point = None
            
            # Sample points for efficiency
            comp_sample = comp_coords[::max(1, len(comp_coords)//5)]
            main_sample = main_component_coords[::max(1, len(main_component_coords)//50)]
            
            for comp_point in comp_sample:
                for main_point in main_sample:
                    # Calculate Euclidean distance in mm
                    dist_vector = (comp_point - main_point) * np.array([self.spacing[2], self.spacing[1], self.spacing[0]])
                    distance = np.linalg.norm(dist_vector)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_comp_point = comp_point
                        best_main_point = main_point
            
            print(f"Component {comp_id}: {voxel_count} voxels, minimum distance: {min_distance:.2f} mm")
            
            # DECISION: Reconnect, Keep, or Remove?
            if (min_distance <= max_reconnect_distance_mm and 
                voxel_count >= min_voxels_for_reconnect):
                # RECONNECT: significant and nearby component
                print(f"  → Reconnected! (distance: {min_distance:.2f} mm)")
                self._create_bridge_to_skeleton(best_comp_point, best_main_point, new_skeleton)
                components_reconnected += 1
                voxels_reconnected += voxel_count
            
            elif voxel_count > max_voxels_for_keep:
                # KEEP: component too large to remove, even if distant
                print(f"  → Kept isolated (too large: {voxel_count} voxels)")
                components_kept_isolated += 1
                voxels_kept += voxel_count
            
            elif remove_tiny_components and voxel_count < min_voxels_for_reconnect:
                # REMOVE: component too small
                print(f"  → Removed (too small: {voxel_count} voxels)")
                new_skeleton[comp_mask] = 0
                components_removed += 1
                voxels_removed += voxel_count
            
            else:
                # KEEP: medium-sized but distant component
                print(f"  → Kept isolated (distance: {min_distance:.2f} mm)")
                components_kept_isolated += 1
                voxels_kept += voxel_count
        
        # Update skeleton
        self.skeleton = new_skeleton.astype(bool)
        
        print(f"\n=== COMPONENT MANAGEMENT RESULTS ===")
        print(f"Reconnected components: {components_reconnected}")
        print(f"Components kept isolated: {components_kept_isolated}")
        print(f"Removed components: {components_removed}")
        print(f"Reconnected voxels: {voxels_reconnected}")
        print(f"Voxels kept isolated: {voxels_kept}")
        print(f"Removed voxels: {voxels_removed}")
        
        if components_reconnected > 0:
            print(f"✓ Reconnected {components_reconnected} components to main structure")
        if components_kept_isolated > 0:
            print(f"✓ Kept {components_kept_isolated} components isolated (significant)")
        if components_removed > 0:
            print(f"✓ Removed {components_removed} components (noise)")
        
        return self.skeleton
    
    def _create_bridge_to_skeleton(self, point1, point2, skeleton_array):
        """Creates a bridge between two points in 3D space and adds it to skeleton"""
        # Convert to integer coordinates
        p1 = np.array(point1, dtype=int)
        p2 = np.array(point2, dtype=int)
        
        # Calculate number of intermediate points (at least 3 points for short bridges)
        distance_pixels = np.linalg.norm(p2 - p1)
        distance_mm = distance_pixels * np.mean(self.spacing)
        num_points = max(3, int(distance_mm / 2))  # Points every ~2mm
        
        # Generate points along the line
        for t in np.linspace(0, 1, num_points):
            intermediate_point = p1 + t * (p2 - p1)
            z, y, x = intermediate_point.astype(int)
            
            # Ensure coordinates are within bounds
            if (0 <= z < skeleton_array.shape[0] and 
                0 <= y < skeleton_array.shape[1] and 
                0 <= x < skeleton_array.shape[2]):
                skeleton_array[z, y, x] = 1
        
        # Add small sphere at endpoints for robustness
        for point in [p1, p2]:
            z, y, x = point[0], point[1], point[2]
            # Small 3x3x3 sphere
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if (0 <= nz < skeleton_array.shape[0] and 
                            0 <= ny < skeleton_array.shape[1] and 
                            0 <= nx < skeleton_array.shape[2]):
                            skeleton_array[nz, ny, nx] = 1
    
    def visualize_connected_components(self, save_path=None, max_components=10, min_voxels=5):
        """Visualizes connected components of skeleton with different colors - IMPROVED VERSION"""
        print("\n=== CONNECTED COMPONENTS VISUALIZATION ===")
        
        if self.connected_components is None:
            raise ValueError("Analyze connected components first with analyze_connected_components()")
        
        # Filter components by minimum size
        large_components = self.component_stats[self.component_stats['voxel_count'] >= min_voxels]
        components_to_plot = min(max_components, len(large_components))
        
        print(f"Visualizing {components_to_plot} components with at least {min_voxels} voxels")
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create colormap for components
        colors = plt.cm.tab20(np.linspace(0, 1, components_to_plot))
        
        total_plotted = 0
        for i in range(components_to_plot):
            comp_id = large_components.iloc[i]['component_id']
            voxel_count = large_components.iloc[i]['voxel_count']
            
            # Find component coordinates
            component_coords = np.argwhere(self.connected_components == comp_id)
            
            if len(component_coords) > 0:
                # For small components, show all points
                if voxel_count < 100:
                    subsample = 1
                    marker_size = 20
                else:
                    # For large components, subsampling
                    subsample = max(1, len(component_coords) // 500)
                    marker_size = 10
                
                component_coords_subsampled = component_coords[::subsample]
                
                ax.scatter(component_coords_subsampled[:, 2], 
                        component_coords_subsampled[:, 1], 
                        component_coords_subsampled[:, 0],
                        c=[colors[i]], marker='o', s=marker_size, alpha=0.8,
                        label=f'Comp {comp_id} ({voxel_count} voxels)')
                
                total_plotted += len(component_coords_subsampled)
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Connected Components of Skeleton\n{components_to_plot} components visualized, {total_plotted} total points')
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=8)
        
        # Set axis limits for zoom on interesting region
        if len(large_components) > 0:
            main_component_id = large_components.iloc[0]['component_id']
            main_component_coords = np.argwhere(self.connected_components == main_component_id)
            if len(main_component_coords) > 0:
                z_mean, y_mean, x_mean = np.mean(main_component_coords, axis=0)
                z_std, y_std, x_std = np.std(main_component_coords, axis=0)
                
                # Zoom on ±3 standard deviations from main component
                ax.set_xlim(x_mean - 3*x_std, x_mean + 3*x_std)
                ax.set_ylim(y_mean - 3*y_std, y_mean + 3*y_std)
                ax.set_zlim(z_mean - 3*z_std, z_mean + 3*z_std)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Component visualization saved: {save_path}")
        
        plt.show()
    
    def visualize_component_size_distribution(self, save_path=None):
        """Visualizes the size distribution of connected components"""
        if self.component_stats is None:
            raise ValueError("Analyze connected components first with analyze_connected_components()")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of component sizes
        axes[0].hist(self.component_stats['voxel_count'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Number of voxels per component')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Connected Components Size Distribution')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Rank-size plot (components ordered by size)
        sizes_sorted = np.sort(self.component_stats['voxel_count'])[::-1]
        ranks = np.arange(1, len(sizes_sorted) + 1)
        
        axes[1].loglog(ranks, sizes_sorted, 'bo-', alpha=0.7)
        axes[1].set_xlabel('Rank (size order)')
        axes[1].set_ylabel('Size (voxels)')
        axes[1].set_title('Rank-Size Distribution of Components')
        axes[1].grid(True, alpha=0.3)
        
        # Add annotations for first 3 components
        for i in range(min(3, len(sizes_sorted))):
            axes[1].annotate(f'#{i+1}', (ranks[i], sizes_sorted[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Size distribution saved: {save_path}")
        
        plt.show()
    
    def build_graph(self):
        """Builds graph from skeleton using skan"""
        print("\n=== Graph Construction ===")
        
        if self.skeleton is None:
            raise ValueError("Compute skeleton first with compute_skeleton()")
        
        # Create skan Skeleton object
        # spacing in order (z, y, x) for skan
        spacing_zyx = (self.spacing[2], self.spacing[1], self.spacing[0])
        self.skeleton_obj = Skeleton(self.skeleton, spacing=spacing_zyx)
        
        # Extract branch information
        self.branch_data = summarize(self.skeleton_obj)
        
        print(f"Number of identified branches: {len(self.branch_data)}")
        print(f"Number of junctions: {self.skeleton_obj.n_paths}")
        
        # Create NetworkX graph for topological analysis
        self.graph = self._create_networkx_graph()
        
        return self.graph
    
    def _create_networkx_graph(self):
        """Creates a NetworkX graph from skeleton"""
        G = nx.Graph()
        
        # Add nodes (endpoints and junction points)
        coordinates = self.skeleton_obj.coordinates
        for idx in range(len(coordinates)):
            pos = coordinates[idx]
            G.add_node(idx, pos=pos)
        
        # Add edges from branches
        for _, row in self.branch_data.iterrows():
            node1 = int(row['node-id-src'])
            node2 = int(row['node-id-dst'])
            length = row['branch-distance']
            G.add_edge(node1, node2, length=length, branch_type=row['branch-type'])
        
        return G
    
    def calculate_branch_lengths(self):
        """
        Calculates the actual length of each branch by measuring the path
        through the skeleton coordinates
        """
        print("\n=== Branch Length Calculation ===")
        
        if self.branch_data is None:
            raise ValueError("Build graph first with build_graph()")
        
        branch_lengths = []
        
        for idx, row in self.branch_data.iterrows():
            branch_idx = idx
            
            # Get branch coordinates
            try:
                coords_indices = self.skeleton_obj.path_coordinates(branch_idx)
                
                if len(coords_indices) < 2:
                    # Single point branch
                    length_mm = 0.0
                else:
                    # Calculate cumulative length along the path
                    length_mm = 0.0
                    for i in range(len(coords_indices) - 1):
                        p1 = coords_indices[i]
                        p2 = coords_indices[i + 1]
                        
                        # Calculate physical distance between consecutive points
                        # coords are in (z, y, x) format
                        dz = (p2[0] - p1[0]) * self.spacing[2]
                        dy = (p2[1] - p1[1]) * self.spacing[1]
                        dx = (p2[2] - p1[2]) * self.spacing[0]
                        
                        segment_length = np.sqrt(dx**2 + dy**2 + dz**2)
                        length_mm += segment_length
                
                branch_lengths.append({
                    'branch_id': branch_idx,
                    'length_mm': length_mm,
                    'num_points': len(coords_indices),
                    'skan_distance_mm': row['branch-distance']  # For comparison
                })
                
            except Exception as e:
                print(f"Warning: Error calculating length for branch {branch_idx}: {e}")
                branch_lengths.append({
                    'branch_id': branch_idx,
                    'length_mm': 0.0,
                    'num_points': 0,
                    'skan_distance_mm': row['branch-distance']
                })
        
        self.branch_lengths_df = pd.DataFrame(branch_lengths)
        
        print(f"\nBranch Length Statistics (mm):")
        print(f"  Total branches: {len(self.branch_lengths_df)}")
        print(f"  Mean length: {self.branch_lengths_df['length_mm'].mean():.2f}")
        print(f"  Median length: {self.branch_lengths_df['length_mm'].median():.2f}")
        print(f"  Min length: {self.branch_lengths_df['length_mm'].min():.2f}")
        print(f"  Max length: {self.branch_lengths_df['length_mm'].max():.2f}")
        print(f"  Total airway length: {self.branch_lengths_df['length_mm'].sum():.2f}")
        
        # Compare with skan distances
        length_diff = np.abs(self.branch_lengths_df['length_mm'] - self.branch_lengths_df['skan_distance_mm'])
        print(f"\nComparison with skan distances:")
        print(f"  Mean difference: {length_diff.mean():.2f} mm")
        print(f"  Max difference: {length_diff.max():.2f} mm")
        
        return self.branch_lengths_df
    
    def analyze_diameters(self):
        """Calculates diameters along each branch using distance transform"""
        print("\n=== Diameter Analysis ===")
        
        if self.branch_data is None:
            raise ValueError("Build graph first with build_graph()")
        
        diameters_list = []
        
        for idx, row in self.branch_data.iterrows():
            branch_idx = idx
            
            # Get branch coordinates
            coords_indices = self.skeleton_obj.path_coordinates(branch_idx)
            
            # Extract distance transform values along the branch
            distances = []
            for coord in coords_indices:
                z, y, x = coord
                if 0 <= z < self.distance_transform.shape[0] and \
                   0 <= y < self.distance_transform.shape[1] and \
                   0 <= x < self.distance_transform.shape[2]:
                    dist = self.distance_transform[z, y, x]
                    distances.append(dist * 2)  # diameter = 2 * radius
            
            if distances:
                diameter_mean = np.mean(distances)
                diameter_std = np.std(distances)
                diameter_min = np.min(distances)
                diameter_max = np.max(distances)
            else:
                diameter_mean = diameter_std = diameter_min = diameter_max = 0
            
            diameters_list.append({
                'branch_id': branch_idx,
                'diameter_mean_mm': diameter_mean,
                'diameter_std_mm': diameter_std,
                'diameter_min_mm': diameter_min,
                'diameter_max_mm': diameter_max
            })
        
        self.diameters_df = pd.DataFrame(diameters_list)
        
        print(f"\nDiameter Statistics (mm):")
        print(f"  Mean: {self.diameters_df['diameter_mean_mm'].mean():.2f}")
        print(f"  Min: {self.diameters_df['diameter_mean_mm'].min():.2f}")
        print(f"  Max: {self.diameters_df['diameter_mean_mm'].max():.2f}")
        
        return self.diameters_df
    
    def merge_branch_metrics(self):
        """
        Merges diameters and lengths into a single comprehensive branch dataframe
        """
        print("\n=== Merging Branch Metrics ===")
        
        if not hasattr(self, 'diameters_df'):
            raise ValueError("Calculate diameters first with analyze_diameters()")
        
        if not hasattr(self, 'branch_lengths_df'):
            raise ValueError("Calculate lengths first with calculate_branch_lengths()")
        
        # Merge on branch_id
        self.branch_metrics_df = pd.merge(
            self.diameters_df,
            self.branch_lengths_df[['branch_id', 'length_mm', 'num_points']],
            on='branch_id',
            how='inner'
        )
        
        # Calculate additional metrics
        self.branch_metrics_df['volume_mm3'] = (
            np.pi * (self.branch_metrics_df['diameter_mean_mm'] / 2) ** 2 * 
            self.branch_metrics_df['length_mm']
        )
        
        self.branch_metrics_df['surface_area_mm2'] = (
            np.pi * self.branch_metrics_df['diameter_mean_mm'] * 
            self.branch_metrics_df['length_mm']
        )
        
        print(f"\nMerged metrics for {len(self.branch_metrics_df)} branches")
        print(f"\nColumns: {list(self.branch_metrics_df.columns)}")
        
        return self.branch_metrics_df
    
    def identify_bifurcations(self):
        """Identifies bifurcations (nodes with degree >= 3)"""
        print("\n=== Bifurcation Identification ===")
        
        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")
        
        bifurcations = []
        
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            
            if degree >= 3:  # Bifurcation (or higher branching point)
                pos = self.graph.nodes[node]['pos']
                
                # Get connected branches
                connected_branches = list(self.graph.edges(node))
                
                bifurcations.append({
                    'node_id': node,
                    'position': pos,
                    'degree': degree,
                    'num_branches': len(connected_branches),
                    'z': pos[0],
                    'y': pos[1],
                    'x': pos[2]
                })
        
        self.bifurcations_df = pd.DataFrame(bifurcations)
        
        print(f"Identified bifurcations: {len(self.bifurcations_df)}")
        if len(self.bifurcations_df) > 0:
            print(f"  Average degree: {self.bifurcations_df['degree'].mean():.2f}")
            print(f"  Max degree: {self.bifurcations_df['degree'].max()}")
        
        return self.bifurcations_df
    
    def visualize_branches_3d(self, save_path=None, colormap='viridis', max_branches=1000, color_by='diameter'):
        """
        Visualizes 3D branches/edges colored by mean diameter or length
        
        Args:
            color_by: 'diameter' or 'length'
        """
        print("\n=== 3D Branches/Edges Visualization ===")
        
        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Calculate metrics first with merge_branch_metrics()")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Choose color metric
        if color_by == 'length':
            color_values = self.branch_metrics_df['length_mm']
            color_label = 'Length (mm)'
        else:
            color_values = self.branch_metrics_df['diameter_mean_mm']
            color_label = 'Mean Diameter (mm)'
        
        # Normalize for colormap
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = plt.colormaps[colormap]
        
        # Plot each branch as line
        plotted_count = 0
        for idx, row in self.branch_metrics_df.iterrows():
            if plotted_count >= max_branches:
                break
                
            branch_id = int(row['branch_id'])
            color_value = row[color_label.split()[0].lower() + '_mm' if color_by == 'length' else 'diameter_mean_mm']
            
            try:
                # Get branch coordinates
                coords = self.skeleton_obj.path_coordinates(branch_id)
                
                if len(coords) > 1:
                    # Extract x, y, z coordinates
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]
                    
                    # Color by selected metric
                    color = cmap(norm(color_value))
                    
                    # Plot branch as line
                    line = ax.plot(x_coords, y_coords, z_coords, 
                                 color=color, linewidth=2, alpha=0.8)
                    plotted_count += 1
                    
            except Exception as e:
                print(f"Warning: Error plotting branch {branch_id}: {e}")
                continue
        
        print(f"Branches plotted: {plotted_count}")
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(color_label, rotation=270, labelpad=20)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Bronchial Tree Branches - Colored by {color_label}\n{plotted_count} branches visualized')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Branch visualization saved: {save_path}")
        
        plt.show()
    
    def visualize_skeleton_3d(self, save_path=None, show_bifurcations=False):
        """Visualizes 3D skeleton with matplotlib"""
        print("\n=== 3D Visualization ===")
        
        if self.skeleton is None:
            raise ValueError("Compute skeleton first")
        
        # Find skeleton point coordinates
        skeleton_coords = np.argwhere(self.skeleton > 0)
        
        # Subsampling for performance (take 1 every N points)
        subsample = max(1, len(skeleton_coords) // 10000)
        skeleton_coords = skeleton_coords[::subsample]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot skeleton
        ax.scatter(skeleton_coords[:, 2], skeleton_coords[:, 1], skeleton_coords[:, 0],
                  c='blue', marker='.', s=1, alpha=0.6, label='Skeleton')
        
        # Plot bifurcations only if explicitly requested
        if show_bifurcations and hasattr(self, 'bifurcations_df') and len(self.bifurcations_df) > 0:
            bif = self.bifurcations_df
            ax.scatter(bif['x'], bif['y'], bif['z'],
                      c='red', marker='o', s=100, alpha=0.8, label='Bifurcations')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Skeleton of Bronchial Tree')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()
    
    def plot_diameter_distribution(self, save_path=None):
        """Plots diameter distribution"""
        if not hasattr(self, 'branch_metrics_df'):
            print("Calculate metrics first with merge_branch_metrics()")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Diameter histogram
        axes[0].hist(self.branch_metrics_df['diameter_mean_mm'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Mean diameter (mm)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Branch Diameters Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Diameter vs branch length
        axes[1].scatter(self.branch_metrics_df['length_mm'], 
                       self.branch_metrics_df['diameter_mean_mm'], 
                       alpha=0.6, s=20)
        axes[1].set_xlabel('Branch length (mm)')
        axes[1].set_ylabel('Mean diameter (mm)')
        axes[1].set_title('Diameter vs Branch Length')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.show()
    
    def plot_length_distribution(self, save_path=None):
        """Plots branch length distribution"""
        if not hasattr(self, 'branch_metrics_df'):
            print("Calculate metrics first with merge_branch_metrics()")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Length histogram
        axes[0].hist(self.branch_metrics_df['length_mm'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0].set_xlabel('Branch length (mm)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Branch Length Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Length vs Volume
        axes[1].scatter(self.branch_metrics_df['length_mm'], 
                       self.branch_metrics_df['volume_mm3'], 
                       alpha=0.6, s=20, c='green')
        axes[1].set_xlabel('Branch length (mm)')
        axes[1].set_ylabel('Branch volume (mm³)')
        axes[1].set_title('Length vs Volume')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.show()
    
    def get_branch_summary(self):
        """Returns a dataframe with summary of all branches"""
        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Calculate and merge metrics first")
        
        summary = self.branch_metrics_df.copy()
        print("\n=== BRANCH SUMMARY (first 10) ===")
        for i, (_, row) in enumerate(summary.iterrows()):
            if i >= 10:
                break
            print(f"Branch {row['branch_id']}: "
                  f"Length = {row['length_mm']:.2f} mm, "
                  f"Mean diameter = {row['diameter_mean_mm']:.2f} mm, "
                  f"Volume = {row['volume_mm3']:.2f} mm³")
        
        return summary
    
    def save_results(self, output_dir):
        """Saves all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save skeleton as NIfTI
        skeleton_sitk = sitk.GetImageFromArray(self.skeleton.astype(np.uint8))
        skeleton_sitk.CopyInformation(self.sitk_image)
        skeleton_path = os.path.join(output_dir, "skeleton.nii.gz")
        sitk.WriteImage(skeleton_sitk, skeleton_path)
        print(f"Skeleton saved: {skeleton_path}")
        
        # Save connected components as NIfTI
        if self.connected_components is not None:
            components_sitk = sitk.GetImageFromArray(self.connected_components.astype(np.int16))
            components_sitk.CopyInformation(self.sitk_image)
            components_path = os.path.join(output_dir, "skeleton_components.nii.gz")
            sitk.WriteImage(components_sitk, components_path)
            print(f"Connected components saved: {components_path}")
        
        # Save CSV with complete branch metrics (diameters + lengths)
        if hasattr(self, 'branch_metrics_df'):
            metrics_path = os.path.join(output_dir, "branch_metrics_complete.csv")
            self.branch_metrics_df.to_csv(metrics_path, index=False)
            print(f"Complete branch metrics saved: {metrics_path}")
        
        # Save CSV with bifurcations
        if hasattr(self, 'bifurcations_df'):
            bifurcations_path = os.path.join(output_dir, "bifurcations.csv")
            self.bifurcations_df.to_csv(bifurcations_path, index=False)
            print(f"Bifurcations saved: {bifurcations_path}")
        
        # Save CSV with component statistics
        if hasattr(self, 'component_stats'):
            components_path = os.path.join(output_dir, "component_statistics.csv")
            self.component_stats.to_csv(components_path, index=False)
            print(f"Component statistics saved: {components_path}")
        
        # Save summary
        summary_path = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=== BRONCHIAL TREE ANALYSIS ===\n\n")
            f.write(f"Input file: {self.mask_path}\n")
            f.write(f"Spacing (x,y,z): {self.spacing} mm\n")
            f.write(f"Shape (z,y,x): {self.mask.shape}\n\n")
            
            if self.skeleton is not None:
                f.write(f"Skeleton voxels: {np.sum(self.skeleton > 0)}\n\n")
            
            if hasattr(self, 'component_stats'):
                f.write("SKELETON CONNECTED COMPONENTS:\n")
                f.write(f"  Total number of components: {len(self.component_stats)}\n")
                f.write(f"  Largest component: {self.component_stats['voxel_count'].iloc[0]} voxels "
                       f"({self.component_stats['volume_mm3'].iloc[0]:.2f} mm³)\n")
                f.write(f"  Smallest component: {self.component_stats['voxel_count'].iloc[-1]} voxels "
                       f"({self.component_stats['volume_mm3'].iloc[-1]:.2f} mm³)\n")
                if hasattr(self, 'skeleton_voxels') and self.skeleton_voxels > 0:
                    f.write(f"  Percentage of voxels in main component: "
                           f"{self.component_stats['voxel_count'].iloc[0] / self.skeleton_voxels * 100:.1f}%\n\n")
                else:
                    skeleton_voxels_current = np.sum(self.skeleton > 0)
                    f.write(f"  Percentage of voxels in main component: "
                           f"{self.component_stats['voxel_count'].iloc[0] / skeleton_voxels_current * 100:.1f}%\n\n")
            
            if hasattr(self, 'branch_metrics_df'):
                f.write("BRANCH METRICS:\n")
                f.write(f"  Number of branches: {len(self.branch_metrics_df)}\n")
                f.write(f"  Mean diameter: {self.branch_metrics_df['diameter_mean_mm'].mean():.2f} mm\n")
                f.write(f"  Min diameter: {self.branch_metrics_df['diameter_mean_mm'].min():.2f} mm\n")
                f.write(f"  Max diameter: {self.branch_metrics_df['diameter_mean_mm'].max():.2f} mm\n")
                f.write(f"  Mean length: {self.branch_metrics_df['length_mm'].mean():.2f} mm\n")
                f.write(f"  Min length: {self.branch_metrics_df['length_mm'].min():.2f} mm\n")
                f.write(f"  Max length: {self.branch_metrics_df['length_mm'].max():.2f} mm\n")
                f.write(f"  Total airway length: {self.branch_metrics_df['length_mm'].sum():.2f} mm\n")
                f.write(f"  Total airway volume: {self.branch_metrics_df['volume_mm3'].sum():.2f} mm³\n\n")
            
            if hasattr(self, 'bifurcations_df'):
                f.write("BIFURCATIONS:\n")
                f.write(f"  Number: {len(self.bifurcations_df)}\n")
                if len(self.bifurcations_df) > 0:
                    f.write(f"  Average degree: {self.bifurcations_df['degree'].mean():.2f}\n")
        
        print(f"Summary saved: {summary_path}")
    
    def run_full_analysis(self, output_dir, visualize=True, 
                         max_reconnect_distance_mm=15.0, 
                         min_voxels_for_reconnect=5,
                         max_voxels_for_keep=100):
        """
        Runs complete analysis pipeline with intelligent component management
        
        Args:
            output_dir: Output directory
            visualize: If True, generates visualizations
            max_reconnect_distance_mm: Maximum distance to reconnect components (mm)
            min_voxels_for_reconnect: Minimum voxels to consider a component for reconnection
            max_voxels_for_keep: Maximum voxels to keep a component isolated
        """
        print("\n" + "="*60)
        print("COMPLETE BRONCHIAL TREE ANALYSIS PIPELINE")
        print("="*60)
        print(f"Component management parameters:")
        print(f"  - Max reconnection distance: {max_reconnect_distance_mm} mm")
        print(f"  - Min voxels for reconnection: {min_voxels_for_reconnect}")
        print(f"  - Max voxels to keep isolated: {max_voxels_for_keep}")
        
        # 1. Skeleton
        self.compute_skeleton()
        
        # 2. Connected components analysis
        self.analyze_connected_components()
        
        # 3. INTELLIGENT COMPONENT MANAGEMENT
        self.smart_component_management(
            max_reconnect_distance_mm=max_reconnect_distance_mm,
            min_voxels_for_reconnect=min_voxels_for_reconnect,
            max_voxels_for_keep=max_voxels_for_keep,
            remove_tiny_components=True
        )
        
        # Reanalyze components after management
        print("\n=== COMPONENT ANALYSIS AFTER MANAGEMENT ===")
        self.analyze_connected_components()
        
        # 4. Graph
        self.build_graph()
        
        # 5. Lengths
        self.calculate_branch_lengths()
        
        # 6. Diameters
        self.analyze_diameters()
        
        # 7. Merge metrics
        self.merge_branch_metrics()
        
        # 8. Bifurcations
        self.identify_bifurcations()
        
        # 9. Save results
        self.save_results(output_dir)
        
        # 10. Visualizations
        if visualize:
            # Traditional visualization (points)
            self.visualize_skeleton_3d(
                save_path=os.path.join(output_dir, "skeleton_3d.png"),
                show_bifurcations=False
            )
            
            # Connected components visualization
            self.visualize_connected_components(
                save_path=os.path.join(output_dir, "connected_components.png")
            )
            
            # Component size distribution visualization
            self.visualize_component_size_distribution(
                save_path=os.path.join(output_dir, "component_size_distribution.png")
            )
            
            # Branches/edges visualization - colored by diameter
            self.visualize_branches_3d(
                save_path=os.path.join(output_dir, "branches_3d_diameter.png"),
                color_by='diameter'
            )
            
            # Branches/edges visualization - colored by length
            self.visualize_branches_3d(
                save_path=os.path.join(output_dir, "branches_3d_length.png"),
                color_by='length'
            )
            
            self.plot_diameter_distribution(
                save_path=os.path.join(output_dir, "diameter_distribution.png")
            )
            
            self.plot_length_distribution(
                save_path=os.path.join(output_dir, "length_distribution.png")
            )
        
        # 11. Summary
        self.get_branch_summary()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED!")
        print("="*60)
        
        return {
            'skeleton': self.skeleton,
            'connected_components': self.connected_components,
            'component_stats': self.component_stats,
            'graph': self.graph,
            'branch_metrics': self.branch_metrics_df,
            'bifurcations': self.bifurcations_df
        }