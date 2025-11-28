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
from collections import deque


class AirwayGraphAnalyzer:
    """
    Analyzes the 3D structure of airways with Weibel generation analysis:
    - Generates 3D skeleton
    - Builds topological graph
    - Calculates diameters and lengths along branches
    - Identifies and analyzes bifurcations
    - Classifies branches by generation (Weibel model)
    - Analyzes diameter tapering across generations
    - Analyzes and manages connected components of the skeleton
    """
    
    def __init__(self, airway_mask_path, spacing=None):
        """
        Args:
            airway_mask_path: Path to .nii.gz airway mask file (bronchi only, no trachea)
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
        
        # Generation analysis
        self.carina_node = None
        self.generation_assignments = None
        self.weibel_analysis_df = None
        
    def compute_skeleton(self):
        """Computes the 3D skeleton of the mask"""
        print("\n=== 3D Skeletonization ===")
        
        # Binarize the mask
        binary_mask = (self.mask > 0).astype(np.uint8)
        
        # Apply skeletonize (works for both 2D and 3D)
        print("Computing 3D skeleton (may take a few minutes)...")
        self.skeleton = skeletonize(binary_mask)
        
        self.skeleton_voxels = np.sum(self.skeleton > 0)
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
        
        # Calculate number of intermediate points
        distance_pixels = np.linalg.norm(p2 - p1)
        distance_mm = distance_pixels * np.mean(self.spacing)
        num_points = max(3, int(distance_mm / 2))
        
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
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if (0 <= nz < skeleton_array.shape[0] and 
                            0 <= ny < skeleton_array.shape[1] and 
                            0 <= nx < skeleton_array.shape[2]):
                            skeleton_array[nz, ny, nx] = 1
    
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
    
    def identify_carina(self):
        """
        Identifica la carina come il nodo con il diametro medio più grande tra i punti di biforcazione
        Restituisce anche le coordinate della carina
        """
        print("\n=== CARINA IDENTIFICATION BY DIAMETER ===")
        
        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")
        
        # Trova nodi con grado >= 2 (potenziali biforcazioni)
        candidate_nodes = [node for node in self.graph.nodes() 
                        if self.graph.degree(node) >= 2]
        
        if len(candidate_nodes) == 0:
            print("WARNING: No bifurcation nodes found!")
            # Fallback: usa il nodo con grado più alto
            degrees = dict(self.graph.degree())
            self.carina_node = max(degrees, key=degrees.get)
            carina_diameter = 0
            carina_position = self.graph.nodes[self.carina_node]['pos']
        else:
            # Calcola diametro medio per ogni candidato
            node_info = []
            for node in candidate_nodes:
                pos = self.graph.nodes[node]['pos']
                z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
                
                # Ottieni diametro alla posizione del nodo
                if (0 <= z < self.distance_transform.shape[0] and
                    0 <= y < self.distance_transform.shape[1] and
                    0 <= x < self.distance_transform.shape[2]):
                    diameter = self.distance_transform[z, y, x] * 2
                else:
                    diameter = 0
                
                # Calcola diametro medio dei rami connessi
                connected_branches = []
                for neighbor in self.graph.neighbors(node):
                    edge = tuple(sorted([node, neighbor]))
                    if 'diameter' in self.graph.edges[edge]:
                        connected_branches.append(self.graph.edges[edge]['diameter'])
                
                avg_branch_diameter = np.mean(connected_branches) if connected_branches else diameter
                
                node_info.append({
                    'node': node,
                    'degree': self.graph.degree(node),
                    'diameter_at_node': diameter,
                    'avg_branch_diameter': avg_branch_diameter,
                    'z': z,
                    'y': y,
                    'x': x,
                    'position': pos
                })
            
            node_info.sort(key=lambda x: (-x['avg_branch_diameter'], -x['z']))
            
            self.carina_node = node_info[0]['node']
            carina_diameter = node_info[0]['avg_branch_diameter']
            carina_position = node_info[0]['position']
            
            print(f"Carina identified: Node {self.carina_node}")
            print(f"  Degree: {node_info[0]['degree']}")
            print(f"  Average branch diameter: {node_info[0]['avg_branch_diameter']:.2f} mm")
            print(f"  Diameter at node: {node_info[0]['diameter_at_node']:.2f} mm")
            print(f"  Position (z,y,x): ({node_info[0]['z']}, {node_info[0]['y']:.1f}, {node_info[0]['x']:.1f})")
            print(f"  Position (world coordinates): ({carina_position[2]:.1f}, {carina_position[1]:.1f}, {carina_position[0]:.1f})")
        
        # Salva informazioni dettagliate sulla carina
        self.carina_info = {
            'node_id': self.carina_node,
            'position': carina_position,
            'avg_branch_diameter': carina_diameter,
            'diameter_at_node': node_info[0]['diameter_at_node'] if len(candidate_nodes) > 0 else 0,
            'degree': node_info[0]['degree'] if len(candidate_nodes) > 0 else self.graph.degree(self.carina_node),
            'coordinates_voxel': (node_info[0]['z'], node_info[0]['y'], node_info[0]['x']) if len(candidate_nodes) > 0 else (int(carina_position[0]), int(carina_position[1]), int(carina_position[2])),
            'coordinates_world': (carina_position[2], carina_position[1], carina_position[0])  # x, y, z
        }

        return self.carina_node, carina_diameter, carina_position
    
    def visualize_with_carina(self, save_path=None, figsize=(16, 12)):
        """
        Visualizzazione del grafo con la carina evidenziata con un PUNTINO ROSSO
        """
        print("\n=== GRAPH VISUALIZATION WITH CARINA ===")
        
        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")
        
        if not hasattr(self, 'carina_node'):
            self.identify_carina()
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot tutti i nodi
        all_nodes = list(self.graph.nodes())
        node_positions = np.array([self.graph.nodes[node]['pos'] for node in all_nodes])
        
        # Colora i nodi per grado
        node_degrees = [self.graph.degree(node) for node in all_nodes]
        node_colors = plt.cm.viridis(np.array(node_degrees) / max(node_degrees))
        
        ax.scatter(node_positions[:, 2], node_positions[:, 1], node_positions[:, 0],
                c=node_colors, s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Plot archi
        for edge in self.graph.edges():
            pos1 = self.graph.nodes[edge[0]]['pos']
            pos2 = self.graph.nodes[edge[1]]['pos']
            ax.plot([pos1[2], pos2[2]], [pos1[1], pos2[1]], [pos1[0], pos2[0]],
                color='gray', linewidth=1, alpha=0.5)
        
        # Evidenzia la CARINA con un PUNTINO ROSSO (anziché stella)
        carina_pos = self.graph.nodes[self.carina_node]['pos']
        ax.scatter(carina_pos[2], carina_pos[1], carina_pos[0],
                c='red', s=200, marker='o', edgecolors='black', linewidths=2,
                label=f'Carina (Ø={self.carina_info["avg_branch_diameter"]:.1f}mm)')
        
        # Aggiungi etichetta alla carina con coordinate
        coord_text = f'CARINA\nX:{carina_pos[2]:.1f}\nY:{carina_pos[1]:.1f}\nZ:{carina_pos[0]:.1f}'
        ax.text(carina_pos[2], carina_pos[1], carina_pos[0] + 10, 
            coord_text, fontsize=10, fontweight='bold', color='red', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title('Bronchial Tree Graph with Carina Highlighted\n(Red dot = Carina, largest diameter bifurcation)')
        ax.legend()
        
        # Aggiungi colorbar per i gradi
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(vmin=min(node_degrees), vmax=max(node_degrees)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Node Degree', rotation=270, labelpad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Graph with carina saved: {save_path}")
        
        plt.show()
        return fig, ax
    
    def assign_generations_weibel(self):
        """
        Assigns Weibel generation numbers to each branch using breadth-first search
        from the carina (generation 0).
        
        Weibel model:
        - Generation 0: Main bronchi (starting from carina)
        - Generation 1+: Each bifurcation increases generation by 1
        - Typical human lungs have ~23 generations
        """
        print("\n=== WEIBEL GENERATION ASSIGNMENT ===")
        
        if self.carina_node is None:
            self.identify_carina_with_diameter()
        
        if self.graph is None:
            raise ValueError("Build graph first")
        
        # Initialize generation assignments
        node_generations = {self.carina_node: -1}
        branch_generations = {}
        
        # Breadth-first search from carina
        queue = deque([(self.carina_node, -1)])
        visited = {self.carina_node}
        
        while queue:
            current_node, current_gen = queue.popleft()
            
            # Visit all neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    # Child node is in next generation if current node is a bifurcation
                    if self.graph.degree(current_node) >= 3:
                        # Bifurcation point: increase generation
                        next_gen = current_gen + 1
                    else:
                        # Continuation: same generation
                        next_gen = current_gen
                    
                    node_generations[neighbor] = next_gen
                    visited.add(neighbor)
                    queue.append((neighbor, next_gen))
                    
                    # Assign generation to the branch (edge)
                    edge = tuple(sorted([current_node, neighbor]))
                    branch_generations[edge] = next_gen
        
        self.generation_assignments = {
            'nodes': node_generations,
            'branches': branch_generations
        }
        
        # Statistics
        max_generation = max(node_generations.values())
        generation_counts = {}
        for gen in node_generations.values():
            generation_counts[gen] = generation_counts.get(gen, 0) + 1
        
        print(f"Generation assignment complete:")
        print(f"  Maximum generation: {max_generation}")
        print(f"  Number of nodes per generation:")
        for gen in sorted(generation_counts.keys()):
            print(f"    Gen {gen}: {generation_counts[gen]} nodes")
        
        return self.generation_assignments
    
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
                    length_mm = 0.0
                else:
                    # Calculate cumulative length
                    length_mm = 0.0
                    for i in range(len(coords_indices) - 1):
                        p1 = coords_indices[i]
                        p2 = coords_indices[i + 1]
                        
                        dz = (p2[0] - p1[0]) * self.spacing[2]
                        dy = (p2[1] - p1[1]) * self.spacing[1]
                        dx = (p2[2] - p1[2]) * self.spacing[0]
                        
                        segment_length = np.sqrt(dx**2 + dy**2 + dz**2)
                        length_mm += segment_length
                
                branch_lengths.append({
                    'branch_id': branch_idx,
                    'length_mm': length_mm,
                    'num_points': len(coords_indices),
                    'skan_distance_mm': row['branch-distance']
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
        Merges diameters, lengths, and generation info into comprehensive branch dataframe
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
        
        # Add generation information
        if self.generation_assignments is not None:
            generations = []
            for idx, row in self.branch_data.iterrows():
                node_src = int(row['node-id-src'])
                node_dst = int(row['node-id-dst'])
                edge = tuple(sorted([node_src, node_dst]))
                
                # Get generation from branch assignments
                gen = self.generation_assignments['branches'].get(edge, np.nan)
                generations.append(gen)
            
            self.branch_metrics_df['generation'] = generations
        
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
    
    def analyze_weibel_tapering(self):
        """
        Analyzes diameter tapering across generations according to Weibel model.
        Computes statistics for each generation and analyzes tapering ratios.
        """
        print("\n" + "="*60)
        print("WEIBEL GENERATION TAPERING ANALYSIS")
        print("="*60)
        
        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Merge branch metrics first")
        
        if 'generation' not in self.branch_metrics_df.columns:
            raise ValueError("Assign generations first with assign_generations_weibel()")
        
        # Group by generation
        generation_stats = []
        
        for gen in sorted(self.branch_metrics_df['generation'].dropna().unique()):
            gen_data = self.branch_metrics_df[self.branch_metrics_df['generation'] == gen]
            
            if len(gen_data) == 0:
                continue
            
            stats = {
                'generation': int(gen),
                'n_branches': len(gen_data),
                'diameter_mean_mm': gen_data['diameter_mean_mm'].mean(),
                'diameter_std_mm': gen_data['diameter_mean_mm'].std(),
                'diameter_median_mm': gen_data['diameter_mean_mm'].median(),
                'diameter_min_mm': gen_data['diameter_mean_mm'].min(),
                'diameter_max_mm': gen_data['diameter_mean_mm'].max(),
                'length_mean_mm': gen_data['length_mm'].mean(),
                'length_std_mm': gen_data['length_mm'].std(),
                'length_median_mm': gen_data['length_mm'].median(),
                'total_volume_mm3': gen_data['volume_mm3'].sum(),
                'total_surface_area_mm2': gen_data['surface_area_mm2'].sum()
            }
            
            generation_stats.append(stats)
        
        self.weibel_analysis_df = pd.DataFrame(generation_stats)
        
        # Calculate tapering ratios (diameter_n / diameter_n+1)
        tapering_ratios = []
        for i in range(len(self.weibel_analysis_df) - 1):
            current_gen = self.weibel_analysis_df.iloc[i]
            next_gen = self.weibel_analysis_df.iloc[i + 1]
            ratio = next_gen['diameter_mean_mm'] / current_gen['diameter_mean_mm']
            tapering_ratios.append({
                'from_generation': int(current_gen['generation']),
                'to_generation': int(next_gen['generation']),
                'diameter_ratio': ratio
            })
        
        self.tapering_ratios_df = pd.DataFrame(tapering_ratios)
        
        # Print results
        print(f"\nGeneration statistics:")
        print(self.weibel_analysis_df.to_string(index=False))
        
        print(f"\n{'='*60}")
        print("TAPERING RATIOS (Weibel Model)")
        print(f"{'='*60}")
        print("\nDiameter reduction ratios between consecutive generations:")
        print(self.tapering_ratios_df.to_string(index=False))
        
        if len(self.tapering_ratios_df) > 0:
            mean_ratio = self.tapering_ratios_df['diameter_ratio'].mean()
            print(f"\nMean tapering ratio: {mean_ratio:.3f}")
            print(f"(Weibel's theoretical model predicts ~0.793 for symmetric branching)")
            
            # Compare with Weibel's model
            weibel_expected = 2**(-1/3)  # ~0.793 for symmetric dichotomous branching
            print(f"\nWeibel's theoretical ratio: {weibel_expected:.3f}")
            print(f"Observed mean ratio: {mean_ratio:.3f}")
            print(f"Difference: {abs(mean_ratio - weibel_expected):.3f}")
        
        return self.weibel_analysis_df, self.tapering_ratios_df
    
    def calculate_distances_from_carina(self):
        """
        Calculates cumulative PATH distances from the carina to each endpoint.
        Uses shortest path along the graph (not Euclidean distance).
        """
        print("\n=== DISTANCE FROM CARINA CALCULATION (PATH-BASED) ===")
        
        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")
        
        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Calculate branch metrics first with merge_branch_metrics()")
        
        if self.carina_node is None:
            self.identify_carina()
        
        print(f"Calculating path distances from carina (Node {self.carina_node})...")
        
        # Calculate shortest path distances from carina to all nodes
        try:
            distances_from_carina = nx.single_source_dijkstra_path_length(
                self.graph, 
                self.carina_node, 
                weight='length'
            )
        except nx.NetworkXError as e:
            print(f"Error calculating distances: {e}")
            print("Using connected component containing carina.")
            connected_nodes = nx.node_connected_component(self.graph, self.carina_node)
            subgraph = self.graph.subgraph(connected_nodes)
            distances_from_carina = nx.single_source_dijkstra_path_length(
                subgraph, 
                self.carina_node, 
                weight='length'
            )
        
        print(f"Calculated path distances for {len(distances_from_carina)} nodes")
        
        # For each branch, determine cumulative distance from carina
        branch_distances = []
        
        for idx, row in self.branch_data.iterrows():
            branch_id = idx
            node_src = int(row['node-id-src'])
            node_dst = int(row['node-id-dst'])
            branch_length = row['branch-distance']
            
            # Get path distances from carina to both endpoints
            dist_src = distances_from_carina.get(node_src, np.nan)
            dist_dst = distances_from_carina.get(node_dst, np.nan)
            
            # Proximal end is closer to carina
            if not np.isnan(dist_src) and not np.isnan(dist_dst):
                proximal_distance = min(dist_src, dist_dst)
                distal_distance = max(dist_src, dist_dst)
                proximal_node = node_src if dist_src < dist_dst else node_dst
                distal_node = node_dst if dist_src < dist_dst else node_src
            elif not np.isnan(dist_src):
                proximal_distance = dist_src
                distal_distance = dist_src + branch_length
                proximal_node = node_src
                distal_node = node_dst
            elif not np.isnan(dist_dst):
                proximal_distance = dist_dst
                distal_distance = dist_dst + branch_length
                proximal_node = node_dst
                distal_node = node_src
            else:
                proximal_distance = np.nan
                distal_distance = np.nan
                proximal_node = node_src
                distal_node = node_dst
            
            branch_distances.append({
                'branch_id': branch_id,
                'proximal_node': proximal_node,
                'distal_node': distal_node,
                'distance_from_carina_proximal_mm': proximal_distance,
                'distance_from_carina_distal_mm': distal_distance,
                'branch_length_mm': branch_length
            })
        
        self.branch_distances_from_carina_df = pd.DataFrame(branch_distances)
        
        # Merge with existing branch metrics
        self.branch_metrics_df = pd.merge(
            self.branch_metrics_df,
            self.branch_distances_from_carina_df[['branch_id', 
                                                'proximal_node', 
                                                'distal_node',
                                                'distance_from_carina_proximal_mm',
                                                'distance_from_carina_distal_mm']],
            on='branch_id',
            how='left'
        )
        
        print("\nPath distance from carina statistics:")
        valid_distances = self.branch_metrics_df['distance_from_carina_distal_mm'].dropna()
        if len(valid_distances) > 0:
            print(f"  Min distance (proximal branches): {valid_distances.min():.2f} mm")
            print(f"  Max distance (distal branches): {valid_distances.max():.2f} mm")
            print(f"  Mean distance: {valid_distances.mean():.2f} mm")
            print(f"  Median distance: {valid_distances.median():.2f} mm")
        
        # Find endpoints
        endpoint_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) == 1]
        endpoint_distances = [distances_from_carina.get(node, np.nan) for node in endpoint_nodes]
        endpoint_distances = [d for d in endpoint_distances if not np.isnan(d)]
        
        if endpoint_distances:
            print(f"\nEndpoint statistics ({len(endpoint_distances)} endpoints):")
            print(f"  Closest endpoint to carina: {min(endpoint_distances):.2f} mm")
            print(f"  Farthest endpoint from carina: {max(endpoint_distances):.2f} mm")
            print(f"  Mean endpoint distance: {np.mean(endpoint_distances):.2f} mm")
        
        return self.branch_metrics_df

    def identify_bifurcations(self):
        """Identifies bifurcations (nodes with degree >= 3)"""
        print("\n=== Bifurcation Identification ===")
        
        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")
        
        bifurcations = []
        
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            
            if degree >= 3:
                pos = self.graph.nodes[node]['pos']
                
                # Get generation if available
                generation = self.generation_assignments['nodes'].get(node, np.nan) if self.generation_assignments else np.nan
                
                # Get connected branches
                connected_branches = list(self.graph.edges(node))
                
                bifurcations.append({
                    'node_id': node,
                    'position': pos,
                    'degree': degree,
                    'num_branches': len(connected_branches),
                    'generation': generation,
                    'z': pos[0],
                    'y': pos[1],
                    'x': pos[2]
                })
        
        self.bifurcations_df = pd.DataFrame(bifurcations)
        
        print(f"Identified bifurcations: {len(self.bifurcations_df)}")
        if len(self.bifurcations_df) > 0:
            print(f"  Average degree: {self.bifurcations_df['degree'].mean():.2f}")
            print(f"  Max degree: {self.bifurcations_df['degree'].max()}")
            
            if 'generation' in self.bifurcations_df.columns:
                gen_counts = self.bifurcations_df['generation'].value_counts().sort_index()
                print(f"\nBifurcations per generation:")
                for gen, count in gen_counts.items():
                    if not np.isnan(gen):
                        print(f"  Generation {int(gen)}: {count} bifurcations")
        
        return self.bifurcations_df
    
    def visualize_graph_with_generations(self, save_path=None, max_generation=None, 
                                       show_node_labels=True, show_branch_labels=True,
                                       node_size=50, font_size=8, figsize=(16, 12)):
        """
        Visualizza il grafo 3D con etichette delle generazioni
        
        Args:
            save_path: Percorso per salvare l'immagine
            max_generation: Generazione massima da visualizzare
            show_node_labels: Mostra le etichette sui nodi
            show_branch_labels: Mostra le etichette sui rami
            node_size: Dimensione dei nodi
            font_size: Dimensione del font
        """
        print("\n=== 3D GRAPH WITH GENERATION LABELS ===")
        
        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")
        
        if self.generation_assignments is None:
            raise ValueError("Assign generations first with assign_generations_weibel()")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Filtra per generazione massima se specificata
        if max_generation is not None:
            nodes_to_plot = [node for node in self.graph.nodes() 
                           if self.generation_assignments['nodes'].get(node, np.nan) <= max_generation]
        else:
            nodes_to_plot = list(self.graph.nodes())
        
        # Prepara i colori per le generazioni
        generations = [self.generation_assignments['nodes'].get(node, -1) for node in nodes_to_plot]
        unique_gens = sorted(set(generations))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_gens)))
        gen_to_color = {gen: colors[i] for i, gen in enumerate(unique_gens)}
        
        # Plot dei nodi
        for node in nodes_to_plot:
            pos = self.graph.nodes[node]['pos']
            gen = self.generation_assignments['nodes'].get(node, -1)
            color = gen_to_color[gen]
            
            # Dimensione del nodo basata sul grado
            degree = self.graph.degree(node)
            size = node_size * (1 + degree * 0.2)  # Nodi con più connessioni sono più grandi
            
            ax.scatter(pos[2], pos[1], pos[0], 
                      c=[color], s=size, alpha=0.8, edgecolors='black', linewidths=0.5)
            
            # Etichetta del nodo (generazione)
            if show_node_labels:
                ax.text(pos[2], pos[1], pos[0], f'{gen}', 
                       fontsize=font_size, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
        
        # Plot degli archi (rami)
        for edge in self.graph.edges():
            node1, node2 = edge
            if node1 in nodes_to_plot and node2 in nodes_to_plot:
                pos1 = self.graph.nodes[node1]['pos']
                pos2 = self.graph.nodes[node2]['pos']
                
                # Colore basato sulla generazione media
                gen1 = self.generation_assignments['nodes'].get(node1, -1)
                gen2 = self.generation_assignments['nodes'].get(node2, -1)
                avg_gen = (gen1 + gen2) / 2
                
                # Trova il colore più vicino
                closest_gen = min(unique_gens, key=lambda x: abs(x - avg_gen))
                color = gen_to_color[closest_gen]
                
                ax.plot([pos1[2], pos2[2]], [pos1[1], pos2[1]], [pos1[0], pos2[0]],
                       color=color, linewidth=2, alpha=0.7)
                
                # Etichetta del ramo (generazione)
                if show_branch_labels:
                    mid_point = [(pos1[0] + pos2[0])/2, 
                                (pos1[1] + pos2[1])/2, 
                                (pos1[2] + pos2[2])/2]
                    branch_gen = self.generation_assignments['branches'].get(tuple(sorted(edge)), np.nan)
                    if not np.isnan(branch_gen):
                        ax.text(mid_point[2], mid_point[1], mid_point[0], f'G{branch_gen}',
                               fontsize=font_size-1, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.1", facecolor='yellow', alpha=0.7))
        
        # Legenda delle generazioni
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=gen_to_color[gen], markersize=8,
                                     label=f'Gen {gen}')
                          for gen in unique_gens]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), 
                 loc='upper left', fontsize=9, title="Generations")
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        
        title = f'Bronchial Tree - Graph with Generation Labels\n{len(nodes_to_plot)} nodes, {len(unique_gens)} generations'
        if max_generation is not None:
            title += f' (up to Gen {max_generation})'
        ax.set_title(title)
        
        # Migliora la visualizzazione 3D
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Graph with generations saved: {save_path}")
        
        plt.show()
        
        return fig, ax

    def visualize_weibel_generations_enhanced(self, save_path=None, max_generation=None, 
                                            show_labels=True, label_frequency=0.3):
        """
        Visualizzazione migliorata delle generazioni di Weibel con etichette
        
        Args:
            save_path: Percorso per salvare l'immagine
            max_generation: Generazione massima da visualizzare
            show_labels: Mostra le etichette delle generazioni
            label_frequency: Frequenza delle etichette (0-1, più alto = più etichette)
        """
        print("\n=== ENHANCED WEIBEL GENERATION VISUALIZATION ===")
        
        if 'generation' not in self.branch_metrics_df.columns:
            raise ValueError("Assign generations first with assign_generations_weibel()")
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Filtra per generazione massima se specificata
        if max_generation is not None:
            plot_data = self.branch_metrics_df[self.branch_metrics_df['generation'] <= max_generation]
        else:
            plot_data = self.branch_metrics_df
        
        # Usa colormap discreta per le generazioni
        max_gen = int(plot_data['generation'].max())
        colors = plt.cm.tab20(np.linspace(0, 1, max_gen + 1))
        
        plotted_count = 0
        labeled_branches = set()
        
        for idx, row in plot_data.iterrows():
            if pd.isna(row['generation']):
                continue
            
            branch_id = int(row['branch_id'])
            generation = int(row['generation'])
            
            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)
                
                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]
                    
                    color = colors[generation % len(colors)]
                    
                    # Plot del ramo
                    line = ax.plot(x_coords, y_coords, z_coords, 
                                  color=color, linewidth=2, alpha=0.8,
                                  label=f'Gen {generation}' if generation not in labeled_branches else "")
                    
                    if generation not in labeled_branches:
                        labeled_branches.add(generation)
                    
                    plotted_count += 1
                    
                    # Aggiungi etichetta della generazione (sul punto medio)
                    if show_labels and np.random.random() < label_frequency:
                        mid_idx = len(coords) // 2
                        if mid_idx < len(coords):
                            mid_point = coords[mid_idx]
                            ax.text(mid_point[2], mid_point[1], mid_point[0], 
                                   f'G{generation}', fontsize=8, ha='center', va='center',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
            except Exception as e:
                continue
        
        print(f"Branches plotted: {plotted_count}")
        
        # Crea legenda
        legend_elements = []
        for gen in sorted(labeled_branches):
            color = colors[gen % len(colors)]
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                            label=f'Generation {gen}'))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), 
                 loc='upper left', fontsize=9, title="Weibel Generations")
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        
        title = f'Bronchial Tree - Weibel Generations\n{plotted_count} branches, {len(labeled_branches)} generations'
        if max_generation is not None:
            title += f' (up to Gen {max_generation})'
        ax.set_title(title)
        
        # Migliora la visualizzazione
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Enhanced Weibel visualization saved: {save_path}")
        
        plt.show()
        
        return fig, ax

    def visualize_weibel_generations(self, save_path=None, max_generation=None):
        """
        Visualizes branches colored by Weibel generation
        """
        print("\n=== WEIBEL GENERATION VISUALIZATION ===")
        
        if 'generation' not in self.branch_metrics_df.columns:
            raise ValueError("Assign generations first with assign_generations_weibel()")
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Filter by max generation if specified
        if max_generation is not None:
            plot_data = self.branch_metrics_df[self.branch_metrics_df['generation'] <= max_generation]
        else:
            plot_data = self.branch_metrics_df
        
        # Use discrete colormap for generations
        max_gen = int(plot_data['generation'].max())
        colors = plt.cm.tab20(np.linspace(0, 1, max_gen + 1))
        
        plotted_count = 0
        for idx, row in plot_data.iterrows():
            if pd.isna(row['generation']):
                continue
            
            branch_id = int(row['branch_id'])
            generation = int(row['generation'])
            
            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)
                
                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]
                    
                    color = colors[generation % len(colors)]
                    
                    ax.plot(x_coords, y_coords, z_coords, 
                           color=color, linewidth=2, alpha=0.8)
                    plotted_count += 1
                    
            except Exception as e:
                continue
        
        print(f"Branches plotted: {plotted_count}")
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], color=colors[i % len(colors)], 
                                     linewidth=2, label=f'Gen {i}')
                          for i in range(max_gen + 1)]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), 
                 loc='upper left', fontsize=9)
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Bronchial Tree - Weibel Generations\n{plotted_count} branches, {max_gen+1} generations')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()
    
    def visualize_weibel_generations_with_carina(self, save_path=None, max_generation=None):
        """
        Visualizza le generazioni di Weibel con la carina evidenziata con PUNTINO
        """
        print("\n=== WEIBEL GENERATIONS WITH CARINA ===")
        
        if 'generation' not in self.branch_metrics_df.columns:
            raise ValueError("Assign generations first with assign_generations_weibel()")
        
        if not hasattr(self, 'carina_node'):
            self.identify_carina()
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Filtra per generazione massima se specificata
        if max_generation is not None:
            plot_data = self.branch_metrics_df[self.branch_metrics_df['generation'] <= max_generation]
        else:
            plot_data = self.branch_metrics_df
        
        # Usa colormap discreta per le generazioni
        max_gen = int(plot_data['generation'].max())
        colors = plt.cm.tab20(np.linspace(0, 1, max_gen + 1))
        
        plotted_count = 0
        for idx, row in plot_data.iterrows():
            if pd.isna(row['generation']):
                continue
            
            branch_id = int(row['branch_id'])
            generation = int(row['generation'])
            
            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)
                
                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]
                    
                    color = colors[generation % len(colors)]
                    
                    ax.plot(x_coords, y_coords, z_coords, 
                        color=color, linewidth=2, alpha=0.8)
                    plotted_count += 1
                    
            except Exception as e:
                continue
        
        # Evidenzia la CARINA con PUNTINO ROSSO
        carina_pos = self.graph.nodes[self.carina_node]['pos']
        ax.scatter(carina_pos[2], carina_pos[1], carina_pos[0],
                c='red', s=300, marker='o', edgecolors='black', linewidths=3,
                label=f'Carina (Ø={self.carina_info["avg_branch_diameter"]:.1f}mm)')
        
        # Aggiungi etichetta alla carina con coordinate
        coord_text = f'CARINA\nX:{carina_pos[2]:.1f}\nY:{carina_pos[1]:.1f}\nZ:{carina_pos[0]:.1f}'
        ax.text(carina_pos[2], carina_pos[1], carina_pos[0] + 15, 
            coord_text, fontsize=11, fontweight='bold', color='red', ha='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))
        
        print(f"Branches plotted: {plotted_count}")
        
        # Crea legenda
        legend_elements = [plt.Line2D([0], [0], color=colors[i % len(colors)], 
                                    linewidth=2, label=f'Gen {i}')
                        for i in range(max_gen + 1)]
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='red', 
                                        markersize=10, label='Carina', linewidth=0))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), 
                loc='upper left', fontsize=10)
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Weibel Generations with Carina\n{plotted_count} branches, Carina = Largest diameter bifurcation')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()
        return fig, ax

    def get_carina_coordinates(self):
        """
        Restituisce le coordinate della carina in diversi sistemi di riferimento
        """
        if not hasattr(self, 'carina_info'):
            self.identify_carina()
        
        carina_info = self.carina_info
        
        coordinates = {
            'node_id': carina_info['node_id'],
            'voxel_coordinates': {
                'z': carina_info['coordinates_voxel'][0],
                'y': carina_info['coordinates_voxel'][1], 
                'x': carina_info['coordinates_voxel'][2]
            },
            'world_coordinates': {
                'x': carina_info['coordinates_world'][0],
                'y': carina_info['coordinates_world'][1],
                'z': carina_info['coordinates_world'][2]
            },
            'physical_coordinates_mm': {
                'x': carina_info['coordinates_world'][0] * self.spacing[0],
                'y': carina_info['coordinates_world'][1] * self.spacing[1],
                'z': carina_info['coordinates_world'][2] * self.spacing[2]
            },
            'diameter_info': {
                'avg_branch_diameter_mm': carina_info['avg_branch_diameter'],
                'diameter_at_node_mm': carina_info['diameter_at_node'],
                'degree': carina_info['degree']
            }
        }
        
        print("\n=== CARINA COORDINATES ===")
        print(f"Node ID: {coordinates['node_id']}")
        print(f"Voxel coordinates (z,y,x): ({coordinates['voxel_coordinates']['z']}, {coordinates['voxel_coordinates']['y']}, {coordinates['voxel_coordinates']['x']})")
        print(f"World coordinates (x,y,z): ({coordinates['world_coordinates']['x']:.1f}, {coordinates['world_coordinates']['y']:.1f}, {coordinates['world_coordinates']['z']:.1f})")
        print(f"Physical coordinates (mm): ({coordinates['physical_coordinates_mm']['x']:.1f}, {coordinates['physical_coordinates_mm']['y']:.1f}, {coordinates['physical_coordinates_mm']['z']:.1f})")
        print(f"Diameter: {coordinates['diameter_info']['avg_branch_diameter_mm']:.2f} mm")
        print(f"Degree: {coordinates['diameter_info']['degree']}")
        
        return coordinates

    def visualize_diameter_analysis_with_carina(self, save_path=None):
        """
        Visualizza l'analisi dei diametri con la carina evidenziata
        """
        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Calculate metrics first")
        
        if not hasattr(self, 'carina_node'):
            self.identify_carina_with_diameter()
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        color_values = self.branch_metrics_df['diameter_mean_mm']
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = plt.cm.viridis
        
        plotted_count = 0
        for idx, row in self.branch_metrics_df.iterrows():
            branch_id = int(row['branch_id'])
            diameter = row['diameter_mean_mm']
            
            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)
                
                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]
                    
                    color = cmap(norm(diameter))
                    
                    ax.plot(x_coords, y_coords, z_coords, 
                        color=color, linewidth=3, alpha=0.8)
                    plotted_count += 1
                    
            except Exception as e:
                continue
        
        # Evidenzia la CARINA in ROSSO
        carina_pos = self.graph.nodes[self.carina_node]['pos']
        ax.scatter(carina_pos[2], carina_pos[1], carina_pos[0],
                c='red', s=500, marker='*', edgecolors='black', linewidths=3,
                label=f'Carina (Ø={self.carina_info["avg_branch_diameter"]:.1f}mm)')
        
        # Aggiungi etichetta alla carina
        ax.text(carina_pos[2], carina_pos[1], carina_pos[0] + 20, 
            f'CARINA\nØ={self.carina_info["avg_branch_diameter"]:.1f}mm', 
            fontsize=12, fontweight='bold', color='red', ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        print(f"Branches plotted: {plotted_count}")
        
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Mean Diameter (mm)', rotation=270, labelpad=20)
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Bronchial Tree - Diameter Analysis\nCarina highlighted in red (largest diameter)')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Diameter visualization with carina saved: {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_weibel_tapering_analysis(self, save_path=None):
        """
        Creates comprehensive visualization of diameter tapering across generations
        """
        if not hasattr(self, 'weibel_analysis_df'):
            raise ValueError("Run analyze_weibel_tapering() first")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Mean diameter by generation
        ax1 = axes[0, 0]
        generations = self.weibel_analysis_df['generation']
        diameters = self.weibel_analysis_df['diameter_mean_mm']
        errors = self.weibel_analysis_df['diameter_std_mm']
        
        ax1.errorbar(generations, diameters, yerr=errors, 
                    marker='o', markersize=8, capsize=5, linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Mean Diameter (mm)')
        ax1.set_title('Airway Diameter Tapering Across Generations\n(Weibel Model)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(generations)
        
        # Add Weibel's theoretical curve
        if len(generations) > 1:
            gen_0_diameter = diameters.iloc[0]
            theoretical_diameters = [gen_0_diameter * (2**(-1/3))**g for g in generations]
            ax1.plot(generations, theoretical_diameters, 
                    'r--', linewidth=2, alpha=0.7, label='Weibel theoretical')
            ax1.legend()
        
        # 2. Diameter tapering ratios
        ax2 = axes[0, 1]
        if hasattr(self, 'tapering_ratios_df') and len(self.tapering_ratios_df) > 0:
            x_labels = [f"{row['from_generation']}→{row['to_generation']}" 
                       for _, row in self.tapering_ratios_df.iterrows()]
            ratios = self.tapering_ratios_df['diameter_ratio']
            
            ax2.bar(range(len(ratios)), ratios, alpha=0.7, edgecolor='black')
            ax2.axhline(y=2**(-1/3), color='r', linestyle='--', 
                       linewidth=2, label='Weibel theoretical (0.793)')
            ax2.set_xlabel('Generation Transition')
            ax2.set_ylabel('Diameter Ratio')
            ax2.set_title('Diameter Reduction Ratios Between Generations')
            ax2.set_xticks(range(len(ratios)))
            ax2.set_xticklabels(x_labels, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Number of branches per generation
        ax3 = axes[1, 0]
        n_branches = self.weibel_analysis_df['n_branches']
        ax3.bar(generations, n_branches, alpha=0.7, edgecolor='black', color='green')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Number of Branches')
        ax3.set_title('Branch Count by Generation')
        ax3.set_xticks(generations)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add theoretical line (2^(generation+1) for perfect dichotomous branching)
        # Generation 0 = 2 main bronchi, then each bifurcates at next generation
        theoretical_branches = [2**(g+1) for g in generations]
        ax3.plot(generations, theoretical_branches, 
                'r--', linewidth=2, alpha=0.7, label='Perfect dichotomy (2^(n+1))')
        ax3.legend()
        
        # 4. Mean branch length by generation
        ax4 = axes[1, 1]
        lengths = self.weibel_analysis_df['length_mean_mm']
        length_errors = self.weibel_analysis_df['length_std_mm']
        
        ax4.errorbar(generations, lengths, yerr=length_errors,
                    marker='s', markersize=8, capsize=5, 
                    linewidth=2, color='purple')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Mean Branch Length (mm)')
        ax4.set_title('Branch Length Across Generations')
        ax4.set_xticks(generations)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Weibel analysis plot saved: {save_path}")
        
        plt.show()
    
    def visualize_distance_from_carina(self, save_path=None, colormap='viridis', max_branches=1000):
        """
        Visualizes branches colored by their PATH distance from the carina
        """
        print("\n=== DISTANCE FROM CARINA VISUALIZATION ===")
        
        if 'distance_from_carina_distal_mm' not in self.branch_metrics_df.columns:
            raise ValueError("Calculate distances from carina first")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        color_values = self.branch_metrics_df['distance_from_carina_distal_mm'].dropna()
        
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = plt.colormaps[colormap]
        
        plotted_count = 0
        for idx, row in self.branch_metrics_df.iterrows():
            if plotted_count >= max_branches:
                break
            
            if pd.isna(row['distance_from_carina_distal_mm']):
                continue
                
            branch_id = int(row['branch_id'])
            distance = row['distance_from_carina_distal_mm']
            
            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)
                
                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]
                    
                    color = cmap(norm(distance))
                    
                    ax.plot(x_coords, y_coords, z_coords, 
                           color=color, linewidth=2, alpha=0.8)
                    plotted_count += 1
                    
            except Exception as e:
                continue
        
        print(f"Branches plotted: {plotted_count}")
        
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Path Distance from Carina (mm)', rotation=270, labelpad=20)
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Bronchial Tree - Path Distance from Carina\n{plotted_count} branches visualized')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()

    def plot_distance_distributions(self, save_path=None):
        """
        Plots distributions of path distances from carina
        """
        if 'distance_from_carina_distal_mm' not in self.branch_metrics_df.columns:
            raise ValueError("Calculate distances from carina first")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histogram of distal distances
        valid_distances = self.branch_metrics_df['distance_from_carina_distal_mm'].dropna()
        axes[0, 0].hist(valid_distances, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('Path Distance from Carina (mm)')
        axes[0, 0].set_ylabel('Number of Branches')
        axes[0, 0].set_title('Distribution of Branch Distances from Carina')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distance vs Diameter
        axes[0, 1].scatter(self.branch_metrics_df['distance_from_carina_distal_mm'],
                          self.branch_metrics_df['diameter_mean_mm'],
                          alpha=0.5, s=20)
        axes[0, 1].set_xlabel('Path Distance from Carina (mm)')
        axes[0, 1].set_ylabel('Mean Diameter (mm)')
        axes[0, 1].set_title('Airway Diameter vs Path Distance from Carina')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distance vs Length
        axes[1, 0].scatter(self.branch_metrics_df['distance_from_carina_distal_mm'],
                          self.branch_metrics_df['length_mm'],
                          alpha=0.5, s=20, color='green')
        axes[1, 0].set_xlabel('Path Distance from Carina (mm)')
        axes[1, 0].set_ylabel('Branch Length (mm)')
        axes[1, 0].set_title('Branch Length vs Path Distance from Carina')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Generation vs Distance (if available)
        if 'generation' in self.branch_metrics_df.columns:
            gen_data = self.branch_metrics_df[['generation', 'distance_from_carina_distal_mm']].dropna()
            axes[1, 1].scatter(gen_data['generation'],
                             gen_data['distance_from_carina_distal_mm'],
                             alpha=0.5, s=20, color='coral')
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Path Distance from Carina (mm)')
            axes[1, 1].set_title('Distance vs Generation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distance distributions saved: {save_path}")
        
        plt.show()
    
    def visualize_skeleton_3d(self, save_path=None, show_bifurcations=False):
        """Visualizes 3D skeleton"""
        print("\n=== 3D Visualization ===")
        
        if self.skeleton is None:
            raise ValueError("Compute skeleton first")
        
        skeleton_coords = np.argwhere(self.skeleton > 0)
        
        subsample = max(1, len(skeleton_coords) // 10000)
        skeleton_coords = skeleton_coords[::subsample]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(skeleton_coords[:, 2], skeleton_coords[:, 1], skeleton_coords[:, 0],
                  c='blue', marker='.', s=1, alpha=0.6, label='Skeleton')
        
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
    
    def visualize_branches_3d(self, save_path=None, colormap='viridis', max_branches=1000, color_by='diameter'):
        """Visualizes 3D branches colored by diameter or length"""
        print("\n=== 3D Branches Visualization ===")
        
        if not hasattr(self, 'branch_metrics_df'):
            raise ValueError("Calculate metrics first")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if color_by == 'length':
            color_values = self.branch_metrics_df['length_mm']
            color_label = 'Length (mm)'
        else:
            color_values = self.branch_metrics_df['diameter_mean_mm']
            color_label = 'Mean Diameter (mm)'
        
        norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
        cmap = plt.colormaps[colormap]
        
        plotted_count = 0
        for idx, row in self.branch_metrics_df.iterrows():
            if plotted_count >= max_branches:
                break
                
            branch_id = int(row['branch_id'])
            color_value = row[color_label.split()[0].lower() + '_mm' if color_by == 'length' else 'diameter_mean_mm']
            
            try:
                coords = self.skeleton_obj.path_coordinates(branch_id)
                
                if len(coords) > 1:
                    z_coords = coords[:, 0]
                    y_coords = coords[:, 1]
                    x_coords = coords[:, 2]
                    
                    color = cmap(norm(color_value))
                    
                    ax.plot(x_coords, y_coords, z_coords, 
                           color=color, linewidth=2, alpha=0.8)
                    plotted_count += 1
                    
            except Exception as e:
                continue
        
        print(f"Branches plotted: {plotted_count}")
        
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(color_label, rotation=270, labelpad=20)
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Bronchial Tree Branches - Colored by {color_label}\n{plotted_count} branches')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()
    
    def visualize_connected_components(self, save_path=None, max_components=10, min_voxels=5):
        """Visualizes connected components"""
        print("\n=== CONNECTED COMPONENTS VISUALIZATION ===")
        
        if self.connected_components is None:
            raise ValueError("Analyze connected components first")
        
        large_components = self.component_stats[self.component_stats['voxel_count'] >= min_voxels]
        components_to_plot = min(max_components, len(large_components))
        
        print(f"Visualizing {components_to_plot} components with at least {min_voxels} voxels")
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.tab20(np.linspace(0, 1, components_to_plot))
        
        total_plotted = 0
        for i in range(components_to_plot):
            comp_id = large_components.iloc[i]['component_id']
            voxel_count = large_components.iloc[i]['voxel_count']
            
            component_coords = np.argwhere(self.connected_components == comp_id)
            
            if len(component_coords) > 0:
                if voxel_count < 100:
                    subsample = 1
                    marker_size = 20
                else:
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
        ax.set_title(f'Connected Components of Skeleton\n{components_to_plot} components, {total_plotted} points')
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Component visualization saved: {save_path}")
        
        plt.show()
    
    def plot_diameter_distribution(self, save_path=None):
        """Plots diameter distribution"""
        if not hasattr(self, 'branch_metrics_df'):
            print("Calculate metrics first")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(self.branch_metrics_df['diameter_mean_mm'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Mean diameter (mm)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Branch Diameters Distribution')
        axes[0].grid(True, alpha=0.3)
        
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
            print("Calculate metrics first")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(self.branch_metrics_df['length_mm'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0].set_xlabel('Branch length (mm)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Branch Length Distribution')
        axes[0].grid(True, alpha=0.3)
        
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
            gen_str = f", Gen {int(row['generation'])}" if 'generation' in row and not pd.isna(row['generation']) else ""
            print(f"Branch {row['branch_id']}: "
                  f"Length = {row['length_mm']:.2f} mm, "
                  f"Mean diameter = {row['diameter_mean_mm']:.2f} mm, "
                  f"Volume = {row['volume_mm3']:.2f} mm³{gen_str}")
        
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
        
        # Save CSV with complete branch metrics
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
        
        # Save Weibel analysis
        if hasattr(self, 'weibel_analysis_df'):
            weibel_path = os.path.join(output_dir, "weibel_generation_analysis.csv")
            self.weibel_analysis_df.to_csv(weibel_path, index=False)
            print(f"Weibel generation analysis saved: {weibel_path}")
        
        if hasattr(self, 'tapering_ratios_df'):
            tapering_path = os.path.join(output_dir, "weibel_tapering_ratios.csv")
            self.tapering_ratios_df.to_csv(tapering_path, index=False)
            print(f"Tapering ratios saved: {tapering_path}")
        
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
                f.write(f"  Total components: {len(self.component_stats)}\n")
                f.write(f"  Largest component: {self.component_stats['voxel_count'].iloc[0]} voxels\n\n")
            
            if hasattr(self, 'branch_metrics_df'):
                f.write("BRANCH METRICS:\n")
                f.write(f"  Number of branches: {len(self.branch_metrics_df)}\n")
                f.write(f"  Mean diameter: {self.branch_metrics_df['diameter_mean_mm'].mean():.2f} mm\n")
                f.write(f"  Min diameter: {self.branch_metrics_df['diameter_mean_mm'].min():.2f} mm\n")
                f.write(f"  Max diameter: {self.branch_metrics_df['diameter_mean_mm'].max():.2f} mm\n")
                f.write(f"  Mean length: {self.branch_metrics_df['length_mm'].mean():.2f} mm\n")
                f.write(f"  Total airway length: {self.branch_metrics_df['length_mm'].sum():.2f} mm\n")
                f.write(f"  Total airway volume: {self.branch_metrics_df['volume_mm3'].sum():.2f} mm³\n\n")
            
            if hasattr(self, 'bifurcations_df'):
                f.write("BIFURCATIONS:\n")
                f.write(f"  Number: {len(self.bifurcations_df)}\n")
                if len(self.bifurcations_df) > 0:
                    f.write(f"  Average degree: {self.bifurcations_df['degree'].mean():.2f}\n\n")
            
            if hasattr(self, 'weibel_analysis_df'):
                f.write("WEIBEL GENERATION ANALYSIS:\n")
                f.write(f"  Maximum generation: {int(self.weibel_analysis_df['generation'].max())}\n")
                f.write(f"  Number of generations: {len(self.weibel_analysis_df)}\n")
                if hasattr(self, 'tapering_ratios_df') and len(self.tapering_ratios_df) > 0:
                    mean_ratio = self.tapering_ratios_df['diameter_ratio'].mean()
                    f.write(f"  Mean tapering ratio: {mean_ratio:.3f}\n")
                    f.write(f"  Weibel theoretical ratio: {2**(-1/3):.3f}\n")
        
        print(f"Summary saved: {summary_path}")
    
    def run_full_analysis(self, output_dir, visualize=True, 
                         max_reconnect_distance_mm=15.0, 
                         min_voxels_for_reconnect=5,
                         max_voxels_for_keep=100):
        """
        Runs complete analysis pipeline with Weibel generation analysis
        """
        print("\n" + "="*60)
        print("COMPLETE BRONCHIAL TREE ANALYSIS PIPELINE")
        print("WITH WEIBEL GENERATION ANALYSIS")
        print("="*60)
        
        # 1. Skeleton
        self.compute_skeleton()
        
        # 2. Connected components
        self.analyze_connected_components()
        
        # 3. Component management
        self.smart_component_management(
            max_reconnect_distance_mm=max_reconnect_distance_mm,
            min_voxels_for_reconnect=min_voxels_for_reconnect,
            max_voxels_for_keep=max_voxels_for_keep,
            remove_tiny_components=True
        )
        
        # Reanalyze after management
        self.analyze_connected_components()
        
        # 4. Build graph
        self.build_graph()
        
        # 5. Identify carina
        self.identify_carina()
        
        # 6. Assign Weibel generations
        self.assign_generations_weibel()
        
        # 7. Calculate lengths
        self.calculate_branch_lengths()
        
        # 8. Calculate diameters
        self.analyze_diameters()
        
        # 9. Merge metrics
        self.merge_branch_metrics()
        
        # 10. Analyze Weibel tapering
        self.analyze_weibel_tapering()
        
        # 11. Calculate path distances from carina
        self.calculate_distances_from_carina()
        
        # 12. Identify bifurcations
        self.identify_bifurcations()
        
        # 13. Save results
        self.save_results(output_dir)
        
        # 14. Visualizations
        if visualize:
            self.visualize_skeleton_3d(
                save_path=os.path.join(output_dir, "skeleton_3d.png"),
                show_bifurcations=False
            )
            
            self.visualize_connected_components(
                save_path=os.path.join(output_dir, "connected_components.png")
            )
            
            self.visualize_branches_3d(
                save_path=os.path.join(output_dir, "branches_3d_diameter.png"),
                color_by='diameter'
            )
            
            self.visualize_branches_3d(
                save_path=os.path.join(output_dir, "branches_3d_length.png"),
                color_by='length'
            )
            
            self.visualize_weibel_generations(
                save_path=os.path.join(output_dir, "weibel_generations.png")
            )
            
            self.plot_weibel_tapering_analysis(
                save_path=os.path.join(output_dir, "weibel_tapering_analysis.png")
            )
            
            self.visualize_distance_from_carina(
                save_path=os.path.join(output_dir, "distance_from_carina.png")
            )
            
            self.plot_distance_distributions(
                save_path=os.path.join(output_dir, "distance_distributions.png")
            )
            
            self.plot_diameter_distribution(
                save_path=os.path.join(output_dir, "diameter_distribution.png")
            )
            
            self.plot_length_distribution(
                save_path=os.path.join(output_dir, "length_distribution.png")
            )

            self.visualize_weibel_generations_with_carina(
                save_path=os.path.join(output_dir, "weibel_generations_with_carina.png")
            )
            
        # 15. Summary
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
            'bifurcations': self.bifurcations_df,
            'weibel_analysis': self.weibel_analysis_df,
            'tapering_ratios': self.tapering_ratios_df
        }