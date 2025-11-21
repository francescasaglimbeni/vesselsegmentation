import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label, binary_dilation, binary_erosion, distance_transform_edt
from scipy.ndimage import generate_binary_structure
import pandas as pd
from skimage.morphology import ball
import os


class SegmentationPreprocessor:
    """
    Pre-processes airway segmentation before skeletonization:
    - Analyzes connected components
    - Keeps only the largest component (main airway tree)
    - Attempts to reconnect nearby disconnected components
    - Cleans small isolated artifacts
    """
    
    def __init__(self, mask_path, spacing=None):
        """
        Args:
            mask_path: Path to .nii.gz airway mask file
            spacing: Tuple (x,y,z) of spacing in mm. If None, reads from image
        """
        self.mask_path = mask_path
        
        # Read image
        print(f"Loading segmentation from: {mask_path}")
        self.sitk_image = sitk.ReadImage(mask_path)
        self.original_mask = sitk.GetArrayFromImage(self.sitk_image)
        
        # Get spacing (z, y, x) -> convert to (x, y, z)
        self.spacing = spacing if spacing else self.sitk_image.GetSpacing()
        print(f"Spacing (x,y,z): {self.spacing} mm")
        print(f"Shape (z,y,x): {self.original_mask.shape}")
        print(f"Positive voxels: {np.sum(self.original_mask > 0)}")
        
        # Results
        self.cleaned_mask = None
        self.connected_components = None
        self.component_stats = None
        self.reconnection_info = []
        
    def analyze_components(self):
        """Analyzes connected components in the original segmentation"""
        print("\n" + "="*60)
        print("CONNECTED COMPONENTS ANALYSIS (PRE-SKELETONIZATION)")
        print("="*60)
        
        # Binarize mask
        binary_mask = (self.original_mask > 0).astype(np.uint8)
        
        # Label connected components with 26-connectivity (face+edge+corner neighbors in 3D)
        structure = generate_binary_structure(3, 3)  # 26-connectivity
        labeled_array, num_features = label(binary_mask, structure=structure)
        
        self.connected_components = labeled_array
        
        print(f"\nFound {num_features} connected components")
        
        if num_features == 0:
            print("WARNING: No components found in mask!")
            return None
        
        # Analyze each component
        component_stats = []
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            voxel_count = np.sum(component_mask)
            
            # Volume in mm³
            voxel_volume = self.spacing[0] * self.spacing[1] * self.spacing[2]
            volume_mm3 = voxel_count * voxel_volume
            
            # Get coordinates
            coords = np.argwhere(component_mask)
            
            if len(coords) > 0:
                # Centroid
                centroid = np.mean(coords, axis=0)
                
                # Bounding box
                z_min, y_min, x_min = np.min(coords, axis=0)
                z_max, y_max, x_max = np.max(coords, axis=0)
                
                # Extent (size in each dimension)
                z_extent = (z_max - z_min) * self.spacing[2]
                y_extent = (y_max - y_min) * self.spacing[1]
                x_extent = (x_max - x_min) * self.spacing[0]
                
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
                    'bbox_max_x': x_max,
                    'extent_z_mm': z_extent,
                    'extent_y_mm': y_extent,
                    'extent_x_mm': x_extent
                })
        
        self.component_stats = pd.DataFrame(component_stats)
        self.component_stats = self.component_stats.sort_values('voxel_count', ascending=False)
        
        # Print statistics
        print(f"\n{'='*60}")
        print("COMPONENT STATISTICS")
        print(f"{'='*60}")
        print(f"Largest component: {self.component_stats['voxel_count'].iloc[0]:,} voxels "
              f"({self.component_stats['volume_mm3'].iloc[0]:.2f} mm³)")
        
        if num_features > 1:
            print(f"2nd largest: {self.component_stats['voxel_count'].iloc[1]:,} voxels "
                  f"({self.component_stats['volume_mm3'].iloc[1]:.2f} mm³)")
            print(f"Smallest: {self.component_stats['voxel_count'].iloc[-1]} voxels "
                  f"({self.component_stats['volume_mm3'].iloc[-1]:.2f} mm³)")
            
            # Calculate size ratios
            total_voxels = np.sum(self.original_mask > 0)
            main_percentage = (self.component_stats['voxel_count'].iloc[0] / total_voxels) * 100
            
            print(f"\nMain component represents {main_percentage:.1f}% of total segmentation")
            print(f"Disconnected components: {num_features - 1}")
            
            # Show top 10 components
            print(f"\n{'Component':<12} {'Voxels':<12} {'Volume (mm³)':<15} {'Extent (mm)'}")
            print("-" * 60)
            for idx, row in self.component_stats.head(10).iterrows():
                extent_str = f"{row['extent_x_mm']:.1f}×{row['extent_y_mm']:.1f}×{row['extent_z_mm']:.1f}"
                print(f"{row['component_id']:<12} {row['voxel_count']:<12,} {row['volume_mm3']:<15.2f} {extent_str}")
        
        return self.component_stats
    
    def keep_largest_component(self):
        """Keeps only the largest connected component"""
        print(f"\n{'='*60}")
        print("KEEPING LARGEST COMPONENT ONLY")
        print(f"{'='*60}")
        
        if self.connected_components is None:
            raise ValueError("Run analyze_components() first")
        
        # Get largest component ID
        largest_id = self.component_stats.iloc[0]['component_id']
        
        # Create mask with only largest component
        self.cleaned_mask = (self.connected_components == largest_id).astype(np.uint8)
        
        original_voxels = np.sum(self.original_mask > 0)
        cleaned_voxels = np.sum(self.cleaned_mask > 0)
        removed_voxels = original_voxels - cleaned_voxels
        
        print(f"Original voxels: {original_voxels:,}")
        print(f"Kept voxels: {cleaned_voxels:,}")
        print(f"Removed voxels: {removed_voxels:,} ({removed_voxels/original_voxels*100:.2f}%)")
        
        return self.cleaned_mask
    
    def reconnect_nearby_components(self, max_distance_mm=10.0, min_component_size=50,
                                   max_components_to_reconnect=10):
        """
        Attempts to reconnect nearby disconnected components to the main tree
        
        Args:
            max_distance_mm: Maximum distance to consider reconnection (mm)
            min_component_size: Minimum size of component to attempt reconnection (voxels)
            max_components_to_reconnect: Maximum number of components to try reconnecting
        """
        print(f"\n{'='*60}")
        print("ATTEMPTING COMPONENT RECONNECTION")
        print(f"{'='*60}")
        print(f"Max reconnection distance: {max_distance_mm} mm")
        print(f"Min component size: {min_component_size} voxels")
        
        if self.connected_components is None:
            raise ValueError("Run analyze_components() first")
        
        if len(self.component_stats) <= 1:
            print("Only one component found - no reconnection needed")
            self.cleaned_mask = (self.original_mask > 0).astype(np.uint8)
            return self.cleaned_mask
        
        # Start with largest component
        main_id = self.component_stats.iloc[0]['component_id']
        main_mask = (self.connected_components == main_id).astype(np.uint8)
        
        # Get boundary of main component using dilation
        main_boundary = binary_dilation(main_mask) & ~main_mask
        main_boundary_coords = np.argwhere(main_boundary)
        
        print(f"\nMain component boundary: {len(main_boundary_coords):,} voxels")
        
        reconnected_count = 0
        kept_isolated_count = 0
        
        # Try to reconnect other components
        for idx in range(1, min(len(self.component_stats), max_components_to_reconnect + 1)):
            comp_row = self.component_stats.iloc[idx]
            comp_id = comp_row['component_id']
            comp_size = comp_row['voxel_count']
            
            if comp_size < min_component_size:
                print(f"\nComponent {comp_id}: {comp_size} voxels - too small, skipping")
                continue
            
            print(f"\nComponent {comp_id}: {comp_size} voxels ({comp_row['volume_mm3']:.2f} mm³)")
            
            # Get component boundary
            comp_mask = (self.connected_components == comp_id).astype(np.uint8)
            comp_boundary = binary_dilation(comp_mask) & ~comp_mask
            comp_boundary_coords = np.argwhere(comp_boundary)
            
            # Find closest points between boundaries
            min_distance_voxels = float('inf')
            best_main_point = None
            best_comp_point = None
            
            # Sample for efficiency
            main_sample = main_boundary_coords[::max(1, len(main_boundary_coords)//1000)]
            comp_sample = comp_boundary_coords[::max(1, len(comp_boundary_coords)//200)]
            
            for comp_point in comp_sample:
                for main_point in main_sample:
                    # Physical distance
                    diff = (comp_point - main_point) * np.array([self.spacing[2], 
                                                                  self.spacing[1], 
                                                                  self.spacing[0]])
                    distance = np.linalg.norm(diff)
                    
                    if distance < min_distance_voxels:
                        min_distance_voxels = distance
                        best_main_point = main_point
                        best_comp_point = comp_point
            
            print(f"  Minimum distance to main component: {min_distance_voxels:.2f} mm")
            
            # Decision: reconnect or keep isolated?
            if min_distance_voxels <= max_distance_mm:
                print(f"  ✓ Reconnecting (distance {min_distance_voxels:.2f} mm ≤ {max_distance_mm} mm)")
                
                # Create bridge
                bridge_mask = self._create_bridge(best_comp_point, best_main_point, comp_mask.shape)
                
                # Add component and bridge to main mask
                main_mask = main_mask | comp_mask | bridge_mask
                
                # Update boundary for next iterations
                main_boundary = binary_dilation(main_mask) & ~main_mask
                main_boundary_coords = np.argwhere(main_boundary)
                
                self.reconnection_info.append({
                    'component_id': comp_id,
                    'size_voxels': comp_size,
                    'distance_mm': min_distance_voxels,
                    'reconnected': True
                })
                
                reconnected_count += 1
            else:
                print(f"  ✗ Keeping isolated (distance {min_distance_voxels:.2f} mm > {max_distance_mm} mm)")
                self.reconnection_info.append({
                    'component_id': comp_id,
                    'size_voxels': comp_size,
                    'distance_mm': min_distance_voxels,
                    'reconnected': False
                })
                kept_isolated_count += 1
        
        self.cleaned_mask = main_mask
        
        print(f"\n{'='*60}")
        print("RECONNECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Components reconnected: {reconnected_count}")
        print(f"Components kept isolated: {kept_isolated_count}")
        print(f"Final mask voxels: {np.sum(self.cleaned_mask):,}")
        
        return self.cleaned_mask
    
    def _create_bridge(self, point1, point2, shape):
        """Creates a cylindrical bridge between two points"""
        p1 = np.array(point1, dtype=int)
        p2 = np.array(point2, dtype=int)
        
        # Calculate number of points along the line
        distance_voxels = np.linalg.norm(p2 - p1)
        num_points = max(3, int(distance_voxels * 2))
        
        # Create empty mask
        bridge = np.zeros(shape, dtype=np.uint8)
        
        # Draw line with thickness
        for t in np.linspace(0, 1, num_points):
            center = p1 + t * (p2 - p1)
            z, y, x = center.astype(int)
            
            # Add small sphere at each point (radius ~2 voxels)
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dz*dz + dy*dy + dx*dx <= 4:  # sphere condition
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if (0 <= nz < shape[0] and 
                                0 <= ny < shape[1] and 
                                0 <= nx < shape[2]):
                                bridge[nz, ny, nx] = 1
        
        return bridge.astype(bool)
    
    def morphological_cleanup(self, remove_small_holes=True, hole_size_mm3=50,
                            smooth_surface=False):
        """
        Applies morphological operations to clean up the mask
        
        Args:
            remove_small_holes: Fill small holes in the segmentation
            hole_size_mm3: Maximum hole size to fill (mm³)
            smooth_surface: Apply slight smoothing (erosion + dilation)
        """
        print(f"\n{'='*60}")
        print("MORPHOLOGICAL CLEANUP")
        print(f"{'='*60}")
        
        if self.cleaned_mask is None:
            self.cleaned_mask = (self.original_mask > 0).astype(np.uint8)
        
        mask = self.cleaned_mask.copy()
        
        if remove_small_holes:
            print(f"Filling holes smaller than {hole_size_mm3} mm³...")
            
            # Invert mask to find holes
            inverted = ~mask.astype(bool)
            
            # Label holes
            labeled_holes, num_holes = label(inverted)
            
            voxel_volume = self.spacing[0] * self.spacing[1] * self.spacing[2]
            max_hole_voxels = int(hole_size_mm3 / voxel_volume)
            
            filled_count = 0
            for hole_id in range(1, num_holes + 1):
                hole_size = np.sum(labeled_holes == hole_id)
                if hole_size < max_hole_voxels:
                    mask[labeled_holes == hole_id] = 1
                    filled_count += 1
            
            print(f"  Filled {filled_count} small holes")
        
        if smooth_surface:
            print("Smoothing surface (erosion + dilation)...")
            # Small erosion followed by dilation
            mask = binary_erosion(mask, structure=ball(1))
            mask = binary_dilation(mask, structure=ball(1))
            print("  Surface smoothed")
        
        self.cleaned_mask = mask.astype(np.uint8)
        
        return self.cleaned_mask
    
    def visualize_components_3d(self, save_path=None, max_components=10):
        """Visualizes the connected components in 3D"""
        if self.connected_components is None:
            raise ValueError("Run analyze_components() first")
        
        print(f"\nVisualizing top {max_components} components...")
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.tab20(np.linspace(0, 1, max_components))
        
        for i in range(min(max_components, len(self.component_stats))):
            comp_id = self.component_stats.iloc[i]['component_id']
            comp_size = self.component_stats.iloc[i]['voxel_count']
            
            coords = np.argwhere(self.connected_components == comp_id)
            
            # Subsample for visualization
            if len(coords) > 5000:
                subsample = len(coords) // 5000
                coords = coords[::subsample]
            
            ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
                      c=[colors[i]], s=10, alpha=0.6,
                      label=f'Comp {comp_id} ({comp_size:,} vox)')
        
        ax.set_xlabel('X (voxel)')
        ax.set_ylabel('Y (voxel)')
        ax.set_zlabel('Z (voxel)')
        ax.set_title(f'Connected Components Analysis\n{len(self.component_stats)} total components')
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def visualize_before_after(self, save_path=None):
        """Visualizes original vs cleaned mask"""
        if self.cleaned_mask is None:
            raise ValueError("Run preprocessing steps first")
        
        fig = plt.figure(figsize=(18, 6))
        
        # Original
        ax1 = fig.add_subplot(131, projection='3d')
        original_coords = np.argwhere(self.original_mask > 0)
        subsample = max(1, len(original_coords) // 5000)
        original_coords = original_coords[::subsample]
        ax1.scatter(original_coords[:, 2], original_coords[:, 1], original_coords[:, 0],
                   c='blue', s=1, alpha=0.5)
        ax1.set_title(f'Original\n({np.sum(self.original_mask > 0):,} voxels)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Cleaned
        ax2 = fig.add_subplot(132, projection='3d')
        cleaned_coords = np.argwhere(self.cleaned_mask > 0)
        subsample = max(1, len(cleaned_coords) // 5000)
        cleaned_coords = cleaned_coords[::subsample]
        ax2.scatter(cleaned_coords[:, 2], cleaned_coords[:, 1], cleaned_coords[:, 0],
                   c='green', s=1, alpha=0.5)
        ax2.set_title(f'Cleaned\n({np.sum(self.cleaned_mask > 0):,} voxels)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Difference
        ax3 = fig.add_subplot(133, projection='3d')
        removed = (self.original_mask > 0) & (self.cleaned_mask == 0)
        removed_coords = np.argwhere(removed)
        if len(removed_coords) > 0:
            subsample = max(1, len(removed_coords) // 5000)
            removed_coords = removed_coords[::subsample]
            ax3.scatter(removed_coords[:, 2], removed_coords[:, 1], removed_coords[:, 0],
                       c='red', s=1, alpha=0.5)
        ax3.set_title(f'Removed\n({np.sum(removed):,} voxels)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def save_cleaned_mask(self, output_path):
        """Saves the cleaned mask as NIfTI"""
        if self.cleaned_mask is None:
            raise ValueError("No cleaned mask available. Run preprocessing first.")
        
        # Create SimpleITK image
        cleaned_sitk = sitk.GetImageFromArray(self.cleaned_mask)
        cleaned_sitk.CopyInformation(self.sitk_image)
        
        # Save
        sitk.WriteImage(cleaned_sitk, output_path)
        print(f"\nCleaned mask saved: {output_path}")
        
        return output_path
    
    def save_report(self, output_dir):
        """Saves a detailed preprocessing report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "preprocessing_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("AIRWAY SEGMENTATION PREPROCESSING REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Input file: {self.mask_path}\n")
            f.write(f"Spacing (x,y,z): {self.spacing} mm\n")
            f.write(f"Shape (z,y,x): {self.original_mask.shape}\n\n")
            
            f.write(f"Original voxels: {np.sum(self.original_mask > 0):,}\n")
            if self.cleaned_mask is not None:
                f.write(f"Cleaned voxels: {np.sum(self.cleaned_mask > 0):,}\n")
                removed = np.sum(self.original_mask > 0) - np.sum(self.cleaned_mask > 0)
                f.write(f"Removed voxels: {removed:,} ({removed/np.sum(self.original_mask > 0)*100:.2f}%)\n\n")
            
            if self.component_stats is not None:
                f.write("\nCONNECTED COMPONENTS:\n")
                f.write(f"Total components found: {len(self.component_stats)}\n")
                f.write(f"Largest component: {self.component_stats['voxel_count'].iloc[0]:,} voxels\n")
                if len(self.component_stats) > 1:
                    f.write(f"2nd largest: {self.component_stats['voxel_count'].iloc[1]:,} voxels\n")
                f.write("\n")
            
            if self.reconnection_info:
                f.write("\nRECONNECTION ATTEMPTS:\n")
                for info in self.reconnection_info:
                    status = "✓ Reconnected" if info['reconnected'] else "✗ Kept isolated"
                    f.write(f"Component {info['component_id']}: {info['size_voxels']} voxels, "
                           f"distance {info['distance_mm']:.2f} mm - {status}\n")
        
        print(f"Report saved: {report_path}")
        
        # Save component statistics CSV
        if self.component_stats is not None:
            csv_path = os.path.join(output_dir, "component_statistics.csv")
            self.component_stats.to_csv(csv_path, index=False)
            print(f"Component stats saved: {csv_path}")
    
    def run_full_preprocessing(self, output_dir, 
                              try_reconnection=True,
                              max_reconnect_distance_mm=10.0,
                              min_component_size=50,
                              visualize=True):
        """
        Runs complete preprocessing pipeline
        
        Args:
            output_dir: Output directory
            try_reconnection: Whether to attempt reconnecting components
            max_reconnect_distance_mm: Max distance for reconnection
            min_component_size: Min size for reconnection attempt
            visualize: Generate visualizations
        """
        print("\n" + "="*60)
        print("AIRWAY SEGMENTATION PREPROCESSING PIPELINE")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Analyze components
        self.analyze_components()
        
        # 2. Decision: reconnect or just keep largest?
        if try_reconnection and len(self.component_stats) > 1:
            self.reconnect_nearby_components(
                max_distance_mm=max_reconnect_distance_mm,
                min_component_size=min_component_size
            )
        else:
            self.keep_largest_component()
        
        # 3. Morphological cleanup
        self.morphological_cleanup(
            remove_small_holes=True,
            hole_size_mm3=50,
            smooth_surface=False
        )
        
        # 4. Save results
        cleaned_path = os.path.join(output_dir, "cleaned_airway_mask.nii.gz")
        self.save_cleaned_mask(cleaned_path)
        self.save_report(output_dir)
        
        # 5. Visualizations
        if visualize:
            self.visualize_components_3d(
                save_path=os.path.join(output_dir, "components_3d.png")
            )
            self.visualize_before_after(
                save_path=os.path.join(output_dir, "before_after.png")
            )
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED!")
        print("="*60)
        print(f"\nCleaned mask ready: {cleaned_path}")
        print("You can now use this for skeletonization")
        
        return self.cleaned_mask, cleaned_path
