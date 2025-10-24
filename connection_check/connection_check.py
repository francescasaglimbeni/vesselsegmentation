"""
Quick analysis script per capire le distanze tra componenti
e suggerire parametri realistici.
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

def analyze_component_distances(vessel_path, spacing, sample_size=1000):
    """
    Analizza le distanze tra componenti disconnesse.
    """
    print("\n" + "="*70)
    print("ANALYZING INTER-COMPONENT DISTANCES")
    print("="*70)
    
    img = sitk.ReadImage(vessel_path)
    mask = sitk.GetArrayFromImage(img).astype(bool)
    
    # Label components
    labeled, num_components = ndimage.label(mask)
    
    if num_components <= 1:
        print("Only one component, no distances to compute")
        return
    
    print(f"\nFound {num_components} components")
    
    # Trova la componente principale (più grande)
    component_sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
    main_component_label = np.argmax(component_sizes) + 1
    main_component_size = component_sizes[main_component_label - 1]
    
    print(f"Main component: {main_component_size} voxels")
    
    # Analizza distanze dei frammenti dalla componente principale
    main_mask = (labeled == main_component_label)
    main_coords = np.array(np.where(main_mask)).T
    main_coords_mm = main_coords * np.array(spacing)
    
    # Build KD-tree
    tree = cKDTree(main_coords_mm)
    
    distances_by_size = {}
    
    for label_idx in range(1, num_components + 1):
        if label_idx == main_component_label:
            continue
        
        fragment_mask = (labeled == label_idx)
        fragment_size = component_sizes[label_idx - 1]
        fragment_coords = np.array(np.where(fragment_mask)).T
        
        # Campiona punti (per velocità)
        if len(fragment_coords) > sample_size:
            indices = np.random.choice(len(fragment_coords), sample_size, replace=False)
            fragment_coords = fragment_coords[indices]
        
        fragment_coords_mm = fragment_coords * np.array(spacing)
        
        # Query distanze minime
        distances, _ = tree.query(fragment_coords_mm, k=1)
        min_distance = distances.min()
        mean_distance = distances.mean()
        
        # Categorizza per dimensione
        if fragment_size < 50:
            category = "tiny"
        elif fragment_size < 200:
            category = "small"
        elif fragment_size < 1000:
            category = "medium"
        else:
            category = "large"
        
        if category not in distances_by_size:
            distances_by_size[category] = []
        
        distances_by_size[category].append({
            'size': fragment_size,
            'min_dist': min_distance,
            'mean_dist': mean_distance
        })
    
    # Print statistics
    print("\n" + "-"*70)
    print("DISTANCE STATISTICS BY FRAGMENT SIZE")
    print("-"*70)
    
    for category in ['tiny', 'small', 'medium', 'large']:
        if category not in distances_by_size:
            continue
        
        data = distances_by_size[category]
        min_dists = [d['min_dist'] for d in data]
        
        print(f"\n{category.upper()} fragments (n={len(data)}):")
        print(f"  Min distance range: {min(min_dists):.2f} - {max(min_dists):.2f} mm")
        print(f"  Mean min distance: {np.mean(min_dists):.2f} ± {np.std(min_dists):.2f} mm")
        print(f"  Median min distance: {np.median(min_dists):.2f} mm")
        print(f"  25th percentile: {np.percentile(min_dists, 25):.2f} mm")
        print(f"  75th percentile: {np.percentile(min_dists, 75):.2f} mm")
    
    # Suggest parameters
    print("\n" + "="*70)
    print("SUGGESTED PARAMETERS")
    print("="*70)
    
    all_distances = []
    for category_data in distances_by_size.values():
        all_distances.extend([d['min_dist'] for d in category_data])
    
    if all_distances:
        p50 = np.percentile(all_distances, 50)
        p75 = np.percentile(all_distances, 75)
        p90 = np.percentile(all_distances, 90)
        
        print(f"\nBased on distance distribution:")
        print(f"  Conservative (reconnect only very close): max_connection_distance_mm = {p50:.1f}")
        print(f"  Moderate (reconnect most fragments): max_connection_distance_mm = {p75:.1f}")
        print(f"  Aggressive (reconnect almost all): max_connection_distance_mm = {p90:.1f}")
        print(f"\n  Suggested: Start with {p50:.1f} mm and verify in Slicer")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of distances
    axes[0].hist(all_distances, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(p50, color='green', linestyle='--', label=f'50th percentile ({p50:.1f}mm)')
    axes[0].axvline(p75, color='orange', linestyle='--', label=f'75th percentile ({p75:.1f}mm)')
    axes[0].axvline(p90, color='red', linestyle='--', label=f'90th percentile ({p90:.1f}mm)')
    axes[0].set_xlabel('Distance to main component (mm)')
    axes[0].set_ylabel('Number of fragments')
    axes[0].set_title('Fragment Distance Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distance vs Size scatter
    sizes = []
    dists = []
    for category_data in distances_by_size.values():
        for d in category_data:
            sizes.append(d['size'])
            dists.append(d['min_dist'])
    
    axes[1].scatter(sizes, dists, alpha=0.5, s=20)
    axes[1].set_xlabel('Fragment Size (voxels)')
    axes[1].set_ylabel('Distance to main component (mm)')
    axes[1].set_title('Distance vs Fragment Size')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(p50, color='green', linestyle='--', alpha=0.5)
    axes[1].axhline(p75, color='orange', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('distance_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: distance_analysis.png")
    plt.close()
    
    return distances_by_size


def analyze_vessel_diameters(vessel_path, centerline_path, spacing):
    """
    Analizza i diametri dei vasi usando le centerlines.
    """
    print("\n" + "="*70)
    print("ANALYZING VESSEL DIAMETERS")
    print("="*70)
    
    vessel_img = sitk.ReadImage(vessel_path)
    vessel_mask = sitk.GetArrayFromImage(vessel_img).astype(bool)
    
    # Distance transform dalla maschera dei vasi
    dist_transform = ndimage.distance_transform_edt(vessel_mask, sampling=spacing)
    
    if centerline_path and os.path.exists(centerline_path):
        centerline_img = sitk.ReadImage(centerline_path)
        centerline_mask = sitk.GetArrayFromImage(centerline_img).astype(bool)
        
        # Estrai diametri lungo le centerlines
        diameters = dist_transform[centerline_mask] * 2  # radius -> diameter
        
        print(f"\nDiameter statistics (from {centerline_mask.sum()} centerline points):")
        print(f"  Min: {diameters.min():.2f} mm")
        print(f"  Max: {diameters.max():.2f} mm")
        print(f"  Mean: {diameters.mean():.2f} mm")
        print(f"  Median: {np.median(diameters):.2f} mm")
        print(f"  25th percentile: {np.percentile(diameters, 25):.2f} mm")
        print(f"  75th percentile: {np.percentile(diameters, 75):.2f} mm")
        
        # Suggest threshold for large vessels
        p75 = np.percentile(diameters, 75)
        p90 = np.percentile(diameters, 90)
        
        print(f"\n  Suggested large_vessel_threshold_mm:")
        print(f"    Conservative (top 10%): {p90:.1f} mm")
        print(f"    Moderate (top 25%): {p75:.1f} mm")
        
        # Histogram
        plt.figure(figsize=(10, 5))
        plt.hist(diameters, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(p75, color='orange', linestyle='--', label=f'75th percentile ({p75:.1f}mm)')
        plt.axvline(p90, color='red', linestyle='--', label=f'90th percentile ({p90:.1f}mm)')
        plt.xlabel('Vessel Diameter (mm)')
        plt.ylabel('Frequency')
        plt.title('Vessel Diameter Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('diameter_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: diameter_analysis.png")
        plt.close()



path_vessels = '1.2.840.113704.1.111.1396.1132404220.7_cleaned.nii.gz'
path_centerlines = '1.2.840.113704.1.111.1396.1132404220.7_centerlines.nii.gz'
spacing = (0.625, 0.625, 0.625)  
spacing = tuple(spacing)
    
# Analyze distances
analyze_component_distances(path_vessels, spacing)

# Analyze diameters if centerlines available
if path_centerlines:
    analyze_vessel_diameters(path_vessels, path_centerlines, spacing)

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Review the generated plots (distance_analysis.png, diameter_analysis.png)")
print("2. Open Slicer and manually verify a few connections")
print("3. Adjust parameters based on your observations")
print("4. Re-run the vessel reconnection with new parameters")
