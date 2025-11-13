import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import kimimaro
from collections import defaultdict

LABEL_NAMES = {1: 'Vessels'}

def snap_to_medial_axis(coords, distance_transform, max_search_radius=3):
    snapped_coords = []
    shape = distance_transform.shape
    
    for coord in coords:
        z, y, x = coord
        
        z_min = max(0, z - max_search_radius)
        z_max = min(shape[0], z + max_search_radius + 1)
        y_min = max(0, y - max_search_radius)
        y_max = min(shape[1], y + max_search_radius + 1)
        x_min = max(0, x - max_search_radius)
        x_max = min(shape[2], x + max_search_radius + 1)
        
        local_patch = distance_transform[z_min:z_max, y_min:y_max, x_min:x_max]
        
        if local_patch.size == 0:
            snapped_coords.append(coord)
            continue
        
        max_idx = np.unravel_index(np.argmax(local_patch), local_patch.shape)
        
        snapped_z = z_min + max_idx[0]
        snapped_y = y_min + max_idx[1]
        snapped_x = x_min + max_idx[2]
        
        snapped_coords.append([snapped_z, snapped_y, snapped_x])
    
    return np.array(snapped_coords, dtype=int)


def count_connected_components(binary_mask):
    """Conta il numero di oggetti connessi nella maschera."""
    labeled_array, num_features = ndimage.label(binary_mask)
    return num_features


def extract_centerline_coords_enhanced(binary_mask: np.ndarray, spacing):
    coords_list = []
    skeleton_objects = []
    
    lbl = binary_mask.astype(np.uint32)
    
    skels = kimimaro.skeletonize(
        lbl,
        anisotropy=spacing,
        dust_threshold=10,
        fix_branching=True,
        progress=False
    )
    
    for s in skels.values():
        if hasattr(s, "vertices") and len(s.vertices) > 0:
            coords_list.append(np.asarray(s.vertices, dtype=int))
            skeleton_objects.append(s)
    
    backend_used = "kimimaro"
    
    if not coords_list:
        return np.empty((0, 3), dtype=int), [], backend_used, 0
    
    n_components = len(coords_list)
    all_coords = np.vstack(coords_list)
    
    return all_coords, skeleton_objects, backend_used, n_components


def sample_centerline_diameters_enhanced(mask_bool: np.ndarray, spacing):
    if not np.any(mask_bool):
        return np.array([]), None, {}
    
    # Conta componenti connesse
    n_objects = count_connected_components(mask_bool)
    
    # Calcola Distance Transform
    DT = ndimage.distance_transform_edt(mask_bool, sampling=spacing)
    
    # Estrai centerline con oggetti skeleton
    cl_coords, skeleton_objects, backend_used, n_components = extract_centerline_coords_enhanced(mask_bool, spacing)
    stats = {
        'n_objects': n_objects,
        'n_components': n_components,
        'n_points_raw': len(cl_coords),
        'n_points_snapped': 0,
        'n_points_filtered': 0,
        'skeleton_coords': cl_coords,
    }
    
    if cl_coords.shape[0] == 0:
        return np.array([]), backend_used, stats
    
    print(f"    Raw centerline points: {len(cl_coords):,}")
    print(f"    Connected objects: {n_objects}")
    print(f"    Skeleton components: {n_components}")
    
    # Snap to medial axis
    cl_coords_snapped = snap_to_medial_axis(cl_coords, DT, max_search_radius=3)
    stats['n_points_snapped'] = len(cl_coords_snapped)
    print(f"    After snapping: {len(cl_coords_snapped):,}")
    
    # Filtra punti fuori dal vaso
    radii = DT[cl_coords_snapped[:, 0], cl_coords_snapped[:, 1], cl_coords_snapped[:, 2]]
    valid_mask = radii > 0.10
    cl_coords_filtered = cl_coords_snapped[valid_mask]
    radii_filtered = radii[valid_mask]
    stats['n_points_filtered'] = len(cl_coords_filtered)
    stats['radii'] = radii_filtered    
    if len(cl_coords_filtered) == 0:
        return np.array([]), backend_used, stats
    
    diameters = 2.0 * radii_filtered
    stats['n_points_resampled'] = len(diameters)
    
    return diameters, backend_used, stats

def analyze_diameters(mask_path, spacing=None, diameter_method='centerline'):
    try:
        img = sitk.ReadImage(mask_path)
        mask_data = sitk.GetArrayFromImage(img)
        
        if spacing is None:
            spacing = img.GetSpacing()[::-1]
            print(f"Spacing from file: {spacing} mm")
        else:
            print(f"Using manual spacing: {spacing} mm")
        
        unique_labels = np.unique(mask_data)
        print(f"\nLabels found in mask: {unique_labels}")
        
        results = {}

        for label in [1, 2]:
            if label not in LABEL_NAMES:
                continue
                
            label_name = LABEL_NAMES[label]
            mask = (mask_data == label)

            if not np.any(mask):
                print(f"\n{label_name}: ⚠️  No voxels found")
                results[label] = None
                continue

            print(f"\n{label_name}:")
            print(f"{'-'*60}")

            if diameter_method == 'centerline':
                diameters, backend_used, stats = sample_centerline_diameters_enhanced(mask, spacing)

                if diameters.size == 0:
                    print(f"  ⚠️ Centerline vuota. Fallback su all_voxels.")
                    dist_transform = ndimage.distance_transform_edt(mask, sampling=spacing)
                    diameters = 2.0 * dist_transform[mask]
                    backend_used = "fallback:all_voxels"
                    stats = {'n_objects': count_connected_components(mask)}
                else:
                    print(f"  ✓ Enhanced analysis complete")
            else:
                dist_transform = ndimage.distance_transform_edt(mask, sampling=spacing)
                diameters = 2.0 * dist_transform[mask]
                backend_used = "all_voxels"
                stats = {'n_objects': count_connected_components(mask)}

            diameters_filtered = diameters[(diameters > 0.3) & (diameters < 20)]

            def pct(x):
                return (np.sum(x) / len(diameters) * 100.0) if len(diameters) > 0 else 0.0

            results[label] = {
                'label_name': label_name,
                'method': diameter_method,
                'diameter_backend': backend_used,
                'processing_stats': stats,
                'all_diameters': diameters,
                'filtered_diameters': diameters_filtered,
                'min': float(diameters.min()) if diameters.size else 0.0,
                'max': float(diameters.max()) if diameters.size else 0.0,
                'mean': float(diameters.mean()) if diameters.size else 0.0,
                'median': float(np.median(diameters)) if diameters.size else 0.0,
                'p10': float(np.percentile(diameters, 10)) if diameters.size else 0.0,
                'p25': float(np.percentile(diameters, 25)) if diameters.size else 0.0,
                'p75': float(np.percentile(diameters, 75)) if diameters.size else 0.0,
                'p90': float(np.percentile(diameters, 90)) if diameters.size else 0.0,
                'pct_very_small': pct(diameters < 1.0),
                'pct_small': pct((diameters >= 1.0) & (diameters < 2.0)),
                'pct_medium': pct((diameters >= 2.0) & (diameters < 5.0)),
                'pct_large': pct(diameters >= 5.0),
                'n_sampled_points': len(diameters)
            }

            print(f"\n  Topological Results:")
            print(f"    Connected objects: {stats.get('n_objects', 'N/A')}")
            print(f"    Skeleton components: {stats.get('n_components', 'N/A')}")
            
            print(f"\n  Diameter Results:")
            print(f"    Backend: {backend_used}")
            print(f"    Sample points: {results[label]['n_sampled_points']:,}")
            print(f"    Diameter range: {results[label]['min']:.2f} - {results[label]['max']:.2f} mm")
            print(f"    Mean: {results[label]['mean']:.2f} mm")
            print(f"    Median: {results[label]['median']:.2f} mm")
            print(f"    25th-75th percentile: {results[label]['p25']:.2f} - {results[label]['p75']:.2f} mm")
            
        return results
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_plots(diameter_stats, output_prefix="diameter_analysis"):
    print(f"\n{'='*70}")
    print(f"CREATING PLOTS")
    print(f"{'='*70}")
    
    if not diameter_stats:
        print("⚠️  Insufficient data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {'Arteries': '#e74c3c', 'Veins': '#3498db'}
    
    for idx, label in enumerate([1, 2]):
        if label not in diameter_stats or diameter_stats[label] is None:
            continue
            
        diam = diameter_stats[label]
        label_name = diam['label_name']
        diameters = diam['all_diameters']
        color = colors.get(label_name, '#3498db')
        
        row = idx
        
        # Histogram
        axes[row, 0].hist(diameters, bins=50, alpha=0.7, color=color, edgecolor='black')
        axes[row, 0].set_xlabel('Diameter (mm)')
        axes[row, 0].set_ylabel('Frequency')
        axes[row, 0].set_title(f'{label_name} - Diameter Distribution')
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_xlim([0, 10])
        
        # Box plot
        bp = axes[row, 1].boxplot([diameters], labels=[label_name], 
                             showfliers=False, patch_artist=True)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        axes[row, 1].set_ylabel('Diameter (mm)')
        axes[row, 1].set_title(f'{label_name} - Box Plot')
        axes[row, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}.png")
    plt.close('all')

def print_final_summary(diameter_stats):
    """Stampa il riepilogo finale."""
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    
    if not diameter_stats:
        print("No data available for summary")
        return
    
    print("\nResults:")
    for label in [1, 2]:
        if label in diameter_stats and diameter_stats[label] is not None:
            diam = diameter_stats[label]
            stats = diam.get('processing_stats', {})
            label_name = diam['label_name']
            
            print(f"\n- {label_name}:")
            print(f"  • Connected objects: {stats.get('n_objects', 'N/A')}")
            print(f"  • Skeleton components: {stats.get('n_components', 'N/A')}")
            print(f"  • Mean diameter: {diam['mean']:.2f} mm (range: {diam['min']:.2f}-{diam['max']:.2f} mm)")
            print(f"  • Sampled points: {diam['n_sampled_points']:,}")

mask_path = '/content/vesselsegmentation/vessels_cleaned/1.2.840.113704.1.111.2604.1126357612.7_cleaned.nii.gz'
spacing = (0.7, 0.7, 0.7)
output_prefix = "diameters_analysis_GT"
DIAMETER_METHOD = 'centerline'

print("="*70)
print("VESSEL DIAMETER ANALYSIS - ENHANCED VERSION")
print("="*70)
print(f"\nInput file: {mask_path}")
print(f"Method: {DIAMETER_METHOD}")
print("\nFeatures:")
print("  ✓ Number of objects")
print("  ✓ Radii analysis")
print("  ✓ 3D skeleton visualization")

diameter_stats = analyze_diameters(mask_path, spacing, diameter_method=DIAMETER_METHOD)

if diameter_stats:
    create_plots(diameter_stats, output_prefix)
    print_final_summary(diameter_stats)

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)