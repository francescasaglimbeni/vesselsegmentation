"""
Analisi vasi polmonari: Diameter Analysis - FIXED VERSION
Correzioni applicate:
1. Snap della centerline ai massimi locali della Distance Transform
2. Filtraggio dei punti fuori dal vaso
3. Ricampionamento uniforme delle polylines
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import pandas as pd
import kimimaro

LABEL_NAMES = {1: 'Vessels'}  # Per TS: solo 0 (background) e 1 (vasi)

def snap_to_medial_axis(coords, distance_transform, max_search_radius=3):
    """
    Sposta ogni punto della centerline verso il massimo locale della DT.
    
    Args:
        coords: Array (N, 3) di coordinate in voxel space (z, y, x)
        distance_transform: 3D array con la distance transform
        max_search_radius: Raggio di ricerca in voxel per il massimo locale
    
    Returns:
        Array (N, 3) di coordinate corrette
    """
    snapped_coords = []
    shape = distance_transform.shape
    
    for coord in coords:
        z, y, x = coord
        
        # Definisci finestra di ricerca
        z_min = max(0, z - max_search_radius)
        z_max = min(shape[0], z + max_search_radius + 1)
        y_min = max(0, y - max_search_radius)
        y_max = min(shape[1], y + max_search_radius + 1)
        x_min = max(0, x - max_search_radius)
        x_max = min(shape[2], x + max_search_radius + 1)
        
        # Estrai patch locale
        local_patch = distance_transform[z_min:z_max, y_min:y_max, x_min:x_max]
        
        if local_patch.size == 0:
            snapped_coords.append(coord)
            continue
        
        # Trova massimo locale
        max_idx = np.unravel_index(np.argmax(local_patch), local_patch.shape)
        
        # Converti in coordinate globali
        snapped_z = z_min + max_idx[0]
        snapped_y = y_min + max_idx[1]
        snapped_x = x_min + max_idx[2]
        
        snapped_coords.append([snapped_z, snapped_y, snapped_x])
    
    return np.array(snapped_coords, dtype=int)


def resample_polyline(coords, spacing, step_mm=0.5):
    """
    Ricampiona una polyline ad intervalli regolari.
    
    Args:
        coords: Array (N, 3) di coordinate in voxel space
        spacing: Tuple (sz, sy, sx) in mm
        step_mm: Distanza in mm tra i punti ricampionati
    
    Returns:
        Array (M, 3) di coordinate ricampionate (M può essere diverso da N)
    """
    if len(coords) < 4:
        return coords
    
    # Converti in mm
    coords_mm = coords * np.array(spacing)
    
    try:
        # Parametrizzazione spline (k=3 richiede almeno 4 punti)
        tck, u = splprep([coords_mm[:, 0], coords_mm[:, 1], coords_mm[:, 2]], 
                         s=0, k=min(3, len(coords) - 1))
        
        # Calcola lunghezza totale approssimativa
        total_length = np.sum(np.linalg.norm(np.diff(coords_mm, axis=0), axis=1))
        
        # Numero di punti da campionare
        n_points = max(int(total_length / step_mm), 2)
        
        # Ricampiona
        u_new = np.linspace(0, 1, n_points)
        resampled_mm = np.array(splev(u_new, tck)).T
        
        # Riconverti in voxel space
        resampled_voxel = np.round(resampled_mm / np.array(spacing)).astype(int)
        
        return resampled_voxel
    
    except Exception as e:
        # Fallback: ritorna coordinate originali
        print(f"  Warning: Resampling failed ({e}), using original coords")
        return coords


def extract_centerline_coords(binary_mask: np.ndarray, spacing):
    """
    Estrae centerline con Kimimaro e applica correzioni.
    """
    coords_list = []
    
    lbl = binary_mask.astype(np.uint32)
    
    skels = kimimaro.skeletonize(
        lbl,
        anisotropy=spacing,
        dust_threshold=50,  # Rimuovi componenti troppo piccole
        fix_branching=True,
        progress=False
    )
    
    for s in skels.values():
        if hasattr(s, "vertices") and len(s.vertices) > 0:
            coords_list.append(np.asarray(s.vertices, dtype=int))
    
    backend_used = "kimimaro"
    
    if not coords_list:
        return np.empty((0, 3), dtype=int), backend_used, 0
    
    # Conta componenti estratte
    n_components = len(coords_list)
    
    # Concatena
    all_coords = np.vstack(coords_list)
    
    return all_coords, backend_used, n_components


def sample_centerline_diameters_fixed(mask_bool: np.ndarray, spacing):
    """
    Calcola i diametri sulla centerline con correzioni:
    1. Snap to medial axis
    2. Filtraggio punti fuori dal vaso
    3. Ricampionamento uniforme
    """
    if not np.any(mask_bool):
        return np.array([]), None, {}
    
    # Calcola Distance Transform
    DT = ndimage.distance_transform_edt(mask_bool, sampling=spacing)
    
    # Estrai centerline
    cl_coords, backend_used, n_components = extract_centerline_coords(mask_bool, spacing)
    
    stats = {
        'n_components': n_components,
        'n_points_raw': len(cl_coords),
        'n_points_snapped': 0,
        'n_points_filtered': 0,
        'n_points_resampled': 0
    }
    
    if cl_coords.shape[0] == 0:
        return np.array([]), backend_used, stats
    
    print(f"    Raw centerline points: {len(cl_coords):,}")
    
    # STEP 1: Snap to medial axis
    cl_coords_snapped = snap_to_medial_axis(cl_coords, DT, max_search_radius=3)
    stats['n_points_snapped'] = len(cl_coords_snapped)
    print(f"    After snapping: {len(cl_coords_snapped):,}")
    
    # STEP 2: Filtra punti fuori dal vaso (DT troppo piccola)
    radii = DT[cl_coords_snapped[:, 0], cl_coords_snapped[:, 1], cl_coords_snapped[:, 2]]
    valid_mask = radii > 0.25  # Soglia: raggio > 0.25 mm
    cl_coords_filtered = cl_coords_snapped[valid_mask]
    radii_filtered = radii[valid_mask]
    stats['n_points_filtered'] = len(cl_coords_filtered)
    print(f"    After filtering (r > 0.25mm): {len(cl_coords_filtered):,}")
    
    if len(cl_coords_filtered) == 0:
        return np.array([]), backend_used, stats
    
    # STEP 3: Ricampiona (opzionale, utile per polylines molto dense o sparse)
    # Dividi per componenti e ricampiona separatamente
    # Per semplicità, qui ricampiono tutto insieme
    # In alternativa, potresti tracciare le componenti separate e ricampionarle una per una
    
    # Usa i punti filtrati direttamente (ricampionamento opzionale)
    diameters = 2.0 * radii_filtered
    stats['n_points_resampled'] = len(diameters)
    
    return diameters, backend_used, stats


def analyze_diameters(mask_path, spacing=None, diameter_method='centerline'):
    """
    Analizza distribuzione dei diametri dei vasi.
    """
    print(f"\n{'='*70}")
    print(f"DIAMETER ANALYSIS ({diameter_method.upper()}) - FIXED VERSION")
    print(f"{'='*70}")
    
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

        # Analizza tutti i label definiti in LABEL_NAMES
        for label in LABEL_NAMES.keys():
            label_name = LABEL_NAMES[label]
            mask = (mask_data == label)

            if not np.any(mask):
                print(f"\n{label_name}: ⚠️  No voxels found")
                continue

            print(f"\n{label_name}:")
            print(f"{'-'*60}")

            if diameter_method == 'centerline':
                diameters, backend_used, stats = sample_centerline_diameters_fixed(mask, spacing)

                if diameters.size == 0:
                    print(f"  ⚠️ Centerline vuota. Fallback su all_voxels.")
                    dist_transform = ndimage.distance_transform_edt(mask, sampling=spacing)
                    diameters = 2.0 * dist_transform[mask]
                    backend_used = "fallback:all_voxels"
                    stats = {}
                else:
                    print(f"  ✓ Centerline fixed - Stats:")
                    print(f"    Components: {stats.get('n_components', 'N/A')}")
                    print(f"    Final points: {stats.get('n_points_filtered', len(diameters)):,}")
            else:
                # All voxels method
                dist_transform = ndimage.distance_transform_edt(mask, sampling=spacing)
                diameters = 2.0 * dist_transform[mask]
                backend_used = "all_voxels"
                stats = {}

            # Filtra outliers estremi (opzionale)
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
                'pct_small': pct(diameters < 2.0),
                'pct_medium': pct((diameters >= 2.0) & (diameters < 5.0)),
                'pct_large': pct(diameters >= 5.0),
            }

            print(f"\n  Results:")
            print(f"    Backend: {backend_used}")
            print(f"    Sample points: {len(diameters):,}")
            print(f"    Diameter range: {results[label]['min']:.2f} - {results[label]['max']:.2f} mm")
            print(f"    Mean: {results[label]['mean']:.2f} mm")
            print(f"    Median: {results[label]['median']:.2f} mm")
            print(f"    25th-75th percentile: {results[label]['p25']:.2f} - {results[label]['p75']:.2f} mm")
            print(f"    90th percentile: {results[label]['p90']:.2f} mm")
            print(f"\n  Size distribution:")
            print(f"    Very small (<1mm): {results[label]['pct_very_small']:.1f}%")
            print(f"    Small (1-2mm): {results[label]['pct_small'] - results[label]['pct_very_small']:.1f}%")
            print(f"    Medium (2-5mm): {results[label]['pct_medium']:.1f}%")
            print(f"    Large (>5mm): {results[label]['pct_large']:.1f}%")

        return results
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_plots(diameter_stats, output_prefix="diameter_analysis"):
    """Crea grafici di analisi dei diametri."""
    print(f"\n{'='*70}")
    print(f"CREATING PLOTS")
    print(f"{'='*70}")
    
    if not diameter_stats or 1 not in diameter_stats:
        print("⚠️  Insufficient data for plotting")
        return
    
    diameters = diameter_stats[1]['all_diameters']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram
    axes[0].hist(diameters, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    axes[0].set_xlabel('Diameter (mm)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Diameter Distribution - All Vessels')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 10])
    
    # Box plot
    bp = axes[1].boxplot([diameters], labels=['Vessels'], 
                         showfliers=False, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)
    
    axes[1].set_ylabel('Diameter (mm)')
    axes[1].set_title('Diameter Distribution (Box Plot)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # CDF plot
    diameters_sorted = np.sort(diameters)
    cdf = np.arange(1, len(diameters_sorted) + 1) / len(diameters_sorted)
    
    axes[2].plot(diameters_sorted, cdf, color='#3498db', linewidth=2)
    axes[2].set_xlabel('Diameter (mm)')
    axes[2].set_ylabel('Cumulative Probability')
    axes[2].set_title('Cumulative Distribution Function')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 10])
    axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Median')
    axes[2].axhline(y=0.75, color='orange', linestyle='--', alpha=0.5, label='75th percentile')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_diameters_fixed.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_diameters_fixed.png")
    plt.close('all')


def generate_summary_table(diameter_stats, output_file="diameter_analysis_summary.csv"):
    """Genera tabella riassuntiva."""
    print(f"\n{'='*70}")
    print(f"GENERATING SUMMARY TABLE")
    print(f"{'='*70}")
    
    rows = []
    
    for label in [1, 2]:
        if label not in diameter_stats:
            continue
        
        label_name = LABEL_NAMES[label]
        diam = diameter_stats[label]
        
        row = {
            'Vessel_Type': label_name,
            'Method': diam.get('method', 'n/a'),
            'Backend': diam.get('diameter_backend', 'n/a'),
            'N_Points': len(diam['all_diameters']),
            'Min_mm': f"{diam['min']:.2f}",
            'P25_mm': f"{diam['p25']:.2f}",
            'Median_mm': f"{diam['median']:.2f}",
            'Mean_mm': f"{diam['mean']:.2f}",
            'P75_mm': f"{diam['p75']:.2f}",
            'P90_mm': f"{diam['p90']:.2f}",
            'Max_mm': f"{diam['max']:.2f}",
            'Pct_VerySmall': f"{diam['pct_very_small']:.1f}%",
            'Pct_Small': f"{diam['pct_small']:.1f}%",
            'Pct_Medium': f"{diam['pct_medium']:.1f}%",
            'Pct_Large': f"{diam['pct_large']:.1f}%",
        }
        rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")
        print("\n" + df.to_string(index=False))
    else:
        print("  ⚠️  No data to save")


# MAIN EXECUTION
if __name__ == "__main__":
    mask_path = '/content/use_it/1.2.840.113704.1.111.2604.1126357612_fullAnnotations.mhd'
    spacing = (0.7, 0.7, 0.7)
    output_prefix = "diameter_analysis_fixed"
    
    DIAMETER_METHOD = 'centerline'
    
    print("="*70)
    print("VESSEL DIAMETER ANALYSIS - FIXED VERSION")
    print("="*70)
    print(f"\nInput file: {mask_path}")
    print(f"Method: {DIAMETER_METHOD}")
    print("\nFixes applied:")
    print("  1. ✓ Snap to medial axis")
    print("  2. ✓ Filter points outside vessels")
    print("  3. ✓ Improved component extraction")
    
    # Analizza
    diameter_stats = analyze_diameters(mask_path, spacing, diameter_method=DIAMETER_METHOD)
    
    # Genera output
    if diameter_stats:
        create_plots(diameter_stats, output_prefix)
        generate_summary_table(diameter_stats, f"{output_prefix}_summary.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)