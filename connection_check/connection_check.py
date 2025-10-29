"""
Quick analysis script per capire le distanze tra componenti
e suggerire parametri realistici.
Versione per analisi di intera cartella con statistiche globali.
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

def analyze_component_distances(vessel_path, spacing, sample_size=1000):
    """
    Analizza le distanze tra componenti disconnesse.
    Restituisce le statistiche per analisi aggregata.
    """
    print(f"\nAnalyzing: {os.path.basename(vessel_path)}")
    
    try:
        img = sitk.ReadImage(vessel_path)
        mask = sitk.GetArrayFromImage(img).astype(bool)
        
        # Label components
        labeled, num_components = ndimage.label(mask)
        
        if num_components <= 1:
            print("  Only one component, no distances to compute")
            return None
        
        print(f"  Found {num_components} components")
        
        # Trova la componente principale (più grande)
        component_sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
        main_component_label = np.argmax(component_sizes) + 1
        main_component_size = component_sizes[main_component_label - 1]
        
        # Analizza distanze dei frammenti dalla componente principale
        main_mask = (labeled == main_component_label)
        main_coords = np.array(np.where(main_mask)).T
        main_coords_mm = main_coords * np.array(spacing)
        
        # Build KD-tree
        tree = cKDTree(main_coords_mm)
        
        distances_by_size = {}
        all_fragment_data = []
        
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
            
            fragment_data = {
                'size': fragment_size,
                'min_dist': min_distance,
                'mean_dist': mean_distance,
                'category': category
            }
            distances_by_size[category].append(fragment_data)
            all_fragment_data.append(fragment_data)
        
        # Statistiche per questo file
        if all_fragment_data:
            all_distances = [d['min_dist'] for d in all_fragment_data]
            stats = {
                'filename': os.path.basename(vessel_path),
                'num_components': num_components,
                'num_fragments': len(all_fragment_data),
                'main_component_size': main_component_size,
                'distance_stats': {
                    'p50': np.percentile(all_distances, 50),
                    'p75': np.percentile(all_distances, 75),
                    'p90': np.percentile(all_distances, 90),
                    'mean': np.mean(all_distances),
                    'std': np.std(all_distances),
                    'min': np.min(all_distances),
                    'max': np.max(all_distances)
                },
                'fragments_by_category': {cat: len(data) for cat, data in distances_by_size.items()},
                'all_fragment_data': all_fragment_data
            }
            
            print(f"  Fragments: {len(all_fragment_data)}")
            print(f"  Distance percentiles - 50th: {stats['distance_stats']['p50']:.2f}mm, "
                  f"75th: {stats['distance_stats']['p75']:.2f}mm, "
                  f"90th: {stats['distance_stats']['p90']:.2f}mm")
            
            return stats
        else:
            return None
            
    except Exception as e:
        print(f"  ERROR processing {vessel_path}: {e}")
        return None

def analyze_vessel_diameters(vessel_path, centerline_path, spacing):
    """
    Analizza i diametri dei vasi usando le centerlines.
    Restituisce le statistiche per analisi aggregata.
    """
    print(f"  Analyzing diameters...")
    
    try:
        vessel_img = sitk.ReadImage(vessel_path)
        vessel_mask = sitk.GetArrayFromImage(vessel_img).astype(bool)
        
        # Distance transform dalla maschera dei vasi
        dist_transform = ndimage.distance_transform_edt(vessel_mask, sampling=spacing)
        
        if centerline_path and os.path.exists(centerline_path):
            centerline_img = sitk.ReadImage(centerline_path)
            centerline_mask = sitk.GetArrayFromImage(centerline_img).astype(bool)
            
            # Estrai diametri lungo le centerlines
            diameters = dist_transform[centerline_mask] * 2  # radius -> diameter
            
            stats = {
                'num_points': centerline_mask.sum(),
                'min_diameter': diameters.min(),
                'max_diameter': diameters.max(),
                'mean_diameter': diameters.mean(),
                'median_diameter': np.median(diameters),
                'p25_diameter': np.percentile(diameters, 25),
                'p75_diameter': np.percentile(diameters, 75),
                'p90_diameter': np.percentile(diameters, 90),
                'all_diameters': diameters
            }
            
            print(f"    Diameters: {stats['num_points']} points, "
                  f"median: {stats['median_diameter']:.2f}mm, "
                  f"75th: {stats['p75_diameter']:.2f}mm")
            
            return stats
        else:
            print(f"    Centerline file not found: {centerline_path}")
            return None
            
    except Exception as e:
        print(f"    ERROR analyzing diameters: {e}")
        return None

def create_global_plots(global_stats):
    """
    Crea due grafici globali con le informazioni sul dataset.
    """
    print("\n" + "="*70)
    print("CREATING GLOBAL PLOTS")
    print("="*70)
    
    # Estrai dati per plotting
    distances_all = []
    diameters_all = []
    fragments_per_case = []
    connectivity_ratios = []
    
    for stats in global_stats:
        if 'distance_stats' in stats:
            distances_all.extend([d['min_dist'] for d in stats['all_fragment_data']])
            fragments_per_case.append(stats['num_fragments'])
            # Rapporto di connettività: 1 / (1 + numero di frammenti)
            connectivity_ratio = 1 / (1 + stats['num_fragments'])
            connectivity_ratios.append(connectivity_ratio)
        
        if 'diameter_stats' in stats:
            diameters_all.extend(stats['diameter_stats']['all_diameters'])
    
    # Plot 1: Distribuzione globale delle distanze e connettività
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    if distances_all:
        # Distribuzione distanze
        axes[0, 0].hist(distances_all, bins=50, edgecolor='black', alpha=0.7, density=True)
        p50 = np.percentile(distances_all, 50)
        p75 = np.percentile(distances_all, 75)
        p90 = np.percentile(distances_all, 90)
        
        axes[0, 0].axvline(p50, color='green', linestyle='--', label=f'50th percentile ({p50:.1f}mm)')
        axes[0, 0].axvline(p75, color='orange', linestyle='--', label=f'75th percentile ({p75:.1f}mm)')
        axes[0, 0].axvline(p90, color='red', linestyle='--', label=f'90th percentile ({p90:.1f}mm)')
        axes[0, 0].set_xlabel('Distance to main component (mm)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Global Fragment Distance Distribution\n(All Cases Combined)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribuzione numero di frammenti per caso
        axes[0, 1].hist(fragments_per_case, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Number of Fragments per Case')
        axes[0, 1].set_ylabel('Number of Cases')
        axes[0, 1].set_title('Fragment Count Distribution\nper Case')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Boxplot delle distanze per caso
        distance_data_by_case = []
        case_names = []
        for stats in global_stats:
            if 'distance_stats' in stats and stats['all_fragment_data']:
                case_distances = [d['min_dist'] for d in stats['all_fragment_data']]
                distance_data_by_case.append(case_distances)
                case_names.append(stats['filename'][:15] + "...")
        
        if distance_data_by_case:
            axes[1, 0].boxplot(distance_data_by_case, labels=case_names, showfliers=False)
            axes[1, 0].set_xticklabels(case_names, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Distance (mm)')
            axes[1, 0].set_title('Distance Distribution by Case')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Connettività per caso
        if connectivity_ratios:
            case_indices = range(len(connectivity_ratios))
            case_labels = [f"Case {i+1}" for i in case_indices]
            
            axes[1, 1].bar(case_indices, connectivity_ratios, alpha=0.7)
            axes[1, 1].set_xlabel('Case')
            axes[1, 1].set_ylabel('Connectivity Ratio\n(1 / (1 + fragments))')
            axes[1, 1].set_title('Connectivity by Case\n(Higher = Better Connected)')
            axes[1, 1].set_xticks(case_indices)
            axes[1, 1].set_xticklabels(case_labels, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Aggiungi valori sulle barre
            for i, v in enumerate(connectivity_ratios):
                axes[1, 1].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('global_connectivity_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Global connectivity analysis saved to: global_connectivity_analysis.png")
    
    # Plot 2: Analisi globale dei diametri
    if diameters_all:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribuzione diametri
        axes[0].hist(diameters_all, bins=50, edgecolor='black', alpha=0.7, density=True)
        p75_diam = np.percentile(diameters_all, 75)
        p90_diam = np.percentile(diameters_all, 90)
        
        axes[0].axvline(p75_diam, color='orange', linestyle='--', 
                       label=f'75th percentile ({p75_diam:.1f}mm)')
        axes[0].axvline(p90_diam, color='red', linestyle='--', 
                       label=f'90th percentile ({p90_diam:.1f}mm)')
        axes[0].set_xlabel('Vessel Diameter (mm)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Global Vessel Diameter Distribution\n(All Cases Combined)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Boxplot diametri per caso
        diameter_data_by_case = []
        diam_case_names = []
        for stats in global_stats:
            if 'diameter_stats' in stats:
                diameter_data_by_case.append(stats['diameter_stats']['all_diameters'])
                diam_case_names.append(stats['filename'][:15] + "...")
        
        if diameter_data_by_case:
            axes[1].boxplot(diameter_data_by_case, labels=diam_case_names, showfliers=False)
            axes[1].set_xticklabels(diam_case_names, rotation=45, ha='right')
            axes[1].set_ylabel('Diameter (mm)')
            axes[1].set_title('Diameter Distribution by Case')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('global_diameter_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Global diameter analysis saved to: global_diameter_analysis.png")
    
    plt.close('all')

def print_global_summary(global_stats):
    """
    Stampa un riepilogo globale delle statistiche.
    """
    print("\n" + "="*70)
    print("GLOBAL DATASET SUMMARY")
    print("="*70)
    
    total_cases = len(global_stats)
    cases_with_fragments = 0
    total_fragments = 0
    all_distances = []
    all_diameters = []
    
    for stats in global_stats:
        if 'distance_stats' in stats:
            cases_with_fragments += 1
            total_fragments += stats['num_fragments']
            all_distances.extend([d['min_dist'] for d in stats['all_fragment_data']])
        
        if 'diameter_stats' in stats:
            all_diameters.extend(stats['diameter_stats']['all_diameters'])
    
    print(f"\nDataset Overview:")
    print(f"  Total cases analyzed: {total_cases}")
    print(f"  Cases with fragments: {cases_with_fragments}")
    print(f"  Total fragments: {total_fragments}")
    
    if all_distances:
        print(f"\nGlobal Distance Statistics:")
        print(f"  Min distance: {np.min(all_distances):.2f} mm")
        print(f"  Max distance: {np.max(all_distances):.2f} mm")
        print(f"  Mean distance: {np.mean(all_distances):.2f} mm")
        print(f"  Median distance: {np.median(all_distances):.2f} mm")
        print(f"  50th percentile: {np.percentile(all_distances, 50):.2f} mm")
        print(f"  75th percentile: {np.percentile(all_distances, 75):.2f} mm")
        print(f"  90th percentile: {np.percentile(all_distances, 90):.2f} mm")
        
        print(f"\nSuggested Global Parameters:")
        print(f"  Conservative: max_connection_distance_mm = {np.percentile(all_distances, 50):.1f}")
        print(f"  Moderate:     max_connection_distance_mm = {np.percentile(all_distances, 75):.1f}")
        print(f"  Aggressive:   max_connection_distance_mm = {np.percentile(all_distances, 90):.1f}")
    
    if all_diameters:
        print(f"\nGlobal Diameter Statistics:")
        print(f"  Min diameter: {np.min(all_diameters):.2f} mm")
        print(f"  Max diameter: {np.max(all_diameters):.2f} mm")
        print(f"  Mean diameter: {np.mean(all_diameters):.2f} mm")
        print(f"  Median diameter: {np.median(all_diameters):.2f} mm")
        print(f"  75th percentile: {np.percentile(all_diameters, 75):.2f} mm")
        print(f"  90th percentile: {np.percentile(all_diameters, 90):.2f} mm")
        
        print(f"\nSuggested Large Vessel Thresholds:")
        print(f"  Conservative (top 10%): large_vessel_threshold_mm = {np.percentile(all_diameters, 90):.1f}")
        print(f"  Moderate (top 25%):     large_vessel_threshold_mm = {np.percentile(all_diameters, 75):.1f}")

"""
Funzione principale per analizzare l'intera cartella.
"""
# Configurazione paths
base_dir = '/content/vesselsegmentation/vessels_cleaned'
spacing = (0.7, 0.7, 0.7)

# Trova tutti i file vessels cleaned
vessel_files = glob.glob(os.path.join(base_dir, '*_cleaned.nii.gz'))

print(f"Found {len(vessel_files)} vessel files to analyze")

global_stats = []

for vessel_path in vessel_files:
    # Costruisci il path della centerline corrispondente
    base_name = vessel_path.replace('_cleaned.nii.gz', '')
    centerline_path = base_name + '_centerlines.nii.gz'
    
    # Analizza distanze
    distance_stats = analyze_component_distances(vessel_path, spacing)
    
    # Analizza diametri
    diameter_stats = None
    if os.path.exists(centerline_path):
        diameter_stats = analyze_vessel_diameters(vessel_path, centerline_path, spacing)
    
    # Combina le statistiche
    if distance_stats:
        if diameter_stats:
            distance_stats['diameter_stats'] = diameter_stats
        global_stats.append(distance_stats)

# Genera report globale
if global_stats:
    print_global_summary(global_stats)
    create_global_plots(global_stats)
    print(f"\nAnalysis complete! Processed {len(global_stats)} cases.")
else:
    print("No valid cases found for analysis.")
