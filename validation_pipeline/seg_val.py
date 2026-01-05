"""
VALIDATION STEP 1: SEGMENTATION QUALITY
========================================

Valida la qualità delle maschere di segmentazione (airways e vessels)
confrontando metriche morfologiche con valori di riferimento dalla letteratura.

CRITERI DI VALIDAZIONE:
- Volume totale
- Lunghezza scheletro
- Componenti connesse
- Ratio gap-filling (solo airways)
- Compattezza strutturale
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label, generate_binary_structure
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# REFERENCE VALUES FROM LITERATURE
# ============================================================================

AIRWAY_REFERENCE = {
    "volume_mm3": {
        "min": 15_000,      # Molto piccolo
        "expected_min": 20_000,  # Range normale
        "expected_max": 80_000,
        "max": 100_000,     # Molto grande
        "source": "Kirby et al. 2012, COPD patients"
    },
    "skeleton_length_mm": {
        "min": 200,
        "expected_min": 300,
        "expected_max": 2_000,
        "max": 3_000,
        "source": "Montaudon et al. 2007"
    },
    "max_components": {
        "optimal": 1,
        "acceptable": 3,
        "warning": 5,
        "critical": 10,
        "source": "Quality control metrics"
    },
    "gap_fill_ratio": {
        "normal": 0.10,      # 10% aggiunto è normale
        "high": 0.20,        # 20% indica gap significativi
        "very_high": 0.40,   # 40% è problematico
        "source": "Empirical thresholds"
    }
}

VESSEL_REFERENCE = {
    "volume_mm3": {
        "min": 40_000,
        "expected_min": 50_000,
        "expected_max": 200_000,
        "max": 300_000,
        "source": "Estpar et al. 2013, healthy subjects"
    },
    "skeleton_length_mm": {
        "min": 800,
        "expected_min": 1_000,
        "expected_max": 5_000,
        "max": 8_000,
        "source": "Pulmonary vascular morphometry studies"
    },
    "max_components": {
        "optimal": 1,
        "acceptable": 20,
        "warning": 50,
        "critical": 100,
        "source": "Quality control (vessels più frammentati)"
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_mask(path):
    """Carica maschera e spacing"""
    img = sitk.ReadImage(path)
    mask = sitk.GetArrayFromImage(img) > 0
    spacing = img.GetSpacing()  # (x, y, z)
    return mask, spacing


def volume_mm3(mask, spacing):
    """Calcola volume in mm³"""
    voxel_volume = np.prod(spacing)
    return float(np.sum(mask) * voxel_volume)


def skeleton_length_mm(mask, spacing, use_3d=True):
    """
    Calcola lunghezza scheletro in mm
    
    IMPROVEMENT: Usa skeletonize con method='3d' per vessels (più accurato)
    """
    if use_3d:
        skeleton = skeletonize(mask, method='lee')
    else:
        skeleton = skeletonize(mask, method='lee')
    
    coords = np.argwhere(skeleton)
    
    if len(coords) < 2:
        return 0.0
    
    # KD-Tree per trovare vicini
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=1.5)
    
    length = 0.0
    spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
    
    for i, j in pairs:
        delta = (coords[i] - coords[j]) * spacing_zyx
        length += np.linalg.norm(delta)
    
    return float(length)


def count_components(mask):
    """Conta componenti connesse con 26-connectivity"""
    structure = generate_binary_structure(3, 3)
    _, num = label(mask, structure)
    return int(num)


def component_size_distribution(mask):
    """
    NUOVO: Analizza distribuzione dimensioni componenti
    Utile per capire quanto è frammentata la segmentazione
    """
    structure = generate_binary_structure(3, 3)
    labeled, num = label(mask, structure)
    
    if num == 0:
        return []
    
    sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
    sizes.sort(reverse=True)
    
    return sizes


def compactness_score(mask, spacing):
    """
    NUOVO: Misura quanto è "compatta" la struttura
    
    Compactness = Volume / (Surface_area^(3/2))
    Valori più alti = più compatta (sfera = massimo)
    """
    from scipy.ndimage import binary_erosion
    
    volume = volume_mm3(mask, spacing)
    
    # Stima superficie tramite erosione
    eroded = binary_erosion(mask)
    surface_voxels = np.sum(mask & ~eroded)
    voxel_area = spacing[0] * spacing[1]  # approssimazione
    surface_area = surface_voxels * voxel_area
    
    if surface_area == 0:
        return 0.0
    
    compactness = volume / (surface_area ** 1.5)
    
    return float(compactness)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_airway_case(refined_path, gap_filled_path=None):
    """
    Validazione completa caso airways
    
    IMPROVEMENTS:
    - Validazione con/senza gap-filling
    - Severity scoring
    - Component analysis
    """
    refined, spacing = load_mask(refined_path)
    
    # Base metrics
    vol_ref = volume_mm3(refined, spacing)
    skel_len = skeleton_length_mm(refined, spacing, use_3d=False)
    components = count_components(refined)
    comp_sizes = component_size_distribution(refined)
    compactness = compactness_score(refined, spacing)
    
    # Gap-filled metrics (se disponibile)
    if gap_filled_path and os.path.exists(gap_filled_path):
        gap, _ = load_mask(gap_filled_path)
        vol_gap = volume_mm3(gap, spacing)
        gap_ratio = (vol_gap - vol_ref) / vol_ref if vol_ref > 0 else np.nan
    else:
        vol_gap = vol_ref
        gap_ratio = 0.0
    
    # ========================================
    # HARD CRITERIA (BLOCKERS)
    # ========================================
    ref = AIRWAY_REFERENCE
    
    volume_ok = vol_gap >= ref["volume_mm3"]["min"]
    skeleton_ok = skel_len >= ref["skeleton_length_mm"]["min"]
    components_ok = components <= ref["max_components"]["critical"]
    
    pipeline_usable = volume_ok and skeleton_ok and components_ok
    
    # ========================================
    # QUALITY SCORING (0-100)
    # ========================================
    quality_scores = []
    
    # Volume score
    vol_score = 0
    if vol_gap < ref["volume_mm3"]["expected_min"]:
        vol_score = 50 * (vol_gap / ref["volume_mm3"]["expected_min"])
    elif vol_gap <= ref["volume_mm3"]["expected_max"]:
        vol_score = 100
    else:
        excess = (vol_gap - ref["volume_mm3"]["expected_max"]) / ref["volume_mm3"]["expected_max"]
        vol_score = max(50, 100 - 50 * excess)
    
    quality_scores.append(("volume", vol_score))
    
    # Skeleton score
    skel_score = 0
    if skel_len < ref["skeleton_length_mm"]["expected_min"]:
        skel_score = 50 * (skel_len / ref["skeleton_length_mm"]["expected_min"])
    elif skel_len <= ref["skeleton_length_mm"]["expected_max"]:
        skel_score = 100
    else:
        excess = (skel_len - ref["skeleton_length_mm"]["expected_max"]) / ref["skeleton_length_mm"]["expected_max"]
        skel_score = max(50, 100 - 50 * excess)
    
    quality_scores.append(("skeleton", skel_score))
    
    # Component score
    comp_score = 0
    if components == ref["max_components"]["optimal"]:
        comp_score = 100
    elif components <= ref["max_components"]["acceptable"]:
        comp_score = 80
    elif components <= ref["max_components"]["warning"]:
        comp_score = 60
    else:
        comp_score = max(0, 40 - 10 * (components - ref["max_components"]["warning"]))
    
    quality_scores.append(("components", comp_score))
    
    # Gap-fill score
    gap_score = 100
    if gap_ratio > ref["gap_fill_ratio"]["very_high"]:
        gap_score = 20
    elif gap_ratio > ref["gap_fill_ratio"]["high"]:
        gap_score = 60
    elif gap_ratio > ref["gap_fill_ratio"]["normal"]:
        gap_score = 80
    
    quality_scores.append(("gap_filling", gap_score))
    
    overall_quality = np.mean([s for _, s in quality_scores])
    
    # ========================================
    # GAP-FILL STATUS
    # ========================================
    if gap_ratio > ref["gap_fill_ratio"]["very_high"]:
        gap_status = "VERY_HIGH"
    elif gap_ratio > ref["gap_fill_ratio"]["high"]:
        gap_status = "HIGH"
    elif gap_ratio > ref["gap_fill_ratio"]["normal"]:
        gap_status = "MODERATE"
    else:
        gap_status = "NORMAL"
    
    # ========================================
    # SEVERITY ASSESSMENT
    # ========================================
    severity_flags = []
    
    if not volume_ok:
        severity_flags.append("CRITICAL_VOLUME_TOO_LOW")
    if not skeleton_ok:
        severity_flags.append("CRITICAL_SKELETON_TOO_SHORT")
    if not components_ok:
        severity_flags.append("CRITICAL_TOO_MANY_COMPONENTS")
    
    if vol_gap < ref["volume_mm3"]["expected_min"]:
        severity_flags.append("WARNING_SMALL_VOLUME")
    if components > ref["max_components"]["acceptable"]:
        severity_flags.append("WARNING_FRAGMENTED")
    if gap_ratio > ref["gap_fill_ratio"]["high"]:
        severity_flags.append("WARNING_HIGH_GAP_FILLING")
    
    # ========================================
    # OUTPUT
    # ========================================
    return {
        # Raw metrics
        "volume_refined_mm3": vol_ref,
        "volume_gap_filled_mm3": vol_gap,
        "gap_fill_ratio": gap_ratio,
        "gap_fill_status": gap_status,
        "skeleton_length_mm": skel_len,
        "connected_components": components,
        "main_component_size": comp_sizes[0] if comp_sizes else 0,
        "component_size_ratio": comp_sizes[1] / comp_sizes[0] if len(comp_sizes) > 1 else 0.0,
        "compactness": compactness,
        
        # Hard flags
        "volume_ok": volume_ok,
        "skeleton_ok": skeleton_ok,
        "components_ok": components_ok,
        "PIPELINE_USABLE": pipeline_usable,
        
        # Quality scores
        "quality_volume": vol_score,
        "quality_skeleton": skel_score,
        "quality_components": comp_score,
        "quality_gap_filling": gap_score,
        "quality_overall": overall_quality,
        
        # Severity
        "severity_flags": ";".join(severity_flags) if severity_flags else "NONE"
    }


def validate_vessel_case(vessel_mask_path):
    """
    Validazione completa caso vessels
    
    IMPROVEMENTS:
    - Component distribution analysis
    - Compactness scoring
    """
    mask, spacing = load_mask(vessel_mask_path)
    
    vol = volume_mm3(mask, spacing)
    skel_len = skeleton_length_mm(mask, spacing, use_3d=True)  # 3D per vessels
    components = count_components(mask)
    comp_sizes = component_size_distribution(mask)
    compactness = compactness_score(mask, spacing)
    
    skel_voxels = np.sum(skeletonize(mask, method='lee'))
    
    # ========================================
    # HARD CRITERIA
    # ========================================
    ref = VESSEL_REFERENCE
    
    volume_ok = vol >= ref["volume_mm3"]["min"]
    skeleton_ok = skel_len >= ref["skeleton_length_mm"]["min"]
    skeleton_size_ok = skel_voxels >= 1_000
    components_ok = components <= ref["max_components"]["critical"]
    
    pipeline_usable = volume_ok and skeleton_ok and skeleton_size_ok and components_ok
    
    # ========================================
    # QUALITY SCORING
    # ========================================
    quality_scores = []
    
    # Volume score
    vol_score = 0
    if vol < ref["volume_mm3"]["expected_min"]:
        vol_score = 50 * (vol / ref["volume_mm3"]["expected_min"])
    elif vol <= ref["volume_mm3"]["expected_max"]:
        vol_score = 100
    else:
        excess = (vol - ref["volume_mm3"]["expected_max"]) / ref["volume_mm3"]["expected_max"]
        vol_score = max(50, 100 - 50 * excess)
    
    quality_scores.append(("volume", vol_score))
    
    # Skeleton score
    skel_score = 0
    if skel_len < ref["skeleton_length_mm"]["expected_min"]:
        skel_score = 50 * (skel_len / ref["skeleton_length_mm"]["expected_min"])
    elif skel_len <= ref["skeleton_length_mm"]["expected_max"]:
        skel_score = 100
    else:
        excess = (skel_len - ref["skeleton_length_mm"]["expected_max"]) / ref["skeleton_length_mm"]["expected_max"]
        skel_score = max(50, 100 - 50 * excess)
    
    quality_scores.append(("skeleton", skel_score))
    
    # Component score (vessels can have more components)
    comp_score = 0
    if components <= ref["max_components"]["optimal"]:
        comp_score = 100
    elif components <= ref["max_components"]["acceptable"]:
        comp_score = 80
    elif components <= ref["max_components"]["warning"]:
        comp_score = 60
    else:
        comp_score = max(0, 40 - 5 * (components - ref["max_components"]["warning"]))
    
    quality_scores.append(("components", comp_score))
    
    overall_quality = np.mean([s for _, s in quality_scores])
    
    # ========================================
    # SEVERITY
    # ========================================
    severity_flags = []
    
    if not volume_ok:
        severity_flags.append("CRITICAL_VOLUME_TOO_LOW")
    if not skeleton_ok:
        severity_flags.append("CRITICAL_SKELETON_TOO_SHORT")
    if not components_ok:
        severity_flags.append("CRITICAL_TOO_MANY_COMPONENTS")
    
    if vol < ref["volume_mm3"]["expected_min"]:
        severity_flags.append("WARNING_SMALL_VOLUME")
    if components > ref["max_components"]["acceptable"]:
        severity_flags.append("WARNING_FRAGMENTED")
    
    # ========================================
    # OUTPUT
    # ========================================
    return {
        # Raw metrics
        "volume_mm3": vol,
        "skeleton_length_mm": skel_len,
        "skeleton_voxels": skel_voxels,
        "connected_components": components,
        "main_component_size": comp_sizes[0] if comp_sizes else 0,
        "component_size_ratio": comp_sizes[1] / comp_sizes[0] if len(comp_sizes) > 1 else 0.0,
        "compactness": compactness,
        
        # Hard flags
        "volume_ok": volume_ok,
        "skeleton_ok": skeleton_ok,
        "skeleton_size_ok": skeleton_size_ok,
        "components_ok": components_ok,
        "PIPELINE_USABLE": pipeline_usable,
        
        # Quality scores
        "quality_volume": vol_score,
        "quality_skeleton": skel_score,
        "quality_components": comp_score,
        "quality_overall": overall_quality,
        
        # Severity
        "severity_flags": ";".join(severity_flags) if severity_flags else "NONE"
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def validate_airways_batch(data_root, output_csv):
    """Valida batch airways"""
    print("="*80)
    print("AIRWAY SEGMENTATION VALIDATION")
    print("="*80)
    
    results = []
    
    for case_id in sorted(os.listdir(data_root)):
        step1 = os.path.join(data_root, case_id, "step1_segmentation")
        
        gap_filled = os.path.join(step1, f"{case_id}_airway_refined_enhanced.nii_gap_filled.nii.gz")
        
        if not os.path.exists(gap_filled):
            print(f"[SKIP] {case_id} - missing gap_filled mask")
            continue
        
        print(f"[VALIDATING] {case_id}")
        
        try:
            row = validate_airway_case(gap_filled, None)
            row["case_id"] = case_id
            results.append(row)
            
            # Print immediate feedback
            if row["PIPELINE_USABLE"]:
                print(f"  ✓ USABLE - Quality: {row['quality_overall']:.1f}/100")
            else:
                print(f"  ✗ UNUSABLE - {row['severity_flags']}")
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print_summary(df, "AIRWAYS")
    
    return df


def validate_vessels_batch(data_root, output_csv):
    """Valida batch vessels"""
    print("="*80)
    print("VESSEL SEGMENTATION VALIDATION")
    print("="*80)
    
    results = []
    
    for case_id in sorted(os.listdir(data_root)):
        step1 = os.path.join(data_root, case_id, "step1_segmentation")
        
        vessel_mask = os.path.join(step1, f"{case_id}_vessels_raw_cleaned.nii.gz")
        
        if not os.path.exists(vessel_mask):
            print(f"[SKIP] {case_id} - vessel mask not found")
            continue
        
        print(f"[VALIDATING] {case_id}")
        
        try:
            row = validate_vessel_case(vessel_mask)
            row["case_id"] = case_id
            results.append(row)
            
            if row["PIPELINE_USABLE"]:
                print(f"  ✓ USABLE - Quality: {row['quality_overall']:.1f}/100")
            else:
                print(f"  ✗ UNUSABLE - {row['severity_flags']}")
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print_summary(df, "VESSELS")
    
    return df


def print_summary(df, structure_type):
    """Print validation summary"""
    print("\n" + "="*80)
    print(f"{structure_type} VALIDATION SUMMARY")
    print("="*80)
    
    if len(df) == 0:
        print("No valid cases found.")
        return
    
    print(f"Total cases: {len(df)}")
    print(f"Usable: {df['PIPELINE_USABLE'].sum()} ({df['PIPELINE_USABLE'].sum()/len(df)*100:.1f}%)")
    print(f"Unusable: {(~df['PIPELINE_USABLE']).sum()}")
    
    print(f"\nQuality Distribution:")
    print(f"  Excellent (>90): {(df['quality_overall'] > 90).sum()}")
    print(f"  Good (70-90): {((df['quality_overall'] >= 70) & (df['quality_overall'] <= 90)).sum()}")
    print(f"  Fair (50-70): {((df['quality_overall'] >= 50) & (df['quality_overall'] < 70)).sum()}")
    print(f"  Poor (<50): {(df['quality_overall'] < 50).sum()}")
    
    print(f"\nMean Quality: {df['quality_overall'].mean():.1f}/100")
    print(f"Median Quality: {df['quality_overall'].median():.1f}/100")
    
    # Severity flags frequency
    all_flags = []
    for flags_str in df['severity_flags']:
        if flags_str != "NONE":
            all_flags.extend(flags_str.split(';'))
    
    if all_flags:
        print(f"\nMost Common Issues:")
        from collections import Counter
        for flag, count in Counter(all_flags).most_common(5):
            print(f"  {flag}: {count} cases")


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_validation_plots(df, structure_type, output_dir):
    """Crea visualizzazioni validation"""
    if len(df) == 0:
        print(f"[SKIP] No valid cases for {structure_type} plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Quality distribution
    axes[0, 0].hist(df['quality_overall'], bins=20, edgecolor='black', color='steelblue', alpha=0.7)
    axes[0, 0].axvline(df['quality_overall'].mean(), color='r', linestyle='--', label=f'Mean: {df["quality_overall"].mean():.1f}')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'{structure_type} Quality Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Usability pie
    usable = df['PIPELINE_USABLE'].sum()
    unusable = len(df) - usable
    axes[0, 1].pie([usable, unusable], labels=['Usable', 'Unusable'], 
                   autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    axes[0, 1].set_title('Pipeline Usability')
    
    # 3. Volume distribution
    if structure_type == "AIRWAYS":
        vol_col = 'volume_gap_filled_mm3'
    else:
        vol_col = 'volume_mm3'
    
    axes[0, 2].hist(df[vol_col]/1000, bins=20, edgecolor='black', color='seagreen', alpha=0.7)
    axes[0, 2].set_xlabel('Volume (x1000 mm³)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title(f'{structure_type} Volume Distribution')
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Skeleton length
    axes[1, 0].hist(df['skeleton_length_mm'], bins=20, edgecolor='black', color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Skeleton Length (mm)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Skeleton Length Distribution')
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Components
    axes[1, 1].hist(df['connected_components'], bins=min(20, df['connected_components'].max()), 
                    edgecolor='black', color='crimson', alpha=0.7)
    axes[1, 1].set_xlabel('Number of Components')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Component Distribution')
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Quality components breakdown
    quality_cols = ['quality_volume', 'quality_skeleton', 'quality_components']
    if structure_type == "AIRWAYS":
        quality_cols.append('quality_gap_filling')
    
    quality_means = [df[col].mean() for col in quality_cols]
    labels = [col.replace('quality_', '').title() for col in quality_cols]
    
    axes[1, 2].bar(labels, quality_means, color=['steelblue', 'seagreen', 'orange', 'crimson'][:len(labels)], 
                   edgecolor='black', alpha=0.7)
    axes[1, 2].set_ylabel('Mean Quality Score')
    axes[1, 2].set_title('Quality Components Breakdown')
    axes[1, 2].set_ylim([0, 100])
    axes[1, 2].grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'{structure_type} Segmentation Validation - Step 1', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'validation_step1_{structure_type.lower()}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved validation plot: {plot_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # ========================================
    # OUTPUT DIRECTORY SETUP
    # ========================================
    OUTPUT_BASE = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\output\step1_segmentation_quality"
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # ========================================
    # AIRWAYS VALIDATION
    # ========================================
    AIRWAY_DATA_ROOT = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\output_results_with_fibrosis"
    AIRWAY_OUTPUT_CSV = os.path.join(OUTPUT_BASE, "airway_validation_step1.csv")
    AIRWAY_PLOT_DIR = OUTPUT_BASE
    
    if os.path.exists(AIRWAY_DATA_ROOT):
        df_airways = validate_airways_batch(AIRWAY_DATA_ROOT, AIRWAY_OUTPUT_CSV)
        create_validation_plots(df_airways, "AIRWAYS", AIRWAY_PLOT_DIR)
    
    # ========================================
    # VESSELS VALIDATION
    # ========================================
    VESSEL_DATA_ROOT = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\vessel_segmentation_new\vessel_output"
    VESSEL_OUTPUT_CSV = os.path.join(OUTPUT_BASE, "vessel_validation_step1.csv")
    VESSEL_PLOT_DIR = OUTPUT_BASE
    
    if os.path.exists(VESSEL_DATA_ROOT):
        df_vessels = validate_vessels_batch(VESSEL_DATA_ROOT, VESSEL_OUTPUT_CSV)
        create_validation_plots(df_vessels, "VESSELS", VESSEL_PLOT_DIR)