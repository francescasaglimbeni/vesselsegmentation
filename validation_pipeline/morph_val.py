"""
VALIDATION STEP 3: MORPHOMETRIC MEASUREMENTS
=============================================

Valida accuratezza delle misure morfometriche:
- Diameter measurements
- Tapering ratios
- Generation/distance distributions
- Murray's law compliance (vessels)

CONFRONTO CON LETTERATURA:
- Diameter ranges per generation
- Tapering ratios (Weibel model)
- Murray's law violations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================================
# REFERENCE VALUES FROM LITERATURE
# ============================================================================

AIRWAY_MORPHOMETRY_REFERENCE = {
    "diameter_mm": {
        "trachea": {"min": 12, "max": 20},
        "main_bronchi": {"min": 8, "max": 15},
        "lobar_bronchi": {"min": 5, "max": 10},
        "segmental": {"min": 3, "max": 7},
        "subsegmental": {"min": 1.5, "max": 5},
        "terminal": {"min": 0.5, "max": 2},
        "source": "Weibel 1963, Horsfield et al. 1971"
    },
    "tapering_ratio": {
        "weibel_theoretical": 0.793,  # 2^(-1/3)
        "expected_min": 0.70,
        "expected_max": 0.88,
        "source": "Weibel symmetric branching model"
    },
    "diameter_by_generation": {
        0: {"mean": 12.0, "std": 2.0},
        5: {"mean": 6.0, "std": 1.5},
        10: {"mean": 3.0, "std": 1.0},
        15: {"mean": 1.5, "std": 0.5},
        20: {"mean": 0.8, "std": 0.3},
        "source": "Averaged airway morphometry studies"
    }
}

VESSEL_MORPHOMETRY_REFERENCE = {
    "diameter_mm": {
        "main_PA": {"min": 20, "max": 30},
        "lobar_arteries": {"min": 8, "max": 15},
        "segmental": {"min": 3, "max": 8},
        "subsegmental": {"min": 1.5, "max": 5},
        "small": {"min": 0.5, "max": 2},
        "source": "Pulmonary vascular morphometry"
    },
    "murray_law_deviation": {
        "acceptable": 0.20,  # ±20% è accettabile
        "warning": 0.30,
        "critical": 0.50,
        "source": "Biomechanical optimality"
    },
    "diameter_by_distance": {
        # From trunk (mm): expected diameter range
        "0-20": {"min": 8, "max": 25},
        "20-50": {"min": 4, "max": 12},
        "50-100": {"min": 2, "max": 8},
        "100+": {"min": 0.5, "max": 4},
        "source": "Distance-based vascular morphometry"
    }
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_airway_morphometry(case_dir):
    """
    Valida misure morfometriche airways
    
    Legge:
    - branch_metrics_complete.csv
    - weibel_generation_analysis.csv
    - weibel_tapering_ratios.csv
    """
    
    step4_dir = os.path.join(case_dir, "step4_analysis")
    
    # Load data
    branch_csv = os.path.join(step4_dir, "branch_metrics_complete.csv")
    weibel_csv = os.path.join(step4_dir, "weibel_generation_analysis.csv")
    tapering_csv = os.path.join(step4_dir, "weibel_tapering_ratios.csv")
    
    if not os.path.exists(branch_csv):
        return create_empty_morphometry_result()
    
    branch_df = pd.read_csv(branch_csv)
    
    # ========================================
    # DIAMETER ANALYSIS
    # ========================================
    ref = AIRWAY_MORPHOMETRY_REFERENCE
    
    diameter_scores = []
    
    # Overall diameter range
    mean_diameter = branch_df['diameter_mean_mm'].mean()
    median_diameter = branch_df['diameter_mean_mm'].median()
    std_diameter = branch_df['diameter_mean_mm'].std()
    
    # Check if diameters are in plausible ranges
    min_plausible = ref["diameter_mm"]["terminal"]["min"]
    max_plausible = ref["diameter_mm"]["trachea"]["max"]
    
    outliers = ((branch_df['diameter_mean_mm'] < min_plausible) | 
                (branch_df['diameter_mean_mm'] > max_plausible)).sum()
    outlier_pct = outliers / len(branch_df) * 100
    
    diameter_quality = 100 - min(outlier_pct * 2, 50)  # Max penalty 50pts
    diameter_scores.append(("diameter_range", diameter_quality))
    
    # ========================================
    # GENERATION ANALYSIS
    # ========================================
    generation_quality = 100
    
    if os.path.exists(weibel_csv) and 'generation' in branch_df.columns:
        weibel_df = pd.read_csv(weibel_csv)
        
        # Check diameter progression by generation
        gen_diameters = branch_df.groupby('generation')['diameter_mean_mm'].mean()
        
        # Should monotonically decrease
        is_decreasing = all(gen_diameters.iloc[i] >= gen_diameters.iloc[i+1] 
                           for i in range(len(gen_diameters)-1))
        
        if not is_decreasing:
            generation_quality -= 20
        
        # Compare with reference values
        ref_gens = ref["diameter_by_generation"]
        deviations = []
        
        for gen, ref_data in ref_gens.items():
            if gen == "source":
                continue
            if gen in gen_diameters.index:
                measured = gen_diameters[gen]
                expected = ref_data["mean"]
                tolerance = ref_data["std"] * 2
                
                deviation = abs(measured - expected) / tolerance
                deviations.append(min(deviation, 2.0))
        
        if deviations:
            avg_deviation = np.mean(deviations)
            generation_quality -= min(avg_deviation * 20, 30)
    
    diameter_scores.append(("generation_progression", max(0, generation_quality)))
    
    # ========================================
    # TAPERING ANALYSIS
    # ========================================
    tapering_quality = 100
    
    if os.path.exists(tapering_csv):
        tapering_df = pd.read_csv(tapering_csv)
        
        mean_tapering = tapering_df['diameter_ratio'].mean()
        weibel_theoretical = ref["tapering_ratio"]["weibel_theoretical"]
        
        deviation = abs(mean_tapering - weibel_theoretical) / weibel_theoretical
        
        if deviation > 0.30:  # >30% deviation
            tapering_quality = 40
        elif deviation > 0.20:
            tapering_quality = 60
        elif deviation > 0.10:
            tapering_quality = 80
    
    diameter_scores.append(("tapering", tapering_quality))
    
    # ========================================
    # OVERALL MORPHOMETRY QUALITY
    # ========================================
    overall_morphometry_quality = np.mean([s for _, s in diameter_scores])
    
    # ========================================
    # SEVERITY FLAGS
    # ========================================
    severity_flags = []
    
    if outlier_pct > 20:
        severity_flags.append("CRITICAL_TOO_MANY_DIAMETER_OUTLIERS")
    elif outlier_pct > 10:
        severity_flags.append("WARNING_DIAMETER_OUTLIERS")
    
    if not is_decreasing:
        severity_flags.append("WARNING_NON_MONOTONIC_TAPERING")
    
    if os.path.exists(tapering_csv):
        deviation = abs(mean_tapering - weibel_theoretical) / weibel_theoretical
        if deviation > 0.30:
            severity_flags.append("WARNING_ABNORMAL_TAPERING")
    
    morphometry_usable = (outlier_pct < 30 and overall_morphometry_quality > 40)
    
    # ========================================
    # RETURN
    # ========================================
    return {
        # Diameter stats
        "mean_diameter_mm": float(mean_diameter),
        "median_diameter_mm": float(median_diameter),
        "std_diameter_mm": float(std_diameter),
        "min_diameter_mm": float(branch_df['diameter_mean_mm'].min()),
        "max_diameter_mm": float(branch_df['diameter_mean_mm'].max()),
        "diameter_outliers_pct": float(outlier_pct),
        
        # Tapering
        "mean_tapering_ratio": float(mean_tapering) if os.path.exists(tapering_csv) else np.nan,
        "tapering_deviation_from_weibel": float(deviation) if os.path.exists(tapering_csv) else np.nan,
        "diameter_progression_monotonic": bool(is_decreasing) if os.path.exists(weibel_csv) else np.nan,
        
        # Quality
        "quality_diameter_range": float(diameter_scores[0][1]),
        "quality_generation_progression": float(diameter_scores[1][1]) if len(diameter_scores) > 1 else np.nan,
        "quality_tapering": float(diameter_scores[2][1]) if len(diameter_scores) > 2 else np.nan,
        "quality_morphometry_overall": float(overall_morphometry_quality),
        
        # Flags
        "MORPHOMETRY_USABLE": morphometry_usable,
        "severity_flags": ";".join(severity_flags) if severity_flags else "NONE"
    }


def validate_vessel_morphometry(case_dir):
    """
    Valida misure morfometriche vessels
    
    Legge:
    - branch_metrics.csv
    - murray_violations.csv (se disponibile)
    - path_metrics.csv
    """
    
    step2_dir = os.path.join(case_dir, "step2_unified_analysis")
    
    branch_csv = os.path.join(step2_dir, "branch_metrics.csv")
    murray_csv = os.path.join(step2_dir, "murray_violations.csv")
    path_csv = os.path.join(step2_dir, "path_metrics.csv")
    
    if not os.path.exists(branch_csv):
        return create_empty_morphometry_result()
    
    branch_df = pd.read_csv(branch_csv)
    
    # ========================================
    # DIAMETER ANALYSIS
    # ========================================
    ref = VESSEL_MORPHOMETRY_REFERENCE
    
    diameter_scores = []
    
    mean_diameter = branch_df['diameter_mm'].mean()
    median_diameter = branch_df['diameter_mm'].median()
    std_diameter = branch_df['diameter_mm'].std()
    
    # Check plausible ranges
    min_plausible = ref["diameter_mm"]["small"]["min"]
    max_plausible = ref["diameter_mm"]["main_PA"]["max"]
    
    outliers = ((branch_df['diameter_mm'] < min_plausible) | 
                (branch_df['diameter_mm'] > max_plausible)).sum()
    outlier_pct = outliers / len(branch_df) * 100
    
    diameter_quality = 100 - min(outlier_pct * 2, 50)
    diameter_scores.append(("diameter_range", diameter_quality))
    
    # ========================================
    # MURRAY'S LAW
    # ========================================
    murray_quality = 100
    murray_violation_rate = 0.0
    mean_murray_deviation = 0.0
    
    if os.path.exists(murray_csv):
        murray_df = pd.read_csv(murray_csv)
        
        if len(murray_df) > 0:
            # Calcola rate violazioni
            # Assumo che murray_df contenga solo violazioni
            # Serve numero totale biforcazioni
            bifurcations = branch_df['degree'].apply(lambda x: x >= 3 if pd.notna(x) else False).sum() if 'degree' in branch_df.columns else len(branch_df) * 0.3
            
            murray_violation_rate = len(murray_df) / max(bifurcations, 1) * 100
            
            if 'ratio_actual_over_expected' in murray_df.columns:
                mean_murray_deviation = abs(murray_df['ratio_actual_over_expected'] - 1.0).mean()
            
            # Scoring
            if murray_violation_rate > 50:
                murray_quality = 30
            elif murray_violation_rate > 30:
                murray_quality = 60
            elif murray_violation_rate > 15:
                murray_quality = 80
    
    diameter_scores.append(("murray_law", murray_quality))
    
    # ========================================
    # PATH TAPERING
    # ========================================
    path_quality = 100
    
    if os.path.exists(path_csv):
        path_df = pd.read_csv(path_csv)
        
        if 'taper_ratio' in path_df.columns:
            # Taper ratio should be > 1 (diametro diminuisce lungo path)
            valid_taper = path_df[path_df['taper_ratio'].notna()]
            
            if len(valid_taper) > 0:
                mean_taper = valid_taper['taper_ratio'].mean()
                
                # Taper ratio ideale ~1.5-3.0
                if mean_taper < 1.0:
                    path_quality = 20  # Diametro aumenta?!
                elif mean_taper < 1.2:
                    path_quality = 60
                elif mean_taper > 5.0:
                    path_quality = 70  # Taper eccessivo
    
    diameter_scores.append(("path_tapering", path_quality))
    
    # ========================================
    # OVERALL
    # ========================================
    overall_morphometry_quality = np.mean([s for _, s in diameter_scores])
    
    # ========================================
    # SEVERITY FLAGS
    # ========================================
    severity_flags = []
    
    if outlier_pct > 20:
        severity_flags.append("CRITICAL_TOO_MANY_DIAMETER_OUTLIERS")
    
    if murray_violation_rate > 50:
        severity_flags.append("WARNING_HIGH_MURRAY_VIOLATIONS")
    
    if os.path.exists(path_csv):
        if mean_taper < 1.0:
            severity_flags.append("CRITICAL_INVERTED_TAPERING")
    
    morphometry_usable = (outlier_pct < 30 and overall_morphometry_quality > 40)
    
    # ========================================
    # RETURN
    # ========================================
    return {
        # Diameter stats
        "mean_diameter_mm": float(mean_diameter),
        "median_diameter_mm": float(median_diameter),
        "std_diameter_mm": float(std_diameter),
        "min_diameter_mm": float(branch_df['diameter_mm'].min()),
        "max_diameter_mm": float(branch_df['diameter_mm'].max()),
        "diameter_outliers_pct": float(outlier_pct),
        
        # Murray's law
        "murray_violation_rate_pct": float(murray_violation_rate),
        "mean_murray_deviation": float(mean_murray_deviation),
        
        # Path tapering
        "mean_path_taper_ratio": float(mean_taper) if os.path.exists(path_csv) else np.nan,
        
        # Quality
        "quality_diameter_range": float(diameter_scores[0][1]),
        "quality_murray_law": float(diameter_scores[1][1]) if len(diameter_scores) > 1 else np.nan,
        "quality_path_tapering": float(diameter_scores[2][1]) if len(diameter_scores) > 2 else np.nan,
        "quality_morphometry_overall": float(overall_morphometry_quality),
        
        # Flags
        "MORPHOMETRY_USABLE": morphometry_usable,
        "severity_flags": ";".join(severity_flags) if severity_flags else "NONE"
    }


def create_empty_morphometry_result():
    """Empty result for failed cases"""
    return {
        "mean_diameter_mm": 0.0,
        "median_diameter_mm": 0.0,
        "std_diameter_mm": 0.0,
        "min_diameter_mm": 0.0,
        "max_diameter_mm": 0.0,
        "diameter_outliers_pct": 0.0,
        "quality_diameter_range": 0.0,
        "quality_morphometry_overall": 0.0,
        "MORPHOMETRY_USABLE": False,
        "severity_flags": "CRITICAL_NO_DATA"
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def validate_airways_morphometry_batch(data_root, output_csv):
    """Batch validation airway morphometry"""
    print("="*80)
    print("AIRWAY MORPHOMETRY VALIDATION - STEP 3")
    print("="*80)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    results = []
    
    for case_id in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, case_id)
        
        if not os.path.isdir(case_dir):
            continue
        
        print(f"[VALIDATING] {case_id}")
        
        try:
            row = validate_airway_morphometry(case_dir)
            row["case_id"] = case_id
            results.append(row)
            
            if row["MORPHOMETRY_USABLE"]:
                print(f"  ✓ USABLE - Quality: {row['quality_morphometry_overall']:.1f}/100")
            else:
                print(f"  ✗ UNUSABLE - {row['severity_flags']}")
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print_summary(df, "AIRWAY MORPHOMETRY")
    
    return df


def validate_vessels_morphometry_batch(data_root, output_csv):
    """Batch validation vessel morphometry"""
    print("="*80)
    print("VESSEL MORPHOMETRY VALIDATION - STEP 3")
    print("="*80)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    results = []
    
    for case_id in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, case_id)
        
        if not os.path.isdir(case_dir):
            continue
        
        print(f"[VALIDATING] {case_id}")
        
        try:
            row = validate_vessel_morphometry(case_dir)
            row["case_id"] = case_id
            results.append(row)
            
            if row["MORPHOMETRY_USABLE"]:
                print(f"  ✓ USABLE - Quality: {row['quality_morphometry_overall']:.1f}/100")
            else:
                print(f"  ✗ UNUSABLE - {row['severity_flags']}")
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print_summary(df, "VESSEL MORPHOMETRY")
    
    return df


def print_summary(df, structure_type):
    """Print summary"""
    print("\n" + "="*80)
    print(f"{structure_type} VALIDATION SUMMARY")
    print("="*80)
    
    print(f"Total cases: {len(df)}")
    print(f"Usable: {df['MORPHOMETRY_USABLE'].sum()} ({df['MORPHOMETRY_USABLE'].sum()/len(df)*100:.1f}%)")
    
    print(f"\nMean Quality: {df['quality_morphometry_overall'].mean():.1f}/100")
    
    print(f"\nDiameter Statistics:")
    print(f"  Mean: {df['mean_diameter_mm'].mean():.2f} ± {df['std_diameter_mm'].mean():.2f} mm")
    print(f"  Outliers: {df['diameter_outliers_pct'].mean():.1f}%")


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_morphometry_plots(df, structure_type, output_dir):
    """Create morphometry validation plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Quality distribution
    axes[0, 0].hist(df['quality_morphometry_overall'], bins=20, edgecolor='black', color='steelblue', alpha=0.7)
    axes[0, 0].axvline(df['quality_morphometry_overall'].mean(), color='r', linestyle='--')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'{structure_type} Morphometry Quality')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Diameter distribution
    axes[0, 1].hist(df['mean_diameter_mm'], bins=20, edgecolor='black', color='seagreen', alpha=0.7)
    axes[0, 1].set_xlabel('Mean Diameter (mm)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Diameter Distribution')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Outlier percentage
    axes[0, 2].hist(df['diameter_outliers_pct'], bins=20, edgecolor='black', color='orange', alpha=0.7)
    axes[0, 2].set_xlabel('Outliers (%)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Diameter Outliers')
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Quality components (if available)
    quality_cols = [col for col in df.columns if col.startswith('quality_') and col != 'quality_morphometry_overall']
    if quality_cols:
        quality_means = [df[col].mean() for col in quality_cols]
        labels = [col.replace('quality_', '').replace('_', ' ').title() for col in quality_cols]
        
        axes[1, 0].bar(labels, quality_means, edgecolor='black', alpha=0.7)
        axes[1, 0].set_ylabel('Mean Quality Score')
        axes[1, 0].set_title('Quality Components')
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(alpha=0.3, axis='y')
    
    # 5. Usability pie
    usable = df['MORPHOMETRY_USABLE'].sum()
    unusable = len(df) - usable
    axes[1, 1].pie([usable, unusable], labels=['Usable', 'Unusable'], 
                   autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    axes[1, 1].set_title('Morphometry Usability')
    
    # 6. Additional metric (structure-specific)
    if "AIRWAY" in structure_type and 'mean_tapering_ratio' in df.columns:
        valid_taper = df[df['mean_tapering_ratio'].notna()]
        axes[1, 2].hist(valid_taper['mean_tapering_ratio'], bins=20, edgecolor='black', color='crimson', alpha=0.7)
        axes[1, 2].axvline(0.793, color='g', linestyle='--', label='Weibel theoretical')
        axes[1, 2].set_xlabel('Tapering Ratio')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Tapering Ratio Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)
    
    elif "VESSEL" in structure_type and 'murray_violation_rate_pct' in df.columns:
        axes[1, 2].hist(df['murray_violation_rate_pct'], bins=20, edgecolor='black', color='purple', alpha=0.7)
        axes[1, 2].set_xlabel('Murray Violation Rate (%)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title("Murray's Law Violations")
        axes[1, 2].grid(alpha=0.3)
    
    plt.suptitle(f'{structure_type} Morphometry Validation - Step 3', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'validation_step3_{structure_type.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved validation plot: {plot_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # AIRWAYS
    AIRWAY_DATA_ROOT = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\output_results_with_fibrosis"
    AIRWAY_OUTPUT_CSV = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\output\step3_morph_val\airway_validation_step3_morphometry.csv"
    AIRWAY_PLOT_DIR = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\output\step3_morph_val"
    
    if os.path.exists(AIRWAY_DATA_ROOT):
        df_airways = validate_airways_morphometry_batch(AIRWAY_DATA_ROOT, AIRWAY_OUTPUT_CSV)
        create_morphometry_plots(df_airways, "AIRWAY MORPHOMETRY", AIRWAY_PLOT_DIR)
    
    # VESSELS
    VESSEL_DATA_ROOT = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\vessel_segmentation_new\vessel_output"
    VESSEL_OUTPUT_CSV = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\output\step3_morph_val\vessel_validation_step3_morphometry.csv"
    VESSEL_PLOT_DIR = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\output\step3_morph_val"
    
    if os.path.exists(VESSEL_DATA_ROOT):
        df_vessels = validate_vessels_morphometry_batch(VESSEL_DATA_ROOT, VESSEL_OUTPUT_CSV)
        create_morphometry_plots(df_vessels, "VESSEL MORPHOMETRY", VESSEL_PLOT_DIR)