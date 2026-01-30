import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\OSIC_validation.csv")
RESULTS_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC_combined")
TRAIN_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\train.csv")
TEST_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\test.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\analyzis_base_results")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_clinical_data():
    """Load FVC and Week data from train.csv and test.csv"""
    print("Loading clinical data (FVC, Weeks)...")
    
    # Load both train and test
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    
    # Combine
    clinical = pd.concat([train, test], ignore_index=True)
    
    print(f"  Loaded {len(clinical)} clinical records")
    print(f"  Unique patients: {clinical['Patient'].nunique()}")
    print(f"  Week range: {clinical['Weeks'].min()} to {clinical['Weeks'].max()}")
    
    return clinical


def load_validation_results():
    """Load validation results and filter RELIABLE cases"""
    print("\nLoading validation results...")
    
    validation = pd.read_csv(VALIDATION_CSV)
    
    # Filter RELIABLE cases
    reliable = validation[validation['status'] == 'RELIABLE'].copy()
    
    print(f"  Total cases: {len(validation)}")
    print(f"  RELIABLE cases: {len(reliable)}")
    print(f"  UNRELIABLE cases: {len(validation[validation['status'] == 'UNRELIABLE'])}")
    
    return reliable


def load_advanced_metrics(case_name):
    """Load advanced metrics JSON for a specific case"""
    json_path = RESULTS_ROOT / case_name / "step4_analysis" / "advanced_metrics.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {case_name}: {e}")
        return None


def load_weibel_data(case_name):
    """Load Weibel generation analysis CSV for a specific case"""
    csv_path = RESULTS_ROOT / case_name / "step4_analysis" / "weibel_generation_analysis.csv"
    
    if not csv_path.exists():
        return None
    
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Warning: Could not load Weibel data for {case_name}: {e}")
        return None


def load_parenchymal_metrics(case_name):
    """Load parenchymal metrics JSON for a specific case"""
    json_path = RESULTS_ROOT / case_name / "step5_parenchymal_metrics" / "parenchymal_metrics.json"
    
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load parenchymal metrics for {case_name}: {e}")
        return None


def extract_patient_id(case_name):
    """Extract patient ID from case name (remove _gaussian suffix if present)"""
    return case_name.replace("_gaussian", "")


def is_smoothed_scan(case_name):
    """Check if this is a smoothed scan (has _gaussian suffix)"""
    return "_gaussian" in case_name


# ============================================================
# DATA INTEGRATION
# ============================================================

def build_integrated_dataset(reliable_cases, clinical_data):
    """Build integrated dataset with validation metrics + clinical data + parenchymal metrics"""
    print("\nBuilding integrated dataset (airway + parenchymal + clinical)...")
    
    rows = []
    cases_with_parenchymal = 0
    
    for idx, case_row in reliable_cases.iterrows():
        case_name = case_row['case']
        patient_id = extract_patient_id(case_name)
        
        # Load advanced metrics
        advanced = load_advanced_metrics(case_name)
        if advanced is None:
            print(f"  Skipping {case_name}: no advanced metrics")
            continue
        
        # Load Weibel data
        weibel = load_weibel_data(case_name)
        
        # Load parenchymal metrics
        parenchymal = load_parenchymal_metrics(case_name)
        has_parenchymal = parenchymal is not None
        
        if has_parenchymal:
            cases_with_parenchymal += 1
        
        # Get clinical data for this patient
        patient_clinical = clinical_data[clinical_data['Patient'] == patient_id]
        
        if len(patient_clinical) == 0:
            print(f"  Warning: No clinical data for {patient_id}")
            continue
        
        # For each time point (week) of this patient
        for _, clinical_row in patient_clinical.iterrows():
            row = {
                'case': case_name,
                'patient': patient_id,
                'is_smoothed': is_smoothed_scan(case_name),
                'week': clinical_row['Weeks'],
                'FVC': clinical_row['FVC'],
                'Percent': clinical_row['Percent'],
                'Age': clinical_row['Age'],
                'Sex': clinical_row['Sex'],
                'SmokingStatus': clinical_row['SmokingStatus'],
                
                # Validation metrics (from CSV)
                'volume_ml': case_row['volume_ml'],
                'mean_tortuosity': advanced.get('mean_tortuosity'),
                
                # NEW: Enhanced peripheral metrics (SOLO QUELLE SPECIFICATE)
                'std_peripheral_diameter_mm': advanced.get('std_peripheral_diameter_mm'),  # NEW: Regolarità diametri
                'central_to_peripheral_diameter_ratio': advanced.get('central_to_peripheral_diameter_ratio'),
                'mean_peripheral_branch_volume_mm3': advanced.get('mean_peripheral_branch_volume_mm3'),
            }
            
            # Add parenchymal metrics if available
            if has_parenchymal:
                row.update({
                    'mean_lung_density_HU': parenchymal.get('mean_lung_density_HU'),
                    'histogram_entropy': parenchymal.get('histogram_entropy'),
                })
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    print(f"  Created dataset with {len(df)} records")
    print(f"  Unique patients: {df['patient'].nunique()}")
    print(f"  Week range: {df['week'].min()} to {df['week'].max()}")
    print(f"  Cases with parenchymal metrics: {cases_with_parenchymal}")
    
    # Report smoothed scans
    if 'is_smoothed' in df.columns:
        n_smoothed = df['is_smoothed'].sum()
        n_smoothed_patients = df[df['is_smoothed']]['patient'].nunique()
        print(f"  Smoothed scans: {n_smoothed_patients} patients (kernel correction applied)")
    
    return df


# ============================================================
# VISUALIZATION
# ============================================================

def plot_metric_vs_percent(df, metric_name, metric_label, output_path):
    """Create scatter plot of metric vs FVC Percent (normalized) with week as color"""
    
    # Remove NaN values
    df_clean = df[[metric_name, 'Percent', 'week']].dropna()
    
    if len(df_clean) == 0:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot with week as color
    scatter = ax.scatter(
        df_clean[metric_name], 
        df_clean['Percent'],
        c=df_clean['week'],
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Week', rotation=270, labelpad=20, fontsize=12)
    
    # Add trend line (with error handling for numerical issues)
    try:
        z = np.polyfit(df_clean[metric_name], df_clean['Percent'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df_clean[metric_name].min(), df_clean[metric_name].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    except (np.linalg.LinAlgError, ValueError) as e:
        # Skip trend line if numerical issues (e.g., near-constant values)
        pass
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(df_clean[metric_name], df_clean['Percent'])
    spearman_r, spearman_p = spearmanr(df_clean[metric_name], df_clean['Percent'])
    
    # Add labels and title
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel('FVC Percent (% of predicted)', fontsize=12)
    ax.set_title(f'{metric_label} vs FVC Percent\n' + 
                 f'Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), Spearman ρ={spearman_r:.3f} (p={spearman_p:.4f})',
                 fontsize=13, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'metric': metric_name,
        'n_samples': len(df_clean),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }


def create_all_plots(df):
    """Create scatter plots for all metrics vs FVC Percent (normalized)"""
    print("\nCreating correlation plots with FVC Percent (normalized)...")
    
    # Define metrics to plot (SOLO QUELLE SPECIFICATE)
    metrics = [
        # CORE AIRWAY METRICS
        ('volume_ml', 'Airway Volume (ml)', 'airway'),
        ('mean_tortuosity', 'Mean Tortuosity', 'airway'),
        
        # PERIPHERAL AIRWAY METRICS (NEW)
        ('std_peripheral_diameter_mm', 'Std Peripheral Diameter (mm) - Regularity', 'airway_peripheral'),
        ('central_to_peripheral_diameter_ratio', 'Central/Peripheral Diameter Ratio', 'airway_peripheral'),
        ('mean_peripheral_branch_volume_mm3', 'Mean Peripheral Branch Volume (mm³)', 'airway_peripheral'),
        
        # PARENCHYMAL METRICS
        ('mean_lung_density_HU', 'Mean Lung Density (HU)', 'parenchymal'),
        ('histogram_entropy', 'Histogram Entropy', 'parenchymal'),
    ]
    
    correlation_results = []
    
    for metric_name, metric_label, metric_type in metrics:
        if metric_name not in df.columns:
            continue
        
        output_path = OUTPUT_DIR / f"{metric_name}_vs_percent.png"
        result = plot_metric_vs_percent(df, metric_name, metric_label, output_path)
        
        if result is not None:
            result['type'] = metric_type
            correlation_results.append(result)
            print(f"  ✓ {metric_name} (r={result['pearson_r']:.3f}, p={result['pearson_p']:.4f})")
    
    return pd.DataFrame(correlation_results)


def plot_correlation_summary(corr_df, output_path):
    """Create summary plot of correlations with FVC Percent (normalized)"""
    
    # Sort by absolute Pearson correlation
    corr_df = corr_df.copy()
    corr_df['abs_pearson_r'] = corr_df['pearson_r'].abs()
    corr_df = corr_df.sort_values('abs_pearson_r', ascending=True)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Color by metric type (airway vs parenchymal vs peripheral)
    def get_color(metric_type):
        if metric_type == 'airway':
            return 'steelblue'
        elif metric_type == 'airway_peripheral':
            return 'mediumseagreen'
        elif metric_type == 'parenchymal':
            return 'coral'
        else:
            return 'gray'
    
    colors = [get_color(t) for t in corr_df.get('type', ['airway']*len(corr_df))]
    
    # Plot 1: Pearson correlation
    ax1.barh(corr_df['metric'], corr_df['pearson_r'], color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    ax1.set_title('Pearson Correlation: Selected Metrics vs FVC Percent\n(normalized for age/sex/height)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Core Airway Metrics'),
        Patch(facecolor='mediumseagreen', alpha=0.7, label='Peripheral Metrics'),
        Patch(facecolor='coral', alpha=0.7, label='Parenchymal Metrics')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Spearman correlation
    ax2.barh(corr_df['metric'], corr_df['spearman_r'], color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Spearman Correlation Coefficient', fontsize=12)
    ax2.set_title('Spearman Correlation: Selected Metrics vs FVC Percent\n(normalized for age/sex/height)', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Correlation summary plot saved to: {output_path}")


def plot_percent_evolution(df, metric_name, metric_label, output_path):
    """Plot FVC Percent evolution by baseline metric quartiles"""
    df_clean = df[[metric_name, 'Percent', 'week', 'patient']].dropna()
    
    if len(df_clean) == 0:
        return None
    
    baseline_metrics = df_clean.groupby('patient')[metric_name].first().to_dict()
    df_clean['baseline_metric'] = df_clean['patient'].map(baseline_metrics)
    
    quartiles = pd.qcut(df_clean['baseline_metric'].rank(method='first'), q=4, 
                        labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    df_clean['quartile'] = quartiles
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['red', 'orange', 'lightgreen', 'darkgreen']
    
    # Plot 1: FVC Percent trajectories
    for quartile, color in zip(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], colors):
        subset = df_clean[df_clean['quartile'] == quartile]
        week_stats = subset.groupby('week')['Percent'].agg(['mean', 'std', 'count']).reset_index()
        week_stats = week_stats[week_stats['count'] >= 3]
        
        if len(week_stats) > 0:
            ax1.plot(week_stats['week'], week_stats['mean'], 'o-', 
                    label=f'{quartile} ({subset["baseline_metric"].mean():.1f})',
                    color=color, linewidth=2, markersize=6)
            ax1.fill_between(week_stats['week'], 
                            week_stats['mean'] - week_stats['std'],
                            week_stats['mean'] + week_stats['std'],
                            alpha=0.2, color=color)
    
    ax1.set_xlabel('Week', fontsize=12)
    ax1.set_ylabel('FVC Percent (% of predicted)', fontsize=12)
    ax1.set_title(f'FVC Percent Evolution by Baseline {metric_label}', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Decline rates
    decline_rates = []
    for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        subset = df_clean[df_clean['quartile'] == quartile]
        patient_rates = []
        for patient in subset['patient'].unique():
            patient_data = subset[subset['patient'] == patient].sort_values('week')
            if len(patient_data) >= 2:
                coeffs = np.polyfit(patient_data['week'], patient_data['Percent'], 1)
                patient_rates.append(coeffs[0])
        
        if len(patient_rates) > 0:
            decline_rates.append({
                'quartile': quartile,
                'mean_decline': np.mean(patient_rates),
                'std_decline': np.std(patient_rates)
            })
    
    decline_df = pd.DataFrame(decline_rates)
    if len(decline_df) > 0:
        x_pos = np.arange(len(decline_df))
        ax2.bar(x_pos, decline_df['mean_decline'], yerr=decline_df['std_decline'],
               alpha=0.7, color=colors[:len(decline_df)], capsize=5)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(decline_df['quartile'])
        ax2.set_ylabel('FVC Percent Decline Rate (%/week)', fontsize=12)
        ax2.set_title(f'Decline Rate by Baseline {metric_label}', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return decline_df


# ============================================================
# AIRWAY vs PARENCHYMAL COMPARISON
# ============================================================

def sensitivity_analysis_smoothed(df, corr_df):
    """Perform sensitivity analysis: correlations with/without smoothed scans"""
    
    if 'is_smoothed' not in df.columns:
        return None
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: SMOOTHED vs NON-SMOOTHED SCANS")
    print("="*80)
    
    n_total = df['patient'].nunique()
    n_smoothed = df[df['is_smoothed']]['patient'].nunique()
    n_original = df[~df['is_smoothed']]['patient'].nunique()
    
    print(f"\nDataset composition:")
    print(f"  Total patients: {n_total}")
    print(f"  Smoothed (kernel corrected): {n_smoothed} ({100*n_smoothed/n_total:.1f}%)")
    print(f"  Original: {n_original} ({100*n_original/n_total:.1f}%)")
    
    # For parenchymal metrics, compare correlations
    if 'type' in corr_df.columns:
        parenchymal_metrics = corr_df[corr_df['type'] == 'parenchymal']['metric'].tolist()
        
        if len(parenchymal_metrics) > 0:
            print(f"\nParenchymal metrics correlation comparison:")
            print(f"{'Metric':<40} {'All (r)':<12} {'Original (r)':<15} {'Smoothed (r)':<15} {'Δr':<10}")
            print("-"*95)
            
            sensitivity_results = []
            
            for metric in parenchymal_metrics:
                if metric not in df.columns or 'Percent' not in df.columns:
                    continue
                
                # All data
                df_all = df[[metric, 'Percent']].dropna()
                r_all, p_all = pearsonr(df_all[metric], df_all['Percent']) if len(df_all) > 10 else (np.nan, np.nan)
                
                # Original only
                df_orig = df[~df['is_smoothed']][[metric, 'Percent']].dropna()
                r_orig, p_orig = pearsonr(df_orig[metric], df_orig['Percent']) if len(df_orig) > 10 else (np.nan, np.nan)
                
                # Smoothed only
                df_smooth = df[df['is_smoothed']][[metric, 'Percent']].dropna()
                r_smooth, p_smooth = pearsonr(df_smooth[metric], df_smooth['Percent']) if len(df_smooth) > 10 else (np.nan, np.nan)
                
                delta_r = abs(r_all - r_orig) if not np.isnan(r_all) and not np.isnan(r_orig) else np.nan
                
                sensitivity_results.append({
                    'metric': metric,
                    'r_all': r_all,
                    'r_original': r_orig,
                    'r_smoothed': r_smooth,
                    'delta_r': delta_r,
                    'n_all': len(df_all),
                    'n_original': len(df_orig),
                    'n_smoothed': len(df_smooth)
                })
                
                print(f"{metric:<40} {r_all:>10.3f}  {r_orig:>13.3f}  {r_smooth:>13.3f}  {delta_r:>8.3f}")
            
            # Summary
            valid_deltas = [r['delta_r'] for r in sensitivity_results if not np.isnan(r['delta_r'])]
            if len(valid_deltas) > 0:
                mean_delta = np.mean(valid_deltas)
                max_delta = np.max(valid_deltas)
                print(f"\nΔr = difference between 'All' and 'Original only' correlations")
                print(f"Mean Δr: {mean_delta:.3f} (average impact of including smoothed scans)")
                print(f"Max Δr: {max_delta:.3f}")
                
                if max_delta < 0.1:
                    print("\n✓ CONCLUSION: Smoothed scans have minimal impact on correlations (Δr < 0.1)")
                    print("  → Safe to include in analysis")
                elif max_delta < 0.2:
                    print("\n⚠ CONCLUSION: Smoothed scans have moderate impact (0.1 < Δr < 0.2)")
                    print("  → Should report sensitivity analysis in results")
                else:
                    print("\n⚠⚠ CONCLUSION: Smoothed scans have significant impact (Δr > 0.2)")
                    print("  → Consider separate analysis or excluding smoothed scans")
            
            return pd.DataFrame(sensitivity_results)
    
    return None


def compare_airway_vs_parenchymal(corr_df):
    """Compare airway vs peripheral vs parenchymal metrics performance"""
    
    if 'type' not in corr_df.columns:
        return
    
    airway_df = corr_df[corr_df['type'] == 'airway'].copy()
    peripheral_df = corr_df[corr_df['type'] == 'airway_peripheral'].copy()
    parenchymal_df = corr_df[corr_df['type'] == 'parenchymal'].copy()
    
    print("\n" + "="*80)
    print("COMPARISON: AIRWAY vs PERIPHERAL vs PARENCHYMAL METRICS")
    print("="*80)
    
    print("\nCORE AIRWAY METRICS:")
    print(f"  Count: {len(airway_df)}")
    if len(airway_df) > 0:
        print(f"  Mean |r|: {airway_df['pearson_r'].abs().mean():.3f}")
        print(f"  Max |r|: {airway_df['pearson_r'].abs().max():.3f}")
        print(f"  Significant (p<0.05): {len(airway_df[airway_df['pearson_p'] < 0.05])}/{len(airway_df)}")
        best_airway = airway_df.loc[airway_df['pearson_r'].abs().idxmax()]
        print(f"  Best: {best_airway['metric']} (r={best_airway['pearson_r']:.3f})")
    
    print("\nPERIPHERAL AIRWAY METRICS:")
    print(f"  Count: {len(peripheral_df)}")
    if len(peripheral_df) > 0:
        print(f"  Mean |r|: {peripheral_df['pearson_r'].abs().mean():.3f}")
        print(f"  Max |r|: {peripheral_df['pearson_r'].abs().max():.3f}")
        print(f"  Significant (p<0.05): {len(peripheral_df[peripheral_df['pearson_p'] < 0.05])}/{len(peripheral_df)}")
        best_peripheral = peripheral_df.loc[peripheral_df['pearson_r'].abs().idxmax()]
        print(f"  Best: {best_peripheral['metric']} (r={best_peripheral['pearson_r']:.3f})")
    
    print("\nPARENCHYMAL METRICS:")
    print(f"  Count: {len(parenchymal_df)}")
    if len(parenchymal_df) > 0:
        print(f"  Mean |r|: {parenchymal_df['pearson_r'].abs().mean():.3f}")
        print(f"  Max |r|: {parenchymal_df['pearson_r'].abs().max():.3f}")
        print(f"  Significant (p<0.05): {len(parenchymal_df[parenchymal_df['pearson_p'] < 0.05])}/{len(parenchymal_df)}")
        best_parenchymal = parenchymal_df.loc[parenchymal_df['pearson_r'].abs().idxmax()]
        print(f"  Best: {best_parenchymal['metric']} (r={best_parenchymal['pearson_r']:.3f})")
    else:
        print("  No parenchymal data available yet")
        print("  → Run compute_parenchymal_metrics.py first to add parenchymal metrics")
    
    # Summary comparison
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    all_metrics = []
    if len(airway_df) > 0:
        all_metrics.append(('Core Airway', airway_df['pearson_r'].abs().mean()))
    if len(peripheral_df) > 0:
        all_metrics.append(('Peripheral', peripheral_df['pearson_r'].abs().mean()))
    if len(parenchymal_df) > 0:
        all_metrics.append(('Parenchymal', parenchymal_df['pearson_r'].abs().mean()))
    
    if all_metrics:
        all_metrics.sort(key=lambda x: x[1], reverse=True)
        print(f"\nMean |r| ranking:")
        for i, (name, mean_r) in enumerate(all_metrics, 1):
            print(f"  {i}. {name}: {mean_r:.3f}")
    
    print("\n" + "="*80)


def analyze_fvc_evolution_by_metric(df, metric_name, metric_label, output_path):
    """
    Analizza evoluzione FVC stratificata per quartili di baseline metric.
    Mostra se baseline alto/basso predice progressione o stabilità malattia.
    """
    print(f"\n{'='*80}")
    print(f"FVC EVOLUTION ANALYSIS: {metric_label}")
    print(f"{'='*80}")
    
    # Get baseline metric (week 0-10)
    baseline = df[(df['week'] >= -5) & (df['week'] <= 10)].copy()
    baseline_metrics = baseline.groupby('patient')[metric_name].first().to_dict()
    
    # Assign baseline to all timepoints
    df_analysis = df.copy()
    df_analysis['baseline_metric'] = df_analysis['patient'].map(baseline_metrics)
    
    # Remove patients without baseline
    df_analysis = df_analysis.dropna(subset=['baseline_metric', 'Percent'])
    
    if len(df_analysis) < 20:
        print(f"⚠ Insufficient data for evolution analysis")
        return None
    
    # Create quartiles based on baseline metric
    try:
        quartiles = pd.qcut(df_analysis['baseline_metric'].rank(method='first'), 
                           q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        df_analysis['quartile'] = quartiles
    except ValueError:
        print(f"⚠ Cannot create quartiles (insufficient unique values)")
        return None
    
    print(f"\nPatients: {df_analysis['patient'].nunique()}")
    print(f"Measurements: {len(df_analysis)}")
    print(f"Week range: {df_analysis['week'].min():.0f} to {df_analysis['week'].max():.0f}")
    
    # Calculate decline rates per patient
    decline_data = []
    
    for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        subset = df_analysis[df_analysis['quartile'] == quartile]
        
        patient_rates = []
        for patient in subset['patient'].unique():
            patient_data = subset[subset['patient'] == patient].sort_values('week')
            
            if len(patient_data) >= 2:
                # Linear regression: FVC% vs week
                try:
                    coeffs = np.polyfit(patient_data['week'], patient_data['Percent'], 1)
                    decline_rate = coeffs[0]  # ml/week
                    
                    # Calculate R² for quality of fit
                    p = np.poly1d(coeffs)
                    yhat = p(patient_data['week'])
                    ss_res = np.sum((patient_data['Percent'] - yhat) ** 2)
                    ss_tot = np.sum((patient_data['Percent'] - patient_data['Percent'].mean()) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    patient_rates.append({
                        'decline_rate': decline_rate,
                        'r2': r2,
                        'n_measurements': len(patient_data)
                    })
                except:
                    continue
        
        if len(patient_rates) > 0:
            rates = [x['decline_rate'] for x in patient_rates]
            baseline_val = subset['baseline_metric'].mean()
            
            decline_data.append({
                'quartile': quartile,
                'baseline_mean': baseline_val,
                'mean_decline': np.mean(rates),
                'std_decline': np.std(rates),
                'median_decline': np.median(rates),
                'n_patients': len(patient_rates),
                'declining_count': sum(1 for r in rates if r < -0.5),  # Decline > 0.5%/week
                'stable_count': sum(1 for r in rates if -0.5 <= r <= 0.5),
                'improving_count': sum(1 for r in rates if r > 0.5)
            })
    
    decline_df = pd.DataFrame(decline_data)
    
    if len(decline_df) == 0:
        print("⚠ Cannot compute decline rates")
        return None
    
    # Print results
    print(f"\n{'Quartile':<15} {'Baseline':<12} {'Decline Rate':<18} {'Status Distribution':<40}")
    print("-" * 85)
    
    for _, row in decline_df.iterrows():
        status_str = f"D:{row['declining_count']} S:{row['stable_count']} I:{row['improving_count']}"
        print(f"{row['quartile']:<15} {row['baseline_mean']:>10.3f}  "
              f"{row['mean_decline']:>7.2f}±{row['std_decline']:>5.2f} %/wk  {status_str:<40}")
    
    # Statistical test: correlation between baseline metric and decline rate
    if len(decline_df) >= 4:
        r_decline, p_decline = pearsonr(decline_df['baseline_mean'], decline_df['mean_decline'])
        print(f"\nBaseline {metric_label} vs Decline Rate:")
        print(f"  r={r_decline:.3f}, p={p_decline:.4f} {'***' if p_decline < 0.001 else '**' if p_decline < 0.01 else '*' if p_decline < 0.05 else ''}")
        
        if r_decline > 0 and p_decline < 0.05:
            print(f"  ✓ PROTETTIVO: Valori alti → minore decline (stabilità)")
        elif r_decline < 0 and p_decline < 0.05:
            print(f"  ⚠ PREDITTIVO: Valori alti → maggiore decline (progressione)")
        else:
            print(f"  → Non predittivo di progressione")
    
    # Visualization - ONLY 2 PLOTS (most important)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#c0392b', '#e67e22', '#f39c12', '#2ecc71']
    
    # Plot 1: FVC% trajectories
    ax1 = axes[0]
    for idx, (quartile, color) in enumerate(zip(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], colors)):
        subset = df_analysis[df_analysis['quartile'] == quartile]
        
        week_stats = subset.groupby('week')['Percent'].agg(['mean', 'std', 'count']).reset_index()
        week_stats = week_stats[week_stats['count'] >= 2]
        
        if len(week_stats) > 0:
            baseline_val = decline_df.iloc[idx]['baseline_mean']
            ax1.plot(week_stats['week'], week_stats['mean'], 'o-', 
                    label=f'{quartile} ({baseline_val:.2f})',
                    color=color, linewidth=2, markersize=4)
            ax1.fill_between(week_stats['week'], 
                            week_stats['mean'] - week_stats['std'],
                            week_stats['mean'] + week_stats['std'],
                            alpha=0.2, color=color)
    
    ax1.set_xlabel('Week', fontsize=12)
    ax1.set_ylabel('FVC Percent (% predicted)', fontsize=12)
    ax1.set_title(f'FVC% Evolution by Baseline {metric_label}', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Decline rates by quartile
    ax2 = axes[1]
    x_pos = np.arange(len(decline_df))
    bars = ax2.bar(x_pos, decline_df['mean_decline'], 
                   yerr=decline_df['std_decline'],
                   color=colors[:len(decline_df)], alpha=0.7, capsize=5,
                   edgecolor='black', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.axhline(y=-0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Decline threshold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(decline_df['quartile'], rotation=45, ha='right')
    ax2.set_ylabel('FVC% Decline Rate (%/week)', fontsize=12)
    ax2.set_title('Mean Decline Rate by Baseline Quartile', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot saved: {output_path}")
    
    return decline_df


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("="*80)
    print("OSIC METRICS vs FVC PERCENT ANALYSIS")
    print("Selected Airway + Parenchymal Metrics")
    print("Correlations with FVC Percent (normalized for age/sex/height)")
    print("="*80)
    
    # Load data
    clinical = load_clinical_data()
    reliable = load_validation_results()
    df = build_integrated_dataset(reliable, clinical)
    
    # Check if we have parenchymal metrics
    parenchymal_cols = [
        'mean_lung_density_HU', 'histogram_entropy'
    ]
    has_parenchymal = any(col in df.columns and df[col].notna().any() for col in parenchymal_cols)
    
    if not has_parenchymal:
        print("\n" + "!"*80)
        print("INFO: No parenchymal metrics found (analyzing airway metrics only)")
        print("To include parenchymal metrics, run: py compute_parenchymal_metrics.py")
        print("!"*80)
    else:
        n_parenchymal = df[parenchymal_cols[0]].notna().sum()
        print(f"\n✓ Parenchymal metrics available for {n_parenchymal} measurements")
    
    # Save integrated dataset
    dataset_path = OUTPUT_DIR / "integrated_dataset.csv"
    df.to_csv(dataset_path, index=False)
    print(f"\n✓ Integrated dataset saved to: {dataset_path}")
    
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"\nSelected airway metrics: measured at week 0 (baseline CT)")
    print(f"FVC measurements: week {df['week'].min()} to {df['week'].max()}")
    print(f"Median follow-up: {df['week'].median():.0f} weeks")
    print(f"Total records: {len(df)}")
    print(f"Unique patients: {df['patient'].nunique()}\n")
    
    # === PART 1: FVC vs METRIC CORRELATIONS ===
    print("="*80)
    print("PART 1: SELECTED METRICS vs FVC PERCENT - CORRELATION ANALYSIS")
    print("FVC Percent = normalized for age, sex, and height")
    print("="*80)
    
    corr_df = create_all_plots(df)
    corr_path = OUTPUT_DIR / "correlation_results.csv"
    corr_df.to_csv(corr_path, index=False)
    
    summary_plot_path = OUTPUT_DIR / "correlation_summary.png"
    plot_correlation_summary(corr_df, summary_plot_path)
    
    # Sensitivity analysis: smoothed vs original scans
    sensitivity_df = sensitivity_analysis_smoothed(df, corr_df)
    if sensitivity_df is not None:
        sensitivity_path = OUTPUT_DIR / "sensitivity_analysis_smoothed.csv"
        sensitivity_df.to_csv(sensitivity_path, index=False)
        print(f"\n✓ Sensitivity analysis saved to: {sensitivity_path}")
    
    # Compare airway vs parenchymal metrics
    compare_airway_vs_parenchymal(corr_df)
    
    # Sort by correlation strength
    corr_df['abs_pearson_r'] = corr_df['pearson_r'].abs()
    corr_sorted = corr_df.sort_values('abs_pearson_r', ascending=False)
    
    # Print top correlations with type
    print(f"\n{'Rank':<6} {'Type':<14} {'Metric':<35} {'Pearson r':<12} {'p-value':<12} {'Sig.':<6}")
    print("-"*88)
    for i, row in enumerate(corr_sorted.itertuples(), 1):
        sig = "***" if row.pearson_p < 0.001 else ("**" if row.pearson_p < 0.01 else ("*" if row.pearson_p < 0.05 else ""))
        metric_type = getattr(row, 'type', 'airway').upper()
        print(f"{i:<6} {metric_type:<14} {row.metric:<35} {row.pearson_r:>10.4f}  {row.pearson_p:>10.4f}  {sig:<6}")
    
    # === PART 2: FVC PERCENT EVOLUTION + DECLINE RATE ===
    print("\n" + "="*80)
    print("PART 2: FVC PERCENT EVOLUTION & DECLINE RATE ANALYSIS")
    print("="*80)
    
    # SOLO LE METRICHE SPECIFICATE
    metrics_to_analyze = [
        # AIRWAY METRICS
        ('volume_ml', 'Airway Volume (ml)'),
        ('mean_tortuosity', 'Mean Tortuosity'),
        ('std_peripheral_diameter_mm', 'Std Peripheral Diameter (mm)'),
        ('central_to_peripheral_diameter_ratio', 'Central/Peripheral Diameter Ratio'),
        ('mean_peripheral_branch_volume_mm3', 'Mean Peripheral Branch Volume (mm³)'),
        
        # PARENCHYMAL METRICS
        ('mean_lung_density_HU', 'Mean Lung Density (HU)'),
        ('histogram_entropy', 'Histogram Entropy'),
    ]
    
    print("\nCreating FVC Percent evolution and decline rate plots...")
    for metric_name, metric_label in metrics_to_analyze:
        if metric_name not in df.columns:
            continue
        
        # FVC Percent evolution plot
        evolution_path = OUTPUT_DIR / f"evolution_{metric_name}.png"
        decline_df = plot_percent_evolution(df, metric_name, metric_label, evolution_path)
        
        if decline_df is not None and len(decline_df) > 0:
            print(f"  {metric_name}")
            for idx, row in decline_df.iterrows():
                print(f"      {row['quartile']}: {row['mean_decline']:.2f} ± {row['std_decline']:.2f} %/week")
    
    # === PART 3: EVOLUTION ANALYSIS FOR KEY METRICS ===
    print("\n" + "="*80)
    print("PART 3: FVC EVOLUTION - Progression vs Stability Prediction")
    print("="*80)
    
    evolution_results = {}
    
    for metric_name, metric_label in metrics_to_analyze:
        if metric_name not in df.columns:
            continue
        
        output_path = OUTPUT_DIR / f"evolution_detailed_{metric_name}.png"
        result = analyze_fvc_evolution_by_metric(df, metric_name, metric_label, output_path)
        
        if result is not None:
            evolution_results[metric_name] = result
            result_path = OUTPUT_DIR / f"evolution_detailed_{metric_name}.csv"
            result.to_csv(result_path, index=False)
            print(f"✓ Saved: {result_path}")
    
    # === FINAL SUMMARY ===
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    significant = corr_sorted[corr_sorted['pearson_p'] < 0.05]
    
    print("\n1. STRONGEST CORRELATIONS (Selected Metrics vs FVC Percent):")
    for i, row in enumerate(corr_sorted.head(5).itertuples(), 1):
        strength = "Strong" if abs(row.pearson_r) >= 0.5 else ("Moderate" if abs(row.pearson_r) >= 0.3 else "Weak")
        direction = "positive" if row.pearson_r > 0 else "negative"
        sig = "***" if row.pearson_p < 0.001 else ("**" if row.pearson_p < 0.01 else ("*" if row.pearson_p < 0.05 else ""))
        metric_type = getattr(row, 'type', 'airway').upper()
        print(f"   {i}. [{metric_type}] {row.metric}: r={row.pearson_r:.3f} ({strength} {direction}) {sig}")
    
    print("\n2. STATISTICALLY SIGNIFICANT CORRELATIONS (p<0.05):")
    print(f"   Total: {len(significant)} out of {len(corr_sorted)} metrics")
    if len(significant) > 0:
        # Breakdown by type
        if 'type' in significant.columns:
            airway_sig = significant[significant['type'] == 'airway']
            peripheral_sig = significant[significant['type'] == 'airway_peripheral']
            parench_sig = significant[significant['type'] == 'parenchymal']
            print(f"   - Core Airway: {len(airway_sig)}")
            print(f"   - Peripheral: {len(peripheral_sig)}")
            print(f"   - Parenchymal: {len(parench_sig)}")
        print("\n   Significant metrics:")
        for idx, row in significant.iterrows():
            metric_type = row.get('type', 'airway').upper()
            print(f"      - [{metric_type}] {row['metric']}: r={row['pearson_r']:.3f} (p={row['pearson_p']:.4f})")
    
    print("\n3. INTERPRETATION:")
    print("   FVC Percent = patient FVC as % of predicted FVC for healthy person")
    print("                 with same age, sex, and height")
    print("   This removes confounding effects of body size and demographics")
    
    print("\n4. DISEASE PROGRESSION:")
    if len(evolution_results) > 0:
        print("   Metrics predicting FVC decline (progression):")
        for metric, result_df in evolution_results.items():
            if len(result_df) >= 3:
                # Check if high baseline → worse decline
                q1_decline = result_df[result_df['quartile'] == 'Q1 (Low)']['mean_decline'].values[0]
                q4_decline = result_df[result_df['quartile'] == 'Q4 (High)']['mean_decline'].values[0]
                
                if q4_decline < q1_decline - 0.3:  # Q4 declines faster
                    print(f"      ⚠ {metric}: High baseline → faster decline (predictive)")
                elif q1_decline < q4_decline - 0.3:  # Q1 declines faster
                    print(f"      ✓ {metric}: Low baseline → faster decline (protective)")
                else:
                    print(f"      → {metric}: No strong progression prediction")
    else:
        print("   (Evolution analysis not completed)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nFiles generated:")
    print(f"  - integrated_dataset.csv: Raw data with selected metrics and FVC Percent")
    print(f"  - correlation_results.csv: Correlation coefficients for selected metrics")
    print(f"  - correlation_summary.png: Visual comparison of all correlations")
    print(f"  - [metric]_vs_percent.png: Scatter plots (metric vs FVC Percent)")
    print(f"  - evolution_[metric].png: FVC Percent evolution & decline rate by quartile")
    print(f"  - evolution_detailed_[metric].png: Detailed evolution analysis")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()