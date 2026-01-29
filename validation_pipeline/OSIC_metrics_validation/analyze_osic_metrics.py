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

VALIDATION_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\OSIC_validation_newmetrics.csv")
RESULTS_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC_newMetrcis")
TRAIN_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\train.csv")
TEST_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\test.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\results_analysis")

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
                'branch_count': case_row['branch_count'],
                'max_generation': case_row['max_generation'],
                'pc_ratio': case_row['pc_ratio'],
                'tapering_ratio': case_row['tapering_ratio'],
                
                # Advanced metrics (from JSON)
                'total_volume_mm3': advanced.get('total_volume_mm3'),
                'mean_tortuosity': advanced.get('mean_tortuosity'),
                
                # NEW: Enhanced peripheral metrics
                'peripheral_volume_mm3': advanced.get('peripheral_volume_mm3'),
                'peripheral_volume_percent': advanced.get('peripheral_volume_percent'),
                'mean_peripheral_diameter_mm': advanced.get('mean_peripheral_diameter_mm'),
                'std_peripheral_diameter_mm': advanced.get('std_peripheral_diameter_mm'),  # NEW: Regolarità diametri
                'mean_peripheral_branch_volume_mm3': advanced.get('mean_peripheral_branch_volume_mm3'),
                'peripheral_branch_density': advanced.get('peripheral_branch_density'),
                'central_to_peripheral_diameter_ratio': advanced.get('central_to_peripheral_diameter_ratio'),
                'diameter_cv': advanced.get('diameter_cv'),
                'mean_central_diameter_mm': advanced.get('mean_central_diameter_mm'),
                
                # NEW: PC ratio normalizzato e wall thickness
                'pc_ratio_normalized': (advanced.get('peripheral_volume_mm3', 0) / advanced.get('total_volume_mm3', 1)) * 100 if advanced.get('total_volume_mm3', 0) > 0 else None,
                'wall_thickness_proxy': advanced.get('mean_diameter_to_length_ratio'),  # Nuovo: spessore parete
                
                # Pi10 metrics (to be verified)
                'pi10': advanced.get('pi10'),
                'pi10_slope': advanced.get('pi10_slope'),
            }
            
            # Add Weibel-specific metrics if available
            if weibel is not None and len(weibel) > 0:
                # Diameter at generation 0 (trachea)
                gen0 = weibel[weibel['generation'] == 0]
                if len(gen0) > 0 and 'mean_diameter' in gen0.columns:
                    row['trachea_diameter'] = gen0['mean_diameter'].mean()
                
                # Diameter at generation 5
                gen5 = weibel[weibel['generation'] == 5]
                if len(gen5) > 0 and 'mean_diameter' in gen5.columns:
                    row['gen5_diameter'] = gen5['mean_diameter'].mean()
            
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
    
    # Define metrics to plot (AIRWAY + PARENCHYMAL)
    metrics = [
        # CORE AIRWAY METRICS
        ('volume_ml', 'Airway Volume (ml)', 'airway'),
        ('total_volume_mm3', 'Total Volume (mm³)', 'airway'),
        ('branch_count', 'Branch Count', 'airway'),
        ('max_generation', 'Max Generation', 'airway'),
        ('pc_ratio', 'Peripheral/Central Ratio', 'airway'),
        ('tapering_ratio', 'Tapering Ratio', 'airway'),
        ('mean_tortuosity', 'Mean Tortuosity', 'airway'),
        
        # NEW: ENHANCED PERIPHERAL METRICS
        ('peripheral_volume_mm3', 'Peripheral Volume (mm³)', 'airway_peripheral'),
        ('peripheral_volume_percent', 'Peripheral Volume (%)', 'airway_peripheral'),
        ('mean_peripheral_diameter_mm', 'Mean Peripheral Diameter (mm)', 'airway_peripheral'),
        ('std_peripheral_diameter_mm', 'Std Peripheral Diameter (mm) - Regularity', 'airway_peripheral'),
        ('mean_peripheral_branch_volume_mm3', 'Mean Peripheral Branch Volume (mm³)', 'airway_peripheral'),
        ('peripheral_branch_density', 'Peripheral Branch Density', 'airway_peripheral'),
        ('central_to_peripheral_diameter_ratio', 'Central/Peripheral Diameter Ratio', 'airway_peripheral'),
        ('diameter_cv', 'Diameter Coefficient of Variation', 'airway_peripheral'),
        ('mean_central_diameter_mm', 'Mean Central Diameter (mm)', 'airway_peripheral'),
        
        # NEW: NORMALIZED & WALL METRICS
        ('pc_ratio_normalized', 'PC Ratio Normalized (% peripheral volume)', 'airway_normalized'),
        ('wall_thickness_proxy', 'Wall Thickness Proxy (diameter/length)', 'airway_wall'),
        
        # PI10 METRICS (to verify)
        ('pi10', 'Pi10 (wall thickness proxy)', 'airway_wall'),
        ('pi10_slope', 'Pi10 Slope', 'airway_wall'),
        
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
        elif metric_type == 'airway_normalized':
            return 'mediumorchid'
        elif metric_type == 'airway_wall':
            return 'darkorange'
        elif metric_type == 'parenchymal':
            return 'coral'
        else:
            return 'gray'
    
    colors = [get_color(t) for t in corr_df.get('type', ['airway']*len(corr_df))]
    
    # Plot 1: Pearson correlation
    ax1.barh(corr_df['metric'], corr_df['pearson_r'], color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    ax1.set_title('Pearson Correlation: All Metrics vs FVC Percent\n(normalized for age/sex/height)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Core Airway Metrics'),
        Patch(facecolor='mediumseagreen', alpha=0.7, label='Peripheral Metrics'),
        Patch(facecolor='mediumorchid', alpha=0.7, label='Normalized Metrics'),
        Patch(facecolor='darkorange', alpha=0.7, label='Wall Thickness Metrics'),
        Patch(facecolor='coral', alpha=0.7, label='Parenchymal Metrics')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Spearman correlation
    ax2.barh(corr_df['metric'], corr_df['spearman_r'], color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Spearman Correlation Coefficient', fontsize=12)
    ax2.set_title('Spearman Correlation: All Metrics vs FVC Percent\n(normalized for age/sex/height)', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Correlation summary plot saved to: {output_path}")

def plot_fvc_evolution(df, metric_name, metric_label, output_path):
    """Plot FVC evolution stratified by baseline metric quartiles"""
    
    df_clean = df[[metric_name, 'FVC', 'week', 'patient']].dropna()
    
    if len(df_clean) == 0:
        return None
    
    # Get baseline metric
    baseline_metrics = df_clean.groupby('patient')[metric_name].first().to_dict()
    df_clean['baseline_metric'] = df_clean['patient'].map(baseline_metrics)
    
    # Quartiles
    quartiles = pd.qcut(df_clean['baseline_metric'].rank(method='first'), q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    df_clean['quartile'] = quartiles
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: FVC trajectories
    colors = ['red', 'orange', 'lightgreen', 'darkgreen']
    
    for quartile, color in zip(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], colors):
        subset = df_clean[df_clean['quartile'] == quartile]
        
        week_stats = subset.groupby('week')['FVC'].agg(['mean', 'std', 'count']).reset_index()
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
    ax1.set_ylabel('FVC (ml)', fontsize=12)
    ax1.set_title(f'FVC Evolution by Baseline {metric_label}\n(Mean ± SD)', fontsize=13, fontweight='bold')
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
                coeffs = np.polyfit(patient_data['week'], patient_data['FVC'], 1)
                decline_rate = coeffs[0]
                patient_rates.append(decline_rate)
        
        if len(patient_rates) > 0:
            decline_rates.append({
                'quartile': quartile,
                'mean_decline': np.mean(patient_rates),
                'std_decline': np.std(patient_rates),
                'n_patients': len(patient_rates)
            })
    
    decline_df = pd.DataFrame(decline_rates)
    
    if len(decline_df) > 0:
        x_pos = np.arange(len(decline_df))
        ax2.bar(x_pos, decline_df['mean_decline'], yerr=decline_df['std_decline'],
               alpha=0.7, color=colors[:len(decline_df)], capsize=5)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(decline_df['quartile'])
        ax2.set_ylabel('FVC Decline Rate (ml/week)', fontsize=12)
        ax2.set_title(f'FVC Decline Rate by Baseline {metric_label}\n(Mean ± SD)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return decline_df


# ============================================================
# TEMPORAL CORRELATION ANALYSIS
# ============================================================

def analyze_correlation_by_week(df, metric_name):
    """Analyze how correlation changes with week"""
    df_clean = df[[metric_name, 'Percent', 'week', 'patient']].dropna()
    
    if len(df_clean) == 0:
        return None
    
    week_bins = [
        (0, 10, 'Week 0-10'),
        (10, 20, 'Week 10-20'),
        (20, 30, 'Week 20-30'),
        (30, 50, 'Week 30-50'),
        (50, 100, 'Week >50')
    ]
    
    results = []
    for min_week, max_week, label in week_bins:
        subset = df_clean[(df_clean['week'] > min_week) & (df_clean['week'] <= max_week)]
        
        if len(subset) < 10:
            continue
        
        try:
            pearson_r, pearson_p = pearsonr(subset[metric_name], subset['Percent'])
            spearman_r, spearman_p = spearmanr(subset[metric_name], subset['Percent'])
            
            results.append({
                'week_range': label,
                'min_week': min_week,
                'max_week': max_week,
                'n_samples': len(subset),
                'n_patients': subset['patient'].nunique(),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            })
        except:
            continue
    
    return pd.DataFrame(results) if results else None


def plot_temporal_analysis(df, metric_name, metric_label, output_path):
    """Create comprehensive temporal analysis plot"""
    temporal_df = analyze_correlation_by_week(df, metric_name)
    
    if temporal_df is None or len(temporal_df) == 0:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Correlation over time
    ax1 = axes[0, 0]
    x_pos = np.arange(len(temporal_df))
    ax1.plot(x_pos, temporal_df['pearson_r'], 'o-', linewidth=2, markersize=8, label='Pearson r', color='blue')
    ax1.plot(x_pos, temporal_df['spearman_r'], 's-', linewidth=2, markersize=8, label='Spearman ρ', color='green')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(temporal_df['week_range'], rotation=45, ha='right')
    ax1.set_ylabel('Correlation Coefficient', fontsize=12)
    ax1.set_title(f'Correlation Over Time\n{metric_label} vs FVC Percent', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample sizes
    ax2 = axes[0, 1]
    ax2.bar(x_pos, temporal_df['n_samples'], alpha=0.7, color='steelblue')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(temporal_df['week_range'], rotation=45, ha='right')
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Size by Time Period', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Scatter colored by week bin
    ax3 = axes[1, 0]
    df_clean = df[[metric_name, 'Percent', 'week']].dropna()
    df_clean['week_bin'] = pd.cut(df_clean['week'], 
                                    bins=[0, 10, 20, 30, 50, 100],
                                    labels=['0-10', '10-20', '20-30', '30-50', '>50'])
    
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(df_clean['week_bin'].cat.categories)))
    for i, (bin_name, color) in enumerate(zip(df_clean['week_bin'].cat.categories, colors_map)):
        subset = df_clean[df_clean['week_bin'] == bin_name]
        if len(subset) > 0:
            ax3.scatter(subset[metric_name], subset['Percent'], 
                       label=f'Week {bin_name}', alpha=0.6, s=40, color=color, edgecolors='black', linewidth=0.5)
    
    ax3.set_xlabel(metric_label, fontsize=12)
    ax3.set_ylabel('FVC Percent (% of predicted)', fontsize=12)
    ax3.set_title('FVC Percent vs Metric (by time period)', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistical significance
    ax4 = axes[1, 1]
    ax4.bar(x_pos, -np.log10(temporal_df['pearson_p']), alpha=0.7, color='coral')
    ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p=0.05')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(temporal_df['week_range'], rotation=45, ha='right')
    ax4.set_ylabel('-log10(p-value)', fontsize=12)
    ax4.set_title('Statistical Significance', fontsize=13, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return temporal_df


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
    wall_df = corr_df[corr_df['type'] == 'airway_wall'].copy()
    parenchymal_df = corr_df[corr_df['type'] == 'parenchymal'].copy()
    
    print("\n" + "="*80)
    print("COMPARISON: AIRWAY vs PERIPHERAL vs WALL vs PARENCHYMAL METRICS")
    print("="*80)
    
    print("\nCORE AIRWAY METRICS:")
    print(f"  Count: {len(airway_df)}")
    if len(airway_df) > 0:
        print(f"  Mean |r|: {airway_df['pearson_r'].abs().mean():.3f}")
        print(f"  Max |r|: {airway_df['pearson_r'].abs().max():.3f}")
        print(f"  Significant (p<0.05): {len(airway_df[airway_df['pearson_p'] < 0.05])}/{len(airway_df)}")
        best_airway = airway_df.loc[airway_df['pearson_r'].abs().idxmax()]
        print(f"  Best: {best_airway['metric']} (r={best_airway['pearson_r']:.3f})")
    
    print("\nPERIPHERAL AIRWAY METRICS (NEW):")
    print(f"  Count: {len(peripheral_df)}")
    if len(peripheral_df) > 0:
        print(f"  Mean |r|: {peripheral_df['pearson_r'].abs().mean():.3f}")
        print(f"  Max |r|: {peripheral_df['pearson_r'].abs().max():.3f}")
        print(f"  Significant (p<0.05): {len(peripheral_df[peripheral_df['pearson_p'] < 0.05])}/{len(peripheral_df)}")
        best_peripheral = peripheral_df.loc[peripheral_df['pearson_r'].abs().idxmax()]
        print(f"  Best: {best_peripheral['metric']} (r={best_peripheral['pearson_r']:.3f})")
    
    print("\nWALL THICKNESS METRICS:")
    print(f"  Count: {len(wall_df)}")
    if len(wall_df) > 0:
        print(f"  Mean |r|: {wall_df['pearson_r'].abs().mean():.3f}")
        print(f"  Max |r|: {wall_df['pearson_r'].abs().max():.3f}")
        print(f"  Significant (p<0.05): {len(wall_df[wall_df['pearson_p'] < 0.05])}/{len(wall_df)}")
        best_wall = wall_df.loc[wall_df['pearson_r'].abs().idxmax()]
        print(f"  Best: {best_wall['metric']} (r={best_wall['pearson_r']:.3f})")
        print(f"  ⚠️ WARNING: Pi10/PiSlope calculation should be verified against paper definition!")
    
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
    if len(wall_df) > 0:
        all_metrics.append(('Wall Thickness', wall_df['pearson_r'].abs().mean()))
    if len(parenchymal_df) > 0:
        all_metrics.append(('Parenchymal', parenchymal_df['pearson_r'].abs().mean()))
    
    if all_metrics:
        all_metrics.sort(key=lambda x: x[1], reverse=True)
        print(f"\nMean |r| ranking:")
        for i, (name, mean_r) in enumerate(all_metrics, 1):
            print(f"  {i}. {name}: {mean_r:.3f}")
    
    print("\n" + "="*80)


def analyze_pc_ratio_problem(df):
    """
    Analizza perché PC ratio ha bassa correlazione nonostante sia teoricamente importante
    """
    print("\n" + "="*80)
    print("SPECIAL ANALYSIS: WHY IS PC RATIO CORRELATION LOW?")
    print("="*80)
    
    if 'pc_ratio' not in df.columns or 'Percent' not in df.columns:
        print("  PC ratio or FVC% not available")
        return
    
    # Baseline data only (week 0)
    baseline = df[df['week'] == 0].copy()
    
    if len(baseline) == 0:
        print("  No baseline (week 0) data available")
        return
    
    # Statistics
    pc_clean = baseline['pc_ratio'].dropna()
    fvc_clean = baseline['Percent'].dropna()
    
    print(f"\nBaseline (Week 0) Statistics:")
    print(f"  Patients: {len(baseline)}")
    print(f"\nPC Ratio distribution:")
    print(f"  Mean: {pc_clean.mean():.3f}")
    print(f"  Median: {pc_clean.median():.3f}")
    print(f"  Std: {pc_clean.std():.3f}")
    print(f"  Range: {pc_clean.min():.3f} - {pc_clean.max():.3f}")
    print(f"  CV: {pc_clean.std() / pc_clean.mean():.3f}")
    
    print(f"\nFVC% distribution:")
    print(f"  Mean: {fvc_clean.mean():.1f}%")
    print(f"  Median: {fvc_clean.median():.1f}%")
    print(f"  Range: {fvc_clean.min():.1f}% - {fvc_clean.max():.1f}%")
    
    # Check for restricted range
    if pc_clean.std() / pc_clean.mean() < 0.3:
        print(f"\n⚠️ PROBLEM IDENTIFIED: Low variability in PC ratio (CV < 0.3)")
        print(f"   → In advanced fibrosis, all patients have low PC ratio")
        print(f"   → Limited dynamic range reduces correlation power")
    
    # Compare with volume metrics
    if 'total_volume_mm3' in baseline.columns and 'peripheral_volume_mm3' in baseline.columns:
        vol_clean = baseline[['total_volume_mm3', 'peripheral_volume_mm3', 'Percent', 'pc_ratio']].dropna()
        
        if len(vol_clean) > 10:
            # Correlations
            r_total_fvc, p_total = pearsonr(vol_clean['total_volume_mm3'], vol_clean['Percent'])
            r_periph_fvc, p_periph = pearsonr(vol_clean['peripheral_volume_mm3'], vol_clean['Percent'])
            r_pc_fvc, p_pc = pearsonr(vol_clean['pc_ratio'], vol_clean['Percent'])
            
            print(f"\n" + "-"*80)
            print("COMPARISON: Ratio vs Absolute Volumes")
            print("-"*80)
            print(f"  Total volume vs FVC%:      r={r_total_fvc:.3f}, p={p_total:.4f}")
            print(f"  Peripheral volume vs FVC%: r={r_periph_fvc:.3f}, p={p_periph:.4f}")
            print(f"  PC ratio vs FVC%:          r={r_pc_fvc:.3f}, p={p_pc:.4f}")
            
            if abs(r_total_fvc) > abs(r_pc_fvc) * 1.5:
                print(f"\n💡 INSIGHT: Total volume has {abs(r_total_fvc)/abs(r_pc_fvc):.1f}x stronger correlation!")
                print(f"   → Absolute volume loss is better predictor than ratio")
                print(f"   → Fibrosis affects both central AND peripheral airways")
    
    # Test normalization
    if 'total_volume_mm3' in baseline.columns:
        baseline_test = baseline[['pc_ratio', 'total_volume_mm3', 'Percent']].dropna()
        
        if len(baseline_test) > 10:
            # Normalize PC ratio by total volume
            baseline_test['pc_ratio_normalized'] = baseline_test['pc_ratio'] * baseline_test['total_volume_mm3'] / 20000  # 20000 mm³ = typical normal
            
            r_norm, p_norm = pearsonr(baseline_test['pc_ratio_normalized'], baseline_test['Percent'])
            r_orig, p_orig = pearsonr(baseline_test['pc_ratio'], baseline_test['Percent'])
            
            print(f"\n" + "-"*80)
            print("NORMALIZATION TEST:")
            print("-"*80)
            print(f"  Original PC ratio:    r={r_orig:.3f}, p={p_orig:.4f}")
            print(f"  Normalized PC ratio:  r={r_norm:.3f}, p={p_norm:.4f}")
            
            if abs(r_norm) > abs(r_orig) * 1.2:
                print(f"  ✓ Normalization improves correlation by {abs(r_norm)/abs(r_orig):.1f}x")
            else:
                print(f"  → Normalization doesn't help significantly")
    
    print("\n" + "="*80)


# ============================================================
# NEW: TEMPORAL GAP & DIAMETER REGULARITY ANALYSIS
# ============================================================

def analyze_diameter_regularity_near_baseline(df):
    """
    Analizza la regolarità dei diametri periferici vicino alla settimana 0.
    Diametri irregolari dovrebbero correlare con FVC% più basso.
    """
    print("\n" + "="*80)
    print("DIAMETER REGULARITY ANALYSIS (Near Baseline)")
    print("Hypothesis: Irregular peripheral diameters → Lower FVC%")
    print("="*80)
    
    # Filter for baseline measurements (week 0-10 per gap temporale)
    baseline = df[(df['week'] >= -5) & (df['week'] <= 10)].copy()
    baseline_clean = baseline[['std_peripheral_diameter_mm', 'diameter_cv', 'Percent', 'patient']].dropna()
    
    if len(baseline_clean) < 10:
        print("⚠ Insufficient data near baseline (week 0-10)")
        return None
    
    print(f"\nAnalyzing {len(baseline_clean)} measurements from {baseline_clean['patient'].nunique()} patients")
    print(f"Week range: {baseline['week'].min():.0f} to {baseline['week'].max():.0f}")
    
    # Correlations
    metrics_to_test = {
        'std_peripheral_diameter_mm': 'Std Peripheral Diameter (mm)',
        'diameter_cv': 'Diameter CV (coefficient of variation)'
    }
    
    results = []
    for metric, label in metrics_to_test.items():
        if metric not in baseline_clean.columns:
            continue
        
        metric_clean = baseline_clean[[metric, 'Percent']].dropna()
        if len(metric_clean) < 10:
            continue
        
        r_pearson, p_pearson = pearsonr(metric_clean[metric], metric_clean['Percent'])
        r_spearman, p_spearman = spearmanr(metric_clean[metric], metric_clean['Percent'])
        
        results.append({
            'metric': metric,
            'label': label,
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'n_samples': len(metric_clean)
        })
        
        print(f"\n{label}:")
        print(f"  Pearson:  r={r_pearson:.3f}, p={p_pearson:.4f} {'***' if p_pearson < 0.001 else '**' if p_pearson < 0.01 else '*' if p_pearson < 0.05 else ''}")
        print(f"  Spearman: ρ={r_spearman:.3f}, p={p_spearman:.4f}")
        
        if r_pearson < 0 and p_pearson < 0.05:
            print(f"  ✓ CONFERMATO: Alta variabilità diametri → FVC% ridotto")
        elif abs(r_pearson) < 0.1:
            print(f"  → Correlazione debole/assente")
    
    # Create visualization
    if len(results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, result in enumerate(results[:2]):
            ax = axes[idx] if len(results) > 1 else axes
            metric = result['metric']
            
            metric_data = baseline_clean[[metric, 'Percent']].dropna()
            
            ax.scatter(metric_data[metric], metric_data['Percent'], 
                      alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
            
            # Trend line
            z = np.polyfit(metric_data[metric], metric_data['Percent'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(metric_data[metric].min(), metric_data[metric].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel(result['label'], fontsize=12)
            ax.set_ylabel('FVC Percent (% predicted)', fontsize=12)
            ax.set_title(f"{result['label']} vs FVC%\n(Baseline: Week 0-10)\nr={result['pearson_r']:.3f}, p={result['pearson_p']:.4f}",
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / "diameter_regularity_baseline.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Plot saved: {output_path}")
    
    return pd.DataFrame(results) if results else None


def analyze_normalized_pc_ratio(df):
    """
    Analizza PC ratio normalizzato (% volume periferico sul totale).
    Dovrebbe correlare meglio con FVC% rispetto al ratio grezzo.
    """
    print("\n" + "="*80)
    print("PC RATIO NORMALIZED ANALYSIS")
    print("Comparing: Raw PC ratio vs Normalized (% peripheral volume)")
    print("="*80)
    
    # Use baseline data (week 0-10)
    baseline = df[(df['week'] >= -5) & (df['week'] <= 10)].copy()
    
    # Need both metrics
    required_cols = ['pc_ratio', 'pc_ratio_normalized', 'Percent', 'patient']
    baseline_clean = baseline[required_cols].dropna()
    
    if len(baseline_clean) < 10:
        print("⚠ Insufficient data for PC ratio comparison")
        return None
    
    print(f"\nAnalyzing {len(baseline_clean)} measurements (Week 0-10)")
    
    # Compare correlations
    r_raw, p_raw = pearsonr(baseline_clean['pc_ratio'], baseline_clean['Percent'])
    r_norm, p_norm = pearsonr(baseline_clean['pc_ratio_normalized'], baseline_clean['Percent'])
    
    print(f"\nRAW PC RATIO (peripheral/central volume):")
    print(f"  Pearson: r={r_raw:.3f}, p={p_raw:.4f} {'***' if p_raw < 0.001 else '**' if p_raw < 0.01 else '*' if p_raw < 0.05 else ''}")
    print(f"  Mean: {baseline_clean['pc_ratio'].mean():.3f}")
    print(f"  Range: {baseline_clean['pc_ratio'].min():.3f} - {baseline_clean['pc_ratio'].max():.3f}")
    
    print(f"\nNORMALIZED PC RATIO (% peripheral volume of total):")
    print(f"  Pearson: r={r_norm:.3f}, p={p_norm:.4f} {'***' if p_norm < 0.001 else '**' if p_norm < 0.01 else '*' if p_norm < 0.05 else ''}")
    print(f"  Mean: {baseline_clean['pc_ratio_normalized'].mean():.1f}%")
    print(f"  Range: {baseline_clean['pc_ratio_normalized'].min():.1f}% - {baseline_clean['pc_ratio_normalized'].max():.1f}%")
    
    # Improvement factor
    improvement = abs(r_norm) / abs(r_raw) if abs(r_raw) > 0 else 1.0
    
    print(f"\n{'='*50}")
    if improvement > 1.2:
        print(f"✓ MIGLIORAMENTO: Normalizzazione migliora correlazione di {improvement:.1f}x")
    elif improvement > 0.9:
        print(f"→ COMPARABILE: Entrambe le metriche hanno correlazione simile")
    else:
        print(f"⚠ PEGGIORE: Normalizzazione riduce correlazione")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw PC ratio
    axes[0].scatter(baseline_clean['pc_ratio'], baseline_clean['Percent'],
                    alpha=0.6, s=60, edgecolors='black', linewidth=0.5, color='steelblue')
    z1 = np.polyfit(baseline_clean['pc_ratio'], baseline_clean['Percent'], 1)
    p1 = np.poly1d(z1)
    x1 = np.linspace(baseline_clean['pc_ratio'].min(), baseline_clean['pc_ratio'].max(), 100)
    axes[0].plot(x1, p1(x1), "r--", alpha=0.8, linewidth=2)
    axes[0].set_xlabel('PC Ratio (peripheral/central)', fontsize=12)
    axes[0].set_ylabel('FVC Percent (% predicted)', fontsize=12)
    axes[0].set_title(f"Raw PC Ratio vs FVC%\nr={r_raw:.3f}, p={p_raw:.4f}", fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Normalized PC ratio
    axes[1].scatter(baseline_clean['pc_ratio_normalized'], baseline_clean['Percent'],
                    alpha=0.6, s=60, edgecolors='black', linewidth=0.5, color='mediumorchid')
    z2 = np.polyfit(baseline_clean['pc_ratio_normalized'], baseline_clean['Percent'], 1)
    p2 = np.poly1d(z2)
    x2 = np.linspace(baseline_clean['pc_ratio_normalized'].min(), baseline_clean['pc_ratio_normalized'].max(), 100)
    axes[1].plot(x2, p2(x2), "r--", alpha=0.8, linewidth=2)
    axes[1].set_xlabel('PC Ratio Normalized (% peripheral volume)', fontsize=12)
    axes[1].set_ylabel('FVC Percent (% predicted)', fontsize=12)
    axes[1].set_title(f"Normalized PC Ratio vs FVC%\nr={r_norm:.3f}, p={p_norm:.4f}", fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "pc_ratio_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot saved: {output_path}")
    
    return {
        'raw_r': r_raw,
        'raw_p': p_raw,
        'norm_r': r_norm,
        'norm_p': p_norm,
        'improvement_factor': improvement
    }


def analyze_wall_thickness(df):
    """
    Analizza lo spessore della parete (wall thickness proxy) e correlazione con FVC%.
    In fibrosi, lo spessore parete aumenta.
    """
    print("\n" + "="*80)
    print("WALL THICKNESS ANALYSIS")
    print("Hypothesis: Increased wall thickness → Lower FVC% (fibrosis)")
    print("="*80)
    
    # Use baseline data
    baseline = df[(df['week'] >= -5) & (df['week'] <= 10)].copy()
    baseline_clean = baseline[['wall_thickness_proxy', 'Percent', 'patient']].dropna()
    
    if len(baseline_clean) < 10:
        print("⚠ Insufficient data for wall thickness analysis")
        return None
    
    print(f"\nAnalyzing {len(baseline_clean)} measurements (Week 0-10)")
    print(f"Patients: {baseline_clean['patient'].nunique()}")
    
    # Correlations
    r_pearson, p_pearson = pearsonr(baseline_clean['wall_thickness_proxy'], baseline_clean['Percent'])
    r_spearman, p_spearman = spearmanr(baseline_clean['wall_thickness_proxy'], baseline_clean['Percent'])
    
    print(f"\nWall Thickness Proxy (diameter/length ratio):")
    print(f"  Pearson:  r={r_pearson:.3f}, p={p_pearson:.4f} {'***' if p_pearson < 0.001 else '**' if p_pearson < 0.01 else '*' if p_pearson < 0.05 else ''}")
    print(f"  Spearman: ρ={r_spearman:.3f}, p={p_spearman:.4f}")
    print(f"  Mean: {baseline_clean['wall_thickness_proxy'].mean():.4f}")
    print(f"  Range: {baseline_clean['wall_thickness_proxy'].min():.4f} - {baseline_clean['wall_thickness_proxy'].max():.4f}")
    
    if r_pearson < 0 and p_pearson < 0.05:
        print(f"\n✓ CONFERMATO: Spessore parete aumentato → FVC% ridotto (fibrosi)")
    elif abs(r_pearson) < 0.1:
        print(f"\n→ Correlazione debole: spessore parete non è predittore forte")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(baseline_clean['wall_thickness_proxy'], baseline_clean['Percent'],
               alpha=0.6, s=60, edgecolors='black', linewidth=0.5, color='darkorange')
    
    # Trend line
    z = np.polyfit(baseline_clean['wall_thickness_proxy'], baseline_clean['Percent'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(baseline_clean['wall_thickness_proxy'].min(), 
                         baseline_clean['wall_thickness_proxy'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel('Wall Thickness Proxy (diameter/length ratio)', fontsize=12)
    ax.set_ylabel('FVC Percent (% predicted)', fontsize=12)
    ax.set_title(f"Wall Thickness vs FVC% (Baseline: Week 0-10)\nr={r_pearson:.3f}, p={p_pearson:.4f}",
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "wall_thickness_vs_fvc.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot saved: {output_path}")
    
    return {
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman,
        'n_samples': len(baseline_clean)
    }


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


def validate_fibrosis_score_correlation(df):
    """
    Valida ENTRAMBI i fibrosis scores correlando con FVC%.
    Confronta: AIRWAY_ONLY (Opzione 1) vs COMBINED (Opzione 2)
    """
    print("\n" + "="*80)
    print("DUAL FIBROSIS SCORE VALIDATION")
    print("Comparing: AIRWAY_ONLY vs COMBINED scoring methods")
    print("="*80)
    
    # Load both fibrosis scores
    fibrosis_scores_airway = {}
    fibrosis_scores_combined = {}
    
    for case_name in df['case'].unique():
        # Try to load fibrosis assessment
        json_path = RESULTS_ROOT / case_name / "step6_fibrosis_assessment" / "fibrosis_assessment.json"
        
        if not json_path.exists():
            continue
        
        try:
            with open(json_path, 'r') as f:
                assessment = json.load(f)
                
                # Get airway_only score
                if 'scoring_methods' in assessment and 'airway_only' in assessment['scoring_methods']:
                    score_airway = assessment['scoring_methods']['airway_only']['fibrosis_score']
                    fibrosis_scores_airway[case_name] = score_airway
                
                # Get combined score
                if 'scoring_methods' in assessment and 'combined' in assessment['scoring_methods']:
                    score_combined = assessment['scoring_methods']['combined']['fibrosis_score']
                    if score_combined is not None:
                        fibrosis_scores_combined[case_name] = score_combined
        except Exception as e:
            continue
    
    if len(fibrosis_scores_airway) == 0:
        print("⚠ No fibrosis scores found")
        return None
    
    print(f"\nScores available:")
    print(f"  Airway-only: {len(fibrosis_scores_airway)} cases")
    print(f"  Combined: {len(fibrosis_scores_combined)} cases")
    
    # Analyze AIRWAY_ONLY score
    print("\n" + "="*80)
    print("AIRWAY_ONLY SCORE (Opzione 1: Pure Airway Morphometry)")
    print("="*80)
    
    df_airway = df.copy()
    df_airway['fibrosis_score'] = df_airway['case'].map(fibrosis_scores_airway)
    df_airway = df_airway.dropna(subset=['fibrosis_score', 'Percent'])
    
    result_airway = None
    if len(df_airway) >= 10:
        r_airway, p_airway = pearsonr(df_airway['fibrosis_score'], df_airway['Percent'])
        r_spear_airway, p_spear_airway = spearmanr(df_airway['fibrosis_score'], df_airway['Percent'])
        
        print(f"\nFIBROSIS SCORE (Airway-only) vs FVC%:")
        print(f"  Pearson:  r={r_airway:.3f}, p={p_airway:.4f} {'***' if p_airway < 0.001 else '**' if p_airway < 0.01 else '*' if p_airway < 0.05 else ''}")
        print(f"  Spearman: ρ={r_spear_airway:.3f}, p={p_spear_airway:.4f}")
        print(f"  Score range: {df_airway['fibrosis_score'].min():.1f} - {df_airway['fibrosis_score'].max():.1f}")
        print(f"  Measurements: {len(df_airway)}, Patients: {df_airway['patient'].nunique()}")
        
        result_airway = {
            'correlation_r': r_airway,
            'correlation_p': p_airway,
            'n_samples': len(df_airway),
            'n_patients': df_airway['patient'].nunique()
        }
    
    # Analyze COMBINED score
    print("\n" + "="*80)
    print("COMBINED SCORE (Opzione 2: Airway + Parenchymal) - RECOMMENDED")
    print("="*80)
    
    df_combined = df.copy()
    df_combined['fibrosis_score'] = df_combined['case'].map(fibrosis_scores_combined)
    df_combined = df_combined.dropna(subset=['fibrosis_score', 'Percent'])
    
    result_combined = None
    if len(df_combined) >= 10:
        r_combined, p_combined = pearsonr(df_combined['fibrosis_score'], df_combined['Percent'])
        r_spear_combined, p_spear_combined = spearmanr(df_combined['fibrosis_score'], df_combined['Percent'])
        
        print(f"\nFIBROSIS SCORE (Combined) vs FVC%:")
        print(f"  Pearson:  r={r_combined:.3f}, p={p_combined:.4f} {'***' if p_combined < 0.001 else '**' if p_combined < 0.01 else '*' if p_combined < 0.05 else ''}")
        print(f"  Spearman: ρ={r_spear_combined:.3f}, p={p_spear_combined:.4f}")
        print(f"  Score range: {df_combined['fibrosis_score'].min():.1f} - {df_combined['fibrosis_score'].max():.1f}")
        print(f"  Measurements: {len(df_combined)}, Patients: {df_combined['patient'].nunique()}")
        
        if r_combined < -0.5 and p_combined < 0.001:
            print(f"\n  ✓✓✓ EXCELLENT: Strong negative correlation (high score → low FVC%)")
        elif r_combined < -0.3 and p_combined < 0.01:
            print(f"\n  ✓✓ GOOD: Moderate negative correlation")
        elif r_combined < -0.15 and p_combined < 0.05:
            print(f"\n  ✓ ACCEPTABLE: Weak but significant correlation")
        else:
            print(f"\n  ⚠ NEEDS IMPROVEMENT: Correlation not strong enough")
        
        result_combined = {
            'correlation_r': r_combined,
            'correlation_p': p_combined,
            'n_samples': len(df_combined),
            'n_patients': df_combined['patient'].nunique()
        }
    
    # COMPARISON
    if result_airway and result_combined:
        print("\n" + "="*80)
        print("COMPARISON: AIRWAY_ONLY vs COMBINED")
        print("="*80)
        
        improvement = abs(result_combined['correlation_r']) - abs(result_airway['correlation_r'])
        
        print(f"\nCorrelation Strength:")
        print(f"  Airway-only: |r|={abs(result_airway['correlation_r']):.3f}")
        print(f"  Combined:    |r|={abs(result_combined['correlation_r']):.3f}")
        print(f"  Improvement: {improvement:+.3f} ({improvement/abs(result_airway['correlation_r'])*100:+.1f}%)")
        
        if improvement > 0.15:
            print(f"\n  ✓✓✓ MAJOR IMPROVEMENT: Combined score is SUBSTANTIALLY better!")
        elif improvement > 0.05:
            print(f"\n  ✓✓ GOOD IMPROVEMENT: Combined score performs better")
        elif improvement > 0:
            print(f"\n  ✓ MINOR IMPROVEMENT: Combined score slightly better")
        else:
            print(f"\n  → Similar performance")
    
    # Visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Airway-only score
    ax1 = plt.subplot(1, 3, 1)
    if len(df_airway) > 0:
        scatter1 = ax1.scatter(df_airway['fibrosis_score'], df_airway['Percent'],
                              c=df_airway['week'], cmap='viridis',
                              alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        z1 = np.polyfit(df_airway['fibrosis_score'], df_airway['Percent'], 1)
        p1 = np.poly1d(z1)
        x_trend1 = np.linspace(df_airway['fibrosis_score'].min(), df_airway['fibrosis_score'].max(), 100)
        ax1.plot(x_trend1, p1(x_trend1), "r--", alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Fibrosis Score - Airway-only', fontsize=11)
        ax1.set_ylabel('FVC Percent (% predicted)', fontsize=11)
        ax1.set_title(f'AIRWAY_ONLY Score vs FVC%\nr={result_airway["correlation_r"]:.3f}, p={result_airway["correlation_p"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Week')
    
    # Plot 2: Combined score
    ax2 = plt.subplot(1, 3, 2)
    if len(df_combined) > 0:
        scatter2 = ax2.scatter(df_combined['fibrosis_score'], df_combined['Percent'],
                              c=df_combined['week'], cmap='viridis',
                              alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        z2 = np.polyfit(df_combined['fibrosis_score'], df_combined['Percent'], 1)
        p2 = np.poly1d(z2)
        x_trend2 = np.linspace(df_combined['fibrosis_score'].min(), df_combined['fibrosis_score'].max(), 100)
        ax2.plot(x_trend2, p2(x_trend2), "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Fibrosis Score - Combined', fontsize=11)
        ax2.set_ylabel('FVC Percent (% predicted)', fontsize=11)
        ax2.set_title(f'COMBINED Score vs FVC%\nr={result_combined["correlation_r"]:.3f}, p={result_combined["correlation_p"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Week')
    
    # Plot 3: Comparison bar chart
    ax3 = plt.subplot(1, 3, 3)
    if result_airway and result_combined:
        methods = ['Airway-only\n(Opzione 1)', 'Combined\n(Opzione 2)']
        correlations = [abs(result_airway['correlation_r']), abs(result_combined['correlation_r'])]
        colors = ['#3498db', '#27ae60']
        
        bars = ax3.bar(methods, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, val in zip(bars, correlations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax3.set_ylabel('|Pearson Correlation|', fontsize=11)
        ax3.set_title('Correlation Strength Comparison', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, max(correlations) * 1.2)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add significance indicators
        ax3.text(0.5, max(correlations) * 1.15, 
                f'Improvement: {improvement:+.3f}',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fibrosis_score_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison plot saved: {output_path}")
    
    # Save results
    comparison_results = {
        'airway_only': result_airway,
        'combined': result_combined
    }
    
    json_path = OUTPUT_DIR / "fibrosis_score_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"✓ Comparison results saved: {json_path}")
    
    # Return combined result (preferred) or airway if combined not available
    return result_combined if result_combined else result_airway
    
    # Plot 2: Decline by score quartile
    if len(decline_score_df) >= 3:
        ax2 = axes[1]
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#c0392b']
        x_pos = np.arange(len(decline_score_df))
        
        bars = ax2.bar(x_pos, decline_score_df['mean_decline'],
                      yerr=decline_score_df['std_decline'],
                      color=colors[:len(decline_score_df)], alpha=0.7,
                      capsize=5, edgecolor='black', linewidth=1)
        
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(decline_score_df['quartile'], rotation=45, ha='right')
        ax2.set_ylabel('FVC% Decline Rate (%/week)', fontsize=12)
        ax2.set_title('FVC Decline by Fibrosis Score Quartile', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data\nfor decline analysis', 
                ha='center', va='center', fontsize=14)
        ax2.axis('off')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "fibrosis_score_validation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot saved: {output_path}")
    
    return {
        'correlation_r': r_all,
        'correlation_p': p_all,
        'n_samples': len(df_with_scores),
        'n_patients': df_with_scores['patient'].nunique(),
        'decline_by_score': decline_score_df
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("="*80)
    print("OSIC METRICS vs FVC PERCENT ANALYSIS")
    print("Airway + Parenchymal Metrics")
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
    print(f"\nAll airway metrics: measured at week 0 (baseline CT)")
    print(f"FVC measurements: week {df['week'].min()} to {df['week'].max()}")
    print(f"Median follow-up: {df['week'].median():.0f} weeks")
    print(f"Total records: {len(df)}")
    print(f"Unique patients: {df['patient'].nunique()}\n")
    
    # === PART 1: FVC vs METRIC CORRELATIONS ===
    print("="*80)
    print("PART 1: METRICS vs FVC PERCENT - CORRELATION ANALYSIS")
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
    
    # Special analysis for PC ratio (understand why it has low correlation)
    analyze_pc_ratio_problem(df)
    
    # === NEW: ADDITIONAL SPECIALIZED ANALYSES ===
    print("\n" + "="*80)
    print("PART 1b: SPECIALIZED ANALYSES (Near Baseline)")
    print("="*80)
    
    # 1. Diameter regularity analysis
    diameter_reg_results = analyze_diameter_regularity_near_baseline(df)
    if diameter_reg_results is not None:
        diameter_reg_path = OUTPUT_DIR / "diameter_regularity_results.csv"
        diameter_reg_results.to_csv(diameter_reg_path, index=False)
        print(f"✓ Diameter regularity results saved to: {diameter_reg_path}")
    
    # 2. PC ratio normalized vs raw comparison
    pc_norm_results = analyze_normalized_pc_ratio(df)
    if pc_norm_results is not None:
        with open(OUTPUT_DIR / "pc_ratio_comparison.json", 'w') as f:
            import json
            json.dump(pc_norm_results, f, indent=2)
        print(f"✓ PC ratio comparison saved")
    
    # 3. Wall thickness analysis
    wall_results = analyze_wall_thickness(df)
    if wall_results is not None:
        with open(OUTPUT_DIR / "wall_thickness_results.json", 'w') as f:
            import json
            json.dump(wall_results, f, indent=2)
        print(f"✓ Wall thickness results saved")
    
    # === NEW: FVC EVOLUTION ANALYSIS ===
    print("\n" + "="*80)
    print("PART 1c: FVC EVOLUTION - Progression vs Stability Prediction")
    print("="*80)
    
    # Analyze evolution for key metrics
    evolution_metrics = [
        ('diameter_cv', 'Diameter Coefficient of Variation'),
        ('std_peripheral_diameter_mm', 'Std Peripheral Diameter (mm)'),
        ('wall_thickness_proxy', 'Wall Thickness Proxy'),
        ('mean_tortuosity', 'Mean Tortuosity'),
        ('pc_ratio_normalized', 'PC Ratio Normalized (%)'),
    ]
    
    evolution_results = {}
    
    for metric, label in evolution_metrics:
        if metric not in df.columns:
            continue
        
        output_path = OUTPUT_DIR / f"evolution_{metric}.png"
        result = analyze_fvc_evolution_by_metric(df, metric, label, output_path)
        
        if result is not None:
            evolution_results[metric] = result
            result_path = OUTPUT_DIR / f"evolution_{metric}.csv"
            result.to_csv(result_path, index=False)
            print(f"✓ Saved: {result_path}")
    
    # === NEW: FIBROSIS SCORE VALIDATION ===
    print("\n" + "="*80)
    print("PART 1d: FIBROSIS SCORE VALIDATION")
    print("="*80)
    
    fibrosis_validation = validate_fibrosis_score_correlation(df)
    if fibrosis_validation is not None:
        with open(OUTPUT_DIR / "fibrosis_score_validation.json", 'w') as f:
            import json
            # Convert DataFrame to dict for JSON serialization
            validation_copy = fibrosis_validation.copy()
            if 'decline_by_score' in validation_copy and isinstance(validation_copy['decline_by_score'], pd.DataFrame):
                validation_copy['decline_by_score'] = validation_copy['decline_by_score'].to_dict(orient='records')
            json.dump(validation_copy, f, indent=2)
        print(f"✓ Fibrosis score validation saved")
    
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
    
    metrics_to_analyze = [
        # AIRWAY METRICS
        ('volume_ml', 'Airway Volume (ml)'),
        ('branch_count', 'Branch Count'),
        ('max_generation', 'Max Generation'),
        ('pc_ratio', 'PC Ratio'),
        ('tapering_ratio', 'Tapering Ratio'),
        ('mean_tortuosity', 'Mean Tortuosity'),
        
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
    
    # === FINAL SUMMARY ===
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    significant = corr_sorted[corr_sorted['pearson_p'] < 0.05]
    
    print("\n1. STRONGEST CORRELATIONS (All Metrics vs FVC Percent):")
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
            parench_sig = significant[significant['type'] == 'parenchymal']
            print(f"   - Airway: {len(airway_sig)}")
            print(f"   - Parenchymal: {len(parench_sig)}")
        print("\n   Top significant metrics:")
        for idx, row in significant.head(10).iterrows():
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
    
    print("\n5. FIBROSIS SCORE VALIDATION:")
    if fibrosis_validation is not None:
        r = fibrosis_validation['correlation_r']
        p = fibrosis_validation['correlation_p']
        
        # Check if we have comparison data (both scores)
        comparison_path = OUTPUT_DIR / "fibrosis_score_comparison.json"
        if comparison_path.exists():
            try:
                import json
                with open(comparison_path, 'r') as f:
                    comparison = json.load(f)
                
                airway_r = comparison['airway_only']['correlation_r']
                airway_p = comparison['airway_only']['correlation_p']
                combined_r = comparison['combined']['correlation_r']
                combined_p = comparison['combined']['correlation_p']
                
                print(f"\n   AIRWAY_ONLY (Opzione 1):")
                print(f"      r={airway_r:.3f}, p={airway_p:.4f}")
                
                print(f"\n   COMBINED (Opzione 2) - RECOMMENDED:")
                print(f"      r={combined_r:.3f}, p={combined_p:.4f}")
                
                improvement = abs(combined_r) - abs(airway_r)
                
                if combined_r < -0.5 and combined_p < 0.001:
                    print(f"   ✓✓✓ EXCELLENT VALIDATION: Strong correlation with FVC%")
                    print(f"   → Combined scoring system STRONGLY validated!")
                elif combined_r < -0.3 and combined_p < 0.01:
                    print(f"   ✓✓ VALIDATED: Moderate correlation with FVC%")
                    print(f"   → Scoring system è COERENTE con funzione polmonare")
                elif combined_r < -0.15 and combined_p < 0.05:
                    print(f"   ✓ PARTIAL: Weak but significant correlation")
                else:
                    print(f"   ⚠ NEEDS IMPROVEMENT")
                
                if improvement > 0.15:
                    print(f"\n   🎯 Combined score shows MAJOR improvement over airway-only (+{improvement:.3f})")
                elif improvement > 0.05:
                    print(f"\n   ✓ Combined score performs better than airway-only (+{improvement:.3f})")
                
            except:
                # Fallback to single score display
                if r < -0.3 and p < 0.01:
                    print(f"   ✓✓ VALIDATED: r={r:.3f}, p={p:.4f}")
                elif r < -0.15 and p < 0.05:
                    print(f"   ✓ PARTIAL: r={r:.3f}, p={p:.4f}")
                else:
                    print(f"   ⚠ NEEDS IMPROVEMENT: r={r:.3f}, p={p:.4f}")
        else:
            # Single score display
            if r < -0.3 and p < 0.01:
                print(f"   ✓✓ VALIDATED: r={r:.3f}, p={p:.4f}")
            elif r < -0.15 and p < 0.05:
                print(f"   ✓ PARTIAL: r={r:.3f}, p={p:.4f}")
            else:
                print(f"   ⚠ NEEDS IMPROVEMENT: r={r:.3f}, p={p:.4f}")
    else:
        print("   (Fibrosis scores not available - run scoring pipeline first)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nFiles generated:")
    print(f"  - integrated_dataset.csv: Raw data with all metrics and FVC Percent")
    print(f"  - correlation_results.csv: Correlation coefficients for all metrics")
    print(f"  - correlation_summary.png: Visual comparison of all correlations")
    print(f"  - fibrosis_score_comparison.png: Airway-only vs Combined score comparison")
    print(f"  - [metric]_vs_percent.png: Scatter plots (metric vs FVC Percent)")
    print(f"  - evolution_[metric].png: FVC Percent evolution & decline rate by quartile")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
