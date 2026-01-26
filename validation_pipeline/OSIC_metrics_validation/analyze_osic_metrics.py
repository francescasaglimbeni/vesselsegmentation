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
RESULTS_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC")
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
        # AIRWAY METRICS
        ('volume_ml', 'Airway Volume (ml)', 'airway'),
        ('branch_count', 'Branch Count', 'airway'),
        ('max_generation', 'Max Generation', 'airway'),
        ('pc_ratio', 'Peripheral/Central Ratio', 'airway'),
        ('tapering_ratio', 'Tapering Ratio', 'airway'),
        ('mean_tortuosity', 'Mean Tortuosity', 'airway'),
        
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
    
    # Color by metric type (airway vs parenchymal)
    colors = ['steelblue' if t == 'airway' else 'coral' for t in corr_df.get('type', ['airway']*len(corr_df))]
    
    # Plot 1: Pearson correlation
    ax1.barh(corr_df['metric'], corr_df['pearson_r'], color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    ax1.set_title('Pearson Correlation: All Metrics vs FVC Percent\n(normalized for age/sex/height)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Airway Metrics'),
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
    """Compare airway vs parenchymal metrics performance"""
    
    if 'type' not in corr_df.columns:
        return
    
    airway_df = corr_df[corr_df['type'] == 'airway'].copy()
    parenchymal_df = corr_df[corr_df['type'] == 'parenchymal'].copy()
    
    print("\n" + "="*80)
    print("COMPARISON: AIRWAY vs PARENCHYMAL METRICS")
    print("="*80)
    
    print("\nAIRWAY METRICS:")
    print(f"  Count: {len(airway_df)}")
    if len(airway_df) > 0:
        print(f"  Mean |r|: {airway_df['pearson_r'].abs().mean():.3f}")
        print(f"  Max |r|: {airway_df['pearson_r'].abs().max():.3f}")
        print(f"  Significant (p<0.05): {len(airway_df[airway_df['pearson_p'] < 0.05])}/{len(airway_df)}")
        best_airway = airway_df.loc[airway_df['pearson_r'].abs().idxmax()]
        print(f"  Best: {best_airway['metric']} (r={best_airway['pearson_r']:.3f})")
    
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
    
    print("\n" + "="*80)


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
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"\nFiles generated:")
    print(f"  - integrated_dataset.csv: Raw data with all metrics and FVC Percent")
    print(f"  - correlation_results.csv: Correlation coefficients for all metrics")
    print(f"  - correlation_summary.png: Visual comparison of all correlations")
    print(f"  - [metric]_vs_percent.png: Scatter plots (metric vs FVC Percent)")
    print(f"  - evolution_[metric].png: FVC Percent evolution & decline rate by quartile")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
