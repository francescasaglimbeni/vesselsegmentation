import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class FibrosisConfig:
    """Configuration for fibrosis severity classification"""
    # FVC% thresholds based on clinical guidelines for IPF
    MILD_THRESHOLD = 80.0      # FVC% >= 80 → Mild fibrosis
    MODERATE_THRESHOLD = 50.0  # FVC% 50-80 → Moderate fibrosis
    # FVC% < 50 → Severe fibrosis
    
    # Expected correlations (clinical literature)
    # When FVC decreases (worsening fibrosis):
    # - PC ratio should decrease (peripheral loss)
    # - Tortuosity may increase (airway distortion)
    # - Branch count should decrease (loss of peripheral branches)
    # - Airway volume may decrease (overall airway loss)
    
    # Tolerance for week 0 estimation (weeks)
    MAX_WEEK_DISTANCE = 8  # Use measurements within ±8 weeks for interpolation


# ============================================================
# FVC DATA LOADING AND PROCESSING
# ============================================================

def load_fvc_data(train_csv_path, test_csv_path):
    """Load and combine FVC data from train and test CSV files"""
    
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Combine both datasets
    fvc_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"✓ Loaded FVC data:")
    print(f"  Train: {len(train_df)} measurements")
    print(f"  Test: {len(test_df)} measurements")
    print(f"  Total: {len(fvc_df)} measurements")
    print(f"  Unique patients: {fvc_df['Patient'].nunique()}")
    
    return fvc_df


def estimate_fvc_at_week0(patient_data, config=FibrosisConfig()):
    """
    Estimate FVC at week 0 (CT scan time) from available measurements
    
    Strategy:
    1. If exact week 0 exists → use it
    2. If measurements around week 0 → interpolate
    3. If only later/earlier → use closest or extrapolate with caution
    
    Returns: (fvc_value, fvc_percent, method_used, confidence)
    """
    
    if len(patient_data) == 0:
        return None, None, "no_data", 0.0
    
    weeks = patient_data['Weeks'].values
    fvc = patient_data['FVC'].values
    fvc_percent = patient_data['Percent'].values
    
    # Case 1: Exact match at week 0
    week0_mask = weeks == 0
    if week0_mask.any():
        idx = np.where(week0_mask)[0][0]
        return fvc[idx], fvc_percent[idx], "exact", 1.0
    
    # Case 2: Measurements on both sides of week 0 (interpolation)
    before_mask = weeks < 0
    after_mask = weeks > 0
    
    if before_mask.any() and after_mask.any():
        # Get closest measurements before and after
        before_weeks = weeks[before_mask]
        before_fvc = fvc[before_mask]
        before_percent = fvc_percent[before_mask]
        
        after_weeks = weeks[after_mask]
        after_fvc = fvc[after_mask]
        after_percent = fvc_percent[after_mask]
        
        # Get the closest point on each side
        idx_before = np.argmax(before_weeks)  # Closest to 0 (least negative)
        idx_after = np.argmin(after_weeks)    # Closest to 0 (smallest positive)
        
        week_before = before_weeks[idx_before]
        week_after = after_weeks[idx_after]
        
        # Check if points are within acceptable range
        if abs(week_before) <= config.MAX_WEEK_DISTANCE and week_after <= config.MAX_WEEK_DISTANCE:
            # Linear interpolation
            fvc_interp = np.interp(0, [week_before, week_after], 
                                   [before_fvc[idx_before], after_fvc[idx_after]])
            percent_interp = np.interp(0, [week_before, week_after],
                                       [before_percent[idx_before], after_percent[idx_after]])
            
            # Confidence based on distance between measurements
            max_gap = week_after - week_before
            confidence = max(0.5, 1.0 - max_gap / (config.MAX_WEEK_DISTANCE * 2))
            
            return fvc_interp, percent_interp, "interpolated", confidence
    
    # Case 3: Only measurements on one side → use closest
    abs_weeks = np.abs(weeks)
    closest_idx = np.argmin(abs_weeks)
    closest_week = weeks[closest_idx]
    
    if abs(closest_week) <= config.MAX_WEEK_DISTANCE:
        confidence = max(0.3, 1.0 - abs(closest_week) / config.MAX_WEEK_DISTANCE)
        method = "closest_before" if closest_week < 0 else "closest_after"
        return fvc[closest_idx], fvc_percent[closest_idx], method, confidence
    
    # Case 4: All measurements far from week 0 → use trend if multiple points
    if len(patient_data) >= 3:
        # Linear regression to estimate trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, fvc_percent)
        
        fvc_percent_est = intercept  # Estimate at week 0
        
        # Estimate absolute FVC (approximate)
        slope_abs, intercept_abs, _, _, _ = stats.linregress(weeks, fvc)
        fvc_est = intercept_abs
        
        # Confidence based on R² and distance to nearest point
        r_squared = r_value ** 2
        min_distance = abs_weeks.min()
        confidence = max(0.1, r_squared * (1.0 - min_distance / 50))  # Decay with distance
        
        return fvc_est, fvc_percent_est, "extrapolated", confidence
    
    # Case 5: Insufficient data
    return fvc[closest_idx], fvc_percent[closest_idx], "insufficient_data", 0.1


def classify_fibrosis_severity(fvc_percent, config=FibrosisConfig()):
    """Classify fibrosis severity based on FVC%"""
    if fvc_percent is None or np.isnan(fvc_percent):
        return "unknown"
    
    if fvc_percent >= config.MILD_THRESHOLD:
        return "mild"
    elif fvc_percent >= config.MODERATE_THRESHOLD:
        return "moderate"
    else:
        return "severe"


def process_fvc_for_all_patients(fvc_df, case_list):
    """
    Process FVC data for all cases in the pipeline output
    
    Returns DataFrame with: Patient, FVC_week0, FVC_percent_week0, Severity, Estimation_method, Confidence
    """
    
    results = []
    
    for case_name in case_list:
        # Extract patient ID from case name (format: ID00xxxxx...)
        patient_id = case_name
        
        # Get all measurements for this patient
        patient_data = fvc_df[fvc_df['Patient'] == patient_id]
        
        if len(patient_data) == 0:
            # No FVC data for this patient
            results.append({
                'case': case_name,
                'patient_id': patient_id,
                'fvc_week0': None,
                'fvc_percent_week0': None,
                'severity': 'unknown',
                'estimation_method': 'no_data',
                'confidence': 0.0,
                'num_measurements': 0
            })
            continue
        
        # Estimate FVC at week 0
        fvc_val, fvc_pct, method, conf = estimate_fvc_at_week0(patient_data)
        
        # Classify severity
        severity = classify_fibrosis_severity(fvc_pct)
        
        results.append({
            'case': case_name,
            'patient_id': patient_id,
            'fvc_week0': fvc_val,
            'fvc_percent_week0': fvc_pct,
            'severity': severity,
            'estimation_method': method,
            'confidence': conf,
            'num_measurements': len(patient_data),
            'age': patient_data['Age'].iloc[0] if 'Age' in patient_data.columns else None,
            'sex': patient_data['Sex'].iloc[0] if 'Sex' in patient_data.columns else None,
            'smoking_status': patient_data['SmokingStatus'].iloc[0] if 'SmokingStatus' in patient_data.columns else None
        })
    
    return pd.DataFrame(results)


# ============================================================
# PIPELINE RESULTS LOADING
# ============================================================

def load_pipeline_metrics(case_dir):
    """Load pipeline metrics for a single case"""
    case_dir = Path(case_dir)
    
    metrics_path = case_dir / "step4_analysis" / "advanced_metrics.json"
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Extract key fibrosis indicators
    result = {
        'pc_ratio': metrics.get('peripheral_to_central_ratio', np.nan),
        'tortuosity': metrics.get('mean_tortuosity', np.nan),
        'airway_volume_ml': metrics.get('total_volume_mm3', 0) / 1000,
    }
    
    # Load branch count if available
    branches_path = case_dir / "step4_analysis" / "branch_metrics_complete.csv"
    if branches_path.exists():
        branches_df = pd.read_csv(branches_path)
        result['branch_count'] = len(branches_df)
    else:
        result['branch_count'] = np.nan
    
    # Load max generation
    weibel_path = case_dir / "step4_analysis" / "weibel_generation_analysis.csv"
    if weibel_path.exists():
        weibel_df = pd.read_csv(weibel_path)
        result['max_generation'] = int(weibel_df['generation'].max()) if len(weibel_df) > 0 else np.nan
    else:
        result['max_generation'] = np.nan
    
    # Load fibrosis score if available
    fibrosis_path = case_dir / "step6_fibrosis_assessment" / "fibrosis_assessment.json"
    if fibrosis_path.exists():
        with open(fibrosis_path, 'r') as f:
            fibrosis = json.load(f)
            result['fibrosis_score'] = fibrosis.get('overall', {}).get('fibrosis_score', np.nan)
            result['fibrosis_stage'] = fibrosis.get('overall', {}).get('stage', 'unknown')
            result['fibrosis_confidence'] = fibrosis.get('overall', {}).get('confidence', np.nan)
    else:
        result['fibrosis_score'] = np.nan
        result['fibrosis_stage'] = 'unknown'
        result['fibrosis_confidence'] = np.nan
    
    return result


def load_all_pipeline_results(output_root):
    """Load pipeline metrics for all cases"""
    output_root = Path(output_root)
    
    results = []
    
    for case_dir in sorted(output_root.iterdir()):
        if not case_dir.is_dir():
            continue
        
        case_name = case_dir.name
        metrics = load_pipeline_metrics(case_dir)
        
        if metrics is None:
            continue
        
        metrics['case'] = case_name
        results.append(metrics)
    
    return pd.DataFrame(results)


# ============================================================
# VALIDATION AND CORRELATION ANALYSIS
# ============================================================

def validate_fvc_coherence(merged_df):
    """
    Validate coherence between FVC severity and CT metrics
    
    Expected patterns for worsening fibrosis (decreasing FVC%):
    - PC ratio should decrease
    - Tortuosity may increase
    - Branch count should decrease
    - Airway volume may decrease
    """
    
    results = []
    
    for idx, row in merged_df.iterrows():
        if row['severity'] == 'unknown' or row['confidence'] < 0.3:
            results.append({
                'case': row['case'],
                'coherence_status': 'insufficient_data',
                'flags': ['low_confidence_or_no_fvc']
            })
            continue
        
        flags = []
        severity = row['severity']
        pc_ratio = row['pc_ratio']
        tortuosity = row['tortuosity']
        branch_count = row['branch_count']
        
        # Expected ranges based on severity
        if severity == 'severe':
            # Severe fibrosis: expect low PC ratio, high tortuosity, low branch count
            if not np.isnan(pc_ratio) and pc_ratio > 0.4:
                flags.append('pc_ratio_high_for_severe')
            if not np.isnan(branch_count) and branch_count > 2000:
                flags.append('branch_count_high_for_severe')
        
        elif severity == 'moderate':
            # Moderate: expect intermediate values
            if not np.isnan(pc_ratio) and pc_ratio > 0.6:
                flags.append('pc_ratio_high_for_moderate')
            if not np.isnan(pc_ratio) and pc_ratio < 0.15:
                flags.append('pc_ratio_low_for_moderate')
        
        elif severity == 'mild':
            # Mild: expect relatively normal values
            if not np.isnan(pc_ratio) and pc_ratio < 0.25:
                flags.append('pc_ratio_low_for_mild')
            if not np.isnan(branch_count) and branch_count < 800:
                flags.append('branch_count_low_for_mild')
        
        # Determine overall coherence
        if len(flags) == 0:
            coherence = 'coherent'
        elif len(flags) <= 1:
            coherence = 'partially_coherent'
        else:
            coherence = 'incoherent'
        
        results.append({
            'case': row['case'],
            'coherence_status': coherence,
            'flags': flags if len(flags) > 0 else ['none']
        })
    
    return pd.DataFrame(results)


def compute_correlations(merged_df):
    """Compute correlation statistics between FVC% and CT metrics"""
    
    # Filter valid data (sufficient confidence)
    valid_df = merged_df[
        (merged_df['confidence'] >= 0.3) & 
        (merged_df['fvc_percent_week0'].notna())
    ].copy()
    
    if len(valid_df) < 5:
        print("⚠ WARNING: Insufficient valid cases for correlation analysis (n < 5)")
        return None
    
    correlations = {}
    metrics = ['pc_ratio', 'tortuosity', 'branch_count', 'airway_volume_ml', 'max_generation', 'fibrosis_score']
    
    for metric in metrics:
        valid_metric_df = valid_df[valid_df[metric].notna()]
        
        if len(valid_metric_df) < 5:
            correlations[metric] = {
                'pearson_r': np.nan,
                'pearson_p': np.nan,
                'spearman_r': np.nan,
                'spearman_p': np.nan,
                'n_samples': len(valid_metric_df)
            }
            continue
        
        # Pearson correlation (linear)
        pearson_r, pearson_p = stats.pearsonr(
            valid_metric_df['fvc_percent_week0'],
            valid_metric_df[metric]
        )
        
        # Spearman correlation (monotonic, robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(
            valid_metric_df['fvc_percent_week0'],
            valid_metric_df[metric]
        )
        
        correlations[metric] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(valid_metric_df),
            'interpretation': interpret_correlation(pearson_r, pearson_p, metric)
        }
    
    return correlations


def interpret_correlation(r, p, metric_name):
    """Interpret correlation coefficient with clinical context"""
    
    if np.isnan(r):
        return "insufficient_data"
    
    if p > 0.05:
        return "not_significant"
    
    # Expected correlations (clinical hypothesis)
    expected_positive = ['pc_ratio', 'branch_count', 'airway_volume_ml', 'max_generation']
    expected_negative = ['tortuosity', 'fibrosis_score']
    
    abs_r = abs(r)
    
    # Strength interpretation
    if abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very_strong"
    
    # Direction check
    if metric_name in expected_positive:
        if r > 0:
            direction = "expected_positive"
        else:
            direction = "unexpected_negative"
    elif metric_name in expected_negative:
        if r < 0:
            direction = "expected_negative"
        else:
            direction = "unexpected_positive"
    else:
        direction = "neutral"
    
    return f"{strength}_{direction}"


# ============================================================
# VISUALIZATION
# ============================================================

def plot_correlations(merged_df, output_dir):
    """Generate correlation plots between FVC% and CT metrics"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid data
    valid_df = merged_df[
        (merged_df['confidence'] >= 0.3) & 
        (merged_df['fvc_percent_week0'].notna())
    ].copy()
    
    if len(valid_df) < 5:
        print("⚠ Skipping plots: insufficient valid cases")
        return
    
    metrics = [
        ('pc_ratio', 'PC Ratio (Peripheral/Central)', 'Expected: Positive correlation'),
        ('tortuosity', 'Mean Tortuosity', 'Expected: Negative correlation'),
        ('branch_count', 'Branch Count', 'Expected: Positive correlation'),
        ('airway_volume_ml', 'Airway Volume (ml)', 'Expected: Positive correlation'),
        ('fibrosis_score', 'Fibrosis Score', 'Expected: Negative correlation')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, label, expected) in enumerate(metrics):
        ax = axes[idx]
        
        plot_df = valid_df[valid_df[metric].notna()]
        
        if len(plot_df) < 5:
            ax.text(0.5, 0.5, f'Insufficient data for {label}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue
        
        # Scatter plot with severity color coding
        severity_colors = {'mild': 'green', 'moderate': 'orange', 'severe': 'red', 'unknown': 'gray'}
        
        for severity in ['mild', 'moderate', 'severe']:
            subset = plot_df[plot_df['severity'] == severity]
            if len(subset) > 0:
                ax.scatter(subset['fvc_percent_week0'], subset[metric], 
                          c=severity_colors[severity], label=severity.capitalize(),
                          s=100, alpha=0.6, edgecolors='black', linewidths=0.5)
        
        # Regression line
        if len(plot_df) >= 5:
            z = np.polyfit(plot_df['fvc_percent_week0'], plot_df[metric], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_df['fvc_percent_week0'].min(), 
                                plot_df['fvc_percent_week0'].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
            
            # Calculate correlation
            r, p_val = stats.pearsonr(plot_df['fvc_percent_week0'], plot_df[metric])
            ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('FVC% at Week 0', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(f'{label}\n{expected}', fontsize=10)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fvc_ct_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Correlation plots saved to: {output_dir / 'fvc_ct_correlations.png'}")


def plot_severity_distributions(merged_df, output_dir):
    """Plot distributions of CT metrics stratified by FVC severity"""
    
    output_dir = Path(output_dir)
    
    valid_df = merged_df[
        (merged_df['confidence'] >= 0.3) & 
        (merged_df['severity'] != 'unknown')
    ].copy()
    
    if len(valid_df) < 5:
        print("⚠ Skipping severity distribution plots: insufficient data")
        return
    
    metrics = ['pc_ratio', 'tortuosity', 'branch_count', 'airway_volume_ml', 'fibrosis_score']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        plot_df = valid_df[valid_df[metric].notna()]
        
        if len(plot_df) < 5:
            ax.text(0.5, 0.5, f'Insufficient data', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Boxplot by severity
        severities = ['mild', 'moderate', 'severe']
        data_to_plot = [plot_df[plot_df['severity'] == sev][metric].values 
                       for sev in severities if len(plot_df[plot_df['severity'] == sev]) > 0]
        labels_to_plot = [sev.capitalize() for sev in severities 
                         if len(plot_df[plot_df['severity'] == sev]) > 0]
        
        if len(data_to_plot) > 0:
            bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
            
            # Color boxes
            colors = ['lightgreen', 'orange', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('Fibrosis Severity (FVC-based)', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} by Severity', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fvc_severity_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Severity distribution plots saved to: {output_dir / 'fvc_severity_distributions.png'}")


# ============================================================
# REPORTING
# ============================================================

def generate_validation_report(merged_df, correlations, coherence_df, output_dir):
    """Generate comprehensive validation report"""
    
    output_dir = Path(output_dir)
    report_path = output_dir / "FVC_VALIDATION_REPORT.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FVC-BASED VALIDATION REPORT\n")
        f.write("Validation of Airway Pipeline Fibrosis Metrics using FVC as Ground Truth\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        f.write("1. DATA SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total cases analyzed: {len(merged_df)}\n")
        f.write(f"Cases with FVC data: {len(merged_df[merged_df['fvc_percent_week0'].notna()])}\n")
        f.write(f"Cases with high-confidence FVC (≥0.5): {len(merged_df[merged_df['confidence'] >= 0.5])}\n")
        f.write(f"Cases with valid CT metrics: {len(merged_df[merged_df['pc_ratio'].notna()])}\n\n")
        
        # FVC estimation methods
        f.write("FVC Estimation Methods:\n")
        method_counts = merged_df['estimation_method'].value_counts()
        for method, count in method_counts.items():
            f.write(f"  - {method}: {count} cases\n")
        f.write("\n")
        
        # Severity distribution
        f.write("2. FIBROSIS SEVERITY DISTRIBUTION (FVC-based)\n")
        f.write("-" * 80 + "\n")
        severity_counts = merged_df['severity'].value_counts()
        for severity, count in severity_counts.items():
            pct = count / len(merged_df) * 100
            f.write(f"  {severity.upper()}: {count} cases ({pct:.1f}%)\n")
        f.write("\n")
        
        # FVC statistics by severity
        valid_df = merged_df[merged_df['fvc_percent_week0'].notna()]
        if len(valid_df) > 0:
            f.write("FVC% Statistics by Severity:\n")
            for severity in ['mild', 'moderate', 'severe']:
                subset = valid_df[valid_df['severity'] == severity]
                if len(subset) > 0:
                    f.write(f"  {severity.upper()}: mean={subset['fvc_percent_week0'].mean():.1f}%, "
                           f"std={subset['fvc_percent_week0'].std():.1f}%, "
                           f"range=[{subset['fvc_percent_week0'].min():.1f}-{subset['fvc_percent_week0'].max():.1f}%]\n")
            f.write("\n")
        
        # Correlation analysis
        f.write("3. CORRELATION ANALYSIS (FVC% vs CT Metrics)\n")
        f.write("-" * 80 + "\n")
        
        if correlations is not None:
            for metric, corr_data in correlations.items():
                f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
                f.write(f"  Pearson r:  {corr_data['pearson_r']:.3f} (p={corr_data['pearson_p']:.4f})\n")
                f.write(f"  Spearman ρ: {corr_data['spearman_r']:.3f} (p={corr_data['spearman_p']:.4f})\n")
                f.write(f"  Sample size: {corr_data['n_samples']}\n")
                f.write(f"  Interpretation: {corr_data['interpretation']}\n")
                
                # Clinical interpretation
                if corr_data['pearson_p'] < 0.05:
                    if metric in ['pc_ratio', 'branch_count', 'airway_volume_ml'] and corr_data['pearson_r'] > 0:
                        f.write(f"  ✓ Significant positive correlation (as expected)\n")
                    elif metric == 'tortuosity' and corr_data['pearson_r'] < 0:
                        f.write(f"  ✓ Significant negative correlation (as expected)\n")
                    else:
                        f.write(f"  ⚠ Significant but unexpected direction\n")
                else:
                    f.write(f"  ○ Not statistically significant\n")
        else:
            f.write("Insufficient data for correlation analysis (n < 5)\n")
        
        f.write("\n")
        
        # Coherence analysis
        f.write("4. COHERENCE ANALYSIS (FVC Severity vs CT Metrics)\n")
        f.write("-" * 80 + "\n")
        coherence_counts = coherence_df['coherence_status'].value_counts()
        for status, count in coherence_counts.items():
            pct = count / len(coherence_df) * 100
            f.write(f"  {status}: {count} cases ({pct:.1f}%)\n")
        f.write("\n")
        
        # Flag analysis
        f.write("Common incoherence flags:\n")
        all_flags = []
        for flags in coherence_df['flags']:
            if isinstance(flags, list):
                all_flags.extend(flags)
        
        if len(all_flags) > 0:
            flag_counts = pd.Series(all_flags).value_counts()
            for flag, count in flag_counts.items():
                if flag != 'none' and flag != 'low_confidence_or_no_fvc':
                    f.write(f"  - {flag}: {count} cases\n")
        else:
            f.write("  No incoherence flags detected\n")
        
        f.write("\n")
        
        # Key findings
        f.write("5. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        if correlations is not None:
            # PC ratio correlation
            pc_corr = correlations.get('pc_ratio', {})
            if pc_corr.get('pearson_p', 1.0) < 0.05:
                f.write(f"✓ PC ratio shows significant correlation with FVC% (r={pc_corr['pearson_r']:.3f})\n")
                f.write(f"  → Pipeline successfully captures peripheral airway loss\n\n")
            else:
                f.write(f"⚠ PC ratio does NOT show significant correlation with FVC%\n")
                f.write(f"  → May need pipeline adjustment or larger sample size\n\n")
        
        coherent_pct = len(coherence_df[coherence_df['coherence_status'] == 'coherent']) / len(coherence_df) * 100
        f.write(f"Coherence rate: {coherent_pct:.1f}% of cases show coherent FVC-CT patterns\n\n")
        
        f.write("6. RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        if correlations is not None:
            pc_corr = correlations.get('pc_ratio', {})
            if pc_corr.get('pearson_p', 1.0) < 0.05 and pc_corr.get('pearson_r', 0) > 0.3:
                f.write("✓ PC ratio is a reliable fibrosis indicator validated by FVC\n")
            else:
                f.write("⚠ PC ratio correlation with FVC is weak - consider:\n")
                f.write("  - Increasing sample size\n")
                f.write("  - Refining peripheral/central region definition\n")
                f.write("  - Investigating outlier cases\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Validation report saved to: {report_path}")


# ============================================================
# MAIN VALIDATION PIPELINE
# ============================================================

def validate_pipeline_with_fvc(
    pipeline_output_dir,
    train_csv_path,
    test_csv_path,
    output_dir
):
    """
    Main validation function: correlate pipeline fibrosis metrics with FVC ground truth
    """
    
    print("\n" + "="*80)
    print("FVC-BASED VALIDATION OF AIRWAY PIPELINE")
    print("="*80 + "\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load FVC data
    print("Step 1: Loading FVC data...")
    fvc_df = load_fvc_data(train_csv_path, test_csv_path)
    print()
    
    # Step 2: Load pipeline results
    print("Step 2: Loading pipeline results...")
    pipeline_df = load_all_pipeline_results(pipeline_output_dir)
    print(f"✓ Loaded pipeline results for {len(pipeline_df)} cases\n")
    
    # Step 3: Process FVC for all pipeline cases
    print("Step 3: Estimating FVC at week 0 for all cases...")
    case_list = pipeline_df['case'].tolist()
    fvc_processed_df = process_fvc_for_all_patients(fvc_df, case_list)
    
    print(f"✓ FVC estimation complete:")
    print(f"  Total cases: {len(fvc_processed_df)}")
    print(f"  With FVC data: {len(fvc_processed_df[fvc_processed_df['fvc_percent_week0'].notna()])}")
    print(f"  High confidence (≥0.5): {len(fvc_processed_df[fvc_processed_df['confidence'] >= 0.5])}")
    print()
    
    # Step 4: Merge pipeline and FVC data
    print("Step 4: Merging pipeline and FVC data...")
    merged_df = pd.merge(pipeline_df, fvc_processed_df, on='case', how='left')
    print(f"✓ Merged dataset: {len(merged_df)} cases\n")
    
    # Step 5: Validate coherence
    print("Step 5: Validating FVC-CT coherence...")
    coherence_df = validate_fvc_coherence(merged_df)
    print(f"✓ Coherence analysis complete\n")
    
    # Step 6: Compute correlations
    print("Step 6: Computing correlations...")
    correlations = compute_correlations(merged_df)
    if correlations is not None:
        print("✓ Correlation analysis complete:")
        for metric, corr in correlations.items():
            if corr['n_samples'] >= 5:
                sig = "***" if corr['pearson_p'] < 0.001 else "**" if corr['pearson_p'] < 0.01 else "*" if corr['pearson_p'] < 0.05 else "ns"
                print(f"  {metric}: r={corr['pearson_r']:.3f} {sig} (n={corr['n_samples']})")
    print()
    
    # Step 7: Generate visualizations
    print("Step 7: Generating visualizations...")
    try:
        plot_correlations(merged_df, output_dir)
        plot_severity_distributions(merged_df, output_dir)
    except Exception as e:
        print(f"⚠ Warning: Visualization generation failed: {e}")
    print()
    
    # Step 8: Save results
    print("Step 8: Saving results...")
    
    # Merge coherence results
    merged_with_coherence = pd.merge(merged_df, coherence_df, on='case', how='left')
    
    # Save main results
    results_csv = output_dir / "fvc_validation_results.csv"
    merged_with_coherence.to_csv(results_csv, index=False)
    print(f"✓ Results saved to: {results_csv}")
    
    # Save correlations
    if correlations is not None:
        corr_df = pd.DataFrame(correlations).T
        corr_csv = output_dir / "fvc_correlation_statistics.csv"
        corr_df.to_csv(corr_csv)
        print(f"✓ Correlation statistics saved to: {corr_csv}")
    
    # Generate report
    generate_validation_report(merged_df, correlations, coherence_df, output_dir)
    print()
    
    print("="*80)
    print("VALIDATION COMPLETE")
    print("="*80 + "\n")
    
    return merged_with_coherence, correlations


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    """Main entry point for FVC validation"""
    
    # Configuration paths
    PIPELINE_OUTPUT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC (correct)")
    TRAIN_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\train.csv")
    TEST_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\test.csv")
    OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\results\fvc_validation_results")
    
    print(f"\n{'='*80}")
    print(f"FVC VALIDATION TOOL")
    print(f"{'='*80}")
    print(f"Pipeline output: {PIPELINE_OUTPUT}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    # Verify paths exist
    if not PIPELINE_OUTPUT.exists():
        print(f"ERROR: Pipeline output directory not found: {PIPELINE_OUTPUT}")
        return
    
    if not TRAIN_CSV.exists():
        print(f"ERROR: Train CSV not found: {TRAIN_CSV}")
        return
    
    if not TEST_CSV.exists():
        print(f"ERROR: Test CSV not found: {TEST_CSV}")
        return
    
    # Run validation
    try:
        results_df, correlations = validate_pipeline_with_fvc(
            PIPELINE_OUTPUT,
            TRAIN_CSV,
            TEST_CSV,
            OUTPUT_DIR
        )
        
        print(f"\n✓ FVC validation completed successfully!")
        print(f"  Results available in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nERROR during validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
