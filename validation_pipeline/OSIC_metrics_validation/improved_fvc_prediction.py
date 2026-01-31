import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

VALIDATION_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\OSIC_validation.csv")
RESULTS_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC_combined")
TRAIN_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\train.csv")
TEST_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\test.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\improved_prediction")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# INTERPOLATION FUNCTION - MIX BETWEEN FRANCESCO LOGIC AND INTERPOLATION GENERIC
def interpolate_fvc_percent(patient_data, target_week, window_weeks=15, max_distance=15):    
    patient_data = patient_data.sort_values('Weeks')
    
    # Check for exact match (within 1 week tolerance)
    exact_match = patient_data[abs(patient_data['Weeks'] - target_week) < 1]
    if len(exact_match) > 0:
        closest = exact_match.iloc[0]
        return {
            'value': closest['Percent'],
            'actual_week': closest['Weeks'],
            'method': 'exact',
            'quality': 'high',
            'n_points_used': 1,
            'distance': abs(closest['Weeks'] - target_week)
        }

    in_window = patient_data[
        abs(patient_data['Weeks'] - target_week) <= window_weeks
    ]
    
    # If there are measurements within the window -> Francesco logic
    if len(in_window) >= 1:
        distances = abs(in_window['Weeks'] - target_week)
        min_distance_idx = distances.idxmin()
        min_distance = distances.min()
        
        # If the closest measurement is ≤ max_distance -> Francesco logic
        if min_distance <= max_distance:
            closest = in_window.loc[min_distance_idx]
            
            # Quality based on distance (stricter)
            if min_distance <= 3:
                quality = 'high'
            elif min_distance <= 8:
                quality = 'medium'
            else:
                quality = 'low'
            
            return {
                'value': closest['Percent'],
                'actual_week': closest['Weeks'],
                'method': 'nearest',
                'quality': quality,
                'n_points_used': len(in_window),
                'distance': min_distance
            }
    
    # If no close measurement or measurement too far ( > max_distance) -> REGRESSION
    if len(patient_data) >= 2:
        weeks = patient_data['Weeks'].values
        percents = patient_data['Percent'].values
        
        # Check if all weeks are identical
        if len(np.unique(weeks)) == 1:
            estimated_value = np.mean(percents)
            distance = abs(weeks[0] - target_week)
            
            if distance > 20:
                quality = 'low'
            elif distance > 10:
                quality = 'medium'
            else:
                quality = 'medium'
            
            return {
                'value': estimated_value,
                'actual_week': weeks[0],
                'method': 'regression_simple',
                'quality': quality,
                'n_points_used': len(patient_data),
                'distance': distance
            }
        
        # Linear regression on ALL data
        slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, percents)
        estimated_value = slope * target_week + intercept
        
        # Calculate distance
        min_week = weeks.min()
        max_week = weeks.max()
        
        if target_week < min_week:
            distance = min_week - target_week
        elif target_week > max_week:
            distance = target_week - max_week
        else:
            distance = 0
        
        # Quality assessment
        if distance == 0:  # Interpolation
            if abs(r_value) > 0.85 and len(patient_data) >= 3:
                quality = 'high'
            elif abs(r_value) > 0.7:
                quality = 'medium'
            else:
                quality = 'low'
        else:  # Extrapolation
            if distance > 15:
                quality = 'low'
            elif distance > 8:
                quality = 'medium'
            elif abs(r_value) > 0.8 and len(patient_data) >= 3:
                quality = 'medium'
            else:
                quality = 'low'
        
        # Find closest actual measurement
        closest_idx = np.argmin(np.abs(weeks - target_week))
        actual_week = weeks[closest_idx]
        
        return {
            'value': estimated_value,
            'actual_week': actual_week,
            'method': 'regression' if distance == 0 else 'extrapolation',
            'quality': quality,
            'n_points_used': len(patient_data),
            'r_value': r_value,
            'distance': distance
        }
    
    # if fail -> no measurements or < 2 measurements
    return {
        'value': np.nan,
        'actual_week': np.nan,
        'method': 'failed',
        'quality': 'failed',
        'n_points_used': len(patient_data),
        'distance': np.nan
    }


def create_interpolated_fvc_dataset(clinical_data):    
    results = []
    
    patient_ids = clinical_data['Patient'].unique()
    print(f"\nProcessing {len(patient_ids)} patients...")
    
    for patient_id in patient_ids:
        patient_data = clinical_data[clinical_data['Patient'] == patient_id]
        demographics = patient_data.iloc[0]
        
        # Week 0 interpolation
        week0_result = interpolate_fvc_percent(
            patient_data, 
            target_week=0, 
            window_weeks=10,
            max_distance=8
        )
        
        # Week 52 interpolation
        week52_result = interpolate_fvc_percent(
            patient_data,
            target_week=52,
            window_weeks=15,
            max_distance=10
        )
        
        # Calculate drop
        if not np.isnan(week0_result['value']) and not np.isnan(week52_result['value']):
            fvc_drop_absolute = week0_result['value'] - week52_result['value']
            fvc_drop_percent = (fvc_drop_absolute / week0_result['value']) * 100 if week0_result['value'] > 0 else np.nan
        else:
            fvc_drop_absolute = np.nan
            fvc_drop_percent = np.nan
        
        results.append({
            'Patient': patient_id,
            'Age': demographics['Age'],
            'Sex': demographics['Sex'],
            'SmokingStatus': demographics['SmokingStatus'],
            
            # Week 0 data
            'FVC_percent_week0': week0_result['value'],
            'week0_actual_week': week0_result['actual_week'],
            'week0_method': week0_result['method'],
            'week0_quality': week0_result['quality'],
            'week0_distance': week0_result['distance'],
            
            # Week 52 data
            'FVC_percent_week52': week52_result['value'],
            'week52_actual_week': week52_result['actual_week'],
            'week52_method': week52_result['method'],
            'week52_quality': week52_result['quality'],
            'week52_distance': week52_result['distance'],
            
            # Calculated values
            'FVC_drop_absolute': fvc_drop_absolute,
            'FVC_drop_percent': fvc_drop_percent,
            
            # Data completeness
            'n_measurements': len(patient_data)
        })
    
    df = pd.DataFrame(results)
    
    # Report statistics
    complete = df.dropna(subset=['FVC_percent_week0', 'FVC_percent_week52'])
    print(f"\nComplete cases (both week0 and week52): {len(complete)} ({100*len(complete)/len(df):.1f}%)")
    
    return df


def load_clinical_data():
    """Load and combine train.csv and test.csv"""
    print("Loading clinical data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    clinical = pd.concat([train, test], ignore_index=True)
    
    print(f"  Loaded {len(clinical)} clinical records")
    print(f"  Unique patients: {clinical['Patient'].nunique()}")
    
    return clinical


def load_validation_results():
    """Load validation results and filter RELIABLE cases"""
    print("\nLoading validation results...")
    validation = pd.read_csv(VALIDATION_CSV)
    reliable = validation[validation['status'] == 'RELIABLE'].copy()
    
    print(f"  Total cases: {len(validation)}")
    print(f"  RELIABLE cases: {len(reliable)}")
    
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
        return None


def extract_patient_id(case_name):
    """Extract patient ID from case name"""
    return case_name.replace("_gaussian", "")


def integrate_airway_and_fvc(reliable_cases, fvc_df):
    """
    Integrate airway metrics with interpolated FVC data.
    Returns only two datasets: all patients and high/medium quality patients.
    """
    print("\n" + "="*80)
    print("INTEGRATING AIRWAY METRICS WITH FVC DATA")
    print("="*80)
    
    rows = []
    
    for idx, case_row in reliable_cases.iterrows():
        case_name = case_row['case']
        patient_id = extract_patient_id(case_name)
        
        # Find FVC data
        patient_fvc = fvc_df[fvc_df['Patient'] == patient_id]
        if len(patient_fvc) == 0:
            continue
        
        patient_fvc = patient_fvc.iloc[0]
        
        # Load metrics
        advanced = load_advanced_metrics(case_name)
        if advanced is None:
            continue
        
        parenchymal = load_parenchymal_metrics(case_name)
        
        # Create row
        row = {
            'patient': patient_id,
            'case': case_name,
            
            # Demographics
            'Age': patient_fvc['Age'],
            'Sex': patient_fvc['Sex'],
            'SmokingStatus': patient_fvc['SmokingStatus'],
            
            # FVC targets
            'FVC_percent_week0': patient_fvc['FVC_percent_week0'],
            'FVC_percent_week52': patient_fvc['FVC_percent_week52'],
            'FVC_drop_absolute': patient_fvc['FVC_drop_absolute'],
            'FVC_drop_percent': patient_fvc['FVC_drop_percent'],
            
            # Quality info
            'week0_quality': patient_fvc['week0_quality'],
            'week52_quality': patient_fvc['week52_quality'],
            
            # Airway metrics
            'volume_ml': case_row['volume_ml'],
            'mean_tortuosity': advanced.get('mean_tortuosity'),
            'std_peripheral_diameter_mm': advanced.get('std_peripheral_diameter_mm'),
            'central_to_peripheral_diameter_ratio': advanced.get('central_to_peripheral_diameter_ratio'),
            'mean_peripheral_branch_volume_mm3': advanced.get('mean_peripheral_branch_volume_mm3'),
        }
        
        # Add parenchymal metrics
        if parenchymal is not None:
            row.update({
                'mean_lung_density_HU': parenchymal.get('mean_lung_density_HU'),
                'histogram_entropy': parenchymal.get('histogram_entropy'),
            })
        
        rows.append(row)
    
    df_all = pd.DataFrame(rows)
    
    # Filter for high/medium quality patients only
    df_quality = df_all[
        (df_all['week0_quality'].isin(['high', 'medium'])) & 
        (df_all['week52_quality'].isin(['high', 'medium']))
    ].copy()
    
    print(f"\nIntegrated dataset:")
    print(f"  All patients: {len(df_all)}")
    print(f"  High/Medium quality patients: {len(df_quality)}")
    
    return df_all, df_quality


def leave_one_out_predict(df, feature_name, target_name):
    """Leave-one-out cross-validation for single-feature prediction"""
    
    valid_data = df[[feature_name, target_name]].dropna()
    if len(valid_data) < 5:
        return None
    
    X = valid_data[feature_name].values.reshape(-1, 1)
    y = valid_data[target_name].values
    
    predictions, actuals, errors = [], [], []
    
    for i in range(len(X)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i].reshape(1, -1)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        
        predictions.append(y_pred)
        actuals.append(y[i])
        errors.append(y_pred - y[i])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    errors = np.array(errors)
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    pearson_r, pearson_p = pearsonr(actuals, predictions)
    
    return {
        'feature': feature_name,
        'target': target_name,
        'n_samples': len(X),
        'predictions': predictions,
        'actuals': actuals,
        'errors': errors,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }


def create_single_feature_plots(df_quality, features, output_dir):
    """
    Create plots for each feature showing predictions for:
    1) FVC@week0
    2) FVC@week52  
    3) %FVC drop at 1year
    Includes correlation plots and Bland-Altman plots
    """
    plots_dir = output_dir / "single_feature_plots"
    plots_dir.mkdir(exist_ok=True)
    
    targets = ['FVC_percent_week0', 'FVC_percent_week52', 'FVC_drop_percent']
    
    all_results = []
    
    for feature in features:
        if feature not in df_quality.columns:
            continue
            
        print(f"\n{'='*60}")
        print(f"Analyzing feature: {feature}")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Single-Feature Prediction: {feature}', fontsize=16, fontweight='bold')
        
        for col, target in enumerate(targets):
            if target not in df_quality.columns:
                continue
                
            result = leave_one_out_predict(df_quality, feature, target)
            if result is None or result['n_samples'] < 5:
                continue
                
            all_results.append(result)
            
            predictions = result['predictions']
            actuals = result['actuals']
            errors = result['errors']
            
            # Row 1: Correlation plots
            ax_corr = axes[0, col]
            
            # Scatter plot
            ax_corr.scatter(actuals, predictions, alpha=0.7, s=60, 
                           edgecolors='black', linewidth=0.5, color='steelblue')
            
            # Perfect prediction line
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            ax_corr.plot([min_val, max_val], [min_val, max_val], 
                        'r--', alpha=0.8, linewidth=2, label='Perfect')
            
            # Regression line
            z = np.polyfit(actuals, predictions, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min_val, max_val, 100)
            ax_corr.plot(x_trend, p(x_trend), 'g-', alpha=0.8, linewidth=2, label='Regression')
            
            ax_corr.set_xlabel(f'Actual {target}', fontsize=12)
            ax_corr.set_ylabel(f'Predicted {target}', fontsize=12)
            ax_corr.set_title(f'{target}\nR² = {result["r2"]:.3f}, MAE = {result["mae"]:.2f}', 
                            fontsize=12, fontweight='bold')
            ax_corr.legend(loc='best', fontsize=10)
            ax_corr.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics_text = f'n = {result["n_samples"]}\nMAE = {result["mae"]:.2f}\nRMSE = {result["rmse"]:.2f}'
            ax_corr.text(0.05, 0.95, metrics_text, transform=ax_corr.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # Row 2: Bland-Altman plots
            ax_ba = axes[1, col]
            
            # Calculate mean and difference
            means = (actuals + predictions) / 2
            differences = predictions - actuals
            
            # Scatter plot
            ax_ba.scatter(means, differences, alpha=0.7, s=60, 
                         edgecolors='black', linewidth=0.5, color='coral')
            
            # Mean difference line
            mean_diff = np.mean(differences)
            ax_ba.axhline(y=mean_diff, color='blue', linestyle='-', linewidth=2, 
                         label=f'Mean diff: {mean_diff:.2f}')
            
            # ±1.96 SD lines
            sd_diff = np.std(differences)
            upper_limit = mean_diff + 1.96 * sd_diff
            lower_limit = mean_diff - 1.96 * sd_diff
            
            ax_ba.axhline(y=upper_limit, color='red', linestyle='--', linewidth=1.5,
                         label=f'+1.96 SD: {upper_limit:.2f}')
            ax_ba.axhline(y=lower_limit, color='red', linestyle='--', linewidth=1.5,
                         label=f'-1.96 SD: {lower_limit:.2f}')
            
            ax_ba.set_xlabel('Mean of Actual and Predicted', fontsize=12)
            ax_ba.set_ylabel('Predicted - Actual', fontsize=12)
            ax_ba.set_title(f'Bland-Altman: {target}', fontsize=12, fontweight='bold')
            ax_ba.legend(loc='best', fontsize=9)
            ax_ba.grid(True, alpha=0.3)
            
            # Print results
            print(f"  {target}: R²={result['r2']:.3f}, MAE={result['mae']:.2f}, n={result['n_samples']}")
        
        plt.tight_layout()
        plot_path = plots_dir / f"{feature}_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save summary of results
    if all_results:
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Feature': result['feature'],
                'Target': result['target'],
                'n_samples': result['n_samples'],
                'R²': result['r2'],
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'Pearson_r': result['pearson_r'],
                'Pearson_p': result['pearson_p']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values(['Feature', 'Target'])
        df_summary.to_csv(output_dir / "single_feature_predictions_summary.csv", index=False)
        
        print(f"\n✓ Saved prediction summary: single_feature_predictions_summary.csv")
    
    return all_results


def main():
    print("\n" + "="*80)
    print("SINGLE-FEATURE FVC PREDICTION ANALYSIS")
    print("="*80)
    print("Analysis: Each feature vs FVC@week0, FVC@week52, %FVC drop")
    print("Method: Leave-one-out cross-validation with linear regression")
    print("Output: Correlation plots + Bland-Altman plots for each feature")
    print("="*80)
    
    # Load data
    clinical = load_clinical_data()
    
    # Create interpolated dataset
    fvc_df = create_interpolated_fvc_dataset(clinical)
    
    # Save interpolated data
    fvc_output = OUTPUT_DIR / "interpolated_fvc_data.csv"
    fvc_df.to_csv(fvc_output, index=False)
    print(f"\n✓ Interpolated FVC data saved: {fvc_output}")
    
    # Load airway metrics
    reliable = load_validation_results()
    
    # Integrate datasets (only all and quality-filtered)
    df_all, df_quality = integrate_airway_and_fvc(reliable, fvc_df)
    
    # Save datasets
    df_all.to_csv(OUTPUT_DIR / "all_patients_dataset.csv", index=False)
    df_quality.to_csv(OUTPUT_DIR / "quality_filtered_dataset.csv", index=False)
    
    print(f"\n✓ Saved datasets:")
    print(f"  All patients: {len(df_all)} patients")
    print(f"  High/Medium quality: {len(df_quality)} patients")
    
    # Define features to analyze
    features = [
        'volume_ml',
        'mean_tortuosity',
        'std_peripheral_diameter_mm',
        'central_to_peripheral_diameter_ratio',
        'mean_peripheral_branch_volume_mm3',
        'mean_lung_density_HU',
        'histogram_entropy'
    ]
    
    # Create single-feature plots using only quality-filtered data
    print(f"\n{'='*80}")
    print("CREATING SINGLE-FEATURE PREDICTION PLOTS")
    print(f"{'='*80}")
    
    all_results = create_single_feature_plots(df_quality, features, OUTPUT_DIR)
    
    # Final summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Datasets saved in: {OUTPUT_DIR}")
    print(f"✓ Single-feature plots saved in: {OUTPUT_DIR / 'single_feature_plots'}")
    print(f"✓ Prediction summary saved: single_feature_predictions_summary.csv")
    
    # Show best performing features
    if all_results:
        print(f"\nTOP PERFORMING FEATURES (by average R² across all targets):")
        
        # Group by feature and calculate average R²
        feature_performance = {}
        for result in all_results:
            feature = result['feature']
            if feature not in feature_performance:
                feature_performance[feature] = []
            feature_performance[feature].append(result['r2'])
        
        # Calculate and display averages
        avg_performance = []
        for feature, r2_values in feature_performance.items():
            avg_r2 = np.mean(r2_values)
            avg_performance.append((feature, avg_r2, len(r2_values)))
        
        # Sort by average R²
        avg_performance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, avg_r2, n_targets in avg_performance[:5]:  # Top 5
            print(f"  {feature:35s}: Avg R² = {avg_r2:.3f} (across {n_targets} targets)")


if __name__ == "__main__":
    main()