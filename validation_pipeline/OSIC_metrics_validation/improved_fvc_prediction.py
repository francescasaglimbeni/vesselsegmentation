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


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\OSIC_validation.csv")
RESULTS_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC_combined")
TRAIN_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\train.csv")
TEST_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\test.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\improved_prediction")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ROBUST FVC INTERPOLATION - COMPROMESSO OTTIMALE
# ============================================================

def interpolate_fvc_percent(patient_data, target_week, window_weeks=15, max_distance=15):
    """
    COMPROMESSO OTTIMALE:
    1. Se misura esatta (±1 settimana) → usa quella
    2. Se misura nella finestra E ≤ max_distance → logica Fra (più vicino)
    3. Altrimenti → regressione lineare su tutti i dati
    
    Aggiunta filtro max_distance per escludere misure "troppo lontane" anche se nella finestra
    """
    
    # Sort by week
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
    
    # COMPROMESSO: cerca misure vicine MA con distanza ≤ max_distance
    in_window = patient_data[
        abs(patient_data['Weeks'] - target_week) <= window_weeks
    ]
    
    # Filtra ulteriormente per max_distance
    if len(in_window) >= 1:
        distances = abs(in_window['Weeks'] - target_week)
        min_distance_idx = distances.idxmin()
        min_distance = distances.min()
        
        # SE la misura più vicina è ≤ max_distance → logica Fra
        if min_distance <= max_distance:
            closest = in_window.loc[min_distance_idx]
            
            # Quality basata su distanza (più stretta)
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
    
    # SE nessuna misura vicina O misura troppo lontana (>max_distance) → REGRESSIONE
    if len(patient_data) >= 2:
        weeks = patient_data['Weeks'].values
        percents = patient_data['Percent'].values
        
        # Check if all weeks are identical
        if len(np.unique(weeks)) == 1:
            estimated_value = np.mean(percents)
            distance = abs(weeks[0] - target_week)
            
            # Quality per regressione semplice
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
        
        # Quality assessment migliorata
        if distance == 0:  # Interpolation
            if abs(r_value) > 0.85 and len(patient_data) >= 3:
                quality = 'high'
            elif abs(r_value) > 0.7:
                quality = 'medium'
            else:
                quality = 'low'
        else:  # Extrapolation
            if distance > 15:  # Più restrittivo
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
    
    # FAILED
    return {
        'value': np.nan,
        'actual_week': np.nan,
        'method': 'failed',
        'quality': 'failed',
        'n_points_used': len(patient_data),
        'distance': np.nan
    }


def create_interpolated_fvc_dataset(clinical_data):
    """
    Create dataset with interpolated FVC values at week 0 and week 52.
    COMPROMESSO: usa parametri più restrittivi per qualità
    """
    print("\n" + "="*80)
    print("CREATING INTERPOLATED FVC DATASET - COMPROMESSO OTTIMALE")
    print("="*80)
    print("Parametri:")
    print("- Week0: finestra=10, max_distance=8 (più restrittivo per baseline)")
    print("- Week52: finestra=15, max_distance=10 (leggermente più permissivo per follow-up)")
    print("- Regressione solo se nessuna misura vicina/sufficientemente vicina")
    print("="*80)
    
    results = []
    
    patient_ids = clinical_data['Patient'].unique()
    print(f"\nProcessing {len(patient_ids)} patients...")
    
    method_counts = {'week0': {}, 'week52': {}}
    
    for patient_id in patient_ids:
        patient_data = clinical_data[clinical_data['Patient'] == patient_id]
        
        demographics = patient_data.iloc[0]
        
        # Parametri più restrittivi per baseline
        week0_result = interpolate_fvc_percent(
            patient_data, 
            target_week=0, 
            window_weeks=10,
            max_distance=8  # Solo misure entro 8 settimane per logica Fra
        )
        
        # Parametri leggermente più permissivi per follow-up
        week52_result = interpolate_fvc_percent(
            patient_data,
            target_week=52,
            window_weeks=15,
            max_distance=10  # Misure entro 10 settimane per logica Fra
        )
        
        # Conta metodi
        method_counts['week0'][week0_result['method']] = method_counts['week0'].get(week0_result['method'], 0) + 1
        method_counts['week52'][week52_result['method']] = method_counts['week52'].get(week52_result['method'], 0) + 1
        
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
            'week0_n_points': week0_result['n_points_used'],
            'week0_distance': week0_result['distance'],
            
            # Week 52 data
            'FVC_percent_week52': week52_result['value'],
            'week52_actual_week': week52_result['actual_week'],
            'week52_method': week52_result['method'],
            'week52_quality': week52_result['quality'],
            'week52_n_points': week52_result['n_points_used'],
            'week52_distance': week52_result['distance'],
            
            # Calculated values
            'FVC_drop_absolute': fvc_drop_absolute,
            'FVC_drop_percent': fvc_drop_percent,
            
            # Data completeness
            'n_measurements': len(patient_data),
            'week_range_min': patient_data['Weeks'].min(),
            'week_range_max': patient_data['Weeks'].max()
        })
    
    df = pd.DataFrame(results)
    
    # Report statistics
    print("\n" + "-"*80)
    print("INTERPOLATION STATISTICS - COMPROMESSO")
    print("-"*80)
    
    print("\nWEEK 0 (Baseline - più restrittivo):")
    print(f"  Total patients: {len(df)}")
    print(f"  Successfully interpolated: {df['FVC_percent_week0'].notna().sum()}")
    print(f"  Methods (max_distance=8 settimane):")
    for method, count in method_counts['week0'].items():
        print(f"    {method:20s}: {count:3d} ({100*count/len(df):5.1f}%)")
    
    print(f"\n  Quality:")
    for quality in ['high', 'medium', 'low', 'failed']:
        count = (df['week0_quality'] == quality).sum()
        print(f"    {quality:15s}: {count:3d} ({100*count/len(df):5.1f}%)")
    
    print(f"\n  Distance statistics:")
    valid_dist = df['week0_distance'].dropna()
    if len(valid_dist) > 0:
        print(f"    Mean: {valid_dist.mean():.1f} weeks, Max: {valid_dist.max():.1f}")
        print(f"    ≤3 weeks (high): {(valid_dist <= 3).sum()}")
        print(f"    ≤8 weeks (medium+): {(valid_dist <= 8).sum()}")
    
    print("\nWEEK 52 (Follow-up - leggermente più permissivo):")
    print(f"  Total patients: {len(df)}")
    print(f"  Successfully interpolated: {df['FVC_percent_week52'].notna().sum()}")
    print(f"  Methods (max_distance=10 settimane):")
    for method, count in method_counts['week52'].items():
        print(f"    {method:20s}: {count:3d} ({100*count/len(df):5.1f}%)")
    
    print(f"\n  Quality:")
    for quality in ['high', 'medium', 'low', 'failed']:
        count = (df['week52_quality'] == quality).sum()
        print(f"    {quality:15s}: {count:3d} ({100*count/len(df):5.1f}%)")
    
    print(f"\n  Distance statistics:")
    valid_dist = df['week52_distance'].dropna()
    if len(valid_dist) > 0:
        print(f"    Mean: {valid_dist.mean():.1f} weeks, Max: {valid_dist.max():.1f}")
        print(f"    ≤5 weeks (high): {(valid_dist <= 5).sum()}")
        print(f"    ≤10 weeks (medium+): {(valid_dist <= 10).sum()}")
    
    # Complete cases
    complete = df.dropna(subset=['FVC_percent_week0', 'FVC_percent_week52'])
    print(f"\nCOMPLETE CASES (both week0 and week52):")
    print(f"  Total: {len(complete)} ({100*len(complete)/len(df):.1f}%)")
    
    # Quality distribution for complete cases
    print(f"\n  Quality distribution for complete cases:")
    high_quality = complete[
        (complete['week0_quality'] == 'high') & 
        (complete['week52_quality'] == 'high')
    ]
    medium_quality = complete[
        (complete['week0_quality'].isin(['high', 'medium'])) & 
        (complete['week52_quality'].isin(['high', 'medium']))
    ]
    print(f"    Both high quality: {len(high_quality)}")
    print(f"    Both high/medium:  {len(medium_quality)}")
    
    # Method combination analysis
    print(f"\n  Method combinations:")
    method_combinations = complete.groupby(['week0_method', 'week52_method']).size().reset_index(name='count')
    for _, row in method_combinations.sort_values('count', ascending=False).iterrows():
        print(f"    {row['week0_method']} + {row['week52_method']}: {row['count']}")
    
    # FVC statistics
    if len(complete) > 0:
        print(f"\nFVC STATISTICS (complete cases):")
        print(f"  FVC% week0:  {complete['FVC_percent_week0'].mean():.1f} ± {complete['FVC_percent_week0'].std():.1f}%")
        print(f"  FVC% week52: {complete['FVC_percent_week52'].mean():.1f} ± {complete['FVC_percent_week52'].std():.1f}%")
        print(f"  Absolute drop: {complete['FVC_drop_absolute'].mean():.1f} ± {complete['FVC_drop_absolute'].std():.1f} points")
        print(f"  Relative drop: {complete['FVC_drop_percent'].mean():.1f} ± {complete['FVC_drop_percent'].std():.1f}%")
    
    return df


# ============================================================
# LOAD DATA (nessuna modifica)
# ============================================================

def load_clinical_data():
    """Load and combine train.csv and test.csv"""
    print("\nLoading clinical data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    clinical = pd.concat([train, test], ignore_index=True)
    
    print(f"  Loaded {len(clinical)} clinical records")
    print(f"  Unique patients: {clinical['Patient'].nunique()}")
    print(f"  Week range: {clinical['Weeks'].min():.0f} to {clinical['Weeks'].max():.0f}")
    
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


# ============================================================
# INTEGRATE DATASETS - MODIFICATA PER ANALISI METODI
# ============================================================

def integrate_airway_and_fvc(reliable_cases, fvc_df):
    """
    Integrate airway metrics with interpolated FVC data.
    Aggiunge analisi per metodi diversi.
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
            
            # Quality and method info
            'week0_quality': patient_fvc['week0_quality'],
            'week52_quality': patient_fvc['week52_quality'],
            'week0_method': patient_fvc['week0_method'],
            'week52_method': patient_fvc['week52_method'],
            'week0_distance': patient_fvc.get('week0_distance', np.nan),
            'week52_distance': patient_fvc.get('week52_distance', np.nan),
            
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
    
    df = pd.DataFrame(rows)
    
    print(f"\nIntegrated dataset:")
    print(f"  Total patients with both airway and FVC data: {len(df)}")
    
    # Filtri MULTIPLI per analisi comparativa
    df_high = df[
        (df['week0_quality'] == 'high') & 
        (df['week52_quality'] == 'high')
    ].copy()
    
    df_medium = df[
        (df['week0_quality'].isin(['high', 'medium'])) & 
        (df['week52_quality'].isin(['high', 'medium']))
    ].copy()
    
    # NUOVO: Filtro per metodo "nearest" per entrambi (logica Fra pura)
    df_nearest_both = df[
        (df['week0_method'] == 'nearest') & 
        (df['week52_method'] == 'nearest')
    ].copy()
    
    # NUOVO: Filtro per distanza massima
    df_close_both = df[
        (df['week0_distance'] <= 5) & 
        (df['week52_distance'] <= 8)
    ].copy()
    
    print(f"\n  Filtered subsets:")
    print(f"    High quality (both high): {len(df_high)}")
    print(f"    Medium+ quality: {len(df_medium)}")
    print(f"    Both nearest (logica Fra): {len(df_nearest_both)}")
    print(f"    Both close (dist ≤5/8 weeks): {len(df_close_both)}")
    
    # Method analysis
    print(f"\n  Method distribution:")
    method_counts = df['week0_method'].value_counts()
    for method, count in method_counts.items():
        if pd.notna(method):
            print(f"    Week0 {method}: {count}")
    
    method_counts = df['week52_method'].value_counts()
    for method, count in method_counts.items():
        if pd.notna(method):
            print(f"    Week52 {method}: {count}")
    
    return df, df_high, df_medium, df_nearest_both, df_close_both


# ============================================================
# LEAVE-ONE-OUT PREDICTION - MODIFICATA PER MULTIPLI DATASET
# ============================================================

def leave_one_out_predict(df, feature_name, target_name, model_type='linear'):
    """Leave-one-out cross-validation"""
    
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
        
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)[0]
        else:
            continue
        
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
    
    # MAPE migliorato
    if 'percent' in target_name.lower():
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.nanmean(np.abs(errors) / np.abs(actuals)) * 100
    else:
        mape = np.mean(np.abs(errors))
    
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
        'mape': mape,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p
    }


def run_predictions_on_datasets(datasets, dataset_names, features, targets):
    """Esegue predizioni su multiple dataset per confronto"""
    all_results = []
    
    for df, name in zip(datasets, dataset_names):
        print(f"\n{'='*60}")
        print(f"PREDICTIONS ON: {name} (n={len(df)})")
        print(f"{'='*60}")
        
        for target in targets:
            print(f"\n  Target: {target}")
            
            for feature in features:
                if feature not in df.columns:
                    continue
                
                valid_count = df[[feature, target]].dropna().shape[0]
                if valid_count < 5:
                    continue
                
                result = leave_one_out_predict(df, feature, target, 'linear')
                if result is None:
                    continue
                
                result['dataset'] = name
                all_results.append(result)
                
                if result['r2'] > 0.1:  # Stampa solo risultati decenti
                    print(f"    {feature}: R²={result['r2']:.3f}, r={result['pearson_r']:.3f}, n={result['n_samples']}")
    
    return all_results


# ============================================================
# VISUALIZATION (nessuna modifica sostanziale)
# ============================================================

def plot_correlation(result, output_path):
    """Plot predicted vs actual"""
    
    predictions = result['predictions']
    actuals = result['actuals']
    feature = result['feature']
    target = result['target']
    r2 = result['r2']
    pearson_r = result['pearson_r']
    dataset = result.get('dataset', '')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(actuals, predictions, alpha=0.7, s=80, 
              edgecolors='black', linewidth=1, color='steelblue')
    
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', alpha=0.8, linewidth=2, label='Perfect prediction')
    
    z = np.polyfit(actuals, predictions, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min_val, max_val, 100)
    ax.plot(x_trend, p(x_trend), 'g-', alpha=0.8, linewidth=2, label='Fitted line')
    
    ax.set_xlabel(f'Actual {target}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Predicted {target}', fontsize=14, fontweight='bold')
    
    title = f'{feature} → {target}\n{dataset}\nR² = {r2:.3f}, r = {pearson_r:.3f}'
    ax.set_title(title, fontsize=15, fontweight='bold')
    
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    metrics_text = (f'n = {result["n_samples"]}\n'
                   f'R² = {r2:.3f}\n'
                   f'MAE = {result["mae"]:.2f}\n'
                   f'RMSE = {result["rmse"]:.2f}\n'
                   f'Pearson r = {pearson_r:.3f}')
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN ANALYSIS - RIVISTA PER COMPROMESSO
# ============================================================

def main():
    print("\n" + "="*80)
    print("FVC PREDICTION ANALYSIS - COMPROMESSO OTTIMALE")
    print("="*80)
    print("Strategia: Logica Fra per misure vicine, regressione per altre")
    print("Parametri più restrittivi per qualità migliore")
    print("="*80)
    
    # Load data
    clinical = load_clinical_data()
    
    # Create interpolated dataset con compromesso
    fvc_df = create_interpolated_fvc_dataset(clinical)
    
    # Save
    fvc_output = OUTPUT_DIR / "interpolated_fvc_compromise.csv"
    fvc_df.to_csv(fvc_output, index=False)
    print(f"\n✓ Interpolated FVC dataset saved: {fvc_output}")
    
    # Load airway metrics
    reliable = load_validation_results()
    
    # Integrate con nuovi filtri
    df_all, df_high, df_medium, df_nearest, df_close = integrate_airway_and_fvc(reliable, fvc_df)
    
    # Save tutti i dataset
    df_all.to_csv(OUTPUT_DIR / "integrated_dataset_all.csv", index=False)
    df_medium.to_csv(OUTPUT_DIR / "integrated_dataset_medium.csv", index=False)
    df_nearest.to_csv(OUTPUT_DIR / "integrated_dataset_nearest.csv", index=False)
    df_close.to_csv(OUTPUT_DIR / "integrated_dataset_close.csv", index=False)
    
    print(f"\n✓ Saved all datasets:")
    print(f"  All: {len(df_all)}")
    print(f"  Medium+: {len(df_medium)}")
    print(f"  Both nearest (Fra): {len(df_nearest)}")
    print(f"  Both close: {len(df_close)}")
    
    # Features e targets
    features = [
        'volume_ml',
        'mean_tortuosity',
        'std_peripheral_diameter_mm',
        'central_to_peripheral_diameter_ratio',
        'mean_peripheral_branch_volume_mm3',
        'mean_lung_density_HU',
        'histogram_entropy'
    ]
    
    targets = [
        'FVC_percent_week0',
        'FVC_percent_week52',
        'FVC_drop_absolute',
        'FVC_drop_percent'
    ]
    
    # ESECUZIONE PREDIZIONI SU MULTIPLI DATASET
    datasets = [df_medium, df_nearest, df_close]
    dataset_names = ['Medium+ Quality', 'Both Nearest (Fra)', 'Both Close']
    
    all_results = run_predictions_on_datasets(datasets, dataset_names, features, targets)
    
    # Crea summary table comparativa
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Dataset': result['dataset'],
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
    
    # Salva summary principale
    main_summary = df_summary[df_summary['Dataset'] == 'Medium+ Quality'].copy()
    main_summary = main_summary.sort_values(['Target', 'R²'], ascending=[True, False])
    main_summary.to_csv(OUTPUT_DIR / "prediction_summary_main.csv", index=False)
    
    # Salva summary comparativa
    df_summary.to_csv(OUTPUT_DIR / "prediction_summary_comparative.csv", index=False)
    
    # Genera plots per i migliori risultati
    print(f"\n{'='*80}")
    print("GENERATING PLOTS FOR BEST RESULTS")
    print(f"{'='*80}")
    
    # Crea directory per plots
    plots_dir = OUTPUT_DIR / "compromise_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Trova i migliori risultati per ogni dataset-target
    for dataset in dataset_names:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        
        for target in targets:
            target_results = [r for r in dataset_results if r['target'] == target]
            if not target_results:
                continue
            
            # Prendi il migliore per R²
            best_result = max(target_results, key=lambda x: x['r2'])
            
            if best_result['r2'] > 0.1:  # Solo se decente
                plot_path = plots_dir / f"{dataset}_{best_result['feature']}_{target}.png"
                plot_correlation(best_result, plot_path)
    
    # ANALISI RIASSUNTIVA FINALE
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS - COMPARISON")
    print(f"{'='*80}")
    
    # Per ogni target, confronta tra dataset
    for target in targets:
        print(f"\n{target}:")
        
        target_results = df_summary[df_summary['Target'] == target]
        
        for dataset in dataset_names:
            dataset_target = target_results[target_results['Dataset'] == dataset]
            if len(dataset_target) > 0:
                best_row = dataset_target.loc[dataset_target['R²'].idxmax()]
                print(f"  {dataset:25s}: {best_row['Feature']:35s} R²={best_row['R²']:.3f}, n={best_row['n_samples']}")
    
    # Consigli finali
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    print("1. Per analisi principale: usa 'Medium+ Quality' dataset")
    print("2. Per verifica robustezza: confronta con 'Both Nearest (Fra)'")
    print("3. Feature più promettenti: mean_lung_density_HU, histogram_entropy")
    print("4. Target più predicibili: FVC_percent_week52, FVC_percent_week0")
    print(f"\n✓ All results saved in: {OUTPUT_DIR}")
    print(f"✓ Comparative analysis saved in: prediction_summary_comparative.csv")
    print(f"{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()