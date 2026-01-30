import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURAZIONE AGGIORNATA
# ============================================================

VALIDATION_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\OSIC_validation.csv")
RESULTS_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC_combined")
PERCENTAGE_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\percentage_result.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\single_feature_prediction")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. FUNZIONI DI CARICAMENTO DATI (modificate)
# ============================================================

def load_fvc_percentage_data():
    """Carica i dati FVC interpolati da Francesco"""
    print("Loading FVC percentage data (Francesco's interpolation)...")
    
    try:
        df = pd.read_csv(PERCENTAGE_CSV)
        print(f"  Loaded {len(df)} patients")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Verifica colonne necessarie
        required_cols = ['Patient', 'FVCpercent(week0)', 'FVCpercent(week42)']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  WARNING: Missing columns: {missing_cols}")
            print(f"  Available columns: {df.columns.tolist()}")
            return None
        
        return df
    except Exception as e:
        print(f"  ERROR loading percentage_result.csv: {e}")
        return None


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
        print(f"  Warning: Could not load {case_name}: {e}")
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
    """Extract patient ID from case name"""
    return case_name.replace("_gaussian", "")


# ============================================================
# 2. PREPARAZIONE DATASET PER PREDIZIONE (modificata)
# ============================================================

def prepare_prediction_dataset(reliable_cases, fvc_percentage_df):
    """
    Crea un dataset per la predizione con:
    - FVC% @week0 (da Francesco)
    - FVC% @week52 (da Francesco)
    - % drop @1year (calcolato)
    - Tutte le metriche airway/parenchymal
    """
    print("\nPreparing prediction dataset...")
    
    rows = []
    patients_with_both_data = 0
    
    # Carica tutti i dati FVC interpolati
    if fvc_percentage_df is None:
        print("  ERROR: No FVC percentage data available")
        return pd.DataFrame()
    
    for idx, case_row in reliable_cases.iterrows():
        case_name = case_row['case']
        patient_id = extract_patient_id(case_name)
        
        # Cerca i dati FVC interpolati per questo paziente
        patient_fvc_data = fvc_percentage_df[fvc_percentage_df['Patient'] == patient_id]
        
        if len(patient_fvc_data) == 0:
            # Paziente non presente nel dataset interpolato
            continue
        
        # Load advanced metrics
        advanced = load_advanced_metrics(case_name)
        if advanced is None:
            continue
        
        # Load parenchymal metrics
        parenchymal = load_parenchymal_metrics(case_name)
        
        # Prendi i dati FVC interpolati
        fvc_data = patient_fvc_data.iloc[0]
        
        # Estrai FVC percentuale a week0 e week52
        fvc_percent_week0 = fvc_data['FVCpercent(week0)']
        
        # Controlla se esiste FVCpercent(week52) o usiamo FVCpercent(week42)
        if 'FVCpercent(week52)' in fvc_data:
            fvc_percent_week52 = fvc_data['FVCpercent(week52)']
        elif 'FVCpercent(week42)' in fvc_data:
            fvc_percent_week52 = fvc_data['FVCpercent(week42)']
            print(f"  Note: Using week42 for patient {patient_id} (week52 not available)")
        else:
            print(f"  Warning: No week52 or week42 data for patient {patient_id}")
            continue
        
        # Calcola % drop FVC a 1 anno
        if fvc_percent_week0 > 0 and not pd.isna(fvc_percent_week0) and not pd.isna(fvc_percent_week52):
            fvc_drop_absolute = fvc_percent_week0 - fvc_percent_week52  # Drop assoluto in punti percentuali
            fvc_drop_percent = (fvc_drop_absolute / fvc_percent_week0) * 100  # Drop relativo in %
        else:
            fvc_drop_absolute = np.nan
            fvc_drop_percent = np.nan
        
        # Create row with ALL available metrics
        row = {
            'patient': patient_id,
            'case': case_name,
            
            # Target variables (da Francesco)
            'FVC_percent_week0': fvc_percent_week0,
            'FVC_percent_week52': fvc_percent_week52,
            'FVC_drop_absolute': fvc_drop_absolute,    # Calo assoluto in punti %
            'FVC_drop_percent': fvc_drop_percent,      # Calo relativo in %
            
            # METRICHE AIRWAY (solo quelle specificate)
            'volume_ml': case_row['volume_ml'],
            'mean_tortuosity': advanced.get('mean_tortuosity'),
            'std_peripheral_diameter_mm': advanced.get('std_peripheral_diameter_mm'),
            'central_to_peripheral_diameter_ratio': advanced.get('central_to_peripheral_diameter_ratio'),
            'mean_peripheral_branch_volume_mm3': advanced.get('mean_peripheral_branch_volume_mm3'),
        }
        
        # Add parenchymal metrics if available
        if parenchymal is not None:
            row.update({
                'mean_lung_density_HU': parenchymal.get('mean_lung_density_HU'),
                'histogram_entropy': parenchymal.get('histogram_entropy'),
            })
        
        rows.append(row)
        patients_with_both_data += 1
    
    df = pd.DataFrame(rows)
    
    if len(df) == 0:
        print("  ERROR: No patients with both FVC data and airway metrics")
        return pd.DataFrame()
    
    # Filtra pazienti con dati completi
    required_cols = ['FVC_percent_week0', 'FVC_percent_week52']
    df_complete = df.dropna(subset=required_cols)
    
    print(f"\nDataset prepared:")
    print(f"  Total patients with airway metrics: {len(reliable_cases)}")
    print(f"  Patients with FVC interpolated data: {patients_with_both_data}")
    print(f"  Complete cases for analysis: {len(df_complete)}")
    
    if len(df_complete) > 0:
        print(f"  FVC week0 range: {df_complete['FVC_percent_week0'].min():.1f}% to {df_complete['FVC_percent_week0'].max():.1f}%")
        print(f"  FVC week52 range: {df_complete['FVC_percent_week52'].min():.1f}% to {df_complete['FVC_percent_week52'].max():.1f}%")
        print(f"  FVC drop range: {df_complete['FVC_drop_absolute'].min():.1f} to {df_complete['FVC_drop_absolute'].max():.1f} points")
        print(f"  % drop range: {df_complete['FVC_drop_percent'].min():.1f}% to {df_complete['FVC_drop_percent'].max():.1f}%")
    
    return df_complete


# ============================================================
# 3. SINGLE-FEATURE PREDICTION CON LEAVE-ONE-OUT
# ============================================================

def leave_one_out_predict(df, feature_name, target_name, model_type='linear'):
    """
    Esegue predizione leave-one-out per una singola feature.
    
    Parameters:
    -----------
    df : DataFrame
        Dati completi
    feature_name : str
        Nome della feature predittiva
    target_name : str
        Nome del target (FVC_percent_week0, FVC_percent_week52, FVC_drop_absolute, FVC_drop_percent)
    model_type : str
        Tipo di modello ('linear', 'poly2', 'poly3')
    
    Returns:
    --------
    results : dict
        Risultati della predizione per ogni paziente
    """
    
    # Filtra dati non NaN
    valid_data = df[[feature_name, target_name]].dropna()
    X = valid_data[feature_name].values.reshape(-1, 1)
    y = valid_data[target_name].values
    
    if len(X) < 10:  # Minimo 10 pazienti per analisi
        print(f"    Skipping: only {len(X)} valid samples")
        return None
    
    # Lista per i risultati
    predictions = []
    actuals = []
    patient_ids = []
    features_used = []
    
    # Leave-One-Out Cross Validation
    print(f"    LOOCV with {len(X)} samples...")
    
    for i in range(len(X)):
        # Training set: tutti tranne il paziente i
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        
        # Test set: solo il paziente i
        X_test = X[i].reshape(1, -1)
        y_test = y[i]
        
        # Crea e addestra il modello
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)[0]
            
        elif model_type == 'poly2':
            # Regressione polinomiale di grado 2
            X_train_poly = np.column_stack([X_train, X_train**2])
            X_test_poly = np.column_stack([X_test, X_test**2])
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)[0]
            
        elif model_type == 'poly3':
            # Regressione polinomiale di grado 3
            X_train_poly = np.column_stack([X_train, X_train**2, X_train**3])
            X_test_poly = np.column_stack([X_test, X_test**2, X_test**3])
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)[0]
        else:
            continue
        
        predictions.append(y_pred)
        actuals.append(y_test)
        patient_ids.append(valid_data.index[i])
        features_used.append(X[i][0])
    
    # Calcola metriche di performance
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Errori
    errors = predictions - actuals
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    
    # Percentuali di errore (per FVC in percentuale)
    if 'FVC_percent' in target_name:
        percentage_errors = (np.abs(errors) / actuals) * 100
        mape = np.mean(percentage_errors[actuals != 0])  # Evita divisione per zero
    else:
        mape = np.mean(np.abs(errors))
    
    # Correlazioni
    pearson_r, pearson_p = pearsonr(actuals, predictions)
    spearman_r, spearman_p = spearmanr(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    results = {
        'feature': feature_name,
        'target': target_name,
        'model_type': model_type,
        'n_samples': len(X),
        'predictions': predictions,
        'actuals': actuals,
        'errors': errors,
        'patient_ids': patient_ids,
        'features_used': features_used,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r2': r2
    }
    
    return results


def analyze_all_features_single_target(df, target_name, model_types=['linear', 'poly2', 'poly3']):
    """
    Analizza tutte le feature per predire un singolo target.
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING ALL FEATURES FOR TARGET: {target_name}")
    print(f"{'='*80}")
    
    # Lista di tutte le feature da testare
    features = [
        'volume_ml',
        'mean_tortuosity',
        'std_peripheral_diameter_mm',
        'central_to_peripheral_diameter_ratio',
        'mean_peripheral_branch_volume_mm3',
        'mean_lung_density_HU',
        'histogram_entropy'
    ]
    
    all_results = []
    feature_results_list = []
    
    for feature in features:
        if feature not in df.columns:
            print(f"\n{'─'*60}")
            print(f"Feature: {feature} - NOT AVAILABLE")
            continue
        
        print(f"\n{'─'*60}")
        print(f"Feature: {feature}")
        print(f"{'─'*60}")
        
        # Controlla dati disponibili
        n_samples = df[[feature, target_name]].dropna().shape[0]
        print(f"  Available samples: {n_samples}")
        
        if n_samples < 10:
            print(f"  Skipping: insufficient data ({n_samples} < 10)")
            continue
        
        feature_results = {'feature': feature, 'target': target_name, 'n_samples': n_samples}
        
        for model_type in model_types:
            result = leave_one_out_predict(df, feature, target_name, model_type)
            
            if result is None:
                print(f"  {model_type}: No valid results")
                continue
            
            # Salva i risultati
            feature_results[f'{model_type}_mse'] = result['mse']
            feature_results[f'{model_type}_mae'] = result['mae']
            feature_results[f'{model_type}_rmse'] = result['rmse']
            feature_results[f'{model_type}_r2'] = result['r2']
            feature_results[f'{model_type}_pearson_r'] = result['pearson_r']
            feature_results[f'{model_type}_pearson_p'] = result['pearson_p']
            
            print(f"  {model_type}:")
            print(f"    MSE = {result['mse']:.2f}, MAE = {result['mae']:.2f}")
            print(f"    R² = {result['r2']:.3f}, Pearson r = {result['pearson_r']:.3f} (p={result['pearson_p']:.4f})")
            
            # Salva i risultati completi per i plot
            if model_type == 'linear':  # Salva solo per linear per i plot
                result['model_type_for_plots'] = model_type
                all_results.append(result)
        
        # Trova il modello migliore per questa feature
        best_model = None
        best_r2 = -np.inf
        
        for model_type in model_types:
            r2_key = f'{model_type}_r2'
            if r2_key in feature_results:
                if feature_results[r2_key] > best_r2:
                    best_r2 = feature_results[r2_key]
                    best_model = model_type
        
        if best_model:
            print(f"\n  ✓ Best model: {best_model} (R² = {best_r2:.3f})")
            feature_results['best_model'] = best_model
            feature_results['best_r2'] = best_r2
        
        feature_results_list.append(feature_results)
    
    return all_results, feature_results_list


# ============================================================
# 4. VISUALIZZAZIONI (come prima)
# ============================================================

def plot_correlation_plot(results, output_path):
    """Plot correlazione tra valori predetti e reali"""
    if results is None:
        return
    
    predictions = results['predictions']
    actuals = results['actuals']
    feature = results['feature']
    target = results['target']
    r2 = results['r2']
    pearson_r = results['pearson_r']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(actuals, predictions, alpha=0.6, s=60, 
                        edgecolors='black', linewidth=0.5, color='steelblue')
    
    # Linea di perfetta predizione (y=x)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', alpha=0.7, linewidth=2, label='Perfect prediction')
    
    # Regression line
    z = np.polyfit(actuals, predictions, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min_val, max_val, 100)
    ax.plot(x_trend, p(x_trend), 'g-', alpha=0.8, linewidth=2, label='Regression line')
    
    ax.set_xlabel(f'Actual {target}', fontsize=14)
    ax.set_ylabel(f'Predicted {target}', fontsize=14)
    ax.set_title(f'{feature} → {target}\nLeave-One-Out Prediction\nR² = {r2:.3f}, Pearson r = {pearson_r:.3f}', 
                 fontsize=16, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Aggiungi testo con metriche
    metrics_text = f'MSE = {results["mse"]:.2f}\nMAE = {results["mae"]:.2f}\nRMSE = {results["rmse"]:.2f}\nMAPE = {results["mape"]:.1f}%'
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Correlation plot saved: {output_path}")


def plot_bland_altman(results, output_path):
    """Plot Bland-Altman per analisi degli errori"""
    if results is None:
        return
    
    predictions = results['predictions']
    actuals = results['actuals']
    feature = results['feature']
    target = results['target']
    
    # Calcola differenze e medie
    differences = predictions - actuals
    means = (predictions + actuals) / 2
    
    # Statistiche
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(means, differences, alpha=0.6, s=50, 
              edgecolors='black', linewidth=0.5, color='coral')
    
    # Linea media
    ax.axhline(y=mean_diff, color='blue', linestyle='-', linewidth=2, 
               label=f'Mean difference: {mean_diff:.2f}')
    
    # Limiti di accordo (mean ± 1.96*SD)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    ax.axhline(y=upper_limit, color='red', linestyle='--', linewidth=1.5,
               label=f'+1.96 SD: {upper_limit:.2f}')
    ax.axhline(y=lower_limit, color='red', linestyle='--', linewidth=1.5,
               label=f'-1.96 SD: {lower_limit:.2f}')
    
    # Area tra i limiti
    ax.fill_between([means.min(), means.max()], lower_limit, upper_limit,
                   alpha=0.1, color='red')
    
    ax.set_xlabel(f'Mean of actual and predicted {target}', fontsize=14)
    ax.set_ylabel('Difference (Predicted - Actual)', fontsize=14)
    ax.set_title(f'Bland-Altman Plot: {feature} → {target}\nLeave-One-Out Prediction', 
                 fontsize=16, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Aggiungi testo con statistiche
    stats_text = (f'Mean difference = {mean_diff:.2f}\n'
                  f'SD of differences = {std_diff:.2f}\n'
                  f'95% Limits of Agreement:\n'
                  f'  [{lower_limit:.2f}, {upper_limit:.2f}]')
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Bland-Altman plot saved: {output_path}")


def plot_error_distribution(results, output_path):
    """Plot distribuzione degli errori"""
    if results is None:
        return
    
    errors = results['errors']
    feature = results['feature']
    target = results['target']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Istogramma degli errori
    ax1.hist(errors, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2,
                label=f'Mean error: {np.mean(errors):.2f}')
    
    ax1.set_xlabel('Prediction Error', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Error Distribution: {feature} → {target}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot per normalità
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot for Normality Check', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Test di normalità
    _, p_value = stats.shapiro(errors)
    ax2.text(0.05, 0.95, f'Shapiro-Wilk p = {p_value:.4f}', transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Error distribution plot saved: {output_path}")


def plot_feature_vs_target(df, feature_name, target_name, output_path):
    """Plot semplice feature vs target con regressione"""
    if feature_name not in df.columns or target_name not in df.columns:
        return
    
    data = df[[feature_name, target_name]].dropna()
    if len(data) < 5:
        return
    
    X = data[feature_name].values
    y = data[target_name].values
    
    # Calcola correlazione
    r, p = pearsonr(X, y)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(X, y, alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='purple')
    
    # Regression line
    z = np.polyfit(X, y, 1)
    p_fit = np.poly1d(z)
    x_trend = np.linspace(X.min(), X.max(), 100)
    ax.plot(x_trend, p_fit(x_trend), 'r-', alpha=0.8, linewidth=2)
    
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel(target_name, fontsize=12)
    ax.set_title(f'{feature_name} vs {target_name}\nPearson r = {r:.3f} (p={p:.4f})', 
                 fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Feature vs target plot saved: {output_path}")


def create_summary_table(all_results, feature_results_list, output_path):
    """Crea tabella riassuntiva delle performance"""
    summary_data = []
    
    # Per i risultati dettagliati
    for result in all_results:
        if isinstance(result, dict) and 'feature' in result and 'target' in result:
            row = {
                'Feature': result['feature'],
                'Target': result['target'],
                'Model': 'linear',
                'n_samples': result.get('n_samples', 0),
                'MSE': result.get('mse', np.nan),
                'MAE': result.get('mae', np.nan),
                'RMSE': result.get('rmse', np.nan),
                'R²': result.get('r2', np.nan),
                'Pearson r': result.get('pearson_r', np.nan),
                'Pearson p': result.get('pearson_p', np.nan)
            }
            summary_data.append(row)
    
    # Per i risultati di tutti i modelli
    for feature_result in feature_results_list:
        if isinstance(feature_result, dict) and 'feature' in feature_result:
            feature = feature_result['feature']
            target = feature_result['target']
            
            for model in ['linear', 'poly2', 'poly3']:
                r2_key = f'{model}_r2'
                if r2_key in feature_result and not pd.isna(feature_result[r2_key]):
                    row = {
                        'Feature': feature,
                        'Target': target,
                        'Model': model,
                        'n_samples': feature_result.get('n_samples', 0),
                        'MSE': feature_result.get(f'{model}_mse', np.nan),
                        'MAE': feature_result.get(f'{model}_mae', np.nan),
                        'RMSE': feature_result.get(f'{model}_rmse', np.nan),
                        'R²': feature_result.get(f'{model}_r2', np.nan),
                        'Pearson r': feature_result.get(f'{model}_pearson_r', np.nan),
                        'Pearson p': feature_result.get(f'{model}_pearson_p', np.nan)
                    }
                    summary_data.append(row)
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        # Salva tabella completa
        df_summary.to_csv(output_path, index=False)
        print(f"  ✓ Complete summary table saved: {output_path}")
        
        # Crea tabella semplificata con solo migliori modelli
        simplified_path = output_path.parent / "prediction_summary_best_models.csv"
        best_models = []
        
        for target in df_summary['Target'].unique():
            for feature in df_summary['Feature'].unique():
                feature_target_data = df_summary[(df_summary['Feature'] == feature) & 
                                                 (df_summary['Target'] == target)]
                if not feature_target_data.empty:
                    best_row = feature_target_data.loc[feature_target_data['R²'].idxmax()]
                    best_models.append(best_row)
        
        df_best = pd.DataFrame(best_models)
        df_best = df_best.sort_values(['Target', 'R²'], ascending=[True, False])
        df_best.to_csv(simplified_path, index=False)
        print(f"  ✓ Best models summary saved: {simplified_path}")
        
        # Print top performers
        print(f"\n{'='*60}")
        print("TOP PERFORMING FEATURES (Best Models):")
        print(f"{'='*60}")
        
        for target in df_best['Target'].unique():
            print(f"\nTarget: {target}")
            print(f"{'-'*40}")
            print(f"{'Feature':<30} {'Model':<10} {'R²':<8} {'MAE':<10}")
            
            target_data = df_best[df_best['Target'] == target].head(5)
            for _, row in target_data.iterrows():
                print(f"{row['Feature']:<30} {row['Model']:<10} {row['R²']:<8.3f} {row['MAE']:<10.2f}")
        
        return df_summary, df_best
    
    return None, None


def plot_comparison_all_features(df_best, output_path):
    """Plot comparativo delle migliori features per ogni target"""
    if df_best is None or len(df_best) == 0:
        return
    
    # Crea plot separato per ogni target
    targets = df_best['Target'].unique()
    
    for target in targets:
        target_data = df_best[df_best['Target'] == target].sort_values('R²', ascending=True)
        
        if len(target_data) == 0:
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(target_data) * 0.4)))
        
        y_pos = np.arange(len(target_data))
        
        # Plot R²
        bars1 = ax1.barh(y_pos, target_data['R²'], alpha=0.7, 
                        color=['green' if x >= 0.3 else ('orange' if x >= 0.1 else 'red') 
                              for x in target_data['R²']])
        
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax1.axvline(x=0.3, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        
        # Aggiungi etichette con valori
        for i, (bar, r2, model) in enumerate(zip(bars1, target_data['R²'], target_data['Model'])):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{r2:.3f} ({model})', va='center', fontsize=9)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(target_data['Feature'])
        ax1.set_xlabel('R² Score', fontsize=12)
        ax1.set_title(f'Best R² Scores for {target}\n(Leave-One-Out Prediction)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot MAE
        bars2 = ax2.barh(y_pos, target_data['MAE'], alpha=0.7,
                        color=['green' if x <= 5 else ('orange' if x <= 10 else 'red') 
                              for x in target_data['MAE']])
        
        # Aggiungi etichette con valori
        for i, (bar, mae) in enumerate(zip(bars2, target_data['MAE'])):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{mae:.2f}', va='center', fontsize=9)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([])
        ax2.set_xlabel('MAE', fontsize=12)
        ax2.set_title(f'MAE for {target}\n(Leave-One-Out Prediction)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        target_output_path = output_path.parent / f"comparison_{target}.png"
        plt.savefig(target_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Comparison plot for {target} saved: {target_output_path}")


# ============================================================
# 5. ANALISI PRINCIPALE (modificata)
# ============================================================

def main():
    print("="*80)
    print("SINGLE-FEATURE PREDICTION ANALYSIS")
    print("Using Francesco's interpolated FVC data")
    print("="*80)
    
    # Carica dati FVC interpolati
    fvc_percentage_df = load_fvc_percentage_data()
    if fvc_percentage_df is None:
        print("\nERROR: Cannot load percentage_result.csv")
        return
    
    # Carica dati airway
    reliable = load_validation_results()
    
    # Prepara dataset finale
    df = prepare_prediction_dataset(reliable, fvc_percentage_df)
    
    if len(df) == 0:
        print("\nERROR: No complete cases found for prediction analysis!")
        return
    
    # Salva dataset
    dataset_path = OUTPUT_DIR / "prediction_dataset_final.csv"
    df.to_csv(dataset_path, index=False)
    print(f"\n✓ Final prediction dataset saved: {dataset_path}")
    
    # Crea cartelle per output
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Analizza tutte le feature per ogni target
    # Usiamo 4 target come richiesto dal professore
    targets = ['FVC_percent_week0', 'FVC_percent_week52', 'FVC_drop_absolute', 'FVC_drop_percent']
    
    all_detailed_results = []  # Per plots dettagliati
    all_feature_results = []   # Per tabella riassuntiva
    
    for target in targets:
        target_dir = plots_dir / target
        target_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR TARGET: {target}")
        print(f"{'='*80}")
        
        # Plot feature vs target (correlazioni semplici)
        print("\n1. Simple correlations (feature vs target):")
        features = ['volume_ml', 'mean_tortuosity', 'std_peripheral_diameter_mm',
                   'central_to_peripheral_diameter_ratio', 'mean_peripheral_branch_volume_mm3',
                   'mean_lung_density_HU', 'histogram_entropy']
        
        for feature in features:
            if feature in df.columns:
                plot_path = target_dir / f"scatter_{feature}_vs_{target}.png"
                plot_feature_vs_target(df, feature, target, plot_path)
        
        # Analisi leave-one-out
        print(f"\n2. Leave-One-Out Prediction Analysis for {target}:")
        detailed_results, feature_results = analyze_all_features_single_target(df, target)
        
        # Crea plots dettagliati per ogni feature (solo modello lineare)
        print(f"\n3. Creating detailed plots for {target}:")
        for result in detailed_results:
            if isinstance(result, dict) and 'feature' in result and 'predictions' in result:
                feature = result['feature']
                
                # Correlation plot
                corr_path = target_dir / f"correlation_{feature}_{target}.png"
                plot_correlation_plot(result, corr_path)
                
                # Bland-Altman plot
                ba_path = target_dir / f"bland_altman_{feature}_{target}.png"
                plot_bland_altman(result, ba_path)
                
                # Error distribution
                err_path = target_dir / f"error_distribution_{feature}_{target}.png"
                plot_error_distribution(result, err_path)
                
                all_detailed_results.append(result)
        
        all_feature_results.extend(feature_results)
    
    # Riassunto finale
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    # Crea tabelle riassuntive
    summary_path = OUTPUT_DIR / "prediction_summary_complete.csv"
    df_summary, df_best = create_summary_table(all_detailed_results, all_feature_results, summary_path)
    
    # Plot comparativo
    if df_best is not None:
        plot_comparison_all_features(df_best, OUTPUT_DIR / "feature_comparison.png")
    
    # Statistiche finali
    print(f"\n{'='*80}")
    print("DATASET STATISTICS:")
    print(f"{'='*80}")
    
    print(f"\nTotal patients with complete data: {len(df)}")
    print(f"\nFVC Statistics (using Francesco's interpolation):")
    print(f"  FVC% @week0: {df['FVC_percent_week0'].mean():.1f} ± {df['FVC_percent_week0'].std():.1f}%")
    print(f"  FVC% @week52: {df['FVC_percent_week52'].mean():.1f} ± {df['FVC_percent_week52'].std():.1f}%")
    print(f"  FVC drop (absolute): {df['FVC_drop_absolute'].mean():.1f} ± {df['FVC_drop_absolute'].std():.1f} points")
    print(f"  FVC drop (relative): {df['FVC_drop_percent'].mean():.1f} ± {df['FVC_drop_percent'].std():.1f}%")
    
    # Analisi progressione
    print(f"\nProgression Analysis:")
    progressors = df[df['FVC_drop_absolute'] > 5]  # >5 points decline = progressione
    print(f"  Patients with >5% points decline: {len(progressors)} ({len(progressors)/len(df)*100:.1f}%)")
    print(f"  Mean decline in progressors: {progressors['FVC_drop_absolute'].mean():.1f} points")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION GUIDE:")
    print(f"{'='*80}")
    print("\n1. R² Interpretation:")
    print("   R² ≥ 0.5: Excellent predictive power")
    print("   0.3 ≤ R² < 0.5: Good predictive power")
    print("   0.1 ≤ R² < 0.3: Fair predictive power")
    print("   R² < 0.1: Poor predictive power")
    
    print("\n2. MAE Interpretation (for FVC%):")
    print("   MAE < 5%: Excellent accuracy")
    print("   5% ≤ MAE < 10%: Good accuracy")
    print("   10% ≤ MAE < 15%: Fair accuracy")
    print("   MAE ≥ 15%: Poor accuracy")
    
    print("\n3. Clinical Significance:")
    print("   FVC decline > 10% in 1 year: Progressive disease")
    print("   FVC decline 5-10%: Stable/mild progression")
    print("   FVC decline < 5%: Stable disease")
    
    print(f"\n{'='*80}")
    print("RESULTS SAVED TO:")
    print(f"{'='*80}")
    print(f"\n1. Dataset:")
    print(f"   {dataset_path}")
    
    print(f"\n2. Summary tables:")
    print(f"   {summary_path}")
    print(f"   {OUTPUT_DIR / 'prediction_summary_best_models.csv'}")
    
    print(f"\n3. Plots directory:")
    print(f"   {plots_dir}/")
    for target in targets:
        print(f"   ├── {target}/")
    
    print(f"\n4. Comparison plots:")
    print(f"   {OUTPUT_DIR / 'comparison_*.png'}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"Answers to professor's questions:")
    print(f"1. Single-feature models ✓")
    print(f"2. FVC@0weeks prediction ✓")
    print(f"3. FVC@1year prediction ✓")
    print(f"4. %FVC drop prediction ✓")
    print(f"5. Leave-one-out validation ✓")
    print(f"6. MSE calculation ✓")
    print(f"7. Correlation & Bland-Altman plots ✓")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()