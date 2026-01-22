import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\osic_correlation_analysis\integrated_dataset.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\multivariate_analysis")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_and_prepare_data():
    """Load integrated dataset and prepare for analysis"""
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    df = pd.read_csv(INPUT_CSV)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Unique patients: {df['patient'].nunique()}")
    print(f"Week range: {df['week'].min()} to {df['week'].max()}")
    
    return df


def create_baseline_dataset(df):
    """Create dataset with baseline (week 0) measurements"""
    print("\nCreating baseline dataset (week 0)...")
    
    # Get first measurement for each patient (baseline)
    baseline = df.groupby('patient').first().reset_index()
    
    print(f"  Baseline patients: {len(baseline)}")
    print(f"  Mean FVC: {baseline['FVC'].mean():.1f} ml")
    
    return baseline


def calculate_decline_rates(df):
    """Calculate FVC decline rate for each patient"""
    print("\nCalculating FVC decline rates...")
    
    decline_data = []
    
    for patient in df['patient'].unique():
        patient_data = df[df['patient'] == patient].sort_values('week')
        
        if len(patient_data) >= 2:
            # Linear fit
            weeks = patient_data['week'].values
            fvc = patient_data['FVC'].values
            
            coeffs = np.polyfit(weeks, fvc, 1)
            decline_rate = coeffs[0]  # ml/week
            
            # Get baseline metrics
            baseline_row = patient_data.iloc[0]
            
            decline_data.append({
                'patient': patient,
                'decline_rate': decline_rate,
                'baseline_FVC': baseline_row['FVC'],
                'n_measurements': len(patient_data),
                'follow_up_weeks': weeks.max() - weeks.min(),
                
                # Airway metrics
                'volume_ml': baseline_row['volume_ml'],
                'branch_count': baseline_row['branch_count'],
                'max_generation': baseline_row['max_generation'],
                'pc_ratio': baseline_row['pc_ratio'],
                'tapering_ratio': baseline_row['tapering_ratio'],
                'mean_tortuosity': baseline_row['mean_tortuosity'],
                'mean_diameter': baseline_row['mean_diameter'],
                'peripheral_volume_ratio': baseline_row['peripheral_volume_ratio'],
                'central_volume_ratio': baseline_row['central_volume_ratio'],
                
                # Demographics
                'Age': baseline_row['Age'],
                'Sex': baseline_row['Sex'],
                'SmokingStatus': baseline_row['SmokingStatus'],
            })
    
    decline_df = pd.DataFrame(decline_data)
    
    print(f"  Patients with decline data: {len(decline_df)}")
    print(f"  Mean decline rate: {decline_df['decline_rate'].mean():.2f} ml/week")
    print(f"  Median decline rate: {decline_df['decline_rate'].median():.2f} ml/week")
    
    return decline_df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_engineered_features(df):
    """Create new features by combining existing metrics"""
    print("\nCreating engineered features...")
    
    original_columns = df.columns.tolist()
    df = df.copy()
    
    # Volume-based combinations
    df['volume_per_branch'] = df['volume_ml'] / df['branch_count']
    df['volume_per_generation'] = df['volume_ml'] / df['max_generation']
    
    # Diameter-based combinations
    df['diameter_volume_product'] = df['mean_diameter'] * df['volume_ml']
    df['diameter_tapering_ratio'] = df['mean_diameter'] * df['tapering_ratio']
    
    # Complexity metrics
    df['branch_density'] = df['branch_count'] / df['volume_ml']
    df['generation_branch_ratio'] = df['max_generation'] / df['branch_count']
    
    # Peripheral/Central combinations
    df['pc_volume_product'] = df['pc_ratio'] * df['volume_ml']
    df['peripheral_branch_density'] = df['peripheral_volume_ratio'] * df['branch_count']
    
    # Tortuosity combinations
    df['tortuosity_volume'] = df['mean_tortuosity'] * df['volume_ml']
    df['tortuosity_tapering'] = df['mean_tortuosity'] * df['tapering_ratio']
    
    # Volume-Tapering interaction (suggested by user)
    df['volume_tapering_product'] = df['volume_ml'] * df['tapering_ratio']
    df['volume_tapering_ratio'] = df['volume_ml'] / (df['tapering_ratio'] + 0.001)
    
    # Composite airway health index
    df['airway_health_index'] = (
        df['volume_ml'] * 0.3 + 
        df['branch_count'] * 0.2 + 
        df['max_generation'] * 0.2 +
        df['tapering_ratio'] * 100 * 0.3
    )
    
    new_features = [col for col in df.columns if col not in original_columns]
    print(f"  Created {len(new_features)} new features")
    
    return df


# ============================================================
# MODEL TRAINING & EVALUATION
# ============================================================

def prepare_features_target(df, target_col, exclude_cols=None):
    """Prepare feature matrix and target vector"""
    if exclude_cols is None:
        exclude_cols = ['case', 'patient', 'week', 'Sex', 'SmokingStatus']
    
    # Exclude target and non-numeric columns
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols + [target_col] 
                   and df[col].dtype in ['float64', 'int64']]
    
    print(f"\n  Initial samples: {len(df)}")
    print(f"  Features selected: {len(feature_cols)}")
    
    # Check for NaN values
    nan_counts = df[feature_cols + [target_col]].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    
    if len(cols_with_nan) > 0:
        print(f"\n  Columns with NaN values:")
        for col, count in cols_with_nan.items():
            print(f"    {col}: {count} NaN values ({count/len(df)*100:.1f}%)")
        
        # Remove columns with too many NaN (>50%)
        cols_to_remove = [col for col in cols_with_nan.index if col in feature_cols and cols_with_nan[col] > len(df) * 0.5]
        if len(cols_to_remove) > 0:
            print(f"\n  Removing {len(cols_to_remove)} columns with >50% NaN:")
            for col in cols_to_remove:
                print(f"    - {col}")
            feature_cols = [col for col in feature_cols if col not in cols_to_remove]
    
    # Remove rows with NaN in features or target
    df_clean = df[feature_cols + [target_col]].dropna()
    
    print(f"  Final samples after dropna: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("\n  ERROR: No samples remaining after removing NaN values!")
        print("  This likely means the target column or critical features have all NaN values.")
        raise ValueError(f"No valid samples for target '{target_col}' after cleaning")
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    return X, y, feature_cols


def train_multiple_regression(X, y, feature_names):
    """Train Multiple Linear Regression"""
    print("\n" + "="*80)
    print("MULTIPLE LINEAR REGRESSION")
    print("="*80)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Predictions
    y_pred = model.predict(X_scaled)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    print(f"\nTraining Performance:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE = {mae:.2f}")
    print(f"\nCross-Validation R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance (absolute coefficients)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nTop 10 Features by Coefficient:")
    print(feature_importance.head(10).to_string(index=False))
    
    return {
        'model': model,
        'scaler': scaler,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'y_pred': y_pred
    }


def train_lasso_regression(X, y, feature_names):
    """Train LASSO regression with cross-validation"""
    print("\n" + "="*80)
    print("LASSO REGRESSION (L1 Regularization)")
    print("="*80)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find best alpha using cross-validation
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_scaled, y)
    
    print(f"\nBest alpha: {lasso_cv.alpha_:.6f}")
    
    # Train final model
    model = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
    model.fit(X_scaled, y)
    
    # Predictions
    y_pred = model.predict(X_scaled)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    print(f"\nTraining Performance:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE = {mae:.2f}")
    print(f"\nCross-Validation R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance (non-zero coefficients)
    non_zero_mask = model.coef_ != 0
    feature_importance = pd.DataFrame({
        'feature': np.array(feature_names)[non_zero_mask],
        'coefficient': model.coef_[non_zero_mask],
        'abs_coefficient': np.abs(model.coef_[non_zero_mask])
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\nFeatures selected: {len(feature_importance)} / {len(feature_names)}")
    print(f"Features eliminated: {len(feature_names) - len(feature_importance)}")
    
    if len(feature_importance) > 0:
        print(f"\nTop Selected Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    return {
        'model': model,
        'scaler': scaler,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'y_pred': y_pred,
        'n_features_selected': len(feature_importance)
    }


def train_random_forest(X, y, feature_names):
    """Train Random Forest for feature importance"""
    print("\n" + "="*80)
    print("RANDOM FOREST REGRESSOR")
    print("="*80)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                  random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    print(f"\nTraining Performance:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAE = {mae:.2f}")
    print(f"\nCross-Validation R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Features by Importance:")
    print(feature_importance.head(10).to_string(index=False))
    
    return {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'feature_importance': feature_importance,
        'y_pred': y_pred
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_feature_importance_comparison(results_dict, output_path):
    """Compare feature importance across models"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Linear Regression
    ax = axes[0]
    lr_imp = results_dict['linear']['feature_importance'].head(15)
    ax.barh(range(len(lr_imp)), lr_imp['abs_coefficient'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(lr_imp)))
    ax.set_yticklabels(lr_imp['feature'], fontsize=9)
    ax.set_xlabel('Absolute Coefficient', fontsize=11)
    ax.set_title('Linear Regression\nTop 15 Features', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # LASSO
    ax = axes[1]
    lasso_imp = results_dict['lasso']['feature_importance'].head(15)
    if len(lasso_imp) > 0:
        ax.barh(range(len(lasso_imp)), lasso_imp['abs_coefficient'], color='coral', alpha=0.7)
        ax.set_yticks(range(len(lasso_imp)))
        ax.set_yticklabels(lasso_imp['feature'], fontsize=9)
        ax.set_xlabel('Absolute Coefficient', fontsize=11)
        ax.set_title(f'LASSO Regression\nSelected {results_dict["lasso"]["n_features_selected"]} Features', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    # Random Forest
    ax = axes[2]
    rf_imp = results_dict['rf']['feature_importance'].head(15)
    ax.barh(range(len(rf_imp)), rf_imp['importance'], color='green', alpha=0.7)
    ax.set_yticks(range(len(rf_imp)))
    ax.set_yticklabels(rf_imp['feature'], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Random Forest\nTop 15 Features', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFeature importance comparison saved to: {output_path}")


def plot_predictions_vs_actual(results_dict, y_true, target_name, output_path):
    """Plot predicted vs actual values for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = [
        ('linear', 'Linear Regression', 'steelblue'),
        ('lasso', 'LASSO', 'coral'),
        ('rf', 'Random Forest', 'green')
    ]
    
    for idx, (model_key, model_name, color) in enumerate(models):
        ax = axes[idx]
        
        y_pred = results_dict[model_key]['y_pred']
        r2 = results_dict[model_key]['r2']
        cv_r2 = results_dict[model_key]['cv_r2_mean']
        cv_std = results_dict[model_key]['cv_r2_std']
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        ax.set_xlabel(f'Actual {target_name}', fontsize=11)
        ax.set_ylabel(f'Predicted {target_name}', fontsize=11)
        ax.set_title(f'{model_name}\nR²={r2:.3f}, CV R²={cv_r2:.3f}±{cv_std:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Predictions plot saved to: {output_path}")


def plot_model_comparison(baseline_results, decline_results, output_path):
    """Compare model performance for baseline FVC and decline rate"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = ['linear', 'lasso', 'rf']
    model_names = ['Linear\nRegression', 'LASSO', 'Random\nForest']
    
    # Baseline FVC - R²
    ax = axes[0, 0]
    r2_baseline = [baseline_results[m]['r2'] for m in models]
    cv_r2_baseline = [baseline_results[m]['cv_r2_mean'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, r2_baseline, width, label='Training R²', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, cv_r2_baseline, width, label='CV R²', alpha=0.8, color='coral')
    
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('Baseline FVC Prediction - R² Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Baseline FVC - RMSE
    ax = axes[0, 1]
    rmse_baseline = [baseline_results[m]['rmse'] for m in models]
    ax.bar(model_names, rmse_baseline, alpha=0.7, color='lightcoral')
    ax.set_ylabel('RMSE (ml)', fontsize=11)
    ax.set_title('Baseline FVC Prediction - RMSE Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Decline Rate - R²
    ax = axes[1, 0]
    r2_decline = [decline_results[m]['r2'] for m in models]
    cv_r2_decline = [decline_results[m]['cv_r2_mean'] for m in models]
    
    ax.bar(x - width/2, r2_decline, width, label='Training R²', alpha=0.8, color='green')
    ax.bar(x + width/2, cv_r2_decline, width, label='CV R²', alpha=0.8, color='orange')
    
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('FVC Decline Rate Prediction - R² Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    # Decline Rate - RMSE
    ax = axes[1, 1]
    rmse_decline = [decline_results[m]['rmse'] for m in models]
    ax.bar(model_names, rmse_decline, alpha=0.7, color='lightgreen')
    ax.set_ylabel('RMSE (ml/week)', fontsize=11)
    ax.set_title('FVC Decline Rate Prediction - RMSE Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plot saved to: {output_path}")


def create_top_features_summary(baseline_results, decline_results, output_path):
    """Create summary of top features across both tasks"""
    
    # Get top features from LASSO (most important for selection)
    baseline_top = set(baseline_results['lasso']['feature_importance'].head(10)['feature'].tolist())
    decline_top = set(decline_results['lasso']['feature_importance'].head(10)['feature'].tolist())
    
    # Features important for both tasks
    common_features = baseline_top.intersection(decline_top)
    
    # Features specific to each task
    baseline_specific = baseline_top - decline_top
    decline_specific = decline_top - baseline_top
    
    # Create summary text
    summary = []
    summary.append("="*80)
    summary.append("TOP FEATURES SUMMARY (LASSO Selection)")
    summary.append("="*80)
    
    summary.append(f"\nIMPORTANT FOR BOTH TASKS ({len(common_features)} features):")
    for feat in sorted(common_features):
        summary.append(f"  - {feat}")
    
    summary.append(f"\nSPECIFIC TO BASELINE FVC ({len(baseline_specific)} features):")
    for feat in sorted(baseline_specific):
        summary.append(f"  - {feat}")
    
    summary.append(f"\nSPECIFIC TO DECLINE RATE ({len(decline_specific)} features):")
    for feat in sorted(decline_specific):
        summary.append(f"  - {feat}")
    
    summary_text = "\n".join(summary)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(summary_text)
    
    print("\n" + summary_text)
    print(f"\nTop features summary saved to: {output_path}")
    
    return {
        'common': list(common_features),
        'baseline_specific': list(baseline_specific),
        'decline_specific': list(decline_specific)
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("="*80)
    print("MULTIVARIATE FVC ANALYSIS")
    print("Multiple Regression + Feature Engineering")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Create engineered features
    df = create_engineered_features(df)
    
    # ======= TASK 1: BASELINE FVC PREDICTION =======
    print("\n" + "="*80)
    print("TASK 1: PREDICTING BASELINE FVC (Week 0)")
    print("="*80)
    
    baseline_df = create_baseline_dataset(df)
    X_baseline, y_baseline, baseline_features = prepare_features_target(
        baseline_df, 'FVC', 
        exclude_cols=['case', 'patient', 'week', 'Percent', 'Sex', 'SmokingStatus']
    )
    
    print(f"\nFeatures: {len(baseline_features)}")
    print(f"Samples: {len(X_baseline)}")
    
    # Train models
    baseline_results = {}
    baseline_results['linear'] = train_multiple_regression(X_baseline, y_baseline, baseline_features)
    baseline_results['lasso'] = train_lasso_regression(X_baseline, y_baseline, baseline_features)
    baseline_results['rf'] = train_random_forest(X_baseline, y_baseline, baseline_features)
    
    # Save results
    plot_feature_importance_comparison(
        baseline_results, 
        OUTPUT_DIR / "baseline_feature_importance.png"
    )
    plot_predictions_vs_actual(
        baseline_results, y_baseline, 'FVC (ml)',
        OUTPUT_DIR / "baseline_predictions.png"
    )
    
    # ======= TASK 2: DECLINE RATE PREDICTION =======
    print("\n" + "="*80)
    print("TASK 2: PREDICTING FVC DECLINE RATE")
    print("="*80)
    
    decline_df = calculate_decline_rates(df)
    X_decline, y_decline, decline_features = prepare_features_target(
        decline_df, 'decline_rate',
        exclude_cols=['patient', 'baseline_FVC', 'n_measurements', 'follow_up_weeks', 
                     'Sex', 'SmokingStatus']
    )
    
    print(f"\nFeatures: {len(decline_features)}")
    print(f"Samples: {len(X_decline)}")
    
    # Train models
    decline_results = {}
    decline_results['linear'] = train_multiple_regression(X_decline, y_decline, decline_features)
    decline_results['lasso'] = train_lasso_regression(X_decline, y_decline, decline_features)
    decline_results['rf'] = train_random_forest(X_decline, y_decline, decline_features)
    
    # Save results
    plot_feature_importance_comparison(
        decline_results,
        OUTPUT_DIR / "decline_feature_importance.png"
    )
    plot_predictions_vs_actual(
        decline_results, y_decline, 'Decline Rate (ml/week)',
        OUTPUT_DIR / "decline_predictions.png"
    )
    
    # ======= COMPARISON AND SUMMARY =======
    plot_model_comparison(
        baseline_results, decline_results,
        OUTPUT_DIR / "model_performance_comparison.png"
    )
    
    top_features = create_top_features_summary(
        baseline_results, decline_results,
        OUTPUT_DIR / "top_features_summary.txt"
    )
    
    # Save detailed results
    results_summary = {
        'baseline_fvc': {
            'linear_r2': baseline_results['linear']['r2'],
            'linear_cv_r2': baseline_results['linear']['cv_r2_mean'],
            'lasso_r2': baseline_results['lasso']['r2'],
            'lasso_cv_r2': baseline_results['lasso']['cv_r2_mean'],
            'lasso_features_selected': baseline_results['lasso']['n_features_selected'],
            'rf_r2': baseline_results['rf']['r2'],
            'rf_cv_r2': baseline_results['rf']['cv_r2_mean'],
        },
        'decline_rate': {
            'linear_r2': decline_results['linear']['r2'],
            'linear_cv_r2': decline_results['linear']['cv_r2_mean'],
            'lasso_r2': decline_results['lasso']['r2'],
            'lasso_cv_r2': decline_results['lasso']['cv_r2_mean'],
            'lasso_features_selected': decline_results['lasso']['n_features_selected'],
            'rf_r2': decline_results['rf']['r2'],
            'rf_cv_r2': decline_results['rf']['cv_r2_mean'],
        }
    }
    
    # Save to JSON
    with open(OUTPUT_DIR / "results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save top features from LASSO
    baseline_results['lasso']['feature_importance'].to_csv(
        OUTPUT_DIR / "baseline_lasso_features.csv", index=False
    )
    decline_results['lasso']['feature_importance'].to_csv(
        OUTPUT_DIR / "decline_lasso_features.csv", index=False
    )
    
    # ======= FINAL SUMMARY =======
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\n1. BASELINE FVC PREDICTION:")
    print(f"   Best CV R²: {max(baseline_results['linear']['cv_r2_mean'], baseline_results['lasso']['cv_r2_mean'], baseline_results['rf']['cv_r2_mean']):.4f}")
    best_baseline = max(baseline_results.items(), key=lambda x: x[1]['cv_r2_mean'])
    print(f"   Best model: {best_baseline[0].upper()}")
    
    print("\n2. DECLINE RATE PREDICTION:")
    print(f"   Best CV R²: {max(decline_results['linear']['cv_r2_mean'], decline_results['lasso']['cv_r2_mean'], decline_results['rf']['cv_r2_mean']):.4f}")
    best_decline = max(decline_results.items(), key=lambda x: x[1]['cv_r2_mean'])
    print(f"   Best model: {best_decline[0].upper()}")
    
    print("\n3. KEY ENGINEERED FEATURES:")
    engineered_features = [
        'volume_per_branch', 'volume_per_generation', 'diameter_volume_product',
        'branch_density', 'pc_volume_product', 'volume_tapering_product',
        'airway_health_index'
    ]
    print("   Created features:")
    for feat in engineered_features:
        print(f"      - {feat}")
    
    print("\n4. MOST IMPORTANT FEATURES (across both tasks):")
    for feat in top_features['common'][:5]:
        print(f"      - {feat}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - baseline_feature_importance.png")
    print("  - baseline_predictions.png")
    print("  - baseline_lasso_features.csv")
    print("  - decline_feature_importance.png")
    print("  - decline_predictions.png")
    print("  - decline_lasso_features.csv")
    print("  - model_performance_comparison.png")
    print("  - top_features_summary.txt")
    print("  - results_summary.json")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
