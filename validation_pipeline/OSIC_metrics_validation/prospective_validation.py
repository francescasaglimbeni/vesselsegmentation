"""
PROSPECTIVE VALIDATION: Predict FVC Severity using Airway + Parenchymal Metrics

Objective: Test if baseline CT metrics can predict severe disease (FVC% < 50%)

Methodology:
- Use baseline (week 0) metrics to predict FVC severity
- Classify patients: Severe (FVC% < 50%) vs Non-severe (FVC% >= 50%)
- Train predictive model with cross-validation
- Calculate AUC (Area Under ROC Curve)
- Clinical usability threshold: AUC >= 0.70

Note: Sample size = 42 patients (< 50 recommended)
      → Using k-fold cross-validation for robust evaluation

Author: Francesca Saglimbeni
Date: January 2026
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

INTEGRATED_DATASET = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\results_analysis\integrated_dataset.csv")
OUTPUT_DIR = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\OSIC_metrics_validation\prospective_validation_results")

# Severity threshold
SEVERE_THRESHOLD_PERCENT = 50.0  # FVC% < 50% = severe disease

# Cross-validation
N_FOLDS = 5  # 5-fold CV (given small sample size)
RANDOM_STATE = 42

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_prepare_data():
    """Load integrated dataset and prepare for classification"""
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    # Load dataset
    df = pd.read_csv(INTEGRATED_DATASET)
    print(f"\nLoaded {len(df)} total measurements")
    print(f"Unique patients: {df['patient'].nunique()}")
    
    # Keep only baseline measurements (first available week for each patient)
    # Note: Not all patients have week 0, so we use the earliest measurement
    df_baseline = df.loc[df.groupby('patient')['week'].idxmin()].copy()
    print(f"\nBaseline (earliest) measurements: {len(df_baseline)}")
    print(f"  Week range: {df_baseline['week'].min()} - {df_baseline['week'].max()}")
    print(f"  Median week: {df_baseline['week'].median()}")
    
    # Create severity label
    df_baseline['severe'] = (df_baseline['Percent'] < SEVERE_THRESHOLD_PERCENT).astype(int)
    
    # Count severe vs non-severe
    severe_count = df_baseline['severe'].sum()
    nonsevere_count = len(df_baseline) - severe_count
    
    print(f"\nSeverity Classification (FVC% < {SEVERE_THRESHOLD_PERCENT}%):")
    print(f"  Severe (FVC% < {SEVERE_THRESHOLD_PERCENT}%): {severe_count} patients ({100*severe_count/len(df_baseline):.1f}%)")
    print(f"  Non-severe (FVC% >= {SEVERE_THRESHOLD_PERCENT}%): {nonsevere_count} patients ({100*nonsevere_count/len(df_baseline):.1f}%)")
    
    if severe_count < 5 or nonsevere_count < 5:
        print("\n⚠ WARNING: Very few patients in one class - results may be unreliable!")
    
    # Select features - ONLY SIGNIFICANT ONES (p < 0.05 from correlation analysis)
    # Based on correlation results:
    # - Airway: volume_ml (r=0.245, p<0.001), mean_tortuosity (r=-0.267, p<0.001)
    # - Parenchymal: percent_ground_glass_opacity (r=-0.213, p=0.0002), 
    #                basal_predominance_index (r=0.200, p=0.0004),
    #                percent_fibrotic_patterns (r=-0.149, p=0.009)
    
    feature_cols = [
        # AIRWAY METRICS (2 significant)
        'volume_ml',              # r=+0.245, p<0.001 ***
        'mean_tortuosity',        # r=-0.267, p<0.001 ***
        
        # PARENCHYMAL METRICS (3 significant)
        'percent_ground_glass_opacity',   # r=-0.213, p=0.0002 ***
        'basal_predominance_index',       # r=+0.200, p=0.0004 ***
        'percent_fibrotic_patterns',      # r=-0.149, p=0.009 **
    ]
    
    print("\nUsing ONLY statistically significant metrics (p < 0.05):")
    print("  Airway metrics: 2")
    print("  Parenchymal metrics: 3")
    
    # Remove features with too many NaN
    available_features = []
    for col in feature_cols:
        if col in df_baseline.columns:
            nan_count = df_baseline[col].isna().sum()
            nan_pct = 100 * nan_count / len(df_baseline)
            if nan_pct < 50:  # Keep if < 50% missing
                available_features.append(col)
                print(f"  ✓ {col}: {nan_pct:.1f}% missing")
            else:
                print(f"  ✗ {col}: {nan_pct:.1f}% missing (excluded)")
    
    print(f"\nUsing {len(available_features)} features for prediction")
    
    # Remove rows with any NaN in selected features
    df_clean = df_baseline[['patient', 'Percent', 'severe'] + available_features].dropna()
    
    print(f"\nAfter removing NaN: {len(df_clean)} patients")
    print(f"  Severe: {df_clean['severe'].sum()}")
    print(f"  Non-severe: {len(df_clean) - df_clean['severe'].sum()}")
    
    # Check class balance
    severe_count = df_clean['severe'].sum()
    if severe_count < 5 or (len(df_clean) - severe_count) < 5:
        print("\n⚠ WARNING: Extremely imbalanced classes - classification may not be meaningful!")
        print(f"  Consider using a different severity threshold or endpoint.")
    
    return df_clean, available_features


# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate_models(X, y, feature_names):
    """Train multiple models with cross-validation and evaluate performance"""
    print("\n" + "="*80)
    print("MODEL TRAINING AND EVALUATION (Cross-Validation)")
    print("="*80)
    
    # Check class balance for CV strategy
    class_counts = np.bincount(y)
    min_class_count = class_counts.min()
    
    if min_class_count < N_FOLDS:
        print(f"\n⚠ WARNING: Only {min_class_count} samples in minority class")
        print(f"  Using Leave-One-Out CV instead of {N_FOLDS}-fold CV")
        cv = LeaveOneOut()
        cv_name = "LOOCV"
    else:
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_name = f"{N_FOLDS}-fold CV"
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=3, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=2)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name} ({cv_name})")
        print("-"*60)
        
        # Get predicted probabilities with cross-validation
        try:
            y_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Calculate AUC
        try:
            auc = roc_auc_score(y, y_proba)
            fpr, tpr, thresholds = roc_curve(y, y_proba)
            
            # Average precision (for imbalanced datasets)
            avg_precision = average_precision_score(y, y_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            # Sensitivity and specificity
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"  AUC: {auc:.3f}")
            print(f"  Average Precision: {avg_precision:.3f}")
            print(f"  Sensitivity (Recall): {sensitivity:.3f}")
            print(f"  Specificity: {specificity:.3f}")
            
            # Clinical usability check
            if auc >= 0.70:
                print(f"  ✓ CLINICALLY USABLE (AUC >= 0.70)")
            else:
                print(f"  ✗ NOT CLINICALLY USABLE (AUC < 0.70)")
            
            # Store results
            results[name] = {
                'model': model,
                'auc': auc,
                'avg_precision': avg_precision,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'y_proba': y_proba,
                'y_pred': y_pred,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            print(f"  ERROR: Could not calculate AUC - {e}")
            results[name] = None
    
    # Feature importance (from Random Forest)
    if 'Random Forest' in results and results['Random Forest'] is not None:
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE (Random Forest)")
        print("="*80)
        
        # Train full model for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=3)
        rf.fit(X_scaled, y)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 10 Most Important Features:")
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
        
        results['feature_importances'] = {
            'names': [feature_names[i] for i in indices],
            'values': importances[indices]
        }
    
    return results, scaler


# ============================================================
# VISUALIZATION
# ============================================================

def plot_roc_curves(results, output_path):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for name, res in results.items():
        if res is not None and 'fpr' in res:
            plt.plot(res['fpr'], res['tpr'], linewidth=2, 
                    label=f"{name} (AUC = {res['auc']:.3f})")
    
    # Diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.50)')
    
    # Clinical threshold line
    plt.axhline(y=0.70, color='red', linestyle=':', linewidth=1, alpha=0.5)
    plt.text(0.7, 0.72, 'AUC = 0.70 (Clinical Threshold)', color='red', fontsize=9)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Predicting Severe Disease (FVC% < 50%)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ ROC curves saved to: {output_path}")


def plot_feature_importance(results, output_path):
    """Plot feature importance"""
    if 'feature_importances' not in results:
        return
    
    names = results['feature_importances']['names'][:15]  # Top 15
    values = results['feature_importances']['values'][:15]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(names)), values, color='steelblue', alpha=0.7)
    plt.yticks(range(len(names)), names)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 15 Most Important Features\n(Random Forest)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature importance plot saved to: {output_path}")


def plot_confusion_matrices(results, output_path):
    """Plot confusion matrices for all models"""
    n_models = sum(1 for res in results.values() if res is not None and 'confusion_matrix' in res)
    
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    idx = 0
    for name, res in results.items():
        if res is not None and 'confusion_matrix' in res:
            cm = res['confusion_matrix']
            
            # Plot
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.set_title(f'{name}\nAUC = {res["auc"]:.3f}', fontsize=12, fontweight='bold')
            ax.set_xticklabels(['Non-severe', 'Severe'])
            ax.set_yticklabels(['Non-severe', 'Severe'])
            
            idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrices saved to: {output_path}")


def plot_metric_distributions(df, output_path):
    """Plot distributions of top metrics by severity"""
    top_metrics = [
        ('mean_tortuosity', 'Mean Tortuosity'),
        ('percent_ground_glass_opacity', '% Ground Glass Opacity'),
        ('volume_ml', 'Airway Volume (ml)'),
        ('basal_predominance_index', 'Basal Predominance Index')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(top_metrics):
        if metric not in df.columns:
            continue
        
        ax = axes[i]
        
        # Box plot
        severe_data = df[df['severe'] == 1][metric].dropna()
        nonsevere_data = df[df['severe'] == 0][metric].dropna()
        
        data_to_plot = [nonsevere_data, severe_data]
        bp = ax.boxplot(data_to_plot, labels=['Non-severe', 'Severe'], patch_artist=True)
        
        # Colors
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('salmon')
        
        # Statistical test
        if len(severe_data) > 0 and len(nonsevere_data) > 0:
            stat, p_value = stats.mannwhitneyu(nonsevere_data, severe_data, alternative='two-sided')
            ax.text(0.5, 0.95, f'p = {p_value:.4f}', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Metric Distributions by Severity (FVC% < 50%)', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metric distributions saved to: {output_path}")


# ============================================================
# REPORTING
# ============================================================

def generate_report(df, results, feature_names, output_path):
    """Generate comprehensive validation report"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PROSPECTIVE VALIDATION REPORT\n")
        f.write("Predicting Severe Disease (FVC% < 50%) from Baseline CT Metrics\n")
        f.write("="*80 + "\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Total patients: {len(df)}\n")
        f.write(f"Severe (FVC% < {SEVERE_THRESHOLD_PERCENT}%): {df['severe'].sum()} ({100*df['severe'].sum()/len(df):.1f}%)\n")
        f.write(f"Non-severe (FVC% >= {SEVERE_THRESHOLD_PERCENT}%): {len(df) - df['severe'].sum()} ({100*(len(df) - df['severe'].sum())/len(df):.1f}%)\n")
        f.write(f"Number of features: {len(feature_names)}\n\n")
        
        # Sample size warning
        f.write("SAMPLE SIZE ASSESSMENT\n")
        f.write("-"*80 + "\n")
        if len(df) < 50:
            f.write(f"⚠ WARNING: Sample size ({len(df)}) is below recommended minimum (50)\n")
            f.write(f"  → Results should be considered PRELIMINARY\n")
            f.write(f"  → Cross-validation used for robust evaluation\n")
        else:
            f.write(f"✓ Sample size ({len(df)}) meets recommended minimum (50)\n")
        f.write("\n")
        
        # Model results
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<25} {'AUC':<10} {'Avg Prec':<10} {'Sens':<10} {'Spec':<10} {'Clinical Use':<15}\n")
        f.write("-"*80 + "\n")
        
        for name, res in results.items():
            if res is not None and 'auc' in res:
                clinical = "✓ USABLE" if res['auc'] >= 0.70 else "✗ NOT USABLE"
                f.write(f"{name:<25} {res['auc']:<10.3f} {res['avg_precision']:<10.3f} "
                       f"{res['sensitivity']:<10.3f} {res['specificity']:<10.3f} {clinical:<15}\n")
        
        f.write("\n")
        
        # Clinical interpretation
        f.write("CLINICAL INTERPRETATION\n")
        f.write("-"*80 + "\n")
        
        best_model = max(results.items(), key=lambda x: x[1]['auc'] if x[1] and 'auc' in x[1] else 0)
        best_auc = best_model[1]['auc'] if best_model[1] else 0
        
        if best_auc >= 0.70:
            f.write(f"✓ SYSTEM IS CLINICALLY USABLE\n")
            f.write(f"  Best model: {best_model[0]}\n")
            f.write(f"  AUC: {best_auc:.3f} (>= 0.70 threshold)\n")
            f.write(f"  The model shows acceptable discriminative ability for clinical use.\n")
        else:
            f.write(f"✗ SYSTEM NOT CLINICALLY USABLE\n")
            f.write(f"  Best model: {best_model[0]}\n")
            f.write(f"  AUC: {best_auc:.3f} (< 0.70 threshold)\n")
            f.write(f"  The model does not meet clinical usability criteria.\n")
        
        f.write("\n")
        
        # Feature importance
        if 'feature_importances' in results:
            f.write("TOP 10 MOST IMPORTANT FEATURES\n")
            f.write("-"*80 + "\n")
            for i in range(min(10, len(results['feature_importances']['names']))):
                name = results['feature_importances']['names'][i]
                value = results['feature_importances']['values'][i]
                f.write(f"  {i+1}. {name}: {value:.3f}\n")
            f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        if len(df) < 50:
            f.write("  1. Increase sample size to at least 50 patients for more robust results\n")
        if best_auc < 0.70:
            f.write("  2. Consider adding more parenchymal metrics (e.g., radiomics features)\n")
            f.write("  3. Explore alternative severity thresholds or continuous FVC prediction\n")
        if df['severe'].sum() < 10 or (len(df) - df['severe'].sum()) < 10:
            f.write("  4. Collect more severe cases for better class balance\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n✓ Validation report saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*80)
    print("PROSPECTIVE VALIDATION: SEVERITY PREDICTION")
    print("="*80)
    print(f"\nObjective: Predict severe disease (FVC% < {SEVERE_THRESHOLD_PERCENT}%)")
    print(f"Method: {N_FOLDS}-fold cross-validation")
    print(f"Clinical threshold: AUC >= 0.70")
    
    # Load data
    df, feature_names = load_and_prepare_data()
    
    if len(df) < 10:
        print("\n⚠ ERROR: Too few patients for reliable analysis (< 10)")
        print("Cannot proceed with validation.")
        return
    
    # Prepare X and y
    X = df[feature_names].values
    y = df['severe'].values
    
    # Train and evaluate models
    results, scaler = train_and_evaluate_models(X, y, feature_names)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_roc_curves(results, OUTPUT_DIR / "roc_curves.png")
    plot_confusion_matrices(results, OUTPUT_DIR / "confusion_matrices.png")
    plot_feature_importance(results, OUTPUT_DIR / "feature_importance.png")
    plot_metric_distributions(df, OUTPUT_DIR / "metric_distributions.png")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'model': name,
            'auc': res.get('auc', np.nan) if res else np.nan,
            'avg_precision': res.get('avg_precision', np.nan) if res else np.nan,
            'sensitivity': res.get('sensitivity', np.nan) if res else np.nan,
            'specificity': res.get('specificity', np.nan) if res else np.nan,
            'clinically_usable': res.get('auc', 0) >= 0.70 if res else False
        }
        for name, res in results.items()
    ])
    
    results_df.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)
    print(f"✓ Model performance saved to: {OUTPUT_DIR / 'model_performance.csv'}")
    
    # Generate report
    generate_report(df, results, feature_names, OUTPUT_DIR / "validation_report.txt")
    
    # Final summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    best_model = max(results.items(), key=lambda x: x[1]['auc'] if x[1] and 'auc' in x[1] else 0)
    best_auc = best_model[1]['auc'] if best_model[1] else 0
    
    print(f"\nBest Model: {best_model[0]}")
    print(f"Best AUC: {best_auc:.3f}")
    
    if best_auc >= 0.70:
        print(f"\n✓ SYSTEM IS CLINICALLY USABLE (AUC >= 0.70)")
    else:
        print(f"\n✗ SYSTEM NOT CLINICALLY USABLE (AUC < 0.70)")
        print(f"  → Consider: more data, additional features, or alternative approaches")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
