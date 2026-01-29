"""
Recalculate BOTH fibrosis scores (airway_only + combined) from existing data.
Uses advanced_metrics.json and parenchymal_metrics.json.

Author: Francesca Saglimbeni
Date: January 29, 2026
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


class DualFibrosisScoreRecalculator:
    """
    Recalculates both airway_only and combined fibrosis scores.
    """
    
    def __init__(self):
        # AIRWAY_ONLY WEIGHTS (Opzione 1)
        self.weights_airway_only = {
            'peripheral_density': 0.35,
            'peripheral_volume': 0.25,
            'pc_ratio': 0.20,
            'tortuosity': 0.15,
            'symmetry': 0.05,
        }
        
        # COMBINED WEIGHTS (Opzione 2)
        self.weights_combined = {
            'parenchymal_entropy': 0.35,
            'parenchymal_density': 0.25,
            'peripheral_density': 0.15,
            'peripheral_volume': 0.15,
            'tortuosity': 0.05,
            'symmetry': 0.05,
        }
        
        # Reference values
        self.reference_values = {
            'pc_ratio': {'mean': 0.45, 'std': 0.15, 'min': 0.25, 'max': 0.65},
            'tortuosity': {'mean': 1.25, 'std': 0.15, 'min': 1.0, 'max': 1.5},
            'symmetry_index': {'mean': 0.85, 'std': 0.10, 'min': 0.70, 'max': 1.0},
        }
    
    def compute_pc_ratio_score(self, pc_ratio):
        """Score based on Peripheral/Central ratio"""
        if np.isnan(pc_ratio):
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        if pc_ratio < 0.15:
            score = 10.0
            interpretation = "Severe peripheral airway loss"
        elif pc_ratio < 0.25:
            score = 7.0 + (0.25 - pc_ratio) / 0.10 * 3.0
            interpretation = "Moderate to severe peripheral loss"
        elif pc_ratio < 0.40:
            score = 4.0 + (0.40 - pc_ratio) / 0.15 * 3.0
            interpretation = "Mild peripheral airway loss"
        elif pc_ratio <= 0.65:
            ref = self.reference_values['pc_ratio']
            z_score = abs(pc_ratio - ref['mean']) / ref['std']
            score = min(3.0, z_score * 1.5)
            interpretation = "Within normal range"
        else:
            score = 2.0 + min(1.0, (pc_ratio - 0.65) / 0.20)
            interpretation = "Unusually high"
        
        return {'raw_score': score, 'value': pc_ratio, 'interpretation': interpretation}
    
    def compute_tortuosity_score(self, tortuosity):
        """Score based on airway tortuosity"""
        if np.isnan(tortuosity):
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        if tortuosity < 1.0:
            score = 1.0 + (1.0 - tortuosity) * 2.0
            interpretation = "Unusually straight airways"
        elif tortuosity <= 1.5:
            ref = self.reference_values['tortuosity']
            z_score = abs(tortuosity - ref['mean']) / ref['std']
            score = min(3.0, z_score * 1.5)
            interpretation = "Within normal range"
        elif tortuosity < 2.0:
            score = 3.0 + (tortuosity - 1.5) / 0.5 * 3.0
            interpretation = "Mild airway distortion"
        elif tortuosity < 2.5:
            score = 6.0 + (tortuosity - 2.0) / 0.5 * 2.0
            interpretation = "Moderate airway distortion"
        else:
            score = 8.0 + min(2.0, (tortuosity - 2.5) / 0.5 * 2.0)
            interpretation = "Severe airway distortion"
        
        return {'raw_score': score, 'value': tortuosity, 'interpretation': interpretation}
    
    def compute_symmetry_score(self, symmetry_index):
        """Score based on left-right symmetry"""
        if np.isnan(symmetry_index):
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        asymmetry = abs(1.0 - symmetry_index)
        
        if asymmetry < 0.15:
            score = 0.0
            interpretation = "Symmetric - bilateral disease"
        elif asymmetry < 0.30:
            score = 0.0 + (asymmetry - 0.15) / 0.15 * 4.0
            interpretation = "Mildly asymmetric"
        elif asymmetry < 0.50:
            score = 4.0 + (asymmetry - 0.30) / 0.20 * 3.0
            interpretation = "Moderate asymmetry"
        else:
            score = 7.0 + min(3.0, (asymmetry - 0.50) / 0.30 * 3.0)
            interpretation = "Severe asymmetry"
        
        return {'raw_score': score, 'value': symmetry_index, 'interpretation': interpretation}
    
    def compute_peripheral_density_score(self, density):
        """Score based on peripheral branch density"""
        if np.isnan(density):
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        if density < 0.03:
            score = 10.0
            interpretation = "Severe peripheral airway loss"
        elif density < 0.05:
            score = 7.0 + (0.05 - density) / 0.02 * 3.0
            interpretation = "Moderate to severe peripheral loss"
        elif density < 0.08:
            score = 4.0 + (0.08 - density) / 0.03 * 3.0
            interpretation = "Mild peripheral airway loss"
        elif density <= 0.15:
            score = max(0.0, 4.0 - (density - 0.08) / 0.07 * 4.0)
            interpretation = "Within normal range"
        else:
            score = 0.0
            interpretation = "Excellent peripheral preservation"
        
        return {'raw_score': score, 'value': density, 'interpretation': interpretation}
    
    def compute_peripheral_volume_score(self, periph_diam, periph_vol, periph_pct):
        """Score based on peripheral volume metrics"""
        scores = []
        details = []
        
        if not np.isnan(periph_diam):
            if periph_diam < 1.5:
                diam_score = 10.0
                details.append("Severe diameter reduction")
            elif periph_diam < 2.0:
                diam_score = 6.0 + (2.0 - periph_diam) / 0.5 * 4.0
                details.append("Moderate diameter reduction")
            elif periph_diam < 2.5:
                diam_score = 3.0 + (2.5 - periph_diam) / 0.5 * 3.0
                details.append("Mild diameter reduction")
            else:
                diam_score = max(0.0, 3.0 - (periph_diam - 2.5) / 1.0 * 3.0)
                details.append("Normal diameter")
            scores.append(diam_score)
        
        if not np.isnan(periph_vol):
            if periph_vol < 10:
                vol_score = 10.0
                details.append("Severe volume loss")
            elif periph_vol < 20:
                vol_score = 6.0 + (20 - periph_vol) / 10 * 4.0
                details.append("Moderate volume loss")
            elif periph_vol < 30:
                vol_score = 3.0 + (30 - periph_vol) / 10 * 3.0
                details.append("Mild volume reduction")
            else:
                vol_score = max(0.0, 3.0 - (periph_vol - 30) / 30 * 3.0)
                details.append("Good volume")
            scores.append(vol_score)
        
        if not np.isnan(periph_pct):
            if periph_pct < 10:
                pct_score = 10.0
            elif periph_pct < 15:
                pct_score = 6.0 + (15 - periph_pct) / 5 * 4.0
            elif periph_pct < 20:
                pct_score = 3.0 + (20 - periph_pct) / 5 * 3.0
            else:
                pct_score = max(0.0, 3.0 - (periph_pct - 20) / 10 * 3.0)
            scores.append(pct_score)
        
        if len(scores) == 0:
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        final_score = np.mean(scores)
        interpretation = "; ".join(details) if details else "Unknown"
        
        return {
            'raw_score': final_score,
            'value': {'diameter': periph_diam, 'branch_volume': periph_vol, 'volume_percent': periph_pct},
            'interpretation': interpretation
        }
    
    def compute_parenchymal_entropy_score(self, entropy):
        """Score based on histogram entropy"""
        if np.isnan(entropy):
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Not available'}
        
        if entropy < 4.0:
            score = 0.0
            interpretation = "Very homogeneous"
        elif entropy < 5.0:
            score = max(0.0, (5.0 - entropy) / 1.0 * 2.0)
            interpretation = "Normal homogeneity"
        elif entropy < 5.5:
            score = 2.0 + (entropy - 5.0) / 0.5 * 2.0
            interpretation = "Mild texture heterogeneity"
        elif entropy < 6.0:
            score = 4.0 + (entropy - 5.5) / 0.5 * 2.0
            interpretation = "Moderate texture heterogeneity"
        elif entropy < 6.5:
            score = 6.0 + (entropy - 6.0) / 0.5 * 2.0
            interpretation = "Marked texture heterogeneity"
        else:
            score = 8.0 + min(2.0, (entropy - 6.5) / 0.5 * 2.0)
            interpretation = "Severe texture heterogeneity"
        
        return {'raw_score': score, 'value': entropy, 'interpretation': interpretation}
    
    def compute_parenchymal_density_score(self, density):
        """Score based on mean lung density (HU)"""
        if np.isnan(density):
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Not available'}
        
        if density < -900:
            score = 0.0
            interpretation = "Very low density (emphysematous)"
        elif density < -850:
            score = max(0.0, (-850 - density) / 50 * 1.0)
            interpretation = "Low density"
        elif density < -750:
            score = 1.0 + (-750 - density) / 100 * 2.0
            interpretation = "Normal density"
        elif density < -700:
            score = 3.0 + (-700 - density) / 50 * 2.0
            interpretation = "Mildly increased density"
        elif density < -650:
            score = 5.0 + (-650 - density) / 50 * 2.0
            interpretation = "Moderately increased density"
        elif density < -600:
            score = 7.0 + (-600 - density) / 50 * 2.0
            interpretation = "Markedly increased density"
        else:
            score = 9.0 + min(1.0, (-600 - density) / 50)
            interpretation = "Severe parenchymal density increase"
        
        return {'raw_score': score, 'value': density, 'interpretation': interpretation}
    
    def recalculate_scores(self, advanced_metrics, parenchymal_metrics):
        """
        Recalculate both airway_only and combined scores.
        
        Returns:
            Dict with both scoring methods
        """
        # Extract airway metrics
        pc_ratio = advanced_metrics.get('peripheral_to_central_ratio', np.nan)
        tortuosity = advanced_metrics.get('mean_tortuosity', np.nan)
        symmetry = advanced_metrics.get('symmetry_index', np.nan)
        periph_density = advanced_metrics.get('peripheral_branch_density', np.nan)
        periph_diam = advanced_metrics.get('mean_peripheral_diameter_mm', np.nan)
        periph_vol = advanced_metrics.get('mean_peripheral_branch_volume_mm3', np.nan)
        periph_pct = advanced_metrics.get('peripheral_volume_percent', np.nan)
        
        # Extract parenchymal metrics
        entropy = parenchymal_metrics.get('histogram_entropy', np.nan) if parenchymal_metrics else np.nan
        lung_density = parenchymal_metrics.get('mean_lung_density_HU', np.nan) if parenchymal_metrics else np.nan
        
        # Compute all component scores
        component_scores = {
            'peripheral_density': self.compute_peripheral_density_score(periph_density),
            'peripheral_volume': self.compute_peripheral_volume_score(periph_diam, periph_vol, periph_pct),
            'pc_ratio': self.compute_pc_ratio_score(pc_ratio),
            'tortuosity': self.compute_tortuosity_score(tortuosity),
            'symmetry': self.compute_symmetry_score(symmetry),
            'parenchymal_entropy': self.compute_parenchymal_entropy_score(entropy),
            'parenchymal_density': self.compute_parenchymal_density_score(lung_density),
        }
        
        # Calculate AIRWAY_ONLY score
        airway_score, airway_stage, airway_conf = self._calculate_score(
            component_scores, self.weights_airway_only
        )
        
        # Calculate COMBINED score
        combined_score, combined_stage, combined_conf = self._calculate_score(
            component_scores, self.weights_combined
        )
        
        # Build result
        result = {
            'overall': {
                'fibrosis_score': float(combined_score if not np.isnan(combined_score) else airway_score),
                'stage': combined_stage if not np.isnan(combined_score) else airway_stage,
                'confidence': float(combined_conf if not np.isnan(combined_score) else airway_conf)
            },
            'scoring_methods': {
                'airway_only': {
                    'fibrosis_score': float(airway_score),
                    'stage': airway_stage,
                    'confidence': float(airway_conf),
                    'description': 'Pure airway morphometry (Opzione 1)'
                },
                'combined': {
                    'fibrosis_score': float(combined_score) if not np.isnan(combined_score) else None,
                    'stage': combined_stage if not np.isnan(combined_score) else None,
                    'confidence': float(combined_conf) if not np.isnan(combined_score) else None,
                    'description': 'Airway + Parenchymal (Opzione 2) - RECOMMENDED'
                }
            },
            'components': {}
        }
        
        # Add component details
        for component, comp_data in component_scores.items():
            weight_airway = self.weights_airway_only.get(component, 0.0)
            weight_combined = self.weights_combined.get(component, 0.0)
            raw_score = comp_data.get('raw_score', 5.0)
            
            result['components'][component] = {
                'raw_score': float(raw_score),
                'weighted_score_airway': float(raw_score * weight_airway * 10.0),
                'weighted_score_combined': float(raw_score * weight_combined * 10.0),
                'value': comp_data.get('value') if not isinstance(comp_data.get('value'), dict) else comp_data.get('value'),
                'interpretation': comp_data.get('interpretation', 'N/A'),
                'weight_airway': float(weight_airway),
                'weight_combined': float(weight_combined)
            }
        
        return result
    
    def _calculate_score(self, component_scores, weights):
        """Helper to calculate score with given weights"""
        total_score = 0.0
        available_weight = 0.0
        
        for component, weight in weights.items():
            if weight == 0.0:
                continue
            
            comp_data = component_scores.get(component)
            if comp_data is None:
                continue
            
            raw_score = comp_data.get('raw_score', 5.0)
            value = comp_data.get('value', np.nan)
            
            # Check if valid
            is_valid = False
            if isinstance(value, dict):
                is_valid = any(not np.isnan(v) for v in value.values() if isinstance(v, (int, float)))
            else:
                is_valid = not np.isnan(value)
            
            if is_valid:
                total_score += raw_score * weight
                available_weight += weight
        
        if available_weight > 0:
            score = (total_score / available_weight) * 10.0
        else:
            score = np.nan
        
        # Classify severity
        if not np.isnan(score):
            if score < 20:
                stage = "Minimal/No fibrosis"
            elif score < 35:
                stage = "Mild fibrosis"
            elif score < 50:
                stage = "Moderate fibrosis"
            elif score < 70:
                stage = "Moderate-severe fibrosis (UIP pattern)"
            else:
                stage = "Severe/Advanced fibrosis"
        else:
            stage = "Unknown"
        
        return score, stage, available_weight


def recalculate_all_scores(results_root):
    """Recalculate both scores for all cases."""
    results_root = Path(results_root)
    
    print("="*80)
    print("RECALCULATING BOTH FIBROSIS SCORES")
    print("="*80)
    print(f"Results directory: {results_root}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nScoring methods:")
    print("  1. AIRWAY_ONLY (Opzione 1): Pure airway morphometry")
    print("  2. COMBINED (Opzione 2): Airway + Parenchymal - RECOMMENDED")
    print("="*80 + "\n")
    
    recalculator = DualFibrosisScoreRecalculator()
    case_dirs = [d for d in results_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    success_count = 0
    error_count = 0
    skip_count = 0
    no_parenchymal = 0
    
    for case_dir in sorted(case_dirs):
        case_name = case_dir.name
        
        # Load advanced metrics
        metrics_path = case_dir / "step4_analysis" / "advanced_metrics.json"
        if not metrics_path.exists():
            skip_count += 1
            continue
        
        try:
            with open(metrics_path, 'r') as f:
                advanced_metrics = json.load(f)
            
            # Load parenchymal metrics (optional)
            parench_path = case_dir / "step5_parenchymal_metrics" / "parenchymal_metrics.json"
            if parench_path.exists():
                with open(parench_path, 'r') as f:
                    parenchymal_metrics = json.load(f)
            else:
                parenchymal_metrics = None
                no_parenchymal += 1
            
            # Recalculate both scores
            assessment = recalculator.recalculate_scores(advanced_metrics, parenchymal_metrics)
            
            # Save
            output_dir = case_dir / "step6_fibrosis_assessment"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / "fibrosis_assessment.json", 'w') as f:
                json.dump(assessment, f, indent=2)
            
            airway_score = assessment['scoring_methods']['airway_only']['fibrosis_score']
            combined_score = assessment['scoring_methods']['combined']['fibrosis_score']
            
            if combined_score is not None:
                print(f"✓ {case_name}: Airway={airway_score:.1f}, Combined={combined_score:.1f}")
            else:
                print(f"✓ {case_name}: Airway={airway_score:.1f}, Combined=N/A (no parenchymal)")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ {case_name}: ERROR - {e}")
            error_count += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total cases: {len(case_dirs)}")
    print(f"Successfully processed: {success_count}")
    print(f"  - With combined score: {success_count - no_parenchymal}")
    print(f"  - Airway-only (no parenchymal): {no_parenchymal}")
    print(f"Errors: {error_count}")
    print(f"Skipped: {skip_count}")
    print("="*80 + "\n")


if __name__ == "__main__":
    results_dir = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC_newMetrcis")
    recalculate_all_scores(results_dir)
