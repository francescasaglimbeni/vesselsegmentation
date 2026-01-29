"""
Recalculate fibrosis scores from existing advanced_metrics.json files.
Used when weights are updated without re-running the entire pipeline.

Author: Francesca Saglimbeni
Date: January 29, 2026
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


class FibrosisScoreRecalculator:
    """
    Recalculates fibrosis scores from existing advanced_metrics.json
    without needing the full AirwayGraphAnalyzer.
    """
    
    def __init__(self):
        # VALIDATED WEIGHTS - OPZIONE 1 (Airway-Only Improved)
        # Updated: Jan 29, 2026 - Re-optimized based on OSIC FVC% correlation study
        self.weights = {
            'peripheral_density': 0.35,
            'peripheral_volume': 0.25,
            'pc_ratio': 0.20,
            'tortuosity': 0.15,
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
            interpretation = "Unusually high (possible compensatory changes)"
        
        return {
            'raw_score': score,
            'value': pc_ratio,
            'interpretation': interpretation
        }
    
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
        
        return {
            'raw_score': score,
            'value': tortuosity,
            'interpretation': interpretation
        }
    
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
            interpretation = "Severe asymmetry - unilateral predominance"
        
        return {
            'raw_score': score,
            'value': symmetry_index,
            'interpretation': interpretation
        }
    
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
        
        return {
            'raw_score': score,
            'value': density,
            'interpretation': interpretation
        }
    
    def compute_peripheral_volume_score(self, periph_diam, periph_vol, periph_pct):
        """Score based on peripheral volume metrics"""
        scores = []
        details = []
        
        # Score 1: Peripheral diameter
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
        
        # Score 2: Peripheral branch volume
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
        
        # Score 3: Peripheral volume percent
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
            'value': {
                'diameter': periph_diam,
                'branch_volume': periph_vol,
                'volume_percent': periph_pct
            },
            'interpretation': interpretation
        }
    
    def recalculate_score(self, advanced_metrics):
        """
        Recalculate fibrosis score from advanced_metrics dict.
        
        Args:
            advanced_metrics: Dict containing advanced airway metrics
            
        Returns:
            Dict with fibrosis assessment
        """
        # Extract metrics
        pc_ratio = advanced_metrics.get('peripheral_to_central_ratio', np.nan)
        tortuosity = advanced_metrics.get('mean_tortuosity', np.nan)
        symmetry = advanced_metrics.get('symmetry_index', np.nan)
        periph_density = advanced_metrics.get('peripheral_branch_density', np.nan)
        periph_diam = advanced_metrics.get('mean_peripheral_diameter_mm', np.nan)
        periph_vol = advanced_metrics.get('mean_peripheral_branch_volume_mm3', np.nan)
        periph_pct = advanced_metrics.get('peripheral_volume_percent', np.nan)
        
        # Compute component scores
        component_scores = {
            'peripheral_density': self.compute_peripheral_density_score(periph_density),
            'peripheral_volume': self.compute_peripheral_volume_score(periph_diam, periph_vol, periph_pct),
            'pc_ratio': self.compute_pc_ratio_score(pc_ratio),
            'tortuosity': self.compute_tortuosity_score(tortuosity),
            'symmetry': self.compute_symmetry_score(symmetry),
        }
        
        # Weighted sum
        total_score = 0.0
        available_weight = 0.0
        
        for component, weight in self.weights.items():
            if weight == 0.0:
                continue
            
            comp_data = component_scores.get(component)
            if comp_data is None:
                continue
            
            raw_score = comp_data.get('raw_score', 5.0)
            value = comp_data.get('value', np.nan)
            
            # Only include if we have valid data
            is_valid = False
            if isinstance(value, dict):
                is_valid = any(not np.isnan(v) for v in value.values() if isinstance(v, (int, float)))
            else:
                is_valid = not np.isnan(value)
            
            if is_valid:
                total_score += raw_score * weight
                available_weight += weight
        
        # Normalize to 0-100 scale
        if available_weight > 0:
            fibrosis_score = (total_score / available_weight) * 10.0
        else:
            fibrosis_score = 50.0
        
        # Classify severity
        if fibrosis_score < 20:
            severity_stage = "Minimal/No fibrosis"
        elif fibrosis_score < 35:
            severity_stage = "Mild fibrosis"
        elif fibrosis_score < 50:
            severity_stage = "Moderate fibrosis"
        elif fibrosis_score < 70:
            severity_stage = "Moderate-severe fibrosis (UIP pattern)"
        else:
            severity_stage = "Severe/Advanced fibrosis"
        
        confidence = available_weight
        
        # Build result
        result = {
            'overall': {
                'fibrosis_score': float(fibrosis_score),
                'stage': severity_stage,
                'confidence': float(confidence)
            },
            'components': {}
        }
        
        for component, comp_data in component_scores.items():
            weight = self.weights.get(component, 0.0)
            raw_score = comp_data.get('raw_score', 5.0)
            
            result['components'][component] = {
                'raw_score': float(raw_score),
                'weighted_score': float(raw_score * weight * 10.0),
                'value': comp_data.get('value') if not isinstance(comp_data.get('value'), dict) else comp_data.get('value'),
                'interpretation': comp_data.get('interpretation', 'N/A'),
                'weight': float(weight)
            }
        
        return result


def recalculate_all_fibrosis_scores(results_root):
    """
    Recalculate fibrosis scores for all cases in results directory.
    
    Args:
        results_root: Path to results directory (e.g., results_OSIC_newMetrcis)
    """
    results_root = Path(results_root)
    
    if not results_root.exists():
        print(f"❌ Results directory not found: {results_root}")
        return
    
    print("="*80)
    print("RECALCULATING FIBROSIS SCORES")
    print("="*80)
    print(f"Results directory: {results_root}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNew weights (Opzione 1 - Airway-Only Improved):")
    
    recalculator = FibrosisScoreRecalculator()
    for component, weight in recalculator.weights.items():
        print(f"  - {component}: {weight:.0%}")
    
    print("\n" + "="*80)
    print("PROCESSING CASES")
    print("="*80 + "\n")
    
    # Find all case directories
    case_dirs = [d for d in results_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for case_dir in sorted(case_dirs):
        case_name = case_dir.name
        
        # Check if advanced_metrics.json exists
        metrics_path = case_dir / "step4_analysis" / "advanced_metrics.json"
        if not metrics_path.exists():
            skip_count += 1
            continue
        
        try:
            # Load advanced metrics
            with open(metrics_path, 'r') as f:
                advanced_metrics = json.load(f)
            
            # Recalculate score
            fibrosis_assessment = recalculator.recalculate_score(advanced_metrics)
            
            # Save to step6_fibrosis_assessment
            output_dir = case_dir / "step6_fibrosis_assessment"
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / "fibrosis_assessment.json"
            with open(output_path, 'w') as f:
                json.dump(fibrosis_assessment, f, indent=2)
            
            score = fibrosis_assessment['overall']['fibrosis_score']
            stage = fibrosis_assessment['overall']['stage']
            
            print(f"✓ {case_name}: Score={score:.1f}, Stage={stage}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {case_name}: ERROR - {e}")
            error_count += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total cases found: {len(case_dirs)}")
    print(f"Successfully recalculated: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped (no metrics): {skip_count}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Recalculate for OSIC dataset
    results_dir = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC_newMetrcis")
    recalculate_all_fibrosis_scores(results_dir)
