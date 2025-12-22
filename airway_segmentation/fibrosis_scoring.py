import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os


class PulmonaryFibrosisScorer:
    """
    Advanced scoring system for pulmonary fibrosis assessment based on airway morphometry.
    
    Scoring components:
    1. Peripheral/Central ratio (P/C) - Loss of peripheral airways
    2. Tortuosity - Airway distortion
    3. Generation coverage - Missing airway generations
    4. Symmetry index - Asymmetric disease distribution
    5. Volume distribution - Abnormal airway volume patterns
    6. Diameter tapering - Abnormal bronchial tapering
    """
    
    def __init__(self, analyzer, verbose=True):
        """
        Args:
            analyzer: AirwayGraphAnalyzer instance with computed metrics
            verbose: Print detailed information
        """
        self.analyzer = analyzer
        self.verbose = verbose
        
        # Reference values (from healthy population studies)
        self.reference_values = {
            'pc_ratio': {'mean': 0.45, 'std': 0.15, 'min': 0.25, 'max': 0.65},
            'tortuosity': {'mean': 1.25, 'std': 0.15, 'min': 1.0, 'max': 1.5},
            'generation_coverage': {'mean': 0.85, 'std': 0.10, 'min': 0.70, 'max': 1.0},
            'symmetry_index': {'mean': 0.85, 'std': 0.10, 'min': 0.70, 'max': 1.0},
            'tapering_ratio': {'mean': 0.79, 'std': 0.05, 'min': 0.70, 'max': 0.88}
        }
        
        # Scoring weights (total = 1.0)
        self.weights = {
            'pc_ratio': 0.30,           # Most important: peripheral loss is hallmark
            'tortuosity': 0.20,          # Distortion indicates fibrotic changes
            'generation_coverage': 0.25, # Missing generations = severe disease
            'symmetry': 0.10,            # Asymmetry suggests focal disease
            'volume_distribution': 0.10,  # Abnormal distribution patterns
            'tapering': 0.05             # Altered tapering indicates remodeling
        }
        
        # Initialize results
        self.component_scores = {}
        self.fibrosis_score = None
        self.severity_stage = None
        self.confidence = None
        
    def compute_pc_ratio_score(self):
        """
        Score based on Peripheral/Central airway ratio.
        Lower P/C ratio = more peripheral airway loss = higher fibrosis score.
        """
        if self.verbose:
            print("\n[1/6] Computing P/C Ratio Score...")
        
        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")
        
        metrics = self.analyzer.advanced_metrics
        pc_ratio = metrics.get('peripheral_to_central_ratio', np.nan)
        
        if np.isnan(pc_ratio):
            if self.verbose:
                print("  ⚠ P/C ratio not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        ref = self.reference_values['pc_ratio']
        
        # Scoring logic:
        # - P/C < 0.15: Severe peripheral loss (score 10)
        # - P/C 0.15-0.25: Moderate loss (score 7-9)
        # - P/C 0.25-0.40: Mild loss (score 4-6)
        # - P/C 0.40-0.65: Normal range (score 0-3)
        # - P/C > 0.65: Unusually high (score 2-3, possible compensatory)
        
        if pc_ratio < 0.15:
            score = 10.0
            interpretation = "Severe peripheral airway loss"
        elif pc_ratio < 0.25:
            # Linear interpolation 0.15-0.25 -> 7-10
            score = 7.0 + (0.25 - pc_ratio) / 0.10 * 3.0
            interpretation = "Moderate to severe peripheral loss"
        elif pc_ratio < 0.40:
            # Linear 0.25-0.40 -> 4-7
            score = 4.0 + (0.40 - pc_ratio) / 0.15 * 3.0
            interpretation = "Mild peripheral airway loss"
        elif pc_ratio <= 0.65:
            # Normal range: Gaussian scoring
            z_score = abs(pc_ratio - ref['mean']) / ref['std']
            score = min(3.0, z_score * 1.5)
            interpretation = "Within normal range"
        else:
            # Unusually high
            score = 2.0 + min(1.0, (pc_ratio - 0.65) / 0.20)
            interpretation = "Unusually high (possible compensatory changes)"
        
        if self.verbose:
            print(f"  P/C Ratio: {pc_ratio:.3f}")
            print(f"  Reference range: {ref['min']:.2f} - {ref['max']:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': pc_ratio,
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }
    
    def compute_tortuosity_score(self):
        """
        Score based on airway tortuosity.
        Higher tortuosity = more distortion = higher fibrosis score.
        """
        if self.verbose:
            print("\n[2/6] Computing Tortuosity Score...")
        
        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")
        
        metrics = self.analyzer.advanced_metrics
        tortuosity = metrics.get('mean_tortuosity', np.nan)
        
        if np.isnan(tortuosity):
            if self.verbose:
                print("  ⚠ Tortuosity not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        ref = self.reference_values['tortuosity']
        
        # Scoring logic:
        # - Tortuosity < 1.0: Unusually straight (score 1-2)
        # - Tortuosity 1.0-1.5: Normal range (score 0-3)
        # - Tortuosity 1.5-2.0: Mild distortion (score 4-6)
        # - Tortuosity 2.0-2.5: Moderate distortion (score 7-8)
        # - Tortuosity > 2.5: Severe distortion (score 9-10)
        
        if tortuosity < 1.0:
            score = 1.0 + (1.0 - tortuosity) * 2.0
            interpretation = "Unusually straight airways"
        elif tortuosity <= 1.5:
            z_score = abs(tortuosity - ref['mean']) / ref['std']
            score = min(3.0, z_score * 1.5)
            interpretation = "Within normal range"
        elif tortuosity <= 2.0:
            # Linear 1.5-2.0 -> 4-6
            score = 4.0 + (tortuosity - 1.5) / 0.5 * 2.0
            interpretation = "Mild airway distortion"
        elif tortuosity <= 2.5:
            # Linear 2.0-2.5 -> 7-8
            score = 7.0 + (tortuosity - 2.0) / 0.5 * 1.0
            interpretation = "Moderate airway distortion"
        else:
            score = 9.0 + min(1.0, (tortuosity - 2.5) / 0.5)
            interpretation = "Severe airway distortion"
        
        if self.verbose:
            print(f"  Mean Tortuosity: {tortuosity:.3f}")
            print(f"  Reference range: {ref['min']:.2f} - {ref['max']:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': tortuosity,
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }
    
    def compute_generation_coverage_score(self):
        """
        Score based on airway generation coverage.
        Lower coverage = more missing generations = higher fibrosis score.
        """
        if self.verbose:
            print("\n[3/6] Computing Generation Coverage Score...")
        
        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")
        
        metrics = self.analyzer.advanced_metrics
        coverage = metrics.get('generation_coverage', np.nan)
        missing_gens = metrics.get('missing_generations', [])
        
        if np.isnan(coverage):
            if self.verbose:
                print("  ⚠ Generation coverage not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        ref = self.reference_values['generation_coverage']
        
        # Scoring logic:
        # - Coverage > 0.90: Excellent (score 0-1)
        # - Coverage 0.80-0.90: Good (score 1-3)
        # - Coverage 0.70-0.80: Fair (score 4-6)
        # - Coverage 0.50-0.70: Poor (score 7-8)
        # - Coverage < 0.50: Very poor (score 9-10)
        
        if coverage > 0.90:
            score = (1.0 - coverage) * 10.0
            interpretation = "Excellent generation coverage"
        elif coverage >= 0.80:
            score = 1.0 + (0.90 - coverage) / 0.10 * 2.0
            interpretation = "Good coverage with minor gaps"
        elif coverage >= 0.70:
            score = 4.0 + (0.80 - coverage) / 0.10 * 2.0
            interpretation = "Fair coverage with moderate gaps"
        elif coverage >= 0.50:
            score = 7.0 + (0.70 - coverage) / 0.20 * 1.0
            interpretation = "Poor coverage with significant gaps"
        else:
            score = 9.0 + min(1.0, (0.50 - coverage) / 0.30)
            interpretation = "Very poor coverage - severe airway loss"
        
        if self.verbose:
            print(f"  Generation Coverage: {coverage:.1%}")
            print(f"  Missing generations: {len(missing_gens)}")
            if len(missing_gens) > 0 and len(missing_gens) <= 10:
                print(f"  Missing: {missing_gens}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': coverage,
            'missing_count': len(missing_gens),
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }
    
    def compute_symmetry_score(self):
        """
        Score based on left/right symmetry.
        Lower symmetry = more asymmetric disease = moderate fibrosis indicator.
        """
        if self.verbose:
            print("\n[4/6] Computing Symmetry Score...")
        
        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")
        
        metrics = self.analyzer.advanced_metrics
        symmetry = metrics.get('symmetry_index', np.nan)
        
        if np.isnan(symmetry):
            if self.verbose:
                print("  ⚠ Symmetry index not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        ref = self.reference_values['symmetry_index']
        
        # Scoring logic:
        # - Symmetry > 0.90: Highly symmetric (score 0-1)
        # - Symmetry 0.80-0.90: Good symmetry (score 1-3)
        # - Symmetry 0.70-0.80: Fair symmetry (score 4-5)
        # - Symmetry 0.50-0.70: Moderate asymmetry (score 6-7)
        # - Symmetry < 0.50: Severe asymmetry (score 8-10)
        
        if symmetry > 0.90:
            score = (1.0 - symmetry) * 10.0
            interpretation = "Highly symmetric - bilateral disease pattern"
        elif symmetry >= 0.80:
            score = 1.0 + (0.90 - symmetry) / 0.10 * 2.0
            interpretation = "Good symmetry with minor differences"
        elif symmetry >= 0.70:
            score = 4.0 + (0.80 - symmetry) / 0.10 * 1.0
            interpretation = "Fair symmetry"
        elif symmetry >= 0.50:
            score = 6.0 + (0.70 - symmetry) / 0.20 * 1.0
            interpretation = "Moderate asymmetry - possible focal disease"
        else:
            score = 8.0 + min(2.0, (0.50 - symmetry) / 0.30 * 2.0)
            interpretation = "Severe asymmetry - unilateral predominance"
        
        if self.verbose:
            print(f"  Symmetry Index: {symmetry:.3f}")
            print(f"  Reference range: {ref['min']:.2f} - {ref['max']:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': symmetry,
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }
    
    def compute_volume_distribution_score(self):
        """
        Score based on volume distribution anomalies.
        Analyzes if volume distribution across generations is abnormal.
        """
        if self.verbose:
            print("\n[5/6] Computing Volume Distribution Score...")
        
        if not hasattr(self.analyzer, 'advanced_metrics'):
            raise ValueError("Run analyzer.compute_advanced_metrics() first")
        
        metrics = self.analyzer.advanced_metrics
        vol_per_gen = metrics.get('volume_per_generation', {})
        
        if len(vol_per_gen) < 3:
            if self.verbose:
                print("  ⚠ Insufficient generation data")
            return {'score': 5.0, 'value': 0, 'interpretation': 'Insufficient data'}
        
        # Analyze volume distribution pattern
        generations = sorted(vol_per_gen.keys())
        volumes = [vol_per_gen[g] for g in generations]
        
        # Expected pattern: volumes should decrease with generation
        # Calculate monotonicity and smoothness
        
        # 1. Check if volume generally decreases
        decreasing_count = sum(1 for i in range(len(volumes)-1) if volumes[i] > volumes[i+1])
        decreasing_ratio = decreasing_count / (len(volumes) - 1) if len(volumes) > 1 else 0
        
        # 2. Check for abrupt drops (>80% decrease between consecutive generations)
        abrupt_drops = 0
        for i in range(len(volumes) - 1):
            if volumes[i] > 0:
                drop_ratio = (volumes[i] - volumes[i+1]) / volumes[i]
                if drop_ratio > 0.80:
                    abrupt_drops += 1
        
        # 3. Check coefficient of variation in early generations (should be moderate)
        if len(volumes) >= 5:
            early_vols = volumes[:5]
            cv = np.std(early_vols) / np.mean(early_vols) if np.mean(early_vols) > 0 else 0
        else:
            cv = 0
        
        # Scoring:
        score = 0.0
        interpretation_parts = []
        
        # Penalty for non-monotonic decrease
        if decreasing_ratio < 0.6:
            score += 3.0
            interpretation_parts.append("erratic volume pattern")
        elif decreasing_ratio < 0.8:
            score += 1.5
            interpretation_parts.append("irregular decrease")
        
        # Penalty for abrupt drops
        if abrupt_drops > 2:
            score += 4.0
            interpretation_parts.append(f"multiple abrupt volume drops ({abrupt_drops})")
        elif abrupt_drops > 0:
            score += 2.0
            interpretation_parts.append(f"abrupt volume drop detected")
        
        # Penalty for high variability
        if cv > 1.5:
            score += 3.0
            interpretation_parts.append("high volume variability")
        elif cv > 1.0:
            score += 1.5
            interpretation_parts.append("moderate volume variability")
        
        score = min(10.0, score)
        
        if len(interpretation_parts) == 0:
            interpretation = "Normal volume distribution pattern"
        else:
            interpretation = "Abnormal pattern: " + ", ".join(interpretation_parts)
        
        if self.verbose:
            print(f"  Generations analyzed: {len(generations)}")
            print(f"  Decreasing ratio: {decreasing_ratio:.2f}")
            print(f"  Abrupt drops: {abrupt_drops}")
            print(f"  Coefficient of variation: {cv:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': decreasing_ratio,
            'abrupt_drops': abrupt_drops,
            'cv': cv,
            'interpretation': interpretation
        }
    
    def compute_tapering_score(self):
        """
        Score based on diameter tapering ratios.
        Abnormal tapering indicates airway remodeling.
        """
        if self.verbose:
            print("\n[6/6] Computing Tapering Score...")
        
        if not hasattr(self.analyzer, 'tapering_ratios_df') or self.analyzer.tapering_ratios_df is None:
            if self.verbose:
                print("  ⚠ Tapering ratios not available")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        if len(self.analyzer.tapering_ratios_df) == 0:
            if self.verbose:
                print("  ⚠ No tapering ratios computed")
            return {'score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        ratios = self.analyzer.tapering_ratios_df['diameter_ratio'].values
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        ref = self.reference_values['tapering_ratio']
        weibel_theoretical = 2**(-1/3)  # ~0.793
        
        # Scoring logic:
        # Ideal tapering should be close to Weibel's theoretical 0.793
        # - Deviation > 0.15: Severe abnormality (score 8-10)
        # - Deviation 0.10-0.15: Moderate abnormality (score 5-7)
        # - Deviation 0.05-0.10: Mild abnormality (score 2-4)
        # - Deviation < 0.05: Normal (score 0-2)
        
        deviation = abs(mean_ratio - weibel_theoretical)
        
        if deviation < 0.05:
            score = deviation / 0.05 * 2.0
            interpretation = "Normal tapering pattern"
        elif deviation < 0.10:
            score = 2.0 + (deviation - 0.05) / 0.05 * 2.0
            interpretation = "Mild tapering abnormality"
        elif deviation < 0.15:
            score = 5.0 + (deviation - 0.10) / 0.05 * 2.0
            interpretation = "Moderate tapering abnormality"
        else:
            score = 8.0 + min(2.0, (deviation - 0.15) / 0.10 * 2.0)
            interpretation = "Severe tapering abnormality"
        
        # Additional penalty for high variability
        if std_ratio > 0.15:
            score += 1.0
            interpretation += " with high variability"
        
        score = min(10.0, score)
        
        if self.verbose:
            print(f"  Mean tapering ratio: {mean_ratio:.3f}")
            print(f"  Weibel theoretical: {weibel_theoretical:.3f}")
            print(f"  Deviation: {deviation:.3f}")
            print(f"  Std deviation: {std_ratio:.3f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': mean_ratio,
            'deviation': deviation,
            'std': std_ratio,
            'interpretation': interpretation,
            'reference': weibel_theoretical
        }
    
    def compute_overall_score(self):
        """
        Computes overall fibrosis score by combining all components.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("COMPUTING OVERALL FIBROSIS SCORE")
            print("="*70)
        
        # Compute all component scores
        self.component_scores = {
            'pc_ratio': self.compute_pc_ratio_score(),
            'tortuosity': self.compute_tortuosity_score(),
            'generation_coverage': self.compute_generation_coverage_score(),
            'symmetry': self.compute_symmetry_score(),
            'volume_distribution': self.compute_volume_distribution_score(),
            'tapering': self.compute_tapering_score()
        }
        
        # Weighted sum
        total_score = 0.0
        available_weight = 0.0
        
        for component, weight in self.weights.items():
            comp_data = self.component_scores[component]
            raw_score = comp_data.get('raw_score', 5.0)
            
            # Only include if we have valid data
            if not np.isnan(comp_data.get('value', np.nan)) or component == 'volume_distribution':
                total_score += raw_score * weight
                available_weight += weight
        
        # Normalize to 0-100 scale
        if available_weight > 0:
            self.fibrosis_score = (total_score / available_weight) * 10.0  # Scale to 0-100
        else:
            self.fibrosis_score = 50.0  # Default if no data
        
        # Determine severity stage
        self.severity_stage = self._classify_severity(self.fibrosis_score)
        
        # Compute confidence based on data availability
        self.confidence = available_weight  # 0-1 scale
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("OVERALL FIBROSIS ASSESSMENT")
            print(f"{'='*70}")
            print(f"\nFibrosis Score: {self.fibrosis_score:.1f}/100")
            print(f"Severity Stage: {self.severity_stage}")
            print(f"Confidence: {self.confidence:.0%}")
            
            print(f"\n{'='*70}")
            print("COMPONENT BREAKDOWN")
            print(f"{'='*70}")
            
            for component, weight in self.weights.items():
                comp_data = self.component_scores[component]
                raw_score = comp_data.get('raw_score', 5.0)
                weighted_score = raw_score * weight * 10.0  # Scale to contribution
                
                print(f"\n{component.replace('_', ' ').title()}:")
                print(f"  Weight: {weight:.0%}")
                print(f"  Raw score: {raw_score:.1f}/10")
                print(f"  Contribution: {weighted_score:.1f} points")
                print(f"  {comp_data.get('interpretation', 'N/A')}")
        
        return self.fibrosis_score, self.severity_stage, self.confidence
    
    def _classify_severity(self, score):
        """
        Classifies severity based on total score.
        """
        if score < 20:
            return "Minimal/No fibrosis"
        elif score < 35:
            return "Mild fibrosis"
        elif score < 50:
            return "Moderate fibrosis"
        elif score < 70:
            return "Moderate-severe fibrosis (UIP pattern)"
        else:
            return "Severe/Advanced fibrosis"
    
    def generate_report(self, output_dir):
        """
        Generates comprehensive fibrosis assessment report.
        """
        if self.fibrosis_score is None:
            raise ValueError("Run compute_overall_score() first")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Text report
        report_path = os.path.join(output_dir, "fibrosis_assessment_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(" "*15 + "PULMONARY FIBROSIS ASSESSMENT REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"OVERALL ASSESSMENT\n")
            f.write("-"*70 + "\n")
            f.write(f"Fibrosis Score: {self.fibrosis_score:.1f}/100\n")
            f.write(f"Severity Classification: {self.severity_stage}\n")
            f.write(f"Assessment Confidence: {self.confidence:.0%}\n\n")
            
            f.write("="*70 + "\n")
            f.write("DETAILED COMPONENT ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            for component, weight in self.weights.items():
                comp_data = self.component_scores[component]
                f.write(f"{component.replace('_', ' ').upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"Weight in assessment: {weight:.0%}\n")
                f.write(f"Raw score: {comp_data.get('raw_score', 'N/A'):.1f}/10\n")
                f.write(f"Measured value: {comp_data.get('value', 'N/A')}\n")
                
                if 'reference_range' in comp_data:
                    ref_range = comp_data['reference_range']
                    f.write(f"Reference range: {ref_range[0]:.2f} - {ref_range[1]:.2f}\n")
                
                f.write(f"Interpretation: {comp_data.get('interpretation', 'N/A')}\n\n")
            
            f.write("="*70 + "\n")
            f.write("CLINICAL INTERPRETATION\n")
            f.write("="*70 + "\n\n")
            
            self._write_clinical_interpretation(f)
            
            f.write("\n" + "="*70 + "\n")
            f.write("DISCLAIMER\n")
            f.write("="*70 + "\n")
            f.write("This is an AUTOMATED ASSESSMENT based on quantitative airway morphometry.\n")
            f.write("It does NOT replace:\n")
            f.write("- Clinical evaluation by a pulmonologist\n")
            f.write("- Complete CT image review by a radiologist\n")
            f.write("- Pulmonary function tests (PFT)\n")
            f.write("- Histopathological diagnosis when indicated\n")
            f.write("- Multidisciplinary discussion (MDD)\n\n")
            f.write("Always correlate with complete clinical picture, imaging patterns,\n")
            f.write("and other diagnostic findings.\n")
        
        print(f"✓ Report saved: {report_path}")
        
        # 2. JSON export
        json_path = os.path.join(output_dir, "fibrosis_assessment.json")
        report_data = {
            'overall': {
                'fibrosis_score': float(self.fibrosis_score),
                'stage': self.severity_stage,
                'confidence': float(self.confidence)
            },
            'components': {}
        }
        
        for component, comp_data in self.component_scores.items():
            report_data['components'][component] = {
                'raw_score': float(comp_data.get('raw_score', np.nan)),
                'weighted_score': float(comp_data.get('raw_score', 5.0) * self.weights[component] * 10.0),
                'value': float(comp_data.get('value', np.nan)) if not isinstance(comp_data.get('value'), (list, dict)) else str(comp_data.get('value')),
                'interpretation': comp_data.get('interpretation', 'N/A')
            }
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"✓ JSON saved: {json_path}")
        
        # 3. Visualization
        self.plot_assessment(save_path=os.path.join(output_dir, "fibrosis_assessment_visualization.png"))
        
        return report_data
    
    def _write_clinical_interpretation(self, file_handle):
        """
        Writes clinical interpretation based on score and components.
        """
        f = file_handle
        score = self.fibrosis_score
        
        if score < 20:
            f.write("No significant fibrotic changes detected based on airway morphometry.\n")
            f.write("Airways appear structurally normal with preserved peripheral airways.\n")
            f.write("\nRecommendation:\n")
            f.write("- No specific follow-up required for fibrosis\n")
            f.write("- Standard screening if risk factors present\n")
        
        elif score < 35:
            f.write("Minimal fibrotic changes detected.\n")
            f.write("Early peripheral airway involvement with preserved central airways.\n")
            
            pc_ratio = self.component_scores['pc_ratio'].get('raw_score', 0)
            if pc_ratio > 5:
                f.write("Note: Peripheral airway loss is the primary finding.\n")
            
            f.write("\nRecommendation:\n")
            f.write("- Monitor for progression with follow-up CT in 6-12 months\n")
            f.write("- Baseline pulmonary function tests\n")
            f.write("- Clinical correlation advised\n")
        
        elif score < 50:
            f.write("Mild fibrosis with measurable structural airway changes.\n")
            f.write("Bilateral peripheral airway involvement detected.\n")
            
            components_high = [k for k, v in self.component_scores.items() 
                             if v.get('raw_score', 0) > 6]
            if components_high:
                f.write(f"Prominent findings: {', '.join([k.replace('_', ' ') for k in components_high])}\n")
            
            f.write("\nRecommendation:\n")
            f.write("- Pulmonologist consultation recommended\n")
            f.write("- Complete pulmonary function tests (PFT) including DLCO\n")
            f.write("- Follow-up CT in 3-6 months\n")
            f.write("- Consider autoimmune workup if clinically indicated\n")
        
        elif score < 70:
            f.write("Moderate fibrosis with features suggesting UIP pattern.\n")
            f.write("Significant peripheral airway loss and structural remodeling.\n")
            
            pc_ratio = self.component_scores['pc_ratio'].get('value', 0.5)
            if pc_ratio < 0.25:
                f.write("ALERT: Severe peripheral airway loss detected (P/C < 0.25)\n")
            
            coverage = self.component_scores['generation_coverage'].get('value', 1.0)
            if coverage < 0.70:
                f.write("ALERT: Significant airway generation loss (>30% missing)\n")
            
            f.write("\nRecommendation:\n")
            f.write("- URGENT pulmonologist referral\n")
            f.write("- Complete diagnostic workup:\n")
            f.write("  * High-resolution CT with complete evaluation\n")
            f.write("  * Comprehensive PFTs with 6-minute walk test\n")
            f.write("  * Consider bronchoscopy ± biopsy\n")
            f.write("- Multidisciplinary discussion recommended\n")
            f.write("- Consider antifibrotic therapy evaluation\n")
        
        else:  # score >= 70
            f.write("Severe/advanced fibrosis detected.\n")
            f.write("Extensive airway destruction with marked peripheral loss.\n")
            f.write("Pattern consistent with advanced fibrotic lung disease.\n")
            
            f.write("\nCRITICAL FINDINGS:\n")
            
            for component, comp_data in self.component_scores.items():
                if comp_data.get('raw_score', 0) >= 8:
                    f.write(f"- {component.replace('_', ' ').title()}: {comp_data.get('interpretation')}\n")
            
            f.write("\nRecommendation:\n")
            f.write("- URGENT pulmonary specialist evaluation\n")
            f.write("- Comprehensive assessment:\n")
            f.write("  * Complete PFT battery\n")
            f.write("  * Echocardiography (evaluate pulmonary hypertension)\n")
            f.write("  * 6-minute walk test with oxygen saturation\n")
            f.write("  * Consider right heart catheterization\n")
            f.write("- Multidisciplinary team discussion MANDATORY\n")
            f.write("- Antifibrotic therapy consideration\n")
            f.write("- Lung transplant evaluation if appropriate\n")
            f.write("- Palliative care consultation if end-stage\n")
    
    def plot_assessment(self, save_path=None):
        """
        Creates comprehensive visualization of fibrosis assessment.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall score gauge (top center)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_score_gauge(ax1)
        
        # 2. Component radar chart (middle left)
        ax2 = fig.add_subplot(gs[1, 0], projection='polar')
        self._plot_radar_chart(ax2)
        
        # 3. Component bar chart (middle center)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_component_bars(ax3)
        
        # 4. Reference comparison (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_reference_comparison(ax4)
        
        # 5. Severity distribution (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_severity_distribution(ax5)
        
        # 6. Key metrics table (bottom center and right)
        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_metrics_table(ax6)
        
        plt.suptitle('Pulmonary Fibrosis Assessment - Airway Morphometry Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()
    
    def _plot_score_gauge(self, ax):
        """Plot overall score as a gauge/speedometer."""
        ax.clear()
        ax.axis('off')
        
        score = self.fibrosis_score
        
        # Draw gauge segments
        theta = np.linspace(np.pi, 0, 100)
        
        # Color segments
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
        boundaries = [0, 20, 35, 50, 70, 100]
        
        for i in range(len(boundaries)-1):
            start_angle = np.pi * (1 - boundaries[i]/100)
            end_angle = np.pi * (1 - boundaries[i+1]/100)
            theta_segment = np.linspace(start_angle, end_angle, 20)
            
            x = np.concatenate([[0], np.cos(theta_segment), [0]])
            y = np.concatenate([[0], np.sin(theta_segment), [0]])
            
            ax.fill(x, y, color=colors[i], alpha=0.6, edgecolor='black', linewidth=2)
        
        # Draw needle
        needle_angle = np.pi * (1 - score/100)
        needle_x = [0, 0.9 * np.cos(needle_angle)]
        needle_y = [0, 0.9 * np.sin(needle_angle)]
        ax.plot(needle_x, needle_y, 'k-', linewidth=4)
        ax.plot(0, 0, 'ko', markersize=15)
        
        # Add text
        ax.text(0, -0.3, f'{score:.1f}', ha='center', va='top', 
               fontsize=36, fontweight='bold')
        ax.text(0, -0.45, f'{self.severity_stage}', ha='center', va='top',
               fontsize=14, style='italic')
        ax.text(0, -0.6, f'Confidence: {self.confidence:.0%}', ha='center', va='top',
               fontsize=11)
        
        # Labels
        ax.text(-0.95, 0, 'Minimal', ha='right', va='center', fontsize=9)
        ax.text(0.95, 0, 'Severe', ha='left', va='center', fontsize=9)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 1.2)
        ax.set_aspect('equal')
    
    def _plot_radar_chart(self, ax):
        """Plot component scores as radar chart."""
        categories = [k.replace('_', ' ').title() for k in self.weights.keys()]
        scores = [self.component_scores[k].get('raw_score', 5.0) for k in self.weights.keys()]
        
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        scores += scores[:1]
        angles += angles[:1]
        
        ax.plot(angles, scores, 'o-', linewidth=2, color='#3498db', label='Patient')
        ax.fill(angles, scores, alpha=0.25, color='#3498db')
        
        # Reference (healthy)
        reference_scores = [2.0] * (N + 1)
        ax.plot(angles, reference_scores, '--', linewidth=1.5, color='#2ecc71', 
               label='Normal range', alpha=0.7)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.set_title('Component Scores\n(0-10 scale)', fontsize=11, pad=20)
    
    def _plot_component_bars(self, ax):
        """Plot component contributions as bar chart."""
        components = list(self.weights.keys())
        contributions = [self.component_scores[k].get('raw_score', 5.0) * 
                        self.weights[k] * 10 for k in components]
        
        colors_map = {
            'pc_ratio': '#e74c3c',
            'tortuosity': '#e67e22',
            'generation_coverage': '#f39c12',
            'symmetry': '#3498db',
            'volume_distribution': '#9b59b6',
            'tapering': '#1abc9c'
        }
        
        colors = [colors_map.get(k, '#95a5a6') for k in components]
        labels = [k.replace('_', ' ').title() for k in components]
        
        bars = ax.barh(labels, contributions, color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, contributions)):
            ax.text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=9)
        
        ax.set_xlabel('Contribution to Total Score', fontsize=10)
        ax.set_title('Component Contributions', fontsize=11, fontweight='bold')
        ax.set_xlim(0, max(contributions) * 1.15)
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_reference_comparison(self, ax):
        """Plot measured values vs reference ranges."""
        components_with_refs = {k: v for k, v in self.component_scores.items() 
                               if 'reference_range' in v}
        
        if not components_with_refs:
            ax.text(0.5, 0.5, 'No reference\ndata available', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        y_pos = np.arange(len(components_with_refs))
        labels = [k.replace('_', ' ').title() for k in components_with_refs.keys()]
        
        for i, (component, comp_data) in enumerate(components_with_refs.items()):
            ref_range = comp_data['reference_range']
            value = comp_data.get('value', np.nan)
            
            if np.isnan(value):
                continue
            
            # Plot reference range as horizontal line
            ax.plot(ref_range, [i, i], 'k-', linewidth=8, alpha=0.2, solid_capstyle='round')
            
            # Plot measured value
            color = '#2ecc71' if ref_range[0] <= value <= ref_range[1] else '#e74c3c'
            ax.plot(value, i, 'o', markersize=12, color=color, markeredgecolor='black', 
                   markeredgewidth=1.5, zorder=3)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_title('Measured vs Reference Range', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    def _plot_severity_distribution(self, ax):
        """Plot severity stage distribution."""
        stages = ['Minimal/No', 'Mild', 'Moderate', 'Moderate-severe', 'Severe/Advanced']
        boundaries = [0, 20, 35, 50, 70, 100]
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
        
        for i in range(len(stages)):
            ax.barh(0, boundaries[i+1] - boundaries[i], left=boundaries[i],
                   color=colors[i], edgecolor='black', linewidth=1, alpha=0.7)
            
            # Add stage label
            mid = (boundaries[i] + boundaries[i+1]) / 2
            ax.text(mid, 0, stages[i], ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Mark patient score
        ax.plot(self.fibrosis_score, 0, 'v', markersize=15, color='blue',
               markeredgecolor='black', markeredgewidth=2, zorder=3)
        ax.text(self.fibrosis_score, -0.5, f'Patient\n{self.fibrosis_score:.1f}',
               ha='center', va='top', fontsize=9, fontweight='bold')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.7, 0.5)
        ax.set_xlabel('Fibrosis Score', fontsize=10)
        ax.set_title('Severity Classification', fontsize=11, fontweight='bold')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    def _plot_metrics_table(self, ax):
        """Plot key metrics as a table."""
        ax.axis('off')
        
        # Prepare data
        table_data = []
        table_data.append(['Overall Score', f'{self.fibrosis_score:.1f}/100', ''])
        table_data.append(['Severity Stage', self.severity_stage, ''])
        table_data.append(['Confidence', f'{self.confidence:.0%}', ''])
        table_data.append(['', '', ''])  # Separator
        
        # Add key component metrics
        for component, comp_data in self.component_scores.items():
            name = component.replace('_', ' ').title()
            value = comp_data.get('value', 'N/A')
            
            if isinstance(value, float):
                if not np.isnan(value):
                    value_str = f'{value:.3f}'
                else:
                    value_str = 'N/A'
            else:
                value_str = str(value)
            
            score = comp_data.get('raw_score', 'N/A')
            if isinstance(score, float):
                score_str = f'{score:.1f}/10'
            else:
                score_str = str(score)
            
            table_data.append([name, value_str, score_str])
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value', 'Score'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.4, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(table_data) + 1):
            if i == 4:  # Separator
                for j in range(3):
                    table[(i, j)].set_facecolor('#ecf0f1')
            elif i < 4:  # Overall metrics
                for j in range(3):
                    table[(i, j)].set_facecolor('#e8f8f5')
                    table[(i, j)].set_text_props(weight='bold')
            else:  # Component metrics
                for j in range(3):
                    table[(i, j)].set_facecolor('#ffffff' if i % 2 == 0 else '#f8f9fa')
        
        ax.set_title('Key Metrics Summary', fontsize=11, fontweight='bold', pad=10)


def integrate_fibrosis_scoring(analyzer, output_dir):
    """
    Integration function for the complete pipeline.
    
    Args:
        analyzer: AirwayGraphAnalyzer instance with computed advanced metrics
        output_dir: Directory to save fibrosis assessment results
    
    Returns:
        scorer: PulmonaryFibrosisScorer instance
        report: Dictionary with assessment results
    """
    print("\n" + "="*70)
    print(" "*15 + "PULMONARY FIBROSIS SCORING SYSTEM")
    print("="*70)
    
    # Ensure advanced metrics are computed
    if not hasattr(analyzer, 'advanced_metrics'):
        print("\nComputing advanced metrics...")
        try:
            analyzer.compute_advanced_metrics()
        except Exception as e:
            print(f"⚠ Warning: Could not compute advanced metrics: {e}")
            # Return default values if metrics computation fails
            return None, None
    
    # Create scorer
    scorer = PulmonaryFibrosisScorer(analyzer, verbose=True)
    
    # Compute overall score
    fibrosis_score, severity_stage, confidence = scorer.compute_overall_score()
    
    # Generate comprehensive report
    report = scorer.generate_report(output_dir)
    
    print(f"\n{'='*70}")
    print("FIBROSIS ASSESSMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Score: {fibrosis_score:.1f}/100")
    print(f"Stage: {severity_stage}")
    print(f"Confidence: {confidence:.0%}")
    print(f"\nResults saved to: {output_dir}")
    
    return scorer, report