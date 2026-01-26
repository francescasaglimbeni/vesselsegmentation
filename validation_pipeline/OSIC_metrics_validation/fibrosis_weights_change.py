"""
Script to recalculate fibrosis scores with VALIDATED weights based on FVC correlation analysis.

NEW WEIGHTS (evidence-based):
- Tortuosity: 0.50 (only metric that correlates with FVC)
- Airway Volume: 0.30 (weak but significant correlation)
- P/C Ratio: 0.05 (REDUCED - no correlation with FVC)
- Symmetry: 0.10 (kept for clinical relevance)
- Generation Coverage: 0.05 (reduced)
- Tapering: 0.00 (REMOVED - no correlation)

This script:
1. Loads existing advanced_metrics.json from each patient
2. Recalculates fibrosis scores with new weights
3. Updates step6_fibrosis_assessment folder
4. Updates COMPLETE_ANALYSIS_REPORT.txt
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

# Import the fibrosis scoring class
sys.path.append(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation")
from fibrosis_scoring import PulmonaryFibrosisScorer


class ValidatedFibrosisScorer:
    """
    VALIDATED version of fibrosis scorer with evidence-based weights
    from FVC correlation analysis.
    
    Standalone implementation that works with pre-computed metrics.
    """
    
    def __init__(self, advanced_metrics, verbose=True):
        """
        Args:
            advanced_metrics: Dictionary with pre-computed advanced metrics
            verbose: Print detailed information
        """
        self.verbose = verbose
        
        # VALIDATED WEIGHTS based on FVC correlation analysis
        self.weights = {
            'tortuosity': 0.50,          # VALIDATED (r=-0.267, p<0.001)
            'airway_volume': 0.30,       # VALIDATED (r=+0.245, p<0.001)
            'pc_ratio': 0.05,            # REDUCED (r=-0.062, NS)
            'symmetry': 0.10,            # Kept for clinical relevance
            'generation_coverage': 0.05, # Reduced
            'tapering': 0.00             # REMOVED (no correlation)
        }
        
        # Reference values (same as original)
        self.reference_values = {
            'pc_ratio': {'mean': 0.45, 'std': 0.15, 'min': 0.25, 'max': 0.65},
            'tortuosity': {'mean': 1.25, 'std': 0.15, 'min': 1.0, 'max': 1.5},
            'generation_coverage': {'mean': 0.85, 'std': 0.10, 'min': 0.70, 'max': 1.0},
            'symmetry_index': {'mean': 0.85, 'std': 0.10, 'min': 0.70, 'max': 1.0},
            'tapering_ratio': {'mean': 0.79, 'std': 0.05, 'min': 0.70, 'max': 0.88}
        }
        
        # Store pre-computed metrics
        self.advanced_metrics = advanced_metrics
        
        # Initialize results
        self.component_scores = {}
        self.fibrosis_score = None
        self.severity_stage = None
        self.confidence = None
    
    def _classify_severity(self, score):
        """Classify severity stage based on score"""
        if score < 20:
            return "Normal"
        elif score < 35:
            return "Minimal"
        elif score < 50:
            return "Mild"
        elif score < 70:
            return "Moderate"
        else:
            return "Severe"
    
    
    def compute_tortuosity_score(self):
        """Score based on mean tortuosity from pre-computed metrics"""
        if self.verbose:
            print("\n[1/6] Computing Tortuosity Score...")
        
        metrics = self.advanced_metrics
        tortuosity = metrics.get('mean_tortuosity', np.nan)
        
        if np.isnan(tortuosity):
            if self.verbose:
                print("  WARNING: Tortuosity not available")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        ref = self.reference_values['tortuosity']
        
        if tortuosity < ref['min']:
            score = 0.0
            interpretation = "Very low tortuosity - excellent airway straightness"
        elif tortuosity < ref['mean']:
            score = 3.0 * (tortuosity - ref['min']) / (ref['mean'] - ref['min'])
            interpretation = "Low tortuosity - good airway straightness"
        elif tortuosity < ref['max']:
            score = 3.0 + 4.0 * (tortuosity - ref['mean']) / (ref['max'] - ref['mean'])
            interpretation = "Moderate tortuosity"
        else:
            score = 7.0 + min(3.0, (tortuosity - ref['max']) / 0.3)
            interpretation = "High tortuosity - suggests fibrotic changes"
        
        score = min(10.0, max(0.0, score))
        
        if self.verbose:
            print(f"  Mean Tortuosity: {tortuosity:.3f}")
            print(f"  Reference: {ref['min']:.2f} - {ref['max']:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': tortuosity,
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }
    
    def compute_pc_ratio_score(self):
        """Score based on peripheral/central ratio"""
        if self.verbose:
            print("\n[2/6] Computing P/C Ratio Score...")
        
        # Use the peripheral_to_central_ratio from advanced metrics
        pc_ratio = self.advanced_metrics.get('peripheral_to_central_ratio', np.nan)
        
        if np.isnan(pc_ratio):
            if self.verbose:
                print("  WARNING: P/C ratio not available")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        ref = self.reference_values['pc_ratio']
        
        if pc_ratio > ref['max']:
            score = 0.0
            interpretation = "Excellent peripheral airway preservation"
        elif pc_ratio > ref['mean']:
            score = 3.0 * (ref['max'] - pc_ratio) / (ref['max'] - ref['mean'])
            interpretation = "Good peripheral distribution"
        elif pc_ratio > ref['min']:
            score = 3.0 + 4.0 * (ref['mean'] - pc_ratio) / (ref['mean'] - ref['min'])
            interpretation = "Reduced peripheral airways"
        else:
            score = 7.0 + min(3.0, (ref['min'] - pc_ratio) / 0.15)
            interpretation = "Severe peripheral airway loss"
        
        score = min(10.0, max(0.0, score))
        
        if self.verbose:
            print(f"  P/C Ratio: {pc_ratio:.3f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': pc_ratio,
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }
    
    def compute_symmetry_score(self):
        """Dummy symmetry score (would need left/right analysis)"""
        if self.verbose:
            print("\n[3/6] Computing Symmetry Score...")
            print("  WARNING: Symmetry analysis not available from advanced metrics")
        
        return {
            'raw_score': 5.0,
            'value': np.nan,
            'interpretation': 'Not computed (requires bilateral analysis)',
            'reference_range': (0.7, 1.0)
        }
    
    def compute_generation_coverage_score(self):
        """Score based on generation coverage"""
        if self.verbose:
            print("\n[4/6] Computing Generation Coverage Score...")
        
        metrics = self.advanced_metrics
        max_gen = metrics.get('max_generation', np.nan)
        
        if np.isnan(max_gen):
            if self.verbose:
                print("  WARNING: Generation data not available")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        # Typical healthy: 10-12 generations visible
        # Fibrotic: <8 generations
        coverage = max_gen / 12.0
        
        ref = self.reference_values['generation_coverage']
        
        if coverage > ref['max']:
            score = 0.0
            interpretation = "Excellent generation coverage"
        elif coverage > ref['mean']:
            score = 3.0 * (ref['max'] - coverage) / (ref['max'] - ref['mean'])
            interpretation = "Good generation coverage"
        elif coverage > ref['min']:
            score = 3.0 + 4.0 * (ref['mean'] - coverage) / (ref['mean'] - ref['min'])
            interpretation = "Reduced generation coverage"
        else:
            score = 7.0 + min(3.0, (ref['min'] - coverage) / 0.2)
            interpretation = "Poor generation coverage"
        
        score = min(10.0, max(0.0, score))
        
        if self.verbose:
            print(f"  Max Generation: {max_gen}")
            print(f"  Coverage: {coverage:.2f}")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': coverage,
            'interpretation': interpretation,
            'reference_range': (ref['min'], ref['max'])
        }
    
    def compute_volume_distribution_score(self):
        """Score based on volume distribution"""
        if self.verbose:
            print("\n[5/6] Computing Volume Distribution Score...")
        
        # Similar to P/C ratio
        return self.compute_pc_ratio_score()
    
    def compute_tapering_score(self):
        """Score based on tapering ratio (REMOVED in validated version)"""
        if self.verbose:
            print("\n[6/6] Computing Tapering Score...")
            print("  REMOVED - no correlation with FVC")
        
        return {
            'raw_score': 0.0,
            'value': np.nan,
            'interpretation': 'Removed from validated scoring',
            'reference_range': (0.7, 0.88)
        }
    
    def compute_airway_volume_score(self):
        """
        NEW COMPONENT: Score based on total airway volume
        Higher volume = better preserved airways = lower fibrosis score
        """
        if self.verbose:
            print("\n[NEW] Computing Airway Volume Score...")
        
        metrics = self.advanced_metrics
        volume = metrics.get('total_volume_mm3', np.nan)
        
        if np.isnan(volume):
            if self.verbose:
                print("  WARNING: Volume not available")
            return {'raw_score': 5.0, 'value': np.nan, 'interpretation': 'Unknown'}
        
        # Reference values (approximate from healthy lungs)
        # Healthy adult: ~20,000-40,000 mm³ of visible airways
        # Fibrotic lung: <15,000 mm³
        
        if volume > 30000:
            score = 0.0
            interpretation = "Excellent airway volume - well preserved"
        elif volume > 20000:
            # Linear 20k-30k -> 0-3
            score = 3.0 * (30000 - volume) / 10000
            interpretation = "Good airway volume"
        elif volume > 15000:
            # Linear 15k-20k -> 3-6
            score = 3.0 + 3.0 * (20000 - volume) / 5000
            interpretation = "Moderate airway volume reduction"
        elif volume > 10000:
            # Linear 10k-15k -> 6-8
            score = 6.0 + 2.0 * (15000 - volume) / 5000
            interpretation = "Significant airway volume loss"
        else:
            # <10k -> 8-10
            score = 8.0 + min(2.0, (10000 - volume) / 5000)
            interpretation = "Severe airway volume loss"
        
        score = min(10.0, max(0.0, score))
        
        if self.verbose:
            print(f"  Total Volume: {volume:.2f} mm³")
            print(f"  Score: {score:.1f}/10")
            print(f"  {interpretation}")
        
        return {
            'raw_score': score,
            'value': volume,
            'interpretation': interpretation,
            'reference_range': (15000, 30000)
        }
    
    def compute_overall_score(self):
        """
        Computes overall fibrosis score with VALIDATED weights
        """
        if self.verbose:
            print("\n" + "="*70)
            print("COMPUTING FIBROSIS SCORE WITH VALIDATED WEIGHTS")
            print("="*70)
        
        # Compute all component scores
        self.component_scores = {
            'tortuosity': self.compute_tortuosity_score(),
            'airway_volume': self.compute_airway_volume_score(),
            'pc_ratio': self.compute_pc_ratio_score(),
            'symmetry': self.compute_symmetry_score(),
            'generation_coverage': self.compute_generation_coverage_score(),
            'volume_distribution': self.compute_volume_distribution_score(),
            'tapering': self.compute_tapering_score()
        }
        
        # Weighted sum
        total_score = 0.0
        available_weight = 0.0
        
        for component, weight in self.weights.items():
            if weight == 0.0:  # Skip tapering
                continue
            
            # Map component names
            comp_key = component
            if component == 'airway_volume':
                comp_key = 'airway_volume'
            
            comp_data = self.component_scores.get(comp_key, 
                                                  self.component_scores.get('volume_distribution'))
            
            if comp_data is None:
                continue
            
            raw_score = comp_data.get('raw_score', 5.0)
            
            # Only include if we have valid data
            valid = not np.isnan(comp_data.get('value', np.nan))
            if valid or component in ['volume_distribution', 'airway_volume']:
                total_score += raw_score * weight
                available_weight += weight
        
        # Normalize to 0-100 scale
        if available_weight > 0:
            self.fibrosis_score = (total_score / available_weight) * 10.0
        else:
            self.fibrosis_score = 50.0
        
        # Determine severity stage
        self.severity_stage = self._classify_severity(self.fibrosis_score)
        
        # Compute confidence
        self.confidence = available_weight
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("VALIDATED FIBROSIS ASSESSMENT")
            print(f"{'='*70}")
            print(f"\nFibrosis Score: {self.fibrosis_score:.1f}/100")
            print(f"Severity Stage: {self.severity_stage}")
            print(f"Confidence: {self.confidence:.0%}")
            
            print(f"\n{'='*70}")
            print("VALIDATED COMPONENT BREAKDOWN")
            print(f"{'='*70}")
            
            for component, weight in self.weights.items():
                if weight == 0.0:
                    print(f"\n{component.replace('_', ' ').title()}:")
                    print(f"  Weight: {weight:.0%} (REMOVED - no FVC correlation)")
                    continue
                
                comp_key = component
                comp_data = self.component_scores.get(comp_key, 
                                                      self.component_scores.get('volume_distribution'))
                
                if comp_data is None:
                    continue
                
                raw_score = comp_data.get('raw_score', 5.0)
                weighted_score = raw_score * weight * 10.0
                
                validation = ""
                if component == 'tortuosity':
                    validation = " [VALIDATED] (r=-0.267, p<0.001)"
                elif component == 'airway_volume':
                    validation = " [VALIDATED] (r=+0.245, p<0.001)"
                elif component == 'pc_ratio':
                    validation = " [REDUCED] (r=-0.062, NS)"
                
                print(f"\n{component.replace('_', ' ').title()}{validation}:")
                print(f"  Weight: {weight:.0%}")
                print(f"  Raw score: {raw_score:.1f}/10")
                print(f"  Contribution: {weighted_score:.1f} points")
                print(f"  {comp_data.get('interpretation', 'N/A')}")
        
        return self.fibrosis_score, self.severity_stage, self.confidence
    
    def generate_report(self, output_dir):
        """Generate fibrosis assessment report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        report_data = {
            'overall': {
                'fibrosis_score': self.fibrosis_score,
                'stage': self.severity_stage,
                'confidence': self.confidence,
                'weights_version': 'VALIDATED (FVC-based)'
            },
            'components': {}
        }
        
        for component, weight in self.weights.items():
            comp_key = component
            if component == 'airway_volume':
                comp_key = 'airway_volume'
            
            comp_data = self.component_scores.get(comp_key, 
                                                  self.component_scores.get('volume_distribution'))
            
            if comp_data is None:
                continue
            
            raw_score = comp_data.get('raw_score', 5.0)
            weighted_score = raw_score * weight * 10.0
            
            report_data['components'][component] = {
                'raw_score': raw_score,
                'weight': weight,
                'weighted_score': weighted_score,
                'value': comp_data.get('value', np.nan),
                'interpretation': comp_data.get('interpretation', 'N/A')
            }
        
        # Save JSON
        json_path = output_dir / "fibrosis_assessment_validated.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
        
        # Save text report
        txt_path = output_dir / "FIBROSIS_REPORT_VALIDATED.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PULMONARY FIBROSIS ASSESSMENT (VALIDATED WEIGHTS)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Fibrosis Score: {self.fibrosis_score:.1f}/100\n")
            f.write(f"Severity Stage: {self.severity_stage}\n")
            f.write(f"Confidence: {self.confidence:.0%}\n")
            f.write(f"Weights: VALIDATED (based on FVC correlation)\n\n")
            
            f.write("="*80 + "\n")
            f.write("COMPONENT BREAKDOWN\n")
            f.write("="*80 + "\n\n")
            
            for component in sorted(report_data['components'].keys(), 
                                   key=lambda x: self.weights.get(x, 0), reverse=True):
                comp_data = report_data['components'][component]
                
                validation = ""
                if component == 'tortuosity':
                    validation = " [VALIDATED]"
                elif component == 'airway_volume':
                    validation = " [VALIDATED]"
                elif component == 'pc_ratio':
                    validation = " [REDUCED]"
                elif component == 'tapering':
                    validation = " [REMOVED]"
                
                f.write(f"{component.replace('_', ' ').title()}{validation}:\n")
                f.write(f"  Weight: {comp_data['weight']:.0%}\n")
                f.write(f"  Raw Score: {comp_data['raw_score']:.1f}/10\n")
                f.write(f"  Contribution: {comp_data['weighted_score']:.1f} points\n")
                f.write(f"  {comp_data['interpretation']}\n\n")
        
        if self.verbose:
            print(f"\nReport saved to: {output_dir}")
        
        return report_data


def recalculate_patient_fibrosis(patient_dir, backup=True):
    """
    Recalculate fibrosis score for a single patient
    
    Args:
        patient_dir: Path to patient directory (e.g., .../ID00020637202178344345685)
        backup: Whether to backup existing results
    
    Returns:
        dict with results or None if failed
    """
    patient_dir = Path(patient_dir)
    patient_name = patient_dir.name
    
    print(f"\n{'='*80}")
    print(f"RECALCULATING: {patient_name}")
    print(f"{'='*80}")
    
    # Paths
    step6_dir = patient_dir / "step6_fibrosis_assessment"
    step4_dir = patient_dir / "step4_analysis"
    report_path = patient_dir / "COMPLETE_ANALYSIS_REPORT.txt"
    
    # Check if directories exist
    if not step4_dir.exists():
        print(f"  WARNING: Missing step4_analysis directory, skipping")
        return None
    
    # Load advanced metrics
    metrics_json = step4_dir / "advanced_metrics.json"
    if not metrics_json.exists():
        print(f"  WARNING: Missing advanced_metrics.json, skipping")
        return None
    
    with open(metrics_json, 'r') as f:
        advanced_metrics = json.load(f)
    
    print(f"  Loaded advanced metrics")
    
    # Backup existing step6 if requested
    if backup and step6_dir.exists():
        backup_dir = patient_dir / f"step6_fibrosis_assessment_OLD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(step6_dir, backup_dir)
        print(f"  Backed up to: {backup_dir.name}")
    
    # Create/recreate step6 directory
    step6_dir.mkdir(parents=True, exist_ok=True)
    
    # Create validated scorer
    scorer = ValidatedFibrosisScorer(advanced_metrics, verbose=True)
    
    # Compute score
    fibrosis_score, severity_stage, confidence = scorer.compute_overall_score()
    
    # Generate report
    report_data = scorer.generate_report(str(step6_dir))
    
    # Update COMPLETE_ANALYSIS_REPORT.txt
    if report_path.exists():
        update_complete_report(report_path, report_data, backup=backup)
    
    print(f"\n  RECALCULATION COMPLETE:")
    print(f"    New Score: {fibrosis_score:.1f}/100 ({severity_stage})")
    print(f"    Confidence: {confidence:.0%}")
    
    return {
        'patient': patient_name,
        'score': fibrosis_score,
        'stage': severity_stage,
        'confidence': confidence
    }


def update_complete_report(report_path, fibrosis_data, backup=True):
    """
    Update COMPLETE_ANALYSIS_REPORT.txt with new fibrosis assessment
    """
    print(f"\n  Updating COMPLETE_ANALYSIS_REPORT.txt...")
    
    # Backup original
    if backup:
        backup_path = report_path.parent / f"{report_path.stem}_OLD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        shutil.copy(report_path, backup_path)
    
    # Read original report
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace fibrosis section
    start_marker = "=" * 80 + "\nPULMONARY FIBROSIS ASSESSMENT\n" + "=" * 80
    end_marker = "=" * 80 + "\nDISCLAIMER\n" + "=" * 80
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx)
    
    if start_idx == -1 or end_idx == -1:
        print(f"    WARNING: Could not find fibrosis section in report")
        return
    
    # Build new fibrosis section
    overall = fibrosis_data['overall']
    components = fibrosis_data['components']
    
    new_section = start_marker + "\n\n"
    new_section += f"FIBROSIS SCORE: {overall['fibrosis_score']:.1f}/100 (VALIDATED WEIGHTS)\n"
    new_section += f"CLASSIFICATION: {overall['stage']}\n"
    new_section += f"CONFIDENCE: {overall['confidence']:.0%}\n\n"
    
    new_section += "VALIDATED WEIGHTS (based on FVC correlation analysis):\n"
    new_section += "-" * 80 + "\n"
    new_section += "[VALIDATED] Tortuosity: 50% (r=-0.267, p<0.001 with FVC)\n"
    new_section += "[VALIDATED] Airway Volume: 30% (r=+0.245, p<0.001 with FVC)\n"
    new_section += "[REDUCED] P/C Ratio: 5% (r=-0.062, NS - REDUCED)\n"
    new_section += "Symmetry: 10% (clinical relevance)\n"
    new_section += "Generation Coverage: 5% (reduced)\n"
    new_section += "[REMOVED] Tapering: 0% (REMOVED - no correlation)\n\n"
    
    new_section += "Component Breakdown:\n"
    new_section += "-" * 80 + "\n"
    
    # Order components by weight
    weight_order = {
        'tortuosity': 0.50,
        'airway_volume': 0.30,
        'symmetry': 0.10,
        'generation_coverage': 0.05,
        'pc_ratio': 0.05,
        'volume_distribution': 0.10,
        'tapering': 0.00
    }
    
    for comp_name in sorted(components.keys(), key=lambda x: weight_order.get(x, 0), reverse=True):
        comp_data = components[comp_name]
        
        validation = ""
        if comp_name == 'tortuosity':
            validation = " [VALIDATED]"
        elif comp_name == 'airway_volume':
            validation = " [VALIDATED]"
        elif comp_name == 'pc_ratio':
            validation = " [REDUCED WEIGHT]"
        elif comp_name == 'tapering':
            validation = " [REMOVED]"
        
        new_section += f"\n{comp_name.replace('_', ' ').title()}{validation}:\n"
        new_section += f"  Contribution: {comp_data['weighted_score']:.1f} points\n"
        new_section += f"  Raw score: {comp_data['raw_score']:.1f}/10\n"
        new_section += f"  Interpretation: {comp_data['interpretation']}\n"
    
    new_section += "\n" + "=" * 80 + "\n"
    new_section += "CLINICAL INTERPRETATION\n"
    new_section += "=" * 80 + "\n\n"
    
    score = overall['fibrosis_score']
    
    if score < 20:
        new_section += "No significant fibrotic changes detected.\n"
        new_section += "Airways appear structurally normal.\n"
    elif score < 35:
        new_section += "Minimal fibrotic changes detected.\n"
        new_section += "Early peripheral airway involvement.\n"
        new_section += "Recommendation: Monitor for progression.\n"
    elif score < 50:
        new_section += "Mild fibrosis with measurable airway changes.\n"
        new_section += "Recommendation: Clinical correlation and follow-up.\n"
    elif score < 70:
        new_section += "Moderate fibrosis with UIP-pattern features.\n"
        new_section += "Recommendation: PFT and specialist consultation.\n"
    else:
        new_section += "Severe/advanced fibrosis detected.\n"
        new_section += "Recommendation: Urgent pulmonary evaluation.\n"
    
    new_section += "\nNOTE: This score uses VALIDATED weights based on correlation\n"
    new_section += "with FVC (Forced Vital Capacity) in pulmonary fibrosis patients.\n"
    new_section += "Tortuosity and airway volume are the primary predictors.\n"
    
    new_section += "\n"
    
    # Replace section
    new_content = content[:start_idx] + new_section + content[end_idx:]
    
    # Write updated report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"    Report updated")


def recalculate_dataset(dataset_dir, backup=True):
    """
    Recalculate fibrosis scores for entire dataset
    
    Args:
        dataset_dir: Path to results directory (e.g., results_OSIC)
        backup: Whether to backup existing results
    
    Returns:
        DataFrame with summary
    """
    dataset_dir = Path(dataset_dir)
    dataset_name = dataset_dir.name
    
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}")
    
    # Find all patient directories
    patient_dirs = [d for d in dataset_dir.iterdir() 
                   if d.is_dir() and (d / "step4_analysis").exists()]
    
    print(f"\nFound {len(patient_dirs)} patient directories")
    
    results = []
    successful = 0
    failed = 0
    
    for patient_dir in sorted(patient_dirs):
        try:
            result = recalculate_patient_fibrosis(patient_dir, backup=backup)
            if result:
                results.append(result)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  ERROR: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"DATASET SUMMARY: {dataset_name}")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if results:
        df = pd.DataFrame(results)
        
        # Save summary
        summary_path = dataset_dir / f"RECALCULATED_FIBROSIS_SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")
        
        # Statistics
        print(f"\nScore Statistics:")
        print(f"  Mean: {df['score'].mean():.1f}")
        print(f"  Median: {df['score'].median():.1f}")
        print(f"  Min: {df['score'].min():.1f}")
        print(f"  Max: {df['score'].max():.1f}")
        
        print(f"\nSeverity Distribution:")
        print(df['stage'].value_counts().to_string())
    
    return results


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print(" "*15 + "FIBROSIS SCORE RECALCULATION")
    print(" "*10 + "WITH VALIDATED WEIGHTS (FVC-based)")
    print("="*80)
    
    print("\nVALIDATED WEIGHTS:")
    print("  [VALIDATED] Tortuosity: 0.50 (r=-0.267, p<0.001)")
    print("  [VALIDATED] Airway Volume: 0.30 (r=+0.245, p<0.001)")
    print("  [REDUCED] P/C Ratio: 0.05 (r=-0.062, NS - REDUCED)")
    print("    Symmetry: 0.10 (clinical relevance)")
    print("    Generation Coverage: 0.05")
    print("  [REMOVED] Tapering: 0.00 (REMOVED)")
    
    # Configuration
    DATASETS = [
        r"X:\Francesca Saglimbeni\tesi\results\results_OSIC",
        r"X:\Francesca Saglimbeni\tesi\results\results_CARVE14"
    ]
    
    BACKUP = True  # Backup existing results before overwriting
    
    print(f"\nDatasets to process: {len(DATASETS)}")
    print(f"Backup enabled: {BACKUP}")
    
    input("\nPress ENTER to start recalculation...")
    
    all_results = {}
    
    for dataset_path in DATASETS:
        if not os.path.exists(dataset_path):
            print(f"\nWARNING: Dataset not found: {dataset_path}")
            continue
        
        results = recalculate_dataset(dataset_path, backup=BACKUP)
        all_results[Path(dataset_path).name] = results
    
    # Final summary
    print("\n" + "="*80)
    print(" "*20 + "RECALCULATION COMPLETE")
    print("="*80)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        print(f"  Patients processed: {len(results)}")
        if results:
            df = pd.DataFrame(results)
            print(f"  Mean score: {df['score'].mean():.1f}/100")
            print(f"  Score range: {df['score'].min():.1f} - {df['score'].max():.1f}")
    
    print("\n" + "="*80)
    print("\nNEXT STEPS:")
    print("1. Review updated step6_fibrosis_assessment folders")
    print("2. Check updated COMPLETE_ANALYSIS_REPORT.txt files")
    print("3. Re-run correlation analysis with new scores")
    print("4. Compare old vs new scores to see impact of validated weights")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()