import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from datetime import datetime
from scipy import ndimage
from scipy.stats import skew, kurtosis
from skimage import feature, filters
from totalsegmentator.python_api import totalsegmentator
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\results\results_OSIC")
VALIDATION_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\air_val\OSIC_validation.csv")
OSIC_DATA_DIR = Path(r"X:\Francesca Saglimbeni\tesi\datasets\OSIC_correct")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_ct_scan_path(case_name):
    """Get CT scan path from OSIC_correct directory"""
    # CT scans are stored as case_name.mhd + case_name.raw
    ct_path = OSIC_DATA_DIR / f"{case_name}.mhd"
    
    if ct_path.exists():
        return ct_path
    
    # Try alternative: might have _gaussian suffix or other variations
    for mhd_file in OSIC_DATA_DIR.glob(f"{case_name}*.mhd"):
        return mhd_file
    
    return None


def load_ct_scan(ct_path):
    """Load CT scan from .mhd file"""
    if not ct_path.exists():
        raise FileNotFoundError(f"CT scan not found: {ct_path}")
    
    image = sitk.ReadImage(str(ct_path))
    array = sitk.GetArrayFromImage(image)  # Shape: (Z, Y, X)
    spacing = image.GetSpacing()  # (x, y, z)
    
    return array, spacing, image


# ============================================================
# LUNG SEGMENTATION
# ============================================================

def segment_lungs_totalsegmentator(ct_path, output_dir, fast=True):
    """
    Segment lungs using TotalSegmentator with 'total' task (same as airway pipeline).
    Uses roi_subset to only segment lung lobes for efficiency.
    
    Args:
        ct_path: Path to MHD file
        output_dir: Directory to save segmentation
        fast: Use fast mode (default True)
    
    Returns:
        lung_mask: Binary mask of lung regions (3D numpy array)
    """
    print("  Segmenting lungs with TotalSegmentator...")
    
    # Convert MHD to NIfTI for TotalSegmentator
    ct_image = sitk.ReadImage(str(ct_path))
    nifti_path = output_dir / "ct_temp.nii.gz"
    sitk.WriteImage(ct_image, str(nifti_path))
    
    # Run TotalSegmentator with 'total' task, segmenting only lung lobes
    print("    Running TotalSegmentator (task='total', roi_subset=lungs)...")
    totalsegmentator(
        str(nifti_path),
        str(output_dir),
        task="total",
        roi_subset=["lung_upper_lobe_left", "lung_lower_lobe_left", 
                    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        fast=fast,
        ml=False  # Save as separate files
    )
    
    # Load and combine the 5 lung lobes
    lung_parts = [
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz"
    ]
    
    lung_mask = None
    
    for part_name in lung_parts:
        part_path = output_dir / part_name
        if part_path.exists():
            part_image = sitk.ReadImage(str(part_path))
            part_array = sitk.GetArrayFromImage(part_image)
            
            if lung_mask is None:
                lung_mask = (part_array > 0).astype(np.uint8)
            else:
                lung_mask = np.logical_or(lung_mask, part_array > 0).astype(np.uint8)
            
            # Cleanup this lobe file
            part_path.unlink(missing_ok=True)
    
    if lung_mask is None:
        raise RuntimeError("TotalSegmentator did not generate lung lobe masks")
    
    print(f"    Lung volume: {np.sum(lung_mask)} voxels")
    
    # Cleanup temporary nifti
    nifti_path.unlink(missing_ok=True)
    
    return lung_mask


# ============================================================
# PARENCHYMAL METRICS COMPUTATION
# ============================================================

class ParenchymalMetricsComputer:
    """
    Compute parenchymal metrics for IPF assessment.
    Only computes the 2 metrics actually used in analysis:
    1. Mean Lung Density (HU)
    2. Histogram Entropy
    """
    
    def __init__(self, ct_array, spacing, lung_mask, verbose=True):
        """
        Args:
            ct_array: CT image in Hounsfield Units (Z, Y, X)
            spacing: Voxel spacing (x, y, z) in mm
            lung_mask: Binary lung segmentation mask
            verbose: Print detailed info
        """
        self.ct_array = ct_array
        self.spacing = spacing
        self.lung_mask = lung_mask
        self.verbose = verbose
        
        # Extract lung region only
        self.lung_hu = ct_array[lung_mask > 0]
        
        # Results
        self.metrics = {}
    
    
    def compute_all_metrics(self):
        """Compute the 2 key parenchymal metrics"""
        print("\n" + "="*60)
        print("COMPUTING PARENCHYMAL METRICS")
        print("="*60)
        
        self.compute_density_metrics()
        self.compute_histogram_features()
        
        print("\n" + "="*60)
        print("PARENCHYMAL METRICS COMPLETE")
        print("="*60)
        
        return self.metrics
    
    
    def compute_density_metrics(self):
        """Compute Mean Lung Density in Hounsfield Units"""
        if self.verbose:
            print("\n[1/2] Computing Mean Lung Density (HU)...")
        
        lung_hu = self.lung_hu
        
        # Formula: Mean Density = (1/N) * Σ(HU_i)
        # where N = total voxels in lung mask
        #       HU_i = Hounsfield Unit value of voxel i
        
        mean_density = float(np.mean(lung_hu))
        
        self.metrics['mean_lung_density_HU'] = mean_density
        
        if self.verbose:
            print(f"    Mean lung density: {mean_density:.1f} HU")
            print(f"    Formula: (1/{len(lung_hu)}) * Σ(HU_i)")
    
    
    def compute_histogram_features(self):
        """Compute Histogram Entropy (Shannon entropy)"""
        if self.verbose:
            print("\n[2/2] Computing Histogram Entropy...")
        
        lung_hu = self.lung_hu
        
        # Create histogram of HU values (100 bins from -1024 to 100 HU)
        hist, bin_edges = np.histogram(lung_hu, bins=100, range=(-1024, 100))
        
        # Normalize histogram to get probability distribution
        hist_normalized = hist / np.sum(hist)
        
        # Remove zero probabilities for entropy calculation
        hist_nonzero = hist_normalized[hist_normalized > 0]
        
        # Shannon entropy formula: Entropy = -Σ p_j * log₂(p_j)
        # where p_j = probability in bin j
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        
        self.metrics['histogram_entropy'] = float(entropy)
        
        if self.verbose:
            print(f"    Histogram entropy: {entropy:.3f}")
            print(f"    Formula: -Σ p_j * log₂(p_j)")
            print(f"    Bins: 100 from -1024 to 100 HU")


# ============================================================
# PROCESSING PIPELINE
# ============================================================

def process_single_case(case_name, case_dir, output_subdir="step5_parenchymal_metrics"):
    """Process a single case to compute parenchymal metrics"""
    print(f"\n{'='*80}")
    print(f"Processing: {case_name}")
    print(f"{'='*80}")
    
    # Step 1: Get original CT scan path
    ct_path = get_ct_scan_path(case_name)
    if ct_path is None:
        print(f"  ERROR: Could not find CT scan in {OSIC_DATA_DIR}")
        return None
    
    print(f"  CT scan: {ct_path}")
    
    if not ct_path.exists():
        print(f"  ERROR: CT scan file not found: {ct_path}")
        return None
    
    # Step 2: Load CT scan
    print(f"\nLoading CT scan...")
    try:
        ct_array, spacing, sitk_image = load_ct_scan(ct_path)
        print(f"  Shape: {ct_array.shape}")
        print(f"  Spacing: {spacing} mm")
        print(f"  HU range: [{ct_array.min():.0f}, {ct_array.max():.0f}]")
    except Exception as e:
        print(f"  ERROR: Failed to load CT scan: {e}")
        return None
    
    # Step 3: Segment lungs with TotalSegmentator
    print(f"\nSegmenting lungs with TotalSegmentator...")
    segmentation_dir = case_dir / output_subdir / "segmentation_temp"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        lung_mask = segment_lungs_totalsegmentator(ct_path, segmentation_dir, fast=True)
    except Exception as e:
        print(f"  ERROR: Lung segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 4: Compute parenchymal metrics (ONLY 2 metrics)
    try:
        computer = ParenchymalMetricsComputer(ct_array, spacing, lung_mask, verbose=True)
        metrics = computer.compute_all_metrics()
    except Exception as e:
        print(f"  ERROR: Metrics computation failed: {e}")
        return None
    
    # Step 5: Save results
    output_dir = case_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    metrics_json_path = output_dir / "parenchymal_metrics.json"
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_json_path}")
    
    # Save lung mask (optional, for visualization)
    mask_path = output_dir / "lung_mask.nii.gz"
    mask_sitk = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
    mask_sitk.CopyInformation(sitk_image)
    sitk.WriteImage(mask_sitk, str(mask_path))
    print(f"✓ Lung mask saved to: {mask_path}")
    
    # Generate summary report
    report_path = output_dir / "parenchymal_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PARENCHYMAL METRICS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Case: {case_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CT scan: {ct_path}\n\n")
        
        f.write("PARENCHYMAL METRICS (2 KEY METRICS)\n")
        f.write("-"*80 + "\n")
        f.write(f"  1. Mean Lung Density: {metrics.get('mean_lung_density_HU', np.nan):.1f} HU\n")
        f.write(f"     Formula: (1/N) * Σ(HU_i) where N = {len(computer.lung_hu)} voxels\n\n")
        
        f.write(f"  2. Histogram Entropy: {metrics.get('histogram_entropy', np.nan):.3f}\n")
        f.write(f"     Formula: -Σ p_j * log₂(p_j)\n")
        f.write(f"     Bins: 100 from -1024 to 100 HU\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-"*80 + "\n")
        f.write("  • Higher Mean Density = Denser lung tissue (fibrosis)\n")
        f.write("  • Higher Entropy = More heterogeneous tissue patterns\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ Report saved to: {report_path}")
    
    return metrics


def process_all_cases(case_filter='RELIABLE'):
    """Process all cases (or only RELIABLE ones)"""
    print("="*80)
    print("PARENCHYMAL METRICS COMPUTATION")
    print("="*80)
    print(f"\nResults directory: {RESULTS_ROOT}")
    print(f"Case filter: {case_filter}")
    print(f"Computing ONLY 2 metrics: Mean Lung Density & Histogram Entropy")
    
    # Load validation data to filter cases
    if case_filter == 'RELIABLE':
        validation = pd.read_csv(VALIDATION_CSV)
        reliable_cases = validation[validation['status'] == 'RELIABLE']['case'].tolist()
        print(f"\nProcessing {len(reliable_cases)} RELIABLE cases")
    else:
        # Process all cases
        reliable_cases = [d.name for d in RESULTS_ROOT.iterdir() if d.is_dir()]
        print(f"\nProcessing all {len(reliable_cases)} cases")
    
    # Process each case
    results = []
    failed_cases = []
    
    for i, case_name in enumerate(reliable_cases, 1):
        print(f"\n\n[{i}/{len(reliable_cases)}] Processing {case_name}")
        
        case_dir = RESULTS_ROOT / case_name
        
        if not case_dir.exists():
            print(f"  WARNING: Case directory not found: {case_dir}")
            failed_cases.append((case_name, "Directory not found"))
            continue
        
        try:
            metrics = process_single_case(case_name, case_dir)
            
            if metrics is not None:
                results.append({
                    'case': case_name,
                    'status': 'SUCCESS',
                    **metrics
                })
            else:
                failed_cases.append((case_name, "Processing returned None"))
        
        except Exception as e:
            print(f"\n  ERROR: Failed to process {case_name}: {e}")
            failed_cases.append((case_name, str(e)))
    
    # Summary
    print("\n\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"\nTotal cases: {len(reliable_cases)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {len(failed_cases)}")
    
    if len(failed_cases) > 0:
        print(f"\nFailed cases:")
        for case_name, reason in failed_cases:
            print(f"  - {case_name}: {reason}")
    
    # Save summary CSV
    if len(results) > 0:
        summary_df = pd.DataFrame(results)
        summary_path = RESULTS_ROOT / f"PARENCHYMAL_METRICS_SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        
        # Print metrics summary
        print(f"\nMetrics Statistics:")
        print(f"  Mean Lung Density: {summary_df['mean_lung_density_HU'].mean():.1f} ± {summary_df['mean_lung_density_HU'].std():.1f} HU")
        print(f"  Histogram Entropy: {summary_df['histogram_entropy'].mean():.3f} ± {summary_df['histogram_entropy'].std():.3f}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80 + "\n")
    
    return results, failed_cases


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute parenchymal metrics for OSIC cases')
    parser.add_argument('--case', type=str, help='Process a specific case (optional)')
    parser.add_argument('--all', action='store_true', help='Process all cases (not just RELIABLE)')
    
    args = parser.parse_args()
    
    if args.case:
        # Process single case
        case_name = args.case
        case_dir = RESULTS_ROOT / case_name
        
        if not case_dir.exists():
            print(f"ERROR: Case directory not found: {case_dir}")
            sys.exit(1)
        
        process_single_case(case_name, case_dir)
    
    else:
        # Process all cases
        case_filter = 'ALL' if args.all else 'RELIABLE'
        process_all_cases(case_filter=case_filter)


if __name__ == "__main__":
    main()