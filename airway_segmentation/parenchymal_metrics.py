"""
Parenchymal Metrics Computation Module
======================================

Computes lung parenchymal metrics for IPF assessment during the airway pipeline.
Only computes the 2 metrics actually used in correlation analysis:
    1. Mean Lung Density (HU)
    2. Histogram Entropy

Integrated into main_pipeline.py to compute all metrics in a single run.
"""

import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import TotalSegmentator, but make it optional
try:
    from totalsegmentator.python_api import totalsegmentator
    TOTALSEGMENTATOR_AVAILABLE = True
except ImportError:
    TOTALSEGMENTATOR_AVAILABLE = False
    print("Warning: TotalSegmentator not available for parenchymal metrics")


# ============================================================
# LUNG SEGMENTATION
# ============================================================

def segment_lungs_totalsegmentator(ct_path, output_dir, fast=True):
    """
    Segment lungs using TotalSegmentator with 'total' task.
    Uses roi_subset to only segment lung lobes for efficiency.
    
    Args:
        ct_path: Path to MHD file or SimpleITK image
        output_dir: Directory to save segmentation
        fast: Use fast mode (default True)
    
    Returns:
        lung_mask: Binary mask of lung regions (3D numpy array)
    """
    if not TOTALSEGMENTATOR_AVAILABLE:
        raise ImportError("TotalSegmentator is not installed. Cannot segment lungs.")
    
    print("  Segmenting lungs with TotalSegmentator...")
    
    # Convert to NIfTI for TotalSegmentator if needed
    if isinstance(ct_path, (str, Path)):
        ct_image = sitk.ReadImage(str(ct_path))
    else:
        ct_image = ct_path  # Already a SimpleITK image
    
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
        if self.verbose:
            print("\n" + "="*60)
            print("COMPUTING PARENCHYMAL METRICS")
            print("="*60)
        
        self.compute_density_metrics()
        self.compute_histogram_features()
        
        if self.verbose:
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
# INTEGRATION FUNCTION FOR PIPELINE
# ============================================================

def integrate_parenchymal_metrics(mhd_path, output_dir, fast_segmentation=True, verbose=True):
    """
    Compute parenchymal metrics and integrate into pipeline.
    This function is called from main_pipeline.py.
    
    Args:
        mhd_path: Path to original MHD CT scan
        output_dir: Directory to save results (step5_parenchymal_metrics)
        fast_segmentation: Use fast mode for TotalSegmentator
        verbose: Print detailed info
    
    Returns:
        Dictionary with parenchymal metrics
    """
    
    if verbose:
        print("\n" + "="*80)
        print("STEP 5: PARENCHYMAL METRICS COMPUTATION")
        print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load CT scan
        if verbose:
            print(f"\nLoading CT scan from: {mhd_path}")
        
        ct_image = sitk.ReadImage(str(mhd_path))
        ct_array = sitk.GetArrayFromImage(ct_image)
        spacing = ct_image.GetSpacing()
        
        if verbose:
            print(f"  Shape: {ct_array.shape}")
            print(f"  Spacing: {spacing} mm")
            print(f"  HU range: [{ct_array.min():.0f}, {ct_array.max():.0f}]")
        
        # Step 2: Segment lungs with TotalSegmentator
        if verbose:
            print(f"\nSegmenting lungs with TotalSegmentator...")
        
        segmentation_dir = output_dir / "segmentation_temp"
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        
        lung_mask = segment_lungs_totalsegmentator(ct_image, segmentation_dir, fast=fast_segmentation)
        
        # Step 3: Compute parenchymal metrics
        if verbose:
            print(f"\nComputing parenchymal metrics...")
        
        computer = ParenchymalMetricsComputer(ct_array, spacing, lung_mask, verbose=verbose)
        metrics = computer.compute_all_metrics()
        
        # Step 4: Save results
        
        # Save metrics JSON
        metrics_json_path = output_dir / "parenchymal_metrics.json"
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        if verbose:
            print(f"\n✓ Metrics saved to: {metrics_json_path}")
        
        # Save lung mask (optional, for visualization)
        mask_path = output_dir / "lung_mask.nii.gz"
        mask_sitk = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
        mask_sitk.CopyInformation(ct_image)
        sitk.WriteImage(mask_sitk, str(mask_path))
        
        if verbose:
            print(f"✓ Lung mask saved to: {mask_path}")
        
        # Generate summary report
        report_path = output_dir / "parenchymal_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PARENCHYMAL METRICS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CT scan: {mhd_path}\n\n")
            
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
        
        if verbose:
            print(f"✓ Report saved to: {report_path}")
            print(f"\n✓ Parenchymal metrics computation complete")
        
        return metrics
    
    except Exception as e:
        if verbose:
            print(f"\n⚠ Warning: Could not compute parenchymal metrics: {e}")
            import traceback
            traceback.print_exc()
        
        # Return None if computation fails
        return None


# ============================================================
# STANDALONE EXECUTION (for testing)
# ============================================================

def main():
    """Standalone execution for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute parenchymal metrics')
    parser.add_argument('mhd_path', type=str, help='Path to MHD file')
    parser.add_argument('--output', type=str, default='parenchymal_output', 
                        help='Output directory')
    parser.add_argument('--fast', action='store_true', 
                        help='Use fast segmentation mode')
    
    args = parser.parse_args()
    
    metrics = integrate_parenchymal_metrics(
        args.mhd_path,
        args.output,
        fast_segmentation=args.fast,
        verbose=True
    )
    
    if metrics:
        print("\n" + "="*80)
        print("METRICS COMPUTED SUCCESSFULLY")
        print("="*80)
        print(f"\nMean Lung Density: {metrics['mean_lung_density_HU']:.1f} HU")
        print(f"Histogram Entropy: {metrics['histogram_entropy']:.3f}")
    else:
        print("\n❌ Failed to compute metrics")


if __name__ == "__main__":
    main()
