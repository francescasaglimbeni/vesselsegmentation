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

# Hounsfield Unit thresholds (standard values from literature)
HU_THRESHOLDS = {
    'lung_window': (-1024, -200),      # General lung tissue
    'ground_glass': (-700, -500),      # Ground glass opacity (GGO)
    'honeycombing': (-900, -700),      # Honeycombing pattern
    'consolidation': (-100, 100),      # Consolidation
    'emphysema': (-1024, -950)         # Emphysema
}


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
    Compute comprehensive parenchymal metrics for IPF assessment.
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
        """Compute all parenchymal metrics"""
        print("\n" + "="*60)
        print("COMPUTING PARENCHYMAL METRICS")
        print("="*60)
        
        self.compute_density_metrics()
        self.compute_pattern_percentages()
        self.compute_histogram_features()
        self.compute_texture_features()
        self.compute_spatial_distribution()
        
        print("\n" + "="*60)
        print("PARENCHYMAL METRICS COMPLETE")
        print("="*60)
        
        return self.metrics
    
    
    def compute_density_metrics(self):
        """Compute basic density metrics in Hounsfield Units"""
        if self.verbose:
            print("\n[1/5] Computing Density Metrics (HU)...")
        
        lung_hu = self.lung_hu
        
        metrics = {
            'mean_lung_density_HU': float(np.mean(lung_hu)),
            'median_lung_density_HU': float(np.median(lung_hu)),
            'std_lung_density_HU': float(np.std(lung_hu)),
            'min_lung_density_HU': float(np.min(lung_hu)),
            'max_lung_density_HU': float(np.max(lung_hu)),
            'q25_lung_density_HU': float(np.percentile(lung_hu, 25)),
            'q75_lung_density_HU': float(np.percentile(lung_hu, 75))
        }
        
        if self.verbose:
            print(f"    Mean lung density: {metrics['mean_lung_density_HU']:.1f} HU")
            print(f"    Median: {metrics['median_lung_density_HU']:.1f} HU")
            print(f"    Std: {metrics['std_lung_density_HU']:.1f} HU")
            print(f"    Range: [{metrics['min_lung_density_HU']:.0f}, {metrics['max_lung_density_HU']:.0f}] HU")
        
        self.metrics.update(metrics)
    
    
    def compute_pattern_percentages(self):
        """
        Compute percentages of different parenchymal patterns.
        Key metrics for IPF: Ground glass opacity, Honeycombing, Emphysema
        """
        if self.verbose:
            print("\n[2/5] Computing Pattern Percentages...")
        
        lung_hu = self.lung_hu
        total_voxels = len(lung_hu)
        
        # Ground Glass Opacity (GGO): -700 to -500 HU
        ggo_voxels = np.sum((lung_hu >= HU_THRESHOLDS['ground_glass'][0]) & 
                            (lung_hu <= HU_THRESHOLDS['ground_glass'][1]))
        ggo_percent = 100.0 * ggo_voxels / total_voxels
        
        # Honeycombing: -900 to -700 HU (air-filled cysts)
        honey_voxels = np.sum((lung_hu >= HU_THRESHOLDS['honeycombing'][0]) & 
                              (lung_hu <= HU_THRESHOLDS['honeycombing'][1]))
        honey_percent = 100.0 * honey_voxels / total_voxels
        
        # Consolidation: -100 to +100 HU
        consol_voxels = np.sum((lung_hu >= HU_THRESHOLDS['consolidation'][0]) & 
                               (lung_hu <= HU_THRESHOLDS['consolidation'][1]))
        consol_percent = 100.0 * consol_voxels / total_voxels
        
        # Emphysema: < -950 HU
        emphysema_voxels = np.sum(lung_hu < HU_THRESHOLDS['emphysema'][1])
        emphysema_percent = 100.0 * emphysema_voxels / total_voxels
        
        # Normal lung tissue: -900 to -500 HU
        normal_voxels = np.sum((lung_hu >= -900) & (lung_hu <= -500))
        normal_percent = 100.0 * normal_voxels / total_voxels
        
        metrics = {
            'percent_ground_glass_opacity': ggo_percent,
            'percent_honeycombing': honey_percent,
            'percent_consolidation': consol_percent,
            'percent_emphysema': emphysema_percent,
            'percent_normal_lung': normal_percent,
            
            # Fibrosis composite score (GGO + Honeycombing + Consolidation)
            'percent_fibrotic_patterns': ggo_percent + honey_percent + consol_percent
        }
        
        if self.verbose:
            print(f"    Ground Glass Opacity (GGO): {ggo_percent:.2f}%")
            print(f"    Honeycombing: {honey_percent:.2f}%")
            print(f"    Consolidation: {consol_percent:.2f}%")
            print(f"    Emphysema: {emphysema_percent:.2f}%")
            print(f"    Normal lung: {normal_percent:.2f}%")
            print(f"    Total fibrotic patterns: {metrics['percent_fibrotic_patterns']:.2f}%")
        
        self.metrics.update(metrics)
    
    
    def compute_histogram_features(self):
        """Compute histogram-based features"""
        if self.verbose:
            print("\n[3/5] Computing Histogram Features...")
        
        lung_hu = self.lung_hu
        
        # Histogram statistics
        hist, bin_edges = np.histogram(lung_hu, bins=100, range=(-1024, 100))
        hist_normalized = hist / np.sum(hist)
        
        # Entropy (measure of heterogeneity)
        hist_nonzero = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        
        # Higher order statistics
        skewness = skew(lung_hu)
        kurt = kurtosis(lung_hu)
        
        # Interquartile range
        iqr = np.percentile(lung_hu, 75) - np.percentile(lung_hu, 25)
        
        metrics = {
            'histogram_entropy': float(entropy),
            'histogram_skewness': float(skewness),
            'histogram_kurtosis': float(kurt),
            'histogram_iqr': float(iqr)
        }
        
        if self.verbose:
            print(f"    Entropy: {entropy:.3f}")
            print(f"    Skewness: {skewness:.3f}")
            print(f"    Kurtosis: {kurt:.3f}")
            print(f"    IQR: {iqr:.1f} HU")
        
        self.metrics.update(metrics)
    
    
    def compute_texture_features(self):
        """
        Compute texture features (radiomics).
        Using robust local statistics instead of GLCM (which is unstable for CT).
        """
        if self.verbose:
            print("\n[4/5] Computing Texture Features (Radiomics)...")
        
        # Select central 20% of slices for texture analysis
        z_center = self.ct_array.shape[0] // 2
        z_range = max(1, self.ct_array.shape[0] // 5)
        z_start = z_center - z_range // 2
        z_end = z_center + z_range // 2
        
        texture_values = {
            'local_std': [],
            'local_range': [],
            'local_mean_gradient': []
        }
        
        for z in range(z_start, z_end):
            if z < 0 or z >= self.ct_array.shape[0]:
                continue
            
            slice_img = self.ct_array[z, :, :]
            slice_mask = self.lung_mask[z, :, :]
            
            if np.sum(slice_mask) < 100:  # Skip slices with little lung tissue
                continue
            
            # Local standard deviation (texture roughness)
            local_std = ndimage.generic_filter(
                slice_img.astype(float),
                np.std,
                size=5,
                mode='constant',
                cval=0
            )
            texture_values['local_std'].append(np.mean(local_std[slice_mask > 0]))
            
            # Local range (max - min in local window)
            local_max = ndimage.maximum_filter(slice_img, size=5)
            local_min = ndimage.minimum_filter(slice_img, size=5)
            local_range = local_max - local_min
            texture_values['local_range'].append(np.mean(local_range[slice_mask > 0]))
            
            # Gradient magnitude (edge strength)
            from scipy.ndimage import sobel
            grad_x = sobel(slice_img.astype(float), axis=0)
            grad_y = sobel(slice_img.astype(float), axis=1)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            texture_values['local_mean_gradient'].append(np.mean(grad_mag[slice_mask > 0]))
        
        # Aggregate texture features
        metrics = {}
        for key, values in texture_values.items():
            if len(values) > 0:
                metrics[f'texture_{key}_mean'] = float(np.mean(values))
                metrics[f'texture_{key}_std'] = float(np.std(values))
            else:
                metrics[f'texture_{key}_mean'] = np.nan
                metrics[f'texture_{key}_std'] = np.nan
        
        if self.verbose:
            print(f"    Local Std Dev: {metrics.get('texture_local_std_mean', np.nan):.1f} HU")
            print(f"    Local Range: {metrics.get('texture_local_range_mean', np.nan):.1f} HU")
            print(f"    Mean Gradient: {metrics.get('texture_local_mean_gradient_mean', np.nan):.1f}")
        
        self.metrics.update(metrics)
    
    
    def compute_spatial_distribution(self):
        """Compute spatial distribution of density patterns"""
        if self.verbose:
            print("\n[5/5] Computing Spatial Distribution...")
        
        # Divide lungs into upper/middle/lower zones
        z_max = self.ct_array.shape[0]
        z_upper = z_max // 3
        z_middle = 2 * z_max // 3
        
        zones = {
            'upper': (0, z_upper),
            'middle': (z_upper, z_middle),
            'lower': (z_middle, z_max)
        }
        
        metrics = {}
        
        for zone_name, (z_start, z_end) in zones.items():
            zone_mask = self.lung_mask[z_start:z_end, :, :]
            zone_hu = self.ct_array[z_start:z_end, :, :][zone_mask > 0]
            
            if len(zone_hu) == 0:
                metrics[f'mean_density_{zone_name}_zone_HU'] = np.nan
                metrics[f'percent_fibrotic_{zone_name}_zone'] = np.nan
                continue
            
            # Mean density in this zone
            metrics[f'mean_density_{zone_name}_zone_HU'] = float(np.mean(zone_hu))
            
            # Percent fibrotic patterns in this zone
            ggo = np.sum((zone_hu >= -700) & (zone_hu <= -500))
            honey = np.sum((zone_hu >= -900) & (zone_hu <= -700))
            consol = np.sum((zone_hu >= -100) & (zone_hu <= 100))
            fibrotic_percent = 100.0 * (ggo + honey + consol) / len(zone_hu)
            metrics[f'percent_fibrotic_{zone_name}_zone'] = float(fibrotic_percent)
        
        # Basal predominance index (lower/upper ratio)
        if not np.isnan(metrics.get('percent_fibrotic_lower_zone', np.nan)) and \
           not np.isnan(metrics.get('percent_fibrotic_upper_zone', np.nan)):
            metrics['basal_predominance_index'] = (
                metrics['percent_fibrotic_lower_zone'] / 
                (metrics['percent_fibrotic_upper_zone'] + 1e-6)
            )
        else:
            metrics['basal_predominance_index'] = np.nan
        
        if self.verbose:
            print(f"    Upper zone: {metrics.get('percent_fibrotic_upper_zone', np.nan):.2f}% fibrotic")
            print(f"    Middle zone: {metrics.get('percent_fibrotic_middle_zone', np.nan):.2f}% fibrotic")
            print(f"    Lower zone: {metrics.get('percent_fibrotic_lower_zone', np.nan):.2f}% fibrotic")
            print(f"    Basal predominance: {metrics.get('basal_predominance_index', np.nan):.2f}")
        
        self.metrics.update(metrics)


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
    
    # Step 4: Compute parenchymal metrics
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
        
        f.write("DENSITY METRICS (Hounsfield Units)\n")
        f.write("-"*80 + "\n")
        f.write(f"  Mean lung density: {metrics.get('mean_lung_density_HU', np.nan):.1f} HU\n")
        f.write(f"  Median: {metrics.get('median_lung_density_HU', np.nan):.1f} HU\n")
        f.write(f"  Std deviation: {metrics.get('std_lung_density_HU', np.nan):.1f} HU\n\n")
        
        f.write("PARENCHYMAL PATTERNS\n")
        f.write("-"*80 + "\n")
        f.write(f"  Ground Glass Opacity: {metrics.get('percent_ground_glass_opacity', np.nan):.2f}%\n")
        f.write(f"  Honeycombing: {metrics.get('percent_honeycombing', np.nan):.2f}%\n")
        f.write(f"  Consolidation: {metrics.get('percent_consolidation', np.nan):.2f}%\n")
        f.write(f"  Emphysema: {metrics.get('percent_emphysema', np.nan):.2f}%\n")
        f.write(f"  Normal lung: {metrics.get('percent_normal_lung', np.nan):.2f}%\n")
        f.write(f"  Total fibrotic: {metrics.get('percent_fibrotic_patterns', np.nan):.2f}%\n\n")
        
        f.write("SPATIAL DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        f.write(f"  Upper zone fibrotic: {metrics.get('percent_fibrotic_upper_zone', np.nan):.2f}%\n")
        f.write(f"  Middle zone fibrotic: {metrics.get('percent_fibrotic_middle_zone', np.nan):.2f}%\n")
        f.write(f"  Lower zone fibrotic: {metrics.get('percent_fibrotic_lower_zone', np.nan):.2f}%\n")
        f.write(f"  Basal predominance: {metrics.get('basal_predominance_index', np.nan):.2f}\n\n")
        
        f.write("TEXTURE FEATURES (Robust Statistics)\n")
        f.write("-"*80 + "\n")
        f.write(f"  Local Std Dev: {metrics.get('texture_local_std_mean', np.nan):.1f} HU\n")
        f.write(f"  Local Range: {metrics.get('texture_local_range_mean', np.nan):.1f} HU\n")
        f.write(f"  Mean Gradient: {metrics.get('texture_local_mean_gradient_mean', np.nan):.1f}\n")
        f.write(f"  Histogram Entropy: {metrics.get('histogram_entropy', np.nan):.3f}\n\n")
        
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
