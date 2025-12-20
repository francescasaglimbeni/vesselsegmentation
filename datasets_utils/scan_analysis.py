import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import json
from pathlib import Path


class CTScanQualityAnalyzer:
    """
    Analizza caratteristiche delle CT scan per determinare perchÃ© la pipeline
    funziona su alcuni dataset (CARVE14) ma fallisce su altri (OSIC).
    
    Focus: caratteristiche che influenzano la robustezza della segmentazione airways
    """
    
    def __init__(self, good_scans_dir, bad_scans_dir, output_dir="analysis_results"):
        """
        Args:
            good_scans_dir: Cartella con scan che danno buoni risultati
            bad_scans_dir: Cartella con scan problematiche
            output_dir: Dove salvare i risultati
        """
        self.good_scans_dir = Path(good_scans_dir)
        self.bad_scans_dir = Path(bad_scans_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.good_results = []
        self.bad_results = []
    
    def analyze_all_scans(self, pattern="*.mhd"):
        """Analizza tutte le scan nelle due cartelle"""
        print("\n" + "="*70)
        print(" "*20 + "CT SCAN QUALITY ANALYSIS")
        print("="*70)
        
        # Analizza scan "buone"
        print(f"\n[GOOD SCANS] Analyzing: {self.good_scans_dir}")
        good_files = list(self.good_scans_dir.glob(pattern))
        print(f"Found {len(good_files)} files")
        
        for scan_path in good_files:
            print(f"\n  Processing: {scan_path.name}")
            result = self.analyze_single_scan(str(scan_path), category="good")
            if result:
                self.good_results.append(result)
        
        # Analizza scan "cattive"
        print(f"\n[BAD SCANS] Analyzing: {self.bad_scans_dir}")
        bad_files = list(self.bad_scans_dir.glob(pattern))
        print(f"Found {len(bad_files)} files")
        
        for scan_path in bad_files:
            print(f"\n  Processing: {scan_path.name}")
            result = self.analyze_single_scan(str(scan_path), category="bad")
            if result:
                self.bad_results.append(result)
        
        # Genera report comparativo
        self.generate_comparative_report()
        self.plot_comparative_analysis()
        
        return self.good_results, self.bad_results
    
    def analyze_single_scan(self, scan_path, category="unknown"):
        """
        Analizza una singola scan CT per caratteristiche critiche
        """
        try:
            # Carica immagine
            img = sitk.ReadImage(scan_path)
            arr = sitk.GetArrayFromImage(img)
            spacing = img.GetSpacing()
            
            # Carica segmentazione airways (se esiste)
            seg_path = scan_path.replace('.mhd', '_airwayfull.nii.gz')
            if not os.path.exists(seg_path):
                # Prova varianti del nome
                seg_path = scan_path.replace('.mhd', '_airway.nii.gz')
            
            if os.path.exists(seg_path):
                seg_img = sitk.ReadImage(seg_path)
                seg_arr = sitk.GetArrayFromImage(seg_img)
                has_segmentation = True
            else:
                seg_arr = None
                has_segmentation = False
                print(f"    Warning: No segmentation found for {scan_path}")
            
            result = {
                'filename': os.path.basename(scan_path),
                'category': category,
                'has_segmentation': has_segmentation
            }
            
            # ============================================================
            # FEATURE 1: IMAGING CHARACTERISTICS (CT quality)
            # ============================================================
            
            # 1.1 Slice thickness (importante per continuitÃ  3D)
            result['slice_thickness_mm'] = spacing[2]
            
            # 1.2 In-plane resolution
            result['pixel_spacing_x_mm'] = spacing[0]
            result['pixel_spacing_y_mm'] = spacing[1]
            
            # 1.3 Matrix size
            result['matrix_z'] = arr.shape[0]
            result['matrix_y'] = arr.shape[1]
            result['matrix_x'] = arr.shape[2]
            
            # 1.4 HU range (per controllo calibrazione)
            result['hu_min'] = float(np.min(arr))
            result['hu_max'] = float(np.max(arr))
            result['hu_mean'] = float(np.mean(arr))
            result['hu_std'] = float(np.std(arr))
            
            # ============================================================
            # FEATURE 2: AIRWAY-SPECIFIC CHARACTERISTICS
            # ============================================================
            
            if has_segmentation:
                # 2.1 Volume totale airways
                airway_voxels = np.sum(seg_arr > 0)
                voxel_volume = spacing[0] * spacing[1] * spacing[2]
                result['airway_volume_mm3'] = airway_voxels * voxel_volume
                result['airway_voxels'] = int(airway_voxels)
                
                # 2.2 Extent verticale (trachea length)
                if airway_voxels > 0:
                    airway_coords = np.argwhere(seg_arr > 0)
                    z_min, z_max = np.min(airway_coords[:, 0]), np.max(airway_coords[:, 0])
                    result['airway_z_extent_slices'] = int(z_max - z_min)
                    result['airway_z_extent_mm'] = (z_max - z_min) * spacing[2]
                    result['airway_z_min'] = int(z_min)
                    result['airway_z_max'] = int(z_max)
                else:
                    result['airway_z_extent_slices'] = 0
                    result['airway_z_extent_mm'] = 0.0
                
                # 2.3 ConnettivitÃ  (numero di componenti connesse)
                labeled, num_components = label(seg_arr > 0)
                result['num_connected_components'] = int(num_components)
                
                # 2.4 Dimensione componente principale
                if num_components > 0:
                    component_sizes = []
                    for i in range(1, num_components + 1):
                        size = np.sum(labeled == i)
                        component_sizes.append(size)
                    component_sizes.sort(reverse=True)
                    
                    result['main_component_voxels'] = int(component_sizes[0])
                    result['main_component_percentage'] = (component_sizes[0] / airway_voxels * 100) if airway_voxels > 0 else 0
                    
                    if num_components > 1:
                        result['second_component_voxels'] = int(component_sizes[1])
                        result['fragmentation_ratio'] = component_sizes[1] / component_sizes[0] if component_sizes[0] > 0 else 0
                    else:
                        result['second_component_voxels'] = 0
                        result['fragmentation_ratio'] = 0.0
                
                # 2.5 Analisi slice-by-slice (per discontinuitÃ  trachea)
                slice_areas = []
                for z in range(seg_arr.shape[0]):
                    slice_2d = seg_arr[z, :, :]
                    area = np.sum(slice_2d > 0)
                    slice_areas.append(area)
                
                slice_areas = np.array(slice_areas)
                non_zero_slices = slice_areas > 0
                
                if np.sum(non_zero_slices) > 0:
                    # Conta "gaps" (slice vuoti tra slice pieni)
                    gaps = 0
                    in_airway = False
                    for area in slice_areas:
                        if area > 0:
                            in_airway = True
                        elif in_airway and area == 0:
                            gaps += 1
                            in_airway = False
                    
                    result['trachea_gaps'] = gaps
                    
                    # VariabilitÃ  area cross-section
                    active_areas = slice_areas[slice_areas > 0]
                    result['slice_area_mean'] = float(np.mean(active_areas))
                    result['slice_area_std'] = float(np.std(active_areas))
                    result['slice_area_cv'] = (np.std(active_areas) / np.mean(active_areas)) if np.mean(active_areas) > 0 else 0
                else:
                    result['trachea_gaps'] = 0
                    result['slice_area_mean'] = 0.0
                    result['slice_area_std'] = 0.0
                    result['slice_area_cv'] = 0.0
                
                # 2.6 Skeleton analysis (per complessitÃ  topologica)
                try:
                    binary_mask = (seg_arr > 0).astype(np.uint8)
                    skeleton = skeletonize(binary_mask)
                    skeleton_voxels = np.sum(skeleton > 0)
                    
                    result['skeleton_voxels'] = int(skeleton_voxels)
                    result['skeleton_to_volume_ratio'] = (skeleton_voxels / airway_voxels) if airway_voxels > 0 else 0
                    
                    # Conta endpoints (terminazioni)
                    from scipy.ndimage import convolve
                    kernel = np.ones((3, 3, 3))
                    kernel[1, 1, 1] = 0
                    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')
                    neighbor_count = neighbor_count * skeleton
                    endpoints = (neighbor_count == 1) & (skeleton > 0)
                    
                    result['skeleton_endpoints'] = int(np.sum(endpoints))
                    
                except Exception as e:
                    print(f"    Warning: Skeleton analysis failed: {e}")
                    result['skeleton_voxels'] = 0
                    result['skeleton_to_volume_ratio'] = 0
                    result['skeleton_endpoints'] = 0
            
            else:
                # Nessuna segmentazione disponibile
                result['airway_volume_mm3'] = 0.0
                result['airway_voxels'] = 0
                result['num_connected_components'] = 0
                result['trachea_gaps'] = 0
            
            # ============================================================
            # FEATURE 3: HU DISTRIBUTION IN AIRWAY REGION
            # ============================================================
            
            if has_segmentation and airway_voxels > 0:
                airway_hu = arr[seg_arr > 0]
                
                result['airway_hu_mean'] = float(np.mean(airway_hu))
                result['airway_hu_std'] = float(np.std(airway_hu))
                result['airway_hu_min'] = float(np.min(airway_hu))
                result['airway_hu_max'] = float(np.max(airway_hu))
                result['airway_hu_median'] = float(np.median(airway_hu))
                
                # Percentuali in range critici
                result['airway_hu_below_minus_950'] = float(np.sum(airway_hu < -950) / len(airway_hu) * 100)
                result['airway_hu_minus_950_to_minus_850'] = float(np.sum((airway_hu >= -950) & (airway_hu < -850)) / len(airway_hu) * 100)
                result['airway_hu_minus_850_to_minus_700'] = float(np.sum((airway_hu >= -850) & (airway_hu < -700)) / len(airway_hu) * 100)
                result['airway_hu_above_minus_700'] = float(np.sum(airway_hu >= -700) / len(airway_hu) * 100)
            
            # ============================================================
            # FEATURE 4: NOISE AND ARTIFACTS
            # ============================================================
            
            # 4.1 Noise estimation (std in una region omogenea - aria intorno al paziente)
            # Prendi gli angoli dell'immagine come reference (dovrebbe essere aria esterna)
            corner_region = arr[0:10, 0:10, 0:10]
            result['background_noise_std'] = float(np.std(corner_region))
            
            # 4.2 Dynamic range (spread tra tessuti)
            result['dynamic_range'] = float(result['hu_max'] - result['hu_min'])
            
            print(f"    âœ“ Analysis complete")
            
            return result
            
        except Exception as e:
            print(f"    âœ— Error analyzing {scan_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_comparative_report(self):
        """Genera report testuale comparativo"""
        print("\n" + "="*70)
        print(" "*20 + "COMPARATIVE ANALYSIS REPORT")
        print("="*70)
        
        if not self.good_results or not self.bad_results:
            print("\nNot enough data for comparison")
            return
        
        good_df = pd.DataFrame(self.good_results)
        bad_df = pd.DataFrame(self.bad_results)
        
        # Salva CSV
        good_df.to_csv(self.output_dir / "good_scans_features.csv", index=False)
        bad_df.to_csv(self.output_dir / "bad_scans_features.csv", index=False)
        print(f"\nâœ“ Saved CSVs: {self.output_dir}")
        
        # Report statistico
        report_path = self.output_dir / "comparative_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(" "*15 + "CT SCAN QUALITY COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Good scans analyzed: {len(self.good_results)}\n")
            f.write(f"Bad scans analyzed: {len(self.bad_results)}\n\n")
            
            f.write("="*70 + "\n")
            f.write("KEY DIFFERENCES\n")
            f.write("="*70 + "\n\n")
            
            # Confronta metriche chiave
            key_metrics = [
                ('slice_thickness_mm', 'Slice Thickness (mm)', False),
                ('pixel_spacing_x_mm', 'Pixel Spacing X (mm)', False),
                ('num_connected_components', 'Connected Components', True),
                ('main_component_percentage', 'Main Component %', False),
                ('trachea_gaps', 'Trachea Gaps', True),
                ('slice_area_cv', 'Slice Area Variability (CV)', True),
                ('airway_hu_mean', 'Airway HU Mean', False),
                ('airway_hu_above_minus_700', 'Airway HU > -700 (%)', True),
                ('skeleton_to_volume_ratio', 'Skeleton/Volume Ratio', False),
                ('fragmentation_ratio', 'Fragmentation Ratio', True),
            ]
            
            for metric, label, is_problem in key_metrics:
                if metric in good_df.columns and metric in bad_df.columns:
                    good_mean = good_df[metric].mean()
                    bad_mean = bad_df[metric].mean()
                    good_std = good_df[metric].std()
                    bad_std = bad_df[metric].std()
                    
                    diff_pct = abs(bad_mean - good_mean) / good_mean * 100 if good_mean != 0 else 0
                    
                    f.write(f"{label}:\n")
                    f.write(f"  Good scans: {good_mean:.3f} Â± {good_std:.3f}\n")
                    f.write(f"  Bad scans:  {bad_mean:.3f} Â± {bad_std:.3f}\n")
                    f.write(f"  Difference: {diff_pct:.1f}%\n")
                    
                    # Valutazione
                    if is_problem and bad_mean > good_mean * 1.5:
                        f.write(f"  âš ï¸ CRITICAL: Bad scans show {diff_pct:.0f}% higher {label}\n")
                    elif diff_pct > 30:
                        f.write(f"  âš ï¸ WARNING: Large difference ({diff_pct:.0f}%)\n")
                    
                    f.write("\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            
            # Raccomandazioni automatiche
            recommendations = []
            
            # Check slice thickness
            if bad_df['slice_thickness_mm'].mean() > good_df['slice_thickness_mm'].mean() * 1.3:
                recommendations.append(
                    "â€¢ SLICE THICKNESS: Bad scans have thicker slices â†’ may cause "
                    "discontinuities in skeleton. Consider interpolation or "
                    "adjusting skeletonization parameters."
                )
            
            # Check connected components
            if 'num_connected_components' in bad_df.columns and 'num_connected_components' in good_df.columns:
                if bad_df['num_connected_components'].mean() > good_df['num_connected_components'].mean() + 2:
                    recommendations.append(
                        "â€¢ FRAGMENTATION: Bad scans show more disconnected components â†’ "
                        "increase reconnection distance or improve trachea removal method."
                    )
            
            # Check HU distribution
            if 'airway_hu_above_minus_700' in bad_df.columns and 'airway_hu_above_minus_700' in good_df.columns:
                if bad_df['airway_hu_above_minus_700'].mean() > good_df['airway_hu_above_minus_700'].mean() * 2:
                    recommendations.append(
                        "â€¢ HU THRESHOLD: Bad scans have higher HU values in airways â†’ "
                        "adapt thresholding strategy (adaptive thresholds or multi-Otsu)."
                    )
            
            # Check gaps
            if 'trachea_gaps' in bad_df.columns and 'trachea_gaps' in good_df.columns:
                if bad_df['trachea_gaps'].mean() > 1:
                    recommendations.append(
                        "â€¢ TRACHEA GAPS: Bad scans show discontinuities â†’ "
                        "implement gap-filling preprocessing step."
                    )
            
            if recommendations:
                for rec in recommendations:
                    f.write(rec + "\n\n")
            else:
                f.write("No critical issues detected in the comparison.\n")
        
        print(f"âœ“ Report saved: {report_path}")
    
    def plot_comparative_analysis(self):
        """Genera grafici comparativi"""
        if not self.good_results or not self.bad_results:
            return
        
        good_df = pd.DataFrame(self.good_results)
        bad_df = pd.DataFrame(self.bad_results)
        
        # Helper function to safely plot if column exists
        def safe_boxplot(ax, column, title, ylabel):
            if column in good_df.columns and column in bad_df.columns:
                # Filter out NaN/zero values if needed
                good_data = good_df[column].dropna()
                bad_data = bad_df[column].dropna()
                
                if len(good_data) > 0 and len(bad_data) > 0:
                    ax.boxplot([good_data, bad_data], labels=['Good', 'Bad'])
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No data\nfor {column}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title + ' (No Data)')
            else:
                ax.text(0.5, 0.5, f'Column {column}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title + ' (N/A)')
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('CT Scan Quality: Good vs Bad Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Slice thickness
        safe_boxplot(axes[0, 0], 'slice_thickness_mm', 'Resolution: Slice Thickness', 'Slice Thickness (mm)')
        
        # Plot 2: Connected components
        safe_boxplot(axes[0, 1], 'num_connected_components', 'Fragmentation: Connected Components', 'Number of Components')
        
        # Plot 3: Main component percentage
        safe_boxplot(axes[0, 2], 'main_component_percentage', 'Main Component Size', 'Percentage (%)')
        
        # Plot 4: Trachea gaps
        safe_boxplot(axes[1, 0], 'trachea_gaps', 'Trachea Continuity: Gaps', 'Number of Gaps')
        
        # Plot 5: Slice area variability
        safe_boxplot(axes[1, 1], 'slice_area_cv', 'Cross-section Variability', 'Coefficient of Variation')
        
        # Plot 6: HU mean in airways
        safe_boxplot(axes[1, 2], 'airway_hu_mean', 'Airway HU Mean', 'HU')
        
        # Plot 7: HU distribution (problematic range)
        safe_boxplot(axes[2, 0], 'airway_hu_above_minus_700', 'Airway HU > -700 (Non-Air)', 'Percentage (%)')
        
        # Plot 8: Skeleton complexity
        safe_boxplot(axes[2, 1], 'skeleton_to_volume_ratio', 'Skeleton/Volume Ratio', 'Ratio')
        
        # Plot 9: Fragmentation ratio
        safe_boxplot(axes[2, 2], 'fragmentation_ratio', 'Fragmentation (2nd/1st Component)', 'Ratio')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / "comparative_plots.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"âœ“ Plots saved: {plot_path}")
        
        plt.show()


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Configura i path
    GOOD_SCANS_DIR = "datasets_utils/good_scans"  # CARVE14 o OSIC buone
    BAD_SCANS_DIR = "datasets_utils/bad_scans"    # OSIC problematiche
    OUTPUT_DIR = "scan_quality_analysis"
    
    # Crea analyzer
    analyzer = CTScanQualityAnalyzer(
        good_scans_dir=GOOD_SCANS_DIR,
        bad_scans_dir=BAD_SCANS_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Analizza tutte le scan
    good_results, bad_results = analyzer.analyze_all_scans(pattern="*.mhd")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: {OUTPUT_DIR}/")
    print("\nCheck these files:")
    print("  â€¢ good_scans_features.csv")
    print("  â€¢ bad_scans_features.csv")
    print("  â€¢ comparative_report.txt")
    print("  â€¢ comparative_plots.png")