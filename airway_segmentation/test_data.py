import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from skimage.filters import threshold_otsu
import json


class CTQualityAnalyzer:
    """
    Analizza qualità CT per identificare caratteristiche che causano problemi a TotalSegmentator
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = []
        
        # TotalSegmentator requirements (da documentazione)
        self.ts_requirements = {
            'min_slices': 400,
            'max_slice_thickness': 0.7,  # mm
            'min_xy_resolution': 0.5,    # mm
            'max_xy_resolution': 1.0,    # mm
            'expected_hu_range': (-1024, 3071),  # Standard CT range
            'min_fov': 300,  # mm (Field of View)
            'max_fov': 600   # mm
        }
    
    def analyze_ct(self, mhd_path):
        """
        Analisi completa di una singola CT
        """
        scan_name = Path(mhd_path).stem
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Analyzing: {scan_name}")
            print('='*80)
        
        metrics = {
            'scan_name': scan_name,
            'file_path': mhd_path,
            'load_success': False
        }
        
        try:
            # Carica immagine
            image = sitk.ReadImage(mhd_path)
            array = sitk.GetArrayFromImage(image)
            
            metrics['load_success'] = True
            
            # 1. BASIC GEOMETRY
            geometry = self._analyze_geometry(image, array)
            metrics.update(geometry)
            
            # 2. INTENSITY ANALYSIS
            intensity = self._analyze_intensity(array)
            metrics.update(intensity)
            
            # 3. NOISE AND ARTIFACTS
            quality = self._analyze_quality(array)
            metrics.update(quality)
            
            # 4. ANATOMICAL CONTENT
            anatomy = self._analyze_anatomy(array)
            metrics.update(anatomy)
            
            # 5. TOTALSEGMENTATOR COMPATIBILITY
            compatibility = self._check_totalsegmentator_compatibility(metrics)
            metrics.update(compatibility)
            
            # 6. QUALITY SCORE
            metrics['quality_score'] = self._compute_quality_score(metrics)
            
            if self.verbose:
                self._print_summary(metrics)
            
        except Exception as e:
            metrics['error'] = str(e)
            if self.verbose:
                print(f"❌ Error: {e}")
        
        self.results.append(metrics)
        return metrics
    
    def _analyze_geometry(self, image, array):
        """Analizza geometria della CT"""
        spacing = image.GetSpacing()  # (x, y, z)
        size = image.GetSize()        # (x, y, z)
        origin = image.GetOrigin()
        direction = image.GetDirection()
        
        # Array shape è (z, y, x)
        num_slices = array.shape[0]
        slice_height = array.shape[1]
        slice_width = array.shape[2]
        
        # Spacing
        pixel_spacing_x = spacing[0]
        pixel_spacing_y = spacing[1]
        slice_thickness = spacing[2]
        
        # Field of View (FOV)
        fov_x = pixel_spacing_x * slice_width
        fov_y = pixel_spacing_y * slice_height
        fov_z = slice_thickness * num_slices
        
        metrics = {
            'num_slices': num_slices,
            'slice_height': slice_height,
            'slice_width': slice_width,
            'pixel_spacing_x': pixel_spacing_x,
            'pixel_spacing_y': pixel_spacing_y,
            'slice_thickness': slice_thickness,
            'fov_x_mm': fov_x,
            'fov_y_mm': fov_y,
            'fov_z_mm': fov_z,
            'total_voxels': array.size,
            'direction_identity': np.allclose(direction, np.eye(3).flatten())
        }
        
        if self.verbose:
            print(f"\n📐 GEOMETRY:")
            print(f"  Dimensions: {slice_width} x {slice_height} x {num_slices} voxels")
            print(f"  Spacing: {pixel_spacing_x:.3f} x {pixel_spacing_y:.3f} x {slice_thickness:.3f} mm")
            print(f"  FOV: {fov_x:.1f} x {fov_y:.1f} x {fov_z:.1f} mm")
            print(f"  Slice thickness: {slice_thickness:.3f} mm")
        
        return metrics
    
    def _analyze_intensity(self, array):
        """Analizza intensità HU"""
        
        # Statistics base
        hu_min = float(array.min())
        hu_max = float(array.max())
        hu_mean = float(array.mean())
        hu_std = float(array.std())
        hu_median = float(np.median(array))
        
        # Percentili
        p01 = float(np.percentile(array, 1))
        p99 = float(np.percentile(array, 99))
        
        # Istogramma per identificare tessuti
        hist, bins = np.histogram(array.flatten(), bins=200, range=(-1024, 1024))
        
        # Identifica picchi principali
        air_range = (-1024, -700)  # Aria
        lung_range = (-700, -200)  # Polmone
        soft_range = (-100, 100)   # Tessuti molli
        bone_range = (200, 1500)   # Osso
        
        air_voxels = np.sum((array >= air_range[0]) & (array <= air_range[1]))
        lung_voxels = np.sum((array >= lung_range[0]) & (array <= lung_range[1]))
        soft_voxels = np.sum((array >= soft_range[0]) & (array <= soft_range[1]))
        bone_voxels = np.sum((array >= bone_range[0]) & (array <= bone_range[1]))
        
        total_voxels = array.size
        
        # Calcola entropì (misura di variabilità)
        hist_norm = hist / hist.sum()
        entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
        
        metrics = {
            'hu_min': hu_min,
            'hu_max': hu_max,
            'hu_mean': hu_mean,
            'hu_std': hu_std,
            'hu_median': hu_median,
            'hu_p01': p01,
            'hu_p99': p99,
            'hu_range': hu_max - hu_min,
            'air_percentage': (air_voxels / total_voxels) * 100,
            'lung_percentage': (lung_voxels / total_voxels) * 100,
            'soft_tissue_percentage': (soft_voxels / total_voxels) * 100,
            'bone_percentage': (bone_voxels / total_voxels) * 100,
            'hu_entropy': entropy,
            'dynamic_range': p99 - p01
        }
        
        if self.verbose:
            print(f"\n📊 INTENSITY (HU):")
            print(f"  Range: [{hu_min:.1f}, {hu_max:.1f}] (mean: {hu_mean:.1f}, std: {hu_std:.1f})")
            print(f"  P1-P99: [{p01:.1f}, {p99:.1f}]")
            print(f"  Air: {metrics['air_percentage']:.1f}%")
            print(f"  Lung: {metrics['lung_percentage']:.1f}%")
            print(f"  Soft tissue: {metrics['soft_tissue_percentage']:.1f}%")
            print(f"  Bone: {metrics['bone_percentage']:.1f}%")
            print(f"  Entropy: {entropy:.2f}")
        
        return metrics
    
    def _analyze_quality(self, array):
        """Analizza rumore e artefatti"""
        
        # 1. NOISE ESTIMATION (su regione aria)
        air_mask = array < -700
        if np.sum(air_mask) > 1000:
            air_std = float(np.std(array[air_mask]))
        else:
            air_std = 0.0
        
        # 2. CONTRAST-TO-NOISE RATIO (CNR)
        # Confronta tessuto molle vs polmone
        lung_mask = (array >= -700) & (array <= -200)
        soft_mask = (array >= -100) & (array <= 100)
        
        if np.sum(lung_mask) > 100 and np.sum(soft_mask) > 100:
            lung_mean = float(np.mean(array[lung_mask]))
            soft_mean = float(np.mean(array[soft_mask]))
            combined_std = float(np.sqrt((np.std(array[lung_mask])**2 + 
                                         np.std(array[soft_mask])**2) / 2))
            cnr = abs(lung_mean - soft_mean) / combined_std if combined_std > 0 else 0
        else:
            cnr = 0.0
        
        # 3. METAL ARTIFACTS (voxels con HU estremi)
        metal_voxels = np.sum(array > 2000)
        metal_percentage = (metal_voxels / array.size) * 100
        
        # 4. TRUNCATION ARTIFACTS (troppi voxels al limite)
        truncated_low = np.sum(array <= -1024)
        truncated_high = np.sum(array >= 3071)
        truncation_percentage = ((truncated_low + truncated_high) / array.size) * 100
        
        # 5. UNIFORMITÀ (coefficiente di variazione per slice)
        slice_means = np.mean(array, axis=(1, 2))
        slice_stds = np.std(array, axis=(1, 2))
        cv_mean = float(np.mean(slice_stds / (slice_means + 1e-6)))
        
        # 6. GRADIENT ANALYSIS (detect motion/misalignment)
        # Calcola differenze tra slices consecutive
        slice_diffs = np.diff(slice_means)
        abrupt_changes = np.sum(np.abs(slice_diffs) > (3 * np.std(slice_diffs)))
        
        metrics = {
            'noise_std': air_std,
            'cnr': cnr,
            'metal_artifact_percentage': metal_percentage,
            'truncation_percentage': truncation_percentage,
            'slice_uniformity_cv': cv_mean,
            'abrupt_slice_changes': int(abrupt_changes),
            'has_metal_artifacts': metal_percentage > 0.1,
            'has_truncation': truncation_percentage > 1.0,
            'has_motion_artifacts': abrupt_changes > (array.shape[0] * 0.05)
        }
        
        if self.verbose:
            print(f"\n🔍 QUALITY:")
            print(f"  Noise (air std): {air_std:.2f} HU")
            print(f"  CNR: {cnr:.2f}")
            print(f"  Metal artifacts: {metal_percentage:.2f}%")
            print(f"  Truncation: {truncation_percentage:.2f}%")
            print(f"  Abrupt slice changes: {abrupt_changes}")
        
        return metrics
    
    def _analyze_anatomy(self, array):
        """Analizza contenuto anatomico"""
        
        # 1. BODY REGION DETECTION
        # Cerca il torso (maggiore quantità di tessuto molle)
        soft_tissue = (array >= -100) & (array <= 100)
        soft_per_slice = np.sum(soft_tissue, axis=(1, 2))
        
        # Trova slice con più tessuto (probabilmente centro torso)
        center_slice_idx = int(np.argmax(soft_per_slice))
        
        # 2. FIELD OF VIEW COVERAGE
        # Controlla se body è troncato
        center_slice = array[center_slice_idx]
        body_mask = center_slice > -500  # Tutto tranne aria esterna
        
        if np.sum(body_mask) > 100:
            body_coords = np.argwhere(body_mask)
            y_min, x_min = body_coords.min(axis=0)
            y_max, x_max = body_coords.max(axis=0)
            
            # Margine ai bordi
            margin_top = y_min
            margin_bottom = array.shape[1] - y_max
            margin_left = x_min
            margin_right = array.shape[2] - x_max
            
            min_margin = min(margin_top, margin_bottom, margin_left, margin_right)
            is_truncated = min_margin < 10  # Meno di 10 pixel di margine
        else:
            is_truncated = False
            min_margin = 0
        
        # 3. LUNG DETECTION
        lung_tissue = (array >= -900) & (array <= -300)
        lung_per_slice = np.sum(lung_tissue, axis=(1, 2))
        
        # Trova range dove ci sono polmoni
        lung_threshold = np.max(lung_per_slice) * 0.3
        lung_slices = np.where(lung_per_slice > lung_threshold)[0]
        
        if len(lung_slices) > 0:
            lung_start = int(lung_slices[0])
            lung_end = int(lung_slices[-1])
            lung_span = lung_end - lung_start
            has_lungs = True
        else:
            lung_start = 0
            lung_end = 0
            lung_span = 0
            has_lungs = False
        
        # 4. TRACHEA/AIRWAY DETECTION
        airway = (array >= -1000) & (array <= -700)
        airway_per_slice = np.sum(airway, axis=(1, 2))
        
        # Cerca pattern tipico: airway superiore (trachea) poi biforcazione
        airway_pattern_score = 0.0
        if has_lungs and lung_span > 50:
            # Nella parte superiore dei polmoni dovrebbe esserci trachea
            upper_airway = np.mean(airway_per_slice[lung_start:lung_start+20])
            lower_airway = np.mean(airway_per_slice[lung_end-20:lung_end])
            if upper_airway > lower_airway * 0.5:
                airway_pattern_score = upper_airway / (lower_airway + 1)
        
        metrics = {
            'center_slice_idx': center_slice_idx,
            'body_truncated': is_truncated,
            'min_body_margin': min_margin,
            'has_lungs': has_lungs,
            'lung_start_slice': lung_start,
            'lung_end_slice': lung_end,
            'lung_span_slices': lung_span,
            'airway_pattern_score': float(airway_pattern_score),
            'likely_chest_ct': has_lungs and lung_span > 100
        }
        
        if self.verbose:
            print(f"\n🫁 ANATOMY:")
            print(f"  Body truncated: {is_truncated} (min margin: {min_margin} px)")
            print(f"  Lungs detected: {has_lungs}")
            if has_lungs:
                print(f"  Lung range: slices {lung_start}-{lung_end} ({lung_span} slices)")
            print(f"  Airway pattern score: {airway_pattern_score:.2f}")
            print(f"  Likely chest CT: {metrics['likely_chest_ct']}")
        
        return metrics
    
    def _check_totalsegmentator_compatibility(self, metrics):
        """Verifica compatibilità con TotalSegmentator"""
        
        issues = []
        warnings = []
        
        # 1. Number of slices
        if metrics['num_slices'] < self.ts_requirements['min_slices']:
            issues.append(f"Too few slices: {metrics['num_slices']} < {self.ts_requirements['min_slices']}")
        
        # 2. Slice thickness
        if metrics['slice_thickness'] > self.ts_requirements['max_slice_thickness']:
            issues.append(f"Slice too thick: {metrics['slice_thickness']:.3f} > {self.ts_requirements['max_slice_thickness']}")
        
        # 3. XY resolution
        avg_xy_spacing = (metrics['pixel_spacing_x'] + metrics['pixel_spacing_y']) / 2
        
        if avg_xy_spacing < self.ts_requirements['min_xy_resolution']:
            warnings.append(f"Very fine resolution: {avg_xy_spacing:.3f} < {self.ts_requirements['min_xy_resolution']}")
        
        if avg_xy_spacing > self.ts_requirements['max_xy_resolution']:
            issues.append(f"Resolution too coarse: {avg_xy_spacing:.3f} > {self.ts_requirements['max_xy_resolution']}")
        
        # 4. FOV
        min_fov = min(metrics['fov_x_mm'], metrics['fov_y_mm'])
        if min_fov < self.ts_requirements['min_fov']:
            warnings.append(f"Small FOV: {min_fov:.1f} < {self.ts_requirements['min_fov']}")
        
        # 5. HU range anomalo
        if metrics['hu_min'] > -1000 or metrics['hu_max'] < 500:
            warnings.append(f"Unusual HU range: [{metrics['hu_min']:.0f}, {metrics['hu_max']:.0f}]")
        
        # 6. Body truncation
        if metrics.get('body_truncated', False):
            warnings.append("Body appears truncated in FOV")
        
        # 7. Low lung content
        if metrics.get('likely_chest_ct', False) and metrics['lung_percentage'] < 10:
            warnings.append(f"Low lung tissue: {metrics['lung_percentage']:.1f}%")
        
        # 8. High noise
        if metrics.get('noise_std', 0) > 50:
            warnings.append(f"High noise: {metrics['noise_std']:.1f} HU")
        
        # 9. Artifacts
        if metrics.get('has_metal_artifacts', False):
            warnings.append("Metal artifacts detected")
        
        if metrics.get('has_motion_artifacts', False):
            warnings.append("Possible motion artifacts")
        
        # 10. Low CNR
        if metrics.get('cnr', 0) < 5:
            warnings.append(f"Low contrast-to-noise: {metrics['cnr']:.2f}")
        
        # VERDICT
        is_compatible = len(issues) == 0
        compatibility_score = 100
        
        compatibility_score -= len(issues) * 30  # Ogni issue -30%
        compatibility_score -= len(warnings) * 10  # Ogni warning -10%
        compatibility_score = max(0, compatibility_score)
        
        result = {
            'ts_compatible': is_compatible,
            'ts_compatibility_score': compatibility_score,
            'ts_issues': issues,
            'ts_warnings': warnings,
            'ts_num_issues': len(issues),
            'ts_num_warnings': len(warnings)
        }
        
        if self.verbose:
            print(f"\n✅ TOTALSEGMENTATOR COMPATIBILITY:")
            print(f"  Compatible: {is_compatible}")
            print(f"  Score: {compatibility_score}/100")
            
            if issues:
                print(f"  ❌ Issues ({len(issues)}):")
                for issue in issues:
                    print(f"    - {issue}")
            
            if warnings:
                print(f"  ⚠️  Warnings ({len(warnings)}):")
                for warning in warnings:
                    print(f"    - {warning}")
        
        return result
    
    def _compute_quality_score(self, metrics):
        """Calcola quality score globale (0-100)"""
        
        score = 100
        
        # Penalità per problemi
        score -= metrics.get('ts_num_issues', 0) * 20
        score -= metrics.get('ts_num_warnings', 0) * 5
        
        # Bonus per caratteristiche positive
        if metrics.get('cnr', 0) > 10:
            score += 5
        
        if metrics.get('lung_percentage', 0) > 20 and metrics.get('lung_percentage', 0) < 60:
            score += 5  # Range tipico per chest CT
        
        if metrics.get('noise_std', 100) < 20:
            score += 5
        
        if not metrics.get('has_metal_artifacts', True):
            score += 5
        
        return max(0, min(100, score))
    
    def _print_summary(self, metrics):
        """Print summary finale"""
        quality = metrics['quality_score']
        
        print(f"\n{'='*80}")
        print(f"QUALITY SCORE: {quality}/100", end='')
        
        if quality >= 80:
            print(" ✅ EXCELLENT")
        elif quality >= 60:
            print(" ⚠️  ACCEPTABLE")
        else:
            print(" ❌ POOR")
        
        print('='*80)
    
    def analyze_batch(self, folder_path, pattern="*.mhd", max_scans=None):
        """Analizza batch di CT"""
        
        print("\n" + "="*80)
        print(" "*25 + "BATCH CT QUALITY ANALYSIS")
        print("="*80)
        
        folder = Path(folder_path)
        mhd_files = list(folder.glob(pattern))
        
        if max_scans:
            mhd_files = mhd_files[:max_scans]
        
        print(f"\nFound {len(mhd_files)} CT scans to analyze")
        
        for i, mhd_path in enumerate(mhd_files, 1):
            print(f"\n[{i}/{len(mhd_files)}]", end=' ')
            self.analyze_ct(str(mhd_path))
        
        return self.get_dataframe()
    
    def get_dataframe(self):
        """Converti risultati in DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        return df
    
    def generate_report(self, output_path="ct_quality_report.txt"):
        """Genera report dettagliato"""
        
        df = self.get_dataframe()
        
        if len(df) == 0:
            print("No data to report")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" "*25 + "CT QUALITY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Summary
            total = len(df)
            compatible = len(df[df['ts_compatible'] == True])
            
            f.write(f"Total scans analyzed: {total}\n")
            f.write(f"TotalSegmentator compatible: {compatible} ({compatible/total*100:.1f}%)\n")
            f.write(f"Incompatible: {total - compatible}\n\n")
            
            # Quality distribution
            excellent = len(df[df['quality_score'] >= 80])
            acceptable = len(df[(df['quality_score'] >= 60) & (df['quality_score'] < 80)])
            poor = len(df[df['quality_score'] < 60])
            
            f.write("Quality Distribution:\n")
            f.write(f"  Excellent (≥80): {excellent} ({excellent/total*100:.1f}%)\n")
            f.write(f"  Acceptable (60-79): {acceptable} ({acceptable/total*100:.1f}%)\n")
            f.write(f"  Poor (<60): {poor} ({poor/total*100:.1f}%)\n\n")
            
            # Common issues
            f.write("="*80 + "\n")
            f.write("COMMON ISSUES\n")
            f.write("="*80 + "\n\n")
            
            all_issues = []
            for issues_list in df['ts_issues'].dropna():
                if isinstance(issues_list, list):
                    all_issues.extend(issues_list)
            
            from collections import Counter
            issue_counts = Counter(all_issues)
            
            for issue, count in issue_counts.most_common():
                f.write(f"  {issue}: {count} scans ({count/total*100:.1f}%)\n")
            
            # Statistics
            f.write("\n" + "="*80 + "\n")
            f.write("STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            numeric_cols = ['num_slices', 'slice_thickness', 'pixel_spacing_x', 
                          'hu_mean', 'cnr', 'lung_percentage', 'quality_score']
            
            for col in numeric_cols:
                if col in df.columns:
                    f.write(f"{col}:\n")
                    f.write(f"  Mean: {df[col].mean():.2f}\n")
                    f.write(f"  Std: {df[col].std():.2f}\n")
                    f.write(f"  Min: {df[col].min():.2f}\n")
                    f.write(f"  Max: {df[col].max():.2f}\n\n")
            
            # Incompatible scans
            incompatible_df = df[df['ts_compatible'] == False]
            
            if len(incompatible_df) > 0:
                f.write("="*80 + "\n")
                f.write("INCOMPATIBLE SCANS\n")
                f.write("="*80 + "\n\n")
                
                for _, row in incompatible_df.iterrows():
                    f.write(f"{row['scan_name']}:\n")
                    f.write(f"  Quality Score: {row['quality_score']}/100\n")
                    
                    if isinstance(row['ts_issues'], list):
                        for issue in row['ts_issues']:
                            f.write(f"  ❌ {issue}\n")
                    
                    if isinstance(row['ts_warnings'], list):
                        for warning in row['ts_warnings']:
                            f.write(f"  ⚠️  {warning}\n")
                    
                    f.write("\n")
        
        print(f"\n✓ Report saved: {output_path}")
    
    def plot_analysis(self, output_dir="ct_quality_plots"):
        """Genera visualizzazioni"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.get_dataframe()
        
        if len(df) == 0:
            print("No data to plot")
            return
        
        # 1. Quality score distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Quality scores
        axes[0, 0].hist(df['quality_score'].dropna(), bins=20, edgecolor='black', color='skyblue')
        axes[0, 0].axvline(80, color='g', linestyle='--', label='Excellent threshold')
        axes[0, 0].axvline(60, color='orange', linestyle='--', label='Acceptable threshold')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Quality Score Distribution')
        axes[0, 0].legend()
        
        # Slice count
        axes[0, 1].hist(df['num_slices'].dropna(), bins=30, edgecolor='black', color='green')
        axes[0, 1].axvline(400, color='r', linestyle='--', label='TS min requirement')
        axes[0, 1].set_xlabel('Number of Slices')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Slice Count Distribution')
        axes[0, 1].legend()
        
        # Slice thickness
        axes[0, 2].hist(df['slice_thickness'].dropna(), bins=30, edgecolor='black', color='orange')
        axes[0, 2].axvline(0.7, color='r', linestyle='--', label='TS max requirement')
        axes[0, 2].set_xlabel('Slice Thickness (mm)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Slice Thickness Distribution')
        axes[0, 2].legend()
        
        # CNR
        axes[1, 0].hist(df['cnr'].dropna(), bins=30, edgecolor='black', color='purple')
        axes[1, 0].set_xlabel('Contrast-to-Noise Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('CNR Distribution')
        
        # Lung percentage
        axes[1, 1].hist(df['lung_percentage'].dropna(), bins=30, edgecolor='black', color='pink')
        axes[1, 1].set_xlabel('Lung Tissue %')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Lung Content Distribution')
        
        # Compatibility pie chart
        if 'ts_compatible' in df.columns:
            compatible_counts = df['ts_compatible'].value_counts()
            if len(compatible_counts) > 0:
                labels = ['Compatible', 'Incompatible']
                colors = ['lightgreen', 'lightcoral']
                axes[1, 2].pie(compatible_counts, 
                            labels=[labels[i] for i in range(len(compatible_counts))], 
                            autopct='%1.1f%%', 
                            colors=[colors[i] for i in range(len(compatible_counts))])
            axes[1, 2].set_title('TotalSegmentator Compatibility')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary_plots.png'), dpi=150)
        plt.close()
        
        # 2. Correlations
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Seleziona colonne numeriche per la matrice di correlazione
        numeric_cols = ['quality_score', 'num_slices', 'slice_thickness', 
                    'pixel_spacing_x', 'pixel_spacing_y', 'hu_mean', 
                    'hu_std', 'cnr', 'lung_percentage', 'noise_std',
                    'ts_compatibility_score']
        
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        corr_matrix = df[numeric_cols].corr()
        
        # Maschera per triangolo superiore
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, ax=ax,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150)
        plt.close()
        
        # 3. Scatter plots importanti
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Quality score vs slice thickness
        if 'slice_thickness' in df.columns and 'quality_score' in df.columns:
            axes[0, 0].scatter(df['slice_thickness'], df['quality_score'], alpha=0.6, color='blue')
            axes[0, 0].axvline(0.7, color='r', linestyle='--', label='TS max requirement')
            axes[0, 0].set_xlabel('Slice Thickness (mm)')
            axes[0, 0].set_ylabel('Quality Score')
            axes[0, 0].set_title('Quality vs Slice Thickness')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Quality score vs number of slices
        if 'num_slices' in df.columns and 'quality_score' in df.columns:
            axes[0, 1].scatter(df['num_slices'], df['quality_score'], alpha=0.6, color='green')
            axes[0, 1].axvline(400, color='r', linestyle='--', label='TS min requirement')
            axes[0, 1].set_xlabel('Number of Slices')
            axes[0, 1].set_ylabel('Quality Score')
            axes[0, 1].set_title('Quality vs Number of Slices')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # CNR vs noise
        if 'cnr' in df.columns and 'noise_std' in df.columns:
            axes[1, 0].scatter(df['noise_std'], df['cnr'], alpha=0.6, color='purple')
            axes[1, 0].set_xlabel('Noise (air std)')
            axes[1, 0].set_ylabel('CNR')
            axes[1, 0].set_title('CNR vs Noise')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Tissue composition
        if all(col in df.columns for col in ['air_percentage', 'lung_percentage', 'soft_tissue_percentage', 'bone_percentage']):
            x = range(len(df))
            width = 0.2
            
            axes[1, 1].bar(x, df['air_percentage'], width, label='Air', color='lightblue')
            axes[1, 1].bar([i + width for i in x], df['lung_percentage'], width, label='Lung', color='lightgreen')
            axes[1, 1].bar([i + 2*width for i in x], df['soft_tissue_percentage'], width, label='Soft Tissue', color='salmon')
            axes[1, 1].bar([i + 3*width for i in x], df['bone_percentage'], width, label='Bone', color='gold')
            
            axes[1, 1].set_xlabel('Scan Index')
            axes[1, 1].set_ylabel('Percentage (%)')
            axes[1, 1].set_title('Tissue Composition')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scatter_plots.png'), dpi=150)
        plt.close()
        
        print(f"\n✓ Plots saved in: {output_dir}")


    def export_to_json(self, output_path="ct_quality_results.json"):
        """Esporta risultati in formato JSON"""
        
        # Converti i risultati in formato serializzabile
        results_json = []
        
        for result in self.results:
            result_copy = result.copy()
            
            # Converti numpy types a Python native types
            for key, value in result_copy.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    result_copy[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    result_copy[key] = float(value)
                elif isinstance(value, np.ndarray):
                    result_copy[key] = value.tolist()
                elif isinstance(value, np.bool_):
                    result_copy[key] = bool(value)
            
            results_json.append(result_copy)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_scans': len(results_json),
                'compatible_scans': len([r for r in results_json if r.get('ts_compatible', False)]),
                'results': results_json
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ JSON results saved: {output_path}")


    def identify_problematic_scans(self, threshold_score=60):
        """Identifica scansioni problematiche"""
        
        df = self.get_dataframe()
        
        if len(df) == 0:
            print("No data available")
            return pd.DataFrame()
        
        # Filtra scansioni problematiche
        problematic = df[
            (df['quality_score'] < threshold_score) | 
            (df['ts_compatible'] == False)
        ].copy()
        
        if len(problematic) == 0:
            print("\nNo problematic scans found!")
            return problematic
        
        print(f"\n{'='*80}")
        print(f"PROBLEMATIC SCANS ({len(problematic)} found)")
        print('='*80)
        
        # Ordina per score più basso
        problematic = problematic.sort_values('quality_score')
        
        for idx, (_, row) in enumerate(problematic.iterrows(), 1):
            print(f"\n{idx}. {row['scan_name']}")
            print(f"   Quality Score: {row['quality_score']}/100")
            print(f"   TS Compatible: {row['ts_compatible']}")
            
            if 'ts_issues' in row and isinstance(row['ts_issues'], list):
                for issue in row['ts_issues']:
                    print(f"   ❌ {issue}")
            
            if 'ts_warnings' in row and isinstance(row['ts_warnings'], list):
                for warning in row['ts_warnings']:
                    print(f"   ⚠️  {warning}")
        
        return problematic


    def get_recommendations(self, scan_name=None):
        """Genera raccomandazioni per scansioni specifiche o globali"""
        
        if scan_name:
            # Trova una scansione specifica
            for result in self.results:
                if result.get('scan_name') == scan_name:
                    return self._generate_scan_recommendations(result)
            
            print(f"Scan '{scan_name}' not found")
            return []
        
        else:
            # Raccomandazioni globali
            df = self.get_dataframe()
            
            if len(df) == 0:
                return []
            
            recommendations = []
            
            # Analisi statistica
            avg_slice_thickness = df['slice_thickness'].mean()
            avg_slices = df['num_slices'].mean()
            compatible_rate = df['ts_compatible'].mean() * 100
            
            if avg_slice_thickness > 0.7:
                recommendations.append(
                    f"Average slice thickness ({avg_slice_thickness:.3f} mm) exceeds TotalSegmentator "
                    f"requirement (0.7 mm max). Consider reconstruction with thinner slices."
                )
            
            if avg_slices < 400:
                recommendations.append(
                    f"Average number of slices ({avg_slices:.0f}) is below TotalSegmentator "
                    f"minimum (400). Ensure full coverage of anatomical regions."
                )
            
            if compatible_rate < 80:
                recommendations.append(
                    f"Only {compatible_rate:.1f}% of scans are compatible with TotalSegmentator. "
                    f"Review acquisition parameters and pre-processing steps."
                )
            
            # Controlla artefatti comuni
            if 'has_metal_artifacts' in df.columns:
                metal_rate = df['has_metal_artifacts'].mean() * 100
                if metal_rate > 20:
                    recommendations.append(
                        f"High rate of metal artifacts ({metal_rate:.1f}% of scans). "
                        f"Consider metal artifact reduction techniques."
                    )
            
            return recommendations


    def _generate_scan_recommendations(self, metrics):
        """Genera raccomandazioni per una singola scansione"""
        
        recommendations = []
        
        # 1. Problemi di compatibilità
        if not metrics.get('ts_compatible', True):
            recommendations.append("⚠️  NOT COMPATIBLE with TotalSegmentator")
        
        # 2. Spessore slice
        if metrics.get('slice_thickness', 0) > 0.7:
            recommendations.append(
                f"❌ Reduce slice thickness: {metrics['slice_thickness']:.3f} mm > 0.7 mm max "
                f"(reconstruct with thinner slices if possible)"
            )
        
        # 3. Numero di slice
        if metrics.get('num_slices', 0) < 400:
            recommendations.append(
                f"❌ Increase number of slices: {metrics['num_slices']} < 400 minimum "
                f"(ensure full coverage of thoracic region)"
            )
        
        # 4. Risoluzione
        avg_spacing = (metrics.get('pixel_spacing_x', 0) + metrics.get('pixel_spacing_y', 0)) / 2
        if avg_spacing > 1.0:
            recommendations.append(
                f"❌ Improve in-plane resolution: {avg_spacing:.3f} mm > 1.0 mm max "
                f"(reconstruct with finer matrix if possible)"
            )
        
        # 5. Rumore
        if metrics.get('noise_std', 0) > 50:
            recommendations.append(
                f"⚠️  High noise level: {metrics['noise_std']:.1f} HU "
                f"(consider noise reduction filters or higher dose protocol)"
            )
        
        # 6. Basso contrasto
        if metrics.get('cnr', 0) < 5:
            recommendations.append(
                f"⚠️  Low contrast-to-noise ratio: {metrics['cnr']:.2f} "
                f"(optimize reconstruction kernel or contrast protocol)"
            )
        
        # 7. Artefatti metallici
        if metrics.get('has_metal_artifacts', False):
            recommendations.append(
                f"⚠️  Metal artifacts detected "
                f"(apply metal artifact reduction algorithm or manual correction)"
            )
        
        # 8. Corpo troncato
        if metrics.get('body_truncated', False):
            recommendations.append(
                f"⚠️  Body appears truncated in FOV "
                f"(reposition patient for complete coverage)"
            )
        
        return recommendations


    def save_analysis_report(self, output_dir="ct_quality_analysis"):
        """Salva report completo"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Genera report testuale
        self.generate_report(os.path.join(output_dir, "report.txt"))
        
        # 2. Esporta DataFrame
        df = self.get_dataframe()
        if len(df) > 0:
            excel_path = os.path.join(output_dir, "results.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Full Results', index=False)
                
                # Aggiungi foglio riassuntivo
                summary_stats = pd.DataFrame({
                    'Metric': ['Total Scans', 'Compatible Scans', 'Incompatible Scans',
                            'Average Quality Score', 'Average Slices', 'Average Thickness',
                            'Compatible Rate (%)'],
                    'Value': [
                        len(df),
                        len(df[df['ts_compatible'] == True]),
                        len(df[df['ts_compatible'] == False]),
                        df['quality_score'].mean(),
                        df['num_slices'].mean(),
                        df['slice_thickness'].mean(),
                        (len(df[df['ts_compatible'] == True]) / len(df)) * 100
                    ]
                })
                summary_stats.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"✓ Excel report saved: {excel_path}")
        
        # 3. Esporta JSON
        self.export_to_json(os.path.join(output_dir, "results.json"))
        
        # 4. Genera plots
        self.plot_analysis(output_dir)
        
        # 5. Identifica scansioni problematiche
        problematic = self.identify_problematic_scans()
        if len(problematic) > 0:
            problematic.to_csv(os.path.join(output_dir, "problematic_scans.csv"), index=False)
        
        print(f"\n{'='*80}")
        print(f"✓ Analysis complete! All files saved in: {output_dir}")
        print('='*80)


def main():
    """Funzione principale per analisi con variabili predefinite"""
    
    # ================= CONFIGURAZIONE =================
    # MODIFICA QUESTE VARIABILI PER LE TUE ANALISI
    input_path = "X:/Francesca Saglimbeni/tesi/vesselsegmentation/airway_segmentation/carve_data"                     # Percorso input (cartella o file .mhd)
    output_dir = "X:/Francesca Saglimbeni/tesi/vesselsegmentation/airway_segmentation/output"       # Cartella output
    max_scans = 100                         # Numero massimo di scansioni (None = tutte)
    pattern = "*.mhd"                        # Pattern file
    verbose = True                           # Output dettagliato
    # =================================================
    
    print(f"{'='*80}")
    print(f"{' '*25}CT QUALITY ANALYZER")
    print(f"{'='*80}")
    
    # Crea l'analizzatore
    analyzer = CTQualityAnalyzer(verbose=verbose)
    
    # Converti in Path
    input_path = Path(input_path)
    
    if input_path.is_dir():
        print(f"\n📁 Analyzing folder: {input_path}")
        print(f"📊 Pattern: {pattern}")
        if max_scans:
            print(f"🔢 Max scans: {max_scans}")
        
        # Analizza batch
        analyzer.analyze_batch(
            folder_path=str(input_path),
            pattern=pattern,
            max_scans=max_scans
        )
        
    elif input_path.is_file() and input_path.suffix.lower() == '.mhd':
        print(f"\n📄 Analyzing single file: {input_path}")
        
        # Analizza singola scansione
        analyzer.analyze_ct(str(input_path))
        
    else:
        print(f"\n❌ Invalid input: {input_path}")
        print("Please provide either:")
        print("1. A folder containing .mhd files")
        print("2. A single .mhd file")
        return
    
    # Salva report completo
    print(f"\n💾 Saving analysis to: {output_dir}")
    analyzer.save_analysis_report(output_dir)
    
    # Mostra raccomandazioni globali
    recommendations = analyzer.get_recommendations()
    if recommendations:
        print(f"\n{'='*80}")
        print("📋 GLOBAL RECOMMENDATIONS")
        print('='*80)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    # Identifica scansioni problematiche
    problematic_df = analyzer.identify_problematic_scans(threshold_score=60)
    if len(problematic_df) > 0:
        print(f"\n{'='*80}")
        print(f"⚠️  FOUND {len(problematic_df)} PROBLEMATIC SCANS")
        print('='*80)
        print("Check 'problematic_scans.csv' in output folder for details")
    
    # Statistiche finali
    df = analyzer.get_dataframe()
    if len(df) > 0:
        compatible_rate = (len(df[df['ts_compatible'] == True]) / len(df)) * 100
        avg_quality = df['quality_score'].mean()
        
        print(f"\n{'='*80}")
        print("📈 ANALYSIS SUMMARY")
        print('='*80)
        print(f"Total scans analyzed: {len(df)}")
        print(f"Average quality score: {avg_quality:.1f}/100")
        print(f"Compatible with TotalSegmentator: {compatible_rate:.1f}%")
        
        # Distribuzione qualità
        excellent = len(df[df['quality_score'] >= 80])
        acceptable = len(df[(df['quality_score'] >= 60) & (df['quality_score'] < 80)])
        poor = len(df[df['quality_score'] < 60])
        
        print(f"\nQuality distribution:")
        print(f"  Excellent (≥80): {excellent} ({excellent/len(df)*100:.1f}%)")
        print(f"  Acceptable (60-79): {acceptable} ({acceptable/len(df)*100:.1f}%)")
        print(f"  Poor (<60): {poor} ({poor/len(df)*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print('='*80)


# Versione alternativa con configurazioni multiple
def main_multiple_folders():
    """Analizza multiple cartelle in sequenza"""
    
    # ================= CONFIGURAZIONE MULTIPLA =================
    # Lista di cartelle da analizzare
    input_folders = [
        "data/ct_scans_folder1",
        "data/ct_scans_folder2",
        "data/ct_scans_folder3"
    ]
    
    output_base_dir = "analysis_results"  # Cartella base per output
    max_scans = 50                        # Analizza solo 50 scansioni per cartella
    pattern = "*.mhd"                     # Pattern file
    verbose = True                        # Output dettagliato
    # ==========================================================
    
    all_results = []
    
    for folder_idx, input_folder in enumerate(input_folders, 1):
        print(f"\n{'='*80}")
        print(f"📂 Analyzing folder {folder_idx}/{len(input_folders)}: {input_folder}")
        print('='*80)
        
        # Crea output directory specifica
        folder_name = Path(input_folder).name
        output_dir = f"{output_base_dir}/{folder_name}_{folder_idx}"
        
        # Analizzatore
        analyzer = CTQualityAnalyzer(verbose=verbose)
        
        # Analizza
        analyzer.analyze_batch(
            folder_path=input_folder,
            pattern=pattern,
            max_scans=max_scans
        )
        
        # Salva risultati individuali
        analyzer.save_analysis_report(output_dir)
        
        # Aggiungi risultati alla lista
        df = analyzer.get_dataframe()
        if len(df) > 0:
            df['source_folder'] = input_folder
            all_results.append(df)
        
        # Pausa tra cartelle (opzionale)
        import time
        if folder_idx < len(input_folders):
            print("\n⏱️  Pausing before next folder...")
            time.sleep(2)
    
    # Combina tutti i risultati
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_output = f"{output_base_dir}/combined_analysis"
        os.makedirs(combined_output, exist_ok=True)
        
        # Salva risultati combinati
        combined_df.to_csv(f"{combined_output}/all_results.csv", index=False)
        
        # Genera report combinato
        print(f"\n{'='*80}")
        print("📊 COMBINED ANALYSIS")
        print('='*80)
        print(f"Total scans across all folders: {len(combined_df)}")
        
        # Statistiche per cartella
        for folder in input_folders:
            folder_df = combined_df[combined_df['source_folder'] == folder]
            if len(folder_df) > 0:
                comp_rate = (len(folder_df[folder_df['ts_compatible'] == True]) / len(folder_df)) * 100
                avg_qual = folder_df['quality_score'].mean()
                print(f"\n{folder}:")
                print(f"  Scans: {len(folder_df)}")
                print(f"  Avg quality: {avg_qual:.1f}/100")
                print(f"  Compatible: {comp_rate:.1f}%")
        
        print(f"\n✅ Combined results saved to: {combined_output}")


# Versione con filtro avanzato
def main_with_filters():
    """Analizza con filtri specifici"""
    
    # ================= CONFIGURAZIONE =================
    input_path = "input"
    output_dir = "filtered_analysis"
    
    # Filtri avanzati
    filters = {
        'min_slices': 300,           # Minimo slice richieste
        'max_thickness': 0.8,        # Massimo spessore slice
        'min_quality_score': 70      # Punteggio qualità minimo
    }
    
    verbose = True
    # =================================================
    
    analyzer = CTQualityAnalyzer(verbose=verbose)
    
    # Analizza
    print(f"Analyzing with filters:")
    for key, value in filters.items():
        print(f"  {key}: {value}")
    
    df = analyzer.analyze_batch(
        folder_path=input_path,
        pattern="*.mhd",
        max_scans=None
    )
    
    if len(df) > 0:
        # Applica filtri
        filtered_df = df[
            (df['num_slices'] >= filters['min_slices']) &
            (df['slice_thickness'] <= filters['max_thickness']) &
            (df['quality_score'] >= filters['min_quality_score'])
        ]
        
        print(f"\n📊 Filter results:")
        print(f"  Total scans: {len(df)}")
        print(f"  Passing filters: {len(filtered_df)} ({len(filtered_df)/len(df)*100:.1f}%)")
        
        # Salva scansioni filtrate
        os.makedirs(output_dir, exist_ok=True)
        filtered_df.to_csv(f"{output_dir}/filtered_scans.csv", index=False)
        
        # Crea analisi solo per scansioni filtrate
        analyzer.results = filtered_df.to_dict('records')
        analyzer.save_analysis_report(output_dir)


if __name__ == "__main__":
    # ================= SCELGI LA FUNZIONE DA ESEGUIRE =================
    
    # Opzione 1: Analisi semplice
    main()
    
    # Opzione 2: Analisi multiple cartelle
    # main_multiple_folders()
    
    # Opzione 3: Analisi con filtri
    # main_with_filters()
    
    # ==================================================================

    