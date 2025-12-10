import numpy as np
import SimpleITK as sitk
import os


class HUIntensityNormalizer:
    """
    Normalizza le intensità HU delle CT prima della segmentazione
    Risolve problemi di rescale slope/intercept e windowing
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Range HU attesi per tessuti toracici
        self.expected_ranges = {
            'air': (-1024, -700),
            'lung': (-900, -300),
            'soft_tissue': (-100, 100),
            'bone': (200, 1500)
        }
        
    def analyze_hu_distribution(self, image_array):
        """Analizza distribuzione HU per diagnostica"""
        hu_min = float(np.min(image_array))
        hu_max = float(np.max(image_array))
        hu_mean = float(np.mean(image_array))
        hu_median = float(np.median(image_array))
        
        # Percentili
        p01 = float(np.percentile(image_array, 1))
        p99 = float(np.percentile(image_array, 99))
        
        return {
            'min': hu_min,
            'max': hu_max,
            'mean': hu_mean,
            'median': hu_median,
            'p01': p01,
            'p99': p99,
            'range': hu_max - hu_min
        }
    
    def detect_hu_anomaly(self, stats):
        """
        Rileva se le HU sono anomale
        Returns: (is_anomalous, anomaly_type, severity)
        """
        mean_hu = stats['mean']
        
        # Caso 1: HU troppo basse (problema OSIC)
        if mean_hu < -700:
            severity = abs(mean_hu + 500) / 500  # Quanto è grave
            return True, 'too_low', min(severity, 3.0)
        
        # Caso 2: HU troppo alte
        if mean_hu > -100:
            severity = abs(mean_hu + 300) / 300
            return True, 'too_high', min(severity, 3.0)
        
        # Caso 3: Range HU troppo ristretto (compressione)
        if stats['range'] < 500:
            return True, 'compressed', 2.0
        
        # Caso 4: Range HU troppo ampio (rumore/metal)
        if stats['range'] > 5000:
            return True, 'excessive_range', 1.5
        
        return False, None, 0.0
    
    def estimate_correction_parameters(self, image_array, stats):
        """
        Stima parametri di correzione basati su istogramma
        """
        # Identifica picco dell'aria (dovrebbe essere ~-1000 HU)
        hist, bins = np.histogram(image_array.flatten(), 
                                  bins=300, 
                                  range=(stats['min'], stats['max']))
        
        # Trova picco principale nell'area bassa (aria)
        air_range_idx = np.where(bins < stats['median'])[0]
        if len(air_range_idx) > 0:
            air_hist = hist[:len(air_range_idx)]
            air_peak_idx = np.argmax(air_hist)
            air_peak_hu = bins[air_peak_idx]
        else:
            air_peak_hu = stats['p01']
        
        # Il picco dell'aria dovrebbe essere a circa -1000 HU
        expected_air_peak = -1000
        offset = expected_air_peak - air_peak_hu
        
        # Stima scaling basato sul range
        expected_range = 2500  # Range tipico CT toracico
        actual_range = stats['p99'] - stats['p01']
        scale = expected_range / actual_range if actual_range > 0 else 1.0
        
        # Limita correzioni estreme
        offset = np.clip(offset, -2000, 2000)
        scale = np.clip(scale, 0.5, 2.0)
        
        return offset, scale
    
    def apply_hu_correction(self, image_array, offset, scale):
        """
        Applica correzione lineare: HU_new = (HU_old * scale) + offset
        """
        corrected = (image_array.astype(np.float32) * scale) + offset
        
        # Clamp a range valido
        corrected = np.clip(corrected, -1024, 3071)
        
        return corrected.astype(image_array.dtype)
    
    def normalize_image(self, image_path, output_path=None, force_correction=False):
        """
        Pipeline completa di normalizzazione
        
        Args:
            image_path: Path to input CT
            output_path: Path to save normalized CT (if None, auto-generate)
            force_correction: Apply correction even if HU seems normal
        
        Returns:
            (normalized_image_path, correction_applied, correction_params)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("HU INTENSITY NORMALIZATION")
            print(f"{'='*70}")
            print(f"Input: {os.path.basename(image_path)}")
        
        # Load image
        sitk_image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(sitk_image)
        
        # Analyze HU distribution
        stats = self.analyze_hu_distribution(image_array)
        
        if self.verbose:
            print(f"\nOriginal HU Statistics:")
            print(f"  Mean: {stats['mean']:.1f}")
            print(f"  Median: {stats['median']:.1f}")
            print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
            print(f"  P1-P99: [{stats['p01']:.1f}, {stats['p99']:.1f}]")
        
        # Detect anomaly
        is_anomalous, anomaly_type, severity = self.detect_hu_anomaly(stats)
        
        if is_anomalous:
            if self.verbose:
                print(f"\n⚠️  HU ANOMALY DETECTED!")
                print(f"  Type: {anomaly_type}")
                print(f"  Severity: {severity:.2f}")
        
        # Apply correction if needed
        correction_applied = False
        correction_params = None
        
        if is_anomalous or force_correction:
            if self.verbose:
                print(f"\nApplying HU correction...")
            
            # Estimate correction
            offset, scale = self.estimate_correction_parameters(image_array, stats)
            
            if self.verbose:
                print(f"  Offset: {offset:.1f} HU")
                print(f"  Scale: {scale:.3f}")
            
            # Apply correction
            corrected_array = self.apply_hu_correction(image_array, offset, scale)
            
            # Verify correction
            corrected_stats = self.analyze_hu_distribution(corrected_array)
            
            if self.verbose:
                print(f"\nCorrected HU Statistics:")
                print(f"  Mean: {corrected_stats['mean']:.1f}")
                print(f"  Median: {corrected_stats['median']:.1f}")
                print(f"  Range: [{corrected_stats['min']:.1f}, {corrected_stats['max']:.1f}]")
            
            # Create output image
            corrected_sitk = sitk.GetImageFromArray(corrected_array)
            corrected_sitk.CopyInformation(sitk_image)
            
            # Save
            if output_path is None:
                base, ext = os.path.splitext(image_path)
                if ext == '.gz':
                    base = os.path.splitext(base)[0]
                    ext = '.nii.gz'
                output_path = f"{base}_normalized{ext}"
            
            sitk.WriteImage(corrected_sitk, output_path)
            
            correction_applied = True
            correction_params = {
                'offset': offset,
                'scale': scale,
                'original_mean': stats['mean'],
                'corrected_mean': corrected_stats['mean'],
                'anomaly_type': anomaly_type,
                'severity': severity
            }
            
            if self.verbose:
                print(f"\n✅ Normalized image saved: {output_path}")
        
        else:
            if self.verbose:
                print(f"\n✅ HU values are normal, no correction needed")
            
            output_path = image_path  # Use original
        
        return output_path, correction_applied, correction_params
    
    def batch_normalize(self, input_folder, output_folder=None, pattern="*.mhd"):
        """
        Normalizza tutte le CT in una cartella
        """
        from pathlib import Path
        
        if output_folder is None:
            output_folder = os.path.join(input_folder, "normalized")
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all files
        folder = Path(input_folder)
        files = list(folder.glob(pattern))
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("BATCH HU NORMALIZATION")
            print(f"{'='*70}")
            print(f"Input folder: {input_folder}")
            print(f"Output folder: {output_folder}")
            print(f"Files found: {len(files)}")
        
        results = []
        
        for i, file_path in enumerate(files, 1):
            if self.verbose:
                print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
            
            output_path = os.path.join(output_folder, file_path.name)
            
            try:
                norm_path, corrected, params = self.normalize_image(
                    str(file_path), 
                    output_path
                )
                
                results.append({
                    'file': file_path.name,
                    'corrected': corrected,
                    'params': params,
                    'output': norm_path,
                    'success': True
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error: {e}")
                
                results.append({
                    'file': file_path.name,
                    'corrected': False,
                    'params': None,
                    'output': None,
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        corrected_count = sum(1 for r in results if r['corrected'])
        success_count = sum(1 for r in results if r['success'])
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("BATCH NORMALIZATION COMPLETE")
            print(f"{'='*70}")
            print(f"Total files: {len(results)}")
            print(f"Successful: {success_count}")
            print(f"Corrected: {corrected_count}")
            print(f"No correction needed: {success_count - corrected_count}")
        
        return results


def main():
    """
    Usage example
    """
    # CONFIGURAZIONE
    input_path = "path/to/osic/data"  # Folder or single file
    output_folder = "path/to/normalized_data"
    
    # Create normalizer
    normalizer = HUIntensityNormalizer(verbose=True)
    
    # Option 1: Single file
    if os.path.isfile(input_path):
        norm_path, corrected, params = normalizer.normalize_image(
            input_path,
            output_path=None  # Auto-generate
        )
        
        if corrected:
            print(f"\n🎯 CORRECTION APPLIED:")
            print(f"  Original mean HU: {params['original_mean']:.1f}")
            print(f"  Corrected mean HU: {params['corrected_mean']:.1f}")
            print(f"  Anomaly: {params['anomaly_type']} (severity {params['severity']:.2f})")
        
        print(f"\n📁 Use this file for pipeline: {norm_path}")
    
    # Option 2: Batch processing
    elif os.path.isdir(input_path):
        results = normalizer.batch_normalize(
            input_path,
            output_folder=output_folder,
            pattern="*.mhd"
        )
        
        # Print summary of corrections
        print(f"\n{'='*70}")
        print("FILES REQUIRING CORRECTION:")
        print(f"{'='*70}")
        
        for r in results:
            if r['corrected'] and r['params']:
                print(f"\n{r['file']}:")
                print(f"  Original HU: {r['params']['original_mean']:.1f}")
                print(f"  Corrected HU: {r['params']['corrected_mean']:.1f}")
                print(f"  Anomaly: {r['params']['anomaly_type']}")


if __name__ == "__main__":
    main()