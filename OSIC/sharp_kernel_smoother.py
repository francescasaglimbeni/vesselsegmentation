"""
Sharp Kernel Smoother for OSIC CT Scans
========================================

Questo script applica filtri di smoothing alle CT scan con reconstruction kernel "sharp"
per ridurre il rumore ad alta frequenza e migliorare la segmentazione automatica.

Il problema: kernel sharp (B60f, B80f, FC51, etc.) introducono rumore che confonde
gli algoritmi di segmentazione come TotalSegmentator.

La soluzione: applicare filtri di smoothing (median, gaussian, bilateral) prima
della segmentazione.

Autore: Pipeline OSIC
Data: 2026-01-16
"""

import os
import sys
import SimpleITK as sitk
import numpy as np
import pydicom
from pathlib import Path
import json
import warnings
import time
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from skimage import filters, feature

warnings.filterwarnings('ignore')


class SharpKernelSmoother:
    """
    Identifica e applica smoothing a CT scan con kernel sharp
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Lista di kernel "sharp" che causano problemi
        # Numeri alti = sharp = rumore
        self.sharp_kernels = [
            'B60f', 'B70f', 'B80f', 'B80s',  # Siemens sharp
            'FC51', 'FC53', 'FC18',          # Toshiba sharp
            'LUNG', 'BONEPLUS',              # Kernel per osso/polmone (molto sharp)
        ]
        
        # Kernel "smooth" che funzionano bene (per riferimento)
        self.smooth_kernels = [
            'B41f', 'B50f',                  # Siemens smooth
            'FC01', 'FC10',                  # Toshiba smooth
            'C', 'F', 'L', 'YC',             # Altri kernel smooth
        ]
    
    
    def is_sharp_kernel(self, kernel_name):
        """
        Determina se un kernel è "sharp" basandosi sul nome
        
        Args:
            kernel_name: Nome del kernel DICOM (es: 'B70f', 'FC51')
        
        Returns:
            (is_sharp, sharpness_score)
            - is_sharp: True se è un kernel sharp
            - sharpness_score: 0-100, quanto è sharp (più alto = più sharp)
        """
        if not kernel_name:
            return False, 0
        
        kernel_upper = str(kernel_name).upper()
        
        # Check 1: È nella lista di kernel sharp conosciuti?
        if kernel_name in self.sharp_kernels:
            return True, 90
        
        # Check 2: Estrai il numero dal kernel (es: B70f -> 70, FC51 -> 51)
        import re
        numbers = re.findall(r'\d+', kernel_upper)
        
        if numbers:
            num = int(numbers[0])
            
            # Siemens: Bxx
            if kernel_upper.startswith('B'):
                # B50 o meno = smooth
                # B60-B90 = sharp
                if num <= 50:
                    return False, num
                else:
                    return True, num
            
            # Toshiba: FCxx
            elif kernel_upper.startswith('FC'):
                # FC01-FC10 = smooth
                # FC18+ = sharp
                if num <= 10:
                    return False, num
                else:
                    return True, num
        
        # Check 3: Keyword-based
        sharp_keywords = ['LUNG', 'BONE', 'EDGE', 'SHARP', 'DETAIL', 'HIGH']
        smooth_keywords = ['SOFT', 'SMOOTH', 'STANDARD', 'STD']
        
        for kw in sharp_keywords:
            if kw in kernel_upper:
                return True, 80
        
        for kw in smooth_keywords:
            if kw in kernel_upper:
                return False, 20
        
        # Default: non sappiamo
        return False, 50
    
    
    def extract_kernel_from_dicom(self, dicom_folder):
        """
        Estrae il ConvolutionKernel da una cartella DICOM
        
        Returns:
            kernel_name (str) o None
        """
        dcm_files = list(Path(dicom_folder).glob("*.dcm"))
        
        if len(dcm_files) == 0:
            return None
        
        # Leggi primo file
        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            
            if hasattr(ds, 'ConvolutionKernel'):
                return ds.ConvolutionKernel
        except:
            pass
        
        return None
    
    
    def apply_median_filter(self, image, radius=1):
        """
        Applica filtro mediano 3D
        
        Il filtro mediano è ottimo per ridurre noise salt-and-pepper
        preservando i bordi.
        
        Args:
            image: SimpleITK Image
            radius: Raggio del kernel (default=1 -> 3x3x3)
        
        Returns:
            SimpleITK Image filtrata
        """
        if self.verbose:
            print(f"  Applicando MedianImageFilter (radius={radius})...")
        
        median_filter = sitk.MedianImageFilter()
        median_filter.SetRadius(int(radius))
        
        filtered = median_filter.Execute(image)
        
        return filtered
    
    
    def apply_gaussian_filter(self, image, sigma=0.5):
        """
        Aplica filtro gaussiano 3D
        
        Il filtro gaussiano smootha uniformemente ma può offuscare i bordi.
        
        Args:
            image: SimpleITK Image
            sigma: Deviazione standard del gaussiano (default=0.5)
        
        Returns:
            SimpleITK Image filtrata
        """
        if self.verbose:
            print(f"  Applicando SmoothingRecursiveGaussianImageFilter (sigma={sigma})...")
        
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(float(sigma))
        
        filtered = gaussian_filter.Execute(image)
        
        return filtered
    
    
    def apply_bilateral_filter(self, image, domain_sigma=2.0, range_sigma=50.0):
        """
        Applica filtro bilaterale 3D
        
        Il filtro bilaterale è il migliore: smootha il noise preservando i bordi.
        È più lento ma più efficace per immagini mediche.
        
        Args:
            image: SimpleITK Image
            domain_sigma: Spatial sigma (quanto smooth nello spazio)
            range_sigma: Range sigma (quanto smooth nell'intensità)
        
        Returns:
            SimpleITK Image filtrata
        """
        if self.verbose:
            print(f"  Applicando BilateralImageFilter (domain={domain_sigma}, range={range_sigma})...")
        
        bilateral_filter = sitk.BilateralImageFilter()
        bilateral_filter.SetDomainSigma(float(domain_sigma))
        bilateral_filter.SetRangeSigma(float(range_sigma))
        
        filtered = bilateral_filter.Execute(image)
        
        return filtered
    
    
    def apply_curvature_flow_filter(self, image, iterations=5, time_step=0.0625):
        """
        Applica filtro Curvature Flow (anisotropic diffusion)
        
        Questo filtro è specifico per immagini mediche: smootha mantenendo
        le strutture anatomiche.
        
        Args:
            image: SimpleITK Image
            iterations: Numero di iterazioni (default=5)
            time_step: Time step per diffusione (default=0.0625)
        
        Returns:
            SimpleITK Image filtrata
        """
        if self.verbose:
            print(f"  Applicando CurvatureFlowImageFilter (iterations={iterations})...")
        
        curvature_filter = sitk.CurvatureFlowImageFilter()
        curvature_filter.SetNumberOfIterations(int(iterations))
        curvature_filter.SetTimeStep(float(time_step))
        
        filtered = curvature_filter.Execute(image)
        
        return filtered
    
    
    def calculate_image_quality_metrics(self, original_image, smoothed_image, method_name=''):
        """
        Calcola metriche di qualità per valutare l'efficacia dello smoothing
        
        Metriche calcolate:
        - SNR (Signal-to-Noise Ratio) in diverse regioni
        - Edge preservation (Canny edge detection)
        - Noise reduction (std delle regioni omogenee)
        - Smoothness (variazione locale)
        - Histogram similarity
        
        Args:
            original_image: SimpleITK Image originale
            smoothed_image: SimpleITK Image dopo smoothing
            method_name: Nome del metodo per log
        
        Returns:
            dict con metriche calcolate
        """
        # Converti in numpy arrays
        orig_array = sitk.GetArrayFromImage(original_image).astype(np.float32)
        smooth_array = sitk.GetArrayFromImage(smoothed_image).astype(np.float32)
        
        metrics = {'method': method_name}
        
        # Seleziona slice centrale per analisi 2D (più veloce)
        mid_slice = orig_array.shape[0] // 2
        orig_slice = orig_array[mid_slice, :, :]
        smooth_slice = smooth_array[mid_slice, :, :]
        
        # 1. SNR in regione polmonare (HU: -900 a -500)
        lung_mask_orig = (orig_array > -900) & (orig_array < -500)
        if lung_mask_orig.sum() > 100:
            lung_signal_orig = np.mean(orig_array[lung_mask_orig])
            lung_noise_orig = np.std(orig_array[lung_mask_orig])
            
            lung_mask_smooth = (smooth_array > -900) & (smooth_array < -500)
            lung_signal_smooth = np.mean(smooth_array[lung_mask_smooth])
            lung_noise_smooth = np.std(smooth_array[lung_mask_smooth])
            
            metrics['lung_snr_original'] = abs(lung_signal_orig / lung_noise_orig) if lung_noise_orig > 0 else 0
            metrics['lung_snr_smoothed'] = abs(lung_signal_smooth / lung_noise_smooth) if lung_noise_smooth > 0 else 0
            metrics['lung_noise_reduction'] = ((lung_noise_orig - lung_noise_smooth) / lung_noise_orig * 100) if lung_noise_orig > 0 else 0
        else:
            metrics['lung_snr_original'] = 0
            metrics['lung_snr_smoothed'] = 0
            metrics['lung_noise_reduction'] = 0
        
        # 2. SNR in tessuto molle (HU: -100 a 100)
        tissue_mask_orig = (orig_array > -100) & (orig_array < 100)
        if tissue_mask_orig.sum() > 100:
            tissue_signal_orig = np.mean(orig_array[tissue_mask_orig])
            tissue_noise_orig = np.std(orig_array[tissue_mask_orig])
            
            tissue_mask_smooth = (smooth_array > -100) & (smooth_array < 100)
            tissue_signal_smooth = np.mean(smooth_array[tissue_mask_smooth])
            tissue_noise_smooth = np.std(smooth_array[tissue_mask_smooth])
            
            metrics['tissue_snr_original'] = abs(tissue_signal_orig / tissue_noise_orig) if tissue_noise_orig > 0 else 0
            metrics['tissue_snr_smoothed'] = abs(tissue_signal_smooth / tissue_noise_smooth) if tissue_noise_smooth > 0 else 0
            metrics['tissue_noise_reduction'] = ((tissue_noise_orig - tissue_noise_smooth) / tissue_noise_orig * 100) if tissue_noise_orig > 0 else 0
        else:
            metrics['tissue_snr_original'] = 0
            metrics['tissue_snr_smoothed'] = 0
            metrics['tissue_noise_reduction'] = 0
        
        # 3. Edge preservation (Canny edges sul slice centrale)
        try:
            # Normalizza per Canny
            orig_norm = ((orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min()) * 255).astype(np.uint8)
            smooth_norm = ((smooth_slice - smooth_slice.min()) / (smooth_slice.max() - smooth_slice.min()) * 255).astype(np.uint8)
            
            edges_orig = feature.canny(orig_norm, sigma=1.0)
            edges_smooth = feature.canny(smooth_norm, sigma=1.0)
            
            # Percentuale di edge preservati
            edges_preserved = np.sum(edges_orig & edges_smooth) / np.sum(edges_orig) * 100 if np.sum(edges_orig) > 0 else 0
            edges_lost = np.sum(edges_orig & ~edges_smooth) / np.sum(edges_orig) * 100 if np.sum(edges_orig) > 0 else 0
            
            metrics['edge_preservation_percent'] = edges_preserved
            metrics['edge_loss_percent'] = edges_lost
        except Exception as e:
            metrics['edge_preservation_percent'] = 0
            metrics['edge_loss_percent'] = 0
        
        # 4. Smoothness (gradiente locale - minore = più smooth)
        grad_orig = np.gradient(orig_slice)
        grad_smooth = np.gradient(smooth_slice)
        
        smoothness_orig = np.sqrt(grad_orig[0]**2 + grad_orig[1]**2).mean()
        smoothness_smooth = np.sqrt(grad_smooth[0]**2 + grad_smooth[1]**2).mean()
        
        metrics['gradient_magnitude_original'] = float(smoothness_orig)
        metrics['gradient_magnitude_smoothed'] = float(smoothness_smooth)
        metrics['smoothness_improvement'] = ((smoothness_orig - smoothness_smooth) / smoothness_orig * 100) if smoothness_orig > 0 else 0
        
        # 5. Histogram similarity (Bhattacharyya distance)
        hist_orig, _ = np.histogram(orig_array.flatten(), bins=100, range=(-1024, 3071))
        hist_smooth, _ = np.histogram(smooth_array.flatten(), bins=100, range=(-1024, 3071))
        
        # Normalizza
        hist_orig = hist_orig / hist_orig.sum()
        hist_smooth = hist_smooth / hist_smooth.sum()
        
        # Bhattacharyya coefficient
        bhattacharyya = np.sum(np.sqrt(hist_orig * hist_smooth))
        metrics['histogram_similarity'] = float(bhattacharyya)  # 1 = identico, 0 = completamente diverso
        
        # 6. PSNR (Peak Signal-to-Noise Ratio)
        mse = np.mean((orig_array - smooth_array) ** 2)
        if mse > 0:
            max_pixel = 3071 - (-1024)  # Range HU
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            metrics['psnr'] = float(psnr)
        else:
            metrics['psnr'] = float('inf')
        
        # 7. Differenza media assoluta (MAE)
        mae = np.mean(np.abs(orig_array - smooth_array))
        metrics['mean_absolute_error'] = float(mae)
        
        return metrics
    
    
    def smooth_image(self, input_path, output_folder, patient_id, method='all'):
        """
        Applica smoothing a un'immagine MHD/RAW
        
        Args:
            input_path: Path al file .mhd input
            output_folder: Cartella output
            patient_id: ID paziente
            method: Metodo di smoothing ('median', 'gaussian', 'bilateral', 'curvature', 'all')
        
        Returns:
            dict con {method_name: {'output_path': path, 'metrics': metrics, 'time': time}}
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Smoothing: {patient_id}")
            print(f"Input: {input_path}")
            print(f"{'='*70}")
        
        # Carica immagine
        try:
            image = sitk.ReadImage(str(input_path))
        except Exception as e:
            print(f"✗ Errore caricamento immagine: {e}")
            return {}
        
        # Crea cartella output
        os.makedirs(output_folder, exist_ok=True)
        
        results = {}
        
        # Applica filtri richiesti
        methods_to_apply = []
        
        if method == 'all':
            methods_to_apply = ['median', 'gaussian', 'bilateral', 'curvature']
        else:
            methods_to_apply = [method]
        
        for filter_method in methods_to_apply:
            try:
                if self.verbose:
                    print(f"\n--- Metodo: {filter_method.upper()} ---")
                
                # Misura tempo di esecuzione
                start_time = time.time()
                
                # Applica filtro
                if filter_method == 'median':
                    filtered = self.apply_median_filter(image, radius=1)
                    suffix = '_median'
                
                elif filter_method == 'gaussian':
                    filtered = self.apply_gaussian_filter(image, sigma=0.5)
                    suffix = '_gaussian'
                
                elif filter_method == 'bilateral':
                    filtered = self.apply_bilateral_filter(image, domain_sigma=2.0, range_sigma=50.0)
                    suffix = '_bilateral'
                
                elif filter_method == 'curvature':
                    filtered = self.apply_curvature_flow_filter(image, iterations=5)
                    suffix = '_curvature'
                
                else:
                    print(f"⚠️  Metodo sconosciuto: {filter_method}")
                    continue
                
                execution_time = time.time() - start_time
                
                # Calcola metriche di qualità
                if self.verbose:
                    print(f"  Calcolando metriche di qualità...")
                
                metrics = self.calculate_image_quality_metrics(image, filtered, filter_method)
                metrics['execution_time_seconds'] = execution_time
                
                # Salva immagine filtrata
                output_path = os.path.join(output_folder, f"{patient_id}{suffix}.mhd")
                sitk.WriteImage(filtered, output_path)
                
                if self.verbose:
                    print(f"  ✅ Salvata: {output_path}")
                    print(f"  ⏱️  Tempo: {execution_time:.2f}s")
                    print(f"  📊 SNR polmone: {metrics['lung_snr_original']:.2f} → {metrics['lung_snr_smoothed']:.2f}")
                    print(f"  📊 Riduzione rumore: {metrics['lung_noise_reduction']:.1f}%")
                    print(f"  📊 Preservazione bordi: {metrics['edge_preservation_percent']:.1f}%")
                
                results[filter_method] = {
                    'output_path': output_path,
                    'metrics': metrics,
                    'time': execution_time
                }
                
            except Exception as e:
                print(f"  ✗ Errore applicando {filter_method}: {e}")
        
        return results
    
    
    def analyze_osic_dataset(self, input_folder):
        """
        Analizza tutto il dataset OSIC per identificare scan con kernel sharp
        
        Args:
            input_folder: Cartella con sottocartelle paziente contenenti DICOM
        
        Returns:
            dict con statistiche e lista pazienti sharp
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("ANALISI KERNEL OSIC DATASET")
            print(f"{'='*70}")
            print(f"Input: {input_folder}")
        
        input_path = Path(input_folder)
        
        if not input_path.exists():
            print(f"✗ Cartella non trovata")
            return None
        
        results = {
            'sharp_patients': [],
            'smooth_patients': [],
            'unknown_patients': [],
        }
        
        # Itera su tutte le sottocartelle
        for patient_folder in input_path.iterdir():
            if not patient_folder.is_dir():
                continue
            
            if patient_folder.name.startswith('.'):
                continue
            
            patient_id = patient_folder.name
            
            # Estrai kernel
            kernel = self.extract_kernel_from_dicom(patient_folder)
            
            if kernel is None:
                results['unknown_patients'].append({
                    'id': patient_id,
                    'folder': str(patient_folder),
                })
                continue
            
            # Determina se è sharp
            is_sharp, score = self.is_sharp_kernel(kernel)
            
            patient_info = {
                'id': patient_id,
                'folder': str(patient_folder),
                'kernel': kernel,
                'sharpness_score': score,
            }
            
            if is_sharp:
                results['sharp_patients'].append(patient_info)
            else:
                results['smooth_patients'].append(patient_info)
        
        # Statistiche
        total = len(results['sharp_patients']) + len(results['smooth_patients']) + len(results['unknown_patients'])
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("RISULTATI ANALISI")
            print(f"{'='*70}")
            print(f"\nTotale pazienti: {total}")
            print(f"  - Sharp kernels: {len(results['sharp_patients'])}")
            print(f"  - Smooth kernels: {len(results['smooth_patients'])}")
            print(f"  - Unknown: {len(results['unknown_patients'])}")
            
            if len(results['sharp_patients']) > 0:
                print(f"\n--- PAZIENTI CON KERNEL SHARP ---")
                for p in sorted(results['sharp_patients'], key=lambda x: x['sharpness_score'], reverse=True):
                    print(f"  {p['id']:30s} | {p['kernel']:15s} | Score: {p['sharpness_score']}")
        
        return results
    
    
    def process_sharp_scans(self, input_folder, mhd_folder, output_folder, method='all'):
        """
        Pipeline completa:
        1. Identifica scan con kernel sharp
        2. Le converte in MHD (se non già fatto)
        3. Applica smoothing
        4. Salva risultati
        
        Args:
            input_folder: Cartella con DICOM originali
            mhd_folder: Cartella con file MHD/RAW già convertiti
            output_folder: Cartella output per versioni smoothate
            method: Metodo di smoothing ('median', 'gaussian', 'bilateral', 'curvature', 'all')
        
        Returns:
            Report con risultati
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("PIPELINE SMOOTHING KERNEL SHARP")
            print(f"{'='*70}")
        
        # Step 1: Identifica scan sharp
        analysis = self.analyze_osic_dataset(input_folder)
        
        if analysis is None:
            return None
        
        sharp_patients = analysis['sharp_patients']
        
        if len(sharp_patients) == 0:
            print("\n✅ Nessun paziente con kernel sharp trovato")
            return analysis
        
        print(f"\n{'='*70}")
        print(f"SMOOTHING {len(sharp_patients)} SCAN CON KERNEL SHARP")
        print(f"{'='*70}")
        
        # Step 2: Applica smoothing
        os.makedirs(output_folder, exist_ok=True)
        
        processed = []
        
        for idx, patient_info in enumerate(sharp_patients, 1):
            patient_id = patient_info['id']
            kernel = patient_info['kernel']
            
            print(f"\n[{idx}/{len(sharp_patients)}] {patient_id} (kernel: {kernel})")
            
            # Cerca file MHD
            mhd_file = os.path.join(mhd_folder, f"{patient_id}.mhd")
            
            if not os.path.exists(mhd_file):
                # Prova con _normalized
                mhd_file = os.path.join(mhd_folder, f"{patient_id}_normalized.mhd")
            
            if not os.path.exists(mhd_file):
                print(f"  ⚠️  File MHD non trovato, skip")
                continue
            
            # Applica smoothing
            results = self.smooth_image(mhd_file, output_folder, patient_id, method=method)
            
            if len(results) > 0:
                patient_info['smoothed_files'] = results
                processed.append(patient_info)
                
                # Aggiungi metriche alla lista globale per analisi comparativa
                for method_name, method_results in results.items():
                    if 'metrics' in method_results:
                        metrics_entry = method_results['metrics'].copy()
                        metrics_entry['patient_id'] = patient_id
                        metrics_entry['kernel'] = kernel
                        if not hasattr(self, 'all_metrics'):
                            self.all_metrics = []
                        self.all_metrics.append(metrics_entry)
        
        # Report finale
        print(f"\n{'='*70}")
        print("RIEPILOGO")
        print(f"{'='*70}")
        print(f"Scan sharp identificate: {len(sharp_patients)}")
        print(f"Scan processate: {len(processed)}")
        print(f"\nFile salvati in: {output_folder}")
        
        # Genera visualizzazioni comparative
        visualizations = {}
        if hasattr(self, 'all_metrics') and len(self.all_metrics) > 0:
            if self.verbose:
                print(f"\n{'='*70}")
                print("GENERAZIONE VISUALIZZAZIONI PER TESI")
                print(f"{'='*70}")
            
            visualizations = self.generate_comparison_visualizations(
                self.all_metrics, 
                output_folder
            )
        
        # Salva report JSON
        report_path = os.path.join(output_folder, 'smoothing_report.json')
        
        report = {
            'analysis': analysis,
            'processed': processed,
            'visualizations': visualizations,
            'settings': {
                'input_folder': input_folder,
                'mhd_folder': mhd_folder,
                'output_folder': output_folder,
                'method': method,
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Report salvato: {report_path}")
        
        return report
    
    
    def generate_comparison_visualizations(self, metrics_data, output_folder):
        """
        Genera visualizzazioni comparative per giustificare la scelta di Gaussian
        
        Args:
            metrics_data: Lista di dict con metriche per ogni paziente/metodo
            output_folder: Cartella dove salvare i grafici
        
        Returns:
            dict con path ai grafici generati
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("GENERAZIONE VISUALIZZAZIONI COMPARATIVE")
            print(f"{'='*70}")
        
        viz_folder = os.path.join(output_folder, 'comparative_analysis')
        os.makedirs(viz_folder, exist_ok=True)
        
        # Converti in DataFrame per analisi
        df = pd.DataFrame(metrics_data)
        
        if len(df) == 0:
            print("⚠️  Nessun dato da visualizzare")
            return {}
        
        visualizations = {}
        
        # Configurazione stile
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = {
            'median': '#e74c3c',
            'gaussian': '#2ecc71',
            'bilateral': '#3498db',
            'curvature': '#f39c12'
        }
        
        # ========== GRAFICO 1: Confronto SNR ==========
        if self.verbose:
            print("  Generando confronto SNR...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # SNR polmone
        methods = df['method'].unique()
        for method in methods:
            method_data = df[df['method'] == method]
            x = range(len(method_data))
            y_orig = method_data['lung_snr_original']
            y_smooth = method_data['lung_snr_smoothed']
            
            color = colors.get(method, '#95a5a6')
            ax1.scatter(y_orig, y_smooth, label=method.capitalize(), 
                       color=color, s=100, alpha=0.7, edgecolors='black')
        
        # Linea diagonale (no change)
        max_val = max(df['lung_snr_original'].max(), df['lung_snr_smoothed'].max())
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='No change')
        
        ax1.set_xlabel('SNR Originale (polmone)', fontsize=12, weight='bold')
        ax1.set_ylabel('SNR Dopo Smoothing (polmone)', fontsize=12, weight='bold')
        ax1.set_title('Signal-to-Noise Ratio: Polmone', fontsize=14, weight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # SNR tessuto
        for method in methods:
            method_data = df[df['method'] == method]
            y_orig = method_data['tissue_snr_original']
            y_smooth = method_data['tissue_snr_smoothed']
            
            color = colors.get(method, '#95a5a6')
            ax2.scatter(y_orig, y_smooth, label=method.capitalize(),
                       color=color, s=100, alpha=0.7, edgecolors='black')
        
        max_val_tissue = max(df['tissue_snr_original'].max(), df['tissue_snr_smoothed'].max())
        ax2.plot([0, max_val_tissue], [0, max_val_tissue], 'k--', alpha=0.3, label='No change')
        
        ax2.set_xlabel('SNR Originale (tessuto)', fontsize=12, weight='bold')
        ax2.set_ylabel('SNR Dopo Smoothing (tessuto)', fontsize=12, weight='bold')
        ax2.set_title('Signal-to-Noise Ratio: Tessuto Molle', fontsize=14, weight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        snr_path = os.path.join(viz_folder, 'snr_comparison.png')
        plt.savefig(snr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations['snr_comparison'] = snr_path
        if self.verbose:
            print(f"    ✓ Salvato: {snr_path}")
        
        # ========== GRAFICO 2: Riduzione rumore vs Preservazione bordi ==========
        if self.verbose:
            print("  Generando trade-off rumore/bordi...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for method in methods:
            method_data = df[df['method'] == method]
            x = method_data['lung_noise_reduction']
            y = method_data['edge_preservation_percent']
            
            color = colors.get(method, '#95a5a6')
            ax.scatter(x, y, label=method.capitalize(), color=color, 
                      s=150, alpha=0.7, edgecolors='black', linewidths=2)
            
            # Aggiungi media
            mean_x = x.mean()
            mean_y = y.mean()
            ax.scatter(mean_x, mean_y, color=color, s=400, marker='*', 
                      edgecolors='black', linewidths=2, zorder=10)
            ax.annotate(f'{method.capitalize()}\n({mean_x:.1f}%, {mean_y:.1f}%)',
                       xy=(mean_x, mean_y), xytext=(10, 10),
                       textcoords='offset points', fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))
        
        # Zona ideale (alto-destra)
        ax.axhline(y=85, color='green', linestyle='--', alpha=0.3, linewidth=2)
        ax.axvline(x=40, color='green', linestyle='--', alpha=0.3, linewidth=2)
        ax.fill_between([40, 100], 85, 100, alpha=0.1, color='green', label='Zona ideale')
        
        ax.set_xlabel('Riduzione Rumore (%)', fontsize=13, weight='bold')
        ax.set_ylabel('Preservazione Bordi (%)', fontsize=13, weight='bold')
        ax.set_title('Trade-off: Riduzione Rumore vs Preservazione Bordi', 
                    fontsize=15, weight='bold', pad=20)
        ax.legend(fontsize=11, loc='lower left')
        ax.grid(alpha=0.3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        
        plt.tight_layout()
        tradeoff_path = os.path.join(viz_folder, 'noise_vs_edges_tradeoff.png')
        plt.savefig(tradeoff_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations['tradeoff'] = tradeoff_path
        if self.verbose:
            print(f"    ✓ Salvato: {tradeoff_path}")
        
        # ========== GRAFICO 3: Metriche aggregate per metodo ==========
        if self.verbose:
            print("  Generando metriche aggregate...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('lung_noise_reduction', 'Riduzione Rumore Polmone (%)', 'higher'),
            ('edge_preservation_percent', 'Preservazione Bordi (%)', 'higher'),
            ('smoothness_improvement', 'Miglioramento Smoothness (%)', 'higher'),
            ('histogram_similarity', 'Similarità Istogramma', 'higher'),
            ('execution_time_seconds', 'Tempo Esecuzione (s)', 'lower'),
            ('psnr', 'PSNR (dB)', 'higher')
        ]
        
        for idx, (metric, title, better) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if metric not in df.columns:
                ax.axis('off')
                continue
            
            # Boxplot per ogni metodo
            data_to_plot = []
            labels_to_plot = []
            colors_to_plot = []
            
            for method in methods:
                method_data = df[df['method'] == method][metric].dropna()
                if len(method_data) > 0:
                    data_to_plot.append(method_data)
                    labels_to_plot.append(method.capitalize())
                    colors_to_plot.append(colors.get(method, '#95a5a6'))
            
            if len(data_to_plot) > 0:
                bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                               notch=True, showmeans=True)
                
                for patch, color in zip(bp['boxes'], colors_to_plot):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Evidenzia il migliore
                means = [np.mean(d) for d in data_to_plot]
                if better == 'higher':
                    best_idx = np.argmax(means)
                else:
                    best_idx = np.argmin(means)
                
                bp['boxes'][best_idx].set_linewidth(3)
                bp['boxes'][best_idx].set_edgecolor('gold')
                
                ax.set_ylabel(title.split('(')[0], fontsize=10, weight='bold')
                ax.set_title(title, fontsize=11, weight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # Rotazione label se necessario
                if idx >= 3:
                    ax.set_xticklabels(labels_to_plot, rotation=15, ha='right')
        
        plt.suptitle('Confronto Metriche di Qualità per Metodo di Smoothing',
                    fontsize=16, weight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        metrics_path = os.path.join(viz_folder, 'metrics_by_method.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations['metrics_aggregate'] = metrics_path
        if self.verbose:
            print(f"    ✓ Salvato: {metrics_path}")
        
        # ========== GRAFICO 4: Tabella riassuntiva con ranking ==========
        if self.verbose:
            print("  Generando tabella riassuntiva...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        # Calcola medie per metodo
        summary_data = []
        for method in methods:
            method_data = df[df['method'] == method]
            summary_data.append({
                'Metodo': method.capitalize(),
                'SNR Polmone': f"{method_data['lung_snr_smoothed'].mean():.2f}",
                'Riduzione Rumore (%)': f"{method_data['lung_noise_reduction'].mean():.1f}",
                'Preservazione Bordi (%)': f"{method_data['edge_preservation_percent'].mean():.1f}",
                'Smoothness (%)': f"{method_data['smoothness_improvement'].mean():.1f}",
                'Similarità Hist.': f"{method_data['histogram_similarity'].mean():.3f}",
                'Tempo (s)': f"{method_data['execution_time_seconds'].mean():.2f}",
                'PSNR (dB)': f"{method_data['psnr'].mean():.1f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Crea tabella
        table_data = [df_summary.columns.tolist()] + df_summary.values.tolist()
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 0.9])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Formattazione header
        for i in range(len(df_summary.columns)):
            cell = table[(0, i)]
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        
        # Formattazione righe (colora metodo migliore)
        for i in range(1, len(table_data)):
            method = table_data[i][0].lower()
            color = colors.get(method, '#ecf0f1')
            
            for j in range(len(df_summary.columns)):
                cell = table[(i, j)]
                if j == 0:
                    cell.set_facecolor(color)
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#ffffff')
        
        # Evidenzia Gaussian
        for i in range(1, len(table_data)):
            if 'gaussian' in table_data[i][0].lower():
                for j in range(len(df_summary.columns)):
                    cell = table[(i, j)]
                    cell.set_linewidth(3)
                    cell.set_edgecolor('gold')
        
        ax.set_title('Tabella Riassuntiva: Confronto Metodi di Smoothing\n(Gaussian evidenziato in oro)',
                    fontsize=14, weight='bold', pad=20)
        
        plt.tight_layout()
        summary_path = os.path.join(viz_folder, 'summary_table.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations['summary_table'] = summary_path
        if self.verbose:
            print(f"    ✓ Salvato: {summary_path}")
        
        # ========== Salva metriche in CSV ==========
        csv_path = os.path.join(viz_folder, 'smoothing_metrics_detailed.csv')
        df.to_csv(csv_path, index=False)
        visualizations['metrics_csv'] = csv_path
        
        if self.verbose:
            print(f"    ✓ Metriche salvate in CSV: {csv_path}")
            print(f"\n  ✅ Tutte le visualizzazioni generate in: {viz_folder}")
        
        return visualizations


# ==================== MAIN ====================

def main():
    """
    Main function per smoothing kernel sharp
    """
    print(f"\n{'='*70}")
    print("SHARP KERNEL SMOOTHER - OSIC DATASET")
    print(f"{'='*70}")
    
    # CONFIGURAZIONE
    input_dicom_folder = r"X:\Francesca Saglimbeni\tesi\datasets\datasetErratiCambiati\DICOM"
    mhd_folder = r"X:\Francesca Saglimbeni\tesi\datasets\datasetErratiCambiati\mhd+raw"
    output_folder = r"X:\Francesca Saglimbeni\tesi\cancellare\OSIC_smoothed"
    
    # Opzioni:
    # - 'median': veloce, buono per noise salt-and-pepper
    # - 'gaussian': veloce, smoothing uniforme
    # - 'bilateral': lento ma ottimo, preserva bordi
    # - 'curvature': lento, specifico per medical imaging
    # - 'all': applica tutti i metodi per confronto
    smoothing_method = 'all'  # Cambia qui per testare metodi diversi
    
    # Crea smoother
    smoother = SharpKernelSmoother(verbose=True)
    
    # Opzione 1: ANALISI SOLA (per vedere chi ha kernel sharp)
    # smoother.analyze_osic_dataset(input_dicom_folder)
    
    # Opzione 2: PIPELINE COMPLETA (analisi + smoothing)
    report = smoother.process_sharp_scans(
        input_folder=input_dicom_folder,
        mhd_folder=mhd_folder,
        output_folder=output_folder,
        method=smoothing_method
    )
    
    if report:
        print(f"\n{'='*70}")
        print("✅ PIPELINE COMPLETATA")
        print(f"{'='*70}")
        print(f"\nProssimi passi:")
        print(f"1. Testa le scan smoothate nella pipeline di segmentazione")
        print(f"2. Confronta risultati con originali")
        print(f"3. Scegli il metodo migliore (median/gaussian/bilateral/curvature)")
    else:
        print(f"\n✗ Pipeline fallita")


if __name__ == "__main__":
    main()
