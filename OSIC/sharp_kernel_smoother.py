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
    
    
    def smooth_image(self, input_path, output_folder, patient_id, method='all'):
        """
        Applica smoothing a un'immagine MHD/RAW
        
        Args:
            input_path: Path al file .mhd input
            output_folder: Cartella output
            patient_id: ID paziente
            method: Metodo di smoothing ('median', 'gaussian', 'bilateral', 'curvature', 'all')
        
        Returns:
            dict con {method_name: output_path}
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
                
                # Salva immagine filtrata
                output_path = os.path.join(output_folder, f"{patient_id}{suffix}.mhd")
                sitk.WriteImage(filtered, output_path)
                
                if self.verbose:
                    print(f"  ✅ Salvata: {output_path}")
                
                results[filter_method] = output_path
                
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
        
        # Report finale
        print(f"\n{'='*70}")
        print("RIEPILOGO")
        print(f"{'='*70}")
        print(f"Scan sharp identificate: {len(sharp_patients)}")
        print(f"Scan processate: {len(processed)}")
        print(f"\nFile salvati in: {output_folder}")
        
        # Salva report JSON
        report_path = os.path.join(output_folder, 'smoothing_report.json')
        
        report = {
            'analysis': analysis,
            'processed': processed,
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


# ==================== MAIN ====================

def main():
    """
    Main function per smoothing kernel sharp
    """
    print(f"\n{'='*70}")
    print("SHARP KERNEL SMOOTHER - OSIC DATASET")
    print(f"{'='*70}")
    
    # CONFIGURAZIONE
    input_dicom_folder = r"X:\Francesca Saglimbeni\tesi\daUsareErrr\DICOM"
    mhd_folder = r"X:\Francesca Saglimbeni\tesi\daUsareErrr\mhd+raw"
    output_folder = r"X:\Francesca Saglimbeni\tesi\daUsareErrr\OSIC_smoothed"
    
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
