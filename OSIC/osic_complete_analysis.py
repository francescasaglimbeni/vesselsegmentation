"""
OSIC Dataset Analysis and Conversion Pipeline
===============================================

Questo script gestisce:
1. Analisi compatibilità CT scan (parametri fisici)
2. Conversione DICOM → MHD/RAW con validazione completa
3. Copia delle scan valide in cartella separata
4. Generazione report dettagliati

Autore: Pipeline OSIC
Data: 2026-01-14
"""

import os
import shutil
import json
import numpy as np
import SimpleITK as sitk
import pydicom
from pathlib import Path
from datetime import datetime
import hashlib
import warnings
import matplotlib
matplotlib.use('Agg')  # Backend non interattivo per salvare grafici
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Sopprime i warning di SimpleITK (gestiamo noi i check)
warnings.filterwarnings('ignore', category=UserWarning, module='SimpleITK')


def convert_numpy_types(obj):
    """
    Converte ricorsivamente tipi numpy in tipi Python nativi per JSON serialization
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class OSICAnalysisPipeline:
    """
    Pipeline completa per analisi, conversione e validazione OSIC dataset
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # PARAMETRI PIÙ FLESSIBILI (come richiesto)
        self.criteria = {
            # Numero di slices: accettiamo da 250 in su (prima era 350)
            'min_slices': 300,
            'max_slices': 1500,
            
            # Slice thickness: più permissivo (prima era 0.6-1.5)
            'min_slice_thickness': 0.5,
            'max_slice_thickness': 1.5,
            
            # XY resolution: range più ampio (prima era 0.4-1.0)
            'min_xy_resolution': 0.3,
            'max_xy_resolution': 1.1,
        }
        
        # Percorsi configurabili
        self.input_folder = "X:/Francesca Saglimbeni/tesi/datasets/dataset_OSIC"
        self.output_folder = "X:/Francesca Saglimbeni/tesi/datasets/OSIC_touse"
        
        # Cartella per scan VALIDE (solo copie, originali restano intatti)
        self.validated_folder = "X:/Francesca Saglimbeni/tesi/datasets/OSIC_validated"
        
        # Statistiche globali
        self.stats = {
            'total_patients': 0,
            'compatible': 0,
            'incompatible': 0,
            'converted': 0,
            'conversion_failed': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'errors': 0,
        }
        
        # Raccolta metriche per TUTTI i pazienti (per analisi pre-filtro)
        self.all_metrics = []
        
        # Conteggio motivazioni incompatibilità
        self.incompatibility_reasons = defaultdict(int)
    
    
    # ==================== SEZIONE 1: GESTIONE SCAN VALIDE ====================
    
    def _copy_to_validated(self, patient_folder, patient_id):
        """
        Copia la cartella paziente VALIDA nella cartella validated
        (originale resta nella cartella input senza modifiche)
        
        Args:
            patient_folder: Path della cartella DICOM del paziente
            patient_id: ID del paziente
        
        Returns:
            Path della cartella copiata
        """
        # Crea cartella validated se non esiste
        os.makedirs(self.validated_folder, exist_ok=True)
        
        # Destinazione
        dest_path = os.path.join(self.validated_folder, patient_id)
        
        # Copia cartella DICOM (NON sposta)
        if os.path.exists(patient_folder):
            try:
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(str(patient_folder), dest_path)
                
                # Crea file di log con info validazione
                log_file = os.path.join(dest_path, "_VALIDATED.txt")
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"VALIDATED SCAN\n")
                    f.write(f"Patient ID: {patient_id}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                
                if self.verbose:
                    print(f"  ✓ Copiato in validated/")
                
                return dest_path
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Errore nella copia: {e}")
                return None
        
        return None
    
    
    def _remove_output_files(self, patient_id):
        """
        Rimuove file MHD/RAW dalla cartella output se generati
        
        Args:
            patient_id: ID del paziente
        """
        mhd_file = os.path.join(self.output_folder, f"{patient_id}.mhd")
        raw_file = os.path.join(self.output_folder, f"{patient_id}.raw")
        normalized_mhd = os.path.join(self.output_folder, f"{patient_id}_normalized.mhd")
        normalized_raw = os.path.join(self.output_folder, f"{patient_id}_normalized.raw")
        
        for file_path in [mhd_file, raw_file, normalized_mhd, normalized_raw]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠️  Errore rimozione {file_path}: {e}")
    
    
    # ==================== SEZIONE 3: ANALISI E FILTRAGGIO DICOM ====================
    
    def _check_slice_uniformity(self, dicom_filenames):
        """
        Verifica che le slice siano uniformemente distanziate
        
        Args:
            dicom_filenames: Lista di path ai file DICOM ordinati
        
        Returns:
            (is_uniform, spacing_std, max_gap)
            - is_uniform: True se le slice sono uniformi
            - spacing_std: Deviazione standard della spaziatura
            - max_gap: Gap massimo tra slice consecutive
        """
        try:
            # Leggi posizioni Z (ImagePositionPatient) da tutti i file
            z_positions = []
            
            for filename in dicom_filenames:
                try:
                    ds = pydicom.dcmread(filename, stop_before_pixels=True)
                    if 'ImagePositionPatient' in ds:
                        z_pos = float(ds.ImagePositionPatient[2])
                        z_positions.append(z_pos)
                except:
                    continue
            
            if len(z_positions) < 2:
                # Non abbastanza slice per verificare
                return True, 0.0, 0.0
            
            # Ordina posizioni Z
            z_positions = sorted(z_positions)
            
            # Calcola distanze tra slice consecutive
            slice_spacings = []
            for i in range(len(z_positions) - 1):
                spacing = abs(z_positions[i+1] - z_positions[i])
                slice_spacings.append(spacing)
            
            if len(slice_spacings) == 0:
                return True, 0.0, 0.0
            
            # Calcola statistiche
            mean_spacing = np.mean(slice_spacings)
            std_spacing = np.std(slice_spacings)
            max_gap = np.max(slice_spacings)
            min_gap = np.min(slice_spacings)
            
            # Criteri di uniformità:
            # 1. La deviazione standard deve essere molto piccola (< 10% del mean)
            # 2. Il gap massimo non deve essere troppo diverso dal minimo
            # 3. Non devono esserci gap > 2x il mean spacing (indica slice mancante)
            
            is_uniform = True
            
            # Check 1: Std dev < 10% del mean
            if mean_spacing > 0 and (std_spacing / mean_spacing) > 0.10:
                is_uniform = False
            
            # Check 2: Max gap non deve essere > 2x mean spacing (slice mancante)
            if mean_spacing > 0 and max_gap > (mean_spacing * 2.0):
                is_uniform = False
            
            # Check 3: Std dev assoluto non deve essere > 1mm
            if std_spacing > 1.0:
                is_uniform = False
            
            return is_uniform, float(std_spacing), float(max_gap)
            
        except Exception as e:
            # In caso di errore, assumiamo non uniforme
            if self.verbose:
                print(f"    ⚠️  Errore verifica uniformità slice: {e}")
            return False, 999.0, 999.0
    
    
    def find_dicom_patients(self):
        """
        Trova tutte le cartelle paziente con DICOM
        
        Returns:
            list di tuple (patient_id, patient_folder_path)
        """
        input_path = Path(self.input_folder)
        
        if not input_path.exists():
            if self.verbose:
                print(f"✗ Input folder non trovata: {self.input_folder}")
            return []
        
        patients = []
        
        for item in input_path.iterdir():
            if not item.is_dir():
                continue
            
            # Salta cartelle speciali
            if item.name.startswith('.') or item.name in ['removed_low_filecount', 'incompatible', 'output']:
                continue
            
            # Verifica se contiene file DICOM
            dcm_files = list(item.glob("*.dcm"))
            if len(dcm_files) > 0:
                patients.append((item.name, item))
        
        return patients
    
    
    def analyze_dicom_series(self, dicom_folder):
        """
        Analizza una serie DICOM e restituisce metriche complete
        
        Returns:
            (metrics, is_compatible, issues)
        """
        try:
            # Carica serie DICOM
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))
            
            if not dicom_names or len(dicom_names) == 0:
                return None, False, ["Nessun file DICOM trovato"]
            
            # Leggi metadati dal primo file
            ds = pydicom.dcmread(dicom_names[0])
            
            # Estrai parametri tecnici
            num_slices = len(dicom_names)
            slice_thickness = float(ds.SliceThickness) if 'SliceThickness' in ds else None
            pixel_spacing = ds.PixelSpacing if 'PixelSpacing' in ds else None
            xy_resolution = float(pixel_spacing[0]) if pixel_spacing else None
            
            # CHECK CRITICO: Verifica uniformità delle slice PRIMA di caricare l'intera immagine
            slice_uniformity_ok, slice_spacing_std, max_gap = self._check_slice_uniformity(dicom_names)
            
            # Carica immagine completa
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            image_array = sitk.GetArrayFromImage(image)
            
            # Crea dizionario metriche
            metrics = {
                'num_slices': num_slices,
                'slice_thickness': slice_thickness,
                'xy_resolution': xy_resolution,
                'image_shape': image_array.shape,
                'spacing': image.GetSpacing(),
                'origin': image.GetOrigin(),
                'direction': image.GetDirection(),
                'dicom_count': len(dicom_names),
                'slice_uniformity': {
                    'is_uniform': slice_uniformity_ok,
                    'spacing_std': slice_spacing_std,
                    'max_gap': max_gap,
                }
            }
            
            # Verifica compatibilità con criteri FLESSIBILI
            is_compatible, issues = self._check_compatibility(metrics)
            
            return metrics, is_compatible, issues
            
        except Exception as e:
            return None, False, [f"Errore durante analisi: {str(e)}"]
    
    
    def _check_compatibility(self, metrics):
        """
        Verifica compatibilità con criteri FLESSIBILI
        
        Returns:
            (is_compatible, issues_list)
        """
        issues = []
        
        # Check 0: CRITICO - Uniformità delle slice (PRIORITÀ MASSIMA)
        if not metrics['slice_uniformity']['is_uniform']:
            max_gap = metrics['slice_uniformity']['max_gap']
            spacing_std = metrics['slice_uniformity']['spacing_std']
            issues.append(f"Slice non uniformi o mancanti (max gap: {max_gap:.2f}mm, std: {spacing_std:.2f}mm)")
        
        # Check 1: Numero slices (PIÙ PERMISSIVO: 250+ invece di 350+)
        if metrics['num_slices'] < self.criteria['min_slices']:
            issues.append(f"Troppe poche slices: {metrics['num_slices']} < {self.criteria['min_slices']}")
        
        # Check 2: Slice thickness (PIÙ PERMISSIVO: 0.5-2.0mm invece di 0.6-1.5mm)
        if metrics['slice_thickness'] is not None:
            if metrics['slice_thickness'] < self.criteria['min_slice_thickness']:
                issues.append(f"Slice troppo sottile: {metrics['slice_thickness']:.3f} < {self.criteria['min_slice_thickness']}")
            elif metrics['slice_thickness'] > self.criteria['max_slice_thickness']:
                issues.append(f"Slice troppo spessa: {metrics['slice_thickness']:.3f} > {self.criteria['max_slice_thickness']}")
        
        # Check 3: XY resolution (PIÙ PERMISSIVO: 0.3-1.2mm invece di 0.4-1.0mm)
        if metrics['xy_resolution'] is not None:
            if metrics['xy_resolution'] < self.criteria['min_xy_resolution']:
                issues.append(f"Risoluzione XY troppo alta: {metrics['xy_resolution']:.3f} < {self.criteria['min_xy_resolution']}")
            elif metrics['xy_resolution'] > self.criteria['max_xy_resolution']:
                issues.append(f"Risoluzione XY troppo bassa: {metrics['xy_resolution']:.3f} > {self.criteria['max_xy_resolution']}")
        
        is_compatible = len(issues) == 0
        
        return is_compatible, issues
    
    
    # ==================== SEZIONE 3: CONVERSIONE CON VALIDAZIONE ====================
    
    def convert_dicom_to_mhd(self, dicom_folder, output_filename, original_metrics):
        """
        Converte DICOM → MHD/RAW con validazione completa
        
        Returns:
            (success, output_path, validation_report)
        """
        try:
            output_path = os.path.join(self.output_folder, output_filename)
            
            # 1. Conversione
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))
            reader.SetFileNames(dicom_names)
            
            dicom_image = reader.Execute()
            
            # Salva in formato MHD/RAW
            sitk.WriteImage(dicom_image, output_path)
            
            # 2. VALIDAZIONE: rileggi il file salvato e confronta
            validation_report = self._validate_conversion(
                dicom_image,
                output_path,
                original_metrics
            )
            
            success = validation_report['all_checks_passed']
            
            return success, output_path, validation_report
            
        except Exception as e:
            validation_report = {
                'all_checks_passed': False,
                'error': str(e)
            }
            return False, None, validation_report
    
    
    def _validate_conversion(self, original_image, mhd_path, original_metrics):
        """
        Valida che la conversione sia al 100% corretta
        
        Controlla:
        - File MHD e RAW esistono
        - Dimensioni corrette
        - Spacing preservato
        - HU preservate (checksum pixel-perfect)
        - Metadati preservati
        
        Returns:
            dict con risultati validazione
        """
        report = {
            'all_checks_passed': True,
            'checks': {}
        }
        
        # Check 1: File esistono
        mhd_file = Path(mhd_path)
        raw_file = mhd_file.with_suffix('.raw')
        
        report['checks']['mhd_exists'] = mhd_file.exists()
        report['checks']['raw_exists'] = raw_file.exists()
        
        if not (mhd_file.exists() and raw_file.exists()):
            report['all_checks_passed'] = False
            return report
        
        # Check 2: Ricarica immagine e confronta
        try:
            reloaded_image = sitk.ReadImage(str(mhd_path))
            
            # Check dimensioni
            orig_size = original_image.GetSize()
            reload_size = reloaded_image.GetSize()
            report['checks']['size_match'] = (orig_size == reload_size)
            
            # Check spacing
            orig_spacing = original_image.GetSpacing()
            reload_spacing = reloaded_image.GetSpacing()
            spacing_diff = [abs(o - r) for o, r in zip(orig_spacing, reload_spacing)]
            report['checks']['spacing_match'] = all(d < 1e-6 for d in spacing_diff)
            report['checks']['spacing_diff'] = spacing_diff
            
            # Check origin
            orig_origin = original_image.GetOrigin()
            reload_origin = reloaded_image.GetOrigin()
            origin_diff = [abs(o - r) for o, r in zip(orig_origin, reload_origin)]
            report['checks']['origin_match'] = all(d < 1e-6 for d in origin_diff)
            
            # Check pixel data (MD5 checksum)
            orig_array = sitk.GetArrayFromImage(original_image)
            reload_array = sitk.GetArrayFromImage(reloaded_image)
            
            orig_md5 = hashlib.md5(orig_array.tobytes()).hexdigest()
            reload_md5 = hashlib.md5(reload_array.tobytes()).hexdigest()
            
            report['checks']['pixel_data_match'] = (orig_md5 == reload_md5)
            report['checks']['orig_md5'] = orig_md5
            report['checks']['reload_md5'] = reload_md5
            
            # Check HU statistics
            orig_hu_mean = float(np.mean(orig_array))
            reload_hu_mean = float(np.mean(reload_array))
            hu_mean_diff = abs(orig_hu_mean - reload_hu_mean)
            
            report['checks']['hu_mean_diff'] = hu_mean_diff
            report['checks']['hu_preserved'] = (hu_mean_diff < 0.01)
            
            # Check file sizes
            mhd_size = mhd_file.stat().st_size
            raw_size = raw_file.stat().st_size
            
            expected_raw_size = np.prod(orig_size) * orig_array.itemsize
            
            report['checks']['mhd_size'] = mhd_size
            report['checks']['raw_size'] = raw_size
            report['checks']['expected_raw_size'] = expected_raw_size
            report['checks']['raw_size_match'] = (raw_size == expected_raw_size)
            
            # Verifica generale
            critical_checks = [
                'size_match',
                'spacing_match',
                'pixel_data_match',
                'raw_size_match'
            ]
            
            for check in critical_checks:
                if not report['checks'].get(check, False):
                    report['all_checks_passed'] = False
            
        except Exception as e:
            report['all_checks_passed'] = False
            report['checks']['reload_error'] = str(e)
        
        return report
    
    
    # ==================== SEZIONE 4: NORMALIZZAZIONE HU ====================
    
    def normalize_hu(self, image_path, output_path=None, use_carve_reference=True):
        """
        Normalizza HU usando riferimento CARVE14 (se disponibile)
        
        CONSERVATIVA: Applica normalizzazione solo se:
        1. La correzione è ragionevole (non troppo estrema)
        2. La scan ha una struttura HU sensata (picchi aria/polmone visibili)
        3. La correzione migliora effettivamente la qualità
        
        Returns:
            (normalized_path, correction_applied, correction_params)
        """
        if self.verbose:
            print(f"\n  Analisi HU...")
        
        # Carica immagine
        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image)
        
        # Analizza HU
        hu_stats = self._calculate_hu_statistics(image_array)
        
        if self.verbose:
            print(f"    HU mean: {hu_stats['mean']:.1f}, air peak: {hu_stats['air_peak']:.1f}")
        
        # STEP 1: Verifica che la scan abbia una struttura interna sensata
        has_valid_structure, structure_issues = self._validate_hu_structure(hu_stats)
        
        if not has_valid_structure:
            if self.verbose:
                print(f"    ⚠️  Struttura HU anomala - NON normalizzo per sicurezza")
                for issue in structure_issues:
                    print(f"       - {issue}")
            return image_path, False, {'skipped': True, 'reason': 'invalid_structure', 'issues': structure_issues}
        
        # STEP 2: Determina se serve correzione
        needs_correction = False
        correction_type = None
        
        if use_carve_reference and self.carve_hu_reference:
            # Confronta con CARVE reference
            carve_mean_avg = self.carve_hu_reference['mean_hu']['avg']
            carve_mean_std = self.carve_hu_reference['mean_hu']['std']
            carve_air_avg = self.carve_hu_reference['air_peak']['avg']
            
            # Considera anomalo se fuori da ±2.5 std dev dal reference (più conservativo)
            diff = abs(hu_stats['mean'] - carve_mean_avg)
            threshold = 2.5 * carve_mean_std
            
            # ANCHE controlla air peak
            air_diff = abs(hu_stats['air_peak'] - carve_air_avg)
            
            if diff > threshold or air_diff > 200:
                needs_correction = True
                correction_type = 'carve_reference_deviation'
        else:
            # Criteri standard (senza CARVE) - PIÙ CONSERVATIVI
            # Solo se chiaramente fuori range fisiologico
            if hu_stats['mean'] < -800:  # Troppo scuro
                needs_correction = True
                correction_type = 'hu_too_low'
            elif hu_stats['mean'] > -50:  # Troppo chiaro
                needs_correction = True
                correction_type = 'hu_too_high'
            # Range -800 a -50 è considerato accettabile anche se non perfetto
        
        if not needs_correction:
            if self.verbose:
                print(f"    ✓ HU accettabili - normalizzazione non necessaria")
            return image_path, False, {'status': 'acceptable', 'no_correction_needed': True}
        
        # STEP 3: Calcola parametri di correzione
        if self.verbose:
            print(f"    ⚠️  HU anomale (mean: {hu_stats['mean']:.1f}) - tipo: {correction_type}")
            print(f"    Valutando correzione...")
        
        offset, scale = self._estimate_hu_correction(hu_stats, self.carve_hu_reference)
        
        # STEP 4: Verifica che la correzione sia ragionevole
        is_reasonable, reason = self._is_correction_reasonable(offset, scale, hu_stats)
        
        if not is_reasonable:
            if self.verbose:
                print(f"    ⚠️  Correzione troppo estrema - NON applico per sicurezza")
                print(f"       Motivo: {reason}")
                print(f"       Offset: {offset:.1f}, Scale: {scale:.3f}")
            return image_path, False, {
                'skipped': True, 
                'reason': 'correction_too_extreme',
                'details': reason,
                'offset': offset,
                'scale': scale
            }
        
        # STEP 5: Applica correzione
        if self.verbose:
            print(f"    Applicando correzione (offset: {offset:.1f}, scale: {scale:.3f})...")
        
        corrected_array = (image_array.astype(np.float32) * scale) + offset
        corrected_array = np.clip(corrected_array, -1024, 3071)
        corrected_array = corrected_array.astype(image_array.dtype)
        
        # STEP 6: Verifica che la correzione abbia migliorato
        corrected_stats = self._calculate_hu_statistics(corrected_array)
        
        improvement_ok = self._verify_correction_improved(hu_stats, corrected_stats, self.carve_hu_reference)
        
        if not improvement_ok:
            if self.verbose:
                print(f"    ⚠️  Correzione non migliora la qualità - NON salvo")
            return image_path, False, {
                'skipped': True,
                'reason': 'no_improvement',
                'original_mean': hu_stats['mean'],
                'corrected_mean': corrected_stats['mean']
            }
        
        # STEP 7: Salva immagine corretta
        corrected_image = sitk.GetImageFromArray(corrected_array)
        corrected_image.CopyInformation(image)
        
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_normalized{ext}"
        
        sitk.WriteImage(corrected_image, output_path)
        
        correction_params = {
            'offset': offset,
            'scale': scale,
            'original_mean': hu_stats['mean'],
            'corrected_mean': corrected_stats['mean'],
            'original_air_peak': hu_stats['air_peak'],
            'corrected_air_peak': corrected_stats['air_peak'],
            'correction_type': correction_type,
        }
        
        if self.verbose:
            print(f"    ✓ HU corrette con successo:")
            print(f"       Mean: {hu_stats['mean']:.1f} → {corrected_stats['mean']:.1f}")
            print(f"       Air peak: {hu_stats['air_peak']:.1f} → {corrected_stats['air_peak']:.1f}")
        
        return output_path, True, correction_params
    
    
    def _validate_hu_structure(self, hu_stats):
        """
        Verifica che la scan abbia una struttura HU sensata
        
        Una CT toracica deve avere:
        - Picco dell'aria visibile (< -500 HU)
        - Tessuto polmonare presente (-900 a -300 HU)
        - Range HU ragionevole
        
        Returns:
            (is_valid, issues_list)
        """
        issues = []
        
        # Check 1: Deve esserci un picco dell'aria identificabile
        if hu_stats['air_peak'] > -500:
            issues.append(f"Picco aria non rilevabile (air_peak: {hu_stats['air_peak']:.1f})")
        
        # Check 2: Deve esserci tessuto polmonare
        if hu_stats['lung_percentage'] < 0.1:
            issues.append(f"Quasi nessun tessuto polmonare ({hu_stats['lung_percentage']:.2f}%)")
        
        # Check 3: Range HU deve essere ragionevole
        if hu_stats['range'] < 500:
            issues.append(f"Range HU troppo ristretto ({hu_stats['range']:.0f})")
        elif hu_stats['range'] > 6000:
            issues.append(f"Range HU troppo ampio ({hu_stats['range']:.0f}) - possibile rumore/artefatti")
        
        # Check 4: La distribuzione deve essere sensata
        # p01 dovrebbe essere vicino all'aria, p99 vicino a osso/contrasto
        if hu_stats['p01'] > -500:
            issues.append(f"P01 troppo alto ({hu_stats['p01']:.0f}) - distribuzione anomala")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    
    def _is_correction_reasonable(self, offset, scale, hu_stats):
        """
        Verifica che la correzione proposta sia ragionevole
        
        Correzioni troppo estreme indicano che la scan ha problemi più gravi
        e la normalizzazione potrebbe peggiorare la situazione
        
        Returns:
            (is_reasonable, reason)
        """
        # Limite 1: Offset non deve essere eccessivo
        # Offset > 500 HU indica problemi gravi di calibrazione
        if abs(offset) > 500:
            return False, f"Offset troppo grande ({offset:.1f} HU) - possibile problema di calibrazione grave"
        
        # Limite 2: Scale non deve essere troppo lontano da 1.0
        # Scale lontano da 1.0 indica che la scala HU è completamente sbagliata
        if scale < 0.8 or scale > 1.25:
            return False, f"Scale factor troppo estremo ({scale:.3f}) - possibile problema di rescale slope"
        
        # Limite 3: La correzione combinata non deve spostare troppo i valori
        # Simula correzione sul mean e air peak
        corrected_mean = (hu_stats['mean'] * scale) + offset
        corrected_air = (hu_stats['air_peak'] * scale) + offset
        
        # Il mean deve finire in un range fisiologico per CT toracico
        if corrected_mean < -900 or corrected_mean > -200:
            return False, f"Mean corretto fuori range fisiologico ({corrected_mean:.1f})"
        
        # L'aria deve finire vicino a -1000 HU
        if corrected_air < -1100 or corrected_air > -850:
            return False, f"Air peak corretto fuori range ({corrected_air:.1f})"
        
        return True, "OK"
    
    
    def _verify_correction_improved(self, original_stats, corrected_stats, carve_reference):
        """
        Verifica che la correzione abbia effettivamente migliorato la qualità
        
        Returns:
            True se la correzione migliora, False altrimenti
        """
        if carve_reference:
            # Confronta distanza da CARVE reference
            target_mean = carve_reference['mean_hu']['avg']
            target_air = carve_reference['air_peak']['avg']
            
            # Distanza originale
            orig_mean_dist = abs(original_stats['mean'] - target_mean)
            orig_air_dist = abs(original_stats['air_peak'] - target_air)
            
            # Distanza corretta
            corr_mean_dist = abs(corrected_stats['mean'] - target_mean)
            corr_air_dist = abs(corrected_stats['air_peak'] - target_air)
            
            # Deve migliorare entrambi o almeno uno senza peggiorare l'altro
            mean_improved = corr_mean_dist < orig_mean_dist * 0.8  # Miglioramento del 20%
            air_improved = corr_air_dist < orig_air_dist * 0.8
            
            mean_worsened = corr_mean_dist > orig_mean_dist * 1.2
            air_worsened = corr_air_dist > orig_air_dist * 1.2
            
            # Accetta se migliora almeno uno senza peggiorare l'altro
            if (mean_improved or air_improved) and not (mean_worsened or air_worsened):
                return True
            else:
                return False
        else:
            # Senza CARVE, verifica che i valori siano in range fisiologico
            mean_ok = -700 <= corrected_stats['mean'] <= -200
            air_ok = -1050 <= corrected_stats['air_peak'] <= -900
            
            return mean_ok and air_ok
    
    
    def _estimate_hu_correction(self, hu_stats, carve_reference):
        """
        Stima parametri di correzione HU
        
        Se carve_reference è disponibile, usa quello come target
        Altrimenti usa euristica basata sul picco dell'aria
        """
        if carve_reference:
            # Usa CARVE come target
            target_mean = carve_reference['mean_hu']['avg']
            current_mean = hu_stats['mean']
            
            offset = target_mean - current_mean
            
            # Scale basato su air peak
            target_air_peak = carve_reference['air_peak']['avg']
            current_air_peak = hu_stats['air_peak']
            
            # Stima scala basata su quanto l'aria è spostata
            if abs(current_air_peak - target_air_peak) > 100:
                scale = target_air_peak / current_air_peak if current_air_peak != 0 else 1.0
            else:
                scale = 1.0
        else:
            # Euristica standard
            expected_air_peak = -1000
            current_air_peak = hu_stats['air_peak']
            offset = expected_air_peak - current_air_peak
            
            # Scale basato sul range
            expected_range = 2500
            actual_range = hu_stats['p99'] - hu_stats['p01']
            scale = expected_range / actual_range if actual_range > 0 else 1.0
        
        # Limita correzioni estreme (più conservativo)
        offset = np.clip(offset, -500, 500)    # Era -2000, 2000
        scale = np.clip(scale, 0.8, 1.25)      # Era 0.5, 2.0
        
        return offset, scale
    
    
    # ==================== SEZIONE 2: PIPELINE COMPLETA ====================
    
    def run_complete_pipeline(self):
        """
        Esegue la pipeline completa:
        1. Trova e analizza pazienti OSIC
        2. Converte DICOM → MHD/RAW con validazione
        3. Copia scan valide in cartella separata
        4. Genera report dettagliato
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("OSIC ANALYSIS AND CONVERSION PIPELINE")
            print(f"{'='*70}")
            print(f"Input: {self.input_folder}")
            print(f"Output: {self.output_folder}")
            print(f"Validated: {self.validated_folder}")
            print(f"\nParametri:")
            print(f"  Min slices: {self.criteria['min_slices']}")
            print(f"  Slice thickness: {self.criteria['min_slice_thickness']}-{self.criteria['max_slice_thickness']}mm")
            print(f"  XY resolution: {self.criteria['min_xy_resolution']}-{self.criteria['max_xy_resolution']}mm")
        
        # Step 1: Trova pazienti OSIC
        if self.verbose:
            print(f"\n{'='*70}")
            print("STEP 1: ANALISI PAZIENTI OSIC")
            print(f"{'='*70}")
        
        patients = self.find_dicom_patients()
        self.stats['total_patients'] = len(patients)
        
        if self.verbose:
            print(f"Pazienti trovati: {len(patients)}")
        
        if len(patients) == 0:
            if self.verbose:
                print(f"✗ Nessun paziente trovato in: {self.input_folder}")
            return None
        
        # Crea cartelle output e validated
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.validated_folder, exist_ok=True)
        
        # Step 2: Processa ogni paziente
        results = {}
        
        for idx, (patient_id, patient_folder) in enumerate(patients, 1):
            if self.verbose:
                print(f"\n{'─'*70}")
                print(f"[{idx}/{len(patients)}] Paziente: {patient_id}")
                print(f"{'─'*70}")
            
            result = self._process_patient(
                patient_id,
                patient_folder
            )
            
            results[patient_id] = result
        
        # Step 4: Genera report finale
        report = self._generate_final_report(results)
        
        # Step 5: Genera visualizzazioni per la tesi PRIMA di salvare JSON
        if self.verbose:
            print(f"\n{'='*70}")
            print("GENERAZIONE VISUALIZZAZIONI PER TESI")
            print(f"{'='*70}")
        
        visualizations = self._generate_visualizations()
        report['visualizations'] = visualizations
        
        # Step 6: Converti tipi numpy in tipi Python per JSON serialization
        report_serializable = convert_numpy_types(report)
        
        # Salva report JSON
        report_json = os.path.join(self.output_folder, "osic_analysis_report.json")
        with open(report_json, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("PIPELINE COMPLETATA")
            print(f"{'='*70}")
            print(f"Report salvato: {report_json}")
        
        return report
    
    
    def _process_patient(self, patient_id, patient_folder):
        """
        Processa un singolo paziente attraverso l'intera pipeline
        Le scan valide vengono COPIATE in validated/, quelle non valide restano nella cartella originale
        """
        result = {
            'patient_id': patient_id,
            'folder': str(patient_folder),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Step 1: Analizza DICOM
        if self.verbose:
            print(f"  Analisi DICOM...")
        
        metrics, is_compatible, issues = self.analyze_dicom_series(patient_folder)
        
        if metrics is None:
            result['status'] = 'error'
            result['error'] = 'Impossibile analizzare DICOM'
            result['issues'] = issues
            self.stats['errors'] += 1
            # Non facciamo nulla - resta nella cartella originale
            return result
        
        result['metrics'] = metrics
        result['compatible'] = is_compatible
        result['issues'] = issues
        
        # Raccogli metriche per TUTTI i pazienti (anche incompatibili)
        self.all_metrics.append({
            'patient_id': patient_id,
            'num_slices': int(metrics['num_slices']) if metrics['num_slices'] is not None else None,
            'slice_thickness': float(metrics['slice_thickness']) if metrics['slice_thickness'] is not None else None,
            'xy_resolution': float(metrics['xy_resolution']) if metrics['xy_resolution'] is not None else None,
            'compatible': bool(is_compatible),
            'issues': issues
        })
        
        if not is_compatible:
            result['status'] = 'incompatible'
            # Registra motivazioni incompatibilità
            for issue in issues:
                # Estrai categoria del problema
                if 'slice' in issue.lower() and ('poche' in issue.lower() or 'slices:' in issue.lower()):
                    self.incompatibility_reasons['num_slices_out_of_range'] += 1
                elif 'slice' in issue.lower() and ('sottile' in issue.lower() or 'spessa' in issue.lower()):
                    self.incompatibility_reasons['slice_thickness_out_of_range'] += 1
                elif 'xy' in issue.lower() or 'risoluzione' in issue.lower():
                    self.incompatibility_reasons['xy_resolution_out_of_range'] += 1
                elif 'uniform' in issue.lower() or 'mancant' in issue.lower():
                    self.incompatibility_reasons['non_uniform_slices'] += 1
                else:
                    self.incompatibility_reasons['other'] += 1
            
            if self.verbose:
                print(f"  ✗ INCOMPATIBILE:")
                for issue in issues:
                    print(f"     - {issue}")
            self.stats['incompatible'] += 1
            # Non facciamo nulla - resta nella cartella originale
            return result
        
        self.stats['compatible'] += 1
        
        if self.verbose:
            print(f"  ✓ Compatibile:")
            print(f"     Slices: {metrics['num_slices']}")
            print(f"     Thickness: {metrics['slice_thickness']:.3f}mm")
            print(f"     XY res: {metrics['xy_resolution']:.3f}mm")
        
        # Step 2: Converti DICOM → MHD/RAW
        if self.verbose:
            print(f"  Conversione DICOM → MHD/RAW...")
        
        output_filename = f"{patient_id}.mhd"
        success, output_path, validation = self.convert_dicom_to_mhd(
            patient_folder,
            output_filename,
            metrics
        )
        
        result['conversion'] = {
            'success': success,
            'output_path': output_path,
            'validation': validation
        }
        
        if not success:
            result['status'] = 'conversion_failed'
            if self.verbose:
                print(f"  ✗ Conversione fallita")
            # Rimuovi eventuali file output parziali
            self._remove_output_files(patient_id)
            self.stats['conversion_failed'] += 1
            # Non facciamo altro - resta nella cartella originale
            return result
        
        self.stats['converted'] += 1
        
        # Verifica validazione
        if validation['all_checks_passed']:
            self.stats['validation_passed'] += 1
            if self.verbose:
                print(f"  ✓ Conversione validata al 100%")
                print(f"     MD5 match: {validation['checks'].get('pixel_data_match', False)}")
                print(f"     Size match: {validation['checks'].get('size_match', False)}")
        else:
            # VALIDAZIONE FALLITA
            result['status'] = 'validation_failed'
            if self.verbose:
                print(f"  ✗ VALIDAZIONE FALLITA:")
                for check, value in validation['checks'].items():
                    if not value and isinstance(value, bool):
                        print(f"     - {check}: FAIL")
            # Rimuovi file output
            self._remove_output_files(patient_id)
            self.stats['validation_failed'] += 1
            # Non facciamo altro - resta nella cartella originale
            return result
        
        # SCAN VALIDATA CON SUCCESSO - COPIA in validated/
        validated_path = self._copy_to_validated(patient_folder, patient_id)
        result['validated_copy'] = validated_path
        
        result['status'] = 'success'
        return result
    
    
    def _generate_final_report(self, results):
        """
        Genera report finale dettagliato
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'criteria': self.criteria,
            'statistics': self.stats,
            'patients': results,
        }
        
        # Aggiungi summary
        compatible_patients = [p for p, r in results.items() if r.get('compatible', False)]
        converted_patients = [p for p, r in results.items() if r.get('conversion', {}).get('success', False)]
        validated_patients = [p for p, r in results.items() 
                            if r.get('conversion', {}).get('validation', {}).get('all_checks_passed', False)]
        
        report['summary'] = {
            'total_patients': self.stats['total_patients'],
            'compatible': len(compatible_patients),
            'converted': len(converted_patients),
            'validation_passed': len(validated_patients),
            'success_rate': len(validated_patients) / max(self.stats['total_patients'], 1) * 100,
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("STATISTICHE FINALI")
            print(f"{'='*70}")
            print(f"Pazienti totali: {report['summary']['total_patients']}")
            print(f"\n✅ SCAN VALIDE (copiate in validated/):")
            print(f"  Compatibili: {report['summary']['compatible']}")
            print(f"  Convertite: {report['summary']['converted']}")
            print(f"  Validate: {report['summary']['validation_passed']}")
            print(f"\n✗ SCAN NON VALIDE (rimaste in originale):")
            print(f"  Parametri incompatibili: {self.stats['incompatible']}")
            print(f"  Conversione fallita: {self.stats['conversion_failed']}")
            print(f"  Validazione fallita: {self.stats['validation_failed']}")
            print(f"\nTOTALE NON VALIDE: {self.stats['incompatible'] + self.stats['conversion_failed'] + self.stats['validation_failed']}")
            print(f"Success rate finale: {report['summary']['validation_passed']/max(report['summary']['total_patients'],1)*100:.1f}%")
        
        return report
    
    
    def _generate_visualizations(self):
        """
        Genera visualizzazioni per la tesi:
        1. Riduzione del dataset (grafico a barre/torta)
        2. Distribuzione metriche pre-filtro (istogrammi)
        
        Returns:
            dict con percorsi ai file generati
        """
        try:
            viz_folder = os.path.join(self.output_folder, "visualizations")
            os.makedirs(viz_folder, exist_ok=True)
            
            visualizations = {}
            
            # Converti metriche in DataFrame per facilitare analisi
            df = pd.DataFrame(self.all_metrics)
            
            if len(df) == 0:
                if self.verbose:
                    print("  ⚠️  Nessuna metrica raccolta")
                return visualizations
            
            if self.verbose:
                print(f"  Processando {len(df)} pazienti per visualizzazioni...")
        
            # Configurazione stile grafici
            plt.style.use('seaborn-v0_8-darkgrid')
            colors_main = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
            
            # ========== GRAFICO 1: Riduzione del dataset ==========
            if self.verbose:
                print(f"  Generando grafico riduzione dataset...")
        
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Subplot 1: Torta - Compatibili vs Incompatibili
            compatible_count = df['compatible'].sum()
            incompatible_count = len(df) - compatible_count
        
            labels = ['Compatible', 'Incompatible']
            sizes = [compatible_count, incompatible_count]
            colors = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0.05)
            
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
            ax1.set_title(f'OSIC Dataset Reduction\nTotal patients: {len(df)}', 
                          fontsize=14, weight='bold', pad=20)
            
            # Subplot 2: Barre - Ragioni incompatibilità
            if len(self.incompatibility_reasons) > 0:
                reasons = list(self.incompatibility_reasons.keys())
                counts = list(self.incompatibility_reasons.values())
                
                # Translate reason names for visualization
                reason_labels = {
                    'num_slices_out_of_range': 'Number of slices\nout of range',
                    'slice_thickness_out_of_range': 'Slice thickness\nout of range',
                    'xy_resolution_out_of_range': 'XY resolution\nout of range',
                    'non_uniform_slices': 'Non-uniform\nslices',
                    'other': 'Other'
                }
                
                display_labels = [reason_labels.get(r, r) for r in reasons]
                
                bars = ax2.bar(range(len(reasons)), counts, color=colors_main[:len(reasons)])
                ax2.set_xticks(range(len(reasons)))
                ax2.set_xticklabels(display_labels, rotation=0, ha='center', fontsize=10)
                ax2.set_ylabel('Number of patients', fontsize=12, weight='bold')
                ax2.set_title('Incompatibility reasons', fontsize=14, weight='bold', pad=20)
                ax2.grid(axis='y', alpha=0.3)
                
                # Aggiungi valori sopra le barre
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=11, weight='bold')
            else:
                ax2.text(0.5, 0.5, 'No incompatibility detected', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.axis('off')
            
            plt.tight_layout()
            reduction_path = os.path.join(viz_folder, "dataset_reduction.png")
            plt.savefig(reduction_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations['dataset_reduction'] = reduction_path
            
            if self.verbose:
                print(f"    ✓ Salvato: {reduction_path}")
            
            # ========== GRAFICO 2: Distribuzione numero slices ==========
            if self.verbose:
                print(f"  Generando distribuzione numero slices...")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Istogramma per tutti i pazienti
            all_slices = df['num_slices'].dropna()
            compatible_slices = df[df['compatible']]['num_slices'].dropna()
            incompatible_slices = df[~df['compatible']]['num_slices'].dropna()
            
            bins = np.arange(0, max(all_slices) + 50, 50)
            
            ax.hist([compatible_slices, incompatible_slices], bins=bins, 
                    label=['Compatible', 'Incompatible'], 
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            
            # Linee verticali per i criteri
            ax.axvline(self.criteria['min_slices'], color='red', linestyle='--', 
                       linewidth=2, label=f"Min slices: {self.criteria['min_slices']}")
            ax.axvline(self.criteria['max_slices'], color='red', linestyle='--', 
                       linewidth=2, label=f"Max slices: {self.criteria['max_slices']}")
            
            ax.set_xlabel('Number of slices', fontsize=12, weight='bold')
            ax.set_ylabel('Number of patients', fontsize=12, weight='bold')
            ax.set_title('Number of slices distribution (pre-filter)', fontsize=14, weight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            # Statistics
            stats_text = f"Mean: {all_slices.mean():.0f}\nMedian: {all_slices.median():.0f}\nStd: {all_slices.std():.0f}"
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            slices_path = os.path.join(viz_folder, "distribution_num_slices.png")
            plt.savefig(slices_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations['num_slices_distribution'] = slices_path
            
            if self.verbose:
                print(f"    ✓ Salvato: {slices_path}")
            
            # ========== GRAFICO 3: Distribuzione slice thickness ==========
            if self.verbose:
                print(f"  Generando distribuzione slice thickness...")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            all_thickness = df['slice_thickness'].dropna()
            compatible_thickness = df[df['compatible']]['slice_thickness'].dropna()
            incompatible_thickness = df[~df['compatible']]['slice_thickness'].dropna()
            
            bins_thickness = np.arange(0, min(max(all_thickness) + 0.2, 3.0), 0.1)
            
            ax.hist([compatible_thickness, incompatible_thickness], bins=bins_thickness,
                    label=['Compatible', 'Incompatible'],
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            
            # Linee verticali per i criteri
            ax.axvline(self.criteria['min_slice_thickness'], color='red', linestyle='--',
                       linewidth=2, label=f"Min: {self.criteria['min_slice_thickness']}mm")
            ax.axvline(self.criteria['max_slice_thickness'], color='red', linestyle='--',
                       linewidth=2, label=f"Max: {self.criteria['max_slice_thickness']}mm")
            
            ax.set_xlabel('Slice thickness (mm)', fontsize=12, weight='bold')
            ax.set_ylabel('Number of patients', fontsize=12, weight='bold')
            ax.set_title('Slice thickness distribution (pre-filter)', fontsize=14, weight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            # Statistics
            stats_text = f"Mean: {all_thickness.mean():.3f}mm\nMedian: {all_thickness.median():.3f}mm\nStd: {all_thickness.std():.3f}mm"
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            thickness_path = os.path.join(viz_folder, "distribution_slice_thickness.png")
            plt.savefig(thickness_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations['slice_thickness_distribution'] = thickness_path
            
            if self.verbose:
                print(f"    ✓ Salvato: {thickness_path}")
            
            # ========== GRAFICO 4: Distribuzione XY resolution ==========
            if self.verbose:
                print(f"  Generando distribuzione XY resolution...")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            all_xy = df['xy_resolution'].dropna()
            compatible_xy = df[df['compatible']]['xy_resolution'].dropna()
            incompatible_xy = df[~df['compatible']]['xy_resolution'].dropna()
            
            bins_xy = np.arange(0, min(max(all_xy) + 0.1, 2.0), 0.05)
            
            ax.hist([compatible_xy, incompatible_xy], bins=bins_xy,
                    label=['Compatible', 'Incompatible'],
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
            
            # Linee verticali per i criteri
            ax.axvline(self.criteria['min_xy_resolution'], color='red', linestyle='--',
                       linewidth=2, label=f"Min: {self.criteria['min_xy_resolution']}mm")
            ax.axvline(self.criteria['max_xy_resolution'], color='red', linestyle='--',
                       linewidth=2, label=f"Max: {self.criteria['max_xy_resolution']}mm")
            
            ax.set_xlabel('XY resolution (mm)', fontsize=12, weight='bold')
            ax.set_ylabel('Number of patients', fontsize=12, weight='bold')
            ax.set_title('XY resolution distribution (pre-filter)', fontsize=14, weight='bold', pad=20)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            # Statistics
            stats_text = f"Mean: {all_xy.mean():.3f}mm\nMedian: {all_xy.median():.3f}mm\nStd: {all_xy.std():.3f}mm"
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            xy_path = os.path.join(viz_folder, "distribution_xy_resolution.png")
            plt.savefig(xy_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations['xy_resolution_distribution'] = xy_path
            
            if self.verbose:
                print(f"    ✓ Salvato: {xy_path}")
            
            # ========== GRAFICO 5: Summary multi-metrica ==========
            if self.verbose:
                print(f"  Generando summary multi-metrica...")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # BoxPlot 1: Numero slices
            ax = axes[0, 0]
            data_slices = [compatible_slices, incompatible_slices]
            bp1 = ax.boxplot(data_slices, labels=['Compatible', 'Incompatible'],
                             patch_artist=True, notch=True)
            for patch, color in zip(bp1['boxes'], ['#2ecc71', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.axhline(self.criteria['min_slices'], color='red', linestyle='--', alpha=0.5)
            ax.set_ylabel('Number of slices', fontsize=11, weight='bold')
            ax.set_title('Boxplot: Number of slices', fontsize=12, weight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # BoxPlot 2: Slice thickness
            ax = axes[0, 1]
            data_thickness = [compatible_thickness, incompatible_thickness]
            bp2 = ax.boxplot(data_thickness, labels=['Compatible', 'Incompatible'],
                             patch_artist=True, notch=True)
            for patch, color in zip(bp2['boxes'], ['#2ecc71', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.axhline(self.criteria['min_slice_thickness'], color='red', linestyle='--', alpha=0.5)
            ax.axhline(self.criteria['max_slice_thickness'], color='red', linestyle='--', alpha=0.5)
            ax.set_ylabel('Slice thickness (mm)', fontsize=11, weight='bold')
            ax.set_title('Boxplot: Slice thickness', fontsize=12, weight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # BoxPlot 3: XY resolution
            ax = axes[1, 0]
            data_xy = [compatible_xy, incompatible_xy]
            bp3 = ax.boxplot(data_xy, labels=['Compatible', 'Incompatible'],
                             patch_artist=True, notch=True)
            for patch, color in zip(bp3['boxes'], ['#2ecc71', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.axhline(self.criteria['min_xy_resolution'], color='red', linestyle='--', alpha=0.5)
            ax.axhline(self.criteria['max_xy_resolution'], color='red', linestyle='--', alpha=0.5)
            ax.set_ylabel('XY resolution (mm)', fontsize=11, weight='bold')
            ax.set_title('Boxplot: XY resolution', fontsize=12, weight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Tabella statistiche comparative
            ax = axes[1, 1]
            ax.axis('off')
            
            table_data = [
                ['Metric', 'Compatible', 'Incompatible', 'All'],
                ['', '', '', ''],
                ['N. patients', f"{compatible_count}", f"{incompatible_count}", f"{len(df)}"],
                ['', '', '', ''],
                ['Slices (mean)', f"{compatible_slices.mean():.0f}", 
                 f"{incompatible_slices.mean():.0f}" if len(incompatible_slices) > 0 else "N/A",
                 f"{all_slices.mean():.0f}"],
                ['Slices (std)', f"{compatible_slices.std():.0f}",
                 f"{incompatible_slices.std():.0f}" if len(incompatible_slices) > 0 else "N/A",
                 f"{all_slices.std():.0f}"],
                ['', '', '', ''],
                ['Thickness (mean)', f"{compatible_thickness.mean():.3f}mm",
                 f"{incompatible_thickness.mean():.3f}mm" if len(incompatible_thickness) > 0 else "N/A",
                 f"{all_thickness.mean():.3f}mm"],
                ['Thickness (std)', f"{compatible_thickness.std():.3f}mm",
                 f"{incompatible_thickness.std():.3f}mm" if len(incompatible_thickness) > 0 else "N/A",
                 f"{all_thickness.std():.3f}mm"],
                ['', '', '', ''],
                ['XY res (mean)', f"{compatible_xy.mean():.3f}mm",
                 f"{incompatible_xy.mean():.3f}mm" if len(incompatible_xy) > 0 else "N/A",
                 f"{all_xy.mean():.3f}mm"],
                ['XY res (std)', f"{compatible_xy.std():.3f}mm",
                 f"{incompatible_xy.std():.3f}mm" if len(incompatible_xy) > 0 else "N/A",
                 f"{all_xy.std():.3f}mm"],
            ]
            
            table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Formattazione header
            for i in range(4):
                cell = table[(0, i)]
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
            
            # Formattazione colori alternati
            for i in range(2, len(table_data)):
                for j in range(4):
                    cell = table[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#ecf0f1')
            
            ax.set_title('Comparative statistics', fontsize=12, weight='bold', pad=20)
            
            plt.suptitle('OSIC Dataset Multi-Metric Summary (Pre-Filter)', 
                        fontsize=16, weight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            
            summary_path = os.path.join(viz_folder, "multi_metric_summary.png")
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations['multi_metric_summary'] = summary_path
            
            if self.verbose:
                print(f"    ✓ Salvato: {summary_path}")
            
            # Salva anche statistiche in CSV
            csv_path = os.path.join(viz_folder, "metrics_summary.csv")
            df.to_csv(csv_path, index=False)
            visualizations['metrics_csv'] = csv_path
        
            if self.verbose:
                print(f"    ✓ Metriche salvate in CSV: {csv_path}")
            
            if self.verbose:
                print(f"\n  ✅ Tutte le visualizzazioni generate in: {viz_folder}")
            
            return visualizations
            
        except Exception as e:
            if self.verbose:
                print(f"\n  ✗ Errore durante generazione visualizzazioni: {e}")
                import traceback
                traceback.print_exc()
            return {}


# ==================== MAIN ====================

def main():
    """
    Esegue la pipeline completa OSIC
    """
    # Crea pipeline
    pipeline = OSICAnalysisPipeline(verbose=True)
    
    # CONFIGURAZIONE (modifica questi percorsi se necessario)
    pipeline.input_folder = "X:/Francesca Saglimbeni/tesi/datasets/dataset_OSIC"
    pipeline.output_folder = "X:/Francesca Saglimbeni/tesi/cancellare/osic_touse"
    pipeline.validated_folder = "X:/Francesca Saglimbeni/tesi/cancellare/OSIC_validated"
    
    # Esegui pipeline completa
    report = pipeline.run_complete_pipeline()
    
    if report:
        print(f"\n✅ Pipeline completata con successo!")
        print(f"   Report: {pipeline.output_folder}/osic_analysis_report.json")
    else:
        print(f"\n✗ Pipeline fallita")


if __name__ == "__main__":
    main()
