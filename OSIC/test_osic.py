import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
import os


class TotalSegmentatorCompatibilityChecker:
    """
    Verifica compatibilità CT con TotalSegmentator e filtra scan problematiche
    Supporta file MHD/NII.GZ e cartelle DICOM
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Criteri di compatibilità basati su TotalSegmentator requirements
        # Calibrati su dati reali OSIC/CARVE
        self.criteria = {
            'min_slices': 100,  # TotalSegmentator funziona da ~100 slice in su
            'max_slice_thickness': 1.5,  # Sopra 1.5mm diventa problematico
            'min_lung_tissue': 0.1,  # Anche scan con poco polmone possono funzionare
            'hu_range_min': (-3000, 200),  # Range molto ampio - TotalSegmentator è robusto
            'max_noise_hu': 2000,  # Anche scan rumorose possono funzionare
        }
    
    def load_dicom_series(self, dicom_folder):
        """
        Carica una serie DICOM da una cartella
        
        Returns:
            (sitk_image, success)
        """
        try:
            # Usa ImageSeriesReader per leggere serie DICOM
            reader = sitk.ImageSeriesReader()
            
            # Trova i file DICOM nella cartella
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_folder))
            
            if len(dicom_names) == 0:
                # Se GetGDCMSeriesFileNames non trova nulla, prova a leggere tutti i .dcm
                dcm_files = list(Path(dicom_folder).glob("*.dcm"))
                if len(dcm_files) == 0:
                    return None, False
                dicom_names = [str(f) for f in sorted(dcm_files)]
            
            reader.SetFileNames(dicom_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            
            sitk_image = reader.Execute()
            
            return sitk_image, True
            
        except Exception as e:
            if self.verbose:
                print(f"  Error loading DICOM: {e}")
            return None, False
    
    def analyze_ct_scan(self, image_path_or_folder):
        """
        Analizza una scan CT e restituisce metriche di qualità
        Supporta sia file (MHD/NII.GZ) che cartelle DICOM
        
        Returns:
            dict con metriche e flag di compatibilità
        """
        try:
            path = Path(image_path_or_folder)
            
            # Determina se è una cartella DICOM o un file
            if path.is_dir():
                # È una cartella - prova a caricare come DICOM
                if self.verbose:
                    print(f"  Loading DICOM series from folder...")
                
                sitk_image, success = self.load_dicom_series(path)
                if not success:
                    return None, False
                    
            else:
                # È un file - carica normalmente
                sitk_image = sitk.ReadImage(str(path))
            
            image_array = sitk.GetArrayFromImage(sitk_image)
            
            # Estrai metadata
            spacing = sitk_image.GetSpacing()
            size = sitk_image.GetSize()
            
            num_slices = image_array.shape[0]
            slice_thickness = spacing[2]
            pixel_spacing_xy = (spacing[0] + spacing[1]) / 2
            
            # Analisi HU
            hu_mean = float(np.mean(image_array))
            hu_std = float(np.std(image_array))
            hu_min = float(np.min(image_array))
            hu_max = float(np.max(image_array))
            
            # Stima tessuto polmonare (HU tra -900 e -300)
            lung_mask = (image_array >= -900) & (image_array <= -300)
            lung_percentage = float(np.sum(lung_mask) / image_array.size * 100)
            
            # Stima FOV (field of view in mm)
            fov_xy = max(size[0] * spacing[0], size[1] * spacing[1])
            
            metrics = {
                'num_slices': num_slices,
                'slice_thickness': slice_thickness,
                'pixel_spacing_xy': pixel_spacing_xy,
                'fov_xy': fov_xy,
                'hu_mean': hu_mean,
                'hu_std': hu_std,
                'hu_min': hu_min,
                'hu_max': hu_max,
                'lung_percentage': lung_percentage,
                'image_shape': image_array.shape,
            }
            
            return metrics, True
            
        except Exception as e:
            if self.verbose:
                print(f"  Error analyzing scan: {e}")
            return None, False
    
    def check_compatibility(self, metrics):
        """
        Verifica compatibilità con TotalSegmentator
        CRITERI AGGIORNATI: basati su test reali, TotalSegmentator è molto robusto
        
        Returns:
            (is_compatible, issues, quality_score, warnings)
        """
        issues = []  # Problemi critici che impediscono segmentazione
        warnings = []  # Avvisi su qualità subottimale ma funzionante
        quality_score = 100
        
        # Check 1: Numero di slice - CRITICO
        if metrics['num_slices'] < self.criteria['min_slices']:
            issues.append(f"Too few slices: {metrics['num_slices']} < {self.criteria['min_slices']}")
            quality_score -= 40
        elif metrics['num_slices'] < 250:
            warnings.append(f"Low slice count: {metrics['num_slices']} (optimal: >250)")
            quality_score -= 10
        
        # Check 2: Spessore slice - CRITICO sopra 1.5mm
        if metrics['slice_thickness'] > self.criteria['max_slice_thickness']:
            issues.append(f"Slice too thick: {metrics['slice_thickness']:.3f} > {self.criteria['max_slice_thickness']}")
            quality_score -= 40
        elif metrics['slice_thickness'] > 1.0:
            warnings.append(f"Thick slices: {metrics['slice_thickness']:.3f}mm (optimal: <1.0)")
            quality_score -= 10
        
        # Check 3: Tessuto polmonare - solo avviso, non critico
        if metrics['lung_percentage'] < self.criteria['min_lung_tissue']:
            issues.append(f"Almost no lung tissue: {metrics['lung_percentage']:.2f}% (scan might be incomplete)")
            quality_score -= 50
        elif metrics['lung_percentage'] < 1.0:
            warnings.append(f"Very little lung tissue: {metrics['lung_percentage']:.2f}%")
            quality_score -= 5
        
        # Check 4: HU range - TotalSegmentator è molto robusto, solo avvisi
        hu_min, hu_max = self.criteria['hu_range_min']
        if not (hu_min <= metrics['hu_mean'] <= hu_max):
            warnings.append(f"Unusual HU mean: {metrics['hu_mean']:.1f}")
            quality_score -= 5
        
        # Check 5: Rumore - solo avviso estremo
        if metrics['hu_std'] > self.criteria['max_noise_hu']:
            warnings.append(f"Extremely high noise: {metrics['hu_std']:.1f} HU")
            quality_score -= 10
        
        quality_score = max(0, quality_score)
        is_compatible = len(issues) == 0
        
        return is_compatible, issues, quality_score, warnings
    
    def find_patient_scans(self, input_folder):
        """
        Trova tutte le scan CT nella cartella input
        Gestisce sia strutture piatte (file diretti) che per-paziente (sottocartelle)
        
        Returns:
            dict: {patient_id: {'path': Path, 'type': 'dicom'|'mhd'|'nifti'}}
        """
        input_path = Path(input_folder)
        patients = {}
        
        # Cerca file diretti nella root
        mhd_files = list(input_path.glob("*.mhd"))
        nii_files = list(input_path.glob("*.nii.gz"))
        
        for f in mhd_files:
            patients[f.stem] = {'path': f, 'type': 'mhd'}
        
        for f in nii_files:
            if not f.name.endswith("_normalized.nii.gz"):
                patients[f.stem] = {'path': f, 'type': 'nifti'}
        
        # Cerca sottocartelle (ogni sottocartella = un paziente)
        for subfolder in input_path.iterdir():
            if not subfolder.is_dir():
                continue
            
            # Salta cartelle speciali
            if subfolder.name.startswith('.') or subfolder.name in ['incompatible_scans', 'preprocessed', 'normalized']:
                continue
            
            patient_id = subfolder.name
            
            # Controlla se contiene file DICOM
            dcm_files = list(subfolder.glob("*.dcm"))
            if len(dcm_files) > 0:
                patients[patient_id] = {
                    'path': subfolder,
                    'type': 'dicom',
                    'dcm_count': len(dcm_files)
                }
                continue
            
            # Controlla se contiene file MHD/NII.GZ
            mhd_in_sub = list(subfolder.glob("*.mhd"))
            nii_in_sub = list(subfolder.glob("*.nii.gz"))
            
            if mhd_in_sub:
                patients[patient_id] = {'path': mhd_in_sub[0], 'type': 'mhd'}
            elif nii_in_sub:
                patients[patient_id] = {'path': nii_in_sub[0], 'type': 'nifti'}
        
        return patients
    
    def process_folder(self, input_folder, output_json=None, move_incompatible=False):
        """
        Processa una cartella di CT scan e genera report
        
        Args:
            input_folder: Cartella con scan CT
                         Supporta: file diretti (.mhd, .nii.gz) o sottocartelle per paziente (con .dcm)
            output_json: Path per salvare il report JSON
            move_incompatible: Se True, sposta scan incompatibili in sottocartella
        
        Returns:
            dict con risultati per ogni scan
        """
        input_path = Path(input_folder)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("TOTALSEGMENTATOR COMPATIBILITY CHECK")
            print(f"{'='*70}")
            print(f"Input folder: {input_folder}")
        
        # Trova tutti i pazienti/scan
        patients = self.find_patient_scans(input_folder)
        
        if self.verbose:
            print(f"Patients/scans found: {len(patients)}")
            print(f"\nBreakdown by type:")
            type_counts = {}
            for p in patients.values():
                t = p['type']
                type_counts[t] = type_counts.get(t, 0) + 1
            for t, count in type_counts.items():
                print(f"  {t.upper()}: {count}")
        
        results = {}
        compatible_count = 0
        incompatible_count = 0
        
        for idx, (patient_id, patient_data) in enumerate(patients.items(), 1):
            if self.verbose:
                print(f"\n[{idx}/{len(patients)}] Analyzing: {patient_id}")
                if patient_data['type'] == 'dicom':
                    print(f"  Type: DICOM series ({patient_data.get('dcm_count', 0)} files)")
                else:
                    print(f"  Type: {patient_data['type'].upper()} file")
            
            # Analizza scan
            metrics, success = self.analyze_ct_scan(patient_data['path'])
            
            if not success:
                results[patient_id] = {
                    'path': str(patient_data['path']),
                    'type': patient_data['type'],
                    'compatible': False,
                    'error': 'Failed to load scan',
                    'quality_score': 0
                }
                incompatible_count += 1
                continue
            
            # Check compatibilità
            is_compatible, issues, quality_score, warnings = self.check_compatibility(metrics)
            
            results[patient_id] = {
                'path': str(patient_data['path']),
                'type': patient_data['type'],
                'compatible': is_compatible,
                'quality_score': quality_score,
                'metrics': metrics,
                'issues': issues,
                'warnings': warnings
            }
            
            if is_compatible:
                compatible_count += 1
                if self.verbose:
                    print(f"  ✅ COMPATIBLE (score: {quality_score})")
                    print(f"     Slices: {metrics['num_slices']}, Thickness: {metrics['slice_thickness']:.3f}mm")
                    print(f"     Lung tissue: {metrics['lung_percentage']:.1f}%, HU mean: {metrics['hu_mean']:.1f}")
                    if warnings:
                        print(f"     ⚠️  Warnings:")
                        for warning in warnings:
                            print(f"        - {warning}")
            else:
                incompatible_count += 1
                if self.verbose:
                    print(f"  ❌ INCOMPATIBLE (score: {quality_score})")
                    for issue in issues:
                        print(f"     - {issue}")
            
            # Sposta cartelle/file incompatibili se richiesto
            if move_incompatible and not is_compatible:
                incompatible_dir = input_path / "incompatible_scans"
                incompatible_dir.mkdir(exist_ok=True)
                
                scan_path = patient_data['path']
                
                if scan_path.is_dir():
                    # Sposta intera cartella DICOM
                    new_path = incompatible_dir / scan_path.name
                    scan_path.rename(new_path)
                    if self.verbose:
                        print(f"     → Moved folder to: {new_path}")
                else:
                    # Sposta file
                    new_path = incompatible_dir / scan_path.name
                    scan_path.rename(new_path)
                    
                    # Sposta .raw associato se è .mhd
                    if scan_path.suffix == '.mhd':
                        raw_file = scan_path.with_suffix('.raw')
                        if raw_file.exists():
                            raw_file.rename(incompatible_dir / raw_file.name)
                    
                    if self.verbose:
                        print(f"     → Moved to: {new_path}")
        
        # Summary
        if self.verbose:
            print(f"\n{'='*70}")
            print("COMPATIBILITY SUMMARY")
            print(f"{'='*70}")
            print(f"Total patients: {len(patients)}")
            print(f"Compatible: {compatible_count} ({compatible_count/len(patients)*100:.1f}%)")
            print(f"Incompatible: {incompatible_count} ({incompatible_count/len(patients)*100:.1f}%)")
            
            # Breakdown per issue
            if incompatible_count > 0:
                print(f"\nMost common issues:")
                issue_counts = {}
                for scan_result in results.values():
                    if not scan_result.get('compatible'):
                        for issue in scan_result.get('issues', []):
                            issue_type = issue.split(':')[0]
                            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                
                for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {issue_type}: {count} scans")
        
        # Salva report JSON
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)
            if self.verbose:
                print(f"\n✅ Report saved to: {output_json}")
        
        return results
    
    def get_compatible_scans(self, results):
        """
        Estrae lista di scan compatibili dai risultati
        
        Returns:
            list of patient IDs that are compatible
        """
        return [patient_id for patient_id, data in results.items() 
                if data.get('compatible', False)]
    
    def generate_copy_script(self, results, output_folder, script_name="copy_compatible_scans.sh"):
        """
        Genera script bash/python per copiare solo le scan compatibili
        """
        compatible_scans = self.get_compatible_scans(results)
        
        # Genera script bash
        with open(script_name, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# Script to copy only compatible scans\n")
            f.write("# Generated by TotalSegmentatorCompatibilityChecker\n\n")
            f.write(f"DEST_DIR=\"{output_folder}\"\n\n")
            f.write("mkdir -p \"$DEST_DIR\"\n\n")
            
            for patient_id in compatible_scans:
                scan_data = results[patient_id]
                source_path = scan_data['path']
                
                if scan_data['type'] == 'dicom':
                    # Copia intera cartella DICOM
                    f.write(f"# Copy DICOM folder for {patient_id}\n")
                    f.write(f"cp -r \"{source_path}\" \"$DEST_DIR/{patient_id}\"\n\n")
                else:
                    # Copia file singolo
                    filename = Path(source_path).name
                    f.write(f"# Copy {scan_data['type']} for {patient_id}\n")
                    f.write(f"cp \"{source_path}\" \"$DEST_DIR/\"\n")
                    
                    # Se è .mhd, copia anche .raw
                    if scan_data['type'] == 'mhd':
                        raw_path = str(source_path).replace('.mhd', '.raw')
                        f.write(f"cp \"{raw_path}\" \"$DEST_DIR/\"\n")
                    f.write("\n")
        
        os.chmod(script_name, 0o755)
        
        if self.verbose:
            print(f"\n✅ Copy script saved to: {script_name}")
            print(f"Usage: ./{script_name}")
        
        # Genera anche script Python equivalente
        py_script = script_name.replace('.sh', '.py')
        with open(py_script, 'w') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write("import shutil\nfrom pathlib import Path\n\n")
            f.write(f"DEST_DIR = Path('{output_folder}')\n")
            f.write("DEST_DIR.mkdir(exist_ok=True, parents=True)\n\n")
            
            for patient_id in compatible_scans:
                scan_data = results[patient_id]
                source_path = scan_data['path']
                
                if scan_data['type'] == 'dicom':
                    f.write(f"# Copy DICOM folder for {patient_id}\n")
                    f.write(f"shutil.copytree('{source_path}', DEST_DIR / '{patient_id}')\n\n")
                else:
                    f.write(f"# Copy {scan_data['type']} for {patient_id}\n")
                    f.write(f"shutil.copy2('{source_path}', DEST_DIR)\n")
                    
                    if scan_data['type'] == 'mhd':
                        raw_path = str(source_path).replace('.mhd', '.raw')
                        f.write(f"shutil.copy2('{raw_path}', DEST_DIR)\n")
                    f.write("\n")
            
            f.write("print('✅ All compatible scans copied!')\n")
        
        os.chmod(py_script, 0o755)
        if self.verbose:
            print(f"✅ Python copy script saved to: {py_script}")


def main():
    """
    Example usage
    """
    # CONFIGURAZIONE
    input_folder = "X:/Francesca Saglimbeni/tesi/train_OSIC"
    output_report = "osic_compatibility_report.json"
    compatible_output = "X:/Francesca Saglimbeni/tesi/train_OSIC_compatible"
    
    # Crea checker
    checker = TotalSegmentatorCompatibilityChecker(verbose=True)
    
    # Analizza tutte le scan
    results = checker.process_folder(
        input_folder,
        output_json=output_report,
        move_incompatible=False  # Cambia a True per spostare file incompatibili
    )
    
    # Genera script per copiare solo le scan compatibili
    checker.generate_copy_script(results, compatible_output)
    
    # Stampa lista scan compatibili
    compatible = checker.get_compatible_scans(results)
    print(f"\n{'='*70}")
    print(f"COMPATIBLE SCANS SUMMARY ({len(compatible)} total)")
    print(f"{'='*70}")
    for patient_id in compatible:
        scan_data = results[patient_id]
        print(f"{patient_id}:")
        print(f"  Type: {scan_data['type']}")
        print(f"  Quality: {scan_data['quality_score']}/100")
        print(f"  Slices: {scan_data['metrics']['num_slices']}")
        print(f"  Thickness: {scan_data['metrics']['slice_thickness']:.3f}mm")
        print()


if __name__ == "__main__":
    main()