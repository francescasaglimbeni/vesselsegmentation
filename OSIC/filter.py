"""
DICOM Filter & Converter
Filtra e converte scan DICOM in base a criteri di qualità (num slices, thickness, resolution)
"""

import os
import shutil
import pydicom
import SimpleITK as sitk
from pathlib import Path


class DICOMFilter:
    def __init__(self, input_folder, output_folder):
        """
        Inizializza il filtro DICOM
        
        Args:
            input_folder: Cartella con sottocartelle DICOM
            output_folder: Cartella output per file MHD/RAW
        """
        # PARAMETRI DI FILTRAGGIO
        self.MIN_SLICES = 340           # Minimo numero di slice (flessibile)
        self.MAX_SLICES = 1000          # Massimo numero di slice
        self.MIN_SLICE_THICKNESS = 0.6  # mm
        self.MAX_SLICE_THICKNESS = 1.5  # mm
        self.MIN_XY_RESOLUTION = 0.4    # mm
        self.MAX_XY_RESOLUTION = 1.0    # mm
        self.MAX_SPACING_NONUNIFORMITY = 5.0  # mm - massima differenza di spacing tra slice
        
        # PERCORSI
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.removed_folder = os.path.join(input_folder, "removed_scans")
        
        # STATISTICHE
        self.stats = {
            'total_folders': 0,
            'kept_after_filecount': 0,
            'converted': 0,
            'skipped_filecount': 0,
            'skipped_metadata': 0,
            'errors': 0
        }
    
    def count_files_in_folder(self, folder_path):
        """Conta tutti i file in una cartella ricorsivamente"""
        if not os.path.exists(folder_path):
            return 0
        
        file_count = 0
        for root, dirs, files in os.walk(folder_path):
            file_count += len(files)
        return file_count
    
    def filter_by_file_count(self, min_files=100):
        """
        Fase 1: Filtra cartelle con troppo pochi file
        """
        print("\n" + "="*70)
        print("FASE 1: Filtro preliminare per numero di file")
        print("="*70)
        
        os.makedirs(self.removed_folder, exist_ok=True)
        folders_to_process = []
        
        for item in sorted(os.listdir(self.input_folder)):
            item_path = os.path.join(self.input_folder, item)
            
            # Salta se non è cartella o è la cartella dei rimossi
            if not os.path.isdir(item_path) or item == os.path.basename(self.removed_folder):
                continue
            
            self.stats['total_folders'] += 1
            file_count = self.count_files_in_folder(item_path)
            
            print(f"\n[{self.stats['total_folders']}] {item}")
            print(f"  File trovati: {file_count}")
            
            if file_count >= min_files:
                print(f"  ✓ ACCETTATO (>= {min_files} file)")
                folders_to_process.append((item, item_path, file_count))
                self.stats['kept_after_filecount'] += 1
            else:
                print(f"  ✗ SCARTATO (< {min_files} file)")
                # Sposta in cartella rimossi
                dest_path = os.path.join(self.removed_folder, item)
                try:
                    shutil.move(item_path, dest_path)
                    print(f"  Spostato in: {self.removed_folder}")
                except Exception as e:
                    print(f"  Errore nello spostamento: {e}")
                self.stats['skipped_filecount'] += 1
        
        print(f"\nRisultato Fase 1:")
        print(f"  Cartelle totali: {self.stats['total_folders']}")
        print(f"  Cartelle valide: {self.stats['kept_after_filecount']}")
        print(f"  Cartelle scartate: {self.stats['skipped_filecount']}")
        
        return folders_to_process
    
    def check_dicom_parameters(self, dicom_files, folder_name):
        """
        Verifica se la serie DICOM rispetta i criteri di qualità
        
        Returns:
            (is_valid, result) - result è un dict con metadati o stringa errore
        """
        try:
            if not dicom_files or len(dicom_files) == 0:
                return False, "Nessun file DICOM valido"
            
            # 1. Controlla numero di slices
            num_slices = len(dicom_files)
            if num_slices < self.MIN_SLICES:
                return False, f"Slices ({num_slices}) < {self.MIN_SLICES}"
            if num_slices > self.MAX_SLICES:
                return False, f"Slices ({num_slices}) > {self.MAX_SLICES}"
            
            # 2. Controlla uniformità spacing tra slice
            # Leggi posizioni Z delle slice
            slice_positions = []
            for dcm_file in dicom_files[:min(50, len(dicom_files))]:  # Controlla prime 50 slice
                try:
                    ds_temp = pydicom.dcmread(dcm_file)
                    if 'ImagePositionPatient' in ds_temp:
                        z_pos = float(ds_temp.ImagePositionPatient[2])
                        slice_positions.append(z_pos)
                except:
                    continue
            
            if len(slice_positions) > 1:
                # Calcola differenze consecutive
                slice_positions.sort()
                spacings = [abs(slice_positions[i+1] - slice_positions[i]) 
                           for i in range(len(slice_positions)-1)]
                
                if spacings:
                    max_spacing = max(spacings)
                    min_spacing = min(spacings)
                    spacing_diff = max_spacing - min_spacing
                    
                    if spacing_diff > self.MAX_SPACING_NONUNIFORMITY:
                        return False, f"Spacing non uniforme: differenza {spacing_diff:.2f} mm > {self.MAX_SPACING_NONUNIFORMITY} mm"
            
            # 3. Leggi metadati dal primo file
            ds = pydicom.dcmread(dicom_files[0])
            
            # 4. Controlla slice thickness
            if 'SliceThickness' not in ds:
                return False, "SliceThickness non presente"
            
            slice_thickness = float(ds.SliceThickness)
            if slice_thickness < self.MIN_SLICE_THICKNESS:
                return False, f"Slice thickness ({slice_thickness:.3f} mm) < {self.MIN_SLICE_THICKNESS} mm"
            if slice_thickness > self.MAX_SLICE_THICKNESS:
                return False, f"Slice thickness ({slice_thickness:.3f} mm) > {self.MAX_SLICE_THICKNESS} mm"
            
            # 5. Controlla XY resolution
            if 'PixelSpacing' not in ds:
                return False, "PixelSpacing non presente"
            
            pixel_spacing = ds.PixelSpacing
            xy_resolution = float(pixel_spacing[0])
            
            if xy_resolution < self.MIN_XY_RESOLUTION:
                return False, f"XY resolution ({xy_resolution:.3f} mm) < {self.MIN_XY_RESOLUTION} mm"
            if xy_resolution > self.MAX_XY_RESOLUTION:
                return False, f"XY resolution ({xy_resolution:.3f} mm) > {self.MAX_XY_RESOLUTION} mm"
            
            # 6. Controlla data type (16-bit signed int)
            if 'BitsAllocated' not in ds or 'PixelRepresentation' not in ds:
                return False, "BitsAllocated o PixelRepresentation mancanti"
            
            bits_allocated = int(ds.BitsAllocated)
            pixel_representation = int(ds.PixelRepresentation)
            
            if bits_allocated != 16 or pixel_representation != 1:
                return False, f"Data type non valido (bits={bits_allocated}, signed={pixel_representation})"
            
            # Tutti i controlli passati
            metadata = {
                'slices': num_slices,
                'thickness': slice_thickness,
                'xy_resolution': xy_resolution,
                'bits': bits_allocated,
                'signed': pixel_representation
            }
            
            return True, metadata
            
        except Exception as e:
            return False, f"Errore lettura metadati: {str(e)}"
    
    def convert_dicom_to_mhd(self, dicom_files, output_filename):
        """
        Converte una serie DICOM in formato MHD/RAW
        
        Returns:
            (success, output_path_or_error)
        """
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            
            # Salva in formato MHD (include automaticamente il .raw)
            output_path = os.path.join(self.output_folder, output_filename)
            sitk.WriteImage(image, output_path)
            
            return True, output_path
        except Exception as e:
            return False, str(e)
    
    def process_dicom_folders(self, folders_list):
        """
        Fase 2: Verifica parametri DICOM e converte quelli validi
        """
        print("\n" + "="*70)
        print("FASE 2: Filtro per parametri DICOM e conversione")
        print("="*70)
        
        print("\nCRITERI DI FILTRO:")
        print(f"  • Slices: {self.MIN_SLICES} - {self.MAX_SLICES}")
        print(f"  • Slice thickness: {self.MIN_SLICE_THICKNESS} - {self.MAX_SLICE_THICKNESS} mm")
        print(f"  • XY resolution: {self.MIN_XY_RESOLUTION} - {self.MAX_XY_RESOLUTION} mm")
        print(f"  • Data type: 16-bit signed int")
        print("-"*70)
        
        # Crea cartella output
        os.makedirs(self.output_folder, exist_ok=True)
        
        # File di log
        log_file = os.path.join(self.output_folder, "conversion_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("LOG CONVERSIONE DICOM FILTRATI\n")
            f.write("="*60 + "\n")
            f.write(f"Parametri di filtro:\n")
            f.write(f"  - Slices: {self.MIN_SLICES} - {self.MAX_SLICES}\n")
            f.write(f"  - Slice thickness: {self.MIN_SLICE_THICKNESS} - {self.MAX_SLICE_THICKNESS} mm\n")
            f.write(f"  - XY resolution: {self.MIN_XY_RESOLUTION} - {self.MAX_XY_RESOLUTION} mm\n")
            f.write(f"  - Data type: 16-bit signed int\n")
            f.write("="*60 + "\n\n")
        
        processed_count = 0
        
        for folder_name, folder_path, file_count in folders_list:
            processed_count += 1
            print(f"\n[{processed_count}/{len(folders_list)}] {folder_name}")
            print(f"  File nella cartella: {file_count}")
            
            try:
                # Trova file DICOM
                reader = sitk.ImageSeriesReader()
                dicom_files = reader.GetGDCMSeriesFileNames(folder_path)
                
                if not dicom_files:
                    print(f"  ✗ Nessun file DICOM valido trovato")
                    self.stats['skipped_metadata'] += 1
                    self._write_to_log(log_file, folder_name, "Nessun file DICOM valido", False)
                    continue
                
                print(f"  File DICOM trovati: {len(dicom_files)}")
                
                # Verifica parametri
                is_valid, result = self.check_dicom_parameters(dicom_files, folder_name)
                
                if not is_valid:
                    print(f"  ✗ SCARTATO: {result}")
                    self.stats['skipped_metadata'] += 1
                    self._write_to_log(log_file, folder_name, result, False)
                    continue
                
                # Parametri validi
                metadata = result
                print(f"  ✓ PARAMETRI VALIDI:")
                print(f"     • Slices: {metadata['slices']}")
                print(f"     • Thickness: {metadata['thickness']:.3f} mm")
                print(f"     • XY resolution: {metadata['xy_resolution']:.3f} mm")
                print(f"     • Data type: {metadata['bits']}-bit signed")
                
                # Converti in MHD
                output_name = f"{folder_name}.mhd"
                success, output_path = self.convert_dicom_to_mhd(dicom_files, output_name)
                
                if success:
                    print(f"  ✓ CONVERTITO: {output_name}")
                    self.stats['converted'] += 1
                    
                    # Salva metadati
                    self._save_metadata(folder_name, metadata, output_path)
                    
                    # Log
                    log_msg = f"Slices: {metadata['slices']}, Thickness: {metadata['thickness']:.3f}mm, XY: {metadata['xy_resolution']:.3f}mm"
                    self._write_to_log(log_file, folder_name, log_msg, True, output_path)
                else:
                    print(f"  ✗ Errore conversione: {output_path}")
                    self.stats['errors'] += 1
                    self._write_to_log(log_file, folder_name, f"Errore conversione: {output_path}", False)
                    
            except Exception as e:
                print(f"  ✗ Errore durante elaborazione: {str(e)}")
                self.stats['errors'] += 1
                self._write_to_log(log_file, folder_name, f"Errore: {str(e)}", False)
    
    def _write_to_log(self, log_file, folder_name, message, converted=False, output_path=""):
        """Scrive nel file di log"""
        with open(log_file, 'a', encoding='utf-8') as f:
            if converted:
                f.write(f"✓ {folder_name}\n")
                f.write(f"  {message}\n")
                f.write(f"  Output: {output_path}\n\n")
            else:
                f.write(f"✗ {folder_name}\n")
                f.write(f"  {message}\n\n")
    
    def _save_metadata(self, folder_name, metadata, output_path):
        """Salva metadati in file separato"""
        meta_file = os.path.join(self.output_folder, f"{folder_name}_metadata.txt")
        with open(meta_file, 'w', encoding='utf-8') as f:
            f.write(f"METADATI DICOM: {folder_name}\n")
            f.write("="*40 + "\n")
            f.write(f"Numero di slices: {metadata['slices']}\n")
            f.write(f"Slice thickness: {metadata['thickness']:.3f} mm\n")
            f.write(f"XY resolution: {metadata['xy_resolution']:.3f} mm\n")
            f.write(f"Bits allocated: {metadata['bits']}\n")
            f.write(f"Signed: {metadata['signed']} (1=signed, 0=unsigned)\n")
            f.write(f"File MHD: {os.path.basename(output_path)}\n")
            f.write(f"File RAW: {os.path.basename(output_path).replace('.mhd', '.raw')}\n")
    
    def generate_final_report(self):
        """Genera report finale"""
        report_file = os.path.join(self.output_folder, "FINAL_REPORT.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("REPORT FINALE - FILTRAGGIO E CONVERSIONE DICOM\n")
            f.write("="*60 + "\n\n")
            
            f.write("STATISTICHE:\n")
            f.write("-"*40 + "\n")
            f.write(f"Cartelle totali analizzate: {self.stats['total_folders']}\n")
            f.write(f"Cartelle dopo filtro file count: {self.stats['kept_after_filecount']}\n")
            f.write(f"  • Scartate (pochi file): {self.stats['skipped_filecount']}\n")
            f.write(f"  • Convertite con successo: {self.stats['converted']}\n")
            f.write(f"  • Scartate (parametri DICOM): {self.stats['skipped_metadata']}\n")
            f.write(f"  • Errori durante elaborazione: {self.stats['errors']}\n\n")
            
            f.write("PARAMETRI DI FILTRAGGIO APPLICATI:\n")
            f.write("-"*40 + "\n")
            f.write(f"Slices: {self.MIN_SLICES} - {self.MAX_SLICES}\n")
            f.write(f"Slice thickness: {self.MIN_SLICE_THICKNESS} - {self.MAX_SLICE_THICKNESS} mm\n")
            f.write(f"XY resolution: {self.MIN_XY_RESOLUTION} - {self.MAX_XY_RESOLUTION} mm\n")
            f.write(f"Data type: 16-bit signed int\n\n")
            
            f.write("PERCORSI:\n")
            f.write("-"*40 + "\n")
            f.write(f"Input: {self.input_folder}\n")
            f.write(f"Output: {self.output_folder}\n")
            f.write(f"Cartelle rimosse: {self.removed_folder}\n\n")
            
            f.write("FILE GENERATI:\n")
            f.write("-"*40 + "\n")
            if os.path.exists(self.output_folder):
                mhd_files = [f for f in os.listdir(self.output_folder) if f.endswith('.mhd')]
                f.write(f"File MHD generati: {len(mhd_files)}\n")
                if mhd_files:
                    f.write("\nLista file:\n")
                    for mhd in sorted(mhd_files):
                        f.write(f"  - {mhd}\n")
        
        return report_file
    
    def run(self):
        """Esegue l'intero processo di filtraggio e conversione"""
        print("\n" + "="*70)
        print("DICOM FILTER & CONVERTER")
        print("="*70)
        
        # Verifica cartella input
        if not os.path.exists(self.input_folder):
            print(f"\n❌ ERRORE: Cartella input non trovata: {self.input_folder}")
            return
        
        print(f"\nCartella input: {self.input_folder}")
        print(f"Cartella output: {self.output_folder}")
        print("\nInizio elaborazione...")
        
        # Fase 1: Filtro per numero di file
        valid_folders = self.filter_by_file_count(min_files=100)
        
        if not valid_folders:
            print("\n⚠️  Nessuna cartella valida trovata dopo il primo filtro!")
            return
        
        # Fase 2: Filtro parametri DICOM e conversione
        self.process_dicom_folders(valid_folders)
        
        # Report finale
        report_file = self.generate_final_report()
        
        print("\n" + "="*70)
        print("ELABORAZIONE COMPLETATA!")
        print("="*70)
        print(f"\nRIEPILOGO:")
        print(f"• Cartelle processate: {self.stats['total_folders']}")
        print(f"• File MHD generati: {self.stats['converted']}")
        
        if self.stats['converted'] > 0:
            success_rate = self.stats['converted'] / max(self.stats['total_folders'], 1) * 100
            print(f"• Percentuale successo: {success_rate:.1f}%")
        
        print(f"\nOUTPUT:")
        print(f"• File MHD/RAW: {self.output_folder}/")
        print(f"• Report finale: {report_file}")
        print(f"• Log dettagliato: {self.output_folder}/conversion_log.txt")
        print(f"• Cartelle rimosse: {self.removed_folder}/")
        print("\n" + "="*70)


def main():
    """
    Script principale
    Modifica i percorsi qui sotto per adattarli al tuo caso
    """
    # CONFIGURAZIONE - MODIFICA QUESTI PERCORSI
    input_folder = "X:/Francesca Saglimbeni/tesi/datasets/OSIC"
    output_folder = "X:/Francesca Saglimbeni/tesi/vesselsegmentation/OSIC/OSIC_final"
    
    # Crea il filtro
    dicom_filter = DICOMFilter(input_folder, output_folder)
    
    # Esegui il processo completo
    dicom_filter.run()


if __name__ == "__main__":
    main()
