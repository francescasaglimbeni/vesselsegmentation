import os
import shutil
import SimpleITK as sitk
import pydicom
from pathlib import Path
import numpy as np

class DICOMFilterConverter:
    def __init__(self):
        # PARAMETRI DI FILTRAGGIO (MODIFICATI)
        self.MIN_SLICES = 350      # > 400 slice (almeno 401)
        self.MAX_SLICES = 1000     # Limite massimo alto
        self.MIN_SLICE_THICKNESS = 0.6    # mm (minimum)
        self.MIN_XY_RESOLUTION = 0.4     # mm
        self.MAX_XY_RESOLUTION = 1.00     # mm
        
        # PERCORSI (modifica questi)
        self.input_folder = "X:/Francesca Saglimbeni/tesi/train_OSIC"  # Cartella con i DICOM
        self.output_folder = "X:/Francesca Saglimbeni/tesi/train_OSIC_output"            # Cartella output
        self.removed_folder = os.path.join(self.input_folder, "removed_low_filecount")
        
        # STATISTICHE
        self.stats = {
            'total_folders': 0,
            'kept_after_filecount': 0,
            'converted': 0,
            'skipped_metadata': 0,
            'skipped_filecount': 0,
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
        Prima fase: rimuove cartelle con troppo pochi file
        (probabilmente non sono serie DICOM complete)
        """
        print("\n" + "=" * 70)
        print("FASE 1: Filtro preliminare per numero di file")
        print("=" * 70)
        
        os.makedirs(self.removed_folder, exist_ok=True)
        
        folders_to_process = []
        
        # Scansiona tutte le sottocartelle
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
        print(f"  Cartelle scartate (pochi file): {self.stats['skipped_filecount']}")
        
        return folders_to_process
    
    def check_dicom_parameters(self, dicom_files, folder_name):
        """
        Verifica se la serie DICOM rispetta tutti i criteri tecnici
        """
        try:
            if not dicom_files or len(dicom_files) == 0:
                return False, "Nessun file DICOM valido"
            
            # 1. Controlla numero di slices
            num_slices = len(dicom_files)
            if num_slices < self.MIN_SLICES:  # Deve essere >= MIN_SLICES (350)
                return False, f"Slices ({num_slices}) < {self.MIN_SLICES} (richiesto >= {self.MIN_SLICES})"
            
            # 2. Leggi metadati dal primo file
            ds = pydicom.dcmread(dicom_files[0])
            
            # 3. Controlla slice thickness
            if 'SliceThickness' not in ds:
                return False, "SliceThickness non presente"
            
            slice_thickness = float(ds.SliceThickness)
            if slice_thickness < self.MIN_SLICE_THICKNESS:
                return False, f"Slice thickness ({slice_thickness} mm) < {self.MIN_SLICE_THICKNESS} mm"
            
            # 4. Controlla XY resolution
            if 'PixelSpacing' not in ds:
                return False, "PixelSpacing non presente"
            
            pixel_spacing = ds.PixelSpacing
            xy_resolution = float(pixel_spacing[0])
            
            if not (self.MIN_XY_RESOLUTION <= xy_resolution <= self.MAX_XY_RESOLUTION):
                return False, f"XY resolution ({xy_resolution} mm) fuori range"
            
            # 5. Controlla data type (16-bit signed int)
            if 'BitsAllocated' not in ds or 'PixelRepresentation' not in ds:
                return False, "Metadati tipo dati mancanti"
            
            bits_allocated = int(ds.BitsAllocated)
            pixel_representation = int(ds.PixelRepresentation)
            
            if bits_allocated != 16 or pixel_representation != 1:
                return False, f"Tipo dati: {bits_allocated}-bit, signed={pixel_representation}"
            
            # Tutti i controlli passati
            metadata_summary = {
                'slices': num_slices,
                'thickness': slice_thickness,
                'xy_resolution': xy_resolution,
                'bits': bits_allocated,
                'signed': pixel_representation
            }
            
            return True, metadata_summary
            
        except Exception as e:
            return False, f"Errore lettura metadati: {str(e)}"
    
    def convert_dicom_series(self, dicom_files, output_filename):
        """
        Converte una serie DICOM in formato MHD/RAW
        """
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            
            # Salva in formato MHD (che include automaticamente il .raw)
            output_path = os.path.join(self.output_folder, output_filename)
            sitk.WriteImage(image, output_path)
            
            return True, output_path
        except Exception as e:
            return False, str(e)
    
    def process_dicom_folders(self, folders_list):
        """
        Seconda fase: controlla i parametri DICOM e converte quelli validi
        """
        print("\n" + "=" * 70)
        print("FASE 2: Filtro per parametri DICOM e conversione")
        print("=" * 70)
        
        print("CRITERI DI FILTRO:")
        print(f"  • Slices: >= {self.MIN_SLICES}")
        print(f"  • Slice thickness: ≥ {self.MIN_SLICE_THICKNESS} mm")
        print(f"  • XY resolution: {self.MIN_XY_RESOLUTION}-{self.MAX_XY_RESOLUTION} mm")
        print(f"  • Data type: 16-bit signed int")
        print("-" * 70)
        
        # Crea cartella output
        os.makedirs(self.output_folder, exist_ok=True)
        
        # File di log dettagliato
        log_file = os.path.join(self.output_folder, "conversion_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("LOG CONVERSIONE DICOM FILTRATI\n")
            f.write("=" * 60 + "\n")
            f.write(f"Parametri di filtro:\n")
            f.write(f"  - Slices: >= {self.MIN_SLICES}\n")
            f.write(f"  - Slice thickness: ≥ {self.MIN_SLICE_THICKNESS} mm\n")
            f.write(f"  - XY resolution: {self.MIN_XY_RESOLUTION}-{self.MAX_XY_RESOLUTION} mm\n")
            f.write(f"  - Data type: 16-bit signed int\n")
            f.write("=" * 60 + "\n\n")
        
        processed_count = 0
        
        for folder_name, folder_path, file_count in folders_list:
            processed_count += 1
            print(f"\n[{processed_count}/{len(folders_list)}] {folder_name}")
            print(f"  File totali nella cartella: {file_count}")
            
            try:
                # Trova file DICOM nella cartella
                reader = sitk.ImageSeriesReader()
                dicom_files = reader.GetGDCMSeriesFileNames(folder_path)
                
                if not dicom_files:
                    print("  ✗ Nessuna serie DICOM trovata")
                    self.stats['skipped_metadata'] += 1
                    self._write_to_log(log_file, folder_name, "Nessuna serie DICOM trovata", False)
                    continue
                
                print(f"  File DICOM trovati: {len(dicom_files)}")
                
                # Controlla parametri DICOM
                is_valid, result = self.check_dicom_parameters(dicom_files, folder_name)
                
                if not is_valid:
                    print(f"  ✗ Parametri non validi: {result}")
                    self.stats['skipped_metadata'] += 1
                    self._write_to_log(log_file, folder_name, result, False)
                    continue
                
                # Parametri validi, procedi con conversione
                metadata = result
                print(f"  ✓ PARAMETRI VALIDI:")
                print(f"     • Slices: {metadata['slices']} (>= {self.MIN_SLICES} ✓)")
                print(f"     • Thickness: {metadata['thickness']:.3f} mm")
                print(f"     • XY resolution: {metadata['xy_resolution']:.3f} mm")
                print(f"     • Data type: {metadata['bits']}-bit signed")
                
                # Converti in MHD
                output_name = f"{folder_name}.mhd"
                success, output_path = self.convert_dicom_series(dicom_files, output_name)
                
                if success:
                    print(f"  ✓ CONVERTITO → {output_path}")
                    self.stats['converted'] += 1
                    
                    # Scrivi dettagli nel log
                    log_details = (f"Slices: {metadata['slices']} (>= {self.MIN_SLICES} ✓), "
                                  f"Thickness: {metadata['thickness']:.3f} mm, "
                                  f"XY-res: {metadata['xy_resolution']:.3f} mm")
                    self._write_to_log(log_file, folder_name, log_details, True, output_path)
                    
                    # Crea anche un file di metadati per riferimento
                    self._save_metadata(folder_name, metadata, output_path)
                else:
                    print(f"  ✗ Errore conversione: {output_path}")
                    self.stats['errors'] += 1
                    self._write_to_log(log_file, folder_name, f"Errore conversione: {output_path}", False)
                    
            except Exception as e:
                print(f"  ✗ Errore durante l'elaborazione: {str(e)}")
                self.stats['errors'] += 1
                self._write_to_log(log_file, folder_name, f"Errore: {str(e)}", False)
    
    def _write_to_log(self, log_file, folder_name, message, converted=False, output_path=""):
        """Scrive una voce nel file di log"""
        with open(log_file, 'a', encoding='utf-8') as f:
            if converted:
                f.write(f"✓ {folder_name}\n")
                f.write(f"  {message}\n")
                f.write(f"  Output: {output_path}\n\n")
            else:
                f.write(f"✗ {folder_name}\n")
                f.write(f"  {message}\n\n")
    
    def _save_metadata(self, folder_name, metadata, output_path):
        """Salva i metadati in un file separato per riferimento"""
        meta_file = os.path.join(self.output_folder, f"{folder_name}_metadata.txt")
        with open(meta_file, 'w', encoding='utf-8') as f:
            f.write(f"METADATI DICOM: {folder_name}\n")
            f.write("=" * 40 + "\n")
            f.write(f"Numero di slices: {metadata['slices']} (>= {self.MIN_SLICES} ✓)\n")
            f.write(f"Slice thickness: {metadata['thickness']} mm\n")
            f.write(f"XY resolution: {metadata['xy_resolution']} mm\n")
            f.write(f"Bits allocated: {metadata['bits']}\n")
            f.write(f"Signed: {metadata['signed']} (1=signed, 0=unsigned)\n")
            f.write(f"File MHD: {os.path.basename(output_path)}\n")
            f.write(f"File RAW: {os.path.basename(output_path).replace('.mhd', '.raw')}\n")
    
    def generate_final_report(self):
        """Genera un report finale del processo"""
        report_file = os.path.join(self.output_folder, "FINAL_REPORT.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("REPORT FINALE - FILTRAGGIO E CONVERSIONE DICOM\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("STATISTICHE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Cartelle totali analizzate: {self.stats['total_folders']}\n")
            f.write(f"Cartelle dopo filtro file count: {self.stats['kept_after_filecount']}\n")
            f.write(f"  • Scartate (pochi file): {self.stats['skipped_filecount']}\n")
            f.write(f"  • Convertite con successo: {self.stats['converted']}\n")
            f.write(f"  • Scartate (parametri DICOM): {self.stats['skipped_metadata']}\n")
            f.write(f"  • Errori durante elaborazione: {self.stats['errors']}\n\n")
            
            f.write("PARAMETRI DI FILTRAGGIO APPLICATI:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Slices per CT: >= {self.MIN_SLICES}\n")
            f.write(f"Slice thickness: ≥ {self.MIN_SLICE_THICKNESS} mm\n")
            f.write(f"XY resolution: {self.MIN_XY_RESOLUTION}-{self.MAX_XY_RESOLUTION} mm\n")
            f.write(f"Data type: 16-bit signed int\n\n")
            
            f.write("PERCORSI:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Input: {self.input_folder}\n")
            f.write(f"Output: {self.output_folder}\n")
            f.write(f"Cartelle rimosse (pochi file): {self.removed_folder}\n\n")
            
            f.write("FILE GENERATI:\n")
            f.write("-" * 40 + "\n")
            if os.path.exists(self.output_folder):
                mhd_files = [f for f in os.listdir(self.output_folder) if f.endswith('.mhd')]
                f.write(f"File MHD generati: {len(mhd_files)}\n")
                
                if mhd_files:
                    f.write("\nLista file convertiti:\n")
                    for mhd in sorted(mhd_files):
                        # Leggi i metadati dal file corrispondente
                        meta_file = mhd.replace('.mhd', '_metadata.txt')
                        if os.path.exists(os.path.join(self.output_folder, meta_file)):
                            with open(os.path.join(self.output_folder, meta_file), 'r') as mf:
                                first_line = mf.readline().strip()
                                slices_line = ""
                                for line in mf:
                                    if "Numero di slices" in line:
                                        slices_line = line.strip()
                                        break
                            f.write(f"  • {mhd} - {slices_line}\n")
                        else:
                            f.write(f"  • {mhd}\n")
        
        return report_file
    
    def run(self):
        """Esegue l'intero processo"""
        print("\n" + "=" * 70)
        print("DICOM FILTER & CONVERTER")
        print("=" * 70)
        print(f"CRITERIO SLICE: >= {self.MIN_SLICES}\n")
        
        # Verifica cartella input
        if not os.path.exists(self.input_folder):
            print(f"ERRORE: Cartella input non trovata: {self.input_folder}")
            print("Modifica 'input_folder' nello script con il percorso corretto.")
            return
        
        print(f"Cartella input: {self.input_folder}")
        print(f"Cartella output: {self.output_folder}")
        print("\nInizio elaborazione...")
        
        # Fase 1: Filtro per numero di file
        valid_folders = self.filter_by_file_count(min_files=100)
        
        if not valid_folders:
            print("\n⚠️  Nessuna cartella valida trovata dopo il primo filtro!")
            print("Controlla che i DICOM siano organizzati in sottocartelle.")
            return
        
        # Fase 2: Filtro parametri DICOM e conversione
        self.process_dicom_folders(valid_folders)
        
        # Report finale
        report_file = self.generate_final_report()
        
        print("\n" + "=" * 70)
        print("ELABORAZIONE COMPLETATA!")
        print("=" * 70)
        print(f"\nRIEPILOGO:")
        print(f"• Cartelle processate: {self.stats['total_folders']}")
        print(f"• File MHD generati: {self.stats['converted']}")
        
        if self.stats['converted'] > 0:
            # Calcola statistiche aggiuntive
            mhd_files = [f for f in os.listdir(self.output_folder) if f.endswith('.mhd')]
            print(f"• File nella cartella output: {len(mhd_files)}")
            
            # Conta quanti hanno esattamente > 400 slice
            slices_count = {}
            for mhd in mhd_files:
                meta_file = mhd.replace('.mhd', '_metadata.txt')
                meta_path = os.path.join(self.output_folder, meta_file)
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as mf:
                        for line in mf:
                            if "Numero di slices" in line:
                                slices = int(line.split(':')[1].strip().split()[0])
                                if slices not in slices_count:
                                    slices_count[slices] = 0
                                slices_count[slices] += 1
                                break
            
            if slices_count:
                max_slices = max(slices_count.keys())
                min_slices = min(slices_count.keys())
                avg_slices = sum(k*v for k, v in slices_count.items()) / sum(slices_count.values())
                
                print(f"• Statistiche slice:")
                print(f"  - Minimo: {min_slices} slice")
                print(f"  - Massimo: {max_slices} slice")
                print(f"  - Media: {avg_slices:.1f} slice")
                print(f"  - Tutti >= {self.MIN_SLICES}: {'SI' if min_slices >= self.MIN_SLICES else 'NO'}")
        
        success_rate = self.stats['converted']/max(self.stats['total_folders'],1)*100
        print(f"• Percentuale successo: {success_rate:.1f}%")
        print(f"\nOUTPUT:")
        print(f"• File MHD/RAW: {self.output_folder}/")
        print(f"• Report finale: {report_file}")
        print(f"• Log dettagliato: {self.output_folder}/conversion_log.txt")
        print(f"• Cartelle rimosse (pochi file): {self.removed_folder}/")
        print("\n" + "=" * 70)

# ========== ESECUZIONE ==========
if __name__ == "__main__":
    # Installa le dipendenze se non le hai già:
    # pip install SimpleITK pydicom numpy
    
    # Crea e avvia il processore
    processor = DICOMFilterConverter()
    
    # MODIFICA QUESTI PERCORSI SE NECESSARIO:
    '''processor.input_folder = "C:/Users/sagli/Downloads/test"  # Cartella con i DICOM
    processor.output_folder = "filtered_mhd_output"            # Dove salvare i MHD
    '''
    # MODIFICA QUESTI PARAMETRI SE NECESSARIO:
    # processor.MIN_SLICES = 401       # > 400
    # processor.MIN_SLICE_THICKNESS = 0.7
    # processor.MIN_XY_RESOLUTION = 0.5
    # processor.MAX_XY_RESOLUTION = 1.0
    
    # Avvia il processo
    processor.run()