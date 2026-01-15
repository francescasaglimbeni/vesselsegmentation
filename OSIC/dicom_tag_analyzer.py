"""
DICOM Tag Analyzer
==================

Analizza i DICOM tag di una cartella di pazienti e trova:
1. Tag comuni a TUTTI i pazienti
2. Valori univoci vs variabili per ogni tag
3. Statistiche sui tag più frequenti
4. Report dettagliato con confronto

Autore: Pipeline OSIC
Data: 2026-01-15
"""

import os
import json
import pydicom
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DicomTagAnalyzer:
    """
    Analizza i DICOM tag di una cartella di pazienti
    """
    
    def __init__(self, input_folder, verbose=True):
        self.input_folder = input_folder
        self.verbose = verbose
        
        # Dizionari per raccogliere dati
        self.patient_tags = {}  # {patient_id: {tag_name: [values]}}
        self.tag_frequency = Counter()  # Quanti pazienti hanno ogni tag
        self.tag_values = defaultdict(lambda: defaultdict(list))  # {tag_name: {patient_id: [values]}}
        
    
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
            if item.name.startswith('.') or item.name in ['removed_low_filecount', 'incompatible', 'output', 'validated']:
                continue
            
            # Verifica se contiene file DICOM
            dcm_files = list(item.glob("*.dcm"))
            if len(dcm_files) > 0:
                patients.append((item.name, item))
        
        return patients
    
    
    def extract_dicom_tags(self, dicom_file):
        """
        Estrae tutti i DICOM tag da un file
        
        Returns:
            dict con {tag_name: value}
        """
        try:
            ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=True)
            
            tags = {}
            
            # Itera su tutti gli elementi del DICOM
            for elem in ds:
                # Nome del tag (es: "PatientName", "StudyDate", etc.)
                tag_name = elem.keyword if elem.keyword else f"Tag_{elem.tag}"
                
                # Valore del tag
                try:
                    if elem.VR == 'SQ':  # Sequence - troppo complesso
                        value = f"<Sequence with {len(elem.value)} items>"
                    elif hasattr(elem.value, '__iter__') and not isinstance(elem.value, (str, bytes)):
                        # Array/list
                        value = str(list(elem.value))
                    else:
                        value = str(elem.value)
                except:
                    value = "<Cannot convert>"
                
                tags[tag_name] = value
            
            return tags
            
        except Exception as e:
            if self.verbose:
                print(f"    ⚠️  Errore lettura {dicom_file.name}: {e}")
            return {}
    
    
    def analyze_patient(self, patient_id, patient_folder):
        """
        Analizza tutti i DICOM di un paziente e raccoglie i tag
        
        Returns:
            dict con {tag_name: [values]} per questo paziente
        """
        dcm_files = list(Path(patient_folder).glob("*.dcm"))
        
        if len(dcm_files) == 0:
            if self.verbose:
                print(f"  ⚠️  Nessun DICOM trovato")
            return {}
        
        # Raccoglie tag da tutti i file del paziente
        patient_tag_values = defaultdict(set)  # {tag_name: set(values)}
        
        # Analizza un campione di file (primi 10 + ultimi 10 + alcuni centrali)
        sample_size = min(30, len(dcm_files))
        if len(dcm_files) <= sample_size:
            sample_files = dcm_files
        else:
            # Campiona strategicamente
            first_10 = dcm_files[:10]
            last_10 = dcm_files[-10:]
            middle_10 = dcm_files[len(dcm_files)//2-5:len(dcm_files)//2+5]
            sample_files = list(set(first_10 + last_10 + middle_10))
        
        for dcm_file in sample_files:
            tags = self.extract_dicom_tags(dcm_file)
            
            for tag_name, value in tags.items():
                patient_tag_values[tag_name].add(value)
        
        # Converti set in list per JSON serializzazione
        result = {}
        for tag_name, values in patient_tag_values.items():
            result[tag_name] = sorted(list(values))
        
        return result
    
    
    def analyze_all_patients(self):
        """
        Analizza tutti i pazienti e raccoglie statistiche
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("DICOM TAG ANALYZER")
            print(f"{'='*70}")
            print(f"Input folder: {self.input_folder}")
        
        # Trova pazienti
        patients = self.find_dicom_patients()
        
        if len(patients) == 0:
            if self.verbose:
                print(f"✗ Nessun paziente trovato")
            return None
        
        if self.verbose:
            print(f"\nPazienti trovati: {len(patients)}")
            print(f"\n{'='*70}")
            print("ANALISI TAG PER PAZIENTE")
            print(f"{'='*70}")
        
        # Analizza ogni paziente
        for idx, (patient_id, patient_folder) in enumerate(patients, 1):
            if self.verbose:
                print(f"\n[{idx}/{len(patients)}] {patient_id}")
            
            patient_tags = self.analyze_patient(patient_id, patient_folder)
            
            if self.verbose:
                print(f"  Tag trovati: {len(patient_tags)}")
            
            # Salva dati paziente
            self.patient_tags[patient_id] = patient_tags
            
            # Aggiorna statistiche globali
            for tag_name, values in patient_tags.items():
                self.tag_frequency[tag_name] += 1
                self.tag_values[tag_name][patient_id] = values
        
        # Genera report
        return self._generate_report(len(patients))
    
    
    def _generate_report(self, total_patients):
        """
        Genera report con analisi tag comuni
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("ANALISI TAG COMUNI")
            print(f"{'='*70}")
        
        # 1. Tag presenti in TUTTI i pazienti
        common_tags = [tag for tag, count in self.tag_frequency.items() 
                      if count == total_patients]
        
        if self.verbose:
            print(f"\nTag comuni a TUTTI i {total_patients} pazienti: {len(common_tags)}")
        
        # 2. Analizza variabilità dei tag comuni
        tag_analysis = {}
        
        for tag_name in sorted(common_tags):
            # Raccogli tutti i valori univoci per questo tag
            all_values = set()
            for patient_id, values in self.tag_values[tag_name].items():
                all_values.update(values)
            
            # Determina se il valore è costante o variabile
            is_constant = (len(all_values) == 1)
            
            tag_analysis[tag_name] = {
                'present_in': self.tag_frequency[tag_name],
                'unique_values': sorted(list(all_values)),
                'is_constant': is_constant,
                'value_count': len(all_values),
            }
        
        # 3. Tag parzialmente presenti (in almeno 50% dei pazienti)
        partial_tags = [tag for tag, count in self.tag_frequency.items() 
                       if count >= total_patients * 0.5 and count < total_patients]
        
        if self.verbose:
            print(f"\nTag presenti in almeno 50% pazienti: {len(partial_tags)}")
        
        # Genera report strutturato
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_folder': self.input_folder,
            'total_patients': total_patients,
            'statistics': {
                'total_unique_tags': len(self.tag_frequency),
                'common_tags_count': len(common_tags),
                'partial_tags_count': len(partial_tags),
            },
            'common_tags': tag_analysis,
            'tag_frequency': dict(self.tag_frequency),
            'patient_tags': self.patient_tags,
        }
        
        # Stampa report dettagliato
        if self.verbose:
            self._print_detailed_report(report, common_tags, partial_tags)
        
        return report
    
    
    def _print_detailed_report(self, report, common_tags, partial_tags):
        """
        Stampa report dettagliato su console
        """
        print(f"\n{'='*70}")
        print("TAG COMUNI A TUTTI I PAZIENTI")
        print(f"{'='*70}")
        
        # Separa tag costanti e variabili
        constant_tags = [tag for tag in common_tags 
                        if report['common_tags'][tag]['is_constant']]
        variable_tags = [tag for tag in common_tags 
                        if not report['common_tags'][tag]['is_constant']]
        
        # 1. Tag COSTANTI (stesso valore per tutti)
        print(f"\n--- TAG COSTANTI (stesso valore per tutti i pazienti) ---")
        print(f"Totale: {len(constant_tags)}\n")
        
        for tag in sorted(constant_tags):
            info = report['common_tags'][tag]
            value = info['unique_values'][0]
            # Tronca valori molto lunghi
            if len(value) > 80:
                value = value[:80] + "..."
            print(f"  {tag:40s} = {value}")
        
        # 2. Tag VARIABILI (valori diversi tra pazienti)
        print(f"\n--- TAG VARIABILI (valori diversi tra pazienti) ---")
        print(f"Totale: {len(variable_tags)}\n")
        
        for tag in sorted(variable_tags):
            info = report['common_tags'][tag]
            print(f"  {tag:40s} ({info['value_count']} valori diversi)")
            
            # Mostra esempi di valori (max 5)
            examples = info['unique_values'][:5]
            for val in examples:
                if len(val) > 60:
                    val = val[:60] + "..."
                print(f"    - {val}")
            
            if len(info['unique_values']) > 5:
                print(f"    ... e altri {len(info['unique_values']) - 5} valori")
        
        # 3. Tag parziali (presenti in >= 50% pazienti)
        if len(partial_tags) > 0:
            print(f"\n{'='*70}")
            print("TAG PARZIALI (presenti in almeno 50% pazienti)")
            print(f"{'='*70}\n")
            
            for tag in sorted(partial_tags, key=lambda t: self.tag_frequency[t], reverse=True):
                count = self.tag_frequency[tag]
                percentage = (count / report['total_patients']) * 100
                print(f"  {tag:40s} - {count}/{report['total_patients']} pazienti ({percentage:.1f}%)")
        
        # 4. Statistiche tag più rari
        rare_tags = [tag for tag, count in self.tag_frequency.items() 
                    if count < report['total_patients'] * 0.5]
        
        if len(rare_tags) > 0:
            print(f"\n{'='*70}")
            print(f"TAG RARI (presenti in < 50% pazienti)")
            print(f"{'='*70}")
            print(f"\nTotale tag rari: {len(rare_tags)}")
            print(f"\nTop 10 tag rari più frequenti:")
            
            sorted_rare = sorted(rare_tags, key=lambda t: self.tag_frequency[t], reverse=True)[:10]
            for tag in sorted_rare:
                count = self.tag_frequency[tag]
                percentage = (count / report['total_patients']) * 100
                print(f"  {tag:40s} - {count}/{report['total_patients']} pazienti ({percentage:.1f}%)")
    
    
    def save_report(self, output_path="dicom_tag_analysis.json"):
        """
        Salva report in JSON
        """
        report = self.analyze_all_patients()
        
        if report is None:
            return None
        
        # Salva JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ Report salvato: {output_path}")
            print(f"{'='*70}")
        
        # Genera anche un report human-readable
        txt_path = output_path.replace('.json', '.txt')
        self._save_human_readable_report(report, txt_path)
        
        return report
    
    
    def _save_human_readable_report(self, report, output_path):
        """
        Salva un report human-readable in formato testo
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DICOM TAG ANALYSIS REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"\nTimestamp: {report['timestamp']}\n")
            f.write(f"Input folder: {report['input_folder']}\n")
            f.write(f"Total patients: {report['total_patients']}\n")
            
            f.write(f"\n{'='*70}\n")
            f.write("STATISTICS\n")
            f.write(f"{'='*70}\n")
            f.write(f"Total unique tags found: {report['statistics']['total_unique_tags']}\n")
            f.write(f"Tags common to ALL patients: {report['statistics']['common_tags_count']}\n")
            f.write(f"Tags in at least 50% patients: {report['statistics']['partial_tags_count']}\n")
            
            f.write(f"\n{'='*70}\n")
            f.write("COMMON TAGS (present in ALL patients)\n")
            f.write(f"{'='*70}\n")
            
            # Separa costanti e variabili
            constant_tags = [tag for tag, info in report['common_tags'].items() 
                           if info['is_constant']]
            variable_tags = [tag for tag, info in report['common_tags'].items() 
                           if not info['is_constant']]
            
            f.write(f"\n--- CONSTANT TAGS ({len(constant_tags)} tags) ---\n")
            f.write("(Same value across all patients)\n\n")
            
            for tag in sorted(constant_tags):
                info = report['common_tags'][tag]
                value = info['unique_values'][0]
                f.write(f"{tag:40s} = {value}\n")
            
            f.write(f"\n--- VARIABLE TAGS ({len(variable_tags)} tags) ---\n")
            f.write("(Different values across patients)\n\n")
            
            for tag in sorted(variable_tags):
                info = report['common_tags'][tag]
                f.write(f"\n{tag} ({info['value_count']} different values):\n")
                
                for val in info['unique_values'][:10]:
                    f.write(f"  - {val}\n")
                
                if len(info['unique_values']) > 10:
                    f.write(f"  ... and {len(info['unique_values']) - 10} more values\n")
            
            f.write(f"\n{'='*70}\n")
            f.write("TAG FREQUENCY TABLE\n")
            f.write(f"{'='*70}\n\n")
            
            sorted_tags = sorted(report['tag_frequency'].items(), 
                               key=lambda x: x[1], reverse=True)
            
            for tag, count in sorted_tags:
                percentage = (count / report['total_patients']) * 100
                f.write(f"{tag:50s} {count:4d}/{report['total_patients']:4d} ({percentage:5.1f}%)\n")
        
        if self.verbose:
            print(f"✅ Report testo salvato: {output_path}")


def main():
    """
    Main function per analisi DICOM tag
    """
    # CONFIGURAZIONE
    input_folder = r"X:\Francesca Saglimbeni\tesi\results\results_OSIC\CORRECT\DICOM"
    output_json = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\OSIC\dicom_tag_analysis.json"
    
    # Crea analyzer
    analyzer = DicomTagAnalyzer(input_folder, verbose=True)
    
    # Esegui analisi e salva report
    report = analyzer.save_report(output_json)
    
    if report:
        print(f"\n✅ Analisi completata!")
        print(f"\nFile generati:")
        print(f"  - {output_json}")
        print(f"  - {output_json.replace('.json', '.txt')}")
    else:
        print(f"\n✗ Analisi fallita")


if __name__ == "__main__":
    main()
