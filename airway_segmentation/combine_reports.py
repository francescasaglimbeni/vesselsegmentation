import os
from pathlib import Path

def extract_report_section(file_path):
    """
    Estrae la sezione del report da BRONCHIAL TREE STATISTICS a CLINICAL INTERPRETATION
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Trova l'inizio della sezione
        start_marker = "=" * 80 + "\nBRONCHIAL TREE STATISTICS\n" + "=" * 80
        start_idx = content.find(start_marker)
        
        if start_idx == -1:
            print(f"  ⚠ Marker iniziale non trovato in {file_path}")
            return None
        
        # Trova la fine della sezione (prima di CLINICAL INTERPRETATION)
        end_marker = "=" * 80 + "\nCLINICAL INTERPRETATION\n" + "=" * 80
        end_idx = content.find(end_marker, start_idx)
        
        if end_idx == -1:
            print(f"  ⚠ Marker finale non trovato in {file_path}")
            return None
        
        # Estrai la sezione
        section = content[start_idx:end_idx]
        return section
    
    except Exception as e:
        print(f"  ✗ Errore nella lettura di {file_path}: {e}")
        return None


def combine_all_reports(base_dir, output_file):
    """
    Combina tutti i report dei pazienti in un unico file
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"✗ Directory non trovata: {base_dir}")
        return
    
    # Trova tutte le sottodirectory dei pazienti (qualsiasi nome di directory che contiene il report)
    patient_dirs = [d for d in base_path.iterdir() if d.is_dir() and (d / "COMPLETE_ANALYSIS_REPORT.txt").exists()]
    patient_dirs = sorted(patient_dirs)
    
    print(f"Trovate {len(patient_dirs)} directory di pazienti")
    print()
    
    combined_content = []
    successful_count = 0
    failed_count = 0
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        report_file = patient_dir / "COMPLETE_ANALYSIS_REPORT.txt"
        
        print(f"Processando: {patient_id}")
        
        if not report_file.exists():
            print(f"  ✗ File report non trovato")
            failed_count += 1
            continue
        
        section = extract_report_section(report_file)
        
        if section:
            # Aggiungi un header per identificare il paziente
            header = f"\n\n{'=' * 80}\n"
            header += f"PAZIENTE: {patient_id}\n"
            header += f"{'=' * 80}\n\n"
            
            combined_content.append(header + section)
            print(f"  ✓ Estratto con successo")
            successful_count += 1
        else:
            failed_count += 1
    
    # Scrivi il file combinato
    if combined_content:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORT COMBINATO - ANALISI MULTI-PAZIENTE\n")
            f.write("=" * 80 + "\n")
            f.write(f"\nTotale pazienti analizzati: {successful_count}\n")
            f.write(f"Data generazione: {Path().absolute()}\n")
            f.write("\n" + "=" * 80 + "\n")
            
            for content in combined_content:
                f.write(content)
        
        print()
        print("=" * 80)
        print(f"✓ Report combinato creato con successo!")
        print(f"  File: {output_path}")
        print(f"  Pazienti processati con successo: {successful_count}")
        print(f"  Pazienti falliti: {failed_count}")
        print("=" * 80)
    else:
        print()
        print("✗ Nessun dato estratto. Impossibile creare il report combinato.")


if __name__ == "__main__":
    # Directory base con i risultati
    base_directory = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\output_results_with_fibrosis"
    
    # File di output
    output_file = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\COMBINED_REPORTS.txt"
    
    print("=" * 80)
    print("SCRIPT DI COMBINAZIONE REPORT")
    print("=" * 80)
    print()
    
    combine_all_reports(base_directory, output_file)
