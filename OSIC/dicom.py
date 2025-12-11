import SimpleITK as sitk
import os
from pathlib import Path

# cartelle
dicom_folder = "X:/Francesca Saglimbeni/tesi/train_OSIC_compatible"
output_folder = "output"

# crea la cartella output se non esiste
os.makedirs(output_folder, exist_ok=True)

# itera su tutte le cartelle nella dicom_folder
for patient_dir in os.listdir(dicom_folder):
    patient_path = os.path.join(dicom_folder, patient_dir)
    
    # salta se non è una cartella
    if not os.path.isdir(patient_path):
        continue
    
    try:
        # leggi la serie DICOM
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(patient_path)
        
        # salta se non ci sono file DICOM
        if not dicom_names:
            print(f"Nessun file DICOM trovato in: {patient_path}")
            continue
        
        reader.SetFileNames(dicom_names)
        
        # leggi come volume 3D
        image = reader.Execute()
        
        # salva in formato MHD/RAW
        output_path = os.path.join(output_folder, f"{patient_dir}.mhd")
        sitk.WriteImage(image, output_path)
        print(f"✓ Convertito: {patient_dir} -> {output_path}")
        
    except Exception as e:
        print(f"✗ Errore nel processare {patient_dir}: {e}")