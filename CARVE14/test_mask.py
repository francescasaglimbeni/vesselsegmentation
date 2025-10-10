import SimpleITK as sitk

def check_and_fix_mask(mask_path):
    # Leggi l'immagine della maschera
    mask = sitk.ReadImage(mask_path)
    
    # Verifica i valori unici nella maschera
    unique_values = sitk.GetArrayFromImage(mask).flatten()
    print(f"Valori unici nella maschera: {set(unique_values)}")
    
    # Se la maschera non è binaria, binarizzala
    if len(set(unique_values)) > 2:
        print("La maschera non è binaria. Eseguiamo una binarizzazione...")
        mask = sitk.BinaryThreshold(mask, 0.5, 255, 1, 0)  # Binarizza la maschera (0 o 1)
        sitk.WriteImage(mask, mask_path)  # Salva la maschera binarizzata
        print("Maschera binarizzata e salvata.")
    
    return mask

# Esegui la verifica sulla maschera
check_and_fix_mask('CARVE14\masks\lung_without_airways.nii')
