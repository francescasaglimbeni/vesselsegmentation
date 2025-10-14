import SimpleITK as sitk
import numpy as np

def load_annotation(annotation_path):
    """
    Carica il file di annotazione in formato .mhd (vasi=1)
    """
    annotation_img = sitk.ReadImage(annotation_path)
    annotation_array = sitk.GetArrayFromImage(annotation_img)
    return annotation_array

def compare_segmentation_with_annotations(segmentation_path, annotation_array):
    """
    Confronta la segmentazione con le annotazioni per determinare quanto i vasi segmentati corrispondono
    a vasi nelle annotazioni, ignorando la distinzione tra vena e arteria.
    """
    # Carica la segmentazione ottenuta (vasi segmentati)
    segmentation_img = sitk.ReadImage(segmentation_path)
    segmentation_array = sitk.GetArrayFromImage(segmentation_img)

    # Crea un array di risultati dove:
    # 1 - Corrisponde a vaso (qualunque tipo),
    # 0 - Non corrisponde (non è un vaso).
    result = np.zeros_like(segmentation_array)

    # Confronta ogni punto segmentato con le annotazioni
    # Se la segmentazione è non zero e l'annotazione è 1 (vaso), marca come corrispondente.
    result[(annotation_array == 1) & (segmentation_array > 0)] = 1

    return result, segmentation_array

def calculate_metrics(result, annotation_array, segmentation_array):
    """
    Calcola l'IoU, la Recall, e mostra FP e FN.
    """
    # Calcola i veri positivi, falsi positivi e falsi negativi
    true_positives = np.sum(result == 1)  # Vasi correttamente segmentati
    false_positives = np.sum((result == 1) & (annotation_array != 1))  # Vasi segmentati erroneamente
    false_negatives = np.sum((result == 0) & (annotation_array == 1))  # Vasi non segmentati

    # IoU = TP / (TP + FP + FN)
    iou = true_positives / (true_positives + false_positives + false_negatives)

    # Recall = TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives)

    # Visualizza il numero di FP e FN
    print(f"Numero di veri positivi (TP): {true_positives}")
    print(f"Numero di falsi positivi (FP): {false_positives}")
    print(f"Numero di falsi negativi (FN): {false_negatives}")

    return {
        'iou': iou,
        'recall': recall,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

# Esempio di utilizzo
annotation_path = '/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.1396.1132404220_fullAnnotations.mhd'  # Percorso del file delle annotazioni
segmentation_path = '/content/vesselsegmentation/vessels_cleaned/1.2.840.113704.1.111.1396.1132404220.7_cleaned.nii.gz'  # Percorso della segmentazione dei vasi

# Carica annotazioni
annotation_array = load_annotation(annotation_path)

# Confronta la segmentazione con le annotazioni
result, segmentation_array = compare_segmentation_with_annotations(segmentation_path, annotation_array)

# Calcola IoU, Recall, FP e FN per la segmentazione dei vasi
metrics = calculate_metrics(result, annotation_array, segmentation_array)
print(f"IoU per i vasi segmentati: {metrics['iou']}")
print(f"Recall per i vasi segmentati: {metrics['recall']}")
