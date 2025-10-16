import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from pathlib import Path


def load_carve_annotation(annotation_path):
    """
    Carica il file di annotazione CARVE in formato .mhd.
    
    Labels CARVE:
        0: background (non vaso)
        1: vein (vena)
        2: artery (arteria)
        -999: unknown/uncertain (non lo so)
    
    Returns:
        annotation_array: Array con labels originali
        annotation_img: Immagine SimpleITK
    """
    annotation_img = sitk.ReadImage(annotation_path)
    annotation_array = sitk.GetArrayFromImage(annotation_img)
    
    # Stampa statistiche annotazioni
    print(f"\nCARVE Annotation Stats:")
    print(f"  Background (0):      {np.sum(annotation_array == 0):>10,} voxels")
    print(f"  Vein (1):            {np.sum(annotation_array == 1):>10,} voxels")
    print(f"  Artery (2):          {np.sum(annotation_array == 2):>10,} voxels")
    print(f"  Unknown (-999):      {np.sum(annotation_array == -999):>10,} voxels")
    print(f"  Total vessels (1+2): {np.sum((annotation_array == 1) | (annotation_array == 2)):>10,} voxels")
    
    return annotation_array, annotation_img


def load_vessel12_annotation(csv_path, reference_image):
    """
    Carica le annotazioni VESSEL12 da file CSV e crea una maschera 3D.
    
    Args:
        csv_path: Path al file CSV con annotazioni (formato: x,y,z,label)
        reference_image: SimpleITK image per ottenere dimensioni e spacing
    
    Returns:
        annotation_array: Array numpy 3D con 1=vaso, 0=non-vaso, -1=non annotato
    """
    # Leggi il CSV
    df = pd.read_csv(csv_path, names=['x', 'y', 'z', 'label'])
    
    # Ottieni dimensioni dall'immagine di riferimento
    size = reference_image.GetSize()  # (X, Y, Z)
    
    # Crea array vuoto (riempito con -1 per distinguere "non annotato" da "non-vaso")
    annotation_array = np.full((size[2], size[1], size[0]), -1, dtype=np.int16)
    
    # Popola l'array con le annotazioni
    # CSV usa coordinate 0-based: x,y,z
    # Array numpy è (Z, Y, X)
    for _, row in df.iterrows():
        x, y, z, label = int(row['x']), int(row['y']), int(row['z']), int(row['label'])
        
        # Verifica che le coordinate siano valide
        if 0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]:
            annotation_array[z, y, x] = label
    
    print(f"\nVESSEL12 Annotation Stats:")
    print(f"  Totale punti annotati: {len(df)}")
    print(f"  Vasi (label=1): {(df['label'] == 1).sum()}")
    print(f"  Non-vasi (label=0): {(df['label'] == 0).sum()}")
    print(f"  Volume annotato: {(annotation_array >= 0).sum()} voxels")
    
    return annotation_array


def compare_segmentation_with_annotations(segmentation_path, annotation_array, 
                                          annotation_type='carve', only_annotated=True):
    """
    Confronta la segmentazione con le annotazioni.
    Per CARVE: confronta se TotalSegmentator identifica come vaso tutto ciò che è vena (1) o arteria (2).
    Per VESSEL12: confronta con annotazioni sparse.
    
    Args:
        segmentation_path: Path alla segmentazione (cleaned vessels)
        annotation_array: Array numpy con annotazioni
        annotation_type: 'carve' o 'vessel12'
        only_annotated: Se True, esclude voxel "unknown" (-999 per CARVE, -1 per VESSEL12)
    
    Returns:
        metrics_dict: Dizionario con tutte le metriche
    """
    # Carica la segmentazione
    segmentation_img = sitk.ReadImage(segmentation_path)
    segmentation_array = sitk.GetArrayFromImage(segmentation_img)
    
    # Verifica dimensioni
    if segmentation_array.shape != annotation_array.shape:
        print(f"\nWARNING: Dimensioni diverse!")
        print(f"  Segmentazione: {segmentation_array.shape}")
        print(f"  Annotazioni: {annotation_array.shape}")
        
        # Ridimensiona se necessario (usa nearest neighbor per mantenere labels)
        seg_img_resampled = sitk.Resample(
            segmentation_img,
            sitk.GetImageFromArray(annotation_array),
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0.0,
            segmentation_img.GetPixelID()
        )
        segmentation_array = sitk.GetArrayFromImage(seg_img_resampled)
        print(f"  Segmentazione ridimensionata a: {segmentation_array.shape}")
    
    # Binarizza la segmentazione
    seg_binary = (segmentation_array > 0).astype(np.uint8)
    
    # Gestione diversa per CARVE e VESSEL12
    if annotation_type == 'carve':
        # CARVE: 0=background, 1=vein, 2=artery, -999=unknown
        
        # Maschera per voxel con annotazione certa (esclude -999 se only_annotated=True)
        if only_annotated:
            mask = annotation_array != -999
        else:
            mask = np.ones_like(annotation_array, dtype=bool)
        
        # Vasi = vene (1) + arterie (2)
        ann_vessels = ((annotation_array == 1) | (annotation_array == 2)).astype(np.uint8)
        
    elif annotation_type == 'vessel12':
        # VESSEL12: 1=vessel, 0=non-vessel, -1=not annotated
        if only_annotated:
            mask = annotation_array >= 0
        else:
            mask = np.ones_like(annotation_array, dtype=bool)
        
        ann_vessels = (annotation_array == 1).astype(np.uint8)
    
    # Calcola confusion matrix solo sui voxel considerati (escludendo unknown)
    tp = np.sum((seg_binary == 1) & (ann_vessels == 1) & mask)
    fp = np.sum((seg_binary == 1) & (ann_vessels == 0) & mask)
    fn = np.sum((seg_binary == 0) & (ann_vessels == 1) & mask)
    tn = np.sum((seg_binary == 0) & (ann_vessels == 0) & mask)
    
    # Calcola metriche
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Quanto dei vasi annotati viene trovato
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Quanto di ciò che trova è corretto
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calcola percentuale di overlap
    overlap_percentage = (tp / np.sum(ann_vessels & mask) * 100) if np.sum(ann_vessels & mask) > 0 else 0
    
    metrics = {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'iou': float(iou),
        'recall': float(recall),
        'precision': float(precision),
        'f1_score': float(f1_score),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'overlap_percentage': float(overlap_percentage),
        'total_annotated_voxels': int(np.sum(mask)),
        'total_vessel_voxels_annotated': int(np.sum(ann_vessels & mask)),
        'total_vessel_voxels_segmented': int(np.sum(seg_binary)),
        'unknown_voxels_excluded': int(np.sum(~mask))
    }
    
    return metrics


def print_metrics(metrics, dataset_name=""):
    """
    Stampa le metriche in modo leggibile.
    """
    print(f"\n{'='*70}")
    print(f"METRICHE DI VALUTAZIONE {dataset_name}")
    print(f"{'='*70}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"  True Positives (TP):  {metrics['true_positives']:>10,}  (vasi annotati trovati da TS)")
    print(f"  False Positives (FP): {metrics['false_positives']:>10,}  (TS trova vasi dove non ci sono)")
    print(f"  False Negatives (FN): {metrics['false_negatives']:>10,}  (vasi annotati NON trovati da TS)")
    print(f"  True Negatives (TN):  {metrics['true_negatives']:>10,}  (background corretto)")
    
    print(f"\n📈 Metriche Principali:")
    print(f"  IoU (Dice coefficient): {metrics['iou']:.4f}  (overlap vasi)")
    print(f"  Recall (Sensitivity):   {metrics['recall']:.4f}  ⭐ QUANTO TROVA DEI VASI ANNOTATI")
    print(f"  Precision:              {metrics['precision']:.4f}  (quanto è accurato)")
    print(f"  F1-Score:               {metrics['f1_score']:.4f}  (bilanciamento generale)")
    print(f"  Accuracy:               {metrics['accuracy']:.4f}  (accuratezza globale)")
    print(f"  Specificity:            {metrics['specificity']:.4f}  (riconoscimento background)")
    
    print(f"\n📦 Informazioni Volume:")
    print(f"  Voxel considerati (no unknown):  {metrics['total_annotated_voxels']:>10,}")
    print(f"  Voxel unknown esclusi:           {metrics['unknown_voxels_excluded']:>10,}")
    print(f"  Voxel vasi (ground truth):       {metrics['total_vessel_voxels_annotated']:>10,}")
    print(f"  Voxel vasi (segmentati da TS):   {metrics['total_vessel_voxels_segmented']:>10,}")
    print(f"  Overlap vasi: {metrics['overlap_percentage']:.2f}%")
    
    # Analisi qualitativa focalizzata su recall
    print(f"\n💡 Analisi per Artery/Vein Classification:")
    if metrics['recall'] >= 0.85:
        print(f"  ✅ OTTIMO: TS trova {metrics['recall']*100:.1f}% dei vasi annotati")
        print(f"     → Affidabile per classificazione arteria/vena")
    elif metrics['recall'] >= 0.70:
        print(f"  ⚠️  BUONO: TS trova {metrics['recall']*100:.1f}% dei vasi annotati")
        print(f"     → Usabile ma con cautela, {metrics['false_negatives']:,} vasi mancanti")
    else:
        print(f"  ❌ PROBLEMATICO: TS trova solo {metrics['recall']*100:.1f}% dei vasi annotati")
        print(f"     → {metrics['false_negatives']:,} vasi non rilevati, NON affidabile per classificazione")
    
    if metrics['precision'] < 0.70:
        print(f"  ⚠️  Precision bassa ({metrics['precision']:.2f}): molti falsi positivi")
        print(f"     → TS identifica vasi dove non ce ne sono ({metrics['false_positives']:,} FP)")
    
    print(f"{'='*70}\n")


def evaluate_carve(segmentation_path, annotation_path, exclude_unknown=True):
    """
    Valuta la segmentazione su dataset CARVE14.
    
    Args:
        segmentation_path: Path alla segmentazione TotalSegmentator
        annotation_path: Path alle annotazioni CARVE (_fullAnnotations.mhd)
        exclude_unknown: Se True, esclude voxel con label -999 dalla valutazione
    """
    print("\n🔬 VALUTAZIONE SU CARVE14")
    print(f"Segmentazione: {segmentation_path}")
    print(f"Annotazioni: {annotation_path}")
    print(f"Esclude unknown (-999): {exclude_unknown}")
    
    annotation_array, _ = load_carve_annotation(annotation_path)
    metrics = compare_segmentation_with_annotations(
        segmentation_path, 
        annotation_array, 
        annotation_type='carve',
        only_annotated=exclude_unknown
    )
    
    print_metrics(metrics, "- CARVE14")
    return metrics


def evaluate_vessel12(segmentation_path, annotation_csv_path, original_image_path):
    """
    Valuta la segmentazione su dataset VESSEL12.
    """
    print("\n🔬 VALUTAZIONE SU VESSEL12")
    print(f"Segmentazione: {segmentation_path}")
    print(f"Annotazioni: {annotation_csv_path}")
    print(f"Immagine originale: {original_image_path}")
    
    # Carica immagine originale per dimensioni
    reference_img = sitk.ReadImage(original_image_path)
    
    # Carica annotazioni da CSV
    annotation_array = load_vessel12_annotation(annotation_csv_path, reference_img)
    
    # Confronta con segmentazione
    metrics = compare_segmentation_with_annotations(
        segmentation_path, 
        annotation_array, 
        annotation_type='vessel12',
        only_annotated=True  # Considera solo punti esplicitamente annotati
    )
    
    print_metrics(metrics, "- VESSEL12")
    return metrics


def batch_evaluate_carve(segmentation_dir, annotation_dir, exclude_unknown=True):
    """
    Valuta tutte le scan CARVE14 in batch.
    """
    results = []
    
    seg_files = list(Path(segmentation_dir).glob('*_cleaned.nii.gz'))
    
    if not seg_files:
        print(f"⚠️  Nessun file *_cleaned.nii.gz trovato in {segmentation_dir}")
        return results
    
    for seg_file in seg_files:
        # Estrai l'ID della scan (rimuovi _cleaned.nii.gz)
        scan_id = seg_file.stem.replace('_cleaned', '')
        
        # Trova file annotazioni corrispondente
        ann_file = Path(annotation_dir) / f"{scan_id}_fullAnnotations.mhd"
        
        if ann_file.exists():
            print(f"\n{'='*70}")
            print(f"Processing: {scan_id}")
            try:
                metrics = evaluate_carve(str(seg_file), str(ann_file), exclude_unknown)
                metrics['scan_id'] = scan_id
                results.append(metrics)
            except Exception as e:
                print(f"❌ Errore durante l'elaborazione di {scan_id}: {e}")
        else:
            print(f"⚠️  Annotazioni non trovate per {scan_id}")
            print(f"    Cercato: {ann_file}")
    
    # Stampa statistiche aggregate
    if results:
        print(f"\n{'='*70}")
        print("STATISTICHE AGGREGATE - CARVE14")
        print(f"{'='*70}")
        df = pd.DataFrame(results)
        print(f"\nMedia metriche su {len(results)} scan:")
        print(f"  RECALL (quanto trova):    {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
        print(f"  PRECISION (quanto accurato): {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
        print(f"  F1-SCORE:                 {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")
        print(f"  IoU:                      {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
        
        print(f"\nMin/Max Recall:")
        print(f"  Migliore: {df['recall'].max():.4f} (scan: {df.loc[df['recall'].idxmax(), 'scan_id']})")
        print(f"  Peggiore: {df['recall'].min():.4f} (scan: {df.loc[df['recall'].idxmin(), 'scan_id']})")
    
    return results


# =============================================================================
# ESEMPI DI UTILIZZO
# =============================================================================

if __name__ == "__main__":
    
    # --- ESEMPIO 1: Singola scan CARVE14 ---
    print("\n" + "="*70)
    print("ESEMPIO 1: Valutazione singola scan CARVE14")
    print("="*70)
    
    carve_seg = '/content/vesselsegmentation/vessels_cleaned/1.2.840.113704.1.111.2604.1126357612.7_cleaned.nii.gz'
    carve_ann = '/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.2604.1126357612_fullAnnotations.mhd'
    
    if os.path.exists(carve_seg) and os.path.exists(carve_ann):
        carve_metrics = evaluate_carve(carve_seg, carve_ann, exclude_unknown=True)
    else:
        print("⚠️  File CARVE14 non trovati, salta questo esempio")
        print(f"    Segmentazione: {carve_seg}")
        print(f"    Annotazioni: {carve_ann}")
    
    '''
    # --- ESEMPIO 2: Singola scan VESSEL12 ---
    print("\n" + "="*70)
    print("ESEMPIO 2: Valutazione singola scan VESSEL12")
    print("="*70)
    
    vessel12_seg = '/content/vesselsegmentation/vessels_cleaned/VESSEL12_01_cleaned.nii.gz'
    vessel12_csv = '/content/vesselsegmentation/VESSEL12/VESSEL12_01-point-annotations.csv'
    vessel12_original = '/content/vesselsegmentation/VESSEL12/VESSEL12_01.mhd'
    
    if os.path.exists(vessel12_seg) and os.path.exists(vessel12_csv) and os.path.exists(vessel12_original):
        vessel12_metrics = evaluate_vessel12(vessel12_seg, vessel12_csv, vessel12_original)
    else:
        print("⚠️  File VESSEL12 non trovati, salta questo esempio")
    
    
    # --- ESEMPIO 3: Batch evaluation CARVE14 ---
    print("\n" + "="*70)
    print("ESEMPIO 3: Valutazione batch CARVE14")
    print("="*70)
    
    seg_dir = '/content/vesselsegmentation/vessels_cleaned'
    ann_dir = '/content/vesselsegmentation/CARVE14'
    
    if os.path.exists(seg_dir) and os.path.exists(ann_dir):
        batch_results = batch_evaluate_carve(seg_dir, ann_dir, exclude_unknown=True)
        
        # Salva risultati in CSV
        if batch_results:
            df_results = pd.DataFrame(batch_results)
            output_csv = '/content/vesselsegmentation/carve_evaluation_results.csv'
            df_results.to_csv(output_csv, index=False)
            print(f"\n✅ Risultati salvati in: {output_csv}")
    else:
        print("⚠️  Directory non trovate, salta batch evaluation")
        print(f"    Segmentazione dir: {seg_dir}")
        print(f"    Annotazioni dir: {ann_dir}")'''