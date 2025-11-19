"""
Script per valutare la QUALITÀ della RECONNECTION confrontando:
1. Ground Truth (CARVE annotations) vs BEFORE reconnection
2. Ground Truth (CARVE annotations) vs AFTER reconnection
3. Analisi delle connessioni: quali sono corrette secondo ground truth?

OBIETTIVO: Verificare se la reconnection MIGLIORA o PEGGIORA la segmentazione
           rispetto alle annotazioni manuali.
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree


def load_carve_annotation(annotation_path):
    """Carica annotazioni CARVE14."""
    annotation_img = sitk.ReadImage(annotation_path)
    annotation_array = sitk.GetArrayFromImage(annotation_img)
    
    print(f"\nCARVE Annotation Stats:")
    print(f"  Background (0):      {np.sum(annotation_array == 0):>10,} voxels")
    print(f"  Vein (1):            {np.sum(annotation_array == 1):>10,} voxels")
    print(f"  Artery (2):          {np.sum(annotation_array == 2):>10,} voxels")
    print(f"  Unknown (-999):      {np.sum(annotation_array == -999):>10,} voxels")
    print(f"  Total vessels (1+2): {np.sum((annotation_array == 1) | (annotation_array == 2)):>10,} voxels")
    
    return annotation_array, annotation_img


def compute_metrics(seg_binary, ann_vessels, mask):
    tp = np.sum((seg_binary == 1) & (ann_vessels == 1) & mask)
    fp = np.sum((seg_binary == 1) & (ann_vessels == 0) & mask)
    fn = np.sum((seg_binary == 0) & (ann_vessels == 1) & mask)
    tn = np.sum((seg_binary == 0) & (ann_vessels == 0) & mask)
    
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Quanto dei vasi annotati viene trovato
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Quanto di ciò che trova è corretto
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'iou': float(iou),
        'recall': float(recall),
        'precision': float(precision),
        'f1_score': float(f1_score),
    }


def analyze_connections(before_array, after_array, gt_array, spacing, mask):
    # Binarizza
    before_binary = (before_array > 0).astype(bool)
    after_binary = (after_array > 0).astype(bool)
    gt_vessels = ((gt_array == 1) | (gt_array == 2)).astype(bool)
    
    # Voxel AGGIUNTI dalla reconnection
    added_voxels = after_binary & ~before_binary
    num_added = np.sum(added_voxels)
    
    print(f"  Total voxels added by reconnection: {num_added:,}")
    
    if num_added == 0:
        return {
            'added_voxels': 0,
            'added_correct': 0,
            'added_incorrect': 0,
            'accuracy_added': 0.0,
            'reconnection_benefit': 0.0
        }
    
    # Verifica correttezza dei voxel aggiunti (rispetto a GT)
    added_correct = added_voxels & gt_vessels & mask  # TP: aggiunti che sono veri vasi
    added_incorrect = added_voxels & ~gt_vessels & mask  # FP: aggiunti che non sono vasi
    
    num_correct = np.sum(added_correct)
    num_incorrect = np.sum(added_incorrect)
    
    accuracy_added = num_correct / num_added if num_added > 0 else 0
    
    gt_vessel_voxels = np.sum(gt_vessels & mask)
    reconnection_benefit = num_correct / gt_vessel_voxels if gt_vessel_voxels > 0 else 0
    
    print(f"  Reconnection benefit (recall improvement): +{100*reconnection_benefit:.3f}%")
    
    return {
        'added_voxels': int(num_added),
        'added_correct': int(num_correct),
        'added_incorrect': int(num_incorrect),
        'accuracy_added': float(accuracy_added),
        'reconnection_benefit': float(reconnection_benefit),
    }


def analyze_component_reconnections(before_array, after_array, gt_array, spacing, mask):
    # Per ogni gruppo di componenti BEFORE che sono diventate 1 componente AFTER
    print("\n=== ANALYZING COMPONENT RECONNECTIONS ===")
    
    before_binary = (before_array > 0).astype(bool)
    after_binary = (after_array > 0).astype(bool)
    gt_vessels = ((gt_array == 1) | (gt_array == 2)).astype(bool)
    
    before_labeled, num_before = ndimage.label(before_binary)
    after_labeled, num_after = ndimage.label(after_binary)
    
    print(f"  Components BEFORE: {num_before}")
    print(f"  Components AFTER: {num_after}")
    print(f"  Component reduction: {num_before - num_after} ({100*(num_before-num_after)/num_before:.1f}%)")
    
    reconnections_analyzed = 0
    correct_reconnections = 0
    incorrect_reconnections = 0
    
    for after_label in range(1, num_after + 1):
        after_comp_mask = (after_labeled == after_label)
        
        before_labels_in_after = np.unique(before_labeled[after_comp_mask])
        before_labels_in_after = before_labels_in_after[before_labels_in_after > 0]
        
        if len(before_labels_in_after) > 1:
            reconnections_analyzed += 1
            merged_before_mask = np.isin(before_labeled, before_labels_in_after)
            gt_in_region = gt_vessels & merged_before_mask
            gt_labeled_region, num_gt_components = ndimage.label(gt_in_region)
            if num_gt_components == 1:
                correct_reconnections += 1
            else:
                incorrect_reconnections += 1
    
    accuracy_component = correct_reconnections / reconnections_analyzed if reconnections_analyzed > 0 else 0
    
    return {
        'num_components_before': int(num_before),
        'num_components_after': int(num_after),
        'component_reduction': int(num_before - num_after),
        'reconnections_analyzed': int(reconnections_analyzed),
        'correct_reconnections': int(correct_reconnections),
        'incorrect_reconnections': int(incorrect_reconnections),
        'component_accuracy': float(accuracy_component),
    }


def evaluate_reconnection(before_path, after_path, gt_path, spacing=(0.625, 0.625, 0.625), exclude_unknown=True):
    # confronta BEFORE vs AFTER vs GROUND TRUTH.    
    # Carica files
    gt_array, gt_img = load_carve_annotation(gt_path)
    
    before_img = sitk.ReadImage(before_path)
    before_array = sitk.GetArrayFromImage(before_img)
    
    after_img = sitk.ReadImage(after_path)
    after_array = sitk.GetArrayFromImage(after_img)
    
    # Verifica dimensioni e resample se necessario
    if before_array.shape != gt_array.shape:
        before_img_resampled = sitk.Resample(before_img, gt_img, sitk.Transform(), 
                                             sitk.sitkNearestNeighbor, 0.0, before_img.GetPixelID())
        before_array = sitk.GetArrayFromImage(before_img_resampled)
        
        after_img_resampled = sitk.Resample(after_img, gt_img, sitk.Transform(),
                                            sitk.sitkNearestNeighbor, 0.0, after_img.GetPixelID())
        after_array = sitk.GetArrayFromImage(after_img_resampled)
    
    # Crea maschera (esclude unknown se richiesto)
    if exclude_unknown:
        mask = gt_array != -999
    else:
        mask = np.ones_like(gt_array, dtype=bool)
    
    gt_vessels = ((gt_array == 1) | (gt_array == 2)).astype(np.uint8)
    print("\n" + "="*70)
    print("📊 METRICS BEFORE RECONNECTION")
    print("="*70)
    
    before_binary = (before_array > 0).astype(np.uint8)
    metrics_before = compute_metrics(before_binary, gt_vessels, mask)
    
    print(f"  IoU:       {metrics_before['iou']:.4f}")
    print(f"  Recall:    {metrics_before['recall']:.4f}  ⭐ (quanto trova dei vasi GT)")
    print(f"  Precision: {metrics_before['precision']:.4f}  (quanto è accurato)")
    print(f"  F1-Score:  {metrics_before['f1_score']:.4f}")
    
    print("\n" + "="*70)
    print("📊 METRICS AFTER RECONNECTION")
    print("="*70)
    
    after_binary = (after_array > 0).astype(np.uint8)
    metrics_after = compute_metrics(after_binary, gt_vessels, mask)
    
    print(f"  IoU:       {metrics_after['iou']:.4f}")
    print(f"  Recall:    {metrics_after['recall']:.4f}  ⭐")
    print(f"  Precision: {metrics_after['precision']:.4f}")
    print(f"  F1-Score:  {metrics_after['f1_score']:.4f}")

    print("\n" + "="*70)
    print(" IMPROVEMENT ANALYSIS (AFTER - BEFORE) ")
    print("="*70)
    
    delta_iou = metrics_after['iou'] - metrics_before['iou']
    delta_recall = metrics_after['recall'] - metrics_before['recall']
    delta_precision = metrics_after['precision'] - metrics_before['precision']
    delta_f1 = metrics_after['f1_score'] - metrics_before['f1_score']
    
    def format_delta(value, is_positive_good=True):
        sign = '+' if value >= 0 else ''
        symbol = '✅' if (value > 0 and is_positive_good) or (value < 0 and not is_positive_good) else '❌' if value != 0 else '➖'
        return f"{sign}{value:+.4f} {symbol}"
    
    print(f"  Δ IoU:       {format_delta(delta_iou)}")
    print(f"  Δ Recall:    {format_delta(delta_recall)}")
    print(f"  Δ Precision: {format_delta(delta_precision, is_positive_good=True)}")
    print(f"  Δ F1-Score:  {format_delta(delta_f1)}")
    
    connection_stats = analyze_connections(before_array, after_array, gt_array, spacing, mask)
    
    component_stats = analyze_component_reconnections(before_array, after_array, gt_array, spacing, mask)
    
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT")
    print("="*70)
    
    # Calcola score composito
    overall_improvement = (delta_recall * 0.5 +  # Recall più importante
                          delta_f1 * 0.3 +       # F1 per bilanciamento
                          delta_iou * 0.2)       # IoU per overlap
    
    reconnection_quality = connection_stats['accuracy_added']
    
    print(f"\n  Overall improvement score: {overall_improvement:+.4f}")
    print(f"  Reconnection quality: {100*reconnection_quality:.1f}% correct paths")
    print(f"  Component accuracy: {100*component_stats['component_accuracy']:.1f}%")
    
    if overall_improvement > 0.01 and reconnection_quality > 0.7:
        verdict = " EXCELLENT - Reconnection significantly improves segmentation"
    elif overall_improvement > 0.005 and reconnection_quality > 0.5:
        verdict = " GOOD - Reconnection provides marginal improvement"
    elif overall_improvement > -0.005 and overall_improvement <= 0.005:
        verdict = " NEUTRAL - Reconnection has minimal impact"
    elif overall_improvement > -0.01:
        verdict = " CAUTION - Reconnection slightly degrades segmentation"
    else:
        verdict = " POOR - Reconnection significantly degrades segmentation"
    
    print(f"\n  {verdict}")
    results = {
        'scan_id': os.path.basename(before_path).replace('_cleaned.nii.gz', '').replace('vessels_before_reconnection', ''),
        
        # Metriche BEFORE
        'iou_before': metrics_before['iou'],
        'recall_before': metrics_before['recall'],
        'precision_before': metrics_before['precision'],
        'f1_before': metrics_before['f1_score'],
        
        # Metriche AFTER
        'iou_after': metrics_after['iou'],
        'recall_after': metrics_after['recall'],
        'precision_after': metrics_after['precision'],
        'f1_after': metrics_after['f1_score'],
        
        # Deltas
        'delta_iou': delta_iou,
        'delta_recall': delta_recall,
        'delta_precision': delta_precision,
        'delta_f1': delta_f1,
        
        # Connection stats
        **connection_stats,
        
        # Component stats
        **component_stats,
        
        # Overall
        'overall_improvement': overall_improvement,
        'verdict': verdict,
    }
    
    return results

before_path = '/content/vesselsegmentation/vessels_cleaned/vessels_before_reconnection.nii.gz'
after_path = '/content/vesselsegmentation/vessels_cleaned/1.2.840.113704.1.111.2604.1126357612.7_cleaned.nii.gz'
gt_path = '/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.2604.1126357612_fullAnnotations.mhd'

if os.path.exists(before_path) and os.path.exists(after_path) and os.path.exists(gt_path):
    results = evaluate_reconnection(before_path, after_path, gt_path)
else:
    print("⚠️  Files not found for single evaluation")
    print(f"    BEFORE: {before_path}")
    print(f"    AFTER: {after_path}")
    print(f"    GT: {gt_path}")