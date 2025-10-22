import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy import ndimage
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import json


def load_carve_annotation(annotation_path, verbose=False):
    annotation_img = sitk.ReadImage(annotation_path)
    annotation_array = sitk.GetArrayFromImage(annotation_img)

    # Statistiche (non stampate, ma restituite per eventuale uso)
    stats = {
        "background": int(np.sum(annotation_array == 0)),
        "vein": int(np.sum(annotation_array == 1)),
        "artery": int(np.sum(annotation_array == 2)),
        "unknown": int(np.sum(annotation_array == -999))
    }
    stats["total_vessels"] = stats["vein"] + stats["artery"] + stats["unknown"]
    return annotation_array, annotation_img, stats


def analyze_error_spatial_distribution(segmentation_array, annotation_array, mask, spacing):
    """Analizza la distribuzione spaziale degli errori (FP e FN)."""
    z_max = segmentation_array.shape[0]
    z_upper = z_max // 3
    z_lower = 2 * z_max // 3

    seg_binary = (segmentation_array > 0).astype(np.uint8)
    ann_vessels = annotation_array 

    regions = {
        'superior': (0, z_upper),
        'middle': (z_upper, z_lower),
        'inferior': (z_lower, z_max)
    }

    stats = {}
    for region_name, (z_start, z_end) in regions.items():
        region_mask = mask.copy()
        region_mask[:z_start] = False
        region_mask[z_end:] = False

        tp = np.sum((seg_binary == 1) & (ann_vessels == 1) & region_mask)
        fp = np.sum((seg_binary == 1) & (ann_vessels == 0) & region_mask)
        fn = np.sum((seg_binary == 0) & (ann_vessels == 1) & region_mask)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        stats[region_name] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'recall': float(recall),
            'precision': float(precision)
        }
    return stats


def analyze_error_by_vessel_size(segmentation_array, annotation_array, mask, spacing):
    """Analizza errori in base alla dimensione dei vasi (piccoli/medi/grandi)."""
    labeled_ann, num_ann = ndimage.label(annotation_array & mask)
    if num_ann == 0:
        return {}

    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    vessel_volumes = ndimage.sum(annotation_array & mask, labeled_ann, range(1, num_ann + 1))
    vessel_volumes_mm3 = np.array(vessel_volumes) * voxel_volume
    vessel_diameters = 2.0 * ((3.0 * vessel_volumes_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0))

    seg_binary = (segmentation_array > 0)

    size_categories = {
        'small': (0, 3.0),      # < 3mm
        'medium': (3.0, 6.0),   # 3-6mm
        'large': (6.0, 100.0)   # > 6mm
    }

    stats = {}
    for cat_name, (min_diam, max_diam) in size_categories.items():
        cat_mask = (vessel_diameters >= min_diam) & (vessel_diameters < max_diam)
        cat_labels = np.where(cat_mask)[0] + 1
        if len(cat_labels) == 0:
            continue

        cat_vessel_mask = np.isin(labeled_ann, cat_labels)
        tp = np.sum(seg_binary & cat_vessel_mask)
        fn = np.sum(~seg_binary & cat_vessel_mask)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        stats[cat_name] = {
            'num_vessels': int(len(cat_labels)),
            'total_voxels': int(np.sum(cat_vessel_mask)),
            'detected_voxels': int(tp),
            'missed_voxels': int(fn),
            'recall': float(recall),
            'avg_diameter_mm': float(vessel_diameters[cat_mask].mean())
        }
    return stats


def create_error_masks(segmentation_array, annotation_array, mask, output_dir, reference_img):
    """
    Crea maschere di errore.
    Output:
        - error_false_positives.nii.gz
        - error_false_negatives.nii.gz
        - error_overlay.nii.gz  (0=TN, 1=TP, 2=FP, 3=FN)
    """
    os.makedirs(output_dir, exist_ok=True)

    seg_binary = (segmentation_array > 0)
    ann_vessels = annotation_array

    tp_mask = (seg_binary & ann_vessels & mask).astype(np.uint8)
    fp_mask = (seg_binary & ~ann_vessels & mask).astype(np.uint8)
    fn_mask = (~seg_binary & ann_vessels & mask).astype(np.uint8)

    # Overlay (0=bg, 1=TP, 2=FP, 3=FN)
    overlay = np.zeros_like(seg_binary, dtype=np.uint8)
    overlay[tp_mask > 0] = 1
    overlay[fp_mask > 0] = 2
    overlay[fn_mask > 0] = 3

    # Salvataggi senza print
    fp_img = sitk.GetImageFromArray(fp_mask)
    fp_img.CopyInformation(reference_img)
    fp_path = os.path.join(output_dir, "error_false_positives.nii.gz")
    sitk.WriteImage(fp_img, fp_path)

    fn_img = sitk.GetImageFromArray(fn_mask)
    fn_img.CopyInformation(reference_img)
    fn_path = os.path.join(output_dir, "error_false_negatives.nii.gz")
    sitk.WriteImage(fn_img, fn_path)

    overlay_img = sitk.GetImageFromArray(overlay)
    overlay_img.CopyInformation(reference_img)
    overlay_path = os.path.join(output_dir, "error_overlay.nii.gz")
    sitk.WriteImage(overlay_img, overlay_path)

    return fp_path, fn_path, overlay_path


def generate_error_report(segmentation_path, annotation_array,
                          output_dir, airway_mask_path=None, lung_mask_path=None,
                          exclude_unknown=False, verbose=False, save_json=True):
    os.makedirs(output_dir, exist_ok=True)

    # Carica segmentazione
    segmentation_img = sitk.ReadImage(segmentation_path)
    segmentation_array = sitk.GetArrayFromImage(segmentation_img)
    spacing = segmentation_img.GetSpacing()[::-1]  # (z, y, x)
    if exclude_unknown:
        mask = annotation_array != -999
    else:
        mask = np.ones_like(annotation_array, dtype=bool)

    ann_vessels = ((annotation_array == 1) |
                    (annotation_array == 2) |
                    (annotation_array == -999)).astype(np.uint8)

    # Statistiche (utili per raccomandazioni)
    n_vein = int(np.sum(annotation_array == 1))
    n_artery = int(np.sum(annotation_array == 2))
    n_unknown = int(np.sum(annotation_array == -999))
    n_background = int(np.sum(annotation_array == 0))
    n_total_vessels = n_vein + n_artery + n_unknown

    carve_stats = {
        "vein": n_vein,
        "artery": n_artery,
        "unknown": n_unknown,
        "total_vessels": n_total_vessels,
        "background": n_background
    }

    airway_mask = None
    if airway_mask_path and os.path.exists(airway_mask_path):
        airway_mask = sitk.GetArrayFromImage(sitk.ReadImage(airway_mask_path)).astype(bool)
    else:
        airway_mask = np.zeros_like(mask, dtype=bool)

    pleura_mask = None
    if lung_mask_path and os.path.exists(lung_mask_path):
        lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_path)).astype(bool)
        lung_eroded = ndimage.binary_erosion(lung_mask, iterations=2)
        pleura_mask = lung_mask & ~lung_eroded

    spatial_stats = analyze_error_spatial_distribution(segmentation_array, ann_vessels, mask, spacing)
    size_stats = analyze_error_by_vessel_size(segmentation_array, ann_vessels, mask, spacing)

    # Maschere di errore (salvate senza print)
    fp_path, fn_path, overlay_path = create_error_masks(
        segmentation_array, ann_vessels, mask, output_dir, segmentation_img
    )

    # NIENTE visualizzazione 2D: rimosso salvataggio PNG

    report_data = {
        'annotation_breakdown': carve_stats,
        'spatial_distribution': spatial_stats,
        'vessel_size_analysis': size_stats,
        'outputs': {
            'overlay_path': overlay_path,
            'false_positives_path': fp_path,
            'false_negatives_path': fn_path
        }
    }

    if save_json:
        report_json_path = os.path.join(output_dir, "error_analysis_report.json")
        with open(report_json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        report_data['outputs']['report_json_path'] = report_json_path

    return report_data


segmentation_path = 'CARVE14/1.2.840.113704.1.111.1396.1132404220.7.mhd'
annotation_path   = 'CARVE14/1.2.840.113704.1.111.1396.1132404220_fullAnnotations.mhd'
airway_mask_path  = 'vessels_cleaned/airways_full.nii.gz'
lung_mask_path    = 'vessels_cleaned/lung_mask_original.nii.gz'
output_dir        = 'vessels_cleaned/error_analysis'

if not (os.path.exists(segmentation_path) and os.path.exists(annotation_path)):
    raise FileNotFoundError(f"Missing file(s). "
                            f"Segmentation: {segmentation_path} | Annotation: {annotation_path}")

segmentation_img = sitk.ReadImage(segmentation_path)
ext = os.path.splitext(annotation_path)[1].lower()
annotation_array, _, _ = load_carve_annotation(annotation_path, verbose=False)

_ = generate_error_report(
    segmentation_path=segmentation_path,
    annotation_array=annotation_array,
    output_dir=output_dir,
    airway_mask_path=airway_mask_path,
    lung_mask_path=lung_mask_path,
    exclude_unknown=True,  
    verbose=False,
    save_json=True
)
