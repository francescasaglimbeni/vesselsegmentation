import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy import ndimage
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import json

def load_vessel12_annotation_csv(csv_path, reference_img, radius_fallback_mm=1.0):
    """
    Carica annotazioni VESSEL12 da CSV e restituisce:
      - ann_array: maschera binaria (np.uint8) con 1=vessel
      - stats: conteggi utili

    Gestisce:
      1) CSV con header: (i,j,k) o (z,y,x) [indici voxel], oppure (x,y,z) [mm]
      2) CSV senza header: 3 o 4 colonne numeriche -> interpretate come (i,j,k) in voxel;
         l'eventuale 4ª colonna viene ignorata (spesso è etichetta/diametro).
    """
    # Primo tentativo: leggi "normalmente"
    df = pd.read_csv(csv_path)
    # Se non troviamo le colonne attese e TUTTE le colonne hanno nomi "numerici" (prima riga usata come header),
    # ricarichiamo senza header.
    def _all_numeric_like(cols):
        try:
            _ = [float(c) for c in cols]
            return True
        except Exception:
            return False

    lower_cols = {c.lower(): c for c in df.columns}
    expected_sets = [
        ('i', 'j', 'k'),
        ('z', 'y', 'x'),
        ('x', 'y', 'z')
    ]
    has_expected = any(all(t in lower_cols for t in s) for s in expected_sets)

    if (not has_expected) and _all_numeric_like(df.columns):
        # ricarica forzando header=None e separatore auto
        df = pd.read_csv(csv_path, header=None)
        df = df.dropna(how='all')
        if df.shape[1] < 3:
            raise ValueError(f"CSV con {df.shape[1]} colonne senza header: servono almeno 3 colonne (i,j,k).")
        # Tieni solo le prime 4 colonne (se esistono)
        if df.shape[1] > 4:
            df = df.iloc[:, :4]
        # Rinomina: 3 colonne -> i,j,k ; 4 colonne -> i,j,k,extra (ignorata)
        if df.shape[1] == 3:
            df.columns = ['i', 'j', 'k']
        else:  # 4 colonne
            df.columns = ['i', 'j', 'k', 'extra']

    # Ricalcola lower_cols dopo l’eventuale reload
    lower_cols = {c.lower(): c for c in df.columns}

    # Determina modalità coordinate
    coord = None
    mode = None  # 'index' o 'physical'
    for s in expected_sets:
        if all(t in lower_cols for t in s):
            coord = tuple(lower_cols[t] for t in s)
            mode = 'index' if s in [('i','j','k'), ('z','y','x')] else 'physical'
            break
    # Se ancora niente, prova il caso "senza header" già mappato a i,j,k
    if coord is None and all(k in lower_cols for k in ('i','j','k')):
        coord = (lower_cols['i'], lower_cols['j'], lower_cols['k'])
        mode = 'index'

    if coord is None:
        raise ValueError(
            f"Colonne coordinate non trovate nel CSV. "
            f"Attese una tra [i,j,k], [z,y,x], [x,y,z] oppure CSV senza header a 3/4 colonne."
        )

    # Geometria riferimento
    size = list(reference_img.GetSize())       
    spacing = list(reference_img.GetSpacing()) 

    ann = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)  # (z,y,x)

    def phys_to_index(pt_xyz):
        return reference_img.TransformPhysicalPointToIndex(tuple(pt_xyz))  # -> (i,j,k)

    from math import ceil
    def paint_sphere(kji, r_mm):
        kz, jy, ix = kji
        rx = max(r_mm / spacing[0], 0.0)
        ry = max(r_mm / spacing[1], 0.0)
        rz = max(r_mm / spacing[2], 0.0)
        wx, wy, wz = ceil(rx), ceil(ry), ceil(rz)
        zmin, zmax = max(0, kz - wz), min(ann.shape[0]-1, kz + wz)
        ymin, ymax = max(0, jy - wy), min(ann.shape[1]-1, jy + wy)
        xmin, xmax = max(0, ix - wx), min(ann.shape[2]-1, ix + wx)
        zz = np.arange(zmin, zmax+1)
        yy = np.arange(ymin, ymax+1)
        xx = np.arange(xmin, xmax+1)
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing='ij')
        if rx > 0 and ry > 0 and rz > 0:
            val = ((X - ix)/rx)**2 + ((Y - jy)/ry)**2 + ((Z - kz)/rz)**2
            mask_loc = (val <= 1.0)
        else:
            mask_loc = (X==ix) & (Y==jy) & (Z==kz)
        ann[Z, Y, X] |= mask_loc.astype(np.uint8)

    # Loop punti
    num_points = 0
    for _, row in df.iterrows():
        try:
            if mode == 'index':
                # Supporta (i,j,k) o (z,y,x)
                a, b, c = row[coord[0]], row[coord[1]], row[coord[2]]
                if coord[0].lower() == 'i':      # (i,j,k)
                    i, j, k = int(round(a)), int(round(b)), int(round(c))
                elif coord[0].lower() == 'z':    # (z,y,x) -> (i,j,k) = (x,y,z)
                    k, j, i = int(round(a)), int(round(b)), int(round(c))
                else:
                    # Se siamo qui con 'index', trattiamo come (i,j,k)
                    i, j, k = int(round(a)), int(round(b)), int(round(c))
            else:
                # 'physical' (x,y,z) in mm
                x, y, z = float(row[coord[0]]), float(row[coord[1]]), float(row[coord[2]])
                i, j, k = phys_to_index((x, y, z))
        except Exception:
            continue  # salta righe non valide

        if not (0 <= i < size[0] and 0 <= j < size[1] and 0 <= k < size[2]):
            continue

        paint_sphere((k, j, i), radius_fallback_mm)
        num_points += 1

    stats = {"num_points": int(num_points), "foreground_voxels": int(ann.sum())}
    return ann.astype(np.uint8), stats


def load_carve_annotation(annotation_path, verbose=False):
    """
    Carica il file di annotazione CARVE (.mhd).

    Labels CARVE:
        0: background (non vaso)
        1: vein (vena)
        2: artery (arteria)
        -999: vessel unknown type (VASO di tipo sconosciuto, NON background!)
    """
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
    ann_vessels = annotation_array  # già binario

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


def analyze_error_proximity_to_structures(segmentation_array, annotation_array, mask,
                                          airway_mask, pleura_mask, spacing):
    """Analizza errori in base alla vicinanza a vie aeree e pleura."""
    seg_binary = (segmentation_array > 0)
    ann_vessels = annotation_array

    fp_mask = seg_binary & ~ann_vessels & mask
    fn_mask = ~seg_binary & ann_vessels & mask

    if airway_mask.any():
        airway_dist = ndimage.distance_transform_edt(~airway_mask, sampling=spacing)
    else:
        airway_dist = np.full_like(seg_binary, 999.0, dtype=float)

    if pleura_mask is not None and pleura_mask.any():
        pleura_dist = ndimage.distance_transform_edt(~pleura_mask, sampling=spacing)
    else:
        pleura_dist = np.full_like(seg_binary, 999.0, dtype=float)

    stats = {
        'fp_near_airways': {
            'within_5mm': int(np.sum(fp_mask & (airway_dist < 5))),
            'within_10mm': int(np.sum(fp_mask & (airway_dist < 10))),
            'total_fp': int(fp_mask.sum())
        },
        'fp_near_pleura': {
            'within_5mm': int(np.sum(fp_mask & (pleura_dist < 5))),
            'within_10mm': int(np.sum(fp_mask & (pleura_dist < 10))),
            'total_fp': int(fp_mask.sum())
        },
        'fn_near_airways': {
            'within_5mm': int(np.sum(fn_mask & (airway_dist < 5))),
            'within_10mm': int(np.sum(fn_mask & (airway_dist < 10))),
            'total_fn': int(fn_mask.sum())
        },
        'fn_near_pleura': {
            'within_5mm': int(np.sum(fn_mask & (pleura_dist < 5))),
            'within_10mm': int(np.sum(fn_mask & (pleura_dist < 10))),
            'total_fn': int(fn_mask.sum())
        }
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


def generate_error_report(segmentation_path, annotation_array, annotation_type,
                          output_dir, airway_mask_path=None, lung_mask_path=None,
                          exclude_unknown=False, verbose=False, save_json=True):
    """
    Genera un report completo di analisi degli errori.
    - NESSUNA stampa su stdout (debug rimosso)
    - Visualizzazione 2D DISATTIVATA (non viene creata né salvata)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carica segmentazione
    segmentation_img = sitk.ReadImage(segmentation_path)
    segmentation_array = sitk.GetArrayFromImage(segmentation_img)
    spacing = segmentation_img.GetSpacing()[::-1]  # (z, y, x)

    # Prepara annotazioni
    if annotation_type == 'carve':
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
    elif annotation_type == 'vessel12':
        mask = (annotation_array >= 0) if exclude_unknown else np.ones_like(annotation_array, dtype=bool)
        ann_vessels = (annotation_array == 1).astype(np.uint8)
        carve_stats = None
    else:
        raise ValueError("annotation_type must be 'carve' or 'vessel12'")

    # Maschere aggiuntive
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

    # Analisi
    spatial_stats = analyze_error_spatial_distribution(segmentation_array, ann_vessels, mask, spacing)
    size_stats = analyze_error_by_vessel_size(segmentation_array, ann_vessels, mask, spacing)
    proximity_stats = analyze_error_proximity_to_structures(
        segmentation_array, ann_vessels, mask,
        airway_mask if airway_mask is not None else np.zeros_like(mask),
        pleura_mask, spacing
    )

    # Raccomandazioni (senza stamparle)
    fp_airways_pct = (proximity_stats['fp_near_airways']['within_5mm'] /
                      proximity_stats['fp_near_airways']['total_fp'] * 100
                      if proximity_stats['fp_near_airways']['total_fp'] > 0 else 0.0)
    fp_pleura_pct = (proximity_stats['fp_near_pleura']['within_5mm'] /
                     proximity_stats['fp_near_pleura']['total_fp'] * 100
                     if proximity_stats['fp_near_pleura']['total_fp'] > 0 else 0.0)
    fn_pleura_pct = (proximity_stats['fn_near_pleura']['within_5mm'] /
                     proximity_stats['fn_near_pleura']['total_fn'] * 100
                     if proximity_stats['fn_near_pleura']['total_fn'] > 0 else 0.0)

    recommendations = []
    if annotation_type == 'carve' and exclude_unknown and carve_stats and carve_stats["unknown"] > 0:
        pct_unknown = carve_stats["unknown"] / carve_stats["total_vessels"] * 100
        recommendations.append(
            f"Excluding {carve_stats['unknown']:,} voxels (-999, {pct_unknown:.1f}% of vessels). "
            f"Consider exclude_unknown=False."
        )

    if 'large' in size_stats and size_stats['large']['recall'] < 0.7:
        recommendations.append(
            f"Large vessels low recall ({size_stats['large']['recall']:.2f}). "
            f"Try lowering lung_erosion_mm and enabling preserve_large_vessels."
        )
    if 'small' in size_stats and size_stats['small']['recall'] < 0.5:
        recommendations.append(
            f"Small vessels low recall ({size_stats['small']['recall']:.2f}). "
            f"Consider a more sensitive model for small calibers."
        )
    if fp_airways_pct > 30:
        recommendations.append(
            f"{fp_airways_pct:.0f}% of FP within 5mm of airways. Consider increasing airway_dilation_mm."
        )
    if fp_pleura_pct > 30:
        recommendations.append(
            f"{fp_pleura_pct:.0f}% of FP near pleura. Consider increasing lung_erosion_mm for small vessels."
        )
    if fn_pleura_pct > 30:
        recommendations.append(
            f"{fn_pleura_pct:.0f}% of FN near pleura. Consider reducing lung_erosion_mm."
        )

    # Maschere di errore (salvate senza print)
    fp_path, fn_path, overlay_path = create_error_masks(
        segmentation_array, ann_vessels, mask, output_dir, segmentation_img
    )

    # NIENTE visualizzazione 2D: rimosso salvataggio PNG

    report_data = {
        'annotation_breakdown': carve_stats,
        'spatial_distribution': spatial_stats,
        'vessel_size_analysis': size_stats,
        'proximity_analysis': proximity_stats,
        'recommendations': recommendations,
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


segmentation_path = '/content/vesselsegmentation/vessels_cleaned/VESSEL12_22_cleaned.nii.gz'
annotation_path   = '/content/vesselsegmentation/VESSEL12/VESSEL12_22_Annotations.csv'
airway_mask_path  = '/content/vesselsegmentation/vessels_cleaned/airways_full.nii.gz'
lung_mask_path    = '/content/vesselsegmentation/vessels_cleaned/lung_mask_original.nii.gz'
output_dir        = '/content/vesselsegmentation/error_analysis'

if not (os.path.exists(segmentation_path) and os.path.exists(annotation_path)):
    raise FileNotFoundError(f"Missing file(s). "
                            f"Segmentation: {segmentation_path} | Annotation: {annotation_path}")

# Carica immagine di riferimento (serve per costruire la maschera dal CSV)
segmentation_img = sitk.ReadImage(segmentation_path)

# Carica annotazioni in base al formato
ext = os.path.splitext(annotation_path)[1].lower()
if ext in ('.mhd', '.mha', '.nii', '.nii.gz'):
    # Caso CARVE-like (immagini)
    annotation_array, _, _ = load_carve_annotation(annotation_path, verbose=False)
    annotation_type = 'carve'
elif ext == '.csv':
    # Caso VESSEL12 (CSV)
    annotation_array, _ann_stats = load_vessel12_annotation_csv(annotation_path, segmentation_img)
    annotation_type = 'vessel12'
else:
    raise ValueError(f"Formato annotazioni non supportato: {ext}")

# Lancia il report
_ = generate_error_report(
    segmentation_path=segmentation_path,
    annotation_array=annotation_array,
    annotation_type=annotation_type,
    output_dir=output_dir,
    airway_mask_path=airway_mask_path,
    lung_mask_path=lung_mask_path,
    exclude_unknown=False,   # irrilevante per VESSEL12, ma lascia pure
    verbose=False,
    save_json=True
)
