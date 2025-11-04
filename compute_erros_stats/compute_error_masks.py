import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy import ndimage
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import json
from math import ceil

# ============================================================
# ============   LOADERS CARVE e VESSEL12   ==================
# ============================================================

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

    stats = {
        "background": int(np.sum(annotation_array == 0)),
        "vein": int(np.sum(annotation_array == 1)),
        "artery": int(np.sum(annotation_array == 2)),
        "unknown": int(np.sum(annotation_array == -999))
    }
    stats["total_vessels"] = stats["vein"] + stats["artery"] + stats["unknown"]
    if verbose:
        print("[CARVE] Stats:", stats)
    return annotation_array, annotation_img, stats


def load_vessel12_annotation_csv(csv_path, reference_img, radius_fallback_mm=1.0, radius_mm=None, verbose=False):
    df = pd.read_csv(csv_path)
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
    # effective radius to use (keep backwards compatibility)
    radius_effective = radius_mm if (radius_mm is not None) else radius_fallback_mm
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

    paint_sphere((k, j, i), radius_effective)
    num_points += 1

    stats = {
        "num_points_total": int(num_points),
        "positive_points": int(num_points),   # VESSEL12 CSV here lists vessel points
        "negative_points": 0,
        "annotated_voxels": int(ann.sum()),
        "positive_voxels": int(ann.sum()),
        "negative_voxels": 0,
        "radius_mm_used": float(radius_effective)
    }
    if verbose:
        print(f"  Using radius {radius_effective} mm -> painted {stats['annotated_voxels']} voxels from {stats['num_points_total']} points")

    return ann.astype(np.uint8), stats

def analyze_error_spatial_distribution(segmentation_array, annotation_array, mask, spacing):
    """Analizza la distribuzione spaziale degli errori (FP e FN) in tre regioni assiali."""
    z_max = segmentation_array.shape[0]
    z_upper = z_max // 3
    z_lower = 2 * z_max // 3

    seg_binary = (segmentation_array > 0).astype(np.uint8)
    ann_vessels = annotation_array.astype(np.uint8)  # già binario

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
            'avg_diameter_mm': float(vessel_diameters[cat_mask].mean()) if np.any(cat_mask) else 0.0
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

    seg_binary = (segmentation_array > 0).astype(bool)
    ann_vessels = annotation_array.astype(bool)
    mask = mask.astype(bool)

    tp_mask = (seg_binary & ann_vessels & mask).astype(np.uint8)
    fp_mask = (seg_binary & (~ann_vessels) & mask).astype(np.uint8)
    fn_mask = ((~seg_binary) & ann_vessels & mask).astype(np.uint8)

    overlay = np.zeros_like(seg_binary, dtype=np.uint8)
    overlay[tp_mask > 0] = 1
    overlay[fp_mask > 0] = 2
    overlay[fn_mask > 0] = 3

    def _save(arr, path):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.CopyInformation(reference_img)
        sitk.WriteImage(img, path)

    fp_path = os.path.join(output_dir, "error_false_positives.nii.gz")
    fn_path = os.path.join(output_dir, "error_false_negatives.nii.gz")
    overlay_path = os.path.join(output_dir, "error_overlay.nii.gz")
    _save(fp_mask, fp_path)
    _save(fn_mask, fn_path)
    _save(overlay, overlay_path)

    return fp_path, fn_path, overlay_path


def show_overlay_grid(overlay, base=None, n_slices=12, title="Error overlay (0=TN, 1=TP, 2=FP, 3=FN)"):
    """
    Mostra un pannello di slice equispaziate con overlay TP/FP/FN.
    Se base è passato (es. intensità CT normalizzate), la usa come sfondo in grigio.
    """
    zdim = overlay.shape[0]
    if zdim == 0:
        return
    idx = np.linspace(0, zdim-1, num=n_slices, dtype=int)

    # Colormap: 0=trasparente, 1/2/3 con colori distinti
    colors = np.array([
        [0,0,0,0.0],    # 0 TN: trasparente
        [0,1,0,0.7],    # 1 TP: verde
        [1,0,0,0.7],    # 2 FP: rosso
        [1,1,0,0.7],    # 3 FN: giallo
    ])
    cmap = ListedColormap(colors)

    ncols = 4
    nrows = int(np.ceil(len(idx)/ncols))
    plt.figure(figsize=(3.5*ncols, 3.5*nrows))
    for p, z in enumerate(idx):
        plt.subplot(nrows, ncols, p+1)
        if base is not None:
            plt.imshow(base[z], cmap='gray', interpolation='none')
        plt.imshow(overlay[z], cmap=cmap, interpolation='none')
        plt.axis('off')
        plt.title(f"z={z}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def generate_error_report(segmentation_path,
                          annotation_array,
                          annotation_type,
                          output_dir,
                          airway_mask_path=None,
                          lung_mask_path=None,
                          exclude_unknown=False,
                          verbose=False,
                          save_json=True,
                          show_plots=True,
                          n_slices=12):
    """
    annotation_type:
      - 'carve'    -> annotation_array = etichette CARVE (0/1/2/-999)
      - 'vessel12' -> annotation_array = {-1=non annotato, 0=non vaso, 1=vaso}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Segmentazione (referenza di salvataggio)
    segmentation_img = sitk.ReadImage(segmentation_path)
    segmentation_array = sitk.GetArrayFromImage(segmentation_img)
    spacing = segmentation_img.GetSpacing()[::-1]  # (z, y, x)

    # Prepara annotazioni in binario e mask "dove valutare"
    if annotation_type == 'carve':
        if exclude_unknown:
            mask = (annotation_array != -999)
        else:
            mask = np.ones_like(annotation_array, dtype=bool)
        ann_vessels = ((annotation_array == 1) |
                       (annotation_array == 2) |
                       (annotation_array == -999)).astype(np.uint8)

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
        # ann_array: -1/0/1
        mask = (annotation_array != -1)
        ann_vessels = (annotation_array == 1).astype(np.uint8)
        carve_stats = None
    else:
        raise ValueError("annotation_type must be 'carve' or 'vessel12'")

    # Maschere extra (opzionali)
    if airway_mask_path and os.path.exists(airway_mask_path):
        airway_mask = sitk.GetArrayFromImage(sitk.ReadImage(airway_mask_path)).astype(bool)
    else:
        airway_mask = np.zeros_like(mask, dtype=bool)

    if lung_mask_path and os.path.exists(lung_mask_path):
        lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_path)).astype(bool)
        lung_eroded = ndimage.binary_erosion(lung_mask, iterations=2)
        pleura_mask = lung_mask & (~lung_eroded)
    else:
        pleura_mask = None

    # Statistiche per regione/diametro
    spatial_stats = analyze_error_spatial_distribution(segmentation_array, ann_vessels, mask, spacing)
    size_stats = analyze_error_by_vessel_size(segmentation_array, ann_vessels, mask, spacing)

    # Maschere di errore (salvataggio)
    fp_path, fn_path, overlay_path = create_error_masks(
        segmentation_array, ann_vessels, mask, output_dir, segmentation_img
    )

    # Visualizzazione a schermo
    if show_plots:
        # Overlay per display
        seg_bin = (segmentation_array > 0).astype(np.uint8)
        overlay_disp = np.zeros_like(seg_bin, dtype=np.uint8)  # 0 TN, 1 TP, 2 FP, 3 FN
        overlay_disp[(seg_bin == 1) & (ann_vessels == 1) & mask] = 1
        overlay_disp[(seg_bin == 1) & (ann_vessels == 0) & mask] = 2
        overlay_disp[(seg_bin == 0) & (ann_vessels == 1) & mask] = 3

        # base: normalizzo intensità se disponibile (non sempre abbiamo la CT; usiamo la segmentazione come sfondo neutro)
        base = None
        try:
            # Se esiste una CT allineata come "background.nii.gz" nella stessa cartella, usiamola (opzionale)
            base_candidate = os.path.join(os.path.dirname(segmentation_path), "background.nii.gz")
            if os.path.exists(base_candidate):
                base_img = sitk.GetImageFromArray(sitk.GetArrayFromImage(sitk.ReadImage(base_candidate)))
                base_arr = sitk.GetArrayFromImage(base_img).astype(np.float32)
                vmin, vmax = np.percentile(base_arr, [1, 99])
                base = np.clip((base_arr - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        except Exception:
            base = None

        show_overlay_grid(overlay_disp, base=base, n_slices=n_slices,
                          title="TP (verde)  FP (rosso)  FN (giallo)")

    # Report JSON
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


# ============================================================
# ========================  ESEMPIO  =========================
# ============================================================
if __name__ == "__main__":
    # ========= ESEMPIO CARVE =========
    # segmentation_path = '/content/vesselsegmentation/vessels_cleaned/1.2.840.113704.1.111.2604.1126357612.7_cleaned.nii.gz'
    # annotation_path = '/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.2604.1126357612_fullAnnotations.mhd'
    # airway_mask_path = '/content/vesselsegmentation/vessels_cleaned/airways_full.nii.gz'
    # lung_mask_path = '/content/vesselsegmentation/vessels_cleaned/lung_mask_original.nii.gz'
    # output_dir = '/content/vesselsegmentation/error_analysis'
    # if os.path.exists(segmentation_path) and os.path.exists(annotation_path):
    #     annotation_array, _, _ = load_carve_annotation(annotation_path, verbose=False)
    #     _ = generate_error_report(
    #         segmentation_path=segmentation_path,
    #         annotation_array=annotation_array,
    #         annotation_type='carve',
    #         output_dir=output_dir,
    #         airway_mask_path=airway_mask_path,
    #         lung_mask_path=lung_mask_path,
    #         exclude_unknown=False,  # include -999 come vessels
    #         verbose=False,
    #         save_json=True,
    #         show_plots=True,        # << mostra a schermo
    #         n_slices=12
    #     )
    # else:
    #     raise FileNotFoundError("File CARVE mancanti.")

    # ========= ESEMPIO VESSEL12 =========
    segmentation_path = '/content/vesselsegmentation/vessels_cleaned/VESSEL12_23_cleaned.nii.gz'
    csv_path         = '/content/vesselsegmentation/VESSEL12/VESSEL12_23_Annotations.csv'
    output_dir       = '/content/vesselsegmentation/error_analysis_VESSEL12'
    radius_mm        = 1.0

    if os.path.exists(segmentation_path) and os.path.exists(csv_path):
        # NOTA: per essere sicuri dell'allineamento, uso la segmentazione come "reference_img"
        ref_img = sitk.ReadImage(segmentation_path)
        ann_labels, stats = load_vessel12_annotation_csv(csv_path, ref_img, radius_mm=radius_mm, verbose=True)
        print(f"  -> Punti: tot={stats['num_points_total']}, += {stats['positive_points']}, -= {stats['negative_points']}")
        print(f"  -> Voxel annotati: {stats['annotated_voxels']:,} (pos={stats['positive_voxels']:,}, neg={stats['negative_voxels']:,})")

        _ = generate_error_report(
            segmentation_path=segmentation_path,
            annotation_array=ann_labels,       # -1/0/1
            annotation_type='vessel12',
            output_dir=output_dir,
            airway_mask_path=None,
            lung_mask_path=None,
            exclude_unknown=False,             # non usato per VESSEL12
            verbose=False,
            save_json=True,
            show_plots=True,                   # << mostra a schermo
            n_slices=12
        )
    else:
        raise FileNotFoundError("File VESSEL12 mancanti.")
