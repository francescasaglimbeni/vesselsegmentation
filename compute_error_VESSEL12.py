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