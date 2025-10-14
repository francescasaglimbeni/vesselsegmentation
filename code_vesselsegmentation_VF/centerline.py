import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def extract_centerline(binary_mask):
    """
    Estrae la scheletrizzazione 3D (centerline) da una maschera binaria
    usando il filtro ITK BinaryThinning3DImageFilter.

    Args:
        binary_mask (ndarray): volume 3D booleano o binario (z, y, x)

    Returns:
        ndarray: maschera binaria del centerline (stesso shape dell'input)
    """
    # Converte in immagine ITK
    itk_img = sitk.GetImageFromArray(binary_mask.astype(np.uint8))
    # Applica il thinning 3D
    thin_img = sitk.BinaryThinning(itk_img)
    # Torna a numpy
    skeleton = sitk.GetArrayFromImage(thin_img)
    return skeleton.astype(bool)


def classify_centerline_voxels(skel):
    """
    Classifica i voxel della centerline in base alla connettività:
    - 'endpoint'     → 1 vicino
    - 'segment'      → 2 vicini
    - 'bifurcation'  → ≥3 vicini
    - 'discard'      → 0 vicini (isolato)
    """
    footprint = np.ones((3, 3, 3), dtype=bool)
    neighbors = ndi.convolve(skel.astype(np.uint8), footprint, mode='constant', cval=0)
    # rimuove il voxel stesso
    neighbors = neighbors - skel.astype(np.uint8)

    coords = np.argwhere(skel)
    types = {}
    for z, y, x in coords:
        n = int(neighbors[z, y, x])
        if n <= 0:
            types[(z, y, x)] = 'discard'
        elif n == 1:
            types[(z, y, x)] = 'endpoint'
        elif n == 2:
            types[(z, y, x)] = 'segment'
        else:
            types[(z, y, x)] = 'bifurcation'
    return types


def groups_from_segments(skel):
    """
    Raggruppa i voxel della centerline in componenti connesse
    (solo quelli etichettati come 'segment').
    """
    lbl, num = ndi.label(skel, structure=np.ones((3, 3, 3), bool))
    groups = []
    for i in range(1, num + 1):
        coords = np.argwhere(lbl == i)
        if coords.size:
            groups.append(coords)
    return groups
