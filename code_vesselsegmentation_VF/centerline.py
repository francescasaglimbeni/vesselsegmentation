
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy import ndimage as ndi

def extract_centerline(binary_mask):
    skel = skeletonize_3d(binary_mask.astype(bool))
    return skel

def classify_centerline_voxels(skel):
    # Count 26-neighborhood connectivity
    footprint = np.ones((3,3,3), dtype=bool)
    neighbors = ndi.convolve(skel.astype(np.uint8), footprint, mode='constant', cval=0)
    # remove self
    neighbors = neighbors - (skel.astype(np.uint8))
    coords = np.argwhere(skel)
    types = {}
    for z,y,x in coords:
        n = int(neighbors[z,y,x])
        if n <= 0:
            types[(z,y,x)] = 'discard'
        elif n == 1:
            types[(z,y,x)] = 'endpoint'
        elif n == 2:
            types[(z,y,x)] = 'segment'
        else:
            types[(z,y,x)] = 'bifurcation'
    return types

def groups_from_segments(skel):
    # Label connected components restricted to 'segment' voxels
    lbl, num = ndi.label(skel, structure=np.ones((3,3,3), bool))
    groups = []
    for i in range(1, num+1):
        coords = np.argwhere(lbl==i)
        if coords.size:
            groups.append(coords)
    return groups
