
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, ball
from .utils import spherical_structuring_element, iso_ball_radius_vox

def segment_lungs(img_hu):
    # Simple threshold and largest components heuristic (research-grade; replace with robust method if needed)
    mask = img_hu < -320  # lung parenchyma threshold
    mask = ndi.binary_opening(mask, structure=ball(2))
    # keep 2 largest connected components
    lbl, num = ndi.label(mask)
    if num == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = np.argsort(sizes)[-2:]
    out = np.isin(lbl, keep)
    out = ndi.binary_closing(out, structure=ball(2))
    out = ndi.binary_fill_holes(out)
    return out

def segment_airway_lumen(img_hu, spacing):
    # Air is ~ -1000 HU. Take connected component to the superior trachea region.
    air = img_hu < -950
    # Seed: top-most slice center
    z0 = 0
    cy = img_hu.shape[1]//2
    cx = img_hu.shape[2]//2
    seed_mask = np.zeros_like(air, dtype=bool)
    seed_mask[z0, max(0,cy-5):cy+6, max(0,cx-5):cx+6] = True
    seeds = air & seed_mask
    if not seeds.any():
        # fallback: any air on top slice
        seeds = air[z0]
        filled = np.zeros_like(air, dtype=bool)
        filled[z0] = seeds
    else:
        filled = np.zeros_like(air, dtype=bool)
        # region growing on air to capture conducting airways
        from collections import deque
        Q = deque(map(tuple, np.argwhere(seeds)))
        visited = np.zeros_like(air, dtype=bool)
        while Q:
            z,y,x = Q.popleft()
            if visited[z,y,x]: 
                continue
            visited[z,y,x]=True
            if not air[z,y,x]: 
                continue
            filled[z,y,x]=True
            for dz in (-1,0,1):
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if dz==dy==dx==0: continue
                        zz,yy,xx = z+dz, y+dy, x+dx
                        if 0<=zz<air.shape[0] and 0<=yy<air.shape[1] and 0<=xx<air.shape[2]:
                            if not visited[zz,yy,xx]:
                                Q.append((zz,yy,xx))
    return filled

def airway_wall_exclusion(airway_lumen_mask, spacing, dilate_vox=3):
    # Paper approximates ~2 mm wall thickness by dilating airway lumen with spherical SE radius of 3 voxels.
    se = spherical_structuring_element(dilate_vox)
    dil = ndi.binary_dilation(airway_lumen_mask, structure=se)
    return dil
