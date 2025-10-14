
import numpy as np
from scipy import ndimage as ndi
from .utils import normalize_vector_field, cylindrical_mask, iso_ball_radius_vox, mm_to_vox

def ridler_calvard_threshold(x):
    """Iterative selection threshold (ISODATA). x: intensities in ROI (float)."""
    x = x.astype(np.float32).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    t = 0.5*(x.min()+x.max())
    for _ in range(25):
        g1 = x[x>t]; g2 = x[x<=t]
        if g1.size==0 or g2.size==0:
            break
        m1 = g1.mean(); m2 = g2.mean()
        t_new = 0.5*(m1+m2)
        if abs(t_new-t) < 1e-3:
            t = t_new; break
        t = t_new
    return float(t)

def region_grow(seed_mask, roi_mask, img, min_vesselness_mask=None):
    """Simple 3D BFS region growing restrained by ROI and optional min vesselness boolean mask."""
    from collections import deque
    visited = np.zeros_like(roi_mask, dtype=bool)
    out = np.zeros_like(roi_mask, dtype=bool)
    Q = deque()
    seeds = np.argwhere(seed_mask & roi_mask)
    for z,y,x in seeds:
        Q.append((int(z),int(y),int(x)))
        visited[z,y,x] = True
    while Q:
        z,y,x = Q.popleft()
        out[z,y,x]=True
        for dz in (-1,0,1):
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dz==dy==dx==0: continue
                    zz,yy,xx = z+dz, y+dy, x+dx
                    if zz<0 or yy<0 or xx<0 or zz>=roi_mask.shape[0] or yy>=roi_mask.shape[1] or xx>=roi_mask.shape[2]:
                        continue
                    if visited[zz,yy,xx]: continue
                    if not roi_mask[zz,yy,xx]: continue
                    if min_vesselness_mask is not None and not min_vesselness_mask[zz,yy,xx]:
                        continue
                    visited[zz,yy,xx]=True
                    Q.append((zz,yy,xx))
    return out

def local_optimal_thresholding(img_hu, centerline_mask, scale_map_mm, e1_field, spacing, min_vesselness_mask, 
                               k_radius=2.5, k_len=3.0, min_vesselness=0.05):
    """
    For each connected segment of centerline voxels, build a cylindrical ROI aligned with e1.
    Radius (mm) = k_radius * sigma_max; half-length (mm) = k_len * sigma_max.
    """
    seg_lbl, num = ndi.label(centerline_mask, structure=np.ones((3,3,3), bool))
    out = np.zeros_like(centerline_mask, dtype=bool)
    for i in range(1, num+1):
        coords = np.argwhere(seg_lbl==i)
        if coords.size==0:
            continue
        # mean axis over the segment
        axes = e1_field[seg_lbl==i]
        axis = axes.mean(axis=0)
        axis = axis / (np.linalg.norm(axis)+1e-9)
        # center as mean of coords
        cz, cy, cx = coords.mean(axis=0)
        # estimate scale as median sigma over the segment
        sigma = float(np.median(scale_map_mm[seg_lbl==i]))
        radius_mm = max(0.7, k_radius * sigma)  # lower bound ~ one slice
        half_len_mm = max(1.4, k_len * sigma)
        radius_vox = float(radius_mm / min(spacing))
        half_len_vox = float(half_len_mm / min(spacing))
        roi = cylindrical_mask(centerline_mask.shape, (cz, cy, cx), axis=(axis[0], axis[1], axis[2]),
                               radius_vox=radius_vox, half_len_vox=half_len_vox)
        # compute local Ridler-Calvard threshold on HU
        local_vals = img_hu[roi]
        t = ridler_calvard_threshold(local_vals)
        if t is None: 
            continue
        # seeds are centerline voxels in this group
        seeds = (seg_lbl==i)
        # Optional vesselness gating
        grown = region_grow(seeds, roi & (img_hu>=t), img_hu, min_vesselness_mask=min_vesselness_mask if min_vesselness>0 else None)
        out |= grown
    return out
