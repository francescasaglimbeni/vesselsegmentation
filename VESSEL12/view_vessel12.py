
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

IMG_PATH  = "dataset/Scans/VESSEL12_23.mhd"
MASK_PATH = "dataset/Lungmasks/VESSEL12_23.mhd"
ANN_PATH  = "dataset/Annotations/VESSEL12_23_Annotations.csv"

HU_MIN, HU_MAX = -1000, 400     
OUTSIDE_HU = -1024               
POINT_SIZE = 12                  

def read_sitk(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img) 
    return img, arr

def ensure_same_grid(fixed_img, moving_img, interp=sitk.sitkNearestNeighbor):
    """Resample moving_img onto fixed_img grid if size/spacing/direction/origin differ."""
    if (fixed_img.GetSize()      == moving_img.GetSize() and
        fixed_img.GetSpacing()   == moving_img.GetSpacing() and
        fixed_img.GetDirection() == moving_img.GetDirection() and
        fixed_img.GetOrigin()    == moving_img.GetOrigin() ):
        return moving_img
    return sitk.Resample(
        moving_img,
        fixed_img,
        sitk.Transform(),
        interp,
        fixed_img.GetOrigin(),
        fixed_img.GetSpacing(),
        fixed_img.GetDirection(),
        0,
        moving_img.GetPixelID()
    )

def normalize_hu(slice2d, vmin=HU_MIN, vmax=HU_MAX):
    s = slice2d.astype(np.float32)
    s = (s - vmin) / float(vmax - vmin)
    return np.clip(s, 0, 1)

def detect_annotation_format(df):
    lc = {c.lower(): c for c in df.columns}
    voxel_sets = [
        ("i", "j", "k"),
        ("x_index","y_index","z_index"),
        ("x_vox","y_vox","z_vox"),
        ("col","row","slice"),  
    ]
    phys_sets = [
        ("x","y","z"),
        ("x_mm","y_mm","z_mm"),
        ("world_x","world_y","world_z"),
        ("coord_x","coord_y","coord_z"),
    ]
    for a,b,c in voxel_sets:
        if a in lc and b in lc and c in lc:
            return "voxel", (lc[a], lc[b], lc[c])
    for a,b,c in phys_sets:
        if a in lc and b in lc and c in lc:
            return "physical", (lc[a], lc[b], lc[c])

    for s in ["slice","k","z_index","z_vox","z"]:
        if s in lc:
            return "slice_only", (lc[s],)
    return None, tuple()

def phys_to_index(sitk_img, xyz_mm_array):
    """xyz (mm) -> voxel indices (x,y,z) via SimpleITK"""
    idx = [sitk_img.TransformPhysicalPointToIndex(tuple(p)) for p in xyz_mm_array]
    return np.array(idx, dtype=int)

img_sitk, vol = read_sitk(IMG_PATH)            
mask_sitk, _mask = read_sitk(MASK_PATH)
mask_sitk = ensure_same_grid(img_sitk, mask_sitk, sitk.sitkNearestNeighbor)
mask = sitk.GetArrayFromImage(mask_sitk)        

if vol.shape != mask.shape:
    raise ValueError(f"CT e maschera hanno forme diverse: {vol.shape} vs {mask.shape}")

seg = vol.copy()
seg[mask <= 0] = OUTSIDE_HU

Z, H, W = seg.shape

ann_df, ann_kind, ann_cols = None, None, ()
if os.path.exists(ANN_PATH):
    try:
        ann_df = pd.read_csv(ANN_PATH)
    except Exception:
        ann_df = pd.read_csv(ANN_PATH, sep=";")
    ann_kind, ann_cols = detect_annotation_format(ann_df)

def get_points_for_slice(z_idx):
    if ann_df is None or ann_kind is None:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if ann_kind == "voxel":
        cx, cy, cz = ann_cols
        xi = ann_df[cx].astype(int).values
        yi = ann_df[cy].astype(int).values
        zi = ann_df[cz].astype(int).values
        m = (zi == z_idx) & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        return xi[m], yi[m]
    elif ann_kind == "physical":
        cx, cy, cz = ann_cols
        pts = ann_df[[cx, cy, cz]].values.astype(float)
        idx_xyz = phys_to_index(img_sitk, pts)  
        xs, ys, zs = idx_xyz[:,0], idx_xyz[:,1], idx_xyz[:,2]
        m = (zs == z_idx) & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        return xs[m], ys[m]
    return np.empty(0, dtype=int), np.empty(0, dtype=int)

z0 = Z // 2
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)

img_artist = ax.imshow(normalize_hu(seg[z0]), origin="lower")
pts_x, pts_y = get_points_for_slice(z0)
scatter_artist = ax.scatter(pts_x, pts_y, s=POINT_SIZE) if len(pts_x) else None

ax.set_title(f"VESSEL12 (lung-segmented) – slice z={z0}/{Z-1}")
ax.axis("off")

ax_z = plt.axes([0.15, 0.05, 0.7, 0.03])
z_slider = Slider(ax=ax_z, label="Slice z", valmin=0, valmax=Z-1, valinit=z0, valstep=1)

def update(val):
    z = int(z_slider.val)
    img_artist.set_data(normalize_hu(seg[z]))
    x, y = get_points_for_slice(z)
    global scatter_artist
    if scatter_artist is None:
        if len(x):
            scatter_artist = ax.scatter(x, y, s=POINT_SIZE)
    else:
        if len(x):
            scatter_artist.set_offsets(np.c_[x, y])
        else:
            scatter_artist.set_offsets(np.empty((0, 2)))
    ax.set_title(f"VESSEL12 (lung-segmented) – slice z={z}/{Z-1}")
    fig.canvas.draw_idle()

z_slider.on_changed(update)

def on_key(event):
    if event.key in ("left", "right"):
        cur = int(z_slider.val)
        dz = -1 if event.key == "left" else 1
        newz = np.clip(cur + dz, 0, Z-1)
        z_slider.set_val(int(newz))  

fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
