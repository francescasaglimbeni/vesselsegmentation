import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

IMG_PATH  = "scans/VESSEL12_23.mhd"
MASK_PATH = "lungmasks/VESSEL12_23.mhd"
ANN_PATH  = "annotations/VESSEL12_23_Annotations.csv"

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

def load_annotations(path):
    """Carica il CSV delle annotazioni gestendo vari formati possibili."""
    if not os.path.exists(path):
        print(f"File annotazioni non trovato: {path}")
        return None
    
    try:
        # Prova prima con header
        df_with_header = pd.read_csv(path)
        
        df = None
        
        # Caso 1: Una sola colonna con valori come stringhe "x,y,z,label"
        if len(df_with_header.columns) == 1:
            col_name = df_with_header.columns[0]
            
            # Controlla se la prima riga è un header da ignorare
            first_val = str(df_with_header[col_name].iloc[0])
            if not any(c.isdigit() for c in first_val):
                df_with_header = pd.read_csv(path, header=None)
                col_name = 0
            
            # Splitta per virgole
            split_data = df_with_header[col_name].str.split(',', expand=True)
            if split_data.shape[1] >= 4:
                df = pd.DataFrame({
                    'x': pd.to_numeric(split_data[0], errors='coerce'),
                    'y': pd.to_numeric(split_data[1], errors='coerce'),
                    'z': pd.to_numeric(split_data[2], errors='coerce'),
                    'label': pd.to_numeric(split_data[3], errors='coerce')
                })
        
        # Caso 2: Già separato in 4 colonne
        elif len(df_with_header.columns) >= 4:
            cols = df_with_header.columns.tolist()
            df = pd.DataFrame({
                'x': pd.to_numeric(df_with_header[cols[0]], errors='coerce'),
                'y': pd.to_numeric(df_with_header[cols[1]], errors='coerce'),
                'z': pd.to_numeric(df_with_header[cols[2]], errors='coerce'),
                'label': pd.to_numeric(df_with_header[cols[3]], errors='coerce')
            })
        
        if df is None:
            print("Impossibile parsare il CSV")
            return None
        
        # Rimuovi eventuali NaN
        df = df.dropna(subset=['x', 'y', 'z', 'label'])

        # Filtra solo le annotazioni con label == 1 (vasi)
        df_vessels = df[df['label'] == 1].copy()
        
        # Restituisci solo i vasi, ma se non ce ne sono restituisci tutto
        return df_vessels if len(df_vessels) > 0 else df
        
    except Exception as e:
        print(f"Errore nel caricamento delle annotazioni: {e}")
        import traceback
        traceback.print_exc()
        return None

def phys_to_index(sitk_img, xyz_mm_array):
    """xyz (mm) -> voxel indices (x,y,z) via SimpleITK"""
    idx = [sitk_img.TransformPhysicalPointToIndex(tuple(p)) for p in xyz_mm_array]
    return np.array(idx, dtype=int)

def detect_coordinate_type(df, vol_shape):
    """Determina se le coordinate sono voxel o fisiche.
    Secondo la documentazione VESSEL12, sono 0-based voxel coordinates.
    """
    if df is None or len(df) == 0:
        return 'voxel'
    
    Z, H, W = vol_shape
    max_x = df['x'].max()
    max_y = df['y'].max()
    max_z = df['z'].max()
    
    # Se i valori sono tutti entro le dimensioni del volume, sono voxel
    if max_x < W and max_y < H and max_z < Z:
        return 'voxel'
    else:
        return 'physical'

# Carica immagini e mask
img_sitk, vol = read_sitk(IMG_PATH)            
mask_sitk, _mask = read_sitk(MASK_PATH)
mask_sitk = ensure_same_grid(img_sitk, mask_sitk, sitk.sitkNearestNeighbor)
mask = sitk.GetArrayFromImage(mask_sitk)        

if vol.shape != mask.shape:
    raise ValueError(f"CT e maschera hanno forme diverse: {vol.shape} vs {mask.shape}")

seg = vol.copy()
seg[mask <= 0] = OUTSIDE_HU

Z, H, W = seg.shape

# Carica le annotazioni
ann_df = load_annotations(ANN_PATH)
coord_type = detect_coordinate_type(ann_df, vol.shape) if ann_df is not None else None
   

def get_points_for_slice(z_idx):
    """Ottiene i punti annotati per una data slice z.
    
    Note: CSV format is x,y,z (0-based voxel coordinates)
    Array shape is (Z, Y, X) in numpy
    For plotting we need (X, Y) coordinates
    """
    if ann_df is None or len(ann_df) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    
    # Le coordinate nel CSV sono x,y,z (0-based voxel)
    # dove z è l'indice della slice
    mask = (ann_df['z'].astype(int) == z_idx)
    if mask.sum() == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    
    # x e y dal CSV corrispondono direttamente a x,y per il plot
    pts_x = ann_df.loc[mask, 'x'].astype(int).values
    pts_y = ann_df.loc[mask, 'y'].astype(int).values
    
    # Filtra punti dentro i bounds dell'immagine
    # W è la dimensione X, H è la dimensione Y
    valid = (pts_x >= 0) & (pts_x < W) & (pts_y >= 0) & (pts_y < H)
    
    return pts_x[valid], pts_y[valid]

# Setup visualizzazione
z0 = Z // 2
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.15)

img_artist = ax.imshow(normalize_hu(seg[z0]), cmap='gray', origin="lower")
pts_x, pts_y = get_points_for_slice(z0)
scatter_artist = ax.scatter(pts_x, pts_y, c='red', s=POINT_SIZE, marker='o', 
                           edgecolors='yellow', linewidths=0.5) if len(pts_x) else None

n_points = len(pts_x)
ax.set_title(f"VESSEL12 (lung-segmented) – slice z={z0}/{Z-1} – {n_points} vasi annotati")
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
            scatter_artist = ax.scatter(x, y, c='red', s=POINT_SIZE, marker='o',
                                       edgecolors='yellow', linewidths=0.5)
    else:
        if len(x):
            scatter_artist.set_offsets(np.c_[x, y])
        else:
            scatter_artist.set_offsets(np.empty((0, 2)))
    
    n_points = len(x)
    ax.set_title(f"VESSEL12 (lung-segmented) – slice z={z}/{Z-1} – {n_points} vasi annotati")
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