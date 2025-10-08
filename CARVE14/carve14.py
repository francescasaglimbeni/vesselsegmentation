import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
import os
from pathlib import Path

# ========= CONFIG =========
# Percorsi
LABEL_MHD = "full_annotations/1.2.840.113704.1.111.208.1137518216_results.mhd"
CT_MHD    = None  # es. r"CARVE14/CT_volume.mhd" per overlay CT

# Nomi etichette (aggiorna se i codici sono diversi)
LABEL_NAMES = {
    2: "Artery",
    1: "Vein"
}

# Colori fissi (RGBA)
COLOR_ARTERY = (1.0, 0.1, 0.1, 1.0)  # rosso
COLOR_VEIN   = (0.1, 0.9, 1.0, 1.0)  # ciano
# ==========================


def read_image(path):
    if path is None:
        return None, None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    return img, arr


# === Lettura ===
lbl_img, lbl = read_image(LABEL_MHD)
if lbl_img is None:
    raise RuntimeError("Label map non trovata.")
z, y, x = lbl.shape

# Imposta i valori negativi (es. -999) come background
lbl[lbl < 0] = 0

ct_img, ct = read_image(CT_MHD)

# Etichette presenti
labels_present = [int(v) for v in np.unique(lbl) if v != 0]
print(f"[INFO] Etichette trovate (dopo filtraggio): {labels_present}")


def build_cmap(label_ids, alpha=0.8):
    """Costruisce la colormap (arterie rosse, vene ciano)."""
    color_map = {0: (0, 0, 0, 0)}
    for lid in label_ids:
        name = LABEL_NAMES.get(lid, "").lower()
        if "artery" in name or "arteria" in name:
            color_map[lid] = COLOR_ARTERY
        elif "vein" in name or "vena" in name:
            color_map[lid] = COLOR_VEIN

    lut_labels = [0] + label_ids
    lut = [color_map[0]] + [color_map[lid] for lid in label_ids]
    lut = [(r, g, b, min(max(a * alpha, 0), 1)) for (r, g, b, a) in lut]
    return ListedColormap(lut), lut_labels


def reindex_labels(slice2d, lut_labels):
    mapping = {lab: idx for idx, lab in enumerate(lut_labels)}
    out = np.zeros_like(slice2d, dtype=np.int32)
    for lab in np.unique(slice2d):
        out[slice2d == lab] = mapping.get(int(lab), 0)
    return out


def apply_window(img2d, center, width):
    vmin = center - width / 2.0
    vmax = center + width / 2.0
    imgw = np.clip(img2d, vmin, vmax)
    return imgw, vmin, vmax


# === FIGURA SOLO ASSIALE ===
iz0 = z // 2
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.subplots_adjust(left=0.15, right=0.98, bottom=0.20, top=0.90)
fig.suptitle("Arterie vs Vene — Vista Assiale", fontsize=14)

# Slider slice e opacità
ax_z = plt.axes([0.15, 0.10, 0.80, 0.03])
sz = Slider(ax_z, "Z", 0, z - 1, valinit=iz0, valfmt="%0.0f")

ax_alpha = plt.axes([0.15, 0.06, 0.35, 0.03])
salpha = Slider(ax_alpha, "Opacity", 0.1, 1.0, valinit=0.8, valfmt="%.2f")

# Slider finestra CT
if ct is not None:
    ax_wc = plt.axes([0.55, 0.06, 0.18, 0.03])
    ax_ww = plt.axes([0.78, 0.06, 0.18, 0.03])
    swc = Slider(ax_wc, "CT C", -700, 200, valinit=-600, valfmt="%0.0f")
    sww = Slider(ax_ww, "CT W", 500, 2000, valinit=1500, valfmt="%0.0f")
else:
    swc = sww = None


def draw(_=None):
    ax.clear()
    iz = int(sz.val)
    alpha = float(salpha.val)
    cmap, lut_labels = build_cmap(labels_present, alpha)

    # CT base
    if ct is not None:
        c2d = ct[iz].astype(np.float32)
        c2w, vmin, vmax = apply_window(c2d, float(swc.val), float(sww.val))
        ax.imshow(c2w, cmap="gray", vmin=vmin, vmax=vmax)
    else:
        ax.imshow(np.zeros((y, x)), cmap="gray", vmin=0, vmax=1)

    # Overlay label
    L2D = lbl[iz]
    Lc = reindex_labels(L2D, lut_labels)
    ax.imshow(Lc, cmap=cmap, interpolation="nearest")
    ax.set_title(f"Slice Z={iz}", fontsize=12)
    ax.axis("off")

    # Legend
    import matplotlib.patches as mpatches
    patches = []
    for lab in [lab for lab in lut_labels if lab != 0]:
        name = LABEL_NAMES.get(lab, f"Label {lab}")
        patches.append(mpatches.Patch(color=cmap(lut_labels.index(lab)), label=name))
    if patches:
        ax.legend(handles=patches, loc="lower right", frameon=True)

    fig.canvas.draw_idle()


# Collego slider
sz.on_changed(draw)
salpha.on_changed(draw)
if swc:
    swc.on_changed(draw)
if sww:
    sww.on_changed(draw)

draw()
plt.show()


# === (Opzionale) — se vuoi salvare i volumi puliti / maschere, decommenta da qui ===
# out_dir = Path("CARVE14")
# out_dir.mkdir(parents=True, exist_ok=True)
#
# # Salva il volume pulito (senza -999)
# out_full = out_dir / "fullAnnotations_converted.nii.gz"
# sitk.WriteImage(lbl_img, str(out_full))
# print(f"[INFO] Volume salvato in: {out_full}")
#
# # Maschere binarie
# mask_artery = (lbl == 1).astype(np.uint8)
# mask_vein   = (lbl == 2).astype(np.uint8)
# mask_all    = (lbl > 0).astype(np.uint8)
#
# for m, name in [(mask_artery, "mask_artery.nii.gz"),
#                 (mask_vein, "mask_vein.nii.gz"),
#                 (mask_all, "mask_vessels_all.nii.gz")]:
#     m_img = sitk.GetImageFromArray(m)
#     m_img.CopyInformation(lbl_img)
#     sitk.WriteImage(m_img, str(out_dir / name))
#     print(f"[INFO] Salvata: {name}")
# === Fine parte opzionale ===
