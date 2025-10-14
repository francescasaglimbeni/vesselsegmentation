import os
import sys
import time
import numpy as np

# --- TS (TotalSegmentator) ---
from code_vesselsegmentation_TS.preprocessing import convert_mhd_to_nifti
from totalsegmentator.python_api import totalsegmentator
from code_vesselsegmentation_TS.postprocessing import process_vessel_segmentation

# --- VS (VesselsFilter) ---
from code_vesselsegmentation_VF.io_mhd import read_mhd, write_nii
from code_vesselsegmentation_VF.vesselness import frangi_vesselness
from code_vesselsegmentation_VF.multiscale_threshold import compute_adaptive_thresholds
from code_vesselsegmentation_VF.centerline import extract_centerline
from code_vesselsegmentation_VF.local_thresholding import local_optimal_thresholding
from code_vesselsegmentation_VF.lung_airway import (
    segment_lungs, segment_airway_lumen, airway_wall_exclusion
)

# =========================
#   CONFIGURAZIONE
# =========================
MODE = "VS"   # "TS" oppure "VS"

# -- Path comuni
MHD_PATH = "/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.208.1137518216.7.mhd"

# -- TS config
NIFTI_DIR = "nifti_scans"
SEG_DIR = "vessels_segmentations"
TS_OUTPUT_DIR = "vessels_cleaned"
TS_FAST = False

# -- VS config
VS_OUT_MASK = "vessels_mask.nii.gz"
VS_NUM_SCALES = 7
VS_SIGMA_MAX_MM = 4.5
VS_TMIN = 0.07
VS_TMAX = 0.17
VS_MIN_VESSELNESS = 0.05
VS_AIRWAY_DILATE_VOX = 3


# ===================================
#   Pipeline TS (TotalSegmentator)
# ===================================
def run_ts_pipeline(mhd_path: str,
                    nifti_dir: str,
                    seg_dir: str,
                    output_dir: str,
                    fast: bool = False) -> None:

    # Step 1: Convert to NIfTI
    print("=== Step 1: Converting MHD to NIfTI ===")
    nifti_path = convert_mhd_to_nifti(mhd_path, nifti_dir)

    # Step 2a: Lung vessels
    print("\n=== Step 2a: Segmenting Lung Vessels (TotalSegmentator) ===")
    totalsegmentator(
        nifti_path,
        seg_dir,
        task='lung_vessels',
        fast=fast
    )
    print("✓ Lung vessels segmentation complete!")

    # Step 2b: Total anatomy
    print("\n=== Step 2b: Segmenting Total Anatomy (TotalSegmentator) ===")
    totalsegmentator(
        nifti_path,
        seg_dir,
        task='total',
        fast=fast
    )
    print("✓ Total anatomy segmentation complete!")

    # Step 3: Verify critical structures
    print("\n=== Step 3: Verifying Output Files ===")
    all_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])

    critical_files = {
        'lung_vessels.nii.gz': 'Intrapulmonary vessels',
        'pulmonary_vein.nii.gz': 'Pulmonary vein (vein seed - alternative)',
        'heart.nii.gz': 'Heart (can extract left atrium)',
        'aorta.nii.gz': 'Aorta (artery reference)'
    }

    print("\nLooking for critical structures:")
    present, missing = [], []
    for filename, description in critical_files.items():
        filepath = os.path.join(seg_dir, filename)
        if os.path.exists(filepath):
            present.append(f"✓ {filename} - {description}")
        else:
            missing.append(f"✗ {filename} - {description}")

    for item in present:
        print(f"  {item}")
    if missing:
        print("\nMissing (may need workarounds):")
        for item in missing:
            print(f"  {item}")

    print("\nSearching for alternative structures:")
    search_terms = {
        'pulmonary': 'Pulmonary vessels',
        'atrium': 'Atrium structures',
        'ventricle': 'Ventricle structures'
    }
    for term, description in search_terms.items():
        matches = [f for f in all_files if term in f.lower()]
        if matches:
            print(f"  Found {description}:")
            for m in matches:
                print(f"    - {m}")

    # Step 4: Post-process
    print("\n=== Step 4: Post-processing ===")
    lung_vessels_path = os.path.join(seg_dir, 'lung_vessels.nii.gz')
    if not os.path.exists(lung_vessels_path):
        print("ERROR: lung_vessels.nii.gz not found!")
        print("The task='lung_vessels' may have failed.")
        print("\nTrying to find alternative vessel files...")
        vessel_candidates = [f for f in all_files if 'vessel' in f.lower() or 'pulmonary' in f.lower()]
        if vessel_candidates:
            print(f"Found potential alternatives: {vessel_candidates}")
            print("You may need to modify postprocessing.py to use these files.")
        return

    cleaned_path, centerlines_path = process_vessel_segmentation(
        seg_dir,
        output_dir,
        nifti_path,
        extract_skeleton=True
    )

    if cleaned_path is None:
        print("\nERROR: Post-processing failed!")
        return


# ===================================
#   Utility scale
# ===================================
def build_scales(spacing, num_scales=7, sigma_min_mm=None, sigma_max_mm=4.5):
    if sigma_min_mm is None:
        sigma_min_mm = min(spacing)  # mm del voxel più piccolo
    return np.geom_space(sigma_min_mm, sigma_max_mm, num_scales)


# ===================================
#   Pipeline VS (vesselsFilter)
# ===================================
def run_vs_pipeline(mhd_path: str,
                    out_mask_path: str,
                    num_scales: int = 7,
                    sigma_max_mm: float = 4.5,
                    tmin: float = 0.07,
                    tmax: float = 0.17,
                    min_vesselness: float = 0.05,
                    airway_dilate_vox: int = 3) -> None:

    # Carica MHD
    img_hu, spacing, origin, direction = read_mhd(mhd_path)
    print(f"Loaded scan shape={img_hu.shape}, spacing={spacing} mm")

    # Maschera polmoni e rimozione parete bronchiale
    lung_mask = segment_lungs(img_hu)
    airway_lumen = segment_airway_lumen(img_hu, spacing)
    airway_wall = airway_wall_exclusion(airway_lumen, spacing, dilate_vox=airway_dilate_vox)
    roi_mask = lung_mask & (~airway_wall)

    # Vesselness multi-scala
    sigmas = build_scales(spacing, num_scales=num_scales, sigma_max_mm=sigma_max_mm)
    V, S, E1 = frangi_vesselness(img_hu, spacing, sigmas, alpha=0.5, beta=0.5, c=70.0)
    V *= roi_mask.astype(np.float32)

    # Soglia adattiva dalla mappa delle scale
    thr_map = compute_adaptive_thresholds(
        S, sigma_min_mm=sigmas[0], sigma_max_mm=sigmas[-1], tmin=tmin, tmax=tmax
    )
    initial = (V >= thr_map) & roi_mask

    # Centerline
    centerline = extract_centerline(initial)

    # Region growing con soglia locale (Ridler-Calvard) entro ROI cilindriche + vincolo di vesselness minima
    minVmask = V >= float(min_vesselness)
    vessels = local_optimal_thresholding(
        img_hu, centerline, S, E1, spacing,
        min_vesselness_mask=minVmask,
        k_radius=2.5, k_len=3.0, min_vesselness=min_vesselness
    )

    # Constrain
    vessels &= roi_mask

    # Salva NIfTI
    write_nii(vessels, spacing, origin, direction, out_mask_path)
    print(f"✓ Saved vessels mask: {out_mask_path}")


def main():
    t0 = time.time()
    if MODE.upper() == "TS":
        run_ts_pipeline(
            mhd_path=MHD_PATH,
            nifti_dir=NIFTI_DIR,
            seg_dir=SEG_DIR,
            output_dir=TS_OUTPUT_DIR,
            fast=TS_FAST
        )
    elif MODE.upper() == "VS":
        run_vs_pipeline(
            mhd_path=MHD_PATH,
            out_mask_path=VS_OUT_MASK,
            num_scales=VS_NUM_SCALES,
            sigma_max_mm=VS_SIGMA_MAX_MM,
            tmin=VS_TMIN,
            tmax=VS_TMAX,
            min_vesselness=VS_MIN_VESSELNESS,
            airway_dilate_vox=VS_AIRWAY_DILATE_VOX
        )
    else:
        raise SystemExit("Valore di MODE non valido. Usa 'TS' oppure 'VS'.")
    dt = time.time() - t0
    print(f"\nDone. Elapsed: {dt:.1f}s")


if __name__ == "__main__":
    main()
