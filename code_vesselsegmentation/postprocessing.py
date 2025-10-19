import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from code_vesselsegmentation.preprocessing import create_spherical_kernel, extract_centerlines


def find_seed_regions(seg_dir):
    seed_candidates = {
        'artery': ['pulmonary_artery.nii.gz', 'aorta.nii.gz'],
        'vein': ['heart_atrium_left.nii.gz', 'pulmonary_vein.nii.gz', 'heart.nii.gz'],
    }

    found_seeds = {'artery': None, 'vein': None}
    for vessel_type, candidates in seed_candidates.items():
        for candidate in candidates:
            path = os.path.join(seg_dir, candidate)
            if os.path.exists(path):
                found_seeds[vessel_type] = path
                break
    return found_seeds['artery'], found_seeds['vein']


def _gather_airway_mask(seg_dir, reference_img):
    airway_mask = None
    found = []
    for fname in os.listdir(seg_dir):
        if not fname.endswith('.nii.gz'):
            continue
        lower = fname.lower()
        if ('bronchus' in lower) or ('trachea' in lower):
            path = os.path.join(seg_dir, fname)
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img).astype(bool)
            if airway_mask is None:
                airway_mask = arr
            else:
                airway_mask |= arr
            found.append(fname)

    if airway_mask is None:
        # No explicit airways found; return an all-false mask with same shape
        airway_mask = np.zeros(sitk.GetArrayFromImage(reference_img).shape, dtype=bool)

    return airway_mask, found


def adaptive_vessel_cleaning(vessel_mask, lung_mask, airway_mask, spacing, 
                             lung_erosion_mm=2.0, airway_dilation_mm=3.0,
                             preserve_large_vessels=True, large_vessel_threshold_mm=5.0):
    stats = {}
    
    # 1. Erodi polmoni (più conservativo: 2mm invece di 3mm)
    erode_kernel = create_spherical_kernel(lung_erosion_mm, spacing)
    lung_mask_eroded = ndimage.binary_erosion(lung_mask, structure=erode_kernel)
    stats['lung_erosion_mm'] = lung_erosion_mm
    
    # 2. Dilata vie aeree per esclusione
    if airway_mask.any():
        airway_kernel = create_spherical_kernel(airway_dilation_mm, spacing)
        airway_mask_dilated = ndimage.binary_dilation(airway_mask, structure=airway_kernel)
    else:
        airway_mask_dilated = airway_mask
    stats['airway_dilation_mm'] = airway_dilation_mm
    
    # 3. Se richiesto, identifica e preserva vasi grandi
    if preserve_large_vessels:
        print(f"  🔍 Identificazione vasi grandi (diametro > {large_vessel_threshold_mm}mm)...")
        
        # Identifica componenti connesse nei vasi originali
        labeled_vessels, num_vessels = ndimage.label(vessel_mask)
        
        # Calcola volume di ogni componente
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        vessel_volumes = ndimage.sum(vessel_mask, labeled_vessels, range(1, num_vessels + 1))
        vessel_volumes_mm3 = np.array(vessel_volumes) * voxel_volume_mm3
        
        # Calcola diametro equivalente (assumendo sfera)
        vessel_diameters = 2.0 * ((3.0 * vessel_volumes_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0))
        
        # Identifica vasi grandi
        large_vessel_labels = np.where(vessel_diameters >= large_vessel_threshold_mm)[0] + 1
        large_vessel_mask = np.isin(labeled_vessels, large_vessel_labels)
        
        stats['num_large_vessels'] = len(large_vessel_labels)
        stats['num_total_vessels'] = num_vessels
        
        print(f"    Trovati {len(large_vessel_labels)} vasi grandi su {num_vessels} totali")
        
        # Per vasi grandi: usa erosione minima o nessuna erosione
        # Erosione molto conservativa solo per rimuovere rumore pleura
        minimal_erosion_kernel = create_spherical_kernel(0.5, spacing)  # Solo 0.5mm
        lung_mask_minimal = ndimage.binary_erosion(lung_mask, structure=minimal_erosion_kernel)
        
        # Pulisci vasi grandi con criteri più permissivi
        large_vessels_clean = large_vessel_mask & lung_mask_minimal & ~airway_mask_dilated
        
        # Pulisci vasi piccoli con criteri standard
        small_vessel_mask = vessel_mask & ~large_vessel_mask
        small_vessels_clean = small_vessel_mask & lung_mask_eroded & ~airway_mask_dilated
        
        # Combina
        vessel_clean = large_vessels_clean | small_vessels_clean
        
        stats['large_vessels_voxels'] = int(large_vessels_clean.sum())
        stats['small_vessels_voxels'] = int(small_vessels_clean.sum())
        
    else:
        # Pulizia standard senza preservazione speciale
        vessel_clean = vessel_mask & lung_mask_eroded & ~airway_mask_dilated
        stats['num_large_vessels'] = 0
        stats['num_total_vessels'] = 0
    
    stats['total_clean_voxels'] = int(vessel_clean.sum())
    stats['original_voxels'] = int(vessel_mask.sum())
    stats['removed_voxels'] = int(vessel_mask.sum() - vessel_clean.sum())
    
    return vessel_clean, stats


def process_vessel_segmentation(seg_dir, output_dir, original_image_path, 
                                min_vessel_voxels=None, min_vessel_diameter_mm=None, 
                                extract_skeleton=True,
                                lung_erosion_mm=2.0,
                                airway_dilation_mm=3.0,
                                preserve_large_vessels=True,
                                large_vessel_threshold_mm=5.0):
    # Read original to derive spacing order (we will use z,y,x order for kernels)
    original_img = sitk.ReadImage(original_image_path)
    spacing = original_img.GetSpacing()[::-1]  # (z, y, x)

    os.makedirs(output_dir, exist_ok=True)

    # --- Locate vessel mask from TotalSegmentator outputs ---
    vessel_path = None
    vessel_candidates = [
        "lung_vessels.nii.gz",
        "vessels.nii.gz",
        "pulmonary_vessels.nii.gz",
    ]
    for candidate in vessel_candidates:
        test_path = os.path.join(seg_dir, candidate)
        if os.path.exists(test_path):
            vessel_path = test_path
            break

    if vessel_path is None:
        print("[process_vessel_segmentation] No vessel mask found in:", seg_dir)
        return None, None, None, None, None

    vessel_img = sitk.ReadImage(vessel_path)
    vessel_mask = sitk.GetArrayFromImage(vessel_img).astype(bool)

    # --- Build lung mask from lobes (fallback: use vessel mask extent) ---
    lung_parts = [
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz",
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz",
    ]
    lung_mask = np.zeros_like(vessel_mask, dtype=bool)
    for part in lung_parts:
        part_path = os.path.join(seg_dir, part)
        if os.path.exists(part_path):
            lung_mask |= sitk.GetArrayFromImage(sitk.ReadImage(part_path)).astype(bool)

    if not lung_mask.any():
        print("[process_vessel_segmentation] Lung lobes not found, falling back to vessel bbox extent.")
        lung_mask = vessel_mask.copy()

    # --- Build COMPLETE airway mask (trachea + all bronchi levels) ---
    airway_mask, airway_found = _gather_airway_mask(seg_dir, vessel_img)
    print(f"[process_vessel_segmentation] Airway parts found: {len(airway_found)} -> {airway_found}")

    # --- Save the FULL airway mask for visualization in Slicer ---
    airways_full_img = sitk.GetImageFromArray(airway_mask.astype(np.uint8))
    airways_full_img.CopyInformation(vessel_img)
    airways_full_path = os.path.join(output_dir, "airways_full.nii.gz")
    sitk.WriteImage(airways_full_img, airways_full_path)
    
    vessel_clean, cleaning_stats = adaptive_vessel_cleaning(
        vessel_mask, lung_mask, airway_mask, spacing,
        lung_erosion_mm=lung_erosion_mm,
        airway_dilation_mm=airway_dilation_mm,
        preserve_large_vessels=preserve_large_vessels,
        large_vessel_threshold_mm=large_vessel_threshold_mm
    )

    # Optionally save the dilated airways version used for vessel cleaning (handy for QC)
    if airway_mask.any():
        airway_kernel = create_spherical_kernel(airway_dilation_mm, spacing)
        airway_mask_dilated = ndimage.binary_dilation(airway_mask, structure=airway_kernel)
        airways_dil_img = sitk.GetImageFromArray(airway_mask_dilated.astype(np.uint8))
        airways_dil_img.CopyInformation(vessel_img)
        airways_dil_path = os.path.join(output_dir, "airways_full_dilated_for_cleaning.nii.gz")
        sitk.WriteImage(airways_dil_img, airways_dil_path)

    # --- Prune small vessels by size threshold(s) ---
    labeled, num_features = ndimage.label(vessel_clean)
    if num_features > 0:
        sizes = ndimage.sum(vessel_clean, labeled, range(1, num_features + 1))
        sizes = np.asarray(sizes, dtype=float)

        keep_mask = np.ones(num_features, dtype=bool)

        # Threshold by voxel count
        if min_vessel_voxels is not None:
            keep_mask &= (sizes >= float(min_vessel_voxels))
            print(f"  Filtering by min voxels: {min_vessel_voxels}")

        # Threshold by estimated physical diameter (mm)
        if min_vessel_diameter_mm is not None:
            voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
            volumes_mm3 = sizes * voxel_volume_mm3
            equiv_diam_mm = 2.0 * ((3.0 * volumes_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0))
            keep_mask &= (equiv_diam_mm >= float(min_vessel_diameter_mm))
            print(f"  Filtering by min diameter: {min_vessel_diameter_mm}mm")

        keep_labels = (np.where(keep_mask)[0] + 1).tolist()
        voxels_before = int(vessel_clean.sum())
        vessel_clean = np.isin(labeled, keep_labels)
        voxels_after = int(vessel_clean.sum())

    # --- Extract centerlines if requested ---
    centerlines = None
    if extract_skeleton:
        centerlines = extract_centerlines(vessel_clean)

    # --- Find seed regions (artery/vein) if available ---
    artery_seed_path, vein_seed_path = find_seed_regions(seg_dir)

    # --- Save cleaned vessels ---
    vessel_clean_img = sitk.GetImageFromArray(vessel_clean.astype(np.uint8))
    vessel_clean_img.CopyInformation(vessel_img)

    vessels_out_path = os.path.join(output_dir, "lung_vessels_cleaned.nii.gz")
    sitk.WriteImage(vessel_clean_img, vessels_out_path)
    
    # --- Save centerlines if extracted ---
    centerlines_path = None
    if centerlines is not None:
        centerlines_img = sitk.GetImageFromArray(centerlines.astype(np.uint8))
        centerlines_img.CopyInformation(vessel_img)
        centerlines_path = os.path.join(output_dir, "vessel_centerlines.nii.gz")
        sitk.WriteImage(centerlines_img, centerlines_path)

  
    erode_kernel = create_spherical_kernel(lung_erosion_mm, spacing)
    lung_mask_eroded = ndimage.binary_erosion(lung_mask, structure=erode_kernel)
    lung_mask_path = os.path.join(output_dir, "lung_mask_eroded.nii.gz")
    lung_mask_img = sitk.GetImageFromArray(lung_mask_eroded.astype(np.uint8))
    lung_mask_img.CopyInformation(vessel_img)
    sitk.WriteImage(lung_mask_img, lung_mask_path)
    
    # Save original lung mask for reference
    lung_mask_orig_path = os.path.join(output_dir, "lung_mask_original.nii.gz")
    lung_mask_orig_img = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
    lung_mask_orig_img.CopyInformation(vessel_img)
    sitk.WriteImage(lung_mask_orig_img, lung_mask_orig_path)

    # --- Save seed regions (artery and vein) if present ---
    if artery_seed_path:
        artery_seed_dest = os.path.join(output_dir, "seed_artery.nii.gz")
        sitk.WriteImage(sitk.ReadImage(artery_seed_path), artery_seed_dest)

    if vein_seed_path:
        vein_seed_dest = os.path.join(output_dir, "seed_vein.nii.gz")
        sitk.WriteImage(sitk.ReadImage(vein_seed_path), vein_seed_dest)

    return vessels_out_path, centerlines_path, lung_mask_path, artery_seed_path, vein_seed_path