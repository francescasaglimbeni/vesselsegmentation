import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from code_vesselsegmentation.preprocessing import create_spherical_kernel, extract_centerlines
from code_vesselsegmentation.vessel_reconnection import (
    reconnect_isolated_vessels, 
    advanced_reconnection_with_centerlines
)


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
        
        # Cerca qualsiasi file che contenga "bronchus" o "trachea"
        if ('bronchus' in lower) or ('trachea' in lower):
            path = os.path.join(seg_dir, fname)
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img).astype(bool)
            
            if airway_mask is None:
                airway_mask = arr
            else:
                airway_mask |= arr  # Unione logica
            
            found.append(fname)

    if airway_mask is None:
        # Nessuna airway trovata: crea maschera vuota
        airway_mask = np.zeros(sitk.GetArrayFromImage(reference_img).shape, dtype=bool)

    return airway_mask, found


def adaptive_vessel_cleaning(vessel_mask, lung_mask, airway_mask, spacing, 
                             lung_erosion_mm=1.0, 
                             airway_dilation_mm=3.0,
                             preserve_large_vessels=True, 
                             large_vessel_threshold_mm=2.5):
    
    stats = {}
    
    erode_kernel = create_spherical_kernel(lung_erosion_mm, spacing)
    lung_mask_eroded = ndimage.binary_erosion(lung_mask, structure=erode_kernel)
    stats['lung_erosion_mm'] = lung_erosion_mm
    
    if airway_mask.any():
        airway_kernel = create_spherical_kernel(airway_dilation_mm, spacing)
        airway_mask_dilated = ndimage.binary_dilation(airway_mask, structure=airway_kernel)
    else:
        airway_mask_dilated = airway_mask
    stats['airway_dilation_mm'] = airway_dilation_mm
    
    if preserve_large_vessels:
        
        labeled_vessels, num_vessels = ndimage.label(vessel_mask)
        
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        vessel_volumes = ndimage.sum(vessel_mask, labeled_vessels, range(1, num_vessels + 1))
        vessel_volumes_mm3 = np.array(vessel_volumes) * voxel_volume_mm3
        
        vessel_diameters = 2.0 * ((3.0 * vessel_volumes_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0))
        
        large_vessel_labels = np.where(vessel_diameters >= large_vessel_threshold_mm)[0] + 1
        large_vessel_mask = np.isin(labeled_vessels, large_vessel_labels)
        
        stats['num_large_vessels'] = len(large_vessel_labels)
        stats['num_total_vessels'] = num_vessels
        
        print(f"  [Adaptive Cleaning] Found {len(large_vessel_labels)} large vessels "
              f"(out of {num_vessels} total)")
        
        minimal_erosion_kernel = create_spherical_kernel(0.5, spacing)  
        lung_mask_minimal = ndimage.binary_erosion(lung_mask, structure=minimal_erosion_kernel)
        
        large_vessels_clean = large_vessel_mask & lung_mask_minimal & ~airway_mask_dilated
        
        small_vessel_mask = vessel_mask & ~large_vessel_mask
        small_vessels_clean = small_vessel_mask & lung_mask_eroded & ~airway_mask_dilated
        
        vessel_clean = large_vessels_clean | small_vessels_clean
        
        stats['large_vessels_voxels'] = int(large_vessels_clean.sum())
        stats['small_vessels_voxels'] = int(small_vessels_clean.sum())
        
    else:
        # Trattamento uniforme (non consigliato)
        vessel_clean = vessel_mask & lung_mask_eroded & ~airway_mask_dilated
        stats['num_large_vessels'] = 0
        stats['num_total_vessels'] = 0
    
    stats['total_clean_voxels'] = int(vessel_clean.sum())
    stats['original_voxels'] = int(vessel_mask.sum())
    stats['removed_voxels'] = int(vessel_mask.sum() - vessel_clean.sum())
    
    return vessel_clean, stats


def process_vessel_segmentation(seg_dir, output_dir, original_image_path, 
                                min_vessel_voxels=20, 
                                min_vessel_diameter_mm=0.5, 
                                extract_skeleton=True,
                                lung_erosion_mm=1.0,
                                airway_dilation_mm=3.0,
                                preserve_large_vessels=True,
                                large_vessel_threshold_mm=2.5,
                                enable_reconnection=True,
                                max_gap_mm=0.5,
                                max_connection_distance_mm=2.0,
                                use_centerline_reconnection=False):
    
    # Leggi immagine originale per spacing
    original_img = sitk.ReadImage(original_image_path)
    spacing = original_img.GetSpacing()[::-1]  # (z, y, x)
    print(f"\nSpacing: {spacing} mm")

    os.makedirs(output_dir, exist_ok=True)
    # Cerca maschera vasi
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
        print("[ERROR] No vessel mask found in:", seg_dir)
        return None, None, None, None, None

    vessel_img = sitk.ReadImage(vessel_path)
    vessel_mask = sitk.GetArrayFromImage(vessel_img).astype(bool)
    print(f"  ✓ Vessel mask loaded: {vessel_mask.sum()} voxels")

    # Costruisci maschera polmonare dai lobi
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
        lung_mask = vessel_mask.copy()

    # Costruisci maschera airways completa
    airway_mask, airway_found = _gather_airway_mask(seg_dir, vessel_img)
    print(f"  ✓ Airway mask: {len(airway_found)} files found → {airway_mask.sum()} voxels")
    
    # Salva airways per visualizzazione
    airways_full_img = sitk.GetImageFromArray(airway_mask.astype(np.uint8))
    airways_full_img.CopyInformation(vessel_img)
    sitk.WriteImage(airways_full_img, os.path.join(output_dir, "airways_full.nii.gz"))
    
    vessel_clean, cleaning_stats = adaptive_vessel_cleaning(
        vessel_mask, lung_mask, airway_mask, spacing,
        lung_erosion_mm=lung_erosion_mm,
        airway_dilation_mm=airway_dilation_mm,
        preserve_large_vessels=preserve_large_vessels,
        large_vessel_threshold_mm=large_vessel_threshold_mm
    )

    # Salva airways dilatate (per QC)
    if airway_mask.any():
        airway_kernel = create_spherical_kernel(airway_dilation_mm, spacing)
        airway_mask_dilated = ndimage.binary_dilation(airway_mask, structure=airway_kernel)
        airways_dil_img = sitk.GetImageFromArray(airway_mask_dilated.astype(np.uint8))
        airways_dil_img.CopyInformation(vessel_img)
        sitk.WriteImage(airways_dil_img, 
                       os.path.join(output_dir, "airways_dilated_for_cleaning.nii.gz"))
    reconnection_stats = {}
    
    if enable_reconnection:
        # Salva stato pre-reconnection per QC
        vessel_before_reconnect = vessel_clean.copy()
        
        # Applica reconnection BASE (conservativa)
        vessel_clean, reconnection_stats = reconnect_isolated_vessels(
            vessel_clean,
            spacing,
            max_gap_mm=max_gap_mm,
            min_isolated_size=20,  
            max_isolated_size=100,
            max_connection_distance_mm=max_connection_distance_mm
        )
        
        vessel_before_img = sitk.GetImageFromArray(vessel_before_reconnect.astype(np.uint8))
        vessel_before_img.CopyInformation(vessel_img)
        sitk.WriteImage(vessel_before_img, 
                       os.path.join(output_dir, "vessels_before_reconnection.nii.gz"))
    
    labeled, num_features = ndimage.label(vessel_clean)
    
    if num_features > 0:
        sizes = ndimage.sum(vessel_clean, labeled, range(1, num_features + 1))
        sizes = np.asarray(sizes, dtype=float)

        keep_mask = np.ones(num_features, dtype=bool)

        # Filtro per numero voxels
        if min_vessel_voxels is not None:
            keep_mask &= (sizes >= float(min_vessel_voxels))
            removed_by_voxels = num_features - keep_mask.sum()
            print(f"  Removed {removed_by_voxels} components (< {min_vessel_voxels} voxels)")

        # Filtro per diametro fisico
        if min_vessel_diameter_mm is not None:
            voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
            volumes_mm3 = sizes * voxel_volume_mm3
            equiv_diam_mm = 2.0 * ((3.0 * volumes_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0))
            
            before_diam_filter = keep_mask.sum()
            keep_mask &= (equiv_diam_mm >= float(min_vessel_diameter_mm))
            removed_by_diameter = before_diam_filter - keep_mask.sum()

        keep_labels = (np.where(keep_mask)[0] + 1).tolist()
        voxels_before = int(vessel_clean.sum())
        vessel_clean = np.isin(labeled, keep_labels)
        voxels_after = int(vessel_clean.sum())
    centerlines = None
    
    if extract_skeleton:
        centerlines = extract_centerlines(vessel_clean)
        
        if enable_reconnection and use_centerline_reconnection and centerlines is not None:
            
            vessel_clean, centerline_reconnect_stats = advanced_reconnection_with_centerlines(
                vessel_clean,
                centerlines,
                spacing,
                max_gap_mm=1.0,  # Leggermente più permissivo per centerlines
                search_radius_mm=5.0
            )
            
            # Rigenera centerlines dopo reconnection
            centerlines = extract_centerlines(vessel_clean)
    
    # Cerca seed regions
    artery_seed_path, vein_seed_path = find_seed_regions(seg_dir)

    # Salva vasi puliti (OUTPUT PRINCIPALE)
    vessel_clean_img = sitk.GetImageFromArray(vessel_clean.astype(np.uint8))
    vessel_clean_img.CopyInformation(vessel_img)
    vessels_out_path = os.path.join(output_dir, "lung_vessels_cleaned.nii.gz")
    sitk.WriteImage(vessel_clean_img, vessels_out_path)
    
    # Salva centerlines
    centerlines_path = None
    if centerlines is not None:
        centerlines_img = sitk.GetImageFromArray(centerlines.astype(np.uint8))
        centerlines_img.CopyInformation(vessel_img)
        centerlines_path = os.path.join(output_dir, "vessel_centerlines.nii.gz")
        sitk.WriteImage(centerlines_img, centerlines_path)

    # Salva maschere polmonari (originale + erosa)
    erode_kernel = create_spherical_kernel(lung_erosion_mm, spacing)
    lung_mask_eroded = ndimage.binary_erosion(lung_mask, structure=erode_kernel)
    
    lung_mask_path = os.path.join(output_dir, "lung_mask_eroded.nii.gz")
    lung_mask_img = sitk.GetImageFromArray(lung_mask_eroded.astype(np.uint8))
    lung_mask_img.CopyInformation(vessel_img)
    sitk.WriteImage(lung_mask_img, lung_mask_path)
    
    lung_mask_orig_path = os.path.join(output_dir, "lung_mask_original.nii.gz")
    lung_mask_orig_img = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
    lung_mask_orig_img.CopyInformation(vessel_img)
    sitk.WriteImage(lung_mask_orig_img, lung_mask_orig_path)

    # Copia seed regions se presenti
    if artery_seed_path:
        artery_seed_dest = os.path.join(output_dir, "seed_artery.nii.gz")
        sitk.WriteImage(sitk.ReadImage(artery_seed_path), artery_seed_dest)

    if vein_seed_path:
        vein_seed_dest = os.path.join(output_dir, "seed_vein.nii.gz")
        sitk.WriteImage(sitk.ReadImage(vein_seed_path), vein_seed_dest)
    for key, value in cleaning_stats.items():
        print(f"  {key}: {value}")
    
    if reconnection_stats:
        for key, value in reconnection_stats.items():
            print(f"  {key}: {value}")
    
    return vessels_out_path, centerlines_path, lung_mask_path, artery_seed_path, vein_seed_path