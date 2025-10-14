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

def process_vessel_segmentation(seg_dir, output_dir, original_image_path, extract_skeleton=True):
    original_img = sitk.ReadImage(original_image_path)
    spacing = original_img.GetSpacing()[::-1]

    os.makedirs(output_dir, exist_ok=True)

    vessel_path = None
    vessel_candidates = ["lung_vessels.nii.gz", "vessels.nii.gz", "pulmonary_vessels.nii.gz"]
    for candidate in vessel_candidates:
        test_path = os.path.join(seg_dir, candidate)
        if os.path.exists(test_path):
            vessel_path = test_path
            break

    if vessel_path is None:
        return None, None, None, None, None

    vessel_img = sitk.ReadImage(vessel_path)
    vessel_mask = sitk.GetArrayFromImage(vessel_img).astype(bool)

    lung_parts = ["lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz", "lung_upper_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz", "lung_lower_lobe_right.nii.gz"]
    lung_mask = np.zeros_like(vessel_mask, dtype=bool)
    for part in lung_parts:
        part_path = os.path.join(seg_dir, part)
        if os.path.exists(part_path):
            lung_mask |= sitk.GetArrayFromImage(sitk.ReadImage(part_path)).astype(bool)

    if not lung_mask.any():
        lung_mask = vessel_mask.copy()

    erode_kernel = create_spherical_kernel(3.0, spacing)
    lung_mask_eroded = ndimage.binary_erosion(lung_mask, structure=erode_kernel)

    airway_parts = ["trachea.nii.gz", "bronchus_left.nii.gz", "bronchus_right.nii.gz"]
    airway_mask = np.zeros_like(vessel_mask, dtype=bool)
    for part in airway_parts:
        part_path = os.path.join(seg_dir, part)
        if os.path.exists(part_path):
            airway_mask |= sitk.GetArrayFromImage(sitk.ReadImage(part_path)).astype(bool)

    if airway_mask.any():
        airway_mask_dilated = ndimage.binary_dilation(airway_mask, structure=create_spherical_kernel(3.0, spacing))

    vessel_clean = vessel_mask & lung_mask_eroded & ~airway_mask_dilated

    centerlines = None
    if extract_skeleton:
        centerlines = extract_centerlines(vessel_clean)

    artery_seed_path, vein_seed_path = find_seed_regions(seg_dir)

    vessel_clean_img = sitk.GetImageFromArray(vessel_clean.astype(np.uint8))
    vessel_clean_img.CopyInformation(vessel_img)

    output_path = os.path.join(output_dir, "lung_vessels_cleaned.nii.gz")
    sitk.WriteImage(vessel_clean_img, output_path)

    # Save centerlines if extracted
    centerlines_path = None
    if centerlines is not None:
        centerlines_img = sitk.GetImageFromArray(centerlines.astype(np.uint8))
        centerlines_img.CopyInformation(vessel_img)
        centerlines_path = os.path.join(output_dir, "vessel_centerlines.nii.gz")
        sitk.WriteImage(centerlines_img, centerlines_path)

    # Save lung mask (eroded)
    lung_mask_path = os.path.join(output_dir, "lung_mask_eroded.nii.gz")
    lung_mask_img = sitk.GetImageFromArray(lung_mask_eroded.astype(np.uint8))
    lung_mask_img.CopyInformation(vessel_img)
    sitk.WriteImage(lung_mask_img, lung_mask_path)

    # Save seed regions (artery and vein)
    if artery_seed_path:
        artery_seed_dest = os.path.join(output_dir, "seed_artery.nii.gz")
        sitk.WriteImage(sitk.ReadImage(artery_seed_path), artery_seed_dest)

    if vein_seed_path:
        vein_seed_dest = os.path.join(output_dir, "seed_vein.nii.gz")
        sitk.WriteImage(sitk.ReadImage(vein_seed_path), vein_seed_dest)

    return output_path, centerlines_path, lung_mask_path, artery_seed_path, vein_seed_path
