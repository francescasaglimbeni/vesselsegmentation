import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

try:
    from skimage.morphology import skeletonize_3d
    _HAS_SKELETONIZE_3D = True
except Exception:
    from skimage.morphology import skeletonize
    _HAS_SKELETONIZE_3D = False

from code_vesselsegmentation.preprocessing import create_spherical_kernel, extract_centerlines


def find_seed_regions(seg_dir):
    """
    Intelligently find seed regions for A/V classification from available structures.
    Returns paths to artery and vein seed masks (or None if not found).
    """
    print("\n--- Searching for A/V seed regions ---")
    
    seed_candidates = {
        'artery': [
            'pulmonary_artery.nii.gz',
            'aorta.nii.gz',  # Backup: aorta connects to systemic arteries
        ],
        'vein': [
            'heart_atrium_left.nii.gz',
            'pulmonary_vein.nii.gz',
            'heart.nii.gz',  # Can extract left atrium region
        ]
    }
    
    found_seeds = {'artery': None, 'vein': None}
    
    for vessel_type, candidates in seed_candidates.items():
        for candidate in candidates:
            path = os.path.join(seg_dir, candidate)
            if os.path.exists(path):
                print(f"  ✓ Found {vessel_type} seed: {candidate}")
                found_seeds[vessel_type] = path
                break
        
        if found_seeds[vessel_type] is None:
            print(f"  ✗ No {vessel_type} seed found in: {candidates}")
    
    return found_seeds['artery'], found_seeds['vein']


def process_vessel_segmentation(seg_dir, output_dir, original_image_path, 
                                extract_skeleton=True):
    """
    Post-process vessel segmentation for artery/vein classification.
    Now with flexible handling of available structures.
    """
    print("\n=== Post-processing vessel segmentation ===")
    
    # Read original image for spacing
    original_img = sitk.ReadImage(original_image_path)
    spacing = original_img.GetSpacing()[::-1]  # (x,y,z) -> (z,y,x)
    print(f"Image spacing (z,y,x): {spacing} mm")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load vessel segmentation (with flexible naming)
    vessel_candidates = [
        "lung_vessels.nii.gz",
        "vessels.nii.gz",
        "pulmonary_vessels.nii.gz"
    ]
    
    vessel_path = None
    for candidate in vessel_candidates:
        test_path = os.path.join(seg_dir, candidate)
        if os.path.exists(test_path):
            vessel_path = test_path
            print(f"✓ Found vessel segmentation: {candidate}")
            break
    
    if vessel_path is None:
        # Last resort: search for any file with 'vessel' in name
        all_files = os.listdir(seg_dir)
        vessel_files = [f for f in all_files if 'vessel' in f.lower() and f.endswith('.nii.gz')]
        
        if vessel_files:
            vessel_path = os.path.join(seg_dir, vessel_files[0])
            print(f"⚠ Using best guess: {vessel_files[0]}")
        else:
            print(f"✗ ERROR: No vessel segmentation found!")
            print(f"   Searched for: {vessel_candidates}")
            print(f"   Available files: {all_files}")
            return None, None
    
    vessel_img = sitk.ReadImage(vessel_path)
    vessel_mask = sitk.GetArrayFromImage(vessel_img).astype(bool)
    print(f"Original vessel voxels: {vessel_mask.sum()}")
    
    # 2. Combine lung lobes
    lung_parts = [
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz", 
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz"
    ]
    
    lung_mask = np.zeros_like(vessel_mask, dtype=bool)
    found_lobes = 0
    
    for part in lung_parts:
        part_path = os.path.join(seg_dir, part)
        if os.path.exists(part_path):
            lobe = sitk.GetArrayFromImage(sitk.ReadImage(part_path)).astype(bool)
            lung_mask |= lobe
            found_lobes += 1
    
    print(f"✓ Combined {found_lobes}/5 lung lobes")
    
    if not lung_mask.any():
        print("⚠ WARNING: No lung lobes found! Using vessel mask as ROI.")
        lung_mask = vessel_mask.copy()
    else:
        print(f"Combined lung mask voxels: {lung_mask.sum()}")
    
    # 3. Erode lung mask to remove pleural vessels
    erode_kernel = create_spherical_kernel(3.0, spacing)
    lung_mask_eroded = ndimage.binary_erosion(lung_mask, structure=erode_kernel)
    print(f"Eroded lung mask voxels: {lung_mask_eroded.sum()}")
    
    # 4. Load and dilate airway mask
    airway_parts = ["trachea.nii.gz", "bronchus_left.nii.gz", "bronchus_right.nii.gz"]
    airway_mask = np.zeros_like(vessel_mask, dtype=bool)
    found_airways = 0
    
    for part in airway_parts:
        part_path = os.path.join(seg_dir, part)
        if os.path.exists(part_path):
            airway = sitk.GetArrayFromImage(sitk.ReadImage(part_path)).astype(bool)
            airway_mask |= airway
            found_airways += 1
    
    print(f"✓ Combined {found_airways}/3 airway structures")
    
    if airway_mask.any():
        dilate_kernel = create_spherical_kernel(3.0, spacing)
        airway_mask_dilated = ndimage.binary_dilation(airway_mask, structure=dilate_kernel)
        print(f"Dilated airway voxels: {airway_mask_dilated.sum()}")
    else:
        print("⚠ WARNING: No airways found. Skipping airway removal.")
        airway_mask_dilated = np.zeros_like(vessel_mask, dtype=bool)
    
    # 5. Clean vessel mask
    vessel_clean = vessel_mask & lung_mask_eroded & ~airway_mask_dilated
    removed = vessel_mask.sum() - vessel_clean.sum()
    print(f"✓ Final cleaned vessel voxels: {vessel_clean.sum()}")
    print(f"  Removed {removed} voxels ({100*removed/max(vessel_mask.sum(), 1):.1f}%)")
    
    # 6. Extract centerlines
    centerlines = None
    if extract_skeleton and vessel_clean.sum() > 0:
        print("\n--- Extracting centerlines ---")
        centerlines = extract_centerlines(vessel_clean, min_branch_length=10)
    
    # 7. Find seed regions for A/V classification
    artery_seed_path, vein_seed_path = find_seed_regions(seg_dir)
    
    # 8. Save all outputs
    vessel_clean_img = sitk.GetImageFromArray(vessel_clean.astype(np.uint8))
    vessel_clean_img.CopyInformation(vessel_img)
    
    output_path = os.path.join(output_dir, "lung_vessels_cleaned.nii.gz")
    sitk.WriteImage(vessel_clean_img, output_path)
    print(f"\n✓ Saved cleaned vessels: {output_path}")
    
    centerlines_path = None
    if centerlines is not None:
        centerlines_img = sitk.GetImageFromArray(centerlines.astype(np.uint8))
        centerlines_img.CopyInformation(vessel_img)
        centerlines_path = os.path.join(output_dir, "vessel_centerlines.nii.gz")
        sitk.WriteImage(centerlines_img, centerlines_path)
        print(f"✓ Saved centerlines: {centerlines_path}")
    
    # Save intermediate masks for inspection
    lung_img = sitk.GetImageFromArray(lung_mask_eroded.astype(np.uint8))
    lung_img.CopyInformation(vessel_img)
    sitk.WriteImage(lung_img, os.path.join(output_dir, "lung_mask_eroded.nii.gz"))
    
    if airway_mask_dilated.any():
        airway_img = sitk.GetImageFromArray(airway_mask_dilated.astype(np.uint8))
        airway_img.CopyInformation(vessel_img)
        sitk.WriteImage(airway_img, os.path.join(output_dir, "airway_mask_dilated.nii.gz"))
    
    # Copy seed regions to output directory for convenience
    if artery_seed_path:
        import shutil
        dest = os.path.join(output_dir, "seed_artery.nii.gz")
        shutil.copy(artery_seed_path, dest)
        print(f"✓ Copied artery seed: {dest}")
    
    if vein_seed_path:
        import shutil
        dest = os.path.join(output_dir, "seed_vein.nii.gz")
        shutil.copy(vein_seed_path, dest)
        print(f"✓ Copied vein seed: {dest}")
    
    # Print summary of what's available for A/V classification
    print("\n📋 A/V Classification Resources:")
    print(f"  {'✓' if centerlines is not None else '✗'} Centerlines extracted")
    print(f"  {'✓' if artery_seed_path else '✗'} Artery seed region available")
    print(f"  {'✓' if vein_seed_path else '✗'} Vein seed region available")
    
    if not artery_seed_path or not vein_seed_path:
        print("\n⚠ WARNING: Missing seed regions for supervised A/V classification!")
        print("  Consider using unsupervised methods (e.g., vessel radius, topology)")
    
    return output_path, centerlines_path