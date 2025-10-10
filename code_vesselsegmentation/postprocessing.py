import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
# skimage may not expose skeletonize_3d in some versions; try import with fallback
try:
    from skimage.morphology import skeletonize_3d
    _HAS_SKELETONIZE_3D = True
except Exception:
    from skimage.morphology import skeletonize
    _HAS_SKELETONIZE_3D = False
from totalsegmentator.python_api import totalsegmentator
# Reuse helper functions from preprocessing to avoid duplication
from code_vesselsegmentation.preprocessing import create_spherical_kernel, extract_centerlines

def process_vessel_segmentation(seg_dir, output_dir, original_image_path, 
                                extract_skeleton=True):
    """
    Post-process vessel segmentation for artery/vein classification:
    1. Combine lung lobes into single lung mask
    2. Remove airway walls from vessel segmentation
    3. Restrict vessels to lung parenchyma
    4. Extract centerlines (optional, recommended for A/V classification)
    5. Save connectivity-preserving masks
    
    Args:
        seg_dir: Directory with TotalSegmentator output
        output_dir: Directory to save processed masks
        original_image_path: Path to original image (for spacing info)
        extract_skeleton: If True, compute vessel centerlines
    """
    print("\n=== Post-processing vessel segmentation ===")
    
    # Read original image for spacing
    original_img = sitk.ReadImage(original_image_path)
    spacing = original_img.GetSpacing()[::-1]  # (x,y,z) -> (z,y,x)
    print(f"Image spacing (z,y,x): {spacing} mm")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load vessel segmentation
    expected_name = "lung_vessels.nii.gz"
    vessel_path = os.path.join(seg_dir, expected_name)
    if not os.path.exists(vessel_path):
        # Try to find any candidate file containing 'vessel' in its name
        candidates = [f for f in os.listdir(seg_dir) if 'vessel' in f.lower() and f.lower().endswith(('.nii', '.nii.gz'))]
        if candidates:
            vessel_path = os.path.join(seg_dir, candidates[0])
            print(f"NOTICE: expected '{expected_name}' not found. Using '{candidates[0]}' instead.")
        else:
            print(f"WARNING: Vessel segmentation not found: {vessel_path}")
            print(f"Contents of '{seg_dir}': {os.listdir(seg_dir)}")
            # Return gracefully so the pipeline can continue or the caller can decide next steps
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
    for part in lung_parts:
        part_path = os.path.join(seg_dir, part)
        if os.path.exists(part_path):
            lobe = sitk.GetArrayFromImage(sitk.ReadImage(part_path)).astype(bool)
            lung_mask |= lobe
            print(f"  Added {part}")
    
    if not lung_mask.any():
        print("WARNING: No lung lobes found! Using original vessel mask.")
        lung_mask = np.ones_like(vessel_mask, dtype=bool)
    else:
        print(f"Combined lung mask voxels: {lung_mask.sum()}")
    
    # 3. Erode lung mask (remove border voxels)
    # Paper uses r=3 voxels, we use ~3mm in physical space
    erode_kernel = create_spherical_kernel(3.0, spacing)
    lung_mask_eroded = ndimage.binary_erosion(lung_mask, structure=erode_kernel)
    print(f"Eroded lung mask voxels: {lung_mask_eroded.sum()}")
    
    # 4. Load and dilate airway mask
    airway_parts = ["trachea.nii.gz", "bronchus_left.nii.gz", "bronchus_right.nii.gz"]
    airway_mask = np.zeros_like(vessel_mask, dtype=bool)
    
    for part in airway_parts:
        part_path = os.path.join(seg_dir, part)
        if os.path.exists(part_path):
            airway = sitk.GetArrayFromImage(sitk.ReadImage(part_path)).astype(bool)
            airway_mask |= airway
            print(f"  Added {part}")
    
    if airway_mask.any():
        # Dilate airways to include ~2mm wall thickness
        dilate_kernel = create_spherical_kernel(3.0, spacing)
        airway_mask_dilated = ndimage.binary_dilation(airway_mask, structure=dilate_kernel)
        print(f"Dilated airway voxels: {airway_mask_dilated.sum()}")
    else:
        print("WARNING: No airways found. Skipping airway removal.")
        airway_mask_dilated = np.zeros_like(vessel_mask, dtype=bool)
    
    # 5. Clean vessel mask: (vessels ∩ lung) - airways
    vessel_clean = vessel_mask & lung_mask_eroded & ~airway_mask_dilated
    print(f"Final cleaned vessel voxels: {vessel_clean.sum()}")
    print(f"Removed {vessel_mask.sum() - vessel_clean.sum()} voxels " +
          f"({100*(1 - vessel_clean.sum()/max(vessel_mask.sum(), 1)):.1f}%)")
    
    # 6. Extract centerlines (critical for A/V classification)
    if extract_skeleton:
        # If skeletonize_3d isn't available, extract_centerlines will fall back to 2D per-slice
        centerlines = extract_centerlines(vessel_clean, min_branch_length=10)
    else:
        centerlines = None
    
    # 7. Save cleaned mask and centerlines
    vessel_clean_img = sitk.GetImageFromArray(vessel_clean.astype(np.uint8))
    vessel_clean_img.CopyInformation(vessel_img)
    
    output_path = os.path.join(output_dir, "lung_vessels_cleaned.nii.gz")
    sitk.WriteImage(vessel_clean_img, output_path)
    print(f"\nSaved cleaned vessels to: {output_path}")
    
    if centerlines is not None:
        centerlines_img = sitk.GetImageFromArray(centerlines.astype(np.uint8))
        centerlines_img.CopyInformation(vessel_img)
        centerlines_path = os.path.join(output_dir, "vessel_centerlines.nii.gz")
        sitk.WriteImage(centerlines_img, centerlines_path)
        print(f"Saved centerlines to: {centerlines_path}")
    
    # Optional: Save intermediate masks for inspection
    lung_img = sitk.GetImageFromArray(lung_mask_eroded.astype(np.uint8))
    lung_img.CopyInformation(vessel_img)
    sitk.WriteImage(lung_img, os.path.join(output_dir, "lung_mask_eroded.nii.gz"))
    
    if airway_mask_dilated.any():
        airway_img = sitk.GetImageFromArray(airway_mask_dilated.astype(np.uint8))
        airway_img.CopyInformation(vessel_img)
        sitk.WriteImage(airway_img, os.path.join(output_dir, "airway_mask_dilated.nii.gz"))
    
    return output_path, centerlines_path if centerlines is not None else None
