import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import kimimaro
from totalsegmentator.python_api import totalsegmentator


def convert_mhd_to_nifti(mhd_path, output_dir):
    """
    Convert MHD file to NIfTI format for TotalSegmentator compatibility.
    """
    image = sitk.ReadImage(mhd_path)
    
    print(f"\nMHD Image Info:")
    print(f"  Size: {image.GetSize()}")
    print(f"  Spacing: {image.GetSpacing()}")
    print(f"  Origin: {image.GetOrigin()}")
    
    array = sitk.GetArrayFromImage(image)
    print(f"  Intensity range: [{array.min()}, {array.max()}]")
    print(f"  Array shape (Z,Y,X): {array.shape}")
    
    base_name = os.path.splitext(os.path.basename(mhd_path))[0]
    nifti_path = os.path.join(output_dir, f"{base_name}.nii.gz")
    
    os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(image, nifti_path)
    print(f"Converted {mhd_path} to {nifti_path}")
    
    return nifti_path

def create_spherical_kernel(radius_mm, spacing):
    """
    Create a spherical structuring element based on physical spacing.
    
    Args:
        radius_mm: Radius in millimeters
        spacing: Voxel spacing (z, y, x) in mm
    """
    # Convert radius to voxels for each dimension
    radius_voxels = [int(np.ceil(radius_mm / s)) for s in spacing]
    
    # Create grid
    z = np.arange(-radius_voxels[0], radius_voxels[0] + 1)
    y = np.arange(-radius_voxels[1], radius_voxels[1] + 1)
    x = np.arange(-radius_voxels[2], radius_voxels[2] + 1)
    
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    
    # Physical distance in mm
    dist = np.sqrt((zz * spacing[0])**2 + 
                   (yy * spacing[1])**2 + 
                   (xx * spacing[2])**2)
    
    return dist <= radius_mm


def extract_centerlines(vessel_mask, spacing=None, min_branch_length=10):
    """
    Extract vessel centerlines using 3D skeletonization.
    
    Args:
        vessel_mask: Binary vessel mask (numpy array)
        min_branch_length: Minimum branch length to keep (voxels)
    
    Returns:
        centerlines: Binary skeleton mask
    """
    print("\n--- Extracting centerlines ---")
    print(f"Input vessel voxels: {vessel_mask.sum()}")

    # Use kimimaro for 3D skeletonization (assume kimimaro is available)
    if spacing is None:
        anisotropy = (1.0, 1.0, 1.0)
    else:
        anisotropy = spacing

    lbl = vessel_mask.astype(np.uint32)
    skels = kimimaro.skeletonize(
        lbl,
        anisotropy=anisotropy,
        dust_threshold=50,
        fix_branching=True,
        progress=False,
    )

    sk_mask = np.zeros_like(lbl, dtype=bool)
    for s in skels.values():
        if hasattr(s, 'vertices') and len(s.vertices) > 0:
            verts = np.asarray(s.vertices, dtype=int)
            sk_mask[verts[:, 0], verts[:, 1], verts[:, 2]] = True

    skeleton = sk_mask
    print(f"kimimaro: raw skeleton voxels: {skeleton.sum()}")
    
    # Optional: Remove small disconnected components
    labeled, num_features = ndimage.label(skeleton)
    if num_features > 1:
        sizes = ndimage.sum(skeleton, labeled, range(1, num_features + 1))
        mask_sizes = sizes >= min_branch_length
        keep_labels = np.where(mask_sizes)[0] + 1
        
        cleaned_skeleton = np.isin(labeled, keep_labels)
        removed = skeleton.sum() - cleaned_skeleton.sum()
        print(f"Removed {removed} voxels from {num_features - len(keep_labels)} small branches")
        skeleton = cleaned_skeleton
    
    print(f"Final skeleton voxels: {skeleton.sum()}")
    return skeleton

