import subprocess
import os
import SimpleITK as sitk
import tempfile
import numpy as np


def convert_mhd_to_nifti(mhd_path, output_dir):
    """
    Convert MHD file to NIfTI format for TotalSegmentator compatibility.
    Args:
    - mhd_path (str): Path to the input .mhd file.
    - output_dir (str): Directory where the converted .nii.gz file will be saved.
    Returns:
    - str: Path to the converted .nii.gz file.
    """
    # Read the MHD image
    image = sitk.ReadImage(mhd_path)
    
    # Print image info for debugging
    print(f"\nMHD Image Info:")
    print(f"  Size: {image.GetSize()}")
    print(f"  Spacing: {image.GetSpacing()}")
    print(f"  Origin: {image.GetOrigin()}")
    print(f"  Direction: {image.GetDirection()}")
    
    array = sitk.GetArrayFromImage(image)
    print(f"  Intensity range: [{array.min()}, {array.max()}]")
    print(f"  Array shape (Z,Y,X): {array.shape}")
    
    # Create output path for NIfTI file in the specified directory
    base_name = os.path.splitext(os.path.basename(mhd_path))[0]
    nifti_path = os.path.join(output_dir, f"{base_name}.nii.gz")
    
    # Check if the output directory exists, create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write as NIfTI
    sitk.WriteImage(image, nifti_path)
    print(f"Converted {mhd_path} to {nifti_path}")
    
    return nifti_path
