# vessel_seg.py
import os
import SimpleITK as sitk
import numpy as np
from totalsegmentator.python_api import totalsegmentator


def run_vessel_segmentation(input_nifti_path, output_dir, fast=False):
    """
    Esegue segmentazione vasi polmonari con TotalSegmentator.
    Restituisce mask combinata di tutti i vasi.
    
    Args:
        input_nifti_path: Path al file NIfTI input
        output_dir: Directory per output TotalSegmentator
        fast: Se True, usa modalità fast (meno accurata ma veloce)
        
    Returns:
        str: Path alla mask combinata vasi
    """
    print("\n=== Running Vessel Segmentation ===")
    print(f"  Input: {input_nifti_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Fast mode: {fast}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Segmentazione task lung_vessels
    print("\n  Running TotalSegmentator (lung_vessels)...")
    totalsegmentator(
        input_nifti_path, 
        output_dir, 
        task='lung_vessels', 
        fast=fast
    )

    # TotalSegmentator produce lung_vessels.nii.gz
    lung_vessels_path = os.path.join(output_dir, "lung_vessels.nii.gz")

    if not os.path.exists(lung_vessels_path):
        raise RuntimeError(
            f"Expected output lung_vessels.nii.gz not found in {output_dir}. "
            f"TotalSegmentator may have failed."
        )

    # Carica e binarizza
    vessels_img = sitk.ReadImage(lung_vessels_path)
    vessels_array = sitk.GetArrayFromImage(vessels_img) > 0

    vessels_voxels = np.sum(vessels_array)
    print(f"\n  ✓ Segmentation complete: {vessels_voxels:,} vessel voxels")

    # Salva mask combinata
    combined_path = os.path.join(output_dir, "vessels_combined.nii.gz")
    
    combined_img = sitk.GetImageFromArray(vessels_array.astype(np.uint8))
    combined_img.CopyInformation(vessels_img)
    sitk.WriteImage(combined_img, combined_path)
    
    print(f"  ✓ Saved combined mask: {combined_path}")

    return combined_path


def run_vessel_segmentation_with_anatomy(input_nifti_path, output_dir, fast=False):
    """
    Esegue segmentazione vasi + strutture anatomiche (per reference).
    Utile se vuoi anche PA e atrio sinistro per visualizzazione.
    
    Args:
        input_nifti_path: Path al file NIfTI input
        output_dir: Directory per output
        fast: Se True, usa modalità fast
        
    Returns:
        dict: {
            'vessels': path vessels,
            'pulmonary_artery': path PA (se disponibile),
            'left_atrium': path LA (se disponibile)
        }
    """
    print("\n=== Running Vessel Segmentation with Anatomy ===")
    
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # 1. Vasi polmonari
    vessels_path = run_vessel_segmentation(input_nifti_path, output_dir, fast=fast)
    results['vessels'] = vessels_path

    # 2. Heart chambers (opzionale, per context anatomico)
    print("\n  Running TotalSegmentator (heartchambers_highres)...")
    
    heart_dir = os.path.join(output_dir, "heart_chambers")
    os.makedirs(heart_dir, exist_ok=True)
    
    try:
        totalsegmentator(
            input_nifti_path,
            heart_dir,
            task='heartchambers_highres',
            fast=fast
        )
        
        # Check output
        pa_path = os.path.join(heart_dir, "pulmonary_artery.nii.gz")
        la_path = os.path.join(heart_dir, "heart_atrium_left.nii.gz")
        
        if os.path.exists(pa_path):
            results['pulmonary_artery'] = pa_path
            print(f"  ✓ Pulmonary artery: {pa_path}")
        
        if os.path.exists(la_path):
            results['left_atrium'] = la_path
            print(f"  ✓ Left atrium: {la_path}")
            
    except Exception as e:
        print(f"  Warning: Heart chambers segmentation failed: {e}")
        print("  Continuing with vessels only...")

    return results