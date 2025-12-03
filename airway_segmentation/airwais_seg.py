import os
import shutil
from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk

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

def segment_airwayfull_from_mhd(mhd_path, output_dir, fast=False):
    """
    Segmenta le vie aeree complete (trachea + bronchi) e salva un file unico
    chiamato `<base>_airwayfull.nii.gz` usando la classe 'lung_trachea_bronchia'.
    
    FIXED: Disabilita multiprocessing per evitare memory leak
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 1) Conversione MHD → NIfTI (per airwayfull) ===")
    nifti_path = convert_mhd_to_nifti(mhd_path, output_dir)

    print("\n=== 2) Segmentazione AIRWAYFULL con TotalSegmentator (task 'lung_vessels') ===")
    print("⚠️  Multiprocessing disabled for stability")
    
    # CRITICAL FIX: Disabilita multiprocessing
    totalsegmentator(
        nifti_path,
        output_dir,
        task="lung_vessels",
        fast=fast,
        nr_thr_resamp=1,      # Single thread for resampling
        nr_thr_saving=1,      # Single thread for saving
        force_split=False,    # Don't split processing
        crop_path=None,       # No intermediate cropping
        skip_saving=False,
    )

    airway_src = os.path.join(output_dir, "lung_trachea_bronchia.nii.gz")
    if not os.path.exists(airway_src):
        raise RuntimeError(
            "ERRORE: TotalSegmentator non ha generato lung_trachea_bronchia.nii.gz"
        )

    base = os.path.splitext(os.path.basename(mhd_path))[0]
    airway_dst = os.path.join(output_dir, f"{base}_airwayfull.nii.gz")

    shutil.move(airway_src, airway_dst)

    print(f"\n✓ AirwayFull estratto: {airway_dst}")
    return airway_dst