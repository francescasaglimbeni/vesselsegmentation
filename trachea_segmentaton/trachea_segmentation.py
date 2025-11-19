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


def segment_trachea_from_mhd(mhd_path, output_dir, fast=False):
    """
    Converte la CT da .mhd a .nii.gz e segmenta SOLO la trachea usando TotalSegmentator (task 'total').
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 1) Conversione MHD → NIfTI (per trachea) ===")
    nifti_path = convert_mhd_to_nifti(mhd_path, output_dir)

    print("\n=== 2) Segmentazione TRACHEA con TotalSegmentator (task 'total') ===")
    totalsegmentator(
        nifti_path,
        output_dir,
        task="total",             # task generale
        roi_subset=["trachea"],   # <-- SOLO trachea
        fast=fast,
    )

    trachea_src = os.path.join(output_dir, "trachea.nii.gz")
    if not os.path.exists(trachea_src):
        raise RuntimeError(
            "ERRORE: TotalSegmentator non ha generato trachea.nii.gz"
        )

    base = os.path.splitext(os.path.basename(mhd_path))[0]
    trachea_dst = os.path.join(output_dir, f"{base}_trachea.nii.gz")

    shutil.move(trachea_src, trachea_dst)

    print(f"\n✓ Trachea estratta: {trachea_dst}")
    return trachea_dst


def segment_airways_from_mhd(mhd_path, output_dir, fast=False):
    """
    Converte la CT da .mhd a .nii.gz e segmenta le VIE AEREE (trachea + bronchi)
    usando il task 'lung_vessels' di TotalSegmentator, prendendo la classe 'lung_trachea_bronchia'.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 1) Conversione MHD → NIfTI (per vie aeree) ===")
    nifti_path = convert_mhd_to_nifti(mhd_path, output_dir)

    print("\n=== 2) Segmentazione VIE AEREE con TotalSegmentator (task 'lung_vessels') ===")
    totalsegmentator(
        nifti_path,
        output_dir,
        task="lung_vessels",                     # <-- task specifico
        roi_subset=["lung_trachea_bronchia"],    # <-- SOLO vie aeree
        fast=fast,
    )

    airways_src = os.path.join(output_dir, "lung_trachea_bronchia.nii.gz")
    if not os.path.exists(airways_src):
        raise RuntimeError(
            "ERRORE: TotalSegmentator non ha generato lung_trachea_bronchia.nii.gz"
        )

    base = os.path.splitext(os.path.basename(mhd_path))[0]
    airways_dst = os.path.join(output_dir, f"{base}_airways.nii.gz")

    shutil.move(airways_src, airways_dst)

    print(f"\n✓ Vie aeree estratte: {airways_dst}")
    return airways_dst


if __name__ == "__main__":
    # <--- modifica qui i path come ti serve
    input_mhd_path = "/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.2604.1126357612.7.mhd"
    output_dir = "/content/vesselsegmentation/output"
    fast_mode = False

    # Trachea (solo struttura 'trachea' dal task 'total')
    trachea_path = segment_trachea_from_mhd(
        input_mhd_path, output_dir, fast=fast_mode
    )

    # Vie aeree (trachea + bronchi dal task 'lung_vessels')
    airways_path = segment_airways_from_mhd(
        input_mhd_path, output_dir, fast=fast_mode
    )

    print("\n=== COMPLETATO ===")
    print(f"File trachea: {trachea_path}")
    print(f"File vie aeree: {airways_path}")
