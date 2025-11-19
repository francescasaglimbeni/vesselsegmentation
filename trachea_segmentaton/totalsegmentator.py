import os
import shutil
from totalsegmentator.python_api import totalsegmentator

from preprocessing.preprocessing import convert_mhd_to_nifti

input_mhd_path = "/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.2604.1126357612.7.mhd"      # <--- modifica qui
output_dir = "/content/vesselsegmentation/output"             
fast_mode = False                                   


def segment_trachea_from_mhd(mhd_path, output_dir, fast=False):
    """
    Converte la CT da .mhd a .nii.gz e segmenta SOLO la trachea usando TotalSegmentator.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 1) Conversione MHD → NIfTI ===")
    nifti_path = convert_mhd_to_nifti(mhd_path, output_dir)

    print("\n=== 2) Segmentazione TRACHEA con TotalSegmentator ===")
    # TotalSegmentator può estrarre SOLO la classe richiesta
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

    # Rinominiamo il file per dare un nome più chiaro
    base = os.path.splitext(os.path.basename(mhd_path))[0]
    trachea_dst = os.path.join(output_dir, f"{base}_trachea.nii.gz")

    shutil.move(trachea_src, trachea_dst)

    print(f"\n✓ Trachea estratta: {trachea_dst}")
    return trachea_dst


trachea_path = segment_trachea_from_mhd(
    input_mhd_path, output_dir, fast=fast_mode
)

print(f"\n=== COMPLETATO ===")
print(f"File finale: {trachea_path}")
