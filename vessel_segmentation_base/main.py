from preprocessing.preprocessing import convert_mhd_to_nifti
from totalsegmentator.python_api import totalsegmentator
from code_vesselsegmentation.postprocessing import process_vessel_segmentation
import os
import tempfile
import shutil
import argparse

def process_single_scan(mhd_path, output_dir):
    """Elabora una singola scansione MHD"""
    # Verifica che il file esista
    if not os.path.isfile(mhd_path):
        print(f"[ERROR] File '{mhd_path}' not found.")
        return
    
    # Crea output folder se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(mhd_path))[0]
    print(f"Processing: {base_name}")
    
    # Step 1: Convert to NIfTI (using temporary directory)
    with tempfile.TemporaryDirectory() as nifti_dir:
        nifti_path = convert_mhd_to_nifti(mhd_path, nifti_dir)
        
        # Segmentazione vasi polmonari
        totalsegmentator(
            nifti_path, 
            nifti_dir, 
            task='lung_vessels', 
            fast=False,  
        )
        
        totalsegmentator(
            nifti_path, 
            nifti_dir, 
            task='total', 
            fast=False,
        )
        
        totalsegmentator(
            nifti_path,
            nifti_dir,
            task='heartchambers_highres',
            fast=False,
        )

        # Process vessel segmentation con parametri OTTIMIZZATI
        cleaned_path, centerlines_path, lung_mask_path, artery_seed_path, vein_seed_path = process_vessel_segmentation(
        nifti_dir,
        output_dir, 
        nifti_path,
        min_vessel_voxels=30,
        min_vessel_diameter_mm=0.6,
        lung_erosion_mm=1.5,
        airway_dilation_mm=4.0,
        preserve_large_vessels=True,
        large_vessel_threshold_mm=2.5,            
        extract_skeleton=True
        )

        if cleaned_path is None:
            print("\n[ERROR] Vessel segmentation failed!")
            return
        
        # Rename output files con basename originale
        
        # Cleaned vessels (output principale)
        final_segmentation_name = base_name + '_cleaned.nii.gz'
        final_segmentation_path = os.path.join(output_dir, final_segmentation_name)
        os.rename(cleaned_path, final_segmentation_path)
        print(f"  ✓ {final_segmentation_name}")

        # MODIFICA: Rinomina i seed invece di copiarli (se esistono)
        if artery_seed_path and os.path.exists(artery_seed_path):
            artery_seed_name = base_name + '_seed_artery.nii.gz'
            artery_seed_final_path = os.path.join(output_dir, artery_seed_name)
            os.rename(artery_seed_path, artery_seed_final_path)  # RENAME invece di COPY
            print(f"  ✓ {artery_seed_name}")

        if vein_seed_path and os.path.exists(vein_seed_path):
            vein_seed_name = base_name + '_seed_vein.nii.gz'
            vein_seed_final_path = os.path.join(output_dir, vein_seed_name)
            os.rename(vein_seed_path, vein_seed_final_path)  # RENAME invece di COPY
            print(f"  ✓ {vein_seed_name}")
            
def process_folder(input_folder, output_dir):
    """Elabora tutti i file MHD in una cartella"""
    # Verifica input folder
    if not os.path.isdir(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' not found or is not a directory.")
        return

    # Crea output folder
    os.makedirs(output_dir, exist_ok=True)

    # Lista file MHD e iterazione
    mhd_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mhd')]
    
    if not mhd_files:
        print(f"[WARNING] No .mhd files found in '{input_folder}'")
        return
    
    print(f"Found {len(mhd_files)} MHD files to process")
    
    for i, mhd_file in enumerate(mhd_files, 1):
        full_mhd_path = os.path.join(input_folder, mhd_file)
        print(f"\n[{i}/{len(mhd_files)}] Processing: {mhd_file}")
        
        process_single_scan(full_mhd_path, output_dir)


def main():
    input_folder = 'CARVE14'
    output_folder = 'vessels_cleaned'
    input_scan = '/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.2604.1126357612.7.mhd'
    #process_folder(input_folder, output_folder)
    process_single_scan(input_scan, output_folder)


if __name__ == "__main__":
    main()