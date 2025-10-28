from code_vesselsegmentation.preprocessing import convert_mhd_to_nifti
from totalsegmentator.python_api import totalsegmentator
from code_vesselsegmentation.postprocessing import process_vessel_segmentation
import os
import tempfile


def main():
    # Paths
    #mhd_path = 'CARVE14/1.2.840.113704.1.111.2604.1126357612.7.mhd'
    
    #make it iterative accross all mhd files in a folder
    input_folder = 'CARVE14'

    # verify input folder exists
    if not os.path.isdir(input_folder):
        print(f"[ERROR] Input folder '{input_folder}' not found or is not a directory.")
        return

    # create output folder once
    output_dir = 'vessels_cleaned'
    os.makedirs(output_dir, exist_ok=True)

    # list .mhd files and iterate using full paths
    mhd_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mhd')]
    for mhd_file in mhd_files:
        full_mhd_path = os.path.join(input_folder, mhd_file)
        # Step 1: Convert to NIfTI (using temporary directory)
        with tempfile.TemporaryDirectory() as nifti_dir:
            nifti_path = convert_mhd_to_nifti(full_mhd_path, nifti_dir)
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

            
            # Process vessel segmentation con parametri OTTIMIZZATI
            cleaned_path, centerlines_path, lung_mask_path, artery_seed_path, vein_seed_path = process_vessel_segmentation(
                nifti_dir,
                output_dir,
                nifti_path,
                min_vessel_voxels=20, # con spacing 0.6mm = ~0.1mm³ → rumore, non vasi
                min_vessel_diameter_mm=0.5, # rimuove componenti con diametro equivalente < 0.5mm 
                lung_erosion_mm=1.0,
                airway_dilation_mm=3.0,
                preserve_large_vessels=True,
                large_vessel_threshold_mm=2.5,            
                extract_skeleton=True,
                #valori riconnessione
                enable_reconnection=False,
                max_gap_mm=1.5,# raggio morphological closing per gap microscopici
                max_connection_distance_mm=3.0,
                use_centerline_reconnection=False
            )

            if cleaned_path is None:
                print("\n[ERROR] Vessel segmentation failed!")
                return
            
            # Rename output files con basename originale
            base_name = os.path.splitext(os.path.basename(mhd_file))[0]
            
            # Cleaned vessels (output principale)
            final_segmentation_name = base_name + '_cleaned.nii.gz'
            final_segmentation_path = os.path.join(output_dir, final_segmentation_name)
            os.rename(cleaned_path, final_segmentation_path)
            print(f"  ✓ {final_segmentation_name}")

            # Centerlines
            if centerlines_path:
                centerlines_name = base_name + '_centerlines.nii.gz'
                centerlines_final_path = os.path.join(output_dir, centerlines_name)
                os.rename(centerlines_path, centerlines_final_path)
                print(f"  ✓ {centerlines_name}")

            '''# Lung mask (eroded)
            if lung_mask_path:
                lung_mask_name = base_name + '_lung_mask_eroded.nii.gz'
                lung_mask_final_path = os.path.join(output_dir, lung_mask_name)
                os.rename(lung_mask_path, lung_mask_final_path)
                print(f"  ✓ {lung_mask_name}")
                
            # Seed regions
            if artery_seed_path:
                artery_seed_name = base_name + '_seed_artery.nii.gz'
                artery_seed_final_path = os.path.join(output_dir, artery_seed_name)
                os.rename(artery_seed_path, artery_seed_final_path)
                print(f"  ✓ {artery_seed_name}")

            if vein_seed_path:
                vein_seed_name = base_name + '_seed_vein.nii.gz'
                vein_seed_final_path = os.path.join(output_dir, vein_seed_name)
                os.rename(vein_seed_path, vein_seed_final_path)
                print(f"  ✓ {vein_seed_name}")'''
        
if __name__ == "__main__":
    main()