'''from code_vesselsegmentation.preprocessing import convert_mhd_to_nifti
from totalsegmentator.python_api import totalsegmentator

def main():
    # Percorso al file MHD
    mhd_path = 'VESSEL12/scans/VESSEL12_01.mhd'
    nifti_path = 'nifti_scans'
    # Cartella temporanea per il file NIfTI
    nifti_path = convert_mhd_to_nifti(mhd_path, nifti_path)   
    # Cartella di output per le segmentazioni
    output_path = 'vessels_segmentations'
        
    # Esegui la segmentazione
    totalsegmentator(nifti_path, output_path, task='lung_vessels')

if __name__ == "__main__":
    main()
'''
from code_vesselsegmentation.preprocessing import convert_mhd_to_nifti
from totalsegmentator.python_api import totalsegmentator
from code_vesselsegmentation.postprocessing import process_vessel_segmentation
import os
def main():
    # Paths
    mhd_path = 'VESSEL12/scans/VESSEL12_01.mhd'
    nifti_dir = 'nifti_scans'
    seg_dir = 'vessels_segmentations'
    output_dir = 'vessels_cleaned'
    
    # Step 1: Convert to NIfTI
    print("=== Step 1: Converting MHD to NIfTI ===")
    nifti_path = convert_mhd_to_nifti(mhd_path, nifti_dir)
    
    # Step 2: Run TotalSegmentator
    print("\n=== Step 2: Running TotalSegmentator ===")
    # Use 'total' task to get heart structures for A/V seed extraction
    totalsegmentator(nifti_path, seg_dir, task='lung_vessels')
    print("TotalSegmentator completed!")
    
    # Verify critical structures are present
    critical_files = [
        'lung_vessels.nii.gz',
        'pulmonary_artery.nii.gz',
        'heart_atrium_left.nii.gz'
    ]
    missing = [f for f in critical_files if not os.path.exists(os.path.join(seg_dir, f))]
    if missing:
        print(f"WARNING: Missing structures for A/V classification: {missing}")
    
    # Step 3: Post-process segmentation
    print("\n=== Step 3: Post-processing ===")
    cleaned_path, centerlines_path = process_vessel_segmentation(
        seg_dir, output_dir, nifti_path, extract_skeleton=True
    )
    
    print("\n=== Pipeline Complete! ===")
    print(f"Cleaned vessels: {cleaned_path}")
    if centerlines_path:
        print(f"Centerlines: {centerlines_path}")
    print("\nNext step: Artery/Vein classification using centerlines + connectivity")


if __name__ == "__main__":
    main()