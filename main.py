from code_vesselsegmentation.preprocessing import convert_mhd_to_nifti
from totalsegmentator.python_api import totalsegmentator
from code_vesselsegmentation.postprocessing import process_vessel_segmentation
import os
import tempfile

def main():
    # Paths
    mhd_path = 'CARVE14/1.2.840.113704.1.111.208.1137518216.7.mhd'
    output_dir = 'vessels_cleaned'

    # Step 1: Convert to NIfTI (using temporary directory)
    with tempfile.TemporaryDirectory() as nifti_dir:
        nifti_path = convert_mhd_to_nifti(mhd_path, nifti_dir)

        # Step 2a: Run TotalSegmentator for lung vessels
        totalsegmentator(nifti_path, nifti_dir, task='lung_vessels', fast=False)

        # Step 2b: Run TotalSegmentator for total anatomy
        totalsegmentator(nifti_path, nifti_dir, task='total', fast=False)

        # Step 3: Post-process and clean segmentation
        cleaned_path, centerlines_path = process_vessel_segmentation(
            nifti_dir, 
            output_dir, 
            nifti_path, 
            extract_skeleton=True
        )

        if cleaned_path is None:
            return

        # Step 4: Save only the final cleaned segmentation
        final_segmentation_name = os.path.splitext(os.path.basename(mhd_path))[0] + '_cleaned.mhd'
        final_segmentation_path = os.path.join(output_dir, final_segmentation_name)
        os.rename(cleaned_path, final_segmentation_path)

        print(f"✓ Saved final cleaned vessel segmentation: {final_segmentation_path}")
        if centerlines_path:
            print(f"✓ Saved centerlines: {centerlines_path}")

if __name__ == "__main__":
    main()
