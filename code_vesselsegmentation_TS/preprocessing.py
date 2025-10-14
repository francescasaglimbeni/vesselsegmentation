import os
import SimpleITK as sitk

def convert_mhd_to_nifti(mhd_path, output_dir):
    image = sitk.ReadImage(mhd_path)
    base_name = os.path.splitext(os.path.basename(mhd_path))[0]
    nifti_path = os.path.join(output_dir, f"{base_name}.nii.gz")
    os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(image, nifti_path)
    return nifti_path
