import os
import SimpleITK as sitk
import numpy as np
from totalsegmentator.python_api import totalsegmentator

def run_vessel_segmentation(input_nifti_path, output_dir, fast=False):
    os.makedirs(output_dir, exist_ok=True)

    totalsegmentator(input_nifti_path, output_dir, task='lung_vessels', fast=fast)

    pulmonary_vein_path = os.path.join(output_dir, "pulmonary_vein.nii.gz")
    pulmonary_artery_path = os.path.join(output_dir, "pulmonary_artery.nii.gz")

    if not (os.path.exists(pulmonary_vein_path) and os.path.exists(pulmonary_artery_path)):
        raise RuntimeError("Missing pulmonary_vein / pulmonary_artery outputs from TotalSegmentator.")

    vein_img = sitk.ReadImage(pulmonary_vein_path)
    art_img = sitk.ReadImage(pulmonary_artery_path)

    vein = sitk.GetArrayFromImage(vein_img) > 0
    art = sitk.GetArrayFromImage(art_img) > 0

    combined = np.logical_or(vein, art).astype(np.uint8)
    out_img = sitk.GetImageFromArray(combined)
    out_img.CopyInformation(vein_img)

    out_path = os.path.join(output_dir, "vessels_combined.nii.gz")
    sitk.WriteImage(out_img, out_path)
    return out_path
