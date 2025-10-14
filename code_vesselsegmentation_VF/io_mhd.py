
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from .utils import voxel_spacing_from_sitk, make_affine_from_sitk

def read_mhd(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing, origin, direction = voxel_spacing_from_sitk(img)
    return arr.astype(np.int16), spacing, origin, direction

def write_nii(mask, spacing, origin, direction, out_path):
    # mask is z,y,x; build affine and save uint8
    A = make_affine_from_sitk(spacing, origin, direction)
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine=A)
    nib.save(nii, out_path)
