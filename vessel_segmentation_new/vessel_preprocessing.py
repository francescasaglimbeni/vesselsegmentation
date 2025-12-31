# (IL TUO FILE ORIGINALE - invariato)
# Se vuoi che lo integri direttamente nella pipeline (al posto della clean base),
# dimmelo e lo patcho.
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, generate_binary_structure, binary_closing, binary_opening, binary_dilation, binary_erosion
from scipy.ndimage import distance_transform_edt
import os

class VesselSegmentationPreprocessor:
    def __init__(self, min_component_size=500, closing_iterations=2, opening_iterations=1, bridge_gaps=True, max_gap_size=3):
        self.min_component_size = min_component_size
        self.closing_iterations = closing_iterations
        self.opening_iterations = opening_iterations
        self.bridge_gaps = bridge_gaps
        self.max_gap_size = max_gap_size

    def load_mask(self, mask_path):
        img = sitk.ReadImage(mask_path)
        arr = sitk.GetArrayFromImage(img)
        return img, arr

    def save_mask(self, arr, ref_img, out_path):
        out_img = sitk.GetImageFromArray(arr.astype(np.uint8))
        out_img.CopyInformation(ref_img)
        sitk.WriteImage(out_img, out_path)

    def remove_small_components(self, binary_mask):
        structure = np.ones((3,3,3), dtype=np.uint8)
        labeled, num = label(binary_mask, structure=structure)
        if num == 0:
            return binary_mask
        sizes = np.array([np.sum(labeled == i) for i in range(1, num+1)])
        keep = np.where(sizes >= self.min_component_size)[0] + 1
        cleaned = np.isin(labeled, keep)
        return cleaned

    def morphological_cleanup(self, mask):
        m = mask.astype(bool)
        if self.closing_iterations > 0:
            for _ in range(self.closing_iterations):
                m = binary_closing(m, structure=np.ones((3,3,3)))
        if self.opening_iterations > 0:
            for _ in range(self.opening_iterations):
                m = binary_opening(m, structure=np.ones((3,3,3)))
        return m

    def bridge_small_gaps(self, mask, spacing_zyx=None):
        # semplice bridging: dilatazione + erosione leggera
        m = mask.astype(bool)
        m2 = binary_dilation(m, iterations=self.max_gap_size)
        m2 = binary_erosion(m2, iterations=self.max_gap_size)
        return m2

    def process(self, mask_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        ref_img, arr = self.load_mask(mask_path)
        binary = (arr > 0)

        step1 = self.remove_small_components(binary)
        step2 = self.morphological_cleanup(step1)
        step3 = self.bridge_small_gaps(step2) if self.bridge_gaps else step2

        out_path = os.path.join(output_dir, "vessels_preprocessed.nii.gz")
        self.save_mask(step3.astype(np.uint8), ref_img, out_path)
        return out_path
