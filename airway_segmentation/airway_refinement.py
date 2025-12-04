import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu


class AirwayRefinementModule:

    def __init__(self, intensity_img, mask, spacing):
        self.img = intensity_img      # CT image (numpy)
        self.mask = mask              # TS segmentation (numpy binary)
        self.spacing = spacing
        self.refined = None

    def _compute_distance(self):
        return distance_transform_edt(self.mask == 0, sampling=self.spacing)

    def _compute_skeleton(self):
        return skeletonize(self.mask.astype(np.uint8))

    def _seed_region_grow(self, seed, max_dist_mm=3):
        sx, sy, sz = self.spacing
        max_dist_vox = int(max_dist_mm / min(sx, sy, sz))
        queue = [seed]
        visited = set([tuple(seed)])
        grown = []

        while queue:
            p = queue.pop()
            grown.append(p)
            z, y, x = p

            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nz, ny, nx = z+dz, y+dy, x+dx
                        if (nz, ny, nx) in visited: continue
                        if not (0 <= nz < self.img.shape[0] and
                                0 <= ny < self.img.shape[1] and
                                0 <= nx < self.img.shape[2]):
                            continue

                        # Check voxel
                        if self.img[nz, ny, nx] < -900:       # Air HU
                            if np.linalg.norm([nz-seed[0], ny-seed[1], nx-seed[2]]) < max_dist_vox:
                                queue.append((nz, ny, nx))
                                visited.add((nz, ny, nx))
        return grown

    def refine(self):
        dt = self._compute_distance()
        skel = self._compute_skeleton()

        refined = self.mask.copy()
        endpoints = np.argwhere(skel)
        
        # 1) region growing at endpoints
        for p in endpoints:
            z, y, x = p
            if np.sum(refined[z-1:z+2, y-1:y+2, x-1:x+2]) < 3:  # tip detection
                grown = self._seed_region_grow((z, y, x))
                for gz, gy, gx in grown:
                    refined[gz, gy, gx] = 1

        # 2) DT-based expansion
        air = self.img < -950
        candidates = (dt < 2.5) & air
        refined = np.logical_or(refined, candidates)

        # 3) small smoothing
        refined = binary_dilation(refined, iterations=1)

        self.refined = refined.astype(np.uint8)
        return self.refined

    def save(self, path, ref_img):
        out = sitk.GetImageFromArray(self.refined)
        out.CopyInformation(ref_img)
        sitk.WriteImage(out, path)
        return path
