
import numpy as np

def compute_adaptive_thresholds(scale_map_mm, sigma_min_mm, sigma_max_mm, tmin=0.07, tmax=0.17):
    """
    Map per-voxel vesselness threshold using scale (σ_max) so that small-scale responses use LOWER thresholds.
    Linear interpolation between tmin (smallest sigma) and tmax (largest sigma).
    """
    s = np.clip(scale_map_mm, sigma_min_mm, sigma_max_mm)
    th = tmin + (tmax - tmin) * (s - sigma_min_mm) / max(1e-9, (sigma_max_mm - sigma_min_mm))
    return th.astype(np.float32)
