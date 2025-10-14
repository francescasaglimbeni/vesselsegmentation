
import numpy as np
from scipy import ndimage as ndi

def voxel_spacing_from_sitk(img):
    spacing = tuple(float(s) for s in img.GetSpacing())  # (x, y, z) in mm
    origin = tuple(float(o) for o in img.GetOrigin())
    direction = tuple(float(d) for d in img.GetDirection())
    return spacing, origin, direction

def make_affine_from_sitk(spacing, origin, direction):
    # Build a 4x4 affine from ITK style origin/spacing/direction
    # direction is 3x3 row-major flattened
    R = np.array(direction, dtype=float).reshape(3,3)
    S = np.diag(spacing)
    A = np.eye(4, dtype=float)
    A[:3,:3] = R @ S
    A[:3, 3] = np.array(origin, dtype=float)
    return A

def spherical_structuring_element(radius_vox):
    r = int(max(1, round(radius_vox)))
    grid = np.indices((2*r+1, 2*r+1, 2*r+1)) - r
    dist = np.sqrt((grid**2).sum(0))
    return (dist <= r).astype(bool)

def cylindrical_mask(shape, center, axis, radius_vox, half_len_vox):
    """Create a binary mask of a finite cylinder in voxel space.
    axis: unit vector in voxel coordinates; center: (z,y,x) center in voxel indices.
    """
    zyx = np.indices(shape)
    vec = np.stack([zyx[0]-center[0], zyx[1]-center[1], zyx[2]-center[2]], axis=0).astype(float)
    axis = np.asarray(axis, dtype=float)
    axis /= (np.linalg.norm(axis)+1e-9)
    # projection length along axis
    proj = vec[0]*axis[0] + vec[1]*axis[1] + vec[2]*axis[2]
    radial2 = (vec[0]-proj*axis[0])**2 + (vec[1]-proj*axis[1])**2 + (vec[2]-proj*axis[2])**2
    return (np.abs(proj) <= half_len_vox) & (radial2 <= radius_vox**2)

def normalize_vector_field(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    return v / n

def mm_to_vox(mm, spacing):
    return np.array(mm) / np.array(spacing)

def iso_ball_radius_vox(radius_mm, spacing):
    # approximate isotropic radius in voxels (use min spacing)
    return float(radius_mm / min(spacing))

def label_connected(mask, connectivity=1):
    return ndi.label(mask, structure=ndi.generate_binary_structure(3, connectivity))

def binary_fill_holes3d(mask):
    return ndi.binary_fill_holes(mask)

def apply_mask(img, mask, outside=0):
    out = img.copy()
    out[~mask] = outside
    return out
