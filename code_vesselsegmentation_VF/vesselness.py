
import numpy as np
from scipy import ndimage as ndi

def hessian_eigen(img, sigma_mm, spacing):
    # Gaussian second derivatives with anisotropic spacing normalization (Lindeberg).
    # Convert sigma in mm to per-axis sigma in voxels.
    sigma_vox = np.array(sigma_mm) / np.array(spacing)
    # scale normalization: multiply derivatives by sigma^2
    Dxx = ndi.gaussian_filter(img, sigma=sigma_vox, order=(2,0,0), mode='nearest') * (sigma_mm[0]**2)
    Dyy = ndi.gaussian_filter(img, sigma=sigma_vox, order=(0,2,0), mode='nearest') * (sigma_mm[1]**2)
    Dzz = ndi.gaussian_filter(img, sigma=sigma_vox, order=(0,0,2), mode='nearest') * (sigma_mm[2]**2)
    Dxy = ndi.gaussian_filter(img, sigma=sigma_vox, order=(1,1,0), mode='nearest') * (sigma_mm[0]*sigma_mm[1])
    Dxz = ndi.gaussian_filter(img, sigma=sigma_vox, order=(1,0,1), mode='nearest') * (sigma_mm[0]*sigma_mm[2])
    Dyz = ndi.gaussian_filter(img, sigma=sigma_vox, order=(0,1,1), mode='nearest') * (sigma_mm[1]*sigma_mm[2])
    # Compose Hessian and eigen-decompose for each voxel using analytical symmetric 3x3 eigendecomp
    # Flatten
    shp = img.shape
    H = np.zeros(shp + (3,3), dtype=float)
    H[...,0,0] = Dzz
    H[...,1,1] = Dyy
    H[...,2,2] = Dxx
    H[...,0,1] = H[...,1,0] = Dyz
    H[...,0,2] = H[...,2,0] = Dxz
    H[...,1,2] = H[...,2,1] = Dxy
    # reshape to (N,3,3)
    H2 = H.reshape(-1,3,3)
    # eigh for symmetric matrices
    w, v = np.linalg.eigh(H2)  # w shape (N,3), v shape (N,3,3) where v[i,:,j] is eigenvector j for matrix i
    # sort eigenvalues by absolute value ascending so that |λ1| <= |λ2| <= |λ3|
    idx = np.argsort(np.abs(w), axis=1)  # shape (N,3) indices per row
    # Use take_along_axis to reorder eigenvalues along last axis
    w_sorted = np.take_along_axis(w, idx, axis=1)  # shape (N,3)
    # For eigenvectors, we need to reorder the last axis (eigenvector axis)
    # prepare indices for take_along_axis with shape (N,1,3) to broadcast over the vector components
    idx_expanded = idx[:, None, :]  # shape (N,1,3)
    v_sorted = np.take_along_axis(v, idx_expanded, axis=2)  # shape (N,3,3)
    # Extract lambdas and first eigenvector
    lam1 = w_sorted[:, 0].reshape(shp)
    lam2 = w_sorted[:, 1].reshape(shp)
    lam3 = w_sorted[:, 2].reshape(shp)
    # v_sorted[..., 0] is the eigenvector corresponding to smallest |lambda|
    e1 = v_sorted[:, :, 0].reshape(shp + (3,))
    return lam1, lam2, lam3, e1

def frangi_vesselness(img_hu, spacing, sigmas_mm, alpha=0.5, beta=0.5, c=70.0):
    # Invert contrast so vessels (bright relative to lung parenchyma after neg HU) become negative λ2, λ3
    img = img_hu.astype(np.float32)
    Vmax = np.zeros_like(img, dtype=np.float32)
    Smax = np.zeros_like(img, dtype=np.float32)
    E1 = np.zeros(img.shape + (3,), dtype=np.float32)
    for s in sigmas_mm:
        sigma_vec = (s, s, s)
        lam1, lam2, lam3, e1 = hessian_eigen(img, sigma_vec, spacing)
        # enforce sign convention: for bright tubular on dark, λ2 and λ3 should be negative
        rb = np.where(lam3==0, 0, lam2/lam3)
        ra = np.where(lam2==0, 0, lam1/np.sqrt(np.abs(lam2*lam3)+1e-12))
        s2 = lam1**2 + lam2**2 + lam3**2
        # vesselness (probability-like) as in paper's modified Frangi
        exp_b = np.exp(-(rb**2)/(2*beta**2))
        exp_a = np.exp(-(ra**2)/(2*alpha**2))
        exp_s = 1.0 - np.exp(-(s2)/(2*c**2))
        v = exp_b * exp_a * exp_s
        # suppress non-tubular where lam2>0 or lam3>0
        v[(lam2>0) | (lam3>0)] = 0.0
        # keep max over scales
        upd = v > Vmax
        Vmax[upd] = v[upd]
        Smax[upd] = s
        E1[upd] = e1[upd]
    return Vmax, Smax, E1
