# vessel_analysis.py
import numpy as np
import networkx as nx
from scipy import ndimage
import pandas as pd

# 26-neighborhood
_NEI = [(dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)]


def build_skeleton_graph(skeleton: np.ndarray, spacing_zyx):
    """
    Graph nodes are skeleton voxels (z,y,x). Edge weights in mm.
    """
    spacing = np.asarray(spacing_zyx, dtype=float)
    coords = np.argwhere(skeleton)
    coord_set = set(map(tuple, coords.tolist()))

    G = nx.Graph()
    for c in coord_set:
        G.add_node(c)

    for z, y, x in coord_set:
        for dz, dy, dx in _NEI:
            n = (z + dz, y + dy, x + dx)
            if n in coord_set:
                w = float(np.linalg.norm(np.array([dz, dy, dx], dtype=float) * spacing))
                G.add_edge((z, y, x), n, weight=w)
    return G


def compute_radius_map_mm(vessel_mask: np.ndarray, spacing_zyx):
    """
    Radius in mm for each voxel inside vessel mask, using EDT.
    """
    rad = ndimage.distance_transform_edt(vessel_mask.astype(bool), sampling=spacing_zyx)
    return rad.astype(np.float32)


def choose_root_from_seed(skeleton: np.ndarray, seed_mask: np.ndarray):
    """
    Choose root node on skeleton closest to any seed voxel.
    Returns tuple (z,y,x) or None.
    """
    if seed_mask is None or seed_mask.sum() == 0:
        return None
    skel_coords = np.argwhere(skeleton)
    if skel_coords.size == 0:
        return None

    seed_coords = np.argwhere(seed_mask)

    _, inds = ndimage.distance_transform_edt(~skeleton, return_indices=True)

    seed_coords = seed_coords[::max(1, len(seed_coords)//5000)]
    proj = np.stack([
        inds[0][seed_coords[:, 0], seed_coords[:, 1], seed_coords[:, 2]],
        inds[1][seed_coords[:, 0], seed_coords[:, 1], seed_coords[:, 2]],
        inds[2][seed_coords[:, 0], seed_coords[:, 1], seed_coords[:, 2]],
    ], axis=1)
    proj = np.unique(proj, axis=0)

    root = tuple(proj[0].tolist())
    return root


def degree_stats(G: nx.Graph):
    deg = np.array([d for _, d in G.degree()], dtype=int) if G.number_of_nodes() > 0 else np.array([], dtype=int)
    endpoints = int((deg == 1).sum())
    bifurcations = int((deg >= 3).sum())
    return endpoints, bifurcations
