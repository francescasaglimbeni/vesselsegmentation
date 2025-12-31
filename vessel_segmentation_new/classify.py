import os
import numpy as np
import SimpleITK as sitk
import networkx as nx
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


class PathBasedVesselClassifier:
    """
    Classifica arterie/vene seguendo lo skeleton lungo i percorsi (path-based)
    invece di usare region growing spaziale.
    """

    def __init__(self, vessel_mask, skeleton, seed_artery, seed_vein, spacing_zyx):
        """
        Args:
            vessel_mask: bool array (z,y,x)
            skeleton: bool array (z,y,x)
            seed_artery: bool array (z,y,x)
            seed_vein: bool array (z,y,x)
            spacing_zyx: tuple/list (z,y,x) in mm
        """
        self.vessel_mask = vessel_mask.astype(bool)
        self.skeleton = skeleton.astype(bool)
        self.seed_artery = seed_artery.astype(bool)
        self.seed_vein = seed_vein.astype(bool)
        self.spacing = np.asarray(spacing_zyx, dtype=float)
        self.shape = self.vessel_mask.shape

        # 26-neighborhood in 3D
        self.neighbors_26 = [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ]

    def build_skeleton_graph(self):
        """Costruisce grafo (networkx) dallo skeleton con pesi metrici in mm."""
        print("  Building skeleton graph...")

        skel_coords = np.argwhere(self.skeleton)
        coord_set = set(map(tuple, skel_coords.tolist()))

        G = nx.Graph()
        for c in coord_set:
            G.add_node(c)

        for z, y, x in coord_set:
            for dz, dy, dx in self.neighbors_26:
                n = (z + dz, y + dy, x + dx)
                if n in coord_set:
                    w = float(np.linalg.norm(np.array([dz, dy, dx], dtype=float) * self.spacing))
                    G.add_edge((z, y, x), n, weight=w)

        print(f"    Graph nodes: {G.number_of_nodes():,}")
        print(f"    Graph edges: {G.number_of_edges():,}")
        return G

    def find_seed_points_on_skeleton(self, seed_mask, label="seed", max_points=2000):
        """
        Trova punti seed che intersecano lo skeleton.
        Se non c'è intersezione, proietta i seed sul punto skeleton più vicino.
        """
        intersection = seed_mask & self.skeleton
        coords = np.argwhere(intersection)

        if coords.shape[0] == 0:
            print(f"    Warning: no direct intersection for {label}, projecting seeds to nearest skeleton points...")
            seed_coords = np.argwhere(seed_mask)
            skel_coords = np.argwhere(self.skeleton)

            if seed_coords.size == 0 or skel_coords.size == 0:
                return []

            # Subsample seed coords se enormi
            stride = max(1, len(seed_coords) // max_points)
            seed_sample = seed_coords[::stride]

            tree = cKDTree(skel_coords)
            dists, idxs = tree.query(seed_sample, k=1)
            # Prendi i più vicini (fino a 50)
            order = np.argsort(dists)
            idxs = idxs[order[: min(50, len(order))]]
            coords = skel_coords[idxs]

        # Unique points
        coords = np.unique(coords, axis=0)
        seed_points = [tuple(c) for c in coords.tolist()]
        print(f"    Found {len(seed_points)} {label} points on skeleton")
        return seed_points

    def _multi_source_dijkstra_min_dist(self, G, sources, cutoff_mm):
        """
        Calcola la minima distanza da qualunque source a ciascun nodo.
        Implementazione:
        - fa Dijkstra da ogni seed e tiene min; più semplice ma può costare.
        - per molte seed, puoi migliorare con un super-source, ma di solito basta.
        """
        min_dist = {}
        for s in sources:
            if s not in G:
                continue
            try:
                lengths = nx.single_source_dijkstra_path_length(G, s, cutoff=cutoff_mm, weight="weight")
            except Exception:
                continue
            for node, dist in lengths.items():
                prev = min_dist.get(node, np.inf)
                if dist < prev:
                    min_dist[node] = dist
        return min_dist

    def classify_by_shortest_paths(self, G, artery_seeds, vein_seeds, max_path_length_mm=400.0):
        """
        Classifica nodi skeleton in base alla distanza di percorso più piccola
        rispetto a seed arterie e seed vene.
        """
        print("  Classifying skeleton by shortest paths...")

        print("    Computing distances from artery seeds...")
        artery_min = self._multi_source_dijkstra_min_dist(G, artery_seeds, cutoff_mm=max_path_length_mm)
        print(f"      Reached {len(artery_min):,} nodes from artery seeds")

        print("    Computing distances from vein seeds...")
        vein_min = self._multi_source_dijkstra_min_dist(G, vein_seeds, cutoff_mm=max_path_length_mm)
        print(f"      Reached {len(vein_min):,} nodes from vein seeds")

        artery_nodes = set()
        vein_nodes = set()

        for node in G.nodes():
            da = artery_min.get(node, np.inf)
            dv = vein_min.get(node, np.inf)

            if da < dv and da < max_path_length_mm:
                artery_nodes.add(node)
            elif dv < da and dv < max_path_length_mm:
                vein_nodes.add(node)

        print(f"    Classified {len(artery_nodes):,} artery skeleton nodes")
        print(f"    Classified {len(vein_nodes):,} vein skeleton nodes")
        return artery_nodes, vein_nodes

    def expand_classification_to_vessels(self, artery_nodes, vein_nodes, tie_break="vein"):
        """
        Espande la classificazione dallo skeleton a tutti i voxel vasi,
        assegnando ciascun voxel al tipo di skeleton più vicino (EDT).
        tie_break: "vein" o "artery" per gestire i tie dist_to_artery == dist_to_vein
        """
        print("  Expanding classification to full vessel mask...")

        artery_skel = np.zeros(self.shape, dtype=bool)
        vein_skel = np.zeros(self.shape, dtype=bool)

        for n in artery_nodes:
            artery_skel[n] = True
        for n in vein_nodes:
            vein_skel[n] = True

        print(f"    Artery skeleton voxels: {artery_skel.sum():,}")
        print(f"    Vein skeleton voxels: {vein_skel.sum():,}")

        if artery_skel.sum() > 0:
            dist_to_artery = distance_transform_edt(~artery_skel, sampling=self.spacing)
        else:
            dist_to_artery = np.full(self.shape, np.inf, dtype=float)

        if vein_skel.sum() > 0:
            dist_to_vein = distance_transform_edt(~vein_skel, sampling=self.spacing)
        else:
            dist_to_vein = np.full(self.shape, np.inf, dtype=float)

        artery_mask = self.vessel_mask & (dist_to_artery < dist_to_vein)
        vein_mask = self.vessel_mask & (dist_to_vein < dist_to_artery)

        # Tie handling
        tie = self.vessel_mask & (dist_to_artery == dist_to_vein)
        if tie.any():
            if tie_break == "artery":
                artery_mask |= tie
            else:
                vein_mask |= tie

        classified = artery_mask | vein_mask
        unclassified = self.vessel_mask & ~classified

        print(f"    Artery mask: {artery_mask.sum():,} voxels")
        print(f"    Vein mask: {vein_mask.sum():,} voxels")
        print(f"    Unclassified: {unclassified.sum():,} voxels ({(unclassified.sum()/max(1,self.vessel_mask.sum()))*100:.2f}%)")

        return artery_mask, vein_mask, unclassified

    def run_classification(self, max_path_length_mm=400.0, tie_break="vein"):
        """Pipeline completa path-based."""
        print("\n=== PATH-BASED VESSEL CLASSIFICATION ===")

        G = self.build_skeleton_graph()
        if G.number_of_nodes() == 0:
            print("ERROR: Empty skeleton graph")
            return None, None, None

        artery_seeds = self.find_seed_points_on_skeleton(self.seed_artery, "artery")
        vein_seeds = self.find_seed_points_on_skeleton(self.seed_vein, "vein")

        if not artery_seeds or not vein_seeds:
            print("ERROR: No seed points on skeleton (both required).")
            return None, None, None

        artery_nodes, vein_nodes = self.classify_by_shortest_paths(
            G, artery_seeds, vein_seeds, max_path_length_mm=max_path_length_mm
        )

        if (not artery_nodes) and (not vein_nodes):
            print("ERROR: No nodes classified.")
            return None, None, None

        artery_mask, vein_mask, unclassified = self.expand_classification_to_vessels(
            artery_nodes, vein_nodes, tie_break=tie_break
        )

        return artery_mask, vein_mask, unclassified


def classify_vessels_pathbased(
    vessel_path,
    seed_artery_path,
    seed_vein_path,
    output_dir,
    max_path_length_mm=400.0,
    tie_break="vein",
    skeleton_path=None,
):
    """
    Helper: carica maschere, skeletonizza 3D e lancia PathBasedVesselClassifier.
    Salva arteries/veins/unclassified + skeleton (opzionale).
    """
    os.makedirs(output_dir, exist_ok=True)

    vessel_img = sitk.ReadImage(vessel_path)
    vessel_mask = sitk.GetArrayFromImage(vessel_img).astype(bool)

    spacing_xyz = vessel_img.GetSpacing()          # (x,y,z)
    spacing_zyx = spacing_xyz[::-1]                # (z,y,x) per array numpy
    print(f"Spacing (xyz): {spacing_xyz} mm | using (zyx)={spacing_zyx} for numpy ops")

    seed_artery = sitk.GetArrayFromImage(sitk.ReadImage(seed_artery_path)).astype(bool)
    seed_vein = sitk.GetArrayFromImage(sitk.ReadImage(seed_vein_path)).astype(bool)

    # Skeleton 3D (skimage)
    from skimage.morphology import skeletonize_3d
    print("Computing 3D skeleton...")
    skeleton = skeletonize_3d(vessel_mask.astype(np.uint8)).astype(bool)
    print(f"  Skeleton voxels: {skeleton.sum():,}")

    if skeleton_path:
        skel_img = sitk.GetImageFromArray(skeleton.astype(np.uint8))
        skel_img.CopyInformation(vessel_img)
        sitk.WriteImage(skel_img, skeleton_path)

    classifier = PathBasedVesselClassifier(
        vessel_mask=vessel_mask,
        skeleton=skeleton,
        seed_artery=seed_artery,
        seed_vein=seed_vein,
        spacing_zyx=spacing_zyx
    )

    artery_mask, vein_mask, unclassified = classifier.run_classification(
        max_path_length_mm=max_path_length_mm,
        tie_break=tie_break
    )

    if artery_mask is None:
        raise RuntimeError("Path-based classification failed.")

    # Save outputs
    def _save(mask, name):
        img = sitk.GetImageFromArray(mask.astype(np.uint8))
        img.CopyInformation(vessel_img)
        out = os.path.join(output_dir, name)
        sitk.WriteImage(img, out)
        return out

    artery_path = _save(artery_mask, "arteries.nii.gz")
    vein_path = _save(vein_mask, "veins.nii.gz")
    unclass_path = _save(unclassified, "unclassified.nii.gz")

    print(f"✓ Saved arteries: {artery_path}")
    print(f"✓ Saved veins: {vein_path}")
    print(f"✓ Saved unclassified: {unclass_path}")

    return artery_path, vein_path, unclass_path
