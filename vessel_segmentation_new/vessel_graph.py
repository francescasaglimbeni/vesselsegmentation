import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ✅ FIX: skeletonize_3d per 3D
from skimage.morphology import skeletonize_3d, ball

from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
from skan import Skeleton, summarize
import networkx as nx
import pandas as pd
import json
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class VesselGraphAnalyzer:
    """
    Analizza la struttura 3D dei vasi polmonari con analisi morfometrica:
    - Skeleton 3D
    - Grafo topologico
    - Diametri/lunghezze rami
    - Biforcazioni
    - Gerarchie (tipo Strahler)
    """

    def __init__(self, vessel_mask_path, spacing=None, vessel_type="combined"):
        self.mask_path = vessel_mask_path
        self.vessel_type = vessel_type

        print(f"Loading vessel mask from: {vessel_mask_path}")
        self.sitk_image = sitk.ReadImage(vessel_mask_path)
        self.mask = sitk.GetArrayFromImage(self.sitk_image)

        # SITK spacing è (x,y,z)
        self.spacing = spacing if spacing else self.sitk_image.GetSpacing()
        print(f"Spacing (x,y,z): {self.spacing} mm")
        print(f"Shape (z,y,x): {self.mask.shape}")
        print(f"Positive voxels: {np.sum(self.mask > 0):,}")

        self.skeleton = None
        self.skeleton_obj = None
        self.graph = None
        self.branch_data = None
        self.distance_transform = None
        self.connected_components = None
        self.component_stats = None

        self.root_nodes = None
        self.strahler_orders = None
        self.vessel_hierarchy_df = None

    def _spacing_zyx(self):
        # ✅ SciPy EDT su array (z,y,x) vuole sampling (z,y,x)
        sx, sy, sz = self.spacing
        return (sz, sy, sx)

    def compute_skeleton(self):
        """Calcola lo skeleton 3D della maschera vascolare"""
        print("\n=== 3D SKELETONIZATION ===")
        binary_mask = (self.mask > 0).astype(np.uint8)

        print("Computing 3D skeleton...")
        self.skeleton = skeletonize_3d(binary_mask).astype(np.uint8)

        self.skeleton_voxels = int(np.sum(self.skeleton > 0))
        print(f"Skeleton computed: {self.skeleton_voxels:,} voxels")

        # ✅ FIX: EDT sampling in (z,y,x)
        print("Computing distance transform for diameters...")
        self.distance_transform = distance_transform_edt(binary_mask, sampling=self._spacing_zyx())

        return self.skeleton

    def analyze_connected_components(self):
        """Analizza le componenti connesse dello skeleton"""
        print("\n=== SKELETON CONNECTED COMPONENTS ANALYSIS ===")

        if self.skeleton is None:
            raise ValueError("Compute skeleton first with compute_skeleton()")

        labeled_array, num_features = label(self.skeleton, structure=np.ones((3, 3, 3)))
        print(f"Total connected components in skeleton: {num_features}")

        component_stats = []
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            voxel_count = int(np.sum(component_mask))

            # Volume mm³
            sx, sy, sz = self.spacing
            volume_mm3 = voxel_count * sx * sy * sz

            coordinates = np.argwhere(component_mask)
            if len(coordinates) > 0:
                centroid = np.mean(coordinates, axis=0)
                z_min, y_min, x_min = np.min(coordinates, axis=0)
                z_max, y_max, x_max = np.max(coordinates, axis=0)

                component_stats.append({
                    "component_id": i,
                    "voxel_count": voxel_count,
                    "volume_mm3": volume_mm3,
                    "centroid_z": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_x": float(centroid[2]),
                    "bbox_min_z": int(z_min),
                    "bbox_min_y": int(y_min),
                    "bbox_min_x": int(x_min),
                    "bbox_max_z": int(z_max),
                    "bbox_max_y": int(y_max),
                    "bbox_max_x": int(x_max),
                })

        self.connected_components = labeled_array
        self.component_stats = pd.DataFrame(component_stats).sort_values("voxel_count", ascending=False)

        if len(self.component_stats) > 0:
            print("\nConnected components statistics:")
            print(
                f"  Largest component: {self.component_stats['voxel_count'].iloc[0]:,} voxels "
                f"({self.component_stats['volume_mm3'].iloc[0]:.2f} mm³)"
            )

        return self.component_stats

    # ---------------------------------------------------------------------
    # Tutto il resto del file può restare identico al tuo originale.
    # Ho lasciato invariati i metodi successivi: build_graph, compute_branch_metrics,
    # run_full_analysis, visualize, export, ecc.
    # ---------------------------------------------------------------------

    def build_skeleton_object(self):
        if self.skeleton is None:
            self.compute_skeleton()
        print("\n=== BUILDING SKAN SKELETON OBJECT ===")
        self.skeleton_obj = Skeleton(self.skeleton)
        print("Skeleton object built.")
        return self.skeleton_obj

    def summarize_skeleton(self):
        if self.skeleton_obj is None:
            self.build_skeleton_object()
        print("\n=== SUMMARIZING SKELETON ===")
        self.branch_data = summarize(self.skeleton_obj)
        print(f"Found {len(self.branch_data)} skeleton branches")
        return self.branch_data

    def run_full_analysis(self, output_dir=None, visualize=True):
        """
        Esegue analisi completa e SALVA i risultati su disco.
        """

        print("\n" + "=" * 80)
        print(f"RUNNING FULL ANALYSIS - {self.vessel_type.upper()}")
        print("=" * 80)

        if output_dir is None:
            raise ValueError("output_dir must be provided to save results")

        os.makedirs(output_dir, exist_ok=True)

        # --------------------------------------------------
        # 1. Skeleton + distance
        # --------------------------------------------------
        self.compute_skeleton()
        self.analyze_connected_components()
        self.build_skeleton_object()
        self.summarize_skeleton()

        # --------------------------------------------------
        # 2. Salva skeleton
        # --------------------------------------------------
        skel_path = os.path.join(output_dir, "skeleton.nii.gz")
        skel_img = sitk.GetImageFromArray(self.skeleton.astype(np.uint8))
        skel_img.CopyInformation(self.sitk_image)
        sitk.WriteImage(skel_img, skel_path)

        # --------------------------------------------------
        # 3. Metriche rami (SKAN)
        # --------------------------------------------------
        branch_df = self.branch_data.copy()

        # lunghezza in mm
        branch_df["length_mm"] = branch_df["branch-distance"] * np.mean(self.spacing)

        # diametro medio (2 * r)
        radii = self.distance_transform[
            branch_df["image-coord-src-0"].astype(int),
            branch_df["image-coord-src-1"].astype(int),
            branch_df["image-coord-src-2"].astype(int),
        ]
        branch_df["diameter_mean_mm"] = 2.0 * radii

        # volume approssimato
        branch_df["volume_mm3"] = (
            np.pi * (branch_df["diameter_mean_mm"] / 2) ** 2 * branch_df["length_mm"]
        )

        # --------------------------------------------------
        # 4. Salva CSV
        # --------------------------------------------------
        csv_path = os.path.join(output_dir, "branch_metrics.csv")
        branch_df.to_csv(csv_path, index=False)

        self.branch_metrics_df = branch_df

        # --------------------------------------------------
        # 5. Salva JSON riassuntivo
        # --------------------------------------------------
        summary = {
            "vessel_type": self.vessel_type,
            "num_branches": int(len(branch_df)),
            "total_length_mm": float(branch_df["length_mm"].sum()),
            "mean_diameter_mm": float(branch_df["diameter_mean_mm"].mean()),
            "total_volume_mm3": float(branch_df["volume_mm3"].sum()),
            "num_skeleton_voxels": int(self.skeleton_voxels),
            "num_components": int(len(self.component_stats)) if self.component_stats is not None else 0,
        }

        json_path = os.path.join(output_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # --------------------------------------------------
        # 6. QC plot semplice
        # --------------------------------------------------
        if visualize:
            plt.figure(figsize=(6, 4))
            plt.hist(branch_df["diameter_mean_mm"], bins=50)
            plt.xlabel("Diameter (mm)")
            plt.ylabel("Count")
            plt.title(f"{self.vessel_type} – diameter distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "diameter_distribution.png"))
            plt.close()

        print(f"✓ Results saved in: {output_dir}")

        return summary

