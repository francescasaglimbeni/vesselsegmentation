# vessel_graph.py
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from skimage.morphology import skeletonize_3d
from scipy.ndimage import distance_transform_edt, label
from skan import Skeleton, summarize
import networkx as nx
import pandas as pd
import json


class ImprovedVesselAnalyzer:
    """
    Analizzatore migliorato con focus su:
    1. Path tracking completo (trunk → endpoints)
    2. Metriche lungo i path (diametro, lunghezza cumulativa, taper)
    3. Fix per componenti disconnesse
    4. Visualizzazioni 3D complete
    5. Check Murray's law sulle biforcazioni
    """

    def __init__(self, vessel_mask_path, spacing=None, num_initial_points=5):
        self.mask_path = vessel_mask_path
        self.num_initial_points = num_initial_points

        print(f"Loading vessel mask: {vessel_mask_path}")
        self.sitk_image = sitk.ReadImage(vessel_mask_path)
        self.mask = sitk.GetArrayFromImage(self.sitk_image)

        self.spacing = spacing if spacing else self.sitk_image.GetSpacing()
        print(f"  Spacing (x,y,z): {self.spacing} mm")
        print(f"  Shape (z,y,x): {self.mask.shape}")
        print(f"  Positive voxels: {np.sum(self.mask > 0):,}")

        self.skeleton = None
        self.distance_transform = None
        self.graph = None
        self.branch_df = None
        self.initial_points = None
        self.paths_df = None
        self.pathology_report = None

        # Murray's law
        self.murray_violations_df = None

    def _spacing_zyx(self):
        sx, sy, sz = self.spacing
        return (sz, sy, sx)

    def compute_skeleton_and_distances(self):
        """Calcola skeleton 3D e distance transform"""
        print("\n=== Computing Skeleton & Distance Transform ===")

        binary_mask = (self.mask > 0).astype(np.uint8)

        print("  Skeletonizing...")
        self.skeleton = skeletonize_3d(binary_mask).astype(np.uint8)
        skel_voxels = np.sum(self.skeleton > 0)
        print(f"    Skeleton: {skel_voxels:,} voxels")

        print("  Computing EDT for precise radii...")
        self.distance_transform = distance_transform_edt(
            binary_mask, sampling=self._spacing_zyx()
        )
        print(f"    Max radius: {self.distance_transform.max():.2f} mm")

        return self.skeleton, self.distance_transform

    def build_graph_with_properties(self):
        """Costruisce grafo con metriche dettagliate"""
        print("\n=== Building Detailed Graph ===")

        if self.skeleton is None:
            self.compute_skeleton_and_distances()

        skel_obj = Skeleton(self.skeleton)
        branch_data = summarize(skel_obj, separator='-')

        self.graph = nx.Graph()

        spacing_avg = np.mean(self.spacing)

        for idx, row in branch_data.iterrows():
            src = (
                int(row['image-coord-src-0']),
                int(row['image-coord-src-1']),
                int(row['image-coord-src-2'])
            )
            dst = (
                int(row['image-coord-dst-0']),
                int(row['image-coord-dst-1']),
                int(row['image-coord-dst-2'])
            )

            length_mm = row['branch-distance'] * spacing_avg

            # IMPROVED: Sample più punti lungo il ramo per diametro accurato
            coords_along_branch = self._sample_branch_coords(src, dst, num_points=20)
            diameters_along = []

            for coord in coords_along_branch:
                z, y, x = coord
                if 0 <= z < self.distance_transform.shape[0] and \
                   0 <= y < self.distance_transform.shape[1] and \
                   0 <= x < self.distance_transform.shape[2]:
                    radius = self.distance_transform[z, y, x]
                    diameters_along.append(2.0 * radius)

            if diameters_along:
                diameter_mean_mm = float(np.mean(diameters_along))
                diameter_std_mm = float(np.std(diameters_along))
                diameter_min_mm = float(np.min(diameters_along))
                diameter_max_mm = float(np.max(diameters_along))
            else:
                r_src = float(self.distance_transform[src])
                r_dst = float(self.distance_transform[dst])
                diameter_mean_mm = float(2.0 * (r_src + r_dst) / 2.0)
                diameter_std_mm = 0.0
                diameter_min_mm = float(2.0 * min(r_src, r_dst))
                diameter_max_mm = float(2.0 * max(r_src, r_dst))

            radius_mean_mm = diameter_mean_mm / 2.0
            volume_mm3 = float(np.pi * (radius_mean_mm ** 2) * length_mm)

            self.graph.add_edge(src, dst,
                                length_mm=float(length_mm),
                                diameter_mm=float(diameter_mean_mm),
                                diameter_std_mm=float(diameter_std_mm),
                                diameter_min_mm=float(diameter_min_mm),
                                diameter_max_mm=float(diameter_max_mm),
                                volume_mm3=float(volume_mm3),
                                branch_id=int(idx))

        print(f"    Graph: {self.graph.number_of_nodes():,} nodes, "
              f"{self.graph.number_of_edges():,} edges")

        # Analizza componenti connesse
        components = list(nx.connected_components(self.graph))
        print(f"    Connected components: {len(components)}")
        if len(components) > 1:
            sizes = [len(c) for c in components]
            print(f"    Component sizes: largest={max(sizes)}, smallest={min(sizes)}")

        self.branch_df = self._extract_branch_dataframe()

        return self.graph

    def _sample_branch_coords(self, src, dst, num_points=20):
        """
        IMPROVED: Campiona più punti lungo il ramo per diametro accurato
        """
        z1, y1, x1 = src
        z2, y2, x2 = dst

        t = np.linspace(0, 1, num_points)

        z_interp = np.round(z1 + t * (z2 - z1)).astype(int)
        y_interp = np.round(y1 + t * (y2 - y1)).astype(int)
        x_interp = np.round(x1 + t * (x2 - x1)).astype(int)

        coords = list(zip(z_interp, y_interp, x_interp))
        return coords

    def _extract_branch_dataframe(self):
        """Estrae metriche branch in DataFrame"""
        branches = []
        for u, v, data in self.graph.edges(data=True):
            branches.append({
                'branch_id': data.get('branch_id', -1),
                'node_src': u,
                'node_dst': v,
                'length_mm': data['length_mm'],
                'diameter_mm': data['diameter_mm'],
                'diameter_std_mm': data.get('diameter_std_mm', 0.0),
                'diameter_min_mm': data.get('diameter_min_mm', 0.0),
                'diameter_max_mm': data.get('diameter_max_mm', 0.0),
                'volume_mm3': data['volume_mm3']
            })

        df = pd.DataFrame(branches)
        return df

    def identify_initial_points(self):
        """
        IMPROVED: Identifica trunks considerando tutte le componenti
        """
        print(f"\n=== Identifying {self.num_initial_points} Initial Points ===")

        if self.graph is None:
            self.build_graph_with_properties()

        if self.graph.number_of_nodes() == 0:
            print("  ERROR: Empty graph")
            return []

        # Per ogni componente connessa, trova il miglior trunk
        components = list(nx.connected_components(self.graph))
        print(f"    Processing {len(components)} connected components")

        all_candidates = []

        for comp_idx, comp_nodes in enumerate(components):
            comp_graph = self.graph.subgraph(comp_nodes)

            # Trova nodo con max(degree * diameter)
            best_node = None
            best_score = -1

            for node in comp_nodes:
                degree = comp_graph.degree(node)
                z, y, x = node
                diameter = float(self.distance_transform[z, y, x] * 2.0)
                score = degree * diameter

                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node:
                z, y, x = best_node
                all_candidates.append({
                    'coords': best_node,
                    'degree': comp_graph.degree(best_node),
                    'diameter': float(self.distance_transform[z, y, x] * 2.0),
                    'component_size': len(comp_nodes),
                    'component_id': comp_idx,
                    'z': z, 'y': y, 'x': x
                })

        # Ordina per component_size * diameter
        all_candidates = sorted(
            all_candidates,
            key=lambda x: x['component_size'] * x['diameter'],
            reverse=True
        )

        # Seleziona top N con distribuzione spaziale
        selected = []
        min_distance = 50

        for candidate in all_candidates:
            if len(selected) >= self.num_initial_points:
                break

            too_close = False
            for sel in selected:
                dist = np.linalg.norm(
                    np.array(candidate['coords']) - np.array(sel['coords'])
                )
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                selected.append(candidate)

        self.initial_points = selected

        print(f"    Identified {len(selected)} initial points:")
        for i, pt in enumerate(selected, 1):
            print(f"      Trunk {i}: component_size={pt['component_size']}, "
                  f"diameter={pt['diameter']:.2f}mm, degree={pt['degree']}")

        return selected

    def extract_all_paths_from_trunks(self):
        """
        Estrae TUTTI i path completi da ogni trunk agli endpoints
        """
        print("\n=== Extracting All Paths from Trunks ===")

        if self.initial_points is None:
            self.identify_initial_points()

        if not self.initial_points:
            print("  ERROR: No initial points")
            return None

        all_paths = []

        for trunk_idx, trunk in enumerate(self.initial_points):
            trunk_node = trunk['coords']
            trunk_id = f"trunk_{trunk_idx + 1}"

            if trunk_node not in self.graph:
                continue

            # Trova la componente connessa di questo trunk
            component = nx.node_connected_component(self.graph, trunk_node)
            subgraph = self.graph.subgraph(component)

            # Trova tutti gli endpoints (degree == 1)
            endpoints = [n for n in subgraph.nodes() if subgraph.degree(n) == 1]

            print(f"    {trunk_id}: {len(endpoints)} endpoints in component")

            # Per ogni endpoint, calcola path dal trunk
            for endpoint in endpoints:
                if endpoint == trunk_node:
                    continue

                try:
                    path_nodes = nx.shortest_path(
                        subgraph, trunk_node, endpoint, weight='length_mm'
                    )

                    # Calcola metriche lungo il path
                    path_info = self._analyze_path(path_nodes, trunk_id)
                    all_paths.append(path_info)

                except nx.NetworkXNoPath:
                    continue

        print(f"    Total paths extracted: {len(all_paths)}")

        # Crea DataFrame paths
        self.paths_df = pd.DataFrame(all_paths)

        return self.paths_df

    def _analyze_path(self, path_nodes, trunk_id):
        """
        Analizza singolo path: calcola metriche cumulative e taper
        """
        path_length_cumulative = 0.0
        diameters = []
        lengths = []
        volumes = []

        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]

            edge_data = self.graph[u][v]
            length = edge_data['length_mm']
            diameter = edge_data['diameter_mm']
            volume = edge_data['volume_mm3']

            path_length_cumulative += length
            diameters.append(diameter)
            lengths.append(length)
            volumes.append(volume)

        taper_ratio = diameters[0] / diameters[-1] if len(diameters) > 0 and diameters[-1] > 0 else np.nan
        diameter_change = diameters[0] - diameters[-1] if len(diameters) > 0 else 0.0
        diameter_change_pct = (diameter_change / diameters[0] * 100) if len(diameters) > 0 and diameters[0] > 0 else 0.0

        return {
            'trunk_id': trunk_id,
            'num_branches': len(diameters),
            'total_length_mm': path_length_cumulative,
            'diameter_start_mm': diameters[0] if diameters else np.nan,
            'diameter_end_mm': diameters[-1] if diameters else np.nan,
            'diameter_mean_mm': np.mean(diameters) if diameters else np.nan,
            'diameter_std_mm': np.std(diameters) if diameters else np.nan,
            'taper_ratio': taper_ratio,
            'diameter_change_mm': diameter_change,
            'diameter_change_pct': diameter_change_pct,
            'total_volume_mm3': sum(volumes),
            'path_nodes': path_nodes
        }

    def compute_generation_from_trunks(self):
        """
        BFS per generazioni, gestisce componenti disconnesse
        """
        print("\n=== Computing Generations from Trunks ===")

        if self.initial_points is None:
            self.identify_initial_points()

        if not self.initial_points:
            print("  ERROR: No initial points")
            return None

        node_to_generation = {node: -1 for node in self.graph.nodes()}

        for i, trunk in enumerate(self.initial_points):
            trunk_node = trunk['coords']
            trunk_id = f'trunk_{i+1}'

            if trunk_node not in self.graph:
                continue

            print(f"    Computing generations from {trunk_id}...")

            node_to_generation[trunk_node] = 0
            queue = [trunk_node]
            visited = {trunk_node}

            while queue:
                node = queue.pop(0)
                gen = node_to_generation[node]

                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        node_to_generation[neighbor] = gen + 1
                        queue.append(neighbor)

            print(f"      Reached {len(visited)} nodes (max gen: {max(node_to_generation[n] for n in visited)})")

        for idx, row in self.branch_df.iterrows():
            u = row['node_src']
            v = row['node_dst']
            gen_u = node_to_generation.get(u, -1)
            gen_v = node_to_generation.get(v, -1)
            self.branch_df.at[idx, 'generation'] = max(gen_u, gen_v)

        unreached = sum(1 for g in node_to_generation.values() if g == -1)
        if unreached > 0:
            print(f"    WARNING: {unreached} nodes unreached (disconnected components)")

        return node_to_generation

    def analyze_diameter_progression_by_generation(self):
        """
        Analisi dettagliata progressione diametro per generazione
        """
        print("\n=== Analyzing Diameter Progression ===")

        if 'generation' not in self.branch_df.columns:
            self.compute_generation_from_trunks()

        valid_branches = self.branch_df[self.branch_df['generation'] >= 0].copy()

        if len(valid_branches) == 0:
            print("  No valid branches with generation data")
            return None

        gen_stats = valid_branches.groupby('generation').agg({
            'diameter_mm': ['mean', 'std', 'min', 'max', 'count'],
            'length_mm': ['mean', 'sum'],
            'volume_mm3': 'sum'
        }).round(2)

        print(f"    Analyzed {len(valid_branches)} branches across {int(valid_branches['generation'].max())} generations")

        gen_means = valid_branches.groupby('generation')['diameter_mm'].mean()
        if len(gen_means) > 1:
            diameter_slope = np.polyfit(gen_means.index, gen_means.values, 1)[0]
            print(f"    Diameter change rate: {diameter_slope:.4f} mm/generation")

        return gen_stats

    # ============================================================
    # MURRAY'S LAW (NUOVO)
    # ============================================================
    def check_murrays_law(self, tolerance=0.2, use_top2_children=True):
        """
        Verifica Murray's Law: d_parent^3 ≈ d_child1^3 + d_child2^3

        NOTE:
        - Il grafo è non orientato, quindi "parent" non è definito.
          Qui usiamo una proxy robusta: parent = ramo con diametro massimo al nodo.
        - Per nodi con >2 figli, di default prendiamo i 2 figli più grandi (top2).
          (Se vuoi, posso estenderlo a somma di tutti i children.)

        Args:
            tolerance: ±tolleranza (0.2 => ±20%)
            use_top2_children: se True, usa i 2 children più grandi come nel tuo snippet.
        Returns:
            pd.DataFrame con violazioni
        """
        if self.graph is None:
            self.build_graph_with_properties()

        violations = []

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))

            # Biforcazione (>=3 connessioni)
            if len(neighbors) >= 3:
                diameters = []
                for n in neighbors:
                    d = self.graph[node][n].get('diameter_mm', None)
                    if d is not None and np.isfinite(d) and d > 0:
                        diameters.append(float(d))

                if len(diameters) < 3:
                    continue

                # proxy parent
                d_parent = max(diameters)

                # children: top2 oppure "tutti tranne parent"
                sorted_d = sorted(diameters, reverse=True)
                children = sorted_d[1:]  # esclude parent

                if use_top2_children:
                    if len(children) < 2:
                        continue
                    d_children = children[:2]
                    expected = d_children[0]**3 + d_children[1]**3
                else:
                    # opzionale: somma di tutti i children
                    expected = sum([dc**3 for dc in children])

                actual = d_parent**3
                ratio = actual / expected if expected > 0 else np.nan

                if not np.isfinite(ratio):
                    continue

                if ratio < (1 - tolerance) or ratio > (1 + tolerance):
                    violations.append({
                        'node': node,
                        'degree': len(neighbors),
                        'd_parent': d_parent,
                        'expected_sum_children_cubed': expected,
                        'actual_parent_cubed': actual,
                        'ratio_actual_over_expected': ratio,
                        'diameters_neighbors': diameters
                    })

        self.murray_violations_df = pd.DataFrame(violations)
        print(f"\n=== Murray's Law Check ===")
        print(f"    Violations found: {len(self.murray_violations_df)} (tolerance ±{tolerance*100:.0f}%)")

        return self.murray_violations_df

    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    def visualize_3d_vessels(self, output_dir, color_by='diameter'):
        """
        Visualizzazione 3D
        """
        print(f"\n=== 3D Visualization (colored by {color_by}) ===")

        if self.graph is None:
            self.build_graph_with_properties()

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        segments = []
        values = []

        for u, v, data in self.graph.edges(data=True):
            z1, y1, x1 = u
            z2, y2, x2 = v
            segments.append([(x1, y1, z1), (x2, y2, z2)])

            if color_by == 'diameter':
                values.append(data['diameter_mm'])
            elif color_by == 'length':
                values.append(data['length_mm'])
            elif color_by == 'generation':
                branch_id = data.get('branch_id', -1)
                if branch_id >= 0 and branch_id < len(self.branch_df) and 'generation' in self.branch_df.columns:
                    gen_val = self.branch_df.iloc[branch_id]['generation']
                    values.append(gen_val if gen_val >= 0 else 0)
                else:
                    values.append(0)
            else:
                values.append(1.0)

        if not segments:
            print("  No segments to plot")
            return

        values = np.array(values)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        if color_by == 'generation':
            values = np.maximum(values, 0)

        vmin, vmax = np.nanmin(values), np.nanmax(values)

        if vmin == vmax:
            vmin = vmin - 0.5 if vmin > 0 else 0.0
            vmax = vmax + 0.5

        vmin = float(np.clip(vmin, -1e10, 1e10))
        vmax = float(np.clip(vmax, -1e10, 1e10))

        norm = Normalize(vmin=vmin, vmax=vmax)

        if color_by == 'diameter':
            cmap = cm.get_cmap('plasma')
        elif color_by == 'length':
            cmap = cm.get_cmap('viridis')
        elif color_by == 'generation':
            cmap = cm.get_cmap('turbo')
        else:
            cmap = cm.get_cmap('viridis')

        for seg, val in zip(segments, values):
            (x1, y1, z1), (x2, y2, z2) = seg
            color = cmap(norm(val))

            if color_by == 'diameter':
                lw = np.clip(val / 3.0, 0.5, 3.0)
            else:
                lw = 1.5

            ax.plot([x1, x2], [y1, y2], [z1, z2],
                    color=color, linewidth=lw, alpha=0.8)

        if self.initial_points:
            trunk_coords = np.array([pt['coords'] for pt in self.initial_points])
            ax.scatter(trunk_coords[:, 2], trunk_coords[:, 1], trunk_coords[:, 0],
                       c='red', s=200, marker='*', edgecolors='black', linewidths=2,
                       label='Initial Points (Trunks)', zorder=10)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)

        if color_by == 'diameter':
            cbar.set_label('Diameter (mm)', fontsize=12)
            title = 'Unified Vessels - Diameter Map'
        elif color_by == 'length':
            cbar.set_label('Branch Length (mm)', fontsize=12)
            title = 'Unified Vessels - Length Distribution'
        elif color_by == 'generation':
            cbar.set_label('Generation', fontsize=12)
            title = 'Unified Vessels - Hierarchical Generations'
        else:
            title = 'Unified Vessels'

        ax.set_xlabel('X (voxel)', fontsize=11)
        ax.set_ylabel('Y (voxel)', fontsize=11)
        ax.set_zlabel('Z (voxel)', fontsize=11)
        ax.set_title(f'{title}\n{len(segments)} branches', fontsize=14, pad=20, weight='bold')
        ax.legend()

        filename = f'vessel_3d_{color_by}.png'
        output_path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    Saved: {filename}")

    def create_path_visualizations(self, output_dir):
        """
        Visualizza path individuali con evoluzione diametro
        """
        if self.paths_df is None or len(self.paths_df) == 0:
            print("  No path data available")
            return

        print("\n=== Creating Path Visualizations ===")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        valid_taper = self.paths_df[np.isfinite(self.paths_df['taper_ratio'])]

        axes[0, 0].hist(valid_taper['taper_ratio'], bins=50, edgecolor='black', color='steelblue')
        axes[0, 0].set_xlabel('Taper Ratio (start/end diameter)', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].set_title('Path Taper Distribution', fontsize=12, weight='bold')
        axes[0, 0].axvline(valid_taper['taper_ratio'].median(), color='r',
                          linestyle='--', label=f'Median: {valid_taper["taper_ratio"].median():.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].hist(self.paths_df['diameter_change_pct'], bins=50,
                        edgecolor='black', color='seagreen')
        axes[0, 1].set_xlabel('Diameter Change (%)', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title('Diameter Reduction Along Paths', fontsize=12, weight='bold')
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].scatter(self.paths_df['total_length_mm'],
                           self.paths_df['diameter_change_mm'],
                           alpha=0.4, s=20, c='purple')
        axes[1, 0].set_xlabel('Path Length (mm)', fontsize=11)
        axes[1, 0].set_ylabel('Diameter Change (mm)', fontsize=11)
        axes[1, 0].set_title('Length vs Diameter Change', fontsize=12, weight='bold')
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].scatter(self.paths_df['num_branches'],
                           valid_taper['taper_ratio'],
                           alpha=0.4, s=20, c='crimson')
        axes[1, 1].set_xlabel('Number of Branches in Path', fontsize=11)
        axes[1, 1].set_ylabel('Taper Ratio', fontsize=11)
        axes[1, 1].set_title('Path Complexity vs Taper', fontsize=12, weight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'path_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print("    Saved: path_analysis.png")

    def create_generation_schematic(self, output_dir):
        """
        Schema hierarchical delle generazioni (diameter vs generation)
        """
        print("\n=== Creating Generation Schematic ===")

        if 'generation' not in self.branch_df.columns:
            print("  No generation data available")
            return

        valid_branches = self.branch_df[self.branch_df['generation'] >= 0].copy()

        if len(valid_branches) == 0:
            print("  No valid generation data")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        gen_stats = valid_branches.groupby('generation')['diameter_mm'].agg(['mean', 'std', 'count'])

        axes[0, 0].errorbar(gen_stats.index, gen_stats['mean'],
                            yerr=gen_stats['std'], marker='o', capsize=5,
                            linewidth=2, markersize=8, color='crimson', alpha=0.7)
        axes[0, 0].fill_between(gen_stats.index,
                                gen_stats['mean'] - gen_stats['std'],
                                gen_stats['mean'] + gen_stats['std'],
                                alpha=0.2, color='crimson')
        axes[0, 0].set_xlabel('Generation', fontsize=12)
        axes[0, 0].set_ylabel('Mean Diameter (mm)', fontsize=12)
        axes[0, 0].set_title('Diameter Progression by Generation', fontsize=13, weight='bold')
        axes[0, 0].grid(alpha=0.3)

        gen_counts = valid_branches['generation'].value_counts().sort_index()

        axes[0, 1].bar(gen_counts.index, gen_counts.values, edgecolor='black', color='steelblue', alpha=0.7)
        axes[0, 1].set_xlabel('Generation', fontsize=12)
        axes[0, 1].set_ylabel('Number of Branches', fontsize=12)
        axes[0, 1].set_title('Branch Distribution by Generation', fontsize=13, weight='bold')
        axes[0, 1].grid(alpha=0.3, axis='y')

        gen_length = valid_branches.groupby('generation')['length_mm'].sum()

        axes[1, 0].bar(gen_length.index, gen_length.values, edgecolor='black', color='seagreen', alpha=0.7)
        axes[1, 0].set_xlabel('Generation', fontsize=12)
        axes[1, 0].set_ylabel('Total Length (mm)', fontsize=12)
        axes[1, 0].set_title('Cumulative Length by Generation', fontsize=13, weight='bold')
        axes[1, 0].grid(alpha=0.3, axis='y')

        gen_volume = valid_branches.groupby('generation')['volume_mm3'].sum()

        axes[1, 1].bar(gen_volume.index, gen_volume.values, edgecolor='black', color='orange', alpha=0.7)
        axes[1, 1].set_xlabel('Generation', fontsize=12)
        axes[1, 1].set_ylabel('Total Volume (mm³)', fontsize=12)
        axes[1, 1].set_title('Cumulative Volume by Generation', fontsize=13, weight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'generation_schematic.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print("    Saved: generation_schematic.png")

    def run_full_analysis(self, output_dir):
        """Pipeline completa con tutte le visualizzazioni"""
        print("\n" + "=" * 80)
        print("FULL VESSEL ANALYSIS WITH PATH TRACKING")
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        # 1-2. Skeleton + Graph
        self.compute_skeleton_and_distances()
        self.build_graph_with_properties()

        # 3. Identify trunks
        self.identify_initial_points()

        # 4. Extract all paths
        self.extract_all_paths_from_trunks()

        # 5. Compute generations
        self.compute_generation_from_trunks()

        # 6. Analyze diameter progression
        gen_stats = self.analyze_diameter_progression_by_generation()

        # 6.5 Murray's law (NUOVO)
        murray_df = self.check_murrays_law(tolerance=0.2, use_top2_children=True)

        # 7. Save outputs
        if self.branch_df is not None:
            csv_path = os.path.join(output_dir, "branch_metrics.csv")
            self.branch_df.to_csv(csv_path, index=False)
            print(f"\n✓ Branch metrics: {csv_path}")

        if self.paths_df is not None:
            paths_csv = os.path.join(output_dir, "path_metrics.csv")
            self.paths_df.drop(columns=['path_nodes']).to_csv(paths_csv, index=False)
            print(f"✓ Path metrics: {paths_csv}")

        if gen_stats is not None:
            gen_csv = os.path.join(output_dir, "generation_stats.csv")
            gen_stats.to_csv(gen_csv)
            print(f"✓ Generation stats: {gen_csv}")

        if murray_df is not None:
            murray_csv = os.path.join(output_dir, "murray_violations.csv")
            murray_df.to_csv(murray_csv, index=False)
            print(f"✓ Murray violations: {murray_csv}")

        # 8. VISUALIZATIONS - ALL REQUESTED
        print("\n=== Generating All Visualizations ===")

        self.visualize_3d_vessels(output_dir, color_by='diameter')
        self.visualize_3d_vessels(output_dir, color_by='length')
        self.visualize_3d_vessels(output_dir, color_by='generation')

        self.create_path_visualizations(output_dir)
        self.create_generation_schematic(output_dir)

        # 9. Summary
        summary = self._create_summary()
        json_path = os.path.join(output_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Complete analysis saved in: {output_dir}")

        return summary

    def _create_summary(self):
        """Crea summary con path statistics + Murray"""
        summary = {
            "analysis_type": "unified_vascular_with_paths",
            "num_trunks": len(self.initial_points) if self.initial_points else 0,
            "morphometry": {},
            "paths": {},
            "hierarchy": {},
            "murray": {}
        }

        if self.branch_df is not None and len(self.branch_df) > 0:
            summary["morphometry"] = {
                "num_branches": int(len(self.branch_df)),
                "total_length_mm": float(self.branch_df['length_mm'].sum()),
                "mean_diameter_mm": float(self.branch_df['diameter_mm'].mean()),
                "total_volume_mm3": float(self.branch_df['volume_mm3'].sum())
            }

        if self.paths_df is not None and len(self.paths_df) > 0:
            summary["paths"] = {
                "num_paths": int(len(self.paths_df)),
                "mean_path_length_mm": float(self.paths_df['total_length_mm'].mean()),
                "mean_taper_ratio": float(self.paths_df['taper_ratio'].mean()),
                "mean_diameter_change_pct": float(self.paths_df['diameter_change_pct'].mean())
            }

        if self.branch_df is not None and 'generation' in self.branch_df.columns:
            valid_gen = self.branch_df[self.branch_df['generation'] >= 0]
            if len(valid_gen) > 0:
                summary["hierarchy"] = {
                    "max_generation": int(valid_gen['generation'].max()),
                    "branches_with_generation": int(len(valid_gen)),
                    "branches_unreachable": int(len(self.branch_df) - len(valid_gen))
                }
            else:
                summary["hierarchy"] = {
                    "max_generation": None,
                    "branches_with_generation": 0,
                    "branches_unreachable": int(len(self.branch_df))
                }

        # Murray summary
        if self.murray_violations_df is not None:
            summary["murray"] = {
                "num_violations": int(len(self.murray_violations_df)),
                "mean_ratio": float(self.murray_violations_df['ratio_actual_over_expected'].mean())
                if len(self.murray_violations_df) > 0 else None,
                "median_ratio": float(self.murray_violations_df['ratio_actual_over_expected'].median())
                if len(self.murray_violations_df) > 0 else None
            }
        else:
            summary["murray"] = {
                "num_violations": None,
                "mean_ratio": None,
                "median_ratio": None
            }

        return summary
