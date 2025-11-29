import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, center_of_mass, binary_erosion, binary_dilation
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from skimage.morphology import skeletonize, ball
from skan import Skeleton, summarize
import networkx as nx
from collections import defaultdict


class RobustCarinaDetector:
    """
    Sistema robusto per la detection della carina con rimozione selettiva della trachea
    Preserva carina e bronchi, rimuove solo il 50% superiore del tronco tracheale
    """
    
    def __init__(self, mask, spacing, verbose=True):
        """
        Args:
            mask: Maschera 3D delle vie aeree (numpy array)
            spacing: Tuple (x,y,z) spacing in mm
            verbose: Print debug information
        """
        self.mask = mask
        self.spacing = spacing
        self.verbose = verbose
        
        # Risultati
        self.cleaned_mask = None
        self.carina_z = None
        self.carina_y = None
        self.carina_x = None
        self.confidence_score = 0.0
        self.detection_method = None
        self.trachea_top_z = None
        self.trachea_length = None
        
    def detect_carina_robust(self):
        """
        Detection robusta della carina usando approccio multi-metodo
        """
        if self.verbose:
            print("\n" + "="*70)
            print(" "*20 + "ROBUST CARINA DETECTION")
            print("="*70)
        
        # STEP 1: Pre-processing conservativo
        self._adaptive_upper_cleaning()
        
        # STEP 2: Applica multipli metodi di detection
        candidates = []
        
        carina_cc = self._detect_by_connected_components()
        if carina_cc is not None:
            candidates.append(('connected_components', carina_cc, 3.0))
        
        carina_diameter = self._detect_by_diameter_analysis()
        if carina_diameter is not None:
            candidates.append(('diameter', carina_diameter, 2.0))
        
        carina_topo = self._detect_by_topology()
        if carina_topo is not None:
            candidates.append(('topology', carina_topo, 2.5))
        
        carina_slice = self._detect_by_slice_analysis()
        if carina_slice is not None:
            candidates.append(('slice_analysis', carina_slice, 1.5))
        
        # STEP 3: Voting e selezione
        if not candidates:
            if self.verbose:
                print("\n⚠ WARNING: No candidates found, using fallback")
            return self._fallback_detection()
        
        best_candidate = self._vote_and_select(candidates)
        
        self.carina_z, self.carina_y, self.carina_x = best_candidate['position']
        self.confidence_score = best_candidate['confidence']
        self.detection_method = best_candidate['method']
        
        if self.verbose:
            print(f"\n✓ CARINA DETECTED:")
            print(f"  Position: z={self.carina_z}, y={self.carina_y}, x={self.carina_x}")
            print(f"  Method: {self.detection_method}")
            print(f"  Confidence: {self.confidence_score:.2f}/5.0")
        
        return self.carina_z, self.carina_y, self.carina_x, self.confidence_score
    
    def _adaptive_upper_cleaning(self):
        """
        Pulizia conservativa della parte superiore
        """
        if self.verbose:
            print("\nStep 1: Adaptive conservative cleaning...")
        
        binary_mask = (self.mask > 0).astype(np.uint8)
        
        labeled, num_objects = label(binary_mask)
        if num_objects == 0:
            self.cleaned_mask = self.mask
            return
        
        sizes = [np.sum(labeled == i) for i in range(1, num_objects + 1)]
        main_component = np.argmax(sizes) + 1
        main_mask = (labeled == main_component)
        
        main_coords = np.argwhere(main_mask)
        if len(main_coords) == 0:
            self.cleaned_mask = self.mask
            return
        
        min_z, max_z = np.min(main_coords[:, 0]), np.max(main_coords[:, 0])
        height = max_z - min_z
        
        # Analisi morfologica
        slice_areas = []
        slice_indices = []
        
        for z in range(max_z, min_z, -1):
            slice_2d = main_mask[z, :, :]
            area = np.sum(slice_2d)
            if area > 0:
                slice_areas.append(area)
                slice_indices.append(z)
        
        if len(slice_areas) < 10:
            self.cleaned_mask = self.mask
            return
        
        # Cerca stabilizzazione
        window = 5
        areas_smooth = np.convolve(slice_areas, np.ones(window)/window, mode='valid')
        
        cutoff_idx = 0
        if len(areas_smooth) > 10:
            derivatives = np.abs(np.diff(areas_smooth))
            stable_region_size = min(20, len(derivatives) // 3)
            stable_derivative = np.mean(derivatives[-stable_region_size:])
            threshold = stable_derivative * 3
            
            for i, deriv in enumerate(derivatives):
                if deriv < threshold:
                    cutoff_idx = i
                    break
        
        if cutoff_idx == 0:
            remove_slices = int(height * 0.10)
            remove_slices = max(3, min(remove_slices, 15))
        else:
            remove_slices = cutoff_idx + 2
            remove_slices = min(remove_slices, int(height * 0.20))
        
        cutoff = slice_indices[min(remove_slices, len(slice_indices)-1)]
        
        cleaned = self.mask.copy()
        cleaned[cutoff:, :, :] = 0
        
        self.cleaned_mask = cleaned
        
        if self.verbose:
            removed_voxels = np.sum(self.mask > 0) - np.sum(cleaned > 0)
            print(f"  Removed: {remove_slices} upper slices (z > {cutoff})")
            print(f"  Removed voxels: {removed_voxels:,} ({removed_voxels/np.sum(self.mask>0)*100:.1f}%)")
    
    def _detect_by_connected_components(self):
        """
        Detection basata su componenti connesse
        """
        if self.verbose:
            print("\nMethod 1: Connected components analysis...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        candidates = []
        
        z_start = binary_mask.shape[0] - 1
        z_min = max(0, int(binary_mask.shape[0] * 0.2))
        
        for z in range(z_start, z_min, -1):
            slice_2d = binary_mask[z, :, :]
            if np.sum(slice_2d) < 50:
                continue
            
            labeled, num_objects = label(slice_2d)
            
            if num_objects >= 2:
                sizes = []
                centroids = []
                for obj_id in range(1, num_objects + 1):
                    obj_mask = (labeled == obj_id)
                    size = np.sum(obj_mask)
                    sizes.append(size)
                    centroids.append(center_of_mass(obj_mask))
                
                sorted_indices = np.argsort(sizes)[::-1]
                
                if len(sizes) >= 2:
                    size1, size2 = sizes[sorted_indices[0]], sizes[sorted_indices[1]]
                    size_ratio = size2 / size1 if size1 > 0 else 0
                    
                    if size_ratio > 0.15:
                        score = size_ratio * (1.0 if num_objects == 2 else 0.8)
                        score *= min(1.0, (size1 + size2) / 800)
                        
                        cent1 = centroids[sorted_indices[0]]
                        cent2 = centroids[sorted_indices[1]]
                        carina_y = int((cent1[0] + cent2[0]) / 2)
                        carina_x = int((cent1[1] + cent2[1]) / 2)
                        
                        candidates.append({
                            'z': z,
                            'y': carina_y,
                            'x': carina_x,
                            'score': score,
                            'num_objects': num_objects,
                            'size_ratio': size_ratio
                        })
        
        if not candidates:
            if self.verbose:
                print("  No valid candidates found")
            return None
        
        good_candidates = [c for c in candidates if c['score'] > 0.3]
        if good_candidates:
            best = max(good_candidates, key=lambda x: x['z'])
        else:
            best = max(candidates, key=lambda x: x['score'])
        
        if self.verbose:
            print(f"  Best: z={best['z']}, score={best['score']:.3f}")
        
        return (best['z'], best['y'], best['x'])
    
    def _detect_by_diameter_analysis(self):
        """
        Detection basata su diametro
        """
        if self.verbose:
            print("\nMethod 2: Diameter analysis...")
        
        try:
            binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
            skeleton = skeletonize(binary_mask)
            if np.sum(skeleton) < 10:
                return None
            
            spacing_zyx = (self.spacing[2], self.spacing[1], self.spacing[0])
            distance_transform = distance_transform_edt(binary_mask, sampling=spacing_zyx)
            
            skeleton_obj = Skeleton(skeleton, spacing=spacing_zyx)
            coordinates = skeleton_obj.coordinates
            
            if len(coordinates) == 0:
                return None
            
            candidates = []
            z_threshold = binary_mask.shape[0] * 0.4
            
            for idx in range(len(coordinates)):
                pos = coordinates[idx]
                z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
                
                if z < z_threshold:
                    continue
                
                if not (0 <= z < distance_transform.shape[0] and
                       0 <= y < distance_transform.shape[1] and
                       0 <= x < distance_transform.shape[2]):
                    continue
                
                diameter = distance_transform[z, y, x] * 2
                
                candidates.append({
                    'z': z,
                    'y': y,
                    'x': x,
                    'diameter': diameter,
                    'score': diameter
                })
            
            if not candidates:
                return None
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_candidates = candidates[:max(5, len(candidates) // 10)]
            best = max(top_candidates, key=lambda x: x['z'])
            
            if self.verbose:
                print(f"  Best: z={best['z']}, diameter={best['diameter']:.2f}mm")
            
            return (best['z'], best['y'], best['x'])
            
        except Exception as e:
            if self.verbose:
                print(f"  Error: {e}")
            return None
    
    def _detect_by_topology(self):
        """
        Detection basata su topologia
        """
        if self.verbose:
            print("\nMethod 3: Topology analysis...")
        
        try:
            binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
            skeleton = skeletonize(binary_mask)
            if np.sum(skeleton) < 10:
                return None
            
            spacing_zyx = (self.spacing[2], self.spacing[1], self.spacing[0])
            skeleton_obj = Skeleton(skeleton, spacing=spacing_zyx)
            branch_data = summarize(skeleton_obj)
            
            G = nx.Graph()
            coordinates = skeleton_obj.coordinates
            
            for idx in range(len(coordinates)):
                pos = coordinates[idx]
                G.add_node(idx, pos=pos)
            
            for _, row in branch_data.iterrows():
                node1 = int(row['node-id-src'])
                node2 = int(row['node-id-dst'])
                length = row['branch-distance']
                G.add_edge(node1, node2, length=length)
            
            if len(G.nodes()) < 3:
                return None
            
            try:
                centrality = nx.betweenness_centrality(G, weight='length')
            except:
                return None
            
            z_threshold = binary_mask.shape[0] * 0.4
            bifurcation_nodes = []
            
            for n in G.nodes():
                if G.degree(n) >= 3:
                    pos = G.nodes[n]['pos']
                    if pos[0] >= z_threshold:
                        bifurcation_nodes.append(n)
            
            if not bifurcation_nodes:
                for n in G.nodes():
                    if G.degree(n) >= 2:
                        pos = G.nodes[n]['pos']
                        if pos[0] >= z_threshold:
                            bifurcation_nodes.append(n)
            
            if not bifurcation_nodes:
                return None
            
            bifurcation_nodes.sort(key=lambda n: G.nodes[n]['pos'][0], reverse=True)
            candidates = bifurcation_nodes[:5]
            
            best_node = max(candidates, key=lambda n: centrality.get(n, 0))
            best_pos = G.nodes[best_node]['pos']
            
            z, y, x = int(best_pos[0]), int(best_pos[1]), int(best_pos[2])
            
            if self.verbose:
                print(f"  Best: z={z}, centrality={centrality[best_node]:.3f}")
            
            return (z, y, x)
            
        except Exception as e:
            if self.verbose:
                print(f"  Error: {e}")
            return None
    
    def _detect_by_slice_analysis(self):
        """
        Analisi slice-by-slice
        """
        if self.verbose:
            print("\nMethod 4: Slice analysis...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        candidates = []
        
        z_start = binary_mask.shape[0] - 1
        z_min = max(0, int(binary_mask.shape[0] * 0.3))
        
        for z in range(z_start, z_min, -1):
            slice_2d = binary_mask[z, :, :]
            area = np.sum(slice_2d)
            
            if area < 30:
                continue
            
            labeled, num_objects = label(slice_2d)
            
            if area > 0:
                coords = np.argwhere(slice_2d)
                y_extent = np.max(coords[:, 0]) - np.min(coords[:, 0])
                x_extent = np.max(coords[:, 1]) - np.min(coords[:, 1])
                extent = (y_extent + 1) * (x_extent + 1)
                compactness = area / extent if extent > 0 else 0
            else:
                compactness = 0
            
            score = 0
            if num_objects == 2:
                score = 2.5 * (1.0 - compactness)
            elif num_objects > 2:
                score = 1.5 * (1.0 - compactness)
            
            if score > 0.4:
                cy, cx = center_of_mass(slice_2d)
                candidates.append({
                    'z': z,
                    'y': int(cy),
                    'x': int(cx),
                    'score': score,
                    'num_objects': num_objects
                })
        
        if not candidates:
            return None
        
        good_candidates = [c for c in candidates if c['score'] > 0.8]
        if good_candidates:
            best = max(good_candidates, key=lambda x: x['z'])
        else:
            best = max(candidates, key=lambda x: x['score'])
        
        if self.verbose:
            print(f"  Best: z={best['z']}, score={best['score']:.3f}")
        
        return (best['z'], best['y'], best['x'])
    
    def _vote_and_select(self, candidates):
        """
        Voting tra candidati
        """
        if self.verbose:
            print(f"\nVoting among {len(candidates)} candidates...")
        
        best_idx = None
        best_consensus_score = -1
        
        for i, (method, pos, weight) in enumerate(candidates):
            consensus = 0
            for j, (_, other_pos, other_weight) in enumerate(candidates):
                if i == j:
                    continue
                
                dist = np.linalg.norm(np.array(pos) - np.array(other_pos))
                
                if dist < 25:
                    consensus += other_weight * (1.0 - dist / 25)
            
            total_score = weight + consensus
            
            if self.verbose:
                print(f"  {method}: pos={pos}, weight={weight:.1f}, "
                      f"consensus={consensus:.2f}, total={total_score:.2f}")
            
            if total_score > best_consensus_score:
                best_consensus_score = total_score
                best_idx = i
        
        best_method, best_pos, best_weight = candidates[best_idx]
        confidence = min(5.0, best_consensus_score / len(candidates))
        
        return {
            'method': best_method,
            'position': best_pos,
            'confidence': confidence
        }
    
    def _fallback_detection(self):
        """
        Fallback conservativo
        """
        if self.verbose:
            print("\nUsing fallback detection...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        coords = np.argwhere(binary_mask)
        
        if len(coords) == 0:
            self.carina_z = binary_mask.shape[0] // 2
            self.carina_y = binary_mask.shape[1] // 2
            self.carina_x = binary_mask.shape[2] // 2
            self.confidence_score = 0.5
            self.detection_method = 'fallback_center'
            return self.carina_z, self.carina_y, self.carina_x, 0.5
        
        min_z, max_z = np.min(coords[:, 0]), np.max(coords[:, 0])
        self.carina_z = int(min_z + (max_z - min_z) * 0.5)
        
        self.carina_y = int(np.mean(coords[:, 1]))
        self.carina_x = int(np.mean(coords[:, 2]))
        
        self.confidence_score = 1.5
        self.detection_method = 'fallback_anatomical'
        
        return self.carina_z, self.carina_y, self.carina_x, 1.5
    
    def _trace_trachea_upward(self):
        """
        NUOVA FUNZIONE: Risale dalla carina per identificare tutta la trachea
        Trova dove inizia (top) e calcola la lunghezza totale
        """
        if self.verbose:
            print("\nTracing trachea upward from carina...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        
        # Parti dalla carina e sali
        current_z = self.carina_z
        trachea_slices = []
        
        # Trova la componente centrale (trachea) alla carina
        carina_slice = binary_mask[current_z, :, :]
        labeled, num_objects = label(carina_slice)
        
        if num_objects == 0:
            if self.verbose:
                print("  ⚠ No objects at carina level")
            return None
        
        # Identifica quale componente è la trachea (quella più vicina al centro della carina)
        min_dist = float('inf')
        trachea_label = 1
        
        for obj_id in range(1, num_objects + 1):
            obj_mask = (labeled == obj_id)
            cy, cx = center_of_mass(obj_mask)
            dist = np.sqrt((cy - self.carina_y)**2 + (cx - self.carina_x)**2)
            if dist < min_dist:
                min_dist = dist
                trachea_label = obj_id
        
        # Ora risali cercando la continuazione di questa componente
        for z in range(current_z, binary_mask.shape[0]):
            slice_2d = binary_mask[z, :, :]
            
            if np.sum(slice_2d) == 0:
                break
            
            # Cerca la componente connessa più centrale
            labeled_z, num_objs = label(slice_2d)
            
            if num_objs == 0:
                break
            
            # Se c'è una sola componente, è probabilmente la trachea
            if num_objs == 1:
                trachea_slices.append(z)
                continue
            
            # Se ci sono multiple componenti, prendi quella più centrale
            min_dist = float('inf')
            found_trachea = False
            
            for obj_id in range(1, num_objs + 1):
                obj_mask = (labeled_z == obj_id)
                cy, cx = center_of_mass(obj_mask)
                
                # Distanza dal centro anatomico (centro del volume)
                center_y = binary_mask.shape[1] / 2
                center_x = binary_mask.shape[2] / 2
                dist = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    found_trachea = True
            
            if found_trachea:
                # Verifica che non sia troppo laterale (rami bronchiali)
                if min_dist < binary_mask.shape[1] * 0.3:  # entro 30% dal centro
                    trachea_slices.append(z)
                else:
                    break
            else:
                break
        
        if len(trachea_slices) < 5:
            if self.verbose:
                print("  ⚠ Trachea too short, using fallback")
            # Fallback: usa distanza anatomica dalla carina
            self.trachea_top_z = min(binary_mask.shape[0] - 1, 
                                     self.carina_z + int(50 / self.spacing[2]))  # ~50mm
            self.trachea_length = self.trachea_top_z - self.carina_z
            return
        
        self.trachea_top_z = max(trachea_slices)
        self.trachea_length = self.trachea_top_z - self.carina_z
        
        if self.verbose:
            print(f"  ✓ Trachea traced:")
            print(f"    Bottom (carina): z={self.carina_z}")
            print(f"    Top: z={self.trachea_top_z}")
            print(f"    Length: {self.trachea_length} slices ({self.trachea_length * self.spacing[2]:.1f} mm)")
    
    def remove_trachea(self):
        """
        Rimuove solo il 50% superiore della trachea preservando carina e bronchi
        """
        if self.carina_z is None:
            raise ValueError("Run detect_carina_robust() first")
        
        # Prima, traccia la trachea verso l'alto
        self._trace_trachea_upward()
        
        if self.trachea_length is None or self.trachea_length < 5:
            if self.verbose:
                print("\n⚠ Cannot trace trachea, using conservative cutoff")
            # Fallback: taglia molto conservativamente
            cutoff_z = min(self.cleaned_mask.shape[0] - 1, 
                          self.carina_z + int(10 / self.spacing[2]))
        else:
            # Calcola 50% della lunghezza della trachea
            half_trachea = self.trachea_length // 2
            cutoff_z = self.carina_z + half_trachea
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("TRACHEA REMOVAL:")
            print(f"{'='*70}")
            print(f"  Carina: z={self.carina_z}")
            print(f"  Trachea top: z={self.trachea_top_z}")
            print(f"  Cutoff (50% point): z={cutoff_z}")
            print(f"  Removing from z={cutoff_z} upward")
        
        bronchi_mask = self.cleaned_mask.copy()
        original_voxels = np.sum(bronchi_mask > 0)
        
        # Rimuovi tutto sopra il cutoff
        bronchi_mask[cutoff_z:, :, :] = 0
        
        # Morfologia leggera per smoothing
        struct = generate_binary_structure(3, 1)
        bronchi_mask_binary = (bronchi_mask > 0).astype(np.uint8)
        bronchi_mask_binary = binary_dilation(bronchi_mask_binary, structure=struct, iterations=1)
        bronchi_mask_binary = binary_erosion(bronchi_mask_binary, structure=struct, iterations=1)
        
        bronchi_mask = bronchi_mask_binary.astype(self.mask.dtype) * np.max(self.mask)
        
        remaining_voxels = np.sum(bronchi_mask > 0)
        removed_voxels = original_voxels - remaining_voxels
        
        if self.verbose:
            print(f"\n✓ Trachea removal complete:")
            print(f"  Original: {original_voxels:,} voxels")
            print(f"  Remaining: {remaining_voxels:,} voxels")
            print(f"  Removed: {removed_voxels:,} ({removed_voxels/original_voxels*100:.1f}%)")
        
        return bronchi_mask, cutoff_z


def integrate_with_pipeline(mask_path, spacing=None, save_visualization=True):
    """
    Integrazione con pipeline
    """
    sitk_image = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_image)
    
    if spacing is None:
        spacing = sitk_image.GetSpacing()
    
    detector = RobustCarinaDetector(mask, spacing, verbose=True)
    carina_z, carina_y, carina_x, confidence = detector.detect_carina_robust()
    
    # Nuova logica: rimuovi solo 50% superiore della trachea
    bronchi_mask, cutoff_z = detector.remove_trachea()
    
    return bronchi_mask, (carina_z, carina_y, carina_x), confidence, cutoff_z


if __name__ == "__main__":
    import os
    
    print("="*70)
    print(" "*15 + "TRACHEA-AWARE CARINA DETECTION")
    print("="*70)
    
    mask_path = "airway_segmentation/1.2.840.113704.1.111.276.1120287318.6_airwayfull.nii.gz"
    
    if os.path.exists(mask_path):
        bronchi_mask, carina_coords, confidence, cutoff_z = integrate_with_pipeline(
            mask_path, 
            spacing=None,
            save_visualization=True
        )
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS:")
        print(f"{'='*70}")
        print(f"Carina: z={carina_coords[0]}, y={carina_coords[1]}, x={carina_coords[2]}")
        print(f"Confidence: {confidence:.2f}/5.0")
        print(f"Cutoff: z={cutoff_z}")
        print(f"Bronchi voxels: {np.sum(bronchi_mask > 0):,}")
        
        output_sitk = sitk.GetImageFromArray(bronchi_mask.astype(np.uint8))
        output_sitk.CopyInformation(sitk.ReadImage(mask_path))
        sitk.WriteImage(output_sitk, "bronchi_with_carina_preserved.nii.gz")
        print(f"\n✓ Saved: bronchi_with_carina_preserved.nii.gz")
    else:
        print(f"\n⚠ File not found: {mask_path}")