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
    Sistema robusto per la detection della carina che gestisce:
    - Rumore nella parte superiore della trachea
    - Variabilità anatomica tra scans
    - Artefatti di segmentazione
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
        
    def detect_carina_robust(self):
        """
        Detection robusta della carina usando approccio multi-metodo
        con voting e validazione
        """
        if self.verbose:
            print("\n" + "="*70)
            print(" "*20 + "ROBUST CARINA DETECTION")
            print("="*70)
        
        # STEP 1: Pre-processing aggressivo per rimuovere rumore superiore
        self._aggressive_upper_cleaning()
        
        # STEP 2: Applica multipli metodi di detection
        candidates = []
        
        # Metodo 1: Analisi componenti connesse (più robusto)
        carina_cc = self._detect_by_connected_components()
        if carina_cc is not None:
            candidates.append(('connected_components', carina_cc, 3.0))  # peso alto
        
        # Metodo 2: Analisi diametro su skeleton
        carina_diameter = self._detect_by_diameter_analysis()
        if carina_diameter is not None:
            candidates.append(('diameter', carina_diameter, 2.0))
        
        # Metodo 3: Analisi biforcazione topologica
        carina_topo = self._detect_by_topology()
        if carina_topo is not None:
            candidates.append(('topology', carina_topo, 2.5))
        
        # Metodo 4: Analisi slice-by-slice (fallback robusto)
        carina_slice = self._detect_by_slice_analysis()
        if carina_slice is not None:
            candidates.append(('slice_analysis', carina_slice, 1.5))
        
        # STEP 3: Voting e selezione del miglior candidato
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
    
    def _aggressive_upper_cleaning(self):
        """
        Pulizia aggressiva della parte superiore per rimuovere rumore
        che potrebbe confondere la detection
        """
        if self.verbose:
            print("\nStep 1: Aggressive upper cleaning...")
        
        binary_mask = (self.mask > 0).astype(np.uint8)
        
        # Trova componente principale
        labeled, num_objects = label(binary_mask)
        if num_objects == 0:
            self.cleaned_mask = self.mask
            return
        
        # Prendi solo la componente più grande
        sizes = [np.sum(labeled == i) for i in range(1, num_objects + 1)]
        main_component = np.argmax(sizes) + 1
        main_mask = (labeled == main_component)
        
        # Trova estensione in Z
        main_coords = np.argwhere(main_mask)
        if len(main_coords) == 0:
            self.cleaned_mask = self.mask
            return
        
        min_z, max_z = np.min(main_coords[:, 0]), np.max(main_coords[:, 0])
        height = max_z - min_z
        
        # RIMOZIONE AGGRESSIVA: rimuovi top 35% (più del 18% precedente)
        remove_slices = int(height * 0.35)
        remove_slices = max(10, min(remove_slices, 40))  # tra 10 e 40 slices
        
        cutoff = max_z - remove_slices
        
        cleaned = self.mask.copy()
        cleaned[cutoff:, :, :] = 0
        
        self.cleaned_mask = cleaned
        
        if self.verbose:
            removed_voxels = np.sum(self.mask > 0) - np.sum(cleaned > 0)
            print(f"  Removed {remove_slices} upper slices (z > {cutoff})")
            print(f"  Removed {removed_voxels:,} voxels ({removed_voxels/np.sum(self.mask>0)*100:.1f}%)")
    
    def _detect_by_connected_components(self):
        """
        Metodo 1: Detection basata su analisi componenti connesse
        Cerca lo slice dove compaiono 2+ componenti separate di dimensioni simili
        """
        if self.verbose:
            print("\nMethod 1: Connected components analysis...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        
        # Cerca dal basso verso l'alto (dalla periferia verso il centro)
        candidates = []
        
        for z in range(binary_mask.shape[0] - 1, max(0, binary_mask.shape[0] - 100), -1):
            slice_2d = binary_mask[z, :, :]
            if np.sum(slice_2d) < 100:  # troppo piccolo
                continue
            
            labeled, num_objects = label(slice_2d)
            
            if num_objects >= 2:
                # Analizza dimensioni degli oggetti
                sizes = []
                centroids = []
                for obj_id in range(1, num_objects + 1):
                    obj_mask = (labeled == obj_id)
                    size = np.sum(obj_mask)
                    sizes.append(size)
                    centroids.append(center_of_mass(obj_mask))
                
                # Ordina per dimensione
                sorted_indices = np.argsort(sizes)[::-1]
                
                if len(sizes) >= 2:
                    size1, size2 = sizes[sorted_indices[0]], sizes[sorted_indices[1]]
                    
                    # Check: i due oggetti principali sono di dimensioni simili?
                    size_ratio = size2 / size1 if size1 > 0 else 0
                    
                    # Score basato su:
                    # 1. Rapporto dimensioni (ideale: 0.5-1.0)
                    # 2. Dimensione assoluta degli oggetti
                    # 3. Numero di oggetti (2 è ideale)
                    
                    if size_ratio > 0.2:  # almeno 20% della dimensione principale
                        score = size_ratio * (1.0 if num_objects == 2 else 0.7)
                        score *= min(1.0, (size1 + size2) / 1000)  # bonus per oggetti grandi
                        
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
        
        # Seleziona il candidato con score più alto
        best = max(candidates, key=lambda x: x['score'])
        
        if self.verbose:
            print(f"  Found {len(candidates)} candidates")
            print(f"  Best: z={best['z']}, score={best['score']:.3f}, "
                  f"objects={best['num_objects']}, ratio={best['size_ratio']:.3f}")
        
        return (best['z'], best['y'], best['x'])
    
    def _detect_by_diameter_analysis(self):
        """
        Metodo 2: Detection basata su analisi diametro lungo skeleton
        Cerca il punto con diametro massimo che sia anche una biforcazione
        """
        if self.verbose:
            print("\nMethod 2: Diameter analysis on skeleton...")
        
        try:
            binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
            
            # Compute skeleton
            skeleton = skeletonize(binary_mask)
            if np.sum(skeleton) < 10:
                return None
            
            # Distance transform
            spacing_zyx = (self.spacing[2], self.spacing[1], self.spacing[0])
            distance_transform = distance_transform_edt(binary_mask, sampling=spacing_zyx)
            
            # Build graph
            skeleton_obj = Skeleton(skeleton, spacing=spacing_zyx)
            coordinates = skeleton_obj.coordinates
            
            if len(coordinates) == 0:
                return None
            
            # Analizza nodi con diametro alto E grado >= 3 (biforcazioni)
            candidates = []
            
            for idx in range(len(coordinates)):
                pos = coordinates[idx]
                z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
                
                if not (0 <= z < distance_transform.shape[0] and
                       0 <= y < distance_transform.shape[1] and
                       0 <= x < distance_transform.shape[2]):
                    continue
                
                diameter = distance_transform[z, y, x] * 2
                
                # Score basato su diametro e posizione
                # Preferisci posizioni centrali (non troppo in alto o basso)
                z_score = 1.0 - abs(z - binary_mask.shape[0] * 0.6) / (binary_mask.shape[0] * 0.4)
                z_score = max(0.0, z_score)
                
                score = diameter * (0.5 + 0.5 * z_score)
                
                candidates.append({
                    'z': z,
                    'y': y,
                    'x': x,
                    'diameter': diameter,
                    'score': score
                })
            
            if not candidates:
                return None
            
            # Prendi top 5% per diametro
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top_candidates = candidates[:max(1, len(candidates) // 20)]
            
            # Tra questi, scegli quello più centrale
            best = min(top_candidates, 
                      key=lambda x: abs(x['z'] - binary_mask.shape[0] * 0.6))
            
            if self.verbose:
                print(f"  Best: z={best['z']}, diameter={best['diameter']:.2f}mm")
            
            return (best['z'], best['y'], best['x'])
            
        except Exception as e:
            if self.verbose:
                print(f"  Error in diameter analysis: {e}")
            return None
    
    def _detect_by_topology(self):
        """
        Metodo 3: Detection basata su analisi topologica del grafo
        Cerca il nodo con massimo betweenness centrality (punto centrale)
        """
        if self.verbose:
            print("\nMethod 3: Topology analysis...")
        
        try:
            binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
            
            # Compute skeleton
            skeleton = skeletonize(binary_mask)
            if np.sum(skeleton) < 10:
                return None
            
            # Build graph
            spacing_zyx = (self.spacing[2], self.spacing[1], self.spacing[0])
            skeleton_obj = Skeleton(skeleton, spacing=spacing_zyx)
            branch_data = summarize(skeleton_obj)
            
            G = nx.Graph()
            coordinates = skeleton_obj.coordinates
            
            # Add nodes
            for idx in range(len(coordinates)):
                pos = coordinates[idx]
                G.add_node(idx, pos=pos)
            
            # Add edges
            for _, row in branch_data.iterrows():
                node1 = int(row['node-id-src'])
                node2 = int(row['node-id-dst'])
                length = row['branch-distance']
                G.add_edge(node1, node2, length=length)
            
            if len(G.nodes()) < 3:
                return None
            
            # Calcola betweenness centrality (indica punto centrale)
            try:
                centrality = nx.betweenness_centrality(G, weight='length')
            except:
                return None
            
            # Filtra solo nodi con grado >= 3 (biforcazioni)
            bifurcation_nodes = [n for n in G.nodes() if G.degree(n) >= 3]
            
            if not bifurcation_nodes:
                bifurcation_nodes = [n for n in G.nodes() if G.degree(n) >= 2]
            
            if not bifurcation_nodes:
                return None
            
            # Tra le biforcazioni, prendi quella con massima centrality
            best_node = max(bifurcation_nodes, key=lambda n: centrality.get(n, 0))
            best_pos = G.nodes[best_node]['pos']
            
            z, y, x = int(best_pos[0]), int(best_pos[1]), int(best_pos[2])
            
            if self.verbose:
                print(f"  Best: z={z}, centrality={centrality[best_node]:.3f}")
            
            return (z, y, x)
            
        except Exception as e:
            if self.verbose:
                print(f"  Error in topology analysis: {e}")
            return None
    
    def _detect_by_slice_analysis(self):
        """
        Metodo 4: Analisi slice-by-slice cercando pattern di biforcazione
        """
        if self.verbose:
            print("\nMethod 4: Slice-by-slice analysis...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        
        candidates = []
        
        # Analizza ogni slice
        for z in range(binary_mask.shape[0] - 1, max(0, binary_mask.shape[0] - 80), -1):
            slice_2d = binary_mask[z, :, :]
            area = np.sum(slice_2d)
            
            if area < 50:
                continue
            
            # Conta componenti connesse
            labeled, num_objects = label(slice_2d)
            
            # Calcola "circularity" (quanto è compatto)
            if area > 0:
                coords = np.argwhere(slice_2d)
                y_extent = np.max(coords[:, 0]) - np.min(coords[:, 0])
                x_extent = np.max(coords[:, 1]) - np.min(coords[:, 1])
                extent = (y_extent + 1) * (x_extent + 1)
                compactness = area / extent if extent > 0 else 0
            else:
                compactness = 0
            
            # Score: preferisci 2 oggetti, bassa compattezza (biforcazione)
            score = 0
            if num_objects == 2:
                score = 2.0 * (1.0 - compactness)
            elif num_objects > 2:
                score = 1.0 * (1.0 - compactness)
            
            if score > 0.5:
                # Trova centro di massa
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
        
        # Prendi il candidato con score più alto
        best = max(candidates, key=lambda x: x['score'])
        
        if self.verbose:
            print(f"  Best: z={best['z']}, score={best['score']:.3f}")
        
        return (best['z'], best['y'], best['x'])
    
    def _vote_and_select(self, candidates):
        """
        Voting tra i candidati per selezionare la posizione più affidabile
        """
        if self.verbose:
            print(f"\nVoting among {len(candidates)} candidates...")
        
        # Calcola distanze tra tutti i candidati
        positions = np.array([c[1] for c in candidates])
        weights = np.array([c[2] for c in candidates])
        
        # Weighted voting usando distanza spaziale
        best_idx = None
        best_consensus_score = -1
        
        for i, (method, pos, weight) in enumerate(candidates):
            # Calcola consensus score: quanto questo candidato è vicino agli altri
            consensus = 0
            for j, (_, other_pos, other_weight) in enumerate(candidates):
                if i == j:
                    continue
                
                # Distanza euclidea (in voxel)
                dist = np.linalg.norm(np.array(pos) - np.array(other_pos))
                
                # Score inversamente proporzionale alla distanza
                if dist < 20:  # entro 20 voxel
                    consensus += other_weight * (1.0 - dist / 20)
            
            # Score totale = peso del metodo + consensus
            total_score = weight + consensus
            
            if self.verbose:
                print(f"  {method}: pos={pos}, weight={weight:.1f}, "
                      f"consensus={consensus:.2f}, total={total_score:.2f}")
            
            if total_score > best_consensus_score:
                best_consensus_score = total_score
                best_idx = i
        
        best_method, best_pos, best_weight = candidates[best_idx]
        
        # Confidence basata sul consensus (normalizza a 0-5)
        confidence = min(5.0, best_consensus_score / len(candidates))
        
        return {
            'method': best_method,
            'position': best_pos,
            'confidence': confidence
        }
    
    def _fallback_detection(self):
        """
        Fallback: usa posizione anatomica standard se tutti i metodi falliscono
        """
        if self.verbose:
            print("\nUsing fallback detection (anatomical position)...")
        
        binary_mask = (self.cleaned_mask > 0).astype(np.uint8)
        coords = np.argwhere(binary_mask)
        
        if len(coords) == 0:
            # Ultimo resort: centro del volume
            self.carina_z = binary_mask.shape[0] // 2
            self.carina_y = binary_mask.shape[1] // 2
            self.carina_x = binary_mask.shape[2] // 2
            self.confidence_score = 0.5
            self.detection_method = 'fallback_center'
            return self.carina_z, self.carina_y, self.carina_x, 0.5
        
        # Usa 60% dall'alto (posizione anatomica tipica della carina)
        min_z, max_z = np.min(coords[:, 0]), np.max(coords[:, 0])
        self.carina_z = int(min_z + (max_z - min_z) * 0.4)
        
        # Centro in y, x
        self.carina_y = int(np.mean(coords[:, 1]))
        self.carina_x = int(np.mean(coords[:, 2]))
        
        self.confidence_score = 1.0
        self.detection_method = 'fallback_anatomical'
        
        return self.carina_z, self.carina_y, self.carina_x, 1.0
    
    def get_trachea_cutoff(self, safety_margin_mm=5):
        """
        Determina il punto di taglio ottimale per rimuovere la trachea
        basato sulla carina detectata
        
        Args:
            safety_margin_mm: Margine di sicurezza sopra la carina (mm)
        
        Returns:
            cutoff_z: Coordinata Z per il taglio
        """
        if self.carina_z is None:
            raise ValueError("Run detect_carina_robust() first")
        
        # Converti margine da mm a voxel
        margin_voxels = int(safety_margin_mm / self.spacing[2])
        
        # Cutoff: carina + margine
        cutoff_z = min(self.mask.shape[0] - 1, self.carina_z + margin_voxels)
        
        if self.verbose:
            print(f"\nTrachea cutoff determination:")
            print(f"  Carina z: {self.carina_z}")
            print(f"  Safety margin: {safety_margin_mm} mm ({margin_voxels} voxels)")
            print(f"  Cutoff z: {cutoff_z}")
        
        return cutoff_z
    
    def remove_trachea(self, safety_margin_mm=5):
        """
        Rimuove la trachea mantenendo i bronchi
        
        Returns:
            bronchi_mask: Maschera con solo i bronchi
            cutoff_z: Coordinata del taglio
        """
        if self.carina_z is None:
            raise ValueError("Run detect_carina_robust() first")
        
        cutoff_z = self.get_trachea_cutoff(safety_margin_mm)
        
        bronchi_mask = self.cleaned_mask.copy()
        
        # Rimuovi tutto sopra il cutoff
        original_voxels = np.sum(bronchi_mask > 0)
        bronchi_mask[cutoff_z:, :, :] = 0
        
        # Preserva piccola regione attorno alla carina per connettività
        preserve_radius = 2
        z0 = max(0, self.carina_z - preserve_radius)
        z1 = min(bronchi_mask.shape[0], self.carina_z + preserve_radius + 1)
        y0 = max(0, self.carina_y - preserve_radius)
        y1 = min(bronchi_mask.shape[1], self.carina_y + preserve_radius + 1)
        x0 = max(0, self.carina_x - preserve_radius)
        x1 = min(bronchi_mask.shape[2], self.carina_x + preserve_radius + 1)
        
        bronchi_mask[z0:z1, y0:y1, x0:x1] = self.cleaned_mask[z0:z1, y0:y1, x0:x1]
        
        # Morfologia minima per connettività
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
    Funzione di integrazione con la pipeline esistente
    
    Args:
        mask_path: Path alla maschera delle vie aeree
        spacing: Tuple (x,y,z) spacing, se None lo legge dall'immagine
        save_visualization: Salva visualizzazioni debug
    
    Returns:
        bronchi_mask: Maschera dei soli bronchi
        carina_coords: Tuple (z, y, x) coordinate della carina
        confidence: Score di confidenza (0-5)
        cutoff_z: Coordinata Z del taglio
    """
    # Load mask
    sitk_image = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_image)
    
    if spacing is None:
        spacing = sitk_image.GetSpacing()
    
    # Detect carina
    detector = RobustCarinaDetector(mask, spacing, verbose=True)
    carina_z, carina_y, carina_x, confidence = detector.detect_carina_robust()
    
    # Remove trachea
    bronchi_mask, cutoff_z = detector.remove_trachea(safety_margin_mm=5)
    
    return bronchi_mask, (carina_z, carina_y, carina_x), confidence, cutoff_z


# ESEMPIO DI UTILIZZO
if __name__ == "__main__":
    import os
    
    print("="*70)
    print(" "*15 + "ROBUST CARINA DETECTION - EXAMPLE")
    print("="*70)
    
    # Parametri
    mask_path = "airway_segmentation/1.2.840.113704.1.111.1396.1132404220.7_airwayfull.nii.gz"
    
    if os.path.exists(mask_path):
        # Usa la funzione di integrazione
        bronchi_mask, carina_coords, confidence, cutoff_z = integrate_with_pipeline(
            mask_path, 
            spacing=None,  # Auto-detect
            save_visualization=True
        )
        
        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"{'='*70}")
        print(f"Carina position: z={carina_coords[0]}, y={carina_coords[1]}, x={carina_coords[2]}")
        print(f"Confidence score: {confidence:.2f}/5.0")
        print(f"Trachea cutoff: z={cutoff_z}")
        print(f"Bronchi voxels: {np.sum(bronchi_mask > 0):,}")
        
        # Salva risultato
        output_sitk = sitk.GetImageFromArray(bronchi_mask.astype(np.uint8))
        output_sitk.CopyInformation(sitk.ReadImage(mask_path))
        sitk.WriteImage(output_sitk, "bronchi_only_robust.nii.gz")
        print(f"\n✓ Saved: bronchi_only_robust.nii.gz")
    else:
        print(f"\n⚠ File not found: {mask_path}")
        print("Please update the mask_path variable with your file path")