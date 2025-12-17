import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from scipy.ndimage import label, gaussian_filter
from skimage.morphology import skeletonize, ball
from skimage.filters import threshold_otsu, threshold_multiotsu
from collections import deque


class EnhancedAirwayRefinementModule:
    """
    Modulo avanzato per il refinement della segmentazione delle vie aeree.
    Miglioramenti rispetto alla versione base:
    - Soglia HU adattiva multi-regione
    - Region growing più sensibile con priorità
    - Recupero vie aeree periferiche sottili
    - Preservazione dettagli piccoli
    """
    
    def __init__(self, intensity_img, mask, spacing, verbose=True):
        self.img = intensity_img.astype(np.int16)  # CT image (numpy)
        self.mask = mask.astype(np.uint8)          # TS segmentation (binary)
        self.spacing = spacing
        self.verbose = verbose
        self.refined = None
        
        # Parametri adattivi
        self.hu_thresholds = self._estimate_adaptive_thresholds()
        self.distance_transform = None
        
    def _estimate_adaptive_thresholds(self):
        """
        Stima soglie HU adattive per diverse regioni anatomiche:
        - Trachea/bronchi principali: HU molto basse (aria pura)
        - Bronchi periferici: HU leggermente più alte (volume parziale)
        - Vie aeree distali: HU ancora più alte (artefatti da volume)
        """
        if self.verbose:
            print("\n[Enhanced Refinement] Estimating adaptive HU thresholds...")
        
        # Analizza solo i voxel mascherati
        flat = self.img[self.mask > 0].flatten()
        flat = flat[(flat > -1200) & (flat < 200)]
        
        if len(flat) < 100:
            # Fallback se pochi voxel
            return {'central': -850, 'intermediate': -750, 'peripheral': -650}
        
        # Usa Otsu multi-level per identificare 3 regioni
        try:
            # threshold_multiotsu(classes=3) restituisce 2 soglie [t0, t1]
            thresholds = threshold_multiotsu(flat, classes=3)

            # Verifica che abbiamo almeno 2 soglie
            if len(thresholds) >= 2:
                # Aggiungi margini di sicurezza usando le soglie disponibili
                t0, t1 = thresholds[0], thresholds[1]
                central_threshold = min(t0 + 100, -700)      # Più conservativo
                intermediate_threshold = min(t1 + 80, -600)  # Vie aeree medie
                # Periferico: usa la seconda soglia come base (non esiste thresholds[2])
                peripheral_threshold = min(t1 + 60, -500)    # Vie periferiche
            elif len(thresholds) == 1:
                # Solo una soglia disponibile, usa valori derivati
                t0 = thresholds[0]
                central_threshold = min(t0 + 100, -700)
                intermediate_threshold = -750
                peripheral_threshold = -650
            else:
                # Nessuna soglia valida, usa defaults
                raise ValueError("No valid thresholds returned")

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Otsu failed ({e}), using defaults")
            central_threshold = -850
            intermediate_threshold = -750
            peripheral_threshold = -650
        
        thresholds_dict = {
            'central': central_threshold,
            'intermediate': intermediate_threshold,
            'peripheral': peripheral_threshold
        }
        
        if self.verbose:
            print(f"  Central airways: HU < {central_threshold:.1f}")
            print(f"  Intermediate: HU < {intermediate_threshold:.1f}")
            print(f"  Peripheral: HU < {peripheral_threshold:.1f}")
        
        return thresholds_dict
    
    def _compute_distance_transform(self):
        """Calcola distance transform per identificare regioni centrali/periferiche"""
        if self.distance_transform is None:
            if self.verbose:
                print("\n[Enhanced Refinement] Computing distance transform...")
            self.distance_transform = distance_transform_edt(
                self.mask == 0, 
                sampling=self.spacing
            )
        return self.distance_transform
    
    def _get_adaptive_threshold(self, position):
        """
        Restituisce soglia HU adattiva in base alla posizione anatomica.
        Vie aeree centrali = soglie più stringenti
        Vie aeree periferiche = soglie più permissive
        """
        dt = self._compute_distance_transform()
        z, y, x = position
        
        # Distanza dal bordo della maschera (mm)
        dist_from_edge = dt[z, y, x]
        
        # Logica adattiva:
        # - Vicino al bordo (< 5mm) → usa soglia periferica
        # - Distanza media (5-15mm) → usa soglia intermedia
        # - Centro (> 15mm) → usa soglia centrale
        if dist_from_edge < 5:
            return self.hu_thresholds['peripheral']
        elif dist_from_edge < 15:
            return self.hu_thresholds['intermediate']
        else:
            return self.hu_thresholds['central']
    
    def _priority_region_grow(self, seeds, max_dist_mm=4.0, max_voxels=2000):
        """
        Region growing con priorità basata su:
        1. Distanza dal seed
        2. Intensità HU (più bassa = più probabile aria)
        3. Connettività alla maschera originale
        """
        if len(seeds) == 0:
            return []
        
        # Priority queue: (priority_score, position)
        from heapq import heappush, heappop
        
        visited = set()
        grown = []
        queue = []
        
        # Inizializza queue con i seeds
        for seed in seeds:
            seed_tuple = tuple(seed)
            if seed_tuple not in visited:
                # Priority = 0 per i seeds (massima priorità)
                heappush(queue, (0.0, seed_tuple))
                visited.add(seed_tuple)
        
        max_dist_vox = int(max_dist_mm / min(self.spacing))
        voxel_count = 0
        
        while queue and voxel_count < max_voxels:
            priority, current_pos = heappop(queue)
            z, y, x = current_pos
            
            # Aggiungi alla regione cresciuta
            grown.append(current_pos)
            voxel_count += 1
            
            # Soglia adattiva per questa posizione
            threshold = self._get_adaptive_threshold(current_pos)
            
            # Esplora vicini 26-connectivity
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        
                        nz, ny, nx = z + dz, y + dy, x + dx
                        neighbor_pos = (nz, ny, nx)
                        
                        # Bounds check
                        if not (0 <= nz < self.img.shape[0] and
                                0 <= ny < self.img.shape[1] and
                                0 <= nx < self.img.shape[2]):
                            continue
                        
                        if neighbor_pos in visited:
                            continue
                        
                        # Check HU threshold
                        hu_value = self.img[nz, ny, nx]
                        if hu_value >= threshold:
                            continue
                        
                        # Calcola priority score
                        # Componenti:
                        # 1. Distanza euclidea dai seeds
                        dist_from_seeds = min([
                            np.linalg.norm(np.array(neighbor_pos) - np.array(seed)) 
                            for seed in seeds
                        ])
                        
                        if dist_from_seeds > max_dist_vox:
                            continue
                        
                        # 2. Bonus per HU molto basse (aria certa)
                        hu_score = max(0, (threshold - hu_value) / 100.0)
                        
                        # 3. Bonus se connesso alla maschera originale
                        connectivity_bonus = 1.0 if self.mask[nz, ny, nx] > 0 else 0.0
                        
                        # Priority score (più basso = più prioritario)
                        priority_score = dist_from_seeds - hu_score - connectivity_bonus
                        
                        heappush(queue, (priority_score, neighbor_pos))
                        visited.add(neighbor_pos)
        
        return grown
    
    def _detect_endpoints_and_tips(self):
        """
        Identifica endpoints e tips delle vie aeree per region growing mirato.
        Usa skeleton per trovare terminazioni.
        """
        if self.verbose:
            print("\n[Enhanced Refinement] Detecting airway endpoints...")
        
        # Compute skeleton
        binary_mask = (self.mask > 0).astype(np.uint8)
        skeleton = skeletonize(binary_mask)
        
        # Conta vicini per ogni voxel dello skeleton
        from scipy.ndimage import convolve
        kernel = np.ones((3, 3, 3))
        kernel[1, 1, 1] = 0
        neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')
        neighbor_count = neighbor_count * skeleton  # Solo su skeleton
        
        # Endpoints = degree 1 (solo 1 vicino)
        endpoints = (neighbor_count == 1) & (skeleton > 0)
        endpoint_coords = np.argwhere(endpoints)
        
        if self.verbose:
            print(f"  Found {len(endpoint_coords)} endpoints")
        
        # Filtra endpoints troppo interni (probabilmente artefatti)
        dt = self._compute_distance_transform()
        valid_endpoints = []
        
        for ep in endpoint_coords:
            z, y, x = ep
            dist_to_edge = dt[z, y, x]
            
            # Endpoints validi sono vicini al bordo della maschera
            if dist_to_edge < 3.0:  # mm
                valid_endpoints.append(ep)
        
        if self.verbose:
            print(f"  Valid endpoints near edge: {len(valid_endpoints)}")
        
        return valid_endpoints
    
    def _recover_thin_airways(self):
        """
        Recupera vie aeree sottili perse dalla segmentazione iniziale.
        Usa morphological closing e analisi di connettività.
        """
        if self.verbose:
            print("\n[Enhanced Refinement] Recovering thin airways...")
        
        # 1. Identifica potenziali gap (buchi piccoli nella maschera)
        binary_mask = (self.mask > 0).astype(np.uint8)
        
        # Closing morfologico per colmare piccoli gap
        from skimage.morphology import binary_closing
        closed = binary_closing(binary_mask, ball(2))
        
        # Identifica i voxel aggiunti dal closing
        gap_candidates = closed & (~binary_mask.astype(bool))
        gap_coords = np.argwhere(gap_candidates)
        
        if len(gap_coords) == 0:
            if self.verbose:
                print("  No gaps to fill")
            return np.zeros_like(self.mask)
        
        # 2. Filtra candidati: mantieni solo quelli con HU da aria
        recovered = np.zeros_like(self.mask)
        recovered_count = 0
        
        for coord in gap_coords:
            z, y, x = coord
            hu_value = self.img[z, y, x]
            threshold = self._get_adaptive_threshold(coord)
            
            # Se HU compatibile con aria, aggiungi
            if hu_value < threshold:
                recovered[z, y, x] = 1
                recovered_count += 1
        
        if self.verbose:
            print(f"  Recovered {recovered_count} thin airway voxels")
        
        return recovered
    
    def _expand_near_skeleton(self):
        """
        Espande la maschera lungo lo skeleton dove manca tessuto.
        Utile per vie aeree collassate o con artefatti da volume parziale.
        """
        if self.verbose:
            print("\n[Enhanced Refinement] Expanding along skeleton...")
        
        binary_mask = (self.mask > 0).astype(np.uint8)
        skeleton = skeletonize(binary_mask)
        
        # Distance transform dalla maschera
        dt = self._compute_distance_transform()
        
        # Candidati: voxel NON nella maschera ma vicini allo skeleton
        skeleton_coords = np.argwhere(skeleton)
        expanded = np.zeros_like(self.mask)
        expanded_count = 0
        
        for skel_pos in skeleton_coords:
            z, y, x = skel_pos
            
            # Esplora vicini
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nz, ny, nx = z + dz, y + dy, x + dx
                        
                        if not (0 <= nz < self.img.shape[0] and
                                0 <= ny < self.img.shape[1] and
                                0 <= nx < self.img.shape[2]):
                            continue
                        
                        # Se già nella maschera, salta
                        if self.mask[nz, ny, nx] > 0:
                            continue
                        
                        # Check HU
                        hu_value = self.img[nz, ny, nx]
                        threshold = self._get_adaptive_threshold((nz, ny, nx))
                        
                        if hu_value < threshold:
                            # Distanza dallo skeleton
                            dist = np.sqrt(dz**2 + dy**2 + dx**2) * np.mean(self.spacing)
                            
                            if dist < 2.0:  # Max 2mm dallo skeleton
                                expanded[nz, ny, nx] = 1
                                expanded_count += 1
        
        if self.verbose:
            print(f"  Expanded {expanded_count} voxels near skeleton")
        
        return expanded
    
    def refine(self):
        """
        Pipeline completa di refinement avanzato.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("ENHANCED AIRWAY REFINEMENT")
            print("="*70)
        
        refined = self.mask.copy()
        initial_voxels = np.sum(refined > 0)
        
        # STEP 1: Region growing da endpoints
        endpoints = self._detect_endpoints_and_tips()
        if len(endpoints) > 0:
            if self.verbose:
                print(f"\n[Step 1] Region growing from {len(endpoints)} endpoints...")
            
            grown_total = 0
            for ep in endpoints:
                grown = self._priority_region_grow(
                    [ep], 
                    max_dist_mm=4.0,
                    max_voxels=500
                )
                for gz, gy, gx in grown:
                    if refined[gz, gy, gx] == 0:
                        refined[gz, gy, gx] = 1
                        grown_total += 1
            
            if self.verbose:
                print(f"  Added {grown_total} voxels from endpoint growing")
        
        # STEP 2: Recupero vie aeree sottili
        thin_airways = self._recover_thin_airways()
        refined = np.logical_or(refined, thin_airways).astype(np.uint8)
        thin_count = np.sum(thin_airways)
        
        # STEP 3: Espansione lungo skeleton
        expanded = self._expand_near_skeleton()
        refined = np.logical_or(refined, expanded).astype(np.uint8)
        expanded_count = np.sum(expanded)
        
        # STEP 4: Distance transform expansion (conservativo)
        if self.verbose:
            print("\n[Step 4] Distance-based expansion...")
        
        dt = self._compute_distance_transform()
        air_hu = self.img < self.hu_thresholds['peripheral']
        
        # Espandi solo dove: (a) vicino alla maschera E (b) HU da aria
        near_mask = (dt < 3.0) & air_hu
        candidates = near_mask & (refined == 0)
        
        refined = np.logical_or(refined, candidates).astype(np.uint8)
        dt_expanded = np.sum(candidates)
        
        if self.verbose:
            print(f"  Added {dt_expanded} voxels from distance-based expansion")
        
        # STEP 5: Smoothing morfologico leggero
        if self.verbose:
            print("\n[Step 5] Morphological smoothing...")
        
        refined = binary_dilation(refined, ball(1))
        refined = binary_erosion(refined, ball(1))
        
        self.refined = refined.astype(np.uint8)
        final_voxels = np.sum(self.refined > 0)
        added_voxels = final_voxels - initial_voxels
        
        if self.verbose:
            print(f"\n" + "="*70)
            print("REFINEMENT SUMMARY")
            print("="*70)
            print(f"Initial voxels: {initial_voxels:,}")
            print(f"Final voxels: {final_voxels:,}")
            print(f"Added: {added_voxels:,} ({added_voxels/initial_voxels*100:.1f}%)")
            print(f"  - Endpoint growing: {grown_total if 'grown_total' in locals() else 0}")
            print(f"  - Thin airways: {thin_count}")
            print(f"  - Skeleton expansion: {expanded_count}")
            print(f"  - DT expansion: {dt_expanded}")
        
        return self.refined
    
    def save(self, path, ref_img):
        """Salva maschera raffinata"""
        if self.refined is None:
            raise ValueError("Run refine() first")
        
        out = sitk.GetImageFromArray(self.refined)
        out.CopyInformation(ref_img)
        sitk.WriteImage(out, path)
        
        if self.verbose:
            print(f"\n✓ Saved refined mask: {path}")
        
        return path