import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

def safe_reconnection_mixed_vessels(
    mask_path,
    spacing,
    max_gap_mm=2.0,
    min_component_size_mm3=50.0,
    output_prefix="safe_reconnection"
):
    """
    Riconnessione SICURA per maschere con arterie+vene miste.
    
    Strategia:
    1. Identifica componenti connesse separate
    2. Per ogni componente, trova endpoint
    3. Connetti SOLO endpoint della stessa componente (o molto vicini)
    4. Valida che non si creino cicli o connessioni anatomicamente impossibili
    """
    
    # Carica maschera
    img = sitk.ReadImage(mask_path)
    mask_data = sitk.GetArrayFromImage(img)
    
    working_mask = (mask_data > 0).astype(np.uint8)
    
    # ========================================
    # STEP 1: Analisi componenti iniziali
    # ========================================
    labeled_initial, n_initial = ndimage.label(working_mask)
    
    print(f"\n{'='*60}")
    print(f"SAFE RECONNECTION - Mixed Vessels")
    print(f"{'='*60}")
    print(f"Initial components: {n_initial:,}")
    
    # Calcola dimensioni componenti
    voxel_volume = np.prod(spacing)
    sizes = ndimage.sum(working_mask, labeled_initial, range(1, n_initial + 1))
    volumes_mm3 = sizes * voxel_volume
    
    # Filtra componenti troppo piccole (probabilmente noise)
    min_size_voxels = int(min_component_size_mm3 / voxel_volume)
    valid_components = np.where(sizes >= min_size_voxels)[0] + 1
    
    print(f"Valid components (>{min_component_size_mm3:.0f}mm³): {len(valid_components)}")
    
    # ========================================
    # STEP 2: Raggruppa componenti per vicinanza spaziale
    # ========================================
    # Idea: componenti della stessa struttura vascolare 
    # tendono ad essere vicine nello spazio
    
    component_centroids = []
    component_labels = []
    
    for comp_id in valid_components:
        comp_mask = (labeled_initial == comp_id)
        coords = np.argwhere(comp_mask)
        
        if len(coords) > 0:
            centroid = np.mean(coords, axis=0) * spacing  # in mm
            component_centroids.append(centroid)
            component_labels.append(comp_id)
    
    component_centroids = np.array(component_centroids)
    
    # ========================================
    # STEP 3: Trova coppie candidate per riconnessione
    # ========================================
    # Usa KDTree sui centroidi per trovare componenti vicine
    
    if len(component_centroids) < 2:
        print("⚠️  Insufficient components for reconnection")
        return working_mask, {'connections_made': 0}
    
    # Threshold: connetti solo se centroidi sono entro 10mm
    # (componenti molto distanti = probabilmente vasi diversi)
    centroid_threshold_mm = 10.0
    
    tree = cKDTree(component_centroids)
    nearby_pairs = tree.query_pairs(r=centroid_threshold_mm)
    
    print(f"Candidate component pairs (centroids <{centroid_threshold_mm}mm): {len(nearby_pairs)}")
    
    # ========================================
    # STEP 4: Per ogni coppia, verifica endpoint
    # ========================================
    
    import kimimaro
    
    connections_made = 0
    connections_rejected = 0
    
    for idx1, idx2 in nearby_pairs:
        comp1_id = component_labels[idx1]
        comp2_id = component_labels[idx2]
        
        # Estrai maschere delle due componenti
        mask1 = (labeled_initial == comp1_id)
        mask2 = (labeled_initial == comp2_id)
        
        # Skeletonizza entrambe
        try:
            skel1 = kimimaro.skeletonize(
                mask1.astype(np.uint32),
                anisotropy=spacing,
                dust_threshold=10,
                progress=False
            )
            skel2 = kimimaro.skeletonize(
                mask2.astype(np.uint32),
                anisotropy=spacing,
                dust_threshold=10,
                progress=False
            )
        except:
            continue
        
        if not skel1 or not skel2:
            continue
        
        # Estrai endpoint
        endpoints1 = get_all_endpoints_from_skeletons(skel1, spacing)
        endpoints2 = get_all_endpoints_from_skeletons(skel2, spacing)
        
        if len(endpoints1) == 0 or len(endpoints2) == 0:
            continue
        
        # Trova la coppia di endpoint più vicina
        distances = cdist(endpoints1, endpoints2, metric='euclidean')
        min_dist = np.min(distances)
        
        # ========================================
        # VALIDAZIONE: Connetti solo se sensato
        # ========================================
        
        # Regola 1: Distanza endpoint < max_gap_mm
        if min_dist > max_gap_mm:
            connections_rejected += 1
            continue
        
        # Regola 2: Entrambe le componenti devono essere "piccole"
        # (grandi componenti connesse = probabilmente già complete)
        vol1 = volumes_mm3[comp1_id - 1]
        vol2 = volumes_mm3[comp2_id - 1]
        
        max_volume_for_connection = 5000  # mm³ (~5ml)
        if vol1 > max_volume_for_connection and vol2 > max_volume_for_connection:
            connections_rejected += 1
            continue
        
        # Regola 3: Orientazione endpoint coerente
        # (gli endpoint dovrebbero "puntare" uno verso l'altro)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        ep1 = endpoints1[min_idx[0]]
        ep2 = endpoints2[min_idx[1]]
        
        # ✅ CONNETTI!
        line_coords = bresenham_3d(
            (ep1 / spacing).astype(int),
            (ep2 / spacing).astype(int)
        )
        
        for coord in line_coords:
            z, y, x = coord
            if (0 <= z < working_mask.shape[0] and 
                0 <= y < working_mask.shape[1] and 
                0 <= x < working_mask.shape[2]):
                working_mask[z, y, x] = 1
        
        connections_made += 1
    
    # ========================================
    # STEP 5: Validazione finale
    # ========================================
    
    labeled_final, n_final = ndimage.label(working_mask)
    reduction = n_initial - n_final
    reduction_pct = (reduction / n_initial * 100) if n_initial > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Components: {n_initial:,} → {n_final:,} (-{reduction:,}, -{reduction_pct:.1f}%)")
    print(f"Connections made: {connections_made}")
    print(f"Connections rejected: {connections_rejected}")
    print(f"Safety ratio: {connections_rejected/(connections_made+connections_rejected+1e-6):.1%}")
    
    # Salva risultato
    out_path = f"{output_prefix}_safe.nii.gz"
    out_img = sitk.GetImageFromArray(working_mask)
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, out_path)
    print(f"\n✓ Saved: {out_path}")
    
    return working_mask, {
        'n_initial': n_initial,
        'n_final': n_final,
        'reduction': reduction,
        'connections_made': connections_made,
        'connections_rejected': connections_rejected
    }


def get_all_endpoints_from_skeletons(skeletons_dict, spacing):
    """Estrai tutti gli endpoint da un dict di skeleton Kimimaro."""
    # Prefer implementation from main vessel_reconnection module to avoid duplicates.
    try:
        from connection_check.vessel_reconnection_GT import get_all_endpoints_from_skeletons as _impl
        return _impl(skeletons_dict, spacing)
    except Exception:
        # Fallback to local implementation if import fails (keeps standalone behavior)
        from collections import defaultdict
        all_endpoints = []
        for skeleton in skeletons_dict.values():
            vertices = skeleton.vertices
            edges = skeleton.edges
            if len(edges) == 0:
                continue
            degree = defaultdict(int)
            for edge in edges:
                degree[edge[0]] += 1
                degree[edge[1]] += 1
            endpoint_indices = [i for i, d in degree.items() if d == 1]
            if len(endpoint_indices) > 0:
                endpoint_coords = vertices[endpoint_indices]
                endpoint_coords_zyx = endpoint_coords[:, [2, 1, 0]] * spacing
                all_endpoints.extend(endpoint_coords_zyx)
        return np.array(all_endpoints) if all_endpoints else np.array([]).reshape(0, 3)


def bresenham_3d(p1, p2):
    """Algoritmo Bresenham 3D (copia dal tuo codice originale)."""
    # Prefer using shared bresenham implementation if available to avoid duplication
    try:
        from connection_check.vessel_reconnection_GT import bresenham_3d as _impl
        return _impl(p1, p2)
    except Exception:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1
        if dx >= dy and dx >= dz:
            p1_inc = 2 * dy - dx
            p2_inc = 2 * dz - dx
            while x1 != x2:
                points.append([x1, y1, z1])
                x1 += xs
                if p1_inc >= 0:
                    y1 += ys
                    p1_inc -= 2 * dx
                if p2_inc >= 0:
                    z1 += zs
                    p2_inc -= 2 * dx
                p1_inc += 2 * dy
                p2_inc += 2 * dz
        elif dy >= dx and dy >= dz:
            p1_inc = 2 * dx - dy
            p2_inc = 2 * dz - dy
            while y1 != y2:
                points.append([x1, y1, z1])
                y1 += ys
                if p1_inc >= 0:
                    x1 += xs
                    p1_inc -= 2 * dy
                if p2_inc >= 0:
                    z1 += zs
                    p2_inc -= 2 * dy
                p1_inc += 2 * dx
                p2_inc += 2 * dz
        else:
            p1_inc = 2 * dy - dz
            p2_inc = 2 * dx - dz
            while z1 != z2:
                points.append([x1, y1, z1])
                z1 += zs
                if p1_inc >= 0:
                    y1 += ys
                    p1_inc -= 2 * dz
                if p2_inc >= 0:
                    x1 += xs
                    p2_inc -= 2 * dz
                p1_inc += 2 * dy
                p2_inc += 2 * dx
        points.append([x2, y2, z2])
        return np.array(points)

mask_path = '/content/vesselsegmentation/vessels_cleaned/1.2.840.113704.1.111.2604.1126357612.7_cleaned.nii.gz'
spacing = (0.7, 0.7, 0.7)

mask_safe, stats = safe_reconnection_mixed_vessels(
    mask_path,
    spacing,
    max_gap_mm=3,  
    min_component_size_mm3=50.0,
    output_prefix="TS_safe_reconnected"
)