import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import ball, binary_closing, skeletonize
from skimage.measure import label as sk_label
import kimimaro
from collections import defaultdict

def get_skeleton_endpoints_kimimaro(skeleton):
    """
    Trova gli endpoint dello skeleton usando l'output di Kimimaro.
    Gli endpoint sono punti con esattamente 1 vicino nello skeleton graph.
    """
    if isinstance(skeleton, kimimaro.Skeleton):
        # Estrai i punti dello skeleton e la topologia
        vertices = skeleton.vertices
        edges = skeleton.edges
        
        # Calcola il grado di ogni vertice (numero di connessioni)
        degree = defaultdict(int)
        for edge in edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        
        # Endpoint: vertici con grado 1
        endpoint_indices = [idx for idx, deg in degree.items() if deg == 1]
        endpoint_coords = vertices[endpoint_indices]
        
        return endpoint_coords, endpoint_indices
    
    else:
        raise ValueError("Input deve essere un oggetto Skeleton di Kimimaro")

def skeletonize_with_kimimaro(mask, spacing):
    """
    Skeletonizzazione 3D robusta usando Kimimaro.
    """
    
    # Kimimaro si aspetta (x,y,z) ma noi abbiamo (z,y,x)
    # Dobbiamo adattare le coordinate
    kimimaro_scale = spacing[::-1]  # Converti a (x,y,z)
    
    # Skeletonizza con Kimimaro - parametri corretti
    try:
        skeletons = kimimaro.skeletonize(
            mask.astype(np.uint8),
            # Kimimaro moderno usa 'voxel_size' invece di 'scale'
            voxel_size=kimimaro_scale,
            dust_threshold=50,  # Rimuovi componenti piccole
            parallel=1,  # Numero di CPU
            fix_branching=True,
            fix_borders=True,
            progress=True
        )
        return skeletons
    except TypeError as e:
        # Fallback per vecchie versioni di Kimimaro
        if "unexpected keyword argument 'voxel_size'" in str(e):
            skeletons = kimimaro.skeletonize(
                mask.astype(np.uint8),
                scale=kimimaro_scale,
                dust_threshold=50,
                parallel=1,
                fix_branching=True,
                fix_borders=True,
                progress=True
            )
            return skeletons
        else:
            raise e

def connect_nearby_endpoints(mask, spacing, max_gap_mm=3.0, min_fragment_size=50):
    """
    Connette SOLO endpoint vicini usando lo skeleton di Kimimaro.
    Molto più conservativo del distance filling.
    """
    
    working_mask = mask.copy()
    
    # 1. Calcola skeleton con Kimimaro
    try:
        skeletons = skeletonize_with_kimimaro(working_mask > 0, spacing)
    except Exception as e:
            # Kimimaro failed: falling back to scikit-image
        return connect_nearby_endpoints_fallback(working_mask, spacing, max_gap_mm)
    
    if not skeletons:
        return working_mask, {'connections_made': 0}
    
    # 2. Estrai endpoint da tutti gli skeleton
    all_endpoint_coords = []
    for skeleton_id, skeleton in skeletons.items():
        try:
            endpoint_coords, _ = get_skeleton_endpoints_kimimaro(skeleton)
            if len(endpoint_coords) > 0:
                # Converti coordinate Kimimaro (x,y,z) al nostro sistema (z,y,x)
                endpoint_coords_zyx = endpoint_coords[:, [2, 1, 0]].astype(int)
                all_endpoint_coords.extend(endpoint_coords_zyx)
        except Exception as e:
            # Warning: failed to process skeleton
            continue
    
    if len(all_endpoint_coords) == 0:
        return working_mask, {'connections_made': 0}
    
    all_endpoint_coords = np.array(all_endpoint_coords)
    
    # 3. Filtra endpoint che sono dentro la maschera
    valid_endpoints = []
    for coord in all_endpoint_coords:
        z, y, x = coord
        if (0 <= z < working_mask.shape[0] and 
            0 <= y < working_mask.shape[1] and 
            0 <= x < working_mask.shape[2]):
            valid_endpoints.append(coord)
    
    valid_endpoints = np.array(valid_endpoints)
    
    if len(valid_endpoints) == 0:
        return working_mask, {'connections_made': 0}
    
    # 4. Converti coordinate in mm per distance threshold
    endpoint_coords_mm = valid_endpoints * spacing
    
    # 5. Trova coppie di endpoint vicini usando KDTree
    tree = cKDTree(endpoint_coords_mm)
    pairs = tree.query_pairs(r=max_gap_mm)
    
    if len(pairs) == 0:
        return working_mask, {'connections_made': 0}
    
    # 6. Per ogni coppia, traccia una linea di connessione
    connections_made = 0
    for idx1, idx2 in pairs:
        p1 = valid_endpoints[idx1]
        p2 = valid_endpoints[idx2]
        
        # Traccia linea 3D tra i due endpoint
        line_coords = bresenham_3d(p1, p2)
        
        # Dilata leggermente la linea per assicurare connessione
        for coord in line_coords:
            z, y, x = coord
            if (0 <= z < working_mask.shape[0] and 
                0 <= y < working_mask.shape[1] and 
                0 <= x < working_mask.shape[2]):
                working_mask[z, y, x] = mask.max()  # Usa lo stesso valore della maschera
        
        connections_made += 1
    
    
    
    # 7. Validazione: conta frammenti prima/dopo
    labeled_before, n_before = ndimage.label(mask > 0)
    labeled_after, n_after = ndimage.label(working_mask > 0)
    
    reduction_pct = (1 - n_after/n_before) * 100 if n_before > 0 else 0
    
    return working_mask, {
        'fragments_before': n_before,
        'fragments_after': n_after,
        'reduction_pct': reduction_pct,
        'connections_made': connections_made,
        'skeletons_found': len(skeletons)
    }

def connect_nearby_endpoints_fallback(mask, spacing, max_gap_mm=3.0):
    """
    Fallback usando skeleton binario tradizionale di scikit-image.
    """
    
    working_mask = mask.copy()
    
    # Skeletonizzazione con scikit-image
    try:
        # skeletonize funziona su immagini 2D, dobbiamo applicarlo slice per slice
        skeleton_binary = np.zeros_like(working_mask, dtype=bool)
        for z in range(working_mask.shape[0]):
            slice_2d = working_mask[z] > 0
            if np.any(slice_2d):
                skeleton_binary[z] = skeletonize(slice_2d)
    except Exception as e:
        # Scikit-image skeletonize failed
        return working_mask, {'connections_made': 0}
    
    # Trova endpoint dallo skeleton binario
    def get_skeleton_endpoints_binary(skeleton):
        struct = ndimage.generate_binary_structure(3, 3)
        neighbor_count = ndimage.convolve(skeleton.astype(int), struct.astype(int), mode='constant')
        endpoints = skeleton & (neighbor_count == 2)
        return endpoints
    
    endpoints = get_skeleton_endpoints_binary(skeleton_binary)
    endpoint_coords = np.argwhere(endpoints)
    
    if len(endpoint_coords) == 0:
        return working_mask, {'connections_made': 0}
    
    # Resto del codice identico...
    endpoint_coords_mm = endpoint_coords * spacing
    tree = cKDTree(endpoint_coords_mm)
    pairs = tree.query_pairs(r=max_gap_mm)
    
    
    if len(pairs) == 0:
        return working_mask, {'connections_made': 0}
    
    connections_made = 0
    for idx1, idx2 in pairs:
        p1 = endpoint_coords[idx1]
        p2 = endpoint_coords[idx2]
        
        line_coords = bresenham_3d(p1, p2)
        
        for coord in line_coords:
            z, y, x = coord
            if (0 <= z < working_mask.shape[0] and 
                0 <= y < working_mask.shape[1] and 
                0 <= x < working_mask.shape[2]):
                working_mask[z, y, x] = mask.max()
        
        connections_made += 1
    
    
    labeled_before, n_before = ndimage.label(mask > 0)
    labeled_after, n_after = ndimage.label(working_mask > 0)
    
    reduction_pct = (1 - n_after/n_before) * 100 if n_before > 0 else 0
    
    return working_mask, {
        'fragments_before': n_before,
        'fragments_after': n_after,
        'reduction_pct': reduction_pct,
        'connections_made': connections_made
    }

def bresenham_3d(p1, p2):
    """
    Algoritmo di Bresenham 3D per tracciare linea tra due punti.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1
    
    # Driving axis is X
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
    
    # Driving axis is Y
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
    
    # Driving axis is Z
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

def create_skeleton_mask_only(
    mask_path,
    label_value=1,
    spacing=None,
    max_gap_mm=3,
    min_component_size_mm3=1.0,
    output_prefix="test_skeleton"
):
    """
    Crea SOLO la maschera skeleton (test_skeleton_label1_skeleton).
    """
    # Carica l'immagine
    img = sitk.ReadImage(mask_path)
    mask_data = sitk.GetArrayFromImage(img)
    
    if spacing is None:
        spacing = img.GetSpacing()[::-1]
    
    
    # Estrai solo la maschera per il label specificato
    if label_value is not None:
        working_mask = (mask_data == label_value).astype(np.uint8) * label_value
    else:
        working_mask = mask_data
    
    # Applica skeleton-guided endpoint connection
    mask_reconnected, stats = connect_nearby_endpoints(
        working_mask, spacing, max_gap_mm
    )
    
    # Analisi componenti connesse
    mask_cleaned, comp_stats = analyze_connected_components(
        mask_reconnected, spacing, label_value, min_component_size_mm3
    )
    
    # Salva la maschera
    out_path = f"{output_prefix}_label{label_value}_skeleton.nii.gz"
    out_img = sitk.GetImageFromArray(mask_cleaned.astype(np.uint8))
    out_img.CopyInformation(img)
    sitk.WriteImage(out_img, out_path)
    
    
    return mask_cleaned, {'reconnection': stats, 'components': comp_stats}

def analyze_connected_components(mask, spacing, label_value=None, min_size_mm3=1.0):
    """Analizza componenti connesse."""
    if label_value is not None:
        working_mask = (mask == label_value)
        label_name = f"Label_{label_value}"
    else:
        working_mask = (mask > 0)
        label_name = "Binary_mask"
    
    labeled, n_components = ndimage.label(working_mask)
    
    if n_components == 0:
        return mask, {}
    
    voxel_volume_mm3 = np.prod(spacing)
    sizes = ndimage.sum(working_mask, labeled, range(1, n_components + 1))
    volumes_mm3 = sizes * voxel_volume_mm3
    
    sorted_indices = np.argsort(volumes_mm3)[::-1]
    sorted_volumes = volumes_mm3[sorted_indices]
    
    for i in range(min(10, len(sorted_volumes))):
        vol = sorted_volumes[i]
        pct = vol / np.sum(volumes_mm3) * 100
    
    min_size_voxels = int(min_size_mm3 / voxel_volume_mm3)
    mask_sizes = sizes >= min_size_voxels
    
    if n_components > 0:
        cleaned_binary = mask_sizes[labeled - 1]
    else:
        cleaned_binary = working_mask
    
    if label_value is not None:
        cleaned_mask = mask.copy()
        cleaned_mask[cleaned_binary] = label_value
        cleaned_mask[~cleaned_binary & (mask == label_value)] = 0
    else:
        cleaned_mask = cleaned_binary.astype(mask.dtype)
    
    n_removed = n_components - ndimage.label(cleaned_binary)[1]
    
    
    return cleaned_mask, {
        'n_components': n_components,
        'total_volume_mm3': np.sum(volumes_mm3),
        'largest_component_mm3': sorted_volumes[0] if len(sorted_volumes) > 0 else 0,
        'n_removed': n_removed
    }

    
# Crea SOLO la maschera skeleton
mask_path = '/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.2604.1126357612_fullAnnotations.mhd'
spacing = (0.7, 0.7, 0.7)
mask_skeleton, stats_skeleton = create_skeleton_mask_only(
    mask_path,
    label_value=2,  # vene
    spacing=spacing,
    max_gap_mm=2.5,
    output_prefix="test_skeleton"
)

