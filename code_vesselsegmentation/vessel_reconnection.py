import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import ball
from collections import deque


def reconnect_isolated_vessels(vessel_mask, spacing, 
                                max_gap_mm=3.0,
                                min_isolated_size=5,
                                max_isolated_size=500,
                                max_connection_distance_mm=5.0):
    stats = {
        'original_components': 0,
        'isolated_fragments': 0,
        'reconnected_fragments': 0,
        'removed_fragments': 0,
        'total_voxels_added': 0
    }
    
    print("\n=== VESSEL RECONNECTION ===")
    original_voxels = vessel_mask.sum()
    
    # STEP 1: Morphological closing per gap piccoli
    print(f"\n[Step 1] Morphological closing (gap <= {max_gap_mm}mm)")
    closing_radius_voxels = [int(np.ceil(max_gap_mm / s)) for s in spacing]
    closing_kernel = ball(max(closing_radius_voxels))
    
    vessel_closed = ndimage.binary_closing(vessel_mask, structure=closing_kernel)
    closed_voxels = vessel_closed.sum() - original_voxels
    print(f"  Added {closed_voxels} voxels via closing")
    stats['total_voxels_added'] += closed_voxels
    
    # STEP 2: Identifica componenti connesse
    print(f"\n[Step 2] Identifying connected components")
    labeled, num_components = ndimage.label(vessel_closed)
    stats['original_components'] = num_components
    
    if num_components <= 1:
        print("  Only one component found, no reconnection needed")
        return vessel_closed, stats
    
    # Calcola dimensioni delle componenti
    component_sizes = ndimage.sum(vessel_closed, labeled, range(1, num_components + 1))
    component_sizes = np.array(component_sizes)
    
    # Identifica la componente principale (la più grande)
    main_component_label = np.argmax(component_sizes) + 1
    main_component_size = component_sizes[main_component_label - 1]
    
    print(f"  Found {num_components} components")
    print(f"  Main component: label={main_component_label}, size={main_component_size} voxels")
    
    # STEP 3: Identifica frammenti isolati da processare
    isolated_labels = []
    for label_idx in range(1, num_components + 1):
        if label_idx == main_component_label:
            continue
        size = component_sizes[label_idx - 1]
        if min_isolated_size <= size <= max_isolated_size:
            isolated_labels.append(label_idx)
    
    stats['isolated_fragments'] = len(isolated_labels)
    print(f"\n[Step 3] Found {len(isolated_labels)} isolated fragments to process")
    
    if len(isolated_labels) == 0:
        return vessel_closed, stats
    
    # STEP 4: Riconnetti frammenti isolati
    print(f"\n[Step 4] Reconnecting fragments (max distance: {max_connection_distance_mm}mm)")
    
    # Estrai coordinate del componente principale
    main_mask = (labeled == main_component_label)
    main_coords = np.array(np.where(main_mask)).T  # (N, 3)
    
    # Costruisci KD-tree per ricerca veloce
    # Converti coordinate in spazio fisico (mm)
    main_coords_mm = main_coords * np.array(spacing)
    tree = cKDTree(main_coords_mm)
    
    vessel_reconnected = vessel_closed.copy()
    
    for fragment_label in isolated_labels:
        fragment_mask = (labeled == fragment_label)
        fragment_coords = np.array(np.where(fragment_mask)).T
        fragment_coords_mm = fragment_coords * np.array(spacing)
        
        # Trova il punto del frammento più vicino al componente principale
        distances, indices = tree.query(fragment_coords_mm, k=1)
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        if min_distance <= max_connection_distance_mm:
            # Riconnetti: traccia linea tra i due punti più vicini
            fragment_point = fragment_coords[min_distance_idx]
            main_point = main_coords[indices[min_distance_idx]]
            
            # Crea path di connessione
            connection_path = create_connection_path(
                fragment_point, main_point, vessel_reconnected.shape
            )
            
            # Aggiungi il path e il frammento
            vessel_reconnected[fragment_mask] = True
            vessel_reconnected[connection_path] = True
            
            added_voxels = connection_path.sum()
            stats['reconnected_fragments'] += 1
            stats['total_voxels_added'] += added_voxels
            
            print(f"  Reconnected fragment {fragment_label}: "
                  f"distance={min_distance:.2f}mm, added={added_voxels} voxels")
        else:
            print(f"  Fragment {fragment_label} too far: {min_distance:.2f}mm > {max_connection_distance_mm}mm")
            stats['removed_fragments'] += 1
    
    final_voxels = vessel_reconnected.sum()
    print(f"\n[Summary]")
    print(f"  Original voxels: {original_voxels}")
    print(f"  Final voxels: {final_voxels}")
    print(f"  Net added: {final_voxels - original_voxels}")
    print(f"  Reconnected: {stats['reconnected_fragments']}/{stats['isolated_fragments']} fragments")
    
    return vessel_reconnected, stats


def create_connection_path(point1, point2, shape):
    # Bresenham 3D per tracciare linea
    path_coords = bresenham_3d(point1, point2)
    
    # Crea maschera
    path_mask = np.zeros(shape, dtype=bool)
    for coord in path_coords:
        z, y, x = coord
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            path_mask[z, y, x] = True
    
    # Dilata leggermente il path per renderlo più robusto (1 voxel)
    path_mask = ndimage.binary_dilation(path_mask, structure=np.ones((3, 3, 3)))
    
    return path_mask


def bresenham_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1
    
    points = []
    
    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            points.append((x1, y1, z1))
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
    
    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            points.append((x1, y1, z1))
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
    
    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            points.append((x1, y1, z1))
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
    
    points.append((x2, y2, z2))
    return points


def advanced_reconnection_with_centerlines(vessel_mask, centerlines, spacing,
                                           max_gap_mm=5.0,
                                           search_radius_mm=10.0):
    stats = {
        'endpoints_found': 0,
        'connections_made': 0,
        'voxels_added': 0
    }
    
    print("\n=== ADVANCED RECONNECTION WITH CENTERLINES ===")
    
    # Trova endpoint delle centerlines (hanno 1 solo vicino)
    endpoint_mask = find_centerline_endpoints(centerlines)
    endpoint_coords = np.array(np.where(endpoint_mask)).T
    stats['endpoints_found'] = len(endpoint_coords)
    
    print(f"Found {len(endpoint_coords)} centerline endpoints")
    
    if len(endpoint_coords) < 2:
        return vessel_mask, stats
    
    # Converti in coordinate fisiche
    endpoint_coords_mm = endpoint_coords * np.array(spacing)
    
    # Costruisci KD-tree
    tree = cKDTree(endpoint_coords_mm)
    
    vessel_reconnected = vessel_mask.copy()
    
    # Per ogni endpoint, cerca endpoint vicini
    for i, coord_mm in enumerate(endpoint_coords_mm):
        # Cerca endpoint entro search_radius
        indices = tree.query_ball_point(coord_mm, search_radius_mm)
        
        for j in indices:
            if i >= j:  # Evita duplicati
                continue
            
            distance = np.linalg.norm(endpoint_coords_mm[i] - endpoint_coords_mm[j])
            
            if distance <= max_gap_mm:
                # Crea connessione
                path = create_connection_path(
                    endpoint_coords[i], 
                    endpoint_coords[j],
                    vessel_mask.shape
                )
                vessel_reconnected[path] = True
                stats['connections_made'] += 1
                stats['voxels_added'] += path.sum()
                
                print(f"  Connected endpoints {i} <-> {j}: distance={distance:.2f}mm")
    
    print(f"\n[Summary] Made {stats['connections_made']} connections, added {stats['voxels_added']} voxels")
    
    return vessel_reconnected, stats


def find_centerline_endpoints(centerlines):
    # Conta vicini per ogni voxel della centerline
    neighbor_kernel = np.ones((3, 3, 3), dtype=int)
    neighbor_kernel[1, 1, 1] = 0  # Escludi il voxel centrale
    
    neighbor_count = ndimage.convolve(
        centerlines.astype(int), 
        neighbor_kernel, 
        mode='constant', 
        cval=0
    )
    
    # Endpoint = centerline voxel con 1 solo vicino
    endpoint_mask = (centerlines) & (neighbor_count == 1)
    
    return endpoint_mask