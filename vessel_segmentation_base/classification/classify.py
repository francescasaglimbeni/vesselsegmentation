import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from collections import deque

def _get_neighbors(coord):
    z, y, x = coord
    neighbors = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                neighbors.append((z + dz, y + dy, x + dx))
    return neighbors


def _is_valid_voxel(coord, shape):
    z, y, x = coord
    return (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2])


def region_growing(vessel_mask, seed_artery, seed_vein, spacing=(1.0, 1.0, 1.0),
                   max_iterations=1000, distance_threshold_mm=5.0):
    artery_mask = seed_artery.copy()
    vein_mask = seed_vein.copy()

    artery_queue = deque()
    vein_queue = deque()

    artery_coords = np.argwhere(seed_artery)
    vein_coords = np.argwhere(seed_vein)

    for coord in artery_coords:
        artery_queue.append((0.0, tuple(coord)))
    for coord in vein_coords:
        vein_queue.append((0.0, tuple(coord)))

    visited = artery_mask | vein_mask
    iteration = 0
    spacing = np.array(spacing)
    shape = vessel_mask.shape

    while (artery_queue or vein_queue) and iteration < max_iterations:
        iteration += 1
        
        if artery_queue:
            dist, coord = artery_queue.popleft()
            if dist > distance_threshold_mm:
                pass
            else:
                for neighbor in _get_neighbors(coord):
                    if not _is_valid_voxel(neighbor, shape):
                        continue
                    if visited[neighbor]:
                        continue
                    if not vessel_mask[neighbor]:
                        continue

                    phys_dist = np.linalg.norm((np.array(neighbor) - np.array(coord)) * spacing)
                    new_dist = dist + phys_dist
                    if new_dist <= distance_threshold_mm:
                        artery_mask[neighbor] = True
                        visited[neighbor] = True
                        artery_queue.append((new_dist, neighbor))

        if vein_queue:
            dist, coord = vein_queue.popleft()
            if dist > distance_threshold_mm:
                pass
            else:
                for neighbor in _get_neighbors(coord):
                    if not _is_valid_voxel(neighbor, shape):
                        continue
                    if visited[neighbor]:
                        continue
                    if not vessel_mask[neighbor]:
                        continue

                    phys_dist = np.linalg.norm((np.array(neighbor) - np.array(coord)) * spacing)
                    new_dist = dist + phys_dist
                    if new_dist <= distance_threshold_mm:
                        vein_mask[neighbor] = True
                        visited[neighbor] = True
                        vein_queue.append((new_dist, neighbor))

        if iteration % 1000 == 0:
            print(f"  Iteration {iteration}: Artery={artery_mask.sum()}, Vein={vein_mask.sum()}")

    return artery_mask, vein_mask


def compute_anatomical_features(unclassified_mask, airway_mask, lung_mask, spacing=(1.0, 1.0, 1.0)):
    features = {}
    spacing = tuple(spacing)

    # Distance to airways
    if airway_mask is not None and airway_mask.any():
        print("  Computing distance to airways...")
        dist_to_airways = ndimage.distance_transform_edt(~airway_mask, sampling=spacing)
        features['dist_to_airways'] = dist_to_airways
    else:
        features['dist_to_airways'] = np.zeros_like(unclassified_mask, dtype=float)

    # Distance to pleura (approx as lung border)
    print("  Computing distance to pleura...")
    lung_eroded = ndimage.binary_erosion(lung_mask, iterations=3)
    lung_border = lung_mask & ~lung_eroded
    dist_to_pleura = ndimage.distance_transform_edt(~lung_border, sampling=spacing)
    features['dist_to_pleura'] = dist_to_pleura

    # Centrality
    print("  Computing centrality features...")
    lung_center = ndimage.distance_transform_edt(lung_mask, sampling=spacing)
    features['centrality'] = lung_center / (lung_center.max() + 1e-8)

    return features


def classify_by_features(unclassified_mask, features, confidence_threshold=0.6):
    dist_airways_norm = features['dist_to_airways'] / (features['dist_to_airways'].max() + 1e-8)
    dist_pleura_norm = features['dist_to_pleura'] / (features['dist_to_pleura'].max() + 1e-8)
    centrality = features['centrality']

    artery_score = (1.0 - dist_airways_norm) * 0.6 + centrality * 0.4
    vein_score = (1.0 - dist_pleura_norm) * 0.6 + (1.0 - centrality) * 0.4

    confident_artery = unclassified_mask & (artery_score > vein_score) & (artery_score > confidence_threshold)
    confident_vein = unclassified_mask & (vein_score > artery_score) & (vein_score > confidence_threshold)

    return confident_artery, confident_vein


def cleanup_classification(mask, min_size=50):
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    keep_labels = np.where(np.array(sizes) >= min_size)[0] + 1
    cleaned = np.isin(labeled, keep_labels)
    removed = mask.sum() - cleaned.sum()
    if removed > 0:
        print(f"  Removed {removed} voxels from {num_features - len(keep_labels)} small components")
    return cleaned


def save_results(artery_mask, vein_mask, unclassified_mask, vessel_mask, output_dir, reference_img_path):
    os.makedirs(output_dir, exist_ok=True)
    ref_img = sitk.ReadImage(reference_img_path)

    if artery_mask is not None:
        artery_img = sitk.GetImageFromArray(artery_mask.astype(np.uint8))
        artery_img.CopyInformation(ref_img)
        artery_path = os.path.join(output_dir, "arteries.nii.gz")
        sitk.WriteImage(artery_img, artery_path)
        print(f"✓ Saved: {artery_path}")

    if vein_mask is not None:
        vein_img = sitk.GetImageFromArray(vein_mask.astype(np.uint8))
        vein_img.CopyInformation(ref_img)
        vein_path = os.path.join(output_dir, "veins.nii.gz")
        sitk.WriteImage(vein_img, vein_path)
        print(f"✓ Saved: {vein_path}")

    if unclassified_mask is not None and unclassified_mask.sum() > 0:
        unclass_img = sitk.GetImageFromArray(unclassified_mask.astype(np.uint8))
        unclass_img.CopyInformation(ref_img)
        unclass_path = os.path.join(output_dir, "unclassified.nii.gz")
        sitk.WriteImage(unclass_img, unclass_path)
        print(f"✓ Saved: {unclass_path}")

    combined = np.zeros_like(vessel_mask, dtype=np.uint8)
    if artery_mask is not None:
        combined[artery_mask] = 1
    if vein_mask is not None:
        combined[vein_mask] = 2
    if unclassified_mask is not None:
        combined[unclassified_mask] = 3

    combined_img = sitk.GetImageFromArray(combined)
    combined_img.CopyInformation(ref_img)
    combined_path = os.path.join(output_dir, "classification_combined.nii.gz")
    sitk.WriteImage(combined_img, combined_path)
    print(f"✓ Saved combined visualization: {combined_path}")
    print("  (1=Arteries, 2=Veins, 3=Unclassified)")


def classify_vessels(vessel_path, seed_artery_path, seed_vein_path,
                     output_dir, airway_path=None, lung_mask_path=None,
                     max_iterations=1000, distance_threshold_mm=5.0,
                     use_anatomical_rules=True, confidence_threshold=0.6):
    """
    Funzione helper per classificare vasi da file.
    
    Args:
        vessel_path: Path alla maschera dei vasi puliti
        seed_artery_path: Path al seed delle arterie
        seed_vein_path: Path al seed delle vene
        output_dir: Directory output
        airway_path: Path alla maschera delle vie aeree (opzionale)
        lung_mask_path: Path alla maschera polmonare (opzionale)
        max_iterations: Iterazioni massime per region growing
        distance_threshold_mm: Soglia distanza per propagazione (mm)
        use_anatomical_rules: Usa regole anatomiche
        confidence_threshold: Soglia confidenza classificazione
    """
    # Carica dati
    print("Loading data...")
    vessel_img = sitk.ReadImage(vessel_path)
    vessel_mask = sitk.GetArrayFromImage(vessel_img).astype(bool)
    spacing = vessel_img.GetSpacing()[::-1]  # (z, y, x)
    
    seed_artery = sitk.GetArrayFromImage(sitk.ReadImage(seed_artery_path)).astype(bool)
    seed_vein = sitk.GetArrayFromImage(sitk.ReadImage(seed_vein_path)).astype(bool)
    
    airway_mask = None
    if airway_path and os.path.exists(airway_path):
        airway_mask = sitk.GetArrayFromImage(sitk.ReadImage(airway_path)).astype(bool)
    
    lung_mask = None
    if lung_mask_path and os.path.exists(lung_mask_path):
        lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_path)).astype(bool)
    
    # Classifica usando funzioni procedurali
    artery_grown, vein_grown = region_growing(
        vessel_mask=vessel_mask,
        seed_artery=seed_artery,
        seed_vein=seed_vein,
        spacing=spacing,
        max_iterations=max_iterations,
        distance_threshold_mm=distance_threshold_mm
    )

    unclassified = vessel_mask & ~artery_grown & ~vein_grown

    if use_anatomical_rules and unclassified.sum() > 0:
        features = compute_anatomical_features(unclassified, airway_mask, lung_mask, spacing=spacing)
        artery_from_rules, vein_from_rules = classify_by_features(unclassified, features, confidence_threshold)
        artery_grown = artery_grown | artery_from_rules
        vein_grown = vein_grown | vein_from_rules

    artery_mask = cleanup_classification(artery_grown, min_size=50)
    vein_mask = cleanup_classification(vein_grown, min_size=50)
    unclassified_mask = vessel_mask & ~artery_mask & ~vein_mask

    # Salva risultati
    save_results(artery_mask, vein_mask, unclassified_mask, vessel_mask, output_dir, vessel_path)

    return artery_mask, vein_mask, unclassified_mask


vessel_path = "/content/vesselsegmentation/vessels_cleaned/1.2.840.113704.1.111.2604.1126357612.7_cleaned.nii.gz"
seed_artery_path = "/content/vesselsegmentation/vessels_cleaned/1.2.840.113704.1.111.2604.1126357612.7_seed_artery.nii.gz"
seed_vein_path = "/content/vesselsegmentation/vessels_cleaned/1.2.840.113704.1.111.2604.1126357612.7_seed_vein.nii.gz"
airway_path = "/content/vesselsegmentation/vessels_cleaned/airways_full.nii.gz"
lung_mask_path = "/content/vesselsegmentation/vessels_cleaned/lung_mask_eroded.nii.gz"
output_dir = "output_classification"

classify_vessels(
    vessel_path,
    seed_artery_path,
    seed_vein_path,
    output_dir,
    airway_path=airway_path,
    lung_mask_path=lung_mask_path,
    max_iterations=2000,
    distance_threshold_mm=7.0,
    use_anatomical_rules=True,
    confidence_threshold=0.65
)
