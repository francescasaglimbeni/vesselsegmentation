import numpy as np
import SimpleITK as sitk
import kimimaro
from collections import defaultdict
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
try:
    from skimage.morphology import skeletonize_3d
    _HAS_SKELETONIZE_3D = True
except Exception:
    # Fallback: use 2D skeletonize per-slice (works on binary masks, less topologically-accurate)
    from skimage.morphology import skeletonize
    _HAS_SKELETONIZE_3D = False

def analyze_skeleton_topology(vessel_mask, spacing, output_prefix="debug"):
    """
    Analisi dettagliata della topologia dello skeleton per debug biforcazioni.
    """
    print(f"\n{'='*70}")
    print("SKELETON TOPOLOGY DEBUG")
    print(f"{'='*70}")
    
    # Skeletonizza
    lbl = vessel_mask.astype(np.uint32)
    skels = kimimaro.skeletonize(
        lbl,
        anisotropy=spacing,
        dust_threshold=10,
        fix_branching=True,
        progress=False
    )
    
    print(f"\nNumber of skeleton objects: {len(skels)}")
    
    all_vertex_degrees = []
    bifurcation_count = 0
    
    for skel_idx, skel_obj in enumerate(skels.values()):
        print(f"\n--- Skeleton Object {skel_idx + 1} ---")
        
        if not hasattr(skel_obj, 'vertices') or len(skel_obj.vertices) == 0:
            print("  No vertices")
            continue
        
        vertices = skel_obj.vertices
        print(f"  Vertices: {len(vertices)}")
        
        if not hasattr(skel_obj, 'edges') or len(skel_obj.edges) == 0:
            print("  No edges - this is just isolated points!")
            continue
        
        edges = skel_obj.edges
        print(f"  Edges: {len(edges)}")
        
        # Analizza degree distribution
        vertex_degree = defaultdict(int)
        for edge in edges:
            vertex_degree[edge[0]] += 1
            vertex_degree[edge[1]] += 1
        
        degrees = list(vertex_degree.values())
        all_vertex_degrees.extend(degrees)
        
        print(f"  Vertex degree distribution:")
        degree_hist = defaultdict(int)
        for d in degrees:
            degree_hist[d] += 1
        
        for degree in sorted(degree_hist.keys()):
            count = degree_hist[degree]
            vertex_type = ""
            if degree == 1:
                vertex_type = " (endpoints)"
            elif degree == 2:
                vertex_type = " (normal path)"
            elif degree >= 3:
                vertex_type = f" (BIFURCATIONS!)"
                bifurcation_count += count
            
            print(f"    Degree {degree}: {count} vertices{vertex_type}")
        
        # Identifica biforcazioni in questo oggetto
        bifurcations_in_obj = [v for v, d in vertex_degree.items() if d >= 3]
        print(f"  Bifurcations in this object: {len(bifurcations_in_obj)}")
        
        if len(bifurcations_in_obj) > 0:
            print(f"  Bifurcation vertex indices: {bifurcations_in_obj[:10]}")  # primi 10
    
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total skeleton objects: {len(skels)}")
    print(f"Total bifurcations across all objects: {bifurcation_count}")
    
    if len(all_vertex_degrees) > 0:
        print(f"\nOverall vertex degree statistics:")
        print(f"  Min degree: {min(all_vertex_degrees)}")
        print(f"  Max degree: {max(all_vertex_degrees)}")
        print(f"  Mean degree: {np.mean(all_vertex_degrees):.2f}")
        print(f"  Median degree: {np.median(all_vertex_degrees):.0f}")
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(all_vertex_degrees, bins=range(1, max(all_vertex_degrees) + 2), 
                 alpha=0.7, edgecolor='black')
        plt.xlabel('Vertex Degree')
        plt.ylabel('Count')
        plt.title('Vertex Degree Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_prefix}_degree_distribution.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved degree distribution plot: {output_prefix}_degree_distribution.png")
        plt.close()
    
    return bifurcation_count


def visualize_skeleton_structure(vessel_mask, spacing, max_component_to_show=5):
    """
    Visualizza la struttura delle componenti connesse.
    """
    print(f"\n{'='*70}")
    print("CONNECTED COMPONENTS ANALYSIS")
    print(f"{'='*70}")
    
    labeled, num_components = ndimage.label(vessel_mask)
    print(f"Number of connected components: {num_components}")
    
    if num_components == 0:
        print("No components found!")
        return
    
    # Dimensioni componenti
    sizes = ndimage.sum(vessel_mask, labeled, range(1, num_components + 1))
    sizes = np.array(sizes)
    
    print(f"\nComponent size statistics:")
    print(f"  Min: {sizes.min():.0f} voxels")
    print(f"  Max: {sizes.max():.0f} voxels")
    print(f"  Mean: {sizes.mean():.0f} voxels")
    print(f"  Median: {np.median(sizes):.0f} voxels")
    
    # Mostra le componenti più grandi
    sorted_indices = np.argsort(sizes)[::-1]  # ordine decrescente
    print(f"\nLargest {min(max_component_to_show, num_components)} components:")
    for i in range(min(max_component_to_show, num_components)):
        comp_label = sorted_indices[i] + 1
        comp_size = sizes[sorted_indices[i]]
        print(f"  Component {comp_label}: {comp_size:.0f} voxels")
    
    # Plot distribuzione dimensioni
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(sizes, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Component Size (voxels)')
    plt.ylabel('Count')
    plt.title('Component Size Distribution')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(sizes[sizes > 50], bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel('Component Size (voxels)')
    plt.ylabel('Count')
    plt.title('Component Size Distribution (> 50 voxels)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('component_size_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved component distribution plot: component_size_distribution.png")
    plt.close()


def check_skeleton_quality(vessel_mask, spacing):
    # Kimimaro skeleton
    lbl = vessel_mask.astype(np.uint32)
    skels = kimimaro.skeletonize(
        lbl,
        anisotropy=spacing,
        dust_threshold=10,
        fix_branching=True,
        progress=False
    )
    
    total_vertices = sum(len(s.vertices) for s in skels.values() if hasattr(s, 'vertices'))
    total_edges = sum(len(s.edges) for s in skels.values() if hasattr(s, 'edges'))
    
    print(f"\nKimimaro skeletonization:")
    print(f"  Total vertices: {total_vertices}")
    print(f"  Total edges: {total_edges}")
    print(f"  Skeleton objects: {len(skels)}")
    
    # Controlla se edges sono presenti
    objects_without_edges = sum(1 for s in skels.values() if not hasattr(s, 'edges') or len(s.edges) == 0)
    print(f"  Objects without edges: {objects_without_edges}")
    
    if objects_without_edges == len(skels):
        print("\n  ⚠️  WARNING: NO skeleton objects have edges!")
        print("  This means the skeleton is just isolated points without connections.")
        print("  Bifurcations cannot be detected without edges.")
        print("\n  Possible causes:")
        print("  - Components too small/fragmented")
        print("  - dust_threshold too high")
        print("  - Vessel mask needs more aggressive reconnection")
    
    return objects_without_edges == 0

mask_path = '/content/vesselsegmentation/CARVE14/1.2.840.113704.1.111.2604.1126357612_fullAnnotations.mhd'
spacing = (0.7, 0.7, 0.7)

print("="*70)
print("BIFURCATION DETECTION DEBUG TOOL")
print("="*70)
print(f"\nAnalyzing: {mask_path}")

# Carica maschera
img = sitk.ReadImage(mask_path)
mask_data = sitk.GetArrayFromImage(img)

# Analizza ogni label
for label in [1, 2]:  # Arterie e Vene
    label_name = "Arteries" if label == 1 else "Veins"
    mask = (mask_data == label)
    
    if not mask.any():
        print(f"\n{label_name}: No voxels found - skipping")
        continue
    
    print(f"\n\n")
    print("="*70)
    print(f"ANALYZING: {label_name}")
    print("="*70)
    
    # 1. Analizza componenti connesse
    visualize_skeleton_structure(mask, spacing)
    
    # 2. Controlla qualità skeleton
    has_edges = check_skeleton_quality(mask, spacing)
    
    # 3. Analizza topologia dettagliata
    if has_edges:
        bifurcation_count = analyze_skeleton_topology(
            mask, spacing, 
            output_prefix=f"debug_{label_name.lower()}"
        )
    else:
        print(f"\n⚠️  Skipping topology analysis - no edges in skeleton")

print(f"\n{'='*70}")
print("DEBUG ANALYSIS COMPLETE")
print(f"{'='*70}")