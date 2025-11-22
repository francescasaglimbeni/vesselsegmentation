"""
Script standalone per testare la rimozione della trachea.
Identifica la carina (biforcazione con diametro maggiore) e rimuove tutto il tessuto superiore.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skan import Skeleton, summarize
import os
import datetime


def load_airway_mask(mask_path):
    """Carica la maschera delle vie aeree"""
    print(f"Loading mask from: {mask_path}")
    sitk_image = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_image)
    spacing = sitk_image.GetSpacing()
    
    print(f"  Spacing (x,y,z): {spacing} mm")
    print(f"  Shape (z,y,x): {mask.shape}")
    print(f"  Positive voxels: {np.sum(mask > 0):,}")
    
    return mask, sitk_image, spacing


def compute_skeleton_and_distances(mask, spacing):
    """Calcola lo skeleton 3D e la distance transform"""
    print("\n=== Computing skeleton ===")
    
    binary_mask = (mask > 0).astype(np.uint8)
    
    print("  Skeletonizing (may take a few minutes)...")
    skeleton = skeletonize(binary_mask)
    print(f"  Skeleton voxels: {np.sum(skeleton > 0):,}")
    
    print("  Computing distance transform...")
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    distance_transform = distance_transform_edt(binary_mask, sampling=spacing_zyx)
    
    return skeleton, distance_transform


def build_skeleton_graph(skeleton, distance_transform, spacing):
    """Costruisce il grafo dello skeleton con informazioni sui diametri"""
    print("\n=== Building skeleton graph ===")
    
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    skeleton_obj = Skeleton(skeleton, spacing=spacing_zyx)
    branch_data = summarize(skeleton_obj)
    
    print(f"  Branches identified: {len(branch_data)}")
    print(f"  Nodes identified: {len(skeleton_obj.coordinates)}")
    
    # Crea dizionario nodi con informazioni
    nodes_info = {}
    coordinates = skeleton_obj.coordinates
    
    for idx in range(len(coordinates)):
        pos = coordinates[idx]
        z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
        
        # Ottieni diametro alla posizione del nodo
        if (0 <= z < distance_transform.shape[0] and
            0 <= y < distance_transform.shape[1] and
            0 <= x < distance_transform.shape[2]):
            diameter = distance_transform[z, y, x] * 2  # diameter = 2 * radius
        else:
            diameter = 0
        
        nodes_info[idx] = {
            'pos': pos,
            'z': z,
            'y': y,
            'x': x,
            'diameter': diameter,
            'degree': 0  # Sarà calcolato dopo
        }
    
    # Calcola il grado di ogni nodo (numero di connessioni)
    for _, row in branch_data.iterrows():
        node_src = int(row['node-id-src'])
        node_dst = int(row['node-id-dst'])
        
        if node_src in nodes_info:
            nodes_info[node_src]['degree'] += 1
        if node_dst in nodes_info:
            nodes_info[node_dst]['degree'] += 1
    
    # Calcola diametri medi per ogni branch
    branch_diameters = {}
    for idx, row in branch_data.iterrows():
        try:
            coords = skeleton_obj.path_coordinates(idx)
            diameters = []
            for coord in coords:
                z, y, x = int(coord[0]), int(coord[1]), int(coord[2])
                if (0 <= z < distance_transform.shape[0] and
                    0 <= y < distance_transform.shape[1] and
                    0 <= x < distance_transform.shape[2]):
                    diameters.append(distance_transform[z, y, x] * 2)
            
            branch_diameters[idx] = np.mean(diameters) if diameters else 0
        except:
            branch_diameters[idx] = 0
    
    return skeleton_obj, branch_data, nodes_info, branch_diameters


def identify_carina(nodes_info, branch_data, branch_diameters):
    """
    Identifica la carina come la biforcazione con il diametro maggiore.
    La carina è tipicamente un nodo con grado >= 3 (biforcazione o triforcazione).
    """
    print("\n=== Identifying carina ===")
    
    # Trova tutti i nodi con grado >= 3 (biforcazioni)
    bifurcation_nodes = [node_id for node_id, info in nodes_info.items() 
                         if info['degree'] >= 3]
    
    print(f"  Found {len(bifurcation_nodes)} bifurcation nodes (degree >= 3)")
    
    if len(bifurcation_nodes) == 0:
        print("  WARNING: No bifurcations found!")
        # Fallback: usa il nodo con grado maggiore
        max_degree_node = max(nodes_info.items(), key=lambda x: x[1]['degree'])
        carina_node = max_degree_node[0]
        print(f"  Fallback: Using node {carina_node} (degree={max_degree_node[1]['degree']})")
    else:
        # Per ogni biforcazione, calcola il diametro medio dei branch connessi
        bifurcation_scores = []
        
        for node_id in bifurcation_nodes:
            node = nodes_info[node_id]
            
            # Trova i branch connessi a questo nodo
            connected_branches = []
            for idx, row in branch_data.iterrows():
                if int(row['node-id-src']) == node_id or int(row['node-id-dst']) == node_id:
                    connected_branches.append(idx)
            
            # Calcola diametro medio dei branch connessi
            branch_diams = [branch_diameters.get(idx, 0) for idx in connected_branches]
            avg_diameter = np.mean(branch_diams) if branch_diams else node['diameter']
            max_diameter = np.max(branch_diams) if branch_diams else node['diameter']
            
            bifurcation_scores.append({
                'node_id': node_id,
                'degree': node['degree'],
                'z': node['z'],
                'y': node['y'],
                'x': node['x'],
                'node_diameter': node['diameter'],
                'avg_branch_diameter': avg_diameter,
                'max_branch_diameter': max_diameter,
                'score': avg_diameter  # Usiamo il diametro medio come score
            })
        
        # Ordina per score (diametro medio) decrescente
        bifurcation_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # La carina è la biforcazione con il diametro maggiore
        carina_info = bifurcation_scores[0]
        carina_node = carina_info['node_id']
        
        print(f"\n  CARINA IDENTIFIED:")
        print(f"    Node ID: {carina_node}")
        print(f"    Position (z,y,x): ({carina_info['z']}, {carina_info['y']}, {carina_info['x']})")
        print(f"    Degree: {carina_info['degree']}")
        print(f"    Node diameter: {carina_info['node_diameter']:.2f} mm")
        print(f"    Avg branch diameter: {carina_info['avg_branch_diameter']:.2f} mm")
        print(f"    Max branch diameter: {carina_info['max_branch_diameter']:.2f} mm")
        
        # Mostra le top 5 biforcazioni per confronto
        print(f"\n  Top 5 bifurcations by diameter:")
        for i, bif in enumerate(bifurcation_scores[:5]):
            marker = " ← CARINA (selected)" if i == 0 else ""
            print(f"    {i+1}. Node {bif['node_id']}: "
                  f"degree={bif['degree']}, "
                  f"avg_diam={bif['avg_branch_diameter']:.2f}mm, "
                  f"z={bif['z']}{marker}")
    
    carina_z = nodes_info[carina_node]['z']
    carina_y = nodes_info[carina_node]['y']
    carina_x = nodes_info[carina_node]['x']
    
    return carina_node, carina_z, carina_y, carina_x


def remove_trachea(mask, carina_z, margin_slices=0):
    """
    Rimuove la trachea mantenendo solo il tessuto dalla carina in poi.
    
    Args:
        mask: Maschera originale
        carina_z: Coordinata z della carina
        margin_slices: Numero di slice aggiuntive da rimuovere sotto la carina (per sicurezza)
    """
    print(f"\n=== Removing trachea ===")
    print(f"  Carina z-coordinate: {carina_z}")
    print(f"  Margin slices below carina: {margin_slices}")
    
    # Crea una copia della maschera
    bronchi_mask = mask.copy()
    
    # Rimuovi tutto sopra (e eventualmente leggermente sotto) la carina
    cutoff_z = carina_z + margin_slices
    print(f"  Removing all voxels with z < {cutoff_z}")
    
    # Conta i voxel prima
    original_voxels = np.sum(mask > 0)
    
    # Rimuovi trachea (tutto sopra cutoff_z)
    bronchi_mask[:cutoff_z, :, :] = 0
    
    # Conta i voxel dopo
    remaining_voxels = np.sum(bronchi_mask > 0)
    removed_voxels = original_voxels - remaining_voxels
    
    print(f"\n  Results:")
    print(f"    Original voxels (trachea + bronchi): {original_voxels:,}")
    print(f"    Remaining voxels (bronchi only): {remaining_voxels:,}")
    print(f"    Removed voxels (trachea): {removed_voxels:,} ({removed_voxels/original_voxels*100:.1f}%)")
    
    return bronchi_mask


def visualize_comparison(original_mask, bronchi_mask, carina_z, carina_y, carina_x, save_path=None):
    """Visualizza confronto tra maschera originale e maschera senza trachea"""
    print("\n=== Generating visualization ===")
    
    fig = plt.figure(figsize=(20, 6))
    
    # 1. Original mask
    ax1 = fig.add_subplot(131, projection='3d')
    original_coords = np.argwhere(original_mask > 0)
    subsample = max(1, len(original_coords) // 5000)
    original_coords = original_coords[::subsample]
    
    ax1.scatter(original_coords[:, 2], original_coords[:, 1], original_coords[:, 0],
               c='blue', s=1, alpha=0.5, label='Airways')
    ax1.scatter([carina_x], [carina_y], [carina_z],
               c='red', s=200, marker='*', edgecolors='black', linewidths=2,
               label='Carina', zorder=10)
    
    ax1.set_xlabel('X (voxel)')
    ax1.set_ylabel('Y (voxel)')
    ax1.set_zlabel('Z (voxel)')
    ax1.set_title(f'Original (Trachea + Bronchi)\n{np.sum(original_mask > 0):,} voxels')
    ax1.legend()
    
    # 2. Bronchi only
    ax2 = fig.add_subplot(132, projection='3d')
    bronchi_coords = np.argwhere(bronchi_mask > 0)
    subsample = max(1, len(bronchi_coords) // 5000)
    bronchi_coords = bronchi_coords[::subsample]
    
    ax2.scatter(bronchi_coords[:, 2], bronchi_coords[:, 1], bronchi_coords[:, 0],
               c='green', s=1, alpha=0.5, label='Bronchi')
    ax2.scatter([carina_x], [carina_y], [carina_z],
               c='red', s=200, marker='*', edgecolors='black', linewidths=2,
               label='Carina', zorder=10)
    
    ax2.set_xlabel('X (voxel)')
    ax2.set_ylabel('Y (voxel)')
    ax2.set_zlabel('Z (voxel)')
    ax2.set_title(f'After Trachea Removal (Bronchi Only)\n{np.sum(bronchi_mask > 0):,} voxels')
    ax2.legend()
    
    # 3. Removed tissue (trachea)
    ax3 = fig.add_subplot(133, projection='3d')
    removed = (original_mask > 0) & (bronchi_mask == 0)
    removed_coords = np.argwhere(removed)
    
    if len(removed_coords) > 0:
        subsample = max(1, len(removed_coords) // 5000)
        removed_coords = removed_coords[::subsample]
        
        ax3.scatter(removed_coords[:, 2], removed_coords[:, 1], removed_coords[:, 0],
                   c='red', s=1, alpha=0.5, label='Removed (Trachea)')
    
    ax3.scatter([carina_x], [carina_y], [carina_z],
               c='yellow', s=200, marker='*', edgecolors='black', linewidths=2,
               label='Carina', zorder=10)
    
    ax3.set_xlabel('X (voxel)')
    ax3.set_ylabel('Y (voxel)')
    ax3.set_zlabel('Z (voxel)')
    ax3.set_title(f'Removed Tissue (Trachea)\n{np.sum(removed):,} voxels')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved: {save_path}")
    
    plt.show()


def visualize_axial_slices(original_mask, bronchi_mask, carina_z, save_path=None):
    """Visualizza alcune slice assiali per vedere il taglio"""
    print("\n=== Generating axial slices ===")
    
    # Seleziona 6 slice: 3 sopra e 3 sotto la carina
    slices_to_show = [
        carina_z - 20,
        carina_z - 10,
        carina_z - 2,
        carina_z + 2,
        carina_z + 10,
        carina_z + 20
    ]
    
    # Filtra slice valide
    slices_to_show = [s for s in slices_to_show if 0 <= s < original_mask.shape[0]]
    
    fig, axes = plt.subplots(2, len(slices_to_show), figsize=(20, 8))
    
    for i, z in enumerate(slices_to_show):
        # Original
        axes[0, i].imshow(original_mask[z, :, :], cmap='gray')
        axes[0, i].set_title(f'Original\nz={z}' + (' (CARINA)' if abs(z - carina_z) < 3 else ''))
        axes[0, i].axis('off')
        
        # After removal
        axes[1, i].imshow(bronchi_mask[z, :, :], cmap='gray')
        axes[1, i].set_title(f'After removal\nz={z}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Axial Slices Comparison (Carina at z={carina_z})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Axial slices saved: {save_path}")
    
    plt.show()


def save_bronchi_mask(bronchi_mask, sitk_image, output_path):
    """Salva la maschera dei bronchi come file NIfTI"""
    print(f"\n=== Saving bronchi mask ===")
    
    # Crea immagine SimpleITK
    bronchi_sitk = sitk.GetImageFromArray(bronchi_mask.astype(np.uint8))
    bronchi_sitk.CopyInformation(sitk_image)
    
    # Salva
    sitk.WriteImage(bronchi_sitk, output_path)
    print(f"  Bronchi mask saved: {output_path}")
    
    return output_path


def save_final_segmentation(bronchi_mask, sitk_image, output_dir, input_filename):
    """Salva la segmentazione finale nella directory specificata"""
    print(f"\n=== Saving final segmentation ===")
    
    # Crea directory se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Crea nome file di output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    base_name = base_name.replace('.nii', '')  # Rimuovi eventuali estensioni .nii
    base_name = base_name.replace('_airwayfull', '')  # Rimuovi suffisso airwayfull
    
    output_filename = f"{base_name}_bronchi_only_{timestamp}.nii.gz"
    output_path = os.path.join(output_dir, output_filename)
    
    # Salva la segmentazione
    final_sitk = sitk.GetImageFromArray(bronchi_mask.astype(np.uint8))
    final_sitk.CopyInformation(sitk_image)
    sitk.WriteImage(final_sitk, output_path)
    
    print(f"  Final segmentation saved: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return output_path


def main():
    """Main function"""
    print("="*80)
    print(" "*25 + "TRACHEA REMOVAL TEST")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Input file
    input_mask_path = "airway_segmentation/1.2.840.113704.1.111.2604.1126357612.7_airwayfull.nii.gz"
    
    # Output directories
    output_dir = "trachea_removal_test"
    final_segmentation_dir = "final_segmentations"  # Directory per le segmentazioni finali
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_segmentation_dir, exist_ok=True)
    
    # Margin: numero di slice sotto la carina da rimuovere (0 = taglia esattamente alla carina)
    margin_slices = 0  # Puoi aumentare a 2-3 se vuoi essere più conservativo
    
    # ========================================================================
    # STEP 1: Load mask
    # ========================================================================
    
    if not os.path.exists(input_mask_path):
        print(f"\n❌ ERROR: Input file not found: {input_mask_path}")
        print("Please update the 'input_mask_path' variable.")
        return
    
    mask, sitk_image, spacing = load_airway_mask(input_mask_path)
    
    # ========================================================================
    # STEP 2: Compute skeleton and distance transform
    # ========================================================================
    
    skeleton, distance_transform = compute_skeleton_and_distances(mask, spacing)
    
    # ========================================================================
    # STEP 3: Build skeleton graph with diameter info
    # ========================================================================
    
    skeleton_obj, branch_data, nodes_info, branch_diameters = build_skeleton_graph(
        skeleton, distance_transform, spacing
    )
    
    # ========================================================================
    # STEP 4: Identify carina
    # ========================================================================
    
    carina_node, carina_z, carina_y, carina_x = identify_carina(
        nodes_info, branch_data, branch_diameters
    )
    
    # ========================================================================
    # STEP 5: Remove trachea
    # ========================================================================
    
    bronchi_mask = remove_trachea(mask, carina_z, margin_slices=margin_slices)
    
    # ========================================================================
    # STEP 6: Save results
    # ========================================================================
    
    # Salva nella directory di test
    test_output_path = os.path.join(output_dir, "bronchi_only_mask.nii.gz")
    save_bronchi_mask(bronchi_mask, sitk_image, test_output_path)
    
    # Salva la segmentazione finale nella directory dedicata
    final_output_path = save_final_segmentation(
        bronchi_mask, sitk_image, final_segmentation_dir, input_mask_path
    )
    
    # ========================================================================
    # STEP 7: Visualizations
    # ========================================================================
    
    visualize_comparison(
        mask, bronchi_mask, carina_z, carina_y, carina_x,
        save_path=os.path.join(output_dir, "comparison_3d.png")
    )
    
    visualize_axial_slices(
        mask, bronchi_mask, carina_z,
        save_path=os.path.join(output_dir, "axial_slices.png")
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print(" "*30 + "COMPLETED!")
    print("="*80)
    print(f"\n✓ Trachea successfully removed!")
    print(f"\nOutput files:")
    print(f"  • Test output: {test_output_path}")
    print(f"  • Final segmentation: {final_output_path}")
    print(f"  • Visualizations: {output_dir}/comparison_3d.png")
    print(f"  • Visualizations: {output_dir}/axial_slices.png")
    print(f"\nCarina location: z={carina_z}, y={carina_y}, x={carina_x}")
    print(f"Voxels removed: {np.sum((mask > 0) & (bronchi_mask == 0)):,}")
    print(f"Voxels remaining: {np.sum(bronchi_mask > 0):,}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()