"""
Script migliorato per la rimozione della trachea.
Usa un approccio multi-planare per identificare la carina analizzando
tutte e tre le direzioni (assiale, coronale, sagittale).
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label, center_of_mass
import os
import datetime


def load_airway_mask(mask_path):
    """Carica la maschera delle vie aeree"""
    print(f"Loading mask from: {mask_path}")
    sitk_image = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_image)
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    
    print(f"  Spacing (x,y,z): {spacing} mm")
    print(f"  Origin (x,y,z): {origin}")
    print(f"  Shape (z,y,x): {mask.shape}")
    print(f"  Positive voxels: {np.sum(mask > 0):,}")
    
    return mask, sitk_image, spacing


def analyze_connectivity_all_directions(mask):
    """
    Analizza la connettività in tutte e tre le direzioni.
    Restituisce informazioni su dove avviene la biforcazione.
    """
    print("\n=== Analyzing connectivity in all directions ===")
    
    results = {}
    
    # DIREZIONE 1: Assiale (lungo z, dall'alto verso il basso)
    print("\n  Analyzing AXIAL direction (z-axis, superior → inferior)...")
    axial_info = []
    for z in range(mask.shape[0]):
        slice_2d = (mask[z, :, :] > 0).astype(np.uint8)
        if np.sum(slice_2d) == 0:
            continue
        
        labeled, num_objects = label(slice_2d)
        object_sizes = []
        for obj_id in range(1, num_objects + 1):
            size = np.sum(labeled == obj_id)
            object_sizes.append(size)
        
        if object_sizes:
            object_sizes.sort(reverse=True)
            axial_info.append({
                'coord': z,
                'num_objects': num_objects,
                'largest': object_sizes[0],
                'second_largest': object_sizes[1] if len(object_sizes) > 1 else 0
            })
    
    results['axial'] = axial_info
    print(f"    Found {len(axial_info)} non-empty slices")
    
    # DIREZIONE 2: Coronale (lungo y, dal posteriore all'anteriore)
    print("\n  Analyzing CORONAL direction (y-axis, posterior → anterior)...")
    coronal_info = []
    for y in range(mask.shape[1]):
        slice_2d = (mask[:, y, :] > 0).astype(np.uint8)
        if np.sum(slice_2d) == 0:
            continue
        
        labeled, num_objects = label(slice_2d)
        object_sizes = []
        for obj_id in range(1, num_objects + 1):
            size = np.sum(labeled == obj_id)
            object_sizes.append(size)
        
        if object_sizes:
            object_sizes.sort(reverse=True)
            coronal_info.append({
                'coord': y,
                'num_objects': num_objects,
                'largest': object_sizes[0],
                'second_largest': object_sizes[1] if len(object_sizes) > 1 else 0
            })
    
    results['coronal'] = coronal_info
    print(f"    Found {len(coronal_info)} non-empty slices")
    
    # DIREZIONE 3: Sagittale (lungo x, da sinistra a destra)
    print("\n  Analyzing SAGITTAL direction (x-axis, left → right)...")
    sagittal_info = []
    for x in range(mask.shape[2]):
        slice_2d = (mask[:, :, x] > 0).astype(np.uint8)
        if np.sum(slice_2d) == 0:
            continue
        
        labeled, num_objects = label(slice_2d)
        object_sizes = []
        for obj_id in range(1, num_objects + 1):
            size = np.sum(labeled == obj_id)
            object_sizes.append(size)
        
        if object_sizes:
            object_sizes.sort(reverse=True)
            sagittal_info.append({
                'coord': x,
                'num_objects': num_objects,
                'largest': object_sizes[0],
                'second_largest': object_sizes[1] if len(object_sizes) > 1 else 0
            })
    
    results['sagittal'] = sagittal_info
    print(f"    Found {len(sagittal_info)} non-empty slices")
    
    return results


def find_bifurcation_point(direction_info, direction_name, min_ratio=0.20):
    """
    Trova il punto di biforcazione in una direzione specifica.
    Cerca dove 1 oggetto diventa 2 oggetti di dimensioni simili.
    """
    candidates = []
    
    for i, info in enumerate(direction_info):
        if info['num_objects'] >= 2:
            ratio = info['second_largest'] / info['largest'] if info['largest'] > 0 else 0
            
            if ratio >= min_ratio:
                # Verifica che prima ci fosse principalmente 1 oggetto
                consistent_single = True
                lookback = min(5, i)
                
                if i >= lookback:
                    for j in range(i - lookback, i):
                        if direction_info[j]['num_objects'] > 2:
                            consistent_single = False
                            break
                
                if consistent_single:
                    candidates.append({
                        'coord': info['coord'],
                        'ratio': ratio,
                        'num_objects': info['num_objects'],
                        'largest': info['largest'],
                        'second_largest': info['second_largest']
                    })
    
    if candidates:
        # Prendi il primo candidato con ratio più alto
        best = max(candidates, key=lambda x: x['ratio'])
        return best
    
    return None


def find_carina_multiplanar(analysis_results, mask):
    """
    Identifica la carina combinando informazioni da tutte e tre le direzioni.
    """
    print("\n=== Identifying carina from multi-planar analysis ===")
    
    bifurcations = {}
    
    # Trova biforcazioni in ogni direzione
    for direction in ['axial', 'coronal', 'sagittal']:
        bif = find_bifurcation_point(analysis_results[direction], direction)
        bifurcations[direction] = bif
        
        if bif:
            print(f"\n  {direction.upper()} bifurcation:")
            print(f"    Coordinate: {bif['coord']}")
            print(f"    Ratio: {bif['ratio']:.3f}")
            print(f"    Objects: {bif['num_objects']}")
            print(f"    Sizes: {bif['largest']}, {bif['second_largest']}")
        else:
            print(f"\n  {direction.upper()}: No clear bifurcation found")
    
    # Strategia: usa la biforcazione assiale (z) come principale
    # ma valida con le altre direzioni
    
    if bifurcations['axial']:
        carina_z = bifurcations['axial']['coord']
        print(f"\n  Using AXIAL bifurcation as primary: z={carina_z}")
    elif bifurcations['coronal']:
        # Se non trovato in assiale, prova a stimare da coronale
        # La carina è tipicamente nel terzo superiore dei polmoni
        carina_z = mask.shape[0] // 3
        print(f"\n  WARNING: Using estimated z={carina_z} (no clear axial bifurcation)")
    else:
        # Fallback
        carina_z = mask.shape[0] // 3
        print(f"\n  WARNING: Using fallback z={carina_z}")
    
    # Trova il centroide della regione alla carina
    carina_slice = (mask[carina_z, :, :] > 0).astype(np.uint8)
    
    if np.sum(carina_slice) > 0:
        labeled, num = label(carina_slice)
        
        if num >= 2:
            # Se ci sono 2+ oggetti, usa il punto medio tra i due più grandi
            sizes = []
            centroids = []
            for obj_id in range(1, num + 1):
                obj_mask = (labeled == obj_id)
                size = np.sum(obj_mask)
                cent = center_of_mass(obj_mask)
                sizes.append(size)
                centroids.append(cent)
            
            # Ordina per dimensione
            sorted_indices = np.argsort(sizes)[::-1]
            
            if len(centroids) >= 2:
                cent1 = centroids[sorted_indices[0]]
                cent2 = centroids[sorted_indices[1]]
                carina_y = int((cent1[0] + cent2[0]) / 2)
                carina_x = int((cent1[1] + cent2[1]) / 2)
            else:
                cent = center_of_mass(carina_slice)
                carina_y = int(cent[0])
                carina_x = int(cent[1])
        else:
            # Un solo oggetto, usa il suo centroide
            cent = center_of_mass(carina_slice)
            carina_y = int(cent[0])
            carina_x = int(cent[1])
    else:
        # Slice vuota, usa il centro
        carina_y = mask.shape[1] // 2
        carina_x = mask.shape[2] // 2
    
    print(f"\n  FINAL CARINA POSITION:")
    print(f"    (z, y, x) = ({carina_z}, {carina_y}, {carina_x})")
    
    return carina_z, carina_y, carina_x, bifurcations


def visualize_multiplanar_analysis(analysis_results, bifurcations, save_path=None):
    """Visualizza l'analisi multi-planare"""
    print("\n=== Generating multi-planar analysis plot ===")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    directions = ['axial', 'coronal', 'sagittal']
    colors = ['blue', 'green', 'purple']
    
    for idx, direction in enumerate(directions):
        info = analysis_results[direction]
        bif = bifurcations[direction]
        
        coords = [i['coord'] for i in info]
        num_objects = [i['num_objects'] for i in info]
        largest = [i['largest'] for i in info]
        second = [i['second_largest'] for i in info]
        
        # Plot 1: Number of objects
        axes[idx, 0].plot(coords, num_objects, color=colors[idx], linewidth=2)
        if bif:
            axes[idx, 0].axvline(bif['coord'], color='red', linestyle='--', linewidth=2,
                                label=f"Bifurcation at {bif['coord']}")
        axes[idx, 0].set_ylabel('Number of objects')
        axes[idx, 0].set_title(f'{direction.upper()} - Object Count')
        axes[idx, 0].grid(True, alpha=0.3)
        if bif:
            axes[idx, 0].legend()
        
        # Plot 2: Object sizes
        axes[idx, 1].plot(coords, largest, color=colors[idx], linewidth=2, label='Largest')
        axes[idx, 1].plot(coords, second, color=colors[idx], linewidth=2, 
                         linestyle='--', alpha=0.7, label='Second')
        if bif:
            axes[idx, 1].axvline(bif['coord'], color='red', linestyle='--', linewidth=2)
        axes[idx, 1].set_ylabel('Object size (pixels)')
        axes[idx, 1].set_title(f'{direction.upper()} - Object Sizes')
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend()
    
    axes[-1, 0].set_xlabel('Coordinate')
    axes[-1, 1].set_xlabel('Coordinate')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Multi-planar plot saved: {save_path}")
    
    plt.show()


def remove_trachea(mask, carina_z, margin_slices=0):
    """Rimuove la trachea"""
    print(f"\n=== Removing trachea ===")
    print(f"  Carina z-coordinate: {carina_z}")
    print(f"  Margin slices: {margin_slices}")
    
    bronchi_mask = mask.copy()
    cutoff_z = carina_z + margin_slices
    
    original_voxels = np.sum(mask > 0)
    bronchi_mask[:cutoff_z, :, :] = 0
    remaining_voxels = np.sum(bronchi_mask > 0)
    removed_voxels = original_voxels - remaining_voxels
    
    print(f"\n  Results:")
    print(f"    Original: {original_voxels:,}")
    print(f"    Remaining: {remaining_voxels:,}")
    print(f"    Removed: {removed_voxels:,} ({removed_voxels/original_voxels*100:.1f}%)")
    
    return bronchi_mask


def visualize_comparison(original_mask, bronchi_mask, carina_z, carina_y, carina_x, save_path=None):
    """Visualizza confronto 3D"""
    print("\n=== Generating 3D visualization ===")
    
    fig = plt.figure(figsize=(20, 6))
    
    # Original
    ax1 = fig.add_subplot(131, projection='3d')
    coords = np.argwhere(original_mask > 0)
    subsample = max(1, len(coords) // 5000)
    coords = coords[::subsample]
    ax1.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
               c='blue', s=1, alpha=0.5)
    ax1.scatter([carina_x], [carina_y], [carina_z],
               c='red', s=200, marker='*', edgecolors='black', linewidths=2)
    ax1.set_title(f'Original\n{np.sum(original_mask > 0):,} voxels')
    
    # Bronchi
    ax2 = fig.add_subplot(132, projection='3d')
    coords = np.argwhere(bronchi_mask > 0)
    subsample = max(1, len(coords) // 5000)
    coords = coords[::subsample]
    ax2.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
               c='green', s=1, alpha=0.5)
    ax2.scatter([carina_x], [carina_y], [carina_z],
               c='red', s=200, marker='*', edgecolors='black', linewidths=2)
    ax2.set_title(f'Bronchi Only\n{np.sum(bronchi_mask > 0):,} voxels')
    
    # Removed
    ax3 = fig.add_subplot(133, projection='3d')
    removed = (original_mask > 0) & (bronchi_mask == 0)
    coords = np.argwhere(removed)
    if len(coords) > 0:
        subsample = max(1, len(coords) // 5000)
        coords = coords[::subsample]
        ax3.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
                   c='red', s=1, alpha=0.5)
    ax3.scatter([carina_x], [carina_y], [carina_z],
               c='yellow', s=200, marker='*', edgecolors='black', linewidths=2)
    ax3.set_title(f'Removed\n{np.sum(removed):,} voxels')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  3D comparison saved: {save_path}")
    
    plt.show()


def visualize_axial_slices(original_mask, bronchi_mask, carina_z, save_path=None):
    """Visualizza slice assiali"""
    print("\n=== Generating axial slices ===")
    
    slices = [carina_z - 20, carina_z - 10, carina_z - 2, 
              carina_z + 2, carina_z + 10, carina_z + 20]
    slices = [s for s in slices if 0 <= s < original_mask.shape[0]]
    
    fig, axes = plt.subplots(2, len(slices), figsize=(20, 8))
    
    for i, z in enumerate(slices):
        axes[0, i].imshow(original_mask[z, :, :], cmap='gray')
        axes[0, i].set_title(f'Original z={z}' + 
                            (' ★' if abs(z - carina_z) < 3 else ''))
        axes[0, i].axis('off')
        
        axes[1, i].imshow(bronchi_mask[z, :, :], cmap='gray')
        axes[1, i].set_title(f'After z={z}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Axial Slices (Carina at z={carina_z})', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Axial slices saved: {save_path}")
    
    plt.show()


def save_final_segmentation(bronchi_mask, sitk_image, output_dir, input_filename):
    """Salva segmentazione finale"""
    print(f"\n=== Saving final segmentation ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    base_name = base_name.replace('.nii', '').replace('_airwayfull', '')
    
    output_filename = f"{base_name}_bronchi_only_{timestamp}.nii.gz"
    output_path = os.path.join(output_dir, output_filename)
    
    final_sitk = sitk.GetImageFromArray(bronchi_mask.astype(np.uint8))
    final_sitk.CopyInformation(sitk_image)
    sitk.WriteImage(final_sitk, output_path)
    
    print(f"  Saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return output_path


def main():
    """Main function"""
    print("="*80)
    print(" "*18 + "TRACHEA REMOVAL - Multi-planar Method")
    print("="*80)
    
    # Configuration
    input_mask_path = "airway_segmentation/1.2.840.113704.1.111.2604.1126357612.7_airwayfull.nii.gz"
    
    output_dir = "trachea_removal_test_v3"
    final_segmentation_dir = "final_segmentations"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_segmentation_dir, exist_ok=True)
    
    margin_slices = 0
    
    # Load
    if not os.path.exists(input_mask_path):
        print(f"\n❌ ERROR: File not found: {input_mask_path}")
        return
    
    mask, sitk_image, spacing = load_airway_mask(input_mask_path)
    
    # Multi-planar analysis
    analysis_results = analyze_connectivity_all_directions(mask)
    
    # Find carina
    carina_z, carina_y, carina_x, bifurcations = find_carina_multiplanar(
        analysis_results, mask
    )
    
    # Visualize analysis
    visualize_multiplanar_analysis(
        analysis_results, bifurcations,
        save_path=os.path.join(output_dir, "multiplanar_analysis.png")
    )
    
    # Remove trachea
    bronchi_mask = remove_trachea(mask, carina_z, margin_slices=margin_slices)
    
    # Save
    test_output_path = os.path.join(output_dir, "bronchi_only_mask.nii.gz")
    final_sitk = sitk.GetImageFromArray(bronchi_mask.astype(np.uint8))
    final_sitk.CopyInformation(sitk_image)
    sitk.WriteImage(final_sitk, test_output_path)
    
    final_output_path = save_final_segmentation(
        bronchi_mask, sitk_image, final_segmentation_dir, input_mask_path
    )
    
    # Visualize
    visualize_comparison(
        mask, bronchi_mask, carina_z, carina_y, carina_x,
        save_path=os.path.join(output_dir, "comparison_3d.png")
    )
    
    visualize_axial_slices(
        mask, bronchi_mask, carina_z,
        save_path=os.path.join(output_dir, "axial_slices.png")
    )
    
    # Summary
    print("\n" + "="*80)
    print(" "*32 + "COMPLETED!")
    print("="*80)
    print(f"\n✓ Analysis complete!")
    print(f"\nCarina found at: (z={carina_z}, y={carina_y}, x={carina_x})")
    print(f"\nBifurcations detected:")
    for direction, bif in bifurcations.items():
        if bif:
            print(f"  • {direction}: coord={bif['coord']}, ratio={bif['ratio']:.3f}")
        else:
            print(f"  • {direction}: not found")
    print(f"\nRemoved: {np.sum((mask > 0) & (bronchi_mask == 0)):,} voxels")
    print(f"Remaining: {np.sum(bronchi_mask > 0):,} voxels")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()