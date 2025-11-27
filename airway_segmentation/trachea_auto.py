import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass, binary_erosion, binary_dilation
from scipy.ndimage import generate_binary_structure
import os
import datetime
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label
from skan import Skeleton, summarize
import networkx as nx


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

def compute_skeleton(mask, spacing):
    """Computes the 3D skeleton of the mask"""
    print("\n=== 3D Skeletonization ===")
    
    # Binarize the mask
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Apply skeletonize (works for both 2D and 3D)
    print("Computing 3D skeleton (may take a few minutes)...")
    skeleton = skeletonize(binary_mask)
    
    skeleton_voxels = np.sum(skeleton > 0)
    print(f"Skeleton computed: {skeleton_voxels} voxels")
    
    # Compute distance transform for diameters
    print("Computing distance transform for diameters...")
    distance_transform = distance_transform_edt(binary_mask, sampling=spacing)
    
    return skeleton, distance_transform, skeleton_voxels

def build_graph(skeleton, spacing):
    """Builds graph from skeleton using skan"""
    print("\n=== Graph Construction ===")
    
    if skeleton is None:
        raise ValueError("Compute skeleton first with compute_skeleton()")
    
    # Create skan Skeleton object
    # spacing in order (z, y, x) for skan
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    skeleton_obj = Skeleton(skeleton, spacing=spacing_zyx)
    
    # Extract branch information
    branch_data = summarize(skeleton_obj)
    
    print(f"Number of identified branches: {len(branch_data)}")
    print(f"Number of junctions: {skeleton_obj.n_paths}")
    
    # Create NetworkX graph for topological analysis
    graph = _create_networkx_graph(skeleton_obj, branch_data)
    
    return graph, skeleton_obj, branch_data

def _create_networkx_graph(skeleton_obj, branch_data):
    """Creates a NetworkX graph from skeleton"""
    G = nx.Graph()
    
    # Add nodes (endpoints and junction points)
    coordinates = skeleton_obj.coordinates
    for idx in range(len(coordinates)):
        pos = coordinates[idx]
        G.add_node(idx, pos=pos)
    
    # Add edges from branches
    for _, row in branch_data.iterrows():
        node1 = int(row['node-id-src'])
        node2 = int(row['node-id-dst'])
        length = row['branch-distance']
        G.add_edge(node1, node2, length=length, branch_type=row['branch-type'])
    
    return G

def identify_carina(graph, distance_transform):
    """
    Identifica la carina come il nodo con il diametro medio più grande tra i punti di biforcazione
    Restituisce anche le coordinate della carina
    """
    print("\n=== CARINA IDENTIFICATION BY DIAMETER ===")
    
    if graph is None:
        raise ValueError("Build graph first with build_graph()")
    
    # Trova nodi con grado >= 2 (potenziali biforcazioni)
    candidate_nodes = [node for node in graph.nodes() 
                    if graph.degree(node) >= 2]
    
    if len(candidate_nodes) == 0:
        print("WARNING: No bifurcation nodes found!")
        # Fallback: usa il nodo con grado più alto
        degrees = dict(graph.degree())
        carina_node = max(degrees, key=degrees.get)
        carina_diameter = 0
        carina_position = graph.nodes[carina_node]['pos']
        node_info = []  # Initialize empty for the fallback case
    else:
        # Calcola diametro medio per ogni candidato
        node_info = []
        for node in candidate_nodes:
            pos = graph.nodes[node]['pos']
            z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
            
            # Ottieni diametro alla posizione del nodo
            if (0 <= z < distance_transform.shape[0] and
                0 <= y < distance_transform.shape[1] and
                0 <= x < distance_transform.shape[2]):
                diameter = distance_transform[z, y, x] * 2
            else:
                diameter = 0
            
            # Calcola diametro medio dei rami connessi
            connected_branches = []
            for neighbor in graph.neighbors(node):
                edge = tuple(sorted([node, neighbor]))
                if 'diameter' in graph.edges[edge]:
                    connected_branches.append(graph.edges[edge]['diameter'])
            
            avg_branch_diameter = np.mean(connected_branches) if connected_branches else diameter
            
            node_info.append({
                'node': node,
                'degree': graph.degree(node),
                'diameter_at_node': diameter,
                'avg_branch_diameter': avg_branch_diameter,
                'z': z,
                'y': y,
                'x': x,
                'position': pos
            })
        
        # Ordina per diametro medio (discendente) poi per coordinata z (ascendente - superiore)
        node_info.sort(key=lambda x: (-x['avg_branch_diameter'], x['z']))
        
        carina_node = node_info[0]['node']
        carina_diameter = node_info[0]['avg_branch_diameter']
        carina_position = node_info[0]['position']
        
        print(f"Carina identified: Node {carina_node}")
        print(f"  Degree: {node_info[0]['degree']}")
        print(f"  Average branch diameter: {node_info[0]['avg_branch_diameter']:.2f} mm")
        print(f"  Diameter at node: {node_info[0]['diameter_at_node']:.2f} mm")
        print(f"  Position (z,y,x): ({node_info[0]['z']}, {node_info[0]['y']:.1f}, {node_info[0]['x']:.1f})")
        print(f"  Position (world coordinates): ({carina_position[2]:.1f}, {carina_position[1]:.1f}, {carina_position[0]:.1f})")
    
    # Handle fallback case where node_info might be empty
    if len(candidate_nodes) == 0:
        carina_info = {
            'node_id': carina_node,
            'position': carina_position,
            'avg_branch_diameter': carina_diameter,
            'diameter_at_node': 0,
            'degree': graph.degree(carina_node),
            'coordinates_voxel': (int(carina_position[0]), int(carina_position[1]), int(carina_position[2])),
            'coordinates_world': (carina_position[2], carina_position[1], carina_position[0])  # x, y, z
        }
    else:
        carina_info = {
            'node_id': carina_node,
            'position': carina_position,
            'avg_branch_diameter': carina_diameter,
            'diameter_at_node': node_info[0]['diameter_at_node'],
            'degree': node_info[0]['degree'],
            'coordinates_voxel': (node_info[0]['z'], node_info[0]['y'], node_info[0]['x']),
            'coordinates_world': (carina_position[2], carina_position[1], carina_position[0])  # x, y, z
        }

    return carina_node, carina_diameter, carina_position, carina_info

def find_exact_carina_position(mask, known_carina_coords):
    """
    Trova la posizione esatta della carina.
    Se sono fornite coordinate note, le usa direttamente.
    Altrimenti, usa l'analisi automatica.
    """
    print("\n=== Finding exact carina position ===")
    
    if known_carina_coords is not None:
        # Converti in interi
        carina_z, carina_y, carina_x = [int(coord) for coord in known_carina_coords]
        print(f"  Using provided carina coordinates:")
        print(f"    z={carina_z}, y={carina_y}, x={carina_x}")
        
        # Verifica che le coordinate siano valide
        if (0 <= carina_z < mask.shape[0] and 
            0 <= carina_y < mask.shape[1] and 
            0 <= carina_x < mask.shape[2]):
            return carina_z, carina_y, carina_x
        else:
            print("  WARNING: Provided coordinates are out of bounds, using automatic detection")
    
    # Metodo automatico di fallback
    print("  Using automatic carina detection...")
    
    # Trova lo slice assiale più basso con almeno 2 componenti connesse
    # NOTA: Z=0 è la parte inferiore (piedi), Z=max è la parte superiore (testa)
    for z in range(mask.shape[0]-1, -1, -1):
        slice_2d = (mask[z, :, :] > 0).astype(np.uint8)
        if np.sum(slice_2d) == 0:
            continue
        
        labeled, num_objects = label(slice_2d)
        if num_objects >= 2:
            # Trova i due oggetti più grandi
            sizes = []
            centroids = []
            for obj_id in range(1, num_objects + 1):
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
                print(f"  Found carina at z={z}")
                return z, carina_y, carina_x
    
    # Fallback: usa il centro del volume
    carina_z = mask.shape[0] // 3
    carina_y = mask.shape[1] // 2
    carina_x = mask.shape[2] // 2
    print(f"  WARNING: Using fallback coordinates: z={carina_z}")
    
    return carina_z, carina_y, carina_x

def refine_carina_with_region_growing(mask, carina_z, carina_y, carina_x, spacing):
    """
    Raffina la posizione della carina usando region growing
    per trovare il punto esatto di biforcazione.
    """
    print("\n=== Refining carina position with region growing ===")
    
    # Crea una maschera binaria
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Definisci una regione di interesse attorno alla carina
    roi_size = 20  # voxels
    z_start = max(0, carina_z - roi_size)
    z_end = min(mask.shape[0], carina_z + roi_size)
    
    best_z = carina_z
    max_objects = 0
    
    # Analizza ogni slice nella ROI
    for z in range(int(z_start), int(z_end)):  # Converti in interi
        slice_2d = binary_mask[z, :, :]
        if np.sum(slice_2d) == 0:
            continue
        
        labeled, num_objects = label(slice_2d)
        
        # Considera solo slice con almeno 2 oggetti di dimensioni simili
        if num_objects >= 2:
            sizes = []
            for obj_id in range(1, num_objects + 1):
                size = np.sum(labeled == obj_id)
                sizes.append(size)
            
            sizes.sort(reverse=True)
            if len(sizes) >= 2 and sizes[1] > sizes[0] * 0.3:  # Secondo oggetto almeno 30% del primo
                if num_objects > max_objects:
                    max_objects = num_objects
                    best_z = z
    
    print(f"  Refined carina z-coordinate: {best_z} (was {carina_z})")
    return best_z, carina_y, carina_x


def remove_trachea_precise(mask, carina_z, carina_y, carina_x, spacing, method='adaptive'):
    """
    Rimuove la trachea in modo preciso usando diverse strategie.
    MODIFICATO: Preserva una porzione significativa della trachea per evitare problemi nel cleaning
    """
    print(f"\n=== Removing trachea with {method} method ===")
    print(f"  Carina position: z={carina_z}, y={carina_y}, x={carina_x}")
    
    bronchi_mask = mask.copy()
    binary_mask = (mask > 0).astype(np.uint8)
    
    if method == 'adaptive':
        # Metodo adattivo: usa analisi delle componenti connesse
        cutoff_z = find_adaptive_cutoff(binary_mask, carina_z, spacing)
        
    elif method == 'curved':
        # Metodo curvo: segue l'anatomia delle vie aeree
        cutoff_z = find_curved_cutoff(binary_mask, carina_z, carina_y, carina_x, spacing)
        
    elif method == 'vertical':
        # Taglio verticale semplice (metodo originale) - MODIFICATO: preserva più trachea
        cutoff_z = min(mask.shape[0] - 1, carina_z + 15)  # MOLTO più alto
        
    elif method == 'horizontal':
        # Taglio orizzontale con margine di sicurezza - MODIFICATO: preserva molto più trachea
        cutoff_z = min(mask.shape[0] - 1, carina_z + 20)  # MOLTO più alto
    
    else:
        cutoff_z = carina_z
    
    print(f"  Cutoff z-coordinate: {cutoff_z}")
    
    # APPLICA LA RIMOZIONE CORRETTA: rimuovi la trachea SOPRA la carina
    original_voxels = np.sum(mask > 0)
    
    # IMPORTANTE: Z cresce dal basso verso l'alto, quindi la trachea è SOPRA la carina
    # MODIFICATO: Rimuovi solo la parte SUPERIORE della trachea, preservando una buona porzione
    bronchi_mask[cutoff_z:, :, :] = 0  # Rimuovi tutto SOPRA il cutoff
    
    # AGGIUNTA: Preserva attivamente una porzione significativa della trachea sopra la carina
    preserved_trachea_height = 50  # Altezza significativa della trachea da preservare (in slice)
    trachea_start_z = max(0, carina_z - preserved_trachea_height)
    
    # Ricostruisci la porzione tracheale preservata sopra la carina
    for z in range(trachea_start_z, carina_z + 1):
        if z < cutoff_z:  # Solo se non è già stato rimosso
            # Trova la componente tracheale in questo slice
            slice_2d = binary_mask[z, :, :]
            if np.sum(slice_2d) > 0:
                labeled, num_objects = label(slice_2d)
                if num_objects > 0:
                    # Per slice sopra la carina, preserva TUTTE le componenti connesse
                    # per mantenere l'integrità strutturale
                    bronchi_mask[z, :, :] = mask[z, :, :]  # Preserva tutto lo slice
    
    # AGGIUNTA: Dilatazione per garantire la connettività
    structure = generate_binary_structure(3, 1)  # Connettività 6-vicinato
    bronchi_mask_binary = (bronchi_mask > 0).astype(np.uint8)
    bronchi_mask_binary = binary_dilation(bronchi_mask_binary, structure=structure, iterations=1)
    bronchi_mask_binary = binary_erosion(bronchi_mask_binary, structure=structure, iterations=1)
    
    # Ricostruisci la maschera finale
    bronchi_mask = bronchi_mask_binary.astype(mask.dtype) * np.max(mask)
    
    remaining_voxels = np.sum(bronchi_mask > 0)
    removed_voxels = original_voxels - remaining_voxels
    
    print(f"\n  Results:")
    print(f"    Original: {original_voxels:,}")
    print(f"    Remaining: {remaining_voxels:,}")
    print(f"    Removed: {removed_voxels:,} ({removed_voxels/original_voxels*100:.1f}%)")
    print(f"    Preserved trachea: {preserved_trachea_height} slices above carina")
    print(f"    Final cutoff: {cutoff_z} (removes only upper trachea)")
    
    return bronchi_mask, cutoff_z


def find_adaptive_cutoff(binary_mask, carina_z, spacing):
    """
    Trova il punto di taglio adattivo analizzando la connettività.
    MODIFICATO: Preserva MOLTA più trachea
    """
    print("  Using adaptive cutoff method...")
    
    # MODIFICATO: Cerca molto più in alto per preservare quasi tutta la trachea
    search_range = 50  # MOLTO più grande
    
    # Cerca il punto dove la trachea si divide chiaramente in bronchi
    # Cerca SOPRA la carina (Z maggiori) poiché Z=0 è in basso
    for z in range(int(carina_z), min(binary_mask.shape[0], int(carina_z) + search_range)):
        slice_2d = binary_mask[z, :, :]
        if np.sum(slice_2d) == 0:
            continue
        
        labeled, num_objects = label(slice_2d)
        
        if num_objects >= 2:
            # Verifica che i due oggetti principali siano ben separati
            sizes = []
            for obj_id in range(1, num_objects + 1):
                size = np.sum(labeled == obj_id)
                sizes.append(size)
            
            sizes.sort(reverse=True)
            # MODIFICATO: Soglia molto bassa per essere ultra-conservativo
            if len(sizes) >= 2 and sizes[1] > sizes[0] * 0.08:  # RIDOTTA soglia all'8%
                print(f"    Found clear bifurcation at z={z}")
                # MODIFICATO: Offset MOLTO maggiore per preservare quasi tutta la trachea
                return min(binary_mask.shape[0] - 1, z + 10)
    
    # MODIFICATO: Fallback con margine MOLTO alto per preservare quasi tutta la trachea
    return min(binary_mask.shape[0] - 1, int(carina_z) + 25)


def find_curved_cutoff(binary_mask, carina_z, carina_y, carina_x, spacing):
    """
    Trova un punto di taglio che segua l'anatomia curva delle vie aeree.
    MODIFICATO: Preserva QUASI TUTTA la trachea
    """
    print("  Using curved cutoff method...")
    
    # Crea una maschera della sola trachea
    trachea_mask = binary_mask.copy()
    
    # Trova la componente connessa principale (trachea)
    labeled, num_objects = label(binary_mask)
    if num_objects == 0:
        return carina_z
    
    # Trova l'oggetto più grande (trachea)
    sizes = []
    for obj_id in range(1, num_objects + 1):
        size = np.sum(labeled == obj_id)
        sizes.append(size)
    
    main_object = np.argmax(sizes) + 1
    trachea_mask = (labeled == main_object)
    
    # Trova il punto più basso e alto della trachea
    trachea_coords = np.argwhere(trachea_mask)
    if len(trachea_coords) == 0:
        return carina_z
    
    min_z = np.min(trachea_coords[:, 0])
    max_z = np.max(trachea_coords[:, 0])
    
    print(f"    Trachea extends from z={min_z} to z={max_z}")
    
    # MODIFICATO: Il punto di taglio è MOLTO più SOPRA la carina per preservare quasi tutta la trachea
    preserved_length = 50  # Lunghezza MOLTO grande della trachea da preservare
    cutoff_z = min(max_z, int(carina_z) + preserved_length)
    
    # Se siamo vicini alla fine, preserva ancora di più
    if cutoff_z > max_z - 5:
        cutoff_z = max_z - 2  # Lascia gli ultimi 2 slice
    
    return cutoff_z


def remove_trachea_3d_region_growing(mask, carina_z, carina_y, carina_x, spacing):
    """
    Rimuove la trachea usando region growing 3D per identificare
    precisamente la regione tracheale.
    MODIFICATO: Preserva QUASI TUTTA la trachea
    """
    print("\n=== Removing trachea with 3D region growing ===")
    
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Inizia dal punto più alto della maschera (Z massimo)
    start_z = binary_mask.shape[0] - 1
    for z in range(binary_mask.shape[0] - 1, -1, -1):
        if np.any(binary_mask[z, :, :]):
            start_z = z
            break
    
    # MODIFICATO: Punto di partenza molto più alto per preservare quasi tutta la trachea
    seed_point = (start_z - 25, carina_y, carina_x)  # MOLTO più alto
    
    # MODIFICATO: Rimuovi solo la parte SUPERIORE della trachea
    preserved_trachea_height = 20  # Preserva MOLTA trachea
    trachea_start_z = min(binary_mask.shape[0], carina_z + preserved_trachea_height)
    
    # Crea maschera finale che preserva quasi tutta la trachea
    bronchi_mask = binary_mask.copy()
    bronchi_mask[trachea_start_z:, :, :] = 0  # Rimuovi solo la parte superiore
    
    # AGGIUNTA: Dilatazione per garantire la connettività
    structure = generate_binary_structure(3, 1)  # Connettività 6-vicinato
    bronchi_mask = binary_dilation(bronchi_mask, structure=structure, iterations=2)
    bronchi_mask = binary_erosion(bronchi_mask, structure=structure, iterations=2)
    
    # Converti al tipo originale
    bronchi_mask = bronchi_mask.astype(mask.dtype) * np.max(mask)
    
    # Statistiche
    original_voxels = np.sum(mask > 0)
    remaining_voxels = np.sum(bronchi_mask > 0)
    removed_voxels = original_voxels - remaining_voxels
    
    print(f"  Results:")
    print(f"    Original: {original_voxels:,}")
    print(f"    Remaining: {remaining_voxels:,}")
    print(f"    Removed: {removed_voxels:,} ({removed_voxels/original_voxels*100:.1f}%)")
    print(f"    Preserved most trachea for robust branch analysis")
    print(f"    Only removed upper trachea above z={trachea_start_z}")
    
    return bronchi_mask


def visualize_precise_removal(original_mask, results_dict, carina_z, carina_y, carina_x, save_path=None):
    """Visualizza confronto tra diversi metodi di rimozione"""
    print("\n=== Generating precise removal visualization ===")
    
    n_methods = len(results_dict)
    fig = plt.figure(figsize=(6 * n_methods, 6))
    
    for idx, (method_name, bronchi_mask) in enumerate(results_dict.items()):
        ax = fig.add_subplot(1, n_methods, idx + 1, projection='3d')
        
        # Visualizza i bronchi rimanenti
        coords = np.argwhere(bronchi_mask > 0)
        if len(coords) > 0:
            subsample = max(1, len(coords) // 3000)
            coords = coords[::subsample]
            ax.scatter(coords[:, 2], coords[:, 1], coords[:, 0],
                     c='green', s=1, alpha=0.6, label='Bronchi + Trachea')
        
        # Visualizza la carina
        ax.scatter([carina_x], [carina_y], [carina_z],
                 c='red', s=200, marker='*', edgecolors='black', 
                 linewidths=2, label='Carina')
        
        ax.set_title(f'{method_name}\n{np.sum(bronchi_mask > 0):,} voxels')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Precise removal visualization saved: {save_path}")
    
    plt.show()


def save_final_segmentation(bronchi_mask, sitk_image, output_dir, input_filename, method_name=""):
    """Salva segmentazione finale"""
    print(f"\n=== Saving final segmentation ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    base_name = base_name.replace('.nii', '').replace('_airwayfull', '')
    
    if method_name:
        output_filename = f"{base_name}_bronchi_{method_name}_{timestamp}.nii.gz"
    else:
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
    print(" "*18 + "PRECISE TRACHEA REMOVAL - PRESERVE MOST TRACHEA")
    print("="*80)
    
    # Configuration
    input_mask_path = "airway_segmentation/1.2.840.113704.1.111.1396.1132404220.7_airwayfull.nii.gz"    
    output_dir = "precise_trachea_removal"
    final_segmentation_dir = "final_segmentations"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_segmentation_dir, exist_ok=True)
    
    # Load
    if not os.path.exists(input_mask_path):
        print(f"\n❌ ERROR: File not found: {input_mask_path}")
        return
    
    mask, sitk_image, spacing = load_airway_mask(input_mask_path)
    skeleton_temp = compute_skeleton(mask, spacing)
    build_graph_temp = build_graph(skeleton_temp[0], spacing)
    carina_temp = identify_carina(build_graph_temp[0], skeleton_temp[1])
    KNOWN_CARINA_COORDS = carina_temp[3]['coordinates_voxel']
    
    # Trova la posizione precisa della carina
    carina_z, carina_y, carina_x = find_exact_carina_position(
        mask, KNOWN_CARINA_COORDS
    )
    
    # Raffina la posizione con region growing
    carina_z, carina_y, carina_x = refine_carina_with_region_growing(
        mask, carina_z, carina_y, carina_x, spacing
    )
    
    # Prova diversi metodi di rimozione
    removal_methods = ['adaptive', 'curved', 'vertical', '3d_growing']
    results = {}
    
    for method in removal_methods:
        if method == '3d_growing':
            results[method] = remove_trachea_3d_region_growing(
                mask, carina_z, carina_y, carina_x, spacing
            )
        else:
            results[method], cutoff_z = remove_trachea_precise(
                mask, carina_z, carina_y, carina_x, spacing, method
            )
    
    # Visualizza confronto
    visualize_precise_removal(
        mask, results, carina_z, carina_y, carina_x,
        save_path=os.path.join(output_dir, "method_comparison_preserved_trachea.png")
    )
    
    # Salva tutti i risultati
    for method, bronchi_mask in results.items():
        save_final_segmentation(
            bronchi_mask, sitk_image, final_segmentation_dir, 
            input_mask_path, method
        )
    
    # Salva anche nel output_dir per riferimento
    best_method = 'curved'  # Metodo raccomandato per preservare la trachea
    test_output_path = os.path.join(output_dir, f"bronchi_{best_method}_preserved_trachea.nii.gz")
    final_sitk = sitk.GetImageFromArray(results[best_method].astype(np.uint8))
    final_sitk.CopyInformation(sitk_image)
    sitk.WriteImage(final_sitk, test_output_path)
    
    # Summary
    print("\n" + "="*80)
    print(" "*32 + "COMPLETED!")
    print("="*80)
    print(f"\n✓ Precise trachea removal complete!")
    print(f"✓ MODIFIED: Preserved SIGNIFICANT trachea portion for robust cleaning")
    print(f"✓ This ensures branch connections won't be lost during subsequent processing")
    print(f"\nFinal carina position: (z={carina_z}, y={carina_y}, x={carina_x})")
    
    print(f"\nComparison of methods:")
    for method, bronchi_mask in results.items():
        original = np.sum(mask > 0)
        remaining = np.sum(bronchi_mask > 0)
        removed = original - remaining
        print(f"  • {method:12}: {remaining:>8,} voxels remaining "
              f"({removed/original*100:5.1f}% removed)")
    
    print(f"\nAll methods preserve SIGNIFICANT trachea portion to prevent cleaning issues")
    print(f"Only upper trachea is removed, maintaining robust branch connections")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()