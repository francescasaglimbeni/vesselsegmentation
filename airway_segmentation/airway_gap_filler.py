import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (distance_transform_edt, binary_dilation, binary_erosion,
                           label, generate_binary_structure, binary_closing,
                           gaussian_filter)
from skimage.morphology import skeletonize, ball, remove_small_holes
from scipy.spatial import cKDTree
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class IntelligentAirwayGapFiller:
    """
    Riempie gap e buchi nella segmentazione delle vie aeree in modo intelligente,
    preservando la topologia anatomica e seguendo il percorso naturale delle vie aeree.
    
    Strategia:
    1. Identifica gap e buchi nella segmentazione
    2. Costruisce skeleton preliminare per capire la topologia
    3. Riempie gap seguendo percorsi anatomicamente plausibili
    4. Valida che i riempimenti seguano HU coerenti con aria
    """
    
    def __init__(self, intensity_img, mask, spacing, verbose=True):
        """
        Args:
            intensity_img: CT image (numpy array, HU values)
            mask: Segmentazione binaria delle vie aeree
            spacing: Tuple (x,y,z) spacing in mm
            verbose: Print progress
        """
        self.img = intensity_img.astype(np.int16)
        self.mask = mask.astype(np.uint8)
        self.spacing = spacing
        self.verbose = verbose
        
        # Risultati
        self.filled_mask = None
        self.gap_analysis = {}
        self.preliminary_skeleton = None
        
    def analyze_gaps_and_holes(self):
        """
        Analizza buchi e disconnessioni nella segmentazione
        """
        if self.verbose:
            print("\n" + "="*70)
            print("GAP AND HOLE ANALYSIS")
            print("="*70)
        
        binary_mask = (self.mask > 0).astype(np.uint8)
        
        # 1. ANALISI COMPONENTI CONNESSE
        structure = generate_binary_structure(3, 3)  # 26-connectivity
        labeled, num_components = label(binary_mask, structure=structure)
        
        component_sizes = []
        for i in range(1, num_components + 1):
            size = np.sum(labeled == i)
            component_sizes.append((i, size))
        
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        if self.verbose:
            print(f"\nConnected components: {num_components}")
            if num_components > 1:
                print(f"  Main component: {component_sizes[0][1]:,} voxels")
                print(f"  Disconnected components: {num_components - 1}")
                for i in range(1, min(6, num_components)):
                    print(f"    Component {i+1}: {component_sizes[i][1]:,} voxels")
        
        # 2. ANALISI BUCHI (HOLES)
        # Chiudi morfologicamente e vedi cosa si aggiunge
        closed = binary_closing(binary_mask, structure=ball(3))
        holes = closed & (~binary_mask.astype(bool))
        num_hole_voxels = np.sum(holes)
        
        if self.verbose:
            print(f"\nHoles detected: {num_hole_voxels:,} voxels")
        
        # 3. STIMA GAP TRA COMPONENTI
        if num_components > 1:
            main_component = (labeled == component_sizes[0][0])
            main_coords = np.argwhere(main_component)
            
            # Calcola distanze minime per componenti disconnesse
            gap_info = []
            for i in range(1, min(6, num_components)):  # Analizza top 5 componenti
                comp_id, comp_size = component_sizes[i]
                comp_coords = np.argwhere(labeled == comp_id)
                
                # Sample per efficienza
                main_sample = main_coords[::max(1, len(main_coords)//200)]
                comp_sample = comp_coords[::max(1, len(comp_coords)//50)]
                
                # KD-Tree per ricerca veloce
                tree = cKDTree(main_sample * np.array([self.spacing[2], 
                                                        self.spacing[1], 
                                                        self.spacing[0]]))
                distances, _ = tree.query(comp_sample * np.array([self.spacing[2],
                                                                    self.spacing[1],
                                                                    self.spacing[0]]))
                min_distance = np.min(distances)
                
                gap_info.append({
                    'component_id': comp_id,
                    'size_voxels': comp_size,
                    'min_distance_mm': min_distance
                })
            
            if self.verbose:
                print(f"\nGap distances to main component:")
                for info in gap_info:
                    print(f"  Component {info['component_id']} "
                          f"({info['size_voxels']:,} voxels): "
                          f"{info['min_distance_mm']:.1f} mm")
        
        self.gap_analysis = {
            'num_components': num_components,
            'component_sizes': component_sizes,
            'num_hole_voxels': num_hole_voxels,
            'gap_info': gap_info if num_components > 1 else []
        }
        
        return self.gap_analysis
    
    def compute_preliminary_skeleton(self):
        """
        Calcola skeleton preliminare per guidare il riempimento dei gap
        """
        if self.verbose:
            print("\n" + "="*70)
            print("COMPUTING PRELIMINARY SKELETON")
            print("="*70)
        
        # Usa componente principale per skeleton
        binary_mask = (self.mask > 0).astype(np.uint8)
        
        structure = generate_binary_structure(3, 3)
        labeled, _ = label(binary_mask, structure=structure)
        
        # Prendi componente più grande
        component_sizes = [(i, np.sum(labeled == i)) for i in range(1, np.max(labeled) + 1)]
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        if len(component_sizes) > 0:
            main_component = (labeled == component_sizes[0][0])
        else:
            main_component = binary_mask
        
        if self.verbose:
            print("Computing skeleton (this may take a moment)...")
        
        self.preliminary_skeleton = skeletonize(main_component)
        
        if self.verbose:
            print(f"Skeleton computed: {np.sum(self.preliminary_skeleton):,} voxels")
        
        return self.preliminary_skeleton
    
    def fill_small_holes(self, max_hole_size_mm3=100):
        """
        Riempie piccoli buchi nella segmentazione
        """
        if self.verbose:
            print("\n" + "="*70)
            print(f"FILLING SMALL HOLES (< {max_hole_size_mm3} mm³)")
            print("="*70)
        
        binary_mask = (self.mask > 0).astype(np.uint8)
        
        # Calcola dimensione massima in voxel
        voxel_volume = self.spacing[0] * self.spacing[1] * self.spacing[2]
        max_hole_voxels = int(max_hole_size_mm3 / voxel_volume)
        
        # Trova buchi (regioni 0 circondate da 1)
        inverted = ~binary_mask.astype(bool)
        labeled_holes, num_holes = label(inverted)
        
        filled = binary_mask.copy()
        filled_count = 0
        filled_voxels = 0
        
        for hole_id in range(1, num_holes + 1):
            hole_mask = (labeled_holes == hole_id)
            hole_size = np.sum(hole_mask)
            
            # Riempie se abbastanza piccolo
            if hole_size <= max_hole_voxels:
                # Verifica HU: riempi solo se HU compatibile con aria
                hole_coords = np.argwhere(hole_mask)
                hu_values = self.img[hole_mask]
                
                # La maggioranza dei voxel deve avere HU da aria
                air_like = np.sum(hu_values < -400)
                
                if air_like / hole_size > 0.5:  # >50% HU da aria
                    filled[hole_mask] = 1
                    filled_count += 1
                    filled_voxels += hole_size
        
        if self.verbose:
            print(f"Filled {filled_count} small holes ({filled_voxels:,} voxels)")
        
        self.filled_mask = filled
        return filled
    
    def connect_nearby_components(self, max_bridge_distance_mm=10.0):
        """
        Connette componenti disconnesse vicine con "ponti" anatomicamente plausibili
        """
        if self.verbose:
            print("\n" + "="*70)
            print(f"CONNECTING NEARBY COMPONENTS (< {max_bridge_distance_mm} mm)")
            print("="*70)
        
        if self.filled_mask is None:
            self.filled_mask = self.mask.copy()
        
        binary_mask = (self.filled_mask > 0).astype(np.uint8)
        
        # Identifica componenti
        structure = generate_binary_structure(3, 3)
        labeled, num_components = label(binary_mask, structure=structure)
        
        if num_components <= 1:
            if self.verbose:
                print("Only one component, no connection needed")
            return self.filled_mask
        
        # Trova componente principale
        component_sizes = [(i, np.sum(labeled == i)) for i in range(1, num_components + 1)]
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        main_id = component_sizes[0][0]
        main_component = (labeled == main_id)
        main_coords = np.argwhere(main_component)
        
        connected_count = 0
        total_bridge_voxels = 0
        
        # Prova a connettere altre componenti
        for comp_id, comp_size in component_sizes[1:]:
            comp_mask = (labeled == comp_id)
            comp_coords = np.argwhere(comp_mask)
            
            # Trova punti più vicini
            min_distance, best_main_pt, best_comp_pt = self._find_closest_points(
                main_coords, comp_coords
            )
            
            if self.verbose:
                print(f"\nComponent {comp_id} ({comp_size:,} voxels): "
                      f"distance = {min_distance:.1f} mm")
            
            if min_distance <= max_bridge_distance_mm:
                # Crea ponte
                bridge = self._create_intelligent_bridge(
                    best_main_pt, best_comp_pt, min_distance
                )
                
                if bridge is not None:
                    self.filled_mask = self.filled_mask | bridge
                    main_coords = np.argwhere(self.filled_mask > 0)  # Aggiorna
                    connected_count += 1
                    bridge_voxels = np.sum(bridge)
                    total_bridge_voxels += bridge_voxels
                    
                    if self.verbose:
                        print(f"  ✓ Connected with bridge ({bridge_voxels} voxels)")
                else:
                    if self.verbose:
                        print(f"  ✗ Bridge not anatomically valid")
            else:
                if self.verbose:
                    print(f"  ✗ Too far to connect")
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Connected {connected_count} components")
            print(f"Total bridge voxels added: {total_bridge_voxels:,}")
        
        return self.filled_mask
    
    def _find_closest_points(self, coords1, coords2):
        """
        Trova i punti più vicini tra due set di coordinate
        """
        # Sample per efficienza
        sample1 = coords1[::max(1, len(coords1)//100)]
        sample2 = coords2[::max(1, len(coords2)//50)]
        
        # Converti in mm
        sample1_mm = sample1 * np.array([self.spacing[2], self.spacing[1], self.spacing[0]])
        sample2_mm = sample2 * np.array([self.spacing[2], self.spacing[1], self.spacing[0]])
        
        # KD-Tree
        tree = cKDTree(sample1_mm)
        distances, indices = tree.query(sample2_mm)
        
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        best_pt1 = sample1[indices[min_idx]]
        best_pt2 = sample2[min_idx]
        
        return min_distance, best_pt1, best_pt2
    
    def _create_intelligent_bridge(self, pt1, pt2, distance_mm):
        """
        Crea un ponte tra due punti, validando che segua HU compatibili con aria
        """
        pt1 = np.array(pt1, dtype=int)
        pt2 = np.array(pt2, dtype=int)
        
        # Numero di punti lungo la linea (più denso per distanze maggiori)
        num_points = max(5, int(distance_mm / np.mean(self.spacing)))
        
        # Genera punti lungo la linea
        bridge_mask = np.zeros_like(self.mask, dtype=bool)
        
        for t in np.linspace(0, 1, num_points):
            center = pt1 + t * (pt2 - pt1)
            z, y, x = center.astype(int)
            
            # Bounds check
            if not (0 <= z < self.mask.shape[0] and
                    0 <= y < self.mask.shape[1] and
                    0 <= x < self.mask.shape[2]):
                continue
            
            # Verifica HU
            hu = self.img[z, y, x]
            
            # Ponte valido solo se HU compatibile con aria (con tolleranza)
            if hu > -300:  # Troppo denso, probabilmente tessuto
                return None  # Bridge non valido
            
            # Aggiungi piccola sfera per robustezza
            for dz in range(-1, 2):
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if (0 <= nz < self.mask.shape[0] and
                            0 <= ny < self.mask.shape[1] and
                            0 <= nx < self.mask.shape[2]):
                            
                            # Controlla anche HU dei vicini
                            if self.img[nz, ny, nx] < -300:
                                bridge_mask[nz, ny, nx] = True
        
        # Valida che il ponte abbia abbastanza voxel
        if np.sum(bridge_mask) < 3:
            return None
        
        return bridge_mask.astype(np.uint8)
    
    def morphological_refinement(self):
        """
        Refinement morfologico per smoothing superficie
        """
        if self.verbose:
            print("\n" + "="*70)
            print("MORPHOLOGICAL REFINEMENT")
            print("="*70)
        
        if self.filled_mask is None:
            self.filled_mask = self.mask.copy()
        
        # Closing per colmare piccoli gap
        filled = binary_closing(self.filled_mask, structure=ball(2))
        
        # Smoothing leggero
        filled = binary_dilation(filled, structure=ball(1))
        filled = binary_erosion(filled, structure=ball(1))
        
        self.filled_mask = filled.astype(np.uint8)
        
        if self.verbose:
            print("Morphological operations complete")
        
        return self.filled_mask
    
    def full_gap_filling_pipeline(self, 
                                  max_hole_size_mm3=100,
                                  max_bridge_distance_mm=10.0,
                                  enable_morphological_refinement=True):
        """
        Pipeline completa di riempimento gap
        """
        if self.verbose:
            print("\n" + "="*80)
            print(" "*20 + "INTELLIGENT GAP FILLING PIPELINE")
            print("="*80)
        
        initial_voxels = np.sum(self.mask > 0)
        
        # Step 1: Analisi
        self.analyze_gaps_and_holes()
        
        # Step 2: Skeleton preliminare (opzionale ma utile)
        # self.compute_preliminary_skeleton()
        
        # Step 3: Riempi piccoli buchi
        self.fill_small_holes(max_hole_size_mm3=max_hole_size_mm3)
        
        # Step 4: Connetti componenti vicine
        self.connect_nearby_components(max_bridge_distance_mm=max_bridge_distance_mm)
        
        # Step 5: Refinement morfologico
        if enable_morphological_refinement:
            self.morphological_refinement()
        
        final_voxels = np.sum(self.filled_mask > 0)
        added_voxels = final_voxels - initial_voxels
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("GAP FILLING RESULTS")
            print(f"{'='*80}")
            print(f"Initial voxels: {initial_voxels:,}")
            print(f"Final voxels: {final_voxels:,}")
            print(f"Added: {added_voxels:,} ({added_voxels/initial_voxels*100:.1f}%)")
        
        return self.filled_mask
    
    def save_filled_mask(self, output_path, reference_image):
        """
        Salva maschera riempita
        """
        if self.filled_mask is None:
            raise ValueError("Run gap filling pipeline first")
        
        filled_sitk = sitk.GetImageFromArray(self.filled_mask.astype(np.uint8))
        filled_sitk.CopyInformation(reference_image)
        sitk.WriteImage(filled_sitk, output_path)
        
        if self.verbose:
            print(f"\n✓ Saved filled mask: {output_path}")
        
        return output_path
    
    def visualize_gaps_filled(self, save_path=None):
        """
        Visualizza differenza prima/dopo
        """
        if self.filled_mask is None:
            raise ValueError("Run gap filling first")
        
        fig = plt.figure(figsize=(18, 6))
        
        # Original
        ax1 = fig.add_subplot(131, projection='3d')
        original_coords = np.argwhere(self.mask > 0)
        subsample = max(1, len(original_coords) // 5000)
        original_coords = original_coords[::subsample]
        ax1.scatter(original_coords[:, 2], original_coords[:, 1], original_coords[:, 0],
                   c='blue', s=1, alpha=0.5)
        ax1.set_title(f'Original\n({np.sum(self.mask > 0):,} voxels)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Filled
        ax2 = fig.add_subplot(132, projection='3d')
        filled_coords = np.argwhere(self.filled_mask > 0)
        subsample = max(1, len(filled_coords) // 5000)
        filled_coords = filled_coords[::subsample]
        ax2.scatter(filled_coords[:, 2], filled_coords[:, 1], filled_coords[:, 0],
                   c='green', s=1, alpha=0.5)
        ax2.set_title(f'Gap Filled\n({np.sum(self.filled_mask > 0):,} voxels)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Added voxels
        ax3 = fig.add_subplot(133, projection='3d')
        added = (self.filled_mask > 0) & (self.mask == 0)
        added_coords = np.argwhere(added)
        if len(added_coords) > 0:
            subsample = max(1, len(added_coords) // 5000)
            added_coords = added_coords[::subsample]
            ax3.scatter(added_coords[:, 2], added_coords[:, 1], added_coords[:, 0],
                       c='red', s=2, alpha=0.7)
        ax3.set_title(f'Added Voxels\n({np.sum(added):,} voxels)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()


# ============================================================================
# INTEGRAZIONE NELLA PIPELINE
# ============================================================================

def integrate_gap_filling_into_pipeline(mhd_path, airway_mask_path, output_dir,
                                       max_hole_size_mm3=100,
                                       max_bridge_distance_mm=10.0):
    """
    Integra gap filling nella pipeline esistente
    
    Args:
        mhd_path: Path al CT originale (per HU values)
        airway_mask_path: Path alla segmentazione airway
        output_dir: Directory output
        max_hole_size_mm3: Dimensione massima buchi da riempire
        max_bridge_distance_mm: Distanza massima per connettere componenti
    
    Returns:
        filled_mask_path: Path alla mask riempita
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Carica CT e mask
    ct_img = sitk.ReadImage(mhd_path)
    ct_np = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()
    
    airway_img = sitk.ReadImage(airway_mask_path)
    airway_np = sitk.GetArrayFromImage(airway_img)
    
    print("\n" + "="*80)
    print(" "*25 + "GAP FILLING MODULE")
    print("="*80)
    print(f"CT image: {mhd_path}")
    print(f"Airway mask: {airway_mask_path}")
    print(f"Spacing: {spacing} mm")
    
    # Crea gap filler
    gap_filler = IntelligentAirwayGapFiller(
        ct_np, 
        airway_np, 
        spacing,
        verbose=True
    )
    
    # Run pipeline
    filled_mask = gap_filler.full_gap_filling_pipeline(
        max_hole_size_mm3=max_hole_size_mm3,
        max_bridge_distance_mm=max_bridge_distance_mm,
        enable_morphological_refinement=True
    )
    
    # Salva
    scan_name = os.path.splitext(os.path.basename(airway_mask_path))[0]
    output_path = os.path.join(output_dir, f"{scan_name}_gap_filled.nii.gz")
    gap_filler.save_filled_mask(output_path, airway_img)
    
    # Visualizza
    viz_path = os.path.join(output_dir, "gap_filling_visualization.png")
    gap_filler.visualize_gaps_filled(save_path=viz_path)
    
    return output_path, gap_filler

'''
# ============================================================================
# ESEMPIO DI USO
# ============================================================================

if __name__ == "__main__":
    # Test con un file
    mhd_path = "path/to/ct.mhd"
    airway_mask_path = "path/to/airway_segmentation.nii.gz"
    output_dir = "gap_filled_output"
    
    filled_path, filler = integrate_gap_filling_into_pipeline(
        mhd_path,
        airway_mask_path,
        output_dir,
        max_hole_size_mm3=100,
        max_bridge_distance_mm=10.0
    )
    
    print(f"\n✓ Gap-filled mask saved: {filled_path}")'''