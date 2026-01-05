"""
VALIDATION STEP 2: SKELETON & GRAPH QUALITY
============================================

Valida qualità dello skeleton e del grafo estratto:
- Topologia del grafo
- Connettività
- Biforcazioni vs endpoints
- Lunghezza branch
- Distribuzione gradi nodi

REFERENCE:
- Strahler order distribution
- Bifurcation ratios
- Branch length distribution
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# ============================================================================
# REFERENCE VALUES
# ============================================================================

AIRWAY_GRAPH_REFERENCE = {
    "bifurcation_to_endpoint_ratio": {
        "expected": 0.5,  # Weibel: ogni biforcazione genera 2 endpoints
        "min": 0.3,
        "max": 0.8,
        "source": "Weibel 1963 - Symmetric branching"
    },
    "mean_branch_length_mm": {
        "min": 5,
        "expected_min": 8,
        "expected_max": 25,
        "max": 40,
        "source": "Airway morphometry literature"
    },
    "max_strahler_order": {
        "min": 3,
        "expected_min": 5,
        "expected_max": 12,
        "max": 15,
        "source": "Hierarchical branching complexity"
    },
    "graph_connectivity": {
        "optimal": 1.0,  # Fully connected
        "acceptable": 0.95,
        "warning": 0.90,
        "source": "Quality metric"
    }
}

VESSEL_GRAPH_REFERENCE = {
    "bifurcation_to_endpoint_ratio": {
        "expected": 0.5,
        "min": 0.2,
        "max": 1.0,
        "source": "Vascular branching patterns"
    },
    "mean_branch_length_mm": {
        "min": 3,
        "expected_min": 5,
        "expected_max": 20,
        "max": 30,
        "source": "Vascular morphometry"
    },
    "max_strahler_order": {
        "min": 3,
        "expected_min": 4,
        "expected_max": 10,
        "max": 12,
        "source": "Hierarchical complexity"
    },
    "graph_connectivity": {
        "optimal": 1.0,
        "acceptable": 0.90,
        "warning": 0.80,
        "source": "Vessels più frammentabili"
    }
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def analyze_skeleton_graph(mask_path, spacing=None, is_airway=True):
    """
    Analizza skeleton e grafo
    
    Returns dict con:
    - skeleton_quality
    - graph_topology
    - branch_metrics
    - quality_scores
    """
    
    # Load mask
    sitk_img = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_img)
    
    if spacing is None:
        spacing = sitk_img.GetSpacing()
    
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    
    # Compute skeleton (skeletonize automatically handles 2D and 3D)
    skeleton = skeletonize(mask > 0)
    
    skel_voxels = np.sum(skeleton)
    
    if skel_voxels < 10:
        return create_empty_result()
    
    # Build graph with skan
    try:
        skel_obj = Skeleton(skeleton, spacing=spacing_zyx)
        branch_data = summarize(skel_obj, separator='-')
    except Exception as e:
        print(f"  Warning: skan failed - {e}")
        return create_empty_result()
    
    if len(branch_data) == 0:
        return create_empty_result()
    
    # ========================================
    # GRAPH TOPOLOGY
    # ========================================
    G = nx.Graph()
    
    coordinates = skel_obj.coordinates
    for idx in range(len(coordinates)):
        G.add_node(idx, pos=coordinates[idx])
    
    for _, row in branch_data.iterrows():
        src = int(row['node-id-src'])
        dst = int(row['node-id-dst'])
        length = row['branch-distance']
        G.add_edge(src, dst, length=length)
    
    # Basic topology
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Degree distribution
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    endpoints = sum(1 for d in degree_values if d == 1)
    bifurcations = sum(1 for d in degree_values if d >= 3)
    
    bif_to_endpoint_ratio = bifurcations / endpoints if endpoints > 0 else 0
    
    # Connectivity
    num_components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    connectivity_ratio = len(largest_cc) / num_nodes
    
    # Branch lengths
    branch_lengths = branch_data['branch-distance'].values * np.mean(spacing)
    mean_branch_length = float(np.mean(branch_lengths))
    median_branch_length = float(np.median(branch_lengths))
    std_branch_length = float(np.std(branch_lengths))
    
    # Strahler order (proxy for complexity)
    try:
        strahler_orders = []
        for node in G.nodes():
            # Simplified Strahler: max degree of neighbors
            neighbor_degrees = [G.degree(n) for n in G.neighbors(node)]
            if neighbor_degrees:
                strahler_orders.append(max(neighbor_degrees))
            else:
                strahler_orders.append(1)
        
        max_strahler = max(strahler_orders) if strahler_orders else 0
    except:
        max_strahler = 0
    
    # ========================================
    # QUALITY SCORING
    # ========================================
    ref = AIRWAY_GRAPH_REFERENCE if is_airway else VESSEL_GRAPH_REFERENCE
    
    quality_scores = {}
    
    # 1. Bifurcation ratio score
    bif_ratio_score = 100
    expected = ref["bifurcation_to_endpoint_ratio"]["expected"]
    if bif_to_endpoint_ratio < ref["bifurcation_to_endpoint_ratio"]["min"]:
        bif_ratio_score = 50 * (bif_to_endpoint_ratio / ref["bifurcation_to_endpoint_ratio"]["min"])
    elif bif_to_endpoint_ratio > ref["bifurcation_to_endpoint_ratio"]["max"]:
        excess = (bif_to_endpoint_ratio - ref["bifurcation_to_endpoint_ratio"]["max"]) / ref["bifurcation_to_endpoint_ratio"]["max"]
        bif_ratio_score = max(50, 100 - 50 * excess)
    
    quality_scores['bifurcation_ratio'] = bif_ratio_score
    
    # 2. Branch length score
    branch_score = 100
    if mean_branch_length < ref["mean_branch_length_mm"]["expected_min"]:
        branch_score = 70 * (mean_branch_length / ref["mean_branch_length_mm"]["expected_min"])
    elif mean_branch_length > ref["mean_branch_length_mm"]["expected_max"]:
        excess = (mean_branch_length - ref["mean_branch_length_mm"]["expected_max"]) / ref["mean_branch_length_mm"]["expected_max"]
        branch_score = max(50, 100 - 30 * excess)
    
    quality_scores['branch_length'] = branch_score
    
    # 3. Complexity score (Strahler)
    complexity_score = 100
    if max_strahler < ref["max_strahler_order"]["expected_min"]:
        complexity_score = 60 * (max_strahler / ref["max_strahler_order"]["expected_min"])
    elif max_strahler > ref["max_strahler_order"]["expected_max"]:
        excess = (max_strahler - ref["max_strahler_order"]["expected_max"]) / ref["max_strahler_order"]["expected_max"]
        complexity_score = max(70, 100 - 20 * excess)
    
    quality_scores['complexity'] = complexity_score
    
    # 4. Connectivity score
    conn_score = 100
    if connectivity_ratio < ref["graph_connectivity"]["warning"]:
        conn_score = 50
    elif connectivity_ratio < ref["graph_connectivity"]["acceptable"]:
        conn_score = 75
    
    quality_scores['connectivity'] = conn_score
    
    overall_quality = np.mean(list(quality_scores.values()))
    
    # ========================================
    # SEVERITY FLAGS
    # ========================================
    severity_flags = []
    
    if connectivity_ratio < ref["graph_connectivity"]["acceptable"]:
        severity_flags.append("WARNING_FRAGMENTED_GRAPH")
    
    if mean_branch_length < ref["mean_branch_length_mm"]["min"]:
        severity_flags.append("CRITICAL_BRANCHES_TOO_SHORT")
    
    if max_strahler < ref["max_strahler_order"]["min"]:
        severity_flags.append("WARNING_LOW_COMPLEXITY")
    
    if bif_to_endpoint_ratio < ref["bifurcation_to_endpoint_ratio"]["min"] or \
       bif_to_endpoint_ratio > ref["bifurcation_to_endpoint_ratio"]["max"]:
        severity_flags.append("WARNING_ABNORMAL_TOPOLOGY")
    
    graph_usable = (connectivity_ratio >= ref["graph_connectivity"]["warning"] and
                    mean_branch_length >= ref["mean_branch_length_mm"]["min"] and
                    max_strahler >= ref["max_strahler_order"]["min"])
    
    # ========================================
    # RETURN
    # ========================================
    return {
        # Skeleton
        "skeleton_voxels": int(skel_voxels),
        
        # Graph topology
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "num_components": int(num_components),
        "connectivity_ratio": float(connectivity_ratio),
        
        # Degree distribution
        "endpoints": int(endpoints),
        "bifurcations": int(bifurcations),
        "bifurcation_to_endpoint_ratio": float(bif_to_endpoint_ratio),
        "mean_degree": float(np.mean(degree_values)),
        "max_degree": int(max(degree_values)),
        
        # Branch metrics
        "num_branches": int(len(branch_lengths)),
        "mean_branch_length_mm": float(mean_branch_length),
        "median_branch_length_mm": float(median_branch_length),
        "std_branch_length_mm": float(std_branch_length),
        "min_branch_length_mm": float(np.min(branch_lengths)),
        "max_branch_length_mm": float(np.max(branch_lengths)),
        
        # Complexity
        "max_strahler_order": int(max_strahler),
        
        # Quality
        "quality_bifurcation_ratio": float(quality_scores['bifurcation_ratio']),
        "quality_branch_length": float(quality_scores['branch_length']),
        "quality_complexity": float(quality_scores['complexity']),
        "quality_connectivity": float(quality_scores['connectivity']),
        "quality_overall": float(overall_quality),
        
        # Flags
        "GRAPH_USABLE": graph_usable,
        "severity_flags": ";".join(severity_flags) if severity_flags else "NONE"
    }


def create_empty_result():
    """Returns empty result for failed cases"""
    return {
        "skeleton_voxels": 0,
        "num_nodes": 0,
        "num_edges": 0,
        "num_components": 0,
        "connectivity_ratio": 0.0,
        "endpoints": 0,
        "bifurcations": 0,
        "bifurcation_to_endpoint_ratio": 0.0,
        "mean_degree": 0.0,
        "max_degree": 0,
        "num_branches": 0,
        "mean_branch_length_mm": 0.0,
        "median_branch_length_mm": 0.0,
        "std_branch_length_mm": 0.0,
        "min_branch_length_mm": 0.0,
        "max_branch_length_mm": 0.0,
        "max_strahler_order": 0,
        "quality_bifurcation_ratio": 0.0,
        "quality_branch_length": 0.0,
        "quality_complexity": 0.0,
        "quality_connectivity": 0.0,
        "quality_overall": 0.0,
        "GRAPH_USABLE": False,
        "severity_flags": "CRITICAL_EMPTY_GRAPH"
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def validate_airways_graphs_batch(data_root, output_csv):
    """Valida batch airway graphs"""
    print("="*80)
    print("AIRWAY GRAPH VALIDATION - STEP 2")
    print("="*80)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    results = []
    
    for case_id in sorted(os.listdir(data_root)):
        step3_dir = os.path.join(data_root, case_id, "step3_preprocessing")
        
        cleaned_mask = os.path.join(step3_dir, "cleaned_airway_mask_complete.nii.gz")
        
        if not os.path.exists(cleaned_mask):
            print(f"[SKIP] {case_id} - missing cleaned mask")
            continue
        
        print(f"[VALIDATING] {case_id}")
        
        try:
            row = analyze_skeleton_graph(cleaned_mask, is_airway=True)
            row["case_id"] = case_id
            results.append(row)
            
            if row["GRAPH_USABLE"]:
                print(f"  ✓ USABLE - Quality: {row['quality_overall']:.1f}/100")
            else:
                print(f"  ✗ UNUSABLE - {row['severity_flags']}")
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print_summary(df, "AIRWAY GRAPHS")
    
    return df


def validate_vessels_graphs_batch(data_root, output_csv):
    """Valida batch vessel graphs"""
    print("="*80)
    print("VESSEL GRAPH VALIDATION - STEP 2")
    print("="*80)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    results = []
    
    for case_id in sorted(os.listdir(data_root)):
        step1_dir = os.path.join(data_root, case_id, "step1_segmentation")
        
        # Try different names
        vessel_mask = None
        for name in [f"{case_id}_vessels_raw_cleaned.nii.gz",
                     f"{case_id}_vessels_raw.nii.gz",
                     "vessels_combined.nii.gz"]:
            path = os.path.join(step1_dir, name)
            if os.path.exists(path):
                vessel_mask = path
                break
        
        if vessel_mask is None:
            print(f"[SKIP] {case_id} - vessel mask not found")
            continue
        
        print(f"[VALIDATING] {case_id}")
        
        try:
            row = analyze_skeleton_graph(vessel_mask, is_airway=False)
            row["case_id"] = case_id
            results.append(row)
            
            if row["GRAPH_USABLE"]:
                print(f"  ✓ USABLE - Quality: {row['quality_overall']:.1f}/100")
            else:
                print(f"  ✗ UNUSABLE - {row['severity_flags']}")
        
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            continue
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print_summary(df, "VESSEL GRAPHS")
    
    return df


def print_summary(df, structure_type):
    """Print validation summary"""
    print("\n" + "="*80)
    print(f"{structure_type} VALIDATION SUMMARY")
    print("="*80)
    
    print(f"Total cases: {len(df)}")
    print(f"Usable: {df['GRAPH_USABLE'].sum()} ({df['GRAPH_USABLE'].sum()/len(df)*100:.1f}%)")
    
    print(f"\nQuality Distribution:")
    print(f"  Excellent (>90): {(df['quality_overall'] > 90).sum()}")
    print(f"  Good (70-90): {((df['quality_overall'] >= 70) & (df['quality_overall'] <= 90)).sum()}")
    print(f"  Fair (50-70): {((df['quality_overall'] >= 50) & (df['quality_overall'] < 70)).sum()}")
    print(f"  Poor (<50): {(df['quality_overall'] < 50).sum()}")
    
    print(f"\nMean Quality: {df['quality_overall'].mean():.1f}/100")
    
    print(f"\nTopology:")
    print(f"  Mean branches: {df['num_branches'].mean():.1f}")
    print(f"  Mean branch length: {df['mean_branch_length_mm'].mean():.1f} mm")
    print(f"  Mean Strahler order: {df['max_strahler_order'].mean():.1f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_graph_validation_plots(df, structure_type, output_dir):
    """Crea plot validazione grafo"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Quality distribution
    axes[0, 0].hist(df['quality_overall'], bins=20, edgecolor='black', color='steelblue', alpha=0.7)
    axes[0, 0].axvline(df['quality_overall'].mean(), color='r', linestyle='--', 
                       label=f'Mean: {df["quality_overall"].mean():.1f}')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'{structure_type} Graph Quality')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Branch length distribution
    axes[0, 1].hist(df['mean_branch_length_mm'], bins=20, edgecolor='black', color='seagreen', alpha=0.7)
    axes[0, 1].set_xlabel('Mean Branch Length (mm)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Branch Length Distribution')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Bifurcation ratio
    axes[0, 2].hist(df['bifurcation_to_endpoint_ratio'], bins=20, edgecolor='black', color='orange', alpha=0.7)
    axes[0, 2].set_xlabel('Bifurcation/Endpoint Ratio')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Topology: Bifurcation Ratio')
    axes[0, 2].grid(alpha=0.3)
    
    # 4. Strahler order
    axes[1, 0].hist(df['max_strahler_order'], bins=range(0, int(df['max_strahler_order'].max())+2), 
                    edgecolor='black', color='crimson', alpha=0.7)
    axes[1, 0].set_xlabel('Max Strahler Order')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Complexity: Strahler Order')
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Connectivity
    axes[1, 1].hist(df['connectivity_ratio']*100, bins=20, edgecolor='black', color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Connectivity (%)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Graph Connectivity')
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Quality components
    quality_cols = ['quality_bifurcation_ratio', 'quality_branch_length', 
                    'quality_complexity', 'quality_connectivity']
    quality_means = [df[col].mean() for col in quality_cols]
    labels = ['Bifurcation', 'Branch Length', 'Complexity', 'Connectivity']
    
    axes[1, 2].bar(labels, quality_means, color=['steelblue', 'seagreen', 'orange', 'purple'], 
                   edgecolor='black', alpha=0.7)
    axes[1, 2].set_ylabel('Mean Quality Score')
    axes[1, 2].set_title('Quality Components')
    axes[1, 2].set_ylim([0, 100])
    axes[1, 2].grid(alpha=0.3, axis='y')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'{structure_type} Graph Validation - Step 2', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'validation_step2_{structure_type.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved validation plot: {plot_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # AIRWAYS
    AIRWAY_DATA_ROOT = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\output_results_with_fibrosis"
    AIRWAY_OUTPUT_CSV = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\output\step2_graph_quality\airway_validation_step2_graph.csv"
    AIRWAY_PLOT_DIR = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\output\step2_graph_quality"
    
    if os.path.exists(AIRWAY_DATA_ROOT):
        df_airways = validate_airways_graphs_batch(AIRWAY_DATA_ROOT, AIRWAY_OUTPUT_CSV)
        create_graph_validation_plots(df_airways, "AIRWAY GRAPHS", AIRWAY_PLOT_DIR)
    
    # VESSELS
    VESSEL_DATA_ROOT = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\vessel_segmentation_new\vessel_output"
    VESSEL_OUTPUT_CSV = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\output\step2_graph_quality\vessel_validation_step2_graph.csv"
    VESSEL_PLOT_DIR = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\output\step2_graph_quality"
    
    if os.path.exists(VESSEL_DATA_ROOT):
        df_vessels = validate_vessels_graphs_batch(VESSEL_DATA_ROOT, VESSEL_OUTPUT_CSV)
        create_graph_validation_plots(df_vessels, "VESSEL GRAPHS", VESSEL_PLOT_DIR)