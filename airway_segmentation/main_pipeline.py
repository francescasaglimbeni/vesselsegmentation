import os
import sys

from preprocessin_cleaning import SegmentationPreprocessor
from airway_graph import AirwayGraphAnalyzer


def main():
    """
    Complete airway analysis pipeline (REVISED ORDER):
    1. Preprocessing and cleaning (remove artifacts, reconnect components)
    2. Skeletonization and initial graph construction
    3. Complete graph analysis on bronchi only (branches, diameters, lengths, bifurcations)
    4. Final summary
    """
    
    print("="*80)
    print(" "*20 + "AIRWAY ANALYSIS PIPELINE (REVISED)")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION - MODIFY THESE PATHS
    # ========================================================================
    
    # Input: Original airway segmentation (trachea + bronchi from TotalSegmentator)
    original_mask_path = "airway_segmentation/segm_cutted.seg.nrrd"
    
    # Output: Root folder where all results will be saved
    output_root = "airway_pipeline_results_v2"
    
    # ========================================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================================
    
    step1_dir = os.path.join(output_root, "step1_preprocessing")
    step2_dir = os.path.join(output_root, "step2_initial_skeleton")
    step3_dir = os.path.join(output_root, "step3_carina_removal")
    step4_dir = os.path.join(output_root, "step4_final_analysis")
    
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(step1_dir, exist_ok=True)
    os.makedirs(step2_dir, exist_ok=True)
    os.makedirs(step3_dir, exist_ok=True)
    os.makedirs(step4_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: PREPROCESSING AND CLEANING
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: PREPROCESSING AND CLEANING")
    print("="*80)
    print("Goal: Clean the segmentation by removing artifacts and disconnected")
    print("      components, and attempt to reconnect nearby components")
    
    if not os.path.exists(original_mask_path):
        print(f"\nERROR: Input file not found: {original_mask_path}")
        print("Please update the 'original_mask_path' variable with the correct path.")
        sys.exit(1)
    
    try:
        preprocessor = SegmentationPreprocessor(original_mask_path)
        
        cleaned_mask, cleaned_path = preprocessor.run_full_preprocessing(
            output_dir=step1_dir,
            try_reconnection=True,          # Attempt to reconnect nearby components
            max_reconnect_distance_mm=10.0, # Maximum distance for reconnection (mm)
            min_component_size=50,          # Minimum size to consider for reconnection
            visualize=True
        )
        
        print(f"\n✓ Step 1 completed successfully!")
        print(f"  Output: {cleaned_path}")
        
    except Exception as e:
        print(f"\n✗ Error in Step 1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: SKELETONIZATION AND INITIAL GRAPH CONSTRUCTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: SKELETONIZATION AND INITIAL GRAPH CONSTRUCTION")
    print("="*80)
    print("Goal: Create 3D skeleton and build initial graph structure")
    print("      (includes trachea + bronchi)")
    
    try:
        analyzer = AirwayGraphAnalyzer(cleaned_path)
        
        # Compute skeleton
        analyzer.compute_skeleton()
        
        # Analyze and manage skeleton components
        analyzer.analyze_connected_components()
        analyzer.smart_component_management(
            max_reconnect_distance_mm=15.0,
            min_voxels_for_reconnect=5,
            max_voxels_for_keep=100,
            remove_tiny_components=True
        )
        
        # Reanalyze after management
        analyzer.analyze_connected_components()
        
        # Build graph
        analyzer.build_graph()
        
        # Save intermediate results
        analyzer.save_results(step2_dir)
        
        print(f"\n✓ Step 2 completed successfully!")
        print(f"  Skeleton created with {np.sum(analyzer.skeleton > 0):,} voxels")
        print(f"  Graph built with {len(analyzer.graph.nodes())} nodes")
        
    except Exception as e:
        print(f"\n✗ Error in Step 2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    
    # ========================================================================
    # STEP 3: COMPLETE GRAPH ANALYSIS ON BRONCHI ONLY
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: COMPLETE GRAPH ANALYSIS ON BRONCHI ONLY")
    print("="*80)
    print("Goal: Re-analyze the bronchial tree (without trachea) to compute:")
    print("      - Branch diameters")
    print("      - Branch lengths")
    print("      - Bifurcation points")
    print("      - Complete topological metrics")
    
    try:
        # Rebuild graph on cropped skeleton
        print("\nRebuilding graph on bronchi-only skeleton...")
        analyzer.compute_skeleton()  # Recompute on updated mask
        analyzer.analyze_connected_components()
        analyzer.build_graph()
        
        # Calculate lengths
        analyzer.calculate_branch_lengths()
        
        # Calculate diameters
        analyzer.analyze_diameters()
        
        # Merge metrics
        analyzer.merge_branch_metrics()
        
        # Identify bifurcations
        analyzer.identify_bifurcations()
        
        # Save final results
        analyzer.save_results(step4_dir)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        analyzer.visualize_skeleton_3d(
            save_path=os.path.join(step4_dir, "bronchi_skeleton_3d.png"),
            show_bifurcations=False
        )
        
        analyzer.visualize_connected_components(
            save_path=os.path.join(step4_dir, "bronchi_components.png")
        )
        
        analyzer.visualize_branches_3d(
            save_path=os.path.join(step4_dir, "bronchi_branches_diameter.png"),
            color_by='diameter'
        )
        
        analyzer.visualize_branches_3d(
            save_path=os.path.join(step4_dir, "bronchi_branches_length.png"),
            color_by='length'
        )
        
        analyzer.plot_diameter_distribution(
            save_path=os.path.join(step4_dir, "diameter_distribution.png")
        )
        
        analyzer.plot_length_distribution(
            save_path=os.path.join(step4_dir, "length_distribution.png")
        )
        
        print(f"\n✓ Step 4 completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error in Step 4: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # STEP 4: FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print(" "*25 + "PIPELINE COMPLETED!")
    print("="*80)
    
    print("\n📁 Results saved in:")
    print(f"  {output_root}/")
    print(f"     ├── step1_preprocessing/")
    print(f"     │   ├── cleaned_airway_mask.nii.gz")
    print(f"     │   ├── preprocessing_report.txt")
    print(f"     │   ├── component_statistics.csv")
    print(f"     │   └── visualizations (PNG)")
    print(f"     │")
    print(f"     ├── step2_initial_skeleton/")
    print(f"     │   ├── skeleton.nii.gz (with trachea)")
    print(f"     │   ├── analysis_summary.txt")
    print(f"     │   └── component_statistics.csv")
    print(f"     │")
    print(f"     └── step3_final_analysis/")
    print(f"         ├── skeleton.nii.gz (bronchi only)")
    print(f"         ├── branch_metrics_complete.csv    ⭐ MAIN OUTPUT")
    print(f"         ├── bifurcations.csv")
    print(f"         ├── component_statistics.csv")
    print(f"         ├── analysis_summary.txt")
    print(f"         └── visualizations (PNG)")
    
    print("\n📊 Key output files:")
    print(f"   • branch_metrics_complete.csv - Contains all bronchial branch measurements:")
    print(f"     - Diameters (mean, min, max, std)")
    print(f"     - Lengths (mm)")
    print(f"     - Volumes (mm³)")
    print(f"     - Surface areas (mm²)")
    print(f"   • branch_classification_trachea_vs_bronchi.csv - Classification of all branches")
    
    if hasattr(analyzer, 'branch_metrics_df'):
        print(f"\n📈 Final Analysis Statistics (BRONCHI ONLY):")
        print(f"   • Total bronchial branches analyzed: {len(analyzer.branch_metrics_df)}")
        print(f"   • Total bronchial tree length: {analyzer.branch_metrics_df['length_mm'].sum():.2f} mm")
        print(f"   • Total bronchial tree volume: {analyzer.branch_metrics_df['volume_mm3'].sum():.2f} mm³")
        print(f"   • Average branch diameter: {analyzer.branch_metrics_df['diameter_mean_mm'].mean():.2f} mm")
        print(f"   • Average branch length: {analyzer.branch_metrics_df['length_mm'].mean():.2f} mm")
        print(f"   • Diameter range: {analyzer.branch_metrics_df['diameter_mean_mm'].min():.2f} - "
              f"{analyzer.branch_metrics_df['diameter_mean_mm'].max():.2f} mm")
        print(f"   • Length range: {analyzer.branch_metrics_df['length_mm'].min():.2f} - "
              f"{analyzer.branch_metrics_df['length_mm'].max():.2f} mm")
    
    if hasattr(analyzer, 'bifurcations_df'):
        print(f"   • Bifurcations detected: {len(analyzer.bifurcations_df)}")
    
   
    print("\n" + "="*80)
    print("Pipeline execution completed successfully! ✓")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Need numpy for the analyzer
    import numpy as np
    main()