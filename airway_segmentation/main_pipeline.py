import os
import sys
import numpy as np

from preprocessin_cleaning import SegmentationPreprocessor
from airway_graph import AirwayGraphAnalyzer


def main():
    """
    Complete airway analysis pipeline with Weibel generation analysis:
    1. Preprocessing with trachea removal (identifies carina, keeps bronchi only)
    2. Component reconnection using PATH-BASED distances
    3. Skeletonization and graph construction on bronchi
    4. Weibel generation assignment and tapering analysis
    5. Complete branch metrics (diameter, length, generation, distance from carina)
    6. Comprehensive visualizations and reports
    """
    
    print("="*80)
    print(" "*15 + "AIRWAY ANALYSIS PIPELINE WITH WEIBEL MODEL")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION - MODIFY THESE PATHS
    # ========================================================================
    
    # Input: Original airway segmentation (trachea + bronchi from TotalSegmentator)
    original_mask_path = "airway_segmentation/segm_cutted.seg.nrrd"
    
    # Output: Root folder where all results will be saved
    output_root = "airway_pipeline_results_weibel"
    
    # ========================================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================================
    
    step1_dir = os.path.join(output_root, "step1_preprocessing_with_trachea_removal")
    step2_dir = os.path.join(output_root, "step2_bronchial_tree_analysis")
    
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(step1_dir, exist_ok=True)
    os.makedirs(step2_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: PREPROCESSING WITH TRACHEA REMOVAL
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: PREPROCESSING WITH TRACHEA REMOVAL")
    print("="*80)
    print("This step will:")
    print("  1. Analyze and clean connected components")
    print("  2. Compute preliminary skeleton")
    print("  3. Identify carina (largest bifurcation)")
    print("  4. Remove trachea (keep only tissue from carina onwards)")
    print("  5. Reconnect nearby components using PATH-BASED distances")
    
    if not os.path.exists(original_mask_path):
        print(f"\n❌ ERROR: Input file not found: {original_mask_path}")
        print("Please update the 'original_mask_path' variable with the correct path.")
        sys.exit(1)
    
    try:
        preprocessor = SegmentationPreprocessor(original_mask_path)
        
        cleaned_mask, cleaned_path = preprocessor.run_full_preprocessing(
            output_dir=step1_dir,
            try_reconnection=True,          # Attempt to reconnect nearby components
            max_reconnect_distance_mm=15.0, # Maximum PATH distance for reconnection (mm)
            min_component_size=50,          # Minimum size to consider for reconnection
            visualize=True
        )
        
        print(f"\n✓ Step 1 completed successfully!")
        print(f"  Output: {cleaned_path}")
        print(f"  Trachea removed: Mask contains BRONCHI ONLY (from carina)")
        
    except Exception as e:
        print(f"\n❌ Error in Step 1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: COMPLETE BRONCHIAL TREE ANALYSIS WITH WEIBEL MODEL
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: BRONCHIAL TREE ANALYSIS WITH WEIBEL MODEL")
    print("="*80)
    print("This step will:")
    print("  1. Compute skeleton on bronchi-only mask")
    print("  2. Build topological graph")
    print("  3. Identify carina as root node")
    print("  4. Assign Weibel generation numbers (breadth-first from carina)")
    print("  5. Calculate branch diameters and lengths")
    print("  6. Analyze diameter tapering across generations")
    print("  7. Calculate PATH-BASED distances from carina")
    print("  8. Identify and classify bifurcations")
    print("  9. Generate comprehensive visualizations and reports")
    
    try:
        analyzer = AirwayGraphAnalyzer(cleaned_path)
        
        # Run complete analysis with Weibel generation analysis
        results = analyzer.run_full_analysis(
            output_dir=step2_dir,
            visualize=True,
            max_reconnect_distance_mm=15.0,
            min_voxels_for_reconnect=5,
            max_voxels_for_keep=100
        )
        
        print(f"\n✓ Step 2 completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in Step 2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print(" "*25 + "PIPELINE COMPLETED!")
    print("="*80)
    
    print("\n📁 Results saved in:")
    print(f"  {output_root}/")
    print(f"     ├── step1_preprocessing_with_trachea_removal/")
    print(f"     │   ├── cleaned_airway_mask_bronchi_only.nii.gz")
    print(f"     │   ├── preprocessing_report.txt")
    print(f"     │   ├── component_statistics.csv")
    print(f"     │   └── visualizations (PNG)")
    print(f"     │")
    print(f"     └── step2_bronchial_tree_analysis/")
    print(f"         ├── skeleton.nii.gz")
    print(f"         ├── branch_metrics_complete.csv        ⭐ MAIN OUTPUT")
    print(f"         ├── weibel_generation_analysis.csv     ⭐ WEIBEL ANALYSIS")
    print(f"         ├── weibel_tapering_ratios.csv         ⭐ TAPERING RATIOS")
    print(f"         ├── bifurcations.csv")
    print(f"         ├── component_statistics.csv")
    print(f"         ├── analysis_summary.txt")
    print(f"         └── visualizations (PNG)")
    
    print("\n📊 Key output files:")
    print(f"   • branch_metrics_complete.csv - Contains all bronchial branch measurements:")
    print(f"     - Diameters (mean, min, max, std)")
    print(f"     - Lengths (mm)")
    print(f"     - Volumes (mm³) and surface areas (mm²)")
    print(f"     - Weibel generation numbers")
    print(f"     - PATH distances from carina (proximal and distal)")
    print(f"   • weibel_generation_analysis.csv - Statistics by generation:")
    print(f"     - Mean/std diameter per generation")
    print(f"     - Mean/std length per generation")
    print(f"     - Number of branches per generation")
    print(f"     - Total volume and surface area per generation")
    print(f"   • weibel_tapering_ratios.csv - Diameter reduction ratios:")
    print(f"     - Ratio between consecutive generations")
    print(f"     - Comparison with Weibel's theoretical model (0.793)")
    
    if hasattr(analyzer, 'branch_metrics_df'):
        print(f"\n📈 Final Analysis Statistics (BRONCHI ONLY):")
        print(f"   • Total bronchial branches: {len(analyzer.branch_metrics_df)}")
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
    
    if hasattr(analyzer, 'weibel_analysis_df'):
        print(f"\n🔬 Weibel Generation Analysis:")
        print(f"   • Maximum generation detected: {int(analyzer.weibel_analysis_df['generation'].max())}")
        print(f"   • Number of generations: {len(analyzer.weibel_analysis_df)}")
        
        if hasattr(analyzer, 'tapering_ratios_df') and len(analyzer.tapering_ratios_df) > 0:
            mean_ratio = analyzer.tapering_ratios_df['diameter_ratio'].mean()
            weibel_theoretical = 2**(-1/3)
            difference = abs(mean_ratio - weibel_theoretical)
            
            print(f"   • Mean diameter tapering ratio: {mean_ratio:.3f}")
            print(f"   • Weibel's theoretical ratio: {weibel_theoretical:.3f}")
            print(f"   • Difference from theory: {difference:.3f}")
            
            if difference < 0.1:
                print(f"   ✓ Good agreement with Weibel's model!")
            else:
                print(f"   ⚠ Deviation from Weibel's model (possible anatomical variations)")
        
        # Show generation-by-generation breakdown
        print(f"\n   Generation breakdown:")
        for _, row in analyzer.weibel_analysis_df.iterrows():
            gen = int(row['generation'])
            n_branches = int(row['n_branches'])
            diameter = row['diameter_mean_mm']
            length = row['length_mean_mm']
            print(f"     Gen {gen}: {n_branches:3d} branches, "
                  f"Ø={diameter:.2f}mm, L={length:.2f}mm")
    
    # Calculate some derived metrics
    if hasattr(analyzer, 'branch_metrics_df') and hasattr(analyzer, 'carina_node'):
        print(f"\n🎯 Distance Analysis (PATH-BASED from carina):")
        if 'distance_from_carina_distal_mm' in analyzer.branch_metrics_df.columns:
            distances = analyzer.branch_metrics_df['distance_from_carina_distal_mm'].dropna()
            if len(distances) > 0:
                print(f"   • Mean distance to branch endpoints: {distances.mean():.2f} mm")
                print(f"   • Max distance (most distal branch): {distances.max():.2f} mm")
                print(f"   • Min distance (most proximal branch): {distances.min():.2f} mm")
    
    print("\n" + "="*80)
    print("Pipeline execution completed successfully! ✓")
    print("\n🔍 KEY IMPROVEMENTS IN THIS VERSION:")
    print("  1. ✓ Trachea automatically removed during preprocessing")
    print("  2. ✓ Component reconnection uses PATH-BASED distances (not Euclidean)")
    print("  3. ✓ Weibel generation analysis with tapering ratios")
    print("  4. ✓ All distances calculated along skeleton paths (physiologically accurate)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()