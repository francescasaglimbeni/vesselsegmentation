import os
import sys
import numpy as np
import datetime
from pathlib import Path
import SimpleITK as sitk

# Import from existing modules
from airwais_seg import segment_airwayfull_from_mhd
from trachea_auto import (
    load_airway_mask, compute_skeleton, build_graph, 
    identify_carina, find_exact_carina_position,
    refine_carina_with_region_growing, remove_trachea_precise,
    visualize_precise_removal, save_final_segmentation
)
from preprocessin_cleaning import SegmentationPreprocessor
from airway_graph import AirwayGraphAnalyzer


class CompleteAirwayPipeline:
    """
    Complete end-to-end airway analysis pipeline:
    1. Segmentation from MHD (TotalSegmentator)
    2. Trachea removal (preserving bronchi)
    3. Preprocessing with component reconnection
    4. Skeletonization and graph construction
    5. Weibel generation analysis
    6. Complete metrics and visualizations
    """
    
    def __init__(self, output_root="output"):
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)
        
    def process_single_scan(self, mhd_path, scan_name=None, 
                           trachea_removal_method='curved',
                           fast_segmentation=False):
        """
        Process a single MHD scan through the complete pipeline
        
        Args:
            mhd_path: Path to .mhd file
            scan_name: Optional name for the scan (defaults to filename)
            trachea_removal_method: Method for trachea removal 
                                   ('adaptive', 'curved', 'vertical', '3d_growing')
            fast_segmentation: Use fast mode for TotalSegmentator
        
        Returns:
            Dictionary with results and paths
        """
        
        # Extract scan name
        if scan_name is None:
            scan_name = Path(mhd_path).stem
        
        print("\n" + "="*80)
        print(f" PROCESSING SCAN: {scan_name}")
        print("="*80)
        
        # Create output directory for this scan
        scan_output_dir = os.path.join(self.output_root, scan_name)
        os.makedirs(scan_output_dir, exist_ok=True)
        
        # Create subdirectories
        step1_dir = os.path.join(scan_output_dir, "step1_segmentation")
        step2_dir = os.path.join(scan_output_dir, "step2_trachea_removal")
        step3_dir = os.path.join(scan_output_dir, "step3_preprocessing")
        step4_dir = os.path.join(scan_output_dir, "step4_analysis")
        
        for d in [step1_dir, step2_dir, step3_dir, step4_dir]:
            os.makedirs(d, exist_ok=True)
        
        results = {
            'scan_name': scan_name,
            'input_path': mhd_path,
            'output_dir': scan_output_dir,
            'success': False,
            'error': None
        }
        
        try:
            # ============================================================
            # STEP 1: SEGMENTATION
            # ============================================================
            print("\n" + "="*80)
            print("STEP 1: AIRWAY SEGMENTATION (TotalSegmentator)")
            print("="*80)
            
            airway_path = segment_airwayfull_from_mhd(
                mhd_path, 
                step1_dir, 
                fast=fast_segmentation
            )
            
            results['airway_segmentation'] = airway_path
            print(f"\n✓ Segmentation complete: {airway_path}")
            
            # ============================================================
            # STEP 2: TRACHEA REMOVAL
            # ============================================================
            print("\n" + "="*80)
            print("STEP 2: TRACHEA REMOVAL")
            print("="*80)
            print(f"Using method: {trachea_removal_method}")
            
            # Load mask
            mask, sitk_image, spacing = load_airway_mask(airway_path)
            
            # Compute preliminary skeleton for carina detection
            skeleton_temp, distance_transform, _ = compute_skeleton(mask, spacing)
            graph_temp, skeleton_obj, branch_data = build_graph(skeleton_temp, spacing)
            
            # Identify carina
            carina_node, carina_diameter, carina_position, carina_info = identify_carina(
                graph_temp, distance_transform
            )
            
            known_carina_coords = carina_info['coordinates_voxel']
            
            # Find exact carina position
            carina_z, carina_y, carina_x = find_exact_carina_position(
                mask, known_carina_coords
            )
            
            # Refine with region growing
            carina_z, carina_y, carina_x = refine_carina_with_region_growing(
                mask, carina_z, carina_y, carina_x, spacing
            )
            
            # Remove trachea
            bronchi_mask, cutoff_z = remove_trachea_precise(
                mask, carina_z, carina_y, carina_x, spacing, 
                method=trachea_removal_method
            )
            
            # Save result
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bronchi_filename = f"{scan_name}_bronchi_{trachea_removal_method}_{timestamp}.nii.gz"
            bronchi_path = os.path.join(step2_dir, bronchi_filename)
            
            bronchi_sitk = sitk.GetImageFromArray(bronchi_mask.astype(np.uint8))
            bronchi_sitk.CopyInformation(sitk_image)
            sitk.WriteImage(bronchi_sitk, bronchi_path)
            
            results['bronchi_mask'] = bronchi_path
            results['carina_coordinates'] = {
                'z': int(carina_z),
                'y': int(carina_y),
                'x': int(carina_x)
            }
            
            print(f"\n✓ Trachea removed: {bronchi_path}")
            print(f"  Carina at: (z={carina_z}, y={carina_y}, x={carina_x})")
            
            # ============================================================
            # STEP 3: PREPROCESSING WITH COMPONENT RECONNECTION
            # ============================================================
            print("\n" + "="*80)
            print("STEP 3: PREPROCESSING & COMPONENT RECONNECTION")
            print("="*80)
            
            preprocessor = SegmentationPreprocessor(bronchi_path)
            
            cleaned_mask, cleaned_path = preprocessor.run_full_preprocessing(
                output_dir=step3_dir,
                try_reconnection=True,
                max_reconnect_distance_mm=15.0,
                min_component_size=50,
                visualize=True
            )
            
            results['cleaned_mask'] = cleaned_path
            print(f"\n✓ Preprocessing complete: {cleaned_path}")
            
            # ============================================================
            # STEP 4: COMPLETE BRONCHIAL TREE ANALYSIS
            # ============================================================
            print("\n" + "="*80)
            print("STEP 4: BRONCHIAL TREE ANALYSIS WITH WEIBEL MODEL")
            print("="*80)
            
            analyzer = AirwayGraphAnalyzer(cleaned_path)
            
            analysis_results = analyzer.run_full_analysis(
                output_dir=step4_dir,
                visualize=True,
                max_reconnect_distance_mm=15.0,
                min_voxels_for_reconnect=5,
                max_voxels_for_keep=100
            )
            
            results['analysis_results'] = analysis_results
            results['analyzer'] = analyzer
            
            print(f"\n✓ Analysis complete!")
            
            # ============================================================
            # GENERATE SUMMARY REPORT
            # ============================================================
            self._generate_summary_report(results, scan_output_dir, analyzer)
            
            results['success'] = True
            
        except Exception as e:
            print(f"\n❌ Error processing {scan_name}: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def process_folder(self, folder_path, pattern="*.mhd", 
                      trachea_removal_method='curved',
                      fast_segmentation=False):
        """
        Process all MHD files in a folder
        
        Args:
            folder_path: Path to folder containing MHD files
            pattern: File pattern to match (default: "*.mhd")
            trachea_removal_method: Method for trachea removal
            fast_segmentation: Use fast mode for TotalSegmentator
        
        Returns:
            List of results dictionaries
        """
        
        print("\n" + "="*80)
        print(f" BATCH PROCESSING: {folder_path}")
        print("="*80)
        
        # Find all MHD files
        folder_path = Path(folder_path)
        mhd_files = list(folder_path.glob(pattern))
        
        if len(mhd_files) == 0:
            print(f"\n❌ No files matching '{pattern}' found in {folder_path}")
            return []
        
        print(f"\nFound {len(mhd_files)} MHD files to process")
        
        all_results = []
        
        for idx, mhd_file in enumerate(mhd_files, 1):
            print(f"\n{'='*80}")
            print(f" SCAN {idx}/{len(mhd_files)}: {mhd_file.name}")
            print(f"{'='*80}")
            
            scan_name = mhd_file.stem
            
            result = self.process_single_scan(
                str(mhd_file),
                scan_name=scan_name,
                trachea_removal_method=trachea_removal_method,
                fast_segmentation=fast_segmentation
            )
            
            all_results.append(result)
        
        # Generate batch summary
        self._generate_batch_summary(all_results, self.output_root)
        
        return all_results
    
    def _generate_summary_report(self, results, output_dir, analyzer):
        """Generate a comprehensive summary report for a single scan"""
        
        report_path = os.path.join(output_dir, "COMPLETE_ANALYSIS_REPORT.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" "*20 + "COMPLETE AIRWAY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Scan name: {results['scan_name']}\n")
            f.write(f"Input file: {results['input_path']}\n")
            f.write(f"Analysis date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("="*80 + "\n")
            f.write("PIPELINE STEPS\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. ✓ Segmentation (TotalSegmentator)\n")
            f.write(f"   Output: {results.get('airway_segmentation', 'N/A')}\n\n")
            
            f.write("2. ✓ Trachea Removal\n")
            f.write(f"   Output: {results.get('bronchi_mask', 'N/A')}\n")
            if 'carina_coordinates' in results:
                carina = results['carina_coordinates']
                f.write(f"   Carina: (z={carina['z']}, y={carina['y']}, x={carina['x']})\n\n")
            
            f.write("3. ✓ Preprocessing & Component Reconnection\n")
            f.write(f"   Output: {results.get('cleaned_mask', 'N/A')}\n\n")
            
            f.write("4. ✓ Bronchial Tree Analysis\n")
            f.write(f"   Output directory: {os.path.join(output_dir, 'step4_analysis')}\n\n")
            
            f.write("="*80 + "\n")
            f.write("BRONCHIAL TREE STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            if hasattr(analyzer, 'branch_metrics_df') and analyzer.branch_metrics_df is not None:
                df = analyzer.branch_metrics_df
                
                f.write(f"Total branches: {len(df)}\n")
                f.write(f"Total bronchial tree length: {df['length_mm'].sum():.2f} mm\n")
                f.write(f"Total bronchial tree volume: {df['volume_mm3'].sum():.2f} mm³\n\n")
                
                f.write(f"Diameter statistics:\n")
                f.write(f"  Mean: {df['diameter_mean_mm'].mean():.2f} mm\n")
                f.write(f"  Min:  {df['diameter_mean_mm'].min():.2f} mm\n")
                f.write(f"  Max:  {df['diameter_mean_mm'].max():.2f} mm\n")
                f.write(f"  Std:  {df['diameter_mean_mm'].std():.2f} mm\n\n")
                
                f.write(f"Length statistics:\n")
                f.write(f"  Mean: {df['length_mm'].mean():.2f} mm\n")
                f.write(f"  Min:  {df['length_mm'].min():.2f} mm\n")
                f.write(f"  Max:  {df['length_mm'].max():.2f} mm\n")
                f.write(f"  Std:  {df['length_mm'].std():.2f} mm\n\n")
            
            if hasattr(analyzer, 'bifurcations_df') and analyzer.bifurcations_df is not None:
                f.write(f"Bifurcations: {len(analyzer.bifurcations_df)}\n\n")
            
            if hasattr(analyzer, 'weibel_analysis_df') and analyzer.weibel_analysis_df is not None:
                f.write("="*80 + "\n")
                f.write("WEIBEL GENERATION ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                weibel_df = analyzer.weibel_analysis_df
                f.write(f"Maximum generation: {int(weibel_df['generation'].max())}\n")
                f.write(f"Number of generations: {len(weibel_df)}\n\n")
                
                f.write("Generation breakdown:\n")
                f.write(f"{'Gen':<5} {'Branches':<10} {'Mean Ø (mm)':<15} {'Mean L (mm)':<15}\n")
                f.write("-"*50 + "\n")
                
                for _, row in weibel_df.iterrows():
                    gen = int(row['generation'])
                    n_branches = int(row['n_branches'])
                    diameter = row['diameter_mean_mm']
                    length = row['length_mean_mm']
                    f.write(f"{gen:<5} {n_branches:<10} {diameter:<15.2f} {length:<15.2f}\n")
                
                f.write("\n")
                
                if hasattr(analyzer, 'tapering_ratios_df') and len(analyzer.tapering_ratios_df) > 0:
                    f.write("\nDiameter tapering ratios:\n")
                    mean_ratio = analyzer.tapering_ratios_df['diameter_ratio'].mean()
                    weibel_theoretical = 2**(-1/3)
                    
                    f.write(f"  Mean observed ratio: {mean_ratio:.3f}\n")
                    f.write(f"  Weibel theoretical:  {weibel_theoretical:.3f}\n")
                    f.write(f"  Difference:          {abs(mean_ratio - weibel_theoretical):.3f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Main results directory: {output_dir}\n\n")
            
            f.write("Key files:\n")
            f.write(f"  • step1_segmentation/          - Initial airway segmentation\n")
            f.write(f"  • step2_trachea_removal/       - Bronchi-only mask\n")
            f.write(f"  • step3_preprocessing/         - Cleaned and reconnected mask\n")
            f.write(f"  • step4_analysis/              - Complete analysis results\n")
            f.write(f"    - branch_metrics_complete.csv\n")
            f.write(f"    - weibel_generation_analysis.csv\n")
            f.write(f"    - weibel_tapering_ratios.csv\n")
            f.write(f"    - bifurcations.csv\n")
            f.write(f"    - skeleton.nii.gz\n")
            f.write(f"    - Multiple visualization PNG files\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\n📄 Summary report saved: {report_path}")
    
    def _generate_batch_summary(self, all_results, output_dir):
        """Generate a summary for batch processing"""
        
        report_path = os.path.join(output_dir, "BATCH_PROCESSING_SUMMARY.txt")
        
        successful = [r for r in all_results if r['success']]
        failed = [r for r in all_results if not r['success']]
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" "*25 + "BATCH PROCESSING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Processing date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total scans processed: {len(all_results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("SUCCESSFUL SCANS\n")
            f.write("="*80 + "\n\n")
            
            for result in successful:
                f.write(f"✓ {result['scan_name']}\n")
                f.write(f"  Output: {result['output_dir']}\n\n")
            
            if failed:
                f.write("="*80 + "\n")
                f.write("FAILED SCANS\n")
                f.write("="*80 + "\n\n")
                
                for result in failed:
                    f.write(f"❌ {result['scan_name']}\n")
                    f.write(f"  Error: {result['error']}\n\n")
        
        print(f"\n📄 Batch summary saved: {report_path}")


def main():
    """Main entry point with argument parsing"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete Airway Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single scan
  python main_pipeline.py --input scan.mhd --output results
  
  # Process folder
  python main_pipeline.py --input scans/ --output results --batch
  
  # Use fast segmentation
  python main_pipeline.py --input scan.mhd --fast
  
  # Change trachea removal method
  python main_pipeline.py --input scan.mhd --method adaptive
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input MHD file or folder containing MHD files'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output root directory (default: output)'
    )
    
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Batch mode: process all MHD files in input folder'
    )
    
    parser.add_argument(
        '--method', '-m',
        default='curved',
        choices=['adaptive', 'curved', 'vertical', '3d_growing'],
        help='Trachea removal method (default: curved)'
    )
    
    parser.add_argument(
        '--fast', '-f',
        action='store_true',
        help='Use fast mode for TotalSegmentator'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        default='*.mhd',
        help='File pattern for batch mode (default: *.mhd)'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = CompleteAirwayPipeline(output_root=args.output)
    
    print("\n" + "="*80)
    print(" "*15 + "COMPLETE AIRWAY ANALYSIS PIPELINE")
    print("="*80)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    print(f"Trachea removal method: {args.method}")
    print(f"Fast segmentation: {args.fast}")
    
    # Process
    if args.batch:
        print(f"\nMode: BATCH PROCESSING")
        print(f"Pattern: {args.pattern}")
        
        if not os.path.isdir(args.input):
            print(f"\n❌ Error: {args.input} is not a directory")
            sys.exit(1)
        
        results = pipeline.process_folder(
            args.input,
            pattern=args.pattern,
            trachea_removal_method=args.method,
            fast_segmentation=args.fast
        )
        
    else:
        print(f"\nMode: SINGLE SCAN")
        
        if not os.path.exists(args.input):
            print(f"\n❌ Error: {args.input} does not exist")
            sys.exit(1)
        
        if os.path.isdir(args.input):
            print(f"\n❌ Error: {args.input} is a directory. Use --batch for folder processing")
            sys.exit(1)
        
        result = pipeline.process_single_scan(
            args.input,
            trachea_removal_method=args.method,
            fast_segmentation=args.fast
        )
        
        if result['success']:
            print("\n" + "="*80)
            print(" "*30 + "SUCCESS!")
            print("="*80)
            print(f"\n✓ Analysis complete for {result['scan_name']}")
            print(f"\n📁 Results saved in: {result['output_dir']}")
        else:
            print("\n" + "="*80)
            print(" "*30 + "FAILED!")
            print("="*80)
            print(f"\n❌ Error: {result['error']}")
            sys.exit(1)
    
    print("\n" + "="*80)
    print("Pipeline execution completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()