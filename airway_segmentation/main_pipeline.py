import os
import sys
import numpy as np
import datetime
from pathlib import Path
import SimpleITK as sitk
from test_robust import EnhancedCarinaDetector  # Cambiato qui
from airway_refinement import AirwayRefinementModule  # ← aggiungi questo
from airwais_seg import segment_airwayfull_from_mhd
from preprocessin_cleaning import SegmentationPreprocessor
from airway_graph import AirwayGraphAnalyzer
from skeleton_cleaner import integrate_skeleton_cleaning


class CompleteAirwayPipeline:
    """
    Complete end-to-end airway analysis pipeline:
    1. Segmentation from MHD (TotalSegmentator)
    2. Trachea removal (preserving bronchi and carina) - ENHANCED METHOD
    3. Preprocessing with component reconnection
    4. Skeletonization and graph construction
    5. Weibel generation analysis
    6. Complete metrics and visualizations
    """
    
    def __init__(self, output_root="output"):
        self.output_root = os.path.abspath(output_root)
        os.makedirs(self.output_root, exist_ok=True)
        
    def process_single_scan(self, mhd_path, scan_name=None, 
                           fast_segmentation=False):
        """
        Process a single MHD scan through the complete pipeline
        
        Args:
            mhd_path: Path to .mhd file
            scan_name: Optional name for the scan (defaults to filename)
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
            
            from airway_refinement import AirwayRefinementModule

            ''' sitk_img = sitk.ReadImage(airway_path)
            img_np = sitk.GetArrayFromImage(sitk_img)
            mask_np = (img_np > 0).astype(np.uint8)  # TS airwayfull mask

            ARM = AirwayRefinementModule(img_np, mask_np, sitk_img.GetSpacing())
            refined_mask = ARM.refine()

            refined_path = os.path.join(step1_dir, f"{scan_name}_airway_refined.nii.gz")
            ARM.save(refined_path, sitk_img)

            airway_path = refined_path   # <-- sostituisce la segmentazione standard
            print("✓ Airway refinement complete")'''
            # 1) Carica CT originale
            ct_img = sitk.ReadImage(mhd_path)
            ct_np = sitk.GetArrayFromImage(ct_img)

            # 2) Carica maschera airwayfull
            airway_img = sitk.ReadImage(airway_path)
            airway_np = sitk.GetArrayFromImage(airway_img)
            mask_np = (airway_np > 0).astype(np.uint8)

            # 3) Refinement con HU e maschera TS
            ARM = AirwayRefinementModule(ct_np, mask_np, ct_img.GetSpacing())
            refined_np = ARM.refine()

            # 4) Salvataggio corretto
            # QUI: usa airway_img come ref_img, NON refined_np
            refined_path = os.path.join(step1_dir, f"{scan_name}_airway_refined.nii.gz")
            ARM.save(refined_path, airway_img)

            # aggiorna il path airway da passare ai passi successivi
            airway_path = refined_path

            print("✓ Airway refinement complete")
            results['airway_segmentation'] = airway_path
            print(f"\n✓ Segmentation complete: {airway_path}")

            
            # ============================================================
            # STEP 2: ENHANCED TRACHEA REMOVAL (NEW METHOD)
            # ============================================================
            print("\n" + "="*80)
            print("STEP 2: ENHANCED TRACHEA REMOVAL")
            print("="*80)
            print("Using ultra-conservative trachea identification method")
            print("Strategy: Pre-cut → Identify trachea → Remove upper 50% only")

            # IMPORTANTE: Importa la funzione che salva i grafici
            from test_robust import integrate_with_pipeline

            # Usa integrate_with_pipeline passando la directory di output corretta
            bronchi_mask, carina_coords, confidence, detector = integrate_with_pipeline(
                airway_path,
                spacing=None,
                save_output=True,
                output_dir=step2_dir  
            )

            # Il file è già salvato come bronchi_enhanced_conservative.nii.gz
            # Rinominalo in bronchi_enhanced.nii.gz per compatibilità
            bronchi_original_path = os.path.join(step2_dir, "bronchi_enhanced_conservative.nii.gz")
            bronchi_filename = f"{scan_name}_bronchi_enhanced.nii.gz"
            bronchi_path = os.path.join(step2_dir, bronchi_filename)
            
            if os.path.exists(bronchi_original_path):
                # Rimuovi il file di destinazione se esiste già
                if os.path.exists(bronchi_path):
                    os.remove(bronchi_path)
                    print(f"✓ Removed existing: {bronchi_path}")
                
                os.rename(bronchi_original_path, bronchi_path)
                print(f"✓ Renamed to: {bronchi_path}")

            # Aggiungi le coordinate della carina ai risultati
            carina_z, carina_y, carina_x = carina_coords
            # ============================================================
            # STEP 3: PREPROCESSING WITH COMPONENT RECONNECTION
            # ============================================================
            print("\n" + "="*80)
            print("STEP 3: PREPROCESSING & COMPONENT RECONNECTION")
            print("="*80)
            print(f"Loading segmentation from: {os.path.abspath(bronchi_path)}")
            
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
            # Dopo preprocessing, PRIMA dell'analisi
            cleaned_skeleton_path, _ = integrate_skeleton_cleaning(
                cleaned_path, 
                step4_dir,
                min_component_size=20,  # Parametro da calibrare!
                max_isolation_distance_mm=15.0,
                min_branch_length_mm=5.0
            )
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
                      fast_segmentation=False):
        """
        Process all MHD files in a folder
        
        Args:
            folder_path: Path to folder containing MHD files
            pattern: File pattern to match (default: "*.mhd")
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
                fast_segmentation=fast_segmentation,
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
            
            f.write("2. ✓ ENHANCED Trachea Removal (Ultra-conservative)\n")
            f.write(f"   Output: {results.get('bronchi_mask', 'N/A')}\n")
            f.write(f"   Method: EnhancedCarinaDetector with pre-cut\n")
            
            if 'carina_coordinates' in results:
                carina = results['carina_coordinates']
                f.write(f"   Carina: (z={carina['z']}, y={carina['y']}, x={carina['x']})\n")
            
            if 'trachea_info' in results:
                trachea = results['trachea_info']
                f.write(f"   Detection method: {trachea.get('detection_method', 'N/A')}\n")
                f.write(f"   Confidence: {trachea.get('confidence', 0):.2f}/5.0\n")
                if trachea.get('trachea_length_mm'):
                    f.write(f"   Trachea identified: z={trachea['trachea_top_z']} to z={trachea['trachea_bottom_z']}\n")
                    f.write(f"   Trachea length: {trachea['trachea_length_mm']:.1f} mm "
                           f"({trachea['trachea_length_slices']} slices)\n")
            f.write("\n")
            
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
            f.write("ENHANCED TRACHEA REMOVAL METHOD\n")
            f.write("="*80 + "\n\n")
            
            f.write("Key improvements:\n")
            f.write("  1. Pre-cut fisso a z=390 per rimuovere trachea cervicale\n")
            f.write("  2. Identificazione intelligente della trachea vera\n")
            f.write("  3. Distinzione trachea/rami bronchiali basata su posizione\n")
            f.write("  4. Rimozione solo della metà superiore della trachea\n")
            f.write("  5. Preservazione completa di carina e tutti i rami bronchiali\n")
            f.write("  6. Approccio ultra-conservativo per massima sicurezza\n\n")
            
            f.write("="*80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Main results directory: {output_dir}\n\n")
            
            f.write("Key files:\n")
            f.write(f"  • step1_segmentation/          - Initial airway segmentation\n")
            f.write(f"  • step2_trachea_removal/       - Bronchi-only mask (ENHANCED method)\n")
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
                f.write(f"  Output: {result['output_dir']}\n")
                if 'trachea_info' in result:
                    trachea = result['trachea_info']
                    f.write(f"  Carina confidence: {trachea.get('confidence', 0):.2f}/5.0\n")
                    f.write(f"  Method: {trachea.get('detection_method', 'N/A')}\n")
                f.write("\n")
            
            if failed:
                f.write("="*80 + "\n")
                f.write("FAILED SCANS\n")
                f.write("="*80 + "\n\n")
                
                for result in failed:
                    f.write(f"❌ {result['scan_name']}\n")
                    f.write(f"  Error: {result['error']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("ENHANCED METHOD INFORMATION\n")
            f.write("="*80 + "\n\n")
            f.write("Trachea removal: Enhanced ultra-conservative method\n")
            f.write("  - Pre-cut fisso a z=390 per trachea cervicale\n")
            f.write("  - Identificazione intelligente della trachea vera\n")
            f.write("  - Distinzione trachea/rami bronchiali\n")
            f.write("  - Rimozione solo metà superiore della trachea\n")
            f.write("  - Preservazione completa carina e rami bronchiali\n")
            f.write("  - Approccio ultra-conservativo per massima sicurezza\n\n")
        
        print(f"\n📄 Batch summary saved: {report_path}")


def main():
    """Main entry point with predefined parameters"""
    
    # ============================================================
    # CONFIGURATION PARAMETERS - MODIFY THESE AS NEEDED
    # ============================================================
    
    # Input path (file or folder)
    INPUT_PATH = "X:/Francesca Saglimbeni/tesi/vesselsegmentation/airway_segmentation/test_data"
    
    # Output directory
    OUTPUT_DIR = "output_results_enhanced"
    
    # Processing mode
    BATCH_MODE = True  # Set to True for folder processing, False for single file
    
    # Enhanced trachea removal parameters
    
    # Segmentation mode
    FAST_SEGMENTATION = False  # Use fast mode for TotalSegmentator
    
    # File pattern for batch mode
    FILE_PATTERN = "*.mhd"
    
    # ============================================================
    # PIPELINE EXECUTION
    # ============================================================
    
    # Create pipeline
    pipeline = CompleteAirwayPipeline(output_root=OUTPUT_DIR)
    
    print("\n" + "="*80)
    print(" "*15 + "ENHANCED AIRWAY ANALYSIS PIPELINE")
    print("="*80)
    print(f"\nInput: {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Trachea removal: Enhanced ultra-conservative method")
    print(f"Fast segmentation: {FAST_SEGMENTATION}")
    print(f"Batch mode: {BATCH_MODE}")
    
    # Process
    if BATCH_MODE:
        print(f"\nMode: BATCH PROCESSING")
        print(f"Pattern: {FILE_PATTERN}")
        
        if not os.path.isdir(INPUT_PATH):
            print(f"\n❌ Error: {INPUT_PATH} is not a directory")
            sys.exit(1)
        
        results = pipeline.process_folder(
            INPUT_PATH,
            pattern=FILE_PATTERN,
            fast_segmentation=FAST_SEGMENTATION,
        )
        
        # Print summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print("\n" + "="*80)
        print(" "*25 + "BATCH PROCESSING COMPLETE")
        print("="*80)
        print(f"\nTotal scans: {len(results)}")
        print(f"✓ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
    else:
        print(f"\nMode: SINGLE SCAN")
        
        if not os.path.exists(INPUT_PATH):
            print(f"\n❌ Error: {INPUT_PATH} does not exist")
            sys.exit(1)
        
        if os.path.isdir(INPUT_PATH):
            print(f"\n❌ Error: {INPUT_PATH} is a directory. Set BATCH_MODE=True for folder processing")
            sys.exit(1)
        
        result = pipeline.process_single_scan(
            INPUT_PATH,
            fast_segmentation=FAST_SEGMENTATION,
        )
        
        if result['success']:
            print("\n" + "="*80)
            print(" "*30 + "SUCCESS!")
            print("="*80)
            print(f"\n✓ Enhanced analysis complete for {result['scan_name']}")
            print(f"\n📁 Results saved in: {result['output_dir']}")
            
            if 'trachea_info' in result:
                trachea = result['trachea_info']
                print(f"\nEnhanced trachea removal details:")
                print(f"  Method: {trachea.get('detection_method', 'N/A')}")
                print(f"  Confidence: {trachea.get('confidence', 0):.2f}/5.0")
                if trachea.get('trachea_length_mm'):
                    print(f"  Trachea identified: z={trachea['trachea_top_z']} to z={trachea['trachea_bottom_z']}")
                    print(f"  Trachea length: {trachea['trachea_length_mm']:.1f} mm")
        else:
            print("\n" + "="*80)
            print(" "*30 + "FAILED!")
            print("="*80)
            print(f"\n❌ Error: {result['error']}")
            sys.exit(1)
    
    print("\n" + "="*80)
    print("Enhanced pipeline execution completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()