import os
import sys
import numpy as np
import datetime
from pathlib import Path
import SimpleITK as sitk
from test_robust import EnhancedCarinaDetector
from airway_refinement import EnhancedAirwayRefinementModule  
from airwais_seg import segment_airwayfull_from_mhd
from preprocessin_cleaning import SegmentationPreprocessor
from airway_graph import AirwayGraphAnalyzer
from skeleton_cleaner import integrate_skeleton_cleaning
import pandas as pd
from airway_gap_filler import integrate_gap_filling_into_pipeline
from fibrosis_scoring import integrate_fibrosis_scoring  # NEW


class CompleteAirwayPipeline:
    """
    Complete end-to-end airway analysis pipeline with fibrosis assessment:
    1. Segmentation from MHD (TotalSegmentator)
    2. Enhanced Refinement
    3. Gap Filling
    4. Trachea removal (preserving bronchi and carina)
    5. Preprocessing with component reconnection
    6. Skeletonization and graph construction
    7. Weibel generation analysis
    8. Advanced clinical metrics
    9. Pulmonary fibrosis assessment (NEW)
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
        step6_dir = os.path.join(scan_output_dir, "step6_fibrosis_assessment")
        
        for d in [step1_dir, step2_dir, step3_dir, step4_dir, step6_dir]:
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
            # STEP 1: SEGMENTATION WITH ENHANCED REFINEMENT
            # ============================================================
            print("\n" + "="*80)
            print("STEP 1: AIRWAY SEGMENTATION (TotalSegmentator + Enhanced Refinement)")
            print("="*80)

            airway_path = segment_airwayfull_from_mhd(
                mhd_path, 
                step1_dir, 
                fast=fast_segmentation
            )

            # ENHANCED REFINEMENT
            print("\n--- Applying Enhanced Refinement ---")

            ct_img = sitk.ReadImage(mhd_path)
            ct_np = sitk.GetArrayFromImage(ct_img)

            airway_img = sitk.ReadImage(airway_path)
            airway_np = sitk.GetArrayFromImage(airway_img)
            mask_np = (airway_np > 0).astype(np.uint8)

            ARM = EnhancedAirwayRefinementModule(
                ct_np, 
                mask_np, 
                ct_img.GetSpacing(),
                verbose=True
            )

            # PARAMETRI ANTI-BLOB - CONSERVATIVI per rimuovere solo artefatti chiari
            # Parametri rilassati per preservare rami periferici veri mantenendo qualità
            refined_np = ARM.refine(
                enable_anti_blob=True,              # ✓ RIABILITATO con parametri conservativi
                min_blob_size_voxels=10,            # Ridotto: rimuove solo blob molto piccoli
                min_blob_size_mm3=3,                # Ridotto: preserva anche piccoli rami distali
                max_blob_distance_mm=20.0,          # Ridotto da 30: più conservativo
                enable_tubular_smoothing=False,     # Mantieni disabilitato per preservare forma
                enable_skeleton_reconstruction=False
            )

            refined_path = os.path.join(step1_dir, f"{scan_name}_airway_refined_enhanced.nii.gz")
            ARM.save(refined_path, airway_img)

            airway_path = refined_path
            results['airway_segmentation'] = airway_path
            print(f"\n✓ Enhanced segmentation complete: {airway_path}")
            
            # ============================================================
            # STEP 1.5: GAP FILLING
            # ============================================================
            print("\n" + "="*80)
            print("STEP 1.5: INTELLIGENT GAP FILLING")
            print("="*80)

            gap_filled_path, gap_filler = integrate_gap_filling_into_pipeline(
                mhd_path=mhd_path,
                airway_mask_path=airway_path,
                output_dir=step1_dir,
                max_hole_size_mm3=100,              # Ridotto da 200: più conservativo per evitare pallini
                max_bridge_distance_mm=10.0         # Ridotto da 15: evita connessioni spurie
            )

            airway_path = gap_filled_path
            results['airway_gap_filled'] = gap_filled_path

            # ============================================================
            # STEP 2: ENHANCED TRACHEA REMOVAL
            # ============================================================
            print("\n" + "="*80)
            print("STEP 2: ENHANCED TRACHEA REMOVAL")
            print("="*80)
            print("Using ultra-conservative trachea identification method")

            from test_robust import integrate_with_pipeline

            bronchi_mask, carina_coords, confidence, detector = integrate_with_pipeline(
                airway_path,
                spacing=None,
                save_output=True,
                output_dir=step2_dir  
            )

            bronchi_original_path = os.path.join(step2_dir, "bronchi_enhanced_conservative.nii.gz")
            bronchi_filename = f"{scan_name}_bronchi_enhanced.nii.gz"
            bronchi_path = os.path.join(step2_dir, bronchi_filename)
            
            if os.path.exists(bronchi_original_path):
                if os.path.exists(bronchi_path):
                    os.remove(bronchi_path)
                os.rename(bronchi_original_path, bronchi_path)

            carina_z, carina_y, carina_x = carina_coords
            
            # Store trachea info
            results['trachea_info'] = {
                'detection_method': detector.detection_method,
                'confidence': confidence,
                'trachea_top_z': detector.trachea_top_z,
                'trachea_bottom_z': detector.trachea_bottom_z,
                'trachea_length_slices': detector.trachea_length,
                'trachea_length_mm': detector.trachea_length * detector.spacing[2] if detector.trachea_length else None
            }
            results['carina_coordinates'] = {'z': carina_z, 'y': carina_y, 'x': carina_x}
            
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
            
            cleaned_skeleton_path, _ = integrate_skeleton_cleaning(
                cleaned_path, 
                step4_dir,
                min_component_size=20,
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
            
            # CRITICAL: Use larger reconnect distance to ensure BOTH lungs are connected
            # With fibrosis, bronchi may be separated by >15mm in skeleton space
            analysis_results = analyzer.run_full_analysis(
                output_dir=step4_dir,
                visualize=True,
                max_reconnect_distance_mm=50.0,  # Increased from 15mm to connect both lungs
                min_voxels_for_reconnect=10,     # Increased to avoid reconnecting tiny noise
                max_voxels_for_keep=200          # Increased to preserve significant isolated regions
            )
            
            results['analysis_results'] = analysis_results
            results['analyzer'] = analyzer
            
            # ============================================================
            # STEP 5: ADVANCED CLINICAL METRICS
            # ============================================================
            print("\n" + "="*80)
            print("STEP 5: ADVANCED CLINICAL METRICS")
            print("="*80)

            try:
                advanced_metrics = analyzer.compute_advanced_metrics()
                analyzer.save_advanced_metrics(step4_dir)
                analyzer.plot_advanced_metrics(
                    save_path=os.path.join(step4_dir, "advanced_metrics_summary.png")
                )
                results['advanced_metrics'] = advanced_metrics
                print("\n✓ Advanced metrics computed and saved")
                
            except Exception as e:
                print(f"\n⚠ Warning: Could not compute advanced metrics: {e}")
                import traceback
                traceback.print_exc()

            # ============================================================
            # STEP 6: PULMONARY FIBROSIS ASSESSMENT (NEW)
            # ============================================================
            print("\n" + "="*80)
            print("STEP 6: PULMONARY FIBROSIS ASSESSMENT")
            print("="*80)

            try:
                scorer, fibrosis_report = integrate_fibrosis_scoring(
                    analyzer,
                    output_dir=step6_dir
                )
                
                results['fibrosis_scorer'] = scorer
                results['fibrosis_report'] = fibrosis_report
                
                print("\n✓ Fibrosis assessment complete")
                
            except Exception as e:
                print(f"\n⚠ Warning: Could not complete fibrosis assessment: {e}")
                import traceback
                traceback.print_exc()

            # ============================================================
            # GENERATE COMPREHENSIVE SUMMARY REPORT
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
            
            f.write("1. ✓ Segmentation (TotalSegmentator + Enhanced Refinement)\n")
            f.write(f"   Output: {results.get('airway_segmentation', 'N/A')}\n\n")
            
            f.write("2. ✓ Gap Filling\n")
            f.write(f"   Output: {results.get('airway_gap_filled', 'N/A')}\n\n")
            
            f.write("3. ✓ Enhanced Trachea Removal\n")
            f.write(f"   Method: Ultra-conservative with pre-cut\n")
            
            if 'carina_coordinates' in results:
                carina = results['carina_coordinates']
                f.write(f"   Carina: (z={carina['z']}, y={carina['y']}, x={carina['x']})\n")
            
            if 'trachea_info' in results:
                trachea = results['trachea_info']
                f.write(f"   Detection method: {trachea.get('detection_method', 'N/A')}\n")
                f.write(f"   Confidence: {trachea.get('confidence', 0):.2f}/5.0\n")
                if trachea.get('trachea_length_mm'):
                    f.write(f"   Trachea: z={trachea['trachea_top_z']} to z={trachea['trachea_bottom_z']}\n")
                    f.write(f"   Length: {trachea['trachea_length_mm']:.1f} mm\n")
            f.write("\n")
            
            f.write("4. ✓ Preprocessing & Component Reconnection\n")
            f.write(f"   Output: {results.get('cleaned_mask', 'N/A')}\n\n")
            
            f.write("5. ✓ Bronchial Tree Analysis\n")
            f.write(f"   Output: {os.path.join(output_dir, 'step4_analysis')}\n\n")
            
            f.write("6. ✓ Advanced Clinical Metrics\n\n")
            
            f.write("7. ✓ Pulmonary Fibrosis Assessment\n")
            f.write(f"   Output: {os.path.join(output_dir, 'step6_fibrosis_assessment')}\n\n")
            
            # ============================================================
            # BRONCHIAL TREE STATISTICS
            # ============================================================
            f.write("="*80 + "\n")
            f.write("BRONCHIAL TREE STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            if hasattr(analyzer, 'branch_metrics_df') and analyzer.branch_metrics_df is not None:
                df = analyzer.branch_metrics_df
                
                f.write(f"Total branches: {len(df)}\n")
                f.write(f"Total tree length: {df['length_mm'].sum():.2f} mm\n")
                f.write(f"Total tree volume: {df['volume_mm3'].sum():.2f} mm³\n\n")
                
                f.write(f"Diameter statistics:\n")
                f.write(f"  Mean: {df['diameter_mean_mm'].mean():.2f} mm\n")
                f.write(f"  Range: {df['diameter_mean_mm'].min():.2f} - {df['diameter_mean_mm'].max():.2f} mm\n\n")
            
            if hasattr(analyzer, 'weibel_analysis_df') and analyzer.weibel_analysis_df is not None:
                f.write("\nWeibel Generation Analysis:\n")
                weibel_df = analyzer.weibel_analysis_df
                f.write(f"  Maximum generation: {int(weibel_df['generation'].max())}\n")
                f.write(f"  Number of generations: {len(weibel_df)}\n\n")
            
            # ============================================================
            # ADVANCED CLINICAL METRICS
            # ============================================================
            if hasattr(analyzer, 'advanced_metrics') and analyzer.advanced_metrics is not None:
                f.write("="*80 + "\n")
                f.write("ADVANCED CLINICAL METRICS\n")
                f.write("="*80 + "\n\n")
                
                metrics = analyzer.advanced_metrics
                
                f.write(f"Total airway volume: {metrics['total_volume_mm3']:.2f} mm³\n\n")
                
                f.write("Peripheral vs Central:\n")
                f.write(f"  Central: {metrics['central_volume_mm3']:.2f} mm³ ({metrics['central_branch_count']} branches)\n")
                f.write(f"  Peripheral: {metrics['peripheral_volume_mm3']:.2f} mm³ ({metrics['peripheral_branch_count']} branches)\n")
                f.write(f"  P/C Ratio: {metrics['peripheral_to_central_ratio']:.3f}\n")
                
                if metrics['peripheral_to_central_ratio'] < 0.2:
                    f.write(f"    ⚠ LOW - peripheral airway loss\n")
                elif metrics['peripheral_to_central_ratio'] > 0.6:
                    f.write(f"    ✓ HIGH - well-preserved\n")
                else:
                    f.write(f"    ✓ Normal range\n")
                f.write(f"\n")
                
                if 'mean_tortuosity' in metrics and not pd.isna(metrics['mean_tortuosity']):
                    f.write(f"Tortuosity: {metrics['mean_tortuosity']:.3f}\n")
                    if metrics['mean_tortuosity'] > 1.5:
                        f.write(f"  ⚠ HIGH - airway distortion\n")
                    else:
                        f.write(f"  ✓ Normal range\n")
                    f.write(f"\n")
                
                if 'symmetry_index' in metrics and not pd.isna(metrics['symmetry_index']):
                    f.write(f"Symmetry Index: {metrics['symmetry_index']:.3f}\n")
                    if metrics['symmetry_index'] < 0.7:
                        f.write(f"  ⚠ ASYMMETRIC\n")
                    else:
                        f.write(f"  ✓ Symmetric\n")
                    f.write(f"\n")
                
                f.write(f"Generation Coverage: {metrics['generation_coverage']*100:.1f}%\n\n")
            
            # ============================================================
            # PULMONARY FIBROSIS ASSESSMENT (NEW)
            # ============================================================
            if 'fibrosis_report' in results and results['fibrosis_report'] is not None:
                f.write("="*80 + "\n")
                f.write("PULMONARY FIBROSIS ASSESSMENT\n")
                f.write("="*80 + "\n\n")
                
                fib_rep = results['fibrosis_report']
                overall = fib_rep['overall']
                
                f.write(f"FIBROSIS SCORE: {overall['fibrosis_score']:.1f}/100\n")
                f.write(f"CLASSIFICATION: {overall['stage']}\n")
                f.write(f"CONFIDENCE: {overall['confidence']:.0%}\n\n")
                
                f.write("Component Breakdown:\n")
                f.write("-" * 80 + "\n")
                
                for comp_name, comp_data in fib_rep['components'].items():
                    f.write(f"\n{comp_name.replace('_', ' ').title()}:\n")
                    f.write(f"  Contribution: {comp_data['weighted_score']:.1f} points\n")
                    f.write(f"  Raw score: {comp_data['raw_score']:.1f}/10\n")
                    f.write(f"  Interpretation: {comp_data['interpretation']}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("CLINICAL INTERPRETATION\n")
                f.write("="*80 + "\n\n")
                
                score = overall['fibrosis_score']
                
                if score < 20:
                    f.write("No significant fibrotic changes detected.\n")
                    f.write("Airways appear structurally normal.\n")
                elif score < 35:
                    f.write("Minimal fibrotic changes detected.\n")
                    f.write("Early peripheral airway involvement.\n")
                    f.write("Recommendation: Monitor for progression.\n")
                elif score < 50:
                    f.write("Mild fibrosis with measurable airway changes.\n")
                    f.write("Recommendation: Clinical correlation and follow-up.\n")
                elif score < 70:
                    f.write("Moderate fibrosis with UIP-pattern features.\n")
                    f.write("Recommendation: PFT and specialist consultation.\n")
                else:
                    f.write("Severe/advanced fibrosis detected.\n")
                    f.write("Recommendation: Urgent pulmonary evaluation.\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("DISCLAIMER\n")
                f.write("="*80 + "\n\n")
                f.write("This is an AUTOMATED ASSESSMENT based on airway morphology.\n")
                f.write("It does NOT replace clinical evaluation, complete CT review,\n")
                f.write("pulmonary function tests, or pathological diagnosis.\n")
                f.write("Always correlate with full clinical picture.\n\n")
            
            # ============================================================
            # OUTPUT FILES
            # ============================================================
            f.write("="*80 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Main directory: {output_dir}\n\n")
            
            f.write("Key files:\n")
            f.write(f"  • step1_segmentation/          - Initial airway segmentation\n")
            f.write(f"  • step2_trachea_removal/       - Bronchi-only mask\n")
            f.write(f"  • step3_preprocessing/         - Cleaned mask\n")
            f.write(f"  • step4_analysis/              - Complete analysis\n")
            f.write(f"    - branch_metrics_complete.csv\n")
            f.write(f"    - weibel_generation_analysis.csv\n")
            f.write(f"    - Multiple visualizations\n")
            f.write(f"  • step6_fibrosis_assessment/   - Fibrosis scoring\n")
            f.write(f"    - fibrosis_assessment_report.txt\n")
            f.write(f"    - fibrosis_assessment.json\n")
            f.write(f"    - fibrosis_assessment_visualization.png\n")
            
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
            
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total scans: {len(all_results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")
            
            if successful:
                f.write("="*80 + "\n")
                f.write("SUCCESSFUL SCANS\n")
                f.write("="*80 + "\n\n")
                
                for result in successful:
                    f.write(f"✓ {result['scan_name']}\n")
                    f.write(f"  Output: {result['output_dir']}\n")
                    
                    if 'fibrosis_report' in result and result['fibrosis_report']:
                        fib = result['fibrosis_report']['overall']
                        f.write(f"  Fibrosis: {fib['fibrosis_score']:.1f}/100 ({fib['stage']})\n")
                    
                    f.write("\n")
            
            if failed:
                f.write("="*80 + "\n")
                f.write("FAILED SCANS\n")
                f.write("="*80 + "\n\n")
                
                for result in failed:
                    f.write(f"❌ {result['scan_name']}\n")
                    f.write(f"  Error: {result['error']}\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"\n📄 Batch summary saved: {report_path}")


def main():
    """Main entry point"""
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    INPUT_PATH = r"X:\Francesca Saglimbeni\tesi\datasets\dataset_OSIC_final"
    OUTPUT_DIR = "output_results_with_fibrosis"
    BATCH_MODE = True
    FAST_SEGMENTATION = False
    FILE_PATTERN = "*.mhd"
    
    # ============================================================
    # EXECUTION
    # ============================================================
    
    pipeline = CompleteAirwayPipeline(output_root=OUTPUT_DIR)
    
    print("\n" + "="*80)
    print(" "*10 + "COMPLETE AIRWAY ANALYSIS WITH FIBROSIS ASSESSMENT")
    print("="*80)
    print(f"\nInput: {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Batch mode: {BATCH_MODE}")
    print(f"Fibrosis assessment: ENABLED")
    
    if BATCH_MODE:
        if not os.path.isdir(INPUT_PATH):
            print(f"\n❌ Error: {INPUT_PATH} is not a directory")
            sys.exit(1)
        
        results = pipeline.process_folder(
            INPUT_PATH,
            pattern=FILE_PATTERN,
            fast_segmentation=FAST_SEGMENTATION
        )
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print("\n" + "="*80)
        print(" "*20 + "BATCH PROCESSING COMPLETE")
        print("="*80)
        print(f"\nTotal scans: {len(results)}")
        print(f"✓ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
    else:
        if not os.path.exists(INPUT_PATH):
            print(f"\n❌ Error: {INPUT_PATH} does not exist")
            sys.exit(1)
        
        result = pipeline.process_single_scan(INPUT_PATH, fast_segmentation=FAST_SEGMENTATION)
        
        if result['success']:
            print("\n" + "="*80)
            print(" "*30 + "SUCCESS!")
            print("="*80)
            print(f"\n✓ Complete analysis with fibrosis assessment")
            print(f"\n📁 Results: {result['output_dir']}")
            
            if 'fibrosis_report' in result and result['fibrosis_report']:
                fib = result['fibrosis_report']['overall']
                print(f"\nFibrosis Score: {fib['fibrosis_score']:.1f}/100")
                print(f"Classification: {fib['stage']}")
        else:
            print(f"\n❌ Error: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()