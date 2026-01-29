import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json


# ============================================================
# CONFIGURATION
# ============================================================

# Add subdirectories to path
WORKSPACE_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(WORKSPACE_ROOT / "airway_segmentation"))
sys.path.insert(0, str(WORKSPACE_ROOT / "validation_pipeline" / "air_val"))
sys.path.insert(0, str(WORKSPACE_ROOT / "validation_pipeline" / "OSIC_metrics_validation"))

# Import pipeline modules
from airway_segmentation.main_pipeline import CompleteAirwayPipeline, batch_process_scans
from validation_pipeline.air_val.air_val import validate_results, generate_validation_report
from validation_pipeline.OSIC_metrics_validation.analyze_osic_metrics import main as analyze_metrics


# Data paths (OSIC dataset only)
OSIC_DATA = WORKSPACE_ROOT / "datasets" / "OSIC_correct"
RESULTS_OSIC = WORKSPACE_ROOT.parent / "results" / "results_OSIC_newMetrcis"

# Validation paths
VALIDATION_OUTPUT = WORKSPACE_ROOT / "validation_pipeline" / "air_val"


# ============================================================
# STEP 1: AIRWAY SEGMENTATION & ANALYSIS PIPELINE
# ============================================================

def run_airway_pipeline(single_scan=None, fast_mode=False):
    """
    Run complete airway pipeline: segmentation → analysis → parenchymal → dual fibrosis scoring
    
    Args:
        dataset: "OSIC" or "CARVE14"
        single_scan: Path to single .mhd file (overrides dataset)
        fast_mode: Use fast TotalSegmentator mode
    
    Returns:
        Results directory path
    """
    print("\n" + "="*100)
    print("STEP 1: COMPLETE AIRWAY ANALYSIS PIPELINE")
    print("Segmentation → Preprocessing → Analysis → Parenchymal Metrics → Dual Fibrosis Scoring")
    print("="*100)
    
    # Determine input and output
    if single_scan:
        input_path = Path(single_scan)
        if not input_path.exists():
            raise FileNotFoundError(f"Scan not found: {input_path}")
        
        output_dir = WORKSPACE_ROOT.parent / "results" / "single_scan_results"
        scan_list = [input_path]
        
        print(f"\nProcessing single scan: {input_path.name}")
        print(f"Output directory: {output_dir}")
        
    else:
        if dataset == "OSIC":
            input_dir = OSIC_DATA
            output_dir = RESULTS_OSIC
        elif dataset == "CARVE14":
            input_dir = CARVE14_DATA
            output_dir = RESULTS_CARVE14
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {input_dir}")
        
        # Find all .mhd files
        scan_list = list(input_dir.glob("*.mhd"))
        
        print(f"\nProcessing {dataset} dataset")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(scan_list)} scans")
    
    if len(scan_list) == 0:
        print("⚠ No scans found!")
        return None
    
    # Create pipeline
    pipeline = CompleteAirwayPipeline(output_root=str(output_dir))
    
    # Process scans
    if single_scan:
        # Single scan
        print(f"\nProcessing: {scan_list[0].name}")
        result = pipeline.process_single_scan(
            str(scan_list[0]),
            scan_name=scan_list[0].stem,
            fast_segmentation=fast_mode
        )
        
        if result['success']:
            print(f"\n✓ Successfully processed: {result['scan_name']}")
        else:
            print(f"\n❌ Failed: {result['error']}")
    
    else:
        # Batch processing
        success_count, total_count = batch_process_scans(
            scan_list,
            output_root=str(output_dir),
            fast_mode=fast_mode
        )
        
        print(f"\n{'='*100}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*100}")
        print(f"Successfully processed: {success_count}/{total_count} scans")
        print(f"Results saved to: {output_dir}")
    
    return output_dir


# ============================================================
# STEP 2: TECHNICAL VALIDATION (RELIABLE/UNRELIABLE)
# ============================================================

def run_technical_validation(results_dir, dataset="OSIC"):
    """
    Validate results against literature values and classify as RELIABLE/UNRELIABLE
    
    Args:
        results_dir: Path to results directory
        dataset: "OSIC" or "CARVE14"
    
    Returns:
        Path to validation CSV
    """
    print("\n" + "="*100)
    print("STEP 2: TECHNICAL VALIDATION")
    print("Classifying results as RELIABLE/UNRELIABLE based on literature ranges")
    print("="*100)
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Run validation
    print(f"\nValidating results from: {results_dir}")
    
    validation_results = validate_results(
        results_dir=str(results_dir),
        output_csv_name=f"{dataset}_validation_newmetrics.csv",
        save_dir=str(VALIDATION_OUTPUT)
    )
    
    # Generate report
    output_csv = VALIDATION_OUTPUT / f"{dataset}_validation_newmetrics.csv"
    
    if output_csv.exists():
        report = generate_validation_report(str(output_csv))
        
        print(f"\n{'='*100}")
        print("VALIDATION SUMMARY")
        print(f"{'='*100}")
        print(f"Total cases: {report['total_cases']}")
        print(f"RELIABLE: {report['reliable_count']} ({report['reliable_percent']:.1f}%)")
        print(f"UNRELIABLE: {report['unreliable_count']} ({report['unreliable_percent']:.1f}%)")
        
        if report['unreliable_count'] > 0:
            print(f"\nMost common issues:")
            for issue, count in report['issue_counts'].items():
                print(f"  - {issue}: {count} cases")
        
        print(f"\nValidation results saved to: {output_csv}")
    
    return output_csv


# ============================================================
# STEP 3: FVC CORRELATION ANALYSIS & DUAL SCORING VALIDATION
# ============================================================

def run_fvc_correlation_analysis():
    """
    Analyze correlations between metrics and FVC%, including dual fibrosis score validation
    
    Compares:
    - AIRWAY_ONLY score (Opzione 1)
    - COMBINED score (Opzione 2 - RECOMMENDED)
    
    Returns:
        Path to analysis results directory
    """
    print("\n" + "="*100)
    print("STEP 3: FVC CORRELATION ANALYSIS & DUAL FIBROSIS SCORE VALIDATION")
    print("Analyzing metrics vs FVC% + comparing AIRWAY_ONLY vs COMBINED scoring")
    print("="*100)
    
    # Change to analysis directory (required for relative paths in script)
    original_dir = os.getcwd()
    analysis_dir = WORKSPACE_ROOT / "validation_pipeline" / "OSIC_metrics_validation"
    os.chdir(analysis_dir)
    
    try:
        # Run analysis
        print(f"\nRunning FVC correlation analysis...")
        print(f"Working directory: {analysis_dir}")
        
        analyze_metrics()
        
        results_dir = analysis_dir / "results_analysis"
        
        print(f"\n{'='*100}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*100}")
        print(f"Results saved to: {results_dir}")
        print(f"\nKey files generated:")
        print(f"  - integrated_dataset.csv: Complete data with all metrics")
        print(f"  - correlation_results.csv: All correlations with FVC%")
        print(f"  - fibrosis_score_comparison.png: AIRWAY_ONLY vs COMBINED comparison")
        print(f"  - correlation_summary.png: Visual summary of all correlations")
        
        return results_dir
    
    finally:
        # Return to original directory
        os.chdir(original_dir)


# ============================================================
# COMPLETE WORKFLOW
# ============================================================

def run_complete_workflow(fast_mode=False):
    """
    Run complete workflow for OSIC dataset:
    1. Airway pipeline (segmentation → analysis → parenchymal → dual scoring)
    2. Technical validation (RELIABLE/UNRELIABLE)
    3. FVC correlation analysis (including dual score comparison)
    
    Args:
        fast_mode: Use fast segmentation mode
    """
    print("\n" + "="*100)
    print("MASTER PIPELINE - COMPLETE AIRWAY ANALYSIS & VALIDATION")
    print("="*100)
    print(f"Dataset: OSIC")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workspace: {WORKSPACE_ROOT}")
    print("="*100)
    
    start_time = datetime.now()
    
    # STEP 1: Airway Pipeline
    results_dir = run_airway_pipeline(dataset="OSIC", fast_mode=fast_mode)
    if results_dir is None:
        print("\n❌ Pipeline failed - aborting")
        return
    
    # STEP 2: Technical Validation
    try:
        validation_csv = run_technical_validation(results_dir, dataset="OSIC")
    except Exception as e:
        print(f"\n⚠ Validation failed: {e}")
        validation_csv = None
    
    # STEP 3: FVC Correlation Analysis
    try:
        analysis_dir = run_fvc_correlation_analysis()
    except Exception as e:
        print(f"\n⚠ FVC analysis failed: {e}")
        analysis_dir = None
    
    # Final summary
    elapsed = datetime.now() - start_time
    
    print("\n" + "="*100)
    print("COMPLETE WORKFLOW FINISHED")
    print("="*100)
    print(f"Total time: {elapsed}")
    print(f"\nResults locations:")
    print(f"  Pipeline results: {results_dir}")
    if validation_csv:
        print(f"  Validation CSV: {validation_csv}")
    if analysis_dir:
        print(f"  FVC analysis: {analysis_dir}")
    print("="*100 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master Pipeline - Complete OSIC Airway Analysis & Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete workflow for OSIC dataset (pipeline + validation)
  python master_pipeline.py
  
  # Fast mode for testing
  python master_pipeline.py --fast
  
  # Process single scan (pipeline only, no validation)
  python master_pipeline.py --single path/to/scan.mhd
        """
    )
    
    # Options
    parser.add_argument(
        '--single',
        type=str,
        metavar='PATH',
        help='Process single .mhd scan (pipeline only, no validation)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use fast segmentation mode (for testing)'
    )
    
    args = parser.parse_args()
    
    # Run appropriate workflow
    try:
        if args.single:
            # Single scan mode (pipeline only, no validation)
            results_dir = run_airway_pipeline(single_scan=args.single, fast_mode=args.fast)
            print(f"\n✓ Single scan processing complete")
            print(f"Results: {results_dir}")
        
        else:
            # Complete OSIC workflow (always pipeline + validation)
            run_complete_workflow(fast_mode=args.fast)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
