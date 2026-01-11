import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class LiteratureReference:
    mean: float
    std: float
    valid_range: Tuple[float, float]
    source: str
    notes: str = ""


# ============================================================
# LITERATURE DATABASE
# ============================================================

LITERATURE = {

    # --- SEGMENTATION ---
    "airway_volume_ml": LiteratureReference(
        mean=180, std=50, valid_range=(80, 350),
        source="Montaudon et al. 2007",
        notes="Total airway tree volume in healthy adults"
    ),

    "surface_area_cm2": LiteratureReference(
        mean=200, std=80, valid_range=(80, 400),
        source="CT morphometry studies",
        notes="Total airway surface area"
    ),

    # --- GRAPH ---
    "branch_count": LiteratureReference(
        mean=1500, std=500, valid_range=(500, 3000),
        source="Weibel 1963",
        notes="Number of airway segments visible in CT"
    ),

    "bifurcation_ratio": LiteratureReference(
        mean=0.15, std=0.05, valid_range=(0.05, 0.30),
        source="Horsfield & Cumming 1968",
        notes="Ratio of bifurcation nodes to total nodes"
    ),

    "avg_branch_length_mm": LiteratureReference(
        mean=12.0, std=5.0, valid_range=(5.0, 25.0),
        source="Weibel & Horsfield models",
        notes="Average length of airway segments"
    ),

    # --- WEIBEL ---
    "max_generation": LiteratureReference(
        mean=18, std=3, valid_range=(12, 23),
        source="Weibel 1963",
        notes="Maximum generation depth in airway tree"
    ),

    "tapering_ratio": LiteratureReference(
        mean=0.793, std=0.05, valid_range=(0.70, 0.88),
        source="Weibel (2^(-1/3))",
        notes="Diameter ratio parent/child (ideal ~0.793)"
    ),

    "diameter_trachea_mm": LiteratureReference(
        mean=18.0, std=2.0, valid_range=(14.0, 22.0),
        source="CT normative data",
        notes="Trachea diameter (generation 0)"
    ),

    "diameter_gen5_mm": LiteratureReference(
        mean=3.5, std=1.0, valid_range=(2.0, 6.0),
        source="Weibel tables",
        notes="Approximate diameter at generation 5"
    ),

    # --- FIBROSIS ---
    "pc_ratio_healthy": LiteratureReference(
        mean=0.45, std=0.15, valid_range=(0.25, 0.65),
        source="IPF quantitative CT studies",
        notes="Peripheral/Central ratio in healthy subjects"
    ),

    "pc_ratio_fibrotic": LiteratureReference(
        mean=0.20, std=0.10, valid_range=(0.05, 0.35),
        source="UIP / IPF CT studies",
        notes="PC ratio in fibrotic disease"
    ),

    "tortuosity": LiteratureReference(
        mean=1.25, std=0.15, valid_range=(1.0, 1.6),
        source="CT morphometry studies",
        notes="Path tortuosity (length/straight distance)"
    )
}


# ============================================================
# DATA LOADING UTILITIES
# ============================================================

def load_case_data(case_dir):
    """Carica i dati di un caso dalle cartelle della pipeline"""
    case_dir = Path(case_dir)
    data = {}
    
    # Load advanced metrics
    metrics_path = case_dir / "step4_analysis" / "advanced_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            data['advanced_metrics'] = json.load(f)
    else:
        raise FileNotFoundError(f"advanced_metrics.json not found in {case_dir}")
    
    # Load Weibel generation analysis
    weibel_path = case_dir / "step4_analysis" / "weibel_generation_analysis.csv"
    if weibel_path.exists():
        data['weibel_df'] = pd.read_csv(weibel_path)
    
    # Load tapering ratios
    taper_path = case_dir / "step4_analysis" / "weibel_tapering_ratios.csv"
    if taper_path.exists():
        data['tapering_df'] = pd.read_csv(taper_path)
    
    # Load branch metrics for graph analysis
    branches_path = case_dir / "step4_analysis" / "branch_metrics_complete.csv"
    if branches_path.exists():
        data['branches_df'] = pd.read_csv(branches_path)
    
    return data


# ============================================================
# CORE VALIDATION UTILITIES
# ============================================================

def check_value(name, value, ref: LiteratureReference):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "FAIL", "Value missing"

    if ref.valid_range[0] <= value <= ref.valid_range[1]:
        return "PASS", "Within literature range"

    if abs(value - ref.mean) <= 2 * ref.std:
        return "WARNING", "Outside range but within 2σ"

    return "FAIL", "Implausible value"


# ============================================================
# VALIDATION STEPS
# ============================================================

def validate_segmentation(advanced_metrics):
    """Valida la qualità della segmentazione delle vie aeree"""
    results = {}
    
    # Volume
    vol_ml = advanced_metrics.get("total_volume_mm3", 0) / 1000
    status, msg = check_value(
        "airway_volume_ml", vol_ml, LITERATURE["airway_volume_ml"]
    )
    results["airway_volume_ml"] = {
        "value": round(vol_ml, 2),
        "status": status,
        "message": msg,
        "reference": f"{LITERATURE['airway_volume_ml'].mean} ± {LITERATURE['airway_volume_ml'].std} ml"
    }

    # Surface area (se disponibile)
    surf_mm2 = advanced_metrics.get("total_surface_area_mm2", None)
    if surf_mm2 is not None:
        surf_cm2 = surf_mm2 / 100
        status, msg = check_value(
            "surface_area_cm2", surf_cm2, LITERATURE["surface_area_cm2"]
        )
        results["surface_area_cm2"] = {
            "value": round(surf_cm2, 2),
            "status": status,
            "message": msg,
            "reference": f"{LITERATURE['surface_area_cm2'].mean} ± {LITERATURE['surface_area_cm2'].std} cm²"
        }

    return results


def validate_graph(case_data):
    """Valida la struttura del grafo delle vie aeree"""
    results = {}
    
    branches_df = case_data.get('branches_df')
    if branches_df is None or len(branches_df) == 0:
        results["error"] = {"status": "FAIL", "message": "Branch data missing"}
        return results

    n_edges = len(branches_df)
    
    # Estimate nodes (branches + endpoints)
    n_nodes = n_edges + 1  # Approssimazione semplice

    # Branch count
    status, msg = check_value(
        "branch_count", n_edges, LITERATURE["branch_count"]
    )
    results["branch_count"] = {
        "value": n_edges,
        "status": status,
        "message": msg,
        "reference": f"{LITERATURE['branch_count'].mean} ± {LITERATURE['branch_count'].std}"
    }

    # Bifurcation ratio (stimato dai dati)
    # Se ci sono generazioni, stimiamo dalle transizioni
    if 'generation' in branches_df.columns:
        generations = branches_df['generation'].value_counts()
        # Bifurcation ratio approssimativo
        if len(generations) > 1:
            bif_ratio = 1.0 / len(generations)  # Stima semplificata
            status, msg = check_value(
                "bifurcation_ratio", bif_ratio, LITERATURE["bifurcation_ratio"]
            )
            results["bifurcation_ratio"] = {
                "value": round(bif_ratio, 3),
                "status": status,
                "message": msg,
                "reference": f"{LITERATURE['bifurcation_ratio'].mean} ± {LITERATURE['bifurcation_ratio'].std}"
            }

    # Average branch length
    if 'length' in branches_df.columns:
        lengths = branches_df['length'].dropna()
        if len(lengths) > 0:
            avg_length = lengths.mean()
            status, msg = check_value(
                "avg_branch_length_mm", avg_length, LITERATURE["avg_branch_length_mm"]
            )
            results["avg_branch_length_mm"] = {
                "value": round(avg_length, 2),
                "status": status,
                "message": msg,
                "reference": f"{LITERATURE['avg_branch_length_mm'].mean} ± {LITERATURE['avg_branch_length_mm'].std} mm"
            }

    return results


def validate_weibel(case_data):
    """Valida la generazione di Weibel e il tapering dei diametri"""
    results = {}

    weibel = case_data.get('weibel_df')
    
    if weibel is None or len(weibel) == 0:
        results["error"] = {"status": "FAIL", "message": "Weibel analysis missing"}
        return results

    # Max generation
    max_gen = int(weibel["generation"].max())
    status, msg = check_value(
        "max_generation", max_gen, LITERATURE["max_generation"]
    )
    results["max_generation"] = {
        "value": max_gen,
        "status": status,
        "message": msg,
        "reference": f"{LITERATURE['max_generation'].mean} ± {LITERATURE['max_generation'].std}"
    }

    # Tapering ratio
    tapering_df = case_data.get('tapering_df')
    if tapering_df is not None and len(tapering_df) > 0 and "diameter_ratio" in tapering_df.columns:
        taper = tapering_df["diameter_ratio"].mean()
        
        status, msg = check_value(
            "tapering_ratio", taper, LITERATURE["tapering_ratio"]
        )
        results["tapering_ratio"] = {
            "value": round(taper, 3),
            "status": status,
            "message": msg,
            "reference": f"{LITERATURE['tapering_ratio'].mean} (ideal 2^(-1/3))"
        }

    # Diameter at generation 0 (trachea)
    gen0 = weibel[weibel["generation"] == 0]
    if len(gen0) > 0 and "mean_diameter" in gen0.columns:
        d_trachea = gen0["mean_diameter"].mean()
        status, msg = check_value(
            "diameter_trachea_mm", d_trachea, LITERATURE["diameter_trachea_mm"]
        )
        results["diameter_trachea_mm"] = {
            "value": round(d_trachea, 2),
            "status": status,
            "message": msg,
            "reference": f"{LITERATURE['diameter_trachea_mm'].mean} ± {LITERATURE['diameter_trachea_mm'].std} mm"
        }

    # Diameter at generation 5
    gen5 = weibel[weibel["generation"] == 5]
    if len(gen5) > 0 and "mean_diameter" in gen5.columns:
        d_gen5 = gen5["mean_diameter"].mean()
        status, msg = check_value(
            "diameter_gen5_mm", d_gen5, LITERATURE["diameter_gen5_mm"]
        )
        results["diameter_gen5_mm"] = {
            "value": round(d_gen5, 2),
            "status": status,
            "message": msg,
            "reference": f"{LITERATURE['diameter_gen5_mm'].mean} ± {LITERATURE['diameter_gen5_mm'].std} mm"
        }

    return results


def validate_fibrosis(advanced_metrics):
    """Valida le metriche di fibrosi (PC ratio, tortuosity)"""
    results = {}

    # PC ratio
    pc = advanced_metrics.get("peripheral_to_central_ratio", np.nan)
    
    # Determina quale riferimento usare (healthy vs fibrotic)
    # Per ora usiamo healthy, ma puoi adattarlo
    ref = LITERATURE["pc_ratio_healthy"]
    
    status, msg = check_value("pc_ratio", pc, ref)
    results["pc_ratio"] = {
        "value": round(pc, 3) if not np.isnan(pc) else None,
        "status": status,
        "message": msg,
        "reference": f"Healthy: {LITERATURE['pc_ratio_healthy'].mean} ± {LITERATURE['pc_ratio_healthy'].std}, Fibrotic: {LITERATURE['pc_ratio_fibrotic'].mean} ± {LITERATURE['pc_ratio_fibrotic'].std}",
        "interpretation": "Lower PC ratio suggests central airway predominance or peripheral loss (possible fibrosis)"
    }

    # Tortuosity
    tort = advanced_metrics.get("mean_tortuosity", np.nan)
    status, msg = check_value("tortuosity", tort, LITERATURE["tortuosity"])
    results["tortuosity"] = {
        "value": round(tort, 3) if not np.isnan(tort) else None,
        "status": status,
        "message": msg,
        "reference": f"{LITERATURE['tortuosity'].mean} ± {LITERATURE['tortuosity'].std}",
        "interpretation": "Higher tortuosity may indicate airway distortion"
    }

    return results


# ============================================================
# SINGLE-CASE VALIDATION
# ============================================================

def validate_single_case(case_data, output_dir):
    """Valida un singolo caso e genera report dettagliato"""
    validation = {}
    
    print(f"\n{'='*60}")
    print(f"Validating case: {Path(output_dir).name}")
    print(f"{'='*60}")

    advanced_metrics = case_data.get('advanced_metrics', {})

    # Step 1: Segmentation
    print("\n[1/4] Validating segmentation...")
    try:
        validation["segmentation"] = validate_segmentation(advanced_metrics)
        print(f"  ✓ Segmentation validation completed")
    except Exception as e:
        validation["segmentation"] = {"error": str(e)}
        print(f"  ✗ Segmentation validation failed: {e}")

    # Step 2: Graph
    print("\n[2/4] Validating graph structure...")
    try:
        validation["graph"] = validate_graph(case_data)
        print(f"  ✓ Graph validation completed")
    except Exception as e:
        validation["graph"] = {"error": str(e)}
        print(f"  ✗ Graph validation failed: {e}")

    # Step 3: Weibel
    print("\n[3/4] Validating Weibel generations and diameters...")
    try:
        validation["weibel"] = validate_weibel(case_data)
        print(f"  ✓ Weibel validation completed")
    except Exception as e:
        validation["weibel"] = {"error": str(e)}
        print(f"  ✗ Weibel validation failed: {e}")

    # Step 4: Fibrosis
    print("\n[4/4] Validating fibrosis metrics...")
    try:
        validation["fibrosis"] = validate_fibrosis(advanced_metrics)
        print(f"  ✓ Fibrosis validation completed")
    except Exception as e:
        validation["fibrosis"] = {"error": str(e)}
        print(f"  ✗ Fibrosis validation failed: {e}")

    # Overall usability assessment
    flags = []
    warnings = 0
    fails = 0
    
    for block_name, block in validation.items():
        for metric_name, v in block.items():
            if isinstance(v, dict) and "status" in v:
                flags.append(v["status"])
                if v["status"] == "WARNING":
                    warnings += 1
                elif v["status"] == "FAIL":
                    fails += 1

    pipeline_usable = fails == 0
    
    validation["SUMMARY"] = {
        "PIPELINE_USABLE": pipeline_usable,
        "total_checks": len(flags),
        "passed": flags.count("PASS"),
        "warnings": warnings,
        "failed": fails
    }

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY:")
    print(f"  Total checks: {len(flags)}")
    print(f"  ✓ Passed: {flags.count('PASS')}")
    print(f"  ⚠ Warnings: {warnings}")
    print(f"  ✗ Failed: {fails}")
    print(f"  Pipeline usable: {'YES' if pipeline_usable else 'NO'}")
    print(f"{'='*60}\n")

    # Save validation report
    output_path = Path(output_dir) / "PIPELINE_VALIDATION.json"
    with open(output_path, "w") as f:
        json.dump(validation, f, indent=2)
    
    print(f"Validation report saved to: {output_path}")

    return validation


# ============================================================
# 🚀 BATCH VALIDATION (INTERA CARTELLA)
# ============================================================

def validate_pipeline_folder(output_root):
    """Valida tutti i casi in una cartella"""
    output_root = Path(output_root)
    
    if not output_root.exists():
        raise FileNotFoundError(f"Output folder not found: {output_root}")
    
    print(f"\n{'#'*80}")
    print(f"# BATCH VALIDATION - AIRWAY ANALYSIS PIPELINE")
    print(f"# Root folder: {output_root}")
    print(f"{'#'*80}\n")
    
    summary = []
    case_dirs = [d for d in sorted(output_root.iterdir()) if d.is_dir()]
    
    print(f"Found {len(case_dirs)} potential cases to validate\n")

    for i, case_dir in enumerate(case_dirs, 1):
        case_name = case_dir.name
        
        try:
            # Look for step4_analysis folder
            analysis_dir = case_dir / "step4_analysis"

            if not analysis_dir.exists():
                print(f"[{i}/{len(case_dirs)}] SKIP: {case_name} - No step4_analysis folder")
                summary.append({
                    "case": case_name,
                    "status": "SKIPPED",
                    "reason": "step4_analysis folder not found"
                })
                continue

            # Load case data from JSON/CSV files
            print(f"[{i}/{len(case_dirs)}] Processing: {case_name}")
            case_data = load_case_data(case_dir)

            # Validate
            validation = validate_single_case(case_data, case_dir)

            # Extract summary
            summary_data = {
                "case": case_name,
                "status": "USABLE" if validation["SUMMARY"]["PIPELINE_USABLE"] else "NOT_USABLE",
                "total_checks": validation["SUMMARY"]["total_checks"],
                "passed": validation["SUMMARY"]["passed"],
                "warnings": validation["SUMMARY"]["warnings"],
                "failed": validation["SUMMARY"]["failed"]
            }
            
            # Add key metrics
            if "segmentation" in validation and "airway_volume_ml" in validation["segmentation"]:
                summary_data["volume_ml"] = validation["segmentation"]["airway_volume_ml"].get("value")
                summary_data["volume_status"] = validation["segmentation"]["airway_volume_ml"].get("status")
            
            if "graph" in validation and "branch_count" in validation["graph"]:
                summary_data["branch_count"] = validation["graph"]["branch_count"].get("value")
                summary_data["branch_count_status"] = validation["graph"]["branch_count"].get("status")
            
            if "weibel" in validation and "max_generation" in validation["weibel"]:
                summary_data["max_generation"] = validation["weibel"]["max_generation"].get("value")
                summary_data["max_generation_status"] = validation["weibel"]["max_generation"].get("status")
            
            if "fibrosis" in validation and "pc_ratio" in validation["fibrosis"]:
                summary_data["pc_ratio"] = validation["fibrosis"]["pc_ratio"].get("value")
                summary_data["pc_ratio_status"] = validation["fibrosis"]["pc_ratio"].get("status")

            summary.append(summary_data)

        except Exception as e:
            print(f"[{i}/{len(case_dirs)}] ERROR: {case_name} - {str(e)}")
            summary.append({
                "case": case_name,
                "status": "ERROR",
                "error": str(e)
            })

    # Create summary dataframe
    df = pd.DataFrame(summary)
    
    # Save to output folder
    summary_csv = output_root / "PIPELINE_VALIDATION_SUMMARY.csv"
    df.to_csv(summary_csv, index=False)
    
    print(f"\n{'#'*80}")
    print(f"# BATCH VALIDATION COMPLETE")
    print(f"{'#'*80}")
    print(f"\nProcessed {len(case_dirs)} cases:")
    print(f"  ✓ Usable: {len(df[df['status'] == 'USABLE'])}")
    print(f"  ✗ Not usable: {len(df[df['status'] == 'NOT_USABLE'])}")
    print(f"  ⊗ Errors: {len(df[df['status'] == 'ERROR'])}")
    print(f"  - Skipped: {len(df[df['status'] == 'SKIPPED'])}")
    print(f"\nSummary saved to: {summary_csv}\n")

    return df

def main():
    """Entry point per la validazione batch della pipeline"""
    
    # Configurazione percorsi
    DATA_ROOT = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\output_results_with_fibrosis")
    OUTPUT_CSV = Path(r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\validation_pipeline\airway_pipeline_validation_summary.csv")
    
    print(f"\n{'='*80}")
    print(f"AIRWAY PIPELINE VALIDATION TOOL")
    print(f"{'='*80}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"{'='*80}\n")

    # Verifica che la cartella esista
    if not DATA_ROOT.exists():
        print(f"ERROR: Data root folder does not exist: {DATA_ROOT}")
        return

    # Esegui validazione batch
    try:
        df = validate_pipeline_folder(DATA_ROOT)
        
        # Salva anche nella cartella validation_pipeline
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\n✓ Validation summary also saved to: {OUTPUT_CSV}")
        
        # Statistiche finali
        print(f"\n{'='*80}")
        print(f"FINAL STATISTICS")
        print(f"{'='*80}")
        
        if 'status' in df.columns:
            print(f"\nCase status distribution:")
            print(df['status'].value_counts())
        
        if 'failed' in df.columns:
            usable_cases = df[df['status'] == 'USABLE']
            if len(usable_cases) > 0:
                print(f"\nQuality metrics for USABLE cases:")
                print(f"  Average failed checks: {usable_cases['failed'].mean():.2f}")
                print(f"  Average warnings: {usable_cases['warnings'].mean():.2f}")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"\nERROR during validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()