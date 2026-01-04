import os
import datetime
import tempfile
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, binary_dilation
from skimage.morphology import skeletonize_3d

from totalsegmentator.python_api import totalsegmentator
from vessel_graph import ImprovedVesselAnalyzer


class UnifiedVesselPipeline:
    """
    Pipeline UNIFICATA per analisi vasi polmonari:
    - Segmentazione combinata vasi
    - Identificazione automatica trunks principali
    - Analisi morfometrica completa da ogni trunk
    - NO distinzione arterie/vene
    """

    def __init__(self, output_root="vessel_output"):
        self.output_root = os.path.abspath(output_root)
        os.makedirs(self.output_root, exist_ok=True)

    def _convert_mhd_to_nifti(self, mhd_path, output_dir):
        """Converte MHD → NIfTI"""
        image = sitk.ReadImage(mhd_path)
        print(f"\nMHD Image Info:")
        print(f"  Size: {image.GetSize()}")
        print(f"  Spacing: {image.GetSpacing()}")

        base_name = os.path.splitext(os.path.basename(mhd_path))[0]
        nifti_path = os.path.join(output_dir, f"{base_name}.nii.gz")

        os.makedirs(output_dir, exist_ok=True)
        sitk.WriteImage(image, nifti_path)
        print(f"Converted to {nifti_path}")

        return nifti_path

    def _light_cleaning(self, vessels_path, output_dir, min_size=50):
        """Cleaning LEGGERO: rimuove solo noise piccolo"""
        print("\n=== Light Cleaning (noise removal) ===")
        
        vessels_img = sitk.ReadImage(vessels_path)
        vessels_np = sitk.GetArrayFromImage(vessels_img)
        original_voxels = np.sum(vessels_np > 0)
        print(f"  Original voxels: {original_voxels:,}")

        binary_vessels = (vessels_np > 0).astype(np.uint8)

        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labeled, num_components = label(binary_vessels, structure=structure)

        if num_components == 0:
            print("  Warning: No components found")
            return vessels_path

        comp_sizes = np.array([np.sum(labeled == i) for i in range(1, num_components + 1)], dtype=int)
        
        cleaned = np.zeros_like(binary_vessels)
        kept_count = 0
        removed_count = 0
        
        for cid in range(1, num_components + 1):
            if comp_sizes[cid - 1] >= min_size:
                cleaned[labeled == cid] = 1
                kept_count += 1
            else:
                removed_count += 1

        cleaned_voxels = np.sum(cleaned > 0)
        removed_voxels = original_voxels - cleaned_voxels

        print(f"  Components: {num_components} total")
        print(f"    Kept: {kept_count} (>= {min_size} voxels)")
        print(f"    Removed as noise: {removed_count} (< {min_size} voxels)")
        print(f"  Voxels: {cleaned_voxels:,} ({removed_voxels:,} removed, {removed_voxels/max(1,original_voxels)*100:.1f}%)")

        if removed_count > 0:
            base_name = os.path.splitext(os.path.basename(vessels_path))[0]
            if base_name.endswith('.nii'):
                base_name = os.path.splitext(base_name)[0]
            
            cleaned_path = os.path.join(output_dir, f"{base_name}_cleaned.nii.gz")
            
            cleaned_img = sitk.GetImageFromArray(cleaned.astype(np.uint8))
            cleaned_img.CopyInformation(vessels_img)
            sitk.WriteImage(cleaned_img, cleaned_path)
            
            print(f"  ✓ Saved cleaned: {cleaned_path}")
            return cleaned_path
        else:
            print(f"  ✓ No noise detected, using original")
            return vessels_path

    def process_single_scan(self, mhd_path, scan_name=None,
                            fast_segmentation=False,
                            apply_light_cleaning=True,
                            num_initial_points=5):
        """
        Processa singola scansione con analisi UNIFICATA
        
        Args:
            num_initial_points: Numero di punti iniziali (trunks) da identificare automaticamente
        """
        
        if scan_name is None:
            scan_name = Path(mhd_path).stem

        print("\n" + "=" * 80)
        print(f" PROCESSING SCAN: {scan_name}")
        print("=" * 80)

        scan_output_dir = os.path.join(self.output_root, scan_name)
        os.makedirs(scan_output_dir, exist_ok=True)

        step1_dir = os.path.join(scan_output_dir, "step1_segmentation")
        step2_dir = os.path.join(scan_output_dir, "step2_unified_analysis")

        for d in [step1_dir, step2_dir]:
            os.makedirs(d, exist_ok=True)

        results = {
            "scan_name": scan_name,
            "input_path": mhd_path,
            "output_dir": scan_output_dir,
            "success": False,
            "error": None
        }

        try:
            # ============================================================
            # STEP 1: SEGMENTAZIONE UNIFICATA VASI
            # ============================================================
            print("\n" + "=" * 80)
            print("STEP 1: UNIFIED VESSEL SEGMENTATION")
            print("=" * 80)

            with tempfile.TemporaryDirectory(prefix="totalseg_") as temp_dir:

                nifti_path = self._convert_mhd_to_nifti(mhd_path, temp_dir)

                # Segmentazione vasi polmonari
                print("\nRunning TotalSegmentator (lung_vessels)...")
                totalsegmentator(nifti_path, temp_dir,
                                task="lung_vessels",
                                fast=fast_segmentation)

                lung_vessels_path = os.path.join(temp_dir, "lung_vessels.nii.gz")
                if not os.path.exists(lung_vessels_path):
                    raise RuntimeError("lung_vessels.nii.gz not found")

                vessels_img = sitk.ReadImage(lung_vessels_path)
                vessels_np = (sitk.GetArrayFromImage(vessels_img) > 0).astype(np.uint8)

                vessels_raw_path = os.path.join(step1_dir, f"{scan_name}_vessels_raw.nii.gz")
                out_img = sitk.GetImageFromArray(vessels_np)
                out_img.CopyInformation(vessels_img)
                sitk.WriteImage(out_img, vessels_raw_path)

                results["vessels_raw"] = vessels_raw_path
                print(f"✓ Raw vessels saved: {vessels_np.sum():,} voxels")

                # ============================================================
                # CLEANING (opzionale)
                # ============================================================
                if apply_light_cleaning:
                    vessels_to_use = self._light_cleaning(
                        vessels_path=vessels_raw_path,
                        output_dir=step1_dir,
                        min_size=50
                    )
                else:
                    print("\n=== Skipping cleaning, using raw vessels ===")
                    vessels_to_use = vessels_raw_path

                results["vessels_final"] = vessels_to_use

            # ============================================================
            # STEP 2: ANALISI MORFOMETRICA UNIFICATA
            # ============================================================
            print("\n" + "=" * 80)
            print("STEP 2: UNIFIED MORPHOMETRIC ANALYSIS")
            print("=" * 80)

            analyzer = ImprovedVesselAnalyzer(
                vessels_to_use,
                num_initial_points=num_initial_points
            )
            
            summary = analyzer.run_full_analysis(step2_dir)
            results["analysis"] = summary

            # ============================================================
            # REPORT FINALE
            # ============================================================
            self._generate_report(results, scan_output_dir, analyzer)

            results["success"] = True
            print("\n" + "=" * 80)
            print("✓ PROCESSING COMPLETED SUCCESSFULLY")
            print("=" * 80)

        except Exception as e:
            print(f"\n✗ Error processing {scan_name}: {e}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
            results["success"] = False

        return results

    def process_folder(self, input_folder, fast_segmentation=False, 
                      apply_light_cleaning=True, num_initial_points=5):
        """Processa cartella con multipli .mhd"""
        input_folder = os.path.abspath(input_folder)

        if not os.path.isdir(input_folder):
            raise ValueError(f"Input folder does not exist: {input_folder}")

        mhd_files = sorted([
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(".mhd")
        ])

        if len(mhd_files) == 0:
            raise RuntimeError(f"No .mhd files found in {input_folder}")

        print("\n" + "=" * 80)
        print(f"FOUND {len(mhd_files)} SCANS TO PROCESS")
        print("=" * 80)

        all_results = {}

        for idx, mhd_path in enumerate(mhd_files, start=1):
            patient_id = Path(mhd_path).stem

            print("\n" + "#" * 80)
            print(f"[{idx}/{len(mhd_files)}] PROCESSING PATIENT: {patient_id}")
            print("#" * 80)

            try:
                results = self.process_single_scan(
                    mhd_path=mhd_path,
                    scan_name=patient_id,
                    fast_segmentation=fast_segmentation,
                    apply_light_cleaning=apply_light_cleaning,
                    num_initial_points=num_initial_points
                )
                all_results[patient_id] = results

            except Exception as e:
                print(f"\n✗ FAILED processing {patient_id}: {e}")
                all_results[patient_id] = {
                    "scan_name": patient_id,
                    "input_path": mhd_path,
                    "success": False,
                    "error": str(e)
                }

        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETED")
        print("=" * 80)

        success = sum(1 for r in all_results.values() if r.get("success"))
        failed = len(all_results) - success
        print(f"✓ Success: {success}")
        print(f"✗ Failed: {failed}")

        return all_results

    def _generate_report(self, results, output_dir, analyzer):
        """Genera report completo analisi unificata"""
        report_path = os.path.join(output_dir, "VESSEL_ANALYSIS_REPORT.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(" " * 20 + "UNIFIED PULMONARY VESSEL ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Scan: {results['scan_name']}\n")
            f.write(f"Input: {results['input_path']}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("ANALYSIS APPROACH:\n")
            f.write("  - Unified vascular tree analysis (no artery/vein distinction)\n")
            f.write("  - Multiple initial points (main trunks) automatically identified\n")
            f.write("  - Distance-based metrics from each trunk\n")
            f.write("  - Precise diameter measurements along entire tree\n")
            f.write("  - Bifurcation and generation analysis\n\n")

            # SUMMARY
            if analyzer.branch_df is not None:
                df = analyzer.branch_df
                f.write("=" * 80 + "\n")
                f.write("MORPHOMETRIC SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Total branches: {len(df)}\n")
                f.write(f"Total length: {df['length_mm'].sum():.2f} mm\n")
                f.write(f"Total volume: {df['volume_mm3'].sum():.2f} mm³\n\n")
                
                f.write("Diameter Statistics (mm):\n")
                f.write(f"  Mean: {df['diameter_mm'].mean():.2f}\n")
                f.write(f"  Median: {df['diameter_mm'].median():.2f}\n")
                f.write(f"  Std Dev: {df['diameter_mm'].std():.2f}\n")
                f.write(f"  Range: {df['diameter_mm'].min():.2f} - {df['diameter_mm'].max():.2f}\n\n")

                if 'generation' in df.columns:
                    f.write(f"Max generation: {df['generation'].max()}\n")
                    f.write(f"Generation distribution:\n")
                    gen_dist = df['generation'].value_counts().sort_index()
                    for gen, count in gen_dist.items():
                        f.write(f"  Gen {gen}: {count} branches\n")
                    f.write("\n")

                if 'strahler_order' in df.columns:
                    f.write(f"Max Strahler order: {df['strahler_order'].max()}\n\n")

            # INITIAL POINTS
            if hasattr(analyzer, 'initial_points') and analyzer.initial_points:
                f.write("=" * 80 + "\n")
                f.write("IDENTIFIED INITIAL POINTS (TRUNKS)\n")
                f.write("=" * 80 + "\n\n")
                
                for i, pt in enumerate(analyzer.initial_points, 1):
                    f.write(f"Trunk {i}: {pt['coords']}\n")
                    f.write(f"  Degree: {pt['degree']}\n")
                    f.write(f"  Local diameter: {pt['diameter']:.2f} mm\n")
                    f.write(f"  Branches downstream: {pt.get('branch_count', 'N/A')}\n\n")

            # PATHOLOGY
            if analyzer.pathology_report:
                f.write("=" * 80 + "\n")
                f.write("PATHOLOGY FINDINGS\n")
                f.write("=" * 80 + "\n\n")
                for i, p in enumerate(analyzer.pathology_report, 1):
                    f.write(f"{i}. {p['type']} - {p['severity']}\n")
                    f.write(f"   {p['description']}\n")
                    if 'clinical_significance' in p:
                        f.write(f"   Clinical: {p['clinical_significance']}\n")
                    f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"\n✓ Report saved: {report_path}")


if __name__ == "__main__":

    # ==============================
    # CONFIGURAZIONE
    # ==============================

    INPUT_FOLDER = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\test_data"
    OUTPUT_FOLDER = "vessel_output"
    FAST_SEGMENTATION = False
    APPLY_LIGHT_CLEANING = True
    NUM_INITIAL_POINTS = 5  # Numero di trunks principali da identificare

    # ==============================
    # ESECUZIONE
    # ==============================

    pipeline = UnifiedVesselPipeline(output_root=OUTPUT_FOLDER)

    pipeline.process_folder(
        input_folder=INPUT_FOLDER,
        fast_segmentation=FAST_SEGMENTATION,
        apply_light_cleaning=APPLY_LIGHT_CLEANING,
        num_initial_points=NUM_INITIAL_POINTS
    )

    print("\n" + "=" * 80)
    print("TIPS:")
    print("  - NUM_INITIAL_POINTS: Number of main trunks to identify (default 5)")
    print("  - Analysis starts from each trunk and propagates through tree")
    print("  - Check 3D visualizations for diameter and distance maps")
    print("  - Pathology detection based on unified vascular criteria")
    print("=" * 80)