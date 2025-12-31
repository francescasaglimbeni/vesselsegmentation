import os
import datetime
import tempfile
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label, binary_dilation

from totalsegmentator.python_api import totalsegmentator

from vessel_graph import VesselGraphAnalyzer
from classify import classify_vessels_pathbased


class ImprovedVesselPipeline:
    """
    Pipeline completa:
    1) Segmentazione vasi (TotalSegmentator) + seed (atrio sinistro / arteria polmonare)
    2) Pulizia CC con 26-connectivity (FIX BUG)
    3) Classificazione path-based arterie/vene seguendo skeleton
    4) Analisi morfometrica combinata + arterie + vene
    """

    def __init__(self, output_root="vessel_output"):
        self.output_root = os.path.abspath(output_root)
        os.makedirs(self.output_root, exist_ok=True)

    def _convert_mhd_to_nifti(self, mhd_path, output_dir):
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

    def _generate_vessel_seeds(self, nifti_dir, output_dir):
        """
        Seed vene: dilatazione atrio sinistro
        Seed arterie: dilatazione arteria polmonare
        """
        print("\n=== Generating Artery/Vein Seeds ===")

        left_atrium_path = os.path.join(nifti_dir, "heart_atrium_left.nii.gz")
        pulmonary_artery_path = os.path.join(nifti_dir, "pulmonary_artery.nii.gz")

        if not all([os.path.exists(p) for p in [left_atrium_path, pulmonary_artery_path]]):
            raise RuntimeError("Missing heart segmentations for seeds (heart_atrium_left / pulmonary_artery).")

        left_atrium = sitk.GetArrayFromImage(sitk.ReadImage(left_atrium_path))
        pulmonary_artery = sitk.GetArrayFromImage(sitk.ReadImage(pulmonary_artery_path))
        ref_img = sitk.ReadImage(left_atrium_path)

        print("  Creating vein seeds from left atrium...")
        vein_seed = binary_dilation(left_atrium > 0, iterations=5)

        print("  Creating artery seeds from pulmonary artery...")
        artery_seed = binary_dilation(pulmonary_artery > 0, iterations=5)

        overlap = vein_seed & artery_seed
        if np.any(overlap):
            print(f"  Resolving {np.sum(overlap):,} overlapping voxels (artery priority)...")
            vein_seed = vein_seed & ~artery_seed

        artery_seed_path = os.path.join(output_dir, "seed_artery.nii.gz")
        vein_seed_path = os.path.join(output_dir, "seed_vein.nii.gz")

        artery_seed_img = sitk.GetImageFromArray(artery_seed.astype(np.uint8))
        artery_seed_img.CopyInformation(ref_img)
        sitk.WriteImage(artery_seed_img, artery_seed_path)

        vein_seed_img = sitk.GetImageFromArray(vein_seed.astype(np.uint8))
        vein_seed_img.CopyInformation(ref_img)
        sitk.WriteImage(vein_seed_img, vein_seed_path)

        print(f"  ✓ Artery seed: {np.sum(artery_seed):,} voxels")
        print(f"  ✓ Vein seed: {np.sum(vein_seed):,} voxels")

        return artery_seed_path, vein_seed_path

    def _clean_vessel_segmentation(
        self,
        vessels_path,
        output_dir,
        min_component_size=100,
        keep_n_largest=None,
        connectivity_26=True
    ):
        """
        FIX CRITICO:
        - label() in 3D senza structure usa 6-connectivity -> spezza i vasi sottili.
        - qui usiamo 26-connectivity per mantenere l'albero connesso.
        """
        print("\n=== Cleaning Vessel Segmentation (3D connectivity fix) ===")

        vessels_img = sitk.ReadImage(vessels_path)
        vessels_np = sitk.GetArrayFromImage(vessels_img)
        print(f"  Original voxels: {np.sum(vessels_np > 0):,}")

        binary_vessels = (vessels_np > 0).astype(np.uint8)

        structure = np.ones((3, 3, 3), dtype=np.uint8) if connectivity_26 else None
        labeled, num_components = label(binary_vessels, structure=structure)

        if num_components == 0:
            raise RuntimeError("No connected components in vessel mask after binarization.")

        comp_sizes = np.array([np.sum(labeled == i) for i in range(1, num_components + 1)], dtype=int)

        cleaned = np.zeros_like(binary_vessels)

        if keep_n_largest is not None:
            keep_ids = np.argsort(comp_sizes)[::-1][:keep_n_largest] + 1
            for cid in keep_ids:
                cleaned[labeled == cid] = 1
            print(f"  Keeping {len(keep_ids)} largest components: sizes={comp_sizes[keep_ids-1].tolist()}")
        else:
            kept = 0
            for cid in range(1, num_components + 1):
                if comp_sizes[cid - 1] >= min_component_size:
                    cleaned[labeled == cid] = 1
                    kept += 1
            print(f"  Components kept: {kept}/{num_components}")

        base_name = os.path.splitext(os.path.basename(vessels_path))[0]
        cleaned_path = os.path.join(output_dir, f"{base_name}_cleaned.nii.gz")

        cleaned_img = sitk.GetImageFromArray(cleaned.astype(np.uint8))
        cleaned_img.CopyInformation(vessels_img)
        sitk.WriteImage(cleaned_img, cleaned_path)

        print(f"  ✓ Cleaned voxels: {np.sum(cleaned > 0):,} → {cleaned_path}")
        return cleaned_path

    def process_single_scan(self, mhd_path, scan_name=None,
                            fast_segmentation=False,
                            max_path_length_mm=400.0):

        if scan_name is None:
            scan_name = Path(mhd_path).stem

        print("\n" + "=" * 80)
        print(f" PROCESSING SCAN: {scan_name}")
        print("=" * 80)

        scan_output_dir = os.path.join(self.output_root, scan_name)
        os.makedirs(scan_output_dir, exist_ok=True)

        step1_dir = os.path.join(scan_output_dir, "step1_segmentation")
        step2_dir = os.path.join(scan_output_dir, "step2_classification")
        step3_dir = os.path.join(scan_output_dir, "step3_analysis_combined")
        step4_dir = os.path.join(scan_output_dir, "step4_analysis_arteries")
        step5_dir = os.path.join(scan_output_dir, "step5_analysis_veins")

        for d in [step1_dir, step2_dir, step3_dir, step4_dir, step5_dir]:
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
            # STEP 1: SEGMENTAZIONE + SEED
            # ============================================================
            print("\n" + "=" * 80)
            print("STEP 1: VESSEL SEGMENTATION + SEED GENERATION")
            print("=" * 80)

            with tempfile.TemporaryDirectory(prefix="totalseg_") as temp_dir:

                nifti_path = self._convert_mhd_to_nifti(mhd_path, temp_dir)

                # --- lung vessels ---
                totalsegmentator(nifti_path, temp_dir,
                                task="lung_vessels",
                                fast=fast_segmentation)

                lung_vessels_path = os.path.join(temp_dir, "lung_vessels.nii.gz")
                if not os.path.exists(lung_vessels_path):
                    raise RuntimeError("lung_vessels.nii.gz not found")

                vessels_img = sitk.ReadImage(lung_vessels_path)
                vessels_np = (sitk.GetArrayFromImage(vessels_img) > 0).astype(np.uint8)

                vessels_raw_path = os.path.join(
                    step1_dir, f"{scan_name}_vessels_raw.nii.gz"
                )
                out_img = sitk.GetImageFromArray(vessels_np)
                out_img.CopyInformation(vessels_img)
                sitk.WriteImage(out_img, vessels_raw_path)

                results["vessels_raw"] = vessels_raw_path
                print(f"✓ Raw vessels: {vessels_np.sum():,} voxels")

                # --- heart for seeds ---
                totalsegmentator(nifti_path, temp_dir,
                                task="heartchambers_highres",
                                fast=fast_segmentation)

                pulmonary_artery_path = os.path.join(temp_dir, "pulmonary_artery.nii.gz")
                left_atrium_path = os.path.join(temp_dir, "heart_atrium_left.nii.gz")

                if not os.path.exists(pulmonary_artery_path):
                    raise RuntimeError("pulmonary_artery.nii.gz not found")
                if not os.path.exists(left_atrium_path):
                    raise RuntimeError("heart_atrium_left.nii.gz not found")

                artery_np = sitk.GetArrayFromImage(
                    sitk.ReadImage(pulmonary_artery_path)) > 0
                vein_np = sitk.GetArrayFromImage(
                    sitk.ReadImage(left_atrium_path)) > 0

                artery_seed = binary_dilation(artery_np, iterations=5)
                vein_seed = binary_dilation(vein_np, iterations=5)

                overlap = artery_seed & vein_seed
                vein_seed[overlap] = 0

                artery_seed_path = os.path.join(step1_dir, "seed_artery.nii.gz")
                vein_seed_path = os.path.join(step1_dir, "seed_vein.nii.gz")

                sitk.WriteImage(
                    sitk.GetImageFromArray(artery_seed.astype(np.uint8)),
                    artery_seed_path
                )
                sitk.WriteImage(
                    sitk.GetImageFromArray(vein_seed.astype(np.uint8)),
                    vein_seed_path
                )

                results["artery_seed"] = artery_seed_path
                results["vein_seed"] = vein_seed_path

            # ============================================================
            # STEP 1.5: CLEANING (QUI NASCE vessels_cleaned_path)
            # ============================================================
            print("\n=== CLEANING VESSEL SEGMENTATION ===")

            vessels_cleaned_path = self._clean_vessel_segmentation(
                vessels_path=results["vessels_raw"],
                output_dir=step1_dir,
                keep_n_largest=1,
                connectivity_26=True
            )

            results["vessels_cleaned"] = vessels_cleaned_path

            # ============================================================
            # STEP 2: CLASSIFICAZIONE PATH-BASED
            # ============================================================
            print("\n" + "=" * 80)
            print("STEP 2: PATH-BASED CLASSIFICATION")
            print("=" * 80)

            skeleton_path = os.path.join(step2_dir, "skeleton.nii.gz")

            artery_path, vein_path, unclass_path = classify_vessels_pathbased(
                vessel_path=vessels_cleaned_path,   # ✅ ORA ESISTE
                seed_artery_path=results["artery_seed"],
                seed_vein_path=results["vein_seed"],
                output_dir=step2_dir,
                max_path_length_mm=max_path_length_mm,
                tie_break="vein",
                skeleton_path=skeleton_path
            )

            # ============================================================
            # STEP 3–5: ANALISI
            # ============================================================
            analyzer_combined = VesselGraphAnalyzer(
                vessels_cleaned_path, vessel_type="combined"
            )
            analyzer_combined.run_full_analysis(step3_dir)

            analyzer_arteries = VesselGraphAnalyzer(
                artery_path, vessel_type="arteries"
            )
            analyzer_arteries.run_full_analysis(step4_dir)

            analyzer_veins = VesselGraphAnalyzer(
                vein_path, vessel_type="veins"
            )
            analyzer_veins.run_full_analysis(step5_dir)

            self._generate_summary_report(results, scan_output_dir)
            results["success"] = True

        except Exception as e:
            print(f"\n❌ Error processing {scan_name}: {e}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
            results["success"] = False

        return results

    def process_folder(self, input_folder, fast_segmentation=False, max_path_length_mm=400.0):
        """
        Itera su una cartella contenente più file .mhd (+ .zraw associati)
        e processa ogni scansione come paziente separato.

        Struttura output:
        output_root/
            └── patient_id/
                ├── step1_segmentation/
                ├── step2_classification/
                ├── step3_analysis_combined/
                ├── step4_analysis_arteries/
                └── step5_analysis_veins/
        """
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
                    max_path_length_mm=max_path_length_mm
                )
                all_results[patient_id] = results

            except Exception as e:
                print(f"\n❌ FAILED processing {patient_id}: {e}")
                all_results[patient_id] = {
                    "scan_name": patient_id,
                    "input_path": mhd_path,
                    "success": False,
                    "error": str(e)
                }

        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETED")
        print("=" * 80)

        # Riassunto finale
        success = sum(1 for r in all_results.values() if r.get("success"))
        failed = len(all_results) - success
        print(f"✓ Success: {success}")
        print(f"✗ Failed: {failed}")

        return all_results

    def _generate_summary_report(self, results, output_dir):
        report_path = os.path.join(output_dir, "COMPLETE_ANALYSIS_REPORT.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(" " * 20 + "COMPLETE VESSEL ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Scan: {results['scan_name']}\n")
            f.write(f"Input: {results['input_path']}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # COMBINED
            if "analyzer_combined" in results:
                analyzer = results["analyzer_combined"]
                f.write("=" * 80 + "\n")
                f.write("COMBINED VESSEL ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                if hasattr(analyzer, "branch_metrics_df") and analyzer.branch_metrics_df is not None:
                    df = analyzer.branch_metrics_df
                    f.write(f"Total branches: {len(df)}\n")
                    f.write(f"Total length: {df['length_mm'].sum():.2f} mm\n")
                    f.write(f"Total volume: {df['volume_mm3'].sum():.2f} mm³\n\n")
                    f.write("Diameter (mm):\n")
                    f.write(f"  Mean: {df['diameter_mean_mm'].mean():.2f}\n")
                    f.write(f"  Median: {df['diameter_mean_mm'].median():.2f}\n")
                    f.write(f"  Range: {df['diameter_mean_mm'].min():.2f} - {df['diameter_mean_mm'].max():.2f}\n\n")

            # ARTERIES
            if "analyzer_arteries" in results:
                analyzer = results["analyzer_arteries"]
                f.write("=" * 80 + "\n")
                f.write("ARTERY ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                if hasattr(analyzer, "branch_metrics_df") and analyzer.branch_metrics_df is not None:
                    df = analyzer.branch_metrics_df
                    f.write(f"Branches: {len(df)}\n")
                    f.write(f"Total length: {df['length_mm'].sum():.2f} mm\n")
                    f.write(f"Total volume: {df['volume_mm3'].sum():.2f} mm³\n")
                    f.write(f"Mean diameter: {df['diameter_mean_mm'].mean():.2f} mm\n\n")

            # VEINS
            if "analyzer_veins" in results:
                analyzer = results["analyzer_veins"]
                f.write("=" * 80 + "\n")
                f.write("VEIN ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                if hasattr(analyzer, "branch_metrics_df") and analyzer.branch_metrics_df is not None:
                    df = analyzer.branch_metrics_df
                    f.write(f"Branches: {len(df)}\n")
                    f.write(f"Total length: {df['length_mm'].sum():.2f} mm\n")
                    f.write(f"Total volume: {df['volume_mm3'].sum():.2f} mm³\n")
                    f.write(f"Mean diameter: {df['diameter_mean_mm'].mean():.2f} mm\n\n")

        print(f"\n✓ Summary report saved: {report_path}")


if __name__ == "__main__":

    # ==============================
    # CONFIGURAZIONE (MODIFICA QUI)
    # ==============================

    INPUT_FOLDER = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\test_data"      # cartella con .mhd + .zraw
    OUTPUT_FOLDER = "vessel_output"             # root output
    FAST_SEGMENTATION = False                   # True = TotalSegmentator fast
    MAX_PATH_LENGTH_MM = 400.0                  # cutoff path-based (mm)

    # ==============================
    # ESECUZIONE PIPELINE
    # ==============================

    pipeline = ImprovedVesselPipeline(
        output_root=OUTPUT_FOLDER
    )

    pipeline.process_folder(
        input_folder=INPUT_FOLDER,
        fast_segmentation=FAST_SEGMENTATION,
        max_path_length_mm=MAX_PATH_LENGTH_MM
    )
