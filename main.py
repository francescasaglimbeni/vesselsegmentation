'''from code_vesselsegmentation.preprocessing import convert_mhd_to_nifti
from totalsegmentator.python_api import totalsegmentator

def main():
    # Percorso al file MHD
    mhd_path = 'VESSEL12/scans/VESSEL12_01.mhd'
    nifti_path = 'nifti_scans'
    # Cartella temporanea per il file NIfTI
    nifti_path = convert_mhd_to_nifti(mhd_path, nifti_path)   
    # Cartella di output per le segmentazioni
    output_path = 'vessels_segmentations'
        
    # Esegui la segmentazione
    totalsegmentator(nifti_path, output_path, task='lung_vessels')

if __name__ == "__main__":
    main()
'''
from code_vesselsegmentation.preprocessing import convert_mhd_to_nifti
from totalsegmentator.python_api import totalsegmentator
from code_vesselsegmentation.postprocessing import process_vessel_segmentation
import os

def main():
    # Paths
    mhd_path = 'VESSEL12/scans/VESSEL12_01.mhd'
    nifti_dir = 'nifti_scans'
    seg_dir = 'vessels_segmentations'
    output_dir = 'vessels_cleaned'
    
    # Step 1: Convert to NIfTI
    print("=== Step 1: Converting MHD to NIfTI ===")
    nifti_path = convert_mhd_to_nifti(mhd_path, nifti_dir)
    
    # Step 2a: Run TotalSegmentator for lung vessels
    print("\n=== Step 2a: Segmenting Lung Vessels ===")
    print("Running task='lung_vessels' to get intrapulmonary vessels...")
    totalsegmentator(
        nifti_path, 
        seg_dir, 
        task='lung_vessels',
        fast=False
    )
    print("✓ Lung vessels segmentation complete!")
    
    # Step 2b: Run TotalSegmentator for total anatomy
    print("\n=== Step 2b: Segmenting Total Anatomy ===")
    print("Running task='total' to get heart structures and major vessels...")
    print("(This will take several minutes...)")
    totalsegmentator(
        nifti_path, 
        seg_dir, 
        task='total',
        fast=False
    )
    print("✓ Total anatomy segmentation complete!")
    
    # Step 3: Verify critical structures
    print("\n=== Step 3: Verifying Output Files ===")
    
    # Check what we actually have
    all_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    
    # Critical files for A/V classification
    critical_files = {
        'lung_vessels.nii.gz': 'Intrapulmonary vessels',
        'pulmonary_vein.nii.gz': 'Pulmonary vein (vein seed - alternative)',
        'heart.nii.gz': 'Heart (can extract left atrium)',
        'aorta.nii.gz': 'Aorta (artery reference)'
    }
    
    print("\nLooking for critical structures:")
    present = []
    missing = []
    
    for filename, description in critical_files.items():
        filepath = os.path.join(seg_dir, filename)
        if os.path.exists(filepath):
            present.append(f"✓ {filename} - {description}")
        else:
            missing.append(f"✗ {filename} - {description}")
    
    for item in present:
        print(f"  {item}")
    
    if missing:
        print("\nMissing (may need workarounds):")
        for item in missing:
            print(f"  {item}")
    
    # Check for specific structures that TotalSegmentator might name differently
    print("\nSearching for alternative structures:")
    search_terms = {
        'pulmonary': 'Pulmonary vessels',
        'atrium': 'Atrium structures',
        'ventricle': 'Ventricle structures'
    }
    
    for term, description in search_terms.items():
        matches = [f for f in all_files if term in f.lower()]
        if matches:
            print(f"  Found {description}:")
            for m in matches:
                print(f"    - {m}")
    
    # Step 4: Post-process
    print("\n=== Step 4: Post-processing ===")
    
    # Check if lung_vessels exists before processing
    lung_vessels_path = os.path.join(seg_dir, 'lung_vessels.nii.gz')
    if not os.path.exists(lung_vessels_path):
        print("ERROR: lung_vessels.nii.gz not found!")
        print("The task='lung_vessels' may have failed.")
        print("\nTrying to find alternative vessel files...")
        
        vessel_candidates = [f for f in all_files if 'vessel' in f.lower() or 'pulmonary' in f.lower()]
        if vessel_candidates:
            print(f"Found potential alternatives: {vessel_candidates}")
            print("You may need to modify postprocessing.py to use these files.")
        return
    
    cleaned_path, centerlines_path = process_vessel_segmentation(
        seg_dir, 
        output_dir, 
        nifti_path, 
        extract_skeleton=True
    )
    
    if cleaned_path is None:
        print("\nERROR: Post-processing failed!")
        return
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("=== Pipeline Complete! ===")
    print("="*70)
    print(f"✓ Cleaned vessels: {cleaned_path}")
    if centerlines_path:
        print(f"✓ Centerlines: {centerlines_path}")
    
    print(f"\n📁 Generated {len(all_files)} anatomical structures")
    print(f"📁 Output directory: {output_dir}/")
    
    output_files = sorted(os.listdir(output_dir))
    for f in output_files:
        print(f"    ├── {f}")
    
    print("\n🔬 Next steps for A/V classification:")
    print("  1. Visualize results in ITK-SNAP or 3D Slicer")
    print("  2. For seed points, you can use:")
    print("     - Arteries: Use 'aorta.nii.gz' or 'pulmonary_vein.nii.gz' (verify which connects to arteries)")
    print("     - Veins: Use 'pulmonary_vein.nii.gz' or 'heart.nii.gz' regions")
    print("  3. Implement connectivity-based labeling from seeds")
    print("  4. Apply transformer network for refinement")
    
    print("\n💡 Note: TotalSegmentator may not distinguish pulmonary artery vs vein perfectly.")
    print("   You'll need to verify connectivity patterns or use anatomical constraints.")


if __name__ == "__main__":
    main()