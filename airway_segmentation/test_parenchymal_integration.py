"""
Quick Test for Parenchymal Metrics Integration
===============================================

This script tests the parenchymal metrics module standalone
before running the full pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from parenchymal_metrics import integrate_parenchymal_metrics

def test_parenchymal_metrics():
    """Test parenchymal metrics on a single scan"""
    
    # Test configuration
    TEST_MHD = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\test_data\ID00378637202298597306391.mhd"
    TEST_OUTPUT = Path(__file__).parent / "test_parenchymal_output"
    
    print("="*80)
    print("TESTING PARENCHYMAL METRICS INTEGRATION")
    print("="*80)
    print(f"\nTest MHD: {TEST_MHD}")
    print(f"Output: {TEST_OUTPUT}")
    
    # Check if test file exists
    test_mhd_path = Path(TEST_MHD)
    if not test_mhd_path.exists():
        print(f"\n❌ ERROR: Test file not found: {TEST_MHD}")
        print("\nPlease update TEST_MHD in this script to point to a valid MHD file.")
        return False
    
    # Run test
    try:
        metrics = integrate_parenchymal_metrics(
            TEST_MHD,
            TEST_OUTPUT,
            fast_segmentation=True,
            verbose=True
        )
        
        if metrics:
            print("\n" + "="*80)
            print("✓ TEST SUCCESSFUL")
            print("="*80)
            print(f"\nComputed metrics:")
            print(f"  Mean Lung Density: {metrics['mean_lung_density_HU']:.1f} HU")
            print(f"  Histogram Entropy: {metrics['histogram_entropy']:.3f}")
            
            print(f"\nOutput files in: {TEST_OUTPUT}")
            print(f"  ✓ parenchymal_metrics.json")
            print(f"  ✓ parenchymal_report.txt")
            print(f"  ✓ lung_mask.nii.gz")
            
            print("\n✅ Parenchymal metrics integration is working correctly!")
            print("✅ You can now run the full pipeline with main_pipeline.py")
            return True
        else:
            print("\n❌ TEST FAILED: Metrics computation returned None")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_parenchymal_metrics()
    sys.exit(0 if success else 1)
