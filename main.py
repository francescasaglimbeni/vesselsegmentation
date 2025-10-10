from code_vesselsegmentation.preprocessing import convert_mhd_to_nifti
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
