import SimpleITK as sitk

# cartella con i DICOM
dicom_dir = "dicom_folder"

# leggi la serie DICOM
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
reader.SetFileNames(dicom_names)

# leggi come volume 3D
image = reader.Execute()

# salva in formato MHD/RAW
sitk.WriteImage(image, "ID00009637202177434476278.mhd")