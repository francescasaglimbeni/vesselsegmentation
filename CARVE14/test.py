import numpy as np
import SimpleITK as sitk

img = sitk.ReadImage("CARVE14/1.2.840.113704.1.111.1396.1132404220_fullAnnotations.mhd")
arr = sitk.GetArrayFromImage(img)

# Ignora valori negativi
vals = np.unique(arr[arr >= 0])
print("Valori unici:", vals)
