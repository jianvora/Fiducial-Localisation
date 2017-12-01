# to run this code, install dicom_numpy, pydicom, natsort(on pip),skimage and mayavi

import dicom_numpy
import dicom
import os
from natsort import natsorted
from mayavi import mlab
import time
from skimage import measure
start_time = time.time()

# Path to the folder having dicom images
PathDicom = "/Users/Parth/Inter_IIT_Tech/2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE2"

lstFilesDCM = []  # create an empty list
# may want to exclude the first dicom image in some files
for __,__, fileList in os.walk(PathDicom):
    for filename in fileList:
    	if filename==".DS_Store":
    		continue
        lstFilesDCM.append(filename)
# the function natsorted() from natsort library does natural sorting
# i.e the files are in the order "IM1,IM2,IM3..." 
# instead of "IM1,IM10,IM100.." which is the lexicographical order
lstFilesDCM = natsorted(lstFilesDCM)

datasets = [dicom.read_file(PathDicom + '/' + f) for f in lstFilesDCM]
RefDs = datasets[0]
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

try:
    voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)
except dicom_numpy.DicomImportException as e:
    # invalid DICOM data
    raise
# arbitrary for now, can be set to different values for CT scan as in Hounsfield unit,
# bone is from +700 to +3000
upper_thresh = 0
lower_thresh = 0

voxel_ndarray[voxel_ndarray > upper_thresh] = 1
voxel_ndarray[voxel_ndarray <= lower_thresh] = 0

verts, faces, normals, values = measure.marching_cubes_lewiner(voxel_ndarray, 0, ConstPixelSpacing)

mlab.triangular_mesh([vert[0] for vert in verts],
                      [vert[1] for vert in verts],
                      [vert[2] for vert in verts], faces)
print("---- %s seconds -----"%(time.time() - start_time))
mlab.show()
