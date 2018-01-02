# to run this code, install dicom_numpy, pydicom, natsort(on pip),skimage and mayavi

from skullFindSurface import *
from skullReconstruct import *
from skullNormalExtraction import *
from skullFindFIducial import *
import time
from mayavi import mlab
import copy
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

ConstPixelSpacing = (1.0, 1.0, 1.0)

# def main():
global ConstPixelSpacing
start_time = time.time()
# change to required path to dicom folder
PathDicom = "../2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE2"
PathDicom = "../2016.05.26 Glass Scan 1 mm/Glass Scan Axial 1.25 mm/DICOM/PA1/ST1/SE2"

data = readDicomData(PathDicom)
print("---- %s seconds -----" % (time.time() - start_time))

voxelData, ConstPixelSpacing = get3DRecon(data)
print(ConstPixelSpacing)
voxelData,ConstPixelSpacing = image_zoom(voxelData,(1,1,3))  ## zooming the image
voxelDataThresh = applyThreshold(copy.deepcopy(voxelData))
print(ConstPixelSpacing)
print("SLices" + str(voxelData.shape))
print("---- %s seconds -----" % (time.time() - start_time))

surfaceVoxels = getSurfaceVoxels(voxelDataThresh)

print("---- %s seconds -----" % (time.time() - start_time))

normals, surfaceVoxelCoord, verts, faces = findSurfaceNormals(copy.deepcopy(
    surfaceVoxels), voxelData, ConstPixelSpacing)


print("---- %s seconds -----" % (time.time() - start_time))

print(surfaceVoxelCoord.shape)
i = 20  # decrease this to sample more points
# red_set = np.random.choice(surfaceVoxelCoord.shape[0], 10000, replace = False)
normals_red = normals[::i]
surfaceVoxelCoord_red = surfaceVoxelCoord[::i]


print("---- %s seconds -----" % (time.time() - start_time))
surfaceVoxelCoord_red = np.uint16(np.float64(surfaceVoxelCoord_red)/ConstPixelSpacing)

print(len(surfaceVoxelCoord_red))

print("---- %s seconds -----" % (time.time() - start_time))
costs, patches = checkFiducial(surfaceVoxelCoord,
              surfaceVoxelCoord_red,normals, ConstPixelSpacing)

# Visualise in Mayavi
visualiseFiducials(costs, patches, surfaceVoxelCoord, verts, faces)

print("---- %s seconds -----" % (time.time() - start_time))
    
# if __name__ == '__main__':
#     main()
