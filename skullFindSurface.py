# to run this code, install dicom_numpy, pydicom, natsort(on pip),skimage and mayavi
import dicom_numpy
import dicom
import os
from natsort import natsorted
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
from skullReconstruct import readDicomData, get3DRecon
from mayavi import mlab
import time
from skimage import measure

def getSurfaceVoxels(voxelData):
	dim = voxelData.shape
	onVoxelsX, onVoxelsY, onVoxelsZ = np.nonzero(voxelData == 1)
	onVoxels = np.stack((onVoxelsX, onVoxelsY, onVoxelsZ), axis=1)

	surfaceVoxels = []

	for i in range(onVoxels.shape[0]):
		x = onVoxels[i][0]
		y = onVoxels[i][1]
		z = onVoxels[i][2]

		xList = [x-1, x, x+1]
		yList = [y-1, y, y+1]
		zList = [z-1, z, z+1]

		xList = [x for x in xList if x>=0 and x<dim[0]]
		yList = [y for y in yList if y>=0 and y<dim[1]]
		zList = [z for z in zList if z>=0 and z<dim[2]]

		neighbours = list(itertools.product(xList, yList, zList))

		for n in neighbours:
			if(voxelData[n[0],n[1],n[2]] == 0):
				surfaceVoxels.append([x,y,z])
				break

		# neighbours = [list(tup) for tup in neighbours]
		# print neighbours

		# break

	return surfaceVoxels

# Path to the folder having dicom images
PathDicom = "./2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE2"
data = readDicomData(PathDicom)
voxelData = get3DRecon(data) 

print(time.asctime(time.localtime(time.time())))
surfaceVoxels = getSurfaceVoxels(voxelData)
surfaceVoxels = np.asarray(surfaceVoxels)
print(time.asctime(time.localtime(time.time())))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(surfaceVoxels[:,0],surfaceVoxels[:,1],surfaceVoxels[:,2])
plt.show()

# verts, faces, normals, values = measure.marching_cubes_lewiner(voxel_ndarray, 0, ConstPixelSpacing)

# mlab.triangular_mesh([vert[0] for vert in verts],
#                       [vert[1] for vert in verts],
#                       [vert[2] for vert in verts], faces)
# print("---- %s seconds -----"%(time.time() - start_time))
# mlab.show()
