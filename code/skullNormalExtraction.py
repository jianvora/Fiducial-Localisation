# import dicom_numpy
# import dicom
# import os
# from natsort import natsorted
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# import itertools

# from mayavi import mlab
# import time
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from skullFindSurface import getSurfaceVoxels

def findSurfaceNormals(surfaceVoxels, voxelData, ConstPixelSpacing):
	# Takes in verts, normals from the Marching Cubes Algorithm
	verts, normals = getSurfaceMesh(voxelData, ConstPixelSpacing)
	# print verts.shape, normals.shape
	# print surfaceVoxels.shape	
	nbrs = NearestNeighbors(n_neighbors=1).fit(verts)
	distances, indices = nbrs.kneighbors(surfaceVoxels)
	# print list(indices)
	surfaceNormals = normals[list(indices),:]
	# print surfaceNormals.shape
	return surfaceNormals
	
def getSurfaceMesh(voxelData, ConstPixelSpacing):
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        voxelData, 0, ConstPixelSpacing)
    return verts, normals
