import numpy as np
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from skullFindSurface import getSurfaceVoxels


def findSurfaceNormals(surfaceVoxels, voxelData, ConstPixelSpacing):
    # Takes in verts, normals from the Marching Cubes Algorithm
    verts, normals, faces = getSurfaceMesh(voxelData, ConstPixelSpacing)

    surfaceVoxels = np.float64(surfaceVoxels) * ConstPixelSpacing

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(verts)
    distances, indices = nbrs.kneighbors(surfaceVoxels)
    surfaceNormals = normals[indices[:]]
    surfaceNormals = surfaceNormals.reshape(
        surfaceNormals.shape[0], surfaceNormals.shape[2])

    surfaceNormals_out,surfaceVoxels_out = getOutwardNormals(surfaceNormals,surfaceVoxels)

    return surfaceNormals_out,surfaceVoxels_out, verts, faces


def getSurfaceMesh(voxelData, ConstPixelSpacing):
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        voxelData, 0, ConstPixelSpacing)
    return verts, normals, faces

def getOutwardNormals(normals, surfels):
    mid = np.average(surfels, 0)
    diff_coord = surfels - mid
    outward_normals = normals[np.sum(diff_coord * normals, 1) > 0]
    outer_surfels = surfels[np.sum(diff_coord * normals, 1) > 0]
    return outward_normals, outer_surfels
