import numpy as np
from scipy.spatial import cKDTree, distance
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from mayavi import mlab
from skullReconstruct import *

vFiducial = np.array([])
fFiducial = np.array([])
kdt = cKDTree(pointCloud)

ConstPixelSpacing = (1, 1, 1)


def rotatePC(vert, normal, theta, phi):
    rotationMatrix = np.array([[[np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)],
                                [-np.sin(p), np.cos(p), 0],
                                [-np.sin(t) * np.cos(p), -np.sin(t) * np.sin(p), np.cos(t)]]
                               for p, t in zip(phi, theta)], dtype=np.float64)

    # print rot[0:2]
    rotatedPatches = np.array([np.matmul(rot, v.T).T for rot,
                               v in zip(rotationMatrix, vert)])
    # rotated_normal = np.matmul(rot, normal.T)
    return rotatedPatches


def getNeighborVoxel(pointCloud, points, r):
    global kdt
    neighbor = kdt.query_ball_point(points, r)
    return neighbor


def checkFiducial(pointCloud, poi, normals, neighbor):
    global vFiducial, fFiducial

    patches = np.array([pointCloud[lst] for lst in neighbor])
    # all patches are translated to origin(or normalised) by subtracting
    # the coordinate of point around which the patch is taken
    normPatches = np.array([patches[i] - poi[i] for i in range(len(poi))])

    phi = np.arctan(normals[:, 1] / normals[:, 0])
    theta = np.arctan(
        np.sqrt(normals[:, 0]**2 + normals[:, 1]**2) / normals[:, 2])
    theta[theta < 0] += np.pi

    alignedPatches = rotatePC(normPatches.copy(), normals.copy(), theta, phi)


    mlab.points3d(alignedPatches[0][:, 0],
                  alignedPatches[0][:, 1], alignedPatches[0][:, 2])
    # mlab.quiver3d(0, 0, 0, 0, 0, 1)
    # mlab.quiver3d(0, 0, 0, norm[0], norm[1], norm[2], color=(0, 1, 0))
    mlab.show()