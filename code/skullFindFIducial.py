import numpy as np
from scipy.spatial import cKDTree, distance
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from mayavi import mlab
from skullReconstruct import *

vFiducial = np.array([])
fFiducial = np.array([])
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
    kdt = cKDTree(pointCloud)
    neighbor = kdt.query_ball_point(points, r)
    return neighbor

# function to compare point clouds u and v


def comparePC(u, v):
    # dist = max(distance.directed_hausdorff(u, v)[0],
    #            distance.directed_hausdorff(v, u)[0])
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(v)
    distances, _ = nbrs.kneighbors(u)
    cost = np.matmul(distances.T, distances)
    return cost


def checkFiducial(pointCloud, poi, normals, neighbor):
    global vFiducial, fFiducial

    patches = np.array([pointCloud[lst] for lst in neighbor])
    # all patches are translated to origin(or normalised) by subtracting
    # the coordinate of point around which the patch is taken
    normPatches = np.array([patches[i] - poi[i] for i in range(len(poi))])

    if vFiducial.size == 0:
        vFiducial, fFiducial, _ = genFiducialModel()

    phi = np.arctan(normals[:, 1] / normals[:, 0])
    theta = np.arctan(
        np.sqrt(normals[:, 0]**2 + normals[:, 1]**2) / normals[:, 2])
    theta[theta < 0] += np.pi

    alignedPatches = rotatePC(normPatches.copy(), normals.copy(), theta, phi)

    cost = np.array([comparePC(vFiducial, alignedPatches[i])
                     for i in range(len(poi))])

    i = np.argmin(cost)

    print cost[i]

    patch = alignedPatches[i]

    # plotting the patch giving minimum cost
    mlab.triangular_mesh(
        vFiducial[:, 0], vFiducial[:, 1], vFiducial[:, 2], fFiducial)
    mlab.points3d(patch[:, 0],
                  patch[:, 1], patch[:, 2])
    # mlab.quiver3d(0, 0, 0, 0, 0, 1)
    # mlab.quiver3d(0, 0, 0, norm[0], norm[1], norm[2], color=(0, 1, 0))
    mlab.show()
