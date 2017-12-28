import numpy as np
from scipy.spatial import cKDTree, distance
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from mayavi import mlab
from skullReconstruct import *
import matplotlib.pyplot as plt
import time


vFiducial = np.array([])
fFiducial = np.array([])
ConstPixelSpacing = (1.0, 1.0, 1.0)


# def rotatePC(vert, normal, theta, phi):
#     rotationMatrix = np.array([[[np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)],
#                                 [-np.sin(p), np.cos(p), 0],
#                                 [-np.sin(t) * np.cos(p), -np.sin(t) * np.sin(p), np.cos(t)]]
#                                for p, t in zip(phi, theta)], dtype=np.float64)

#     # print rot[0:2]
#     rotatedPatches = np.array([np.matmul(rot, v.T).T for rot,
#                                v in zip(rotationMatrix, vert)])
#     # rotated_normal = np.matmul(rot, normal.T)
#     return rotatedPatches


def getNeighborVoxel(pointCloud, points, r):
    kdt = cKDTree(pointCloud)
    neighbor = kdt.query_ball_point(points, r)
    return neighbor

# function to compare point clouds u and v


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    ## assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    # display the point cloud, after initial transformation

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, mean_error, i


def apply_affine(A, init_pose):
    m = A.shape[1]
    src = np.ones((m + 1, A.shape[0]))
    src[:m, :] = np.copy(A.T)
    if init_pose is not None:
        src = np.dot(init_pose, src)
    return src[:m, :].T


def find_init_transfo(evec1, evec2):
    e_cross = np.cross(evec1, evec2)
    e_cross1 = e_cross[0]
    e_cross2 = e_cross[1]
    e_cross3 = e_cross[2]
    i = np.identity(3)
    v = np.zeros((3, 3))
    v[1, 0] = e_cross3
    v[2, 0] = -e_cross2
    v[0, 1] = -e_cross3
    v[2, 1] = e_cross1
    v[0, 2] = e_cross2
    v[1, 2] = -e_cross1
    v2 = np.dot(v, v)
    c = np.dot(evec1, evec2)
    # will not work in case angle of rotation is exactly 180 degrees
    R = i + v + (v2 / (1 + c))
    T = np.identity(4)
    T[0:3, 0:3] = R
    T = np.transpose(T)
    R = np.transpose(R)
    #com = [img.resolution[0]*len(img)/2,img.resolution[1]*len(img[0])/2,img.resolution[2]*len(img[0][0])/2]
    [tx, ty, tz] = [0, 0, 0]
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def genPHI(patch, size=15):
    res = 1
    size *= res
    if size % 2 == 0:
        size += 1

    center = np.uint8(size / 2)

    patch += (center, center, 0)

    depthMap = np.ones((size, size)) * (-100)

    for i in range(patch.shape[0]):
        depthMap[np.uint8(patch[i, 0] * res), np.uint8(patch[i, 1] * res)] = max(
            (depthMap[np.uint8(patch[i, 0] * res), np.uint8(patch[i, 1] * res)]), (patch[i, 2]))
    depthMap[depthMap == (-100)] = 0

    return depthMap


def comparePHI(PHI1, PHI2):
    cost = np.correlate(PHI1, PHI2)
    return cost


def checkFiducial(pointCloud, poi, normals, neighbor):
    global vFiducial, fFiducial
    start_time = time.time()

    patches = np.array([pointCloud[lst] for lst in neighbor])
    # all patches are translated to origin(or normalised) by subtracting
    # the coordinate of point around which the patch is taken
    normPatches = np.array([patches[i] - poi[i] for i in range(len(poi))])

    if vFiducial.size == 0:
        vFiducial = genFiducialModel(pointCloud, normals)

    """
    phi = np.arctan(normals[:, 1] / normals[:, 0])
    theta = np.arctan(
        np.sqrt(normals[:, 0]**2 + normals[:, 1]**2) / normals[:, 2])
    theta[theta < 0] += np.pi

    alignedPatches = rotatePC(normPatches.copy(), normals.copy(), theta, phi)

    """
    print("---- %s seconds -----" % (time.time() - start_time))

    alignedPatches = []
    for i in range(len(poi)):
        affine_T = find_init_transfo(normals[i], np.array([0, 0, 1]))
        alignedPatches.append(np.array(apply_affine(normPatches[i], affine_T)))
    alignedPatches = np.array(alignedPatches)

    print("---- %s seconds -----" % (time.time() - start_time))

    cost = []

    for i in range(len(poi)):
        if(i % 20 == 0):
            print("ICP: "),
            print(i)
        if(len(alignedPatches[i]) > 50):
            cost.append(
                icp(alignedPatches[i], vFiducial, max_iterations=10)[1])

    print("END OF ICP")

    cost_sorted = np.sort(cost)
    print(cost_sorted)
    print("")
    print("")

    print (str(cost_sorted[0]) + " " + str(cost_sorted[1]) + " " + str(
        cost_sorted[2]) + " " + str(cost_sorted[3]) + " " + str(cost_sorted[4]))

    colormap = np.random.rand(10, 3)
    for i in range(10):
        patch = patches[cost.index(cost_sorted[i])]
        mlab.points3d(patch[:, 0], patch[:, 1], patch[:, 2], color=colormap[i])

    mlab.points3d(pointCloud[::10, 0], pointCloud[::10, 1],
                  pointCloud[::10, 2], color=(1, 0, 0))
    # mlab.quiver3d(0, 0, 0, 0, 0, 1)
    # mlab.quiver3d(0, 0, 0, norm[0], norm[1], norm[2], color=(0, 1, 0))
    mlab.show()
