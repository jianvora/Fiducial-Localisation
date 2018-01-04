import numpy as np
from scipy.spatial import cKDTree, distance
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from mayavi import mlab
from skullReconstruct import *
import matplotlib.pyplot as plt
import copy
import time


vFiducial = np.array([])
fFiducial = np.array([])
ConstPixelSpacing = (1.0, 1.0, 1.0)


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

    ## assert src.shape == dst.shape

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

    ##assert A.shape == B.shape

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
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
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
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    ## display the point cloud, after initial transformation
    
    prev_error = 0
    error_arr = []
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        error_arr.append(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error


    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)
    min_error = np.array(error_arr).min()
    
    return T, min_error/len(distances), i

def apply_affine(A,init_pose):
    m = A.shape[1]
    src = np.ones((m+1,A.shape[0]))
    src[:m,:] = np.copy(A.T)
    if init_pose is not None:
        src = np.dot(init_pose, src)
    return src[:m,:].T
   
def find_init_transfo(evec1,evec2):
    e_cross = np.cross(evec1,evec2)
    e_cross1 = e_cross[0]
    e_cross2 = e_cross[1]
    e_cross3 = e_cross[2]
    i = np.identity(3)
    v = np.zeros((3,3))
    v[1,0] = e_cross3
    v[2,0] = -e_cross2
    v[0,1] = -e_cross3
    v[2,1] = e_cross1
    v[0,2] = e_cross2
    v[1,2] = -e_cross1
    v2 = np.dot(v,v)
    c = np.dot(evec1,evec2)
    R = i + v + (v2/(1+c)) ## will not work in case angle of rotation is exactly 180 degrees
    T = np.identity(4)
    T[0:3,0:3] = R
    T = np.transpose(T)
    R = np.transpose(R)
    #com = [img.resolution[0]*len(img)/2,img.resolution[1]*len(img[0])/2,img.resolution[2]*len(img[0][0])/2]
    [tx,ty,tz] = [0,0,0]
    T[0,3] = tx
    T[1,3] = ty
    T[2,3] = tz
    return T

def genFiducialModel(surfaceVoxelCoord, normals, point,neighbor, PixelSpacing):
    # point = np.float64(point)*PixelSpacing
    # neighbor = getNeighborVoxel(
    #      surfaceVoxelCoord, point, r=4.8)
    # neighbor = np.array(neighbor)
    # print(neighbor-neighbor1)
    patch = surfaceVoxelCoord[neighbor]
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(patch)
    distances, indices = neigh.kneighbors(
        point.reshape(1,-1), return_distance=True)
    center = patch[indices][0, 0]
    orignal_patch = copy.deepcopy(patch)
    patch -= center
    point -= center
    # print(neighbor)
    # norm = np.zeros((1,3),dtype='float')
    # for i in range(len(indices[0])):
    #     norm += normals[neighbor[indices[0][i]]].reshape(-1,3)
    norm = normals[neighbor[indices]].reshape(-1,3)
    norm = np.sum(norm,axis=0)/1
    #norm  = norm/4
    norm = norm.reshape(1,-1)

    T = find_init_transfo(np.array([0.0, 0.0, 1.0]), copy.deepcopy(norm[0]))
    patchch = apply_affine(copy.deepcopy(patch), T)
    alignedNorm = apply_affine(copy.deepcopy(norm), T)
    return patchch,alignedNorm[0],orignal_patch

def genFiducialModel_old(PixelSpacing):
    global ConstPixelSpacing
    ConstPixelSpacing = PixelSpacing
    innerD = 4  # mm
    outerD = 14 * ConstPixelSpacing[0]  # mm
    height = 2  # mm
    print outerD
    mPixel = np.uint8(np.round(outerD / ConstPixelSpacing[0]))
    if mPixel % 2 != 0:
        mPixel += 16
    else:
        mPixel += 15
    # mLayer = np.uint8(np.round(height / ConstPixelSpacing[2]) + 1)
    mLayer = height / ConstPixelSpacing[2]
    # print height / ConstPixelSpacing[2]

    fiducial = np.zeros((mPixel, mPixel, int(mLayer) + 2))
    for l in range(fiducial.shape[2]):
        for i in range(mPixel):
            for j in range(mPixel):
                d = np.sqrt(((i - (mPixel - 1) * 0.5) * ConstPixelSpacing[0])**2 +
                            ((j - (mPixel - 1) * 0.5) * ConstPixelSpacing[1])**2)
                if d <= outerD * 0.5 and d >= innerD * 0.5 and l <= mLayer:
                    fiducial[i, j, l] = 1
                elif d > (outerD * 0.5) and d < ((outerD * 0.5) + 1) and l <= mLayer:
                    fiducial[i, j, l] = 1 - (d - (outerD * 0.5))
                elif d < innerD * 0.5 and d < ((innerD * 0.5) - 1) and l <= mLayer:
                    fiducial[i, j, l] = 1 + (d - (innerD * 0.5))
                # elif l > mLayer and l < mLayer + 1 and d <= outerD * 0.5 and d >= innerD * 0.5:
                #     fiducial[i, j, l] = 1 - (l - mLayer)
    disk = np.zeros((fiducial.shape[0], fiducial.shape[1]))
    for i in range(fiducial.shape[0]):
        for j in range(fiducial.shape[1]):
            d = np.sqrt(((i - (mPixel - 1) * 0.5) * ConstPixelSpacing[0])**2 +
                        ((j - (mPixel - 1) * 0.5) * ConstPixelSpacing[1])**2)
            if d <= innerD * 0.5:
                disk[i, j] = 1
            # elif d > innerD * 0.5 and d < ((innerD * 0.5) + 1) and l <= mLayer:
                    # fiducial[i, j, l] = 1 - (d - (innerD * 0.5))
    x, y = np.where(disk==1)
    z = np.zeros(x.size)
    x = np.float64(x)*ConstPixelSpacing[0]
    y = np.float64(y)*ConstPixelSpacing[1]
    x -= np.sum(x)/x.size
    y -= np.sum(y)/y.size
    vert = np.stack([x,y,z],axis=1)

    # fiducial = np.insert(fiducial, 0, np.ones((mPixel, mPixel)), axis=2)
    vertFiducial, fFiducial, nFiducial, valFiducial = measure.marching_cubes_lewiner(
        fiducial, 0, ConstPixelSpacing)

    vertFiducial = vertFiducial - np.sum(
        vertFiducial[vertFiducial[:, 2] <= 0],
        axis=0) / vertFiducial[vertFiducial[:, 2] <= 0].shape[0]
    vertFiducial = np.append(vertFiducial, vert, axis=0)
    # vertFiducial = np.append(vertFiducial, np.array([[0, 0, 0]]), axis=0)
    # nFiducial = np.append(nFiducial, np.array([[0, 0, 1]]), axis=0)
    # mlab.triangular_mesh(
    # vertFiducial[:, 0], vertFiducial[:, 1], vertFiducial[:, 2], fFiducial)
    # mlab.points3d(vertFiducial[:, 0], vertFiducial[:, 1], vertFiducial[:, 2])
    # mlab.points3d(x,y,z,color=(0,1,0))
    # mlab.show()
    return vertFiducial, fFiducial, nFiducial



def checkFiducial(pointCloud, poi, normalstotal, PixelSpacing):
    global vFiducial, fFiducial,ConstPixelSpacing
    start_time = time.time()
    ConstPixelSpacing = PixelSpacing
    
    if vFiducial.size == 0:
        #vFiducial,_,_ = genFiducialModel(pointCloud, normalstotal,np.array([385,201,3*97]), ConstPixelSpacing)
        vFiducial,_,_ = genFiducialModel_old(ConstPixelSpacing)
    alignedPatches = []
    patches =  []
    point = np.float64(copy.deepcopy(poi))*ConstPixelSpacing
    neighbor1 = getNeighborVoxel(pointCloud, point, r=4.8)
    neighbor1 = np.array(neighbor1)
    
    for i in range(len(point)):
        algiP, aligN, P = genFiducialModel(pointCloud, normalstotal, point[i],np.array(neighbor1[i]).astype(int), ConstPixelSpacing)
        alignedPatches.append(algiP)
        patches.append(P)
        if(i%20 == 0):
            print("POI "+str(i)+" "),
            print(time.time()-start_time)
    patches = np.array(patches) ## contains the orignal patch
    alignedPatches = np.array(alignedPatches) ## contains the transformed patch

    
    cost = []

    count = 0
    for i in range(len(point)):
        if(i%200 == 0):
            print("ICP: "),
            print(i)
        if(len(alignedPatches[i])>800):
            cost.append(icp(alignedPatches[i], vFiducial,max_iterations=1)[1])
        else:
            count += 1
            cost.append(100)  ## a very high value
    print("END OF ICP")
    print(str(count)+ " of small point clouds detected")
    #cost = np.array([nearest_neighbor(vFiducial, alignedPatches[i])
                     #for i in range(len(poi))])
    

    cost_sorted = np.sort(cost)
    print("")
    print("")
    for i in range(40):
        print(cost_sorted[i]),
        print(" "),
        print(cost.index(cost_sorted[i])),
        print(" "),
        print(alignedPatches[i].size)

    
    colormap = np.random.rand(30,3)
    mlab.points3d(vFiducial[:,0],vFiducial[:,1],vFiducial[:,2])
    for i in range(30):
        patch = patches[cost.index(cost_sorted[i])]
        mlab.points3d(patch[:,0],patch[:,1],patch[:,2],color=tuple(colormap[i]))


    mlab.points3d(pointCloud[::80,0],pointCloud[::80,1],pointCloud[::80,2], color=(1,0,0))
    # mlab.quiver3d(0, 0, 0, 0, 0, 1)
    # mlab.quiver3d(0, 0, 0, norm[0], norm[1], norm[2], color=(0, 1, 0))
    mlab.show()
