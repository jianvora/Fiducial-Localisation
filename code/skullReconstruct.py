import dicom_numpy
import dicom
import os
from natsort import natsorted
import numpy as np
import copy
from skimage import measure
from skullFindFIducialLoya import *

ConstPixelSpacing = (1, 1, 1)


def readDicomData(path):
    lstFilesDCM = []
    # may want to exclude the first dicom image in some files
    for root, directory, fileList in os.walk(path):
        for filename in fileList:
            if filename == ".DS_Store":  # or filename == "IM1":
                continue
            lstFilesDCM.append(filename)
    '''
    the function natsorted() from natsort library does natural sorting
    i.e the files are in the order "IM1,IM2,IM3..."
    instead of "IM1,IM10,IM100.." which is the lexicographical order
    '''
    lstFilesDCM = natsorted(lstFilesDCM)
    data = [dicom.read_file(path + '/' + f) for f in lstFilesDCM]
    return data


def makeCompatible(dicomData, prec=5):
    for i in range(len(dicomData)):
        a = dicomData[i].ImageOrientationPatient
        print a
        a[0] = round(a[0], prec)
        a[1] = round(a[1], prec)
        a[2] = round(a[2], prec)
        a[3] = round(a[3], prec)
        dicomData[i].ImageOrientationPatient = a


def get3DRecon(data):
    global ConstPixelSpacing
    RefDs = data[0]

    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(
        RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    try:
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
    except dicom_numpy.DicomImportException as e:
        # invalid DICOM data
        print("Handling incompatible dicom slices")
        makeCompatible(data, prec=5)
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
        # raise NameError(
        #     'Unable to do 3D reconstruction. Slice missing? or incompatible slice data?')

    return voxel_ndarray, ConstPixelSpacing


def applyThreshold(voxelData):
    # arbitrary for now, can be set to different values for CT scan as in Hounsfield unit,
    # bone is from +700 to +3000
    upper_thresh = 0
    lower_thresh = 0
    voxel = voxelData
    voxel[voxel > upper_thresh] = 1
    voxel[voxel <= lower_thresh] = 0

    return voxel


def extractFiducialModel(surfaceVoxelCoord, normals, point):
    point = np.float64(point) * ConstPixelSpacing
    neighbor = getNeighborVoxel(
        surfaceVoxelCoord, point, r=9)
    neighbor = np.array(neighbor)
    patch = surfaceVoxelCoord[neighbor]
    n = 8
    neigh = NearestNeighbors(n_neighbors=n)
    neigh.fit(patch)
    distances, indices = neigh.kneighbors(
        point.reshape(1, -1), return_distance=True)
    center = patch[indices][0, 0]
    original = patch.copy()
    # print center
    patch -= center
    point -= center
    # print point
    norm = normals[neighbor[indices]].reshape(-1, 3)
    norm = np.sum(norm, axis=0) / n
    norm = norm.reshape(1, -1)
    # norm = norm.reshape(1,-1)
    # print norm
    T = find_init_transfo(np.array([0, 0, 1]), copy.deepcopy(norm[0]))
    patch = apply_affine(patch, T)
    alignedNorm = apply_affine(copy.deepcopy(norm), T)
    return patch, alignedNorm[0], original

def genFiducialModel():
    global ConstPixelSpacing
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



# def genFiducialModel():
#     global ConstPixelSpacing
#     innerD = 5  # mm
#     outerD = 14 * ConstPixelSpacing[0]  # mm
#     height = 2  # mm
#     # print outerD
#     mPixel = np.uint8(np.round(outerD / ConstPixelSpacing[0]))
#     if mPixel % 2 != 0:
#         mPixel += 16
#     else:
#         mPixel += 15
#     # mLayer = np.uint8(np.round(height / ConstPixelSpacing[2]) + 1)
#     mLayer = height / ConstPixelSpacing[2]
#     # mLayer+=1
#     # print height / ConstPixelSpacing[2]

#     fiducial = np.zeros((mPixel, mPixel, int(mLayer) + 3))
#     print int(mLayer)
#     for l in range(fiducial.shape[2]):
#         for i in range(fiducial.shape[0]):
#             for j in range(fiducial.shape[1]):
#                 d = np.sqrt(((i - (mPixel - 1) * 0.5) * ConstPixelSpacing[0])**2 +
#                             ((j - (mPixel - 1) * 0.5) * ConstPixelSpacing[1])**2)
#                 # if l==0 and d<=innerD*0.5:
#                 # fiducial[i, j, l] = 1
#                 # elif d > (innerD * 0.5) and d < ((innerD * 0.5) + 1) and l == 0 :
#                 # fiducial[i, j, l] = 1 - (d - (innerD * 0.5))
#                 if d <= outerD * 0.5 and d >= innerD * 0.5 and l * ConstPixelSpacing[2] <= height:
#                     fiducial[i, j, l] = 1
#                 elif d > (outerD * 0.5) and d < ((outerD * 0.5) + 1) and l * ConstPixelSpacing[2] <= height:
#                     fiducial[i, j, l] = 1 - (d - (outerD * 0.5))
#                 elif d < innerD * 0.5 and d > ((innerD * 0.5) - 1) and l * ConstPixelSpacing[2] <= height:
#                     fiducial[i, j, l] = 1 + (d - (innerD * 0.5))
#                 # elif l* ConstPixelSpacing[2] > height and l* ConstPixelSpacing[2] < height + 1 and d <= outerD * 0.5 and d >= innerD * 0.5:
#                 #     fiducial[i, j, l] = 1 - (l * ConstPixelSpacing[2] - height)
#                 # elif l* ConstPixelSpacing[2] > height and l* ConstPixelSpacing[2] < height + 1 and d > outerD * 0.5 and d < outerD * 0.5 + 1:
#                 #     fiducial[i, j, l] = 1 - (l* ConstPixelSpacing[2] - height)
#                 # elif l* ConstPixelSpacing[2] > height and l* ConstPixelSpacing[2] < height + 1 and d > innerD * 0.5 - 1 and d < innerD * 0.5:
#                 #     fiducial[i, j, l] = 1 - (l* ConstPixelSpacing[2] - height)

#     # fiducial = np.insert(fiducial, 0, np.ones((fiducial.shape[0], fiducial.shape[1])), axis=2)
#     vertFiducial, fFiducial, nFiducial, valFiducial = measure.marching_cubes_lewiner(
#         fiducial, 0, ConstPixelSpacing)
#     x, y, z = np.where(fiducial > 0)
#     # print z
#     vertFiducial = vertFiducial - np.sum(
#         vertFiducial[vertFiducial[:, 2] <= 0],
#         axis=0) / vertFiducial[vertFiducial[:, 2] <= 0].shape[0]

#     # vertFiducial = np.append(vertFiducial, np.array([[0, 0, 0]]), axis=0)
#     # nFiducial = np.append(nFiducial, np.array([[0, 0, 1]]), axis=0)
#     # mlab.triangular_mesh(
#         # vertFiducial[:, 0], vertFiducial[:, 1], vertFiducial[:, 2], fFiducial)
#     mlab.points3d(x,y,z)
#     # mlab.points3d(vertFiducial[:, 0], vertFiducial[:, 1], vertFiducial[:, 2])
#     mlab.show()
#     return vertFiducial, fFiducial, nFiducial


def genFiducialPHI():
    global ConstPixelSpacing
    innerD = 3.5  # mm
    outerD = 8  # mm
    # height = 2  # mm

    mPixel = np.uint8(np.round(outerD / ConstPixelSpacing[0]))
    if mPixel % 2 != 0:
        mPixel += 16
    else:
        mPixel += 15
    mPixel = 15
    # mLayer = np.uint8(np.round(height / ConstPixelSpacing[2]) + 2)

    fiducial = np.zeros((mPixel, mPixel))
    # for l in range(mLayer):
    for i in range(mPixel):
        for j in range(mPixel):
            d = np.sqrt(((i - (mPixel - 1) * 0.5) * ConstPixelSpacing[0])**2 +
                        ((j - (mPixel - 1) * 0.5) * ConstPixelSpacing[1])**2)
            if d <= outerD * 0.5 and d >= innerD * 0.5:  # and l < mLayer - 1:
                fiducial[i, j] = 1
            elif d > outerD * 0.5 and d < ((outerD * 0.5) + 1):
                fiducial[i, j] = 1 - (d - (outerD * 0.5))
            elif d < innerD * 0.5 and d < ((innerD * 0.5) - 1):
                fiducial[i, j] = 1 + (d - (innerD * 0.5))
    return fiducial
