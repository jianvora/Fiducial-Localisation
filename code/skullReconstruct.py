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

    neighbor = getNeighborVoxel(
        surfaceVoxelCoord, point * ConstPixelSpacing, r=7)
    neighbor = np.array(neighbor)
    patch = surfaceVoxelCoord[neighbor]
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(patch)
    distances, indices = neigh.kneighbors(
        point * ConstPixelSpacing, return_distance=True)
    center = patch[indices][0, 0]
    # print center
    patch -= center
    point = (point*ConstPixelSpacing)-center
    print point
    norm = normals[neighbor[indices]]
    # norm = norm.reshape(1,-1)
    # print norm
    T = find_init_transfo(np.array([0, 0, 1]), copy.deepcopy(norm[0,0]))
    patch = apply_affine(patch, T)
    alignedNorm = apply_affine(copy.deepcopy(norm[0]), T)
    return patch, alignedNorm[0]


def genFiducialModel():
    global ConstPixelSpacing
    innerD = 4  # mm
    outerD = 11  # mm
    height = 2  # mm

    mPixel = np.uint8(np.round(outerD / ConstPixelSpacing[0]))
    if mPixel % 2 != 0:
        mPixel += 2
    else:
        mPixel += 1
    mLayer = np.uint8(np.round(height / ConstPixelSpacing[2]) + 1)

    fiducial = np.zeros((mPixel, mPixel, mLayer))
    for l in range(mLayer):
        for i in range(mPixel):
            for j in range(mPixel):
                d = np.sqrt(((i - (mPixel - 1) * 0.5)*ConstPixelSpacing[0])**2 +
                            ((j - (mPixel - 1) * 0.5)*ConstPixelSpacing[1])**2)
                if d <= outerD * 0.5 and d >= innerD * 0.5 and l < mLayer - 1:
                    fiducial[i, j, l] = 1

    vertFiducial, fFiducial, nFiducial, valFiducial = measure.marching_cubes_lewiner(
        fiducial, 0, ConstPixelSpacing)

    vertFiducial = vertFiducial - np.sum(
        vertFiducial[vertFiducial[:, 2] <= 0],
        axis=0) / vertFiducial[vertFiducial[:, 2] <= 0].shape[0]

    # vertFiducial = np.append(vertFiducial, np.array([[0, 0, 0]]), axis=0)
    # nFiducial = np.append(nFiducial, np.array([[0, 0, 1]]), axis=0)
    return vertFiducial, fFiducial, nFiducial