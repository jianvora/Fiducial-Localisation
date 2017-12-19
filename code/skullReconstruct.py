import dicom_numpy
import dicom
import os
from natsort import natsorted
import numpy as np

def readDicomData(path):
    lstFilesDCM = []
    # may want to exclude the first dicom image in some files
    for root, directory, fileList in os.walk(path):
        for filename in fileList:
            if filename == ".DS_Store" or filename == "IM1":
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
        a[0] = round(a[0], prec)
        a[1] = round(a[1], prec)
        a[2] = round(a[2], prec)
        a[3] = round(a[3], prec)
        dicomData[i].ImageOrientationPatient = a

def get3DRecon(data):

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
