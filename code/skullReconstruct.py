# -*- coding: utf-8 -*-
"""
Autonomous localization of fiducial markers for IGNS.
This script contains utilities for handling DICOM data
and reconstructing 3D scans.

Authors: P. Khirwadkar, H. Loya, D. Shah, R. Chaudhry,
A. Ghosh & S. Goel (For Inter IIT Technical Meet 2018)
Copyright © 2018 Indian Institute of Technology, Bombay
"""
import dicom_numpy
import dicom
import os
import copy
from natsort import natsorted
import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage import measure
from skullFindFIducial import *
import scipy.ndimage as nd
ConstPixelSpacing = (1.0, 1.0, 1.0)


def readDicomData(path):
    """
    Reads the files specified in path, and returns DICOM data
    corresponding to the files.
    """
    lstFilesDCM = []
    for root, directory, fileList in os.walk(path):
        for filename in fileList:
            if filename == ".DS_Store":
                continue
            lstFilesDCM.append(filename)

    lstFilesDCM = natsorted(lstFilesDCM)  # Normally lexicographic!
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
    """
    Performs 3D reconstruction from the given DICOM data, and
    returns a voxel array and the pixel spacing factors.
    """
    global ConstPixelSpacing
    RefDs = data[0]

    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(
        RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    try:
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
    except dicom_numpy.DicomImportException as e:
        # Invalid DICOM data
        print("Handling incompatible dicom slices")
        makeCompatible(data, prec=5)
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
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


def interpolate_image(A, factor):
    """
    Interpolate object A by _factor_ and resample.
    """
    global ConstPixelSpacing
    A = copy.deepcopy(A)
    PixelSpacing = []
    for i in range(3):
        PixelSpacing.append(ConstPixelSpacing[i] / factor[i])
    ConstPixelSpacing = tuple(PixelSpacing)
    print("Interpolating image by " + str(factor))
    Atrans = nd.interpolation.zoom(A, zoom=factor)
    Atrans = np.array(Atrans, dtype='float32')

    return Atrans, ConstPixelSpacing
