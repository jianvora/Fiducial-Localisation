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
from skullFindFiducial import *
import scipy.ndimage as nd
from dicom.contrib import pydicom_series

ConstPixelSpacing = (1.0, 1.0, 1.0)
def GenerateTransform(slice_datasets):
    image_orientation = slice_datasets.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    row_spacing, column_spacing = slice_datasets.PixelSpacing
    slice_spacing = slice_datasets.SliceThickness

    transform = np.identity(4, dtype=np.float32)

    transform[:3, 0] = row_cosine*column_spacing
    transform[:3, 1] = column_cosine*row_spacing
    transform[:3, 2] = slice_cosine*slice_spacing

    transform[:3, 3] = slice_datasets.ImagePositionPatient
    return transform

def _extract_cosines(image_orientation):
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine

# def remove_keymap_conflicts(new_keys_set):
#     for prop in plt.rcParams:
#         if prop.startswith('keymap.'):
#             keys = plt.rcParams[prop]
#             remove_list = set(keys) & new_keys_set
#             for key in remove_list:
#                 keys.remove(key)
                
# def multi_slice_viewer(volume):
#     remove_keymap_conflicts({'j', 'k'})
#     fig, ax = plt.subplots()
#     ax.volume = volume
#     ax.index = volume.shape[0] // 2
#     print(volume.shape)
#     ax.imshow(volume[ax.index],cmap = plt.get_cmap('gray'))
#     fig.canvas.mpl_connect('key_press_event', process_key)

# def process_key(event):
#     fig = event.canvas.figure
#     ax = fig.axes[0]
#     if event.key == 'j':
#         previous_slice(ax)
#     elif event.key == 'k':
#         next_slice(ax)
#     fig.canvas.draw()

# def previous_slice(ax):
#     volume = ax.volume
#     ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
#     ax.images[0].set_array(volume[ax.index])

# def next_slice(ax):
#     volume = ax.volume
#     ax.index = (ax.index + 1) % volume.shape[0]
#     ax.images[0].set_array(volume[ax.index])


# def makeCompatible(dicomData, prec=5):
#     for i in range(len(dicomData)):
#         a = dicomData[i].ImageOrientationPatient
#         #print dicomData[i].pixel_array
#         a[0] = round(a[0], prec)
#         a[1] = round(a[1], prec)
#         a[2] = round(a[2], prec)
#         a[3] = round(a[3], prec)
#         dicomData[i].ImageOrientationPatient = a



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
        a[0] = round(a[0], prec)
        a[1] = round(a[1], prec)
        a[2] = round(a[2], prec)
        a[3] = round(a[3], prec)
        dicomData[i].ImageOrientationPatient = a


def get3DRecon(data, path):
    """
    Performs 3D reconstruction from the given DICOM data, and
    returns a voxel array and the pixel spacing factors.
    """
    global ConstPixelSpacing
    # del(data[0])
    # RefDs = data[0]

    # ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(
        # RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    series = pydicom_series.read_files(path, False, True) # second to not show progress bar, third to retrieve data
    # print len(series)

    seiesNum = 1
    voxel_ndarray = series[seiesNum].get_pixel_array()
    print voxel_ndarray.shape

    info = series[seiesNum].info
    ijk_to_xyz = GenerateTransform(info)
    print(ijk_to_xyz)
    # _, ijk_to_xyz = dicom_numpy.combine_slices(data)
    # print(ijk_to_xyz)
    sliceSpacing = abs(data[0].ImagePositionPatient[2] - data[1].ImagePositionPatient[2])
    ConstPixelSpacing = [sliceSpacing, data[1].PixelSpacing[0], data[1].PixelSpacing[1]]

    # try:
    #     voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
    # except dicom_numpy.DicomImportException as e:
    #     # Invalid DICOM data
    #     print("Handling incompatible dicom slices")
    #     # voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
    #     try:
    #         # makeCompatible(data, prec=5)
    #         voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
    #     except:
    #         print("Handling incompatible dicom slices")
    #         voxel_ndarray = []
    #         for i in range(len(data)):
    #             # print data[i].pixel_array.shape
    #             # assert False, "Stop Code"
    #             voxel_ndarray.append(data[i].pixel_array)
    #         voxel_ndarray = np.stack(voxel_ndarray, axis=2)
    #         # multi_slice_viewer(voxel_ndarray)
    #         # plt.show()
    #         ijk_to_xyz = np.eye(4)

    sliceSpacing = abs(data[0].ImagePositionPatient[2] - data[1].ImagePositionPatient[2])
    ConstPixelSpacing = [sliceSpacing, data[0].PixelSpacing[0], data[0].PixelSpacing[1]]

    
    return voxel_ndarray, ConstPixelSpacing, ijk_to_xyz


def applyThreshold(voxelData):
    """
    Thresholding for the bone value in a real scan (Hounsfield unit).
    Bone is from 700 to 3000.
    """
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
