import dicom_numpy
import dicom
import os
from natsort import natsorted
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools

from mayavi import mlab
import time
from skimage import measure

def readDicomData(path):
	lstFilesDCM = []
	# may want to exclude the first dicom image in some files
	for root, directory, fileList in os.walk(path):
		for filename in fileList:
			if filename==".DS_Store":
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

def get3DRecon(data):
	RefDs = data[0]
	ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

	try:
	    voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
	except dicom_numpy.DicomImportException as e:
	    # invalid DICOM data
	    raise NameError('Unable to do 3D reconstruction. Slice missing? or incompatible slice data?')
	# arbitrary for now, can be set to different values for CT scan as in Hounsfield unit,
	# bone is from +700 to +3000
	upper_thresh = 0
	lower_thresh = 0

	voxel_ndarray[voxel_ndarray > upper_thresh] = 1
	voxel_ndarray[voxel_ndarray <= lower_thresh] = 0
	return voxel_ndarray
