# -*- coding: utf-8 -*-
"""
Autonomous localization of fiducial markers for IGNS.
This script runs the routine for the autonomous
detection and localization of fiducial markers affixed
on the skull, using a DICOM scan as input.
Package dependencies: dicom_numpy, pydicom, natsort,
scipy, sklearn, skimage and mayavi.

Authors: P. Khirwadkar, H. Loya, D. Shah, R. Chaudhry,
A. Ghosh & S. Goel (For Inter IIT Technical Meet 2018)
Copyright © 2018 Indian Institute of Technology, Bombay
"""

from skullReconstruct import *
from skullNormalExtraction import *
from skullFindFiducial import *
import time
from mayavi import mlab
import copy

ConstPixelSpacing = (1.0, 1.0, 1.0)

start_time = time.time()

PathDicom = "/home/j_69/Fiducial Localization - MRI Scans/pvc/Sequential Scan/DICOM/PA1/ST1/SE4"

data = readDicomData(PathDicom)
voxelData, ConstPixelSpacing, ijk_to_xyz = get3DRecon(data)
print("Constant Pixel Spacing: " + str(ConstPixelSpacing))
voxelData, ConstPixelSpacing = interpolate_image(
    voxelData, (1, 1, 3))  # interpolating the image
voxelDataThresh = applyThreshold(copy.deepcopy(voxelData))
print(ConstPixelSpacing)
print("---- %s seconds ----- Extracted %s Slices!" %
      (time.time() - start_time, str(voxelData.shape)))

surfaceVoxels = getSurfaceVoxels(voxelDataThresh)
# surfaceVoxels has i,j,k indices
print("---- %s seconds ----- Extracted Surface Voxels!" %
      (time.time() - start_time))

normals, surfaceVoxelCoord, verts, faces = findSurfaceNormals(copy.deepcopy(
    surfaceVoxels), voxelData, ConstPixelSpacing)
# surfaceVoxelCoord = surfaceVoxel*ConstPixelSpacing

print("---- %s seconds ----- Extracted %s Surface Normals!" %
      (time.time() - start_time, len(surfaceVoxelCoord)))

sampling_factor = 20
normals_sample = normals[::sampling_factor]
surfaceVoxelCoord_sample = surfaceVoxelCoord[::sampling_factor]
# surfaceVoxelCoord_sample = sub-sampled version of surfaceVoxelCoord

surfaceVoxelCoord_sample = np.uint16(np.float64(
    surfaceVoxelCoord_sample) / ConstPixelSpacing)

print("---- %s seconds ----- Sampled %s Voxels!" %
      (time.time() - start_time, len(surfaceVoxelCoord_sample)))
costs, patches = checkFiducial(surfaceVoxelCoord,
                               surfaceVoxelCoord_sample, normals, ConstPixelSpacing)

print("---- %s seconds ----- Finished comparing with Fiducial Model!" %
      (time.time() - start_time))

# Visualise in Mayavi
visualiseFiducials(costs, patches, surfaceVoxelCoord_sample, surfaceVoxelCoord, verts, faces, ijk_to_xyz, normals)

# for i in range(num_markers):
# 	patch = patches[indices[i]]
# 	mlab.points3d(patch[:, 0], patch[:, 1], patch[
# 	                  :, 2], color=tuple(colormap[labels[i]]))
