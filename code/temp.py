import numpy as np
from scipy.spatial import cKDTree, distance
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift
from mayavi import mlab
from skullReconstruct import *
import matplotlib.pyplot as plt
import copy
import time


def visualiseFiducials(cost, patches, points, pointCloud, verts, faces, num_markers=100, show_skull=True, show_markers=True):
    """
    Collects the top __ fiducial markers, filters them using the Mean Shift algorithm,
    and then renders with the original 3D scan, on Mayavi for
    visualisation and verification.
    """
    indices = filterFiducials(cost, patches, points, num_markers)
    print len(indices)
    # cost_sorted = np.sort(cost)
    colormap = np.random.rand(100, 3)
    if show_skull:
        mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts], faces)
    if show_markers:
        # Plot the top __ markers!
        for i in range(len(indices)):
            patch = patches[indices[i]]
            mlab.points3d(patch[:, 0], patch[:, 1], patch[
                          :, 2], color=tuple(colormap[i]))
            
    mlab.show()