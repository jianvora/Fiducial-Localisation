# to run this code, install dicom_numpy, pydicom, natsort(on pip),skimage and mayavi

from skullFindSurface import *
from skullReconstruct import *
from skullNormalExtraction import *
from skullFindFIducial import *
from visualiseTools import *
import time
from mayavi import mlab
import copy
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

ConstPixelSpacing = (1, 1, 1)


def main():
    global ConstPixelSpacing
    start_time = time.time()
    # change to required path to dicom folder
    PathDicom = "../../2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE2"

    data = readDicomData(PathDicom)
    print("---- %s seconds -----" % (time.time() - start_time))

    voxelData, ConstPixelSpacing = get3DRecon(data)
    voxelDataThresh = applyThreshold(copy.deepcopy(voxelData))

    print("---- %s seconds -----" % (time.time() - start_time))

    surfaceVoxels = getSurfaceVoxels(voxelDataThresh)
    surfaceVoxelCoord = surfaceVoxels * ConstPixelSpacing
    # only plotting 1/10th of points to speed up rendering
    # the points are represented as spheres hence skull surface may look bubbly
    print("---- %s seconds -----" % (time.time() - start_time))

    normals = findSurfaceNormals(copy.deepcopy(
        surfaceVoxels), voxelData, ConstPixelSpacing)

    # print normals.shape

    print("---- %s seconds -----" % (time.time() - start_time))

    # normals = np.squeeze(normals)
    # visualiseNormals(verts, normals, ConstPixelSpacing, res=100)
    # ------ uncomment this to view surface mesh ----------
    # vert, normals, faces = getSurfaceMesh(voxelData, ConstPixelSpacing)
    # mlab.triangular_mesh(vert[:, 0], vert[:, 1], vert[:, 2], faces)
    # print("---- %s seconds -----" % (time.time() - start_time))

    i = 20  # decrease this to sample more points
    neighbor = getNeighborVoxel(surfaceVoxelCoord, surfaceVoxelCoord[::i], r=6)
    # neighbor = getNeighborVoxel(vert, vert[::i], r=6)

    print("---- %s seconds -----" % (time.time() - start_time))

    checkFiducial(surfaceVoxelCoord,
                  surfaceVoxelCoord[::i], normals[::i], neighbor)
    # checkFiducial(vert, vert[::i], normals[::i], neighbor)

    print("---- %s seconds -----" % (time.time() - start_time))
    # mlab.points3d(surfaceVoxelCoord[:, 0], surfaceVoxelCoord[:, 1],
    #               surfaceVoxelCoord[:, 2])
    # -----------------------------------------------------
    # mlab.show()
    # -------- uncomment this to view in matplotlib -----------
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(surfaceVoxels[:, 0], surfaceVoxels[:, 1], surfaceVoxels[:, 2])
    # print(time.asctime(time.localtime(time.time())))
    # plt.show()


if __name__ == '__main__':
    main()
