# to run this code, install dicom_numpy, pydicom, natsort(on pip),skimage and mayavi

from skullFindSurface import *
from skullReconstruct import *
from skullNormalExtraction import *
import time
from mayavi import mlab
import copy
# # import matplotlib
# # matplotlib.use("TkAgg")
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt


def main():
    start_time = time.time()

    # change to required path to dicom folder
    PathDicom = "../../2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE2"

    data = readDicomData(PathDicom)
    print("---- %s seconds -----" % (time.time() - start_time))

    voxelData, ConstPixelSpacing = get3DRecon(data)
    voxelDataThresh = applyThreshold(copy.deepcopy(voxelData))

    print("---- %s seconds -----" % (time.time() - start_time))

    surfaceVoxels = getSurfaceVoxels(voxelDataThresh)
    # only plotting 1/10th of points to speed up rendering
    # the points are represented as spheres hence skull surface may look bubbly
    mlab.points3d(surfaceVoxels[::10, 0] * ConstPixelSpacing[0], surfaceVoxels[::10, 1]
                  * ConstPixelSpacing[1], surfaceVoxels[::10, 2] * ConstPixelSpacing[2])
    # mlab.plot3d(np.expand_dims(surfaceVoxels[:,0],axis=1),np.expand_dims(surfaceVoxels[:,1],axis=1),np.expand_dims(surfaceVoxels[:,2],axis=1))

    print("---- %s seconds -----" % (time.time() - start_time))

    normals = findSurfaceNormals(copy.deepcopy(
        surfaceVoxels), voxelData, ConstPixelSpacing)

    print("---- %s seconds -----" % (time.time() - start_time))

    normals = np.squeeze(normals)
    mlab.quiver3d(surfaceVoxels[::100, 0] * ConstPixelSpacing[0], surfaceVoxels[::100, 1] * ConstPixelSpacing[1], surfaceVoxels[::100, 2] * ConstPixelSpacing[2],
                  normals[::100, 0], normals[::100, 1], normals[::100, 2])

    # ------ uncomment this to view surface mesh ----------
    # vert,_, faces = getSurfaceMesh(voxelData, ConstPixelSpacing)
    # mlab.triangular_mesh(vert[:,0],vert[:,1],vert[:,2],faces)
    # -----------------------------------------------------
    print("---- %s seconds -----" % (time.time() - start_time))

    mlab.show()
    # -------- uncomment this to view in matplotlib -----------
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(surfaceVoxels[:, 0], surfaceVoxels[:, 1], surfaceVoxels[:, 2])
    # print(time.asctime(time.localtime(time.time())))
    # plt.show()


if __name__ == '__main__':
    main()
