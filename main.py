# to run this code, install dicom_numpy, pydicom, natsort(on pip),skimage and mayavi

from skullFindSurface import getSurfaceVoxels
from skullReconstruct import get3DRecon, readDicomData, applyThreshold
import time
from skimage import measure
from mayavi import mlab


def main():
    start_time = time.time()

    PathDicom = "/Users/Parth/Inter_IIT_Tech/2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE2"

    data = readDicomData(PathDicom)
    print("---- %s seconds -----" % (time.time() - start_time))

    voxelData, ConstPixelSpacing = get3DRecon(data)
    #voxelDataThresh = applyThreshold(voxelData)
    print("---- %s seconds -----" % (time.time() - start_time))

    # surfaceVoxels = getSurfaceVoxels(voxelDataThresh)
    print("---- %s seconds -----" % (time.time() - start_time))

    verts, faces, normals, values = measure.marching_cubes_lewiner(voxelData, 0, ConstPixelSpacing)
    print("---- %s seconds -----" % (time.time() - start_time))

    mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces)
    mlab.quiver3d(verts[::100,0], verts[::100,1], verts[::100,2],
               normals[::100,0], normals[::100,1], normals[::100,2])

    print("---- %s seconds -----" % (time.time() - start_time))
    
    mlab.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(surfaceVoxels[:, 0], surfaceVoxels[:, 1], surfaceVoxels[:, 2])
    # print(time.asctime(time.localtime(time.time())))
    # plt.show()


if __name__ == '__main__':
    main()
