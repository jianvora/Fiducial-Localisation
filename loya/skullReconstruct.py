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

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
                
def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    print(volume.shape)
    ax.imshow(volume[ax.index],cmap = plt.get_cmap('gray'))
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def makeCompatible(dicomData, prec=5):
    for i in range(len(dicomData)):
        a = dicomData[i].ImageOrientationPatient
        #print dicomData[i].pixel_array
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
        try:
            voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(data)
        except:
            voxel_ndarray = []
            for i in range(len(data)):
                voxel_ndarray.append(data[i].pixel_array)
            voxel_ndarray = np.array(voxel_ndarray)
            multi_slice_viewer(voxel_ndarray)
            plt.show()

            ijk_to_xyz = np.eye(4)
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

def image_zoom(A,zoomfactor):
    ### zoom and filter
    ## resample A, based on B's resolution
    global ConstPixelSpacing
    #ConstPixelSpacing = tuple([y/zoomfactor for y in ConstPixelSpacing])
    A = copy.deepcopy(A)
    PixelSpacing = []
    for i in range(3):
        PixelSpacing.append(ConstPixelSpacing[i]/zoomfactor[i])
    ConstPixelSpacing = tuple(PixelSpacing)
    print("Zooming image by " + str(zoomfactor))
    Atrans = nd.interpolation.zoom(A, zoom=zoomfactor)
    Atrans = np.array(Atrans,dtype='float32')


    return Atrans,ConstPixelSpacing


