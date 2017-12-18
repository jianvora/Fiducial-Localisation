import dicom
import vtk
import numpy as np
import time
start_time = time.time()

# Path to the folder containing dicom images
PathDicom = "/Users/Parth/Inter_IIT_Tech/2016.06.27 PVC Skull Model/Sequential Scan/DICOM/PA1/ST1/SE2"


reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(PathDicom)
reader.Update()

_extent = reader.GetDataExtent()
ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
ConstPixelSpacing = reader.GetPixelSpacing()

threshold = vtk.vtkImageThreshold()
threshold.SetInputConnection(reader.GetOutputPort())
threshold.ThresholdByLower(0)
threshold.ReplaceInOn()
threshold.SetInValue(0)  # set all values below 0 to 0
threshold.ReplaceOutOn()
threshold.SetOutValue(1)  # set all values above 0 to 1
threshold.Update()

dmc = vtk.vtkDiscreteMarchingCubes()
dmc.SetInputConnection(threshold.GetOutputPort())
dmc.GenerateValues(1, 1, 1)
dmc.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(dmc.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1.0, 1.0, 1.0)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

renWin.SetSize(200, 200)

# This allows the interactor to initalize itself. It has to be
# called before an event loop.
iren.Initialize()

# We'll zoom in a little by accessing the camera and invoking a "Zoom"
# method on it.
renderer.ResetCamera()
renderer.GetActiveCamera().Zoom(1.5)
renWin.Render()
print("---- %s seconds -----"%(time.time() - start_time))
# Start the event loop.
iren.Start()