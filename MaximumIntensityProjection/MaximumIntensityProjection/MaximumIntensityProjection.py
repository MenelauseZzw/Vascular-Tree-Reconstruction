#!/usr/bin/env python
import argparse
import os.path
import vtk

def CreateRenderer(renWin):
    renderer = vtk.vtkRenderer()
    renWin.AddRenderer(renderer)
    return renderer

class DataSource:
    def __init__(self, sourceDirName):
        self.originalVolumeReader                  = self.CreateMetaImageReader(os.path.join(sourceDirName, "original_image.mhd"))
        self.objectnessMeasureVolumeReader         = self.CreateMetaImageReader(os.path.join(sourceDirName, "ObjectnessMeasureVolume.mhd"))
        self.objectnessMeasureVolumeTangentsReader = self.CreatePolyDataReader(os.path.join(sourceDirName, "ObjectnessMeasureVolumeTangents.vtp"))
        self.nonMaximumSuppressionVolumeReader = self.CreateMetaImageReader(os.path.join(sourceDirName, "NonMaximumSuppressionVolume.mhd"))
        #self.groundTruthTreeReader             = self.CreatePolyDataReader(os.path.join(sourceDirName, "GroundTruthTree.vtp"))
        #self.tangentsReader                    = self.CreatePolyDataReader(os.path.join(sourceDirName, "NonMaximumSuppressionVolumeTangents.vtp"))
        self.tangentsCurvReader                = self.CreatePolyDataReader(os.path.join(sourceDirName, "1.95/NonMaximumSuppressionCurvVolumeEMST.vtp"))
        #self.resultTreeReader                  = self.CreatePolyDataReader(os.path.join(sourceDirName, "NonMaximumSuppressionCurvVolumeEMST.vtp"))

    def CreatePolyDataReader(self, polyDataFileName):
        polyDataReader = vtk.vtkXMLPolyDataReader()
        polyDataReader.SetFileName(polyDataFileName)
        return polyDataReader

    def CreateMetaImageReader(self, imageFileName):
        metaImageReader = vtk.vtkMetaImageReader()
        metaImageReader.SetFileName(imageFileName)
        return metaImageReader

    def GetOriginalVolumeReader(self): 
        return self.originalVolumeReader

    def GetObjectnessMeasureVolumeReader(self):
        return self.objectnessMeasureVolumeReader
   
    def GetObjectnessMeasureVolumeTangentsReader(self):
        return self.objectnessMeasureVolumeTangentsReader

    def GetNonMaximumSuppressionVolumeReader(self):
        return self.nonMaximumSuppressionVolumeReader

    def GetGroundTruthTreeReader(self):
        return self.groundTruthTreeReader

    def GetTangentsReader(self):
        return self.tangentsReader

    def GetTangentsCurvReader(self):
        return self.tangentsCurvReader

    def GetResultTreeReader(self):
        return self.resultTreeReader

class PresentationModel:
    def GetDataSource(self):
        return self.src

    def SetDataSource(self, src):
        self.src = src

    def CreateVolumeRayCastMapper(self, volumeSrc):
	volumeMapper = vtk.vtkSmartVolumeMapper()
	volumeMapper.SetInputConnection(volumeSrc.GetOutputPort())
	volumeMapper.SetRequestedRenderModeToRayCast()	
	volumeMapper.SetBlendModeToMaximumIntensity()
	volumeMapper.SetInterpolationModeToLinear ()

        return volumeMapper

    def CreatePolyDataMapper(self, polyDataProvider):
        polyDataMapper = vtk.vtkPolyDataMapper()
        polyDataMapper.SetInputConnection(polyDataProvider.GetOutputPort())
        return polyDataMapper

    def CreateOriginalVolume(self, volumeReader):
        volumeReader.Update()

        valueMin,valueMax = volumeReader.GetOutput().GetPointData().GetArray('MetaImage').GetValueRange()


        toUnsignedShort = vtk.vtkImageShiftScale()
        toUnsignedShort.ClampOverflowOn()
        toUnsignedShort.SetShift(-valueMin)
        toUnsignedShort.SetScale(512.0 / (valueMax - valueMin))
        toUnsignedShort.SetInputConnection(volumeReader.GetOutputPort())
        toUnsignedShort.SetOutputScalarTypeToUnsignedShort()

        return toUnsignedShort

    def CreateObjectnessMeasureVolume(self, volumeReader):
        volumeReader.Update()

        objectnessMeasure = vtk.vtkImageExtractComponents()
        objectnessMeasure.SetInputConnection(volumeReader.GetOutputPort())
        objectnessMeasure.SetComponents(0)
        objectnessMeasure.Update()

        valueMin,valueMax = objectnessMeasure.GetOutput().GetPointData().GetArray('MetaImage').GetValueRange()
        valueMin = 0.0

        toUnsignedShort = vtk.vtkImageShiftScale()
        toUnsignedShort.ClampOverflowOn()
        toUnsignedShort.SetShift(-valueMin)
        toUnsignedShort.SetScale(512.0 / (valueMax - valueMin))
        toUnsignedShort.SetInputConnection(objectnessMeasure.GetOutputPort())
        toUnsignedShort.SetOutputScalarTypeToUnsignedShort()

        return toUnsignedShort

    def CreateVolumeProperty(self):
        volumeColor = vtk.vtkColorTransferFunction()
        volumeColor.AddRGBPoint(0,      0.0, 0.0, 0.0)
        volumeColor.AddRGBPoint(71.78,  0.25,0.25,0.25)
        volumeColor.AddRGBPoint(143.56, 0.5, 0.5, 0.5)
        volumeColor.AddRGBPoint(215.34, 0.75,0.75,0.75)
        volumeColor.AddRGBPoint(286.00, 1.0, 1.0, 1.0)
        volumeColor.AddRGBPoint(512,    1.0, 1.0, 1.0)

        volumeScalarOpacity = vtk.vtkPiecewiseFunction()
        volumeScalarOpacity.AddPoint(0,     0.0)
        volumeScalarOpacity.AddPoint(512, 1.0)

        volumeGradientOpacity = vtk.vtkPiecewiseFunction()
        volumeGradientOpacity.AddPoint(0,   1.0)
        volumeGradientOpacity.AddPoint(512, 1.0)

        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(volumeColor)
        volumeProperty.SetScalarOpacity(volumeScalarOpacity)
        volumeProperty.SetGradientOpacity(volumeGradientOpacity)
        volumeProperty.SetInterpolationTypeToLinear()
        volumeProperty.ShadeOn()

        volumeProperty.SetAmbient(0.30)
        volumeProperty.SetDiffuse(0.60)
        volumeProperty.SetSpecular(0.20)

        return volumeProperty

    def CreateVolumeActor(self, volumeMapper, volumeProperty):
        actor = vtk.vtkVolume()
        actor.SetMapper(volumeMapper)
        actor.SetProperty(volumeProperty)
        return actor

    def CreateOriginalVolumeActor(self):
        volumeReader = self.src.GetOriginalVolumeReader()
        volume       = self.CreateOriginalVolume(volumeReader)
        return self.CreateVolumeActor(self.CreateVolumeRayCastMapper(volume), self.CreateVolumeProperty())

    def CreateObjectnessMeasureVolumeActor(self):
        volumeReader = self.src.GetObjectnessMeasureVolumeReader()
        volume       = self.CreateObjectnessMeasureVolume(volumeReader)
        return self.CreateVolumeActor(self.CreateVolumeRayCastMapper(volume), self.CreateVolumeProperty())

    def CreateNonMaximumSuppressionVolumeActor(self):
        volumeReader = self.src.GetNonMaximumSuppressionVolumeReader()
        volume       = self.CreateObjectnessMeasureVolume(volumeReader)
        return self.CreateVolumeActor(self.CreateVolumeRayCastMapper(volume), self.CreateVolumeProperty())

    def CreateGroundTruthTreeActor(self):
        polyDataReader = self.src.GetGroundTruthTreeReader()
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(0, 1, 0)
        actor.GetProperty().SetLineWidth(0.5)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

    def CreatebjectnessMeasureVolumeTangentsActor(self):
        polyDataReader = self.src.GetObjectnessMeasureVolumeTangentsReader() 
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(1, 0, 0)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

    def CreateTangentsActor(self):
        polyDataReader = self.src.GetTangentsReader() 
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(1, 0, 0)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

    def CreateTangentsCurvActor(self):
        polyDataReader = self.src.GetTangentsCurvReader() 
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(85 / 255., 170 / 255., 255 / 255.)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

    def CreateResultTreeActor(self):
        polyDataReader = self.src.GetResultTreeReader()
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(1, 1, 0)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

def OnKeyPress(obj, e):
    keySym = obj.GetKeySym()
    if keySym == 's':
         print 's'
    return


def DoMaximumIntensityProjection(args):
    sourceDirName = args.sourceDirName

    renWin = vtk.vtkRenderWindow()
    #renWin.FullScreenOn()
    renWin.SetSize(1000, 800)
    renWin.LineSmoothingOn()

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    src = DataSource(sourceDirName)

    presModel = PresentationModel()
    presModel.SetDataSource(src)

    originalVolumeActor = presModel.CreateOriginalVolumeActor()

    numRows, numCols = 2, 2
    viewports = []

    for row in xrange(numRows - 1, -1, -1):
        ymin = float(row) / numRows
	ymax = (row + 1.0) / numRows

        for col in xrange(numCols):
            xmin = float(col) / numCols
            xmax = (col + 1.0) / numCols
            viewports.append((xmin,ymin,xmax,ymax))

    originalVolumeRenderer = CreateRenderer(renWin)
    originalVolumeRenderer.SetViewport(viewports[0])

    originalVolumeRenderer.AddVolume(originalVolumeActor)

    objectnessMeasureVolumeRenderer = CreateRenderer(renWin)
    objectnessMeasureVolumeRenderer.AddVolume(presModel.CreateObjectnessMeasureVolumeActor())
    #objectnessMeasureVolumeRenderer.AddActor(presModel.CreatebjectnessMeasureVolumeTangentsActor())
    objectnessMeasureVolumeRenderer.SetViewport(viewports[1])

    nonMaximumSuppressionVolumeRenderer = CreateRenderer(renWin)
    nonMaximumSuppressionVolumeRenderer.AddVolume(presModel.CreateNonMaximumSuppressionVolumeActor())
    nonMaximumSuppressionVolumeRenderer.SetViewport(viewports[2])

    #tangentsRenderer = CreateRenderer(renWin)
    #tangentsRenderer.AddActor(presModel.CreateTangentsActor())
    #tangentsRenderer.AddActor(presModel.CreateGroundTruthTreeActor())
    #tangentsRenderer.SetViewport((0, 0, 1 / 3., 0.5))

    #tangentsCurvRenderer = CreateRenderer(renWin)
    #tangentsCurvRenderer.AddActor(presModel.CreateTangentsCurvActor())
    #tangentsCurvRenderer.SetViewport(viewports[5])

    #resultTreeRenderer = CreateRenderer(renWin)
    #resultTreeRenderer.AddActor(presModel.CreateResultTreeActor())
    #resultTreeRenderer.AddActor(presModel.CreateGroundTruthTreeActor())
    #resultTreeRenderer.SetViewport((2 / 3., 0, 1, 0.5))

    objectnessMeasureVolumeRenderer.SetActiveCamera(originalVolumeRenderer.GetActiveCamera())
    nonMaximumSuppressionVolumeRenderer.SetActiveCamera(originalVolumeRenderer.GetActiveCamera())
    #tangentsRenderer.SetActiveCamera(originalVolumeRenderer.GetActiveCamera())
    #tangentsCurvRenderer.SetActiveCamera(originalVolumeRenderer.GetActiveCamera())
    #resultTreeRenderer.SetActiveCamera(originalVolumeRenderer.GetActiveCamera())

    camera =  originalVolumeRenderer.GetActiveCamera()
    c = originalVolumeActor.GetCenter()
    
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.SetPosition(c[0], c[1], c[2] + 10) 
    camera.SetViewUp(0, 1, 0)

    camera.SetViewAngle(30)
    #camera.SetParallelScale(647.113669754781)
    #camera.SetParallelProjection(0)

    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetCurrentRenderer(originalVolumeRenderer)
    style.SetMotionFactor(5)
    iren.SetInteractorStyle(style)

    iren.AddObserver('KeyPressEvent', OnKeyPress)

    iren.Initialize()
    renWin.Render()
    iren.Start()

if __name__ == '__main__':
    # create the top-level parser
    argparser = argparse.ArgumentParser()
    argparser.add_argument('sourceDirName')
    argparser.set_defaults(func=DoMaximumIntensityProjection)
    # parse the args and call whatever function was selected
    args = argparser.parse_args()
    args.func(args)


