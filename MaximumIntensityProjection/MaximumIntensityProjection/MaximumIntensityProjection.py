#!/usr/bin/env python
import argparse
import os.path
import vtk

def CreateRenderer(renWin):
    renderer = vtk.vtkRenderer()
    renWin.AddRenderer(renderer)
    return renderer

class DataSource:
    def __init__(self, args):
        sourceDirName = args.sourceDirName

        self.originalVolumeReader = self.CreateMetaImageReader(os.path.join(sourceDirName, args.originalVolumeFileName))
        self.groundTruthPolyDataReader = self.CreatePolyDataReader(os.path.join(sourceDirName, args.groundTruthPolyDataFileName))

        self.objectnessMeasureVolumeReader = self.CreateMetaImageReader(os.path.join(sourceDirName, args.objectnessMeasureVolumeFileName))
        self.objectnessMeasureVolumePolyDataReader = self.CreatePolyDataReader(os.path.join(sourceDirName, args.objectnessMeasureVolumePolyDataFileName))

        self.nonMaximumSuppressionVolumeReader = self.CreateMetaImageReader(os.path.join(sourceDirName, args.nonMaximumSuppressionVolumeFileName))
        self.nonMaximumSuppressionVolumePolyDataReader = self.CreatePolyDataReader(os.path.join(sourceDirName, args.nonMaximumSuppressionVolumePolyDataFileName))

        self.nonMaximumSuppressionCurvVolumePolyDataReader = self.CreatePolyDataReader(os.path.join(sourceDirName, args.nonMaximumSuppressionCurvVolumePolyDataFileName))

        self.resultPolyDataReader = self.CreatePolyDataReader(os.path.join(sourceDirName, args.resultPolyDataFileName))
        
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

    def GetGroundTruthTreePolyDataReader(self):
        return self.groundTruthPolyDataReader

    def GetObjectnessMeasureVolumeReader(self):
        return self.objectnessMeasureVolumeReader

    def GetObjectnessMeasureVolumePolyDataReader(self):
        return self.objectnessMeasureVolumePolyDataReader

    def GetNonMaximumSuppressionVolumeReader(self):
        return self.nonMaximumSuppressionVolumeReader

    def GetNonMaximumSuppressionVolumePolyDataReader(self):
        return self.nonMaximumSuppressionVolumePolyDataReader

    def GetNonMaximumSuppressionCurvVolumePolyDataReader(self):
        return self.nonMaximumSuppressionCurvVolumePolyDataReader

    def GetResultPolyDataReader(self):
        return self.resultPolyDataReader

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
        volumeMapper.SetInterpolationModeToLinear()

        return volumeMapper

    def CreatePolyDataMapper(self, polyDataProvider):
        polyDataMapper = vtk.vtkPolyDataMapper()
        polyDataMapper.SetInputConnection(polyDataProvider.GetOutputPort())
        polyDataMapper.SetScalarModeToUseCellData()
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
        volume = self.CreateOriginalVolume(volumeReader)
        return self.CreateVolumeActor(self.CreateVolumeRayCastMapper(volume), self.CreateVolumeProperty())

    def CreateGroundTruthTreePolyDataActor(self):
        polyDataReader = self.src.GetGroundTruthTreePolyDataReader()
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(1 / 3., 1, 0.5)
        actor.GetProperty().SetOpacity(0.8)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

    def CreateObjectnessMeasureVolumeActor(self):
        volumeReader = self.src.GetObjectnessMeasureVolumeReader()
        volume = self.CreateObjectnessMeasureVolume(volumeReader)
        return self.CreateVolumeActor(self.CreateVolumeRayCastMapper(volume), self.CreateVolumeProperty())

    def CreateObjectnessMeasureVolumePolyDataActor(self):
        polyDataReader = self.src.GetObjectnessMeasureVolumePolyDataReader()
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(1, 1, 0.5)
        actor.GetProperty().SetOpacity(0.8)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

    def CreateNonMaximumSuppressionVolumeActor(self):
        volumeReader = self.src.GetNonMaximumSuppressionVolumeReader()
        volume = self.CreateObjectnessMeasureVolume(volumeReader)
        return self.CreateVolumeActor(self.CreateVolumeRayCastMapper(volume), self.CreateVolumeProperty())

    def CreateNonMaximumSuppressionVolumePolyDataActor(self):
        polyDataReader = self.src.GetNonMaximumSuppressionVolumePolyDataReader()
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(1 / 3., 2 / 3., 1)
        actor.GetProperty().SetOpacity(0.8)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

    def CreateNonMaximumSuppressionCurvVolumePolyDataActor(self):
        polyDataReader = self.src.GetNonMaximumSuppressionCurvVolumePolyDataReader()
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(1, 2 / 3., 1)
        actor.GetProperty().SetOpacity(0.8)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

    def CreateResultPolyDataActor(self):
        polyDataReader = self.src.GetResultPolyDataReader()
        actor = vtk.vtkActor()
        actor.GetProperty().SetColor(1, 1, 1)
        actor.GetProperty().SetOpacity(1)
        actor.SetMapper(self.CreatePolyDataMapper(polyDataReader))
        return actor

class Application:
    def __init__(self):
        renWin = vtk.vtkRenderWindow()
        renWin.LineSmoothingOn()
        renWin.SetSize(1280, 720)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        self.iren = iren
        self.renWin = renWin
    
    def RenderOutputToImage(self):
        w2i = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkPNGWriter()

        w2i.SetInput(self.renWin)
        w2i.Update()
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.SetFileName('image.png')
        writer.Write()

    def OnKeyPress(self, obj, e):
        keySym = obj.GetKeySym()
        renWin = self.renWin

        def TogglePolyDataActorVisibility(polyDataActor):
            polyDataActor.SetVisibility(not polyDataActor.GetVisibility())
            renWin.Render()

        if '1' == keySym:
            TogglePolyDataActorVisibility(self.groundTruthTreePolyDataActor)
        elif '2' == keySym:
            TogglePolyDataActorVisibility(self.objectnessMeasureVolumePolyDataActor)
        elif '3' == keySym:
            TogglePolyDataActorVisibility(self.nonMaximumSuppressionVolumePolyDataActor)
        elif '4' == keySym:
            TogglePolyDataActorVisibility(self.nonMaximumSuppressionCurvVolumePolyDataActor)
        elif 'S' == keySym:
            self.RenderOutputToImage()

    def Run(self, args):
        renWin = self.renWin
        iren = self.iren

        src = DataSource(args)
        

        pm = PresentationModel()
        pm.SetDataSource(src)

        originalVolumeActor = pm.CreateOriginalVolumeActor()

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


        self.originalVolumeRenderer = originalVolumeRenderer

        self.objectnessMeasureVolumeActor = pm.CreateObjectnessMeasureVolumeActor()
        self.objectnessMeasureVolumePolyDataActor = pm.CreateObjectnessMeasureVolumePolyDataActor()
        self.nonMaximumSuppressionVolumePolyDataActor = pm.CreateNonMaximumSuppressionVolumePolyDataActor()
        self.nonMaximumSuppressionCurvVolumePolyDataActor = pm.CreateNonMaximumSuppressionCurvVolumePolyDataActor()
        self.groundTruthTreePolyDataActor = pm.CreateGroundTruthTreePolyDataActor()
        self.resultPolyDataActor = pm.CreateResultPolyDataActor()

        self.groundTruthTreePolyDataActor.VisibilityOff()
        self.objectnessMeasureVolumePolyDataActor.VisibilityOff()
        self.nonMaximumSuppressionVolumePolyDataActor.VisibilityOff()
        self.nonMaximumSuppressionCurvVolumePolyDataActor.VisibilityOff()

        originalVolumeRenderer.AddVolume(originalVolumeActor)
        originalVolumeRenderer.AddActor(self.groundTruthTreePolyDataActor)
        originalVolumeRenderer.AddActor(self.objectnessMeasureVolumePolyDataActor)
        originalVolumeRenderer.AddActor(self.nonMaximumSuppressionVolumePolyDataActor)
        originalVolumeRenderer.AddActor(self.nonMaximumSuppressionCurvVolumePolyDataActor)

        originalVolumeRenderer.AddActor(self.objectnessMeasureVolumePolyDataActor)

        objectnessMeasureVolumeRenderer = CreateRenderer(renWin)
        objectnessMeasureVolumeRenderer.AddVolume(self.objectnessMeasureVolumeActor)

        objectnessMeasureVolumeRenderer.AddActor(self.groundTruthTreePolyDataActor)
        objectnessMeasureVolumeRenderer.AddActor(self.objectnessMeasureVolumePolyDataActor)
        objectnessMeasureVolumeRenderer.AddActor(self.nonMaximumSuppressionVolumePolyDataActor)
        objectnessMeasureVolumeRenderer.AddActor(self.nonMaximumSuppressionCurvVolumePolyDataActor)

        objectnessMeasureVolumeRenderer.SetViewport(viewports[1])

        self.objectnessMeasureVolumeRenderer = objectnessMeasureVolumeRenderer

        self.nonMaximumSuppressionVolumeActor = pm.CreateNonMaximumSuppressionVolumeActor()

        nonMaximumSuppressionVolumeRenderer = CreateRenderer(renWin)

        self.nonMaximumSuppressionVolumeRenderer = nonMaximumSuppressionVolumeRenderer

        nonMaximumSuppressionVolumeRenderer.AddVolume(self.nonMaximumSuppressionVolumeActor)

        nonMaximumSuppressionVolumeRenderer.AddActor(self.groundTruthTreePolyDataActor)
        nonMaximumSuppressionVolumeRenderer.AddActor(self.objectnessMeasureVolumePolyDataActor)
        nonMaximumSuppressionVolumeRenderer.AddActor(self.nonMaximumSuppressionVolumePolyDataActor)
        nonMaximumSuppressionVolumeRenderer.AddActor(self.nonMaximumSuppressionCurvVolumePolyDataActor)
        nonMaximumSuppressionVolumeRenderer.SetViewport(viewports[2])

        resultRenderer = CreateRenderer(renWin)

        resultRenderer.AddActor(self.resultPolyDataActor)
        resultRenderer.AddActor(self.groundTruthTreePolyDataActor)
        resultRenderer.AddActor(self.objectnessMeasureVolumePolyDataActor)
        resultRenderer.AddActor(self.nonMaximumSuppressionVolumePolyDataActor)
        resultRenderer.AddActor(self.nonMaximumSuppressionCurvVolumePolyDataActor)
        resultRenderer.SetViewport(viewports[3])

        objectnessMeasureVolumeRenderer.SetActiveCamera(originalVolumeRenderer.GetActiveCamera())
        nonMaximumSuppressionVolumeRenderer.SetActiveCamera(originalVolumeRenderer.GetActiveCamera())
        resultRenderer.SetActiveCamera(originalVolumeRenderer.GetActiveCamera())

        camera = originalVolumeRenderer.GetActiveCamera()
        c = originalVolumeActor.GetCenter()
    
        camera.ParallelProjectionOff()

        camera.SetFocalPoint(c[0], c[1], c[2])
        camera.SetPosition(c[0], c[1], c[2] + 10) 
        camera.SetViewAngle(30)
        camera.SetViewUp(0, 1, 0)

        style = vtk.vtkInteractorStyleTrackballCamera()
        style.SetCurrentRenderer(originalVolumeRenderer)
        style.SetMotionFactor(5)
        iren.SetInteractorStyle(style)

        iren.AddObserver('KeyPressEvent', self.OnKeyPress)

        iren.Initialize()
        renWin.Render()
        iren.Start()

if __name__ == '__main__':
    app = Application()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('sourceDirName')
    argparser.add_argument('--originalVolumeFileName', default='original_image.mhd')
    argparser.add_argument('--groundTruthPolyDataFileName', default='tree_structure.vtp')
    argparser.add_argument('--objectnessMeasureVolumeFileName', default='ObjectnessMeasureVolume.mhd')
    argparser.add_argument('--objectnessMeasureVolumePolyDataFileName', default='ObjectnessMeasureVolumeTangents.vtp')
    argparser.add_argument('--nonMaximumSuppressionVolumeFileName', default='NonMaximumSuppressionVolume.mhd')
    argparser.add_argument('--nonMaximumSuppressionVolumePolyDataFileName', default='NonMaximumSuppressionVolumeTangents.vtp')
    argparser.add_argument('--nonMaximumSuppressionCurvVolumePolyDataFileName', default='1.95/NonMaximumSuppressionCurvVolumeTangents.vtp')
    argparser.add_argument('--resultPolyDataFileName', default='1.95/NonMaximumSuppressionCurvVolumeEMST.vtp')
    argparser.set_defaults(func=app.Run)

    args = argparser.parse_args()
    args.func(args)


