import datetime
import math
import vtk

class Application:
    def __init__(self):
        renWin = vtk.vtkRenderWindow()
        renWin.LineSmoothingOff()
        renWin.SetSize(1280, 720)

        iren = vtk.vtkRenderWindowInteractor()
        iren.AddObserver('KeyPressEvent', self.OnKeyPress)
        iren.SetRenderWindow(renWin)

        self.commandsByKeySym = dict()
        self.renWin = renWin
        self.iren = iren

    def CreateRenderer(self):
        renderer = vtk.vtkRenderer()
        self.renWin.AddRenderer(renderer)
        return renderer

    def OnKeyPress(self, obj, e):
        keySym = obj.GetKeySym()
        if keySym in self.commandsByKeySym:
            for command in self.commandsByKeySym[keySym]:
                command()
            self.iren.Render()
        elif keySym == 'space':
            self.SaveCommand()

    def SaveCommand(self):
        windowToImage = vtk.vtkWindowToImageFilter()
        windowToImage.SetInput(self.renWin)
        windowToImage.Update()

        now = datetime.datetime.now()
        imageFileName = now.strftime('IMG-%Y%m%d-%H%M%S.png')

        imageWriter = vtk.vtkPNGWriter()
        imageWriter.SetInputData(windowToImage.GetOutput())
        imageWriter.SetFileName(imageFileName)
        imageWriter.Write()

    def SetInteractorStyle(self, style):
        self.iren.SetInteractorStyle(style)

    def RegisterToggleVisibilityCommand(self, polyDataActor, keySym):
        def ToggleVisibility():
            polyDataActor.SetVisibility(not polyDataActor.GetVisibility())
            
        if keySym in self.commandsByKeySym:
            self.commandsByKeySym[keySym].append(ToggleVisibility)
        else:
            self.commandsByKeySym[keySym] = [ToggleVisibility]

    def Start(self):
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

class ApplicationBuilder:
    class Volume:
        def CreateVolumeRayCastMapper(self, volumeSrc):
            volumeMapper = vtk.vtkSmartVolumeMapper()

            volumeMapper.SetInputConnection(volumeSrc.GetOutputPort())
            volumeMapper.SetRequestedRenderModeToRayCast()
            volumeMapper.SetBlendModeToMaximumIntensity()
            volumeMapper.SetInterpolationModeToNearestNeighbor()

            return volumeMapper

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

            volumeProp = vtk.vtkVolumeProperty()
            volumeProp.SetColor(volumeColor)
            volumeProp.SetScalarOpacity(volumeScalarOpacity)
            volumeProp.SetGradientOpacity(volumeGradientOpacity)
            volumeProp.SetInterpolationTypeToLinear()
            volumeProp.ShadeOn()

            volumeProp.SetAmbient(0.30)
            volumeProp.SetDiffuse(0.60)
            volumeProp.SetSpecular(0.20)

            return volumeProp

        def CreateVolume(self):
            volumeSrc = self.CreateVolumeSource()

            volumeMapper = self.CreateVolumeRayCastMapper(volumeSrc)
            volumeProp = self.CreateVolumeProperty()

            volume = vtk.vtkVolume()
            volume.SetMapper(volumeMapper)
            volume.SetProperty(volumeProp)
            return volume

    class ScalarVolume(Volume):
        def __init__(self, imageFileName, row, column):
            self.ImageFileName = imageFileName
            self.Row = row
            self.Column = column

        def CreateVolumeSource(self):
            imageReader = vtk.vtkMetaImageReader()
            imageReader.SetFileName(self.ImageFileName)
            imageReader.Update()

            valueMin,valueMax = imageReader.GetOutput().GetPointData().GetArray('MetaImage').GetValueRange()

            toUnsignedShort = vtk.vtkImageShiftScale()
            toUnsignedShort.ClampOverflowOn()
            toUnsignedShort.SetShift(-valueMin)
            toUnsignedShort.SetScale(512.0 / (valueMax - valueMin))
            toUnsignedShort.SetInputConnection(imageReader.GetOutputPort())
            toUnsignedShort.SetOutputScalarTypeToUnsignedShort()

            return toUnsignedShort

    class VectorVolumeComp(Volume):
        def __init__(self, imageFileName, componentIndex, row, column):
            self.ImageFileName = imageFileName
            self.ComponentIndex = componentIndex
            self.Row = row
            self.Column = column

        def CreateVolumeSource(self):
            imageReader = vtk.vtkMetaImageReader()
            imageReader.SetFileName(self.ImageFileName)

            compSrc = vtk.vtkImageExtractComponents()
            compSrc.SetInputConnection(imageReader.GetOutputPort())
            compSrc.SetComponents(self.ComponentIndex)
            compSrc.Update()

            valueMin,valueMax = compSrc.GetOutput().GetPointData().GetArray('MetaImage').GetValueRange()
            valueMin = 0.0

            toUnsignedShort = vtk.vtkImageShiftScale()
            toUnsignedShort.ClampOverflowOn()
            toUnsignedShort.SetShift(-valueMin)
            if valueMax - valueMin != 0:
                toUnsignedShort.SetScale(512.0 / (valueMax - valueMin)) #randomly cause error that "zerodivision"
            toUnsignedShort.SetInputConnection(compSrc.GetOutputPort())
            toUnsignedShort.SetOutputScalarTypeToUnsignedShort()

            return toUnsignedShort

    class PolyDataFile:
        def __init__(self, polyDataFileName, keySym, red, green, blue):
            self.PolyDataFileName = polyDataFileName
            self.KeySym = keySym
            self.Red = red
            self.Green = green
            self.Blue = blue

        def CreatePolyDataSource(self):
            polyDataReader = vtk.vtkXMLPolyDataReader()
            polyDataReader.SetFileName(self.PolyDataFileName)
            return polyDataReader

        def CreatePolyDataMapper(self):
            polyDataSrc = self.CreatePolyDataSource()

            polyDataMapper = vtk.vtkPolyDataMapper()
            polyDataMapper.SetInputConnection(polyDataSrc.GetOutputPort())
            polyDataMapper.SetScalarModeToUseCellData()
            return polyDataMapper

        def CreatePolyDataActor(self):
            polyDataMapper = self.CreatePolyDataMapper()

            actor = vtk.vtkActor()
            actor.GetProperty().SetColor(self.Red / 255., self.Green / 255., self.Blue / 255.)
            actor.GetProperty().SetOpacity(1)
            actor.SetMapper(polyDataMapper)
            return actor

    def __init__(self):
        self.volumes = []
        self.polyDataFiles = []

    def AddScalarVolume(self, imageFileName, row, column):
        self.volumes.append(self.ScalarVolume(imageFileName, row, column))

    def AddVectorVolumeComp(self, imageFileName, componentIndex, row, column):
        self.volumes.append(self.VectorVolumeComp(imageFileName, componentIndex, row, column))

    def AddPolyDataFile(self, polyDataFileName, keySym, red, green, blue):
        self.polyDataFiles.append(self.PolyDataFile(polyDataFileName, keySym, red, green, blue))

    def CreateViewport(self, row, numRows, column, numColumns):
        xmin = float(column) / numColumns
        xmax = (column + 1.) / numColumns
        row = numRows - 1 - row
        ymin = float(row) / numRows
        ymax = (row + 1.) / numRows
        return (xmin,ymin,xmax,ymax)

    def BuildApp(self):
        maxRow = 0
        maxColumn = 0

        volumesByPosition = dict()

        for volume in self.volumes:
            if volume.Row > maxRow:
                maxRow = volume.Row
            if volume.Column > maxColumn:
                maxColumn = volume.Column

            pos = (volume.Row,volume.Column)
            if pos in volumesByPosition:
                volumesByPosition[pos].append(volume) #modified by Rex
            else:
                volumesByPosition[pos] = [volume]

        app = Application()

        numRows = maxRow + 1
        numColumns = maxColumn + 1

        firstRenderer = None

        for row in xrange(numRows):
            for column in xrange(numColumns):
                print(row,column)
                renderer = app.CreateRenderer()
                viewport = self.CreateViewport(row, numRows, column, numColumns)
                renderer.SetViewport(viewport) 

                pos = (row,column)
                if pos in volumesByPosition:
                    volumes = volumesByPosition[pos]
                    for volume in volumes:
                        volumeProp = volume.CreateVolume()
                        renderer.AddVolume(volumeProp)
                
                if firstRenderer is None:
                    firstRenderer = renderer
                    camera = firstRenderer.GetActiveCamera()
                
                    c = volumeProp.GetCenter()

                    camera.ParallelProjectionOff()
                    camera.SetFocalPoint(c[0], c[1], c[2])
                    camera.SetPosition(c[0], c[1], c[2] + 10) 
                    camera.SetViewAngle(30)
                    camera.SetViewUp(0, 1, 0)
                else:
                    renderer.SetActiveCamera(firstRenderer.GetActiveCamera())
                
                for polyDataFile in self.polyDataFiles:
                    polyDataActor = polyDataFile.CreatePolyDataActor()
                    polyDataActor.VisibilityOff()
                    renderer.AddActor(polyDataActor)
                    app.RegisterToggleVisibilityCommand(polyDataActor, polyDataFile.KeySym)

        style = vtk.vtkInteractorStyleTrackballCamera()
        style.SetCurrentRenderer(firstRenderer)
        style.SetMotionFactor(5)

        app.SetInteractorStyle(style)

        return app
