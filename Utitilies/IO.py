import h5py
import numpy as np
import vtk

def readRawFile(filename, shape):
    rawData      = np.fromfile(filename, dtype=np.float32)
    rawData      = np.reshape(rawData, (-1, 5)) # each row consists of rsp(1), dir(3) and rad(1)

    responses    = rawData[:, 0]

    ignor        = np.isclose(responses, 0, atol=1e-2) # ignor is 1-D array
    responses    = responses[~ignor]
    tangentLines = rawData[:, 1:4]
    tangentLines = tangentLines[~ignor]
    radiuses     = rawData[:, 4]
    radiuses     = radiuses[~ignor]

    ignor        = np.reshape(ignor, shape) # ignor is 3-D array
    zs,ys,xs     = np.where(~ignor)
    measurements = np.column_stack(reversed(xs,ys,zs))

    tangentLinesPoints1 = np.empty_like(measurements, dtype=np.double)
    tangentLinesPoints2 = np.empty_like(measurements, dtype=np.double)

    for i in xrange(len(measurements)):
        p  = measurements[i]
        lp = tangentLines[i]
        tangentLinesPoints1[i] = p + lp
        tangentLinesPoints2[i] = p - lp
    
    dataset = dict()

    dataset['measurements']        = measurements
    dataset['tangentLinesPoints1'] = tangentLinesPoints1
    dataset['tangentLinesPoints2'] = tangentLinesPoints2
    dataset['radiuses']            = radiuses
    dataset['responses']           = responses

    return dataset

def readH5(filename, n_dims=3):
    dataset = dict()

    with h5py.File(filename, mode='r') as f:
        for name in ['indices1', 'indices2', 'radiuses']:
            if name in f:
                dataset[name] = f[name][()]

        for name in ['positions', 'measurements', 'tangentLinesPoints1', 'tangentLinesPoints2']:
            if name in f:
                item = f[name][()]
                item = np.reshape(item, newshape=(-1,n_dims))
                dataset[name] = item

    return dataset

def writeH5(filename, dataset):
    with h5py.File(filename, mode='w') as f:
        for name in dataset:
            item = dataset[name]
            f.create_dataset(name, data=item.flatten())

def writeParaView(filename, dataset):
    positions = dataset['positions']

    points = vtk.vtkPoints()
    for p in positions:
        points.InsertNextPoint(p)
    
    graph = vtk.vtkMutableUndirectedGraph()

    graph.SetNumberOfVertices(points.GetNumberOfPoints())
    graph.SetPoints(points)
    
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    for ind1,ind2 in zip(indices1, indices2):
        graph.AddGraphEdge(ind1, ind2)

    graphToPolyData = vtk.vtkGraphToPolyData()
    graphToPolyData.SetInputData(graph)

    polyDataWriter = vtk.vtkXMLPolyDataWriter()
    polyDataWriter.SetInputConnection(graphToPolyData.GetOutputPort())
    polyDataWriter.SetFileName(filename)
    polyDataWriter.Write()