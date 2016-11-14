import argparse
import IO
import MinimumSpanningTree
import matplotlib.pyplot as plt
import maxflow
import os.path
import numpy as np
import numpy.linalg as linalg
import vtk

from sklearn.neighbors import KDTree,radius_neighbors_graph
from xml.etree import ElementTree

def doConvertRawToH5(args):
    dirname  = args.dirname
    basename = args.basename

    filename = os.path.join(dirname, basename)
    dataset  = IO.readRawFile(filename, shape=(101,101,101))

    measurements        = dataset['measurements']

    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    mask = indices1 < indices2
    
    indices1 = indices1[mask]
    indices2 = indices2[mask]

    dataset['indices1'] = np.hstack((indices1, indices2))
    dataset['indices2'] = np.hstack((indices2, indices1))

    weights = np.full(len(indices1) + len(indices2), 20, dtype=np.float)

    dataset['weights']  = weights

    filename, _ = os.path.splitext(filename)
    filename    = filename + '.h5'
    
    IO.writeH5File(filename, dataset)

def getArcCenter(p, lp, q):
    chord = q - p
    radialLine = chord * lp.dot(lp) - lp * lp.dot(chord) # radialLine = lp x (chord x lp) = chord * ||lp||^2 - lp * <lp,chord>
    t = 0.5 * chord.dot(chord) / chord.dot(radialLine)
    if np.isinf(t):
        Cpq = np.array((np.inf, np.inf, np.inf))
    else:
        Cpq = p + t * radialLine
    return Cpq

def getArcRadius(p, Cpq):
    return np.sqrt((p - Cpq).dot(p - Cpq))

def doConvertRawToH5NoBifurc(dirname):
    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    nodeTypes = dataset['nodeTypes']

    bifurcs  = positions[nodeTypes == 'b']
    bifurcnn = KDTree(bifurcs)

    filename = os.path.join(dirname, 'canny2_image.raw')
    dataset  = IO.readRawFile(filename, shape=(101,101,101))

    measurements        = dataset['measurements']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses            = dataset['radiuses']
    responses           = dataset['responses']
 
    dist, ind = bifurcnn.query(measurements, k=1)

    ignor = dist[:,0] < 1

    measurements        = measurements[~ignor]
    tangentLinesPoints1 = tangentLinesPoints1[~ignor]
    tangentLinesPoints2 = tangentLinesPoints2[~ignor]
    radiuses            = radiuses[~ignor]
    responses           = responses[~ignor]

    dataset = dict()

    dataset['measurements']        = measurements
    dataset['tangentLinesPoints1'] = tangentLinesPoints1
    dataset['tangentLinesPoints2'] = tangentLinesPoints2
    dataset['radiuses']            = radiuses

    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    mask = indices1 < indices2
    
    indices1 = indices1[mask]
    indices2 = indices2[mask]

    dataset['indices1'] = np.hstack((indices1, indices2))
    dataset['indices2'] = np.hstack((indices2, indices1))
    
    weights = np.full(len(indices1) + len(indices2), 2.0, dtype=np.float)

    dataset['weights'] = weights

    filename = os.path.join(dirname, 'canny2_image_nobifurc.h5')
    IO.writeH5File(filename, dataset)

def doComputeArcRadiuses(dirname, pointsArrName):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset  = IO.readH5File(filename)

    pointsArr           = dataset[pointsArrName]
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    indices1            = dataset['indices1']
    indices2            = dataset['indices2']

    n = len(pointsArr)

    arcRadiuses = [[] for _ in xrange(n)]

    for index1,index2 in zip(indices1, indices2):
        p   = pointsArr[index1]
        q   = pointsArr[index2]
        lp  = tangentLinesPoints2[index1] - tangentLinesPoints1[index1]
        Cpq = getArcCenter(p, lp, q)

        arcRadius = getArcRadius(p, Cpq)
        arcRadiuses[index1].append(arcRadius)

    arcRadiusesMean   = np.empty(n, dtype=np.double)
    arcRadiusesStdDev = np.empty(n, dtype=np.double)

    for i in xrange(n):
        arcRadiusesMean[i]   = np.mean(np.reciprocal(arcRadiuses[i]))
        arcRadiusesStdDev[i] = np.std(np.reciprocal(arcRadiuses[i]))

    dataset['arcRadiusesMean']   = arcRadiusesMean
    dataset['arcRadiusesStdDev'] = arcRadiusesStdDev

    IO.writeH5File(filename, dataset)

def doConvertH5ToParaView(dirname):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset = IO.readH5File(filename)

    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.vtp')
    IO.writeParaView(filename, dataset)

def createLine(p, q):
    lineSrc = vtk.vtkLineSource()
    lineSrc.SetPoint1(p)
    lineSrc.SetPoint2(q)
    lineSrc.Update()
    return lineSrc.GetOutput()

def createSplinePolyData(spline, num_points):
    points = vtk.vtkPoints()

    for t in np.linspace(0, 1, num_points):
        points.InsertNextPoint(spline(t))

    polyLine = vtk.vtkPolyLineSource()
    polyLine.SetNumberOfPoints(points.GetNumberOfPoints())
    polyLine.SetPoints(points)
    polyLine.Update()

    polyData = polyLine.GetOutput()
    return polyData

# Hermite basis functions
def h00(t):
    s = 1 - t
    return (1 + 2 * t) * s * s

def h10(t):
    s = 1 - t
    return t * s * s

def h01(t):
    return t * t * (3 - 2 * t)

def h11(t):
    return t * t * (t - 1)

def createSpline(p0, m0, p1, m1):
    return lambda t: h00(t) * p0 + h10(t) * m0 + h01(t) * p1 + h11(t) * m1

def splineLength(spline, num_points):
    ts = np.linspace(0.0, 1.0, num_points, dtype=np.double)
    points = spline(ts[:,np.newaxis])
    splineLen = np.linalg.norm(points[:-1] - points[1:], axis=1).sum()
    return splineLen

def createGraphPolyData(pointsArr, indices1, indices2, weightsArr):
    points = vtk.vtkPoints()

    for p in pointsArr:
        points.InsertNextPoint(p)
    
    graph = vtk.vtkMutableUndirectedGraph()

    graph.SetNumberOfVertices(points.GetNumberOfPoints())
    graph.SetPoints(points)

    for index1,index2 in zip(indices1, indices2):
        graph.AddGraphEdge(index1, index2)

    weights = vtk.vtkDoubleArray()
    weights.SetName('Weights')
    
    for w in weightsArr:
        weights.InsertNextValue(w)

    graphToPolyData = vtk.vtkGraphToPolyData()
    graphToPolyData.SetInputData(graph)
    graphToPolyData.Update()

    polyData = graphToPolyData.GetOutput()
    polyData.GetCellData().AddArray(weights)

    return polyData

def createCircularPolyData(pointsArr, tangentLinesPoints1, tangentLinesPoints2, radiusesArr):
    polygonSrc = vtk.vtkRegularPolygonSource()
    polygonSrc.GeneratePolygonOff()

    appendPolyData = vtk.vtkAppendPolyData()

    for i,p in enumerate(pointsArr):
        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        radius = radiusesArr[i]

        polygonSrc.SetCenter(p)
        polygonSrc.SetNormal(lp)
        polygonSrc.SetNumberOfSides(36)
        polygonSrc.SetRadius(radius)
        polygonSrc.Update()

        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(polygonSrc.GetOutput())
        appendPolyData.AddInputData(polyData)
    
    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    return polyData

def doCreateGraphPolyDataFile(dirname, pointsArrName, weightsArrName):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset  = IO.readH5File(filename)

    pointsArr  = dataset[pointsArrName]
    indices1   = dataset['indices1']
    indices2   = dataset['indices2']
    weightsArr = dataset[weightsArrName]

    graphPolyData = createGraphPolyData(pointsArr, indices1, indices2, weightsArr)
    
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.vtp')
    IO.writePolyDataFile(filename, graphPolyData)

def doCreateCircularPolyDataFile(dirname, pointsArrName, radiusesArrName):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset  = IO.readH5File(filename)

    pointsArr           = dataset[pointsArrName]
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiusesArr         = dataset[radiusesArrName]

    polyData = createCircularPolyData(pointsArr, tangentLinesPoints1, tangentLinesPoints2, radiusesArr)
    
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.vtp')
    IO.writePolyDataFile(filename, polyData)

def doCreateSplinePolyDataFile(dirname):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset  = IO.readH5File(filename)

    positions           = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    
    indices1            = dataset['indices1']
    indices2            = dataset['indices2']

    n = len(positions)

    appendPolyData = vtk.vtkAppendPolyData()

    for i,k in zip(indices1, indices2):
        p  = positions[i]
        q  = positions[k]
        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

        lp = lp / linalg.norm(lp)
        lq = lq / linalg.norm(lq)
        dist = linalg.norm(p - q)
        lp = lp * dist
        lq = lq * dist

        spline = createSpline(p, lp, q, lq)
        splinePolyData = createSplinePolyData(spline, num_points=100)

        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(splinePolyData)
        appendPolyData.AddInputData(polyData)
    
    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.vtp')
    IO.writePolyDataFile(filename, polyData)

def doMST(dirname):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset  = IO.readH5File(filename)

    measurements        = dataset['measurements']
    positions           = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses            = dataset['radiuses']
    
    indices1            = dataset['indices1']
    indices2            = dataset['indices2']

    n = len(positions)

    G = dict()

    for i,k in zip(indices1, indices2):
        p  = positions[i]
        q  = positions[k]
        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

        lp = lp / linalg.norm(lp)
        lq = lq / linalg.norm(lq)
        dist = linalg.norm(p - q)
        lp = lp * dist
        lq = lq * dist

        spline    = createSpline(p, lp, q, lq)
        splineLen = splineLength(spline, num_points=100)
        
        if not i in G:
            G[i] = dict()

        if not k in G:
            G[k] = dict()

        G[i][k] = dist
        G[k][i] = dist

    T = MinimumSpanningTree.MinimumSpanningTree(G)

    deg = np.full(n, 0, dtype=np.int)

    for i,k in T:
        deg[i] += 1
        deg[k] += 1

    
    appendPolyData = vtk.vtkAppendPolyData()

    for i in xrange(n):
        p  = positions[i]
        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        lp = lp / (2 * linalg.norm(lp))

        polyData = createLine(p - lp, p + lp)
        appendPolyData.AddInputData(polyData)
    
    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.vtp')
    IO.writePolyDataFile(filename, polyData)

def doGraphCut(dirname):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset  = IO.readH5File(filename)

    positions           = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    
    indices1            = dataset['indices1']
    indices2            = dataset['indices2']

    n = len(positions)
    g = maxflow.Graph[float]()

    nodes = g.add_nodes(n)

    appendPolyData = vtk.vtkAppendPolyData()

    indices1 = []
    indices2 = []

    for i in xrange(n):
        for k in xrange(i + 1, n):
            if linalg.norm(positions[i] - positions[k]) < 2:
                indices1.append(i)
                indices2.append(k)

    for i,k in zip(indices1, indices2):
        if k < i: continue

        p  = positions[i]
        q  = positions[k]
        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

        lp = lp / linalg.norm(lp)
        lq = lq / linalg.norm(lq)
        dist = linalg.norm(p - q)
        lp = lp * dist
        lq = lq * dist
   
        for lpsgn,lqsgn in [(-lp,-lq), (-lp, lq), (lp,-lq), (lp, lq)]:
            spline    = createSpline(p,lpsgn, q, lqsgn)
            splineLen = splineLength(spline, num_points=100)

        A = splineLength(createSpline(p,-lp, q,-lq), num_points=100)
        B = splineLength(createSpline(p,-lp, q, lq), num_points=100)
        C = splineLength(createSpline(p, lp, q,-lq), num_points=100)
        D = splineLength(createSpline(p, lp, q, lq), num_points=100)

        #assert A + D <= B + C

        g.add_tedge(nodes[i], C, A)
        g.add_tedge(nodes[k], D - C, 0)
        g.add_edge(nodes[i], nodes[k], B + C - A - D, 0)

    flow = g.maxflow()

    G = dict()

    for i,k in zip(indices1, indices2):
        if k < i: continue

        p  = positions[i]
        q  = positions[k]
        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

        lp = lp / linalg.norm(lp)
        lq = lq / linalg.norm(lq)
        dist = linalg.norm(p - q)
        
        lp = lp * dist
        lq = lq * dist

        if g.get_segment(nodes[i]) == 1:
            lpsgn =  lp
        else:
            lpsgn = -lp

        if g.get_segment(nodes[k]) == 1:
            lqsgn =  lq
        else:
            lqsgn = -lq
   
        spline    = createSpline(p, lpsgn, q, lqsgn)
        splineLen = splineLength(spline, num_points=100)

        if not i in G:
            G[i] = dict()

        if not k in G:
            G[k] = dict()
        
        G[i][k] = splineLen
        G[k][i] = splineLen

    T = MinimumSpanningTree.MinimumSpanningTree(G)

    for i,k in T:
        p  = positions[i]
        q  = positions[k]
        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

        lp = lp / linalg.norm(lp)
        lq = lq / linalg.norm(lq)
        dist = linalg.norm(p - q)
        
        lp = lp * dist
        lq = lq * dist

        if g.get_segment(nodes[i]) == 1:
            lpsgn =  lp
        else:
            lpsgn = -lp

        if g.get_segment(nodes[k]) == 1:
            lqsgn =  lq
        else:
            lqsgn = -lq
   
        spline = createSpline(p, lpsgn, q, lqsgn)
        splinePolyData = createSplinePolyData(spline, num_points=100)

        polyData = vtk.vtkPolyData()
        polyData.DeepCopy(splinePolyData)
        appendPolyData.AddInputData(polyData)
        
    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.vtp')
    IO.writePolyDataFile(filename, polyData)

if __name__ == '__main__':
    # create the top-level parser
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers()

    # create the parser for the "doConvertRawToH5" command
    subparser = subparsers.add_parser('doConvertRawToH5')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doConvertRawToH5)

    # parse the args and call whatever function was selected
    args = argparser.parse_args()
    args.func(args)

   
    #doConvertRawToH5(dirname)
    #doConvertRawToH5NoBifurc(dirname)
    #doComputeArcRadiuses(dirname, pointsArrName='measurements')
    #doCreateCircularPolyDataFile(dirname, pointsArrName='measurements', radiusesArrName='arcRadiusesMean')
    
    #doMST(dirname)
    #doCreateSplinePolyDataFile(dirname)
    #doComputeArcRadiuses(dirname, pointsArrName='positions')
    #doCreateCircularPolyDataFile(dirname, pointsArrName='positions', radiusesArrName='arcRadiusesMean')
    #doCreateGraphPolyDataFile(dirname, pointsArrName='positions', weightsArrName='arcRadiuses')
    #doConvertH5ToParaView(dirname)
