import argparse
import IO
import MinimumSpanningTree
import matplotlib
import pandas as pd

# http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import maxflow
import os.path
import re
import numpy as np
import numpy.linalg as linalg
import scipy.integrate as integrate
import vtk

from sklearn.neighbors import KDTree,NearestNeighbors,radius_neighbors_graph
from xml.etree import ElementTree

def doConvertRawToH5(args):
    dirname = args.dirname
    basename = args.basename
    shape = tuple(args.shape)
    weight = args.weight

    filename = os.path.join(dirname, basename)
    dataset = IO.readRawFile(filename, shape=shape)

    measurements = dataset['measurements']

    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    dataset['indices1'] = indices1
    dataset['indices2'] = indices2

    weights = np.full(len(indices1), weight, dtype=np.float)

    dataset['weights'] = weights

    filename, _ = os.path.splitext(filename)
    filename = filename + '.h5'
    
    IO.writeH5File(filename, dataset)

def doConvertRawToH5Ignor(args):
    dirname = args.dirname
    basename = args.basename
    shape = tuple(args.shape)
    weight = args.weight

    filename = os.path.join(dirname, basename)
    dataset = IO.readRawFile(filename, shape=shape)

    measurements = dataset['measurements']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses = dataset['radiuses']
    responses = dataset['responses']
 
    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    ignor = np.full_like(radiuses, True, dtype=np.bool)
    ignor[indices1] = False
    ignor[indices2] = False

    measurements = measurements[~ignor]
    tangentLinesPoints1 = tangentLinesPoints1[~ignor]
    tangentLinesPoints2 = tangentLinesPoints2[~ignor]
    radiuses = radiuses[~ignor]
    responses = responses[~ignor]

    dataset = dict()

    dataset['measurements'] = measurements
    dataset['tangentLinesPoints1'] = tangentLinesPoints1
    dataset['tangentLinesPoints2'] = tangentLinesPoints2
    dataset['radiuses'] = radiuses
    
    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    dataset['indices1'] = indices1
    dataset['indices2'] = indices2
    
    weights = np.full(len(indices1), weight, dtype=np.float)

    dataset['weights'] = weights

    filename, _ = os.path.splitext(filename)
    filename = filename + '.h5'

    IO.writeH5File(filename, dataset)

def doConvertRawToH5NoBifurc(args):
    dirname = args.dirname
    basename = args.basename

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    nodeTypes = dataset['nodeTypes']

    bifurcs = positions[nodeTypes == 'b']
    bifurcnn = KDTree(bifurcs)

    filename = os.path.join(dirname, basename)
    dataset = IO.readRawFile(filename, shape=(101,101,101))

    measurements = dataset['measurements']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses = dataset['radiuses']
    responses = dataset['responses']
 
    dist, _ = bifurcnn.query(measurements, k=1)
    ignor = dist[:,0] < (np.sqrt(3) + 2) / 2

    measurements = measurements[~ignor]
    tangentLinesPoints1 = tangentLinesPoints1[~ignor]
    tangentLinesPoints2 = tangentLinesPoints2[~ignor]
    radiuses = radiuses[~ignor]
    responses = responses[~ignor]

    dataset = dict()

    dataset['measurements'] = measurements
    dataset['tangentLinesPoints1'] = tangentLinesPoints1
    dataset['tangentLinesPoints2'] = tangentLinesPoints2
    dataset['radiuses'] = radiuses

    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    dataset['indices1'] = indices1
    dataset['indices2'] = indices2
    
    weights = np.full(len(indices1), 2.0, dtype=np.float)

    dataset['weights'] = weights

    filename, _ = os.path.splitext(filename)
    filename = filename + '_nobifurc.h5'

    IO.writeH5File(filename, dataset)

def createGraphPolyData(points, indices1, indices2, connectedComponents):
    pointsArr = vtk.vtkPoints()

    for p in points:
        pointsArr.InsertNextPoint(p)
    
    graph = vtk.vtkMutableUndirectedGraph()

    graph.SetNumberOfVertices(pointsArr.GetNumberOfPoints())
    graph.SetPoints(pointsArr)

    for index1,index2 in zip(indices1, indices2):
        graph.AddGraphEdge(index1, index2)

    graphToPolyData = vtk.vtkGraphToPolyData()
    graphToPolyData.SetInputData(graph)
    graphToPolyData.Update()

    polyData = graphToPolyData.GetOutput()


    colorIndexArray = vtk.vtkDoubleArray()
    colorIndexArray.SetNumberOfValues(graph.GetNumberOfVertices())
    colorIndexArray.SetName("ConnectedComponent")

    for i in xrange(graph.GetNumberOfVertices()):
        colorIndexArray.SetValue(i, connectedComponents[i])

    polyData.GetPointData().SetScalars(colorIndexArray)

    return polyData

def doCreateGraphPolyDataFile(args):
    dirname = args.dirname
    basename = args.basename
    pointsArrName = args.points

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    connectedComponentsName = 'connectedComponentsIndices'

    positions = dataset[pointsArrName]
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']
    connectedComponentsIndices = dataset[connectedComponentsName] if connectedComponentsName in dataset else np.zeros(len(positions), dtype="double")

    polyData = createGraphPolyData(positions, indices1, indices2, connectedComponentsIndices)
    
    filename, _ = os.path.splitext(filename)
    filename = filename + '.vtp'
    IO.writePolyDataFile(filename, polyData)

def doEMST(args):
    dirname = args.dirname
    basename = args.basename
    maxradius = args.maxradius
    pointsArrName = args.points

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    positions = dataset[pointsArrName]

    n = len(positions)
    G = dict()

    for i in xrange(n):
        p = positions[i]
        for k in xrange(i + 1, n):
            q = positions[k]
            dist = linalg.norm(p - q)
            if dist > maxradius: continue

            if not i in G:
                G[i] = dict()

            if not k in G:
                G[k] = dict()

            G[i][k] = dist
            G[k][i] = dist

    T = MinimumSpanningTree.MinimumSpanningTree(G)
    indices1,indices2 = zip(*T)

    dataset['indices1'] = np.array(indices1, dtype=np.int)
    dataset['indices2'] = np.array(indices2, dtype=np.int)
    
    filename, _ = os.path.splitext(filename)
    filename = filename + 'EMST.h5'

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
    return linalg.norm(p - Cpq)

def getArcLength(p, q, Cpq):
    arcRad = getArcRadius(p, Cpq)

    pMinusCpq = p - Cpq
    qMinusCpq = q - Cpq

    pMinusCpq /= linalg.norm(pMinusCpq)
    qMinusCpq /= linalg.norm(qMinusCpq)

    arcLen = np.arccos(pMinusCpq.dot(qMinusCpq)) * arcRad
    return arcLen

def doArcMST(args):
    dirname = args.dirname
    basename = args.basename
    maxradius = args.maxradius

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    positions = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']

    tangentLines = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)

    n = len(positions)
    G = dict()

    for i in xrange(n):
        p = positions[i]
        lp = tangentLines[i]

        for k in xrange(i + 1, n):
            q = positions[k]
            dist = linalg.norm(p - q)
            if dist > maxradius: continue
            lq = tangentLines[k]

            Cpq = getArcCenter(p, lp, q)
            Cqp = getArcCenter(q, lq, p)
            
            arcLen1 = getArcLength(p, q, Cpq)
            arcLen2 = getArcLength(q, p, Cqp)

            arcLen = (arcLen1 + arcLen2) / 2

            if not i in G:
                G[i] = dict()

            if not k in G:
                G[k] = dict()

            G[i][k] = arcLen
            G[k][i] = arcLen

    T = MinimumSpanningTree.MinimumSpanningTree(G)
    indices1,indices2 = zip(*T)

    dataset['indices1'] = np.array(indices1, dtype=np.int)
    dataset['indices2'] = np.array(indices2, dtype=np.int)

    filename, _ = os.path.splitext(filename)
    filename = filename + 'ArcMST.h5'

    IO.writeH5File(filename, dataset)

def createArcPolyData(p, q, Cpq):
    arcSrc = vtk.vtkArcSource()
    arcSrc.SetPoint1(p)
    arcSrc.SetPoint2(q)
    arcSrc.SetCenter(Cpq)
    arcSrc.SetResolution(36)
    arcSrc.Update()
    polyData = arcSrc.GetOutput()
    return polyData

def doCreateArcsPolyDataFile(args):
    dirname = args.dirname
    basename = args.basename

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    positions = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']

    tangentLines = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)

    n = len(positions)

    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    appendPolyData = vtk.vtkAppendPolyData()

    for i,k in zip(indices1,indices2):
        p = positions[i]
        q = positions[k]

        lp = tangentLines[i]
        lq = tangentLines[k]

        Cpq = getArcCenter(p, lp, q)
        Cqp = getArcCenter(q, lq, p)

        arcLen1 = getArcLength(p, q, Cpq)
        arcLen2 = getArcLength(q, p, Cqp)

        if arcLen1 < arcLen2:
            polyData = createArcPolyData(p, q, Cpq)
        else:
            polyData = createArcPolyData(q, p, Cqp)

        appendPolyData.AddInputData(polyData)

    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()


    filename, _ = os.path.splitext(filename)
    filename = filename + '.vtp'

    IO.writePolyDataFile(filename, polyData)

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

def getCubicSpline(p0, m0, p1, m1):
    return lambda t: h00(t) * p0 + h10(t) * m0 + h01(t) * p1 + h11(t) * m1

def getSplineLength(spline, num_points):
    ts = np.linspace(0.0, 1.0, num_points, dtype=np.double)
    points = spline(ts[:,np.newaxis])
    splineLen = np.linalg.norm(points[:-1] - points[1:], axis=1).sum()
    return splineLen

def doCubicSplineMST(args):
    dirname = args.dirname
    basename = args.basename
    maxradius = args.maxradius

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    positions = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']

    tangentLines = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)

    n = len(positions)
    G = dict()

    for i in xrange(n):
        p = positions[i]
        lp = tangentLines[i]
        for k in xrange(i + 1, n):
            q = positions[k]
            lq = tangentLines[k]
            dist = linalg.norm(p - q)
            if dist > maxradius: continue

            minLength = np.inf

            for lpsgn,lqsgn in ((-1,-1),(-1, 1),(1,-1),(1, 1)):
                cubicSpline = getCubicSpline(p, dist * lpsgn * lp, q, dist * lqsgn * lq)
                splineLength = getSplineLength(cubicSpline, num_points=100)
                if splineLength < minLength:
                    minLength = splineLength

            if not i in G:
                G[i] = dict()

            if not k in G:
                G[k] = dict()

            G[i][k] = minLength
            G[k][i] = minLength

    T = MinimumSpanningTree.MinimumSpanningTree(G)
    indices1,indices2 = zip(*T)

    dataset['indices1'] = np.array(indices1, dtype=np.int)
    dataset['indices2'] = np.array(indices2, dtype=np.int)
    
    filename, _ = os.path.splitext(filename)
    filename = filename + 'CubicSplineMST.h5'

    IO.writeH5File(filename, dataset)

def createSplinePolyData(spline, num_points):
    points = vtk.vtkPoints()

    for t in np.linspace(0.0, 1.0, num_points):
        points.InsertNextPoint(spline(t))

    polyLineSrc = vtk.vtkPolyLineSource()
    polyLineSrc.SetNumberOfPoints(points.GetNumberOfPoints())
    polyLineSrc.SetPoints(points)
    polyLineSrc.Update()
    polyData = polyLineSrc.GetOutput()
    return polyData

def doCreateCubicSplinePolyDataFile(args):
    dirname = args.dirname
    basename = args.basename

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    positions = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']

    tangentLines = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)
    
    n = len(positions)

    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    appendPolyData = vtk.vtkAppendPolyData()

    for i,k in zip(indices1, indices2):
        p = positions[i]
        q = positions[k]

        lp = tangentLines[i]
        lq = tangentLines[k]

        dist = linalg.norm(p - q)

        spline = None
        minLength = np.inf

        for lpsgn,lqsgn in ((-1,-1),(-1, 1),(1,-1),(1, 1)):
            cubicSpline = getCubicSpline(p, dist * lpsgn * lp, q, dist * lqsgn * lq)
            splineLength = getSplineLength(cubicSpline, num_points=100)
            if splineLength < minLength:
                minLength = splineLength
                spline = cubicSpline

        splinePolyData = createSplinePolyData(spline, num_points=100)
        appendPolyData.AddInputData(splinePolyData)
    
    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    filename, _ = os.path.splitext(filename)
    filename = filename + '.vtp'

    IO.writePolyDataFile(filename, polyData)

def doConvertRawToH5Responses(args):
    dirname = args.dirname
    basename = args.basename
    shape = tuple(args.shape)
    weight = args.weight

    filename = os.path.join(dirname, basename)
    dataset = IO.readRawFile(filename, shape=shape)

    measurements = dataset['measurements']
    radiuses = dataset['radiuses']
    responses = dataset['responses']

    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    # conn = radius_neighbors_graph(measurements, radius=(1 + np.sqrt(2)) / 2,
    # metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    # Sort in ascending order
    # indices1, indices2 = zip(*sorted(zip(indices1,indices2)))

    # Sort in descending order
    # indices1, indices2 = zip(*sorted(zip(indices1,indices2), key=lambda x:
    # x[1], reverse=True))
    # indices1, indices2 = zip(*sorted(zip(indices1,indices2), key=lambda x:
    # x[0]))

    indices1 = np.array(indices1, dtype=np.int)
    indices2 = np.array(indices2, dtype=np.int)

    dataset['indices1'] = indices1
    dataset['indices2'] = indices2
    dataset['radiuses'] = radiuses

    weights = np.multiply(weight, responses[indices1])

    dataset['weights'] = weights

    filename, _ = os.path.splitext(filename)
    filename = filename + '.h5'
    
    IO.writeH5File(filename, dataset)

def doROC(args):
    dirname = args.dirname
    basename1 = args.basename1
    basename2 = args.basename2

    filename1 = os.path.join(dirname, basename1)

    stgd1 = np.genfromtxt(filename1, delimiter=',', usecols=1, skip_header=1) # source-to-target-graphs-distance
    sglr1 = np.genfromtxt(filename1, delimiter=',', usecols=2, skip_header=1) # source-graphs-length-ratio
    tglr1 = np.genfromtxt(filename1, delimiter=',', usecols=3, skip_header=1) # target-graphs-length-ratio
    
    filename2 = os.path.join(dirname, basename2)

    stgd2 = np.genfromtxt(filename2, delimiter=',', usecols=1, skip_header=1) # source-to-target-graphs-distance
    sglr2 = np.genfromtxt(filename2, delimiter=',', usecols=2, skip_header=1) # source-graphs-length-ratio
    tglr2 = np.genfromtxt(filename2, delimiter=',', usecols=3, skip_header=1) # target-graphs-length-ratio

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(16, 6)

    ax1.set_xlabel('source-to-target-graphs-distance')
    ax1.set_ylabel('source-graphs-length-ratio')

    ax1.axis((0, np.maximum(stgd1.max(), stgd2.max()), 0, 100))
    ax1.scatter(stgd1, 100 * sglr1, 2.0, color='green', marker=',', label=basename1)
    ax1.scatter(stgd2, 100 * sglr2, 2.0, color='blue', marker=',', label=basename2)
    ax1.legend(loc=4)

    ax2.set_xlabel('source-to-target-graphs-distance')
    ax2.set_ylabel('target-graphs-length-ratio')

    ax2.axis((0, np.maximum(stgd1.max(), stgd2.max()), 0, 100))
    ax2.scatter(stgd1, 100 * tglr1, 2.0, color='green', marker=',', label=basename1)
    ax2.scatter(stgd2, 100 * tglr2, 2.0, color='blue', marker=',', label=basename2)
    ax2.legend(loc=4)

    basename1,_ = os.path.splitext(basename1)
    basename2,_ = os.path.splitext(basename2)

    filename = os.path.join(dirname, basename1 + '.' + basename2 + 'ROC.png')
    fig.savefig(filename)

def doAUC(args):
    dirname = args.dirname
    basename = args.basename

    filename = os.path.join(dirname, basename)

    sourceToTargetGraphsDistance = np.genfromtxt(filename, delimiter=',', usecols=1, skip_header=1) # source-to-target-graphs-distance
    sourceGraphsLengthRatio = np.genfromtxt(filename, delimiter=',', usecols=2, skip_header=1) # source-graphs-length-ratio
    targetGraphsLengthRatio = np.genfromtxt(filename, delimiter=',', usecols=3, skip_header=1) # target-graphs-length-ratio

    print('{:18.16f}'.format(integrate.trapz(sourceGraphsLengthRatio, sourceToTargetGraphsDistance)))
    print('{:18.16f}'.format(integrate.trapz(targetGraphsLengthRatio, sourceToTargetGraphsDistance)))

def createLinePolyData(s, t):
    lineSrc = vtk.vtkLineSource()
    lineSrc.SetPoint1(s)
    lineSrc.SetPoint2(t)
    lineSrc.Update()
    polyData = lineSrc.GetOutput()
    return polyData

def createTangentsPolyData(tangentLinesPoints1, tangentLinesPoints2):
    appendPolyData = vtk.vtkAppendPolyData()

    for p,q in zip(tangentLinesPoints1, tangentLinesPoints2):
        polyData = createLinePolyData(p,q)
        appendPolyData.AddInputData(polyData)

    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    return polyData

def doCreateTangentsPolyDataFile(args):
    dirname = args.dirname
    basename = args.basename
    pointsArrName = args.points

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    positions = dataset[pointsArrName]
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    scale = dataset['radiuses']

    tangentLines = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)
    
    n = len(positions)

    appendPolyData = vtk.vtkAppendPolyData()
    scaleDataArray = vtk.vtkDoubleArray()
    scaleDataArray.SetNumberOfValues(n * 2)
    scaleDataArray.SetName("Radius")

    for i in xrange(n):
        p = positions[i]
        s = p + 0.5 * tangentLines[i]
        t = p - 0.5 * tangentLines[i]

        linePolyData = createLinePolyData(s,t)
        appendPolyData.AddInputData(linePolyData)
        scaleDataArray.SetValue(i * 2, scale[i])
        scaleDataArray.SetValue(i * 2 + 1, scale[i])
    
    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    polyData.GetPointData().SetScalars(scaleDataArray)

    filename, _ = os.path.splitext(filename)
    filename = filename + 'Tangents.vtp'
    IO.writePolyDataFile(filename, polyData)

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

def doCreateRadiusesPolyDataFile(args):
    dirname = args.dirname
    basename = args.basename
    pointsArrName = args.points

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    points = dataset[pointsArrName]
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses = dataset['radiuses']

    polyData = createCircularPolyData(points, tangentLinesPoints1, tangentLinesPoints2, radiuses)
    
    filename, _ = os.path.splitext(filename)
    filename = filename + 'Radiuses.vtp'
    IO.writePolyDataFile(filename, polyData)

def doMST(dirname):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset = IO.readH5File(filename)

    measurements = dataset['measurements']
    positions = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses = dataset['radiuses']
    
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    n = len(positions)
    G = dict()

    for i,k in zip(indices1, indices2):
        p = positions[i]
        q = positions[k]
        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

        lp = lp / linalg.norm(lp)
        lq = lq / linalg.norm(lq)
        dist = linalg.norm(p - q)
        lp = lp * dist
        lq = lq * dist

        spline = createSpline(p, lp, q, lq)
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
        p = positions[i]
        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        lp = lp / (2 * linalg.norm(lp))

        polyData = createLine(p - lp, p + lp)
        appendPolyData.AddInputData(polyData)
    
    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.vtp')
    IO.writePolyDataFile(filename, polyData)

#def doGraphCut(dirname):
#    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
#    dataset = IO.readH5File(filename)

#    positions = dataset['positions']
#    tangentLinesPoints1 = dataset['tangentLinesPoints1']
#    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    
#    indices1 = dataset['indices1']
#    indices2 = dataset['indices2']

#    n = len(positions)
#    g = maxflow.Graph[float]()

#    nodes = g.add_nodes(n)

#    appendPolyData = vtk.vtkAppendPolyData()

#    indices1 = []
#    indices2 = []

#    for i in xrange(n):
#        for k in xrange(i + 1, n):
#            if linalg.norm(positions[i] - positions[k]) < 2:
#                indices1.append(i)
#                indices2.append(k)

#    for i,k in zip(indices1, indices2):
#        if k < i: continue

#        p = positions[i]
#        q = positions[k]
#        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
#        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

#        lp = lp / linalg.norm(lp)
#        lq = lq / linalg.norm(lq)
#        dist = linalg.norm(p - q)
#        lp = lp * dist
#        lq = lq * dist
   
#        for lpsgn,lqsgn in [(-lp,-lq), (-lp, lq), (lp,-lq), (lp, lq)]:
#            spline = createSpline(p,lpsgn, q, lqsgn)
#            splineLen = splineLength(spline, num_points=100)

#        A = splineLength(createSpline(p,-lp, q,-lq), num_points=100)
#        B = splineLength(createSpline(p,-lp, q, lq), num_points=100)
#        C = splineLength(createSpline(p, lp, q,-lq), num_points=100)
#        D = splineLength(createSpline(p, lp, q, lq), num_points=100)

#        #assert A + D <= B + C

#        g.add_tedge(nodes[i], C, A)
#        g.add_tedge(nodes[k], D - C, 0)
#        g.add_edge(nodes[i], nodes[k], B + C - A - D, 0)

#    flow = g.maxflow()

#    G = dict()

#    for i,k in zip(indices1, indices2):
#        if k < i: continue

#        p = positions[i]
#        q = positions[k]
#        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
#        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

#        lp = lp / linalg.norm(lp)
#        lq = lq / linalg.norm(lq)
#        dist = linalg.norm(p - q)
        
#        lp = lp * dist
#        lq = lq * dist

#        if g.get_segment(nodes[i]) == 1:
#            lpsgn = lp
#        else:
#            lpsgn = -lp

#        if g.get_segment(nodes[k]) == 1:
#            lqsgn = lq
#        else:
#            lqsgn = -lq
   
#        spline = createSpline(p, lpsgn, q, lqsgn)
#        splineLen = splineLength(spline, num_points=100)

#        if not i in G:
#            G[i] = dict()

#        if not k in G:
#            G[k] = dict()
        
#        G[i][k] = splineLen
#        G[k][i] = splineLen

#    T = MinimumSpanningTree.MinimumSpanningTree(G)

#    for i,k in T:
#        p = positions[i]
#        q = positions[k]
#        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
#        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

#        lp = lp / linalg.norm(lp)
#        lq = lq / linalg.norm(lq)
#        dist = linalg.norm(p - q)
        
#        lp = lp * dist
#        lq = lq * dist

#        if g.get_segment(nodes[i]) == 1:
#            lpsgn = lp
#        else:
#            lpsgn = -lp

#        if g.get_segment(nodes[k]) == 1:
#            lqsgn = lq
#        else:
#            lqsgn = -lq
   
#        spline = createSpline(p, lpsgn, q, lqsgn)
#        splinePolyData = createSplinePolyData(spline, num_points=100)

#        polyData = vtk.vtkPolyData()
#        polyData.DeepCopy(splinePolyData)
#        appendPolyData.AddInputData(polyData)
        
#    appendPolyData.Update()
#    polyData = appendPolyData.GetOutput()

#    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.vtp')
#    IO.writePolyDataFile(filename, polyData)
def doAnalyzeLabeling(args):
    dirname = args.dirname
    basename = args.basename

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    nodeTypes = dataset['nodeTypes']

    bifurcs = positions[nodeTypes == 'b']
    bifurcnn = KDTree(bifurcs)

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    positions = dataset['positions']
    sourceIndices = dataset['sourceIndices']
    targetIndices = dataset['targetIndices']

    dist, _ = bifurcnn.query(positions[sourceIndices], k=1)
    nearBifurc = dist[:,0] < 1e-2

    dist = linalg.norm(positions[sourceIndices] - positions[targetIndices], axis=1)
    print np.mean(dist),np.std(dist),np.mean(dist[nearBifurc]),np.std(dist[nearBifurc]),np.count_nonzero(nearBifurc)

def projectOntoSourceTree(sOrig, tOrig, pOrig):
    s = sOrig[:,np.newaxis,:] # s.shape = (numPoints1, 1L, numDimensions)
    t = tOrig[:,np.newaxis,:] # t.shape = (numPoints1, 1L, numDimensions)
    p = pOrig[np.newaxis,:,:] # p.shape = (1L, numPoints2, numDimensions)

    sMinusT = s - t
    sMinusTSq = np.sum(sMinusT * sMinusT, axis=2)

    sMinusP = s - p
    sMinusPDotSMinusT = np.sum(sMinusP * sMinusT, axis=2)
    
    lambd = sMinusPDotSMinusT / sMinusTSq
    lambd = lambd[:,:,np.newaxis]
    
    # proj[i,k] is projection of point p[k] onto line between points s[i] and
    # t[i]
    proj = s - lambd * sMinusT

    # dist[i,k] is distance between point p[k] and line between points s[i] and
    # t[i]
    dist = linalg.norm(proj - p, axis=2)
    
    # ignore points which projections do not belong to corresponding intervals
    ignor = np.logical_or(lambd < 0, lambd > 1)
    ignor = ignor[:,:,0]
    
    sDist = linalg.norm(s - p, axis=2)
    tDist = linalg.norm(t - p, axis=2)

    dist = np.where(ignor, np.minimum(sDist, tDist), dist)

    # closIndex[k] is index i such that an interval between points s[i] and
    # t[i] of source tree is the closest one to point p[k]
    closIndex = np.argmin(dist, axis=0)

    # closProj[k] is projection of point p[k] to the closest interval of source
    # tree
    closProj = np.array([proj[closIndex[I]][I] for I in np.ndindex(closIndex.shape)])

    # closLambd[k] is projection coefficient corresponding to closProj[k]
    closLambd = np.array([lambd[closIndex[I]][I] for I in np.ndindex(closIndex.shape)]) 

    # errors[k] = ||pOrig[k] - closProj[k]||
    errors = linalg.norm(closProj - pOrig, axis=1)

    return (closIndex, closProj, closLambd, errors)

def doProjectionOntoSourceTreePolyDataFile(args):
    dirname = args.dirname
    targetFileBasename = args.targetFileBasename
    positionsDataSet = args.positions
    
    sourceFilename = os.path.join(dirname, 'tree_structure.xml')
    targetFilename = os.path.join(dirname, targetFileBasename)

    sourceDataset = IO.readGxlFile(sourceFilename)
    targetDataset = IO.readH5File(targetFilename)

    positions = sourceDataset['positions']
    indices1 = sourceDataset['indices1']
    indices2 = sourceDataset['indices2']

    sOrig = positions[indices1]
    tOrig = positions[indices2]

    pOrig = targetDataset[positionsDataSet]

    closIndex,closProj,closLambd,errors = projectOntoSourceTree(sOrig, tOrig, pOrig)

    print 'errors.mean       = ', np.mean(errors)
    print 'errors.stdDev     = ', np.std(errors)
    print 'errors.median     = ', np.median(errors)

    numIndices = len(indices1)

    positions = []
    indices1 = []
    indices2 = []

    for index in xrange(numIndices):
        mask = np.equal(closIndex, index)

        orderedByLambda = sorted(zip(closLambd[mask], pOrig[mask]))
        if (len(orderedByLambda) == 0): continue

        orderedLambd, orderedProj = zip(*orderedByLambda)

        startIndex = len(positions)
        numProjections = len(orderedProj)

        positions.extend(orderedProj)
        indices1.extend(xrange(startIndex, startIndex + numProjections - 1))
        indices2.extend(xrange(startIndex + 1, startIndex + numProjections))

    polyData = createGraphPolyData(positions, indices1, indices2)
    
    filename, _ = os.path.splitext(targetFileBasename)
    filename = os.path.join(dirname, filename + 'OptimalTree.vtp')
    IO.writePolyDataFile(filename, polyData)

    positions = []
    positions.extend(pOrig)
    positions.extend(closProj)

    pOrigLen = len(pOrig)
    indices1 = list(xrange(0, pOrigLen))
    indices2 = list(xrange(pOrigLen, len(positions)))

    polyData = createGraphPolyData(positions, indices1, indices2)
    
    filename, _ = os.path.splitext(targetFileBasename)
    filename = os.path.join(dirname, filename + 'ProjectionOntoSourceTree.vtp')
    IO.writePolyDataFile(filename, polyData)

def doProjectionOntoSourceTreeCsv(args):
    dirname = args.dirname
    targetFileBasename = args.targetFileBasename
    prependStringRow = args.prependStringRow
    pointsArrName = args.points
    
    sourceFilename = os.path.join(dirname, 'tree_structure.xml')
    targetFilename = os.path.join(dirname, targetFileBasename)

    sourceDataset = IO.readGxlFile(sourceFilename)
    targetDataset = IO.readH5File(targetFilename)

    positions = sourceDataset['positions']
    indices1 = sourceDataset['indices1']
    indices2 = sourceDataset['indices2']

    sOrig = positions[indices1]
    tOrig = positions[indices2]
    pOrig = targetDataset[pointsArrName]

    _,_,_,errors = projectOntoSourceTree(sOrig, tOrig, pOrig)

    errMean = np.mean(errors)
    errStdDev = np.std(errors)
    errMedian = np.median(errors)
    
    err25Perc = np.percentile(errors,q=25)
    err75Perc = np.percentile(errors,q=75)
    err95Perc = np.percentile(errors,q=95)
    err100Perc = np.percentile(errors,q=100)

    print '{0}{1},{2},{3},{4},{5},{6},{7}'.format(prependStringRow, errMean, errStdDev, errMedian, err25Perc, err75Perc, err95Perc, err100Perc)

def doErrorBar(args):
    dirname = args.dirname
    basename = args.basename
    
    filename = os.path.join(dirname, basename)

    weight = np.genfromtxt(filename, delimiter=',', usecols=0, skip_header=1) # weight
    errMean = np.genfromtxt(filename, delimiter=',', usecols=1, skip_header=1) # error-mean
    errStdDev = np.genfromtxt(filename, delimiter=',', usecols=2, skip_header=1) # error-stdDev
    errMedian = np.genfromtxt(filename, delimiter=',', usecols=3, skip_header=1) # error-median

    fig = plt.figure()
    fig.set_size_inches(12, 6)

    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.0)

    ax.set_xlabel('weight')
    ax.set_ylabel('error')

    ax.errorbar(x=weight, y=errMean, yerr=errStdDev, label='mean/stdDev')
    ax.plot(weight, errMedian, label='median')
    ax.legend(loc=2)
    
    filename,_ = os.path.splitext(filename)
    fig.savefig(filename + '.png')

def doCreateDatabaseProjectionOntoSourceTreeCsvFile(args):
    dirname = args.dirname
    basename = args.basename
    firstNum = args.firstNum
    lastNum = args.lastNum

    df = pd.DataFrame()

    items = dict()
    curvatureWeightColumnName = 'curvatureWeight'

    for num in xrange(firstNum, lastNum + 1):
        filename = os.path.join(dirname, 'image{0:03d}'.format(num), basename)
        rows = np.genfromtxt(filename, delimiter=',', skip_header=1)

        errorMeanColumnName = 'errorMean{0:03d}'.format(num)
        errorStdDevColumnName = 'errorStdDev{0:03d}'.format(num)

        curvatureWeightColumn = []
        errorMeanColumn = []
        errorStdDevColumn = []

        for r in rows:
            curvatureWeight = r[0]
            errorMean = r[1]
            errorStdDev = r[2]

            curvatureWeightColumn.append(curvatureWeight)
            errorMeanColumn.append(errorMean)
            errorStdDevColumn.append(errorStdDev)
           
        items[curvatureWeightColumnName] = curvatureWeightColumn
        items[errorMeanColumnName] = errorMeanColumn
        items[errorStdDevColumnName] = errorStdDevColumn

    df = pd.DataFrame(items)
    df = df.set_index(curvatureWeightColumnName)

    filename = os.path.join(dirname, 'DatabaseProjectionOntoSourceTree.csv')
    df.to_csv(filename)

def createProjectionsOntoGroundTruthTree(sOrig, tOrig, pOrig, closestIndices=None):
    s = sOrig[:,np.newaxis,:] # s.shape = (numPoints1, 1L, numDimensions)
    t = tOrig[:,np.newaxis,:] # t.shape = (numPoints1, 1L, numDimensions)
    p = pOrig[np.newaxis,:,:] # p.shape = (1L, numPoints2, numDimensions)

    sMinusT = s - t
    sMinusTSq = np.sum(sMinusT * sMinusT, axis=2)

    sMinusP = s - p
    #sMinusPDotSMinusT = np.sum(sMinusP * sMinusT, axis=2)
    sMinusPDotSMinusT = np.einsum('ijk,ijk->ij',sMinusP, sMinusT)

    lambd = sMinusPDotSMinusT / sMinusTSq
    lambd = lambd[:,:,np.newaxis]

    # proj[i,k] is projection of point p[k] onto line between points s[i] and
    # t[i]
    # dist[i,k] is distance between point p[k] and line between points s[i] and
    # t[i]
    proj = s - lambd * sMinusT
    dist = linalg.norm(proj - p, axis=2)

    # ignore points which projections do not belong to corresponding intervals
    ignor = np.logical_or(lambd < 0, lambd > 1)
    ignor = ignor.reshape(dist.shape)

    sDist = linalg.norm(s - p, axis=2)
    tDist = linalg.norm(t - p, axis=2)

    dist = np.where(ignor, np.minimum(sDist, tDist), dist)

    # closestIndices[k] is index i such that the distance between point p[k]
    # and
    # interval between points s[i] and t[i] is the minimal among all intervals
    if closestIndices is None:
        closestIndices = np.argmin(dist, axis=0)
    else:
        np.argmin(dist, axis=0, out=closestIndices)

    # closestPoints[k] is the closest to p[k] point belonging to the interval
    # between points
    # s[closestIndices[k]] and t[closestIndices[k]]
    closestPoints = np.array([(sOrig[closestIndices[I]] if sDist[closestIndices[I]][I] < tDist[closestIndices[I]][I] else tOrig[closestIndices[I]]) \
        if ignor[closestIndices[I]][I] else proj[closestIndices[I]][I] for I in np.ndindex(closestIndices.shape)])

    return closestPoints

def doCreateProjectionsOntoGroundTruthTreeGraphPolyDataFile(args):
    dirname = args.dirname
    basename = args.basename
    pointsArrName = args.points

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    sOrig = positions[indices1]
    tOrig = positions[indices2]

    filename = os.path.join(dirname, basename)
    dataset = IO.readRawFile(filename, shape=(101,101,101))

    points = dataset[pointsArrName]
    numberOfPoints = len(points)
    projections = createProjectionsOntoGroundTruthTree(sOrig, tOrig, points)

    indices1 = list(xrange(numberOfPoints))
    indices2 = list(xrange(numberOfPoints, 2 * numberOfPoints))

    pointsArr = np.vstack((points, projections))
    polyData = createGraphPolyData(pointsArr, indices1, indices2)

    filename, _ = os.path.splitext(filename)
    filename = filename + 'ProjectionsOntoGroundTruthTree.vtp'
    IO.writePolyDataFile(filename, polyData)

def doAnalyzeNonMaximumSuppressionVolumeCsv(args):
    dirname = args.dirname
    basenameOrig = args.basenameOrig
    basename = args.basename
    pointsArrName = args.points
    thresholdAbove = args.thresholdAbove

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    sOrig = positions[indices1]
    tOrig = positions[indices2]

    filenameOrig = os.path.join(dirname, basenameOrig)
    filename = os.path.join(dirname, basename)

    datasetOrig = IO.readRawFile(filenameOrig, shape=(101,101,101))
    dataset = IO.readRawFile(filename, shape=(101,101,101))

    pointsOrig = datasetOrig[pointsArrName]
    points = dataset[pointsArrName]

    responsesOrig = datasetOrig['responses']

    numberOfPointsOrig = len(pointsOrig)
    numberOfPoints = len(points)

    projectionsOrig = createProjectionsOntoGroundTruthTree(sOrig, tOrig, pointsOrig)
    distancesOrig = linalg.norm(pointsOrig - projectionsOrig, axis=1)

    # anything farther that thresholdAbove is considered to be negative
    ignorOrig = distancesOrig > thresholdAbove
    positives, = np.where(~ignorOrig)
    negatives, = np.where(ignorOrig)

    numberOfNegatives = len(negatives) # the number of real negative cases in the data
    numberOfPositives = len(positives) # the number of real positive cases in the data
    assert numberOfPositives + numberOfNegatives == numberOfPointsOrig

    # pairwiseRel[i,k] is True when pointsOrig[i] and points[k] are the same
    # points
    pairwiseRel = np.all(np.equal(pointsOrig[:,np.newaxis,:], points[np.newaxis,:,:]), axis=2)
    indicesOrig,indices = np.nonzero(pairwiseRel)
    assert np.array_equal(pointsOrig[indicesOrig], points[indices])

    # tp is positives that presented in both volumes
    truePositives = np.intersect1d(positives, indicesOrig)
    # fn is those positives not presented in indicesOrig
    falsNegatives = np.setdiff1d(positives, indicesOrig)

    numberOfTruePos = len(truePositives)
    numberOfFalsNeg = len(falsNegatives)
    assert numberOfTruePos + numberOfFalsNeg == numberOfPositives

    # fp is negatives that presented in both volumes
    falsPositives = np.intersect1d(negatives, indicesOrig)
    # tn is those negatives not presented in incidesOrig
    trueNegatives = np.setdiff1d(negatives, indicesOrig)

    numberOfTrueNeg = len(trueNegatives)
    numberOfFalsPos = len(falsPositives)
    assert numberOfTrueNeg + numberOfFalsPos == numberOfNegatives
    assert numberOfTruePos + numberOfFalsPos == numberOfPoints

    precision = float(numberOfTruePos) / (numberOfTruePos + numberOfFalsPos)
    recall = float(numberOfTruePos) / (numberOfTruePos + numberOfFalsNeg)
    accuracy = (float(numberOfTruePos) + numberOfTrueNeg) / (numberOfPositives + numberOfNegatives)
    fmeasure = 2 * (precision * recall) / (precision + recall)

    print "numberOfPositives,numberOfNegatives,numberOfTruePositives,numberOfFalsePositives,numberOfFalseNegatives,numberOfTrueNegatives,precision,recall,accuracy,fmeasure"
    print ",".join(str(i) for i in (numberOfPositives,numberOfNegatives,numberOfTruePos,numberOfFalsPos,numberOfFalsNeg,numberOfTrueNeg,precision,recall,accuracy,fmeasure))

    print np.mean(responsesOrig[positives]),np.std(responsesOrig[positives])
    print np.mean(responsesOrig[negatives]),np.std(responsesOrig[negatives])

    print np.mean(responsesOrig[truePositives]),np.std(responsesOrig[truePositives])
    print np.mean(responsesOrig[falsPositives]),np.std(responsesOrig[falsPositives])

def doCreateDistanceToClosestPointsCsv(args):
    dirname = args.dirname
    basename = args.basename
    voxelWidth = args.voxelWidth
    pointsArrName = args.points
    doOutputHeader = args.doOutputHeader
    prependHeaderStr = args.prependHeaderStr
    prependRowStr = args.prependRowStr
    minRadiusIncl = args.minRadiusIncl
    maxRadiusExcl = args.maxRadiusExcl

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    positions = voxelWidth * positions

    indices1 = dataset['indices1']
    indices2 = dataset['indices2']
    radiuses = dataset['radiusPrime']

    sOrig = positions[indices1]
    tOrig = positions[indices2]

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    points = dataset[pointsArrName]
    numberOfPoints = len(points)

    closestIndices = np.empty((numberOfPoints,), dtype='int32')
    closestPoints = createProjectionsOntoGroundTruthTree(sOrig, tOrig, points, closestIndices)
    closestRadiuses = radiuses[closestIndices]

    distancesSq = np.sum(np.square(closestPoints - points), axis=1)
    ignor = np.logical_or(closestRadiuses < minRadiusIncl, closestRadiuses >= maxRadiusExcl)
    distancesSq = distancesSq[~ignor]
    distances = np.sqrt(distancesSq)

    num = len(distances)
    sum = np.sum(distances)
    ssd = np.sum(distancesSq)
    ave = sum / num # ave = np.mean(distances)
    msd = ssd / num # np.mean(distancesSq)
    var = msd - ave * ave # var = np.var(distances)
    std = np.sqrt(var) # std = np.std(distances)
    med = np.median(distances)
    p25 = np.percentile(distances, q=25)
    p75 = np.percentile(distances, q=75)
    p95 = np.percentile(distances, q=95)
    p99 = np.percentile(distances, q=99)
    max = np.max(distances)
    
    keyValPairs = [(name,eval(name)) for name in ('num','ave','std','med','var','sum','ssd','p25','p75','p95','p99','max')]

    if (doOutputHeader):
        print prependHeaderStr + (",".join(kvp[0].upper() for kvp in keyValPairs))

    print prependRowStr + (",".join(str(kvp[1]) for kvp in keyValPairs))

def doCreateCoDirectionalityWithClosestPointsCsv(args):
    dirname = args.dirname
    basename = args.basename
    pointsArrName = args.points
    doOutputHeader = args.doOutputHeader
    prependHeaderStr = args.prependHeaderStr
    prependRowStr = args.prependRowStr

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    sOrig = positions[indices1]
    tOrig = positions[indices2]

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    points = dataset[pointsArrName]
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    
    numberOfPoints = len(points)

    closestIndices = np.zeros(points.shape[:1], dtype='int64')
    closestPoints = createProjectionsOntoGroundTruthTree(sOrig, tOrig, points, closestIndices)

    tangentLinesOrig = sOrig[closestIndices] - tOrig[closestIndices]
    tangentLinesOrig = tangentLinesOrig / linalg.norm(tangentLinesOrig, axis=1, keepdims=True)

    tangentLines = tangentLinesPoints1 - tangentLinesPoints2
    tangentLines = tangentLines / linalg.norm(tangentLines, axis=1, keepdims=True)

    xs = np.abs(np.sum(tangentLines * tangentLinesOrig, axis=1)) # cos(u,v)
    ys = np.sqrt(1 - np.square(xs)) # sin(u,v)

    num = numberOfPoints

    sumXs = np.sum(xs)
    sumYs = np.sum(ys)

    aveXs = sumXs / num
    aveYs = sumYs / num

    circularMean = np.arctan2(aveYs, aveXs) # https://en.wikipedia.org/wiki/Mean_of_circular_quantities

    keyValPairs = [(name,eval(name)) for name in ('num','circularMean','aveXs','aveYs','sumXs','sumYs')]

    if (doOutputHeader):
        print prependHeaderStr + (",".join(kvp[0].upper() for kvp in keyValPairs))

    print prependRowStr + (",".join(str(kvp[1]) for kvp in keyValPairs))

def doCreateDistanceClosestInnerNodesCsv(args):
    dirname = args.dirname
    basename = args.basename
    pointsArrName = args.points
    doOutputHeader = args.doOutputHeader
    prependHeaderStr = args.prependHeaderStr
    prependRowStr = args.prependRowStr

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positionsOrig = dataset['positions']
    nodeTypes = dataset['nodeTypes']

    pointsOrig = positionsOrig[nodeTypes == 'b']

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    positions = dataset[pointsArrName]
    numberOfPoints = len(positions)

    identity = np.identity(numberOfPoints)
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']
    
    deg = np.zeros(numberOfPoints, dtype=np.int)

    for i,k in zip(indices1, indices2):
        deg[i] += 1
        deg[k] += 1
    
    points = positions[deg > 2]

    pointsTree = KDTree(points)
    distances,neighbors = pointsTree.query(pointsOrig)
    neighbors = neighbors.flatten()

    pointsOrigTree = KDTree(pointsOrig)
    distancesOrig,neighborsOrig = pointsOrigTree.query(points)
    neighborsOrig = neighborsOrig.flatten()
    
    maskOrig = neighborsOrig[neighbors] == np.arange(len(neighbors))
    mask = neighbors[neighborsOrig] == np.arange(len(neighborsOrig))

    assert np.count_nonzero(maskOrig) == np.count_nonzero(mask)

    num = len(distances)
    sum = np.sum(distances)
    ave = np.mean(distances)
    ssd = np.sum(np.square(distances))
    std = np.std(distances)
    med = np.median(distances)
    max = np.max(distances)

    numOrig = len(distancesOrig)
    sumOrig = np.sum(distancesOrig)
    aveOrig = np.mean(distancesOrig)
    ssdOrig = np.sum(np.square(distancesOrig))
    stdOrig = np.std(distancesOrig)
    medOrig = np.median(distancesOrig)
    maxOrig = np.max(distancesOrig)
    
    distancesComb = distances[maskOrig]
    assert len(np.setdiff1d(distances[maskOrig],distancesOrig[mask])) == 0

    numComb = len(distancesComb)
    sumComb = np.sum(distancesComb)
    aveComb = np.mean(distancesComb)
    ssdComb = np.sum(np.square(distancesComb))
    stdComb = np.std(distancesComb)
    medComb = np.median(distancesComb)
    maxComb = np.max(distancesComb)

    assert maxComb <= max
    assert maxComb <= maxOrig

    keyValPairs = [(name,eval(name)) for name in ('num','sum','ave','ssd','std','med', 'max', 'numOrig','sumOrig','aveOrig','ssdOrig','stdOrig','medOrig', 'maxOrig', 'numComb','sumComb','aveComb','ssdComb','stdComb','medComb', 'maxComb')]

    if (doOutputHeader):
        print prependHeaderStr + (",".join(kvp[0].upper() for kvp in keyValPairs))

    print prependRowStr + (",".join(str(kvp[1]) for kvp in keyValPairs))

def doCreateFrangiDistanceComparisonChart(args):
    dirname = args.dirname
    basenames = args.basenames
    argDataSetName = args.arg
    valDataSetName = args.val

    plt.figure(dpi=600, figsize=(20,20))

    plt.xlim(0, 800)
    plt.ylim(0, 1)

    plt.xlabel(argDataSetName)
    plt.ylabel(valDataSetName)

    for basename,colorname in zip(basenames, ('red','blue','green','purple')):
        filename = os.path.join(dirname, basename)
        if os.path.exists(filename):
            labelname,_ = os.path.splitext(basename)
            dataset = pd.read_csv(filename)
            argDataSet = dataset[argDataSetName]
            valDataSet = dataset[valDataSetName]
            plt.scatter(argDataSet, valDataSet, color=colorname, label=labelname, marker='.')

    plt.legend(loc='upper left')

    filename = valDataSetName + 'Vs' + argDataSetName + '.png'
    filename = os.path.join(dirname, filename)
    plt.savefig(filename)

def doCreateTreeStructureH5File(args):
    dirname = args.dirname
    voxelWidth = args.voxelWidth
    
    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)
     
    positions = dataset['positions']
    indices1 = dataset['indices1']   
    indices2 = dataset['indices2']   
    radiuses = dataset['radiusPrime']
    
    positions = voxelWidth * positions

    dataset = dict()

    dataset['positions'] = positions
    dataset['indices1'] = indices1
    dataset['indices2'] = indices2
    dataset['radiuses'] = radiuses

    filename = os.path.join(dirname, 'tree_structure.h5')
    IO.writeH5File(filename, dataset)

def doConvertTubeTKFileToH5File(args):
    dirname = args.dirname
    basename = args.basename
    voxelWidth = args.voxelWidth

    filename = os.path.join(dirname, basename)
    allPoints = []    

    with open(filename, 'r') as file:
        str = file.read()
        pattern = re.compile("NPoints = (?P<NPoints>\d+)\nPoints = \n(?P<Points>(\d.*\n?)+)\n")

        for m in re.finditer(pattern, str):
            NPoints = int(m.group('NPoints'))
            Points = m.group('Points').split('\n')
            assert NPoints == len(Points)
            allPoints.extend(Points)

    pattern = re.compile("^(?P<x>{fpn})\s(?P<y>{fpn})\s(?P<z>{fpn})\s(?P<r>{fpn})\s.*\s(?P<tx>{fpn})\s(?P<ty>{fpn})\s(?P<tz>{fpn})\s(?P<a1>{fpn})\s(?P<a2>{fpn})\s(?P<a3>{fpn})\s(?P<red>\d+)\s(?P<green>\d+)\s(?P<blue>\d+)\s(?P<alpha>\d+)\s(?P<id>\d+)\s$".format(fpn="[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"))

    positions = []
    radiuses = []
    tangentLinesPoints1 = []
    tangentLinesPoints2 = []

    for str in allPoints:
        m = re.match(pattern, str)

        x = float(m.group('x')) * voxelWidth
        y = float(m.group('y')) * voxelWidth
        z = float(m.group('z')) * voxelWidth
        r = float(m.group('r')) * voxelWidth

        tx = float(m.group('tx')) * voxelWidth
        ty = float(m.group('ty')) * voxelWidth
        tz = float(m.group('tz')) * voxelWidth

        id = int(m.group('id'))

        positions.append(x)
        positions.append(y)
        positions.append(z)

        radiuses.append(r)

        tangentLinesPoints1.append(x + tx)
        tangentLinesPoints1.append(y + ty)
        tangentLinesPoints1.append(z + tz)

        tangentLinesPoints2.append(x - tx)
        tangentLinesPoints2.append(y - ty)
        tangentLinesPoints2.append(z - tz)

    filename, _ = os.path.splitext(filename)
    filename = filename + '.h5'

    dataset = dict()

    dataset['positions'] = np.array(positions, dtype=np.double)
    dataset['radiuses'] = np.array(radiuses, dtype=np.double)
    dataset['tangentLinesPoints1'] = np.array(tangentLinesPoints1, dtype=np.double)
    dataset['tangentLinesPoints2'] = np.array(tangentLinesPoints2, dtype=np.double)
    dataset['measurements'] = np.array([], dtype=np.double)
    dataset['objectnessMeasure'] = np.array([], dtype=np.double)

    IO.writeH5File(filename, dataset)

def doComputeLengthOfMinimumSpanningTree(args):
    dirname = args.dirname
    basename = args.basename
    minRadiusIncl = args.minRadiusIncl
    maxRadiusExcl = args.maxRadiusExcl

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    positions = dataset['positions']
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']
    radiuses = dataset['radiuses']

    ignor = np.logical_or(radiuses < minRadiusIncl, radiuses >= maxRadiusExcl)
    
    indices1 = indices1[~ignor]
    indices2 = indices2[~ignor]

    lengthOfMinimumSpanningTree = linalg.norm(positions[indices1] - positions[indices2], axis=1).sum()
    print lengthOfMinimumSpanningTree

def doPlotDistanceToClosestPointsAgainstLengthOfMinimumSpanningTree(args):
    dirname = args.dirname
    basename = args.basename

    filename = os.path.join(dirname, basename)

    thresholds = np.array([0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20],dtype=float)
    distancesToClosestPoints = np.array([100,90,80,70,60,50,40,30,20,10],dtype=float)
    distancesToClosestPointsStd = np.array([1,2,1,2,1,2,1,2,1,2],dtype=float)
    lengthsOfMinimumSpanningTree = np.array([303.3745022,279.3026574,257.1022683,237.4869511,220.1730432,204.0034839,189.0027731,175.2272198,162.1135914,150.3167844],dtype=float)
    lengthsOfMinimumSpanningTreeStd = np.array([5.510502504,5.796899314,5.740989963,5.669363902,5.722441026,5.620959512,5.884145791,6.00747231,6.098954469,6.020057755],dtype=float)
    
    plt.errorbar(lengthsOfMinimumSpanningTree, distancesToClosestPoints, xerr=lengthsOfMinimumSpanningTreeStd, yerr=distancesToClosestPointsStd, color='green')
    plt.show()

    pass

def analyzeResult(df, voxelSize):
    outResult = pd.DataFrame(columns=['ThresholdValue','AverageErrorInVoxelSize','LengthsRatioOfReconstructedTreeWithinGivenRadiuses','RatioOfPointsWithin05Voxel','RatioOfPointsWithin1Voxel'])

    for i, (ThresholdValue,g) in enumerate(df.groupby(['ThresholdValue'])):
        AverageErrorInVoxelSize = g.AverageError.mean() / voxelSize
        LengthsRatioOfReconstructedTreeWithinGivenRadiuses = (g.LengthOfTreeWithinGivenRadiuses / g.LengthOfGroundTruthTree).mean()
        RatioOfPointsWithin05Voxel = (g.NumberOfPointsWithin05Voxel / g.NumberOfPoints).mean()
        RatioOfPointsWithin1Voxel = (g.NumberOfPointsWithin1Voxel / g.NumberOfPoints).mean()
        
        outResult.loc[i] = (ThresholdValue,AverageErrorInVoxelSize,LengthsRatioOfReconstructedTreeWithinGivenRadiuses,RatioOfPointsWithin05Voxel,RatioOfPointsWithin1Voxel)

    return outResult

def doAnalyzeOurResultCsv(args):
    dirname   = args.dirname
    basename  = args.basename
    voxelSize = args.voxelSize
    parValue  = args.parValue

    df = pd.read_csv(os.path.join(dirname, basename))
    df = df[df.ParValue == parValue]

    outResult = analyzeResult(df, voxelSize)
    outResult.to_csv(os.path.join(dirname, 'OutResult.csv'), index=False)

def doAnalyzeTheirResultCsv(args):
    dirname   = args.dirname
    basename  = args.basename
    voxelSize = args.voxelSize

    df = pd.read_csv(os.path.join(dirname, basename))
    
    outResult = analyzeResult(df, voxelSize)
    outResult.to_csv(os.path.join(dirname, 'OutResult.csv'), index=False)

def doComputeDistanceToClosestPointsCsv(args):
    dirname = args.dirname
    basename = args.basename
    voxelWidth = args.voxelWidth
    pointsArrName = args.points
    doOutputHeader = args.doOutputHeader
    prependHeaderStr = args.prependHeaderStr
    prependRowStr = args.prependRowStr
    minRadiusIncl = args.minRadiusIncl
    maxRadiusExcl = args.maxRadiusExcl

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    positions = voxelWidth * positions

    indicesOrig1 = dataset['indices1']
    indicesOrig2 = dataset['indices2']
    radiuses = dataset['radiusPrime']

    sOrig = positions[indicesOrig1]
    tOrig = positions[indicesOrig2]

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    points = dataset[pointsArrName]
 
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    numberOfPoints = len(points)

    closestIndices = np.empty((numberOfPoints,), dtype='int32')
    closestPoints = createProjectionsOntoGroundTruthTree(sOrig, tOrig, points, closestIndices)
    closestRadiuses = radiuses[closestIndices]

    distancesSq = np.sum(np.square(closestPoints - points), axis=1)
    ignor = np.logical_or(closestRadiuses < minRadiusIncl, closestRadiuses >= maxRadiusExcl)
    distancesSq = distancesSq[~ignor]
    distanceToClosestPoints = np.sqrt(distancesSq)

    LengthOfTree                    = np.sum(linalg.norm(points[index1] - points[index2]) for index1,index2 in zip(indices1,indices2))
    LengthOfTreeWithinGivenRadiuses = np.sum(linalg.norm(points[index1] - points[index2]) for index1,index2 in zip(indices1,indices2) if not (ignor[index1] or ignor[index2]))

    NumberOfPoints = len(distanceToClosestPoints)
    sum = np.sum(distanceToClosestPoints)
    ssd = np.sum(distancesSq)
    AverageError = sum / NumberOfPoints # ave = np.mean(distances)
    msd = ssd / NumberOfPoints # np.mean(distancesSq)
    var = msd - AverageError * AverageError # var = np.var(distances)
    StandardDeviation = np.sqrt(var) # std = np.std(distances)
    Median       = np.median(distanceToClosestPoints)
    Percentile25 = np.percentile(distanceToClosestPoints, q=25)
    Percentile75 = np.percentile(distanceToClosestPoints, q=75)
    Percentile95 = np.percentile(distanceToClosestPoints, q=95)
    Percentile99 = np.percentile(distanceToClosestPoints, q=99)
    MaximumError = np.max(distanceToClosestPoints)

    NumberOfPointsWithin05Voxel = np.count_nonzero(distanceToClosestPoints <= 0.5 * voxelWidth)
    NumberOfPointsWithin1Voxel  = np.count_nonzero(distanceToClosestPoints <= voxelWidth)

    keyValPairs = [(name,eval(name)) for name in ('NumberOfPoints', 'AverageError', 'StandardDeviation', 'MaximumError', 'Median', 'Percentile25','Percentile75','Percentile95','Percentile99',
        'NumberOfPointsWithin05Voxel', 'NumberOfPointsWithin1Voxel', 'LengthOfTreeWithinGivenRadiuses','LengthOfTree',)]

    if (doOutputHeader):
        print prependHeaderStr + (",".join(kvp[0] for kvp in keyValPairs))

    print prependRowStr + (",".join(str(kvp[1]) for kvp in keyValPairs))

def resamplePoints(points, indices1, indices2, samplingStep):
    resampledPoints  = []
    resampledIndices = []

    for k,(index1,index2) in enumerate(zip(indices1, indices2)):
        point1 = points[index1]
        point2 = points[index2]
        num = int(np.ceil(linalg.norm(point1 - point2) / samplingStep) + 1)

        for i in xrange(num + 1):
            lambd  = i / float(num)
            interp = (1 - lambd) * point1 + lambd * point2
            resampledPoints.append(interp)
            resampledIndices.append(k)

    return np.array(resampledPoints),np.array(resampledIndices)

def doComputeOverlapMeasure(args):
    dirname       = args.dirname
    basename      = args.basename
    voxelSize     = args.voxelWidth
    samplingStep  = args.samplingStep
    pointsArrName = args.points

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset  = IO.readGxlFile(filename)

    positions = dataset['positions']
    positions = voxelSize * positions

    indicesOrig1 = dataset['indices1']
    indicesOrig2 = dataset['indices2']
    radiuses     = dataset['radiusPrime'] 

    filename = os.path.join(dirname, basename)
    dataset = IO.readH5File(filename)

    points = dataset[pointsArrName]
 
    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    pointsOrig,indicesOrig = resamplePoints(positions, indicesOrig1, indicesOrig2, samplingStep)
    points,indices         = resamplePoints(points, indices1, indices2, samplingStep)

    learner = NearestNeighbors(n_neighbors=1)
    learner.fit(points)

    dist,_ = learner.kneighbors(pointsOrig)
    dist   = np.ravel(dist)

    closerThanRadiusOrig = dist < radiuses[indicesOrig]
    numCloserThanRadiusOrig = np.count_nonzero(closerThanRadiusOrig) # number of points {p_i} \in GroundTruthTree such that {q_j} is closest \in ReconstructedTree and ||p_i - q_j|| < radiusAt(p_i)
    numOrig = len(pointsOrig)
    
    learner = NearestNeighbors(n_neighbors=1)
    learner.fit(pointsOrig)

    dist,closestIndices = learner.kneighbors(points)
    dist           = np.ravel(dist)
    closestIndices = np.ravel(closestIndices)

    closerThanRadius = dist < radiuses[indicesOrig[closestIndices]]

    numCloserThanRadius = np.count_nonzero(closerThanRadius) # number of points {q_j} \in ReconstructedTree such that {p_i} is closest \in GroundTruthTree and ||p_i - q_j|| < radiusAt(p_i)
    num = len(points)
    
    print '{0},{1},{2},{3},{4},{5},{6}'.format(
        numCloserThanRadiusOrig, numOrig, numCloserThanRadiusOrig / float(numOrig), 
        numCloserThanRadius, num, numCloserThanRadius / float(num),
        (numCloserThanRadiusOrig + numCloserThanRadius) / float(numOrig + num))

    pointsTP1 = []
    pointsTP2 = []

    pointsFP1 = []
    pointsFP2 = []

    for i in xrange(1, len(indicesOrig)):
        if indicesOrig[i - 1] == indicesOrig[i]:
            point1 = pointsOrig[i - 1]
            point2 = pointsOrig[i]

            if closerThanRadiusOrig[i - 1] and closerThanRadiusOrig[i]:
                pointsTP1.append(point1)
                pointsTP2.append(point2)
            else:
                pointsFP1.append(point1)
                pointsFP2.append(point2)

    polyData   = createTangentsPolyData(pointsTP1, pointsTP2)
    basename,_ = os.path.splitext(basename)
    filename = os.path.join(dirname, basename + 'TPGT.vtp')

    IO.writePolyDataFile(filename, polyData)

    polyData   = createTangentsPolyData(pointsFP1, pointsFP2)
    basename,_ = os.path.splitext(basename)
    filename = os.path.join(dirname, basename + 'FPGT.vtp')

    IO.writePolyDataFile(filename, polyData)

    pointsTP1 = []
    pointsTP2 = []

    pointsFP1 = []
    pointsFP2 = []

    for i in xrange(1, len(indices)):
        if indices[i - 1] == indices[i]:
            point1 = points[i - 1]
            point2 = points[i]

            if closerThanRadius[i - 1] and closerThanRadius[i]:
                pointsTP1.append(point1)
                pointsTP2.append(point2)
            else:
                pointsFP1.append(point1)
                pointsFP2.append(point2)

    polyData   = createTangentsPolyData(pointsTP1, pointsTP2)
    basename,_ = os.path.splitext(basename)
    filename = os.path.join(dirname, basename + 'TPRT.vtp')

    IO.writePolyDataFile(filename, polyData)

    polyData   = createTangentsPolyData(pointsFP1, pointsFP2)
    basename,_ = os.path.splitext(basename)
    filename = os.path.join(dirname, basename + 'FPRT.vtp')

    IO.writePolyDataFile(filename, polyData)

if __name__ == '__main__':
    # create the top-level parser
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers()

    # create the parser for the "doConvertRawToH5" command
    subparser = subparsers.add_parser('doConvertRawToH5')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--shape', nargs=3, type=int, default=[101,101,101])
    subparser.add_argument('--weight', type=float, default=1.0)
    subparser.add_argument('--thresholdBelow', type=float, default=0.05)
    subparser.set_defaults(func=doConvertRawToH5)

    # create the parser for the "doConvertRawToH5Ignor" command
    subparser = subparsers.add_parser('doConvertRawToH5Ignor')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--shape', nargs=3, type=int, default=[101,101,101])
    subparser.add_argument('--weight', default=1.0)
    subparser.set_defaults(func=doConvertRawToH5Ignor)

    # create the parser for the "doConvertRawToH5NoBifurc" command
    subparser = subparsers.add_parser('doConvertRawToH5NoBifurc')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doConvertRawToH5NoBifurc)

    # create the parser for the "doConvertRawToH5Responses" command
    subparser = subparsers.add_parser('doConvertRawToH5Responses')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--shape', nargs=3, type=int, default=[101,101,101])
    subparser.add_argument('--weight', default=1.0, type=float)
    subparser.set_defaults(func=doConvertRawToH5Responses)

    # create the parser for the "doCreateGraphPolyDataFile" command
    subparser = subparsers.add_parser('doCreateGraphPolyDataFile')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--points', default='positions')
    subparser.set_defaults(func=doCreateGraphPolyDataFile)

    # create the parser for the "doEMST" command
    subparser = subparsers.add_parser('doEMST')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--maxradius', default=np.inf)
    subparser.add_argument('--points', default='positions')
    subparser.set_defaults(func=doEMST)

    # create the parser for the "doArcMST" command
    subparser = subparsers.add_parser('doArcMST')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--maxradius', default=np.inf)
    subparser.set_defaults(func=doArcMST)

    # create the parser for the "doCreateArcsPolyDataFile" command
    subparser = subparsers.add_parser('doCreateArcsPolyDataFile')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doCreateArcsPolyDataFile)

    # create the parser for the "doCubicSplineMST" command
    subparser = subparsers.add_parser('doCubicSplineMST')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--maxradius', default=np.inf)
    subparser.set_defaults(func=doCubicSplineMST)

    # create the parser for the "doCreateCubicSplinePolyDataFile" command
    subparser = subparsers.add_parser('doCreateCubicSplinePolyDataFile')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doCreateCubicSplinePolyDataFile)

    # create the parser for the "doCreateTangentsPolyDataFile" command
    subparser = subparsers.add_parser('doCreateTangentsPolyDataFile')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--points', default='positions')
    subparser.set_defaults(func=doCreateTangentsPolyDataFile)

    # create the parser for the "doCreateRadiusesPolyDataFile" command
    subparser = subparsers.add_parser('doCreateRadiusesPolyDataFile')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--points', default='positions')
    subparser.set_defaults(func=doCreateRadiusesPolyDataFile)

    # create the parser for the "doROC" command
    subparser = subparsers.add_parser('doROC')
    subparser.add_argument('dirname')
    subparser.add_argument('basename1')
    subparser.add_argument('basename2')
    subparser.set_defaults(func=doROC)

    # create the parser for the "doAUC" command
    subparser = subparsers.add_parser('doAUC')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doAUC)

    # create the parser for the "doAnalyzeLabeling" command
    subparser = subparsers.add_parser('doAnalyzeLabeling')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doAnalyzeLabeling)

    # create the parser for the "doProjectionOntoSourceTreePolyDataFile"
    # command
    subparser = subparsers.add_parser('doProjectionOntoSourceTreePolyDataFile')
    subparser.add_argument('dirname')
    subparser.add_argument('targetFileBasename')
    subparser.add_argument('--positions', default='positions')
    subparser.set_defaults(func=doProjectionOntoSourceTreePolyDataFile)

    # create the parser for the "doProjectionOntoSourceTreeCsv" command
    subparser = subparsers.add_parser('doProjectionOntoSourceTreeCsv')
    subparser.add_argument('dirname')
    subparser.add_argument('targetFileBasename')
    subparser.add_argument('prependStringRow')
    subparser.add_argument('--points', default='positions')
    subparser.set_defaults(func=doProjectionOntoSourceTreeCsv)

    # create the parser for the
    # "doCreateDatabaseProjectionOntoSourceTreeCsvFile" command
    subparser = subparsers.add_parser('doCreateDatabaseProjectionOntoSourceTreeCsvFile')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('firstNum', type=int)
    subparser.add_argument('lastNum', type=int)
    subparser.set_defaults(func=doCreateDatabaseProjectionOntoSourceTreeCsvFile)

    # create the parser for the "doErrorBar" command
    subparser = subparsers.add_parser('doErrorBar')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doErrorBar)

    # create the parser for the
    # "doCreateProjectionsOntoGroundTruthTreeGraphPolyDataFile" command
    subparser = subparsers.add_parser('doCreateProjectionsOntoGroundTruthTreeGraphPolyDataFile')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--points', default='positions')
    subparser.set_defaults(func=doCreateProjectionsOntoGroundTruthTreeGraphPolyDataFile)

    # create the parser for the "doAnalyzeNonMaximumSuppressionVolumeCsv"
    # command
    subparser = subparsers.add_parser('doAnalyzeNonMaximumSuppressionVolumeCsv')
    subparser.add_argument('dirname')
    subparser.add_argument('basenameOrig')
    subparser.add_argument('basename')
    subparser.add_argument('--points', default='positions')
    subparser.add_argument('--thresholdAbove', type=float, default=0.5)
    subparser.set_defaults(func=doAnalyzeNonMaximumSuppressionVolumeCsv)

    # create the parser for the "doCreateDistanceToClosestPointsCsv" command
    subparser = subparsers.add_parser('doCreateDistanceToClosestPointsCsv')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('voxelWidth', type=float)
    subparser.add_argument('--points', default='positions')
    subparser.add_argument('--doOutputHeader', default=False, action='store_true')
    subparser.add_argument('--prependHeaderStr', default="")
    subparser.add_argument('--prependRowStr', default="")
    subparser.add_argument('--minRadiusIncl', default=0, type=float)
    subparser.add_argument('--maxRadiusExcl', default=np.inf, type=float)
    subparser.set_defaults(func=doCreateDistanceToClosestPointsCsv)

    # create the parser for the "doCreateCoDirectionalityWithClosestPointsCsv"
    # command
    subparser = subparsers.add_parser('doCreateCoDirectionalityWithClosestPointsCsv')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--points', default='positions')
    subparser.add_argument('--doOutputHeader', default=False, action='store_true')
    subparser.add_argument('--prependHeaderStr', default="")
    subparser.add_argument('--prependRowStr', default="")
    subparser.set_defaults(func=doCreateCoDirectionalityWithClosestPointsCsv)

    # create the parser for the "doCreateDistanceClosestInnerNodesCsv" command
    subparser = subparsers.add_parser('doCreateDistanceClosestInnerNodesCsv')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--points', default='positions')
    subparser.add_argument('--doOutputHeader', default=False, action='store_true')
    subparser.add_argument('--prependHeaderStr', default="")
    subparser.add_argument('--prependRowStr', default="")
    subparser.set_defaults(func=doCreateDistanceClosestInnerNodesCsv)

    # create the parser for the "doCreateFrangiDistanceComparisonChart" command
    subparser = subparsers.add_parser('doCreateFrangiDistanceComparisonChart')
    subparser.add_argument('dirname')
    subparser.add_argument('basenames', nargs='+')
    subparser.add_argument('--arg')
    subparser.add_argument('--val')
    subparser.set_defaults(func=doCreateFrangiDistanceComparisonChart)

    # create the parser for the "doCreateTreeStructureH5File" command
    subparser = subparsers.add_parser('doCreateTreeStructureH5File')
    subparser.add_argument('dirname')
    subparser.add_argument('voxelWidth', type=float)
    subparser.set_defaults(func=doCreateTreeStructureH5File)

    # create the parser for the "doConvertTubeTKFileToH5File" command
    subparser = subparsers.add_parser('doConvertTubeTKFileToH5File')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('voxelWidth', type=float)
    subparser.set_defaults(func=doConvertTubeTKFileToH5File)

    # create the parser for the "doComputeLengthOfMinimumSpanningTree" command
    subparser = subparsers.add_parser('doComputeLengthOfMinimumSpanningTree')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--minRadiusIncl', default=0, type=float)
    subparser.add_argument('--maxRadiusExcl', default=np.inf, type=float)
    subparser.set_defaults(func=doComputeLengthOfMinimumSpanningTree)

    # create the parser for the
    # "doPlotDistanceToClosestPointsAgainstLengthOfMinimumSpanningTree" command
    subparser = subparsers.add_parser('doPlotDistanceToClosestPointsAgainstLengthOfMinimumSpanningTree')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doPlotDistanceToClosestPointsAgainstLengthOfMinimumSpanningTree)

    # create the parser for the "doAnalyzeOurResultCsv" command
    subparser = subparsers.add_parser('doAnalyzeOurResultCsv')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('voxelSize', type=float)
    subparser.add_argument('parValue', type=float)
    subparser.set_defaults(func=doAnalyzeOurResultCsv)

    # create the parser for the "doAnalyzeTheirResultCsv" command
    subparser = subparsers.add_parser('doAnalyzeTheirResultCsv')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('voxelSize', type=float)
    subparser.set_defaults(func=doAnalyzeTheirResultCsv)

    # create the parser for the "doComputeDistanceToClosestPointsCsv" command
    subparser = subparsers.add_parser('doComputeDistanceToClosestPointsCsv')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('voxelWidth', type=float)
    subparser.add_argument('--points', default='positions')
    subparser.add_argument('--doOutputHeader', default=False, action='store_true')
    subparser.add_argument('--prependHeaderStr', default="")
    subparser.add_argument('--prependRowStr', default="")
    subparser.add_argument('--minRadiusIncl', default=0, type=float)
    subparser.add_argument('--maxRadiusExcl', default=np.inf, type=float)
    subparser.set_defaults(func=doComputeDistanceToClosestPointsCsv)

    # create the parser for the "doComputeOverlapMeasure" command
    subparser = subparsers.add_parser('doComputeOverlapMeasure')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('voxelWidth', type=float)
    subparser.add_argument('samplingStep', type=float)
    subparser.add_argument('--points', default='positions')
    subparser.set_defaults(func=doComputeOverlapMeasure)

    # parse the args and call whatever function was selected
    args = argparser.parse_args()
    args.func(args)
