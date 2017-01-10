import argparse
import IO
import MinimumSpanningTree
import matplotlib

# http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
#import maxflow
import os.path
import numpy as np
import numpy.linalg as linalg
import scipy.integrate as integrate
import vtk

from sklearn.neighbors import KDTree,radius_neighbors_graph
from xml.etree import ElementTree

def doConvertRawToH5(args):
    dirname  = args.dirname
    basename = args.basename
    weight   = args.weight

    filename = os.path.join(dirname, basename)
    dataset  = IO.readRawFile(filename, shape=(101,101,101))

    measurements        = dataset['measurements']

    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    dataset['indices1'] = indices1
    dataset['indices2'] = indices2

    weights = np.full(len(indices1), weight, dtype=np.float)

    dataset['weights']  = weights

    filename, _ = os.path.splitext(filename)
    filename    = filename + '.h5'
    
    IO.writeH5File(filename, dataset)

def doConvertRawToH5Ignor(args):
    dirname  = args.dirname
    basename = args.basename
    weight   = args.weight

    filename = os.path.join(dirname, basename)
    dataset  = IO.readRawFile(filename, shape=(101,101,101))

    measurements        = dataset['measurements']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses            = dataset['radiuses']
    responses           = dataset['responses']
 
    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    ignor = np.full_like(radiuses, True, dtype=np.bool)
    ignor[indices1] = False
    ignor[indices2] = False

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

    dataset['indices1'] = indices1
    dataset['indices2'] = indices2
    
    weights = np.full(len(indices1), weight, dtype=np.float)

    dataset['weights'] = weights

    filename, _ = os.path.splitext(filename)
    filename    = filename + '_ignor.h5'

    IO.writeH5File(filename, dataset)

def doConvertRawToH5NoBifurc(args):
    dirname  = args.dirname
    basename = args.basename

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    nodeTypes = dataset['nodeTypes']

    bifurcs  = positions[nodeTypes == 'b']
    bifurcnn = KDTree(bifurcs)

    filename = os.path.join(dirname, basename)
    dataset  = IO.readRawFile(filename, shape=(101,101,101))

    measurements        = dataset['measurements']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses            = dataset['radiuses']
    responses           = dataset['responses']
 
    dist, _ = bifurcnn.query(measurements, k=1)
    ignor   = dist[:,0] < (np.sqrt(3) + 2) / 2

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

    dataset['indices1'] = indices1
    dataset['indices2'] = indices2
    
    weights = np.full(len(indices1), 2.0, dtype=np.float)

    dataset['weights'] = weights

    filename, _ = os.path.splitext(filename)
    filename    = filename + '_nobifurc.h5'

    IO.writeH5File(filename, dataset)

def createGraphPolyData(points, indices1, indices2):
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
    return polyData

def doCreateGraphPolyDataFile(args):
    dirname   = args.dirname
    basename  = args.basename
    positions = args.positions

    filename = os.path.join(dirname, basename)
    dataset  = IO.readH5File(filename)

    positions  = dataset[positions]
    indices1   = dataset['indices1']
    indices2   = dataset['indices2']

    polyData   = createGraphPolyData(positions, indices1, indices2)
    
    filename, _ = os.path.splitext(filename)
    filename    = filename + '.vtp'
    IO.writePolyDataFile(filename, polyData)

def doEMST(args):
    dirname   = args.dirname
    basename  = args.basename
    maxradius = args.maxradius

    filename  = os.path.join(dirname, basename)
    dataset   = IO.readH5File(filename)

    positions = dataset['positions']

    n = len(positions)
    G = dict()

    for i in xrange(n):
        p = positions[i]
        for k in xrange(i+1, n):
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
    filename    = filename + 'EMST.h5'

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
    dirname   = args.dirname
    basename  = args.basename
    maxradius = args.maxradius

    filename  = os.path.join(dirname, basename)
    dataset   = IO.readH5File(filename)

    positions           = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']

    tangentLines  = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)

    n = len(positions)
    G = dict()

    for i in xrange(n):
        p  = positions[i]
        lp = tangentLines[i]

        for k in xrange(i+1, n):
            q  = positions[k]
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
    filename    = filename + 'ArcMST.h5'

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
    dirname   = args.dirname
    basename  = args.basename

    filename  = os.path.join(dirname, basename)
    dataset   = IO.readH5File(filename)

    positions           = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']

    tangentLines  = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)

    n = len(positions)

    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    appendPolyData = vtk.vtkAppendPolyData()

    for i,k in zip(indices1,indices2):
        p  = positions[i]
        q  = positions[k]

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
    filename    = filename + '.vtp'

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
    dirname   = args.dirname
    basename  = args.basename
    maxradius = args.maxradius

    filename  = os.path.join(dirname, basename)
    dataset   = IO.readH5File(filename)

    positions           = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']

    tangentLines  = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)

    n = len(positions)
    G = dict()

    for i in xrange(n):
        p  = positions[i]
        lp = tangentLines[i]
        for k in xrange(i+1, n):
            q  = positions[k]
            lq = tangentLines[k]
            dist = linalg.norm(p - q)
            if dist > maxradius: continue

            minLength = np.inf

            for lpsgn,lqsgn in ((-1,-1),(-1, 1),(1,-1),(1, 1)):
                cubicSpline  = getCubicSpline(p, dist * lpsgn * lp, q, dist * lqsgn * lq)
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
    filename    = filename + 'CubicSplineMST.h5'

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
    dirname   = args.dirname
    basename  = args.basename

    filename  = os.path.join(dirname, basename)
    dataset   = IO.readH5File(filename)

    positions           = dataset['positions']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']

    tangentLines  = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)
    
    n = len(positions)

    indices1 = dataset['indices1']
    indices2 = dataset['indices2']

    appendPolyData = vtk.vtkAppendPolyData()

    for i,k in zip(indices1, indices2):
        p  = positions[i]
        q  = positions[k]

        lp = tangentLines[i]
        lq = tangentLines[k]

        dist = linalg.norm(p - q)

        spline = None
        minLength = np.inf

        for lpsgn,lqsgn in ((-1,-1),(-1, 1),(1,-1),(1, 1)):
            cubicSpline  = getCubicSpline(p, dist * lpsgn * lp, q, dist * lqsgn * lq)
            splineLength = getSplineLength(cubicSpline, num_points=100)
            if splineLength < minLength:
                minLength = splineLength
                spline    = cubicSpline

        splinePolyData = createSplinePolyData(spline, num_points=100)
        appendPolyData.AddInputData(splinePolyData)
    
    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    filename, _ = os.path.splitext(filename)
    filename    = filename + '.vtp'

    IO.writePolyDataFile(filename, polyData)

def doConvertRawToH5Weights(args):
    dirname  = args.dirname
    basename = args.basename
    shape    = tuple(args.shape)

    filename = os.path.join(dirname, basename)
    dataset  = IO.readRawFile(filename, shape=shape)

    measurements     = dataset['measurements']
    radiuses         = dataset['radiuses']
    responses        = dataset['responses']

    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    # conn = radius_neighbors_graph(measurements, radius=(1 + np.sqrt(2)) / 2, metric='euclidean', include_self=False)
    indices1, indices2 = np.nonzero(conn)

    # Sort in ascending order
    # indices1, indices2 = zip(*sorted(zip(indices1,indices2)))

    # Sort in descending order
    # indices1, indices2 = zip(*sorted(zip(indices1,indices2), key=lambda x: x[1], reverse=True))
    # indices1, indices2 = zip(*sorted(zip(indices1,indices2), key=lambda x: x[0]))

    indices1 = np.array(indices1, dtype=np.int)
    indices2 = np.array(indices2, dtype=np.int)

    dataset['indices1'] = indices1
    dataset['indices2'] = indices2
    dataset['radiuses'] = radiuses

    # weights = np.full_like(indices1, 2.0, dtype=np.double)
    weights = 2.0 * responses[indices1] # data15
    # weights = 6.0 * responses[indices1] # 6.0 is 'better' than 9.0
    # weights = 4.0 * responses[indices2] indices1 works 'better' than indices2
    # weights = 4.0 * (responses[indices1] + responses[indices2])

    dataset['weights']  = weights

    filename, _ = os.path.splitext(filename)
    filename    = filename + '.h5'
    
    IO.writeH5File(filename, dataset)

def doROC(args):
    dirname   = args.dirname
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

    basename1,_  = os.path.splitext(basename1)
    basename2,_  = os.path.splitext(basename2)

    filename = os.path.join(dirname, basename1 + '.' + basename2 + 'ROC.png')
    fig.savefig(filename)

def doAUC(args):
    dirname  = args.dirname
    basename = args.basename

    filename = os.path.join(dirname, basename)

    sourceToTargetGraphsDistance = np.genfromtxt(filename, delimiter=',', usecols=1, skip_header=1) # source-to-target-graphs-distance
    sourceGraphsLengthRatio      = np.genfromtxt(filename, delimiter=',', usecols=2, skip_header=1) # source-graphs-length-ratio
    targetGraphsLengthRatio      = np.genfromtxt(filename, delimiter=',', usecols=3, skip_header=1) # target-graphs-length-ratio

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
    dirname   = args.dirname
    basename  = args.basename
    pointsArrName   = args.points

    filename = os.path.join(dirname, basename)
    dataset  = IO.readH5File(filename)

    positions           = dataset[pointsArrName]
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']

    tangentLines  = tangentLinesPoints2 - tangentLinesPoints1
    tangentLines /= linalg.norm(tangentLines, axis=1, keepdims=True)
    
    n = len(positions)

    appendPolyData = vtk.vtkAppendPolyData()

    for i in xrange(n):
        p  = positions[i]
        s = p + 0.5 * tangentLines[i]
        t = p - 0.5 * tangentLines[i]

        linePolyData = createLinePolyData(s,t)
        appendPolyData.AddInputData(linePolyData)
    
    appendPolyData.Update()
    polyData = appendPolyData.GetOutput()

    filename, _ = os.path.splitext(filename)
    filename    = filename + 'Tangents.vtp'
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

def doCreateCircularPolyDataFile(args):
    dirname         = args.dirname
    basename        = args.basename
    pointsArrName   = args.points
    radiusesArrName = args.radiuses

    filename = os.path.join(dirname, basename)
    dataset  = IO.readH5File(filename)

    pointsArr           = dataset[pointsArrName]
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiusesArr         = dataset[radiusesArrName]

    polyData = createCircularPolyData(pointsArr, tangentLinesPoints1, tangentLinesPoints2, radiusesArr)
    
    filename, _ = os.path.splitext(filename)
    filename    = filename + 'Circular.vtp'
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

#def doGraphCut(dirname):
#    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
#    dataset  = IO.readH5File(filename)

#    positions           = dataset['positions']
#    tangentLinesPoints1 = dataset['tangentLinesPoints1']
#    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    
#    indices1            = dataset['indices1']
#    indices2            = dataset['indices2']

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

#        p  = positions[i]
#        q  = positions[k]
#        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
#        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

#        lp = lp / linalg.norm(lp)
#        lq = lq / linalg.norm(lq)
#        dist = linalg.norm(p - q)
#        lp = lp * dist
#        lq = lq * dist
   
#        for lpsgn,lqsgn in [(-lp,-lq), (-lp, lq), (lp,-lq), (lp, lq)]:
#            spline    = createSpline(p,lpsgn, q, lqsgn)
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

#        p  = positions[i]
#        q  = positions[k]
#        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
#        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

#        lp = lp / linalg.norm(lp)
#        lq = lq / linalg.norm(lq)
#        dist = linalg.norm(p - q)
        
#        lp = lp * dist
#        lq = lq * dist

#        if g.get_segment(nodes[i]) == 1:
#            lpsgn =  lp
#        else:
#            lpsgn = -lp

#        if g.get_segment(nodes[k]) == 1:
#            lqsgn =  lq
#        else:
#            lqsgn = -lq
   
#        spline    = createSpline(p, lpsgn, q, lqsgn)
#        splineLen = splineLength(spline, num_points=100)

#        if not i in G:
#            G[i] = dict()

#        if not k in G:
#            G[k] = dict()
        
#        G[i][k] = splineLen
#        G[k][i] = splineLen

#    T = MinimumSpanningTree.MinimumSpanningTree(G)

#    for i,k in T:
#        p  = positions[i]
#        q  = positions[k]
#        lp = tangentLinesPoints2[i] - tangentLinesPoints1[i]
#        lq = tangentLinesPoints2[k] - tangentLinesPoints1[k]

#        lp = lp / linalg.norm(lp)
#        lq = lq / linalg.norm(lq)
#        dist = linalg.norm(p - q)
        
#        lp = lp * dist
#        lq = lq * dist

#        if g.get_segment(nodes[i]) == 1:
#            lpsgn =  lp
#        else:
#            lpsgn = -lp

#        if g.get_segment(nodes[k]) == 1:
#            lqsgn =  lq
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
    dirname   = args.dirname
    basename  = args.basename

    filename = os.path.join(dirname, 'tree_structure.xml')
    dataset = IO.readGxlFile(filename)

    positions = dataset['positions']
    nodeTypes = dataset['nodeTypes']

    bifurcs  = positions[nodeTypes == 'b']
    bifurcnn = KDTree(bifurcs)

    filename  = os.path.join(dirname, basename)
    dataset   = IO.readH5File(filename)

    positions     = dataset['positions']
    sourceIndices = dataset['sourceIndices']
    targetIndices = dataset['targetIndices']

    dist, _    = bifurcnn.query(positions[sourceIndices], k=1)
    nearBifurc = dist[:,0] < 1e-2

    dist = linalg.norm(positions[sourceIndices] - positions[targetIndices], axis=1)
    print np.mean(dist),np.std(dist),np.mean(dist[nearBifurc]),np.std(dist[nearBifurc]),np.count_nonzero(nearBifurc)

def projectOntoSourceTree(sOrig, tOrig, pOrig):
    s = sOrig[:,np.newaxis,:] # s.shape = (numPoints1, 1L, numDimensions)
    t = tOrig[:,np.newaxis,:] # t.shape = (numPoints1, 1L, numDimensions)
    p = pOrig[np.newaxis,:,:] # p.shape = (1L, numPoints2, numDimensions)

    sMinusT   = s - t
    sMinusTSq = np.sum(sMinusT * sMinusT, axis=2)

    sMinusP   = s - p
    sMinusPDotSMinusT = np.sum(sMinusP * sMinusT, axis=2)
    
    lambd = sMinusPDotSMinusT / sMinusTSq
    lambd = lambd[:,:,np.newaxis]
    
    # proj[i,k] is projection of point p[k] onto line between points s[i] and t[i]
    proj  = s - lambd * sMinusT

    # dist[i,k] is distance between point p[k] and line between points s[i] and t[i]
    dist  = linalg.norm(proj - p, axis=2)
    
    # ignore points which projections do not belong to corresponding intervals
    ignor = np.logical_or(lambd < 0, lambd > 1)
    ignor = ignor[:,:,0]
    
    sDist = linalg.norm(s - p, axis=2)
    tDist = linalg.norm(t - p, axis=2)

    dist = np.where(ignor, np.minimum(sDist, tDist), dist)

    # closIndex[k] is index i such that an interval between points s[i] and t[i] of source tree is the closest one to point p[k]
    closIndex = np.argmin(dist, axis=0)

    # closProj[k] is projection of point p[k] to the closest interval of source tree
    closProj  = np.array([proj[closIndex[I]][I] for I in np.ndindex(closIndex.shape)])

    # closLambd[k] is projection coefficient corresponding to closProj[k]
    closLambd = np.array([lambd[closIndex[I]][I] for I in np.ndindex(closIndex.shape)]) 

    # errors[k] = ||pOrig[k] - closProj[k]||    
    errors = linalg.norm(closProj - pOrig, axis=1)

    return (closIndex, closProj, closLambd, errors)

def doProjectionOntoSourceTreePolyDataFile(args):
    dirname            = args.dirname
    targetFileBasename = args.targetFileBasename
    positionsDataSet   = args.positions
    
    sourceFilename = os.path.join(dirname, 'tree_structure.xml')
    targetFilename = os.path.join(dirname, targetFileBasename)

    sourceDataset = IO.readGxlFile(sourceFilename)
    targetDataset = IO.readH5File(targetFilename)

    positions   = sourceDataset['positions']
    indices1    = sourceDataset['indices1']
    indices2    = sourceDataset['indices2']

    sOrig = positions[indices1]
    tOrig = positions[indices2]

    pOrig = targetDataset[positionsDataSet]

    closIndex,closProj,closLambd,errors = projectOntoSourceTree(sOrig, tOrig, pOrig)

    print 'errors.mean       = ', np.mean(errors)
    print 'errors.stdDev     = ', np.std(errors)
    print 'errors.median     = ', np.median(errors)

    numIndices = len(indices1)

    positions = []
    indices1  = []
    indices2  = []

    for index in xrange(numIndices):
        mask = np.equal(closIndex, index)

        orderedByLambda = sorted(zip(closLambd[mask], pOrig[mask]))
        if (len(orderedByLambda) == 0): continue

        orderedLambd, orderedProj = zip(*orderedByLambda)

        startIndex     = len(positions)
        numProjections = len(orderedProj)

        positions.extend(orderedProj)
        indices1.extend(xrange(startIndex, startIndex + numProjections - 1))
        indices2.extend(xrange(startIndex + 1, startIndex + numProjections))

    polyData   = createGraphPolyData(positions, indices1, indices2)
    
    filename, _ = os.path.splitext(targetFileBasename)
    filename    = os.path.join(dirname, filename + 'Opt.vtp')
    IO.writePolyDataFile(filename, polyData)

    positions = []
    positions.extend(pOrig)
    positions.extend(closProj)

    pOrigLen = len(pOrig)
    indices1 = list(xrange(0, pOrigLen))
    indices2 = list(xrange(pOrigLen, len(positions)))

    polyData   = createGraphPolyData(positions, indices1, indices2)
    
    filename, _ = os.path.splitext(targetFileBasename)
    filename    = os.path.join(dirname, filename + 'Proj.vtp')
    IO.writePolyDataFile(filename, polyData)

def doProjectionOntoSourceTreeCsv(args):
    dirname             = args.dirname
    targetFileBasename  = args.targetFileBasename
    prependStringRow    = args.prependStringRow
    pointsArrName       = args.points
    
    sourceFilename = os.path.join(dirname, 'tree_structure.xml')
    targetFilename = os.path.join(dirname, targetFileBasename)

    sourceDataset = IO.readGxlFile(sourceFilename)
    targetDataset = IO.readH5File(targetFilename)

    positions   = sourceDataset['positions']
    indices1    = sourceDataset['indices1']
    indices2    = sourceDataset['indices2']

    sOrig = positions[indices1]
    tOrig = positions[indices2]
    pOrig = targetDataset[pointsArrName]

    _,_,_,errors = projectOntoSourceTree(sOrig, tOrig, pOrig)

    errMean    = np.mean(errors)
    errStdDev  = np.std(errors)
    errMedian  = np.median(errors)
    
    err25Perc  = np.percentile(errors,q=25)
    err75Perc  = np.percentile(errors,q=75)
    err95Perc  = np.percentile(errors,q=95)
    err100Perc = np.percentile(errors,q=100)

    print '{0}{1},{2},{3},{4},{5},{6},{7}'.format(prependStringRow, errMean, errStdDev, errMedian, err25Perc, err75Perc, err95Perc, err100Perc)
    
def doErrorBar(args):
    dirname  = args.dirname
    basename = args.basename
    
    filename = os.path.join(dirname, basename)

    weight    = np.genfromtxt(filename, delimiter=',', usecols=0, skip_header=1) # weight
    errMean   = np.genfromtxt(filename, delimiter=',', usecols=1, skip_header=1) # error-mean
    errStdDev = np.genfromtxt(filename, delimiter=',', usecols=2, skip_header=1) # error-stdDev
    errMedian = np.genfromtxt(filename, delimiter=',', usecols=3, skip_header=1) # error-median

    fig = plt.figure()
    fig.set_size_inches(12, 6)

    ax  = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.0)

    ax.set_xlabel('weight')
    ax.set_ylabel('error')

    ax.errorbar(x=weight, y=errMean, yerr=errStdDev, label='mean/stdDev')
    ax.plot(weight, errMedian, label='median')
    ax.legend(loc=2)
    
    filename,_ = os.path.splitext(filename)
    fig.savefig(filename + '.png')

if __name__ == '__main__':
    # create the top-level parser
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers()

    # create the parser for the "doConvertRawToH5" command
    subparser = subparsers.add_parser('doConvertRawToH5')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--weight', default=1.0)
    subparser.set_defaults(func=doConvertRawToH5)

    # create the parser for the "doConvertRawToH5Ignor" command
    subparser = subparsers.add_parser('doConvertRawToH5Ignor')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--weight', default=1.0)
    subparser.set_defaults(func=doConvertRawToH5Ignor)

    # create the parser for the "doConvertRawToH5NoBifurc" command
    subparser = subparsers.add_parser('doConvertRawToH5NoBifurc')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doConvertRawToH5NoBifurc)

    # create the parser for the "doConvertRawToH5Weights" command
    subparser = subparsers.add_parser('doConvertRawToH5Weights')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--shape', nargs=3, type=int, default=[101,101,101])
    subparser.set_defaults(func=doConvertRawToH5Weights)

    # create the parser for the "doCreateGraphPolyDataFile" command
    subparser = subparsers.add_parser('doCreateGraphPolyDataFile')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--positions', default='positions')
    subparser.set_defaults(func=doCreateGraphPolyDataFile)

    # create the parser for the "doEMST" command
    subparser = subparsers.add_parser('doEMST')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--maxradius', default=np.inf)
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

    # create the parser for the "doCreateCircularPolyDataFile" command
    subparser = subparsers.add_parser('doCreateCircularPolyDataFile')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.add_argument('--points', default='positions')
    subparser.add_argument('--radiuses', default='radiuses')
    subparser.set_defaults(func=doCreateCircularPolyDataFile)

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

    # create the parser for the "doProjectionOntoSourceTreePolyDataFile" command
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

    # create the parser for the "doErrorBar" command
    subparser = subparsers.add_parser('doErrorBar')
    subparser.add_argument('dirname')
    subparser.add_argument('basename')
    subparser.set_defaults(func=doErrorBar)

    # parse the args and call whatever function was selected
    args = argparser.parse_args()

    args.func(args)
