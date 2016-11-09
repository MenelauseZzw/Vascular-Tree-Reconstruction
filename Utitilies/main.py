import argparse
import IO
import MinimumSpanningTree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree,radius_neighbors_graph
import os.path
from xml.etree import ElementTree

argparser = argparse.ArgumentParser()
argparser.add_argument('dirname')

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

    ignor = dist[:,0] < 2 * radiuses

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

def doConvertRawToH5(dirname):
    filename = os.path.join(dirname, 'canny2_image.raw')
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

    filename = os.path.join(dirname, 'canny2_image.h5')
    IO.writeH5File(filename, dataset)

def doPlotHist(dirname):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset  = IO.readH5File(filename)

    measurements        = dataset['measurements']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses            = dataset['radiuses']
    positions           = dataset['positions']

    indices1            = dataset['indices1']
    indices2            = dataset['indices2']

    n = len(positions)
    
    curv = [[] for _ in xrange(n)]

    for i,k in zip(indices1, indices2):
        p   = positions[i]
        q   = positions[k]
        lp  = tangentLinesPoints2[i] - tangentLinesPoints1[i]
        Cpq = getArcCenter(p, lp, q)

        arcRad = getArcRadius(p, Cpq)
        curv[i].append(1 / arcRad)

    mean   = [None for _ in xrange(n)]
    stdDev = [None for _ in xrange(n)]

    for i in xrange(n):
        if len(curv[i]) != 0:
            mean[i]   = np.mean(curv[i])
            stdDev[i] = np.std(curv[i])

    bins = np.linspace(0, 1, 100)
    plt.hist(mean, bins, alpha=0.5)
    plt.show()

def doConvertH5ToParaView(dirname):
    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.h5')
    dataset = IO.readH5File(filename)

    filename = os.path.join(dirname, 'canny2_image_nobifurc_curv.vtp')
    IO.writeParaView(filename, dataset)

if __name__ == '__main__':
    args = argparser.parse_args()
    dirname = args.dirname
   
    #doConvertRawToH5(dirname)
    #doConvertRawToH5NoBifurc(dirname)
    doPlotHist(dirname)
    #doConvertH5ToParaView(dirname)
