import argparse
import IO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree,radius_neighbors_graph
import os.path
from xml.etree import ElementTree

argparser = argparse.ArgumentParser()
argparser.add_argument('dirname')

def doConvertRawToH5(dirname):
    filename = os.path.join(dirname, 'frangi_image.raw')
    dataset = IO.readRaw(filename, dims=(101,101,101))

    filename = os.path.join(dirname, 'tree_structure.xml')

    treestr = ElementTree.parse(filename).getroot()
    bifurc = []

    for e in treestr.findall("./graph/node/attr[@name=' nodeType']/[string=' bifurication ']/../attr[@name=' position']/tup"):
        tup = tuple(float(item.text) for item in e.getchildren())
        bifurc.append(tup)

    measurements        = dataset['measurements']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses            = dataset['radiuses']

    bifurcnn = KDTree(bifurc)
    dist, ind = bifurcnn.query(measurements, k=1)

    ignorRad = (np.sqrt(3) + 2) / 2
    ignor = (dist < ignorRad)

    mask = ~ignor.flatten()
    measurements        = measurements[mask]
    tangentLinesPoints1 = tangentLinesPoints1[mask]
    tangentLinesPoints2 = tangentLinesPoints2[mask]
    radiuses            = radiuses[mask]

    conn = radius_neighbors_graph(measurements, radius=(np.sqrt(3) + 2) / 2, metric='euclidean', include_self=False)
    (indices1, indices2) = np.nonzero(conn)

    dataset['measurements']        = measurements
    dataset['tangentLinesPoints1'] = tangentLinesPoints1
    dataset['tangentLinesPoints2'] = tangentLinesPoints2
    dataset['radiuses']            = radiuses

    dataset['indices1'] = np.hstack((indices1, indices2))
    dataset['indices2'] = np.hstack((indices2, indices1))

    filename = os.path.join(dirname, 'frangi_image_nobifurc.h5')
    IO.writeH5(filename, dataset)

def doConvertH5ToParaView(dirname):
    filename = os.path.join(dirname, 'frangi_image_nobifurc_curv.h5')
    dataset = IO.readH5(filename)

    filename = os.path.join(dirname, 'frangi_image_nobifurc_curv.vtp')
    IO.writeParaView(filename, dataset)

if __name__ == '__main__':
    args = argparser.parse_args()
    dirname = args.dirname
    
    #doConvertRawToH5(dirname)
    doConvertH5ToParaView(dirname)