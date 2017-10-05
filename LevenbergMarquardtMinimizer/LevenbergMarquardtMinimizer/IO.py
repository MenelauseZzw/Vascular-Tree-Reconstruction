import h5py
import numpy as np

def readH5File(filename, n_dims=3):
    dataset = dict()

    with h5py.File(filename, mode='r') as f:
        for name in ['indices1', 'indices2', 'radiuses', 'objectnessMeasure']:
            if name in f:
                dataset[name] = f[name][()]

        for name in ['positions', 'measurements', 'tangentLinesPoints1', 'tangentLinesPoints2']:
            if name in f:
                item = f[name][()]
                item = np.reshape(item, newshape=(-1,n_dims))
                dataset[name] = item

    return dataset

def writeH5File(filename, dataset):
    with h5py.File(filename, mode='w') as f:
        for name in dataset:
            item = dataset[name]
            f.create_dataset(name, data=item.flatten())
