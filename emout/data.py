from pathlib import Path

import h5py
import numpy as np


class Emout:
    def __init__(self, emses_dir):
        if not isinstance(emses_dir, Path):
            emses_dir = Path(emses_dir)
        self.emses_dir = emses_dir

        for h5file_path in self.emses_dir.glob('*.h5'):
            name = str(h5file_path.name).replace('00_0000.h5', '')
            setattr(self, name, GridData3dSeries(h5file_path, name))


class GridData3dSeries:
    def __init__(self, filename, name):
        self.h5 = h5py.File(filename, 'r')
        self.group = self.h5[list(self.h5.keys())[0]]
        self.index2key = {int(key): key for key in self.group.keys()}

        self.name = name
    
    def close(self):
        self.h5.close()

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError()
        if index not in self.index2key:
            raise IndexError()
        key = self.index2key[index]
        name = "{} {}".format(self.name, index)
        return GridData3d(np.array(self.group[key]), name=name)

    def __iter__(self):
        indexes = sorted(self.index2key.keys())
        for index in indexes:
            yield self[index]


class GridData3d(np.ndarray):
    def __new__(cls, input_array, name=None):
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
