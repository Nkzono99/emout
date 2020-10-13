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

    def __len__(self):
        return len(self.index2key)


class GridData3d(np.ndarray):
    def __new__(cls, input_array, name=None, xslice=None, yslice=None, zslice=None, slice_axes=None):
        obj = np.asarray(input_array).view(cls)
        obj.name = name

        if xslice is None:
            xslice = slice(0, obj.shape[2], 1)
        if yslice is None:
            yslice = slice(0, obj.shape[1], 1)
        if zslice is None:
            zslice = slice(0, obj.shape[0], 1)
        obj.slices = [zslice, yslice, xslice]
        if slice_axes is None:
            slice_axes = [0, 1, 2]
        obj.slice_axes = slice_axes

        return obj
    
    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item, )
        
        new_obj = super().__getitem__(item)

        if isinstance(new_obj, GridData3d):
            self.__add_slices(new_obj, item)

        return new_obj
    
    def __add_slices(self, new_obj, item):
        slices = [*self.slices]
        axes = [*self.slice_axes]
        for i, axis in enumerate(axes):
            if i < len(item):
                slice_obj = item[i]
            else:
                continue

            if not isinstance(slice_obj, slice):
                slice_obj = slice(slice_obj, slice_obj+1, 1)
                axes[i] = -1

            obj_start = slice_obj.start
            obj_stop = slice_obj.stop
            obj_step = slice_obj.step

            new_start = self.slices[axis].start
            new_stop = self.slices[axis].stop
            new_step = self.slices[axis].step
    
            if obj_start is not None:
                if obj_start < 0:
                    obj_start = self.shape[i] + obj_start
                new_start += self.slices[axis].step * obj_start
        
            if slice_obj.stop is not None:
                if obj_stop < 0:
                    obj_stop = self.shape[i] + obj_stop
                new_stop = self.slices[axis].start + self.slices[axis].step * obj_stop
            
            if obj_step is not None:
                new_step *= obj_step
            
            slices[axis] = slice(new_start, new_stop, new_step)

        axes = [axis for axis in axes if axis != -1]
        setattr(new_obj, 'slices', slices)
        setattr(new_obj, 'slice_axes', axes)
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.slices = getattr(obj, 'slices', None)
        self.slice_axes = getattr(obj, 'slice_axes', None)
    
    @property
    def xslice(self):
        return self.slices[2]
    
    @property
    def yslice(self):
        return self.slices[1]
    
    @property
    def zslice(self):
        return self.slices[0]
