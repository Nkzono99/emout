from pathlib import Path

import h5py
import numpy as np

from emout.utils import Units, Plasmainp, UnitConversionKey


class Emout:
    def __init__(self, directory, inpfilename='plasma.inp'):
        if not isinstance(directory, Path):
            directory = Path(directory)
        self.directory = directory

        for h5file_path in self.directory.glob('*.h5'):
            name = str(h5file_path.name).replace('00_0000.h5', '')
            setattr(self, name, GridDataSeries(h5file_path, name))

        if inpfilename is not None:
            self._inp = Plasmainp(directory / inpfilename)
            convkey = UnitConversionKey.load(directory / inpfilename)
            if convkey is not None:
                self._unit = Units(dx=convkey.dx, to_c=convkey.to_c)
    
    @property
    def inp(self):
        try:
            return self._inp
        except AttributeError:
            return None
    
    @property
    def unit(self):
        try:
            return self._unit
        except:
            return None


class GridDataSeries:
    def __init__(self, filename, name):
        self.h5 = h5py.File(str(filename), 'r')
        self.group = self.h5[list(self.h5.keys())[0]]
        self.index2key = {int(key): key for key in self.group.keys()}

        self.name = name

    def close(self):
        self.h5.close()

    def time_series(self, x, y, z):
        series = []
        indexes = sorted(self.index2key.keys())
        for index in indexes:
            key = self.index2key[index]
            series.append(self.group[key][z, y, x])
        return np.array(series)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError()
        if index not in self.index2key:
            raise IndexError()
        key = self.index2key[index]
        name = "{} {}".format(self.name, index)
        return GridData(np.array(self.group[key]), name=name)

    def __iter__(self):
        indexes = sorted(self.index2key.keys())
        for index in indexes:
            yield self[index]

    def __len__(self):
        return len(self.index2key)


class GridData(np.ndarray):
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

        if isinstance(new_obj, GridData):
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
                new_stop = self.slices[axis].start + \
                    self.slices[axis].step * obj_stop

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
