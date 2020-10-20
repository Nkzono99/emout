from pathlib import Path

import h5py
import numpy as np

from emout.utils import Units, Plasmainp, UnitConversionKey
import emout.plot as emplt
import emout.utils as utils
import re


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


class Data(np.ndarray):
    def __new__(cls, input_array, name=None, xslice=None, yslice=None, zslice=None, slice_axes=None):
        obj = np.asarray(input_array).view(cls)
        obj.name = name

        if xslice is None:
            xslice = slice(0, obj.shape[2], 1)
        if yslice is None:
            yslice = slice(0, obj.shape[1], 1)
        if zslice is None:
            zslice = slice(0, obj.shape[0], 1)
        if slice_axes is None:
            slice_axes = [0, 1, 2]

        obj.slices = [zslice, yslice, xslice]
        obj.slice_axes = slice_axes

        return obj

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item, )

        new_obj = super().__getitem__(item)

        if not isinstance(new_obj, Data):
            return new_obj

        self.__add_slices(new_obj, item)

        if len(new_obj.shape) == 1:
            if isinstance(new_obj, LineData):
                return new_obj
            return LineData(new_obj,
                            name=new_obj.name,
                            xslice=new_obj.xslice,
                            yslice=new_obj.yslice,
                            zslice=new_obj.zslice,
                            slice_axes=new_obj.slice_axes)
        elif len(new_obj.shape) == 2:
            if isinstance(new_obj, SlicedData):
                return new_obj
            return SlicedData(new_obj,
                              name=new_obj.name,
                              xslice=new_obj.xslice,
                              yslice=new_obj.yslice,
                              zslice=new_obj.zslice,
                              slice_axes=new_obj.slice_axes)
        elif len(new_obj.shape) == 3:
            if isinstance(new_obj, GridData):
                return new_obj
            return GridData(new_obj,
                            name=new_obj.name,
                            xslice=new_obj.xslice,
                            yslice=new_obj.yslice,
                            zslice=new_obj.zslice,
                            slice_axes=new_obj.slice_axes)
        else:
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

    @ property
    def xslice(self):
        return self.slices[2]

    @ property
    def yslice(self):
        return self.slices[1]

    @ property
    def zslice(self):
        return self.slices[0]

    @ property
    def use_axes(self):
        to_axis = {2: 'x', 1: 'y', 0: 'z'}
        return list(map(lambda a: to_axis[a], self.slice_axes))

    def plot(self, **kwargs):
        raise NotImplementedError()


class GridData(Data):
    def plot(mode='auto', x=None, y=None, z=None, **kwargs):
        if mode == 'auto':
            mode = ''.join(sorted(self.use_axes))
        pass


class SlicedData(Data):
    def plot(self, mode='auto', **kwargs):
        if mode == 'auto':
            mode = ''.join(sorted(self.use_axes))

        if not re.match(r'x[yz]|y[xz]|z[xy]', mode):
            raise Exception(
                'Error: mode "{mode}" cannot be used with SlicedData'.format(mode=mode))
        if mode[0] not in self.use_axes or mode[1] not in self.use_axes:
            raise Exception(
                'Error: mode "{mode}" cannot be used because {mode}-axis does not exist in this data.'.format(mode=mode))
        if len(self.shape) != 2:
            raise Exception(
                'Error: mode "{mode}" cannot be used because data is not 2dim shape.'.format(mode=mode))

        # x: 2, y: 1, z:0
        axis1 = self.slice_axes[self.use_axes.index(mode[0])]
        axis2 = self.slice_axes[self.use_axes.index(mode[1])]
        x = np.arange(*utils.slice2tuple(self.slices[axis1]))
        y = np.arange(*utils.slice2tuple(self.slices[axis2]))

        kwargs['xlabel'] = kwargs.get('xlabel', None) or mode[0]
        kwargs['ylabel'] = kwargs.get('ylabel', None) or mode[1]
        kwargs['title'] = kwargs.get('title', None) or self.name

        mesh = np.meshgrid(x, y)
        if axis1 > axis2:
            emplt.plot_2dmap(self, mesh=mesh, **kwargs)
        else:
            emplt.plot_2dmap(self.T, mesh=mesh, **kwargs)


class LineData(Data):
    def plot(self, mode='auto', **kwargs):
        if mode == 'auto':
            mode = ''.join(sorted(self.use_axes))

        if not re.match(r'[xyz]', mode):
            raise Exception(
                'Error: mode "{mode}" cannot be used with LineData'.format(mode=mode))
        if mode not in self.use_axes:
            raise Exception(
                'Error: mode "{mode}" cannot be used because {mode}-axis does not exist in this data.'.format(mode=mode))
        if len(self.shape) != 1:
            raise Exception(
                'Error: mode "{mode}" cannot be used because data is not 1dim shape.'.format(mode=mode))

        axis = self.slice_axes[0]
        horizon_data = np.arange(*utils.slice2tuple(self.slices[axis]))

        kwargs['xlabel'] = kwargs.get('xlabel', None) or mode
        kwargs['ylabel'] = kwargs.get('ylabel', None) or self.name

        emplt.plot_line(self, x=horizon_data, **kwargs)
