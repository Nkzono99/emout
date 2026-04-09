"""Base Data ndarray subclass for EMSES grid data.

:class:`Data` provides axis metadata, unit translators, slicing, and
plot helpers shared by all dimensioned subclasses.
"""

import re
import warnings
from os import PathLike
from pathlib import Path
from typing import Callable, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import emout.utils as utils
from emout.plot.animation_plot import ANIMATER_PLOT_MODE, FrameUpdater
from emout.utils import DataFileInfo
from emout.utils.util import apply_offset

_REMOTE_PLOT_HANDLED = object()


class Data(np.ndarray):
    """Dimensioned ndarray subclass for EMSES grid data.

    Attributes
    ----------
    datafile : DataFileInfo
        Data file metadata.
    name : str
        Field name.
    slices : list of slice
        Sub-range on each axis (t, z, y, x).
    slice_axes : list of int
        Mapping from array dimensions to original axes (0: t, 1: z, 2: y, 3: x).
    axisunits : list of UnitTranslator or None
        Per-axis unit translators.
    valunit : UnitTranslator or None
        Value unit translator.
    """

    def __new__(
        cls,
        input_array,
        filename=None,
        name=None,
        xslice=None,
        yslice=None,
        zslice=None,
        tslice=None,
        slice_axes=None,
        axisunits=None,
        valunit=None,
    ):
        """Create a new Data instance.

        Parameters
        ----------
        input_array : array_like
            Source NumPy array
        filename : path-like, optional
            Path to the originating HDF5 file
        name : str, optional
            Field name
        xslice : slice, optional
            Sub-range along the x axis
        yslice : slice, optional
            Sub-range along the y axis
        zslice : slice, optional
            Sub-range along the z axis
        tslice : slice, optional
            Sub-range along the time axis
        slice_axes : list of int, optional
            Mapping from array dimensions to original axes
            (0: t, 1: z, 2: y, 3: x)
        axisunits : list of UnitTranslator, optional
            Per-axis (t, z, y, x) unit translators
        valunit : UnitTranslator, optional
            Value unit translator

        Returns
        -------
        Data
            Newly created instance.
        """
        obj = np.asarray(input_array).view(cls)

        if obj.ndim < 4 and slice_axes is None:
            raise ValueError(
                f"Data base class requires a 4-D array (t, z, y, x), "
                f"got shape {obj.shape}. Use Data1d/Data2d/Data3d/Data4d instead."
            )

        obj.datafile = DataFileInfo(filename)
        obj.name = name

        obj.axisunits = axisunits
        obj.valunit = valunit

        if xslice is None:
            xslice = slice(0, obj.shape[3], 1)
        if yslice is None:
            yslice = slice(0, obj.shape[2], 1)
        if zslice is None:
            zslice = slice(0, obj.shape[1], 1)
        if tslice is None:
            tslice = slice(0, obj.shape[0], 1)
        if slice_axes is None:
            slice_axes = [0, 1, 2, 3]

        obj.slices = [tslice, zslice, yslice, xslice]
        obj.slice_axes = slice_axes

        return obj

    def __repr__(self):
        name = self.name or "unnamed"
        unit = self.valunit.unit if self.valunit else "raw"
        return f"<{type(self).__name__}: name={name!r}, shape={self.shape}, unit={unit}>"

    def __getitem__(self, item):
        """Retrieve an element or sub-array by index or slice.

        Parameters
        ----------
        item : int, slice, or tuple
            Index expression

        Returns
        -------
        Data or scalar
            Sliced data or scalar value.
        """
        if not isinstance(item, tuple):
            item = (item,)

        new_obj = super().__getitem__(item)

        if not isinstance(new_obj, Data):
            return new_obj

        self.__add_slices(new_obj, item)

        params = {
            "filename": new_obj.filename,
            "name": new_obj.name,
            "xslice": new_obj.xslice,
            "yslice": new_obj.yslice,
            "zslice": new_obj.zslice,
            "tslice": new_obj.tslice,
            "slice_axes": new_obj.slice_axes,
            "axisunits": new_obj.axisunits,
            "valunit": new_obj.valunit,
        }

        # Lazy imports to avoid circular dependency
        from ._data1d import Data1d
        from ._data2d import Data2d
        from ._data3d import Data3d
        from ._data4d import Data4d

        if len(new_obj.shape) == 1:
            if isinstance(new_obj, Data1d):
                return new_obj
            return Data1d(new_obj, **params)
        elif len(new_obj.shape) == 2:
            if isinstance(new_obj, Data2d):
                return new_obj
            return Data2d(new_obj, **params)
        elif len(new_obj.shape) == 3:
            if isinstance(new_obj, Data3d):
                return new_obj
            return Data3d(new_obj, **params)
        elif len(new_obj.shape) == 4:
            if isinstance(new_obj, Data4d):
                return new_obj
            return Data4d(new_obj, **params)
        else:
            return new_obj

    def __add_slices(self, new_obj, item):
        """Propagate sub-range metadata to a newly sliced object.

        Parameters
        ----------
        new_obj : Data
            Newly created data object
        item : int or slice or tuple of (int or slice)
            Index expression used for slicing
        """
        slices = [*self.slices]
        axes = [*self.slice_axes]
        for i, axis in enumerate(axes):
            if i < len(item):
                slice_obj = item[i]
            else:
                continue

            if not isinstance(slice_obj, slice):
                slice_obj = slice(slice_obj, slice_obj + 1, 1)
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
        setattr(new_obj, "slices", slices)
        setattr(new_obj, "slice_axes", axes)

    def __array_finalize__(self, obj):
        """Inherit metadata from the source array during view casting.

        Parameters
        ----------
        obj : ndarray or None
            Source array whose attributes are copied
        """
        if obj is None:
            return
        self.datafile = getattr(obj, "datafile", None)
        self.name = getattr(obj, "name", None)
        self.slices = getattr(obj, "slices", None)
        self.slice_axes = getattr(obj, "slice_axes", None)
        self.axisunits = getattr(obj, "axisunits", None)
        self.valunit = getattr(obj, "valunit", None)
        self._emout_dir = getattr(obj, "_emout_dir", None)
        self._emout_open_kwargs = getattr(obj, "_emout_open_kwargs", None)

    @property
    def filename(self) -> Path:
        """Return the source file path.

        Returns
        -------
        Path
            Path to the originating HDF5 file.
        """
        return self.datafile.filename

    @property
    def directory(self) -> Path:
        """Return the parent directory of the source file.

        Returns
        -------
        Path
            Parent directory path.
        """
        return self.datafile.directory

    @property
    def xslice(self) -> slice:
        """Return the sub-range along the x axis.

        Returns
        -------
        slice
            Sub-range along the x axis.
        """
        return self.slices[3]

    @property
    def yslice(self) -> slice:
        """Return the sub-range along the y axis.

        Returns
        -------
        slice
            Sub-range along the y axis.
        """
        return self.slices[2]

    @property
    def zslice(self) -> slice:
        """Return the sub-range along the z axis.

        Returns
        -------
        slice
            Sub-range along the z axis.
        """
        return self.slices[1]

    @property
    def tslice(self) -> slice:
        """Return the sub-range along the time axis.

        Returns
        -------
        slice
            Sub-range along the time axis.
        """
        return self.slices[0]

    def axis(self, ax: int) -> np.ndarray:
        """Return the coordinate array for the given array dimension.

        Parameters
        ----------
        ax : int
            Array dimension index

        Returns
        -------
        np.ndarray
            1-D coordinate array for the requested axis.
        """
        index = self.slice_axes[ax]
        axis_slice = self.slices[index]
        return np.arange(*utils.slice2tuple(axis_slice))

    @property
    def x(self) -> np.ndarray:
        """X-axis coordinate array in grid units.

        Returns
        -------
        np.ndarray
            X-axis coordinates.
        """
        return np.arange(*utils.slice2tuple(self.xslice))

    @property
    def y(self) -> np.ndarray:
        """Y-axis coordinate array in grid units.

        Returns
        -------
        np.ndarray
            Y-axis coordinates.
        """
        return np.arange(*utils.slice2tuple(self.yslice))

    @property
    def z(self) -> np.ndarray:
        """Z-axis coordinate array in grid units.

        Returns
        -------
        np.ndarray
            Z-axis coordinates.
        """
        return np.arange(*utils.slice2tuple(self.zslice))

    @property
    def t(self) -> np.ndarray:
        """Time-axis coordinate array in grid units.

        Returns
        -------
        np.ndarray
            Time-axis coordinates.
        """
        slc = self.tslice
        maxlen = (slc.stop - slc.start) // slc.step
        return np.array(utils.range_with_slice(self.tslice, maxlen=maxlen))

    @property
    def x_si(self) -> np.ndarray:
        """X-axis coordinate array in SI units.

        Returns
        -------
        np.ndarray
            X-axis coordinates in SI units.
        """
        return self.axisunits[3].reverse(self.x)

    @property
    def y_si(self) -> np.ndarray:
        """Y-axis coordinate array in SI units.

        Returns
        -------
        np.ndarray
            Y-axis coordinates in SI units.
        """
        return self.axisunits[2].reverse(self.y)

    @property
    def z_si(self) -> np.ndarray:
        """Z-axis coordinate array in SI units.

        Returns
        -------
        np.ndarray
            Z-axis coordinates in SI units.
        """
        return self.axisunits[1].reverse(self.z)

    @property
    def t_si(self) -> np.ndarray:
        """Time-axis coordinate array in SI units.

        Returns
        -------
        np.ndarray
            Time-axis coordinates in SI units.
        """
        return self.axisunits[0].reverse(self.t)

    @property
    def val_si(self) -> "Data":
        """Field values converted to SI units.

        Returns
        -------
        Data
            Field values in SI units.
        """
        return self.valunit.reverse(self)

    @property
    def use_axes(self) -> List[str]:
        """Return axis labels corresponding to each array dimension.

        Returns
        -------
        list of str
            Axis labels such as ``['x']``, ``['x', 'z']``, etc.
        """
        to_axis = {3: "x", 2: "y", 1: "z", 0: "t"}
        return [to_axis[a] for a in self.slice_axes]

    def masked(
        self, mask: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]
    ) -> "Data":
        """Return a copy with masked elements set to NaN.

        Parameters
        ----------
        mask : numpy.ndarray or callable
            Boolean mask array, or a callable that returns one

        Returns
        -------
        Data
            Copy of the data with masked elements replaced by NaN.
        """
        masked = self.copy()
        if isinstance(mask, np.ndarray):
            masked[mask] = np.nan
        else:
            masked[mask(masked)] = np.nan
        return masked

    def to_numpy(self) -> np.ndarray:
        """Convert to a plain NumPy ndarray."""
        return np.array(self)

    def _to_recipe_index(self):
        """Reconstruct a GridDataSeries[index]-style tuple from slices."""
        result = []
        for s in self.slices:
            if s.stop - s.start == s.step:  # single element
                result.append(s.start)
            else:
                result.append(s)
        return tuple(result)

    def _get_remote_open_kwargs(self):
        remote_kwargs = getattr(self, "_emout_open_kwargs", None)
        if remote_kwargs is not None:
            return dict(remote_kwargs)

        emout_dir = getattr(self, "_emout_dir", None)
        if emout_dir is None:
            return None
        return {"directory": str(emout_dir)}

    def _try_remote_plot(self, **plot_kwargs):
        """Record a plot command if inside remote_figure(); otherwise use data transfer mode."""
        remote_kwargs = self._get_remote_open_kwargs()

        # --- Inside remote_figure() context: record command only ---
        from emout.distributed.remote_figure import (
            is_recording,
            record_field_plot,
            request_session,
        )
        if is_recording():
            request_session(remote_kwargs)
            recipe_index = self._to_recipe_index()
            record_field_plot(self.name, recipe_index, plot_kwargs, emout_kwargs=remote_kwargs)
            return _REMOTE_PLOT_HANDLED

        # --- Outside remote_figure + Dask session available: data transfer mode ---
        if remote_kwargs is None:
            return None
        from emout.distributed.remote_render import get_or_create_session
        session = get_or_create_session(emout_kwargs=remote_kwargs)
        if session is None:
            return None

        recipe_index = self._to_recipe_index()
        payload = session.fetch_field(self.name, recipe_index, emout_kwargs=remote_kwargs).result()

        local_data = type(self)(
            payload["array"],
            name=payload["name"],
            axisunits=payload["axisunits"],
            valunit=payload["valunit"],
        )
        local_data.slices = payload["slices"]
        local_data.slice_axes = payload["slice_axes"]
        local_data._emout_dir = None  # prevent recursion
        local_data._emout_open_kwargs = None
        return local_data.plot(**plot_kwargs)

    def plot(self, **kwargs):
        """Plot the data (subclass-specific)."""
        raise NotImplementedError()

    def build_frame_updater(
        self,
        axis: int = 0,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        use_si: bool = True,
        vmin: float = None,
        vmax: float = None,
        **kwargs,
    ) -> FrameUpdater:
        """Build a frame updater for animation.

        Parameters
        ----------
        axis : int, optional
            Axis along which to animate, by default 0
        title : str, optional
            Plot title (uses the field name when None), by default None
        notitle : bool, optional
            If True, suppress the automatic title, by default False
        offsets : tuple of (float or str), optional
            Axis offsets for x, y, z ('left': start at 0, 'center': centre
            at 0, 'right': end at 0, float: shift by value), by default None
        use_si : bool
            If True, use SI units; otherwise use EMSES grid units,
            by default False
        vmin : float, optional
            Colour-map minimum, by default None
        vmax : float, optional
            Colour-map maximum, by default None
        """
        if use_si:
            vmin = vmin if vmin is not None else self.valunit.reverse(self.min())
            vmax = vmax if vmax is not None else self.valunit.reverse(self.max())
        else:
            vmin = vmin if vmin is not None else self.min()
            vmax = vmax if vmax is not None else self.max()

        updater = FrameUpdater(
            self, axis, title, notitle, offsets, use_si, vmin=vmin, vmax=vmax, **kwargs
        )

        return updater

    def gifplot(
        self,
        fig: Union[plt.Figure, None] = None,
        axis: int = 0,
        mode: str = None,
        action: ANIMATER_PLOT_MODE = "to_html",
        filename: PathLike = None,
        show: bool = False,
        savefilename: PathLike = None,
        interval: int = 200,
        repeat: bool = True,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        use_si: bool = True,
        vmin: float = None,
        vmax: float = None,
        to_html: bool = False,
        return_updater: bool = False,
        **kwargs,
    ):
        """Create and run an animation over one axis.

        Parameters
        ----------
        fig : plt.Figure or None, optional
            Target figure
        axis : int, optional
            Axis to animate along
        mode : str, optional
            Plot mode forwarded to the frame updater
        action : ANIMATER_PLOT_MODE, optional
            Output action type
        filename : path-like, optional
            Destination file path for saving the animation
        show : bool, optional
            If True, display the animation interactively
        savefilename : path-like, optional
            Destination file name (deprecated alias for *filename*)
        interval : int, optional
            Frame interval in milliseconds
        repeat : bool, optional
            If True, loop the animation
        title : str or None, optional
            Plot title
        notitle : bool, optional
            If True, suppress the automatic frame-number title
        offsets : tuple of (float or str) or None, optional
            Per-axis offsets for x, y, z
        use_si : bool, optional
            If True, use SI units
        vmin : float, optional
            Colour-map minimum
        vmax : float, optional
            Colour-map maximum
        to_html : bool, optional
            Deprecated. Equivalent to ``action='to_html'``.
        return_updater : bool, optional
            Deprecated. Equivalent to ``action='frames'``.
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying function.

        Returns
        -------
        object
            Animation object, HTML string, or FrameUpdater depending on *action*.
        """
        if to_html:
            warnings.warn(
                "The 'to_html' flag is deprecated. "
                "Please use gifplot(action='to_html') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if return_updater:
            warnings.warn(
                "The 'return_updater' flag is deprecated. "
                "Please use gifplot(action='frames') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            action = "frames"

        if mode is None:
            updater = self.build_frame_updater(
                axis, title, notitle, offsets, use_si, vmin, vmax, **kwargs
            )
        else:
            updater = self.build_frame_updater(
                axis, title, notitle, offsets, use_si, vmin, vmax, mode=mode, **kwargs
            )

        if action == "frames":
            return updater

        animator = updater.to_animator()

        return animator.plot(
            fig=fig,
            action=action,
            filename=filename,
            show=show,
            savefilename=savefilename,
            interval=interval,
            repeat=repeat,
            to_html=to_html,
        )


