"""Miscellaneous helpers: regex dict, file-info parsing, interpolation, and slicing."""

import re
from pathlib import Path
from typing import Union

import numpy as np
import scipy.interpolate as interp
from matplotlib.animation import PillowWriter, writers


def interp2d(mesh, n, **kwargs):
    """Perform bilinear interpolation on a 2-D array.

    Parameters
    ----------
    mesh : object
        2-D mesh data to interpolate.
    n : object
        Upsampling factor for each axis.
    **kwargs : dict
        Additional keyword arguments forwarded to
        ``scipy.interpolate.griddata``.

    Returns
    -------
    object
        Interpolated 2-D array.
    """
    ny, nx = mesh.shape

    if (mesh == mesh[0, 0]).all():
        return np.zeros((int(ny * n), int(nx * n))) + mesh[0, 0]

    x_sparse = np.linspace(0, 1, nx)
    y_sparse = np.linspace(0, 1, ny)

    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)

    x_dense = np.linspace(0, 1, int(nx * n))
    y_dense = np.linspace(0, 1, int(ny * n))
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

    points = (X_sparse.flatten(), Y_sparse.flatten())
    value = mesh.flatten()
    points_dense = (X_dense.flatten(), Y_dense.flatten())

    mesh_dense = interp.griddata(points, value, points_dense, **kwargs)

    return mesh_dense.reshape(X_dense.shape)


def slice2tuple(slice_obj: slice):
    """Convert a slice object to a tuple.

    Parameters
    ----------
    slice_obj : slice
        Slice object.

    Returns
    -------
    (start, stop, step) : int
        Tuple containing the slice information.
    """
    start = slice_obj.start
    stop = slice_obj.stop
    step = slice_obj.step
    return (start, stop, step)


def range_with_slice(slice_obj, maxlen):
    """Create a range from a slice object.

    Parameters
    ----------
    slice_obj : slice
        Slice object.
    maxlen : int
        Maximum length (used when slice values are negative).

    Returns
    -------
    generator
        Range generator.
    """
    start = slice_obj.start or 0
    if start < 0:
        start = maxlen + start

    stop = slice_obj.stop or maxlen
    if stop < 0:
        stop = maxlen + stop

    step = slice_obj.step or 1
    return range(start, stop, step)


def apply_offset(
    line: "np.ndarray",
    offset: Union[float, str],
) -> "np.ndarray":
    """Apply a positional offset to a coordinate array.

    Parameters
    ----------
    line : numpy.ndarray
        Coordinate values to shift.
    offset : float or str
        ``"left"`` sets the first element to 0, ``"center"`` centres on
        the middle element, ``"right"`` sets the last element to 0.
        A numeric value is added directly.

    Returns
    -------
    numpy.ndarray
        Shifted array (modified in-place when possible).
    """
    flat = line.ravel()
    if offset == "left":
        line -= flat[0]
    elif offset == "center":
        line -= flat[len(flat) // 2]
    elif offset == "right":
        line -= flat[-1]
    else:
        line += offset
    return line


class RegexDict(dict):
    """Dictionary that supports regular expression keys."""

    def __getitem__(self, key):
        """Retrieve the value for a key, falling back to regex matching.

        Parameters
        ----------
        key : object
            Literal key or string to match against regex keys.
        Returns
        -------
        object
            Value associated with the matched key.
        """
        if super().__contains__(key):
            return super().__getitem__(key)

        for regex in self:
            if re.fullmatch(regex, key):
                return self[regex]

        raise KeyError()

    def __contains__(self, key):
        """Check whether a key matches any literal or regex key.

        Parameters
        ----------
        key : object
            Key to look up.
        Returns
        -------
        object
            ``True`` if the key matches.
        """
        if super().__contains__(key):
            return True

        for regex in self:
            if re.fullmatch(regex, key):
                return True

        return False

    def get(self, key, default=None):
        """Return the value for key if present, otherwise return *default*.

        Parameters
        ----------
        key : object
            Key to look up.
        default : object, optional
            Default value returned when the key is not found.
        Returns
        -------
        object
            Value associated with the key or the default.
        """
        try:
            return self[key]
        except (KeyError, IndexError):
            return default


class DataFileInfo:
    """Manage metadata about a data file."""

    def __init__(self, filename):
        """Create a data-file info object.

        Parameters
        ----------
        filename : str or Path
            File name or path.
        """
        if filename is None:
            self._filename = None
            return
        if not isinstance(filename, Path):
            filename = Path(filename)
        self._filename = filename

    @property
    def filename(self):
        """Return the file name.

        Returns
        -------
        Path
            File name.
        """
        return self._filename

    @property
    def directory(self):
        """Return the absolute path of the parent directory.

        Returns
        -------
        Path
            Absolute directory path.
        """
        if self._filename is None:
            return None
        return (self._filename / "../").resolve()

    @property
    def abspath(self):
        """Return the absolute path of the file.

        Returns
        -------
        Path or None
            Absolute file path, or ``None`` if no filename was set.
        """
        if self._filename is None:
            return None
        return self._filename.resolve()

    def __str__(self):
        """Return the string representation.

        Returns
        -------
        str
            String representation of the file path.
        """
        return str(self._filename)


@writers.register("quantized-pillow")
class QuantizedPillowWriter(PillowWriter):
    """PillowWriter wrapper that quantises each frame to 256 colours."""

    def grab_frame(self, **savefig_kwargs):
        """Grab a frame and quantise it to 256 colours.

        Parameters
        ----------
        **savefig_kwargs : dict
            Additional keyword arguments forwarded to ``savefig``.

        Returns
        -------
        None
            No return value.
        """
        super().grab_frame(**savefig_kwargs)
        self._frames[-1] = self._frames[-1].convert("RGB").quantize()


def hole_mask(inp, reverse=False):
    """Generate a boolean mask for a rectangular hole region.

    Parameters
    ----------
    inp : object
        Input parameter object providing grid and hole dimensions.
    reverse : bool, optional
        If ``True``, invert the mask before returning.
    Returns
    -------
    object
        Boolean mask array.
    """
    shape = (inp.nz + 1, inp.ny + 1, inp.nx + 1)
    xl = int(inp.xlrechole[0])
    xu = int(inp.xurechole[0])
    yl = int(inp.ylrechole[0])
    yu = int(inp.yurechole[0])
    zu = int(inp.zssurf)
    zl = int(inp.zlrechole[1])

    mask = np.ones(shape, dtype=bool)
    mask[zu:, :, :] = False
    mask[zl:zu, yl : yu + 1, xl : xu + 1] = False
    return (not reverse) == mask
