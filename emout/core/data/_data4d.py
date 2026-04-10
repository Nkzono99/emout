"""Four-dimensional (t, z, y, x) grid data container."""

from typing import Literal

import numpy as np

from ._base import Data


class Data4d(Data):
    """Four-dimensional (t, z, y, x) grid data container."""

    def __new__(cls, input_array, **kwargs):
        """Create a new Data4d instance.

        Parameters
        ----------
        input_array : array_like
            Source NumPy array
        **kwargs : dict
            Additional keyword arguments forwarded to ``Data.__new__``.

        Returns
        -------
        Data4d
            Newly created instance.
        """
        obj = np.asarray(input_array).view(cls)

        if obj.ndim != 4:
            raise ValueError(f"Data4d requires a 4-D array (t, z, y, x), got shape {obj.shape}")

        if "xslice" not in kwargs:
            kwargs["xslice"] = slice(0, obj.shape[3], 1)
        if "yslice" not in kwargs:
            kwargs["yslice"] = slice(0, obj.shape[2], 1)
        if "zslice" not in kwargs:
            kwargs["zslice"] = slice(0, obj.shape[1], 1)
        if "tslice" not in kwargs:
            kwargs["tslice"] = slice(0, obj.shape[0], 1)
        if "slice_axes" not in kwargs:
            kwargs["slice_axes"] = [0, 1, 2, 3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(self, mode: Literal["auto"] = "auto", **kwargs):
        """Plot four-dimensional data (not yet implemented).

        Parameters
        ----------
        mode : {'auto'}, optional
            Plot mode. Currently only ``'auto'`` is accepted.
        **kwargs : dict
            Reserved for future extensions; currently unused.

        Returns
        -------
        None
            Not implemented.
        """
        raise NotImplementedError("Data4d.plot() is not yet implemented.")
