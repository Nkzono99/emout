"""One-dimensional line data container."""

from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import emout.plot.basic_plot as emplt
import emout.utils as utils
from emout.utils.util import apply_offset

from ._base import Data, _REMOTE_PLOT_HANDLED


class Data1d(Data):
    """One-dimensional line data container."""

    def __new__(cls, input_array, **kwargs):
        """Create a new Data1d instance.

        Parameters
        ----------
        input_array : array_like
            Source NumPy array
        **kwargs : dict
            Additional keyword arguments forwarded to ``Data.__new__``.

        Returns
        -------
        Data1d
            Newly created instance.
        """
        obj = np.asarray(input_array).view(cls)

        if "xslice" not in kwargs:
            kwargs["xslice"] = slice(0, obj.shape[-1], 1)
        if "yslice" not in kwargs:
            kwargs["yslice"] = slice(0, 1, 1)
        if "zslice" not in kwargs:
            kwargs["zslice"] = slice(0, 1, 1)
        if "tslice" not in kwargs:
            kwargs["tslice"] = slice(0, 1, 1)
        if "slice_axes" not in kwargs:
            kwargs["slice_axes"] = [3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(
        self,
        show: bool = False,
        use_si: bool = True,
        offsets: Union[Tuple[Union[float, str], Union[float, str]], None] = None,
        **kwargs,
    ):
        """Plot one-dimensional data as a line.

        Parameters
        ----------
        show : bool
            If True, display the plot (suppresses return value), by default False
        use_si : bool
            If True, use SI units; otherwise use EMSES grid units,
            by default True
        offsets : tuple of (float or str), optional
            Offsets for the x and y axes ('left': start at 0, 'center':
            centre at 0, 'right': end at 0, float: shift by value),
            by default None
        savefilename : str, optional
            Output file name, by default None
        vmin : float, optional
            Minimum value, by default None
        vmax : float, optional
            Maximum value, by default None
        figsize : tuple of float, optional
            Figure size, by default None
        xlabel : str, optional
            Horizontal axis label, by default None
        ylabel : str, optional
            Vertical axis label, by default None
        label : str, optional
            Series label, by default None
        title : str, optional
            Title, by default None

        Returns
        -------
        Line2D or None
            Line object (None when saved or shown)

        Raises
        ------
        ValueError
            If the data is not one-dimensional.
        """
        remote = self._try_remote_plot(show=show, use_si=use_si, offsets=offsets, **kwargs)
        if remote is _REMOTE_PLOT_HANDLED:
            return None
        if remote is not None:
            return remote

        import emout.plot.basic_plot as emplt

        if self.valunit is None:
            use_si = False

        if len(self.shape) != 1:
            raise ValueError("cannot plot because data is not 1-dimensional")

        axis = self.slice_axes[0]
        x = np.arange(*utils.slice2tuple(self.slices[axis]))
        y = self

        # "EMSES Unit" to "Physical Unit"
        if use_si:
            xunit = self.axisunits[axis]

            x = xunit.reverse(x)
            y = self.valunit.reverse(y)

            _xlabel = "{} [{}]".format(self.use_axes[0], xunit.unit)
            _ylabel = "{} [{}]".format(self.name, self.valunit.unit)
        else:
            _xlabel = self.use_axes[0]
            _ylabel = self.name

        if offsets is not None:
            x = apply_offset(x, offsets[0])
            y = apply_offset(y, offsets[1])

        kwargs["xlabel"] = kwargs.get("xlabel", None) or _xlabel
        kwargs["ylabel"] = kwargs.get("ylabel", None) or _ylabel

        line = emplt.plot_line(y, x=x, **kwargs)

        if show:
            plt.show()
            return None
        else:
            return line


