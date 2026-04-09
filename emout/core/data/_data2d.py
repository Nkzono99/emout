"""Two-dimensional grid data container."""

import re
from typing import Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import emout.plot.basic_plot as emplt
import emout.utils as utils
from emout.utils.util import apply_offset

from ._base import Data, _REMOTE_PLOT_HANDLED


class Data2d(Data):
    """Two-dimensional grid data container."""

    def __new__(cls, input_array, **kwargs):
        """Create a new Data2d instance.

        Parameters
        ----------
        input_array : array_like
            Source NumPy array
        **kwargs : dict
            Additional keyword arguments forwarded to ``Data.__new__``.

        Returns
        -------
        Data2d
            Newly created instance.
        """
        obj = np.asarray(input_array).view(cls)

        if "xslice" not in kwargs:
            kwargs["xslice"] = slice(0, obj.shape[-1], 1)
        if "yslice" not in kwargs:
            kwargs["yslice"] = slice(0, obj.shape[0], 1)
        if "zslice" not in kwargs:
            kwargs["zslice"] = slice(0, 1, 1)
        if "tslice" not in kwargs:
            kwargs["tslice"] = slice(0, 1, 1)
        if "slice_axes" not in kwargs:
            kwargs["slice_axes"] = [2, 3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(
        self,
        axes: Literal["auto", "xy", "yz", "zx", "yx", "zy", "xy"] = "auto",
        show: bool = False,
        use_si: bool = True,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        mode: Literal["cm", "cm+cont", "cont"] = "cm",
        **kwargs,
    ):
        """Plot two-dimensional data.

        Parameters
        ----------
        axes : str, optional
            Axis pair to plot ('xy', 'zx', etc.), by default 'auto'
        show : bool
            If True, display the plot (suppresses file output), by default False
        use_si : bool
            If True, use SI units; otherwise use EMSES grid units,
            by default True
        offsets : tuple of (float or str), optional
            Per-axis offsets for x, y, z ('left': start at 0, 'center':
            centre at 0, 'right': end at 0, float: shift by value),
            by default None
        mode : str
            Plot type ('cm': colour map, 'cont': contour, 'surf': surface)
        **kwargs : dict
            Additional arguments forwarded to the low-level plot function
            (``plot_2dmap`` for 'cm'/'cm+cont', ``plot_2d_contour`` for
            'cont', ``plot_surface`` for 'surf').
        mesh : tuple of numpy.ndarray, optional
            Mesh grid, by default None
        savefilename : str, optional
            Output file name (None to skip saving), by default None
        cmap : matplotlib.Colormap or str or None, optional
            Colour map, by default cm.coolwarm
        vmin : float, optional
            Minimum value, by default None
        vmax : float, optional
            Maximum value, by default None
        figsize : tuple of float, optional
            Figure size, by default None
        xlabel : str, optional
            X-axis label, by default None
        ylabel : str, optional
            Y-axis label, by default None
        title : str, optional
            Title, by default None
        interpolation : str, optional
            Interpolation method, by default 'bilinear'
        dpi : int, optional
            Resolution (ignored when figsize is set), by default 10

        Returns
        -------
        AxesImage or None
            Plot image data (None when saved or shown)

        Raises
        ------
        ValueError
            If the axes parameter is invalid.
        ValueError
            If the requested axis does not exist in the data.
        ValueError
            If the data is not two-dimensional.
        """
        remote = self._try_remote_plot(
            axes=axes, show=show, use_si=use_si, offsets=offsets, mode=mode, **kwargs,
        )
        if remote is _REMOTE_PLOT_HANDLED:
            return None
        if remote is not None:
            return remote

        import emout.plot.basic_plot as emplt

        if self.valunit is None:
            use_si = False

        if axes == "auto":
            axes = "".join(sorted(self.use_axes))

        if not re.match(r"x[yzt]|y[xzt]|z[xyt]|t[xyz]", axes):
            raise ValueError(
                f'axes "{axes}" cannot be used with Data2d'
            )
        if axes[0] not in self.use_axes or axes[1] not in self.use_axes:
            raise ValueError(
                f'axes "{axes}" cannot be used because the axis does not exist in this data'
            )
        if len(self.shape) != 2:
            raise ValueError(
                f'axes "{axes}" cannot be used because data is not 2-dimensional (shape={self.shape})'
            )

        # x: 3, y: 2, z:1 t:0
        axis1 = self.slice_axes[self.use_axes.index(axes[0])]
        axis2 = self.slice_axes[self.use_axes.index(axes[1])]

        x = np.arange(*utils.slice2tuple(self.slices[axis1]))
        y = np.arange(*utils.slice2tuple(self.slices[axis2]))
        z = self if axis1 > axis2 else self.T  # transpose for 'xz' etc.

        if use_si:
            xunit = self.axisunits[axis1]
            yunit = self.axisunits[axis2]

            x = xunit.reverse(x)
            y = yunit.reverse(y)
            z = self.valunit.reverse(z)

            _xlabel = "{} [{}]".format(axes[0], xunit.unit)
            _ylabel = "{} [{}]".format(axes[1], yunit.unit)
            _title = "{} [{}]".format(self.name, self.valunit.unit)
        else:
            _xlabel = axes[0]
            _ylabel = axes[1]
            _title = self.name


        kwargs["xlabel"] = kwargs.get("xlabel", None) or _xlabel
        kwargs["ylabel"] = kwargs.get("ylabel", None) or _ylabel
        kwargs["title"] = kwargs.get("title", None) or _title

        if mode == "surf":
            mesh = np.meshgrid(x, y)

            kwargs["zlabel"] = kwargs.get("zlabel", None) or _title
            val = z
            if "x" not in self.use_axes:
                y, z = mesh
                x = self.x_si[0] if use_si else self.x[0]
                x = np.zeros_like(mesh[0]) + x
            elif "y" not in self.use_axes:
                x, z = mesh
                y = self.y_si[0] if use_si else self.y[0]
                y = np.zeros_like(mesh[0]) + y
            elif "z" not in self.use_axes:
                x, y = mesh
                z = self.z_si[0] if use_si else self.z[0]
                z = np.zeros_like(mesh[0]) + z

            if offsets is not None:
                x = apply_offset(x, offsets[0])
                y = apply_offset(y, offsets[1])
                z = apply_offset(z, offsets[2])
                val = apply_offset(val, offsets[3])

            imgs = [emplt.plot_surface(x, y, z, val, **kwargs)]
        else:
            if offsets is not None:
                x = apply_offset(x, offsets[0])
                y = apply_offset(y, offsets[1])
                z = apply_offset(z, offsets[2])
            mesh = np.meshgrid(x, y)

            imgs = []
            if "cm" in mode and "cont" in mode:
                savefilename = kwargs.get("savefilename", None)
                kwargs["savefilename"] = None
                img = emplt.plot_2dmap(z, mesh=mesh, **kwargs)
                kwargs["savefilename"] = savefilename
                img2 = emplt.plot_2d_contour(z, mesh=mesh, **kwargs)
                imgs = [img, img2]
            elif "cm" in mode:
                img = emplt.plot_2dmap(z, mesh=mesh, **kwargs)
                imgs.append(img)
            elif "cont" in mode:
                img = emplt.plot_2d_contour(z, mesh=mesh, **kwargs)
                imgs.append(img)

        if show:
            plt.show()
            return None
        else:
            return imgs[0] if len(imgs) == 1 else imgs

    def cmap(self, **kwargs):
        """Plot two-dimensional data as a colour map.

        Shortcut equivalent to ``plot(mode='cm')``.  Accepts the same
        arguments as :py:meth:`plot` except ``mode`` (raises
        :class:`TypeError` if supplied).

        Returns
        -------
        matplotlib.image.AxesImage or list or None
            Same as :py:meth:`plot`.
        """
        if "mode" in kwargs:
            raise TypeError(
                "Data2d.cmap() does not accept 'mode'; call plot(mode=...) directly instead"
            )
        return self.plot(mode="cm", **kwargs)

    def contour(self, **kwargs):
        """Plot two-dimensional data as contour lines.

        Shortcut equivalent to ``plot(mode='cont')``.  Accepts the same
        arguments as :py:meth:`plot` except ``mode`` (raises
        :class:`TypeError` if supplied).

        Returns
        -------
        matplotlib.contour.QuadContourSet or list or None
            Same as :py:meth:`plot`.
        """
        if "mode" in kwargs:
            raise TypeError(
                "Data2d.contour() does not accept 'mode'; call plot(mode=...) directly instead"
            )
        return self.plot(mode="cont", **kwargs)

    def plot_pyvista(
        self,
        use_si: bool = True,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        show: bool = False,
        plotter=None,
        cmap: str = "viridis",
        clim: Union[Tuple[float, float], None] = None,
        show_edges: bool = False,
        add_scalar_bar: bool = True,
        **kwargs,
    ):
        """Render two-dimensional data as a plane in 3-D space with PyVista."""
        from emout.plot.pyvista_plot import plot_scalar_plane

        if self.valunit is None:
            use_si = False

        return plot_scalar_plane(
            self,
            plotter=plotter,
            use_si=use_si,
            offsets=offsets,
            show=show,
            cmap=cmap,
            clim=clim,
            show_edges=show_edges,
            add_scalar_bar=add_scalar_bar,
            **kwargs,
        )

    def plot3d(self, *args, **kwargs):
        """Alias for :meth:`plot_pyvista`."""
        return self.plot_pyvista(*args, **kwargs)


