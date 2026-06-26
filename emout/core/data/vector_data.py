"""Multi-component vector field wrappers.

:class:`VectorData` bundles two or three :class:`~emout.core.data.data.Data`
components and provides unified plotting for 2-D quiver / streamline and
3-D PyVista visualisation.
"""

from __future__ import annotations

import re
import warnings
from os import PathLike
from typing import Any, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import emout.plot.basic_plot as emplt
import emout.utils as utils
from emout.plot.animation_plot import ANIMATER_PLOT_MODE, FrameUpdater
from emout.utils import UnitTranslator
from emout.utils.util import apply_offset

from ._base import _REMOTE_PLOT_HANDLED


def _infer_component_axes(objs: List[Any], name=None) -> tuple[str, ...]:
    axes = []
    for component in objs:
        comp_name = getattr(component, "name", None)
        if not comp_name or str(comp_name)[-1] not in "xyz":
            axes = []
            break
        axes.append(str(comp_name)[-1])
    if len(axes) == len(objs) and len(set(axes)) == len(axes):
        return tuple(axes)

    if name:
        suffix = str(name)[-len(objs) :]
        if len(suffix) == len(objs) and set(suffix) <= set("xyz") and len(set(suffix)) == len(suffix):
            return tuple(suffix)

    return tuple("xyz"[: len(objs)])


class VectorData(utils.Group):
    """Multi-component vector field container.

    Wraps 2 or 3 :class:`~emout.core.data.data.Data` arrays (x, y[, z])
    and delegates axis metadata, slicing, and plotting to the underlying
    components.

    Parameters
    ----------
    objs : list of Data
        Vector components (length 2 or 3).
    name : str, optional
        Display name for the vector field.
    attrs : dict, optional
        Additional attributes inherited by the wrapper.
    """

    def __init__(self, objs: List[Any], name=None, attrs=None, component_axes=None):
        """Initialise a VectorData instance.

        Parameters
        ----------
        objs : list of Data
            Vector components (length 2 or 3).
        name : str, optional
            Display name for the vector field.
        attrs : dict, optional
            Additional attributes inherited by the wrapper.
        component_axes : tuple of {'x', 'y', 'z'}, optional
            Vector component axis for each object. Inferred from component
            names or the vector name when omitted.
        """
        if len(objs) not in (2, 3):
            raise ValueError("VectorData requires 2 or 3 components.")
        x_data = objs[0]
        y_data = objs[1]
        z_data = objs[2] if len(objs) == 3 else None

        if attrs is None:
            attrs = dict()

        if name:
            attrs["name"] = name
        elif "name" in attrs:
            pass
        elif hasattr(x_data, "name"):
            attrs["name"] = x_data.name
        else:
            attrs["name"] = ""
        if component_axes is None:
            component_axes = attrs.get("component_axes") or _infer_component_axes(objs, attrs.get("name"))
        attrs["component_axes"] = tuple(component_axes)

        super().__init__(list(objs), attrs=attrs)
        self.x_data = x_data
        self.y_data = y_data
        if z_data is not None:
            self.z_data = z_data

    def __repr__(self) -> str:
        n = len(self.objs)
        return f"<VectorData: name={self.name!r}, components={n}, shape={self.shape}>"

    @property
    def component_axes(self) -> tuple[str, ...]:
        """Return the vector component axis attached to each component."""
        return tuple(self.attrs["component_axes"])

    def _component_for_axis(self, axis: str):
        if axis not in self.component_axes:
            raise ValueError(f'axes "{axis}" cannot be used because this vector has no {axis!r} component')
        return self.objs[self.component_axes.index(axis)]

    def _require_local_data_access(self, operation: str) -> None:
        for component in self.objs:
            require = getattr(component, "_require_local_data_access", None)
            if require is None:
                continue
            target_name = getattr(component, "_target_name", None)
            target = target_name() if callable(target_name) else getattr(component, "name", self.name)
            require(operation, target)

    def _record_article_access(self, kind: str, kwargs: dict[str, Any] | None = None) -> None:
        from emout.article import record_data_access

        for component in self.objs:
            record_data_access(component, kind=kind, kwargs=kwargs)

    def _to_recipe_index(self):
        recipe = getattr(self.x_data, "_to_recipe_index", None)
        if callable(recipe):
            return tuple(recipe())
        return tuple(getattr(self.x_data, "slices", ()))

    def _get_remote_open_kwargs(self):
        remote_kwargs = getattr(self.x_data, "_emout_open_kwargs", None)
        if remote_kwargs is not None:
            return dict(remote_kwargs)
        emout_dir = getattr(self.x_data, "_emout_dir", None)
        if emout_dir is None:
            return None
        return {"directory": str(emout_dir)}

    def _local_data_access_disabled(self) -> bool:
        return any(getattr(component, "_local_data_access_disabled", lambda: False)() for component in self.objs)

    def _try_remote_plot(self, **plot_kwargs):
        """Record or dispatch vector plots when remote rendering is available."""
        remote_kwargs = self._get_remote_open_kwargs()

        from emout.distributed.remote_figure import (
            is_recording,
            record_field_plot,
            request_session,
        )

        if is_recording():
            session = request_session(remote_kwargs)
            if session is None and self._local_data_access_disabled():
                self._require_local_data_access("record a remote vector plot without a remote session")
            record_field_plot(self.name, self._to_recipe_index(), plot_kwargs, emout_kwargs=remote_kwargs)
            return _REMOTE_PLOT_HANDLED

        if not self._local_data_access_disabled():
            return None

        if remote_kwargs is None:
            self._require_local_data_access("plot vector field locally")

        from emout.distributed.remote_render import _await_remote, display_image, get_or_create_session

        session = get_or_create_session(emout_kwargs=remote_kwargs)
        if session is None:
            self._require_local_data_access("plot vector field without a remote session")

        ax = plot_kwargs.pop("ax", None)
        img = _await_remote(
            session.render_field(
                self.name,
                self._to_recipe_index(),
                emout_kwargs=remote_kwargs,
                **plot_kwargs,
            )
        )
        display_image(img, ax=ax)
        return _REMOTE_PLOT_HANDLED

    def negate(self) -> "VectorData":
        """Return a new VectorData with all components sign-flipped.

        Returns
        -------
        VectorData
            Negated copy.

        Examples
        --------
        >>> data.bxyz[-1].negate().plot()
        """
        negated = [comp.negate() for comp in self.objs]
        return VectorData(negated, name=self.name, attrs=dict(self.attrs), component_axes=self.component_axes)

    def scale(self, factor: float) -> "VectorData":
        """Return a new VectorData with all components scaled.

        Parameters
        ----------
        factor : float
            Multiplicative factor.

        Returns
        -------
        VectorData
            Scaled copy.
        """
        scaled = [comp.scale(factor) for comp in self.objs]
        return VectorData(scaled, name=self.name, attrs=dict(self.attrs), component_axes=self.component_axes)

    def __setattr__(self, key, value):
        """Set an attribute, routing component data to the internal dict.

        Parameters
        ----------
        key : str
            Attribute name
        value : object
            Value to set
        """
        if key in ("x_data", "y_data", "z_data"):
            super().__dict__[key] = value
            return
        super().__setattr__(key, value)

    @property
    def name(self) -> str:
        """Return the human-readable name of this vector field.

        Returns
        -------
        str
            Human-readable name of this vector field.
        """
        return self.attrs["name"]

    @property
    def valunit(self) -> UnitTranslator:
        """Return the unit translator for the field values.

        Returns
        -------
        UnitTranslator
            Unit translator for the field values.
        """
        return self.objs[0].valunit

    @property
    def axisunits(self) -> UnitTranslator:
        """Return per-axis unit translators.

        Returns
        -------
        list of UnitTranslator
            Per-axis unit translators ``[-1]=x, [-2]=y, [-3]=z``.
        """
        return self.objs[0].axisunits

    @property
    def slice_axes(self) -> np.ndarray:
        """Return the mapping from current array axes to original data axes.

        Returns
        -------
        np.ndarray
            Integer array mapping current array axes to original data axes.
        """
        return self.objs[0].slice_axes

    @property
    def slices(self) -> np.ndarray:
        """Return the slice objects describing the current sub-range on each axis.

        Returns
        -------
        np.ndarray
            Slice objects describing the current sub-range on each axis.
        """
        return self.objs[0].slices

    @property
    def shape(self) -> tuple:
        """Return the shape of the underlying component arrays.

        Returns
        -------
        tuple
            Shape tuple of the underlying component arrays.
        """
        return self.objs[0].shape

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions (excluding vector components)."""
        return self.objs[0].ndim

    def materialize(self) -> "VectorData":
        """Materialize lazy vector components and return a vector wrapper."""
        components = []
        changed = False
        for component in self.objs:
            materialize = getattr(component, "materialize", None)
            if callable(materialize):
                component = materialize()
                changed = True
            components.append(component)
        if not changed:
            return self
        return VectorData(components, name=self.name, attrs=dict(self.attrs), component_axes=self.component_axes)

    def to_numpy(self, stack_axis: int = 0) -> np.ndarray:
        """Return vector components stacked into a NumPy array.

        Parameters
        ----------
        stack_axis : int, default 0
            Axis used for the vector-component dimension. The default
            returns arrays shaped ``(component, ...)``.
        """
        arrays = []
        for component in self.objs:
            to_numpy = getattr(component, "to_numpy", None)
            if callable(to_numpy):
                arrays.append(to_numpy())
            else:
                self._require_local_data_access("convert vector field to a NumPy array")
                arrays.append(np.asarray(component))
        return np.stack(arrays, axis=stack_axis)

    def build_frame_updater(
        self,
        axis: int = 0,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        use_si: bool = True,
        **kwargs,
    ):
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
        """
        self._require_local_data_access("build a local vector animation frame updater")
        updater = FrameUpdater(self, axis, title, notitle, offsets, use_si, **kwargs)

        return updater

    def gifplot(
        self,
        fig: Union[plt.Figure, None] = None,
        axis: int = 0,
        action: ANIMATER_PLOT_MODE = "to_html",
        filename: PathLike = None,
        interval: int = 200,
        repeat: bool = True,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        use_si: bool = True,
        show: bool = False,
        savefilename: PathLike = None,
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
        action : ANIMATER_PLOT_MODE, optional
            Output action type
        filename : path-like, optional
            Destination file path for saving the animation
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
        show : bool, optional
            If True, display the animation interactively
        savefilename : path-like, optional
            Destination file name (deprecated alias for *filename*)
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
        if return_updater:
            warnings.warn(
                "The 'return_updater' flag is deprecated. Please use gifplot(action='frames') instead.",
                DeprecationWarning,
            )
            action = "frames"

        updater = self.build_frame_updater(axis, title, notitle, offsets, use_si, **kwargs)

        if action == "frames":
            return updater

        animator = updater.to_animator([[[updater]]])

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

    def plot(
        self,
        *args,
        **kwargs,
    ):
        """Plot vector data.

        Delegates to :meth:`plot2d` for 2-D data and :meth:`plot3d` for
        3-D data.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :meth:`plot2d`.
            Common arguments: *mode*, *axes*, *show*, *use_si*, *offsets*.
        **kwargs : dict
            Keyword arguments forwarded to :meth:`plot2d`, which in turn
            delegates to ``plot_2d_vector`` or ``plot_2d_streamline``.
            Accepts ``mesh``, ``savefilename``, ``scale``, ``scaler``,
            ``skip``, ``easy_to_read``, ``color``, ``cmap``, ``norm``,
            ``vmin``, ``vmax``, ``density``, ``figsize``, ``xlabel``,
            ``ylabel``, ``title``, ``dpi``, etc.

        Returns
        -------
        object
            Return value of the delegated plotting method.
        """
        if self.x_data.ndim == 2:
            return self.plot2d(
                *args,
                **kwargs,
            )
        if self.x_data.ndim == 3:
            return self.plot3d(
                *args,
                **kwargs,
            )
        raise NotImplementedError(f"VectorData.plot is not implemented for ndim={self.x_data.ndim}.")

    def plot2d(
        self,
        mode: Literal["stream", "vec"] = "stream",
        axes: Literal["auto", "xy", "yz", "zx", "yx", "zy", "xy"] = "auto",
        show: bool = False,
        use_si: bool = True,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        **kwargs,
    ):
        """Plot two-dimensional vector data.

        Parameters
        ----------
        mode : str
            Plot type ('vec': quiver, 'stream': streamline), by default 'stream'
        axes : str, optional
            Axis pair to plot ('xy', 'zx', etc.), by default 'auto'
        show : bool
            If True, display the plot, by default False
        use_si : bool
            If True, use SI units; otherwise use EMSES grid units,
            by default False
        offsets : tuple of (float or str), optional
            Per-axis offsets ('left', 'center', 'right', or float),
            by default None
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
            mode=mode,
            axes=axes,
            show=show,
            use_si=use_si,
            offsets=offsets,
            **kwargs,
        )
        if remote is _REMOTE_PLOT_HANDLED:
            return None
        self._require_local_data_access("plot vector field locally")
        self._record_article_access(
            "plot",
            {
                "vector": self.name,
                "mode": mode,
                "axes": axes,
                "show": show,
                "use_si": use_si,
                "offsets": offsets,
                **kwargs,
            },
        )
        if self.objs[0].valunit is None:
            use_si = False

        if axes == "auto":
            axes = "".join(sorted(self.objs[0].use_axes))

        if not re.match(r"x[yzt]|y[xzt]|z[xyt]|t[xyz]", axes):
            raise ValueError(f'axes "{axes}" cannot be used with 2D vector data')
        if axes[0] not in self.objs[0].use_axes or axes[1] not in self.objs[0].use_axes:
            raise ValueError(f'axes "{axes}" cannot be used because the axis does not exist in this data')
        if len(self.objs[0].shape) != 2:
            raise ValueError(f'axes "{axes}" cannot be used because data is not 2-dimensional')
        component1 = self._component_for_axis(axes[0])
        component2 = self._component_for_axis(axes[1])

        # x: 3, y: 2, z:1 t:0
        axis1 = self.objs[0].slice_axes[self.objs[0].use_axes.index(axes[0])]
        axis2 = self.objs[0].slice_axes[self.objs[0].use_axes.index(axes[1])]

        x = np.arange(*utils.slice2tuple(self.objs[0].slices[axis1]), dtype=float)
        y = np.arange(*utils.slice2tuple(self.objs[0].slices[axis2]), dtype=float)

        if use_si:
            xunit = self.objs[0].axisunits[axis1]
            yunit = self.objs[0].axisunits[axis2]
            valunit = self.objs[0].valunit

            x = xunit.reverse(x)
            y = yunit.reverse(y)

            _xlabel = "{} [{}]".format(axes[0], xunit.unit)
            _ylabel = "{} [{}]".format(axes[1], yunit.unit)
            _title = "{} [{}]".format(self.name, valunit.unit)

            x_data = component1.val_si
            y_data = component2.val_si
        else:
            _xlabel = axes[0]
            _ylabel = axes[1]
            _title = self.name

            x_data = component1
            y_data = component2

        if offsets is not None:
            x = apply_offset(x.astype(float), offsets[0])
            y = apply_offset(y.astype(float), offsets[1])

        kwargs["xlabel"] = kwargs.get("xlabel", None) or _xlabel
        kwargs["ylabel"] = kwargs.get("ylabel", None) or _ylabel
        kwargs["title"] = kwargs.get("title", None) or _title

        mesh = np.meshgrid(x, y)
        if "vec" in mode:
            img = emplt.plot_2d_vector(x_data, y_data, mesh=mesh, **kwargs)
        elif "stream" in mode:
            img = emplt.plot_2d_streamline(x_data, y_data, mesh=mesh, **kwargs)

        if show:
            plt.show()
            return None
        else:
            return img

    def plot_pyvista(
        self,
        mode: Literal["stream", "streamline", "vec", "quiver"] = "stream",
        show: bool = False,
        use_si: bool = True,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        plotter=None,
        filename=None,
        savefilename=None,
        **kwargs,
    ):
        """Render three-dimensional vector field with PyVista."""
        self._require_local_data_access("render vector field with PyVista locally")
        self._record_article_access(
            "plot_pyvista",
            {"vector": self.name, "mode": mode, "show": show, "use_si": use_si, "offsets": offsets, **kwargs},
        )
        if self.x_data.ndim != 3:
            raise ValueError("plot_pyvista on VectorData requires 3D component data.")
        if len(self.objs) < 3 or not hasattr(self, "z_data"):
            raise ValueError("plot_pyvista on VectorData requires 3 components (x, y, z).")
        x_component = self._component_for_axis("x")
        y_component = self._component_for_axis("y")
        z_component = self._component_for_axis("z")

        if self.objs[0].valunit is None:
            use_si = False

        if mode in ("vec", "quiver"):
            from emout.plot.pyvista_plot import plot_vector_quiver3d

            return plot_vector_quiver3d(
                x_component,
                y_component,
                z_component,
                plotter=plotter,
                use_si=use_si,
                offsets=offsets,
                show=show,
                filename=filename,
                savefilename=savefilename,
                **kwargs,
            )

        if mode in ("stream", "streamline"):
            from emout.plot.pyvista_plot import plot_vector_streamlines3d

            return plot_vector_streamlines3d(
                x_component,
                y_component,
                z_component,
                plotter=plotter,
                use_si=use_si,
                offsets=offsets,
                show=show,
                filename=filename,
                savefilename=savefilename,
                **kwargs,
            )

        raise ValueError(f'Unsupported mode "{mode}" for VectorData.plot_pyvista.')

    def plot3d_mpl(
        self,
        mode: Literal["stream", "streamline", "vec", "quiver"] = "stream",
        use_si: bool = True,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        ax=None,
        **kwargs,
    ):
        """Plot a 3-D vector field with matplotlib.

        For ``mode='stream'`` (default), field lines are traced using
        :func:`scipy.integrate.solve_ivp` and drawn on a
        :class:`mpl_toolkits.mplot3d.Axes3D`.

        Parameters
        ----------
        mode : {'stream', 'streamline', 'vec', 'quiver'}
            Plot type.
        use_si : bool, default True
            Convert to SI units when available.
        offsets : tuple of (float or str), optional
            Per-axis offsets.
        ax : Axes3D, optional
            Target axes.
        **kwargs
            Forwarded to the underlying plot function
            (e.g. ``n_seeds``, ``seed_points``, ``max_length``,
            ``step_size``, ``color``, ``cmap``, ``linewidth``).

        Returns
        -------
        Axes3D
        """
        remote = self._try_remote_plot(
            mode=mode,
            use_si=use_si,
            offsets=offsets,
            ax=ax,
            **kwargs,
        )
        if remote is _REMOTE_PLOT_HANDLED:
            return None
        self._require_local_data_access("plot vector field locally")
        self._record_article_access(
            "plot3d",
            {"vector": self.name, "mode": mode, "use_si": use_si, "offsets": offsets, **kwargs},
        )
        if self.x_data.ndim != 3:
            raise ValueError("plot3d_mpl requires 3-D component data.")
        if len(self.objs) < 3 or not hasattr(self, "z_data"):
            raise ValueError("plot3d_mpl requires 3 components (x, y, z).")
        x_component = self._component_for_axis("x")
        y_component = self._component_for_axis("y")
        z_component = self._component_for_axis("z")

        if self.objs[0].valunit is None:
            use_si = False

        if use_si:
            xunit = self.objs[0].axisunits[-1]
            yunit = self.objs[0].axisunits[-2]
            zunit = self.objs[0].axisunits[-3]

            x = xunit.reverse(np.arange(*utils.slice2tuple(self.objs[0].slices[3]), dtype=float))
            y = yunit.reverse(np.arange(*utils.slice2tuple(self.objs[0].slices[2]), dtype=float))
            z = zunit.reverse(np.arange(*utils.slice2tuple(self.objs[0].slices[1]), dtype=float))

            valunit = self.objs[0].valunit
            _xlabel = f"x [{xunit.unit}]"
            _ylabel = f"y [{yunit.unit}]"
            _zlabel = f"z [{zunit.unit}]"
            _title = f"{self.name} [{valunit.unit}]"

            x_data = x_component.val_si
            y_data = y_component.val_si
            z_data = z_component.val_si
        else:
            x = np.arange(self.x_data.shape[2], dtype=float)
            y = np.arange(self.x_data.shape[1], dtype=float)
            z = np.arange(self.x_data.shape[0], dtype=float)

            _xlabel = "x"
            _ylabel = "y"
            _zlabel = "z"
            _title = self.name

            x_data = x_component
            y_data = y_component
            z_data = z_component

        if offsets is not None:
            x = apply_offset(x.astype(float), offsets[0])
            y = apply_offset(y.astype(float), offsets[1])
            z = apply_offset(z.astype(float), offsets[2])

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        mesh = (
            np.transpose(X, (2, 1, 0)),
            np.transpose(Y, (2, 1, 0)),
            np.transpose(Z, (2, 1, 0)),
        )

        kwargs["xlabel"] = kwargs.get("xlabel") or _xlabel
        kwargs["ylabel"] = kwargs.get("ylabel") or _ylabel
        kwargs["zlabel"] = kwargs.get("zlabel") or _zlabel
        kwargs["title"] = kwargs.get("title") or _title

        if mode in ("stream", "streamline"):
            return emplt.plot_3d_streamline(
                x_data,
                y_data,
                z_data,
                ax=ax,
                mesh=mesh,
                **kwargs,
            )
        elif mode in ("vec", "quiver"):
            return emplt.plot_3d_quiver(
                x_data,
                y_data,
                z_data,
                ax3d=ax,
                mesh=mesh,
                **kwargs,
            )
        else:
            raise ValueError(f'Unsupported mode "{mode}" for plot3d_mpl.')

    def plot3d(
        self,
        mode: Literal["stream", "streamline", "vec", "quiver"] = "stream",
        backend: Literal["mpl", "pyvista"] = "pyvista",
        **kwargs,
    ):
        """Plot a 3-D vector field.

        Parameters
        ----------
        mode : {'stream', 'streamline', 'vec', 'quiver'}
            Plot type.
        backend : {'mpl', 'pyvista'}, default 'pyvista'
            ``'mpl'`` uses matplotlib 3-D axes;
            ``'pyvista'`` uses the PyVista renderer.
        **kwargs
            Forwarded to :meth:`plot3d_mpl` or :meth:`plot_pyvista`.

        Returns
        -------
        object
        """
        if backend == "pyvista":
            return self.plot_pyvista(mode=mode, **kwargs)
        return self.plot3d_mpl(mode=mode, **kwargs)


VectorData2d = VectorData
VectorData3d = VectorData
