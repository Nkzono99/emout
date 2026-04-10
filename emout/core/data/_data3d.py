"""Three-dimensional (z, y, x) grid data container."""

from os import PathLike
from pathlib import Path
from typing import Literal, Tuple, Union

import numpy as np

from emout.utils.util import apply_offset

from ._base import Data


class Data3d(Data):
    """Three-dimensional (z, y, x) grid data container."""

    def __new__(cls, input_array, **kwargs):
        """Create a new Data3d instance.

        Parameters
        ----------
        input_array : array_like
            Source NumPy array
        **kwargs : dict
            Additional keyword arguments forwarded to ``Data.__new__``.

        Returns
        -------
        Data3d
            Newly created instance.
        """
        obj = np.asarray(input_array).view(cls)

        if obj.ndim != 3:
            raise ValueError(f"Data3d requires a 3-D array (z, y, x), got shape {obj.shape}")

        if "xslice" not in kwargs:
            kwargs["xslice"] = slice(0, obj.shape[2], 1)
        if "yslice" not in kwargs:
            kwargs["yslice"] = slice(0, obj.shape[1], 1)
        if "zslice" not in kwargs:
            kwargs["zslice"] = slice(0, obj.shape[0], 1)
        if "tslice" not in kwargs:
            kwargs["tslice"] = slice(0, 1, 1)
        if "slice_axes" not in kwargs:
            kwargs["slice_axes"] = [1, 2, 3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(
        self,
        mode: Literal["auto"] = "auto",
        use_si: bool = True,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        *args,
        **kwargs,
    ):
        """Plot three-dimensional data.

        Currently only ``mode='cont'`` is implemented, which delegates to
        :func:`emout.plot.contour3d.contour3d`.

        Parameters
        ----------
        mode : {'auto', 'cont'}, optional
            Plot mode. ``'auto'`` selects ``'cont'``.
        use_si : bool, optional
            If True, convert to SI units before plotting.
        offsets : tuple of (float or str) or None, optional
            Origin offsets ``(x, y, z)``. Strings ``'left'``,
            ``'center'``, ``'right'`` are also accepted.
        *args : tuple
            Positional arguments forwarded to ``contour3d``.
            Typically the first element is ``levels`` (a sequence of
            iso-surface values).
        **kwargs : dict
            Keyword arguments forwarded to ``contour3d``, including
            ``ax``, ``bounds_xyz``, ``roi_zyx``, ``opacity``, ``step``,
            ``title``, ``save``, ``show``, ``xlabel``, ``ylabel``,
            ``zlabel``, ``clabel``, etc.

        Returns
        -------
        tuple of (Figure, Axes) or None
            ``(fig, ax)`` for ``mode='cont'``; ``None`` for unsupported
            modes.
        """
        if mode == "auto":
            mode = "cont"

        if mode == "cont":
            from emout.plot.contour3d import contour3d

            if use_si:
                data3d = self.val_si
                dx = self.axisunits[-1].reverse(1.0)
            else:
                data3d = self
                dx = 1.0

            if offsets is not None:
                origin_xyz = (
                    apply_offset(0.0, offsets[0]),
                    apply_offset(0.0, offsets[1]),
                    apply_offset(0.0, offsets[2]),
                )
            else:
                origin_xyz = (0.0, 0.0, 0.0)

            fig, ax = contour3d(data3d, dx, origin_xyz=origin_xyz, *args, **kwargs)

            return fig, ax

    def plot_pyvista(
        self,
        mode: Literal["box", "volume", "slice", "contour"] = "box",
        use_si: bool = True,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        show: bool = False,
        plotter=None,
        cmap: str = "viridis",
        clim: Union[Tuple[float, float], None] = None,
        opacity: Union[float, str] = "sigmoid",
        contour_levels: Union[int, np.ndarray] = 8,
        add_outline: bool = True,
        outline_color: str = "white",
        add_scalar_bar: bool = True,
        **kwargs,
    ):
        """Render three-dimensional data with PyVista."""
        from emout.plot.pyvista_plot import plot_scalar_volume

        if self.valunit is None:
            use_si = False

        return plot_scalar_volume(
            self,
            mode=mode,
            plotter=plotter,
            use_si=use_si,
            offsets=offsets,
            show=show,
            cmap=cmap,
            clim=clim,
            opacity=opacity,
            contour_levels=contour_levels,
            add_outline=add_outline,
            outline_color=outline_color,
            add_scalar_bar=add_scalar_bar,
            **kwargs,
        )

    def plot3d(self, *args, **kwargs):
        """Alias for :meth:`plot_pyvista`."""
        return self.plot_pyvista(*args, **kwargs)

    def to_vtk(
        self,
        filename: PathLike,
        use_si: bool = True,
        array_name: str = None,
    ) -> Path:
        """Export to VTK ImageData format (``.vti``).

        Parameters
        ----------
        filename : path-like
            Destination file path. The ``.vti`` extension is appended
            automatically when missing.
        use_si : bool, default True
            Convert values and grid spacing to SI units.
        array_name : str, optional
            Name of the scalar array in the VTK file.
            Defaults to :attr:`name`.

        Returns
        -------
        Path
            Path to the written file.
        """
        filepath = Path(filename)
        if filepath.suffix != ".vti":
            filepath = filepath.with_suffix(".vti")

        array_name = array_name or self.name or "data"

        if use_si and self.valunit is not None:
            data = np.asarray(self.val_si, dtype=np.float64)
            dx = float(self.axisunits[-1].reverse(1.0))
            dy = float(self.axisunits[-2].reverse(1.0))
            dz = float(self.axisunits[-3].reverse(1.0))
        else:
            data = np.asarray(self, dtype=np.float64)
            dx = dy = dz = 1.0

        try:
            import pyvista as pv

            nz, ny, nx = data.shape
            grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(dx, dy, dz))
            grid.cell_data[array_name] = data.ravel(order="F")
            grid.save(str(filepath))
        except ImportError:
            _write_vti_xml(filepath, data, dx, dy, dz, array_name)

        return filepath

    def plot_surfaces(
        self,
        surfaces,
        *,
        ax=None,
        use_si: bool = True,
        vmin: Union[float, None] = None,
        vmax: Union[float, None] = None,
        **kwargs,
    ):
        """Overlay explicit mesh boundaries on a 3-D scalar field.

        Wraps ``self`` as a :class:`~emout.plot.surface_cut.Field3D` and
        passes it to :func:`emout.plot.surface_cut.plot_surfaces`.
        Designed for one-line calls such as
        ``data.phisp[-1].plot_surfaces(data.boundaries.mesh().render(), vmin=0, vmax=10)``.

        Parameters
        ----------
        surfaces
            A :class:`~emout.plot.surface_cut.RenderItem`,
            :class:`~emout.plot.surface_cut.MeshSurface3D`,
            :class:`~emout.core.boundaries.Boundary`,
            :class:`~emout.core.boundaries.BoundaryCollection`, or a
            sequence thereof.  A bare ``MeshSurface3D`` is wrapped with
            default render style.  A ``Boundary`` / ``BoundaryCollection``
            is converted via ``mesh(use_si=use_si)`` then ``render()``,
            so ``data.phisp[-1].plot_surfaces(data.boundaries)`` works
            directly.
        ax : matplotlib.axes.Axes, optional
            Target 3-D axes.  Created automatically if not supplied.
        use_si : bool, optional
            If True (default), convert data and grid spacing to SI units
            before plotting.  Also propagated to boundary mesh generation.
            Falls back to False when no unit conversion key is available.
        vmin, vmax : float, optional
            Colour-map range.
        **kwargs : dict
            Additional keyword arguments forwarded to
            :func:`~emout.plot.surface_cut.plot_surfaces` (e.g.
            ``bounds``, ``cmap_name``, ``contour_levels``).

        Returns
        -------
        tuple
            ``(cmap, norm)`` returned by ``plot_surfaces``.
        """
        # If a Dask session is running, fetch the 3-D array from the worker
        # and render locally (so ax.set_xlabel() etc. can be applied afterwards)
        remote_kwargs = self._get_remote_open_kwargs()
        if remote_kwargs is not None:
            from emout.distributed.remote_render import get_or_create_session

            session = get_or_create_session(emout_kwargs=remote_kwargs)
            if session is not None:
                recipe_index = self._to_recipe_index()
                payload = session.fetch_field(self.name, recipe_index, emout_kwargs=remote_kwargs).result()
                local_data = Data3d(
                    payload["array"],
                    name=payload["name"],
                    axisunits=payload["axisunits"],
                    valunit=payload["valunit"],
                )
                local_data.slices = payload["slices"]
                local_data.slice_axes = payload["slice_axes"]
                local_data._emout_dir = None  # prevent recursion
                local_data._emout_open_kwargs = None
                return local_data.plot_surfaces(
                    surfaces,
                    ax=ax,
                    use_si=use_si,
                    vmin=vmin,
                    vmax=vmax,
                    **kwargs,
                )

        import matplotlib.pyplot as plt

        from emout.core.boundaries import Boundary, BoundaryCollection
        from emout.plot.surface_cut import (
            Field3D,
            MeshSurface3D,
            RenderItem,
            UniformCellCenteredGrid,
            plot_surfaces as _plot_surfaces,
        )

        effective_si = bool(use_si) and getattr(self, "valunit", None) is not None

        if effective_si:
            data = np.asarray(self.val_si, dtype=np.float64)
            dx = float(self.axisunits[-1].reverse(1.0))
            dy = float(self.axisunits[-2].reverse(1.0))
            dz = float(self.axisunits[-3].reverse(1.0))
        else:
            data = np.asarray(self, dtype=np.float64)
            dx = dy = dz = 1.0

        nz, ny, nx = data.shape
        grid = UniformCellCenteredGrid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
        field = Field3D(grid, data)

        def _wrap(item):
            if isinstance(item, RenderItem):
                return item
            if isinstance(item, MeshSurface3D):
                return item.render()
            if isinstance(item, (Boundary, BoundaryCollection)):
                return item.render(use_si=effective_si)
            return item

        # A bare BoundaryCollection is iterable, but we want to treat it as a
        # single composite — match the (RenderItem, MeshSurface3D) branch.
        single_types = (RenderItem, MeshSurface3D, Boundary, BoundaryCollection)
        if isinstance(surfaces, single_types):
            items = _wrap(surfaces)
        else:
            items = [_wrap(s) for s in surfaces]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        return _plot_surfaces(
            ax,
            field=field,
            surfaces=items,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )


def _write_vti_xml(filepath, data, dx, dy, dz, array_name):
    """Write a VTK ImageData (.vti) XML file without PyVista.

    Parameters
    ----------
    filepath : Path
        Destination file path.
    data : np.ndarray
        3-D data array in (z, y, x) order.
    dx, dy, dz : float
        Grid spacing.
    array_name : str
        Scalar array name.
    """
    import base64
    import struct

    nz, ny, nx = data.shape
    flat = data.astype(np.float64).ravel(order="F")
    raw = flat.tobytes()
    nbytes = len(raw)
    header = struct.pack("<I", nbytes)
    encoded = base64.b64encode(header + raw).decode("ascii")

    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" '
        f'Origin="0 0 0" Spacing="{dx} {dy} {dz}">\n'
        f'    <Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n'
        "      <CellData>\n"
        f'        <DataArray type="Float64" Name="{array_name}" '
        f'format="binary">\n'
        f"          {encoded}\n"
        "        </DataArray>\n"
        "      </CellData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )

    with open(filepath, "w") as f:
        f.write(xml)
