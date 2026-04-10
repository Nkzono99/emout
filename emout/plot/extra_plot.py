"""Supplementary plotting helpers for boundary cross-sections with holes."""

from typing import Tuple, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from emout import Emout

from emout.utils.emsesinp import InpFile
from emout.utils.units import Units


def plot_surface_with_hole(data_xyz, inp, add_colorbar=True, show=False, vrange="minmax", **kwargs):
    """Render a 3-D surface with a rectangular hole.

    Internally calls `data_xyz[...].plot(mode="surf", **kwargs)` multiple
    times to draw the top, side, and bottom surfaces.

    Parameters
    ----------
    data_xyz : Data3d
        3-D data to render.
    inp : InpFile
        Input parameters containing hole geometry (`xlrechole`, etc.).
    add_colorbar : bool, optional
        Whether to add a colorbar on the last (bottom) surface plot.
    show : bool, optional
        If `True`, call `plt.show()` at the end.
    vrange : {'minmax', 'center'}, optional
        Method for determining the color-scale range.
        `'center'` uses a symmetric range centered at 0.
    **kwargs : dict
        Arguments forwarded to `Data2d.plot(mode="surf")`.
        Primarily `use_si`, `offsets`, `cmap`, `vmin`, `vmax`, `xlabel`,
        `ylabel`, `zlabel`, `title`, `ninterp`, `function`, `figsize`, `dpi`,
        and additional `mpl_toolkits.mplot3d.Axes3D.plot_surface` arguments.

    Returns
    -------
    None
    """
    xl = int(inp.xlrechole[1])
    xu = int(inp.xurechole[1])
    yl = int(inp.ylrechole[1])
    yu = int(inp.yurechole[1])
    zl = int(inp.zlrechole[1])
    zu = int(inp.zurechole[0])

    if kwargs.get("use_si", False):
        vmax = data_xyz.val_si.max()
        vmin = data_xyz.val_si.min()
    else:
        vmax = data_xyz.max()
        vmin = data_xyz.min()

    if vrange == "minmax":
        pass
    elif vrange == "center":
        vmax = max(abs(vmax), abs(vmin))
        vmin = -vmax

    kwargs["vmax"] = kwargs.get("vmax", None) or vmax
    kwargs["vmin"] = kwargs.get("vmin", None) or vmin

    # plot top surface
    data_xyz[zu, : yl + 1, :].plot(mode="surf", **kwargs)
    data_xyz[zu, yu:, :].plot(mode="surf", **kwargs)
    data_xyz[zu, yl : yu + 1, : xl + 1].plot(mode="surf", **kwargs)
    data_xyz[zu, yl : yu + 1, xu:].plot(mode="surf", **kwargs)

    # plot wall
    data_xyz[zl : zu + 1, yl, xl : xu + 1].plot(mode="surf", **kwargs)
    data_xyz[zl : zu + 1, yu, xl : xu + 1].plot(mode="surf", **kwargs)
    data_xyz[zl : zu + 1, yl : yu + 1, xl].plot(mode="surf", **kwargs)
    data_xyz[zl : zu + 1, yl : yu + 1, xu].plot(mode="surf", **kwargs)

    # plot bottom
    data_xyz[zl, yl : yu + 1, xl : xu + 1].plot(mode="surf", add_colorbar=add_colorbar, **kwargs)

    if show:
        plt.show()


def plot_hole_line(
    inp_or_data: Union[InpFile, "Emout"],
    unit: Units = None,
    use_si: bool = True,
    offsets: Tuple[int, int] = (0, 0),
    axis="xz",
    color="black",
    linewidth=None,
    fix_lims=True,
):
    """Draw a rectangular hole outline on a 2-D plane.

    Parameters
    ----------
    inp_or_data : Union[InpFile, "Emout"]
        `InpFile` or `Emout` object.
    unit : Units, optional
        Unit conversion object. Auto-detected when `inp_or_data` is `Emout`.
    use_si : bool, optional
        If `True`, draw in SI units.
    offsets : Tuple[int, int], optional
        Display offset in `(x, y)` directions.
    axis : str, optional
        Cross-section axis to draw. Currently only `"xz"` is supported.
    color : str, optional
        Line color.
    linewidth : float, optional
        Line width.
    fix_lims : bool, optional
        If `True`, fix axis limits before drawing.

    Returns
    -------
    list[matplotlib.lines.Line2D]
        Line objects returned by `plt.plot`.
    """
    if isinstance(inp_or_data, InpFile):
        inp: InpFile = inp_or_data
    else:
        data = inp_or_data
        inp = data.inp
        unit = data.unit

    if fix_lims:
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())

    xl = inp.xlrechole[1]
    xu = inp.xurechole[1]
    zl = inp.zlrechole[1] + 0.5
    zu = inp.zurechole[0]
    nx = inp.nx

    if axis == "xz":
        xs = np.array([0.0, xl, xl, xu, xu, nx - 1])
        ys = np.array([zu, zu, zl, zl, zu, zu])

    if use_si and unit is not None:
        xs = unit.length.reverse(xs)
        ys = unit.length.reverse(ys)

    im = plt.plot(xs + offsets[0], ys + offsets[1], color=color, linewidth=linewidth)

    return im


def plot_line_of_hole_half(inp, off, unit):
    """Draw auxiliary lines for a half-section hole shape on a 3-D axis.

    Parameters
    ----------
    inp : InpFile
        Input parameters containing hole definition.
    off : int
        Offset width for extended display.
    unit : UnitTranslator
        Unit conversion object.

    Returns
    -------
    None
    """
    xl = int(inp.xlrechole[1])
    xu = int(inp.xurechole[1])
    yl = int(inp.ylrechole[1])
    yu = int(inp.yurechole[1])
    zl = int(inp.zlrechole[1])
    zu = int(inp.zurechole[0])

    xc = (xl + xu) // 2
    yc = (yl + yu) // 2

    xc_si = unit.reverse(xc)
    yc_si = unit.reverse(yc)
    zu_si = unit.reverse(zu)

    ax = plt.gca()

    surf_points = np.array(
        [
            [xc, yl - off, zu],
            [xc, yl, zu],
            [xl, yl, zu],
            [xl, yu, zu],
            [xc, yu, zu],
            [xc, yu + off, zu],
            [xc - off, yu + off, zu],
            [xc - off, yl - off, zu],
            [xc, yl - off, zu],
        ]
    )

    bottom_points = np.array([[xc, yl, zl], [xc, yu, zl], [xl, yu, zl], [xl, yl, zl], [xc, yl, zl]])

    bottom2_points = np.array(
        [
            [xc, yl - off, zl - off // 2],
            [xc, yu + off, zl - off // 2],
            [xc - off, yu + off, zl - off // 2],
            [xc - off, yl - off, zl - off // 2],
            [xc, yl - off, zl - off // 2],
        ]
    )

    lines = np.array(
        [
            [[xc, yl, zl], [xc, yl, zu]],
            [[xl, yl, zl], [xl, yl, zu]],
            [[xl, yu, zl], [xl, yu, zu]],
            [[xc, yu, zl], [xc, yu, zu]],
            [[xc, yl - off, zl - off // 2], [xc, yl - off, zu]],
            [[xc, yu + off, zl - off // 2], [xc, yu + off, zu]],
            [[xc - off, yu + off, zl - off // 2], [xc - off, yu + off, zu]],
            [[xc - off, yl - off, zl - off // 2], [xc - off, yl - off, zu]],
        ]
    )

    surf_points = unit.reverse(surf_points)
    bottom_points = unit.reverse(bottom_points)
    bottom2_points = unit.reverse(bottom2_points)
    lines = unit.reverse(lines)

    def parse(points, offsets):
        """Apply offsets to a point array and return x, y, z.

        Parameters
        ----------
        points : object
            Coordinate array of shape `(N, 3)`.
        offsets : object
            Per-axis offsets.

        Returns
        -------
        object
            Offset-adjusted x, y, z arrays.
        """
        x = points[:, 0] + offsets[0]
        y = points[:, 1] + offsets[1]
        z = points[:, 2] + offsets[2]
        return x, y, z

    offsets = (-xc_si, -yc_si, -zu_si)
    ax.plot(*parse(surf_points, offsets), color="black")
    ax.plot(*parse(bottom_points, offsets), color="black")
    ax.plot(*parse(bottom2_points, offsets), color="black")

    for line in lines:
        ax.plot(*parse(line, offsets), color="black")


def plot_surface_with_hole_half(data_xyz, inp, off=10, add_colorbar=True, show=False, vrange="minmax", **kwargs):
    """Render a half cross-section surface of a rectangular hole.

    Parameters
    ----------
    data_xyz : Data3d
        3-D data to render.
    inp : InpFile
        Input parameters containing hole geometry (`xlrechole`, etc.).
    off : int, optional
        Offset width for the cross-section.
    add_colorbar : bool, optional
        Whether to add a colorbar on the last (bottom) surface plot.
    show : bool, optional
        If `True`, call `plt.show()` at the end.
    vrange : {'minmax', 'center'}, optional
        Method for determining the color-scale range.
    **kwargs : dict
        Arguments forwarded to `Data2d.plot(mode="surf")`.
        Primarily `use_si`, `offsets`, `cmap`, `vmin`, `vmax`, `xlabel`,
        `ylabel`, `zlabel`, `title`, `ninterp`, `function`, `figsize`, `dpi`,
        and additional `mpl_toolkits.mplot3d.Axes3D.plot_surface` arguments.

    Returns
    -------
    None

    Notes
    -----
    An `ax3d` is created internally and set to `kwargs["ax3d"]`.
    Any `ax3d` passed by the caller will be overwritten.
    """
    xl = int(inp.xlrechole[1])
    xu = int(inp.xurechole[1])
    yl = int(inp.ylrechole[1])
    yu = int(inp.yurechole[1])
    zl = int(inp.zlrechole[1])
    zu = int(inp.zurechole[0])

    xlen = (xu - xl) / 2 + off
    ylen = yu - yl + off * 2
    zlen = zu - zl + off

    lenmax = max(max(xlen, ylen), zlen)
    box_aspect = (xlen / lenmax, ylen / lenmax, zlen / lenmax)

    ax = plt.gcf().add_subplot(projection="3d")
    plt.sca(ax)
    ax.set_box_aspect(box_aspect)

    xc = (xl + xu) // 2

    if kwargs.get("use_si", False):
        vmax = data_xyz.val_si.max()
        vmin = data_xyz.val_si.min()
    else:
        vmax = data_xyz.max()
        vmin = data_xyz.min()

    if vrange == "minmax":
        pass
    elif vrange == "center":
        vmax = max(abs(vmax), abs(vmin))
        vmin = -vmax

    kwargs["vmax"] = kwargs.get("vmax", None) or vmax
    kwargs["vmin"] = kwargs.get("vmin", None) or vmin

    kwargs["ax3d"] = ax

    # plot top surface
    data_xyz[zu, yl - off : yl + 1, xl - off : xc + 1].plot(mode="surf", **kwargs)
    data_xyz[zu, yu : yu + off, xl - off : xc + 1].plot(mode="surf", **kwargs)
    data_xyz[zu, yl : yu + 1, xl - off : xl + 1].plot(mode="surf", **kwargs)

    # plot wall
    data_xyz[zl : zu + 1, yl, xl : xc + 1].plot(mode="surf", **kwargs)
    data_xyz[zl : zu + 1, yu, xl : xc + 1].plot(mode="surf", **kwargs)
    data_xyz[zl : zu + 1, yl : yu + 1, xl].plot(mode="surf", **kwargs)

    # innner wall
    def alltrue(x):
        """Return a mask that passes all finite values through.

        Parameters
        ----------
        x : object
            Input array or coordinate.

        Returns
        -------
        object
            Boolean mask where all values are True.
        """
        return x == x

    masked = data_xyz.masked(alltrue)

    masked[zl - off // 2 : zu + 1, yl - off : yl + 1, xc].plot(mode="surf", **kwargs)
    masked[zl - off // 2 : zu + 1, yu : yu + off, xc].plot(mode="surf", **kwargs)
    masked[zl - off // 2 : zu + 1, yl - off, xl - off : xc + 1].plot(mode="surf", **kwargs)
    masked[zl - off // 2 : zu + 1, yu + off - 1, xl - off : xc + 1].plot(mode="surf", **kwargs)
    masked[zl - off // 2 : zl + 1, yl : yu + 1, xc].plot(mode="surf", **kwargs)

    # plot bottom
    data_xyz[zl, yl : yu + 1, xl : xc + 1].plot(
        mode="surf", add_colorbar=add_colorbar, xlabel="x[m]", ylabel="y[m]", zlabel="z[m]", **kwargs
    )

    if show:
        plt.show()
