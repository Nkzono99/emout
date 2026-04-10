"""Low-level 2-D plotting helpers for scalar and vector fields.

Functions in this module accept raw numpy arrays plus axis / unit
metadata and produce matplotlib figures.  They are called by the
:meth:`Data.plot` and :meth:`VectorData.plot` convenience methods.
"""

import copy

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import emout.utils as utils

_r = 0.9
_d = 0.5
mycmap = mcolors.LinearSegmentedColormap(
    "gray-jet",
    {
        "red": (
            (0.00, 0.2, 0.2),
            (_d * (1 - _r), 0.3, 0.3),
            (0.35 * _r + (1 - _r), 0, 0),
            (0.66 * _r + (1 - _r), 1, 1),
            (0.89 * _r + (1 - _r), 1, 1),
            (1.00, 0.5, 0.5),
        ),
        "green": (
            (0.00, 0.2, 0.2),
            (_d * (1 - _r), 0.3, 0.3),
            (0.125 * _r + (1 - _r), 0, 0),
            (0.375 * _r + (1 - _r), 1, 1),
            (0.640 * _r + (1 - _r), 1, 1),
            (0.910 * _r + (1 - _r), 0, 0),
            (1.000, 0, 0),
        ),
        "blue": (
            (0.00, 0.2, 0.2),
            (_d * (1 - _r), 0.3, 0.3),
            (0.00 * _r + (1 - _r), 0.5, 0.5),
            (0.11 * _r + (1 - _r), 1, 1),
            (0.34 * _r + (1 - _r), 1, 1),
            (0.65 * _r + (1 - _r), 0, 0),
            (1.00, 0, 0),
        ),
    },
)


def figsize_with_2d(data2d, dpi=10):
    """Compute figure size from 2-D data shape.

    Parameters
    ----------
    data2d : numpy.ndarray
        2-D data array.
    dpi : int, optional
        Pixels per data point, by default 10.

    Returns
    -------
    (float, float)
        Figure size in inches.
    """
    px = 1 / plt.rcParams["figure.dpi"] * dpi
    figsize = (data2d.shape[1] * px, data2d.shape[0] * px)
    return figsize


def plot_2dmap(
    data2d,
    mesh=None,
    savefilename=None,
    cmap=mycmap,
    mask_color="gray",
    vmin=None,
    vmax=None,
    figsize=None,
    xlabel=None,
    ylabel=None,
    title=None,
    interpolation="bilinear",
    dpi=10,
    colorbar_label="",
    cbargs=None,
    add_colorbar=True,
    **kwargs,
):
    """Plot a 2-D colormap.

    Parameters
    ----------
    data2d : numpy.ndarray
        2-D data array.
    mesh : (numpy.ndarray, numpy.ndarray), optional
        Mesh grid, by default None.
    savefilename : str, optional
        Output file path (None to skip saving), by default None.
    cmap : matplotlib.Colormap or str or None, optional
        Colormap, by default cm.coolwarm.
    mask_color : str
        Color for masked positions, by default 'gray'.
    vmin : float, optional
        Minimum value, by default None.
    vmax : float, optional
        Maximum value, by default None.
    figsize : (float, float), optional
        Figure size, by default None.
    xlabel : str, optional
        X-axis label, by default None.
    ylabel : str, optional
        Y-axis label, by default None.
    title : str, optional
        Title, by default None.
    interpolation : str, optional
        Interpolation method, by default 'bilinear'.
    dpi : int, optional
        Resolution (ignored when figsize is specified), by default 10.
    add_colorbar: bool, optional
        If True, add a colorbar, by default True.

    Returns
    -------
    AxesImage or None
        Plotted image data (None when saved to file).
    """
    if cbargs is None:
        cbargs = {}

    if mesh is None:
        x = list(range(data2d.shape[1]))
        y = list(range(data2d.shape[0]))
        mesh = np.meshgrid(x, y)

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = copy.copy(plt.get_cmap(str(cmap)))
        else:
            cmap = copy.copy(cmap)
        cmap.set_bad(color=mask_color)

    extent = [mesh[0][0, 0], mesh[0][-1, -1], mesh[1][0, 0], mesh[1][-1, -1]]
    img = plt.imshow(
        data2d,
        interpolation=interpolation,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="auto",
        **kwargs,
    )
    if add_colorbar:
        if "cb" in cbargs:
            cb = plt.colorbar(label=colorbar_label, **cbargs["cb"])
        else:
            cb = plt.colorbar(label=colorbar_label)

    if add_colorbar and "others" in cbargs and "yticklabels" in cbargs["others"]:
        cb.ax.set_yticklabels(cbargs["others"]["yticklabels"])

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savefilename is not None:
        plt.gcf().savefig(savefilename)
        plt.close(plt.gcf())
        return None
    else:
        return img


def plot_2d_contour(
    data2d,
    mesh=None,
    levels=None,
    colors=None,
    cmap=None,
    alpha=1,
    vmin=None,
    vmax=None,
    savefilename=None,
    figsize=None,
    xlabel=None,
    ylabel=None,
    title=None,
    dpi=10,
    fmt="%1.1f",
    fontsize=12,
    **kwargs,
):
    """Plot 2-D contour lines.

    Parameters
    ----------
    data2d : numpy.ndarray
        2-D data array.
    mesh : (numpy.ndarray, numpy.ndarray), optional
        Mesh grid, by default None.
    levels : int
        Number of contour levels, by default None.
    alpha : float
        Opacity (0.0 to 1.0), by default 1.
    savefilename : str, optional
        Output file path (None to skip saving), by default None.
    cmap : matplotlib.Colormap or str or None, optional
        Colormap, by default None.
    mask_color : str
        Color for masked positions, by default 'gray'.
    vmin : float, optional
        Minimum value, by default None.
    vmax : float, optional
        Maximum value, by default None.
    figsize : (float, float), optional
        Figure size, by default None.
    xlabel : str, optional
        X-axis label, by default None.
    ylabel : str, optional
        Y-axis label, by default None.
    title : str, optional
        Title, by default None.
    interpolation : str, optional
        Interpolation method, by default 'bilinear'.
    dpi : int, optional
        Resolution (ignored when figsize is specified), by default 10.
    fmt : str
        Contour label format, by default '%1.1f'.
    fontsize : str
        Contour label font size, by default 12.

    Returns
    -------
    AxesImage or None
        Plotted image data (None when saved to file).
    """
    if mesh is None:
        x = list(range(data2d.shape[1]))
        y = list(range(data2d.shape[0]))
        mesh = np.meshgrid(x, y)

    kwargs = {
        "alpha": alpha,
        "vmin": vmin,
        "vmax": vmax,
    }
    if cmap is None:
        kwargs["colors"] = colors if colors is not None else ["black"]
    else:
        kwargs["cmap"] = cmap
    if levels is not None:
        kwargs["levels"] = levels
    cont = plt.contour(*mesh, data2d, **kwargs)
    cont.clabel(fmt=fmt, fontsize=fontsize)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savefilename is not None:
        plt.gcf().savefig(savefilename)
        plt.close(plt.gcf())
        return None
    else:
        return cont


def plot_surface(
    x,
    y,
    z,
    value,
    ax3d=None,
    add_colorbar=False,
    savefilename=None,
    cmap=cm.jet,
    mask_color="gray",
    vmin=None,
    vmax=None,
    figsize=None,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    title=None,
    ninterp=1,
    function="linear",
    dpi=10,
    colorbar_label="",
    **kwargs,
):
    """Plot a 3-D surface.

    Parameters
    ----------
    x : (numpy.ndarray, numpy.ndarray), optional
        X-coordinate mesh.
    y : (numpy.ndarray, numpy.ndarray), optional
        Y-coordinate mesh.
    z : (numpy.ndarray, numpy.ndarray), optional
        Z-coordinate mesh.
    val : (numpy.ndarray, numpy.ndarray), optional
        Value mesh.
    ax3d : Axes3D
        Axes3D object, by default None.
    savefilename : str, optional
        Output file path (None to skip saving), by default None.
    cmap : matplotlib.Colormap or str or None, optional
        Colormap, by default cm.coolwarm.
    vmin : float, optional
        Minimum value, by default None.
    vmax : float, optional
        Maximum value, by default None.
    figsize : (float, float), optional
        Figure size, by default None.
    xlabel : str, optional
        X-axis label, by default None.
    ylabel : str, optional
        Y-axis label, by default None.
    zlabel : str, optional
        Z-axis label, by default None.
    title : str, optional
        Title, by default None.
    dpi : int, optional
        Resolution (ignored when figsize is specified), by default 10.

    Returns
    -------
    AxesImage or None
        Plotted image data (None when saved to file).
    """
    if savefilename is not None:
        if figsize is None:
            fig = plt.figure()
        else:
            if figsize == "auto":
                figsize = figsize_with_2d(x, dpi=dpi)
            fig = plt.figure(figsize=figsize)
    else:
        fig = plt.gcf()

    if ax3d is None:
        ax3d = plt.gcf().add_subplot(projection="3d")

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = copy.copy(plt.get_cmap(str(cmap)))
        else:
            cmap = copy.copy(cmap)
        cmap.set_bad(color=mask_color)

    if ninterp is not None:
        x = utils.interp2d(x, ninterp, method=function)
        y = utils.interp2d(y, ninterp, method=function)
        z = utils.interp2d(z, ninterp, method=function)
        value = utils.interp2d(value, ninterp)

    if vmin is None:
        vmin = value.min()
    if vmax is None:
        vmax = value.max()

    norm = matplotlib.colors.Normalize(vmin, vmax)
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    value_colors = mappable.to_rgba(value)

    surf = ax3d.plot_surface(
        x,
        y,
        z,
        facecolors=value_colors,
        vmin=vmin,
        vmax=vmax,
        shade=False,
        **kwargs,
    )
    if add_colorbar:
        plt.colorbar(mappable, ax=ax3d, label=colorbar_label)

    if title is not None:
        ax3d.set_title(title)
    if xlabel is not None:
        ax3d.set_xlabel(xlabel)
    if ylabel is not None:
        ax3d.set_ylabel(ylabel)
    if zlabel is not None:
        ax3d.set_zlabel(zlabel)

    if savefilename is not None:
        fig.savefig(savefilename)
        plt.close(fig)
        return None
    else:
        return surf


def plot_line(
    data1d,
    x=None,
    savefilename=None,
    vmin=None,
    vmax=None,
    figsize=None,
    xlabel=None,
    ylabel=None,
    label=None,
    title=None,
    **kwargs,
):
    """Plot 1-D data.

    Parameters
    ----------
    data1d : array-like or scalar
        1-D data to plot.
    x : array-like or scalar
        1-D data for the horizontal axis, by default None.
    savefilename : str, optional
        Output file path, by default None.
    vmin : float, optional
        Minimum value, by default None.
    vmax : float, optional
        Maximum value, by default None.
    figsize : (float, float), optional
        Figure size, by default None.
    xlabel : str, optional
        Horizontal axis label, by default None.
    ylabel : str, optional
        Vertical axis label, by default None.
    label : str, optional
        Legend label, by default None.
    title : str, optional
        Title, by default None.

    Returns
    -------
    Line2D or None
        Line object representing the plotted data (None when saved to file).
    """
    if savefilename is not None:
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)

    if x is None:
        line = plt.plot(data1d, label=label, **kwargs)
    else:
        line = plt.plot(x, data1d, label=label, **kwargs)
    plt.ylim([vmin, vmax])

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savefilename is not None:
        fig.savefig(savefilename)
        plt.close(fig)
        return None
    else:
        return line


def plot_2d_vector(
    x_data2d,
    y_data2d,
    mesh=None,
    savefilename=None,
    scale=1,
    scaler="standard",
    skip=1,
    easy_to_read=True,
    figsize=None,
    xlabel=None,
    ylabel=None,
    title=None,
    dpi=10,
    cmap=None,
    **kwargs,
):
    """Plot a 2-D vector field.

    Parameters
    ----------
    x_data2d, y_data2d : numpy.ndarray
        2-D data arrays for vector components.
    mesh : (numpy.ndarray, numpy.ndarray), optional
        Mesh grid, by default None.
    savefilename : str, optional
        Output file path (None to skip saving), by default None.
    color : str
        Vector color, by default None.
    scale : float
        Vector magnitude scale factor (multiplied to final size), by default 1.
    skip : int
        Plotting data interval, by default 1.
    easy_to_read : bool
        If True, scale vectors to a legible size, by default True.
    figsize : (float, float), optional
        Figure size, by default None.
    xlabel : str, optional
        X-axis label, by default None.
    ylabel : str, optional
        Y-axis label, by default None.
    title : str, optional
        Title, by default None.
    interpolation : str, optional
        Interpolation method, by default 'bilinear'.
    dpi : int, optional
        Resolution (ignored when figsize is specified), by default 10.

    Returns
    -------
    AxesImage or None
        Plotted image data (None when saved to file).
    """
    fig = None
    if savefilename is not None:
        if figsize is None:
            fig = plt.figure()
        else:
            if figsize == "auto":
                figsize = figsize_with_2d(x_data2d, dpi=dpi)
            fig = plt.figure(figsize=figsize)

    if mesh is None:
        x = list(range(x_data2d.shape[1]))
        y = list(range(x_data2d.shape[0]))
        mesh = np.meshgrid(x, y)

    x = mesh[0]
    y = mesh[1]
    U = np.array(x_data2d)
    V = np.array(y_data2d)

    x_skip = skip if isinstance(skip, int) else skip[0]
    y_skip = skip if isinstance(skip, int) else skip[1]
    x = x[::y_skip, ::x_skip]
    y = y[::y_skip, ::x_skip]
    U = U[::y_skip, ::x_skip]
    V = V[::y_skip, ::x_skip]

    norm = np.sqrt(U**2 + V**2)

    if scaler == "standard":
        norm_max = np.nanmax(np.abs(norm))
        U /= norm_max
        V /= norm_max

    elif scaler == "normal":
        U /= norm
        V /= norm

    elif scaler == "log":
        U = U / norm * np.log(norm + 1)
        V = V / norm * np.log(norm + 1)

    # Linear scaling to a legible size
    if easy_to_read:
        dx = (x.max() - x.min()) / x.shape[0]
        multiplier = dx * 1.2
        norm_mean = np.nanmean(np.sqrt(U**2 + V**2))
        U *= scale / norm_mean * multiplier
        V *= scale / norm_mean * multiplier

    if cmap is None:
        img = plt.quiver(
            x,
            y,
            U,
            V,
            angles="xy",
            scale_units="xy",
            scale=1,
            **kwargs,
        )
    else:
        img = plt.quiver(
            x,
            y,
            U,
            V,
            np.sqrt(U**2 + V**2),
            angles="xy",
            scale_units="xy",
            scale=1,
            cmap=cmap,
            **kwargs,
        )

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savefilename is not None:
        fig.savefig(savefilename)
        plt.close(fig)
        return None
    else:
        return img


def plot_2d_streamline(
    x_data2d,
    y_data2d,
    mesh=None,
    savefilename=None,
    skip=1,
    figsize=None,
    xlabel=None,
    ylabel=None,
    title=None,
    dpi=10,
    color=None,
    cmap=None,
    norm="linear",
    vmin=None,
    vmax=None,
    density=1,
    **kwargs,
):
    """Plot a 2-D streamline field.

    Parameters
    ----------
    x_data2d, y_data2d : numpy.ndarray
        2-D data arrays for vector components.
    mesh : (numpy.ndarray, numpy.ndarray), optional
        Mesh grid, by default None.
    savefilename : str, optional
        Output file path (None to skip saving), by default None.
    color : str
        Streamline color, by default None.
    scale : float
        Vector magnitude scale factor (multiplied to final size), by default 1.
    skip : int
        Plotting data interval, by default 1.
    easy_to_read : bool
        If True, scale vectors to a legible size, by default True.
    figsize : (float, float), optional
        Figure size, by default None.
    xlabel : str, optional
        X-axis label, by default None.
    ylabel : str, optional
        Y-axis label, by default None.
    title : str, optional
        Title, by default None.
    interpolation : str, optional
        Interpolation method, by default 'bilinear'.
    dpi : int, optional
        Resolution (ignored when figsize is specified), by default 10.

    Returns
    -------
    AxesImage or None
        Plotted image data (None when saved to file).
    """
    fig = None
    if savefilename is not None:
        if figsize is None:
            fig = plt.figure()
        else:
            if figsize == "auto":
                figsize = figsize_with_2d(x_data2d, dpi=dpi)
            fig = plt.figure(figsize=figsize)

    if mesh is None:
        x = list(range(x_data2d.shape[1]))
        y = list(range(x_data2d.shape[0]))
        mesh = np.meshgrid(x, y)

    x = mesh[0]
    y = mesh[1]
    U = np.array(x_data2d)
    V = np.array(y_data2d)

    x_skip = skip if isinstance(skip, int) else skip[0]
    y_skip = skip if isinstance(skip, int) else skip[1]
    x = x[::y_skip, ::x_skip]
    y = y[::y_skip, ::x_skip]
    U = U[::y_skip, ::x_skip]
    V = V[::y_skip, ::x_skip]

    if cmap:
        length = np.sqrt(U**2 + V**2)
        vmin = vmin or length.min()
        vmax = vmax or length.max()

        if norm == "linear":
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        elif norm == "log":
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        elif norm == "centered":
            norm = mcolors.CenteredNorm()
        elif norm == "symlog":
            norm = mcolors.SymLogNorm(vmin=vmin, vmax=vmax)

        img = plt.streamplot(
            x,
            y,
            U,
            V,
            color=length,
            cmap=cmap,
            norm=norm,
            density=density,
            **kwargs,
        )
    else:
        img = plt.streamplot(
            x,
            y,
            U,
            V,
            color=color,
            density=density,
            **kwargs,
        )

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savefilename is not None:
        fig.savefig(savefilename)
        plt.close(fig)
        return None
    else:
        return img
