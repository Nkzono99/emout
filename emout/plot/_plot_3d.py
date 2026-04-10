"""3-D plotting helpers for vector fields.

Provides :func:`plot_3d_quiver` and :func:`plot_3d_streamline`.
"""

import numpy as np
import matplotlib.pyplot as plt

from ._plot_2d import figsize_with_2d


def plot_3d_quiver(
    x_data3d,
    y_data3d,
    z_data3d,
    ax3d=None,
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
    """Plot a 3-D quiver (vector) field.

    Parameters
    ----------
    x_data2d, y_data2d : numpy.ndarray
        3-D data arrays for vector components.
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
                figsize = figsize_with_2d(x_data3d[:, 0, :], dpi=dpi)
            fig = plt.figure(figsize=figsize)

    if ax3d is None:
        ax3d = fig.add_subplot(projection="3d")

    if mesh is None:
        x = list(range(x_data3d.shape[1]))
        y = list(range(x_data3d.shape[0]))
        z = list(range(x_data3d.shape[0]))
        mesh = np.meshgrid(x, y, z)

    x = mesh[0]
    y = mesh[1]
    z = mesh[2]
    U = np.array(x_data3d)
    V = np.array(y_data3d)
    W = np.array(z_data3d)

    x_skip = skip if isinstance(skip, int) else skip[0]
    y_skip = skip if isinstance(skip, int) else skip[1]
    z_skip = skip if isinstance(skip, int) else skip[2]
    x = x[::z_skip, ::y_skip, ::x_skip]
    y = y[::z_skip, ::y_skip, ::x_skip]
    z = z[::z_skip, ::y_skip, ::x_skip]
    U = U[::z_skip, ::y_skip, ::x_skip]
    V = V[::z_skip, ::y_skip, ::x_skip]
    W = W[::z_skip, ::y_skip, ::x_skip]

    norm = np.sqrt(U**2 + V**2)

    if scaler == "standard":
        norm_max = np.nanmax(np.abs(norm))
        U /= norm_max
        V /= norm_max
        W /= norm_max

    elif scaler == "normal":
        U /= norm
        V /= norm
        W /= norm

    elif scaler == "log":
        U = U / norm * np.log(norm + 1)
        V = V / norm * np.log(norm + 1)
        W = W / norm * np.log(norm + 1)

    # Linear scaling to a legible size
    if easy_to_read:
        dx = (x.max() - x.min()) / x.shape[0]
        multiplier = dx * 1.2
        norm_mean = np.nanmean(np.sqrt(U**2 + V**2))
        U *= scale / norm_mean * multiplier
        V *= scale / norm_mean * multiplier
        W *= scale / norm_mean * multiplier

    if cmap is None:
        img = ax3d.quiver(
            x,
            y,
            z,
            U,
            V,
            W,
            angles="xy",
            scale_units="xy",
            scale=1,
            **kwargs,
        )
    else:
        img = ax3d.quiver(
            x,
            y,
            z,
            U,
            V,
            W,
            np.sqrt(U**2 + V**2 + W**2),
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


def plot_3d_streamline(
    x_data3d,
    y_data3d,
    z_data3d,
    ax=None,
    mesh=None,
    n_seeds=50,
    seed_points=None,
    max_length=200.0,
    step_size=0.5,
    color=None,
    cmap="viridis",
    colorby="magnitude",
    linewidth=1.0,
    alpha=0.8,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    title=None,
    show=False,
    savefilename=None,
    figsize=None,
    **kwargs,
):
    """Plot 3-D streamlines of a vector field using matplotlib.

    Field lines are traced from *seed_points* (or randomly placed seeds)
    by integrating the normalised vector field with
    :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    x_data3d, y_data3d, z_data3d : numpy.ndarray
        3-D arrays of shape ``(nz, ny, nx)`` for each vector component.
    ax : Axes3D, optional
        Target 3-D axes.  Created automatically when ``None``.
    mesh : tuple of numpy.ndarray, optional
        ``(X, Y, Z)`` coordinate arrays.
    n_seeds : int, default 50
        Number of random seed points (ignored when *seed_points* is set).
    seed_points : numpy.ndarray, optional
        Explicit seed points of shape ``(N, 3)`` as ``(x, y, z)``.
    max_length : float, default 200.0
        Maximum integration arc length per streamline.
    step_size : float, default 0.5
        RK45 maximum step size in grid units.
    color : str, optional
        Uniform line colour.  Overrides *cmap*/*colorby*.
    cmap : str, default ``"viridis"``
        Colour map when *colorby* is ``"magnitude"``.
    colorby : {``"magnitude"``, ``None``}, default ``"magnitude"``
        Colour each segment by the local field magnitude.
    linewidth : float, default 1.0
    alpha : float, default 0.8
    xlabel, ylabel, zlabel, title : str, optional
    show : bool, default False
    savefilename : path-like, optional
    figsize : tuple of float, optional
    **kwargs
        Forwarded to ``ax.plot()``.

    Returns
    -------
    Axes3D
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import RegularGridInterpolator

    Fx = np.asarray(x_data3d, dtype=np.float64)
    Fy = np.asarray(y_data3d, dtype=np.float64)
    Fz = np.asarray(z_data3d, dtype=np.float64)
    nz, ny, nx = Fx.shape

    if mesh is not None:
        xc = mesh[0][0, 0, :]
        yc = mesh[1][0, :, 0]
        zc = mesh[2][:, 0, 0]
    else:
        xc = np.arange(nx, dtype=float)
        yc = np.arange(ny, dtype=float)
        zc = np.arange(nz, dtype=float)

    interp_x = RegularGridInterpolator((zc, yc, xc), Fx, bounds_error=False, fill_value=0.0)
    interp_y = RegularGridInterpolator((zc, yc, xc), Fy, bounds_error=False, fill_value=0.0)
    interp_z = RegularGridInterpolator((zc, yc, xc), Fz, bounds_error=False, fill_value=0.0)

    def field_func(t, pos):
        pt = np.array([[pos[2], pos[1], pos[0]]])  # (z, y, x)
        vx = interp_x(pt).item()
        vy = interp_y(pt).item()
        vz = interp_z(pt).item()
        mag = np.sqrt(vx**2 + vy**2 + vz**2)
        if mag < 1e-30:
            return [0.0, 0.0, 0.0]
        return [vx / mag, vy / mag, vz / mag]

    if seed_points is None:
        rng = np.random.RandomState(42)
        seed_points = np.column_stack(
            [
                rng.uniform(xc[0], xc[-1], n_seeds),
                rng.uniform(yc[0], yc[-1], n_seeds),
                rng.uniform(zc[0], zc[-1], n_seeds),
            ]
        )

    lines = []
    for seed in seed_points:
        try:
            sol = solve_ivp(
                field_func,
                [0, max_length],
                seed,
                method="RK45",
                max_step=step_size,
                dense_output=False,
            )
            if sol.success and sol.y.shape[1] > 1:
                lines.append(sol.y)
        except Exception:
            continue

    if not lines:
        import warnings

        warnings.warn("No streamlines could be traced from the given seeds.")

    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    use_cmap = color is None and colorby == "magnitude"
    if use_cmap:
        all_mags = []
        for line in lines:
            pts_zyx = np.column_stack([line[2], line[1], line[0]])
            mags = np.sqrt(interp_x(pts_zyx) ** 2 + interp_y(pts_zyx) ** 2 + interp_z(pts_zyx) ** 2)
            all_mags.append(mags)
        mag_min = min(m.min() for m in all_mags) if all_mags else 0.0
        mag_max = max(m.max() for m in all_mags) if all_mags else 1.0
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=mag_min, vmax=mag_max))

    for i, line in enumerate(lines):
        xs, ys, zs = line[0], line[1], line[2]
        if use_cmap:
            pts_zyx = np.column_stack([zs, ys, xs])
            mags = np.sqrt(interp_x(pts_zyx) ** 2 + interp_y(pts_zyx) ** 2 + interp_z(pts_zyx) ** 2)
            for j in range(len(xs) - 1):
                c = sm.to_rgba(mags[j])
                ax.plot(
                    xs[j : j + 2],
                    ys[j : j + 2],
                    zs[j : j + 2],
                    color=c,
                    linewidth=linewidth,
                    alpha=alpha,
                    **kwargs,
                )
        else:
            ax.plot(
                xs,
                ys,
                zs,
                color=color or f"C{i % 10}",
                linewidth=linewidth,
                alpha=alpha,
                **kwargs,
            )

    if use_cmap and lines:
        plt.colorbar(sm, ax=ax, shrink=0.6, label="magnitude")

    ax.set_xlabel(xlabel or "x")
    ax.set_ylabel(ylabel or "y")
    ax.set_zlabel(zlabel or "z")
    if title:
        ax.set_title(title)

    if savefilename is not None:
        plt.savefig(savefilename)
        if fig is not None:
            plt.close(fig)
        return ax

    if show:
        plt.show()

    return ax
