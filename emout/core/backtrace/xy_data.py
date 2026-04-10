"""Lightweight x-y pair data containers for backtrace visualisation.

:class:`XYData` holds a single named curve, while :class:`MultiXYData`
groups several curves for overlay plotting.
"""

from typing import Any, Iterator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class XYData:
    """Named x-y curve with optional unit labels and a plot helper."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = "x",
        ylabel: str = "y",
        title: Optional[str] = None,
        units=None,
    ):
        """Initialize the x-y curve data.

        Parameters
        ----------
        x : np.ndarray
            X coordinates or X component values
        y : np.ndarray
            Y coordinates or Y component values
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        title : str or None, optional
            Plot title
        units : object, optional
            Unit conversion information
        """
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("XYData: x and y must both be 1-D arrays")
        if len(x) != len(y):
            raise ValueError("XYData: x and y must have the same length")

        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title or f"{xlabel} vs {ylabel}"
        self.units = units

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate, yielding ``(x, y)`` arrays for tuple unpacking.

        Returns
        -------
        Iterator[Tuple[np.ndarray, np.ndarray]]
            Iterator yielding (x, y) arrays.
        """
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        """Return a string representation.

        Returns
        -------
        str
            Human-readable summary.
        """
        return f"<XYData: len={len(self.x)}, xlabel={self.xlabel}, ylabel={self.ylabel}>"

    def plot(self, ax: Any = None, use_si=True, gap=None, offsets=None, **plot_kwargs) -> Any:
        """Plot the x-y curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes. If ``None``, uses the current axes.
        use_si : bool, default True
            Convert to SI units when available.
        gap : float, optional
            Insert NaN breaks where consecutive-point distance exceeds *gap*.
        offsets : tuple of (float or str), optional
            ``(x_offset, y_offset)`` applied after unit conversion.
        **plot_kwargs
            Forwarded to :func:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from emout.utils.util import apply_offset

        if ax is None:
            ax = plt.gca()

        xs = self.x.copy()
        ys = self.y.copy()

        xlabel = self.xlabel
        ylabel = self.ylabel

        if self.units and use_si:
            xs = self.units[0].reverse(xs)
            ys = self.units[1].reverse(ys)
            xlabel = f"{xlabel} [{self.units[0].unit}]"
            ylabel = f"{ylabel} [{self.units[1].unit}]"

        if offsets is not None:
            xs = apply_offset(xs, offsets[0])
            ys = apply_offset(ys, offsets[1])

        if gap:
            xs, ys = _insert_nans_for_gaps(xs, ys, gap)

        ax.plot(xs, ys, **plot_kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title(self.title)

        return ax


class MultiXYData:
    """Collection of :class:`XYData` curves for overlay plotting."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        last_indexes: np.ndarray,
        xlabel: str = "x",
        ylabel: str = "y",
        title: Optional[str] = None,
        units=None,
    ):
        """Initialize the multi-series x-y data.

        Parameters
        ----------
        x : np.ndarray
            X coordinate array, shape ``(N_series, N_points)``
        y : np.ndarray
            Y coordinate array, shape ``(N_series, N_points)``
        last_indexes : np.ndarray
            End index (exclusive) of valid data for each series
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        title : str or None, optional
            Plot title
        units : object, optional
            Unit conversion information
        """
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("MultiXYData: x and y must be 2-D arrays of shape (N_series, N_points)")
        if x.shape != y.shape:
            raise ValueError("MultiXYData: x and y must have the same shape")

        self.x = x
        self.y = y
        self.last_indexes = last_indexes
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title or f"{xlabel} vs {ylabel}"
        self.units = units

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate, yielding ``(x, y)`` arrays for tuple unpacking.

        Returns
        -------
        Iterator[Tuple[np.ndarray, np.ndarray]]
            Iterator yielding (x, y) arrays.
        """
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        """Return a string representation.

        Returns
        -------
        str
            Human-readable summary.
        """
        return (
            f"<MultiXYData: n_series={self.x.shape[0]}, n_points={self.x.shape[1]}, "
            f"xlabel={self.xlabel}, ylabel={self.ylabel}>"
        )

    def plot(self, ax: Any = None, use_si=True, gap=None, offsets=None, **plot_kwargs) -> Any:
        """Plot all series as overlaid line plots.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes.
        use_si : bool, default True
            Convert to SI units when available.
        gap : float, optional
            Insert NaN breaks where consecutive-point distance exceeds *gap*.
        offsets : tuple of (float or str), optional
            ``(x_offset, y_offset)`` applied after unit conversion.
        **plot_kwargs
            Forwarded to :func:`matplotlib.axes.Axes.plot`.
            ``alpha`` may be a scalar or a per-series array of length *N*.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from emout.utils.util import apply_offset

        if ax is None:
            ax = plt.gca()

        n_series = self.x.shape[0]
        alpha_arr = plot_kwargs.get("alpha", None)

        for i in range(n_series):
            iend = self.last_indexes[i]
            xs = self.x[i, :iend].copy()
            ys = self.y[i, :iend].copy()

            if self.units and use_si:
                xs = self.units[0].reverse(xs)
                ys = self.units[1].reverse(ys)

            if offsets is not None:
                xs = apply_offset(xs, offsets[0])
                ys = apply_offset(ys, offsets[1])

            if gap:
                xs, ys = _insert_nans_for_gaps(xs, ys, gap)

            if alpha_arr is not None and hasattr(alpha_arr, "__len__") and len(alpha_arr) == n_series:
                alpha_i = float(alpha_arr[i])
                kw = {**plot_kwargs, "alpha": alpha_i}
                ax.plot(xs, ys, **kw)

            else:
                ax.plot(xs, ys, **plot_kwargs)

        xlabel = self.xlabel
        ylabel = self.ylabel

        if self.units and use_si:
            xlabel = f"{xlabel} [{self.units[0].unit}]"
            ylabel = f"{ylabel} [{self.units[1].unit}]"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(self.title)

        return ax


def _insert_nans_for_gaps(x: np.ndarray, y: np.ndarray, gap: float):
    """Insert NaN breaks where consecutive-point distance exceeds the gap threshold.

    Parameters
    ----------
    x : np.ndarray
        X coordinate array
    y : np.ndarray
        Y coordinate array
    gap : float
        Distance threshold above which a trajectory gap is assumed.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(x, y)`` arrays with NaN breaks inserted at gaps.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = x.shape[0]
    if N < 2:
        return x.copy(), y.copy()

    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)

    new_x = [x[0]]
    new_y = [y[0]]
    for i in range(N - 1):
        if dist[i] > gap:
            new_x.append(np.nan)
            new_y.append(np.nan)
        new_x.append(x[i + 1])
        new_y.append(y[i + 1])

    return np.array(new_x), np.array(new_y)
