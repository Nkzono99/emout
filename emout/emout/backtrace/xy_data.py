from typing import Any, Iterator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class XYData:
    """
    単一シリーズ (x, y) を保持し、plot() で x vs y を描くヘルパークラス。
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = "x",
        ylabel: str = "y",
        title: Optional[str] = None,
        units=None,
    ):
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("XYData: x, y はいずれも一次元配列である必要があります")
        if len(x) != len(y):
            raise ValueError("XYData: x, y は同じ長さである必要があります")

        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title or f"{xlabel} vs {ylabel}"
        self.units = units

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return (
            f"<XYData: len={len(self.x)}, xlabel={self.xlabel}, ylabel={self.ylabel}>"
        )

    def plot(self, ax: Any = None, use_si=True, **plot_kwargs) -> Any:
        """
        折れ線プロット: x vs y
        - ax: matplotlib.axes.Axes を渡すとその上にプロット
        - plot_kwargs: matplotlib.pyplot.plot に渡すキーワード (例: color, linestyle, label, alpha, など)
        """
        if ax is None:
            ax = plt.gca()

        xs = self.x
        ys = self.y

        xlabel = self.xlabel
        ylabel = self.ylabel

        if self.units and use_si:
            xs = self.units[0].reverse(xs)
            ys = self.units[1].reverse(ys)
            xlabel = f"{xlabel} [{self.units[0].unit}]"
            ylabel = f"{ylabel} [{self.units[1].unit}]"

        ax.plot(xs, ys, **plot_kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title(self.title)

        return ax


class MultiXYData:
    """
    複数シリーズ (x[i,:], y[i,:]) を保持し、plot() で全シリーズをプロットするヘルパークラス。
    """

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
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(
                "MultiXYData: x, y は 2D 配列 (N_series, N_points) である必要があります"
            )
        if x.shape != y.shape:
            raise ValueError("MultiXYData: x, y は同じ形状である必要があります")

        self.x = x
        self.y = y
        self.last_indexes = last_indexes
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title or f"{xlabel} vs {ylabel}"
        self.units = units

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return (
            f"<MultiXYData: n_series={self.x.shape[0]}, n_points={self.x.shape[1]}, "
            f"xlabel={self.xlabel}, ylabel={self.ylabel}>"
        )

    def plot(self, ax: Any = None, use_si=True, **plot_kwargs) -> Any:
        """
        複数シリーズを重ねて折れ線プロットする。
        - ax: matplotlib.axes.Axes を渡すとその上にプロット
        - plot_kwargs: matplotlib.pyplot.plot に渡す追加キーワード
            * color, linestyle, label, etc.
            * alpha: スカラー値 OR 各系列ごと (長さ N_series の 1D array)
        """
        if ax is None:
            ax = plt.gca()

        n_series = self.x.shape[0]
        alpha_arr = plot_kwargs.get("alpha", None)

        for i in range(n_series):
            iend = self.last_indexes[i]
            xs = self.x[i, :iend]
            ys = self.y[i, :iend]

            if self.units and use_si:
                xs = self.units[0].reverse(xs)
                ys = self.units[1].reverse(ys)

            if (
                alpha_arr is not None
                and hasattr(alpha_arr, "__len__")
                and len(alpha_arr) == n_series
            ):
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
