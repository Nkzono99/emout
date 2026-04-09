"""Energy-flux probability distribution results.

:class:`ProbabilityResult` computes and visualises energy-resolved pitch-angle
distributions, while :class:`HeatmapData` wraps 2-D histogram payloads.
"""

from typing import Any, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cn

from emout.utils.eflux import compute_energy_flux_histogram

_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


class HeatmapData:
    """2-D histogram data with axis labels and a plot helper.

    Stores bin edges, counts, and axis metadata produced by
    :class:`ProbabilityResult` computations.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        xlabel: str = "X",
        ylabel: str = "Y",
        title: str = "Heatmap",
        units=None,
    ):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        X : np.ndarray
            x 軸格子配列です。
        Y : np.ndarray
            y 軸格子配列です。
        Z : np.ndarray
            z 軸格子配列です。
        xlabel : str, optional
            x 軸ラベル。
        ylabel : str, optional
            y 軸ラベル。
        title : str, optional
            タイトル文字列。
        units : object, optional
            単位変換情報です。
        """
        if X.ndim != 2 or Y.ndim != 2 or Z.ndim != 2:
            raise ValueError(
                "HeatmapData: X, Y, Z はすべて 2D 配列である必要があります"
            )

        if X.shape != Y.shape or X.shape != Z.shape:
            raise ValueError("HeatmapData: X, Y, Z は同じ形状を持つ必要があります")

        self.X = X
        self.Y = Y
        self.Z = Z
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.units = units

    def __repr__(self):
        """文字列表現を返す。
        
        Returns
        -------
        str
            文字列表現。
        """
        return f"<HeatmapData: shape={self.Z.shape}, xlabel={self.xlabel}, ylabel={self.ylabel}>"

    def plot(self, ax=None, cmap="viridis", use_si=True, offsets=None, **plot_kwargs):
        """Plot the heatmap with :func:`pcolormesh`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes. If ``None``, uses the current axes.
        cmap : str, default ``"viridis"``
            Colormap name.
        use_si : bool, default True
            Convert axes to SI units when unit info is available.
        offsets : tuple of (float or str), optional
            ``(x_offset, y_offset)`` applied to the grid axes.
            Accepts ``"left"``, ``"center"``, ``"right"`` or a numeric shift.
        **plot_kwargs
            Forwarded to :func:`matplotlib.axes.Axes.pcolormesh`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from emout.utils.util import apply_offset

        if ax is None:
            ax = plt.gca()

        X = self.X.copy()
        Y = self.Y.copy()

        xlabel = self.xlabel
        ylabel = self.ylabel

        if self.units and use_si:
            X = self.units[0].reverse(X)
            Y = self.units[1].reverse(Y)

            xlabel = f"{xlabel} [{self.units[0].unit}]"
            ylabel = f"{ylabel} [{self.units[1].unit}]"

        if offsets is not None:
            X = apply_offset(X, offsets[0])
            Y = apply_offset(Y, offsets[1])

        mesh = ax.pcolormesh(X, Y, self.Z, cmap=cmap, **plot_kwargs)

        plt.colorbar(mesh, ax=ax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(self.title)

        return ax


class ProbabilityResult:
    """
    get_probabilities が返す結果をまとめるクラス。
    任意の 2 変数平面 (var1, var2) に対して HeatmapData を返す。
    内部では 6 次元グリッドを保存し、任意の2軸を切り出して可視化可能。
    """

    _AXES = ["x", "y", "z", "vx", "vy", "vz"]
    _BACKEND_AXES = ["z", "y", "x", "vz", "vy", "vx"]

    def __init__(
        self,
        phases: np.ndarray,
        probabilities: np.ndarray,
        dims: Sequence[int],
        ret_particles,
        particles,
        ispec: int,
        inp,
        unit=None,
    ):
        """
        Parameters
        ----------
        phases_flat : numpy.ndarray, shape = (N_points, 6)
            PhaseGrid.create_grid() で出力されたフラット配列
        probabilities : numpy.ndarray, shape = (N_points,)
            get_probabilities で出力された確率フラット配列
        dims : Sequence[int], length=6
            各軸 (x, y, z, vx, vy, vz) の要素数。例えば (nx, ny, nz, nvx, nvy, nvz)。
        """
        if len(dims) != 6:
            raise ValueError(
                "dims は 6 要素 (nx, ny, nz, nvx, nvy, nvz) のタプル／リストである必要があります"
            )
        self.dims = tuple(dims)

        self.phases = phases
        self.probabilities = probabilities
        self.ret_particles = ret_particles
        self.particles = particles
        self.ispec = ispec
        self.inp = inp
        self.unit = unit

    def __iter__(self) -> Iterator[Any]:
        """
        タプルアンパック対応 (phases_nd, prob_nd, ret_particles) を返すための定義。
        """
        yield self.phases
        yield self.probabilities
        yield self.ret_particles

    def __repr__(self) -> str:
        """文字列表現を返す。
        
        Returns
        -------
        str
            文字列表現。
        """
        return (
            f"<ProbabilityResult: grid_dims={self.dims}, "
            f"axes={ProbabilityResult._AXES}>"
        )

    def _phases_nd(self) -> np.ndarray:
        """phase grid を shape=(nx, ny, nz, nvx, nvy, nvz, 6) に正規化する。"""
        phases = np.asarray(self.phases)
        expected_size = int(np.prod(self.dims)) * len(ProbabilityResult._AXES)
        if phases.size != expected_size:
            raise ValueError(
                "phases の要素数が dims で表される 6D グリッドと一致しません"
            )

        backend_shape = tuple(
            self.dims[ProbabilityResult._AXES.index(axis)]
            for axis in ProbabilityResult._BACKEND_AXES
        )
        canonical_shape = tuple(self.dims)

        if phases.ndim == 7:
            if phases.shape == (*canonical_shape, len(ProbabilityResult._AXES)):
                return phases
            if phases.shape == (*backend_shape, len(ProbabilityResult._AXES)):
                return np.transpose(
                    phases,
                    axes=[
                        ProbabilityResult._BACKEND_AXES.index(axis)
                        for axis in ProbabilityResult._AXES
                    ]
                    + [6],
                )
            raise ValueError("phases の shape が期待される 6D グリッド形状と一致しません")

        phases = phases.reshape(*backend_shape, len(ProbabilityResult._AXES))
        return np.transpose(
            phases,
            axes=[
                ProbabilityResult._BACKEND_AXES.index(axis)
                for axis in ProbabilityResult._AXES
            ]
            + [6],
        )

    def _probabilities_nd(self) -> np.ndarray:
        """probability grid を shape=(nx, ny, nz, nvx, nvy, nvz) に正規化する。"""
        probabilities = np.asarray(self.probabilities)
        expected_size = int(np.prod(self.dims))
        if probabilities.size != expected_size:
            raise ValueError(
                "probabilities の要素数が dims で表される 6D グリッドと一致しません"
            )

        backend_shape = tuple(
            self.dims[ProbabilityResult._AXES.index(axis)]
            for axis in ProbabilityResult._BACKEND_AXES
        )
        canonical_shape = tuple(self.dims)

        if probabilities.ndim == 6:
            if probabilities.shape == canonical_shape:
                return probabilities
            if probabilities.shape == backend_shape:
                return np.transpose(
                    probabilities,
                    axes=[
                        ProbabilityResult._BACKEND_AXES.index(axis)
                        for axis in ProbabilityResult._AXES
                    ],
                )
            raise ValueError(
                "probabilities の shape が期待される 6D グリッド形状と一致しません"
            )

        probabilities = probabilities.reshape(*backend_shape)
        return np.transpose(
            probabilities,
            axes=[
                ProbabilityResult._BACKEND_AXES.index(axis)
                for axis in ProbabilityResult._AXES
            ],
        )

    def _axis_values(self, axis: str, phases=None) -> np.ndarray:
        """指定軸の 1D 座標列を返す。"""
        idx = ProbabilityResult._AXES.index(axis)
        if phases is None:
            phases = self._phases_nd()
        return np.moveaxis(phases[..., idx], idx, 0).reshape(self.dims[idx], -1)[:, 0]

    def _integrate_axis(
        self,
        values: np.ndarray,
        axis_name: str,
        current_axes: Sequence[str],
        phases=None,
    ) -> np.ndarray:
        """指定軸を座標付き台形積分で潰す。単一点軸はそのまま取り出す。"""
        axis_idx = current_axes.index(axis_name)
        coords = self._axis_values(axis_name, phases=phases)

        if coords.size <= 1:
            return np.take(values, indices=0, axis=axis_idx)

        if coords[0] > coords[-1]:
            values = np.flip(values, axis=axis_idx)
            coords = coords[::-1]

        return _TRAPEZOID(values, x=coords, axis=axis_idx)

    def pair(self, var1: str, var2: str) -> HeatmapData:
        """
        任意の 2 変数 (var1, var2) を取り出し、HeatmapData を返す。
        var1, var2 は 'x','y','z','vx','vy','vz' のいずれか。
        指定しなかった軸は座標付き台形積分で 2 次元へ射影する。
        """
        if var1 not in ProbabilityResult._AXES or var2 not in ProbabilityResult._AXES:
            raise KeyError(
                f"Allowed axes = {ProbabilityResult._AXES}, but got '{var1}', '{var2}'"
            )
        if var1 == var2:
            raise ValueError("var1 と var2 は異なる軸を指定してください")

        idx1 = ProbabilityResult._AXES.index(var1)
        idx2 = ProbabilityResult._AXES.index(var2)

        if self.unit:
            u1 = self.unit.v if var1.startswith("v") else self.unit.length
            u2 = self.unit.v if var2.startswith("v") else self.unit.length
            units = (u1, u2)
        else:
            units = None

        phases = self._phases_nd()
        probabilities = self._probabilities_nd()
        current_axes = list(ProbabilityResult._AXES)
        integrated = probabilities

        for axis_name in ProbabilityResult._AXES:
            if axis_name in (var1, var2):
                continue
            integrated = self._integrate_axis(
                integrated, axis_name, current_axes, phases=phases
            )
            current_axes.remove(axis_name)

        axis1_values = self._axis_values(var1, phases=phases)
        axis2_values = self._axis_values(var2, phases=phases)

        X, Y = np.meshgrid(axis1_values, axis2_values, indexing="xy")
        Z = np.moveaxis(
            integrated,
            (current_axes.index(var2), current_axes.index(var1)),
            (0, 1),
        )

        xlabel = var1
        ylabel = var2
        title = f"{var1} vs {var2} Probability"

        return HeatmapData(
            X, Y, Z, xlabel=xlabel, ylabel=ylabel, title=title, units=units
        )

    def __getattr__(self, name: str) -> Any:
        """
        属性アクセスを「ペア名」と解釈できるようにする。
        例:
          result.xz   → pair("x","z")
          result.vxvy → pair("vx","vy")
          result.yvx  → pair("y","vx")
        """
        for key1 in ProbabilityResult._AXES:
            if name.startswith(key1):
                rest = name[len(key1) :]
                if rest in ProbabilityResult._AXES and rest != key1:
                    return self.pair(key1, rest)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def energy_spectrum(self, energy_bins=None):
        """エネルギースペクトルを計算する。
        
        Parameters
        ----------
        energy_bins : object, optional
            エネルギーヒストグラムのビン設定です。
            整数ならビン数、配列ならビン境界を指定します。
        Returns
        -------
        object
            処理結果です。
        """
        phases = self._phases_nd()
        velocities = phases[..., 3:6].reshape(-1, 3)
        velocities = self.unit.v.reverse(velocities)

        mass = abs(self.unit.m.reverse(cn.e / self.inp.qm[self.ispec]))

        if self.inp.nflag_emit[self.ispec] == 2:  # PE
            J0 = self.unit.J.reverse(self.inp.curf[0])
            a = self.unit.v.reverse(self.inp.path[2])
            n0 = J0 / (2 * a) * np.sqrt(np.pi / 2) / cn.e
        else:
            wp = self.unit.f.reverse(self.inp.wp[self.ispec])
            n0 = wp**2 * mass * cn.epsilon_0 / cn.e**2

        probabilities = self._probabilities_nd().reshape(-1) * n0

        hist, bin_edges = compute_energy_flux_histogram(
            velocities,
            np.nan_to_num(probabilities, 0),
            mass=mass,
            energy_bins=energy_bins,
        )

        return hist, bin_edges

    def plot_energy_spectrum(self, energy_bins=None, scale="log"):
        """プロット処理を行う。
        
        Parameters
        ----------
        energy_bins : object, optional
            エネルギーヒストグラムのビン設定です。
            整数ならビン数、配列ならビン境界を指定します。
        scale : str, optional
            表示スケール係数です。
        Returns
        -------
        None
            戻り値はありません。
        """
        hist, bin_edges = self.energy_spectrum(energy_bins=energy_bins)

        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        plt.step(centers, hist, color="black", linestyle="solid")
        
        plt.xlabel('Energy [eV]')
        plt.ylabel('Energy flux [$eV m^{-2} s^{-1}$]')

        plt.xscale(scale)
