from typing import Any, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cn

from emout.utils.eflux import compute_energy_flux_histogram


class HeatmapData:
    """
    2D グリッド上の値を保持し、pcolormesh で描画するヘルパークラス。
    X, Y, Z はすべて 2D numpy.ndarray で同じ形状である必要があります。
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
        return f"<HeatmapData: shape={self.Z.shape}, xlabel={self.xlabel}, ylabel={self.ylabel}>"

    def plot(self, ax=None, cmap="viridis", use_si=True, **plot_kwargs):
        """
        pcolormesh によるヒートマップを描画する。
        - ax: matplotlib.axes.Axes を渡すか、None の場合は新規 Figure/Axes を作成
        - cmap: カラーマップ（例：'viridis', 'plasma', etc.）
        - plot_kwargs: pcolormesh に渡す追加キーワード (例：shading="auto")
        """
        if ax is None:
            ax = plt.gca()

        X = self.X
        Y = self.Y

        xlabel = self.xlabel
        ylabel = self.ylabel

        if self.units and use_si:
            X = self.units[0].reverse(X)
            Y = self.units[1].reverse(Y)

            xlabel = f"{xlabel} [{self.units[0].unit}]"
            ylabel = f"{ylabel} [{self.units[1].unit}]"

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
        return (
            f"<ProbabilityResult: grid_dims={self.dims}, "
            f"axes={ProbabilityResult._AXES}>"
        )

    def pair(self, var1: str, var2: str) -> HeatmapData:
        """
        任意の 2 変数 (var1, var2) を取り出し、HeatmapData を返す。
        var1, var2 は 'x','y','z','vx','vy','vz' のいずれか。
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

        X = self.phases[:, :, :, :, :, :, idx1].reshape(
            self.dims[idx2], self.dims[idx1]
        )
        Y = self.phases[:, :, :, :, :, :, idx2].reshape(
            self.dims[idx2], self.dims[idx1]
        )
        Z = self.probabilities.reshape(self.dims[idx2], self.dims[idx1])

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
        phases = self.phases
        velocities = phases[:, :, :, :, :, :, 3:6].reshape(-1, 3)
        velocities = self.unit.v.reverse(velocities)

        mass = abs(self.unit.m.reverse(cn.e / self.inp.qm[self.ispec]))

        if self.inp.nflag_emit[self.ispec] == 2:  # PE
            J0 = self.unit.J.reverse(self.inp.curf[0])
            a = self.unit.v.reverse(self.inp.path[2])
            n0 = J0 / (2 * a) * np.sqrt(np.pi / 2) / cn.e
        else:
            wp = self.unit.f.reverse(self.inp.wp[self.ispec])
            n0 = wp**2 * mass * cn.epsilon_0 / cn.e**2

        probabilities = self.probabilities * n0

        hist, bin_edges = compute_energy_flux_histogram(
            velocities,
            np.nan_to_num(probabilities, 0),
            mass=mass,
            energy_bins=energy_bins,
        )

        return hist, bin_edges

    def plot_energy_spectrum(self, energy_bins=None, scale="log"):
        hist, bin_edges = self.energy_spectrum(energy_bins=energy_bins)

        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        plt.step(centers, hist, color="black", linestyle="solid")
        
        plt.xlabel('Energy [eV]')
        plt.ylabel('Energy flux [$eV m^{-2} s^{-1}$]')

        plt.xscale(scale)
