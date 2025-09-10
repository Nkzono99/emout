"""
エネルギーフラックス計算およびピッチ角分類の機能をまとめたライブラリ。
エネルギーは eV 単位で扱う。
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.constants import e as e_charge


def get_indices_in_pitch_range(
    velocities: np.ndarray,
    B: np.ndarray,
    a_deg: float,
    b_deg: float,
    direction: str = 'both',
) -> np.ndarray:
    """
    速度ベクトル群と磁場ベクトルから、ピッチ角が [a_deg, b_deg] の範囲にある粒子のインデックスを返す。

    Parameters
    ----------
    velocities : np.ndarray, shape (N, 3)
        各粒子の速度ベクトル (m/s)。N はサンプル数。
    B : np.ndarray, shape (3,)
        磁場ベクトル (T) または方向ベクトル。大きさが 0 でないこと。
    a_deg : float
        ピッチ角の下限 (度)。0° ≤ a_deg < b_deg ≤ 180° の範囲で指定。
    b_deg : float
        ピッチ角の上限 (度)。
    direction : str, default='both'
        ピッチ角の符号方向を指定:
          - 'both': 内積の符号にかかわらずすべての粒子
          - 'pos' : 磁場と同方向のみ (v·B > 0)
          - 'neg' : 磁場と逆方向のみ (v·B < 0)

    Returns
    -------
    idx : np.ndarray
        指定した角度範囲かつ方向条件を満たす粒子のインデックス配列。
    """

    if not (0.0 <= a_deg < b_deg <= 180.0):
        raise ValueError(f"a_deg={a_deg}, b_deg={b_deg} の指定が不適切です。0 ≤ a < b ≤ 180")
    if direction not in ('both', 'pos', 'neg'):
        raise ValueError(f"direction='{direction}' は 'both','pos','neg' のいずれかで指定してください。")

    a_rad = np.deg2rad(a_deg)
    b_rad = np.deg2rad(b_deg)
    cos_a = np.cos(a_rad)
    cos_b = np.cos(b_rad)

    speeds = np.linalg.norm(velocities, axis=1)
    B_norm = np.linalg.norm(B)
    if B_norm == 0:
        raise ValueError("磁場ベクトル B の大きさがゼロです。")

    dot_vB = velocities.dot(B)
    cos_theta = np.zeros_like(dot_vB)
    nz = speeds > 0
    cos_theta[nz] = dot_vB[nz] / (speeds[nz] * B_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    mask_angle = (cos_theta >= cos_b) & (cos_theta <= cos_a)
    if direction == 'pos':
        mask_dir = dot_vB > 0
    elif direction == 'neg':
        mask_dir = dot_vB < 0
    else:
        mask_dir = np.ones_like(dot_vB, dtype=bool)

    idx = np.where(mask_angle & mask_dir)[0]
    return idx


def compute_energy_flux_histogram(
    velocities: np.ndarray,
    probs: np.ndarray,
    mass: float,
    energy_bins: Union[int, np.ndarray],
):
    speeds = np.linalg.norm(velocities, axis=1)
    energies_J = 0.5 * mass * speeds**2
    energies_eV = energies_J / e_charge

    if energy_bins is None:
        energy_bins = 30

    if isinstance(energy_bins, int):
        _, bin_edges = np.histogram(energies_eV, bins=energy_bins)
        bins = bin_edges
    else:
        bins = energy_bins.copy()

    energy_flux = energies_eV * speeds * probs

    E_cls = energies_eV
    w_cls = energy_flux
    hist, bin_edges = np.histogram(E_cls, bins=bins, weights=w_cls)
    
    return hist, bin_edges


def compute_energy_flux_histograms(
    velocities: np.ndarray,
    probs: np.ndarray,
    B: np.ndarray,
    mass: float,
    energy_bins: Union[int, np.ndarray],
    pitch_ranges: Union[list[tuple[float, float, str]], None] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    速度ベクトル群と存在確率配列から、ユーザーが指定するピッチ角区間および方向ごとに
    エネルギー x エネルギーフラックスのヒストグラムを返す。
    エネルギーは eV 単位で計算する。

    Parameters
    ----------
    velocities : np.ndarray, shape (N, 3)
        各粒子の速度ベクトル (m/s)。N はサンプル数。
    probs : np.ndarray, shape (N,)
        各速度ベクトルに対応する存在確率や重み。
    B : np.ndarray, shape (3,)
        磁場ベクトル (T) または方向ベクトル。
    mass : float
        粒子質量 (kg)。
    energy_bins : int or np.ndarray, shape (M+1,)
        - int の場合: numpy.histogram に自動生成を任せる。
        - np.ndarray の場合: ビン境界をそのまま使う。
    pitch_ranges : list of (a_deg, b_deg, direction) | None
        ピッチ角範囲および方向を指定するリスト。各タプルは
          (a_deg, b_deg, direction)
        の形式で、 direction は 'both','pos','neg' のいずれか。
        None を指定するとデフォルトの 6 クラス分けを使用。

    Returns
    -------
    histograms : dict[str, (hist, bin_edges)]
        キーは f"{a_deg:02.0f}-{b_deg:02.0f}_{direction}" の形式。
        値は (hist, bin_edges) のタプルで、
        - hist: 各ビンにおけるエネルギーフラックス(eV x v x prob)の合計。shape=(M,)
        - bin_edges: eV 単位のビン境界配列。shape=(M+1,)
    """

    N = velocities.shape[0]
    if probs.shape[0] != N:
        raise ValueError("`velocities` と `probs` の長さが一致しません。")

    # 速さとエネルギー (eV) を計算
    speeds = np.linalg.norm(velocities, axis=1)
    energies_J = 0.5 * mass * speeds**2
    energies_eV = energies_J / e_charge

    # energy_bins が int の場合は、numpy.histogram に自動で bin_edges を生成させる
    if isinstance(energy_bins, int):
        _, bin_edges = np.histogram(energies_eV, bins=energy_bins)
        bins = bin_edges
    else:
        bins = energy_bins.copy()
    M = len(bins) - 1

    # 各粒子のエネルギーフラックス重み = eV  x  speed  x  prob
    energy_flux = energies_eV * speeds * probs

    if pitch_ranges is None:
        pitch_ranges = [
            (0.0, 30.0, 'pos'),
            (0.0, 30.0, 'neg'),
            (30.0, 60.0, 'pos'),
            (30.0, 60.0, 'neg'),
            (60.0, 180.0, 'pos'),
            (60.0, 180.0, 'neg'),
        ]

    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (a_deg, b_deg, direction) in pitch_ranges:
        idx = get_indices_in_pitch_range(
            velocities=velocities,
            B=B,
            a_deg=a_deg,
            b_deg=b_deg,
            direction=direction
        )
        if idx.size > 0:
            E_cls = energies_eV[idx]
            w_cls = energy_flux[idx]
            hist, bin_edges = np.histogram(E_cls, bins=bins, weights=w_cls)
        else:
            hist = np.zeros(M, dtype=float)
            bin_edges = bins.copy()

        key = f"{int(a_deg):02d}-{int(b_deg):02d}_{direction}"
        histograms[key] = (hist, bin_edges)

    return histograms


def plot_energy_fluxes(
    velocities_list: list[np.ndarray],
    x: np.ndarray,
    mass: float,
    energy_bins: Union[int, np.ndarray],
    use_probs: bool = False,
    probs_list: list[np.ndarray] | None = None,
    cmap: str = 'viridis',
) -> tuple[plt.Figure, plt.Axes]:
    """
    複数系列にわたる速度ベクトルリストから、2D ヒートマップ(x vs Energy、カラースケールはエネルギーフラックス)を描画する。
    各系列のエネルギーフラックスは eV x v x (prob) の合計としてヒストグラム化する。

    Parameters
    ----------
    velocities_list : list[np.ndarray]
        長さ T のリストで、要素は shape=(NT, 3) の速度ベクトル配列。
    x : np.ndarray, shape (T,)
        各速度リストに対応する x 軸の値 (汎用的に使用可)。
    mass : float
        粒子質量 (kg)。
    energy_bins : int or np.ndarray, shape (M+1,)
        - int の場合: numpy.histogram に自動生成を任せる (全系列を通じた energies_eV から)。
        - np.ndarray の場合: ビン境界をそのまま使う。
    use_probs : bool, default=False
        True の場合、probs_list から各系列ごとに存在確率を読み込んでエネルギーフラックス重みに含める。
        False の場合は probs = np.ones(NT) とみなす (eV x v のみを重みとする)。
    probs_list : list[np.ndarray] | None, default=None
        長さ T のリストで、要素は shape=(NT,) の存在確率配列。
        use_probs=True の場合に必須。
    cmap : str, default='viridis'
        Matplotlib のカラーマップ名。

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        作成した Figure と Axes を返します。
    """

    T = len(velocities_list)
    if x.shape[0] != T:
        raise ValueError("`x` の長さと `velocities_list` の長さが一致しません。")

    if use_probs:
        if probs_list is None or len(probs_list) != T:
            raise ValueError("use_probs=True の場合、probs_list を長さ T のリストで渡してください。")

    # 全系列の energies_eV をまとめて bin_edges を取得
    all_energies_eV = []
    for v_arr in velocities_list:
        speeds = np.linalg.norm(v_arr, axis=1)
        energies_eV = (0.5 * mass * speeds**2) / e_charge
        all_energies_eV.append(energies_eV)
    all_energies_eV = np.concatenate(all_energies_eV)

    if isinstance(energy_bins, int):
        _, bin_edges = np.histogram(all_energies_eV, bins=energy_bins)
        bins = bin_edges
    else:
        bins = energy_bins.copy()
    M = len(bins) - 1

    energy_centers = 0.5 * (bins[:-1] + bins[1:])
    E_map = np.zeros((M, T), dtype=float)

    for j in range(T):
        v_arr = velocities_list[j]
        speeds = np.linalg.norm(v_arr, axis=1)
        energies_eV = (0.5 * mass * speeds**2) / e_charge

        if use_probs:
            probs = probs_list[j]
            if probs.shape[0] != v_arr.shape[0]:
                raise ValueError(f"probs_list[{j}] の長さが velocities_list[{j}] と一致しません。")
        else:
            probs = np.ones_like(speeds)

        weights = energies_eV * speeds * probs
        hist, _ = np.histogram(energies_eV, bins=bins, weights=weights)
        E_map[:, j] = hist

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        E_map,
        origin='lower',
        aspect='auto',
        extent=[x[0], x[-1], energy_centers[0], energy_centers[-1]],
        norm=LogNorm(vmin=E_map[E_map > 0].min(), vmax=E_map.max()),
        cmap=cmap,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Energy Flux [eV·(m/s)·prob] (log scale)')

    ax.set_xlabel('x')
    ax.set_ylabel('Energy [eV]')
    ax.set_title('x vs Energy-Flux Map')

    plt.tight_layout()
    return fig, ax


def plot_energy_flux(
    velocities: np.ndarray,
    probs: np.ndarray,
    B: np.ndarray,
    mass: float,
    energy_bins: Union[int, np.ndarray],
    pitch_ranges: list[tuple[float, float, str]] | None = None,
    cmap: str = 'plasma',
) -> tuple[plt.Figure, plt.Axes]:
    """
    単一データセットの速度データから、全粒子および指定したピッチ角区間ごとの
    エネルギー x エネルギーフラックス分布を重ねてプロットする。
    エネルギーは eV 単位で計算する。

    Parameters
    ----------
    velocities : np.ndarray, shape (N, 3)
        各粒子の速度ベクトル (m/s)。N はサンプル数。
    probs : np.ndarray, shape (N,)
        각速度벡터에 대응하는存在확률や重み。
    B : np.ndarray, shape (3,)
        磁場ベクトル (T) または方向ベクトル。
    mass : float
        粒子質量 (kg)。
    energy_bins : int or np.ndarray, shape (M+1,)
        - int の場合: numpy.histogram に自動生成を任せる。
        - np.ndarray の場合: ビン境界をそのまま使う。
    pitch_ranges : list of (a_deg, b_deg, direction) | None
        ピッチ角範囲および方向を指定するリスト。None の場合はデフォルトの 6 クラス分け。
    cmap : str, default='plasma'
        Matplotlib のカラーマップ名。

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        作成した Figure と Axes を返します。
    """
    speeds = np.linalg.norm(velocities, axis=1)
    energies_eV = (0.5 * mass * speeds**2) / e_charge
    energy_flux = energies_eV * speeds * probs

    if isinstance(energy_bins, int):
        _, bin_edges = np.histogram(energies_eV, bins=energy_bins)
        bins = bin_edges
    else:
        bins = energy_bins.copy()
    M = len(bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])

    total_hist, _ = np.histogram(energies_eV, bins=bins, weights=energy_flux)

    hists = compute_energy_flux_histograms(
        velocities=velocities,
        probs=probs,
        B=B,
        mass=mass,
        energy_bins=bins,
        pitch_ranges=pitch_ranges,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(centers, total_hist, where='mid', label='All', color='black', linewidth=1.5)

    keys = sorted(hists.keys())
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(keys)))

    for color, key in zip(colors, keys):
        hist, _ = hists[key]
        ax.step(centers, hist, where='mid', label=key, color=color)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Energy  x  Flux  x  prob (arb.)')
    ax.set_title('Energy-Flux Distribution')
    ax.legend(fontsize='small', loc='upper right', ncol=2)
    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 100 系列分のサンプルデータ
    T = 50
    x = np.linspace(0.0, 5.0, T)
    velocities_list = []
    probs_list = []

    np.random.seed(0)
    for _ in range(T):
        N_t = 200
        vels = np.random.normal(loc=0.0, scale=1e6, size=(N_t, 3))
        velocities_list.append(vels)
        probs_list.append(np.random.rand(N_t))

    m_e = 9.10938356e-31
    B = np.array([0.0, 0.0, 5e-9])

    # energy_bins に整数を渡した場合の動作テスト
    energy_bins_int = 30

    # 1) get_indices_in_pitch_range のテスト
    sample_vels = velocities_list[0]
    idx_20_50_pos = get_indices_in_pitch_range(
        velocities=sample_vels, B=B, a_deg=20.0, b_deg=50.0, direction='pos'
    )
    print("20°-50° 同向 粒子数:", idx_20_50_pos.size)

    # 2) compute_energy_flux_histograms のテスト（int 指定）
    hists = compute_energy_flux_histograms(
        velocities=sample_vels,
        probs=probs_list[0],
        B=B,
        mass=m_e,
        energy_bins=energy_bins_int,
    )
    print("取得できたヒストグラムのキー:", list(hists.keys()))

    # 3) x vs Energy-Flux Map のテスト（int 指定）
    fig1, ax1 = plot_energy_fluxes(
        velocities_list=velocities_list,
        x=x,
        mass=m_e,
        energy_bins=energy_bins_int,
        use_probs=True,
        probs_list=probs_list,
        cmap='plasma'
    )
    plt.show(fig1)

    # 4) Energy-Flux Distribution のテスト（int 指定）
    fig2, ax2 = plot_energy_flux(
        velocities=sample_vels,
        probs=probs_list[0],
        B=B,
        mass=m_e,
        energy_bins=energy_bins_int,
    )
    plt.show(fig2)
