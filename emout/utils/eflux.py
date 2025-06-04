"""
エネルギーフラックス計算およびピッチ角分類の機能をまとめたライブラリ。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


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

    # 1) 引数チェック
    if not (0.0 <= a_deg < b_deg <= 180.0):
        raise ValueError(f"a_deg={a_deg}, b_deg={b_deg} の指定が不適切です。0 ≤ a < b ≤ 180")
    if direction not in ('both', 'pos', 'neg'):
        raise ValueError(f"direction='{direction}' は 'both','pos','neg' のいずれかで指定してください。")

    # 2) ピッチ角の余弦しきい値に変換
    #    ピッチ角 θ は [0, 180]。cosθ は [-1, +1]。
    #    たとえば a_deg=30, b_deg=60 のとき
    #      cos(a) = cos(30°), cos(b) = cos(60°)
    #    だが、cos 関数は単調減少なので、
    #      θ ∈ [a, b] に該当するのは cosθ ∈ [cos(b), cos(a)] となる点に注意。
    a_rad = np.deg2rad(a_deg)
    b_rad = np.deg2rad(b_deg)
    cos_a = np.cos(a_rad)  # cos(30°) ≈ 0.866
    cos_b = np.cos(b_rad)  # cos(60°) ≈ 0.5

    # 3) 各粒子の cosθ = (v·B) / (|v||B|) を計算
    speeds = np.linalg.norm(velocities, axis=1)  # shape=(N,)
    B_norm = np.linalg.norm(B)
    if B_norm == 0:
        raise ValueError("磁場ベクトル B の大きさがゼロです。")

    dot_vB = velocities.dot(B)  # shape=(N,)
    cos_theta = np.zeros_like(dot_vB)

    # |v|=0 を除外して計算
    nz = speeds > 0
    cos_theta[nz] = dot_vB[nz] / (speeds[nz] * B_norm)

    # 数値誤差で cosθ が [-1,1] をはみ出すことがあるのでクリッピング
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 4) cosθ が [cos(b), cos(a)] の範囲にあるかどうかを判定
    #    cos は θ=0→1, θ=90→0, θ=180→-1 の順に減少する。よって
    #      θ ∈ [a, b] ↔ cosθ ∈ [cos(b), cos(a)]  ただし cos(b) ≤ cos(a)
    mask_angle = (cos_theta >= cos_b) & (cos_theta <= cos_a)

    # 5) direction 条件でフィルタ
    if direction == 'pos':
        mask_dir = dot_vB > 0
    elif direction == 'neg':
        mask_dir = dot_vB < 0
    else:  # both
        mask_dir = np.ones_like(dot_vB, dtype=bool)

    # 6) 両方を満たすインデックスを返す
    idx = np.where(mask_angle & mask_dir)[0]
    return idx


def compute_energy_flux_histograms(
    velocities: np.ndarray,
    probs: np.ndarray,
    B: np.ndarray,
    mass: float,
    energy_bins: np.ndarray,
    pitch_ranges: list[tuple[float, float, str]] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    速度ベクトル群と存在確率配列から、ユーザーが指定するピッチ角区間および方向ごとに
    エネルギーxエネルギーフラックスのヒストグラムを返す。

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
    energy_bins : np.ndarray, shape (M+1,)
        エネルギー軸のビン境界値。
    pitch_ranges : list of (a_deg, b_deg, direction) | None
        ピッチ角範囲および方向を指定するリスト。各タプルは
          (a_deg, b_deg, direction)
        の形式で、 direction は 'both','pos','neg' のいずれか。
        例:
          pitch_ranges = [
              (0.0, 30.0, 'pos'),   # 0~30° で同方向
              (0.0, 30.0, 'neg'),   # 0~30° で逆方向
              (30.0, 60.0, 'pos'),  # 30~60° で同方向
              … 
          ]
        None を指定すると、デフォルトで
          [
            (0.0, 30.0, 'pos'), (0.0, 30.0, 'neg'),
            (30.0, 60.0, 'pos'), (30.0, 60.0, 'neg'),
            (60.0,180.0,'pos'), (60.0,180.0,'neg')
          ]
        の 6 パターンを使う(従来の fixed クラス分割と同等）。
    
    Returns
    -------
    histograms : dict[str, (hist, bin_edges)]
        キーは f"{a_deg:02.0f}-{b_deg:02.0f}_{direction}" の形式の文字列。
        値は (hist, bin_edges) のタプルで、hist は長さ M の配列、
        bin_edges は入力の energy_bins と同じ配列 (M+1)。
    """

    N = velocities.shape[0]
    if probs.shape[0] != N:
        raise ValueError("`velocities` と `probs` の長さが一致しません。")

    # デフォルトの pitch_ranges を用意(従来の 6 クラス分け）
    if pitch_ranges is None:
        pitch_ranges = [
            (0.0, 30.0, 'pos'),
            (0.0, 30.0, 'neg'),
            (30.0, 60.0, 'pos'),
            (30.0, 60.0, 'neg'),
            (60.0, 180.0, 'pos'),
            (60.0, 180.0, 'neg'),
        ]

    # 各粒子の速度ノルムとエネルギーを計算
    speeds = np.linalg.norm(velocities, axis=1)  # shape=(N,)
    energies = 0.5 * mass * speeds**2            # shape=(N,)

    # 各粒子のエネルギーフラックス重み = E * v * prob
    energy_flux = energies * speeds * probs      # shape=(N,)

    # 結果格納用 dict
    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    M = len(energy_bins) - 1

    for (a_deg, b_deg, direction) in pitch_ranges:
        # 汎用関数でインデックスを取得
        idx = get_indices_in_pitch_range(
            velocities=velocities,
            B=B,
            a_deg=a_deg,
            b_deg=b_deg,
            direction=direction
        )

        # ビンごとにヒストグラムを作成
        if idx.size > 0:
            # エネルギーと重みを抽出
            E_cls = energies[idx]
            w_cls = energy_flux[idx]
            hist, bin_edges = np.histogram(E_cls, bins=energy_bins, weights=w_cls)
        else:
            hist = np.zeros(M, dtype=float)
            bin_edges = energy_bins.copy()

        # キーとして文字列を生成(例: "00-30_pos"）
        key = f"{int(a_deg):02d}-{int(b_deg):02d}_{direction}"
        histograms[key] = (hist, bin_edges)

    return histograms


def plot_time_energy_flux_map(
    velocities_list: list[np.ndarray],
    times: np.ndarray,
    mass: float,
    energy_bins: np.ndarray,
    use_probs: bool = False,
    probs_list: list[np.ndarray] | None = None,
    cmap: str = 'viridis',
) -> tuple[plt.Figure, plt.Axes]:
    """
    複数時刻にわたる速度ベクトルリストから、2D ヒートマップ(Time vs Energy、カラースケールはエネルギーフラックス)を描画する。
    各時刻のエネルギーフラックスは Exvx(prob) の合計としてヒストグラム化する。

    Parameters
    ----------
    velocities_list : list[np.ndarray]
        長さ T のリストで、要素は shape=(N_t, 3) の速度ベクトル配列。
    times : np.ndarray, shape (T,)
        各速度リストに対応する時刻配列 (秒等)。
    mass : float
        粒子質量 (kg)。
    energy_bins : np.ndarray, shape (M+1,)
        エネルギー軸のビン境界値 (J や eV)。
    use_probs : bool, default=False
        True の場合、probs_list から各時刻ごとに存在確率を読み込んでエネルギーフラックス重みに含める。
        False の場合は probs = np.ones(N_t) とみなす(Exv のみを重みとする）。
    probs_list : list[np.ndarray] | None, default=None
        長さ T のリストで、要素は shape=(N_t,) の存在確率配列。
        use_probs=True の場合に必須。各時刻 j の粒子数 N_t は velocities_list[j].shape[0] と一致する必要あり。
    cmap : str, default='viridis'
        Matplotlib のカラーマップ名。

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        作成した Figure と Axes を返します。
    """

    T = len(velocities_list)
    if times.shape[0] != T:
        raise ValueError("`times` の長さと `velocities_list` の長さが一致しません。")

    if use_probs:
        if probs_list is None or len(probs_list) != T:
            raise ValueError("use_probs=True の場合、probs_list を長さ T のリストで渡してください。")

    M = len(energy_bins) - 1
    energy_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])
    E_map = np.zeros((M, T), dtype=float)

    for j in range(T):
        v_arr = velocities_list[j]
        if v_arr.ndim != 2 or v_arr.shape[1] != 3:
            raise ValueError(f"velocities_list[{j}] の形状が (N_j,3) ではありません。")

        speeds = np.linalg.norm(v_arr, axis=1)
        energies = 0.5 * mass * speeds**2

        if use_probs:
            probs = probs_list[j]
            if probs.shape[0] != v_arr.shape[0]:
                raise ValueError(f"probs_list[{j}] の長さが velocities_list[{j}] と一致しません。")
        else:
            probs = np.ones_like(speeds)

        weights = energies * speeds * probs
        hist, _ = np.histogram(energies, bins=energy_bins, weights=weights)
        E_map[:, j] = hist

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        E_map,
        origin='lower',
        aspect='auto',
        extent=[times[0], times[-1], energy_centers[0], energy_centers[-1]],
        norm=LogNorm(vmin=E_map[E_map > 0].min(), vmax=E_map.max()),
        cmap=cmap,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Energy Flux (arb. unit, log scale)')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Energy [J]')
    ax.set_title('Time vs Energy-Flux Map')

    # ax.set_yscale('log')  # 必要に応じてアンコメント
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 100 時刻分
    T = 50
    times = np.linspace(0.0, 5.0, T)
    velocities_list = []
    probs_list = []

    np.random.seed(0)
    for _ in range(T):
        N_t = 200
        vels = np.random.normal(loc=0.0, scale=1e6, size=(N_t, 3))
        velocities_list.append(vels)
        probs_list.append(np.random.rand(N_t))

    # 粒子質量：電子
    m_e = 9.10938356e-31

    # エネルギービン：対数空間で 30 ビン
    Emin = 1e-18
    Emax = 1e-15
    energy_bins = np.logspace(np.log10(Emin), np.log10(Emax), 31)

    # --- 1) get_indices_in_pitch_range のテスト ---
    sample_vels = velocities_list[0]
    B = np.array([0.0, 0.0, 5e-9])
    # たとえばピッチ角 20°～50° の同方向 (pos)
    idx_20_50_pos = get_indices_in_pitch_range(
        velocities=sample_vels, B=B, a_deg=20.0, b_deg=50.0, direction='pos'
    )
    print("20°-50° 同向 粒子数:", idx_20_50_pos.size)

    # --- 2) compute_energy_flux_histograms のテスト ---
    #    デフォルトなら 6 クラス分け (0-30,30-60,60-180) x (pos,neg)
    #    または pitch_ranges を指定して任意区間を取得可能
    hists = compute_energy_flux_histograms(
        velocities=sample_vels,
        probs=probs_list[0],
        B=B,
        mass=m_e,
        energy_bins=energy_bins,
        # 例えば (10-20° 同向), (10-20° 逆向) だけ計算したいとき
        # pitch_ranges=[(10.0, 20.0, 'pos'), (10.0, 20.0, 'neg')]
    )

    # キーを列挙して確認
    print("取得できたヒストグラムのキー:", list(hists.keys()))

    # --- 3) 時系列マップ描画のテスト ---
    fig, ax = plot_time_energy_flux_map(
        velocities_list=velocities_list,
        times=times,
        mass=m_e,
        energy_bins=energy_bins,
        use_probs=True,
        probs_list=probs_list,
        cmap='plasma'
    )
    plt.show()
