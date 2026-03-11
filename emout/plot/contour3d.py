# isosurface_mpl_roi.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

DxLike = Union[float, Sequence[float]]
Bounds = Tuple[Tuple[Optional[float], Optional[float]],
               Tuple[Optional[float], Optional[float]],
               Tuple[Optional[float], Optional[float]]]
LevelFormatter = Union[str, Callable[[float], str]]


def _as_spacing_xyz(dx: DxLike) -> Tuple[float, float, float]:
    """格子間隔を `(dx, dy, dz)` 形式へ正規化する。
    
    Parameters
    ----------
    dx : DxLike
        格子間隔です。スカラーなら `(dx, dx, dx)`、3 要素なら
        `(dx, dy, dz)` として解釈します。
    Returns
    -------
    Tuple[float, float, float]
        `(dx, dy, dz)` の 3 要素タプルです。
    """
    if np.isscalar(dx):
        d = float(dx)
        return (d, d, d)
    t = tuple(float(v) for v in dx)
    if len(t) != 3:
        raise ValueError("dx must be a scalar or (dx, dy, dz).")
    return t


def _sanitize_volume(vol: np.ndarray) -> np.ndarray:
    """3 次元ボリューム配列を描画可能な値へ正規化する。
    
    Parameters
    ----------
    vol : np.ndarray
        3 次元ボリューム配列です。
    Returns
    -------
    np.ndarray
        NaN/Inf を処理した浮動小数型の `(nz, ny, nx)` 配列です。
    """
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"data3d must be 3D array (nz, ny, nx). Got ndim={vol.ndim}.")
    if not np.issubdtype(vol.dtype, np.floating):
        vol = vol.astype(np.float32, copy=False)
    if not np.isfinite(vol).all():
        vol = np.nan_to_num(vol, nan=0.0,
                            posinf=np.finfo(vol.dtype).max,
                            neginf=np.finfo(vol.dtype).min)
    return vol


def _format_level_value(
    value: float,
    *,
    fmt: Optional[LevelFormatter] = None,
    sigfigs: Optional[int] = None,
) -> str:
    """等値面レベル値を表示文字列に整形する。
    
    Parameters
    ----------
    value : float
        値。
    fmt : Optional[LevelFormatter], optional
        フォーマット指定文字列です。
    sigfigs : Optional[int], optional
        `fmt` 未指定時に使う有効桁数です。
    Returns
    -------
    str
        文字列表現です。
    """
    if callable(fmt):
        return str(fmt(value))

    if isinstance(fmt, str):
        # old-style format like '%1.2f'
        if "%" in fmt:
            try:
                return fmt % value
            except (TypeError, ValueError):
                pass
        # format mini-language, e.g. '.3g'
        try:
            return format(value, fmt)
        except (TypeError, ValueError):
            pass
        # str.format style, e.g. '{:.3g}' or '{value:.3g}'
        try:
            return fmt.format(value)
        except (IndexError, KeyError, ValueError):
            return fmt.format(value=value)

    if sigfigs is not None:
        if sigfigs < 1:
            raise ValueError("clabel_sigfigs must be >= 1.")
        return f"{value:.{sigfigs}g}"
    return f"{value:g}"


def _resolve_shared_exponent(levels: Sequence[float], shared_exponent: Union[str, int, None]) -> int:
    """ラベル表示用の共通指数を決定する。
    
    Parameters
    ----------
    levels : Sequence[float]
        等値面ラベルに使うレベル値の配列です。
    shared_exponent : Union[str, int, None]
        共通指数の設定値です。
    Returns
    -------
    int
        使用する共通指数です。
    """
    if shared_exponent is None:
        return 0
    if isinstance(shared_exponent, int):
        return shared_exponent
    if shared_exponent != "auto":
        raise ValueError("clabel_shared_exponent must be None, 'auto', or int.")

    finite_levels = np.asarray(levels, dtype=np.float64)
    finite_levels = finite_levels[np.isfinite(finite_levels)]
    if finite_levels.size == 0:
        return 0

    max_abs = np.max(np.abs(finite_levels))
    if max_abs == 0.0:
        return 0
    return int(np.floor(np.log10(max_abs)))


def _slice_from_bounds_1d(
    vmin: Optional[float],
    vmax: Optional[float],
    origin: float,
    d: float,
    n: int,
    axis_name: str,
) -> Tuple[slice, float]:
    """1 次元境界指定からインデックス範囲を計算する。
    
    Parameters
    ----------
    vmin : Optional[float]
        表示範囲の最小値。
    vmax : Optional[float]
        表示範囲の最大値。
    origin : float
        回転・平行移動の基準点です。
    d : float
        格子間隔です。
    n : int
        サンプル数または格子点数です。
    axis_name : str
        軸名（`x`/`y`/`z`）です。
    Returns
    -------
    Tuple[slice, float]
        `(インデックススライス, 新しい軸原点)` を返します。
    """
    if (vmin is not None) and (vmax is not None) and (vmax < vmin):
        raise ValueError(f"{axis_name} bounds invalid: max < min")

    # index i corresponds to x = origin + i*d (node-based)
    i0 = 0 if vmin is None else int(np.ceil((vmin - origin) / d))
    i1 = n if vmax is None else int(np.floor((vmax - origin) / d) + 1)  # end-exclusive

    i0 = max(0, min(n, i0))
    i1 = max(0, min(n, i1))

    if i1 - i0 < 2:
        raise ValueError(
            f"{axis_name} ROI too small (needs >=2 samples). got [{i0}:{i1}] over n={n}"
        )

    new_origin = origin + i0 * d
    return slice(i0, i1), new_origin


def _apply_roi(
    vol_zyx: np.ndarray,
    dx_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    bounds_xyz: Optional[Bounds],
    roi_zyx: Optional[Tuple[slice, slice, slice]],
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """ROI 指定に基づいて配列と原点を切り出す。
    
    Parameters
    ----------
    vol_zyx : np.ndarray
        (z, y, x) 順のボリューム配列です。
    dx_xyz : Tuple[float, float, float]
        (x, y, z) 順の格子間隔です。
    origin_xyz : Tuple[float, float, float]
        (x, y, z) 順の原点座標です。
    bounds_xyz : Optional[Bounds]
        `((xmin, xmax), (ymin, ymax), (zmin, zmax))` 形式の切り出し範囲です。
        各要素に `None` を指定するとその軸方向は全範囲を使用します。
    roi_zyx : Optional[Tuple[slice, slice, slice]]
        配列 index で直接指定する `(z, y, x)` 順のスライスです。
        指定した場合は `bounds_xyz` より優先されます。
    Returns
    -------
    Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]
        `(切り出し後のボリューム, 格子間隔 dx_xyz, 更新後の原点 origin_xyz)` を返します。
    """
    nz, ny, nx = vol_zyx.shape
    dx, dy, dz = dx_xyz
    x0, y0, z0 = origin_xyz

    if roi_zyx is not None:
        sz, sy, sx = roi_zyx
        sub = vol_zyx[sz, sy, sx]

        # slice.start can be None
        iz0 = 0 if sz.start is None else int(sz.start)
        iy0 = 0 if sy.start is None else int(sy.start)
        ix0 = 0 if sx.start is None else int(sx.start)

        new_origin = (x0 + ix0 * dx, y0 + iy0 * dy, z0 + iz0 * dz)
        return sub, dx_xyz, new_origin

    if bounds_xyz is None:
        return vol_zyx, dx_xyz, origin_xyz

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds_xyz

    sx, new_x0 = _slice_from_bounds_1d(xmin, xmax, x0, dx, nx, "x")
    sy, new_y0 = _slice_from_bounds_1d(ymin, ymax, y0, dy, ny, "y")
    sz, new_z0 = _slice_from_bounds_1d(zmin, zmax, z0, dz, nz, "z")

    sub = vol_zyx[sz, sy, sx]
    return sub, dx_xyz, (new_x0, new_y0, new_z0)


def contour3d(
    data3d: np.ndarray,          # (nz, ny, nx)
    dx: DxLike,                  # scalar or (dx, dy, dz) in x,y,z
    levels: Sequence[float],
    *,
    ax=None,
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    # ROI options (choose one)
    bounds_xyz: Optional[Bounds] = None,  # ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    roi_zyx: Optional[Tuple[slice, slice, slice]] = None,  # (slice_z, slice_y, slice_x)
    # rendering
    opacity: float = 0.35,
    step: int = 1,
    title: Optional[str] = None,
    save: Optional[str] = None,
    show: bool = True,
    xlabel: str = None,
    ylabel: str = None,
    zlabel: str = None,
    clabel: bool = False,
    clabel_fmt: Optional[LevelFormatter] = None,
    clabel_fontsize: float = 10,
    clabel_sigfigs: Optional[int] = None,
    clabel_shared_exponent: Union[str, int, None] = None,
    clabel_text_kwargs: Optional[Dict[str, Any]] = None,
    clabel_exponent_pos: Tuple[float, float] = (1.02, 0.98),
    clabel_exponent_text: Optional[str] = None,
    clabel_exponent_kwargs: Optional[Dict[str, Any]] = None,
):
    """3 次元ボリュームの等値面を Matplotlib で描画する。

    入力配列の軸順序は `(z, y, x)` を想定します。`bounds_xyz`（物理座標）または
    `roi_zyx`（インデックススライス）で描画範囲を切り出せます。

    Parameters
    ----------
    data3d : np.ndarray
        描画対象の 3 次元配列。形状は `(nz, ny, nx)`。
    dx : DxLike
        格子間隔。スカラー指定時は `(dx, dy, dz)` として扱います。
    levels : Sequence[float]
        描画する等値面レベルの配列です。
    ax : object, optional
        描画先の 3D Axes。`None` の場合は新規作成します。
    origin_xyz : Tuple[float, float, float], optional
        物理座標系での原点オフセット `(x0, y0, z0)`。
    bounds_xyz : Optional[Bounds], optional
        物理座標での ROI。`((xmin, xmax), (ymin, ymax), (zmin, zmax))`。
    roi_zyx : Optional[Tuple[slice, slice, slice]], optional
        インデックスでの ROI。`(slice_z, slice_y, slice_x)`。
    opacity : float, optional
        等値面の透過率（0.0〜1.0）。
    step : int, optional
        ROI 後の間引きステップ。`step=2` なら各軸 2 点ごとにサンプリングします。
    title : Optional[str], optional
        グラフタイトル。
    save : Optional[str], optional
        保存先ファイルパス。指定時は `fig.savefig` を実行します。
    show : bool, optional
        `True` の場合は `plt.show()` を呼び出します。
    xlabel : str, optional
        x 軸ラベル。
    ylabel : str, optional
        y 軸ラベル。
    zlabel : str, optional
        z 軸ラベル。
    clabel : bool, optional
        `True` の場合は等値面近傍にレベル値ラベルを描画します。
    clabel_fmt : Optional[LevelFormatter], optional
        ラベル文字列のフォーマット指定。
    clabel_fontsize : float, optional
        等値面ラベルのフォントサイズ。
    clabel_sigfigs : Optional[int], optional
        ラベル表示の有効桁数（`clabel_fmt` 未指定時）。
    clabel_shared_exponent : Union[str, int, None], optional
        ラベルで共通指数を使う指定。`'auto'` で自動決定します。
    clabel_text_kwargs : Optional[Dict[str, Any]], optional
        各等値面ラベル文字の `ax.text` 追加引数。
    clabel_exponent_pos : Tuple[float, float], optional
        共通指数テキストの 2D 軸座標位置 `(x, y)`。
    clabel_exponent_text : Optional[str], optional
        共通指数表示のテキスト。`None` の場合は自動生成します。
    clabel_exponent_kwargs : Optional[Dict[str, Any]], optional
        共通指数テキスト描画時の `ax.text2D` 追加引数。

    Returns
    -------
    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        描画に使用した `(fig, ax)` を返します。
    """
    if not levels:
        raise ValueError("levels must be non-empty.")
    if bounds_xyz is not None and roi_zyx is not None:
        raise ValueError("Specify only one of bounds_xyz or roi_zyx.")
    if not (0.0 <= opacity <= 1.0):
        raise ValueError("opacity must be in [0, 1].")
    step = int(step)
    if step < 1:
        raise ValueError("step must be >= 1.")
    if clabel and clabel_fontsize <= 0:
        raise ValueError("clabel_fontsize must be > 0.")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    vol = _sanitize_volume(data3d)
    dx_xyz = _as_spacing_xyz(dx)

    # ROI crop first (fast + predictable)
    vol, (dx_, dy_, dz_), (x0, y0, z0) = _apply_roi(vol, dx_xyz, origin_xyz, bounds_xyz, roi_zyx)

    # Optional downsample after ROI
    if step > 1:
        vol = vol[::step, ::step, ::step]
        dx_, dy_, dz_ = dx_ * step, dy_ * step, dz_ * step

    nz, ny, nx = vol.shape
    spacing_zyx = (dz_, dy_, dx_)  # array axis order: z, y, x

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = plt.gcf()

    label_text_kwargs = {} if clabel_text_kwargs is None else dict(clabel_text_kwargs)
    exponent_kwargs = {} if clabel_exponent_kwargs is None else dict(clabel_exponent_kwargs)
    shared_exp = _resolve_shared_exponent(levels, clabel_shared_exponent)
    scale = 10.0 ** shared_exp if shared_exp != 0 else 1.0

    for lv in levels:
        verts_zyx, faces, _, _ = marching_cubes(
            vol, level=float(lv), spacing=spacing_zyx, allow_degenerate=False
        )

        # (z,y,x) -> (x,y,z) and apply origin
        verts_xyz = verts_zyx[:, [2, 1, 0]]
        verts_xyz[:, 0] += x0
        verts_xyz[:, 1] += y0
        verts_xyz[:, 2] += z0

        mesh = Poly3DCollection(verts_xyz[faces], alpha=opacity)
        ax.add_collection3d(mesh)

        if clabel:
            label_level = float(lv) / scale
            label_text = _format_level_value(
                label_level,
                fmt=clabel_fmt,
                sigfigs=clabel_sigfigs,
            )
            # label near the center of each surface
            center = verts_xyz.mean(axis=0)
            ax.text(
                float(center[0]),
                float(center[1]),
                float(center[2]),
                label_text,
                fontsize=clabel_fontsize,
                **label_text_kwargs,
            )

    # ax.set_xlim(x0, x0 + (nx - 1) * dx_)
    # ax.set_ylim(y0, y0 + (ny - 1) * dy_)
    # ax.set_zlim(z0, z0 + (nz - 1) * dz_)
    
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)
        
    if title:
        ax.set_title(title)

    if clabel and shared_exp != 0:
        exponent_text = clabel_exponent_text or rf"$\times 10^{{{shared_exp}}}$"
        x_pos, y_pos = clabel_exponent_pos
        ax.text2D(
            x_pos,
            y_pos,
            exponent_text,
            transform=ax.transAxes,
            fontsize=clabel_fontsize,
            ha="left",
            va="top",
            **exponent_kwargs,
        )

    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
