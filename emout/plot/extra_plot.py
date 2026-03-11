from typing import Tuple, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from emout import Emout

from emout.utils.emsesinp import InpFile
from emout.utils.units import Units


def plot_surface_with_hole(
    data_xyz, inp, add_colorbar=True, show=False, vrange="minmax", **kwargs
):
    """矩形ホール付きの 3D サーフェスを描画する。

    内部では `data_xyz[...].plot(mode="surf", **kwargs)` を複数回呼び出して
    上面・側面・底面を描画します。

    Parameters
    ----------
    data_xyz : Data3d
        描画対象の 3 次元データです。
    inp : InpFile
        ホール形状（`xlrechole` など）を含む入力パラメータです。
    add_colorbar : bool, optional
        最後の底面描画時にカラーバーを追加するかどうか。
    show : bool, optional
        `True` の場合は最後に `plt.show()` を呼び出します。
    vrange : {'minmax', 'center'}, optional
        カラースケール範囲の決定方法です。
        `'center'` の場合は 0 中心の対称レンジにします。
    **kwargs : dict
        `Data2d.plot(mode="surf")` へ渡す引数です。
        主に `use_si`, `offsets`, `cmap`, `vmin`, `vmax`, `xlabel`, `ylabel`,
        `zlabel`, `title`, `ninterp`, `function`, `figsize`, `dpi`,
        および `mpl_toolkits.mplot3d.Axes3D.plot_surface` の追加引数を指定できます。

    Returns
    -------
    None
        戻り値はありません。
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
    data_xyz[zl, yl : yu + 1, xl : xu + 1].plot(
        mode="surf", add_colorbar=add_colorbar, **kwargs
    )

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
    """矩形ホール輪郭線を 2D 平面上に描画する。

    Parameters
    ----------
    inp_or_data : Union[InpFile, "Emout"]
        `InpFile` または `Emout` オブジェクト。
    unit : Units, optional
        単位変換オブジェクト。`inp_or_data` が `Emout` の場合は自動取得します。
    use_si : bool, optional
        `True` の場合は SI 単位系で描画します。
    offsets : Tuple[int, int], optional
        `(x, y)` 方向の表示オフセット。
    axis : str, optional
        描画する断面軸。現在は `"xz"` のみ対応しています。
    color : str, optional
        線色。
    linewidth : float, optional
        線幅。
    fix_lims : bool, optional
        `True` の場合は描画前の軸範囲を固定します。

    Returns
    -------
    list[matplotlib.lines.Line2D]
        `plt.plot` が返す線オブジェクト。
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
    yl = inp.ylrechole[1]
    yu = inp.yurechole[1]
    zl = inp.zlrechole[1] + 0.5
    zu = inp.zurechole[0]
    nx = inp.nx
    ny = inp.ny
    nz = inp.nz

    if axis == "xz":
        xs = np.array([0.0, xl, xl, xu, xu, nx - 1])
        ys = np.array([zu, zu, zl, zl, zu, zu])

    if use_si and unit is not None:
        xs = unit.length.reverse(xs)
        ys = unit.length.reverse(ys)

    im = plt.plot(xs + offsets[0], ys + offsets[1], color=color, linewidth=linewidth)

    return im


def plot_line_of_hole_half(inp, off, unit):
    """半断面ホール形状の補助線を 3D 軸に描画する。

    Parameters
    ----------
    inp : InpFile
        ホール定義を含む入力パラメータ。
    off : int
        拡張表示に使うオフセット幅。
    unit : UnitTranslator
        単位変換オブジェクト。

    Returns
    -------
    None
        戻り値はありません。
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

    bottom_points = np.array(
        [[xc, yl, zl], [xc, yu, zl], [xl, yu, zl], [xl, yl, zl], [xc, yl, zl]]
    )

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
        """点群にオフセットを適用して x, y, z を返す。
        
        Parameters
        ----------
        points : object
            `(N, 3)` 形式の座標配列です。
        offsets : object
            軸方向のオフセット。
        
        Returns
        -------
        object
            処理結果です。
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


def plot_surface_with_hole_half(
    data_xyz, inp, off=10, add_colorbar=True, show=False, vrange="minmax", **kwargs
):
    """矩形ホールの半断面サーフェスを描画する。

    Parameters
    ----------
    data_xyz : Data3d
        描画対象の 3 次元データです。
    inp : InpFile
        ホール形状（`xlrechole` など）を含む入力パラメータです。
    off : int, optional
        断面のオフセット幅です。
    add_colorbar : bool, optional
        最後の底面描画時にカラーバーを追加するかどうか。
    show : bool, optional
        `True` の場合は最後に `plt.show()` を呼び出します。
    vrange : {'minmax', 'center'}, optional
        カラースケール範囲の決定方法です。
    **kwargs : dict
        `Data2d.plot(mode="surf")` へ渡す引数です。
        主に `use_si`, `offsets`, `cmap`, `vmin`, `vmax`, `xlabel`, `ylabel`,
        `zlabel`, `title`, `ninterp`, `function`, `figsize`, `dpi`,
        および `mpl_toolkits.mplot3d.Axes3D.plot_surface` の追加引数を指定できます。

    Returns
    -------
    None
        戻り値はありません。

    Notes
    -----
    本関数内で `ax3d` は自動生成され、`kwargs["ax3d"]` に設定されます。
    そのため、呼び出し側で `ax3d` を渡した場合は上書きされます。
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

    ax = plt.gcf().add_subplot(projection='3d')
    plt.sca(ax)
    ax.set_box_aspect(box_aspect)

    xc = (xl + xu) // 2
    yc = (xl + xu) // 2
    zc = (xl + xu) // 2

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
        """有限値をそのまま通すマスク関数を返す。
        
        Parameters
        ----------
        x : object
            x 座標または x 成分。
        
        Returns
        -------
        object
            処理結果です。
        """
        return x == x

    masked = data_xyz.masked(alltrue)

    masked[zl - off // 2 : zu + 1, yl - off : yl + 1, xc].plot(mode="surf", **kwargs)
    masked[zl - off // 2 : zu + 1, yu : yu + off, xc].plot(mode="surf", **kwargs)
    masked[zl - off // 2 : zu + 1, yl - off, xl - off : xc + 1].plot(
        mode="surf", **kwargs
    )
    masked[zl - off // 2 : zu + 1, yu + off - 1, xl - off : xc + 1].plot(
        mode="surf", **kwargs
    )
    masked[zl - off // 2 : zl + 1, yl : yu + 1, xc].plot(mode="surf", **kwargs)

    # plot bottom
    data_xyz[zl, yl : yu + 1, xl : xc + 1].plot(
        mode="surf",
        add_colorbar=add_colorbar,
        xlabel="x[m]",
        ylabel="y[m]",
        zlabel="z[m]",
        **kwargs
    )

    if show:
        plt.show()
