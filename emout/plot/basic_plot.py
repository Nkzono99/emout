import copy

import emout.utils as utils
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
import matplotlib.colors as mcolors

_r = 0.98
_d = 0.5
mycmap = mcolors.LinearSegmentedColormap('gray-jet', {
    'red':   ((0.00, 0.3, 0.3),
              (_d*(1-_r), 0.3, 0.3),
              (0.35*_r+(1-_r), 0, 0),
              (0.66*_r+(1-_r), 1, 1),
              (0.89*_r+(1-_r), 1, 1),
              (1.00, 0.5, 0.5)),
    'green': ((0.00, 0.3, 0.3),
              (_d*(1-_r), 0.3, 0.3),
              (0.125*_r+(1-_r), 0, 0),
              (0.375*_r+(1-_r), 1, 1),
              (0.640*_r+(1-_r), 1, 1),
              (0.910*_r+(1-_r), 0, 0),
              (1.000, 0, 0)),
    'blue':  ((0.00, 0.3, 0.3),
              (_d*(1-_r), 0.3, 0.3),
              (0.00*_r+(1-_r), 0.5, 0.5),
              (0.11*_r+(1-_r), 1, 1),
              (0.34*_r+(1-_r), 1, 1),
              (0.65*_r+(1-_r), 0, 0),
              (1.00, 0, 0))
})


def figsize_with_2d(data2d, dpi=10):
    """2次元データから図のサイズを計算する.

    Parameters
    ----------
    data2d : numpy.ndarray
        2次元データ
    dpi : int, optional
        1データを何pixelで表すか, by default 10

    Returns
    -------
    (float, float)
        図のサイズ
    """
    px = 1/plt.rcParams['figure.dpi'] * dpi
    figsize = (data2d.shape[1]*px, data2d.shape[0]*px)
    return figsize


def plot_2dmap(data2d,
               mesh=None,
               savefilename=None,
               cmap=mycmap,
               mask_color='gray',
               vmin=None,
               vmax=None,
               figsize=None,
               xlabel=None,
               ylabel=None,
               title=None,
               interpolation='bilinear',
               dpi=10,
               **kwargs):
    """2次元カラーマップをプロットする.

    Parameters
    ----------
    data2d : numpy.ndarray
        2次元データ
    mesh : (numpy.ndarray, numpy.ndarray), optional
        メッシュ, by default None
    savefilename : str, optional
        保存するファイル名(Noneの場合保存しない), by default None
    cmap : matplotlib.Colormap or str or None, optional
        カラーマップ, by default cm.coolwarm
    mask_color : str
        マスクされた位置の色, by default 'gray'
    vmin : float, optional
        最小値, by default None
    vmax : float, optional
        最大値, by default None
    figsize : (float, float), optional
        図のサイズ, by default None
    xlabel : str, optional
        x軸のラベル, by default None
    ylabel : str, optional
        y軸のラベル, by default None
    title : str, optional
        タイトル, by default None
    interpolation : str, optional
        用いる補間方法, by default 'bilinear'
    dpi : int, optional
        解像度(figsizeが指定された場合は無視される), by default 10

    Returns
    -------
    AxesImage or None
        プロットしたimageデータ(保存した場合None)
    """
    if savefilename is not None:
        if figsize is None:
            fig = plt.figure()
        else:
            if figsize == 'auto':
                figsize = figsize_with_2d(data2d, dpi=dpi)
            fig = plt.figure(figsize=figsize)

    if mesh is None:
        x = list(range(data2d.shape[1]))
        y = list(range(data2d.shape[0]))
        mesh = np.meshgrid(x, y)

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = copy.copy(cm.get_cmap(str(cmap)))
        else:
            cmap = copy.copy(cmap)
        cmap.set_bad(color=mask_color)

    extent = [mesh[0][0, 0], mesh[0][-1, -1],
              mesh[1][0, 0], mesh[1][-1, -1]]
    img = plt.imshow(data2d,
                     interpolation=interpolation,
                     cmap=cmap,
                     origin='lower',
                     vmin=vmin,
                     vmax=vmax,
                     extent=extent,
                     aspect='auto')
    plt.colorbar()

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


def plot_2d_contour(data2d,
                    mesh=None,
                    levels=None,
                    colors=['black'],
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
                    fmt='%1.1f',
                    fontsize=12,
                    **kwargs):
    """2次元等高線をプロットする.

    Parameters
    ----------
    data2d : numpy.ndarray
        2次元データ
    mesh : (numpy.ndarray, numpy.ndarray), optional
        メッシュ, by default None
    levels : int
        等高線数, by default None
    alpha : float
        透明度(0.0~1.0), by default 1
    savefilename : str, optional
        保存するファイル名(Noneの場合保存しない), by default None
    cmap : matplotlib.Colormap or str or None, optional
        カラーマップ, by default None
    mask_color : str
        マスクされた位置の色, by default 'gray'
    vmin : float, optional
        最小値, by default None
    vmax : float, optional
        最大値, by default None
    figsize : (float, float), optional
        図のサイズ, by default None
    xlabel : str, optional
        x軸のラベル, by default None
    ylabel : str, optional
        y軸のラベル, by default None
    title : str, optional
        タイトル, by default None
    interpolation : str, optional
        用いる補間方法, by default 'bilinear'
    dpi : int, optional
        解像度(figsizeが指定された場合は無視される), by default 10
    fmt : str
        clabelの形式, by default '%1.1f'
    fontsize : str
        clabelのフォントサイズ, by default 12

    Returns
    -------
    AxesImage or None
        プロットしたimageデータ(保存した場合None)
    """
    if savefilename is not None:
        if figsize is None:
            fig = plt.figure()
        else:
            if figsize == 'auto':
                figsize = figsize_with_2d(data2d, dpi=dpi)
            fig = plt.figure(figsize=figsize)

    if mesh is None:
        x = list(range(data2d.shape[1]))
        y = list(range(data2d.shape[0]))
        mesh = np.meshgrid(x, y)

    kwargs = {
        'alpha': alpha,
        'vmin': vmin,
        'vmax': vmax,
    }
    print(cmap)
    if cmap is None:
        kwargs['colors'] = colors
    else:
        kwargs['cmap'] = cmap
    if levels is not None:
        kwargs['levels'] = levels
    cont = plt.contour(*mesh, data2d, **kwargs)
    cont.clabel(fmt=fmt, fontsize=fontsize)

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
        return cont


def plot_surface(x, y, z, value,
                 ax3d=None,
                 add_colorbar=False,
                 savefilename=None,
                 cmap=cm.jet,
                 mask_color='gray',
                 vmin=None,
                 vmax=None,
                 figsize=None,
                 xlabel=None,
                 ylabel=None,
                 zlabel=None,
                 title=None,
                 ninterp=1,
                 function='linear',
                 dpi=10):
    """3次元表面プロットをする.

    Parameters
    ----------
    x : (numpy.ndarray, numpy.ndarray), optional
        x座標のメッシュ
    y : (numpy.ndarray, numpy.ndarray), optional
        y座標のメッシュ
    z : (numpy.ndarray, numpy.ndarray), optional
        z座標のメッシュ
    val : (numpy.ndarray, numpy.ndarray), optional
        値のメッシュ
    ax3d : Axes3D
        Axes3Dオブジェクト, by default None
    savefilename : str, optional
        保存するファイル名(Noneの場合保存しない), by default None
    cmap : matplotlib.Colormap or str or None, optional
        カラーマップ, by default cm.coolwarm
    vmin : float, optional
        最小値, by default None
    vmax : float, optional
        最大値, by default None
    figsize : (float, float), optional
        図のサイズ, by default None
    xlabel : str, optional
        x軸のラベル, by default None
    ylabel : str, optional
        y軸のラベル, by default None
    zlabel : str, optional
        z軸のラベル, by default None
    title : str, optional
        タイトル, by default None
    dpi : int, optional
        解像度(figsizeが指定された場合は無視される), by default 10

    Returns
    -------
    AxesImage or None
        プロットしたimageデータ(保存した場合None)
    """
    if savefilename is not None:
        if figsize is None:
            fig = plt.figure()
        else:
            if figsize == 'auto':
                figsize = figsize_with_2d(x, dpi=dpi)
            fig = plt.figure(figsize=figsize)
    else:
        fig = plt.gcf()

    if ax3d is None:
        ax3d = fig.gca(projection='3d')

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = copy.copy(cm.get_cmap(str(cmap)))
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

    surf = ax3d.plot_surface(x, y, z,
                             facecolors=value_colors,
                             vmin=vmin,
                             vmax=vmax,
                             shade=False)
    if add_colorbar:
        plt.colorbar(mappable, ax=ax3d)

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


def plot_line(data1d,
              x=None,
              savefilename=None,
              vmin=None,
              vmax=None,
              figsize=None,
              xlabel=None,
              ylabel=None,
              label=None,
              title=None):
    """1次元データをプロットする.

    Parameters
    ----------
    data1d : array-like or scalar
        プロットする1次元データ
    x : array-like or scalar
        横軸となる1次元データ, by default None
    savefilename : str, optional
        保存するファイル名, by default None
    vmin : float, optional
        最小値, by default None
    vmax : float, optional
        最大値, by default None
    figsize : (float, float), optional
        図のサイズ, by default None
    xlabel : str, optional
        横軸のラベル, by default None
    ylabel : str, optional
        縦軸のラベル, by default None
    label : str, optional
        ラベル, by default None
    title : str, optional
        タイトル, by default None

    Returns
    -------
    Line2D or None
        プロットデータを表す線オブジェクト(保存した場合None)
    """
    if savefilename is not None:
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)

    if x is None:
        line = plt.plot(data1d, label=label)
    else:
        line = plt.plot(x, data1d, label=label)
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


def plot_2d_vector(x_data2d,
                   y_data2d,
                   mesh=None,
                   savefilename=None,
                   color=None,
                   scale=1,
                   scaler='standard',
                   skip=1,
                   easy_to_read=True,
                   figsize=None,
                   xlabel=None,
                   ylabel=None,
                   title=None,
                   dpi=10):
    """2次元ベクトル図をプロットする.

    Parameters
    ----------
    x_data2d, y_data2d : numpy.ndarray
        2次元データ
    mesh : (numpy.ndarray, numpy.ndarray), optional
        メッシュ, by default None
    savefilename : str, optional
        保存するファイル名(Noneの場合保存しない), by default None
    color : str
        ベクトルの色, by default None
    scale : float
        ベクトルの大きさ係数(最終的な大きさにこの値を掛ける), by default 1
    skip : int
        プロットするデータ間隔, by default 1
    easy_to_read : bool
        ベクトルを見やすい大きさにスケーリングするならTrue, by default True
    figsize : (float, float), optional
        図のサイズ, by default None
    xlabel : str, optional
        x軸のラベル, by default None
    ylabel : str, optional
        y軸のラベル, by default None
    title : str, optional
        タイトル, by default None
    interpolation : str, optional
        用いる補間方法, by default 'bilinear'
    dpi : int, optional
        解像度(figsizeが指定された場合は無視される), by default 10

    Returns
    -------
    AxesImage or None
        プロットしたimageデータ(保存した場合None)
    """
    fig = None
    if savefilename is not None:
        if figsize is None:
            fig = plt.figure()
        else:
            if figsize == 'auto':
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

    x_skip = skip if type(skip) == int else skip[0]
    y_skip = skip if type(skip) == int else skip[1]
    x = x[::y_skip, ::x_skip]
    y = y[::y_skip, ::x_skip]
    U = U[::y_skip, ::x_skip]
    V = V[::y_skip, ::x_skip]

    norm = np.sqrt(U**2 + V**2)

    if scaler == 'standard':
        norm_max = np.nanmax(np.abs(norm))
        U /= norm_max
        V /= norm_max

    elif scaler == 'normal':
        U /= norm
        V /= norm

    elif scaler == 'log':
        U = U / norm * np.log(norm+1)
        V = V / norm * np.log(norm+1)

    # 見やすい大きさに線形スケーリングを行う
    if easy_to_read:
        dx = (x.max() - x.min()) / x.shape[0]
        multiplier = dx * 1.2
        norm_mean = np.nanmean(np.sqrt(U**2 + V**2))
        U *= scale / norm_mean * multiplier
        V *= scale / norm_mean * multiplier

    img = plt.quiver(x,
                     y,
                     U,
                     V,
                     angles='xy',
                     scale_units='xy',
                     scale=1,
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
