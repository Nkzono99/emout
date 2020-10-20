import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_2dmap(data2d,
               mesh=None,
               savefilename=None,
               cmap=cm.coolwarm,
               vmin=None,
               vmax=None,
               figsize=None,
               xlabel=None,
               ylabel=None,
               title=None,
               interpolation='bilinear',
               dpi=10):
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
    """
    if savefilename is not None:
        if figsize is None:
            px = 1/plt.rcParams['figure.dpi'] * dpi
            figsize = (data2d.shape[1]*px, data2d.shape[0]*px)
        fig = plt.figure(figsize=figsize)

    if mesh is None:
        x = list(range(data2d.shape[1]))
        y = list(range(data2d.shape[0]))
        mesh = np.meshgrid(x, y)

    extent = [mesh[0][0, 0], mesh[0][-1, -1],
              mesh[1][0, 0], mesh[1][-1, -1]]
    plt.imshow(data2d,
               interpolation=interpolation,
               cmap=cmap,
               origin='lower',
               vmin=vmin,
               vmax=vmax,
               extent=extent)
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
    """
    if savefilename is not None:
        fig = plt.figure(figsize=figsize)

    if x is None:
        plt.plot(data1d, label=label)
    else:
        plt.plot(x, data1d, label=label)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.ylim([vmin, vmax])

    if savefilename is not None:
        fig.savefig(savefilename)
        plt.close(fig)