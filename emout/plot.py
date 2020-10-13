import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def plot_2dmap(data2d, mesh=None, savefilename=None, cmap=cm.coolwarm,  vmin=None, vmax=None, figsize=None, xlabel=None, ylabel=None):
    if savefilename is not None:
        if figsize is None:
            px = 1/plt.rcParams['figure.dpi'] * 5
            figsize = (data2d.shape[1]*px, data2d.shape[0]*px)
        fig = plt.figure(figsize=figsize)

    if mesh is None:
        x = list(range(data2d.shape[1]))
        y = list(range(data2d.shape[0]))
        mesh = np.meshgrid(x, y)

    plt.pcolor(mesh[0], mesh[1], data2d,
               cmap=cmap, vmin=vmin, vmax=vmax,
               shading='auto',
               label=data2d.name)
    plt.colorbar()
    plt.legend()

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savefilename is not None:
        fig.savefig(savefilename)


def plot_xy(data, z, savefilename=None, cmap=cm.coolwarm,  vmin=None, vmax=None, figsize=None):
    x = list(range(data.shape[2]))
    y = list(range(data.shape[1]))
    mesh = np.meshgrid(x, y)
    plot_2dmap(data[z, :, :],
               mesh=mesh,
               savefilename=savefilename,
               cmap=cmap,
               vmin=vmin,
               vmax=vmax,
               figsize=figsize)


def plot_yz(data, x, savefilename=None, cmap=cm.coolwarm,  vmin=None, vmax=None, figsize=None):
    y = list(range(data.shape[1]))
    z = list(range(data.shape[0]))
    mesh = np.meshgrid(y, z)
    plot_2dmap(data[:, :, x],
               mesh=mesh,
               savefilename=savefilename,
               cmap=cmap,
               vmin=vmin,
               vmax=vmax,
               figsize=figsize)


def plot_xz(data, y, savefilename=None, cmap=cm.coolwarm,  vmin=None, vmax=None, figsize=None):
    x = list(range(data.shape[2]))
    z = list(range(data.shape[0]))
    mesh = np.meshgrid(x, z)
    plot_2dmap(data[:, y, :],
               mesh=mesh,
               savefilename=savefilename,
               cmap=cmap,
               vmin=vmin,
               vmax=vmax,
               figsize=figsize)


def plot_line(data1d, savefilename=None, vmin=None, vmax=None, figsize=None):
    if savefilename is not None:
        fig = plt.figure(figsize=figsize)

    plt.plot(data1d)

    if savefilename is not None:
        fig.savefig(savefilename)
