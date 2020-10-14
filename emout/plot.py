import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from emout.data import GridData3d
from scipy.interpolate import griddata


def plot_2dmap(data2d,
               mesh=None,
               savefilename=None,
               cmap=cm.coolwarm,
               vmin=None,
               vmax=None,
               figsize=None,
               xlabel=None,
               ylabel=None,
               nintrp=5,
               dpi=10):
    if savefilename is not None:
        if figsize is None:
            px = 1/plt.rcParams['figure.dpi'] * dpi
            figsize = (data2d.shape[1]*px, data2d.shape[0]*px)
        fig = plt.figure(figsize=figsize)

    if mesh is None:
        x = list(range(data2d.shape[1]))
        y = list(range(data2d.shape[0]))
        mesh = np.meshgrid(x, y)

    plt.imshow(data2d,
               interpolation='bilinear',
               cmap=cmap,
               origin='lower',
               vmin=vmin,
               vmax=vmax)

    if hasattr(data2d, 'name'):
        plt.title(data2d.name)

    plt.colorbar()

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savefilename is not None:
        fig.savefig(savefilename)


def plot_xy(data3d, z, savefilename=None, cmap=cm.coolwarm,  vmin=None, vmax=None, figsize=None):
    if isinstance(data3d, GridData3d):
        x = list(range(data3d.xslice.start,
                       data3d.xslice.stop, data3d.xslice.step))
        y = list(range(data3d.yslice.start,
                       data3d.yslice.stop, data3d.yslice.step))
    else:
        x = list(range(data3d.shape[2]))
        y = list(range(data3d.shape[1]))
    mesh = np.meshgrid(x, y)
    plot_2dmap(data3d[z, :, :],
               mesh=mesh,
               savefilename=savefilename,
               cmap=cmap,
               vmin=vmin,
               vmax=vmax,
               figsize=figsize,
               xlabel='x',
               ylabel='y')


def plot_yz(data3d, x, savefilename=None, cmap=cm.coolwarm,  vmin=None, vmax=None, figsize=None):
    if isinstance(data3d, GridData3d):
        y = list(range(data3d.yslice.start,
                       data3d.yslice.stop, data3d.yslice.step))
        z = list(range(data3d.zslice.start,
                       data3d.zslice.stop, data3d.zslice.step))
    else:
        y = list(range(data3d.shape[1]))
        z = list(range(data3d.shape[0]))
    mesh = np.meshgrid(y, z)
    plot_2dmap(data3d[:, :, x],
               mesh=mesh,
               savefilename=savefilename,
               cmap=cmap,
               vmin=vmin,
               vmax=vmax,
               figsize=figsize,
               xlabel='y',
               ylabel='z')


def plot_xz(data3d, y, savefilename=None, cmap=cm.coolwarm,  vmin=None, vmax=None, figsize=None):
    if isinstance(data3d, GridData3d):
        x = list(range(data3d.xslice.start,
                       data3d.xslice.stop, data3d.xslice.step))
        z = list(range(data3d.zslice.start,
                       data3d.zslice.stop, data3d.zslice.step))
    else:
        x = list(range(data3d.shape[2]))
        z = list(range(data3d.shape[0]))
    mesh = np.meshgrid(x, z)
    plot_2dmap(data3d[:, y, :],
               mesh=mesh,
               savefilename=savefilename,
               cmap=cmap,
               vmin=vmin,
               vmax=vmax,
               figsize=figsize,
               xlabel='x',
               ylabel='z')


def plot_line(data1d, x=None, savefilename=None, vmin=None, vmax=None, figsize=None, xlabel=None, ylabel=None):
    if savefilename is not None:
        fig = plt.figure(figsize=figsize)

    name = data2d.name if hasattr(data2d, 'name') else None

    if x is None:
        plt.plot(data1d, label=name)
    else:
        plt.plot(x, data1d, label=name)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if savefilename is not None:
        fig.savefig(savefilename)


def plot_xline(data3d, y, z, savefilename=None, vmin=None, vmax=None, figsize=None):
    if isinstance(data3d, GridData3d):
        x = list(range(data3d.xslice.start,
                       data3d.xslice.stop, data3d.xslice.step))
    else:
        x = list(range(data3d.shape[2]))
    plot_line(data3d[z, y, :],
              x=x,
              savefilename=savefilename,
              vmin=vmin,
              vmax=vmax,
              figsize=figsize,
              xlabel='x',
              ylabel=data3d.name)


def plot_yline(data3d, x, z, savefilename=None, vmin=None, vmax=None, figsize=None):
    if isinstance(data3d, GridData3d):
        y = list(range(data3d.yslice.start,
                       data3d.yslice.stop, data3d.yslice.step))
    else:
        y = list(range(data3d.shape[1]))
    plot_line(data3d[z, :, x],
              x=y,
              savefilename=savefilename,
              vmin=vmin,
              vmax=vmax,
              figsize=figsize,
              xlabel='y',
              ylabel=data3d.name)


def plot_zline(data3d, x, y, savefilename=None, vmin=None, vmax=None, figsize=None):
    if isinstance(data3d, GridData3d):
        z = list(range(data3d.zslice.start,
                       data3d.zslice.stop, data3d.zslice.step))
    else:
        z = list(range(data3d.shape[0]))
    plot_line(data3d[:, y, x],
              x=z,
              savefilename=savefilename,
              vmin=vmin,
              vmax=vmax,
              figsize=figsize,
              xlabel='z',
              ylabel=data3d.name)
