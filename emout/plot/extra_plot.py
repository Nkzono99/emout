import matplotlib.pyplot as plt
import numpy as np


def plot_surface_with_hole(data_xyz, inp, add_colorbar=True, show=False, vrange='minmax', **kwargs):
    xl = int(inp.xlrechole[1])
    xu = int(inp.xurechole[1])
    yl = int(inp.ylrechole[1])
    yu = int(inp.yurechole[1])
    zl = int(inp.zlrechole[1])
    zu = int(inp.zurechole[0])

    if kwargs.get('use_si', False):
        vmax = data_xyz.val_si.max()
        vmin = data_xyz.val_si.min()
    else:
        vmax = data_xyz.max()
        vmin = data_xyz.min()

    if vrange == 'minmax':
        pass
    elif vrange == 'center':
        vmax = max(abs(vmax), abs(vmin))
        vmin = -vmax

    kwargs['vmax'] = kwargs.get('vmax', None) or vmax
    kwargs['vmin'] = kwargs.get('vmin', None) or vmin

    # plot top surface
    data_xyz[zu, :yl+1, :].plot(mode='surf', **kwargs)
    data_xyz[zu, yu:, :].plot(mode='surf', **kwargs)
    data_xyz[zu, yl:yu+1, :xl+1].plot(mode='surf', **kwargs)
    data_xyz[zu, yl:yu+1, xu:].plot(mode='surf', **kwargs)

    # plot wall
    data_xyz[zl:zu+1, yl, xl:xu+1].plot(mode='surf', **kwargs)
    data_xyz[zl:zu+1, yu, xl:xu+1].plot(mode='surf', **kwargs)
    data_xyz[zl:zu+1, yl:yu+1, xl].plot(mode='surf', **kwargs)
    data_xyz[zl:zu+1, yl:yu+1, xu].plot(mode='surf', **kwargs)

    # plot bottom
    data_xyz[zl, yl:yu+1, xl:xu+1].plot(
        mode='surf',
        add_colorbar=add_colorbar,
        **kwargs)

    if show:
        plt.show()


def plot_line_of_hole_half(inp, off, unit):
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

    surf_points = np.array([
        [xc, yl-off, zu],
        [xc, yl, zu],
        [xl, yl, zu],
        [xl, yu, zu],
        [xc, yu, zu],
        [xc, yu+off, zu],
        [xc-off, yu+off, zu],
        [xc-off, yl-off, zu],
        [xc, yl-off, zu]
    ])

    bottom_points = np.array([
        [xc, yl, zl],
        [xc, yu, zl],
        [xl, yu, zl],
        [xl, yl, zl],
        [xc, yl, zl]
    ])

    bottom2_points = np.array([
        [xc, yl-off, zl-off//2],
        [xc, yu+off, zl-off//2],
        [xc-off, yu+off, zl-off//2],
        [xc-off, yl-off, zl-off//2],
        [xc, yl-off, zl-off//2]
    ])

    lines = np.array([
        [[xc, yl, zl], [xc, yl, zu]],
        [[xl, yl, zl], [xl, yl, zu]],
        [[xl, yu, zl], [xl, yu, zu]],
        [[xc, yu, zl], [xc, yu, zu]],
        [[xc, yl-off, zl-off//2], [xc, yl-off, zu]],
        [[xc, yu+off, zl-off//2], [xc, yu+off, zu]],
        [[xc-off, yu+off, zl-off//2], [xc-off, yu+off, zu]],
        [[xc-off, yl-off, zl-off//2], [xc-off, yl-off, zu]]
    ])

    surf_points = unit.reverse(surf_points)
    bottom_points = unit.reverse(bottom_points)
    bottom2_points = unit.reverse(bottom2_points)
    lines = unit.reverse(lines)

    def parse(points, offsets):
        x = points[:, 0] + offsets[0]
        y = points[:, 1] + offsets[1]
        z = points[:, 2] + offsets[2]
        return x, y, z

    offsets = (-xc_si, -yc_si, -zu_si)
    ax.plot(*parse(surf_points, offsets), color='black')
    ax.plot(*parse(bottom_points, offsets), color='black')
    ax.plot(*parse(bottom2_points, offsets), color='black')

    for line in lines:
        ax.plot(*parse(line, offsets), color='black')


def plot_surface_with_hole_half(data_xyz, inp, off=10, add_colorbar=True, show=False, vrange='minmax', **kwargs):
    xl = int(inp.xlrechole[1])
    xu = int(inp.xurechole[1])
    yl = int(inp.ylrechole[1])
    yu = int(inp.yurechole[1])
    zl = int(inp.zlrechole[1])
    zu = int(inp.zurechole[0])

    xlen = (xu - xl)/2 + off
    ylen = yu - yl + off * 2
    zlen = zu - zl + off

    lenmax = max(max(xlen, ylen), zlen)
    box_aspect = (xlen/lenmax, ylen/lenmax, zlen/lenmax)

    plt.gca(projection='3d').set_box_aspect(box_aspect)

    xc = (xl + xu) // 2
    yc = (xl + xu) // 2
    zc = (xl + xu) // 2

    if kwargs.get('use_si', False):
        vmax = data_xyz.val_si.max()
        vmin = data_xyz.val_si.min()
    else:
        vmax = data_xyz.max()
        vmin = data_xyz.min()

    if vrange == 'minmax':
        pass
    elif vrange == 'center':
        vmax = max(abs(vmax), abs(vmin))
        vmin = -vmax

    kwargs['vmax'] = kwargs.get('vmax', None) or vmax
    kwargs['vmin'] = kwargs.get('vmin', None) or vmin

    # plot top surface
    data_xyz[zu, yl-off:yl+1, xl-off:xc+1].plot(mode='surf', **kwargs)
    data_xyz[zu, yu:yu+off, xl-off:xc+1].plot(mode='surf', **kwargs)
    data_xyz[zu, yl:yu+1, xl-off:xl+1].plot(mode='surf', **kwargs)

    # plot wall
    data_xyz[zl:zu+1, yl, xl:xc+1].plot(mode='surf', **kwargs)
    data_xyz[zl:zu+1, yu, xl:xc+1].plot(mode='surf', **kwargs)
    data_xyz[zl:zu+1, yl:yu+1, xl].plot(mode='surf', **kwargs)

    # innner wall
    def alltrue(x):
        return x == x

    masked = data_xyz.masked(alltrue)

    masked[zl-off//2:zu+1, yl-off:yl+1, xc].plot(mode='surf', **kwargs)
    masked[zl-off//2:zu+1, yu:yu+off, xc].plot(mode='surf', **kwargs)
    masked[zl-off//2:zu+1, yl-off, xl-off:xc+1].plot(mode='surf', **kwargs)
    masked[zl-off//2:zu+1, yu+off-1, xl-off:xc+1].plot(mode='surf', **kwargs)
    masked[zl-off//2:zl+1, yl:yu+1, xc].plot(mode='surf', **kwargs)

    # plot bottom
    data_xyz[zl, yl:yu+1, xl:xc+1].plot(
        mode='surf',
        add_colorbar=add_colorbar,
        xlabel='x[m]',
        ylabel='y[m]',
        zlabel='z[m]',
        **kwargs)

    if show:
        plt.show()
