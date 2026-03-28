import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from emout.plot.surface_cut import (
    Bounds3D,
    BoxMeshSurface,
    CylinderMeshSurface,
    Field3D,
    HollowCylinderMeshSurface,
    RenderItem,
    UniformCellCenteredGrid,
    plot_surfaces,
)

matplotlib.use("Agg")


def _make_field():
    grid = UniformCellCenteredGrid(nx=8, ny=7, nz=6, dx=1.0, dy=1.0, dz=1.0)
    z = grid.z_centers()[:, None, None]
    y = grid.y_centers()[None, :, None]
    x = grid.x_centers()[None, None, :]
    data = x + 2.0 * y + 3.0 * z
    return Field3D(grid, data)


def test_box_mesh_surface_selects_requested_face():
    surface = BoxMeshSurface(
        0.0,
        2.0,
        0.0,
        3.0,
        0.0,
        4.0,
        faces=("zmax",),
        resolution=(3, 4),
    )

    V, F = surface.mesh()

    assert V.shape == (12, 3)
    assert F.shape == (12, 3)
    assert np.allclose(V[:, 2], 4.0)
    assert np.isclose(V[:, 0].min(), 0.0)
    assert np.isclose(V[:, 0].max(), 2.0)
    assert np.isclose(V[:, 1].min(), 0.0)
    assert np.isclose(V[:, 1].max(), 3.0)


def test_cylinder_mesh_surface_side_has_constant_radius():
    surface = CylinderMeshSurface(
        center=(1.0, -2.0, 0.5),
        axis="z",
        radius=1.5,
        length=4.0,
        parts=("side",),
        ntheta=18,
        naxial=5,
    )

    V, F = surface.mesh()
    r = np.sqrt((V[:, 0] - 1.0) ** 2 + (V[:, 1] + 2.0) ** 2)

    assert F.shape[0] == 18 * (5 - 1) * 2
    assert np.allclose(r, 1.5)
    assert np.isclose(V[:, 2].min(), -1.5)
    assert np.isclose(V[:, 2].max(), 2.5)


def test_hollow_cylinder_mesh_surface_top_cap_is_annulus():
    surface = HollowCylinderMeshSurface(
        center=(0.0, 0.0, 1.0),
        axis="z",
        outer_radius=2.0,
        inner_radius=0.75,
        length=6.0,
        parts=("top",),
        ntheta=24,
        nradial=4,
    )

    V, F = surface.mesh()
    r = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)

    assert F.size > 0
    assert np.allclose(V[:, 2], 4.0)
    assert np.isclose(r.min(), 0.75)
    assert np.isclose(r.max(), 2.0)


def test_plot_surfaces_accepts_explicit_mesh_surfaces():
    field = _make_field()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    bounds = Bounds3D((0.0, 8.0), (0.0, 7.0), (0.0, 6.0))
    surfaces = [
        RenderItem(
            surface=BoxMeshSurface(
                1.0,
                5.0,
                1.0,
                4.0,
                1.0,
                3.0,
                faces=("zmax",),
                resolution=(6, 5),
            ),
            style="field",
        ),
        RenderItem(
            surface=CylinderMeshSurface(
                center=(3.0, 3.0, 2.0),
                axis="z",
                radius=1.0,
                length=2.0,
                parts=("side",),
                ntheta=20,
                naxial=4,
            ),
            style="solid",
            solid_color="tab:orange",
            alpha=0.5,
        ),
    ]

    cmap, norm = plot_surfaces(
        ax,
        field=field,
        surfaces=surfaces,
        bounds=bounds,
        contour_levels=[8.0, 10.0, 12.0],
    )

    assert cmap is not None
    assert norm.vmin < norm.vmax
    assert len(ax.collections) >= 2

    plt.close(fig)


def test_plot_surfaces_can_render_solid_mesh_without_field():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    plot_surfaces(
        ax,
        field=None,
        surfaces=RenderItem(
            surface=BoxMeshSurface(
                -1.0,
                1.0,
                -2.0,
                2.0,
                0.0,
                3.0,
                faces=("xmax", "zmax"),
            ),
            style="solid",
            solid_color="0.7",
        ),
    )

    assert len(ax.collections) == 1
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    assert xlim[0] < -1.0 and xlim[1] > 1.0
    assert ylim[0] < -2.0 and ylim[1] > 2.0
    assert zlim[0] < 0.0 and zlim[1] > 3.0

    plt.close(fig)
