import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from emout.plot.surface_cut import (
    Bounds3D,
    BoxMeshSurface,
    CylinderMeshSurface,
    Field3D,
    HollowCylinderMeshSurface,
    RectangleMeshSurface,
    RenderItem,
    SphereMeshSurface,
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


def test_hollow_cylinder_mesh_surface_top_cap_is_rect_with_hole():
    surface = HollowCylinderMeshSurface(
        center=(0.0, 0.0, 1.0),
        axis="z",
        width=4.0,
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
    # Inner edge is the hole circle of radius 0.75.
    assert np.isclose(r.min(), 0.75)
    # The rectangle bounds the cap at |x|,|y| <= 2.0 with corners at r=2√2.
    assert np.isclose(V[:, 0].max(), 2.0)
    assert np.isclose(V[:, 0].min(), -2.0)
    assert np.isclose(V[:, 1].max(), 2.0)
    assert np.isclose(V[:, 1].min(), -2.0)
    assert np.isclose(r.max(), 2.0 * np.sqrt(2.0))


def test_hollow_cylinder_mesh_surface_accepts_rectangular_width():
    surface = HollowCylinderMeshSurface(
        center=(0.0, 0.0, 0.0),
        axis="z",
        width=(6.0, 4.0),
        inner_radius=1.0,
        length=2.0,
        parts=("outer", "top", "bottom", "inner"),
        ntheta=16,
        nradial=3,
        naxial=2,
        nwall=2,
    )

    V, F = surface.mesh()

    assert F.size > 0
    # Outer rectangular bounds follow width_u=6, width_v=4.
    assert np.isclose(V[:, 0].max(), 3.0)
    assert np.isclose(V[:, 0].min(), -3.0)
    assert np.isclose(V[:, 1].max(), 2.0)
    assert np.isclose(V[:, 1].min(), -2.0)
    # Axial extent is length=2 centered on z=0.
    assert np.isclose(V[:, 2].max(), 1.0)
    assert np.isclose(V[:, 2].min(), -1.0)


def test_cylinder_mesh_surface_half_section_has_open_ends():
    full = CylinderMeshSurface(
        center=(0.0, 0.0, 0.0),
        axis="z",
        radius=1.0,
        length=2.0,
        parts=("side",),
        ntheta=16,
        naxial=3,
    )
    half = CylinderMeshSurface(
        center=(0.0, 0.0, 0.0),
        axis="z",
        radius=1.0,
        length=2.0,
        parts=("side",),
        ntheta=16,
        naxial=3,
        theta_range=(0.0, np.pi),
    )

    Vf, Ff = full.mesh()
    Vh, Fh = half.mesh()

    # The half has roughly half the triangle count.
    assert Fh.shape[0] < Ff.shape[0]
    # Every vertex is in the y >= 0 half-space (with numerical tolerance).
    assert Vh[:, 1].min() >= -1e-9
    # And includes the y=0 edge.
    assert np.isclose(Vh[:, 1].min(), 0.0)


def test_hollow_cylinder_mesh_surface_theta_range_limits_cap():
    surface = HollowCylinderMeshSurface(
        center=(0.0, 0.0, 0.0),
        axis="z",
        width=4.0,
        inner_radius=0.5,
        length=2.0,
        parts=("top",),
        ntheta=17,
        nradial=3,
        theta_range=(0.0, 0.5 * np.pi),
    )

    V, F = surface.mesh()

    assert F.size > 0
    # Quadrant: all vertices have x >= 0 and y >= 0.
    assert V[:, 0].min() >= -1e-9
    assert V[:, 1].min() >= -1e-9


def test_rectangle_mesh_surface_builds_flat_panel():
    surface = RectangleMeshSurface(
        center=(1.0, 2.0, 3.0),
        axis="z",
        width=(4.0, 2.0),
        resolution=(3, 5),
    )

    V, F = surface.mesh()

    assert V.shape == (15, 3)
    assert F.shape[0] == (5 - 1) * (3 - 1) * 2
    assert np.allclose(V[:, 2], 3.0)
    assert np.isclose(V[:, 0].min(), -1.0)
    assert np.isclose(V[:, 0].max(), 3.0)
    assert np.isclose(V[:, 1].min(), 1.0)
    assert np.isclose(V[:, 1].max(), 3.0)


def test_sphere_mesh_surface_points_lie_on_sphere():
    surface = SphereMeshSurface(
        center=(0.5, -1.0, 2.0),
        radius=3.0,
        ntheta=24,
        nphi=13,
    )

    V, F = surface.mesh()
    d = np.linalg.norm(V - np.array([0.5, -1.0, 2.0]), axis=1)

    assert F.size > 0
    assert np.allclose(d, 3.0, atol=1e-9)
    assert np.isclose(V[:, 2].max(), 2.0 + 3.0)
    assert np.isclose(V[:, 2].min(), 2.0 - 3.0)


def test_sphere_mesh_surface_hemisphere_has_phi_range():
    surface = SphereMeshSurface(
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        ntheta=16,
        nphi=9,
        phi_range=(0.0, 0.5 * np.pi),
    )

    V, F = surface.mesh()

    assert F.size > 0
    assert V[:, 2].min() >= -1e-9
    assert np.isclose(V[:, 2].max(), 1.0)


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


def test_data3d_plot_surfaces_accepts_boundary_collection(tmp_path):
    """Verify Data3d.plot_surfaces auto-wraps Boundary / BoundaryCollection.

    This is the one-line API entrypoint:
        data.phisp[-1].plot_surfaces(data.boundaries)
    or
        data.phisp[-1].plot_surfaces(data.boundaries[0] + data.boundaries[1])
    """
    from emout.emout.boundaries import BoundaryCollection
    from emout.emout.data.data import Data3d
    from emout.utils import InpFile, Units
    from emout.utils.units import UnitTranslator

    # Build an inp with two boundaries.
    inp_path = tmp_path / "plasma.inp"
    inp_path.write_text(
        "!!key dx=[0.1],to_c=[10000.0]\n"
        "&ptcond\n"
        "    boundary_type = 'complex'\n"
        "    boundary_types(1) = 'sphere'\n"
        "    boundary_types(2) = 'sphere'\n"
        "    sphere_origin(:, 1) = 2.0, 2.0, 2.0\n"
        "    sphere_radius(1) = 0.5\n"
        "    sphere_origin(:, 2) = 5.0, 5.0, 3.0\n"
        "    sphere_radius(2) = 0.7\n"
        "/\n"
    )
    inp = InpFile(inp_path)
    unit = Units(dx=0.1, to_c=10000.0)
    boundaries = BoundaryCollection(inp, unit)

    # Build a Data3d fixture covering the simulation domain.
    nx, ny, nz = 8, 8, 6
    data = np.arange(nx * ny * nz, dtype=np.float64).reshape(nz, ny, nx)
    length_unit = UnitTranslator(0.1, 1.0)  # dx=0.1m
    val_unit = UnitTranslator(1.0, 1.0)
    d3 = Data3d(
        data,
        filename="dummy.h5",
        axisunits=[None, length_unit, length_unit, length_unit],
        valunit=val_unit,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 1) Bare BoundaryCollection (treated as a single composite).
    cmap, _norm = d3.plot_surfaces(boundaries, ax=ax, use_si=True)
    assert cmap is not None

    # 2) Boundary + Boundary returns a BoundaryCollection — also auto-wrapped.
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    cmap2, _norm2 = d3.plot_surfaces(boundaries[0] + boundaries[1], ax=ax2, use_si=True)
    assert cmap2 is not None

    # 3) A single Boundary.
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection="3d")
    cmap3, _norm3 = d3.plot_surfaces(boundaries[0], ax=ax3, use_si=True)
    assert cmap3 is not None

    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)


def test_data3d_plot_surfaces_wraps_field_and_mesh():
    from emout.emout.data.data import Data3d
    from emout.utils.units import UnitTranslator

    nx, ny, nz = 8, 7, 6
    data = np.arange(nx * ny * nz, dtype=np.float64).reshape(nz, ny, nx)

    # Unit translators: grid→SI is divide-by-10 (dx=0.1 m equivalent).
    length_unit = UnitTranslator(0.1, 1.0)  # from_unit=0.1 m, to_unit=1 grid
    val_unit = UnitTranslator(1.0, 1.0)  # no scaling
    d3 = Data3d(
        data,
        filename="dummy.h5",
        axisunits=[None, length_unit, length_unit, length_unit],
        valunit=val_unit,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    mesh = BoxMeshSurface(
        0.1, 0.4, 0.1, 0.4, 0.05, 0.35, faces=("zmax",)
    )
    # Pass a bare MeshSurface3D — plot_surfaces should wrap it in a RenderItem.
    cmap, norm = d3.plot_surfaces(
        mesh,
        ax=ax,
        use_si=True,
        vmin=0.0,
        vmax=10.0,
    )

    assert cmap is not None
    assert norm.vmin == 0.0 and norm.vmax == 10.0
    assert len(ax.collections) >= 1
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
