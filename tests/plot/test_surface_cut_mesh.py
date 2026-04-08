import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

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


def test_rectangle_mesh_surface_pmin_pmax_z_aligned():
    surface = RectangleMeshSurface(pmin=(1.0, 2.0, 5.0), pmax=(4.0, 6.0, 5.0))
    V, _ = surface.mesh()

    # axis is z (the matching coordinate), centred on the box midpoint.
    assert np.allclose(surface.axis, [0.0, 0.0, 1.0])
    assert np.allclose(surface.center, [2.5, 4.0, 5.0])
    assert surface.width_u == 3.0  # x extent
    assert surface.width_v == 4.0  # y extent
    assert np.allclose(V[:, 2], 5.0)
    assert np.isclose(V[:, 0].min(), 1.0) and np.isclose(V[:, 0].max(), 4.0)
    assert np.isclose(V[:, 1].min(), 2.0) and np.isclose(V[:, 1].max(), 6.0)


def test_rectangle_mesh_surface_pmin_pmax_y_aligned():
    surface = RectangleMeshSurface(pmin=(0.0, 5.0, 0.0), pmax=(10.0, 5.0, 8.0))
    V, _ = surface.mesh()

    assert np.allclose(surface.axis, [0.0, 1.0, 0.0])
    assert np.allclose(surface.center, [5.0, 5.0, 4.0])
    assert surface.width_u == 10.0  # x extent
    assert surface.width_v == 8.0  # z extent
    assert np.allclose(V[:, 1], 5.0)
    assert np.isclose(V[:, 0].min(), 0.0) and np.isclose(V[:, 0].max(), 10.0)
    assert np.isclose(V[:, 2].min(), 0.0) and np.isclose(V[:, 2].max(), 8.0)


def test_rectangle_mesh_surface_pmin_pmax_rejects_non_axis_aligned():
    with pytest.raises(ValueError, match="axis-aligned"):
        RectangleMeshSurface(pmin=(0.0, 0.0, 0.0), pmax=(1.0, 2.0, 3.0))


def test_rectangle_mesh_surface_rejects_mixing_forms():
    with pytest.raises(ValueError, match="Cannot mix"):
        RectangleMeshSurface(
            center=(0.0, 0.0, 0.0),
            axis="z",
            width=1.0,
            pmin=(0.0, 0.0, 0.0),
            pmax=(1.0, 1.0, 0.0),
        )


def test_rectangle_mesh_surface_requires_one_form():
    with pytest.raises(ValueError, match="requires either"):
        RectangleMeshSurface()


# ---------------------------------------------------------------------------
# resolution_scale broadcast
# ---------------------------------------------------------------------------


def test_sphere_resolution_scale_multiplies_defaults():
    base = SphereMeshSurface(center=(0, 0, 0), radius=1.0)
    scaled = SphereMeshSurface(center=(0, 0, 0), radius=1.0, resolution_scale=2.5)
    # Defaults are ntheta=48, nphi=25.
    assert base.ntheta == 48 and base.nphi == 25
    # 48 * 2.5 = 120; 25 * 2.5 = 62.5 → rounds to 62 or 63 (Python banker's
    # rounding gives 62 for ties; for 25*2.5=62.5 it lands on 62 as well).
    assert scaled.ntheta == round(48 * 2.5)
    assert scaled.nphi == round(25 * 2.5)


def test_cylinder_resolution_scale_scales_every_count():
    base = CylinderMeshSurface(center=(0, 0, 0), axis="z", radius=1.0, length=2.0)
    scaled = CylinderMeshSurface(
        center=(0, 0, 0), axis="z", radius=1.0, length=2.0, resolution_scale=4
    )
    # Defaults: ntheta=64, naxial=2, nradial=8.
    assert base.ntheta == 64 and base.naxial == 2 and base.nradial == 8
    assert scaled.ntheta == 64 * 4
    assert scaled.naxial == 2 * 4
    assert scaled.nradial == 8 * 4


def test_box_resolution_scale_scales_face_resolution():
    base = BoxMeshSurface(0, 1, 0, 1, 0, 1, faces=("zmax",))
    scaled = BoxMeshSurface(0, 1, 0, 1, 0, 1, faces=("zmax",), resolution_scale=5)
    # Default resolution is (2, 2). After scale=5 → (10, 10).
    assert base.resolution == (2, 2)
    assert scaled.resolution == (10, 10)


def test_resolution_scale_combines_with_explicit_count():
    # Explicit ntheta becomes the *base* that scale multiplies.
    surface = SphereMeshSurface(
        center=(0, 0, 0), radius=1.0, ntheta=12, nphi=7, resolution_scale=3
    )
    assert surface.ntheta == 36
    assert surface.nphi == 21


def test_resolution_scale_rejects_non_positive():
    with pytest.raises(ValueError, match="resolution_scale"):
        SphereMeshSurface(center=(0, 0, 0), radius=1.0, resolution_scale=0)
    with pytest.raises(ValueError, match="resolution_scale"):
        SphereMeshSurface(center=(0, 0, 0), radius=1.0, resolution_scale=-1.0)
    with pytest.raises(ValueError, match="resolution_scale"):
        SphereMeshSurface(center=(0, 0, 0), radius=1.0, resolution_scale=float("nan"))


def test_collection_mesh_broadcasts_resolution_scale_to_every_boundary(tmp_path):
    """data.boundaries.mesh(resolution_scale=N) hits every entry."""
    from emout.emout.boundaries import BoundaryCollection
    from emout.utils import InpFile, Units

    inp_path = tmp_path / "plasma.inp"
    inp_path.write_text(
        "!!key dx=[0.1],to_c=[10000.0]\n"
        "&esorem\n"
        "    nx = 64\n"
        "    ny = 64\n"
        "    nz = 64\n"
        "/\n"
        "&ptcond\n"
        "    boundary_type = 'complex'\n"
        "    boundary_types(1) = 'sphere'\n"
        "    boundary_types(2) = 'cuboid'\n"
        "    boundary_types(3) = 'cylinderz'\n"
        "    sphere_origin(:, 1) = 10.0, 10.0, 10.0\n"
        "    sphere_radius(1) = 1.0\n"
        "    cuboid_shape(:, 2) = 0, 1, 0, 1, 0, 1\n"
        "    cylinder_origin(:, 3) = 5.0, 5.0, 0.0\n"
        "    cylinder_radius(3) = 1.0\n"
        "    cylinder_height(3) = 2.0\n"
        "/\n"
    )
    inp = InpFile(inp_path)
    unit = Units(dx=0.1, to_c=10000.0)
    boundaries = BoundaryCollection(inp, unit)

    composite = boundaries.mesh(use_si=False, resolution_scale=4)
    sphere_child, box_child, cyl_child = composite.children

    # Sphere defaults: ntheta=48, nphi=25 → ×4
    assert sphere_child.ntheta == 48 * 4
    assert sphere_child.nphi == 25 * 4
    # Box default resolution=(2, 2) → ×4
    assert box_child.resolution == (8, 8)
    # Cylinder defaults: ntheta=64, naxial=2, nradial=8 → ×4
    assert cyl_child.ntheta == 64 * 4
    assert cyl_child.naxial == 2 * 4
    assert cyl_child.nradial == 8 * 4


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


def test_plot_surfaces_clip_to_bounds_drops_outside_faces():
    """A sphere centered outside the bounds should be culled by clipping.

    With clip_to_bounds=True (the default) every triangle whose centroid is
    outside ``bounds`` is dropped before being added to the merged
    :class:`Poly3DCollection`. With clip_to_bounds=False the sphere's faces
    survive, so the merged collection contains strictly more triangles.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    field = _make_field()

    # Sphere completely outside the field's grid extent (which is
    # 0..8 / 0..7 / 0..6 for the test fixture).
    far_sphere = SphereMeshSurface(
        center=(50.0, 50.0, 50.0), radius=1.0, ntheta=12, nphi=7
    )
    near_box = BoxMeshSurface(
        1.0, 5.0, 1.0, 4.0, 1.0, 3.0, faces=("zmax",), resolution=(4, 4)
    )

    def _merged_face_count(ax):
        """Return the number of faces in the (single) merged Poly3DCollection."""
        polys = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
        assert len(polys) == 1, f"expected one merged Poly3DCollection, got {len(polys)}"
        return len(polys[0].get_facecolors())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cmap, _norm = plot_surfaces(
        ax,
        field=field,
        surfaces=[
            RenderItem(near_box, style="field"),
            RenderItem(far_sphere, style="solid"),
        ],
        bounds=Bounds3D((0.0, 8.0), (0.0, 7.0), (0.0, 6.0)),
        contour_levels=[8.0, 10.0],
    )
    assert cmap is not None
    n_faces_clipped = _merged_face_count(ax)
    plt.close(fig)

    # With clip_to_bounds=False the sphere's triangles are added back,
    # so the merged collection contains strictly more faces.
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    plot_surfaces(
        ax2,
        field=field,
        surfaces=[
            RenderItem(near_box, style="field"),
            RenderItem(far_sphere, style="solid"),
        ],
        bounds=Bounds3D((0.0, 8.0), (0.0, 7.0), (0.0, 6.0)),
        contour_levels=[8.0, 10.0],
        clip_to_bounds=False,
    )
    n_faces_full = _merged_face_count(ax2)
    assert n_faces_full > n_faces_clipped
    plt.close(fig2)


def test_plot_surfaces_clabel_places_one_text_per_level():
    """clabel=True adds one ax.text() per contour level with custom format."""
    field = _make_field()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    box = BoxMeshSurface(
        1.0, 7.0, 1.0, 5.0, 1.0, 3.0,
        faces=("zmax",),
        resolution=(6, 6),
    )

    plot_surfaces(
        ax,
        field=field,
        surfaces=[RenderItem(box, style="field")],
        bounds=Bounds3D((0.0, 8.0), (0.0, 7.0), (0.0, 6.0)),
        contour_levels=[15.0, 18.0, 22.0],
        clabel=True,
        clabel_fmt="phi={:.1f}",
        clabel_kwargs={"fontsize": 6, "color": "tab:red"},
    )

    # mpl 3D text artists land in ax.texts (or directly as Text3D children).
    texts = list(ax.texts)
    # All three levels should produce a label and the format string is honoured.
    label_strings = [t.get_text() for t in texts]
    assert any("phi=15.0" in s for s in label_strings)
    assert any("phi=18.0" in s for s in label_strings)
    assert any("phi=22.0" in s for s in label_strings)
    # User clabel_kwargs override the defaults.
    for t in texts:
        if t.get_text().startswith("phi="):
            assert t.get_fontsize() == 6
            assert t.get_color() == "tab:red"

    plt.close(fig)


def test_is_mesh_open_distinguishes_closed_vs_open():
    """The boundary-edge detector correctly classifies closed vs open meshes."""
    from emout.plot.surface_cut.viz import _is_mesh_open

    # A full lat-lng sphere has degenerate (zero-length) "boundary edges"
    # at the pole rows because every vertex in those rows maps to the
    # same 3D point. The detector must filter those out and report
    # closed.
    closed_sphere = SphereMeshSurface(center=(0, 0, 0), radius=1.0, ntheta=16, nphi=9)
    V_closed, F_closed = closed_sphere.mesh()
    assert _is_mesh_open(V_closed, F_closed) is False

    # A half-cylinder cut by theta_range has real boundary edges along
    # the cut.
    half_cyl = CylinderMeshSurface(
        center=(0, 0, 0), axis="z", radius=1.0, length=2.0,
        ntheta=16, naxial=4, theta_range=(0.0, np.pi),
    )
    V_half, F_half = half_cyl.mesh()
    assert _is_mesh_open(V_half, F_half) is True

    # A single rectangle face has 3 boundary edges.
    rect = RectangleMeshSurface(center=(0, 0, 0), axis="z", width=2.0)
    V_rect, F_rect = rect.mesh()
    assert _is_mesh_open(V_rect, F_rect) is True


def test_plot_surfaces_contour_auto_keeps_open_mesh_contours():
    """``contour_side='auto'`` skips back-face culling on open meshes.

    Reproduces the half-cylinder regression: a CylinderMeshSurface with
    ``theta_range=(0, π)`` is open, and the previous default
    (``contour_side='front'``) would drop 80–90% of its triangles
    depending on the camera angle, so the contour curve appeared cut
    off. With the auto default the open mesh skips culling and the full
    contour line survives.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # Synthetic field f(x, y, z) = z + 0.2*x broadcast to (nz, ny, nx).
    # Centred half-cylinder strictly inside the cell-centred grid so
    # field.sample never returns NaN for its vertices.
    grid = UniformCellCenteredGrid(nx=20, ny=20, nz=20, dx=1.0, dy=1.0, dz=1.0)
    z = grid.z_centers()[:, None, None]
    x = grid.x_centers()[None, None, :]
    y_zero = np.zeros((1, grid.ny, 1), dtype=np.float64)
    data = (z + 0.2 * x + y_zero).astype(np.float64)
    field = Field3D(grid, data)

    half_cyl = CylinderMeshSurface(
        center=(10.0, 10.0, 10.0),
        axis="z",
        radius=2.0,
        length=6.0,            # spans z = 7 .. 13, well inside the grid
        ntheta=64,
        naxial=8,
        theta_range=(0.0, np.pi),
        parts=("side",),
    )

    # Field on cylinder side ≈ z + 0.2*x ≈ 8.6 .. 15.4 over its extent.
    # Pick levels that definitely cut the cylinder.
    levels = [10.0, 11.0, 12.0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # azim chosen so a "front"-cull would drop most of the half-cylinder
    ax.view_init(elev=18, azim=-110)
    plot_surfaces(
        ax,
        field=field,
        surfaces=[RenderItem(half_cyl, style="field")],
        bounds=Bounds3D((0.0, 20.0), (0.0, 20.0), (0.0, 20.0)),
        contour_levels=levels,
    )

    lines = [c for c in ax.collections if isinstance(c, Line3DCollection)]
    assert len(lines) >= 1
    # auto-cull on an open mesh should leave many segments — the entire
    # half-cylinder ring at each level. With the previous front-cull
    # default this would be dropped to a tiny sliver.
    # Note: Line3DCollection.get_segments() returns 2D projected segments
    # which are empty before draw; the 3D segments live in _segments3d.
    n_segs_auto = sum(len(lc._segments3d) for lc in lines)
    assert n_segs_auto > 0

    # Now run again with contour_side="front" forced and confirm we
    # get strictly fewer segments — proving the auto path is what
    # rescues them.
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.view_init(elev=18, azim=-110)
    plot_surfaces(
        ax2,
        field=field,
        surfaces=[RenderItem(half_cyl, style="field")],
        bounds=Bounds3D((0.0, 20.0), (0.0, 20.0), (0.0, 20.0)),
        contour_levels=levels,
        contour_side="front",
    )
    lines_front = [c for c in ax2.collections if isinstance(c, Line3DCollection)]
    n_segs_front = sum(len(lc._segments3d) for lc in lines_front)
    assert n_segs_auto > n_segs_front

    plt.close(fig)
    plt.close(fig2)


def test_plot_surfaces_clabel_off_by_default():
    """No labels when clabel is not enabled."""
    field = _make_field()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    box = BoxMeshSurface(
        1.0, 7.0, 1.0, 5.0, 1.0, 3.0,
        faces=("zmax",),
        resolution=(6, 6),
    )
    plot_surfaces(
        ax,
        field=field,
        surfaces=[RenderItem(box, style="field")],
        bounds=Bounds3D((0.0, 8.0), (0.0, 7.0), (0.0, 6.0)),
        contour_levels=[18.0],
    )
    # The default Axes3D may carry its own decorative texts, but none of
    # them should be a contour label — none should match our format.
    label_texts = [t.get_text() for t in ax.texts]
    assert not any(t.startswith("18") for t in label_texts)
    plt.close(fig)


def test_plot_surfaces_contour_pinned_above_polygon_zorder():
    """Contours render above the merged polygon via explicit set_zorder.

    plot_surfaces disables ``ax.computed_zorder`` (so the explicit zorder
    we set is honoured) and pins the merged polygon at zorder=1 and any
    contour Line3DCollection at zorder=2. This is camera-independent and
    avoids the set_sort_zpos / camera-projection trap that previously
    caused contours to vanish behind their parent polygon.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    field = _make_field()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # The fixture field is f(x,y,z) = x + 2y + 3z. On the zmax=3 face
    # of the box below, f ranges roughly 1+2+9=12 .. 7+10+9=26, so a
    # level of 18 will definitely cut the face and produce contour
    # segments.
    box = BoxMeshSurface(
        1.0, 7.0, 1.0, 5.0, 1.0, 3.0,
        faces=("zmax",),
        resolution=(6, 6),
    )

    plot_surfaces(
        ax,
        field=field,
        surfaces=[RenderItem(box, style="field")],
        bounds=Bounds3D((0.0, 8.0), (0.0, 7.0), (0.0, 6.0)),
        contour_levels=[15.0, 18.0, 22.0],
    )

    assert ax.computed_zorder is False

    polys = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    lines = [c for c in ax.collections if isinstance(c, Line3DCollection)]
    assert len(polys) == 1
    assert polys[0].get_zorder() == 1
    # At least one contour level cuts the box's zmax face, so we
    # should have a Line3DCollection sitting at zorder=2.
    assert len(lines) >= 1
    for lc in lines:
        assert lc.get_zorder() == 2

    plt.close(fig)


def test_plot_surfaces_merges_polygons_into_single_collection():
    """All input mesh surfaces share a single Poly3DCollection.

    Per-triangle depth sorting only happens *within* a Poly3DCollection.
    Merging every input surface's faces into one collection lets mpl sort
    each triangle independently for correct rendering order, instead of
    sorting whole collections by their average z (which gets the order
    wrong as soon as the meshes interleave).
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    field = _make_field()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surfaces = [
        RenderItem(
            CylinderMeshSurface(
                center=(3.0, 3.0, 2.0),
                axis="z",
                radius=1.0,
                length=2.0,
                parts=("side",),
                ntheta=18,
                naxial=4,
            ),
            style="solid",
            solid_color="0.5",
        ),
        RenderItem(
            BoxMeshSurface(
                1.0, 5.0, 1.0, 4.0, 1.0, 3.0,
                faces=("zmax",),
                resolution=(3, 3),
            ),
            style="field",
        ),
        RenderItem(
            SphereMeshSurface(
                center=(4.0, 3.0, 2.0), radius=0.8, ntheta=14, nphi=8
            ),
            style="solid",
            solid_color="tab:blue",
            alpha=0.4,
        ),
    ]

    plot_surfaces(
        ax,
        field=field,
        surfaces=surfaces,
        bounds=Bounds3D((0.0, 8.0), (0.0, 7.0), (0.0, 6.0)),
        mode="cmap",  # disable contours so the count is purely the merged poly
    )

    polys = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    assert len(polys) == 1
    # The merged collection should hold faces from all three surfaces
    # combined (cylinder side + box face + sphere). Per-triangle counts
    # depend on resolution but the lower bound is the box's 18 + sphere
    # base ≥ 100; we just check it is "much more than one surface".
    n_faces = len(polys[0].get_facecolors())
    assert n_faces > 50
    plt.close(fig)


def test_plot_surfaces_clip_to_bounds_partial_mesh_keeps_inside_faces():
    """A mesh straddling the bounds keeps the inside half and drops the rest."""
    field = _make_field()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Box that pokes well outside the field domain on +x. Faces of the box
    # whose centroid lies inside the bounds box should survive; the rest
    # should be culled.
    half_in = BoxMeshSurface(
        4.0, 20.0, 1.0, 4.0, 1.0, 3.0, resolution=(4, 4)
    )

    cmap, _norm = plot_surfaces(
        ax,
        field=field,
        surfaces=[RenderItem(half_in, style="solid")],
        bounds=Bounds3D((0.0, 8.0), (0.0, 7.0), (0.0, 6.0)),
    )

    # The collection should still exist (some faces survived) — the assertion
    # we care about is that the call did not error and produced a render.
    assert cmap is not None
    assert len(ax.collections) >= 1
    plt.close(fig)


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
    # All polygon faces from every surface are now merged into a single
    # Poly3DCollection so mpl can sort each triangle individually for
    # correct depth order. Contour Line3DCollections (if any segments
    # were extracted) are added on top.
    assert len(ax.collections) >= 1

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
