"""Tests for :mod:`emout.emout.boundaries`.

These tests build an :class:`emout.utils.InpFile` from a temporary namelist
file (matching the MPIEMSES3D finbound format) and instantiate
:class:`emout.emout.boundaries.BoundaryCollection` directly, without going
through the full ``Emout`` facade. That keeps the tests independent of the
output-file discovery logic and lets us cover sparse/indexed parameter
access, per-boundary overrides, unit conversion, and composite mesh
generation in isolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emout.emout.boundaries import (
    BoundaryCollection,
    CircleBoundary,
    CuboidBoundary,
    CylinderBoundary,
    CylinderHoleBoundary,
    DiskBoundary,
    FlatSurfaceBoundary,
    PlaneWithCircleBoundary,
    RectangleBoundary,
    RectangleHoleBoundary,
    SphereBoundary,
)
from emout.plot.surface_cut import (
    BoxMeshSurface,
    CircleMeshSurface,
    CompositeMeshSurface,
    CylinderMeshSurface,
    DiskMeshSurface,
    MeshSurface3D,
    PlaneWithCircleMeshSurface,
    RectangleMeshSurface,
    SphereMeshSurface,
)
from emout.utils import InpFile, Units


_FINBOUND_NAMELIST = """\
&ptcond
    boundary_type = 'complex'
    boundary_types(1) = 'sphere'
    boundary_types(2) = 'cuboid'
    boundary_types(3) = 'cylinderz'
    boundary_types(4) = 'diskz'
    boundary_types(5) = 'open-cylinderx'
    boundary_types(6) = 'rectangle'
    boundary_types(7) = 'circlez'
    sphere_origin(:, 1) = 10.0, 20.0, 30.0
    sphere_radius(1) = 4.0
    cuboid_shape(:, 2) = 1.0, 5.0, 2.0, 6.0, 3.0, 7.0
    cylinder_origin(:, 3) = 8.0, 8.0, 0.0
    cylinder_radius(3) = 2.0
    cylinder_height(3) = 5.0
    disk_origin(:, 4) = 12.0, 12.0, 1.0
    disk_radius(4) = 3.0
    disk_inner_radius(4) = 1.0
    disk_height(4) = 0.5
    cylinder_origin(:, 5) = 0.0, 16.0, 16.0
    cylinder_radius(5) = 1.5
    cylinder_height(5) = 20.0
    rectangle_shape(:, 6) = 1.0, 5.0, 2.0, 6.0, 4.0, 4.0
    circle_origin(:, 7) = 15.0, 15.0, 8.0
    circle_radius(7) = 2.5
/
"""


@pytest.fixture
def inp(tmp_path: Path) -> InpFile:
    path = tmp_path / "plasma.inp"
    path.write_text("!!key dx=[0.1],to_c=[10000.0]\n" + _FINBOUND_NAMELIST)
    return InpFile(path)


@pytest.fixture
def unit() -> Units:
    # dx = 0.1 m, to_c arbitrary. length.ratio = 1/dx = 10
    return Units(dx=0.1, to_c=10000.0)


@pytest.fixture
def boundaries(inp: InpFile, unit: Units) -> BoundaryCollection:
    return BoundaryCollection(inp, unit)


# ---------------------------------------------------------------------------
# Collection basics
# ---------------------------------------------------------------------------


def test_collection_builds_expected_types(boundaries: BoundaryCollection):
    assert len(boundaries) == 7
    classes = [type(b) for b in boundaries]
    assert classes == [
        SphereBoundary,
        CuboidBoundary,
        CylinderBoundary,
        DiskBoundary,
        CylinderBoundary,
        RectangleBoundary,
        CircleBoundary,
    ]
    assert boundaries[0].btype == "sphere"
    assert boundaries[2].btype == "cylinderz"
    assert boundaries[4].btype == "open-cylinderx"


def test_collection_repr_and_iteration(boundaries: BoundaryCollection):
    text = repr(boundaries)
    assert "sphere" in text and "cylinderz" in text
    for i, b in enumerate(boundaries):
        assert b.index == i
        assert b.fortran_index == i + 1


def test_collection_empty_when_unsupported_top_level_type(tmp_path: Path, unit: Units):
    path = tmp_path / "plasma.inp"
    path.write_text(
        "&ptcond\n    boundary_type = 'none'\n/\n"
    )
    inp = InpFile(path)
    coll = BoundaryCollection(inp, unit)
    assert len(coll) == 0
    assert not coll


def test_collection_skips_unsupported_complex_types(tmp_path: Path, unit: Units):
    path = tmp_path / "plasma.inp"
    path.write_text(
        "&ptcond\n"
        "    boundary_type = 'complex'\n"
        "    boundary_types(1) = 'hyperboloid-hole'\n"
        "    boundary_types(2) = 'sphere'\n"
        "    sphere_origin(:, 2) = 1.0, 2.0, 3.0\n"
        "    sphere_radius(2) = 0.5\n"
        "/\n"
    )
    inp = InpFile(path)
    coll = BoundaryCollection(inp, unit)
    # Only the sphere is supported — hyperboloid-hole is still unhandled.
    assert len(coll) == 1
    assert isinstance(coll[0], SphereBoundary)
    assert coll[0].index == 1  # original 0-based position preserved
    assert any(
        "hyperboloid-hole" in entry for entry in (f"{s[1]}" for s in coll.skipped)
    )


# ---------------------------------------------------------------------------
# Per-boundary mesh construction
# ---------------------------------------------------------------------------


def test_sphere_boundary_reads_origin_and_radius(boundaries: BoundaryCollection):
    sphere = boundaries[0]
    mesh_obj = sphere.mesh()
    assert isinstance(mesh_obj, SphereMeshSurface)
    assert np.allclose(mesh_obj.center, [10.0, 20.0, 30.0])
    assert np.isclose(mesh_obj.radius, 4.0)


def test_sphere_boundary_use_si_converts_lengths(boundaries: BoundaryCollection):
    sphere_si = boundaries[0].mesh(use_si=True)
    # dx = 0.1 m ⇒ length.ratio = 10 ⇒ grid→SI is divide-by-10.
    assert np.allclose(sphere_si.center, [1.0, 2.0, 3.0])
    assert np.isclose(sphere_si.radius, 0.4)


def test_sphere_boundary_respects_overrides(boundaries: BoundaryCollection):
    m = boundaries[0].mesh(ntheta=96, nphi=48)
    assert m.ntheta == 96
    assert m.nphi == 48


def test_cuboid_boundary_builds_full_box(boundaries: BoundaryCollection):
    box = boundaries[1].mesh()
    assert isinstance(box, BoxMeshSurface)
    assert box.xmin == 1.0 and box.xmax == 5.0
    assert box.ymin == 2.0 and box.ymax == 6.0
    assert box.zmin == 3.0 and box.zmax == 7.0
    # Default: all 6 faces.
    assert set(box.faces) == {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}


def test_cylinder_boundary_z_closed(boundaries: BoundaryCollection):
    cyl = boundaries[2].mesh()
    assert isinstance(cyl, CylinderMeshSurface)
    # Axis letter is the last character of the btype.
    assert np.allclose(cyl.axis, [0.0, 0.0, 1.0])
    assert np.allclose(cyl.center, [8.0, 8.0, 0.0])
    assert cyl.radius == 2.0
    assert cyl.tmin == 0.0 and cyl.tmax == 5.0
    # Closed: all three parts.
    assert set(cyl.parts) == {"side", "top", "bottom"}


def test_disk_boundary_reads_inner_outer(boundaries: BoundaryCollection):
    disk = boundaries[3].mesh()
    assert isinstance(disk, DiskMeshSurface)
    assert disk.outer_radius == 3.0
    assert disk.inner_radius == 1.0
    assert disk.tmin == 0.0 and disk.tmax == 0.5
    assert np.allclose(disk.center, [12.0, 12.0, 1.0])


def test_open_cylinder_boundary_x_only_side(boundaries: BoundaryCollection):
    open_cyl = boundaries[4].mesh()
    assert isinstance(open_cyl, CylinderMeshSurface)
    assert np.allclose(open_cyl.axis, [1.0, 0.0, 0.0])
    assert tuple(open_cyl.parts) == ("side",)


def test_rectangle_boundary_flat_face_detected(boundaries: BoundaryCollection):
    rect = boundaries[5].mesh()
    assert isinstance(rect, BoxMeshSurface)
    # zmin == zmax ⇒ flat z face.
    assert rect.zmin == rect.zmax == 4.0
    assert tuple(rect.faces) == ("zmin",)


def test_circle_boundary_builds_disc(boundaries: BoundaryCollection):
    circle = boundaries[6].mesh()
    assert isinstance(circle, CircleMeshSurface)
    assert np.allclose(circle.center, [15.0, 15.0, 8.0])
    assert circle.radius == 2.5
    # circlez ⇒ axis is z.
    assert np.allclose(circle.axis, [0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Composite mesh API
# ---------------------------------------------------------------------------


def test_collection_mesh_returns_composite(boundaries: BoundaryCollection):
    composite = boundaries.mesh()
    assert isinstance(composite, CompositeMeshSurface)
    assert len(composite.children) == len(boundaries)
    V, F = composite.mesh()
    assert V.shape[1] == 3
    assert F.shape[1] == 3
    assert V.shape[0] > 0 and F.shape[0] > 0


def test_collection_mesh_use_si_applies_to_all(boundaries: BoundaryCollection):
    composite = boundaries.mesh(use_si=True)
    # The child SphereMeshSurface should reflect SI coordinates.
    sphere_child = composite.children[0]
    assert np.isclose(sphere_child.radius, 0.4)
    assert np.allclose(sphere_child.center, [1.0, 2.0, 3.0])


def test_collection_mesh_per_boundary_overrides(boundaries: BoundaryCollection):
    composite = boundaries.mesh(per={0: {"ntheta": 64}, 2: {"naxial": 6}})
    sphere_child = composite.children[0]
    cyl_child = composite.children[2]
    assert sphere_child.ntheta == 64
    assert cyl_child.naxial == 6


# ---------------------------------------------------------------------------
# MeshSurface3D + operator and render()
# ---------------------------------------------------------------------------


def test_mesh_add_operator_creates_composite():
    a = SphereMeshSurface(center=(0.0, 0.0, 0.0), radius=1.0, ntheta=12, nphi=7)
    b = SphereMeshSurface(center=(5.0, 0.0, 0.0), radius=1.0, ntheta=12, nphi=7)
    composite = a + b
    assert isinstance(composite, CompositeMeshSurface)
    assert composite.children == (a, b)
    V, _ = composite.mesh()
    # Vertices span both spheres.
    assert V[:, 0].max() > 4.0
    assert V[:, 0].min() < 1.0


def test_mesh_add_flattens_nested_composites():
    a = SphereMeshSurface(center=(0, 0, 0), radius=1.0, ntheta=12, nphi=7)
    b = SphereMeshSurface(center=(5, 0, 0), radius=1.0, ntheta=12, nphi=7)
    c = SphereMeshSurface(center=(0, 5, 0), radius=1.0, ntheta=12, nphi=7)
    combined = (a + b) + c
    assert isinstance(combined, CompositeMeshSurface)
    assert combined.children == (a, b, c)


def test_mesh_render_returns_render_item():
    from emout.plot.surface_cut import RenderItem

    m = SphereMeshSurface(center=(0, 0, 0), radius=1.0, ntheta=12, nphi=7)
    item = m.render(style="solid", solid_color="0.5", alpha=0.4)
    assert isinstance(item, RenderItem)
    assert item.surface is m
    assert item.style == "solid"
    assert item.solid_color == "0.5"
    assert item.alpha == 0.4


# ---------------------------------------------------------------------------
# Boundary __add__ composition
# ---------------------------------------------------------------------------


def test_boundary_add_returns_collection(boundaries: BoundaryCollection):
    combined = boundaries[0] + boundaries[1]
    assert isinstance(combined, BoundaryCollection)
    assert len(combined) == 2
    assert combined[0] is boundaries[0]
    assert combined[1] is boundaries[1]
    # Unit propagates from the first boundary that has one.
    assert combined.unit is boundaries[0].unit


def test_boundary_add_three_flattens(boundaries: BoundaryCollection):
    combined = boundaries[0] + boundaries[1] + boundaries[2]
    assert isinstance(combined, BoundaryCollection)
    assert len(combined) == 3


def test_boundary_add_collection(boundaries: BoundaryCollection):
    pair = boundaries[0] + boundaries[1]
    triple = boundaries[2] + pair
    assert isinstance(triple, BoundaryCollection)
    assert len(triple) == 3
    # Order: left side first, then the right collection's children.
    assert triple[0] is boundaries[2]
    assert triple[1] is boundaries[0]
    assert triple[2] is boundaries[1]


def test_boundary_add_collection_then_mesh_use_si(boundaries: BoundaryCollection):
    composite_mesh = (boundaries[0] + boundaries[1]).mesh(use_si=True)
    # The Sphere child should have its center scaled to SI metres.
    sphere_child = composite_mesh.children[0]
    assert isinstance(sphere_child, SphereMeshSurface)
    # dx = 0.1 ⇒ grid→SI is divide-by-10. boundaries[0] is sphere at (10, 20, 30).
    assert np.allclose(sphere_child.center, [1.0, 2.0, 3.0])


def test_boundary_sum_via_python_sum(boundaries: BoundaryCollection):
    # sum() uses __radd__ with start=0.
    combined = sum([boundaries[0], boundaries[3], boundaries[6]])
    assert isinstance(combined, BoundaryCollection)
    assert len(combined) == 3
    assert combined[0] is boundaries[0]
    assert combined[1] is boundaries[3]
    assert combined[2] is boundaries[6]


def test_boundary_add_meshsurface_yields_composite(boundaries: BoundaryCollection):
    extra = SphereMeshSurface(center=(0.0, 0.0, 0.0), radius=1.0, ntheta=12, nphi=7)
    composite = boundaries[0] + extra
    assert isinstance(composite, CompositeMeshSurface)
    # First child is whatever boundaries[0].mesh() built (a SphereMeshSurface).
    assert isinstance(composite.children[0], SphereMeshSurface)
    assert composite.children[1] is extra


def test_collection_from_boundaries_preserves_order_and_unit(boundaries: BoundaryCollection, unit: Units):
    explicit = type(boundaries).from_boundaries(
        [boundaries[2], boundaries[0]], unit=unit
    )
    assert isinstance(explicit, BoundaryCollection)
    assert len(explicit) == 2
    assert explicit[0] is boundaries[2]
    assert explicit[1] is boundaries[0]
    assert explicit.unit is unit
    # The composite mesh works without an `inp` reference.
    composite_mesh = explicit.mesh(use_si=True)
    assert isinstance(composite_mesh, CompositeMeshSurface)


def test_boundary_render_returns_render_item(boundaries: BoundaryCollection):
    from emout.plot.surface_cut import RenderItem

    item = boundaries[0].render(use_si=True, style="solid", solid_color="0.4", alpha=0.6)
    assert isinstance(item, RenderItem)
    assert isinstance(item.surface, SphereMeshSurface)
    assert item.style == "solid"
    assert item.solid_color == "0.4"
    assert item.alpha == 0.6
    # use_si propagated through.
    assert np.allclose(item.surface.center, [1.0, 2.0, 3.0])


def test_collection_render_returns_render_item(boundaries: BoundaryCollection):
    from emout.plot.surface_cut import RenderItem

    item = (boundaries[0] + boundaries[1]).render(use_si=True, style="solid", alpha=0.5)
    assert isinstance(item, RenderItem)
    assert isinstance(item.surface, CompositeMeshSurface)
    assert item.style == "solid"
    assert item.alpha == 0.5


# ---------------------------------------------------------------------------
# plane-with-circle* and legacy *-hole / flat-surface boundaries
# ---------------------------------------------------------------------------


def _make_inp(tmp_path: Path, ptcond_body: str, *, include_key: bool = True) -> InpFile:
    path = tmp_path / "plasma.inp"
    header = "!!key dx=[0.1],to_c=[10000.0]\n" if include_key else ""
    path.write_text(header + ptcond_body)
    return InpFile(path)


def test_plane_with_circle_boundary_reads_origin_radius_and_uses_domain(tmp_path: Path, unit: Units):
    inp = _make_inp(
        tmp_path,
        """\
&esorem
    nx = 64
    ny = 48
    nz = 32
/
&ptcond
    boundary_type = 'complex'
    boundary_types(1) = 'plane-with-circlez'
    plane_with_circle_origin(:, 1) = 32.0, 24.0, 16.0
    plane_with_circle_radius(1) = 4.0
/
""",
    )
    coll = BoundaryCollection(inp, unit)
    assert len(coll) == 1
    assert isinstance(coll[0], PlaneWithCircleBoundary)

    mesh_obj = coll[0].mesh()
    assert isinstance(mesh_obj, PlaneWithCircleMeshSurface)
    # Center copied verbatim, plane spans the full (nx, ny) rectangle.
    assert np.allclose(mesh_obj.center, [32.0, 24.0, 16.0])
    assert np.isclose(mesh_obj.width_u, 64.0)
    assert np.isclose(mesh_obj.width_v, 48.0)
    assert np.isclose(mesh_obj.inner_radius, 4.0)
    # plane-with-circlez ⇒ normal is z.
    assert np.allclose(mesh_obj.axis, [0.0, 0.0, 1.0])


def test_plane_with_circle_boundary_use_si_converts_all_lengths(tmp_path: Path, unit: Units):
    inp = _make_inp(
        tmp_path,
        """\
&esorem
    nx = 64
    ny = 48
    nz = 32
/
&ptcond
    boundary_type = 'complex'
    boundary_types(1) = 'plane-with-circlex'
    plane_with_circle_origin(:, 1) = 10.0, 20.0, 30.0
    plane_with_circle_radius(1) = 2.0
/
""",
    )
    mesh_obj = BoundaryCollection(inp, unit)[0].mesh(use_si=True)
    # dx = 0.1 ⇒ divide-by-10. plane-with-circlex → width is (ny, nz).
    assert np.allclose(mesh_obj.center, [1.0, 2.0, 3.0])
    assert np.isclose(mesh_obj.inner_radius, 0.2)
    assert np.isclose(mesh_obj.width_u, 4.8)  # ny=48 grid → 4.8 m
    assert np.isclose(mesh_obj.width_v, 3.2)  # nz=32 grid → 3.2 m


def test_flat_surface_legacy_single_body_builds_one_rectangle(tmp_path: Path, unit: Units):
    inp = _make_inp(
        tmp_path,
        """\
&esorem
    nx = 40
    ny = 30
    nz = 20
/
&ptcond
    boundary_type = 'flat-surface'
    zssurf = 12.5
    zsbuf = 1.0
/
""",
    )
    coll = BoundaryCollection(inp, unit)
    assert len(coll) == 1
    assert isinstance(coll[0], FlatSurfaceBoundary)

    rect = coll[0].mesh()
    assert isinstance(rect, RectangleMeshSurface)
    assert np.allclose(rect.center, [20.0, 15.0, 12.5])
    assert rect.width_u == 40.0 and rect.width_v == 30.0
    # Normal is +z for a flat-surface.
    assert np.allclose(rect.axis, [0.0, 0.0, 1.0])


def test_flat_surface_inside_complex_mode_also_works(tmp_path: Path, unit: Units):
    inp = _make_inp(
        tmp_path,
        """\
&esorem
    nx = 10
    ny = 10
    nz = 10
/
&ptcond
    boundary_type = 'complex'
    boundary_types(1) = 'flat-surface'
    boundary_types(2) = 'sphere'
    zssurf = 6.0
    sphere_origin(:, 2) = 5.0, 5.0, 8.0
    sphere_radius(2) = 1.0
/
""",
    )
    coll = BoundaryCollection(inp, unit)
    assert len(coll) == 2
    assert isinstance(coll[0], FlatSurfaceBoundary)
    assert isinstance(coll[1], SphereBoundary)

    flat_mesh = coll[0].mesh()
    assert isinstance(flat_mesh, RectangleMeshSurface)
    assert flat_mesh.center[2] == 6.0


def test_rectangle_hole_boundary_builds_open_top_box(tmp_path: Path, unit: Units):
    inp = _make_inp(
        tmp_path,
        """\
&esorem
    nx = 64
    ny = 64
    nz = 64
/
&ptcond
    boundary_type = 'rectangle-hole'
    zssurf = 50.0
    xlrechole(1) = 20.0
    xurechole(1) = 40.0
    ylrechole(1) = 25.0
    yurechole(1) = 35.0
    zlrechole(2) = 10.0
/
""",
    )
    coll = BoundaryCollection(inp, unit)
    assert len(coll) == 1
    assert isinstance(coll[0], RectangleHoleBoundary)

    box = coll[0].mesh()
    assert isinstance(box, BoxMeshSurface)
    assert (box.xmin, box.xmax) == (20.0, 40.0)
    assert (box.ymin, box.ymax) == (25.0, 35.0)
    assert (box.zmin, box.zmax) == (10.0, 50.0)
    # Top face (zmax) is intentionally omitted — it is the pit opening.
    assert "zmax" not in box.faces
    assert {"xmin", "xmax", "ymin", "ymax", "zmin"} <= set(box.faces)


def test_cylinder_hole_boundary_builds_open_top_cylinder(tmp_path: Path, unit: Units):
    inp = _make_inp(
        tmp_path,
        """\
&esorem
    nx = 64
    ny = 64
    nz = 64
/
&ptcond
    boundary_type = 'cylinder-hole'
    zssurf = 40.0
    xlrechole(1) = 28.0
    xurechole(1) = 36.0
    ylrechole(1) = 28.0
    yurechole(1) = 36.0
    zlrechole(2) = 5.0
/
""",
    )
    coll = BoundaryCollection(inp, unit)
    assert len(coll) == 1
    assert isinstance(coll[0], CylinderHoleBoundary)

    cyl = coll[0].mesh()
    assert isinstance(cyl, CylinderMeshSurface)
    # center is the pit base at z=zlrechole, radius = (xu-xl)/2, height = zssurf-zl.
    assert np.allclose(cyl.center, [32.0, 32.0, 5.0])
    assert np.isclose(cyl.radius, 4.0)
    assert cyl.tmin == 0.0 and cyl.tmax == 35.0
    # Top cap (opening) is not rendered.
    assert set(cyl.parts) == {"side", "bottom"}


def test_rectangle_hole_use_si_converts_all_lengths(tmp_path: Path, unit: Units):
    inp = _make_inp(
        tmp_path,
        """\
&esorem
    nx = 64
    ny = 64
    nz = 64
/
&ptcond
    boundary_type = 'rectangle-hole'
    zssurf = 50.0
    xlrechole(1) = 20.0
    xurechole(1) = 40.0
    ylrechole(1) = 25.0
    yurechole(1) = 35.0
    zlrechole(2) = 10.0
/
""",
    )
    box = BoundaryCollection(inp, unit)[0].mesh(use_si=True)
    # dx=0.1 ⇒ divide-by-10.
    assert box.xmin == 2.0 and box.xmax == 4.0
    assert box.ymin == 2.5 and box.ymax == 3.5
    assert box.zmin == 1.0 and box.zmax == 5.0
