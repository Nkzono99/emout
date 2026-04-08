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
    DiskBoundary,
    RectangleBoundary,
    SphereBoundary,
)
from emout.plot.surface_cut import (
    BoxMeshSurface,
    CircleMeshSurface,
    CompositeMeshSurface,
    CylinderMeshSurface,
    DiskMeshSurface,
    MeshSurface3D,
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


def test_collection_empty_when_not_complex(tmp_path: Path, unit: Units):
    path = tmp_path / "plasma.inp"
    path.write_text(
        "&ptcond\n    boundary_type = 'flat-surface'\n    zssurf = 60.0\n/\n"
    )
    inp = InpFile(path)
    coll = BoundaryCollection(inp, unit)
    assert len(coll) == 0
    assert not coll


def test_collection_skips_unsupported_types(tmp_path: Path, unit: Units):
    path = tmp_path / "plasma.inp"
    path.write_text(
        "&ptcond\n"
        "    boundary_type = 'complex'\n"
        "    boundary_types(1) = 'flat-surface'\n"
        "    boundary_types(2) = 'sphere'\n"
        "    sphere_origin(:, 2) = 1.0, 2.0, 3.0\n"
        "    sphere_radius(2) = 0.5\n"
        "/\n"
    )
    inp = InpFile(path)
    coll = BoundaryCollection(inp, unit)
    # Only the sphere is supported.
    assert len(coll) == 1
    assert isinstance(coll[0], SphereBoundary)
    assert coll[0].index == 1  # original 0-based position preserved
    assert any("flat-surface" in entry for entry in (f"{s[1]}" for s in coll.skipped))


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
