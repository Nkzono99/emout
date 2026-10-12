"""Inspect real marching-cubes surfaces through their collection constructor."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from emout.core.data import Data3d
from emout.utils import UnitTranslator
from emout.utils.util import apply_offset


@pytest.fixture
def surface_vertices(monkeypatch):
    from mpl_toolkits.mplot3d import art3d

    original_init = art3d.Poly3DCollection.__init__
    vertices = []

    def capture(self, polygons, *args, **kwargs):
        vertices.append(np.asarray(polygons).reshape(-1, 3).copy())
        original_init(self, polygons, *args, **kwargs)

    monkeypatch.setattr(art3d.Poly3DCollection, "__init__", capture)
    return vertices


@pytest.mark.parametrize(
    "item",
    [
        (slice(2, 8), slice(3, 9), slice(4, 16)),
        (slice(2, 8, 2), slice(3, 9, 2), slice(4, 16, 2)),
        (slice(1, 10, 3), slice(2, 10, 2), slice(3, 18, 4)),
        (slice(8, 1, -2), slice(9, 2, -3), slice(16, 3, -2)),
    ],
)
@pytest.mark.parametrize("use_si", [False, True])
@pytest.mark.parametrize("offsets", [None, (1.5, -2.0, 3.0), ("left", "center", "right")])
def test_grid_contours_have_analytic_coordinates(surface_vertices, item, use_si, offsets):
    field = Data3d(
        np.broadcast_to(np.arange(20.0), (10, 10, 20)).copy(),
        axisunits=[None, UnitTranslator(5, 1), UnitTranslator(3, 1), UnitTranslator(2, 1)],
        valunit=UnitTranslator(4, 1),
    )
    roi = field[item]
    value_scale = 4 if use_si else 1
    fig, ax = roi.plot(coordinates="grid", use_si=use_si, offsets=offsets, levels=[8.0 * value_scale], show=False)
    assert len(ax.collections) == 1
    xyz = surface_vertices[0]
    for dim, axis_name in enumerate(("x", "y", "z")):
        line = getattr(roi, axis_name + ("_si" if use_si else "")).astype(float)
        shifted = line.copy() if offsets is None else apply_offset(line.copy(), offsets[dim])
        if dim == 0:
            expected = 8.0 * (2 if use_si else 1) + shifted[0] - line[0]
            np.testing.assert_allclose(xyz[:, dim], expected)
        else:
            np.testing.assert_allclose([xyz[:, dim].min(), xyz[:, dim].max()], [shifted.min(), shifted.max()])
    plt.close(fig)


def test_grid_contours_use_raw_coordinates_without_units(surface_vertices):
    field = Data3d(np.broadcast_to(np.arange(20.0), (10, 10, 20)).copy())
    roi = field[2:8:2, 3:9:2, 4:16:2]
    fig, _ = roi.plot(coordinates="grid", levels=[8.0], show=False)
    xyz = surface_vertices[0]
    np.testing.assert_allclose(xyz[:, 0], 8)
    np.testing.assert_allclose(xyz.min(axis=0), [8, 3, 2])
    np.testing.assert_allclose(xyz.max(axis=0), [8, 7, 6])
    plt.close(fig)


def test_local_contours_keep_legacy_origin_and_spacing(surface_vertices):
    field = Data3d(np.broadcast_to(np.arange(20.0), (10, 10, 20)).copy())
    roi = field[2:8:2, 3:9:2, 4:16:2]
    fig, _ = roi.plot(use_si=False, levels=[8.0], show=False)
    xyz = surface_vertices[0]
    np.testing.assert_allclose(xyz.min(axis=0), [2, 0, 0])
    np.testing.assert_allclose(xyz.max(axis=0), [2, 2, 2])
    plt.close(fig)


def test_grid_contours_forward_physical_bounds(surface_vertices):
    field = Data3d(np.broadcast_to(np.arange(20.0), (10, 10, 20)).copy())
    roi = field[2:10:2, 1:10:2, 4:16:2]
    fig, _ = roi.plot(
        coordinates="grid",
        use_si=False,
        levels=[8.0],
        bounds_xyz=((6, 12), (3, 7), (4, 8)),
        show=False,
    )
    xyz = surface_vertices[0]
    np.testing.assert_allclose(xyz.min(axis=0), [8, 3, 4])
    np.testing.assert_allclose(xyz.max(axis=0), [8, 7, 8])
    plt.close(fig)


def test_contours_reject_unknown_coordinate_mode():
    with pytest.raises(ValueError, match="coordinates"):
        Data3d(np.zeros((3, 3, 3))).plot(coordinates="unknown", use_si=False, levels=[0])
