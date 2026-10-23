"""Read exported VTK files back and compare values at their spatial cells."""

import sys

import numpy as np
import pytest

from emout.core.data import Data3d
from emout.utils import Units


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("use_si", [False, True])
@pytest.mark.parametrize(
    "selection",
    [
        (slice(None),) * 3,
        (slice(2, 8), slice(3, 10), slice(4, 12)),
        (slice(2, 10, 2), slice(3, 12, 3), slice(4, 15, 2)),
        (slice(9, 1, -2), slice(11, 2, -3), slice(14, 3, -2)),
        (slice(2, 3), slice(3, 4), slice(4, 5)),
    ],
    ids=["full", "crop", "stride", "reverse", "single-cell"],
)
def test_vtk_grid_values_and_coordinates(tmp_path, monkeypatch, selection, use_si, fallback):
    pv = pytest.importorskip("pyvista")
    z, y, x = np.indices((12, 14, 16))
    units = Units(0.5, 10000.0)
    field = Data3d(
        100 * z + 10 * y + x,
        name="phi",
        axisunits=[units.t, units.length, units.length, units.length],
        valunit=units.phi,
    )[selection]
    expected = np.asarray(field.val_si if use_si else field)
    lines = [field.x, field.y, field.z]
    steps = [field.slices[axis].step for axis in (3, 2, 1)]
    for index, step in enumerate(steps):
        if step < 0:
            expected = np.flip(expected, axis=2 - index)
    scale = 0.5 if use_si else 1.0
    origin = np.array([min(line) for line in lines]) * scale
    spacing = np.abs(steps) * scale

    with monkeypatch.context() as patch:
        if fallback:
            patch.setitem(sys.modules, "pyvista", None)
        path = field.to_vtk(tmp_path / "field.vti", use_si=use_si, ordering="vtk", coordinates="grid")
    mesh = pv.read(path)
    np.testing.assert_allclose(mesh.cell_data["phi"], expected.ravel(order="C"))
    np.testing.assert_allclose(mesh.origin, origin)
    np.testing.assert_allclose(mesh.spacing, spacing)
    np.testing.assert_allclose(mesh.cell_centers().points[0], origin + spacing / 2)


@pytest.mark.parametrize("fallback", [False, True])
def test_vtk_default_retains_legacy_export(tmp_path, monkeypatch, fallback):
    pv = pytest.importorskip("pyvista")
    field = Data3d(np.arange(120).reshape(4, 5, 6), name="rho")[1:4, 1:5:2, 2:6]
    with monkeypatch.context() as patch:
        if fallback:
            patch.setitem(sys.modules, "pyvista", None)
        path = field.to_vtk(tmp_path / "legacy.vti", use_si=False)
    mesh = pv.read(path)
    np.testing.assert_array_equal(mesh.cell_data["rho"], np.asarray(field).ravel(order="F"))
    assert mesh.origin == (0, 0, 0)
    assert mesh.spacing == (1, 1, 1)


def test_vtk_grid_without_unit_metadata(tmp_path):
    pv = pytest.importorskip("pyvista")
    field = Data3d(np.ones((8, 9, 10)))[2:6, 3:7:2, 4:8:2]
    mesh = pv.read(field.to_vtk(tmp_path / "raw.vti", coordinates="grid", ordering="vtk"))
    assert mesh.origin == (4, 3, 2)
    assert mesh.spacing == (2, 2, 1)


@pytest.mark.parametrize("kwargs", [{"ordering": "wrong"}, {"coordinates": "wrong"}])
def test_vtk_invalid_modes_do_not_write(tmp_path, kwargs):
    with pytest.raises(ValueError):
        Data3d(np.ones((2, 3, 4))).to_vtk(tmp_path / "bad.vti", **kwargs)
    assert not (tmp_path / "bad.vti").exists()
