"""Grid-aware flat-surface meshes retain interior field variation."""

import numpy as np
import pytest

from emout import Emout


@pytest.mark.parametrize("use_si", [False, True])
def test_flat_surface_grid_resolution_samples_interior(tmp_path, use_si):
    (tmp_path / "plasma.inp").write_text(
        "!!key dx=[0.5],to_c=[10000.0]\n"
        "&tmgrid\n nx=4, ny=3, nz=2, dt=0.1\n/\n"
        "&ptcond\n boundary_type='flat-surface', zssurf=1\n/\n"
    )
    data = Emout(tmp_path)
    coarse, _ = data.boundaries.mesh(use_si=use_si).mesh()
    vertices, faces = data.boundaries.mesh(use_si=use_si, per={0: {"resolution": "grid"}}).mesh()
    if use_si:
        coarse = data.unit.length.trans(coarse)
        vertices = data.unit.length.trans(vertices)
    assert coarse.shape == (4, 3)
    assert vertices.shape == (5 * 4, 3)
    assert len(faces) == 2 * 4 * 3
    np.testing.assert_allclose(np.unique(vertices[:, 0]), np.arange(5))
    np.testing.assert_allclose(np.unique(vertices[:, 1]), np.arange(4))
    np.testing.assert_allclose(vertices[:, 2], 1)

    def phi(points):
        x, y = points[:, 0], points[:, 1]
        return x * (4 - x) + y * (3 - y)

    np.testing.assert_allclose(phi(coarse), 0)
    assert phi(vertices).max() == 6


def test_flat_surface_explicit_resolution_still_wins(tmp_path):
    (tmp_path / "plasma.inp").write_text(
        "&tmgrid\n nx=4, ny=3, nz=2\n/\n&ptcond\n boundary_type='flat-surface', zssurf=1\n/\n"
    )
    mesh = Emout(tmp_path).boundaries[0].mesh(use_si=False, resolution=(7, 8))
    assert mesh.mesh()[0].shape == (56, 3)
