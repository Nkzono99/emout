"""Periodic electric relocation must preserve the mean and translation symmetry."""

import h5py
import numpy as np
import pytest

from emout import Emout
from emout.core.relocation.electric import relocated_electric_field


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_duplicate_endpoint_preserves_mean_and_periodic_shifts(axis):
    unique = np.array([-1.0, 1.0, 1.0, -1.0])

    def relocate(samples):
        line = np.r_[samples, samples[0]]
        shape = [1, 1, 1]
        shape[axis] = len(line)
        field = np.broadcast_to(line.reshape(shape), [len(line) if i == axis else 3 for i in range(3)]).copy()
        result = relocated_electric_field(field, axis, "periodic", periodic_endpoint="duplicate")
        return np.moveaxis(result, axis, -1)[0, 0]

    actual = relocate(unique)
    np.testing.assert_array_equal(actual, [-1, 0, 1, 0, -1])
    assert actual[:-1].mean() == unique.mean()
    np.testing.assert_array_equal(relocate(np.roll(unique, 1))[:-1], np.roll(actual[:-1], 1))


def test_public_relocation_separates_legacy_cache_and_reopens(tmp_path):
    (tmp_path / "plasma.inp").write_text(
        "&tmgrid\n nx=4, ny=2, nz=2, dt=0.002, ifdiag=1\n/\n&esorem\n mtd_vbnd=0,0,0\n/\n"
    )
    with h5py.File(tmp_path / "ex00_0000.h5", "w") as output:
        output.create_dataset("ex/0000", data=np.broadcast_to([-1.0, 1.0, 1.0, -1.0, -1.0], (3, 3, 5)))
    legacy = Emout(tmp_path)
    np.testing.assert_array_equal(legacy.rex[0, 0, 0], [0, 0, 1, 0, 0])
    old_cache = (tmp_path / "rex00_0000.h5").read_bytes()

    corrected = Emout(tmp_path, electric_relocation="periodic")
    np.testing.assert_array_equal(corrected.rex[0, 0, 0], [-1, 0, 1, 0, -1])
    assert (tmp_path / "rex00_0000.h5").read_bytes() == old_cache
    assert (tmp_path / ".emout/relocated/periodic/rex00_0000.h5").exists()
    assert corrected.rex._emout_open_kwargs["electric_relocation"] == "periodic"
    reopened = Emout(**corrected._remote_open_kwargs)
    np.testing.assert_array_equal(reopened.rex[0, 0, 0], [-1, 0, 1, 0, -1])


def test_invalid_electric_relocation_mode(tmp_path):
    with pytest.raises(ValueError, match="electric_relocation"):
        Emout(tmp_path, electric_relocation="wrong")
