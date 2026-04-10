"""Tests for emout.core.relocation (electric and magnetic field relocation)."""

import numpy as np
import pytest

from emout.core.relocation.electric import relocated_electric_field
from emout.core.relocation.magnetic import relocated_magnetic_field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_field(value, shape=(5, 6, 7)):
    """Return a 3-D array filled with *value*."""
    return np.full(shape, value, dtype=float)


def _ramp_along_axis(axis, shape=(5, 6, 7)):
    """Return a 3-D array whose values increase linearly along *axis*.

    For axis=0 the values are 0,1,...,shape[0]-1 broadcast across
    the other dimensions.
    """
    idx = [np.newaxis, np.newaxis, np.newaxis]
    idx[axis] = slice(None)
    vals = np.arange(shape[axis], dtype=float)[tuple(idx)]
    return np.broadcast_to(vals, shape).copy()


# ===================================================================
# Electric-field relocation
# ===================================================================


class TestRelocatedElectricFieldUniform:
    """A uniform field should stay uniform after relocation (interior)."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("btype", ["periodic", "dirichlet", "neumann"])
    def test_interior_unchanged(self, axis, btype):
        ef = _uniform_field(3.0)
        ref = relocated_electric_field(ef, axis, btype)

        # Interior slice: indices 1..-1 along the target axis
        slc = [slice(None)] * 3
        slc[axis] = slice(1, -1)
        np.testing.assert_allclose(ref[tuple(slc)], 3.0)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_uniform_periodic_boundary(self, axis):
        ef = _uniform_field(5.0)
        ref = relocated_electric_field(ef, axis, "periodic")
        # For a uniform field, periodic boundaries should also be 5.0
        slc0 = [slice(None)] * 3
        slc0[axis] = 0
        slcm = [slice(None)] * 3
        slcm[axis] = -1
        np.testing.assert_allclose(ref[tuple(slc0)], 5.0)
        np.testing.assert_allclose(ref[tuple(slcm)], 5.0)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_uniform_neumann_boundary(self, axis):
        ef = _uniform_field(5.0)
        ref = relocated_electric_field(ef, axis, "neumann")
        # Neumann sets boundary to 0
        slc0 = [slice(None)] * 3
        slc0[axis] = 0
        slcm = [slice(None)] * 3
        slcm[axis] = -1
        np.testing.assert_allclose(ref[tuple(slc0)], 0.0)
        np.testing.assert_allclose(ref[tuple(slcm)], 0.0)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_uniform_dirichlet_boundary(self, axis):
        ef = _uniform_field(5.0)
        ref = relocated_electric_field(ef, axis, "dirichlet")
        # Dirichlet copies adjacent interior value
        slc0 = [slice(None)] * 3
        slc0[axis] = 0
        slcm = [slice(None)] * 3
        slcm[axis] = -1
        slc1 = [slice(None)] * 3
        slc1[axis] = 1
        slcm2 = [slice(None)] * 3
        slcm2[axis] = -2
        np.testing.assert_allclose(ref[tuple(slc0)], ef[tuple(slc1)])
        np.testing.assert_allclose(ref[tuple(slcm)], ef[tuple(slcm2)])


class TestRelocatedElectricFieldRamp:
    """Ramp input: interior values should be average of neighbours."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_interior_averaging(self, axis):
        shape = (8, 9, 10)
        ef = _ramp_along_axis(axis, shape)
        ref = relocated_electric_field(ef, axis, "periodic")

        slc = [slice(None)] * 3
        slc[axis] = slice(1, -1)
        expected_slc_lo = [slice(None)] * 3
        expected_slc_lo[axis] = slice(0, -2)
        expected_slc_hi = [slice(None)] * 3
        expected_slc_hi[axis] = slice(1, -1)

        expected = 0.5 * (ef[tuple(expected_slc_lo)] + ef[tuple(expected_slc_hi)])
        np.testing.assert_allclose(ref[tuple(slc)], expected)


class TestRelocatedElectricFieldShape:
    """Output shape must match input shape."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("btype", ["periodic", "dirichlet", "neumann"])
    def test_shape_preserved(self, axis, btype):
        shape = (4, 5, 6)
        ef = np.random.default_rng(42).random(shape)
        ref = relocated_electric_field(ef, axis, btype)
        assert ref.shape == shape


class TestRelocatedElectricFieldPeriodic:
    """Periodic boundary should wrap correctly."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_periodic_boundary_wrap(self, axis):
        shape = (6, 7, 8)
        rng = np.random.default_rng(123)
        ef = rng.random(shape)
        ref = relocated_electric_field(ef, axis, "periodic")

        # boundary[0] = 0.5 * (ef[-2] + ef[1]) along axis
        slc0 = [slice(None)] * 3
        slc0[axis] = 0
        slcm2 = [slice(None)] * 3
        slcm2[axis] = -2
        slc1 = [slice(None)] * 3
        slc1[axis] = 1
        expected = 0.5 * (ef[tuple(slcm2)] + ef[tuple(slc1)])
        np.testing.assert_allclose(ref[tuple(slc0)], expected)

        # boundary[-1] should equal boundary[0] for periodic
        slcm1 = [slice(None)] * 3
        slcm1[axis] = -1
        np.testing.assert_allclose(ref[tuple(slcm1)], expected)


# ===================================================================
# Magnetic-field relocation
# ===================================================================


class TestRelocatedMagneticFieldUniform:
    """Uniform magnetic field stays uniform after 4-point averaging."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize(
        "btypes",
        [
            ("periodic", "periodic"),
            ("dirichlet", "dirichlet"),
            ("neumann", "neumann"),
            ("periodic", "neumann"),
            ("dirichlet", "periodic"),
        ],
    )
    def test_uniform_field(self, axis, btypes):
        shape = (5, 6, 7)
        bf = _uniform_field(4.0, shape)
        rbf = relocated_magnetic_field(bf, axis, btypes)
        # Uniform field -> everywhere 4.0 for periodic/neumann
        # dirichlet may flip sign at boundary
        if "dirichlet" not in btypes:
            np.testing.assert_allclose(rbf, 4.0)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_uniform_all_periodic(self, axis):
        bf = _uniform_field(2.5, (5, 6, 7))
        rbf = relocated_magnetic_field(bf, axis, ("periodic", "periodic"))
        np.testing.assert_allclose(rbf, 2.5)


class TestRelocatedMagneticFieldShape:
    """Output shape must match input shape."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize(
        "btypes",
        [
            ("periodic", "periodic"),
            ("dirichlet", "neumann"),
            ("neumann", "dirichlet"),
        ],
    )
    def test_shape_preserved(self, axis, btypes):
        shape = (4, 5, 6)
        bf = np.random.default_rng(0).random(shape)
        rbf = relocated_magnetic_field(bf, axis, btypes)
        assert rbf.shape == shape


class TestRelocatedMagneticFieldAveraging:
    """Interior averaging uses 4 surrounding points in the orthogonal plane."""

    def test_manual_small_array_axis0(self):
        """Hand-compute the expected result for a tiny 3x3x3 array, axis=0."""
        bf = np.arange(27, dtype=float).reshape(3, 3, 3)
        rbf = relocated_magnetic_field(bf, axis=0, btypes=("periodic", "periodic"))

        # The extended array (bfe) has shape (3, 4, 4).
        # For interior region bfe[:, 1:-1, 1:-1] = bf[:, :-1, :-1]
        # Then periodic boundaries are applied.
        # rbf = 0.25 * (bfe[:,:-1,:-1] + bfe[:,1:,:-1] + bfe[:,:-1,1:] + bfe[:,1:,1:])
        # Check that the result has correct shape
        assert rbf.shape == (3, 3, 3)

        # Verify against explicit computation
        # Build extended array manually
        axis = 0
        axis1 = 1
        axis2 = 2
        bfe = np.empty((3, 4, 4))
        bfe[:, 1:-1, 1:-1] = bf[:, :-1, :-1]

        # btypes[0] = periodic -> axis2 boundaries
        bfe[:, 1:-1, 0] = bfe[:, 1:-1, -2]
        bfe[:, 1:-1, -1] = bfe[:, 1:-1, 1]

        # btypes[1] = periodic -> axis1 boundaries
        bfe[:, 0, :] = bfe[:, -2, :]
        bfe[:, -1, :] = bfe[:, 1, :]

        expected = 0.25 * (bfe[:, :-1, :-1] + bfe[:, 1:, :-1] + bfe[:, :-1, 1:] + bfe[:, 1:, 1:])
        np.testing.assert_allclose(rbf, expected)


class TestRelocatedMagneticFieldBoundaryTypes:
    """Each boundary type (periodic/dirichlet/neumann) for each orthogonal axis."""

    @pytest.mark.parametrize(
        "btypes",
        [
            ("dirichlet", "periodic"),
            ("periodic", "dirichlet"),
            ("neumann", "periodic"),
            ("periodic", "neumann"),
            ("dirichlet", "neumann"),
            ("neumann", "dirichlet"),
        ],
    )
    def test_mixed_btypes_no_crash(self, btypes):
        """All btypes combinations should run without error."""
        shape = (5, 6, 7)
        bf = np.random.default_rng(99).random(shape)
        rbf = relocated_magnetic_field(bf, axis=1, btypes=btypes)
        assert rbf.shape == shape
        assert np.all(np.isfinite(rbf))


class TestRelocatedMagneticFieldDirichletSign:
    """Dirichlet boundary flips the sign of the ghost cell."""

    def test_dirichlet_axis2_sign_flip(self):
        """For dirichlet on axis2 (btypes[0]), the ghost is -interior."""
        shape = (4, 5, 6)
        bf = np.ones(shape, dtype=float)
        rbf = relocated_magnetic_field(bf, axis=0, btypes=("dirichlet", "periodic"))
        # Interior should be 1.0, but boundary cells along axis2 will
        # have contributions from -1 ghost cells
        assert rbf.shape == shape
        # The last column along axis2 should differ from 1.0
        # (because bfe[..., -1] = -bfe[..., -2] for dirichlet)
        assert not np.allclose(rbf[:, :, -1], 1.0)


class TestRelocatedMagneticFieldNeumannCopy:
    """Neumann boundary copies interior to ghost (no sign flip)."""

    def test_neumann_uniform(self):
        """Uniform field + neumann should stay uniform everywhere."""
        shape = (5, 6, 7)
        bf = _uniform_field(3.0, shape)
        rbf = relocated_magnetic_field(bf, axis=2, btypes=("neumann", "neumann"))
        np.testing.assert_allclose(rbf, 3.0)
