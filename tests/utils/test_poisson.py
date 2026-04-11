"""Tests for emout.utils.poisson — FFT-based Poisson solver."""

import warnings

import numpy as np
import pytest

from emout.utils.poisson import (
    DirichletPoissonBoundary,
    FFT3d,
    NeumannPoissonBoundary,
    PeriodicPoissonBoundary,
    poisson,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_shape(nx, ny, nz):
    """Return grid shape (nz+1, ny+1, nx+1)."""
    return (nz + 1, ny + 1, nx + 1)


# ---------------------------------------------------------------------------
# 1. Default arguments: boundary_types and boundary_values initialised
# ---------------------------------------------------------------------------


class TestDefaultArguments:
    """Cover lines 50-53: default boundary_types/boundary_values."""

    def test_default_boundary_types_and_values(self):
        """Calling with only rho and dx should work (all-periodic, zero BV)."""
        shape = _grid_shape(8, 8, 8)
        rho = np.zeros(shape)
        phi = poisson(rho, dx=1.0)
        assert phi.shape == shape
        np.testing.assert_allclose(phi, 0.0)


# ---------------------------------------------------------------------------
# 2. btypes short-string parsing
# ---------------------------------------------------------------------------


class TestBtypesParsing:
    """Cover lines 56-62: short-string btypes → boundary_types expansion."""

    @pytest.mark.parametrize("btypes_str", ["ppp", "ddd", "nnn", "pdn", "dpn"])
    def test_btypes_string_accepted(self, btypes_str):
        """Each valid short string should run without error."""
        shape = _grid_shape(8, 8, 8)
        rho = np.zeros(shape)
        phi = poisson(rho, dx=1.0, btypes=btypes_str)
        assert phi.shape == shape

    def test_btypes_overrides_boundary_types(self):
        """btypes should take precedence over boundary_types."""
        shape = _grid_shape(8, 8, 8)
        rho = np.zeros(shape)
        # Pass contradictory values; btypes wins.
        phi_btypes = poisson(
            rho,
            dx=1.0,
            boundary_types=["dirichlet", "dirichlet", "dirichlet"],
            btypes="ppp",
        )
        phi_periodic = poisson(
            rho,
            dx=1.0,
            boundary_types=["periodic", "periodic", "periodic"],
        )
        np.testing.assert_allclose(phi_btypes, phi_periodic)


# ---------------------------------------------------------------------------
# 3. All-periodic with known analytic solution (sin wave)
# ---------------------------------------------------------------------------


class TestPeriodicAnalytic:
    """Verify against a known analytic solution on a periodic domain.

    For rho = sin(2*pi*kx*x/Lx) on a periodic domain [0, Lx),
    the solution of d^2 phi/dx^2 = -rho / epsilon_0 is
    phi = rho / (epsilon_0 * (2*pi*kx/Lx)^2).
    Here we work in normalised units (epsilon_0=1, dx=1) and only vary x.
    """

    def test_sinusoidal_source_periodic(self):
        nx, ny, nz = 32, 4, 4
        shape = _grid_shape(nx, ny, nz)
        dx = 1.0
        eps0 = 1.0

        # Wave with mode kx=1 along x
        kx = 1
        Lx = nx  # period in grid units (nx cells)
        x = np.arange(nx + 1, dtype=float)

        rho_1d = np.sin(2.0 * np.pi * kx * x / Lx)
        rho = np.zeros(shape)
        rho[:, :, :] = rho_1d[np.newaxis, np.newaxis, :]

        phi = poisson(rho, dx=dx, epsilon_0=eps0)

        # Expected: phi = rho / (2*sin(pi*kx/nx))^2
        # Because modified wave number for periodic is -[2*sin(pi*k/n)]^2
        # and the solver does phik = rhok / modified_wn, with rho_target = -rho/eps0*dx^2
        # Effectively phi = rho / (2*sin(pi*kx/nx))^2  for a single mode
        wn_sq = (2.0 * np.sin(np.pi * kx / nx)) ** 2
        phi_expected = rho / wn_sq
        # The periodic solver copies phi[-1]=phi[0]; adjust expected accordingly.
        phi_expected[:, :, -1] = phi_expected[:, :, 0]

        np.testing.assert_allclose(phi, phi_expected, atol=1e-10)

    def test_periodic_solver_avoids_runtime_and_complex_warnings(self):
        """The singular zero mode should be handled without divide/cast warnings."""
        nx, ny, nz = 32, 4, 4
        shape = _grid_shape(nx, ny, nz)
        x = np.arange(nx + 1, dtype=float)

        rho = np.zeros(shape)
        rho[:, :, :] = np.sin(2.0 * np.pi * x / nx)[np.newaxis, np.newaxis, :]

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            phi = poisson(rho, dx=1.0, epsilon_0=1.0)

        assert np.isrealobj(phi)


# ---------------------------------------------------------------------------
# 4. Mixed boundary types
# ---------------------------------------------------------------------------


class TestMixedBoundaries:
    """Cover mixed boundary combinations and all boundary class code paths."""

    def test_periodic_dirichlet_neumann(self):
        """btypes='pdn' should produce output of correct shape."""
        shape = _grid_shape(8, 8, 8)
        rho = np.zeros(shape)
        phi = poisson(rho, dx=1.0, btypes="pdn")
        assert phi.shape == shape
        np.testing.assert_allclose(phi, 0.0)

    def test_dirichlet_periodic_neumann(self):
        shape = _grid_shape(8, 8, 8)
        rho = np.zeros(shape)
        phi = poisson(rho, dx=1.0, btypes="dpn")
        assert phi.shape == shape

    def test_neumann_neumann_periodic(self):
        shape = _grid_shape(8, 8, 8)
        rho = np.zeros(shape)
        phi = poisson(rho, dx=1.0, btypes="nnp")
        assert phi.shape == shape


# ---------------------------------------------------------------------------
# 5. All-Dirichlet with non-zero boundary values
# ---------------------------------------------------------------------------


class TestDirichletNonZeroBV:
    """Cover boundary value transposition (lines 84-85) and correction (line 120).

    Boundary corrections are applied in x, y, z order; later axes overwrite
    at shared edges/corners, so we verify interior slices only.
    """

    def test_dirichlet_nonzero_boundaries(self):
        nx, ny, nz = 8, 8, 8
        shape = _grid_shape(nx, ny, nz)
        rho = np.zeros(shape)

        bv_x = (1.0, 2.0)
        bv_y = (0.0, 0.0)
        bv_z = (0.0, 0.0)
        phi = poisson(
            rho,
            dx=1.0,
            boundary_types=["dirichlet", "dirichlet", "dirichlet"],
            boundary_values=[bv_x, bv_y, bv_z],
            epsilon_0=1.0,
        )
        assert phi.shape == shape
        # x-boundary on interior y/z (edges overwritten by later y/z corrections)
        inner = slice(1, -1)
        np.testing.assert_allclose(phi[inner, inner, 0], bv_x[0], atol=1e-10)
        np.testing.assert_allclose(phi[inner, inner, -1], bv_x[1], atol=1e-10)
        # y-boundary values (overwritten at z-edges)
        np.testing.assert_allclose(phi[inner, 0, :], bv_y[0], atol=1e-10)
        np.testing.assert_allclose(phi[inner, -1, :], bv_y[1], atol=1e-10)
        # z-boundary values (applied last, full plane)
        np.testing.assert_allclose(phi[0, :, :], bv_z[0], atol=1e-10)
        np.testing.assert_allclose(phi[-1, :, :], bv_z[1], atol=1e-10)

    def test_dirichlet_all_zero_bv(self):
        """All-zero Dirichlet BVs with zero rho gives zero potential."""
        nx, ny, nz = 16, 4, 4
        shape = _grid_shape(nx, ny, nz)
        rho = np.zeros(shape)

        phi = poisson(
            rho,
            dx=1.0,
            boundary_types=["dirichlet", "dirichlet", "dirichlet"],
            boundary_values=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            epsilon_0=1.0,
        )
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. All-Neumann boundaries
# ---------------------------------------------------------------------------


class TestNeumannBoundaries:
    """Cover Neumann boundary class and the all-periodic/neumann zero-freq
    guard (lines 112-113)."""

    def test_all_neumann_zero_source(self):
        shape = _grid_shape(8, 8, 8)
        rho = np.zeros(shape)
        phi = poisson(
            rho,
            dx=1.0,
            boundary_types=["neumann", "neumann", "neumann"],
            epsilon_0=1.0,
        )
        assert phi.shape == shape
        # With zero source and zero Neumann BV, phi should be constant (zero).
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)

    def test_periodic_neumann_zero_freq(self):
        """Lines 112-113: if all BCs are periodic or neumann, phik[0,0,0]=0."""
        shape = _grid_shape(8, 8, 8)
        rho = np.zeros(shape)
        phi = poisson(
            rho,
            dx=1.0,
            boundary_types=["periodic", "neumann", "neumann"],
            epsilon_0=1.0,
        )
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Output shape matches input for various sizes
# ---------------------------------------------------------------------------


class TestOutputShape:
    @pytest.mark.parametrize("nx,ny,nz", [(4, 4, 4), (8, 16, 4), (16, 8, 32)])
    def test_output_shape_periodic(self, nx, ny, nz):
        shape = _grid_shape(nx, ny, nz)
        rho = np.zeros(shape)
        phi = poisson(rho, dx=1.0)
        assert phi.shape == shape

    @pytest.mark.parametrize("nx,ny,nz", [(4, 4, 4), (8, 8, 8)])
    def test_output_shape_dirichlet(self, nx, ny, nz):
        shape = _grid_shape(nx, ny, nz)
        rho = np.zeros(shape)
        phi = poisson(rho, dx=1.0, btypes="ddd")
        assert phi.shape == shape


# ---------------------------------------------------------------------------
# 8. Scalar dx parameter
# ---------------------------------------------------------------------------


class TestDxScalar:
    """dx affects the magnitude of the potential."""

    def test_dx_scaling(self):
        """phi scales as dx^2 (from rho_target = -rho/eps0 * dx^2)."""
        nx, ny, nz = 16, 4, 4
        shape = _grid_shape(nx, ny, nz)
        eps0 = 1.0

        kx = 1
        x = np.arange(nx + 1, dtype=float)
        rho_1d = np.sin(2.0 * np.pi * kx * x / nx)
        rho = np.zeros(shape)
        rho[:, :, :] = rho_1d[np.newaxis, np.newaxis, :]

        phi1 = poisson(rho, dx=1.0, epsilon_0=eps0)
        phi2 = poisson(rho, dx=2.0, epsilon_0=eps0)

        # phi should scale as dx^2
        np.testing.assert_allclose(phi2, phi1 * 4.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 9. Boundary class unit tests
# ---------------------------------------------------------------------------


class TestBoundaryClasses:
    """Direct tests on individual boundary classes."""

    def test_periodic_target_slice(self):
        b = PeriodicPoissonBoundary(axis=0)
        assert b.get_target_slice() == slice(0, -1)

    def test_dirichlet_target_slice(self):
        b = DirichletPoissonBoundary(axis=1)
        assert b.get_target_slice() == slice(1, -1)

    def test_neumann_target_slice(self):
        b = NeumannPoissonBoundary(axis=2)
        assert b.get_target_slice() == slice(None, None)

    def test_periodic_modified_wave_number_zero(self):
        b = PeriodicPoissonBoundary(axis=0)
        # k=0 → wn = 0
        assert b.modified_wave_number(0, 8) == 0.0

    def test_periodic_modified_wave_number_upper_half(self):
        """Covers the else branch (k > n/2) in periodic modified_wave_number."""
        b = PeriodicPoissonBoundary(axis=0)
        n = 8
        k_upper = 6  # > n/2 = 4
        k_mirror = n - k_upper  # = 2
        wn_upper = b.modified_wave_number(k_upper, n)
        wn_mirror = b.modified_wave_number(k_mirror, n)
        np.testing.assert_allclose(wn_upper, wn_mirror)

    def test_dirichlet_modified_wave_number(self):
        b = DirichletPoissonBoundary(axis=0)
        n = 8
        wn = b.modified_wave_number(0, n)
        expected = 2.0 * (np.cos(np.pi * 1 / (n + 1)) - 1.0)
        np.testing.assert_allclose(wn, expected)

    def test_neumann_modified_wave_number(self):
        b = NeumannPoissonBoundary(axis=0)
        n = 8
        wn = b.modified_wave_number(0, n)
        # k=0 → cos(0) - 1 = 0
        assert wn == 0.0

    def test_periodic_correct_boundary_copies(self):
        """PeriodicPoissonBoundary.correct_boundary_values copies phi[...,0] to phi[...,-1].

        axis=2 means _get_slices_at selects along dimension 2 (last dim).
        """
        b = PeriodicPoissonBoundary(axis=2, boundary_values=(0.0, 0.0))
        phi = np.random.rand(5, 5, 5)
        phi[:, :, 0] = 42.0
        b.correct_boundary_values(phi)
        np.testing.assert_array_equal(phi[:, :, -1], phi[:, :, 0])

    def test_dirichlet_correct_boundary_values(self):
        """axis=2 selects dimension 2 (last dim)."""
        b = DirichletPoissonBoundary(axis=2, boundary_values=(10.0, 20.0))
        phi = np.zeros((3, 3, 5))
        b.correct_boundary_values(phi)
        np.testing.assert_allclose(phi[:, :, 0], 10.0)
        np.testing.assert_allclose(phi[:, :, -1], 20.0)

    def test_dirichlet_transpose_boundary_values(self):
        """axis=0 selects dimension 0 (first dim)."""
        b = DirichletPoissonBoundary(axis=0, boundary_values=(3.0, 7.0))
        data = np.ones((4, 4, 4))
        b.transpose_boundary_values(data, dx=1.0)
        np.testing.assert_allclose(data[0, :, :], 1.0 - 3.0)
        np.testing.assert_allclose(data[-1, :, :], 1.0 - 7.0)

    def test_neumann_transpose_boundary_values(self):
        """axis=0 selects dimension 0 (first dim)."""
        b = NeumannPoissonBoundary(axis=0, boundary_values=(2.0, 5.0))
        data = np.ones((4, 4, 4))
        dx = 0.5
        b.transpose_boundary_values(data, dx=dx)
        np.testing.assert_allclose(data[0, :, :], 1.0 - 2.0 * dx)
        np.testing.assert_allclose(data[-1, :, :], 1.0 + 5.0 * dx)


# ---------------------------------------------------------------------------
# 10. FFT3d class
# ---------------------------------------------------------------------------


class TestFFT3d:
    """Direct tests on the FFT3d wrapper."""

    def test_roundtrip(self):
        """forward then backward should return the original data."""
        import scipy.fft

        forwards = [scipy.fft.fft, scipy.fft.fft, scipy.fft.fft]
        backwards = [scipy.fft.ifft, scipy.fft.ifft, scipy.fft.ifft]
        fft3d = FFT3d(forwards, backwards)

        data = np.random.rand(4, 4, 4)
        result = fft3d.backward(fft3d.forward(data))
        np.testing.assert_allclose(result.real, data, atol=1e-12)
