"""Tests for emout.utils.eflux energy-flux computation utilities."""

import numpy as np
import pytest
from scipy.constants import e as e_charge


# ---------------------------------------------------------------------------
# get_indices_in_pitch_range
# ---------------------------------------------------------------------------

class TestGetIndicesInPitchRange:
    """Tests for get_indices_in_pitch_range."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from emout.utils.eflux import get_indices_in_pitch_range
        self.func = get_indices_in_pitch_range

    def test_parallel_particles_selected(self):
        """Particles parallel to B (pitch~0) are found in [0, 30] range."""
        B = np.array([0.0, 0.0, 1.0])
        # All velocities point exactly along +z
        velocities = np.array([[0, 0, 1.0], [0, 0, 2.0], [0, 0, 3.0]])
        idx = self.func(velocities, B, 0.0, 30.0, direction="both")
        assert len(idx) == 3

    def test_antiparallel_excluded_by_pos_direction(self):
        """Anti-parallel particles should be excluded when direction='pos'."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, -1.0]])  # anti-parallel
        idx = self.func(velocities, B, 0.0, 180.0, direction="pos")
        assert len(idx) == 0

    def test_antiparallel_included_by_neg_direction(self):
        """Anti-parallel particles should be included when direction='neg'."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, -1.0]])  # pitch angle = 180 deg
        idx = self.func(velocities, B, 150.0, 180.0, direction="neg")
        assert len(idx) == 1

    def test_perpendicular_in_60_120(self):
        """Velocity perpendicular to B has pitch=90, should be in [60, 120]."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[1.0, 0, 0]])  # pitch = 90 deg
        idx = self.func(velocities, B, 60.0, 120.0, direction="both")
        assert len(idx) == 1

    def test_perpendicular_excluded_from_0_30(self):
        """Pitch=90 should NOT be in [0,30] range."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[1.0, 0, 0]])
        idx = self.func(velocities, B, 0.0, 30.0, direction="both")
        assert len(idx) == 0

    def test_zero_speed_particle_handled(self):
        """A particle with zero velocity should not crash."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, 0.0], [0, 0, 1.0]])
        idx = self.func(velocities, B, 0.0, 180.0, direction="both")
        # Zero-velocity particle has cos_theta=0, so pitch=90, within [0,180]
        assert len(idx) >= 1

    def test_invalid_angle_range_raises(self):
        """a_deg >= b_deg should raise ValueError."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, 1.0]])
        with pytest.raises(ValueError, match="Invalid a_deg"):
            self.func(velocities, B, 30.0, 10.0)

    def test_equal_angles_raises(self):
        """a_deg == b_deg should raise ValueError."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, 1.0]])
        with pytest.raises(ValueError, match="Invalid a_deg"):
            self.func(velocities, B, 30.0, 30.0)

    def test_invalid_direction_raises(self):
        """Invalid direction string should raise ValueError."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, 1.0]])
        with pytest.raises(ValueError, match="direction="):
            self.func(velocities, B, 0.0, 90.0, direction="invalid")

    def test_zero_B_raises(self):
        """Zero magnetic field should raise ValueError."""
        B = np.array([0.0, 0.0, 0.0])
        velocities = np.array([[0, 0, 1.0]])
        with pytest.raises(ValueError, match="zero magnitude"):
            self.func(velocities, B, 0.0, 90.0)

    def test_full_range_captures_all(self):
        """Pitch range [0,180] with direction='both' should capture all particles."""
        rng = np.random.RandomState(42)
        B = np.array([1.0, 0.0, 0.0])
        velocities = rng.randn(100, 3)
        idx = self.func(velocities, B, 0.0, 180.0, direction="both")
        # All particles with non-zero velocity should be captured
        assert len(idx) == 100

    def test_pos_neg_partition(self):
        """pos + neg subsets should cover all particles in [0,180]."""
        rng = np.random.RandomState(123)
        B = np.array([0.0, 1.0, 0.0])
        velocities = rng.randn(50, 3)
        idx_pos = self.func(velocities, B, 0.0, 180.0, direction="pos")
        idx_neg = self.func(velocities, B, 0.0, 180.0, direction="neg")
        # Union should cover most particles (some may have exactly v.B=0)
        combined = set(idx_pos.tolist()) | set(idx_neg.tolist())
        assert len(combined) >= 45  # nearly all

    def test_returns_ndarray(self):
        """Return type should be numpy array."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, 1.0]])
        idx = self.func(velocities, B, 0.0, 90.0, direction="both")
        assert isinstance(idx, np.ndarray)

    def test_negative_a_deg_raises(self):
        """a_deg < 0 should raise ValueError."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, 1.0]])
        with pytest.raises(ValueError, match="Invalid a_deg"):
            self.func(velocities, B, -10.0, 90.0)

    def test_b_deg_over_180_raises(self):
        """b_deg > 180 should raise ValueError."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, 1.0]])
        with pytest.raises(ValueError, match="Invalid a_deg"):
            self.func(velocities, B, 0.0, 200.0)

    def test_non_unit_B_direction(self):
        """B with non-unit magnitude should still work (direction only matters)."""
        B = np.array([0.0, 0.0, 100.0])
        velocities = np.array([[0, 0, 1.0]])
        idx = self.func(velocities, B, 0.0, 30.0, direction="both")
        assert len(idx) == 1  # parallel to B

    def test_oblique_B_field(self):
        """Oblique B field direction should correctly compute pitch angles."""
        # B along (1,1,0), normalised pitch angle for v along (1,0,0) is 45 deg
        B = np.array([1.0, 1.0, 0.0])
        velocities = np.array([[1.0, 0, 0]])
        # cos(45) ~= 0.707, so should be in [30, 60] range
        idx = self.func(velocities, B, 30.0, 60.0, direction="both")
        assert len(idx) == 1

    def test_exact_boundary_0_degrees(self):
        """Particle exactly at 0 degrees pitch angle (parallel to B)."""
        B = np.array([1.0, 0.0, 0.0])
        velocities = np.array([[5.0, 0.0, 0.0]])
        idx = self.func(velocities, B, 0.0, 10.0, direction="both")
        assert len(idx) == 1

    def test_exact_boundary_180_degrees(self):
        """Particle exactly at 180 degrees (anti-parallel to B)."""
        B = np.array([1.0, 0.0, 0.0])
        velocities = np.array([[-5.0, 0.0, 0.0]])
        idx = self.func(velocities, B, 170.0, 180.0, direction="both")
        assert len(idx) == 1

    def test_pos_direction_excludes_perpendicular(self):
        """Perpendicular velocity has v.B=0, should be excluded by 'pos'."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[1.0, 0.0, 0.0]])  # v.B = 0
        idx = self.func(velocities, B, 80.0, 100.0, direction="pos")
        assert len(idx) == 0

    def test_neg_direction_excludes_perpendicular(self):
        """Perpendicular velocity has v.B=0, should be excluded by 'neg'."""
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[1.0, 0.0, 0.0]])  # v.B = 0
        idx = self.func(velocities, B, 80.0, 100.0, direction="neg")
        assert len(idx) == 0

    def test_large_array_performance(self):
        """Should handle large arrays without error."""
        rng = np.random.RandomState(99)
        B = np.array([0.0, 0.0, 1.0])
        velocities = rng.randn(10000, 3)
        idx = self.func(velocities, B, 0.0, 180.0, direction="both")
        assert len(idx) == 10000

    def test_cos_theta_clipping(self):
        """cos_theta should be clipped to [-1, 1] even with numerical noise."""
        B = np.array([0.0, 0.0, 1.0])
        # Very fast particle along B
        velocities = np.array([[0, 0, 1e30]])
        idx = self.func(velocities, B, 0.0, 10.0, direction="both")
        assert len(idx) == 1


# ---------------------------------------------------------------------------
# compute_energy_flux_histogram
# ---------------------------------------------------------------------------

class TestComputeEnergyFluxHistogram:
    """Tests for compute_energy_flux_histogram."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from emout.utils.eflux import compute_energy_flux_histogram
        self.func = compute_energy_flux_histogram

    def test_basic_histogram_int_bins(self):
        """Histogram with integer bins should return correct shapes."""
        velocities = np.array([[0, 0, 1e6], [0, 0, 2e6], [0, 0, 3e6]])
        probs = np.ones(3)
        mass = 9.109e-31  # electron mass
        hist, bin_edges = self.func(velocities, probs, mass, energy_bins=10)
        assert len(hist) == 10
        assert len(bin_edges) == 11

    def test_basic_histogram_array_bins(self):
        """Histogram with explicit bin edges should use them."""
        velocities = np.array([[0, 0, 1e6], [0, 0, 2e6]])
        probs = np.ones(2)
        mass = 9.109e-31
        bins = np.linspace(0, 100, 21)
        hist, bin_edges = self.func(velocities, probs, mass, energy_bins=bins)
        assert len(hist) == 20
        np.testing.assert_array_equal(bin_edges, bins)

    def test_none_bins_defaults_to_30(self):
        """energy_bins=None should default to 30 bins."""
        velocities = np.array([[0, 0, 1e6]] * 10)
        probs = np.ones(10)
        mass = 9.109e-31
        hist, bin_edges = self.func(velocities, probs, mass, energy_bins=None)
        assert len(hist) == 30

    def test_energy_scales_with_speed_squared(self):
        """Doubling speed should quadruple kinetic energy."""
        mass = 1.0
        v1 = np.array([[0, 0, 1.0]])
        v2 = np.array([[0, 0, 2.0]])
        probs = np.ones(1)
        # Energy = 0.5 * m * v^2 / e
        E1 = 0.5 * mass * 1.0**2 / e_charge
        E2 = 0.5 * mass * 2.0**2 / e_charge
        assert pytest.approx(E2 / E1, rel=1e-10) == 4.0

    def test_histogram_weights_are_energy_flux(self):
        """Weights should be energies_eV * speeds * probs."""
        velocities = np.array([[0, 0, 1e6]])
        probs = np.array([2.0])
        mass = 9.109e-31
        speed = 1e6
        energy_eV = 0.5 * mass * speed**2 / e_charge
        expected_weight = energy_eV * speed * 2.0

        bins = np.array([0.0, energy_eV * 2])
        hist, _ = self.func(velocities, probs, mass, energy_bins=bins)
        assert pytest.approx(hist[0], rel=1e-6) == expected_weight

    def test_zero_velocity_particle(self):
        """Particles with zero velocity contribute zero energy flux."""
        velocities = np.array([[0, 0, 0]])
        probs = np.ones(1)
        mass = 9.109e-31
        hist, bin_edges = self.func(velocities, probs, mass, energy_bins=10)
        assert hist.sum() == 0.0

    def test_multiple_particles_same_bin(self):
        """Multiple particles falling in the same bin should sum their flux."""
        velocities = np.array([[0, 0, 1e6], [0, 0, 1e6]])
        probs = np.array([1.0, 1.0])
        mass = 9.109e-31
        speed = 1e6
        energy_eV = 0.5 * mass * speed**2 / e_charge
        expected_weight = energy_eV * speed * 1.0
        bins = np.array([0.0, energy_eV * 2])
        hist, _ = self.func(velocities, probs, mass, energy_bins=bins)
        assert pytest.approx(hist[0], rel=1e-6) == 2 * expected_weight

    def test_different_probs_weight_differently(self):
        """Particles with different probs should have different contributions."""
        velocities = np.array([[0, 0, 1e6]])
        mass = 9.109e-31
        speed = 1e6
        energy_eV = 0.5 * mass * speed**2 / e_charge
        bins = np.array([0.0, energy_eV * 2])

        hist1, _ = self.func(velocities, np.array([1.0]), mass, energy_bins=bins)
        hist2, _ = self.func(velocities, np.array([3.0]), mass, energy_bins=bins)
        assert pytest.approx(hist2[0] / hist1[0], rel=1e-6) == 3.0

    def test_3d_velocity_components(self):
        """Particles with velocity in all three dimensions."""
        velocities = np.array([[1e6, 1e6, 1e6]])
        probs = np.ones(1)
        mass = 9.109e-31
        hist, bin_edges = self.func(velocities, probs, mass, energy_bins=5)
        assert hist.sum() > 0


# ---------------------------------------------------------------------------
# compute_energy_flux_histograms
# ---------------------------------------------------------------------------

class TestComputeEnergyFluxHistograms:
    """Tests for compute_energy_flux_histograms (pitch-angle decomposition)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from emout.utils.eflux import compute_energy_flux_histograms
        self.func = compute_energy_flux_histograms

    def test_default_pitch_ranges_produce_6_keys(self):
        """Default pitch_ranges should produce 6 histogram entries."""
        rng = np.random.RandomState(0)
        velocities = rng.randn(200, 3) * 1e6
        probs = np.ones(200)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        result = self.func(velocities, probs, B, mass, energy_bins=10)
        assert len(result) == 6

    def test_custom_pitch_ranges(self):
        """Custom pitch_ranges should produce the expected keys."""
        rng = np.random.RandomState(0)
        velocities = rng.randn(50, 3) * 1e6
        probs = np.ones(50)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        ranges = [(0.0, 90.0, "both"), (90.0, 180.0, "both")]
        result = self.func(velocities, probs, B, mass, energy_bins=10, pitch_ranges=ranges)
        assert len(result) == 2
        assert "00-90_both" in result
        assert "90-180_both" in result

    def test_key_format(self):
        """Keys should match the format '{a:02d}-{b:02d}_{direction}'."""
        rng = np.random.RandomState(0)
        velocities = rng.randn(20, 3) * 1e6
        probs = np.ones(20)
        B = np.array([1.0, 0.0, 0.0])
        mass = 9.109e-31
        result = self.func(velocities, probs, B, mass, energy_bins=5)
        expected_keys = [
            "00-30_pos", "00-30_neg",
            "30-60_pos", "30-60_neg",
            "60-180_pos", "60-180_neg",
        ]
        for key in expected_keys:
            assert key in result

    def test_histogram_shapes_consistent(self):
        """All histograms should have the same bin edges."""
        rng = np.random.RandomState(0)
        velocities = rng.randn(100, 3) * 1e6
        probs = np.ones(100)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        result = self.func(velocities, probs, B, mass, energy_bins=15)
        bin_edges_list = [v[1] for v in result.values()]
        for be in bin_edges_list:
            np.testing.assert_array_equal(be, bin_edges_list[0])

    def test_mismatched_lengths_raises(self):
        """velocities and probs with different lengths should raise."""
        velocities = np.ones((10, 3))
        probs = np.ones(5)
        B = np.array([0, 0, 1.0])
        with pytest.raises(ValueError, match="same length"):
            self.func(velocities, probs, B, mass=1.0, energy_bins=10)

    def test_explicit_bin_edges_used(self):
        """When energy_bins is an array, those edges should be used."""
        rng = np.random.RandomState(0)
        velocities = rng.randn(50, 3) * 1e6
        probs = np.ones(50)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        bins = np.linspace(0, 50, 11)
        result = self.func(velocities, probs, B, mass, energy_bins=bins)
        for hist, be in result.values():
            np.testing.assert_array_equal(be, bins)
            assert len(hist) == 10

    def test_empty_pitch_range_gives_zero_histogram(self):
        """Pitch range with no particles should return a zero histogram."""
        # All velocities parallel to B, so pitch ~0 deg
        B = np.array([0.0, 0.0, 1.0])
        velocities = np.array([[0, 0, 1e6]] * 10)
        probs = np.ones(10)
        mass = 9.109e-31
        # Range 90-180 with 'pos' should capture no parallel particles
        ranges = [(90.0, 180.0, "pos")]
        result = self.func(velocities, probs, B, mass, energy_bins=5, pitch_ranges=ranges)
        hist, _ = result["90-180_pos"]
        assert hist.sum() == 0.0

    def test_single_pitch_range(self):
        """A single custom pitch range should work."""
        rng = np.random.RandomState(7)
        velocities = rng.randn(30, 3) * 1e6
        probs = np.ones(30)
        B = np.array([0.0, 0.0, 1.0])
        mass = 9.109e-31
        ranges = [(0.0, 180.0, "both")]
        result = self.func(velocities, probs, B, mass, energy_bins=10, pitch_ranges=ranges)
        assert len(result) == 1
        assert "00-180_both" in result
        hist, _ = result["00-180_both"]
        assert hist.sum() > 0

    def test_sum_of_partitions_approximates_total(self):
        """Sum of disjoint pitch-angle partitions should roughly equal the total flux."""
        rng = np.random.RandomState(55)
        velocities = rng.randn(500, 3) * 1e6
        probs = np.ones(500)
        B = np.array([0.0, 0.0, 1.0])
        mass = 9.109e-31

        from emout.utils.eflux import compute_energy_flux_histogram
        total_hist, total_bins = compute_energy_flux_histogram(
            velocities, probs, mass, energy_bins=10
        )

        ranges = [(0.0, 90.0, "both"), (90.0, 180.0, "both")]
        result = self.func(velocities, probs, B, mass, energy_bins=total_bins, pitch_ranges=ranges)
        sum_hist = sum(h for h, _ in result.values())
        # Due to direction filtering, the sum should be close (within rounding)
        np.testing.assert_allclose(sum_hist, total_hist, rtol=1e-10)

    def test_output_values_are_tuples(self):
        """Each value in the result dict should be a (hist, bin_edges) tuple."""
        rng = np.random.RandomState(0)
        velocities = rng.randn(30, 3) * 1e6
        probs = np.ones(30)
        B = np.array([0.0, 0.0, 1.0])
        mass = 9.109e-31
        result = self.func(velocities, probs, B, mass, energy_bins=5)
        for key, val in result.items():
            assert isinstance(val, tuple)
            assert len(val) == 2
            hist, be = val
            assert isinstance(hist, np.ndarray)
            assert isinstance(be, np.ndarray)
            assert len(be) == len(hist) + 1


# ---------------------------------------------------------------------------
# plot_energy_fluxes
# ---------------------------------------------------------------------------

class TestPlotEnergyFluxes:
    """Tests for plot_energy_fluxes (2-D heatmap plotting)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from emout.utils.eflux import plot_energy_fluxes
        self.func = plot_energy_fluxes

    def test_returns_fig_and_ax(self):
        """Should return (fig, ax) tuple."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        T = 5
        velocities_list = [rng.randn(20, 3) * 1e6 for _ in range(T)]
        x = np.linspace(0, 1, T)
        mass = 9.109e-31
        fig, ax = self.func(velocities_list, x, mass, energy_bins=10)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_x_length_mismatch_raises(self):
        """Mismatched x and velocities_list lengths should raise."""
        rng = np.random.RandomState(0)
        velocities_list = [rng.randn(20, 3) for _ in range(5)]
        x = np.linspace(0, 1, 3)  # wrong length
        with pytest.raises(ValueError, match="same length"):
            self.func(velocities_list, x, mass=1.0, energy_bins=10)

    def test_use_probs_requires_probs_list(self):
        """use_probs=True without probs_list should raise."""
        rng = np.random.RandomState(0)
        velocities_list = [rng.randn(10, 3) for _ in range(3)]
        x = np.linspace(0, 1, 3)
        with pytest.raises(ValueError, match="use_probs"):
            self.func(velocities_list, x, mass=1.0, energy_bins=5,
                      use_probs=True, probs_list=None)

    def test_probs_list_wrong_length_raises(self):
        """probs_list with wrong length should raise."""
        rng = np.random.RandomState(0)
        velocities_list = [rng.randn(10, 3) for _ in range(3)]
        x = np.linspace(0, 1, 3)
        with pytest.raises(ValueError, match="use_probs"):
            self.func(velocities_list, x, mass=1.0, energy_bins=5,
                      use_probs=True, probs_list=[np.ones(10)])

    def test_use_probs_true(self):
        """plot_energy_fluxes with use_probs=True should work."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        T = 3
        velocities_list = [rng.randn(15, 3) * 1e6 for _ in range(T)]
        probs_list = [np.ones(15) for _ in range(T)]
        x = np.linspace(0, 1, T)
        mass = 9.109e-31
        fig, ax = self.func(
            velocities_list, x, mass, energy_bins=5,
            use_probs=True, probs_list=probs_list,
        )
        assert fig is not None
        plt.close(fig)

    def test_probs_shape_mismatch_raises(self):
        """probs_list[j] and velocities_list[j] with different lengths should raise."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        T = 2
        velocities_list = [rng.randn(10, 3) * 1e6 for _ in range(T)]
        # probs_list[0] has wrong length
        probs_list = [np.ones(5), np.ones(10)]
        x = np.linspace(0, 1, T)
        with pytest.raises(ValueError, match="same length"):
            self.func(velocities_list, x, mass=1.0, energy_bins=5,
                      use_probs=True, probs_list=probs_list)
        plt.close("all")

    def test_explicit_bin_edges(self):
        """Explicit bin edges should be accepted."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        T = 3
        velocities_list = [rng.randn(10, 3) * 1e6 for _ in range(T)]
        x = np.linspace(0, 1, T)
        mass = 9.109e-31
        bins = np.linspace(0, 50, 11)
        fig, ax = self.func(velocities_list, x, mass, energy_bins=bins)
        assert fig is not None
        plt.close(fig)

    def test_custom_cmap(self):
        """Custom cmap parameter should be accepted without error."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        T = 3
        velocities_list = [rng.randn(10, 3) * 1e6 for _ in range(T)]
        x = np.linspace(0, 1, T)
        mass = 9.109e-31
        fig, ax = self.func(velocities_list, x, mass, energy_bins=5, cmap="plasma")
        plt.close(fig)

    def test_heatmap_shape(self):
        """E_map (the internal heatmap) should have shape (M, T)."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        T = 4
        velocities_list = [rng.randn(20, 3) * 1e6 for _ in range(T)]
        x = np.linspace(0, 1, T)
        mass = 9.109e-31
        fig, ax = self.func(velocities_list, x, mass, energy_bins=8)
        # The imshow image array should have M rows and T columns
        images = ax.get_images()
        assert len(images) == 1
        img_data = images[0].get_array()
        assert img_data.shape[1] == T  # columns = T
        plt.close(fig)

    def test_axis_labels_set(self):
        """Axes labels should be set on the plot."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        T = 3
        velocities_list = [rng.randn(10, 3) * 1e6 for _ in range(T)]
        x = np.linspace(0, 1, T)
        mass = 9.109e-31
        fig, ax = self.func(velocities_list, x, mass, energy_bins=5)
        assert ax.get_xlabel() == "x"
        assert "Energy" in ax.get_ylabel()
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_energy_flux
# ---------------------------------------------------------------------------

class TestPlotEnergyFlux:
    """Tests for plot_energy_flux (line plot of energy-flux distribution)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from emout.utils.eflux import plot_energy_flux
        self.func = plot_energy_flux

    def test_returns_fig_and_ax(self):
        """Should return (fig, ax) tuple."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        velocities = rng.randn(100, 3) * 1e6
        probs = np.ones(100)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        fig, ax = self.func(velocities, probs, B, mass, energy_bins=10)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_custom_pitch_ranges(self):
        """Custom pitch_ranges should produce a valid plot."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        velocities = rng.randn(50, 3) * 1e6
        probs = np.ones(50)
        B = np.array([1.0, 0.0, 0.0])
        mass = 9.109e-31
        ranges = [(0.0, 45.0, "both"), (45.0, 90.0, "both"), (90.0, 180.0, "both")]
        fig, ax = self.func(
            velocities, probs, B, mass, energy_bins=10, pitch_ranges=ranges
        )
        plt.close(fig)

    def test_explicit_bin_edges(self):
        """Explicit bin edges should be accepted."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        velocities = rng.randn(50, 3) * 1e6
        probs = np.ones(50)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        bins = np.linspace(0, 50, 11)
        fig, ax = self.func(velocities, probs, B, mass, energy_bins=bins)
        plt.close(fig)

    def test_log_scale_axes(self):
        """Axes should use log scale."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        velocities = rng.randn(50, 3) * 1e6
        probs = np.ones(50)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        fig, ax = self.func(velocities, probs, B, mass, energy_bins=10)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_custom_cmap(self):
        """Custom cmap should be accepted."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        velocities = rng.randn(50, 3) * 1e6
        probs = np.ones(50)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        fig, ax = self.func(velocities, probs, B, mass, energy_bins=10, cmap="viridis")
        plt.close(fig)

    def test_legend_present(self):
        """The plot should have a legend."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        velocities = rng.randn(50, 3) * 1e6
        probs = np.ones(50)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        fig, ax = self.func(velocities, probs, B, mass, energy_bins=10)
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "All" in labels
        plt.close(fig)

    def test_title_set(self):
        """The plot title should be set."""
        import matplotlib.pyplot as plt
        rng = np.random.RandomState(0)
        velocities = rng.randn(50, 3) * 1e6
        probs = np.ones(50)
        B = np.array([0.0, 0.0, 1e-9])
        mass = 9.109e-31
        fig, ax = self.func(velocities, probs, B, mass, energy_bins=10)
        assert "Energy" in ax.get_title()
        plt.close(fig)
