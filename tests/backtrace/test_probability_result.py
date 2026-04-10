from types import SimpleNamespace

import numpy as np

from emout.core.backtrace import probability_result as probability_result_module
from emout.core.backtrace.probability_result import ProbabilityResult


class _IdentityConverter:
    def __init__(self, unit: str):
        self.unit = unit

    def reverse(self, value):
        return value


_TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _integrate_expected(probabilities_nd, axis_values, var1, var2):
    current_axes = list(ProbabilityResult._AXES)
    integrated = probabilities_nd

    for axis_name in ProbabilityResult._AXES:
        if axis_name in (var1, var2):
            continue

        axis_idx = current_axes.index(axis_name)
        coords = axis_values[axis_name]

        if coords.size <= 1:
            integrated = np.take(integrated, indices=0, axis=axis_idx)
        else:
            if coords[0] > coords[-1]:
                integrated = np.flip(integrated, axis=axis_idx)
                coords = coords[::-1]
            integrated = _TRAPEZOID(integrated, x=coords, axis=axis_idx)

        current_axes.remove(axis_name)

    return np.moveaxis(
        integrated,
        (current_axes.index(var2), current_axes.index(var1)),
        (0, 1),
    )


def _make_probability_result(*, flatten_phases: bool, flatten_probabilities: bool, unit=None, inp=None):
    axis_values = {
        "x": np.array([10.0, 20.0]),
        "y": np.array([-3.0, 7.0]),
        "z": np.array([100.0, 200.0]),
        "vx": np.array([-2.0, 4.0]),
        "vy": np.array([1.0, 9.0]),
        "vz": np.array([0.5, 1.5, 2.5]),
    }
    dims = tuple(len(axis_values[axis]) for axis in ProbabilityResult._AXES)
    grids = np.meshgrid(
        *(axis_values[axis] for axis in ProbabilityResult._AXES),
        indexing="ij",
    )
    phases_nd = np.stack(grids, axis=-1)
    probabilities_nd = np.arange(1, np.prod(dims) + 1, dtype=float).reshape(dims)

    backend_axes = ProbabilityResult._BACKEND_AXES
    backend_phases = np.transpose(
        phases_nd,
        axes=[ProbabilityResult._AXES.index(axis) for axis in backend_axes] + [6],
    )
    backend_probabilities = np.transpose(
        probabilities_nd,
        axes=[ProbabilityResult._AXES.index(axis) for axis in backend_axes],
    )

    phases = backend_phases.reshape(-1, len(ProbabilityResult._AXES))
    probabilities = backend_probabilities.reshape(-1)

    if not flatten_phases:
        phases = phases_nd
    if not flatten_probabilities:
        probabilities = probabilities_nd

    result = ProbabilityResult(
        phases=phases,
        probabilities=probabilities,
        dims=dims,
        ret_particles=None,
        particles=None,
        ispec=0,
        inp=inp,
        unit=unit,
    )

    return result, axis_values, probabilities_nd


def test_pair_integrates_unspecified_axes_from_flat_grid():
    result, axis_values, probabilities_nd = _make_probability_result(
        flatten_phases=True,
        flatten_probabilities=True,
    )

    heatmap = result.pair("x", "vz")
    expected = _integrate_expected(probabilities_nd, axis_values, "x", "vz")

    np.testing.assert_array_equal(heatmap.X[0], axis_values["x"])
    np.testing.assert_array_equal(heatmap.Y[:, 0], axis_values["vz"])
    np.testing.assert_array_equal(heatmap.Z, expected)


def test_pair_accepts_backend_axis_order_from_phase_grid():
    axis_values = {
        "x": np.array([10.0, 20.0]),
        "y": np.array([-3.0, 7.0]),
        "z": np.array([100.0, 200.0]),
        "vx": np.array([-2.0, 4.0]),
        "vy": np.array([1.0, 9.0]),
        "vz": np.array([0.5, 1.5, 2.5]),
    }
    dims = tuple(len(axis_values[axis]) for axis in ProbabilityResult._AXES)
    backend_axes = ProbabilityResult._BACKEND_AXES
    backend_grids = np.meshgrid(
        *(axis_values[axis] for axis in backend_axes),
        indexing="ij",
    )
    backend_phases = np.stack(
        [backend_grids[backend_axes.index(axis)] for axis in ProbabilityResult._AXES],
        axis=-1,
    )
    probabilities_backend = np.arange(
        1,
        np.prod(dims) + 1,
        dtype=float,
    ).reshape(*(len(axis_values[axis]) for axis in backend_axes))

    result = ProbabilityResult(
        phases=backend_phases,
        probabilities=probabilities_backend.reshape(-1),
        dims=dims,
        ret_particles=None,
        particles=None,
        ispec=0,
        inp=None,
        unit=None,
    )

    heatmap = result.pair("x", "vz")
    expected = _integrate_expected(
        np.transpose(
            probabilities_backend,
            axes=[backend_axes.index(axis) for axis in ProbabilityResult._AXES],
        ),
        axis_values,
        "x",
        "vz",
    )

    np.testing.assert_array_equal(heatmap.X[0], axis_values["x"])
    np.testing.assert_array_equal(heatmap.Y[:, 0], axis_values["vz"])
    np.testing.assert_array_equal(heatmap.Z, expected)


def test_pair_accepts_nd_inputs_and_keeps_requested_axis_order():
    result, axis_values, probabilities_nd = _make_probability_result(
        flatten_phases=False,
        flatten_probabilities=False,
    )

    heatmap = result.pair("vz", "x")
    expected = _integrate_expected(probabilities_nd, axis_values, "vz", "x")

    np.testing.assert_array_equal(heatmap.X[0], axis_values["vz"])
    np.testing.assert_array_equal(heatmap.Y[:, 0], axis_values["x"])
    np.testing.assert_array_equal(heatmap.Z, expected)


def test_pair_keeps_singleton_axes_without_zeroing_integral():
    axis_values = {
        "x": np.array([10.0, 20.0]),
        "y": np.array([0.0]),
        "z": np.array([100.0]),
        "vx": np.array([-2.0, 4.0]),
        "vy": np.array([1.0]),
        "vz": np.array([0.5, 1.5, 2.5]),
    }
    dims = tuple(len(axis_values[axis]) for axis in ProbabilityResult._AXES)
    grids = np.meshgrid(
        *(axis_values[axis] for axis in ProbabilityResult._AXES),
        indexing="ij",
    )
    phases_nd = np.stack(grids, axis=-1)
    probabilities_nd = np.arange(1, np.prod(dims) + 1, dtype=float).reshape(dims)

    result = ProbabilityResult(
        phases=phases_nd,
        probabilities=probabilities_nd,
        dims=dims,
        ret_particles=None,
        particles=None,
        ispec=0,
        inp=None,
        unit=None,
    )

    heatmap = result.pair("x", "vz")
    expected = _integrate_expected(probabilities_nd, axis_values, "x", "vz")

    assert np.all(expected > 0)
    np.testing.assert_array_equal(heatmap.Z, expected)


def test_energy_spectrum_flattens_normalized_probability_grid(monkeypatch):
    unit = SimpleNamespace(
        v=_IdentityConverter("m/s"),
        m=_IdentityConverter("kg"),
        J=_IdentityConverter("A/m^2"),
        f=_IdentityConverter("Hz"),
    )
    inp = SimpleNamespace(
        qm=[-1.0],
        nflag_emit=[0],
        wp=[2.0],
        curf=[1.0],
        path=[0.0, 0.0, 1.0],
    )
    result, _, _ = _make_probability_result(
        flatten_phases=True,
        flatten_probabilities=False,
        unit=unit,
        inp=inp,
    )
    captured = {}

    def _fake_compute_energy_flux_histogram(velocities, probs, mass, energy_bins):
        captured["velocities_shape"] = velocities.shape
        captured["probs_shape"] = probs.shape
        captured["energy_bins"] = energy_bins
        return np.array([1.0]), np.array([0.0, 1.0])

    monkeypatch.setattr(
        probability_result_module,
        "compute_energy_flux_histogram",
        _fake_compute_energy_flux_histogram,
    )

    hist, bin_edges = result.energy_spectrum(energy_bins=4)

    assert captured["velocities_shape"] == (np.prod(result.dims), 3)
    assert captured["probs_shape"] == (np.prod(result.dims),)
    assert captured["energy_bins"] == 4
    np.testing.assert_array_equal(hist, np.array([1.0]))
    np.testing.assert_array_equal(bin_edges, np.array([0.0, 1.0]))
