"""Tests for backtrace result containers, XYData, and solver wrapper.

Covers:
- emout/core/backtrace/xy_data.py       (XYData, MultiXYData, _insert_nans_for_gaps)
- emout/core/backtrace/backtrace_result.py (BacktraceResult)
- emout/core/backtrace/multi_backtrace_result.py (MultiBacktraceResult)
- emout/core/backtrace/solver_wrapper.py (BacktraceWrapper)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI

from emout.core.backtrace.xy_data import (
    MultiXYData,
    XYData,
    _insert_nans_for_gaps,
)
from emout.core.backtrace.backtrace_result import BacktraceResult
from emout.core.backtrace.multi_backtrace_result import MultiBacktraceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backtrace_result(n_steps=10, unit=None):
    """Build a BacktraceResult with deterministic data."""
    ts = np.arange(n_steps, dtype=float)
    probability = np.linspace(0.0, 1.0, n_steps)
    positions = np.column_stack([
        np.linspace(0, 1, n_steps),
        np.linspace(1, 2, n_steps),
        np.linspace(2, 3, n_steps),
    ])
    velocities = np.column_stack([
        np.linspace(10, 20, n_steps),
        np.linspace(20, 30, n_steps),
        np.linspace(30, 40, n_steps),
    ])
    return BacktraceResult(ts, probability, positions, velocities, unit=unit)


def _make_multi_backtrace_result(n_traj=3, n_steps=10, unit=None):
    """Build a MultiBacktraceResult with deterministic data."""
    ts_list = np.tile(np.arange(n_steps, dtype=float), (n_traj, 1))
    probabilities = np.linspace(0.5, 1.0, n_traj)
    positions_list = np.random.RandomState(42).randn(n_traj, n_steps, 3)
    velocities_list = np.random.RandomState(99).randn(n_traj, n_steps, 3)
    last_indexes = np.full(n_traj, n_steps, dtype=int)
    return MultiBacktraceResult(
        ts_list, probabilities, positions_list, velocities_list,
        last_indexes, unit=unit,
    )


class _IdentityConverter:
    """Minimal unit converter that returns the value unchanged."""

    def __init__(self, unit_str: str):
        self.unit = unit_str

    def reverse(self, v):
        return v

    def trans(self, v):
        return v


def _make_unit():
    """Build a SimpleNamespace mimicking the emout unit object."""
    return SimpleNamespace(
        t=_IdentityConverter("s"),
        length=_IdentityConverter("m"),
        v=_IdentityConverter("m/s"),
    )


# ===================================================================
# XYData
# ===================================================================

class TestXYData:
    """Tests for the XYData container."""

    def test_construction_stores_arrays(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        xy = XYData(x, y, xlabel="xx", ylabel="yy", title="my title")
        np.testing.assert_array_equal(xy.x, x)
        np.testing.assert_array_equal(xy.y, y)
        assert xy.xlabel == "xx"
        assert xy.ylabel == "yy"
        assert xy.title == "my title"

    def test_default_title_from_labels(self):
        xy = XYData(np.zeros(3), np.zeros(3), xlabel="a", ylabel="b")
        assert xy.title == "a vs b"

    def test_construction_rejects_2d_arrays(self):
        with pytest.raises(ValueError, match="1-D"):
            XYData(np.zeros((3, 2)), np.zeros((3, 2)))

    def test_construction_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            XYData(np.zeros(3), np.zeros(4))

    def test_iter_unpacking(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        xy = XYData(x, y)
        x_out, y_out = xy
        np.testing.assert_array_equal(x_out, x)
        np.testing.assert_array_equal(y_out, y)

    def test_repr(self):
        xy = XYData(np.zeros(5), np.ones(5), xlabel="t", ylabel="x")
        r = repr(xy)
        assert "XYData" in r
        assert "len=5" in r
        assert "xlabel=t" in r
        assert "ylabel=x" in r

    def test_plot_basic(self):
        """plot() should return an Axes and set labels/title."""
        import matplotlib.pyplot as plt

        xy = XYData(
            np.array([0.0, 1.0, 2.0]),
            np.array([3.0, 4.0, 5.0]),
            xlabel="time",
            ylabel="pos",
            title="Trajectory",
        )
        fig, ax = plt.subplots()
        ret = xy.plot(ax=ax, use_si=False)
        assert ret is ax
        assert ax.get_xlabel() == "time"
        assert ax.get_ylabel() == "pos"
        assert ax.get_title() == "Trajectory"
        plt.close(fig)

    def test_plot_with_units(self):
        """When units are provided and use_si=True, labels include unit."""
        import matplotlib.pyplot as plt

        units = (_IdentityConverter("m"), _IdentityConverter("m/s"))
        xy = XYData(
            np.array([0.0, 1.0]),
            np.array([2.0, 3.0]),
            xlabel="x",
            ylabel="vx",
            units=units,
        )
        fig, ax = plt.subplots()
        xy.plot(ax=ax, use_si=True)
        assert "[m]" in ax.get_xlabel()
        assert "[m/s]" in ax.get_ylabel()
        plt.close(fig)

    def test_plot_without_units_uses_plain_labels(self):
        """When units=None, labels are plain variable names."""
        import matplotlib.pyplot as plt

        xy = XYData(
            np.array([0.0, 1.0]),
            np.array([2.0, 3.0]),
            xlabel="x",
            ylabel="y",
        )
        fig, ax = plt.subplots()
        xy.plot(ax=ax, use_si=True)
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        plt.close(fig)

    def test_plot_creates_axes_if_none(self):
        """plot(ax=None) should use plt.gca()."""
        import matplotlib.pyplot as plt

        xy = XYData(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
        plt.figure()
        ret = xy.plot(use_si=False)
        assert ret is not None
        plt.close("all")


# ===================================================================
# MultiXYData
# ===================================================================

class TestMultiXYData:
    """Tests for the MultiXYData container."""

    def test_construction(self):
        x = np.zeros((3, 5))
        y = np.ones((3, 5))
        li = np.array([5, 5, 5])
        mxy = MultiXYData(x, y, li, xlabel="a", ylabel="b", title="T")
        assert mxy.x.shape == (3, 5)
        assert mxy.title == "T"

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            MultiXYData(np.zeros(5), np.zeros(5), np.array([5]))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            MultiXYData(np.zeros((3, 5)), np.zeros((2, 5)), np.array([5, 5]))

    def test_default_title(self):
        mxy = MultiXYData(
            np.zeros((2, 4)), np.ones((2, 4)),
            np.array([4, 4]), xlabel="a", ylabel="b",
        )
        assert mxy.title == "a vs b"

    def test_iter_unpacking(self):
        x = np.zeros((2, 3))
        y = np.ones((2, 3))
        li = np.array([3, 3])
        mxy = MultiXYData(x, y, li)
        x_out, y_out = mxy
        np.testing.assert_array_equal(x_out, x)
        np.testing.assert_array_equal(y_out, y)

    def test_repr(self):
        mxy = MultiXYData(
            np.zeros((4, 7)), np.ones((4, 7)),
            np.full(4, 7), xlabel="xx", ylabel="yy",
        )
        r = repr(mxy)
        assert "MultiXYData" in r
        assert "n_series=4" in r
        assert "n_points=7" in r

    def test_plot_basic(self):
        import matplotlib.pyplot as plt

        x = np.tile(np.linspace(0, 1, 5), (3, 1))
        y = np.random.RandomState(0).randn(3, 5)
        li = np.full(3, 5)
        mxy = MultiXYData(x, y, li, xlabel="x", ylabel="y", title="Multi")
        fig, ax = plt.subplots()
        ret = mxy.plot(ax=ax, use_si=False)
        assert ret is ax
        assert ax.get_title() == "Multi"
        # There should be 3 line objects
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_plot_respects_last_indexes(self):
        """Each series should only plot up to its last_index."""
        import matplotlib.pyplot as plt

        x = np.tile(np.arange(10, dtype=float), (2, 1))
        y = np.ones((2, 10))
        li = np.array([3, 7])  # first series truncated at 3, second at 7
        mxy = MultiXYData(x, y, li)
        fig, ax = plt.subplots()
        mxy.plot(ax=ax, use_si=False)
        assert len(ax.lines[0].get_xdata()) == 3
        assert len(ax.lines[1].get_xdata()) == 7
        plt.close(fig)

    def test_plot_per_series_alpha(self):
        """alpha can be a per-series array."""
        import matplotlib.pyplot as plt

        n = 3
        x = np.tile(np.arange(5, dtype=float), (n, 1))
        y = np.ones((n, 5))
        li = np.full(n, 5)
        mxy = MultiXYData(x, y, li)
        fig, ax = plt.subplots()
        mxy.plot(ax=ax, use_si=False, alpha=[0.2, 0.5, 0.8])
        # Just verify no crash and lines are plotted
        assert len(ax.lines) == n
        plt.close(fig)


# ===================================================================
# _insert_nans_for_gaps
# ===================================================================

class TestInsertNansForGaps:
    """Tests for the gap-breaking utility function."""

    def test_no_gaps(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        xo, yo = _insert_nans_for_gaps(x, y, gap=5.0)
        np.testing.assert_array_equal(xo, x)
        np.testing.assert_array_equal(yo, y)

    def test_gap_inserts_nan(self):
        x = np.array([0.0, 1.0, 10.0, 11.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        xo, yo = _insert_nans_for_gaps(x, y, gap=5.0)
        # Between index 1 and 2 there's a gap of 9 > 5 => NaN inserted
        assert len(xo) == 5
        assert np.isnan(xo[2])
        assert np.isnan(yo[2])

    def test_single_point(self):
        x = np.array([5.0])
        y = np.array([3.0])
        xo, yo = _insert_nans_for_gaps(x, y, gap=1.0)
        np.testing.assert_array_equal(xo, x)

    def test_two_points_with_gap(self):
        x = np.array([0.0, 100.0])
        y = np.array([0.0, 0.0])
        xo, yo = _insert_nans_for_gaps(x, y, gap=1.0)
        assert len(xo) == 3
        assert np.isnan(xo[1])

    def test_all_gaps(self):
        x = np.array([0.0, 10.0, 20.0])
        y = np.array([0.0, 0.0, 0.0])
        xo, yo = _insert_nans_for_gaps(x, y, gap=5.0)
        # Two gaps => 2 NaN insertions, total length = 3 + 2 = 5
        assert len(xo) == 5

    def test_empty_array(self):
        x = np.array([])
        y = np.array([])
        xo, yo = _insert_nans_for_gaps(x, y, gap=1.0)
        assert len(xo) == 0


# ===================================================================
# BacktraceResult
# ===================================================================

class TestBacktraceResult:
    """Tests for BacktraceResult construction and data access."""

    def test_construction(self):
        r = _make_backtrace_result(n_steps=5)
        assert r.ts.shape == (5,)
        assert r.probability.shape == (5,)
        assert r.positions.shape == (5, 3)
        assert r.velocities.shape == (5, 3)

    def test_construction_validates_positions_shape(self):
        ts = np.zeros(5)
        prob = np.zeros(5)
        pos_bad = np.zeros((5, 2))  # should be (5, 3)
        vel = np.zeros((5, 3))
        with pytest.raises(ValueError, match="positions"):
            BacktraceResult(ts, prob, pos_bad, vel)

    def test_construction_validates_velocities_shape(self):
        ts = np.zeros(5)
        prob = np.zeros(5)
        pos = np.zeros((5, 3))
        vel_bad = np.zeros((4, 3))  # wrong N
        with pytest.raises(ValueError, match="velocities"):
            BacktraceResult(ts, prob, pos, vel_bad)

    def test_construction_validates_positions_1d(self):
        ts = np.zeros(5)
        prob = np.zeros(5)
        pos_bad = np.zeros(15)  # 1-D, not (5,3)
        vel = np.zeros((5, 3))
        with pytest.raises(ValueError, match="positions"):
            BacktraceResult(ts, prob, pos_bad, vel)

    def test_iter_unpacking(self):
        r = _make_backtrace_result(n_steps=4)
        ts, prob, pos, vel = r
        np.testing.assert_array_equal(ts, r.ts)
        np.testing.assert_array_equal(prob, r.probability)
        np.testing.assert_array_equal(pos, r.positions)
        np.testing.assert_array_equal(vel, r.velocities)

    def test_repr(self):
        r = _make_backtrace_result(n_steps=7)
        s = repr(r)
        assert "BacktraceResult" in s
        assert "n_steps=7" in s

    def test_pair_tx(self):
        """pair('t', 'x') returns an XYData with ts as x and x-positions as y."""
        r = _make_backtrace_result(n_steps=5)
        xy = r.pair("t", "x")
        assert isinstance(xy, XYData)
        np.testing.assert_array_equal(xy.x, r.ts)
        np.testing.assert_array_equal(xy.y, r.positions[:, 0])

    def test_pair_yz(self):
        r = _make_backtrace_result(n_steps=5)
        xy = r.pair("y", "z")
        np.testing.assert_array_equal(xy.x, r.positions[:, 1])
        np.testing.assert_array_equal(xy.y, r.positions[:, 2])

    def test_pair_xvz(self):
        r = _make_backtrace_result(n_steps=5)
        xy = r.pair("x", "vz")
        np.testing.assert_array_equal(xy.x, r.positions[:, 0])
        np.testing.assert_array_equal(xy.y, r.velocities[:, 2])

    def test_pair_invalid_key(self):
        r = _make_backtrace_result()
        with pytest.raises(KeyError, match="Allowed keys"):
            r.pair("foo", "x")
        with pytest.raises(KeyError, match="Allowed keys"):
            r.pair("x", "bar")

    def test_pair_with_units(self):
        unit = _make_unit()
        r = _make_backtrace_result(n_steps=5, unit=unit)
        xy = r.pair("t", "x")
        assert xy.units is not None
        assert xy.units[0].unit == "s"
        assert xy.units[1].unit == "m"

    def test_pair_velocity_units(self):
        unit = _make_unit()
        r = _make_backtrace_result(n_steps=5, unit=unit)
        xy = r.pair("vx", "vy")
        assert xy.units[0].unit == "m/s"
        assert xy.units[1].unit == "m/s"

    def test_pair_without_unit(self):
        r = _make_backtrace_result(n_steps=5, unit=None)
        xy = r.pair("t", "x")
        assert xy.units is None

    def test_getattr_shorthand_tx(self):
        r = _make_backtrace_result(n_steps=5)
        xy = r.tx
        assert isinstance(xy, XYData)
        np.testing.assert_array_equal(xy.x, r.ts)
        np.testing.assert_array_equal(xy.y, r.positions[:, 0])

    def test_getattr_shorthand_tvx(self):
        r = _make_backtrace_result(n_steps=5)
        xy = r.tvx
        np.testing.assert_array_equal(xy.x, r.ts)
        np.testing.assert_array_equal(xy.y, r.velocities[:, 0])

    def test_getattr_shorthand_xvy(self):
        r = _make_backtrace_result(n_steps=5)
        xy = r.xvy
        np.testing.assert_array_equal(xy.x, r.positions[:, 0])
        np.testing.assert_array_equal(xy.y, r.velocities[:, 1])

    def test_getattr_shorthand_yz(self):
        r = _make_backtrace_result(n_steps=5)
        xy = r.yz
        np.testing.assert_array_equal(xy.x, r.positions[:, 1])
        np.testing.assert_array_equal(xy.y, r.positions[:, 2])

    def test_getattr_invalid_raises(self):
        r = _make_backtrace_result()
        with pytest.raises(AttributeError, match="no attribute"):
            _ = r.foobar

    def test_pair_labels_and_title(self):
        r = _make_backtrace_result()
        xy = r.pair("x", "vz")
        assert xy.xlabel == "x"
        assert xy.ylabel == "vz"
        assert xy.title == "x vs vz"


# ===================================================================
# MultiBacktraceResult
# ===================================================================

class TestMultiBacktraceResult:
    """Tests for MultiBacktraceResult construction, sampling, and pairing."""

    def test_construction(self):
        r = _make_multi_backtrace_result(n_traj=4, n_steps=8)
        assert r.ts_list.shape == (4, 8)
        assert r.probabilities.shape == (4,)
        assert r.positions_list.shape == (4, 8, 3)
        assert r.velocities_list.shape == (4, 8, 3)
        assert r.last_indexes.shape == (4,)

    def test_construction_validates_ts_list_ndim(self):
        with pytest.raises(ValueError, match="2-D"):
            MultiBacktraceResult(
                np.zeros(10),  # 1-D, should be 2-D
                np.zeros(2),
                np.zeros((2, 5, 3)),
                np.zeros((2, 5, 3)),
                np.array([5, 5]),
            )

    def test_construction_validates_probabilities_shape(self):
        with pytest.raises(ValueError, match="probabilities"):
            MultiBacktraceResult(
                np.zeros((3, 5)),
                np.zeros(4),  # wrong length, should be 3
                np.zeros((3, 5, 3)),
                np.zeros((3, 5, 3)),
                np.array([5, 5, 5]),
            )

    def test_construction_validates_positions_list_shape(self):
        with pytest.raises(ValueError, match="positions_list"):
            MultiBacktraceResult(
                np.zeros((3, 5)),
                np.zeros(3),
                np.zeros((3, 5, 2)),  # last dim should be 3
                np.zeros((3, 5, 3)),
                np.array([5, 5, 5]),
            )

    def test_construction_validates_velocities_list_shape(self):
        with pytest.raises(ValueError, match="velocities_list"):
            MultiBacktraceResult(
                np.zeros((3, 5)),
                np.zeros(3),
                np.zeros((3, 5, 3)),
                np.zeros((2, 5, 3)),  # wrong N_traj
                np.array([5, 5, 5]),
            )

    def test_iter_unpacking(self):
        r = _make_multi_backtrace_result(n_traj=2, n_steps=6)
        ts, probs, pos, vel, li = r
        np.testing.assert_array_equal(ts, r.ts_list)
        np.testing.assert_array_equal(probs, r.probabilities)
        np.testing.assert_array_equal(pos, r.positions_list)
        np.testing.assert_array_equal(vel, r.velocities_list)
        np.testing.assert_array_equal(li, r.last_indexes)

    def test_repr(self):
        r = _make_multi_backtrace_result(n_traj=3, n_steps=7)
        s = repr(r)
        assert "MultiBacktraceResult" in s
        assert "n_traj=3" in s
        assert "n_steps=7" in s

    # --- sample() ---

    def test_sample_int(self):
        r = _make_multi_backtrace_result(n_traj=10, n_steps=5)
        sub = r.sample(3, random_state=42)
        assert isinstance(sub, MultiBacktraceResult)
        assert sub.ts_list.shape[0] == 3
        assert sub.positions_list.shape[0] == 3
        assert sub.velocities_list.shape[0] == 3
        assert sub.probabilities.shape == (3,)
        assert sub.last_indexes.shape == (3,)

    def test_sample_int_reproducible(self):
        r = _make_multi_backtrace_result(n_traj=10, n_steps=5)
        sub1 = r.sample(3, random_state=42)
        sub2 = r.sample(3, random_state=42)
        np.testing.assert_array_equal(sub1.ts_list, sub2.ts_list)

    def test_sample_list(self):
        r = _make_multi_backtrace_result(n_traj=5, n_steps=4)
        sub = r.sample([0, 2, 4])
        assert sub.ts_list.shape[0] == 3
        np.testing.assert_array_equal(sub.ts_list[0], r.ts_list[0])
        np.testing.assert_array_equal(sub.ts_list[1], r.ts_list[2])
        np.testing.assert_array_equal(sub.ts_list[2], r.ts_list[4])

    def test_sample_slice(self):
        r = _make_multi_backtrace_result(n_traj=6, n_steps=4)
        sub = r.sample(slice(1, 4))
        assert sub.ts_list.shape[0] == 3

    def test_sample_range(self):
        r = _make_multi_backtrace_result(n_traj=6, n_steps=4)
        sub = r.sample(range(0, 4, 2))
        assert sub.ts_list.shape[0] == 2

    def test_sample_invalid_k(self):
        r = _make_multi_backtrace_result(n_traj=3, n_steps=4)
        with pytest.raises(ValueError, match="k must satisfy"):
            r.sample(10)

    def test_sample_invalid_index(self):
        r = _make_multi_backtrace_result(n_traj=3, n_steps=4)
        with pytest.raises(IndexError, match="out of range"):
            r.sample([0, 10])

    def test_sample_invalid_type(self):
        r = _make_multi_backtrace_result(n_traj=3, n_steps=4)
        # A non-iterable, non-int, non-slice, non-range type should raise
        with pytest.raises(TypeError, match="must be int"):
            r.sample(3.14)

    def test_sample_preserves_unit(self):
        unit = _make_unit()
        r = _make_multi_backtrace_result(n_traj=5, n_steps=4, unit=unit)
        sub = r.sample(2, random_state=0)
        assert sub.unit is unit

    # --- pair() ---

    def test_pair_tx(self):
        r = _make_multi_backtrace_result(n_traj=3, n_steps=5)
        mxy = r.pair("t", "x")
        assert isinstance(mxy, MultiXYData)
        np.testing.assert_array_equal(mxy.x, r.ts_list)
        np.testing.assert_array_equal(mxy.y, r.positions_list[:, :, 0])

    def test_pair_yvz(self):
        r = _make_multi_backtrace_result(n_traj=2, n_steps=4)
        mxy = r.pair("y", "vz")
        np.testing.assert_array_equal(mxy.x, r.positions_list[:, :, 1])
        np.testing.assert_array_equal(mxy.y, r.velocities_list[:, :, 2])

    def test_pair_invalid_key(self):
        r = _make_multi_backtrace_result()
        with pytest.raises(KeyError, match="Allowed keys"):
            r.pair("bad", "x")

    def test_pair_labels_without_unit(self):
        r = _make_multi_backtrace_result(n_traj=2, n_steps=4, unit=None)
        mxy = r.pair("t", "x")
        assert mxy.xlabel == "t"
        assert mxy.ylabel == "x"

    def test_pair_labels_with_unit(self):
        unit = _make_unit()
        r = _make_multi_backtrace_result(n_traj=2, n_steps=4, unit=unit)
        mxy = r.pair("t", "x")
        assert "[m]" in mxy.xlabel or "[m]" in mxy.ylabel
        assert "vs" in mxy.title

    # --- __getattr__ ---

    def test_getattr_shorthand_tx(self):
        r = _make_multi_backtrace_result(n_traj=2, n_steps=4)
        mxy = r.tx
        assert isinstance(mxy, MultiXYData)

    def test_getattr_shorthand_xvy(self):
        r = _make_multi_backtrace_result(n_traj=2, n_steps=4)
        mxy = r.xvy
        np.testing.assert_array_equal(mxy.x, r.positions_list[:, :, 0])
        np.testing.assert_array_equal(mxy.y, r.velocities_list[:, :, 1])

    def test_getattr_invalid(self):
        r = _make_multi_backtrace_result()
        with pytest.raises(AttributeError, match="no attribute"):
            _ = r.nonexistent


# ===================================================================
# BacktraceWrapper (mocked solver)
# ===================================================================

class TestBacktraceWrapper:
    """Tests for BacktraceWrapper using mocked external solver modules."""

    def _make_wrapper(self, dt=0.01):
        """Create a BacktraceWrapper with mocked inp and unit."""
        from emout.core.backtrace.solver_wrapper import BacktraceWrapper

        inp = SimpleNamespace(dt=dt)
        unit = _make_unit()
        return BacktraceWrapper(
            directory="/fake/dir",
            inp=inp,
            unit=unit,
        )

    def test_get_backtrace_delegates_and_returns_result(self, monkeypatch):
        """get_backtrace should call the backend and wrap in BacktraceResult."""
        wrapper = self._make_wrapper(dt=0.01)

        n_steps = 20
        fake_ts = np.arange(n_steps, dtype=float)
        fake_prob = np.ones(n_steps)
        fake_pos = np.random.randn(n_steps, 3)
        fake_vel = np.random.randn(n_steps, 3)

        mock_particle_cls = MagicMock()

        # Mock run_backend to return the fake data
        def fake_run_backend(func, **kwargs):
            return (fake_ts, fake_prob, fake_pos, fake_vel)

        monkeypatch.setattr(
            "emout.core.backtrace.solver_wrapper.run_backend",
            fake_run_backend,
        )

        # Mock the vdsolverf imports inside get_backtrace
        import sys
        mock_vdsolverf_core = MagicMock()
        mock_vdsolverf_emses = MagicMock()
        sys.modules["vdsolverf"] = MagicMock()
        sys.modules["vdsolverf.core"] = mock_vdsolverf_core
        sys.modules["vdsolverf.emses"] = mock_vdsolverf_emses

        try:
            result = wrapper.get_backtrace(
                position=np.array([1.0, 2.0, 3.0]),
                velocity=np.array([4.0, 5.0, 6.0]),
                ispec=0,
                istep=-1,
                max_step=100,
            )
            assert isinstance(result, BacktraceResult)
            np.testing.assert_array_equal(result.ts, fake_ts)
            np.testing.assert_array_equal(result.positions, fake_pos)
            assert result.unit is wrapper.unit
        finally:
            sys.modules.pop("vdsolverf", None)
            sys.modules.pop("vdsolverf.core", None)
            sys.modules.pop("vdsolverf.emses", None)

    def test_get_backtrace_uses_inp_dt_when_dt_is_none(self, monkeypatch):
        """When dt=None, should fall back to self.inp.dt."""
        wrapper = self._make_wrapper(dt=0.05)

        n_steps = 5
        captured_kwargs = {}

        def fake_run_backend(func, **kwargs):
            captured_kwargs.update(kwargs)
            return (
                np.arange(n_steps, dtype=float),
                np.ones(n_steps),
                np.zeros((n_steps, 3)),
                np.zeros((n_steps, 3)),
            )

        monkeypatch.setattr(
            "emout.core.backtrace.solver_wrapper.run_backend",
            fake_run_backend,
        )

        import sys
        sys.modules["vdsolverf"] = MagicMock()
        sys.modules["vdsolverf.core"] = MagicMock()
        sys.modules["vdsolverf.emses"] = MagicMock()

        try:
            wrapper.get_backtrace(
                position=np.array([0.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.0, 0.0]),
                dt=None,
            )
            assert captured_kwargs["dt"] == 0.05
        finally:
            sys.modules.pop("vdsolverf", None)
            sys.modules.pop("vdsolverf.core", None)
            sys.modules.pop("vdsolverf.emses", None)

    def test_get_backtraces_delegates_and_returns_multi_result(self, monkeypatch):
        """get_backtraces should call the backend and return MultiBacktraceResult."""
        wrapper = self._make_wrapper(dt=0.01)

        n_traj, n_steps = 3, 10
        fake_ts_list = np.zeros((n_traj, n_steps))
        fake_probs = np.ones(n_traj)
        fake_pos_list = np.zeros((n_traj, n_steps, 3))
        fake_vel_list = np.zeros((n_traj, n_steps, 3))
        fake_last = np.full(n_traj, n_steps)

        def fake_run_backend(func, *args, **kwargs):
            return (fake_ts_list, fake_probs, fake_pos_list, fake_vel_list, fake_last)

        monkeypatch.setattr(
            "emout.core.backtrace.solver_wrapper.run_backend",
            fake_run_backend,
        )

        import sys
        sys.modules["vdsolverf"] = MagicMock()
        sys.modules["vdsolverf.core"] = MagicMock()
        sys.modules["vdsolverf.emses"] = MagicMock()

        try:
            positions = np.zeros((n_traj, 3))
            velocities = np.ones((n_traj, 3))
            result = wrapper.get_backtraces(positions, velocities)
            assert isinstance(result, MultiBacktraceResult)
            assert result.ts_list.shape == (n_traj, n_steps)
            assert result.unit is wrapper.unit
        finally:
            sys.modules.pop("vdsolverf", None)
            sys.modules.pop("vdsolverf.core", None)
            sys.modules.pop("vdsolverf.emses", None)

    def test_get_backtraces_validates_shape_mismatch(self, monkeypatch):
        """get_backtraces should raise ValueError on mismatched shapes."""
        wrapper = self._make_wrapper()

        import sys
        sys.modules["vdsolverf"] = MagicMock()
        sys.modules["vdsolverf.core"] = MagicMock()
        sys.modules["vdsolverf.emses"] = MagicMock()

        try:
            with pytest.raises(ValueError, match="same shape"):
                wrapper.get_backtraces(
                    np.zeros((3, 3)),
                    np.zeros((4, 3)),
                )
        finally:
            sys.modules.pop("vdsolverf", None)
            sys.modules.pop("vdsolverf.core", None)
            sys.modules.pop("vdsolverf.emses", None)

    def test_get_backtraces_from_particles(self, monkeypatch):
        """get_backtraces_from_particles should accept pre-built particles."""
        wrapper = self._make_wrapper(dt=0.01)

        n_traj, n_steps = 2, 5
        fake_ts_list = np.zeros((n_traj, n_steps))
        fake_probs = np.ones(n_traj)
        fake_pos_list = np.zeros((n_traj, n_steps, 3))
        fake_vel_list = np.zeros((n_traj, n_steps, 3))
        fake_last = np.full(n_traj, n_steps)

        def fake_run_backend(func, *args, **kwargs):
            return (fake_ts_list, fake_probs, fake_pos_list, fake_vel_list, fake_last)

        monkeypatch.setattr(
            "emout.core.backtrace.solver_wrapper.run_backend",
            fake_run_backend,
        )

        import sys
        sys.modules["vdsolverf"] = MagicMock()
        sys.modules["vdsolverf.core"] = MagicMock()
        sys.modules["vdsolverf.emses"] = MagicMock()

        try:
            fake_particles = [MagicMock(), MagicMock()]
            result = wrapper.get_backtraces_from_particles(fake_particles)
            assert isinstance(result, MultiBacktraceResult)
            assert result.ts_list.shape[0] == n_traj
        finally:
            sys.modules.pop("vdsolverf", None)
            sys.modules.pop("vdsolverf.core", None)
            sys.modules.pop("vdsolverf.emses", None)


# ===================================================================
# Integration: BacktraceResult -> XYData -> plot
# ===================================================================

class TestBacktraceResultPlotIntegration:
    """End-to-end test: BacktraceResult -> pair() -> XYData.plot()."""

    def test_pair_and_plot(self):
        import matplotlib.pyplot as plt

        r = _make_backtrace_result(n_steps=20)
        xy = r.pair("t", "x")
        fig, ax = plt.subplots()
        ret = xy.plot(ax=ax, use_si=False)
        assert ret is ax
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_shorthand_and_plot(self):
        import matplotlib.pyplot as plt

        r = _make_backtrace_result(n_steps=15)
        fig, ax = plt.subplots()
        ret = r.tvx.plot(ax=ax, use_si=False)
        assert ret is ax
        plt.close(fig)

    def test_pair_with_units_and_plot(self):
        import matplotlib.pyplot as plt

        unit = _make_unit()
        r = _make_backtrace_result(n_steps=10, unit=unit)
        xy = r.pair("x", "vy")
        fig, ax = plt.subplots()
        xy.plot(ax=ax, use_si=True)
        assert "[m]" in ax.get_xlabel()
        assert "[m/s]" in ax.get_ylabel()
        plt.close(fig)

    def test_plot_with_gap(self):
        """Verify that gap parameter causes NaN insertion in the plot data."""
        import matplotlib.pyplot as plt

        ts = np.array([0.0, 1.0, 2.0, 100.0, 101.0])
        prob = np.zeros(5)
        pos = np.column_stack([ts, np.zeros(5), np.zeros(5)])
        vel = np.zeros((5, 3))
        r = BacktraceResult(ts, prob, pos, vel)
        xy = r.pair("t", "x")
        fig, ax = plt.subplots()
        xy.plot(ax=ax, use_si=False, gap=50.0)
        # The plotted data should have a NaN break
        line_data = ax.lines[0].get_xdata()
        assert len(line_data) > 5  # NaN was inserted
        plt.close(fig)


class TestMultiBacktraceResultPlotIntegration:
    """End-to-end: MultiBacktraceResult -> pair() -> MultiXYData.plot()."""

    def test_pair_and_plot(self):
        import matplotlib.pyplot as plt

        r = _make_multi_backtrace_result(n_traj=4, n_steps=10)
        mxy = r.pair("t", "y")
        fig, ax = plt.subplots()
        ret = mxy.plot(ax=ax, use_si=False)
        assert ret is ax
        assert len(ax.lines) == 4
        plt.close(fig)

    def test_shorthand_and_plot(self):
        import matplotlib.pyplot as plt

        r = _make_multi_backtrace_result(n_traj=2, n_steps=8)
        fig, ax = plt.subplots()
        r.xvy.plot(ax=ax, use_si=False)
        assert len(ax.lines) == 2
        plt.close(fig)
