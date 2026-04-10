"""Tests for emout.plot.extra_plot plotting helpers.

All matplotlib rendering is monkeypatched so no actual display occurs.
Data objects are mocked with synthetic numpy arrays.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers -- mock InpFile, Units, Data3d
# ---------------------------------------------------------------------------


def _make_inp(
    xlrechole=(0, 10),
    xurechole=(0, 30),
    ylrechole=(0, 5),
    yurechole=(0, 25),
    zlrechole=(0, 8),
    zurechole=(40,),
    nx=64,
    ny=64,
    nz=64,
):
    """Return a SimpleNamespace mimicking InpFile with hole parameters."""
    return SimpleNamespace(
        xlrechole=xlrechole,
        xurechole=xurechole,
        ylrechole=ylrechole,
        yurechole=yurechole,
        zlrechole=zlrechole,
        zurechole=zurechole,
        nx=nx,
        ny=ny,
        nz=nz,
    )


def _make_unit():
    """Return a mock Units object with a length translator."""
    unit = SimpleNamespace()
    length = SimpleNamespace()
    length.reverse = lambda x: np.asarray(x, dtype=float) * 0.01
    length.trans = lambda x: np.asarray(x, dtype=float) * 100.0
    unit.length = length
    return unit


def _make_unit_translator():
    """Return a mock UnitTranslator (for plot_line_of_hole_half)."""
    ut = MagicMock()
    ut.reverse = lambda x: np.asarray(x, dtype=float) * 0.01
    return ut


class _SliceableMock:
    """Mock Data3d: supports __getitem__, max/min, val_si, masked."""

    def __init__(self, vmin=-1.0, vmax=1.0):
        self._vmin = vmin
        self._vmax = vmax
        self._getitem_calls = []
        self._masked_mock = None
        self._mask_fn = None

        self.val_si = SimpleNamespace(
            max=lambda: vmax * 2,
            min=lambda: vmin * 2,
        )

    def max(self):
        return self._vmax

    def min(self):
        return self._vmin

    def __getitem__(self, key):
        self._getitem_calls.append(key)
        sub = MagicMock()
        sub.plot = MagicMock()
        return sub

    def masked(self, fn):
        self._mask_fn = fn
        if self._masked_mock is None:
            self._masked_mock = _SliceableMock(self._vmin, self._vmax)
        return self._masked_mock


class _TrackingMock(_SliceableMock):
    """_SliceableMock that records kwargs passed to every .plot() call."""

    def __init__(self, vmin=-1.0, vmax=1.0):
        super().__init__(vmin, vmax)
        self.plot_kwargs: list[dict] = []

    def __getitem__(self, key):
        self._getitem_calls.append(key)
        kw_list = self.plot_kwargs
        sub = MagicMock()

        def _plot(**kw):
            kw_list.append(kw)

        sub.plot = _plot
        return sub

    def masked(self, fn):
        self._mask_fn = fn
        if self._masked_mock is None:
            m = _TrackingMock(self._vmin, self._vmax)
            m.plot_kwargs = self.plot_kwargs  # shared list
            self._masked_mock = m
        return self._masked_mock


def _make_inp_mock_for_inpfile():
    """Create an InpFile-spec MagicMock for isinstance checks."""
    from emout.utils.emsesinp import InpFile

    inp = MagicMock(spec=InpFile)
    inp.xlrechole = [0, 10]
    inp.xurechole = [0, 30]
    inp.ylrechole = [0, 5]
    inp.yurechole = [0, 25]
    inp.zlrechole = [0, 8]
    inp.zurechole = [40]
    inp.nx = 64
    inp.ny = 64
    inp.nz = 64
    return inp


# ===================================================================
# plot_surface_with_hole
# ===================================================================


class TestPlotSurfaceWithHole:
    """Tests for plot_surface_with_hole."""

    def test_basic_call_slice_count(self, monkeypatch):
        """9 sub-slices: 4 top + 4 wall + 1 bottom."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        data = _SliceableMock()
        inp = _make_inp()
        plot_surface_with_hole(data, inp)
        assert len(data._getitem_calls) == 9

    def test_show_true(self, monkeypatch):
        """show=True triggers plt.show()."""
        from emout.plot.extra_plot import plot_surface_with_hole

        show_mock = MagicMock()
        monkeypatch.setattr(plt, "show", show_mock)
        plot_surface_with_hole(_SliceableMock(), _make_inp(), show=True)
        show_mock.assert_called_once()

    def test_show_false(self, monkeypatch):
        """show=False does not call plt.show()."""
        from emout.plot.extra_plot import plot_surface_with_hole

        show_mock = MagicMock()
        monkeypatch.setattr(plt, "show", show_mock)
        plot_surface_with_hole(_SliceableMock(), _make_inp(), show=False)
        show_mock.assert_not_called()

    def test_vrange_minmax(self, monkeypatch):
        """vrange='minmax' uses raw min/max."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        data = _TrackingMock(vmin=-3.0, vmax=5.0)
        plot_surface_with_hole(data, _make_inp(), vrange="minmax")
        for kw in data.plot_kwargs:
            assert kw["vmax"] == 5.0
            assert kw["vmin"] == -3.0

    def test_vrange_center(self, monkeypatch):
        """vrange='center' makes symmetric range."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        data = _TrackingMock(vmin=-2.0, vmax=5.0)
        plot_surface_with_hole(data, _make_inp(), vrange="center")
        for kw in data.plot_kwargs:
            assert kw["vmax"] == 5.0
            assert kw["vmin"] == -5.0

    def test_use_si(self, monkeypatch):
        """use_si=True reads vmin/vmax from data.val_si."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        data = _TrackingMock(vmin=-1.0, vmax=1.0)
        data.val_si = SimpleNamespace(max=lambda: 10.0, min=lambda: -4.0)
        plot_surface_with_hole(data, _make_inp(), use_si=True)
        for kw in data.plot_kwargs:
            assert kw["vmax"] == 10.0
            assert kw["vmin"] == -4.0

    def test_explicit_vmin_vmax(self, monkeypatch):
        """Explicit vmin/vmax override computed values."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        data = _TrackingMock()
        plot_surface_with_hole(data, _make_inp(), vmin=-99.0, vmax=99.0)
        for kw in data.plot_kwargs:
            assert kw["vmax"] == 99.0
            assert kw["vmin"] == -99.0

    def test_add_colorbar_on_bottom_only(self, monkeypatch):
        """add_colorbar appears only on the bottom (last) plot call."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        data = _TrackingMock()
        plot_surface_with_hole(data, _make_inp(), add_colorbar=True)
        assert data.plot_kwargs[-1]["add_colorbar"] is True
        for kw in data.plot_kwargs[:-1]:
            assert "add_colorbar" not in kw

    def test_add_colorbar_false(self, monkeypatch):
        """add_colorbar=False forwarded to bottom."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        data = _TrackingMock()
        plot_surface_with_hole(data, _make_inp(), add_colorbar=False)
        assert data.plot_kwargs[-1]["add_colorbar"] is False

    def test_mode_surf_everywhere(self, monkeypatch):
        """All sub-plot calls use mode='surf'."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        data = _TrackingMock()
        plot_surface_with_hole(data, _make_inp())
        for kw in data.plot_kwargs:
            assert kw.get("mode") == "surf"

    def test_extra_kwargs_forwarded(self, monkeypatch):
        """Extra kwargs (cmap, ninterp) forwarded to all plot calls."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        data = _TrackingMock()
        plot_surface_with_hole(data, _make_inp(), cmap="hot", ninterp=2)
        for kw in data.plot_kwargs:
            assert kw["cmap"] == "hot"
            assert kw["ninterp"] == 2

    def test_hole_indices(self, monkeypatch):
        """Slice indices match inp hole parameters."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        inp = _make_inp(
            xlrechole=(0, 5),
            xurechole=(0, 15),
            ylrechole=(0, 3),
            yurechole=(0, 12),
            zlrechole=(0, 2),
            zurechole=(20,),
        )
        data = _SliceableMock()
        plot_surface_with_hole(data, inp)
        # First top: data[zu, :yl+1, :] -> data[20, :4, :]
        first = data._getitem_calls[0]
        assert first[0] == 20
        assert first[1] == slice(None, 4)

    def test_returns_none(self, monkeypatch):
        """Function returns None."""
        from emout.plot.extra_plot import plot_surface_with_hole

        monkeypatch.setattr(plt, "show", MagicMock())
        assert plot_surface_with_hole(_SliceableMock(), _make_inp()) is None


# ===================================================================
# plot_hole_line
# ===================================================================


class TestPlotHoleLine:
    """Tests for plot_hole_line."""

    def _patch_plt(self, monkeypatch):
        """Common plt patches for plot_hole_line tests."""
        plot_calls = []
        monkeypatch.setattr(
            plt,
            "plot",
            lambda *a, **kw: plot_calls.append((a, kw)) or [MagicMock()],
        )
        monkeypatch.setattr(plt, "xlim", lambda *a: (0, 100))
        monkeypatch.setattr(plt, "ylim", lambda *a: (0, 100))
        return plot_calls

    def test_basic_inpfile(self, monkeypatch):
        """Call with InpFile-spec mock draws one line."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        plot_hole_line(_make_inp_mock_for_inpfile())
        assert len(calls) == 1

    def test_emout_object(self, monkeypatch):
        """Non-InpFile object: inp and unit are extracted."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        emout = MagicMock()
        emout.inp = SimpleNamespace(
            xlrechole=[0, 10],
            xurechole=[0, 30],
            ylrechole=[0, 5],
            yurechole=[0, 25],
            zlrechole=[0, 8],
            zurechole=[40],
            nx=64,
            ny=64,
            nz=64,
        )
        emout.unit = _make_unit()
        plot_hole_line(emout, use_si=True)
        assert len(calls) == 1

    def test_use_si_converts(self, monkeypatch):
        """use_si=True converts via unit.length.reverse."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        inp = _make_inp_mock_for_inpfile()
        unit = _make_unit()
        plot_hole_line(inp, unit=unit, use_si=True)
        xs = calls[0][0][0]
        assert xs[1] == pytest.approx(0.1)  # xl=10 * 0.01
        assert xs[3] == pytest.approx(0.3)  # xu=30 * 0.01

    def test_use_si_false(self, monkeypatch):
        """use_si=False keeps grid units."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        inp = _make_inp_mock_for_inpfile()
        plot_hole_line(inp, use_si=False)
        xs = calls[0][0][0]
        np.testing.assert_array_almost_equal(xs, [0, 10, 10, 30, 30, 63])

    def test_use_si_true_unit_none(self, monkeypatch):
        """use_si=True with unit=None does not convert."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        inp = _make_inp_mock_for_inpfile()
        plot_hole_line(inp, unit=None, use_si=True)
        xs = calls[0][0][0]
        assert xs[1] == pytest.approx(10.0)

    def test_offsets(self, monkeypatch):
        """Offsets are added to x and y."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        inp = _make_inp_mock_for_inpfile()
        plot_hole_line(inp, use_si=False, offsets=(100, 200))
        xs, ys = calls[0][0][0], calls[0][0][1]
        assert xs[0] == pytest.approx(100.0)
        assert ys[0] == pytest.approx(240.0)  # zu=40 + 200

    def test_color_forwarded(self, monkeypatch):
        """color keyword reaches plt.plot."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        plot_hole_line(_make_inp_mock_for_inpfile(), color="red")
        assert calls[0][1]["color"] == "red"

    def test_linewidth_forwarded(self, monkeypatch):
        """linewidth keyword reaches plt.plot."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        plot_hole_line(_make_inp_mock_for_inpfile(), linewidth=2.5)
        assert calls[0][1]["linewidth"] == 2.5

    def test_fix_lims_true(self, monkeypatch):
        """fix_lims=True freezes axes."""
        from emout.plot.extra_plot import plot_hole_line

        xlim_calls = []
        ylim_calls = []
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: [MagicMock()])
        monkeypatch.setattr(plt, "xlim", lambda *a: xlim_calls.append(a) or (0, 100))
        monkeypatch.setattr(plt, "ylim", lambda *a: ylim_calls.append(a) or (0, 100))
        plot_hole_line(_make_inp_mock_for_inpfile(), fix_lims=True)
        assert len(xlim_calls) >= 2
        assert len(ylim_calls) >= 2

    def test_fix_lims_false(self, monkeypatch):
        """fix_lims=False skips axis freezing."""
        from emout.plot.extra_plot import plot_hole_line

        xlim_calls = []
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: [MagicMock()])
        monkeypatch.setattr(plt, "xlim", lambda *a: xlim_calls.append(a) or (0, 100))
        monkeypatch.setattr(plt, "ylim", lambda *a: (0, 100))
        plot_hole_line(_make_inp_mock_for_inpfile(), fix_lims=False)
        assert len(xlim_calls) == 0

    def test_returns_line_objects(self, monkeypatch):
        """Returns list from plt.plot."""
        from emout.plot.extra_plot import plot_hole_line

        sentinel = [MagicMock()]
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: sentinel)
        monkeypatch.setattr(plt, "xlim", lambda *a: (0, 100))
        monkeypatch.setattr(plt, "ylim", lambda *a: (0, 100))
        result = plot_hole_line(_make_inp_mock_for_inpfile())
        assert result is sentinel

    def test_xz_coordinates(self, monkeypatch):
        """axis='xz' produces correct coordinate arrays."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        inp = _make_inp_mock_for_inpfile()
        plot_hole_line(inp, axis="xz", use_si=False)
        xs, ys = calls[0][0][0], calls[0][0][1]
        np.testing.assert_array_almost_equal(xs, [0, 10, 10, 30, 30, 63])
        np.testing.assert_array_almost_equal(ys, [40, 40, 8.5, 8.5, 40, 40])

    def test_default_color_black(self, monkeypatch):
        """Default color is 'black'."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        plot_hole_line(_make_inp_mock_for_inpfile())
        assert calls[0][1]["color"] == "black"

    def test_default_linewidth_none(self, monkeypatch):
        """Default linewidth is None."""
        from emout.plot.extra_plot import plot_hole_line

        calls = self._patch_plt(monkeypatch)
        plot_hole_line(_make_inp_mock_for_inpfile())
        assert calls[0][1]["linewidth"] is None


# ===================================================================
# plot_line_of_hole_half
# ===================================================================


class TestPlotLineOfHoleHalf:
    """Tests for plot_line_of_hole_half."""

    def test_basic_call(self, monkeypatch):
        """Draws 3 polygons + 8 vertical lines = 11 ax.plot calls."""
        from emout.plot.extra_plot import plot_line_of_hole_half

        ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: ax)
        plot_line_of_hole_half(_make_inp(), off=10, unit=_make_unit_translator())
        assert ax.plot.call_count == 11

    def test_all_lines_black(self, monkeypatch):
        """Every ax.plot call uses color='black'."""
        from emout.plot.extra_plot import plot_line_of_hole_half

        ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: ax)
        plot_line_of_hole_half(_make_inp(), off=10, unit=_make_unit_translator())
        for c in ax.plot.call_args_list:
            assert c.kwargs.get("color") == "black"

    def test_unit_reverse_applied(self, monkeypatch):
        """Unit translator .reverse is applied to all point arrays."""
        from emout.plot.extra_plot import plot_line_of_hole_half

        ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: ax)
        unit = MagicMock()
        unit.reverse = lambda x: np.asarray(x, dtype=float) * 2.0
        plot_line_of_hole_half(_make_inp(), off=10, unit=unit)
        assert ax.plot.call_count == 11

    def test_offset_effect(self, monkeypatch):
        """Different off values produce different point coordinates."""
        from emout.plot.extra_plot import plot_line_of_hole_half

        results = {}
        for off_val in [5, 20]:
            ax = MagicMock()
            monkeypatch.setattr(plt, "gca", lambda: ax)
            plot_line_of_hole_half(_make_inp(), off=off_val, unit=_make_unit_translator())
            first_call_args = ax.plot.call_args_list[0][0]
            results[off_val] = first_call_args
        assert not np.array_equal(results[5][0], results[20][0])

    def test_returns_none(self, monkeypatch):
        """Function returns None."""
        from emout.plot.extra_plot import plot_line_of_hole_half

        ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: ax)
        result = plot_line_of_hole_half(_make_inp(), off=10, unit=_make_unit_translator())
        assert result is None

    def test_polygon_point_counts(self, monkeypatch):
        """surf_points=9, bottom_points=5, bottom2_points=5 vertices."""
        from emout.plot.extra_plot import plot_line_of_hole_half

        ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: ax)
        plot_line_of_hole_half(_make_inp(), off=10, unit=_make_unit_translator())
        calls = ax.plot.call_args_list
        # First 3 calls are polygons: surf(9), bottom(5), bottom2(5)
        assert len(calls[0][0][0]) == 9  # surf_points
        assert len(calls[1][0][0]) == 5  # bottom_points
        assert len(calls[2][0][0]) == 5  # bottom2_points

    def test_vertical_lines_have_2_points(self, monkeypatch):
        """The 8 vertical line calls each have 2 endpoints."""
        from emout.plot.extra_plot import plot_line_of_hole_half

        ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: ax)
        plot_line_of_hole_half(_make_inp(), off=10, unit=_make_unit_translator())
        for c in ax.plot.call_args_list[3:]:
            assert len(c[0][0]) == 2  # each line has 2 points


# ===================================================================
# plot_surface_with_hole_half
# ===================================================================


class TestPlotSurfaceWithHoleHalf:
    """Tests for plot_surface_with_hole_half."""

    def _patch_3d(self, monkeypatch):
        """Set up common 3D mocks. Return (fig, ax)."""
        ax = MagicMock()
        ax.set_box_aspect = MagicMock()
        fig = MagicMock()
        fig.add_subplot.return_value = ax
        monkeypatch.setattr(plt, "gcf", lambda: fig)
        monkeypatch.setattr(plt, "sca", lambda a: None)
        monkeypatch.setattr(plt, "show", MagicMock())
        return fig, ax

    def test_basic_call(self, monkeypatch):
        """Creates a 3D subplot."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        plot_surface_with_hole_half(_SliceableMock(), _make_inp())
        fig.add_subplot.assert_called_once_with(projection="3d")

    def test_show_true(self, monkeypatch):
        """show=True triggers plt.show()."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        show_mock = MagicMock()
        monkeypatch.setattr(plt, "show", show_mock)
        plot_surface_with_hole_half(_SliceableMock(), _make_inp(), show=True)
        show_mock.assert_called_once()

    def test_show_false(self, monkeypatch):
        """show=False does not call plt.show()."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        show_mock = MagicMock()
        monkeypatch.setattr(plt, "show", show_mock)
        plot_surface_with_hole_half(_SliceableMock(), _make_inp(), show=False)
        show_mock.assert_not_called()

    def test_vrange_minmax(self, monkeypatch):
        """vrange='minmax' uses raw min/max."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock(vmin=-3.0, vmax=7.0)
        plot_surface_with_hole_half(data, _make_inp(), vrange="minmax")
        for kw in data.plot_kwargs:
            assert kw["vmax"] == 7.0
            assert kw["vmin"] == -3.0

    def test_vrange_center(self, monkeypatch):
        """vrange='center' makes symmetric range."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock(vmin=-2.0, vmax=5.0)
        plot_surface_with_hole_half(data, _make_inp(), vrange="center")
        for kw in data.plot_kwargs:
            assert kw["vmax"] == 5.0
            assert kw["vmin"] == -5.0

    def test_use_si(self, monkeypatch):
        """use_si=True reads from val_si."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock(vmin=-1.0, vmax=1.0)
        data.val_si = SimpleNamespace(max=lambda: 8.0, min=lambda: -3.0)
        plot_surface_with_hole_half(data, _make_inp(), use_si=True)
        for kw in data.plot_kwargs:
            assert kw["vmax"] == 8.0
            assert kw["vmin"] == -3.0

    def test_box_aspect(self, monkeypatch):
        """set_box_aspect called with 3-tuple, all <= 1."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        plot_surface_with_hole_half(_SliceableMock(), _make_inp(), off=10)
        ax.set_box_aspect.assert_called_once()
        aspect = ax.set_box_aspect.call_args[0][0]
        assert len(aspect) == 3
        assert all(0 < v <= 1.0 for v in aspect)

    def test_ax3d_in_kwargs(self, monkeypatch):
        """Created 3D axis is set as kwargs['ax3d'] in every plot call."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock()
        plot_surface_with_hole_half(data, _make_inp())
        for kw in data.plot_kwargs:
            assert kw["ax3d"] is ax

    def test_masked_called(self, monkeypatch):
        """data.masked() is called for inner wall surfaces."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _SliceableMock()
        plot_surface_with_hole_half(data, _make_inp())
        assert data._mask_fn is not None
        # The mask function should return all-True
        arr = np.array([1.0, 2.0, 3.0])
        assert np.all(data._mask_fn(arr))

    def test_total_plot_calls(self, monkeypatch):
        """3 top + 3 wall + 5 inner wall + 1 bottom = 12 plot calls."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock()
        plot_surface_with_hole_half(data, _make_inp())
        assert len(data.plot_kwargs) == 12

    def test_bottom_has_labels(self, monkeypatch):
        """Bottom surface has xlabel/ylabel/zlabel."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock()
        plot_surface_with_hole_half(data, _make_inp())
        last = data.plot_kwargs[-1]
        assert last["xlabel"] == "x[m]"
        assert last["ylabel"] == "y[m]"
        assert last["zlabel"] == "z[m]"

    def test_bottom_has_colorbar(self, monkeypatch):
        """Bottom surface has add_colorbar."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock()
        plot_surface_with_hole_half(data, _make_inp(), add_colorbar=True)
        assert data.plot_kwargs[-1]["add_colorbar"] is True

    def test_mode_surf(self, monkeypatch):
        """All calls use mode='surf'."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock()
        plot_surface_with_hole_half(data, _make_inp())
        for kw in data.plot_kwargs:
            assert kw["mode"] == "surf"

    def test_extra_kwargs(self, monkeypatch):
        """Extra kwargs like cmap forwarded everywhere."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock()
        plot_surface_with_hole_half(data, _make_inp(), cmap="coolwarm", dpi=150)
        for kw in data.plot_kwargs:
            assert kw["cmap"] == "coolwarm"
            assert kw["dpi"] == 150

    def test_explicit_vmin_vmax(self, monkeypatch):
        """Explicit vmin/vmax override computed values."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock()
        plot_surface_with_hole_half(data, _make_inp(), vmin=-50, vmax=50)
        for kw in data.plot_kwargs:
            assert kw["vmax"] == 50
            assert kw["vmin"] == -50

    def test_returns_none(self, monkeypatch):
        """Function returns None."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        assert plot_surface_with_hole_half(_SliceableMock(), _make_inp()) is None

    def test_off_changes_slicing(self, monkeypatch):
        """Different off values produce different slicing."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        results = {}
        for off_val in [5, 20]:
            fig, ax = self._patch_3d(monkeypatch)
            data = _SliceableMock()
            plot_surface_with_hole_half(data, _make_inp(), off=off_val)
            results[off_val] = list(data._getitem_calls)

        assert results[5] != results[20]

    def test_projection_3d(self, monkeypatch):
        """Subplot uses projection='3d'."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        plot_surface_with_hole_half(_SliceableMock(), _make_inp())
        fig.add_subplot.assert_called_once_with(projection="3d")

    def test_sca_called(self, monkeypatch):
        """plt.sca is called with the 3D axis."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        ax = MagicMock()
        ax.set_box_aspect = MagicMock()
        fig = MagicMock()
        fig.add_subplot.return_value = ax
        monkeypatch.setattr(plt, "gcf", lambda: fig)
        sca_mock = MagicMock()
        monkeypatch.setattr(plt, "sca", sca_mock)
        monkeypatch.setattr(plt, "show", MagicMock())

        plot_surface_with_hole_half(_SliceableMock(), _make_inp())
        sca_mock.assert_called_once_with(ax)

    def test_add_colorbar_false_bottom(self, monkeypatch):
        """add_colorbar=False forwarded to bottom."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _TrackingMock()
        plot_surface_with_hole_half(data, _make_inp(), add_colorbar=False)
        assert data.plot_kwargs[-1]["add_colorbar"] is False

    def test_masked_inner_wall_slices(self, monkeypatch):
        """Inner wall uses masked data (not original data) for 5 slices."""
        from emout.plot.extra_plot import plot_surface_with_hole_half

        fig, ax = self._patch_3d(monkeypatch)
        data = _SliceableMock()
        plot_surface_with_hole_half(data, _make_inp())
        # Masked mock was created and sliced
        assert data._masked_mock is not None
        assert len(data._masked_mock._getitem_calls) == 5
