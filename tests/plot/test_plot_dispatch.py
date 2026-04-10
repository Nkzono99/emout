"""Tests for plot dispatch logic across the emout.plot package.

Tests focus on argument parsing, mode selection, coordinate setup, and
control-flow branching.  Matplotlib rendering is monkeypatched so no
actual display or file-writing occurs.
"""

import collections
import copy
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_figure():
    """Return a MagicMock that behaves like a matplotlib Figure."""
    fig = MagicMock()
    ax = MagicMock()
    fig.add_subplot.return_value = ax
    return fig


def _mock_ax3d():
    """Return a MagicMock pretending to be an Axes3D."""
    ax = MagicMock()
    ax.plot_surface.return_value = MagicMock()
    return ax


# ===================================================================
# _plot_2d  --  plot_2dmap
# ===================================================================

class TestPlot2dmap:
    """Tests for emout.plot._plot_2d.plot_2dmap."""

    def test_basic_no_mesh(self, monkeypatch):
        """Default call: mesh is auto-generated, imshow is called."""
        from emout.plot._plot_2d import plot_2dmap

        mock_img = MagicMock()
        monkeypatch.setattr(plt, "imshow", lambda *a, **kw: mock_img)
        monkeypatch.setattr(plt, "colorbar", lambda **kw: MagicMock())

        z = np.random.rand(8, 12)
        result = plot_2dmap(z)
        assert result is mock_img

    def test_with_mesh(self, monkeypatch):
        """Explicit mesh is used for extent calculation."""
        from emout.plot._plot_2d import plot_2dmap

        imshow_kwargs = {}
        def _imshow(*a, **kw):
            imshow_kwargs.update(kw)
            return MagicMock()

        monkeypatch.setattr(plt, "imshow", _imshow)
        monkeypatch.setattr(plt, "colorbar", lambda **kw: MagicMock())

        z = np.random.rand(5, 6)
        x = np.arange(6, dtype=float) * 2
        y = np.arange(5, dtype=float) * 3
        mesh = np.meshgrid(x, y)

        plot_2dmap(z, mesh=mesh)
        extent = imshow_kwargs["extent"]
        assert extent[0] == pytest.approx(0.0)
        assert extent[1] == pytest.approx(10.0)

    def test_vmin_vmax_forwarded(self, monkeypatch):
        """vmin/vmax are passed through to imshow."""
        from emout.plot._plot_2d import plot_2dmap

        imshow_kwargs = {}
        def _imshow(*a, **kw):
            imshow_kwargs.update(kw)
            return MagicMock()

        monkeypatch.setattr(plt, "imshow", _imshow)
        monkeypatch.setattr(plt, "colorbar", lambda **kw: MagicMock())

        z = np.random.rand(4, 4)
        plot_2dmap(z, vmin=-3.0, vmax=3.0)
        assert imshow_kwargs["vmin"] == -3.0
        assert imshow_kwargs["vmax"] == 3.0

    def test_savefilename_returns_none(self, monkeypatch, tmp_path):
        """When savefilename is given, figure is saved and None returned."""
        from emout.plot._plot_2d import plot_2dmap

        monkeypatch.setattr(plt, "imshow", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(plt, "colorbar", lambda **kw: MagicMock())
        mock_fig = _mock_figure()
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "close", lambda f: None)

        z = np.random.rand(4, 4)
        result = plot_2dmap(z, savefilename=str(tmp_path / "out.png"))
        assert result is None
        mock_fig.savefig.assert_called_once()

    def test_title_xlabel_ylabel_set(self, monkeypatch):
        """Labels are forwarded to plt.title/xlabel/ylabel."""
        from emout.plot._plot_2d import plot_2dmap

        called = {}
        monkeypatch.setattr(plt, "imshow", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(plt, "colorbar", lambda **kw: MagicMock())
        monkeypatch.setattr(plt, "title", lambda t: called.update(title=t))
        monkeypatch.setattr(plt, "xlabel", lambda t: called.update(xlabel=t))
        monkeypatch.setattr(plt, "ylabel", lambda t: called.update(ylabel=t))

        z = np.random.rand(4, 4)
        plot_2dmap(z, title="T", xlabel="X", ylabel="Y")
        assert called["title"] == "T"
        assert called["xlabel"] == "X"
        assert called["ylabel"] == "Y"

    def test_no_labels_when_none(self, monkeypatch):
        """Labels are not set when all are None."""
        from emout.plot._plot_2d import plot_2dmap

        label_calls = []
        monkeypatch.setattr(plt, "imshow", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(plt, "colorbar", lambda **kw: MagicMock())
        monkeypatch.setattr(plt, "title", lambda t: label_calls.append("title"))
        monkeypatch.setattr(plt, "xlabel", lambda t: label_calls.append("xlabel"))
        monkeypatch.setattr(plt, "ylabel", lambda t: label_calls.append("ylabel"))

        z = np.random.rand(4, 4)
        plot_2dmap(z)
        assert label_calls == []

    def test_cmap_none(self, monkeypatch):
        """cmap=None passes cmap=None to imshow."""
        from emout.plot._plot_2d import plot_2dmap

        imshow_kw = {}
        monkeypatch.setattr(plt, "imshow", lambda *a, **kw: imshow_kw.update(kw) or MagicMock())
        monkeypatch.setattr(plt, "colorbar", lambda **kw: MagicMock())

        z = np.random.rand(4, 4)
        plot_2dmap(z, cmap=None)
        assert imshow_kw["cmap"] is None

    def test_cmap_string(self, monkeypatch):
        """cmap='viridis' is resolved via plt.get_cmap."""
        from emout.plot._plot_2d import plot_2dmap

        imshow_kw = {}
        monkeypatch.setattr(plt, "imshow", lambda *a, **kw: imshow_kw.update(kw) or MagicMock())
        monkeypatch.setattr(plt, "colorbar", lambda **kw: MagicMock())

        z = np.random.rand(4, 4)
        plot_2dmap(z, cmap="viridis")
        assert imshow_kw["cmap"] is not None

    def test_add_colorbar_false(self, monkeypatch):
        """add_colorbar=False skips colorbar."""
        from emout.plot._plot_2d import plot_2dmap

        cb_count = [0]
        monkeypatch.setattr(plt, "imshow", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(plt, "colorbar", lambda **kw: cb_count.__setitem__(0, cb_count[0] + 1) or MagicMock())

        z = np.random.rand(4, 4)
        plot_2dmap(z, add_colorbar=False)
        assert cb_count[0] == 0

    def test_cbargs_with_cb_key(self, monkeypatch):
        """cbargs with 'cb' key is forwarded to plt.colorbar."""
        from emout.plot._plot_2d import plot_2dmap

        cb_kw = {}
        monkeypatch.setattr(plt, "imshow", lambda *a, **kw: MagicMock())
        def _colorbar(**kw):
            cb_kw.update(kw)
            return MagicMock()
        monkeypatch.setattr(plt, "colorbar", _colorbar)

        z = np.random.rand(4, 4)
        plot_2dmap(z, cbargs={"cb": {"shrink": 0.5}})
        assert cb_kw.get("shrink") == 0.5

    def test_interpolation_forwarded(self, monkeypatch):
        """Interpolation keyword reaches imshow."""
        from emout.plot._plot_2d import plot_2dmap

        imshow_kw = {}
        monkeypatch.setattr(plt, "imshow", lambda *a, **kw: imshow_kw.update(kw) or MagicMock())
        monkeypatch.setattr(plt, "colorbar", lambda **kw: MagicMock())

        z = np.random.rand(4, 4)
        plot_2dmap(z, interpolation="nearest")
        assert imshow_kw["interpolation"] == "nearest"


# ===================================================================
# _plot_2d  --  plot_2d_contour
# ===================================================================

class TestPlot2dContour:
    """Tests for emout.plot._plot_2d.plot_2d_contour."""

    def test_basic(self, monkeypatch):
        """Basic contour plot returns contour object."""
        from emout.plot._plot_2d import plot_2d_contour

        mock_cont = MagicMock()
        monkeypatch.setattr(plt, "contour", lambda *a, **kw: mock_cont)

        z = np.random.rand(6, 6)
        result = plot_2d_contour(z)
        assert result is mock_cont
        mock_cont.clabel.assert_called_once()

    def test_auto_mesh(self, monkeypatch):
        """Without explicit mesh, indices are used."""
        from emout.plot._plot_2d import plot_2d_contour

        contour_args = []
        mock_cont = MagicMock()
        def _contour(*a, **kw):
            contour_args.extend(a)
            return mock_cont
        monkeypatch.setattr(plt, "contour", _contour)

        z = np.random.rand(5, 7)
        plot_2d_contour(z)
        # mesh[0] and mesh[1] plus the data should be passed
        assert len(contour_args) == 3  # X, Y, data

    def test_explicit_mesh(self, monkeypatch):
        """Explicit mesh is used directly."""
        from emout.plot._plot_2d import plot_2d_contour

        contour_args = []
        mock_cont = MagicMock()
        def _contour(*a, **kw):
            contour_args.extend(a)
            return mock_cont
        monkeypatch.setattr(plt, "contour", _contour)

        z = np.random.rand(5, 7)
        x = np.arange(7, dtype=float)
        y = np.arange(5, dtype=float)
        mesh = np.meshgrid(x, y)
        plot_2d_contour(z, mesh=mesh)
        np.testing.assert_array_equal(contour_args[0], mesh[0])
        np.testing.assert_array_equal(contour_args[1], mesh[1])

    def test_colors_default_black(self, monkeypatch):
        """Without cmap, default colors=['black'] is used."""
        from emout.plot._plot_2d import plot_2d_contour

        contour_kw = {}
        mock_cont = MagicMock()
        def _contour(*a, **kw):
            contour_kw.update(kw)
            return mock_cont
        monkeypatch.setattr(plt, "contour", _contour)

        z = np.random.rand(5, 5)
        plot_2d_contour(z)
        assert contour_kw["colors"] == ["black"]

    def test_cmap_overrides_colors(self, monkeypatch):
        """When cmap is given, 'colors' key is absent and 'cmap' is set."""
        from emout.plot._plot_2d import plot_2d_contour

        contour_kw = {}
        mock_cont = MagicMock()
        def _contour(*a, **kw):
            contour_kw.update(kw)
            return mock_cont
        monkeypatch.setattr(plt, "contour", _contour)

        z = np.random.rand(5, 5)
        plot_2d_contour(z, cmap="viridis")
        assert "colors" not in contour_kw
        assert contour_kw["cmap"] == "viridis"

    def test_levels_forwarded(self, monkeypatch):
        """levels parameter is forwarded."""
        from emout.plot._plot_2d import plot_2d_contour

        contour_kw = {}
        mock_cont = MagicMock()
        def _contour(*a, **kw):
            contour_kw.update(kw)
            return mock_cont
        monkeypatch.setattr(plt, "contour", _contour)

        z = np.random.rand(5, 5)
        plot_2d_contour(z, levels=10)
        assert contour_kw["levels"] == 10

    def test_savefilename(self, monkeypatch, tmp_path):
        """savefilename causes save and returns None."""
        from emout.plot._plot_2d import plot_2d_contour

        mock_cont = MagicMock()
        monkeypatch.setattr(plt, "contour", lambda *a, **kw: mock_cont)
        mock_fig = _mock_figure()
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "close", lambda f: None)

        z = np.random.rand(5, 5)
        result = plot_2d_contour(z, savefilename=str(tmp_path / "c.png"))
        assert result is None
        mock_fig.savefig.assert_called_once()

    def test_alpha_vmin_vmax(self, monkeypatch):
        """alpha, vmin, vmax are forwarded."""
        from emout.plot._plot_2d import plot_2d_contour

        contour_kw = {}
        mock_cont = MagicMock()
        def _contour(*a, **kw):
            contour_kw.update(kw)
            return mock_cont
        monkeypatch.setattr(plt, "contour", _contour)

        z = np.random.rand(5, 5)
        plot_2d_contour(z, alpha=0.5, vmin=-1, vmax=1)
        assert contour_kw["alpha"] == 0.5
        assert contour_kw["vmin"] == -1
        assert contour_kw["vmax"] == 1

    def test_fmt_fontsize(self, monkeypatch):
        """fmt and fontsize are forwarded to clabel."""
        from emout.plot._plot_2d import plot_2d_contour

        mock_cont = MagicMock()
        monkeypatch.setattr(plt, "contour", lambda *a, **kw: mock_cont)

        z = np.random.rand(5, 5)
        plot_2d_contour(z, fmt="%1.2f", fontsize=14)
        mock_cont.clabel.assert_called_once_with(fmt="%1.2f", fontsize=14)

    def test_title_xlabel_ylabel(self, monkeypatch):
        """Labels are set when provided."""
        from emout.plot._plot_2d import plot_2d_contour

        called = {}
        mock_cont = MagicMock()
        monkeypatch.setattr(plt, "contour", lambda *a, **kw: mock_cont)
        monkeypatch.setattr(plt, "title", lambda t: called.update(title=t))
        monkeypatch.setattr(plt, "xlabel", lambda t: called.update(xlabel=t))
        monkeypatch.setattr(plt, "ylabel", lambda t: called.update(ylabel=t))

        z = np.random.rand(5, 5)
        plot_2d_contour(z, title="T", xlabel="X", ylabel="Y")
        assert called == {"title": "T", "xlabel": "X", "ylabel": "Y"}

    def test_custom_colors(self, monkeypatch):
        """Explicit colors list overrides the default."""
        from emout.plot._plot_2d import plot_2d_contour

        contour_kw = {}
        mock_cont = MagicMock()
        def _contour(*a, **kw):
            contour_kw.update(kw)
            return mock_cont
        monkeypatch.setattr(plt, "contour", _contour)

        z = np.random.rand(5, 5)
        plot_2d_contour(z, colors=["red", "blue"])
        assert contour_kw["colors"] == ["red", "blue"]


# ===================================================================
# _plot_2d  --  plot_line
# ===================================================================

class TestPlotLine:
    """Tests for emout.plot._plot_2d.plot_line."""

    def test_basic(self, monkeypatch):
        """Basic call with no x returns line object."""
        from emout.plot._plot_2d import plot_line

        mock_line = MagicMock()
        plot_args = []
        def _plot(*a, **kw):
            plot_args.extend(a)
            return mock_line
        monkeypatch.setattr(plt, "plot", _plot)
        monkeypatch.setattr(plt, "ylim", lambda v: None)

        data = np.arange(10, dtype=float)
        result = plot_line(data)
        assert result is mock_line
        assert len(plot_args) == 1  # just data

    def test_with_x(self, monkeypatch):
        """When x is given, plot(x, data) is called."""
        from emout.plot._plot_2d import plot_line

        plot_args = []
        def _plot(*a, **kw):
            plot_args.extend(a)
            return MagicMock()
        monkeypatch.setattr(plt, "plot", _plot)
        monkeypatch.setattr(plt, "ylim", lambda v: None)

        data = np.arange(10, dtype=float)
        x = np.linspace(0, 1, 10)
        plot_line(data, x=x)
        assert len(plot_args) == 2
        np.testing.assert_array_equal(plot_args[0], x)
        np.testing.assert_array_equal(plot_args[1], data)

    def test_ylim_set(self, monkeypatch):
        """vmin/vmax are forwarded to plt.ylim."""
        from emout.plot._plot_2d import plot_line

        ylim_args = []
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(plt, "ylim", lambda v: ylim_args.append(v))

        data = np.arange(10, dtype=float)
        plot_line(data, vmin=-5, vmax=5)
        assert ylim_args[0] == [-5, 5]

    def test_label_forwarded(self, monkeypatch):
        """label keyword is forwarded to plt.plot."""
        from emout.plot._plot_2d import plot_line

        plot_kw = {}
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: plot_kw.update(kw) or MagicMock())
        monkeypatch.setattr(plt, "ylim", lambda v: None)

        data = np.arange(10, dtype=float)
        plot_line(data, label="my label")
        assert plot_kw["label"] == "my label"

    def test_savefilename(self, monkeypatch, tmp_path):
        """savefilename causes save and returns None."""
        from emout.plot._plot_2d import plot_line

        mock_fig = _mock_figure()
        monkeypatch.setattr(plt, "figure", lambda **kw: mock_fig)
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(plt, "ylim", lambda v: None)
        monkeypatch.setattr(plt, "close", lambda f: None)

        data = np.arange(10, dtype=float)
        result = plot_line(data, savefilename=str(tmp_path / "l.png"))
        assert result is None
        mock_fig.savefig.assert_called_once()

    def test_savefilename_with_figsize(self, monkeypatch, tmp_path):
        """savefilename + figsize creates figure with size."""
        from emout.plot._plot_2d import plot_line

        fig_kw = {}
        mock_fig = _mock_figure()
        def _figure(**kw):
            fig_kw.update(kw)
            return mock_fig
        monkeypatch.setattr(plt, "figure", _figure)
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(plt, "ylim", lambda v: None)
        monkeypatch.setattr(plt, "close", lambda f: None)

        data = np.arange(10, dtype=float)
        plot_line(data, savefilename=str(tmp_path / "l.png"), figsize=(10, 5))
        assert fig_kw["figsize"] == (10, 5)

    def test_title_xlabel_ylabel(self, monkeypatch):
        """Labels are set when provided."""
        from emout.plot._plot_2d import plot_line

        called = {}
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(plt, "ylim", lambda v: None)
        monkeypatch.setattr(plt, "title", lambda t: called.update(title=t))
        monkeypatch.setattr(plt, "xlabel", lambda t: called.update(xlabel=t))
        monkeypatch.setattr(plt, "ylabel", lambda t: called.update(ylabel=t))

        data = np.arange(10, dtype=float)
        plot_line(data, title="T", xlabel="X", ylabel="Y")
        assert called == {"title": "T", "xlabel": "X", "ylabel": "Y"}

    def test_extra_kwargs_forwarded(self, monkeypatch):
        """Extra kwargs like color are forwarded."""
        from emout.plot._plot_2d import plot_line

        plot_kw = {}
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: plot_kw.update(kw) or MagicMock())
        monkeypatch.setattr(plt, "ylim", lambda v: None)

        data = np.arange(10, dtype=float)
        plot_line(data, color="red", linestyle="--")
        assert plot_kw["color"] == "red"
        assert plot_kw["linestyle"] == "--"


# ===================================================================
# _plot_2d  --  plot_2d_vector
# ===================================================================

class TestPlot2dVector:
    """Tests for emout.plot._plot_2d.plot_2d_vector."""

    def test_basic(self, monkeypatch):
        """Basic quiver plot returns image."""
        from emout.plot._plot_2d import plot_2d_vector

        mock_img = MagicMock()
        monkeypatch.setattr(plt, "quiver", lambda *a, **kw: mock_img)

        U = np.random.rand(8, 8)
        V = np.random.rand(8, 8)
        result = plot_2d_vector(U, V)
        assert result is mock_img

    def test_skip_int(self, monkeypatch):
        """Integer skip downsamples uniformly."""
        from emout.plot._plot_2d import plot_2d_vector

        quiver_args = []
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            return MagicMock()
        monkeypatch.setattr(plt, "quiver", _quiver)

        U = np.random.rand(8, 8)
        V = np.random.rand(8, 8)
        plot_2d_vector(U, V, skip=2)
        # After skip=2 on 8x8, we get 4x4 mesh
        assert quiver_args[0].shape == (4, 4)

    def test_skip_tuple(self, monkeypatch):
        """Tuple skip uses different x/y skip."""
        from emout.plot._plot_2d import plot_2d_vector

        quiver_args = []
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            return MagicMock()
        monkeypatch.setattr(plt, "quiver", _quiver)

        U = np.random.rand(12, 8)
        V = np.random.rand(12, 8)
        plot_2d_vector(U, V, skip=(2, 3))
        assert quiver_args[0].shape == (4, 4)

    def test_scaler_standard(self, monkeypatch):
        """Standard scaler normalizes by max norm."""
        from emout.plot._plot_2d import plot_2d_vector

        quiver_args = []
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            return MagicMock()
        monkeypatch.setattr(plt, "quiver", _quiver)

        U = np.ones((4, 4)) * 3.0
        V = np.ones((4, 4)) * 4.0
        plot_2d_vector(U, V, scaler="standard", easy_to_read=False)
        # norm_max = 5.0
        # The raw U was divided by norm_max
        # quiver_args[2] is U, quiver_args[3] is V
        np.testing.assert_allclose(quiver_args[2], 3.0 / 5.0, atol=1e-10)
        np.testing.assert_allclose(quiver_args[3], 4.0 / 5.0, atol=1e-10)

    def test_scaler_normal(self, monkeypatch):
        """Normal scaler normalizes each vector to unit length."""
        from emout.plot._plot_2d import plot_2d_vector

        quiver_args = []
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            return MagicMock()
        monkeypatch.setattr(plt, "quiver", _quiver)

        U = np.ones((4, 4)) * 3.0
        V = np.ones((4, 4)) * 4.0
        plot_2d_vector(U, V, scaler="normal", easy_to_read=False)
        norm = np.sqrt(quiver_args[2] ** 2 + quiver_args[3] ** 2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)

    def test_scaler_log(self, monkeypatch):
        """Log scaler applies log(norm+1) scaling."""
        from emout.plot._plot_2d import plot_2d_vector

        quiver_args = []
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            return MagicMock()
        monkeypatch.setattr(plt, "quiver", _quiver)

        U = np.ones((4, 4)) * 3.0
        V = np.ones((4, 4)) * 4.0
        plot_2d_vector(U, V, scaler="log", easy_to_read=False)
        # After log scaling U = (3/5)*log(6), V = (4/5)*log(6)
        expected_u = 3.0 / 5.0 * np.log(6.0)
        np.testing.assert_allclose(quiver_args[2], expected_u, atol=1e-10)

    def test_cmap_quiver(self, monkeypatch):
        """With cmap, quiver receives color array."""
        from emout.plot._plot_2d import plot_2d_vector

        quiver_args = []
        quiver_kw = {}
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            quiver_kw.update(kw)
            return MagicMock()
        monkeypatch.setattr(plt, "quiver", _quiver)

        U = np.random.rand(4, 4)
        V = np.random.rand(4, 4)
        plot_2d_vector(U, V, cmap="viridis")
        assert quiver_kw["cmap"] == "viridis"
        # 5 positional args: x, y, U, V, magnitude
        assert len(quiver_args) == 5


# ===================================================================
# _plot_2d  --  plot_2d_streamline
# ===================================================================

class TestPlot2dStreamline:
    """Tests for emout.plot._plot_2d.plot_2d_streamline."""

    def test_basic_no_cmap(self, monkeypatch):
        """Without cmap, streamplot is called with color=None."""
        from emout.plot._plot_2d import plot_2d_streamline

        stream_kw = {}
        mock_stream = MagicMock()
        def _streamplot(*a, **kw):
            stream_kw.update(kw)
            return mock_stream
        monkeypatch.setattr(plt, "streamplot", _streamplot)

        U = np.random.rand(8, 8)
        V = np.random.rand(8, 8)
        result = plot_2d_streamline(U, V)
        assert result is mock_stream
        assert stream_kw.get("color") is None

    def test_with_cmap_linear(self, monkeypatch):
        """With cmap, magnitude coloring and linear norm are used."""
        from emout.plot._plot_2d import plot_2d_streamline

        stream_kw = {}
        mock_stream = MagicMock()
        def _streamplot(*a, **kw):
            stream_kw.update(kw)
            return mock_stream
        monkeypatch.setattr(plt, "streamplot", _streamplot)

        U = np.random.rand(8, 8)
        V = np.random.rand(8, 8)
        plot_2d_streamline(U, V, cmap="viridis", norm="linear")
        assert stream_kw["cmap"] == "viridis"
        assert "norm" in stream_kw

    def test_with_cmap_log(self, monkeypatch):
        """cmap + norm='log' uses LogNorm."""
        from emout.plot._plot_2d import plot_2d_streamline
        import matplotlib.colors as mcolors

        stream_kw = {}
        def _streamplot(*a, **kw):
            stream_kw.update(kw)
            return MagicMock()
        monkeypatch.setattr(plt, "streamplot", _streamplot)

        U = np.random.rand(8, 8) + 0.1
        V = np.random.rand(8, 8) + 0.1
        plot_2d_streamline(U, V, cmap="viridis", norm="log")
        assert isinstance(stream_kw["norm"], mcolors.LogNorm)

    def test_with_cmap_centered(self, monkeypatch):
        """cmap + norm='centered' uses CenteredNorm."""
        from emout.plot._plot_2d import plot_2d_streamline
        import matplotlib.colors as mcolors

        stream_kw = {}
        def _streamplot(*a, **kw):
            stream_kw.update(kw)
            return MagicMock()
        monkeypatch.setattr(plt, "streamplot", _streamplot)

        U = np.random.rand(8, 8)
        V = np.random.rand(8, 8)
        plot_2d_streamline(U, V, cmap="viridis", norm="centered")
        assert isinstance(stream_kw["norm"], mcolors.CenteredNorm)

    def test_skip(self, monkeypatch):
        """Skip parameter downsamples the grid."""
        from emout.plot._plot_2d import plot_2d_streamline

        stream_args = []
        def _streamplot(*a, **kw):
            stream_args.extend(a)
            return MagicMock()
        monkeypatch.setattr(plt, "streamplot", _streamplot)

        U = np.random.rand(8, 8)
        V = np.random.rand(8, 8)
        plot_2d_streamline(U, V, skip=2)
        assert stream_args[0].shape == (4, 4)

    def test_density_forwarded(self, monkeypatch):
        """density parameter is forwarded."""
        from emout.plot._plot_2d import plot_2d_streamline

        stream_kw = {}
        def _streamplot(*a, **kw):
            stream_kw.update(kw)
            return MagicMock()
        monkeypatch.setattr(plt, "streamplot", _streamplot)

        U = np.random.rand(8, 8)
        V = np.random.rand(8, 8)
        plot_2d_streamline(U, V, density=2.5)
        assert stream_kw["density"] == 2.5

    def test_savefilename(self, monkeypatch, tmp_path):
        """savefilename triggers save and returns None."""
        from emout.plot._plot_2d import plot_2d_streamline

        mock_fig = _mock_figure()
        monkeypatch.setattr(plt, "figure", lambda **kw: mock_fig)
        monkeypatch.setattr(plt, "streamplot", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(plt, "close", lambda f: None)

        U = np.random.rand(8, 8)
        V = np.random.rand(8, 8)
        result = plot_2d_streamline(U, V, savefilename=str(tmp_path / "s.png"))
        assert result is None
        mock_fig.savefig.assert_called_once()


# ===================================================================
# _plot_2d  --  plot_surface
# ===================================================================

class TestPlotSurface:
    """Tests for emout.plot._plot_2d.plot_surface."""

    def test_basic(self, monkeypatch):
        """Basic surface plot returns surf object."""
        from emout.plot._plot_2d import plot_surface

        ax3d = _mock_ax3d()
        mock_surf = MagicMock()
        ax3d.plot_surface.return_value = mock_surf
        monkeypatch.setattr(plt, "gcf", lambda: _mock_figure())
        monkeypatch.setattr(plt, "colorbar", lambda *a, **kw: MagicMock())

        x = np.arange(6, dtype=float).reshape(2, 3)
        y = np.arange(6, dtype=float).reshape(2, 3)
        z = np.arange(6, dtype=float).reshape(2, 3)
        val = np.random.rand(2, 3)
        result = plot_surface(x, y, z, val, ax3d=ax3d, ninterp=None)
        assert result is mock_surf

    def test_ninterp_calls_interp2d(self, monkeypatch):
        """ninterp triggers interpolation on all arrays."""
        from emout.plot._plot_2d import plot_surface

        interp_calls = [0]
        original_interp = None
        try:
            import emout.utils as eutils
            original_interp = eutils.interp2d
        except Exception:
            pass

        def _fake_interp(arr, n, **kw):
            interp_calls[0] += 1
            return arr  # return same shape for simplicity

        monkeypatch.setattr("emout.utils.interp2d", _fake_interp)
        ax3d = _mock_ax3d()
        monkeypatch.setattr(plt, "gcf", lambda: _mock_figure())

        x = np.arange(6, dtype=float).reshape(2, 3)
        y = np.arange(6, dtype=float).reshape(2, 3)
        z = np.arange(6, dtype=float).reshape(2, 3)
        val = np.random.rand(2, 3)
        plot_surface(x, y, z, val, ax3d=ax3d, ninterp=2)
        assert interp_calls[0] == 4  # x, y, z, value

    def test_add_colorbar(self, monkeypatch):
        """add_colorbar=True calls plt.colorbar."""
        from emout.plot._plot_2d import plot_surface

        cb_calls = [0]
        monkeypatch.setattr(plt, "colorbar", lambda *a, **kw: cb_calls.__setitem__(0, cb_calls[0] + 1) or MagicMock())
        ax3d = _mock_ax3d()
        monkeypatch.setattr(plt, "gcf", lambda: _mock_figure())

        x = np.arange(6, dtype=float).reshape(2, 3)
        y = np.arange(6, dtype=float).reshape(2, 3)
        z = np.arange(6, dtype=float).reshape(2, 3)
        val = np.random.rand(2, 3)
        plot_surface(x, y, z, val, ax3d=ax3d, ninterp=None, add_colorbar=True)
        assert cb_calls[0] == 1

    def test_labels_set(self, monkeypatch):
        """xlabel/ylabel/zlabel/title are set on ax3d."""
        from emout.plot._plot_2d import plot_surface

        ax3d = _mock_ax3d()
        monkeypatch.setattr(plt, "gcf", lambda: _mock_figure())
        monkeypatch.setattr(plt, "colorbar", lambda *a, **kw: MagicMock())

        x = np.arange(6, dtype=float).reshape(2, 3)
        y = np.arange(6, dtype=float).reshape(2, 3)
        z = np.arange(6, dtype=float).reshape(2, 3)
        val = np.random.rand(2, 3)
        plot_surface(x, y, z, val, ax3d=ax3d, ninterp=None,
                     xlabel="X", ylabel="Y", zlabel="Z", title="T")
        ax3d.set_xlabel.assert_called_once_with("X")
        ax3d.set_ylabel.assert_called_once_with("Y")
        ax3d.set_zlabel.assert_called_once_with("Z")
        ax3d.set_title.assert_called_once_with("T")

    def test_savefilename_returns_none(self, monkeypatch, tmp_path):
        """savefilename triggers save and returns None."""
        from emout.plot._plot_2d import plot_surface

        mock_fig = _mock_figure()
        ax3d = _mock_ax3d()
        mock_fig.add_subplot.return_value = ax3d
        monkeypatch.setattr(plt, "figure", lambda **kw: mock_fig)
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "close", lambda f: None)

        x = np.arange(6, dtype=float).reshape(2, 3)
        y = np.arange(6, dtype=float).reshape(2, 3)
        z = np.arange(6, dtype=float).reshape(2, 3)
        val = np.random.rand(2, 3)
        result = plot_surface(
            x, y, z, val, ninterp=None,
            savefilename=str(tmp_path / "surf.png")
        )
        assert result is None
        mock_fig.savefig.assert_called_once()

    def test_cmap_string(self, monkeypatch):
        """cmap as string is resolved."""
        from emout.plot._plot_2d import plot_surface

        ax3d = _mock_ax3d()
        monkeypatch.setattr(plt, "gcf", lambda: _mock_figure())

        x = np.arange(6, dtype=float).reshape(2, 3)
        y = np.arange(6, dtype=float).reshape(2, 3)
        z = np.arange(6, dtype=float).reshape(2, 3)
        val = np.random.rand(2, 3)
        # Should not raise
        plot_surface(x, y, z, val, ax3d=ax3d, ninterp=None, cmap="viridis")


# ===================================================================
# _plot_2d  --  figsize_with_2d
# ===================================================================

class TestFigsizeWith2d:
    """Tests for figsize_with_2d."""

    def test_basic(self):
        from emout.plot._plot_2d import figsize_with_2d
        data = np.zeros((10, 20))
        fs = figsize_with_2d(data, dpi=10)
        assert len(fs) == 2
        assert fs[0] > 0
        assert fs[1] > 0
        # width should be proportional to shape[1]
        assert fs[0] > fs[1]  # 20 > 10


# ===================================================================
# _plot_3d  --  plot_3d_quiver
# ===================================================================

class TestPlot3dQuiver:
    """Tests for emout.plot._plot_3d.plot_3d_quiver."""

    def test_basic_with_ax(self, monkeypatch):
        """Providing ax3d avoids figure creation."""
        from emout.plot._plot_3d import plot_3d_quiver

        ax3d = MagicMock()
        ax3d.quiver.return_value = MagicMock()

        U = np.random.rand(4, 4, 4)
        V = np.random.rand(4, 4, 4)
        W = np.random.rand(4, 4, 4)
        result = plot_3d_quiver(U, V, W, ax3d=ax3d)
        ax3d.quiver.assert_called_once()
        assert result is not None

    def test_skip_int(self, monkeypatch):
        """Integer skip downsamples all axes."""
        from emout.plot._plot_3d import plot_3d_quiver

        quiver_args = []
        ax3d = MagicMock()
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            return MagicMock()
        ax3d.quiver = _quiver

        U = np.random.rand(8, 8, 8)
        V = np.random.rand(8, 8, 8)
        W = np.random.rand(8, 8, 8)
        plot_3d_quiver(U, V, W, ax3d=ax3d, skip=2)
        # x, y, z, U, V, W -> 6 args
        assert quiver_args[0].shape == (4, 4, 4)

    def test_skip_tuple(self, monkeypatch):
        """Tuple skip uses per-axis skip."""
        from emout.plot._plot_3d import plot_3d_quiver

        quiver_args = []
        ax3d = MagicMock()
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            return MagicMock()
        ax3d.quiver = _quiver

        U = np.random.rand(6, 8, 12)
        V = np.random.rand(6, 8, 12)
        W = np.random.rand(6, 8, 12)
        plot_3d_quiver(U, V, W, ax3d=ax3d, skip=(2, 4, 3))
        # mesh generated from shape[1]=8 (x), shape[0]=6 (y,z)
        # meshgrid(8,6,6) -> (6,8,6), then [::3, ::4, ::2] -> (2,2,3)
        assert quiver_args[0].shape == (2, 2, 3)

    def test_scaler_standard(self, monkeypatch):
        """Standard scaler normalizes by 2D norm max (U,V only)."""
        from emout.plot._plot_3d import plot_3d_quiver

        quiver_args = []
        ax3d = MagicMock()
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            return MagicMock()
        ax3d.quiver = _quiver

        U = np.ones((4, 4, 4)) * 3.0
        V = np.ones((4, 4, 4)) * 4.0
        W = np.ones((4, 4, 4)) * 2.0
        plot_3d_quiver(U, V, W, ax3d=ax3d, scaler="standard", easy_to_read=False)
        # norm = sqrt(U^2+V^2) = 5.0 for all, norm_max=5.0
        np.testing.assert_allclose(quiver_args[3], 3.0 / 5.0, atol=1e-10)
        np.testing.assert_allclose(quiver_args[4], 4.0 / 5.0, atol=1e-10)
        np.testing.assert_allclose(quiver_args[5], 2.0 / 5.0, atol=1e-10)

    def test_cmap_adds_color_arg(self, monkeypatch):
        """With cmap, quiver receives a color magnitude array."""
        from emout.plot._plot_3d import plot_3d_quiver

        quiver_args = []
        quiver_kw = {}
        ax3d = MagicMock()
        def _quiver(*a, **kw):
            quiver_args.extend(a)
            quiver_kw.update(kw)
            return MagicMock()
        ax3d.quiver = _quiver

        U = np.random.rand(4, 4, 4)
        V = np.random.rand(4, 4, 4)
        W = np.random.rand(4, 4, 4)
        plot_3d_quiver(U, V, W, ax3d=ax3d, cmap="viridis")
        assert quiver_kw["cmap"] == "viridis"
        # 7 args: x, y, z, U, V, W, magnitude
        assert len(quiver_args) == 7

    def test_labels_set(self, monkeypatch):
        """title/xlabel/ylabel are set via plt when provided."""
        from emout.plot._plot_3d import plot_3d_quiver

        called = {}
        monkeypatch.setattr(plt, "title", lambda t: called.update(title=t))
        monkeypatch.setattr(plt, "xlabel", lambda t: called.update(xlabel=t))
        monkeypatch.setattr(plt, "ylabel", lambda t: called.update(ylabel=t))

        ax3d = MagicMock()
        ax3d.quiver.return_value = MagicMock()

        U = np.random.rand(4, 4, 4)
        V = np.random.rand(4, 4, 4)
        W = np.random.rand(4, 4, 4)
        plot_3d_quiver(U, V, W, ax3d=ax3d, title="T", xlabel="X", ylabel="Y")
        assert called["title"] == "T"
        assert called["xlabel"] == "X"
        assert called["ylabel"] == "Y"


# ===================================================================
# _plot_3d  --  plot_3d_streamline
# ===================================================================

class TestPlot3dStreamline:
    """Tests for emout.plot._plot_3d.plot_3d_streamline."""

    def test_basic_with_seed_points(self, monkeypatch):
        """Providing explicit seeds avoids random generation."""
        from emout.plot._plot_3d import plot_3d_streamline

        mock_fig = _mock_figure()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        monkeypatch.setattr(plt, "figure", lambda **kw: mock_fig)
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "show", lambda: None)

        Fx = np.ones((4, 4, 4))
        Fy = np.zeros((4, 4, 4))
        Fz = np.zeros((4, 4, 4))
        seeds = np.array([[1.5, 1.5, 1.5]])

        result = plot_3d_streamline(Fx, Fy, Fz, seed_points=seeds, show=False)
        assert result is mock_ax

    def test_no_lines_warns(self, monkeypatch):
        """When no streamlines can be traced, a warning is issued."""
        from emout.plot._plot_3d import plot_3d_streamline
        from scipy.integrate import solve_ivp as _real_solve_ivp

        mock_fig = _mock_figure()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        monkeypatch.setattr(plt, "figure", lambda **kw: mock_fig)
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "show", lambda: None)

        # Make solve_ivp always raise so no lines are produced
        def _failing_solve_ivp(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr("scipy.integrate.solve_ivp", _failing_solve_ivp)

        Fx = np.ones((4, 4, 4))
        Fy = np.zeros((4, 4, 4))
        Fz = np.zeros((4, 4, 4))
        seeds = np.array([[1.5, 1.5, 1.5]])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plot_3d_streamline(Fx, Fy, Fz, seed_points=seeds, show=False)
            assert any("No streamlines" in str(x.message) for x in w)

    def test_uniform_color(self, monkeypatch):
        """color='red' skips cmap logic and uses uniform color."""
        from emout.plot._plot_3d import plot_3d_streamline

        mock_fig = _mock_figure()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        monkeypatch.setattr(plt, "figure", lambda **kw: mock_fig)
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "show", lambda: None)

        Fx = np.ones((4, 4, 4))
        Fy = np.zeros((4, 4, 4))
        Fz = np.zeros((4, 4, 4))
        seeds = np.array([[1.5, 1.5, 1.5]])

        result = plot_3d_streamline(
            Fx, Fy, Fz, seed_points=seeds, color="red", show=False
        )
        assert result is mock_ax
        # At least one plot call should have been made
        assert mock_ax.plot.called

    def test_labels_set(self, monkeypatch):
        """xlabel/ylabel/zlabel/title are set on ax."""
        from emout.plot._plot_3d import plot_3d_streamline

        mock_fig = _mock_figure()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        monkeypatch.setattr(plt, "figure", lambda **kw: mock_fig)
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "show", lambda: None)

        Fx = np.ones((4, 4, 4))
        Fy = np.zeros((4, 4, 4))
        Fz = np.zeros((4, 4, 4))
        seeds = np.array([[1.5, 1.5, 1.5]])

        plot_3d_streamline(
            Fx, Fy, Fz, seed_points=seeds, show=False,
            xlabel="XX", ylabel="YY", zlabel="ZZ", title="TT",
        )
        mock_ax.set_xlabel.assert_called_once_with("XX")
        mock_ax.set_ylabel.assert_called_once_with("YY")
        mock_ax.set_zlabel.assert_called_once_with("ZZ")
        mock_ax.set_title.assert_called_once_with("TT")

    def test_savefilename(self, monkeypatch, tmp_path):
        """savefilename triggers savefig."""
        from emout.plot._plot_3d import plot_3d_streamline

        mock_fig = _mock_figure()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        monkeypatch.setattr(plt, "figure", lambda **kw: mock_fig)
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "savefig", lambda fn: None)
        monkeypatch.setattr(plt, "close", lambda f: None)
        monkeypatch.setattr(plt, "show", lambda: None)

        Fx = np.ones((4, 4, 4))
        Fy = np.zeros((4, 4, 4))
        Fz = np.zeros((4, 4, 4))
        seeds = np.array([[1.5, 1.5, 1.5]])

        result = plot_3d_streamline(
            Fx, Fy, Fz, seed_points=seeds,
            savefilename=str(tmp_path / "stream3d.png"), show=False,
        )
        assert result is mock_ax

    def test_default_labels(self, monkeypatch):
        """Default labels are 'x', 'y', 'z' when not specified."""
        from emout.plot._plot_3d import plot_3d_streamline

        mock_fig = _mock_figure()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        monkeypatch.setattr(plt, "figure", lambda **kw: mock_fig)
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "show", lambda: None)

        Fx = np.ones((4, 4, 4))
        Fy = np.zeros((4, 4, 4))
        Fz = np.zeros((4, 4, 4))
        seeds = np.array([[1.5, 1.5, 1.5]])

        plot_3d_streamline(Fx, Fy, Fz, seed_points=seeds, show=False)
        mock_ax.set_xlabel.assert_called_once_with("x")
        mock_ax.set_ylabel.assert_called_once_with("y")
        mock_ax.set_zlabel.assert_called_once_with("z")

    def test_with_existing_ax(self, monkeypatch):
        """When ax is provided, no new figure is created."""
        from emout.plot._plot_3d import plot_3d_streamline

        mock_ax = MagicMock()
        figure_calls = [0]
        orig_figure = plt.figure
        def _figure(**kw):
            figure_calls[0] += 1
            return _mock_figure()
        monkeypatch.setattr(plt, "figure", _figure)
        monkeypatch.setattr(plt, "gcf", _mock_figure)
        monkeypatch.setattr(plt, "show", lambda: None)

        Fx = np.ones((4, 4, 4))
        Fy = np.zeros((4, 4, 4))
        Fz = np.zeros((4, 4, 4))
        seeds = np.array([[1.5, 1.5, 1.5]])

        result = plot_3d_streamline(Fx, Fy, Fz, ax=mock_ax, seed_points=seeds, show=False)
        assert result is mock_ax
        assert figure_calls[0] == 0


# ===================================================================
# animation_plot  --  flatten_list
# ===================================================================

class TestFlattenList:
    """Tests for flatten_list."""

    def test_flat(self):
        from emout.plot.animation_plot import flatten_list
        assert list(flatten_list([1, 2, 3])) == [1, 2, 3]

    def test_nested(self):
        from emout.plot.animation_plot import flatten_list
        assert list(flatten_list([[1, [2]], 3])) == [1, 2, 3]

    def test_string_not_expanded(self):
        from emout.plot.animation_plot import flatten_list
        result = list(flatten_list(["abc", [1]]))
        assert result == ["abc", 1]

    def test_deeply_nested(self):
        from emout.plot.animation_plot import flatten_list
        result = list(flatten_list([[[[[1]]]]]))
        assert result == [1]


# ===================================================================
# animation_plot  --  Animator
# ===================================================================

class TestAnimator:
    """Tests for the Animator class."""

    def test_shape_single(self):
        from emout.plot.animation_plot import Animator
        updater = MagicMock()
        updater.__len__ = lambda self: 5
        animator = Animator([[[updater]]])
        assert animator.shape == (1, 1)

    def test_shape_multi(self):
        from emout.plot.animation_plot import Animator
        u1 = MagicMock()
        u2 = MagicMock()
        animator = Animator([[[u1], [u2]], [[u1]]])
        assert animator.shape == (2, 2)

    def test_frames_min(self):
        from emout.plot.animation_plot import Animator, FrameUpdater

        # Create mock updaters that look like FrameUpdater
        u1 = MagicMock(spec=FrameUpdater)
        u1.__len__ = lambda self: 10
        u2 = MagicMock(spec=FrameUpdater)
        u2.__len__ = lambda self: 5

        animator = Animator([[[u1, u2]]])
        assert animator.frames == 5

    def test_frames_empty_raises(self):
        from emout.plot.animation_plot import Animator
        animator = Animator([[]])
        with pytest.raises(ValueError, match="no elements"):
            _ = animator.frames

    def test_plot_action_return(self, monkeypatch):
        """action='return' returns (fig, ani) tuple."""
        from emout.plot.animation_plot import Animator, FrameUpdater
        import matplotlib.animation as animation

        u = MagicMock(spec=FrameUpdater)
        u.__len__ = lambda self: 3
        animator = Animator([[[u]]])

        mock_fig = _mock_figure()
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "clf", lambda: None)
        monkeypatch.setattr(plt, "subplot", lambda *a: None)

        mock_ani = MagicMock()
        monkeypatch.setattr(
            animation, "FuncAnimation",
            lambda fig, func, **kw: mock_ani,
        )

        result = animator.plot(action="return")
        assert result == (mock_fig, mock_ani)

    def test_plot_deprecated_show(self, monkeypatch):
        """show=True triggers deprecation warning."""
        from emout.plot.animation_plot import Animator, FrameUpdater
        import matplotlib.animation as animation

        u = MagicMock(spec=FrameUpdater)
        u.__len__ = lambda self: 3
        animator = Animator([[[u]]])

        mock_fig = _mock_figure()
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "show", lambda: None)
        monkeypatch.setattr(plt, "clf", lambda: None)
        monkeypatch.setattr(plt, "subplot", lambda *a: None)
        monkeypatch.setattr(
            animation, "FuncAnimation",
            lambda fig, func, **kw: MagicMock(),
        )

        with pytest.warns(DeprecationWarning, match="show"):
            animator.plot(show=True)

    def test_plot_deprecated_savefilename(self, monkeypatch, tmp_path):
        """savefilename triggers deprecation warning."""
        from emout.plot.animation_plot import Animator, FrameUpdater
        import matplotlib.animation as animation

        u = MagicMock(spec=FrameUpdater)
        u.__len__ = lambda self: 3
        animator = Animator([[[u]]])

        mock_fig = _mock_figure()
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "clf", lambda: None)
        monkeypatch.setattr(plt, "subplot", lambda *a: None)

        mock_ani = MagicMock()
        monkeypatch.setattr(
            animation, "FuncAnimation",
            lambda fig, func, **kw: mock_ani,
        )

        with pytest.warns(DeprecationWarning, match="savefilename"):
            animator.plot(savefilename=str(tmp_path / "out.gif"))
        mock_ani.save.assert_called_once()

    def test_plot_deprecated_to_html(self, monkeypatch):
        """to_html=True triggers deprecation warning."""
        from emout.plot.animation_plot import Animator, FrameUpdater
        import matplotlib.animation as animation

        u = MagicMock(spec=FrameUpdater)
        u.__len__ = lambda self: 3
        animator = Animator([[[u]]])

        mock_fig = _mock_figure()
        monkeypatch.setattr(plt, "gcf", lambda: mock_fig)
        monkeypatch.setattr(plt, "clf", lambda: None)
        monkeypatch.setattr(plt, "subplot", lambda *a: None)

        mock_ani = MagicMock()
        mock_ani.to_jshtml.return_value = "<html></html>"
        monkeypatch.setattr(
            animation, "FuncAnimation",
            lambda fig, func, **kw: mock_ani,
        )

        # Need to mock IPython HTML
        mock_html_cls = MagicMock()
        monkeypatch.setitem(
            __import__("sys").modules,
            "IPython.display",
            SimpleNamespace(HTML=mock_html_cls),
        )

        with pytest.warns(DeprecationWarning, match="to_html"):
            animator.plot(to_html=True)


# ===================================================================
# animation_plot  --  FrameUpdater
# ===================================================================

class TestFrameUpdater:
    """Tests for the FrameUpdater class."""

    def _make_data(self, shape=(5, 8, 8)):
        """Create a mock data object that looks like Data3d."""
        data = MagicMock()
        data.shape = shape
        data.valunit = MagicMock()
        data.name = "test"
        data.slice_axes = {0: 0, 1: 1, 2: 2}
        data.slices = {0: slice(None), 1: slice(None), 2: slice(None)}
        data.axisunits = {0: MagicMock(), 1: MagicMock(), 2: MagicMock()}
        data.axisunits[0].reverse.return_value = 1.0
        data.axisunits[0].unit = "m"
        return data

    def test_len(self):
        from emout.plot.animation_plot import FrameUpdater
        data = self._make_data()
        fu = FrameUpdater(data, axis=0)
        assert len(fu) == 5

    def test_to_animator(self):
        from emout.plot.animation_plot import FrameUpdater, Animator
        data = self._make_data()
        fu = FrameUpdater(data, axis=0)
        animator = fu.to_animator()
        assert isinstance(animator, Animator)

    def test_to_animator_custom_layout(self):
        from emout.plot.animation_plot import FrameUpdater, Animator
        data = self._make_data()
        fu = FrameUpdater(data, axis=0)
        fu2 = FrameUpdater(data, axis=0)
        animator = fu.to_animator(layout=[[[fu, fu2]]])
        assert isinstance(animator, Animator)
        assert animator.shape == (1, 1)

    def test_offseted_left(self):
        from emout.plot.animation_plot import FrameUpdater
        data = self._make_data()
        fu = FrameUpdater(data, axis=0)
        line = np.array([10.0, 20.0, 30.0])
        result = fu._offseted(line, "left")
        np.testing.assert_array_equal(result, [0, 10, 20])

    def test_offseted_center(self):
        from emout.plot.animation_plot import FrameUpdater
        data = self._make_data()
        fu = FrameUpdater(data, axis=0)
        line = np.array([10.0, 20.0, 30.0])
        result = fu._offseted(line, "center")
        np.testing.assert_array_equal(result, [-10, 0, 10])

    def test_offseted_right(self):
        from emout.plot.animation_plot import FrameUpdater
        data = self._make_data()
        fu = FrameUpdater(data, axis=0)
        line = np.array([10.0, 20.0, 30.0])
        result = fu._offseted(line, "right")
        np.testing.assert_array_equal(result, [-20, -10, 0])

    def test_offseted_numeric(self):
        from emout.plot.animation_plot import FrameUpdater
        data = self._make_data()
        fu = FrameUpdater(data, axis=0)
        line = np.array([10.0, 20.0, 30.0])
        result = fu._offseted(line, 5.0)
        np.testing.assert_array_equal(result, [15, 25, 35])

    def test_valunit_none_disables_si(self):
        """When valunit is None, use_si is forced to False."""
        from emout.plot.animation_plot import FrameUpdater
        data = self._make_data()
        data.valunit = None
        fu = FrameUpdater(data, axis=0, use_si=True)
        assert fu.use_si is False

    def test_title_default_from_name(self):
        """Default title uses data.name."""
        from emout.plot.animation_plot import FrameUpdater
        data = self._make_data()
        data.name = "phi"
        fu = FrameUpdater(data, axis=0)
        assert fu.title == "phi"


# ===================================================================
# plot_cross_sections
# ===================================================================

class TestPlotCrossSections:
    """Tests for plot_cross_sections."""

    def _make_data(self, boundary_type="complex", boundary_types=None):
        """Create a mock data object with inp."""
        data = MagicMock()
        data.inp = MagicMock()
        data.inp.boundary_type = boundary_type
        data.inp.boundary_types = boundary_types or []
        data.unit = MagicMock()
        data.unit.length = MagicMock()
        data.unit.length.reverse = lambda x: x * 2.0
        return data

    def test_no_inp_returns_ax(self, monkeypatch):
        """No inp -> return gca."""
        from emout.plot.plot_cross_sections import plot_cross_sections

        data = MagicMock()
        data.inp = None
        mock_ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: mock_ax)
        result = plot_cross_sections(data)
        assert result is mock_ax

    def test_non_complex_returns_ax(self, monkeypatch):
        """Non-complex boundary_type -> return gca."""
        from emout.plot.plot_cross_sections import plot_cross_sections

        data = self._make_data(boundary_type="simple")
        mock_ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: mock_ax)
        result = plot_cross_sections(data)
        assert result is mock_ax

    def test_flat_surface_dispatches(self, monkeypatch):
        """flat-surface boundary dispatches to _handle_flat."""
        from emout.plot.plot_cross_sections import plot_cross_sections

        data = self._make_data(boundary_types=["flat-surface"])
        data.inp.zssurf = 10.0
        mock_ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: mock_ax)

        result = plot_cross_sections(data, axis="x", use_si=False)
        # For axis='x', a horizontal line is drawn
        mock_ax.axhline.assert_called_once_with(10.0)

    def test_flat_surface_z_axis_noop(self, monkeypatch):
        """flat-surface on z axis is a no-op."""
        from emout.plot.plot_cross_sections import plot_cross_sections

        data = self._make_data(boundary_types=["flat-surface"])
        data.inp.zssurf = 10.0
        mock_ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: mock_ax)

        result = plot_cross_sections(data, axis="z", use_si=False)
        mock_ax.axhline.assert_not_called()

    def test_rectangle_hole_y_axis(self, monkeypatch):
        """rectangle-hole on y axis draws outline."""
        from emout.plot.plot_cross_sections import plot_cross_sections

        data = self._make_data(boundary_types=["rectangle-hole"])
        data.inp.xlrechole = [0, 10]
        data.inp.xurechole = [0, 20]
        data.inp.ylrechole = [0, 5]
        data.inp.yurechole = [0, 15]
        data.inp.zlrechole = [0, 3]
        data.inp.zurechole = [0, 8]
        data.inp.nx = 30
        data.inp.ny = 25
        mock_ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: mock_ax)

        # coord=10 is within yl..yu = 5..15
        result = plot_cross_sections(data, axis="y", coord=10, use_si=False)
        mock_ax.plot.assert_called_once()

    def test_rectangle_hole_z_axis(self, monkeypatch):
        """rectangle-hole on z axis draws outline."""
        from emout.plot.plot_cross_sections import plot_cross_sections

        data = self._make_data(boundary_types=["rectangle-hole"])
        data.inp.xlrechole = [0, 10]
        data.inp.xurechole = [0, 20]
        data.inp.ylrechole = [0, 5]
        data.inp.yurechole = [0, 15]
        data.inp.zlrechole = [0, 3]
        data.inp.zurechole = [0, 8]
        data.inp.nx = 30
        data.inp.ny = 25
        mock_ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: mock_ax)

        # coord=5 is within zl..zu = 3..8
        result = plot_cross_sections(data, axis="z", coord=5, use_si=False)
        mock_ax.plot.assert_called_once()

    def test_rectangle_hole_y_axis_uses_converted_grid_endpoint(self, monkeypatch):
        """SI conversion should apply to ``nx - 1`` / ``ny - 1`` before plotting."""
        from emout.plot.plot_cross_sections import plot_cross_sections

        data = self._make_data(boundary_types=["rectangle-hole"])
        data.inp.xlrechole = [0, 10]
        data.inp.xurechole = [0, 20]
        data.inp.ylrechole = [0, 5]
        data.inp.yurechole = [0, 15]
        data.inp.zlrechole = [0, 3]
        data.inp.zurechole = [0, 8]
        data.inp.nx = 30
        data.inp.ny = 25
        data.unit.length.reverse = lambda x: x * 0.5
        mock_ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: mock_ax)

        plot_cross_sections(data, axis="y", coord=10, use_si=True)

        xs, zs = mock_ax.plot.call_args.args[:2]
        np.testing.assert_allclose(xs, np.array([0.0, 5.0, 5.0, 10.0, 10.0, 14.5]))
        np.testing.assert_allclose(zs, np.array([4.0, 4.0, 1.5, 1.5, 4.0, 4.0]))

    def test_rectangle_hole_outside_range(self, monkeypatch):
        """rectangle-hole with coord outside range draws nothing."""
        from emout.plot.plot_cross_sections import plot_cross_sections

        data = self._make_data(boundary_types=["rectangle-hole"])
        data.inp.xlrechole = [0, 10]
        data.inp.xurechole = [0, 20]
        data.inp.ylrechole = [0, 5]
        data.inp.yurechole = [0, 15]
        data.inp.zlrechole = [0, 3]
        data.inp.zurechole = [0, 8]
        data.inp.nx = 30
        data.inp.ny = 25
        mock_ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: mock_ax)

        # coord=100 outside any range
        result = plot_cross_sections(data, axis="z", coord=100, use_si=False)
        mock_ax.plot.assert_not_called()


# ===================================================================
# plot_cross_sections  --  _plot_spheres
# ===================================================================

class TestPlotSpheres:
    """Tests for _plot_spheres helper."""

    def test_sphere_in_range(self, monkeypatch):
        """Sphere within range adds a circle patch."""
        from emout.plot.plot_cross_sections import _plot_spheres

        mock_ax = MagicMock()
        monkeypatch.setattr(plt, "gca", lambda: mock_ax)

        # Sphere at (5,5,5) radius 3, z-plane at z=5 -> full circle
        result = _plot_spheres([(5, 5, 5, 3)], axis="z", coord=5, ax=mock_ax)
        mock_ax.add_patch.assert_called_once()

    def test_sphere_out_of_range(self, monkeypatch):
        """Sphere outside range adds no patch."""
        from emout.plot.plot_cross_sections import _plot_spheres

        mock_ax = MagicMock()
        _plot_spheres([(5, 5, 5, 3)], axis="z", coord=100, ax=mock_ax)
        mock_ax.add_patch.assert_not_called()

    def test_sphere_x_axis(self, monkeypatch):
        """Sphere cross-section along x axis."""
        from emout.plot.plot_cross_sections import _plot_spheres

        mock_ax = MagicMock()
        _plot_spheres([(5, 5, 5, 3)], axis="x", coord=5, ax=mock_ax)
        mock_ax.add_patch.assert_called_once()

    def test_sphere_y_axis(self, monkeypatch):
        """Sphere cross-section along y axis."""
        from emout.plot.plot_cross_sections import _plot_spheres

        mock_ax = MagicMock()
        _plot_spheres([(5, 5, 5, 3)], axis="y", coord=5, ax=mock_ax)
        mock_ax.add_patch.assert_called_once()

    def test_multiple_spheres(self, monkeypatch):
        """Multiple spheres in range all get patches."""
        from emout.plot.plot_cross_sections import _plot_spheres

        mock_ax = MagicMock()
        _plot_spheres([(5, 5, 5, 3), (10, 10, 5, 2)], axis="z", coord=5, ax=mock_ax)
        assert mock_ax.add_patch.call_count == 2


# ===================================================================
# contour3d  --  helper functions
# ===================================================================

class TestContour3dHelpers:
    """Tests for contour3d helper functions."""

    def test_as_spacing_scalar(self):
        from emout.plot.contour3d import _as_spacing_xyz
        assert _as_spacing_xyz(2.0) == (2.0, 2.0, 2.0)

    def test_as_spacing_tuple(self):
        from emout.plot.contour3d import _as_spacing_xyz
        assert _as_spacing_xyz((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)

    def test_as_spacing_invalid(self):
        from emout.plot.contour3d import _as_spacing_xyz
        with pytest.raises(ValueError, match="scalar"):
            _as_spacing_xyz((1.0, 2.0))

    def test_sanitize_volume_3d(self):
        from emout.plot.contour3d import _sanitize_volume
        vol = np.ones((3, 4, 5), dtype=int)
        result = _sanitize_volume(vol)
        assert np.issubdtype(result.dtype, np.floating)

    def test_sanitize_volume_nan(self):
        from emout.plot.contour3d import _sanitize_volume
        vol = np.ones((3, 4, 5))
        vol[0, 0, 0] = np.nan
        result = _sanitize_volume(vol)
        assert np.isfinite(result).all()

    def test_sanitize_volume_wrong_ndim(self):
        from emout.plot.contour3d import _sanitize_volume
        with pytest.raises(ValueError, match="3D"):
            _sanitize_volume(np.ones((3, 4)))

    def test_slice_from_bounds_1d_basic(self):
        from emout.plot.contour3d import _slice_from_bounds_1d
        sl, new_o = _slice_from_bounds_1d(1.0, 3.0, 0.0, 1.0, 10, "x")
        assert sl == slice(1, 4)
        assert new_o == pytest.approx(1.0)

    def test_slice_from_bounds_1d_none(self):
        from emout.plot.contour3d import _slice_from_bounds_1d
        sl, new_o = _slice_from_bounds_1d(None, None, 0.0, 1.0, 10, "x")
        assert sl == slice(0, 10)
        assert new_o == pytest.approx(0.0)

    def test_slice_from_bounds_1d_invalid(self):
        from emout.plot.contour3d import _slice_from_bounds_1d
        with pytest.raises(ValueError, match="invalid"):
            _slice_from_bounds_1d(5.0, 1.0, 0.0, 1.0, 10, "x")

    def test_slice_from_bounds_1d_too_small(self):
        from emout.plot.contour3d import _slice_from_bounds_1d
        with pytest.raises(ValueError, match="too small"):
            _slice_from_bounds_1d(0.0, 0.1, 0.0, 1.0, 10, "x")

    def test_apply_roi_none(self):
        from emout.plot.contour3d import _apply_roi
        vol = np.ones((5, 6, 7))
        dx = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        result_vol, result_dx, result_origin = _apply_roi(vol, dx, origin, None, None)
        assert result_vol.shape == (5, 6, 7)

    def test_apply_roi_with_roi_zyx(self):
        from emout.plot.contour3d import _apply_roi
        vol = np.ones((10, 10, 10))
        dx = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        roi = (slice(2, 5), slice(1, 4), slice(3, 8))
        result_vol, _, result_origin = _apply_roi(vol, dx, origin, None, roi)
        assert result_vol.shape == (3, 3, 5)
        assert result_origin == (3.0, 1.0, 2.0)

    def test_apply_roi_with_bounds(self):
        from emout.plot.contour3d import _apply_roi
        vol = np.ones((10, 10, 10))
        dx = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        bounds = ((2.0, 5.0), (1.0, 4.0), (3.0, 7.0))
        result_vol, _, _ = _apply_roi(vol, dx, origin, bounds, None)
        assert result_vol.shape[0] >= 2  # at least 2 samples along each axis


# ===================================================================
# contour3d  --  validation
# ===================================================================

class TestContour3dValidation:
    """Tests for contour3d input validation."""

    def test_empty_levels(self):
        from emout.plot.contour3d import contour3d
        with pytest.raises(ValueError, match="non-empty"):
            contour3d(np.ones((5, 5, 5)), dx=1.0, levels=[])

    def test_both_bounds_and_roi(self):
        from emout.plot.contour3d import contour3d
        with pytest.raises(ValueError, match="only one"):
            contour3d(
                np.ones((5, 5, 5)),
                dx=1.0,
                levels=[0.5],
                bounds_xyz=((None, None), (None, None), (None, None)),
                roi_zyx=(slice(None), slice(None), slice(None)),
            )

    def test_invalid_opacity(self):
        from emout.plot.contour3d import contour3d
        with pytest.raises(ValueError, match="opacity"):
            contour3d(np.ones((5, 5, 5)), dx=1.0, levels=[0.5], opacity=1.5)

    def test_invalid_step(self):
        from emout.plot.contour3d import contour3d
        with pytest.raises(ValueError, match="step"):
            contour3d(np.ones((5, 5, 5)), dx=1.0, levels=[0.5], step=0)

    def test_invalid_clabel_fontsize(self):
        from emout.plot.contour3d import contour3d
        with pytest.raises(ValueError, match="clabel_fontsize"):
            contour3d(
                np.ones((5, 5, 5)),
                dx=1.0,
                levels=[0.5],
                clabel=True,
                clabel_fontsize=-1,
            )


# ===================================================================
# contour3d  --  _format_level_value edge cases
# ===================================================================

class TestFormatLevelValueEdgeCases:
    """Additional edge-case tests for _format_level_value."""

    def test_callable_fmt(self):
        from emout.plot.contour3d import _format_level_value
        result = _format_level_value(3.14, fmt=lambda v: f"val={v:.1f}")
        assert result == "val=3.1"

    def test_format_mini_language(self):
        from emout.plot.contour3d import _format_level_value
        result = _format_level_value(3.14159, fmt=".3g")
        assert result == "3.14"

    def test_str_format_style(self):
        from emout.plot.contour3d import _format_level_value
        result = _format_level_value(3.14, fmt="{value:.2f}")
        assert result == "3.14"

    def test_default_no_fmt_no_sigfigs(self):
        from emout.plot.contour3d import _format_level_value
        result = _format_level_value(100.0)
        assert result == "100"

    def test_resolve_shared_exponent_invalid(self):
        from emout.plot.contour3d import _resolve_shared_exponent
        with pytest.raises(ValueError, match="auto"):
            _resolve_shared_exponent([1.0], "bogus")


# ===================================================================
# extra_plot  --  plot_energies
# ===================================================================

class TestExtraPlot:
    """Tests for extra_plot functions.

    Since these rely on heavy domain objects (Emout, InpFile) we test
    that the module can be imported and that basic validation works.
    """

    def test_module_imports(self):
        """extra_plot module imports cleanly."""
        import emout.plot.extra_plot  # noqa: F401

    def test_plot_surface_with_hole_exists(self):
        """plot_surface_with_hole function is importable."""
        from emout.plot.extra_plot import plot_surface_with_hole
        assert callable(plot_surface_with_hole)

    def test_plot_surface_with_hole_half_exists(self):
        """plot_surface_with_hole_half is importable."""
        from emout.plot.extra_plot import plot_surface_with_hole_half
        assert callable(plot_surface_with_hole_half)


# ===================================================================
# extra_plot  --  plot_hole_line
# ===================================================================

class TestPlotHoleLine:
    """Tests for extra_plot.plot_hole_line."""

    def _make_inp_mock(self, monkeypatch):
        """Create a mock InpFile that passes isinstance check."""
        from emout.utils.emsesinp import InpFile
        inp = MagicMock(spec=InpFile)
        inp.xlrechole = [0, 10]
        inp.xurechole = [0, 20]
        inp.ylrechole = [0, 5]
        inp.yurechole = [0, 15]
        inp.zlrechole = [0, 3]
        inp.zurechole = [8]
        inp.nx = 30
        inp.ny = 25
        inp.nz = 40
        return inp

    def test_basic_no_si(self, monkeypatch):
        """Plot hole outline without SI conversion."""
        from emout.plot.extra_plot import plot_hole_line

        plot_calls = []
        monkeypatch.setattr(plt, "xlim", lambda *a: (0, 10))
        monkeypatch.setattr(plt, "ylim", lambda *a: (0, 10))
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: plot_calls.append((a, kw)) or [MagicMock()])

        inp = self._make_inp_mock(monkeypatch)
        result = plot_hole_line(inp, use_si=False)
        assert len(plot_calls) == 1

    def test_with_si_conversion(self, monkeypatch):
        """Plot hole outline with SI conversion."""
        from emout.plot.extra_plot import plot_hole_line

        plot_calls = []
        monkeypatch.setattr(plt, "xlim", lambda *a: (0, 10))
        monkeypatch.setattr(plt, "ylim", lambda *a: (0, 10))
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: plot_calls.append((a, kw)) or [MagicMock()])

        unit = MagicMock()
        unit.length = MagicMock()
        unit.length.reverse = lambda x: x * 0.001

        inp = self._make_inp_mock(monkeypatch)
        result = plot_hole_line(inp, unit=unit, use_si=True)
        assert len(plot_calls) == 1

    def test_fix_lims_false(self, monkeypatch):
        """fix_lims=False skips xlim/ylim fixing."""
        from emout.plot.extra_plot import plot_hole_line

        lim_calls = [0]
        def _xlim(*a):
            lim_calls[0] += 1
            return (0, 10)
        monkeypatch.setattr(plt, "xlim", _xlim)
        monkeypatch.setattr(plt, "ylim", lambda *a: (0, 10))
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: [MagicMock()])

        inp = self._make_inp_mock(monkeypatch)
        plot_hole_line(inp, use_si=False, fix_lims=False)
        assert lim_calls[0] == 0

    def test_emout_data_input(self, monkeypatch):
        """Accepts Emout-like data object and extracts inp and unit."""
        from emout.plot.extra_plot import plot_hole_line

        monkeypatch.setattr(plt, "xlim", lambda *a: (0, 10))
        monkeypatch.setattr(plt, "ylim", lambda *a: (0, 10))
        monkeypatch.setattr(plt, "plot", lambda *a, **kw: [MagicMock()])

        inp = MagicMock()
        inp.xlrechole = [0, 10]
        inp.xurechole = [0, 20]
        inp.ylrechole = [0, 5]
        inp.yurechole = [0, 15]
        inp.zlrechole = [0, 3]
        inp.zurechole = [8]
        inp.nx = 30
        inp.ny = 25
        inp.nz = 40

        # NOT an InpFile instance -> else branch
        data = MagicMock()
        data.inp = inp
        data.unit = MagicMock()
        data.unit.length = MagicMock()
        data.unit.length.reverse = lambda x: x

        plot_hole_line(data, use_si=False)


# ===================================================================
# Smoke test: all public functions importable
# ===================================================================

class TestImports:
    """Verify that all expected public symbols are importable."""

    def test_plot_2d_imports(self):
        from emout.plot._plot_2d import (
            plot_2dmap,
            plot_2d_contour,
            plot_line,
            plot_surface,
            plot_2d_vector,
            plot_2d_streamline,
            figsize_with_2d,
        )

    def test_plot_3d_imports(self):
        from emout.plot._plot_3d import plot_3d_quiver, plot_3d_streamline

    def test_animation_imports(self):
        from emout.plot.animation_plot import (
            flatten_list,
            Animator,
            FrameUpdater,
        )

    def test_extra_plot_imports(self):
        from emout.plot.extra_plot import (
            plot_hole_line,
            plot_surface_with_hole,
            plot_surface_with_hole_half,
            plot_line_of_hole_half,
        )

    def test_cross_sections_imports(self):
        from emout.plot.plot_cross_sections import (
            plot_cross_sections,
            _plot_spheres,
        )

    def test_contour3d_imports(self):
        from emout.plot.contour3d import (
            contour3d,
            _as_spacing_xyz,
            _sanitize_volume,
            _format_level_value,
            _resolve_shared_exponent,
            _slice_from_bounds_1d,
            _apply_roi,
        )
