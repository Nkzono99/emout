"""Tests for VectorData plot dispatch and argument handling.

Focuses on the uncovered plot methods: plot, plot2d, plot3d_mpl,
plot_pyvista, plot3d, gifplot, and build_frame_updater.
"""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from emout.core.data._data2d import Data2d
from emout.core.data._data3d import Data3d
from emout.core.data.vector_data import VectorData
from emout.utils import UnitTranslator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit():
    """Create a minimal UnitTranslator for testing."""
    return UnitTranslator(1.0, 1.0, name="length", unit="m")


def _make_val_unit(unit_str="V/m"):
    """Create a minimal value UnitTranslator."""
    return UnitTranslator(1.0, 1.0, name="field", unit=unit_str)


def _make_2d_vec(shape=(4, 5), name="v", with_units=False):
    """Create a 2-component 2-D VectorData for testing."""
    arr_x = np.random.RandomState(0).rand(*shape).astype(np.float32)
    arr_y = np.random.RandomState(1).rand(*shape).astype(np.float32)

    kwargs = dict(name="vx")
    if with_units:
        au = [_make_unit()] * 4  # t, z, y, x
        vu = _make_val_unit("V/m")
        kwargs["axisunits"] = au
        kwargs["valunit"] = vu

    vx = Data2d(arr_x, **kwargs)
    vy = Data2d(arr_y, name="vy",
                **({k: v for k, v in kwargs.items() if k != "name"}))
    return VectorData([vx, vy], name=name)


def _make_3d_vec(shape=(3, 4, 5), name="B", with_units=False):
    """Create a 3-component 3-D VectorData for testing."""
    rng = np.random.RandomState(42)
    arrs = [rng.rand(*shape).astype(np.float32) for _ in range(3)]

    kwargs = {}
    if with_units:
        au = [_make_unit()] * 4
        vu = _make_val_unit("T")
        kwargs["axisunits"] = au
        kwargs["valunit"] = vu

    names = ["Bx", "By", "Bz"]
    comps = [Data3d(a, name=n, **kwargs) for a, n in zip(arrs, names)]
    return VectorData(comps, name=name)


# ---------------------------------------------------------------------------
# Construction & basic properties
# ---------------------------------------------------------------------------

class TestVectorDataConstruction:
    def test_requires_2_or_3_components(self):
        vx = Data2d(np.zeros((3, 4)), name="vx")
        with pytest.raises(ValueError, match="2 or 3"):
            VectorData([vx])

    def test_2d_vector_properties(self):
        vec = _make_2d_vec()
        assert vec.shape == (4, 5)
        assert vec.ndim == 2
        assert vec.name == "v"
        assert len(vec.objs) == 2

    def test_3d_vector_properties(self):
        vec = _make_3d_vec()
        assert vec.shape == (3, 4, 5)
        assert vec.ndim == 3
        assert vec.name == "B"
        assert len(vec.objs) == 3
        assert hasattr(vec, "z_data")

    def test_repr(self):
        vec = _make_2d_vec()
        r = repr(vec)
        assert "VectorData" in r
        assert "components=2" in r

    def test_negate(self):
        vec = _make_2d_vec()
        neg = vec.negate()
        np.testing.assert_array_almost_equal(
            np.array(neg.x_data), -np.array(vec.x_data)
        )

    def test_scale(self):
        vec = _make_2d_vec()
        scaled = vec.scale(3.0)
        np.testing.assert_array_almost_equal(
            np.array(scaled.x_data), np.array(vec.x_data) * 3.0
        )

    def test_4_components_raises(self):
        arr = np.zeros((3, 4), dtype=np.float32)
        comps = [Data2d(arr, name=f"v{i}") for i in range(4)]
        with pytest.raises(ValueError, match="2 or 3"):
            VectorData(comps)


# ---------------------------------------------------------------------------
# plot() dispatch
# ---------------------------------------------------------------------------

class TestPlotDispatch:
    """Test that plot() dispatches to plot2d or plot3d_mpl based on ndim."""

    def test_2d_dispatches_to_plot2d(self, monkeypatch):
        vec = _make_2d_vec()
        mock = MagicMock(return_value="plot2d_result")
        monkeypatch.setattr(VectorData, "plot2d", mock)
        result = vec.plot(mode="vec", show=False)
        mock.assert_called_once()
        assert result == "plot2d_result"

    def test_3d_dispatches_to_plot3d_mpl(self, monkeypatch):
        vec = _make_3d_vec()
        mock = MagicMock(return_value="plot3d_result")
        monkeypatch.setattr(VectorData, "plot3d_mpl", mock)
        result = vec.plot(mode="vec", show=False)
        mock.assert_called_once()
        assert result == "plot3d_result"

    def test_1d_raises(self):
        """1-D data should raise NotImplementedError."""
        from emout.core.data._data1d import Data1d
        d1 = Data1d(np.zeros(5), name="v1")
        d2 = Data1d(np.zeros(5), name="v2")
        vec = VectorData([d1, d2], name="v")
        with pytest.raises(NotImplementedError, match="ndim=1"):
            vec.plot()

    def test_plot_forwards_kwargs(self, monkeypatch):
        """Extra kwargs should be forwarded to the delegated method."""
        vec = _make_2d_vec()
        mock = MagicMock(return_value="img")
        monkeypatch.setattr(VectorData, "plot2d", mock)
        vec.plot(mode="vec", show=False, density=5, custom_arg="hello")
        call_kwargs = mock.call_args
        assert "density" in call_kwargs.kwargs or "density" in dict(zip(
            ["mode", "show", "density", "custom_arg"],
            call_kwargs.args[1:] if len(call_kwargs.args) > 1 else []
        ))


# ---------------------------------------------------------------------------
# plot2d
# ---------------------------------------------------------------------------

class TestPlot2d:
    """Test plot2d argument parsing and dispatch."""

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="stream_img")
    def test_stream_mode(self, mock_stream):
        vec = _make_2d_vec()
        result = vec.plot2d(mode="stream", use_si=False)
        mock_stream.assert_called_once()
        assert result == "stream_img"

    @patch("emout.plot.basic_plot.plot_2d_vector", return_value="vec_img")
    def test_vec_mode(self, mock_vec):
        vec = _make_2d_vec()
        result = vec.plot2d(mode="vec", use_si=False)
        mock_vec.assert_called_once()
        assert result == "vec_img"

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_auto_axes(self, mock_stream):
        vec = _make_2d_vec()
        vec.plot2d(mode="stream", axes="auto", use_si=False)
        mock_stream.assert_called_once()

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_explicit_axes_xy(self, mock_stream):
        vec = _make_2d_vec()
        vec.plot2d(mode="stream", axes="xy", use_si=False)
        mock_stream.assert_called_once()

    def test_invalid_axes_raises(self):
        vec = _make_2d_vec()
        with pytest.raises(ValueError, match="cannot be used"):
            vec.plot2d(axes="ab", use_si=False)

    def test_nonexistent_axis_raises(self):
        vec = _make_2d_vec()
        # Data2d with default slice_axes=[2,3] -> use_axes=['y','x']
        # 'tz' requires 't' and 'z' which don't exist
        with pytest.raises(ValueError, match="does not exist"):
            vec.plot2d(axes="tz", use_si=False)

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_show_true_returns_none(self, mock_stream):
        vec = _make_2d_vec()
        with patch("matplotlib.pyplot.show"):
            result = vec.plot2d(mode="stream", show=True, use_si=False)
        assert result is None

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_use_si_with_units(self, mock_stream):
        vec = _make_2d_vec(with_units=True)
        vec.plot2d(mode="stream", use_si=True)
        call_kwargs = mock_stream.call_args[1]
        assert "xlabel" in call_kwargs
        assert "[m]" in call_kwargs["xlabel"]

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_use_si_false_when_no_valunit(self, mock_stream):
        """When valunit is None, use_si should be forced to False."""
        vec = _make_2d_vec(with_units=False)
        vec.plot2d(mode="stream", use_si=True)  # should not crash
        call_kwargs = mock_stream.call_args[1]
        # Without units, labels should be plain axis names
        assert "[" not in call_kwargs.get("xlabel", "")

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_offsets_applied(self, mock_stream):
        vec = _make_2d_vec()
        vec.plot2d(mode="stream", use_si=False, offsets=("left", "center", 0.0))
        mock_stream.assert_called_once()

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_custom_kwargs_forwarded(self, mock_stream):
        vec = _make_2d_vec()
        vec.plot2d(mode="stream", use_si=False, density=2, title="custom_title")
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["density"] == 2
        assert call_kwargs["title"] == "custom_title"

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_mesh_passed_to_plotter(self, mock_stream):
        """The mesh argument should be computed and passed."""
        vec = _make_2d_vec()
        vec.plot2d(mode="stream", use_si=False)
        call_kwargs = mock_stream.call_args[1]
        assert "mesh" in call_kwargs
        mesh = call_kwargs["mesh"]
        assert len(mesh) == 2  # meshgrid produces 2 arrays

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_si_labels_contain_unit(self, mock_stream):
        """With SI units, labels should include the unit string."""
        vec = _make_2d_vec(with_units=True)
        vec.plot2d(mode="stream", use_si=True)
        call_kwargs = mock_stream.call_args[1]
        assert "[m]" in call_kwargs["xlabel"]
        assert "[m]" in call_kwargs["ylabel"]
        assert "[V/m]" in call_kwargs["title"]

    @patch("emout.plot.basic_plot.plot_2d_streamline", return_value="img")
    def test_no_si_labels_are_plain(self, mock_stream):
        """Without SI, labels should be plain axis names."""
        vec = _make_2d_vec()
        vec.plot2d(mode="stream", use_si=False)
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["xlabel"] in ("x", "y")
        assert call_kwargs["ylabel"] in ("x", "y")

    @patch("emout.plot.basic_plot.plot_2d_vector", return_value="img")
    def test_explicit_xlabel_overrides(self, mock_vec):
        """User-provided xlabel should override the default."""
        vec = _make_2d_vec()
        vec.plot2d(mode="vec", use_si=False, xlabel="custom X")
        call_kwargs = mock_vec.call_args[1]
        assert call_kwargs["xlabel"] == "custom X"


# ---------------------------------------------------------------------------
# plot3d_mpl
# ---------------------------------------------------------------------------

class TestPlot3dMpl:
    """Test plot3d_mpl argument parsing and dispatch."""

    @patch("emout.plot.basic_plot.plot_3d_streamline", return_value="ax3d")
    def test_stream_mode(self, mock_stream):
        vec = _make_3d_vec()
        result = vec.plot3d_mpl(mode="stream", use_si=False)
        mock_stream.assert_called_once()
        assert result == "ax3d"

    @patch("emout.plot.basic_plot.plot_3d_quiver", return_value="ax3d")
    def test_vec_mode(self, mock_quiver):
        vec = _make_3d_vec()
        result = vec.plot3d_mpl(mode="vec", use_si=False)
        mock_quiver.assert_called_once()
        assert result == "ax3d"

    @patch("emout.plot.basic_plot.plot_3d_quiver", return_value="ax3d")
    def test_quiver_alias(self, mock_quiver):
        vec = _make_3d_vec()
        vec.plot3d_mpl(mode="quiver", use_si=False)
        mock_quiver.assert_called_once()

    @patch("emout.plot.basic_plot.plot_3d_streamline", return_value="ax3d")
    def test_streamline_alias(self, mock_stream):
        vec = _make_3d_vec()
        vec.plot3d_mpl(mode="streamline", use_si=False)
        mock_stream.assert_called_once()

    def test_unsupported_mode_raises(self):
        vec = _make_3d_vec()
        with pytest.raises(ValueError, match="Unsupported mode"):
            vec.plot3d_mpl(mode="scatter", use_si=False)

    def test_2d_data_raises(self):
        vec = _make_2d_vec()
        with pytest.raises(ValueError, match="3-D"):
            vec.plot3d_mpl(use_si=False)

    def test_2_component_raises(self):
        """plot3d_mpl requires 3 components."""
        arr = np.random.rand(3, 4, 5).astype(np.float32)
        vx = Data3d(arr, name="vx")
        vy = Data3d(arr, name="vy")
        vec = VectorData([vx, vy], name="v")
        with pytest.raises(ValueError, match="3 components"):
            vec.plot3d_mpl(use_si=False)

    @patch("emout.plot.basic_plot.plot_3d_streamline", return_value="ax3d")
    def test_use_si_with_units(self, mock_stream):
        vec = _make_3d_vec(with_units=True)
        vec.plot3d_mpl(mode="stream", use_si=True)
        call_kwargs = mock_stream.call_args[1]
        assert "[m]" in call_kwargs["xlabel"]
        assert "[T]" in call_kwargs["title"]

    @patch("emout.plot.basic_plot.plot_3d_streamline", return_value="ax3d")
    def test_use_si_false_labels(self, mock_stream):
        vec = _make_3d_vec()
        vec.plot3d_mpl(mode="stream", use_si=False)
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["xlabel"] == "x"
        assert call_kwargs["ylabel"] == "y"
        assert call_kwargs["zlabel"] == "z"

    @patch("emout.plot.basic_plot.plot_3d_streamline", return_value="ax3d")
    def test_offsets_applied(self, mock_stream):
        vec = _make_3d_vec()
        vec.plot3d_mpl(mode="stream", use_si=False,
                       offsets=("left", "center", "right"))
        mock_stream.assert_called_once()
        call_kwargs = mock_stream.call_args[1]
        # mesh should be provided
        assert "mesh" in call_kwargs

    @patch("emout.plot.basic_plot.plot_3d_streamline", return_value="ax3d")
    def test_custom_kwargs_forwarded(self, mock_stream):
        vec = _make_3d_vec()
        vec.plot3d_mpl(mode="stream", use_si=False, n_seeds=10, title="custom")
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["n_seeds"] == 10
        assert call_kwargs["title"] == "custom"

    @patch("emout.plot.basic_plot.plot_3d_streamline", return_value="ax3d")
    def test_mesh_has_three_components(self, mock_stream):
        """The mesh tuple should have 3 arrays for 3-D data."""
        vec = _make_3d_vec()
        vec.plot3d_mpl(mode="stream", use_si=False)
        call_kwargs = mock_stream.call_args[1]
        mesh = call_kwargs["mesh"]
        assert len(mesh) == 3

    @patch("emout.plot.basic_plot.plot_3d_streamline", return_value="ax3d")
    def test_si_labels_all_axes(self, mock_stream):
        """With SI, xlabel/ylabel/zlabel should all contain [m]."""
        vec = _make_3d_vec(with_units=True)
        vec.plot3d_mpl(mode="stream", use_si=True)
        call_kwargs = mock_stream.call_args[1]
        assert "[m]" in call_kwargs["xlabel"]
        assert "[m]" in call_kwargs["ylabel"]
        assert "[m]" in call_kwargs["zlabel"]

    @patch("emout.plot.basic_plot.plot_3d_streamline", return_value="ax3d")
    def test_no_valunit_forces_si_false(self, mock_stream):
        """When valunit is None, use_si should be forced to False."""
        vec = _make_3d_vec(with_units=False)
        vec.plot3d_mpl(mode="stream", use_si=True)
        call_kwargs = mock_stream.call_args[1]
        # Without units, labels should be plain axis names
        assert call_kwargs["xlabel"] == "x"

    @patch("emout.plot.basic_plot.plot_3d_quiver", return_value="ax3d")
    def test_ax_parameter_forwarded(self, mock_quiver):
        """The ax parameter should be forwarded to the plot function."""
        vec = _make_3d_vec()
        mock_ax = MagicMock()
        vec.plot3d_mpl(mode="quiver", use_si=False, ax=mock_ax)
        call_kwargs = mock_quiver.call_args[1]
        assert call_kwargs["ax3d"] is mock_ax


# ---------------------------------------------------------------------------
# plot_pyvista
# ---------------------------------------------------------------------------

class TestPlotPyvista:
    """Test plot_pyvista argument validation and dispatch."""

    def test_2d_data_raises(self):
        vec = _make_2d_vec()
        with pytest.raises(ValueError, match="3D"):
            vec.plot_pyvista()

    def test_2_component_raises(self):
        arr = np.random.rand(3, 4, 5).astype(np.float32)
        vx = Data3d(arr, name="vx")
        vy = Data3d(arr, name="vy")
        vec = VectorData([vx, vy], name="v")
        with pytest.raises(ValueError, match="3 components"):
            vec.plot_pyvista()

    def test_unsupported_mode_raises(self):
        vec = _make_3d_vec()
        with pytest.raises(ValueError, match="Unsupported mode"):
            vec.plot_pyvista(mode="scatter", use_si=False)

    @patch("emout.plot.pyvista_plot.plot_vector_quiver3d", return_value="pv_result")
    def test_vec_mode_dispatch(self, mock_quiver):
        vec = _make_3d_vec()
        result = vec.plot_pyvista(mode="vec", use_si=False)
        mock_quiver.assert_called_once()
        assert result == "pv_result"

    @patch("emout.plot.pyvista_plot.plot_vector_quiver3d", return_value="pv_result")
    def test_quiver_mode_dispatch(self, mock_quiver):
        vec = _make_3d_vec()
        result = vec.plot_pyvista(mode="quiver", use_si=False)
        mock_quiver.assert_called_once()

    @patch("emout.plot.pyvista_plot.plot_vector_streamlines3d", return_value="pv_result")
    def test_stream_mode_dispatch(self, mock_stream):
        vec = _make_3d_vec()
        result = vec.plot_pyvista(mode="stream", use_si=False)
        mock_stream.assert_called_once()

    @patch("emout.plot.pyvista_plot.plot_vector_streamlines3d", return_value="pv_result")
    def test_streamline_mode_dispatch(self, mock_stream):
        vec = _make_3d_vec()
        result = vec.plot_pyvista(mode="streamline", use_si=False)
        mock_stream.assert_called_once()

    @patch("emout.plot.pyvista_plot.plot_vector_quiver3d", return_value="pv_result")
    def test_no_valunit_forces_si_false(self, mock_quiver):
        """When valunit is None, use_si should be forced to False."""
        vec = _make_3d_vec(with_units=False)
        vec.plot_pyvista(mode="vec", use_si=True)
        call_kwargs = mock_quiver.call_args[1]
        assert call_kwargs["use_si"] is False

    @patch("emout.plot.pyvista_plot.plot_vector_quiver3d", return_value="pv_result")
    def test_plotter_forwarded(self, mock_quiver):
        """The plotter parameter should be forwarded."""
        vec = _make_3d_vec()
        mock_plotter = MagicMock()
        vec.plot_pyvista(mode="vec", use_si=False, plotter=mock_plotter)
        call_kwargs = mock_quiver.call_args[1]
        assert call_kwargs["plotter"] is mock_plotter

    @patch("emout.plot.pyvista_plot.plot_vector_streamlines3d", return_value="pv_result")
    def test_extra_kwargs_forwarded(self, mock_stream):
        """Extra kwargs should be forwarded to the pyvista plot function."""
        vec = _make_3d_vec()
        vec.plot_pyvista(mode="stream", use_si=False, n_points=100)
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["n_points"] == 100


# ---------------------------------------------------------------------------
# plot3d (backend dispatch)
# ---------------------------------------------------------------------------

class TestPlot3d:
    """Test plot3d backend dispatch."""

    def test_mpl_backend(self, monkeypatch):
        vec = _make_3d_vec()
        mock = MagicMock(return_value="mpl")
        monkeypatch.setattr(VectorData, "plot3d_mpl", mock)
        result = vec.plot3d(mode="stream", backend="mpl")
        mock.assert_called_once()
        assert result == "mpl"
        assert mock.call_args[1]["mode"] == "stream"

    def test_pyvista_backend(self, monkeypatch):
        vec = _make_3d_vec()
        mock = MagicMock(return_value="pv")
        monkeypatch.setattr(VectorData, "plot_pyvista", mock)
        result = vec.plot3d(mode="vec", backend="pyvista")
        mock.assert_called_once()
        assert result == "pv"
        assert mock.call_args[1]["mode"] == "vec"

    def test_default_backend_is_mpl(self, monkeypatch):
        vec = _make_3d_vec()
        mock = MagicMock(return_value="mpl")
        monkeypatch.setattr(VectorData, "plot3d_mpl", mock)
        vec.plot3d(mode="vec")
        mock.assert_called_once()

    def test_kwargs_forwarded_to_mpl(self, monkeypatch):
        """Extra kwargs should be forwarded to plot3d_mpl."""
        vec = _make_3d_vec()
        mock = MagicMock(return_value="mpl")
        monkeypatch.setattr(VectorData, "plot3d_mpl", mock)
        vec.plot3d(mode="stream", backend="mpl", n_seeds=5)
        call_kwargs = mock.call_args[1]
        assert call_kwargs["n_seeds"] == 5

    def test_kwargs_forwarded_to_pyvista(self, monkeypatch):
        """Extra kwargs should be forwarded to plot_pyvista."""
        vec = _make_3d_vec()
        mock = MagicMock(return_value="pv")
        monkeypatch.setattr(VectorData, "plot_pyvista", mock)
        vec.plot3d(mode="vec", backend="pyvista", show=True)
        call_kwargs = mock.call_args[1]
        assert call_kwargs["show"] is True


# ---------------------------------------------------------------------------
# build_frame_updater
# ---------------------------------------------------------------------------

class TestBuildFrameUpdater:
    """Test build_frame_updater returns a FrameUpdater."""

    def test_returns_frame_updater(self):
        from emout.plot.animation_plot import FrameUpdater

        # Need a 3-D VectorData with an extra leading axis
        rng = np.random.RandomState(0)
        shape_3d = (3, 4, 5)
        comps = [Data3d(rng.rand(*shape_3d).astype(np.float32), name=n)
                 for n in ("Bx", "By", "Bz")]
        vec = VectorData(comps, name="B")

        updater = vec.build_frame_updater(axis=0, use_si=False)
        assert isinstance(updater, FrameUpdater)
        assert updater.data is vec
        assert updater.axis == 0

    def test_title_override(self):
        rng = np.random.RandomState(0)
        shape_3d = (3, 4, 5)
        comps = [Data3d(rng.rand(*shape_3d).astype(np.float32), name=n)
                 for n in ("Bx", "By", "Bz")]
        vec = VectorData(comps, name="B")
        updater = vec.build_frame_updater(title="custom", use_si=False)
        assert updater.title == "custom"

    def test_notitle_flag(self):
        rng = np.random.RandomState(0)
        shape_3d = (3, 4, 5)
        comps = [Data3d(rng.rand(*shape_3d).astype(np.float32), name=n)
                 for n in ("Bx", "By", "Bz")]
        vec = VectorData(comps, name="B")
        updater = vec.build_frame_updater(notitle=True, use_si=False)
        assert updater.notitle is True

    def test_default_title_uses_name(self):
        """When title is None, it should use the data's name."""
        rng = np.random.RandomState(0)
        shape_3d = (3, 4, 5)
        comps = [Data3d(rng.rand(*shape_3d).astype(np.float32), name=n)
                 for n in ("Bx", "By", "Bz")]
        vec = VectorData(comps, name="B")
        updater = vec.build_frame_updater(title=None, use_si=False)
        assert updater.title == "B"

    def test_offsets_stored(self):
        rng = np.random.RandomState(0)
        shape_3d = (3, 4, 5)
        comps = [Data3d(rng.rand(*shape_3d).astype(np.float32), name=n)
                 for n in ("Bx", "By", "Bz")]
        vec = VectorData(comps, name="B")
        offsets = ("left", "center", "right")
        updater = vec.build_frame_updater(offsets=offsets, use_si=False)
        assert updater.offsets == offsets

    def test_kwargs_stored(self):
        """Extra kwargs should be stored and forwarded during animation."""
        rng = np.random.RandomState(0)
        shape_3d = (3, 4, 5)
        comps = [Data3d(rng.rand(*shape_3d).astype(np.float32), name=n)
                 for n in ("Bx", "By", "Bz")]
        vec = VectorData(comps, name="B")
        updater = vec.build_frame_updater(use_si=False, mode="vec")
        assert updater.kwargs.get("mode") == "vec"


# ---------------------------------------------------------------------------
# gifplot
# ---------------------------------------------------------------------------

class TestGifplot:
    """Test gifplot argument handling and FrameUpdater creation."""

    def _make_vec(self):
        rng = np.random.RandomState(0)
        shape = (3, 4, 5)
        comps = [Data3d(rng.rand(*shape).astype(np.float32), name=n)
                 for n in ("Bx", "By", "Bz")]
        return VectorData(comps, name="B")

    def test_return_updater_deprecated(self):
        vec = self._make_vec()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = vec.gifplot(return_updater=True, use_si=False)
            # Should emit DeprecationWarning
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "return_updater" in str(dep_warnings[0].message)

        from emout.plot.animation_plot import FrameUpdater
        assert isinstance(result, FrameUpdater)

    def test_action_frames(self):
        vec = self._make_vec()
        from emout.plot.animation_plot import FrameUpdater
        result = vec.gifplot(action="frames", use_si=False)
        assert isinstance(result, FrameUpdater)

    def test_action_return_calls_animator(self):
        vec = self._make_vec()
        with patch("emout.plot.animation_plot.Animator.plot",
                   return_value=("fig", "ani")) as mock_plot:
            result = vec.gifplot(action="return", use_si=False)
        mock_plot.assert_called_once()
        assert result == ("fig", "ani")

    def test_gifplot_axis_parameter(self):
        """axis parameter should be passed to FrameUpdater."""
        vec = self._make_vec()
        result = vec.gifplot(action="frames", axis=1, use_si=False)
        assert result.axis == 1

    def test_gifplot_title_parameter(self):
        """title parameter should be passed to FrameUpdater."""
        vec = self._make_vec()
        result = vec.gifplot(action="frames", title="My Title", use_si=False)
        assert result.title == "My Title"

    def test_gifplot_notitle_parameter(self):
        """notitle parameter should be passed to FrameUpdater."""
        vec = self._make_vec()
        result = vec.gifplot(action="frames", notitle=True, use_si=False)
        assert result.notitle is True

    def test_gifplot_kwargs_forwarded(self):
        """Extra kwargs should be forwarded through to FrameUpdater."""
        vec = self._make_vec()
        result = vec.gifplot(action="frames", use_si=False, mode="vec")
        assert result.kwargs.get("mode") == "vec"

    def test_animator_plot_called_with_params(self):
        """Animator.plot should receive the interval, repeat, etc."""
        vec = self._make_vec()
        with patch("emout.plot.animation_plot.Animator.plot",
                   return_value=("fig", "ani")) as mock_plot:
            vec.gifplot(action="return", use_si=False,
                        interval=100, repeat=False)
        call_kwargs = mock_plot.call_args[1]
        assert call_kwargs["interval"] == 100
        assert call_kwargs["repeat"] is False


# ---------------------------------------------------------------------------
# VectorData2d / VectorData3d aliases
# ---------------------------------------------------------------------------

class TestAliases:
    def test_vectordata2d_alias(self):
        from emout.core.data.vector_data import VectorData2d
        assert VectorData2d is VectorData

    def test_vectordata3d_alias(self):
        from emout.core.data.vector_data import VectorData3d
        assert VectorData3d is VectorData


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_name_from_attrs(self):
        """If name is not given, it should be pulled from attrs."""
        arr = np.zeros((4, 5), dtype=np.float32)
        vx = Data2d(arr, name="vx")
        vy = Data2d(arr, name="vy")
        vec = VectorData([vx, vy], attrs={"name": "field"})
        assert vec.name == "field"

    def test_name_from_component(self):
        """If no name and no attrs name, use the first component's name."""
        arr = np.zeros((4, 5), dtype=np.float32)
        vx = Data2d(arr, name="vx")
        vy = Data2d(arr, name="vy")
        vec = VectorData([vx, vy])
        assert vec.name == "vx"

    def test_name_empty_fallback(self):
        """If nothing has a name, default to empty string."""
        arr = np.zeros((4, 5), dtype=np.float32)
        vx = Data2d(arr)
        vy = Data2d(arr)
        # Data with name=None: hasattr(vx, "name") is True.
        # VectorData.__init__ checks hasattr -> True, uses x_data.name (None).
        # The attrs["name"] is set to None in this case.
        vec = VectorData([vx, vy])
        # name may be None when the component name is None
        assert vec.name is None or vec.name == ""

    def test_setattr_component_data(self):
        """Setting x_data/y_data/z_data should use __dict__ directly."""
        vec = _make_3d_vec()
        new_data = Data3d(np.ones((3, 4, 5), dtype=np.float32), name="new")
        vec.x_data = new_data
        assert vec.x_data is new_data

    def test_slice_axes_delegates(self):
        vec = _make_2d_vec()
        np.testing.assert_array_equal(vec.slice_axes, vec.objs[0].slice_axes)

    def test_slices_delegates(self):
        vec = _make_2d_vec()
        for v_s, c_s in zip(vec.slices, vec.objs[0].slices):
            assert v_s == c_s

    def test_valunit_delegates(self):
        vec = _make_2d_vec(with_units=True)
        assert vec.valunit is vec.objs[0].valunit

    def test_axisunits_delegates(self):
        vec = _make_2d_vec(with_units=True)
        assert vec.axisunits is vec.objs[0].axisunits

    def test_negate_preserves_name(self):
        """Negate should preserve the vector field name."""
        vec = _make_2d_vec(name="E")
        neg = vec.negate()
        assert neg.name == "E"

    def test_scale_preserves_name(self):
        """Scale should preserve the vector field name."""
        vec = _make_2d_vec(name="E")
        scaled = vec.scale(2.0)
        assert scaled.name == "E"

    def test_negate_3d(self):
        """Negate should work on 3-component VectorData."""
        vec = _make_3d_vec()
        neg = vec.negate()
        np.testing.assert_array_almost_equal(
            np.array(neg.z_data), -np.array(vec.z_data)
        )

    def test_scale_3d(self):
        """Scale should work on 3-component VectorData."""
        vec = _make_3d_vec()
        scaled = vec.scale(0.5)
        np.testing.assert_array_almost_equal(
            np.array(scaled.z_data), np.array(vec.z_data) * 0.5
        )

    def test_shape_delegates_to_first_component(self):
        """Shape should match the first component."""
        vec = _make_3d_vec(shape=(2, 6, 8))
        assert vec.shape == (2, 6, 8)

    def test_ndim_delegates(self):
        """ndim should match the first component."""
        vec = _make_2d_vec()
        assert vec.ndim == 2
        vec3 = _make_3d_vec()
        assert vec3.ndim == 3
