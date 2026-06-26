import numpy as np

from emout.core.data.data import Data2d, Data3d
from emout.core.data.vector_data import VectorData
import emout.plot.pyvista_plot as pvplot
import emout.plot._pyvista_scalar as pvscalar


class _FakePlotter:
    def __init__(self):
        self.meshes = []
        self.scalar_bar_title = None
        self.bounds_kwargs = None
        self.axes_added = False
        self.screenshot_path = None

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append((mesh, kwargs))

    def add_scalar_bar(self, title=None):
        self.scalar_bar_title = title

    def show_bounds(self, **kwargs):
        self.bounds_kwargs = kwargs

    def add_axes(self):
        self.axes_added = True

    def screenshot(self, filename):
        self.screenshot_path = filename


def test_data2d_plot_pyvista_delegates(monkeypatch):
    """Data2d.plot_pyvista が helper へ委譲されることを確認する。"""
    data2d = Data2d(np.zeros((4, 5)), name="phi", filename="dummy.h5")
    surfaces = object()
    calls = []

    def _fake_plot_scalar_plane(*args, **kwargs):
        calls.append(kwargs)
        return "plane-plotter"

    monkeypatch.setattr(pvplot, "plot_scalar_plane", _fake_plot_scalar_plane)

    assert data2d.plot_pyvista(show=False, surfaces=surfaces) == "plane-plotter"
    assert calls[-1]["surfaces"] is surfaces
    assert data2d.plot3d(show=False) == "plane-plotter"


def test_plot_scalar_plane_saves_filename_without_mesh_kwarg(monkeypatch, tmp_path):
    plotter = _FakePlotter()
    mesh = object()
    filename = tmp_path / "slice.png"

    monkeypatch.setattr(
        pvscalar,
        "create_plane_mesh",
        lambda *args, **kwargs: (mesh, "phi", {"x": "x", "y": "y", "z": "z"}, "phi"),
    )

    returned = pvscalar.plot_scalar_plane(
        object(),
        plotter=plotter,
        filename=filename,
        show_edges=True,
    )

    assert returned is plotter
    assert plotter.screenshot_path == str(filename)
    assert plotter.meshes[0][0] is mesh
    assert plotter.meshes[0][1]["show_edges"] is True
    assert "filename" not in plotter.meshes[0][1]


def test_data3d_plot_pyvista_delegates(monkeypatch):
    """Data3d.plot_pyvista が helper へ委譲されることを確認する。"""
    data3d = Data3d(np.zeros((3, 4, 5)), name="phi", filename="dummy.h5")
    calls = []

    def _fake_plot_scalar_volume(*args, **kwargs):
        calls.append(kwargs)
        return "volume-plotter"

    monkeypatch.setattr(pvplot, "plot_scalar_volume", _fake_plot_scalar_volume)

    levels = [0.25, 0.5]
    surfaces = object()
    assert data3d.plot_pyvista(mode="contour", levels=levels, surfaces=surfaces, show=False) == "volume-plotter"
    assert calls[-1]["contour_levels"] is levels
    assert calls[-1]["surfaces"] is surfaces
    assert data3d.plot3d(show=False) == "volume-plotter"


def test_vector3d_plot_pyvista_modes(monkeypatch):
    """VectorData.plot_pyvista の stream/quiver 切替を確認する。"""
    vx = Data3d(np.zeros((3, 4, 5)), name="vx", filename="dummy.h5")
    vy = Data3d(np.zeros((3, 4, 5)), name="vy", filename="dummy.h5")
    vz = Data3d(np.zeros((3, 4, 5)), name="vz", filename="dummy.h5")
    vec = VectorData([vx, vy, vz], name="vxyz")

    monkeypatch.setattr(
        pvplot,
        "plot_vector_streamlines3d",
        lambda *args, **kwargs: "stream-plotter",
    )
    monkeypatch.setattr(
        pvplot,
        "plot_vector_quiver3d",
        lambda *args, **kwargs: "quiver-plotter",
    )

    assert vec.plot_pyvista(mode="stream", show=False) == "stream-plotter"
    assert vec.plot_pyvista(mode="quiver", show=False) == "quiver-plotter"


def test_vector_plot_dispatches_to_plot3d(monkeypatch):
    """VectorData.plot が 3D の場合に plot3d へ委譲することを確認する。"""
    vx = Data3d(np.zeros((3, 4, 5)), name="vx", filename="dummy.h5")
    vy = Data3d(np.zeros((3, 4, 5)), name="vy", filename="dummy.h5")
    vz = Data3d(np.zeros((3, 4, 5)), name="vz", filename="dummy.h5")
    vec = VectorData([vx, vy, vz], name="vxyz")

    monkeypatch.setattr(VectorData, "plot3d", lambda *args, **kwargs: "delegated")

    assert vec.plot(mode="stream", show=False) == "delegated"
