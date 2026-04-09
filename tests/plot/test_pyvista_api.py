import numpy as np

from emout.core.data.data import Data2d, Data3d
from emout.core.data.vector_data import VectorData
import emout.plot.pyvista_plot as pvplot


def test_data2d_plot_pyvista_delegates(monkeypatch):
    """Data2d.plot_pyvista が helper へ委譲されることを確認する。"""
    data2d = Data2d(np.zeros((4, 5)), name="phi", filename="dummy.h5")

    def _fake_plot_scalar_plane(*args, **kwargs):
        return "plane-plotter"

    monkeypatch.setattr(pvplot, "plot_scalar_plane", _fake_plot_scalar_plane)

    assert data2d.plot_pyvista(show=False) == "plane-plotter"
    assert data2d.plot3d(show=False) == "plane-plotter"


def test_data3d_plot_pyvista_delegates(monkeypatch):
    """Data3d.plot_pyvista が helper へ委譲されることを確認する。"""
    data3d = Data3d(np.zeros((3, 4, 5)), name="phi", filename="dummy.h5")

    def _fake_plot_scalar_volume(*args, **kwargs):
        return "volume-plotter"

    monkeypatch.setattr(pvplot, "plot_scalar_volume", _fake_plot_scalar_volume)

    assert data3d.plot_pyvista(show=False) == "volume-plotter"
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

    monkeypatch.setattr(VectorData, "plot3d_mpl", lambda *args, **kwargs: "delegated")

    assert vec.plot(mode="stream", show=False) == "delegated"
