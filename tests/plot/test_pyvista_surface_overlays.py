import numpy as np

from emout.plot import _pyvista_helpers as helpers
from emout.plot.surface_cut import MeshSurface3D, RenderItem


class _SimpleSurface(MeshSurface3D):
    def mesh(self):
        return (
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            np.array([[0, 1, 2]], dtype=int),
        )


class _FakePolyData:
    def __init__(self, points, faces):
        self.points = np.asarray(points)
        self.faces = np.asarray(faces)


class _FakePyVista:
    PolyData = _FakePolyData


class _FakePlotter:
    def __init__(self):
        self.meshes = []
        self.screenshot_path = None
        self.show_kwargs = None
        self.html_path = None

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append((mesh, kwargs))

    def screenshot(self, filename):
        self.screenshot_path = filename

    def show(self, **kwargs):
        self.show_kwargs = kwargs

    def export_html(self, filename):
        self.html_path = filename


def test_show_bounds_uses_current_pyvista_titles():
    class _Plotter:
        def __init__(self):
            self.kwargs = None

        def show_bounds(self, **kwargs):
            self.kwargs = kwargs

    plotter = _Plotter()

    helpers._show_bounds(plotter, {"x": "x [m]", "y": "y [m]", "z": "z [m]"})

    assert plotter.kwargs == {
        "xtitle": "x [m]",
        "ytitle": "y [m]",
        "ztitle": "z [m]",
    }


def test_save_or_show_plotter_saves_screenshot(tmp_path):
    plotter = _FakePlotter()
    filename = tmp_path / "view.png"

    returned = helpers._save_or_show_plotter(plotter, filename=filename)

    assert returned is plotter
    assert plotter.screenshot_path == str(filename)
    assert plotter.show_kwargs is None


def test_save_or_show_plotter_show_uses_screenshot_argument(tmp_path):
    plotter = _FakePlotter()
    filename = tmp_path / "view.png"

    helpers._save_or_show_plotter(plotter, filename=filename, show=True)

    assert plotter.screenshot_path is None
    assert plotter.show_kwargs == {"screenshot": str(filename)}


def test_save_or_show_plotter_exports_html(tmp_path):
    plotter = _FakePlotter()
    filename = tmp_path / "view.html"

    helpers._save_or_show_plotter(plotter, filename=filename)

    assert plotter.html_path == str(filename)
    assert plotter.screenshot_path is None


def test_save_or_show_plotter_rejects_conflicting_filenames(tmp_path):
    plotter = _FakePlotter()

    try:
        helpers._save_or_show_plotter(
            plotter,
            filename=tmp_path / "a.png",
            savefilename=tmp_path / "b.png",
        )
    except ValueError as exc:
        assert "filename and savefilename" in str(exc)
    else:
        raise AssertionError("conflicting filenames should raise")


def test_add_surface_overlays_converts_meshsurface(monkeypatch):
    monkeypatch.setattr(helpers, "_require_pyvista", lambda: _FakePyVista)
    plotter = _FakePlotter()

    returned = helpers._add_surface_overlays(
        plotter,
        _SimpleSurface(),
        offsets=(1.0, None, "left"),
        surface_color="red",
        surface_opacity=0.25,
        show_edges=True,
    )

    assert returned is plotter
    mesh, kwargs = plotter.meshes[0]
    np.testing.assert_allclose(
        mesh.points,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )
    np.testing.assert_array_equal(mesh.faces, np.array([3, 0, 1, 2]))
    assert kwargs["color"] == "red"
    assert kwargs["opacity"] == 0.25
    assert kwargs["show_edges"] is True


def test_add_surface_overlays_uses_render_item_style(monkeypatch):
    monkeypatch.setattr(helpers, "_require_pyvista", lambda: _FakePyVista)
    plotter = _FakePlotter()
    item = RenderItem(_SimpleSurface(), solid_color="green", alpha=0.4)

    helpers._add_surface_overlays(plotter, item)

    _, kwargs = plotter.meshes[0]
    assert kwargs["color"] == "green"
    assert kwargs["opacity"] == 0.4


def test_add_surface_overlays_builds_boundary_collection_mesh(monkeypatch):
    monkeypatch.setattr(helpers, "_require_pyvista", lambda: _FakePyVista)
    plotter = _FakePlotter()
    calls = []

    class _BoundaryCollectionLike:
        def mesh(self, *, use_si=True, per=None):
            calls.append((use_si, per))
            return _SimpleSurface()

    per = {0: {"resolution": 8}}
    helpers._add_surface_overlays(
        plotter,
        _BoundaryCollectionLike(),
        use_si=False,
        per=per,
    )

    assert calls == [(False, per)]
    assert len(plotter.meshes) == 1
