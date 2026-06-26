import numpy as np

import emout.plot._pyvista_vector as pvvector


class _FakePolyData:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)


class _FakePyVista:
    PolyData = _FakePolyData


class _FakeStreamline:
    def __init__(self):
        self.n_points = 2
        self.array_names = ["vectors"]
        self.data = {"vectors": np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])}
        self.tube_kwargs = None

    def __getitem__(self, name):
        return self.data[name]

    def __setitem__(self, name, value):
        self.data[name] = value
        self.array_names.append(name)

    def tube(self, **kwargs):
        self.tube_kwargs = kwargs
        return self


class _FakeMesh:
    bounds = (0.0, 2.0, 10.0, 14.0, -1.0, 1.0)
    center = (1.0, 12.0, 0.0)

    def __init__(self):
        self.source_points = None
        self.streamline_kwargs = None

    def streamlines_from_source(self, source, **kwargs):
        self.source_points = np.asarray(source.points)
        self.streamline_kwargs = kwargs
        return _FakeStreamline()

    def streamlines(self, **kwargs):
        self.streamline_kwargs = kwargs
        return _FakeStreamline()

    def outline(self):
        return "outline"


class _FakePlotter:
    def __init__(self):
        self.meshes = []

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append((mesh, kwargs))

    def show_bounds(self, **kwargs):
        pass

    def add_axes(self):
        pass


def test_make_streamline_source_volume_places_points_across_domain():
    mesh = _FakeMesh()

    source = pvvector._make_streamline_source(
        mesh,
        _FakePyVista,
        seed_mode="volume",
        n_points=8,
    )

    assert source.points.shape == (8, 3)
    assert {0.0, 2.0} == set(source.points[:, 0])
    assert {10.0, 14.0} == set(source.points[:, 1])
    assert {-1.0, 1.0} == set(source.points[:, 2])


def test_make_streamline_source_plane_uses_center_plane():
    mesh = _FakeMesh()

    source = pvvector._make_streamline_source(
        mesh,
        _FakePyVista,
        seed_mode="plane",
        seed_plane="xy",
        n_points=9,
    )

    assert source.points.shape == (9, 3)
    np.testing.assert_allclose(source.points[:, 2], 0.0)
    assert {0.0, 1.0, 2.0} == set(source.points[:, 0])
    assert {10.0, 12.0, 14.0} == set(source.points[:, 1])


def test_make_streamline_source_uses_explicit_seed_points():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    source = pvvector._make_streamline_source(
        _FakeMesh(),
        _FakePyVista,
        seed_mode="volume",
        seed_points=points,
    )

    np.testing.assert_allclose(source.points, points)


def test_make_streamline_source_surface_uses_surface_points(monkeypatch):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    class _SurfacePolyData:
        def __init__(self):
            self.points = points

    monkeypatch.setattr(
        pvvector,
        "_surface_to_polydata",
        lambda *args, **kwargs: _SurfacePolyData(),
    )

    source = pvvector._make_streamline_source(
        _FakeMesh(),
        _FakePyVista,
        seed_mode="surface",
        seed_surface=object(),
        n_points=2,
    )

    np.testing.assert_allclose(source.points, points[[0, 2]])


def test_tube_streamline_mesh_uses_magnitude_scalars():
    streamline = _FakeStreamline()

    result = pvvector._tube_streamline_mesh(
        streamline,
        tube_radius="magnitude",
        magnitude_name="magnitude",
        default_radius=0.2,
        tube_radius_factor=4.0,
    )

    assert result is streamline
    assert streamline.tube_kwargs == {
        "radius": 0.2,
        "scalars": "magnitude",
        "radius_factor": 4.0,
    }


def test_plot_vector_streamlines3d_plane_seed_uses_source(monkeypatch):
    mesh = _FakeMesh()
    plotter = _FakePlotter()

    monkeypatch.setattr(pvvector, "_require_pyvista", lambda: _FakePyVista)
    monkeypatch.setattr(
        pvvector,
        "create_vector_mesh3d",
        lambda *args, **kwargs: (mesh, "vectors", "magnitude", {"x": "x", "y": "y", "z": "z"}),
    )

    returned = pvvector.plot_vector_streamlines3d(
        object(),
        object(),
        object(),
        plotter=plotter,
        seed_mode="plane",
        seed_plane="xz",
        n_points=4,
        add_scalar_bar=False,
    )

    assert returned is plotter
    np.testing.assert_allclose(mesh.source_points[:, 1], 12.0)
    assert mesh.streamline_kwargs["vectors"] == "vectors"
