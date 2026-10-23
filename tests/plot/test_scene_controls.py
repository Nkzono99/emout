"""Check coordinates, clipped paths, and opacity using real PyVista meshes."""

from types import SimpleNamespace

import numpy as np
import pytest

from emout.core.backtrace.trace_result import TraceResult
from emout.core.data import Data3d
from emout.utils import Units


class SceneRecorder:
    def __init__(self):
        self.meshes = []
        self.bounds_options = None
        self.camera_position = None
        self.scalar_bar_options = None

    @property
    def bounds(self):
        points = np.concatenate([mesh.points for mesh, _ in self.meshes])
        return np.column_stack([points.min(axis=0), points.max(axis=0)]).ravel()

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append((mesh, kwargs))

    add_volume = add_mesh

    def add_axes(self):
        pass

    def add_scalar_bar(self, **kwargs):
        self.scalar_bar_options = kwargs

    def show_bounds(self, **kwargs):
        self.bounds_options = kwargs

    def remove_bounds_axes(self):
        self.bounds_options = None


@pytest.mark.parametrize("mode", ["box", "volume"])
def test_scalar_scene_controls_use_coordinate_units_and_do_not_reach_mesh(mode):
    pytest.importorskip("pyvista")
    units = Units(0.5, 10000)
    data = Data3d(
        np.arange(60).reshape(3, 4, 5),
        axisunits=[units.t, units.length, units.length, units.length],
        valunit=units.phi,
    )
    scene = SceneRecorder()
    data.plot3d(
        mode=mode,
        plotter=scene,
        camera="low",
        bounds_kwargs={"font_size": 10, "n_xlabels": 4},
        scalar_bar_args={"vertical": True, "position_x": 0.82, "title_font_size": 12},
    )
    assert scene.bounds_options["xtitle"] == "x [m]"
    assert scene.bounds_options["font_size"] == 10
    assert scene.bounds_options["n_xlabels"] == 4
    assert scene.camera_position is not None
    assert scene.scalar_bar_options["vertical"] is True
    assert scene.scalar_bar_options["position_x"] == 0.82
    for _, options in scene.meshes:
        assert "camera" not in options
        assert "bounds_kwargs" not in options
        assert "scalar_bar_args" not in options
    data.plot3d(plotter=scene, show_bounds=False, camera_position=[(5, 2, 1), (0, 0, 0), (0, 0, 1)])
    assert scene.bounds_options is None
    assert scene.camera_position[0] == (5, 2, 1)


def _payload(points, count=1, unit=None):
    return SimpleNamespace(
        positions_list=np.tile(points, (count, 1, 1)),
        last_indexes=np.full(count, len(points)),
        unit=unit,
    )


@pytest.mark.parametrize(
    "scale,expected",
    [("linear", [0.05, 0.05, 0.05, 0.05]), ("normalize", [0.05, 0.1, 1.0, 0.05]), ("log", [0.05, 2 / 3, 1.0, 0.05])],
)
def test_probability_alpha_floor_and_scaling(scale, expected):
    pytest.importorskip("pyvista")
    trace = TraceResult(
        "backward",
        backward_traces=_payload([[0, 0, 0], [1, 0, 0]], count=4),
        probabilities=SimpleNamespace(probabilities=[1e-6, 1e-4, 1e-3, 0]),
    )
    scene = SceneRecorder()
    trace.plot3d(plotter=scene, alpha_scale=scale, min_alpha=0.05)
    np.testing.assert_allclose([options["opacity"] for _, options in scene.meshes], expected)


def test_clip_intersections_are_in_display_units_and_both_directions_have_colours():
    pytest.importorskip("pyvista")
    unit = Units(0.5, 10000)
    points = [[-2, 0, 0], [0, 0, 0], [2, 0, 0]]
    trace = TraceResult("both", backward_traces=_payload(points, unit=unit), forward_traces=_payload(points, unit=unit))
    scene = SceneRecorder()
    trace.plot3d(
        plotter=scene,
        direction="both",
        offsets=(10, 0, 0),
        clip_bounds=(9.5, 10.5, -1, 1, -1, 1),
        backward_color="blue",
        forward_color="red",
        alpha=0.01,
        min_alpha=0.1,
        show_bounds=True,
    )
    assert len(scene.meshes) == 2
    for mesh, options in scene.meshes:
        np.testing.assert_allclose(mesh.points, [[9.5, 0, 0], [10, 0, 0], [10.5, 0, 0]])
        assert options["opacity"] == 0.1
    assert [options["color"] for _, options in scene.meshes] == ["blue", "red"]
    assert scene.bounds_options["xtitle"] == "x [m]"


def test_clipping_does_not_connect_paths_across_excluded_region():
    pytest.importorskip("pyvista")
    points = [[-2, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0], [0, 0, 0]]
    trace = TraceResult("backward", backward_traces=_payload(points))
    scene = SceneRecorder()
    trace.plot3d(plotter=scene, clip_bounds=(-1, 1, -1, 1, -1, 1))
    assert len(scene.meshes) == 2
    np.testing.assert_allclose(scene.meshes[0][0].points, [[-1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(scene.meshes[1][0].points, [[0, 1, 0], [0, 0, 0]])


@pytest.mark.parametrize("options", [{"min_alpha": -1}, {"alpha_scale": "bad"}, {"clip_bounds": (1, -1, 0, 1, 0, 1)}])
def test_invalid_trace_controls(options):
    pytest.importorskip("pyvista")
    trace = TraceResult("backward", backward_traces=_payload([[0, 0, 0], [1, 0, 0]]))
    with pytest.raises(ValueError):
        trace.plot3d(plotter=SceneRecorder(), **options)
