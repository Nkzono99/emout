"""``remote_figure`` recording: ``plt.subplots`` layouts (#106) and argument snapshots (#105)."""

import importlib.util
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10)
    or importlib.util.find_spec("dask") is None
    or importlib.util.find_spec("distributed") is None,
    reason="distributed runtime is available only with Python >= 3.10 and dask/distributed installed",
)

SUBPLOTS_LAYOUTS = [
    pytest.param((), {}, id="default"),
    pytest.param((1, 2), {}, id="1x2"),
    pytest.param((2, 1), {}, id="2x1"),
    pytest.param((2, 2), {}, id="2x2"),
    pytest.param((1, 1), {"squeeze": False}, id="1x1-unsqueezed"),
    pytest.param((), {"nrows": 2, "ncols": 3, "squeeze": False}, id="keywords"),
]

PROXY_KINDS = ["figure", "axes", "axis", "spine", "colorbar"]


class FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class RecordingSession:
    """Stand-in session that keeps the commands sent for replay."""

    def __init__(self):
        self.commands = []

    def replay_figure(self, commands, fmt="png", dpi=150):
        self.commands.extend(commands)
        return FakeFuture(b"")


@pytest.fixture
def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


@pytest.fixture
def recording_session(monkeypatch):
    from emout.distributed import remote_render

    monkeypatch.setattr(remote_render, "display_image", lambda *args, **kwargs: None)
    return RecordingSession()


@pytest.fixture(scope="module")
def dask_session():
    """A real ``RemoteSession`` actor on an in-process Dask worker."""
    from distributed import Client, LocalCluster

    from emout.distributed.remote_render import clear_sessions, get_or_create_session

    with LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        protocol="inproc",
        dashboard_address=None,
        memory_limit=0,
    ) as cluster:
        with Client(cluster):
            try:
                yield get_or_create_session()
            finally:
                clear_sessions()


@pytest.fixture
def rendered_lines(monkeypatch):
    """Collect the y data of every line, per axes, each time a figure is saved."""
    from matplotlib.figure import Figure

    rendered = []
    savefig = Figure.savefig

    def observe(self, *args, **kwargs):
        rendered.append([[line.get_ydata().tolist() for line in ax.lines] for ax in self.axes])
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", observe)
    return rendered


def _axes_shape(axes):
    return np.shape(axes) if isinstance(axes, np.ndarray) else None


def _proxy(kind):
    from emout.distributed.remote_figure import AxesProxy, ColorbarProxy, FigureProxy

    ax = AxesProxy("ax_0", "fig_0")
    return {
        "figure": FigureProxy("fig_0"),
        "axes": ax,
        "axis": ax.xaxis,
        "spine": ax.spines["left"],
        "colorbar": ColorbarProxy("cbar_0", ax),
    }[kind]


# ---------------------------------------------------------------------------
# plt.subplots() layouts (#106)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("args, kwargs", SUBPLOTS_LAYOUTS)
def test_subplots_returns_axes_shaped_like_matplotlib(plt, recording_session, args, kwargs):
    from emout.distributed.remote_figure import AxesProxy, FigureProxy, remote_figure

    _, local_axes = plt.subplots(*args, **kwargs)

    with remote_figure(session=recording_session):
        fig, axes = plt.subplots(*args, **kwargs)

    assert isinstance(fig, FigureProxy)
    assert _axes_shape(axes) == _axes_shape(local_axes)
    assert all(isinstance(ax, AxesProxy) for ax in np.ravel(axes))


@pytest.mark.parametrize("args, kwargs", SUBPLOTS_LAYOUTS)
def test_real_session_draws_every_subplots_panel(plt, dask_session, rendered_lines, tmp_path, args, kwargs):
    from emout.distributed.remote_figure import remote_figure

    output = tmp_path / "panels.png"

    with remote_figure(session=dask_session, savefilepath=output):
        _, axes = plt.subplots(*args, **kwargs)
        panels = list(np.ravel(axes))
        for index, ax in enumerate(panels):
            ax.plot([0, 1], [index, index + 1])

    assert rendered_lines[-1] == [[[index, index + 1]] for index in range(len(panels))]
    assert output.stat().st_size > 0


@pytest.mark.parametrize("args, kwargs", SUBPLOTS_LAYOUTS)
def test_real_session_keeps_the_same_current_panel_as_matplotlib(
    plt, dask_session, rendered_lines, tmp_path, args, kwargs
):
    from emout.distributed.remote_figure import remote_figure

    _, local_axes = plt.subplots(*args, **kwargs)
    current = list(np.ravel(local_axes)).index(plt.gca())

    with remote_figure(session=dask_session, savefilepath=tmp_path / "current.png"):
        _, axes = plt.subplots(*args, **kwargs)
        plt.gca().plot([0, 1], [7, 8])

    panels = len(np.ravel(axes))
    assert rendered_lines[-1] == [[[7, 8]] if index == current else [] for index in range(panels)]


def test_subplots_axes_record_calls_against_their_own_panel(plt, recording_session):
    from emout.distributed.remote_figure import remote_figure

    with remote_figure(session=recording_session):
        _, axes = plt.subplots(1, 2)
        axes[1].axhline(0.5, color="red")
        axes[0].set_title("left")

    calls = [command for command in recording_session.commands if command[0] == "axes_call"]
    assert [(command[1], command[2]) for command in calls] == [
        (axes[1]._remote_plot_id, "axhline"),
        (axes[0]._remote_plot_id, "set_title"),
    ]
    assert axes[0]._remote_plot_id != axes[1]._remote_plot_id


# ---------------------------------------------------------------------------
# Proxies answer protocol lookups as "not implemented"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", PROXY_KINDS)
@pytest.mark.parametrize(
    "protocol",
    ["__array_struct__", "__array_interface__", "__array__", "__len__", "__iter__", "__deepcopy__", "__fspath__"],
)
def test_proxies_do_not_claim_protocol_methods(kind, protocol):
    assert not hasattr(_proxy(kind), protocol)


@pytest.mark.parametrize("kind", PROXY_KINDS)
def test_proxies_are_stored_as_single_objects_in_arrays(kind):
    proxy = _proxy(kind)

    arr = np.array([[proxy, proxy]], dtype=object)

    assert arr.shape == (1, 2)
    assert arr[0, 0] is proxy


@pytest.mark.parametrize("kind", PROXY_KINDS)
def test_proxies_still_record_ordinary_methods(plt, recording_session, kind):
    from emout.distributed.remote_figure import remote_figure

    with remote_figure(session=recording_session):
        _proxy(kind).set_label("recorded")

    assert any("set_label" in command for command in recording_session.commands)


# ---------------------------------------------------------------------------
# Array arguments are recorded with their values at call time (#105)
# ---------------------------------------------------------------------------


def test_axes_call_keeps_array_values_passed_before_in_place_update(plt, recording_session):
    from emout.distributed.remote_figure import remote_figure

    x = np.array([0.0, 1.0])
    y = np.array([2.0, 3.0])

    with remote_figure(session=recording_session):
        ax = plt.subplot(1, 1, 1)
        ax.plot(x, y)
        y *= 2
        ax.plot(x, y)

    first, second = (command for command in recording_session.commands if command[0] == "axes_call")
    np.testing.assert_array_equal(first[3][1], [2.0, 3.0])
    np.testing.assert_array_equal(second[3][1], [4.0, 6.0])


def test_keyword_and_nested_array_values_are_recorded_at_call_time(plt, recording_session):
    from emout.distributed.remote_figure import remote_figure

    ticks = np.array([0.0, 1.0])
    colors = np.array([0.5, 0.5])

    with remote_figure(session=recording_session):
        plt.xticks(ticks=ticks, labels=["a", "b"])
        plt.legend(handles=[], bbox_to_anchor=[colors])
        ticks += 10
        colors += 10

    xticks, legend = (command for command in recording_session.commands if command[0] == "plt")
    np.testing.assert_array_equal(xticks[3]["ticks"], [0.0, 1.0])
    np.testing.assert_array_equal(legend[3]["bbox_to_anchor"][0], [0.5, 0.5])


def test_real_session_draws_array_values_passed_before_in_place_update(plt, dask_session, rendered_lines, tmp_path):
    from emout.distributed.remote_figure import remote_figure

    y = np.array([2.0, 3.0])

    with remote_figure(session=dask_session, savefilepath=tmp_path / "lines.png"):
        ax = plt.subplot(1, 1, 1)
        ax.plot([0.0, 1.0], y)
        y *= 2
        ax.plot([0.0, 1.0], y)

    assert rendered_lines[-1] == [[[2.0, 3.0], [4.0, 6.0]]]
