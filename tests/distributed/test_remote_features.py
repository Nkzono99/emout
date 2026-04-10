import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10)
    or importlib.util.find_spec("dask") is None
    or importlib.util.find_spec("distributed") is None,
    reason="distributed runtime is available only with Python >= 3.10 and dask/distributed installed",
)


class FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def test_fetch_field_returns_numpy_payload():
    from emout.distributed.remote_render import RemoteSession
    from emout.core.data.data import Data2d

    data = Data2d(np.arange(4).reshape(2, 2), filename="dummy.h5", name="phisp")

    class Holder:
        def __getitem__(self, index):
            assert index == (slice(None), slice(None))
            return data

    fake_emout = SimpleNamespace(phisp=Holder())
    fake_session = SimpleNamespace(
        _instances={"default": fake_emout},
        _resolve=lambda emout_kwargs=None: fake_emout,
    )

    payload = RemoteSession.fetch_field(
        fake_session,
        "phisp",
        (slice(None), slice(None)),
    )

    assert np.array_equal(payload["array"], np.arange(4).reshape(2, 2))
    assert payload["name"] == "phisp"


def test_remote_figure_auto_creates_session_from_recorded_field_plot(monkeypatch):
    from emout.distributed import remote_figure
    from emout.distributed import remote_render
    from emout.core.data.data import Data2d

    commands_seen = []
    displayed = []
    open_kwargs = {
        "directory": "/tmp/input",
        "output_directory": "/tmp/output",
        "input_path": "/tmp/input/plasma.toml",
        "append_directories": [],
        "inpfilename": "plasma.toml",
    }

    class FakeSession:
        def replay_figure(self, commands, fmt="png", dpi=150):
            commands_seen.append((commands, fmt, dpi))
            return FakeFuture(b"png-bytes")

    fake_session = FakeSession()

    monkeypatch.setattr(
        remote_render,
        "get_or_create_session",
        lambda *args, **kwargs: fake_session,
    )
    monkeypatch.setattr(
        remote_render,
        "display_image",
        lambda img_bytes, ax=None: displayed.append((img_bytes, ax)),
    )

    data = Data2d(np.zeros((2, 2)), filename="dummy.h5", name="phisp")
    data._emout_open_kwargs = open_kwargs

    with remote_figure():
        assert data.plot() is None

    assert len(commands_seen) == 1
    commands, fmt, dpi = commands_seen[0]
    assert len(commands) == 1
    kind, name, recipe_index, plot_kwargs, emout_kwargs = commands[0]
    assert kind == "field_plot"
    assert name == "phisp"
    assert recipe_index == (0, 0, slice(0, 2, 1), slice(0, 2, 1))
    assert plot_kwargs["axes"] == "auto"
    assert plot_kwargs["mode"] == "cm"
    assert plot_kwargs["use_si"] is True
    assert emout_kwargs == open_kwargs
    assert fmt == "png"
    assert dpi == 150
    assert displayed == [(b"png-bytes", None)]


def test_get_or_create_session_returns_shared_session(monkeypatch):
    """All calls return the same shared session regardless of emout_kwargs."""
    import dask.distributed

    from emout.distributed import remote_render

    remote_render.clear_sessions()
    submissions = []

    class FakeClient:
        def submit(self, cls, *args, **kwargs):
            submissions.append((cls, args, kwargs))
            return FakeFuture(object())

    monkeypatch.setattr(
        dask.distributed,
        "default_client",
        lambda: FakeClient(),
    )

    kwargs1 = {
        "directory": "/tmp/input",
        "output_directory": "/tmp/output",
        "input_path": "/tmp/input/a.toml",
        "append_directories": [],
        "inpfilename": "a.toml",
    }
    kwargs2 = {
        "directory": "/tmp/input",
        "output_directory": "/tmp/output",
        "input_path": "/tmp/input/b.toml",
        "append_directories": [],
        "inpfilename": "b.toml",
    }

    session1 = remote_render.get_or_create_session(emout_kwargs=kwargs1)
    session2 = remote_render.get_or_create_session(emout_kwargs=kwargs1)
    session3 = remote_render.get_or_create_session(emout_kwargs=kwargs2)

    # Shared session: all three are the same object, only 1 Actor created
    assert session1 is session2
    assert session3 is session1
    assert len(submissions) == 1


def test_backtrace_wrapper_remote_uses_full_emout_open_kwargs(monkeypatch):
    from emout.distributed import remote_render
    from emout.core.backtrace.solver_wrapper import BacktraceWrapper

    calls = []

    class FakeSession:
        def compute_probabilities(self, key, **kwargs):
            calls.append(("compute", key, kwargs))
            return FakeFuture(True)

    fake_session = FakeSession()

    monkeypatch.setattr(
        remote_render,
        "get_or_create_session",
        lambda *args, **kwargs: calls.append(("session", kwargs)) or fake_session,
    )
    monkeypatch.setattr(remote_render, "_next_key", lambda prefix="result": f"{prefix}_0")

    remote_kwargs = {
        "directory": "/tmp/input",
        "output_directory": "/tmp/output",
        "input_path": "/tmp/input/plasma.toml",
        "append_directories": [],
        "inpfilename": "plasma.toml",
    }
    wrapper = BacktraceWrapper(
        directory="/tmp/output",
        inp=SimpleNamespace(dt=0.1),
        unit=None,
        remote_open_kwargs=remote_kwargs,
    )

    result = wrapper.get_probabilities(
        (0.0, 1.0, 2),
        (0.0, 1.0, 2),
        (0.0, 1.0, 2),
        (0.0, 1.0, 2),
        (0.0, 1.0, 2),
        (0.0, 1.0, 2),
    )

    assert isinstance(result, remote_render.RemoteProbabilityResult)
    assert calls[0] == ("session", {"emout_kwargs": remote_kwargs, "emout_dir": "/tmp/output"})
    assert calls[1][0] == "compute"
    assert calls[1][1] == "prob_0"
    assert calls[1][2]["emout_kwargs"] == remote_kwargs
    assert calls[1][2]["remote"] is False


def test_remote_heatmap_fetch_returns_local_heatmap(monkeypatch):
    """RemoteHeatmap.fetch() should transfer data and return a local HeatmapData."""
    from emout.distributed.remote_render import RemoteHeatmap
    from emout.core.backtrace.probability_result import HeatmapData

    X = np.arange(6).reshape(2, 3).astype(float)
    Y = np.arange(6).reshape(2, 3).astype(float) * 2
    Z = np.ones((2, 3))

    class FakeSession:
        def fetch_heatmap_data(self, key, var1, var2):
            return FakeFuture(
                {
                    "X": X,
                    "Y": Y,
                    "Z": Z,
                    "xlabel": "vx",
                    "ylabel": "vz",
                    "title": "vx vs vz",
                    "units": None,
                }
            )

    hm = RemoteHeatmap(FakeSession(), "k", "vx", "vz")
    local = hm.fetch()

    assert isinstance(local, HeatmapData)
    assert np.array_equal(local.X, X)
    assert np.array_equal(local.Z, Z)
    assert local.xlabel == "vx"


def test_remote_xy_data_fetch_returns_local_xy_data(monkeypatch):
    """RemoteXYData.fetch() should transfer data and return a local XYData."""
    from emout.distributed.remote_render import RemoteXYData
    from emout.core.backtrace.xy_data import XYData

    x = np.linspace(0, 1, 10)
    y = np.sin(x)

    class FakeSession:
        def fetch_xy_data(self, key, var1, var2):
            return FakeFuture(
                {
                    "x": x,
                    "y": y,
                    "xlabel": "x",
                    "ylabel": "vz",
                    "title": "x vs vz",
                    "units": None,
                }
            )

    proxy = RemoteXYData(FakeSession(), "k", "x", "vz")
    local = proxy.fetch()

    assert isinstance(local, XYData)
    assert np.array_equal(local.x, x)
    assert local.ylabel == "vz"


def test_remote_heatmap_plot_outside_recording_renders_on_server():
    """RemoteHeatmap.plot() outside remote_figure() should render on worker."""
    from emout.distributed.remote_render import RemoteHeatmap

    render_calls = []

    class FakeSession:
        def render_pair(self, key, var1, var2, fmt="png", dpi=150, **kw):
            render_calls.append((key, var1, var2))
            # Return 1x1 white PNG
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from io import BytesIO

            fig, ax = plt.subplots()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            return FakeFuture(buf.getvalue())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    hm = RemoteHeatmap(FakeSession(), "k", "vx", "vz")
    hm.plot(ax=ax)
    plt.close(fig)

    assert len(render_calls) == 1
    assert render_calls[0] == ("k", "vx", "vz")


def test_stop_cluster_can_shutdown_by_address(monkeypatch):
    import dask.distributed

    from emout.distributed import client as client_module

    events = []

    class FakeClient:
        def __init__(self, address, timeout="5s"):
            events.append(("connect", address, timeout))

        def shutdown(self):
            events.append("shutdown")

        def close(self):
            events.append("close")

    monkeypatch.setattr(dask.distributed, "Client", FakeClient)
    monkeypatch.setattr(client_module, "clear_sessions", lambda: events.append("clear"))
    monkeypatch.setattr(client_module, "_global_cluster", None)

    client_module.stop_cluster("tcp://127.0.0.1:8786")

    assert events == [
        ("connect", "tcp://127.0.0.1:8786", "5s"),
        "shutdown",
        "close",
        "clear",
    ]
