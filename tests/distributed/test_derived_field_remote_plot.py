"""Remote plotting must draw locally derived field values, not the source file (#103)."""

import importlib.util
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from emout.core.data.data import Data2d, Data3d


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10)
    or importlib.util.find_spec("dask") is None
    or importlib.util.find_spec("distributed") is None,
    reason="distributed runtime is available only with Python >= 3.10 and dask/distributed installed",
)

OPEN_KWARGS = {"directory": "/tmp/input"}


class FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def _source_field(shape=(2, 3)):
    """A Data2d that looks like a slice materialized from an Emout source."""
    data = Data2d(np.arange(1.0, 1.0 + np.prod(shape)).reshape(shape), filename="dummy.h5", name="phisp")
    data._emout_open_kwargs = OPEN_KWARGS
    return data


@pytest.fixture
def emout_dir(tmp_path):
    (tmp_path / "plasma.inp").write_text("&tmgrid nx=3,ny=2,nz=1,dt=0.002,ifdiag=1 /\n")
    with h5py.File(tmp_path / "phisp00_0000.h5", "w") as f:
        f.create_dataset("phisp/0000", data=np.arange(1.0, 25.0).reshape(2, 3, 4))
    return tmp_path


# ---------------------------------------------------------------------------
# Source replay state
# ---------------------------------------------------------------------------


def test_materialized_slices_replay_source(emout_dir):
    from emout import Emout

    data = Emout(emout_dir)
    field = data.phisp[0, 0, :, :]

    assert field._replays_source()
    assert field[1:, :]._replays_source()
    assert field.copy()._replays_source()
    data.phisp.close()


@pytest.mark.parametrize(
    "derive",
    [
        pytest.param(lambda d: d.scale(10), id="scale"),
        pytest.param(lambda d: d.negate(), id="negate"),
        pytest.param(lambda d: d.flip(1), id="flip"),
        pytest.param(lambda d: d.mirror(1), id="mirror"),
        pytest.param(lambda d: d.tile(1, 1), id="tile"),
        pytest.param(lambda d: d.masked(lambda x: x > 3), id="masked"),
        pytest.param(lambda d: -d, id="neg-operator"),
        pytest.param(lambda d: d * 10, id="mul-operator"),
        pytest.param(lambda d: np.abs(d), id="ufunc"),
        pytest.param(lambda d: d.scale(10)[1:, :], id="slice-of-derived"),
        pytest.param(lambda d: d.scale(10).copy(), id="copy-of-derived"),
    ],
)
def test_derived_values_do_not_replay_source(derive):
    field = _source_field()

    derived = derive(field)

    assert not derived._replays_source()
    assert field._replays_source()


def test_in_place_writes_stop_replay_for_shared_views():
    field = _source_field()
    row = field[0, :]
    other_row = field[1, :]

    row *= 2

    assert not row._replays_source()
    assert not field._replays_source()
    assert not other_row._replays_source()


def test_item_assignment_stops_replay_but_leaves_copies_source():
    field = _source_field()
    copy = field.copy()

    copy[0, 0] = -1.0

    assert not copy._replays_source()
    assert field._replays_source()

    field[0, 0] = -1.0
    assert not field._replays_source()


def test_time_mean_does_not_replay_source(emout_dir):
    from emout import Emout

    data = Emout(emout_dir)
    mean = data.phisp.lazy[:].mean(axis=0).materialize()

    assert not mean._replays_source()
    data.phisp.close()


# ---------------------------------------------------------------------------
# Data transfer (compatibility) mode
# ---------------------------------------------------------------------------


def test_data_transfer_mode_plots_derived_values_locally(monkeypatch):
    from emout.distributed import remote_render

    class FailSession:
        def fetch_field(self, *args, **kwargs):
            raise AssertionError("derived values must not be refetched from the source")

    monkeypatch.setattr(remote_render, "get_or_create_session", lambda *args, **kwargs: FailSession())

    assert _source_field().scale(10)._try_remote_plot(use_si=False) is None


def test_data3d_plot_surfaces_uses_derived_values_locally(monkeypatch):
    from emout.distributed import remote_render

    data = Data3d(np.arange(24, dtype=float).reshape(2, 3, 4), filename="dummy.h5", name="phisp")
    data._emout_open_kwargs = OPEN_KWARGS
    derived = data.negate()
    seen = []

    monkeypatch.setattr(
        remote_render,
        "get_or_create_session",
        lambda *args, **kwargs: pytest.fail("derived values must not be refetched from the source"),
    )
    monkeypatch.setattr(Data3d, "_plot_surfaces_local", lambda self, surfaces, **kwargs: seen.append(np.asarray(self)))

    derived.plot_surfaces([], use_si=False)

    np.testing.assert_array_equal(seen[0], np.asarray(derived))


def test_real_dask_data_transfer_mode_plots_derived_values(emout_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from distributed import Client, LocalCluster

    from emout import Emout
    from emout.distributed.remote_render import clear_sessions

    data = Emout(emout_dir)
    field = data.phisp[0, 0, :, :]
    in_place = field.copy()
    in_place *= 3
    cases = {
        "source": field,
        "scale": field.scale(10),
        "negate": field.negate(),
        "neg-operator": -field,
        "in-place": in_place,
    }

    try:
        with LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            protocol="inproc",
            dashboard_address=None,
            memory_limit=0,
        ) as cluster:
            with Client(cluster):
                for name, values in cases.items():
                    image = values.plot(use_si=False)
                    np.testing.assert_array_equal(np.asarray(image.get_array()), np.asarray(values), err_msg=name)
                    plt.close(image.figure)
    finally:
        clear_sessions()
        data.phisp.close()


# ---------------------------------------------------------------------------
# remote_figure recording mode
# ---------------------------------------------------------------------------


def _record_commands(monkeypatch, draw):
    """Run *draw* inside ``remote_figure`` and return the recorded commands."""
    import matplotlib

    matplotlib.use("Agg")
    from emout.distributed import remote_render
    from emout.distributed.remote_figure import remote_figure

    recorded = []

    class RecordingSession:
        def replay_figure(self, commands, fmt="png", dpi=150):
            recorded.extend(commands)
            return FakeFuture(b"")

    monkeypatch.setattr(remote_render, "get_or_create_session", lambda *args, **kwargs: RecordingSession())
    monkeypatch.setattr(remote_render, "display_image", lambda *args, **kwargs: None)

    with remote_figure():
        draw()
    return [command for command in recorded if command[0] != "figure_call"]


def _replay(commands, emout=None):
    from emout.distributed.remote_render import RemoteSession

    session = SimpleNamespace(_resolve=lambda emout_kwargs=None: emout)
    return RemoteSession.replay_figure(session, commands)


def test_remote_figure_keeps_recipe_replay_for_source_values(monkeypatch):
    commands = _record_commands(monkeypatch, lambda: _source_field().plot(use_si=False))

    assert [command[0] for command in commands] == ["field_plot"]


def test_remote_figure_ships_derived_values(monkeypatch):
    field = _source_field()
    derived = field.negate()
    plotted = []

    commands = _record_commands(monkeypatch, lambda: derived.plot(use_si=False))

    assert [command[0] for command in commands] == ["local_field_plot"]
    np.testing.assert_array_equal(commands[0][1]["array"], np.asarray(derived))

    monkeypatch.setattr(Data2d, "plot", lambda self, *args, **kwargs: plotted.append(np.asarray(self)))
    _replay(commands)
    np.testing.assert_array_equal(plotted[0], np.asarray(derived))


def test_remote_figure_ships_derived_surface_values(monkeypatch):
    data = Data3d(np.arange(24, dtype=float).reshape(2, 3, 4), filename="dummy.h5", name="phisp")
    data._emout_open_kwargs = OPEN_KWARGS
    derived = data.scale(2)
    plotted = []

    commands = _record_commands(monkeypatch, lambda: derived.plot_surfaces([], use_si=False, vmin=0, vmax=1))

    assert [command[0] for command in commands] == ["local_plot_surfaces"]

    monkeypatch.setattr(
        Data3d, "_plot_surfaces_local", lambda self, surfaces, **kwargs: plotted.append(np.asarray(self))
    )
    _replay(commands)
    np.testing.assert_array_equal(plotted[0], np.asarray(derived))


def test_remote_figure_ships_derived_vector_values(monkeypatch):
    from emout.core.data.vector_data import VectorData

    ex = _source_field()
    ex.name = "ex"
    ez = _source_field()
    ez.name = "ez"
    derived = VectorData([ex, ez], name="exz").negate()
    plotted = []

    commands = _record_commands(monkeypatch, lambda: derived.plot(use_si=False))

    assert [command[0] for command in commands] == ["local_field_plot"]

    monkeypatch.setattr(VectorData, "plot", lambda self, *args, **kwargs: plotted.append(self))
    _replay(commands)
    np.testing.assert_array_equal(np.asarray(plotted[0].x_data), np.asarray(derived.x_data))
    assert plotted[0].component_axes == derived.component_axes
