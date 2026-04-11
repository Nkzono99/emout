import importlib.util
import sys
from pathlib import Path
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


class FakeActorSession:
    def __init__(self, emout=None):
        self._cache = {}
        self._instances = {"default": emout} if emout is not None else {}
        self._drops = []

    def _render_to_bytes(self, draw_fn, fmt="png", dpi=150):
        from emout.distributed.remote_render import RemoteSession

        return RemoteSession._render_to_bytes(draw_fn, fmt=fmt, dpi=dpi)

    def _resolve(self, emout_kwargs=None):
        if not self._instances:
            raise RuntimeError("No fake Emout is registered")
        return next(iter(self._instances.values()))

    def _decode_remote_value(self, value):
        from emout.distributed.remote_render import RemoteSession

        return RemoteSession._decode_remote_value(self, value)

    def cache_emout_attr(self, key, emout_kwargs, name):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.cache_emout_attr(self, key, emout_kwargs, name))

    def cache_attr(self, key, parent_key, name):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.cache_attr(self, key, parent_key, name))

    def cache_getitem(self, key, parent_key, index):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.cache_getitem(self, key, parent_key, index))

    def call_cached(self, key, parent_key, args=(), kwargs=None):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.call_cached(self, key, parent_key, args=args, kwargs=kwargs))

    def call_method(self, key, parent_key, method_name, args=(), kwargs=None):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.call_method(self, key, parent_key, method_name, args=args, kwargs=kwargs))

    def apply_function(self, key, parent_key, func, args=(), kwargs=None):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.apply_function(self, key, parent_key, func, args=args, kwargs=kwargs))

    def apply_unary_operator(self, key, parent_key, operator_name):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.apply_unary_operator(self, key, parent_key, operator_name))

    def apply_binary_operator(self, key, parent_key, operator_name, other=None, reverse=False):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(
            RemoteSession.apply_binary_operator(
                self,
                key,
                parent_key,
                operator_name,
                other=other,
                reverse=reverse,
            )
        )

    def apply_ufunc(self, key, ufunc_name, method, inputs=(), kwargs=None):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.apply_ufunc(self, key, ufunc_name, method, inputs=inputs, kwargs=kwargs))

    def apply_array_function(self, key, func_name, args=(), kwargs=None):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.apply_array_function(self, key, func_name, args=args, kwargs=kwargs))

    def compute_probabilities(self, key, emout_kwargs=None, **kwargs):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.compute_probabilities(self, key, emout_kwargs=emout_kwargs, **kwargs))

    def compute_backtrace(self, key, emout_kwargs=None, **kwargs):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.compute_backtrace(self, key, emout_kwargs=emout_kwargs, **kwargs))

    def compute_backtraces(self, key, emout_kwargs=None, **kwargs):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.compute_backtraces(self, key, emout_kwargs=emout_kwargs, **kwargs))

    def fetch_object(self, key):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.fetch_object(self, key))

    def render_cached_plot(self, key, fmt="png", dpi=150, **plot_kwargs):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.render_cached_plot(self, key, fmt=fmt, dpi=dpi, **plot_kwargs))

    def get_cached_plot_surfaces_metadata(self, key, use_si=True, vmin=None, vmax=None, cmap_name="jet"):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(
            RemoteSession.get_cached_plot_surfaces_metadata(
                self,
                key,
                use_si=use_si,
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
            )
        )

    def render_cached_plot_surfaces(self, key, surfaces, fmt="png", dpi=150, use_si=True, **plot_kwargs):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(
            RemoteSession.render_cached_plot_surfaces(
                self,
                key,
                surfaces,
                fmt=fmt,
                dpi=dpi,
                use_si=use_si,
                **plot_kwargs,
            )
        )

    def replay_figure(self, commands, fmt="png", dpi=150):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.replay_figure(self, commands, fmt=fmt, dpi=dpi))

    def render_gifplot_html(self, key, **gifplot_kwargs):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.render_gifplot_html(self, key, **gifplot_kwargs))

    def render_gifplot_bytes(self, key, fmt="gif", **gifplot_kwargs):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.render_gifplot_bytes(self, key, fmt=fmt, **gifplot_kwargs))

    def render_gifplot_save(self, key, filename, **gifplot_kwargs):
        from emout.distributed.remote_render import RemoteSession

        return FakeFuture(RemoteSession.render_gifplot_save(self, key, filename, **gifplot_kwargs))

    def drop(self, key):
        self._drops.append(key)
        self._cache.pop(key, None)
        return FakeFuture(True)


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


def test_emout_remote_returns_remote_proxy(monkeypatch):
    from emout.core.facade import Emout
    from emout.distributed import remote_render

    fake_session = object()
    emout = Emout.__new__(Emout)
    emout._dir_inspector = SimpleNamespace(
        _input_directory=Path("/tmp/input"),
        append_directories=[],
        inpfilename="plasma.toml",
        main_directory=Path("/tmp/output"),
        input_path=Path("/tmp/input/plasma.toml"),
    )

    monkeypatch.setattr(remote_render, "get_or_create_session", lambda *args, **kwargs: fake_session)

    proxy = emout.remote()

    assert isinstance(proxy, remote_render.RemoteEmout)
    assert proxy._session is fake_session


def test_remote_ref_supports_dynamic_method_calls():
    from emout.distributed.remote_render import RemoteEmout, RemoteRef

    array = np.arange(12).reshape(3, 4)
    session = FakeActorSession(emout=SimpleNamespace(phisp=array))
    remote_data = RemoteEmout(session, {"directory": "/tmp/input"})

    sl = remote_data.phisp[1]
    mean_ref = sl.mean()

    assert isinstance(sl, RemoteRef)
    assert isinstance(mean_ref, RemoteRef)
    assert mean_ref.fetch() == pytest.approx(array[1].mean())
    assert sl.shape.fetch() == array[1].shape


def test_remote_ref_supports_operator_and_numpy_style_expressions():
    from emout.distributed.remote_render import RemoteEmout

    array = np.arange(12, dtype=float).reshape(3, 4)
    session = FakeActorSession(emout=SimpleNamespace(exz=array, phisp=array + 10.0))
    remote_data = RemoteEmout(session, {"directory": "/tmp/input"})

    sl = remote_data.exz[1]
    neg_ref = -sl
    expr_ref = (remote_data.phisp[1] + 2.0) / np.abs(neg_ref)
    mean_ref = np.mean(expr_ref)

    expected = ((array[1] + 10.0) + 2.0) / np.abs(-array[1])

    assert np.allclose(neg_ref.fetch(), -array[1])
    assert np.allclose(expr_ref.fetch(), expected)
    assert mean_ref.fetch() == pytest.approx(expected.mean())


def test_remote_ref_supports_scalar_conversion_helpers():
    from emout.distributed.remote_render import RemoteEmout

    session = FakeActorSession(emout=SimpleNamespace(inp=SimpleNamespace(ny=6, ratio=np.array(1.5))))
    remote_data = RemoteEmout(session, {"directory": "/tmp/input"})

    assert int(remote_data.inp.ny // 2) == 3
    assert float(remote_data.inp.ratio) == pytest.approx(1.5)


def test_remote_emout_backtrace_returns_specialized_probability_proxy():
    from emout.distributed.remote_render import RemoteBacktraceWrapper, RemoteEmout, RemoteProbabilityResult

    class DummyBacktrace:
        def __init__(self):
            self.calls = []

        def get_probabilities(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"kind": "probability"}

    backtrace = DummyBacktrace()
    session = FakeActorSession(emout=SimpleNamespace(backtrace=backtrace))
    remote_data = RemoteEmout(session, {"directory": "/tmp/input"})

    wrapper = remote_data.backtrace
    result = wrapper.get_probabilities(1, 2, 3, 4, 5, 6)

    assert isinstance(wrapper, RemoteBacktraceWrapper)
    assert isinstance(result, RemoteProbabilityResult)
    assert result.fetch() == {"kind": "probability"}
    assert backtrace.calls[0][1]["remote"] is False


def test_remote_emout_backtrace_returns_specialized_backtrace_proxies():
    from emout.core.backtrace.backtrace_result import BacktraceResult
    from emout.core.backtrace.multi_backtrace_result import MultiBacktraceResult
    from emout.distributed.remote_render import RemoteBacktraceResult, RemoteEmout
    from emout.distributed.remote_render import RemoteXYData

    single = BacktraceResult(
        ts=np.arange(4, dtype=float),
        probability=np.ones(4, dtype=float),
        positions=np.arange(12, dtype=float).reshape(4, 3),
        velocities=np.arange(12, dtype=float).reshape(4, 3) * 0.1,
        unit=None,
    )
    multi = MultiBacktraceResult(
        ts_list=np.arange(12, dtype=float).reshape(3, 4),
        probabilities=np.array([1.0, 0.5, 0.25]),
        positions_list=np.arange(36, dtype=float).reshape(3, 4, 3),
        velocities_list=np.arange(36, dtype=float).reshape(3, 4, 3) * 0.1,
        last_indexes=np.array([3, 2, 1]),
        unit=None,
    )

    class DummyBacktrace:
        def get_backtrace(self, *args, **kwargs):
            return single

        def get_backtraces(self, *args, **kwargs):
            return multi

    session = FakeActorSession(emout=SimpleNamespace(backtrace=DummyBacktrace()))
    remote_data = RemoteEmout(session, {"directory": "/tmp/input"})

    single_result = remote_data.backtrace.get_backtrace(np.zeros(3), np.ones(3))
    multi_result = remote_data.backtrace.get_backtraces(np.zeros((3, 3)), np.ones((3, 3)))
    sampled = multi_result.sample(1, random_state=0)

    assert isinstance(single_result, RemoteBacktraceResult)
    assert isinstance(single_result.tx, RemoteXYData)
    assert single_result.fetch() is single
    assert isinstance(multi_result, RemoteBacktraceResult)
    assert isinstance(multi_result.tvx, RemoteXYData)
    assert isinstance(sampled, RemoteBacktraceResult)
    assert sampled.fetch().ts_list.shape[0] == 1


def test_remote_scope_auto_drops_registered_refs():
    from emout.distributed.remote_render import RemoteRef, remote_scope

    session = FakeActorSession()
    session._cache = {"ref_1": 1, "ref_2": 2}

    with remote_scope():
        ref1 = RemoteRef(session, "ref_1")
        ref2 = RemoteRef(session, "ref_2")
        assert ref1.fetch() == 1
        assert ref2.fetch() == 2

    assert session._drops == ["ref_2", "ref_1"]


def test_remote_scope_can_nest_with_remote_figure(monkeypatch):
    from emout.distributed.remote_figure import remote_figure
    from emout.distributed import remote_render
    from emout.distributed.remote_render import RemoteRef, remote_scope

    displayed = []

    class PlotObject:
        def __init__(self):
            self.calls = []

        def plot(self, **plot_kwargs):
            import matplotlib.pyplot as plt

            self.calls.append(plot_kwargs)
            plt.plot([0.0, 1.0], [0.0, 1.0], color=plot_kwargs.get("color", "blue"))

    plot_object = PlotObject()
    session = FakeActorSession()
    session._cache["plot_0"] = plot_object

    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    with remote_scope():
        ref = RemoteRef(session, "plot_0")
        with remote_figure():
            assert ref.plot(color="red") is None

    assert plot_object.calls == [{"color": "red"}]
    assert len(displayed) == 1
    assert displayed[0][0]
    assert session._drops == ["plot_0"]


def test_remote_figure_replays_subplot_field_plot_and_axes_overlay(monkeypatch):
    from emout.distributed import remote_figure
    from emout.distributed import remote_render
    from emout.core.data.data import Data2d

    displayed = []
    open_kwargs = {
        "directory": "/tmp/input",
        "output_directory": "/tmp/output",
        "input_path": "/tmp/input/plasma.toml",
        "append_directories": [],
        "inpfilename": "plasma.toml",
    }

    data = Data2d(np.arange(4, dtype=float).reshape(2, 2), filename="dummy.h5", name="phisp")
    data._emout_open_kwargs = open_kwargs
    replay_data = Data2d(np.arange(4, dtype=float).reshape(2, 2), filename="dummy.h5", name="phisp")
    replay_data._emout_open_kwargs = None

    class Holder:
        def __getitem__(self, index):
            return replay_data

    session = FakeActorSession(emout=SimpleNamespace(phisp=Holder()))

    monkeypatch.setattr(remote_render, "get_or_create_session", lambda *args, **kwargs: session)
    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with remote_figure():
        ax = plt.subplot(111)
        assert data.plot() is None
        ax.axhline(0.5, color="red", linewidth=0.5)

    assert len(displayed) == 1
    assert displayed[0][0]


def test_remote_figure_supports_expression_style_overlays_with_remote_refs(monkeypatch):
    from emout.core.data.data import Data4d
    from emout.distributed import remote_render
    from emout.distributed.remote_figure import remote_figure
    from emout.distributed.remote_render import RemoteEmout

    displayed = []
    phisp = Data4d(np.arange(2 * 4 * 5 * 6, dtype=float).reshape(2, 4, 5, 6), filename="dummy.h5", name="phisp")
    exz = Data4d(np.arange(2 * 4 * 5 * 6, dtype=float).reshape(2, 4, 5, 6) + 1.0, filename="dummy.h5", name="exz")

    class Holder:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, index):
            return self._data[index]

    session = FakeActorSession(emout=SimpleNamespace(phisp=Holder(phisp), exz=Holder(exz)))
    remote_data = RemoteEmout(session, {"directory": "/tmp/input"})

    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with remote_figure(session=session):
        plt.figure(figsize=(18, 16))
        remote_data.phisp[-1, 1:4, 2, :].plot()
        (-remote_data.exz[-1, 1:4, 2, :]).plot()

    assert len(displayed) == 1
    assert displayed[0][0]


def test_remote_figure_replays_figure_add_axes_plot3d_and_tick_params(monkeypatch):
    from emout.distributed.remote_figure import remote_figure
    from emout.distributed import remote_render

    displayed = []
    session = FakeActorSession()

    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with remote_figure(session=session):
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d")
        ax.plot3D([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], color="black", linewidth=0.5)
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_tick_params(pad=4)

    assert len(displayed) == 1
    assert displayed[0][0]


def test_remote_figure_supports_surface_cut_helpers(monkeypatch):
    from emout.distributed.remote_figure import remote_figure
    from emout.distributed import remote_render
    from emout.plot.surface_cut import (
        Bounds3D,
        BoxMeshSurface,
        Field3D,
        RenderItem,
        UniformCellCenteredGrid,
        add_colorbar,
        plot_surfaces,
    )

    displayed = []
    session = FakeActorSession()

    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = UniformCellCenteredGrid(nx=6, ny=5, nz=4, dx=1.0, dy=1.0, dz=1.0)
    z = grid.z_centers()[:, None, None]
    y = grid.y_centers()[None, :, None]
    x = grid.x_centers()[None, None, :]
    field = Field3D(grid, x + 2.0 * y + 3.0 * z)
    surface = RenderItem(
        BoxMeshSurface(0.0, 5.0, 0.0, 4.0, 0.0, 3.0, faces=("zmax",), resolution=(4, 4)),
        style="field",
    )

    with remote_figure(session=session):
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8], projection="3d")
        cax = fig.add_axes([0.78, 0.15, 0.04, 0.7])
        ax.view_init(elev=25, azim=-40)
        cmap, norm = plot_surfaces(
            ax,
            field=field,
            surfaces=[surface],
            bounds=Bounds3D((0.0, 6.0), (0.0, 5.0), (0.0, 4.0)),
            vmin=0.0,
            vmax=25.0,
            contour_levels=[5.0, 10.0],
        )
        cbar = add_colorbar(fig, ax=None, cmap=cmap, norm=norm, cax=cax)
        cbar.set_label("phi")
        cbar.ax.tick_params(labelsize=8)
        assert ax.elev == 25
        assert ax.azim == -40

    assert len(displayed) == 1
    assert displayed[0][0]


def test_remote_figure_records_data3d_plot_surfaces_without_fetch_field(monkeypatch):
    from emout.distributed.remote_figure import remote_figure
    from emout.distributed import remote_render
    from emout.core.data.data import Data3d
    from emout.plot.surface_cut import BoxMeshSurface, add_colorbar

    displayed = []
    open_kwargs = {
        "directory": "/tmp/input",
        "output_directory": "/tmp/output",
        "input_path": "/tmp/input/plasma.toml",
        "append_directories": [],
        "inpfilename": "plasma.toml",
    }

    array = np.arange(6 * 5 * 4, dtype=float).reshape(4, 5, 6)
    data3d = Data3d(array, filename="dummy.h5", name="phisp")
    data3d._emout_open_kwargs = open_kwargs
    replay_data3d = Data3d(array, filename="dummy.h5", name="phisp")
    replay_data3d._emout_open_kwargs = None

    class Holder:
        def __getitem__(self, index):
            assert index == (0, slice(0, 4, 1), slice(0, 5, 1), slice(0, 6, 1))
            return replay_data3d

    session = FakeActorSession(emout=SimpleNamespace(phisp=Holder()))

    def _fail_fetch_field(*args, **kwargs):
        raise AssertionError("fetch_field should not be used while recording Data3d.plot_surfaces()")

    monkeypatch.setattr(session, "fetch_field", _fail_fetch_field, raising=False)
    monkeypatch.setattr(remote_render, "get_or_create_session", lambda *args, **kwargs: session)
    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with remote_figure():
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")
        cax = fig.add_axes([0.82, 0.15, 0.04, 0.7])
        cmap, norm = data3d.plot_surfaces(
            BoxMeshSurface(0.0, 5.0, 0.0, 4.0, 0.0, 3.0, faces=("zmax",), resolution=(4, 4)),
            ax=ax,
            use_si=False,
            vmin=0.0,
            vmax=120.0,
            contour_levels=[20.0, 40.0],
        )
        cbar = add_colorbar(fig, ax=None, cmap=cmap, norm=norm, cax=cax)
        cbar.set_label("phi")

    assert len(displayed) == 1
    assert displayed[0][0]


def test_remote_figure_replays_pyplot_colorbar_proxy(monkeypatch):
    from emout.distributed.remote_figure import remote_figure
    from emout.distributed import remote_render

    displayed = []
    session = FakeActorSession()

    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    with remote_figure(session=session):
        fig = plt.figure(figsize=(4, 3))
        cax = fig.add_axes([0.82, 0.15, 0.05, 0.7])
        mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="viridis")
        cbar = plt.colorbar(mappable, cax=cax)
        cbar.set_label("phi")
        cbar.ax.tick_params(labelsize=8)

    assert len(displayed) == 1
    assert displayed[0][0]


def test_remote_ref_plot_surfaces_outside_recording_renders_on_server(monkeypatch):
    from emout.core.data.data import Data3d
    from emout.distributed import remote_render
    from emout.distributed.remote_render import RemoteRef
    from emout.plot.surface_cut import BoxMeshSurface

    displayed = []
    session = FakeActorSession()
    session._cache["field_0"] = Data3d(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6), name="phisp")

    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    ref = RemoteRef(session, "field_0")
    cmap, norm = ref.plot_surfaces(
        BoxMeshSurface(0.0, 5.0, 0.0, 4.0, 0.0, 3.0, faces=("zmax",), resolution=(4, 4)),
        use_si=False,
        cmap_name="plasma",
    )

    assert len(displayed) == 1
    assert displayed[0][0]
    assert cmap.name == "plasma"
    assert norm.vmin < norm.vmax


def test_remote_ref_plot_surfaces_replays_with_remote_surface_refs(monkeypatch):
    from emout.core.data.data import Data3d
    from emout.distributed import remote_render
    from emout.distributed.remote_figure import remote_figure
    from emout.distributed.remote_render import RemoteRef, remote_scope
    from emout.plot.surface_cut import BoxMeshSurface, add_colorbar

    displayed = []
    session = FakeActorSession()
    session._cache["field_0"] = Data3d(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6), name="phisp")
    session._cache["surface_0"] = BoxMeshSurface(0.0, 5.0, 0.0, 4.0, 0.0, 3.0, faces=("zmax",), resolution=(4, 4))

    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with remote_scope():
        field = RemoteRef(session, "field_0")
        surfaces = RemoteRef(session, "surface_0")
        with remote_figure(session=session):
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111, projection="3d")
            cax = fig.add_axes([0.82, 0.15, 0.04, 0.7])
            cmap, norm = field.plot_surfaces(
                surfaces,
                ax=ax,
                use_si=False,
                cmap_name="viridis",
            )
            cbar = add_colorbar(fig, ax=None, cmap=cmap, norm=norm, cax=cax)
            cbar.set_label("phi")

    assert len(displayed) == 1
    assert displayed[0][0]
    assert session._drops == ["surface_0", "field_0"]


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


def test_try_remote_plot_keeps_implicit_data_transfer_mode(monkeypatch):
    from emout.core.data.data import Data2d
    from emout.distributed import remote_render

    open_kwargs = {
        "directory": "/tmp/input",
        "output_directory": "/tmp/output",
        "input_path": "/tmp/input/plasma.toml",
        "append_directories": [],
        "inpfilename": "plasma.toml",
    }
    data = Data2d(np.arange(4, dtype=float).reshape(2, 2), filename="dummy.h5", name="phisp")
    data._emout_open_kwargs = open_kwargs
    payload = {
        "array": np.arange(4, dtype=float).reshape(2, 2),
        "name": "phisp",
        "slices": data.slices,
        "slice_axes": data.slice_axes,
        "axisunits": data.axisunits,
        "valunit": data.valunit,
    }
    seen = []

    class FakeSession:
        def fetch_field(self, attr_name, recipe_index, emout_kwargs=None):
            seen.append((attr_name, recipe_index, emout_kwargs))
            return FakeFuture(payload)

    monkeypatch.setattr(remote_render, "get_or_create_session", lambda *args, **kwargs: FakeSession())
    monkeypatch.setattr(
        Data2d,
        "plot",
        lambda self, **kwargs: {"kwargs": kwargs, "remote_kwargs": getattr(self, "_emout_open_kwargs", None)},
    )

    result = data._try_remote_plot(cmap="magma")

    assert result == {"kwargs": {"cmap": "magma"}, "remote_kwargs": None}
    assert seen == [("phisp", (0, 0, slice(0, 2, 1), slice(0, 2, 1)), open_kwargs)]


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


class _DummyGifplottable:
    """Stand-in for a cached object whose ``gifplot()`` the session renders.

    We intentionally do not reuse the real :class:`Data` stack here: the
    remote dispatcher is what these tests exercise, not gifplot itself
    (which is covered extensively by ``tests/data/test_data_extended.py``).
    """

    def __init__(self):
        self.calls = []

    def gifplot(self, action="to_html", filename=None, **kwargs):
        self.calls.append({"action": action, "filename": filename, **kwargs})
        if action == "to_html":
            from IPython.display import HTML

            return HTML("<div class='mock-gif'>frames</div>")
        if action == "save":
            if filename is None:
                raise AssertionError("filename must be forwarded for action='save'")
            # Minimal GIF89a payload so tests can assert the magic header
            with open(filename, "wb") as fh:
                fh.write(b"GIF89a" + b"\x00" * 10)
            return None
        if action == "frames":
            return "frame-updater"
        raise ValueError(f"unsupported action: {action}")


_needs_ipython = pytest.mark.skipif(
    importlib.util.find_spec("IPython") is None,
    reason="gifplot(action='to_html') requires IPython",
)


@_needs_ipython
def test_remote_ref_gifplot_to_html_returns_ipython_html():
    from IPython.display import HTML

    from emout.distributed.remote_render import RemoteRef

    session = FakeActorSession()
    cached = _DummyGifplottable()
    session._cache["anim_0"] = cached

    ref = RemoteRef(session, "anim_0")
    result = ref.gifplot()

    assert isinstance(result, HTML)
    assert "mock-gif" in result.data
    assert cached.calls == [{"action": "to_html", "filename": None}]


@_needs_ipython
def test_remote_ref_gifplot_to_html_forwards_extra_kwargs():
    from emout.distributed.remote_render import RemoteRef

    session = FakeActorSession()
    cached = _DummyGifplottable()
    session._cache["anim_1"] = cached

    ref = RemoteRef(session, "anim_1")
    ref.gifplot(interval=50, repeat=False, axis=0)

    assert cached.calls[0]["interval"] == 50
    assert cached.calls[0]["repeat"] is False
    assert cached.calls[0]["axis"] == 0
    assert cached.calls[0]["action"] == "to_html"


def test_remote_ref_gifplot_save_writes_file_on_worker(tmp_path):
    from emout.distributed.remote_render import RemoteRef

    session = FakeActorSession()
    cached = _DummyGifplottable()
    session._cache["anim_2"] = cached

    target = tmp_path / "out.gif"
    ref = RemoteRef(session, "anim_2")
    result = ref.gifplot(action="save", filename=target)

    assert result is None
    assert target.exists()
    assert target.read_bytes().startswith(b"GIF89a")
    # filename is forwarded as str so the worker does not need a pathlib.Path
    assert cached.calls[0]["filename"] == str(target)


def test_remote_ref_gifplot_bytes_returns_gif_payload():
    from emout.distributed.remote_render import RemoteRef

    session = FakeActorSession()
    cached = _DummyGifplottable()
    session._cache["anim_3"] = cached

    ref = RemoteRef(session, "anim_3")
    payload = ref.gifplot(action="bytes")

    assert isinstance(payload, bytes)
    assert payload.startswith(b"GIF89a")
    # The worker used a temporary file; the filename must not leak back
    assert cached.calls[0]["action"] == "save"


def test_remote_emout_phisp_slice_gifplot_chain_bytes(tmp_path):
    """End-to-end: ``rdata.phisp[:, 100, :, :].gifplot(action='bytes')``.

    Exercises the compound expression that composes ``RemoteEmout.__getattr__``
    (caches ``data.phisp``), ``RemoteRef.__getitem__`` (caches the Data3d
    slice on the worker), and ``RemoteRef.gifplot`` (dispatches through
    :meth:`RemoteSession.render_gifplot_bytes`).
    """
    from emout.distributed.remote_render import RemoteEmout

    class _DummySeries:
        """Mimics ``GridDataSeries[tuple_index]`` well enough for this test."""

        def __init__(self):
            self.slice_calls = []

        def __getitem__(self, index):
            self.slice_calls.append(index)
            return _DummyGifplottable()

    series = _DummySeries()
    session = FakeActorSession(emout=SimpleNamespace(phisp=series))
    rdata = RemoteEmout(session, {"directory": "/tmp/input"})

    payload = rdata.phisp[:, 100, :, :].gifplot(action="bytes")

    # (slice(None), 100, slice(None), slice(None)) must reach the worker
    # untouched — ``RemoteRef._encode_remote_arg`` passes slices through.
    assert series.slice_calls == [(slice(None), 100, slice(None), slice(None))]
    assert isinstance(payload, bytes)
    assert payload.startswith(b"GIF89a")


def test_remote_ref_gifplot_rejects_unsupported_actions():
    from emout.distributed.remote_render import RemoteRef

    session = FakeActorSession()
    session._cache["anim_4"] = _DummyGifplottable()

    ref = RemoteRef(session, "anim_4")
    for bad_action in ("show", "return", "frames"):
        with pytest.raises(ValueError, match=bad_action):
            ref.gifplot(action=bad_action)

    with pytest.raises(ValueError, match="filename"):
        ref.gifplot(action="save")  # save requires filename

    with pytest.raises(ValueError, match="filename"):
        ref.gifplot(action="bytes", filename="x.gif")

    with pytest.raises(ValueError, match="filename"):
        ref.gifplot(filename="x.gif")  # to_html does not accept filename


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
