import numpy as np
import pytest
import h5py

import emout
from emout.local_data_policy import LocalDataAccessDisabledError


class FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def _write_grid_h5(path, name, timesteps=2, shape=(4, 4, 4)):
    with h5py.File(path, "w") as h5:
        group = h5.create_group(name)
        for index in range(timesteps):
            group.create_dataset(f"{index:04}", data=np.ones(shape) * index, dtype="f")


def _write_relocation_inp(path):
    path.write_text(
        """
&tmgrid
    dt = 0.002
    nx = 4
    ny = 4
    nz = 4
    nstep = 2
/
&mpi
    nodes(1:3) = 1, 1, 1
/
&emissn
    nspec = 1
    npc = 1
/
&esorem
    mtd_vbnd(1:3) = 0, 0, 0
/
""",
        encoding="utf-8",
    )


def test_context_policy_blocks_existing_emout_local_field_reads(data):
    with emout.local_data_policy("remote_required"):
        selection = data.phisp[1]

        assert data.phisp.shape == (5, 100, 30, 30)
        assert isinstance(selection, emout.data.GridDataSelection)
        assert selection.shape == (100, 30, 30)

        with pytest.raises(LocalDataAccessDisabledError, match="materialize field data locally"):
            selection.materialize()

        with pytest.raises(LocalDataAccessDisabledError, match="NumPy array"):
            np.asarray(selection)

        with pytest.raises(LocalDataAccessDisabledError, match="time series"):
            data.phisp.time_series(0, 0, 0)


def test_global_policy_blocks_by_default_but_context_can_allow(emdir):
    emout.disable_local_data_access()
    try:
        data = emout.Emout(emdir)

        assert isinstance(data.phisp[1], emout.data.GridDataSelection)

        with emout.local_data_policy("allow"):
            assert isinstance(data.phisp[1], emout.data.Data3d)
    finally:
        emout.reset_local_data_policy()


def test_emout_local_data_policy_override_wins_over_global(emdir):
    emout.disable_local_data_access()
    try:
        data = emout.Emout(emdir, local_data_policy="allow")

        assert isinstance(data.phisp[1], emout.data.Data3d)
    finally:
        emout.reset_local_data_policy()


def test_env_policy_blocks_local_field_reads(emdir, monkeypatch):
    monkeypatch.setenv("EMOUT_LOCAL_DATA_POLICY", "remote_required")
    emout.reset_local_data_policy()

    data = emout.Emout(emdir)

    assert isinstance(data.phisp[1], emout.data.GridDataSelection)


def test_explicit_remote_required_wins_over_allowing_context(emdir):
    data = emout.Emout(emdir, local_data_policy="remote_required")

    with emout.local_data_policy("allow"):
        selection = data.phisp[1]

    assert isinstance(selection, emout.data.GridDataSelection)
    with pytest.raises(LocalDataAccessDisabledError):
        selection.materialize()


def test_remote_open_kwargs_force_worker_side_local_access_allowed(emdir):
    data = emout.Emout(emdir, local_data_policy="remote_required")

    assert data._remote_open_kwargs["local_data_policy"] == "allow"


def test_disabled_selection_plot_renders_remotely_without_fetching_field(data, monkeypatch):
    from emout.distributed import remote_render

    calls = []
    displayed = []

    class FakeSession:
        def render_field(self, attr_name, recipe_index, emout_kwargs=None, **plot_kwargs):
            calls.append((attr_name, recipe_index, emout_kwargs, plot_kwargs))
            return FakeFuture(b"png-bytes")

    monkeypatch.setattr(remote_render, "get_or_create_session", lambda *args, **kwargs: FakeSession())
    monkeypatch.setattr(remote_render, "display_image", lambda img_bytes, ax=None: displayed.append((img_bytes, ax)))

    with emout.local_data_policy("remote_required"):
        result = data.phisp[1, :, 0, :].plot(cmap="magma")

    assert result is None
    assert len(calls) == 1
    attr_name, recipe_index, emout_kwargs, plot_kwargs = calls[0]
    assert attr_name == "phisp"
    assert recipe_index == (1, slice(0, 100, 1), 0, slice(0, 30, 1))
    assert emout_kwargs["local_data_policy"] == "allow"
    assert plot_kwargs["cmap"] == "magma"
    assert displayed == [(b"png-bytes", None)]


def test_disabled_remote_figure_without_session_raises(data, monkeypatch):
    from emout.distributed import remote_render
    from emout.distributed.remote_figure import remote_figure

    monkeypatch.setattr(remote_render, "get_or_create_session", lambda *args, **kwargs: None)

    with emout.local_data_policy("remote_required"):
        with pytest.raises(LocalDataAccessDisabledError, match="without a remote session"):
            with remote_figure():
                data.phisp[1, :, 0, :].plot(use_si=False)


def test_disabled_policy_blocks_local_relocated_field_creation(tmp_path):
    _write_grid_h5(tmp_path / "ex00_0000.h5", "ex")
    _write_relocation_inp(tmp_path / "plasma.inp")
    data = emout.Emout(tmp_path)

    with emout.local_data_policy("remote_required"):
        with pytest.raises(LocalDataAccessDisabledError, match="relocated field"):
            data.rex

    assert not (tmp_path / "rex00_0000.h5").exists()


def test_disabled_policy_blocks_existing_data3d_local_special_paths(data, tmp_path, monkeypatch):
    from emout.distributed import remote_render

    monkeypatch.setattr(remote_render, "get_or_create_session", lambda *args, **kwargs: None)
    snapshot = data.phisp[1]

    with emout.local_data_policy("remote_required"):
        with pytest.raises(LocalDataAccessDisabledError, match="remote session"):
            snapshot.plot(use_si=False)

        with pytest.raises(LocalDataAccessDisabledError, match="VTK"):
            snapshot.to_vtk(tmp_path / "field.vti", use_si=False)

        with pytest.raises(LocalDataAccessDisabledError, match="surfaces"):
            snapshot.plot_surfaces([], use_si=False)


def test_disabled_policy_blocks_existing_vector_local_plot():
    vx = emout.data.Data2d(np.ones((2, 2)), name="vx")
    vy = emout.data.Data2d(np.ones((2, 2)), name="vy")
    vector = emout.data.VectorData([vx, vy], name="vxy")

    with emout.local_data_policy("remote_required"):
        with pytest.raises(LocalDataAccessDisabledError, match="vector field"):
            vector.plot(use_si=False)
