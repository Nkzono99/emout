"""Tests for article data recording and replay."""

import json

import h5py
import numpy as np
import pytest

import emout
from emout.article import ArticleReplayEmout
from tests.conftest import create_inpfile

_BOUNDARY_INP = """!!key dx=[0.1],to_c=[10000.0]
&tmgrid
    nstep = 2
/
&mpi
    nspec = 1
    npc = 1
/
&ptcond
    boundary_type = 'complex'
    boundary_types(1) = 'sphere'
    sphere_origin(:, 1) = 1.0, 2.0, 3.0
    sphere_radius(1) = 0.5
/
"""


def _write_field(directory, name, offset=0.0):
    with h5py.File(directory / f"{name}00_0000.h5", "w") as h5:
        group = h5.create_group(name)
        for index in range(3):
            values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
            group.create_dataset(f"{index:04}", data=values + offset + index * 100)


@pytest.fixture
def article_sim(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    _write_field(sim, "phisp")
    _write_field(sim, "ex", offset=10)
    _write_field(sim, "ez", offset=20)
    create_inpfile(sim / "plasma.inp")
    return sim


def _record_empty_article(sim, records, article_name="fig_meta"):
    emout.Emout(
        sim,
        article_mode="record",
        article_records_path=records,
        article_name=article_name,
    )
    return emout.Emout(
        sim,
        article_mode="replay",
        article_records_path=records,
        article_name=article_name,
    )


def test_record_and_replay_to_numpy(article_sim, tmp_path):
    records = tmp_path / "records"

    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig1",
    )
    recorded = data.phisp[-1, :, 1, :].to_numpy()

    replay = emout.Emout(
        article_sim,
        article_mode="replay",
        article_records_path=records,
        article_name="fig1",
    )
    assert isinstance(replay, ArticleReplayEmout)
    replayed = replay.phisp[-1, :, 1, :].to_numpy()

    np.testing.assert_array_equal(replayed, recorded)

    manifest_path = next(records.glob("datasets/*/fig1/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["records"][0]
    assert record["kind"] == "materialize"
    assert record["field"] == "phisp"
    assert record["slice_axes"] == [1, 3]


def test_record_deduplicates_same_field_selection(article_sim, tmp_path):
    records = tmp_path / "records"

    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_dedupe",
    )
    data.phisp[-1, :, 1, :].to_numpy()
    data.phisp[-1, :, 1, :].plot(show=False, use_si=False)

    record_dir = next(records.glob("datasets/*/fig_dedupe"))
    manifest = json.loads((record_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 1
    with h5py.File(record_dir / "data.h5", "r") as h5:
        assert list(h5["records"]) == ["record_000000"]


def test_record_dataset_write_failure_keeps_existing_bundle(article_sim, tmp_path, monkeypatch):
    records = tmp_path / "records"
    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_atomic",
    )
    data.phisp[-1, :, 1, :].to_numpy()
    record_dir = next(records.glob("datasets/*/fig_atomic"))
    manifest_path = record_dir / "manifest.json"

    import emout.article as article_mod

    def fail_h5_write(*args, **kwargs):
        raise RuntimeError("h5 write failed")

    monkeypatch.setattr(article_mod.h5py, "File", fail_h5_write)

    with pytest.raises(RuntimeError, match="h5 write failed"):
        data.phisp[0, :, 1, :].to_numpy()

    monkeypatch.undo()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 1
    assert not (record_dir / ".data.h5.tmp").exists()
    with h5py.File(record_dir / "data.h5", "r") as h5:
        assert list(h5["records"]) == ["record_000000"]


def test_replay_rejects_unsupported_manifest_schema(article_sim, tmp_path):
    records = tmp_path / "records"
    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_schema",
    )
    data.phisp[-1, :, 1, :].to_numpy()

    manifest_path = next(records.glob("datasets/*/fig_schema/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        emout.Emout(
            article_sim,
            article_mode="replay",
            article_records_path=records,
            article_name="fig_schema",
        )


def test_replay_rejects_invalid_manifest_record(article_sim, tmp_path):
    records = tmp_path / "records"
    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_bad_record",
    )
    data.phisp[-1, :, 1, :].to_numpy()

    manifest_path = next(records.glob("datasets/*/fig_bad_record/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["records"][0]["selector"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="selector"):
        emout.Emout(
            article_sim,
            article_mode="replay",
            article_records_path=records,
            article_name="fig_bad_record",
        )


def test_replay_rejects_manifest_missing_dataset(article_sim, tmp_path):
    records = tmp_path / "records"
    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_missing_dataset",
    )
    data.phisp[-1, :, 1, :].to_numpy()

    data_path = next(records.glob("datasets/*/fig_missing_dataset/data.h5"))
    with h5py.File(data_path, "a") as h5:
        del h5["records/record_000000"]

    with pytest.raises(ValueError, match="dataset"):
        emout.Emout(
            article_sim,
            article_mode="replay",
            article_records_path=records,
            article_name="fig_missing_dataset",
        )


def test_replay_rejects_manifest_shape_mismatch(article_sim, tmp_path):
    records = tmp_path / "records"
    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_shape_mismatch",
    )
    data.phisp[-1, :, 1, :].to_numpy()

    manifest_path = next(records.glob("datasets/*/fig_shape_mismatch/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["shape"] = [999]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="shape"):
        emout.Emout(
            article_sim,
            article_mode="replay",
            article_records_path=records,
            article_name="fig_shape_mismatch",
        )


def test_record_and_replay_plot(article_sim, tmp_path):
    records = tmp_path / "records"

    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_plot",
    )
    image = data.phisp[-1, :, 1, :].plot(show=False, use_si=False)

    replay = emout.Emout(
        article_sim,
        article_mode="replay",
        article_records_path=records,
        article_name="fig_plot",
    )
    replayed_image = replay.phisp[-1, :, 1, :].plot(show=False, use_si=False)

    assert type(replayed_image) is type(image)


def test_record_and_replay_vector_components(article_sim, tmp_path):
    records = tmp_path / "records"

    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_vector",
    )
    data.exz[-1, :, 1, :].plot(mode="vec", axes="xz", show=False, use_si=False)

    replay = emout.Emout(
        article_sim,
        article_mode="replay",
        article_records_path=records,
        article_name="fig_vector",
    )
    vector = replay.exz[-1, :, 1, :]

    np.testing.assert_array_equal(vector.x_data.to_numpy(), data.ex[-1, :, 1, :].to_numpy())
    np.testing.assert_array_equal(vector.y_data.to_numpy(), data.ez[-1, :, 1, :].to_numpy())
    assert vector.component_axes == ("x", "z")


def test_replay_vector_resolution_does_not_mask_scalar_field(article_sim, tmp_path):
    records = tmp_path / "records"

    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_scalar_vectorish",
    )
    data.ex[-1, :, 1, :].to_numpy()

    replay = emout.Emout(
        article_sim,
        article_mode="replay",
        article_records_path=records,
        article_name="fig_scalar_vectorish",
    )

    with pytest.raises(AttributeError, match="not recorded"):
        replay.exz


def test_environment_switches_emout_to_record_and_replay(article_sim, tmp_path, monkeypatch):
    records = tmp_path / "records"
    monkeypatch.setenv("EMOUT_ARTICLE_MODE", "record")
    monkeypatch.setenv("EMOUT_ARTICLE_RECORDS_PATH", str(records))
    monkeypatch.setenv("EMOUT_ARTICLE_NAME", "fig_env")

    data = emout.Emout(article_sim)
    recorded = data.phisp[-1, 0, :, :].to_numpy()

    monkeypatch.setenv("EMOUT_ARTICLE_MODE", "replay")
    replay = emout.Emout(article_sim)
    replayed = replay.phisp[-1, 0, :, :].to_numpy()

    np.testing.assert_array_equal(replayed, recorded)


def test_replay_returns_source_slice_before_script_transform(article_sim, tmp_path):
    records = tmp_path / "records"

    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig_transform",
    )
    recorded = (-data.phisp[-1, :, 1, :]).to_numpy()

    replay = emout.Emout(
        article_sim,
        article_mode="replay",
        article_records_path=records,
        article_name="fig_transform",
    )
    replayed = (-replay.phisp[-1, :, 1, :]).to_numpy()

    np.testing.assert_array_equal(replayed, recorded)


def test_replay_falls_back_to_single_matching_article_name(article_sim, tmp_path):
    records = tmp_path / "records"

    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig1",
    )
    recorded = data.phisp[-1, :, 1, :].to_numpy()

    moved_source = tmp_path / "moved" / "sim"
    with pytest.warns(RuntimeWarning, match="falling back"):
        replay = emout.Emout(
            moved_source,
            article_mode="replay",
            article_records_path=records,
            article_name="fig1",
        )
    replayed = replay.phisp[-1, :, 1, :].to_numpy()

    np.testing.assert_array_equal(replayed, recorded)


def test_replay_missing_slice_raises_clear_error(article_sim, tmp_path):
    records = tmp_path / "records"
    data = emout.Emout(
        article_sim,
        article_mode="record",
        article_records_path=records,
        article_name="fig1",
    )
    data.phisp[-1, :, 1, :].to_numpy()

    replay = emout.Emout(
        article_sim,
        article_mode="replay",
        article_records_path=records,
        article_name="fig1",
    )
    with pytest.raises(KeyError, match="not recorded"):
        replay.phisp[0, :, 1, :].to_numpy()


def test_record_copies_plasma_inp_and_toml_for_replay(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / "plasma.inp").write_text(_BOUNDARY_INP, encoding="utf-8")
    (sim / "plasma.toml").write_text("[tmgrid]\nnx = 4\nny = 5\n", encoding="utf-8")

    replay = _record_empty_article(sim, tmp_path / "records")

    assert replay.inp is not None
    assert replay.is_valid()
    assert replay.toml.tmgrid.nx == 4
    record_dir = next((tmp_path / "records").glob("datasets/*/fig_meta"))
    assert (record_dir / "plasma.inp").exists()
    assert (record_dir / "plasma.toml").exists()
    source = json.loads((record_dir.parent / "source.json").read_text(encoding="utf-8"))
    assert set(source["recorded_files"]) == {"plasma.inp", "plasma.toml"}
    assert len(source["recorded_files"]["plasma.inp"]) == 64


def test_replay_rejects_recorded_file_hash_mismatch(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / "plasma.inp").write_text(_BOUNDARY_INP, encoding="utf-8")

    _record_empty_article(sim, tmp_path / "records", article_name="fig_hash")
    record_dir = next((tmp_path / "records").glob("datasets/*/fig_hash"))
    (record_dir / "plasma.inp").write_text(_BOUNDARY_INP + "\n! tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        emout.Emout(
            sim,
            article_mode="replay",
            article_records_path=tmp_path / "records",
            article_name="fig_hash",
        )


def test_replay_boundaries_from_recorded_input(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / "plasma.inp").write_text(_BOUNDARY_INP, encoding="utf-8")

    replay = _record_empty_article(sim, tmp_path / "records", article_name="fig_boundary")

    assert len(replay.boundaries) == 1
    assert replay.boundaries.types == ("sphere",)
    ax = replay.boundaries.plot(use_si=False)
    assert ax.name == "3d"


def test_replay_unsupported_api_raises_clear_error(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / "plasma.inp").write_text(_BOUNDARY_INP, encoding="utf-8")

    replay = _record_empty_article(sim, tmp_path / "records", article_name="fig_api")

    with pytest.raises(NotImplementedError, match="Article replay"):
        replay.remote()
    with pytest.raises(NotImplementedError, match="Article replay"):
        replay.particle(1)
    with pytest.raises(NotImplementedError, match="Article replay"):
        replay.backtrace


def test_record_copies_diagnostic_files_for_replay(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / "plasma.inp").write_text(_BOUNDARY_INP, encoding="utf-8")
    (sim / "icur").write_text("0 0.0 0.0\n2 1.0 2.0\n", encoding="utf-8")
    (sim / "pbody").write_text("2 10 20\n", encoding="utf-8")

    replay = _record_empty_article(sim, tmp_path / "records", article_name="fig_diag")

    assert list(replay.icur.columns) == ["1_step", "1_body1", "1_body1_ema"]
    assert replay.icur.iloc[-1]["1_step"] == 2
    assert list(replay.pbody.columns) == ["step", "body1", "body2"]
    assert replay.pbody.iloc[0]["body2"] == 20
