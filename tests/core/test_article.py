"""Tests for article data recording and replay."""

import json

import h5py
import numpy as np
import pytest

import emout
from emout.article import ArticleReplayEmout
from tests.conftest import create_inpfile


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
