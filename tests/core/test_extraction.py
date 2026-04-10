"""Tests for emout.core.extraction.EmoutDataExtraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pandas as pd
import pytest

from emout.core.extraction import EmoutDataExtraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_data(directory, convkey=None):
    """Build a mock Emout object with the minimum interface used by
    EmoutDataExtraction.

    Parameters
    ----------
    directory : Path
        Simulated output directory.
    convkey : object, optional
        Unit conversion key for the mock inp.

    Returns
    -------
    MagicMock
        Mock that satisfies the Emout interface used by extraction.
    """
    mock = MagicMock()
    mock.directory = Path(directory)
    mock.inp.convkey = convkey
    mock.inp.save = MagicMock()
    mock.unit = MagicMock()
    mock.is_valid.return_value = True
    mock.icur = pd.DataFrame({"step": [0, 1], "current": [1.0, 2.0]})
    mock.pbody = pd.DataFrame({"step": [0, 1], "count": [100, 200]})
    mock.backtrace = MagicMock()
    return mock


def _make_grid_data(shape=(4, 5, 6)):
    """Return a small numpy array mimicking a grid-data time-series slice."""
    return np.random.default_rng(42).random(shape)


# ===================================================================
# Initialization
# ===================================================================


class TestEmoutDataExtractionInit:
    """Constructor creates the extract directory and saves plasma.inp."""

    def test_creates_extract_dir(self, tmp_path):
        sim_dir = tmp_path / "sim" / "run01"
        sim_dir.mkdir(parents=True)
        root = tmp_path / "extracted"
        mock = _make_mock_data(sim_dir)

        ext = EmoutDataExtraction(root, mock, nparent=1)
        assert ext.extract_dir.exists()

    def test_saves_plasma_inp(self, tmp_path):
        sim_dir = tmp_path / "sim" / "run01"
        sim_dir.mkdir(parents=True)
        mock = _make_mock_data(sim_dir)

        ext = EmoutDataExtraction(tmp_path / "out", mock, nparent=1)
        mock.inp.save.assert_called_once()

    def test_root_stored_as_path(self, tmp_path):
        sim_dir = tmp_path / "run"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)

        ext = EmoutDataExtraction(str(tmp_path / "out"), mock, nparent=0)
        assert isinstance(ext._root, Path)


# ===================================================================
# Properties delegated to Emout
# ===================================================================


class TestDelegatedProperties:
    """Properties that pass through to the wrapped Emout."""

    @pytest.fixture
    def ext(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)
        return EmoutDataExtraction(tmp_path / "out", mock, nparent=0)

    def test_directory(self, ext):
        assert isinstance(ext.directory, Path)

    def test_inp(self, ext):
        assert ext.inp is not None

    def test_unit(self, ext):
        assert ext.unit is not None

    def test_is_valid(self, ext):
        assert ext.is_valid() is True

    def test_icur(self, ext):
        df = ext.icur
        assert isinstance(df, pd.DataFrame)
        assert "step" in df.columns

    def test_pbody(self, ext):
        df = ext.pbody
        assert isinstance(df, pd.DataFrame)
        assert "step" in df.columns

    def test_backtrace(self, ext):
        bt = ext.backtrace
        assert bt is not None


# ===================================================================
# extract_dir computation
# ===================================================================


class TestExtractDir:
    """extract_dir derives the path from directory + nparent."""

    def test_nparent_0(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)
        root = tmp_path / "out"
        ext = EmoutDataExtraction(root, mock, nparent=0)
        # nparent=0 -> extract_dir == root
        assert ext.extract_dir == root

    def test_nparent_1(self, tmp_path):
        sim_dir = tmp_path / "project" / "run42"
        sim_dir.mkdir(parents=True)
        mock = _make_mock_data(sim_dir)
        root = tmp_path / "out"
        ext = EmoutDataExtraction(root, mock, nparent=1)
        assert ext.extract_dir == root / "run42"

    def test_nparent_2(self, tmp_path):
        sim_dir = tmp_path / "project" / "campaign" / "run42"
        sim_dir.mkdir(parents=True)
        mock = _make_mock_data(sim_dir)
        root = tmp_path / "out"
        ext = EmoutDataExtraction(root, mock, nparent=2)
        assert ext.extract_dir == root / "campaign" / "run42"


# ===================================================================
# __getattr__ — dynamic attribute access
# ===================================================================


class TestGetattr:
    """__getattr__ delegates to Emout and triggers save_hdf5."""

    @pytest.fixture
    def ext(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)
        # Make the mock return a numpy array for any getattr call
        shape = (2, 4, 5, 6)
        arr = np.ones(shape)
        mock.__getattr__ = lambda self, name: arr
        # We need to configure __getattr__ on the mock correctly
        mock.configure_mock(**{})
        # Override getattr to return our array
        type(mock).__getattr__ = lambda self, name: arr
        return EmoutDataExtraction(tmp_path / "out", mock, nparent=0)

    def test_scalar_field_triggers_save(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)
        shape = (2, 4, 5, 6)
        arr = np.ones(shape)

        ext = EmoutDataExtraction(tmp_path / "out", mock, nparent=0)

        # Patch save_hdf5 to track calls
        with patch.object(ext, "save_hdf5") as mock_save:
            # Make getattr on mock._data return the array
            mock.phisp = arr
            result = ext.__getattr__("phisp")
            mock_save.assert_called_once_with("phisp")

    def test_vector_field_triggers_two_saves(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)

        ext = EmoutDataExtraction(tmp_path / "out", mock, nparent=0)

        with patch.object(ext, "save_hdf5") as mock_save:
            mock.exz = MagicMock()
            ext.__getattr__("exz")
            # "exz" matches (.+)(x)(z) pattern -> saves "ex" and "ez"
            assert mock_save.call_count == 2
            mock_save.assert_any_call("ex")
            mock_save.assert_any_call("ez")

    def test_longer_vector_name_parsed(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)

        ext = EmoutDataExtraction(tmp_path / "out", mock, nparent=0)

        with patch.object(ext, "save_hdf5") as mock_save:
            mock.j1xy = MagicMock()
            ext.__getattr__("j1xy")
            # "j1xy" -> name="j1", axis1="x", axis2="y"
            mock_save.assert_any_call("j1x")
            mock_save.assert_any_call("j1y")


# ===================================================================
# save_hdf5
# ===================================================================


class TestSaveHdf5:
    """save_hdf5 writes last-timestep data to an HDF5 file."""

    def test_creates_hdf5_file(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)

        # Configure mock so that mock.phisp[-1, :, :, :] returns a 3D array
        shape_3d = (4, 5, 6)
        data_3d = np.random.default_rng(0).random(shape_3d)
        field_mock = MagicMock()
        field_mock.__getitem__ = MagicMock(return_value=data_3d)
        mock.phisp = field_mock

        ext = EmoutDataExtraction(tmp_path / "out", mock, nparent=0)
        ext.save_hdf5("phisp")

        h5_path = ext.extract_dir / "phisp00_0000.h5"
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            assert "phisp" in f
            assert "0000" in f["phisp"]
            np.testing.assert_array_equal(f["phisp"]["0000"][()], data_3d)

    def test_skips_existing_file(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)

        ext = EmoutDataExtraction(tmp_path / "out", mock, nparent=0)

        # Pre-create the file
        h5_path = ext.extract_dir / "ex00_0000.h5"
        h5_path.touch()

        # save_hdf5 should return early without writing
        ext.save_hdf5("ex")
        # File should still be empty (just touched, not a valid HDF5)
        assert h5_path.stat().st_size == 0

    def test_multiple_fields_independent(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)

        shape_3d = (3, 4, 5)
        data_ex = np.ones(shape_3d)
        data_ey = np.ones(shape_3d) * 2

        mock_ex = MagicMock()
        mock_ex.__getitem__ = MagicMock(return_value=data_ex)
        mock.ex = mock_ex

        mock_ey = MagicMock()
        mock_ey.__getitem__ = MagicMock(return_value=data_ey)
        mock.ey = mock_ey

        ext = EmoutDataExtraction(tmp_path / "out", mock, nparent=0)
        ext.save_hdf5("ex")
        ext.save_hdf5("ey")

        assert (ext.extract_dir / "ex00_0000.h5").exists()
        assert (ext.extract_dir / "ey00_0000.h5").exists()

        with h5py.File(ext.extract_dir / "ex00_0000.h5", "r") as f:
            np.testing.assert_array_equal(f["ex"]["0000"][()], data_ex)
        with h5py.File(ext.extract_dir / "ey00_0000.h5", "r") as f:
            np.testing.assert_array_equal(f["ey"]["0000"][()], data_ey)


# ===================================================================
# is_valid delegation
# ===================================================================


class TestIsValid:
    """is_valid() delegates to the wrapped Emout."""

    def test_returns_true_when_valid(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)
        mock.is_valid.return_value = True

        ext = EmoutDataExtraction(tmp_path / "out", mock, nparent=0)
        assert ext.is_valid() is True

    def test_returns_false_when_invalid(self, tmp_path):
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        mock = _make_mock_data(sim_dir)
        mock.is_valid.return_value = False

        ext = EmoutDataExtraction(tmp_path / "out", mock, nparent=0)
        assert ext.is_valid() is False
