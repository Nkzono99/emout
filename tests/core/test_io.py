"""Tests for :mod:`emout.core.io` — DirectoryInspector and GridDataLoader.

Uses ``tmp_path`` to create minimal EMSES-like directory structures
with ``create_h5file()`` and ``create_inpfile()`` from conftest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest

from emout.core.io.directory import DirectoryInspector
from emout.core.io.grid import GridDataLoader
from emout.utils import InpFile, Units

# Re-use helpers from conftest (they are importable as regular functions).
from tests.conftest import create_h5file, create_inpfile, nml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_INP = """\
!!key dx=[0.5],to_c=[10000.0]
&tmgrid
    dt = 0.002
    nx = 64
    ny = 64
    nz = 512
    nstep = 100
/
&mpi
    nodes(1:3) = 4, 4, 32
/
&emissn
    nspec = 2
    npc = 1
/
&esorem
    mtd_vbnd(1:3) = 0, 0, 0
/
"""


def _write_inp(directory: Path, content: str = _MINIMAL_INP) -> Path:
    path = directory / "plasma.inp"
    path.write_text(content)
    return path


def _write_icur(directory: Path, last_step: int = 100, *, nspec: int = 2, npc: int = 1) -> Path:
    """Write a minimal ``icur`` file with one header line and a last line."""
    icur_path = directory / "icur"
    cols_per_spec = 1 + npc * 2  # step + npc*(body + ema)
    # Build two lines: line 1 and line for last_step
    lines = []
    for step in [1, last_step]:
        vals = [str(step)] + ["0.0"] * (cols_per_spec * nspec - 1)
        # Pad to have nspec*cols_per_spec columns
        while len(vals) < cols_per_spec * nspec:
            vals.append("0.0")
        lines.append("  ".join(vals))
    icur_path.write_text("\n".join(lines) + "\n")
    return icur_path


def _write_pbody(directory: Path, npc: int = 1) -> Path:
    pbody_path = directory / "pbody"
    # step + npc+1 body columns
    cols = ["1"] + ["100"] * (npc + 1)
    pbody_path.write_text("  ".join(cols) + "\n")
    return pbody_path


# ===================================================================
# DirectoryInspector.__init__
# ===================================================================


class TestDirectoryInspectorInit:
    def test_basic_init_without_inp(self, tmp_path: Path):
        di = DirectoryInspector(tmp_path, inpfilename=None)
        assert di.main_directory == tmp_path.resolve()
        assert di.inp is None
        assert di.unit is None

    def test_init_loads_inp(self, tmp_path: Path):
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)
        assert di.inp is not None
        assert di.unit is not None

    def test_init_with_string_directory(self, tmp_path: Path):
        _write_inp(tmp_path)
        di = DirectoryInspector(str(tmp_path))
        assert di.main_directory == tmp_path.resolve()
        assert di.inp is not None

    def test_init_with_input_path(self, tmp_path: Path):
        inp_dir = tmp_path / "inputs"
        inp_dir.mkdir()
        inp_path = _write_inp(inp_dir)

        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        di = DirectoryInspector(
            tmp_path,
            input_path=inp_path,
            output_directory=out_dir,
        )
        assert di.input_path == inp_path.resolve()
        assert di.main_directory == out_dir.resolve()
        assert di.inp is not None

    def test_init_with_output_directory(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path, output_directory=out_dir)
        assert di.main_directory == out_dir.resolve()

    def test_init_with_missing_inp_file(self, tmp_path: Path):
        # plasma.inp doesn't exist — should silently skip.
        di = DirectoryInspector(tmp_path)
        assert di.inp is None


# ===================================================================
# _fetch_append_directories
# ===================================================================


class TestFetchAppendDirectories:
    def test_auto_discovers_numbered_dirs(self, tmp_path: Path):
        main = tmp_path / "run"
        main.mkdir()
        _write_inp(main)

        # Candidates are <main>_2, <main>_3 (sibling directories).
        # Each needs a plasma.inp (so is_valid can read nstep) and icur.
        for i in [2, 3]:
            d = tmp_path / f"run_{i}"
            d.mkdir()
            _write_inp(d)
            _write_icur(d, last_step=100)

        di = DirectoryInspector(main, append_directories="auto")
        assert isinstance(di.append_directories, list)
        assert len(di.append_directories) == 2

    def test_auto_stops_when_dir_missing(self, tmp_path: Path):
        main = tmp_path / "run"
        main.mkdir()
        _write_inp(main)
        # No run_2 exists
        di = DirectoryInspector(main, append_directories="auto")
        assert di.append_directories == []

    def test_explicit_append_dirs(self, tmp_path: Path):
        main = tmp_path / "main"
        main.mkdir()
        _write_inp(main)

        extra = tmp_path / "extra"
        extra.mkdir()

        di = DirectoryInspector(main, append_directories=[extra])
        assert len(di.append_directories) == 1
        assert di.append_directories[0] == extra.resolve()


# ===================================================================
# _load_inpfile and _load_from_inp
# ===================================================================


class TestLoadInpFile:
    def test_load_from_default_plasma_inp(self, tmp_path: Path):
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)
        assert di.inp is not None
        assert di.inpfilename == "plasma.inp"

    def test_skip_when_inpfilename_none(self, tmp_path: Path):
        di = DirectoryInspector(tmp_path, inpfilename=None)
        assert di.inp is None

    def test_explicit_nonexistent_inp_file(self, tmp_path: Path):
        di = DirectoryInspector(tmp_path, inpfilename="missing.inp")
        assert di.inp is None

    def test_load_explicit_inp_file(self, tmp_path: Path):
        path = tmp_path / "custom.inp"
        path.write_text(_MINIMAL_INP)
        di = DirectoryInspector(tmp_path, inpfilename="custom.inp")
        assert di.inp is not None

    def test_toml_file_without_toml2inp(self, tmp_path: Path):
        # Write a plasma.toml but toml2inp is not available
        toml_path = tmp_path / "plasma.toml"
        toml_path.write_text("[tmgrid]\nnx = 10\n")
        # Also write a plasma.inp so loading can proceed
        _write_inp(tmp_path)
        with patch("shutil.which", return_value=None):
            di = DirectoryInspector(tmp_path)
        # Should still load the .inp that exists
        assert di.inp is not None


# ===================================================================
# is_valid
# ===================================================================


class TestIsValid:
    def test_valid_directory(self, tmp_path: Path):
        _write_inp(tmp_path)
        _write_icur(tmp_path, last_step=100)
        di = DirectoryInspector(tmp_path)
        assert di.is_valid() is True

    def test_invalid_wrong_step(self, tmp_path: Path):
        _write_inp(tmp_path)
        _write_icur(tmp_path, last_step=50)  # nstep=100 but icur says 50
        di = DirectoryInspector(tmp_path)
        assert di.is_valid() is False

    def test_invalid_no_icur(self, tmp_path: Path):
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)
        assert di.is_valid() is False

    def test_valid_uses_last_append_dir(self, tmp_path: Path):
        main = tmp_path / "main"
        main.mkdir()
        _write_inp(main)

        append = tmp_path / "append"
        append.mkdir()
        _write_icur(append, last_step=100)

        di = DirectoryInspector(main, append_directories=[append])
        assert di.is_valid() is True


# ===================================================================
# read_icur_as_dataframe
# ===================================================================


class TestReadIcur:
    def test_read_icur(self, tmp_path: Path):
        _write_inp(tmp_path)
        _write_icur(tmp_path, last_step=100, nspec=2, npc=1)
        di = DirectoryInspector(tmp_path)
        df = di.read_icur_as_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # two lines in the icur file

    def test_read_icur_no_inp_raises(self, tmp_path: Path):
        di = DirectoryInspector(tmp_path, inpfilename=None)
        with pytest.raises(RuntimeError, match="read_icur"):
            di.read_icur_as_dataframe()

    def test_read_icur_no_file_raises(self, tmp_path: Path):
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)
        with pytest.raises(FileNotFoundError):
            di.read_icur_as_dataframe()


# ===================================================================
# read_pbody_as_dataframe
# ===================================================================


class TestReadPbody:
    def test_read_pbody(self, tmp_path: Path):
        _write_inp(tmp_path)
        _write_pbody(tmp_path, npc=1)
        di = DirectoryInspector(tmp_path)
        df = di.read_pbody_as_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "step" in df.columns
        assert "body1" in df.columns

    def test_read_pbody_no_inp_raises(self, tmp_path: Path):
        di = DirectoryInspector(tmp_path, inpfilename=None)
        with pytest.raises(RuntimeError, match="read_pbody"):
            di.read_pbody_as_dataframe()

    def test_read_pbody_no_file_raises(self, tmp_path: Path):
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)
        with pytest.raises(FileNotFoundError):
            di.read_pbody_as_dataframe()


# ===================================================================
# GridDataLoader._find_h5file
# ===================================================================


class TestGridDataLoaderFindH5:
    def test_find_existing_h5file(self, tmp_path: Path):
        create_h5file(tmp_path / "phisp00_0000.h5", "phisp", 2, (4, 4, 4))
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)
        loader = GridDataLoader(di, {})
        result = loader._find_h5file(tmp_path, "phisp")
        assert result.name == "phisp00_0000.h5"

    def test_find_missing_h5file_raises(self, tmp_path: Path):
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)
        loader = GridDataLoader(di, {})
        with pytest.raises(FileNotFoundError):
            loader._find_h5file(tmp_path, "nonexistent")


# ===================================================================
# GridDataLoader.load — scalar pipeline
# ===================================================================


class TestGridDataLoaderLoad:
    def test_load_scalar_field(self, tmp_path: Path):
        create_h5file(tmp_path / "phisp00_0000.h5", "phisp", 3, (4, 4, 4))
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)
        loader = GridDataLoader(di, {})
        series = loader.load("phisp")
        assert series is not None
        assert len(series) == 3

    def test_load_with_append_directory(self, tmp_path: Path):
        main = tmp_path / "main"
        main.mkdir()
        create_h5file(main / "nd1p00_0000.h5", "nd1p", 2, (4, 4, 4))
        _write_inp(main)

        append = tmp_path / "append"
        append.mkdir()
        create_h5file(append / "nd1p00_0000.h5", "nd1p", 3, (4, 4, 4))

        di = DirectoryInspector(main, append_directories=[append])
        loader = GridDataLoader(di, {})
        series = loader.load("nd1p")
        # Chained: 2 + 3 - 1 overlap = 4 timesteps
        assert len(series) == 4

    def test_load_missing_field_raises(self, tmp_path: Path):
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)
        loader = GridDataLoader(di, {})
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent")

    def test_load_with_unit_map(self, tmp_path: Path):
        create_h5file(tmp_path / "ex00_0000.h5", "ex", 2, (4, 4, 4))
        _write_inp(tmp_path)
        di = DirectoryInspector(tmp_path)

        # Provide a mock unit map
        unit_map = {
            "t": lambda di: None,
            "axis": lambda di: None,
            "ex": lambda di: None,
        }
        loader = GridDataLoader(di, unit_map)
        series = loader.load("ex")
        assert series is not None

    def test_load_without_units(self, tmp_path: Path):
        # Write inp without conversion key
        path = tmp_path / "plasma.inp"
        path.write_text("&tmgrid\n    nx = 8\n/\n")
        create_h5file(tmp_path / "phisp00_0000.h5", "phisp", 2, (4, 4, 4))
        di = DirectoryInspector(tmp_path)
        assert di.unit is None
        loader = GridDataLoader(di, {})
        series = loader.load("phisp")
        assert series is not None

    def test_load_append_dir_missing_file_skipped(self, tmp_path: Path):
        main = tmp_path / "main"
        main.mkdir()
        create_h5file(main / "nd1p00_0000.h5", "nd1p", 2, (4, 4, 4))
        _write_inp(main)

        append = tmp_path / "append"
        append.mkdir()
        # Don't create h5 in append dir — should be silently skipped

        di = DirectoryInspector(main, append_directories=[append])
        loader = GridDataLoader(di, {})
        series = loader.load("nd1p")
        assert len(series) == 2  # Only main directory data
