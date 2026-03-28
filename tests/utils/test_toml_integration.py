"""TOML ファイル読み込みの統合テスト.

Emout / DirectoryInspector を通じて plasma.toml を読み込み、
既存の InpFile API がすべて動作することを確認する。
"""

import f90nml
import h5py
import numpy as np
import pytest

import emout

# テスト用 plasma.inp 内容 (conftest.py と同じパラメータ)
INP_NML = """!!key dx=[0.5],to_c=[10000.0]
&tmgrid
    dt = 0.0020000000000000005
    nx = 64
    ny = 64
    nz = 512
/
&mpi
    nodes(1:3) = 4, 4, 32
/
"""

# テスト用 V1 plasma.toml
V1_TOML = """\
[meta.unit_conversion]
dx = 0.5
to_c = 10000.0

[tmgrid]
dt = 0.002
nx = 64
ny = 64
nz = 512

[mpi]
nodes = [4, 4, 32]
"""

# テスト用 V2 plasma.toml
V2_TOML = """\
[meta]
format_version = 2

[meta.unit_conversion]
dx = 0.5
to_c = 10000.0

[[species]]
wp = 2.1
qm = -1.0
npin = 5242880
path = 44.24

[[species]]
wp = 0.049
qm = 0.000545
npin = 5242880
path = 1.03

[plasma]
cv = 1000.0

[tmgrid]
dt = 0.002
nx = 64
ny = 64
nz = 512

[mpi]
nodes = [4, 4, 32]
"""

# TOML で異なる値を持つバージョン (優先確認用)
TOML_PRIORITY = """\
[meta.unit_conversion]
dx = 1.0
to_c = 20000.0

[tmgrid]
dt = 0.005
nx = 128
ny = 128
nz = 256
"""


def _create_h5file(path, name, timesteps, shape):
    h5 = h5py.File(str(path), "w")
    group = h5.create_group(name)
    for i in range(timesteps):
        group.create_dataset(f"{i:04}", data=np.zeros(shape), dtype="f")
    h5.close()


def _create_inp(path):
    nml = f90nml.reads(INP_NML)
    with open(str(path), "w", encoding="utf-8") as f:
        f.write("!!key dx=[0.5],to_c=[10000.0]\n")
        f90nml.write(nml, f, force=True)


def _setup_emdir(tmpdir, param_file_content=None, param_file_name=None):
    """最小限の EMSES ディレクトリを作成する。"""
    _create_h5file(tmpdir / "phisp00_0000.h5", "phisp", 3, (10, 10, 10))

    if param_file_content is not None and param_file_name is not None:
        with open(str(tmpdir / param_file_name), "w", encoding="utf-8") as f:
            f.write(param_file_content)


# ---------------------------------------------------------------------------
# V1 TOML テスト
# ---------------------------------------------------------------------------


class TestEmoutWithTomlV1:
    def test_getitem_with_group(self, tmp_path):
        _setup_emdir(tmp_path, V1_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.inp["tmgrid"]["nx"] == 64
        assert data.inp["tmgrid"]["ny"] == 64
        assert data.inp["tmgrid"]["nz"] == 512

    def test_getitem_without_group(self, tmp_path):
        _setup_emdir(tmp_path, V1_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.inp["nx"] == 64
        assert data.inp["nodes"] == [4, 4, 32]

    def test_getattr_with_group(self, tmp_path):
        _setup_emdir(tmp_path, V1_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.inp.tmgrid.nx == 64
        assert data.inp.mpi.nodes == [4, 4, 32]

    def test_getattr_without_group(self, tmp_path):
        _setup_emdir(tmp_path, V1_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.inp.nx == 64
        assert data.inp.ny == 64
        assert data.inp.nz == 512
        assert data.inp.nodes == [4, 4, 32]

    def test_unit_initialized(self, tmp_path):
        _setup_emdir(tmp_path, V1_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.unit is not None
        assert data.inp.dx == 0.5
        assert data.inp.to_c == 10000.0


# ---------------------------------------------------------------------------
# V2 TOML テスト
# ---------------------------------------------------------------------------


class TestEmoutWithTomlV2:
    def test_species_flattened(self, tmp_path):
        _setup_emdir(tmp_path, V2_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.inp["plasma"]["wp"] == [2.1, 0.049]
        assert data.inp["intp"]["qm"] == [-1.0, 0.000545]
        assert data.inp["intp"]["npin"] == [5242880, 5242880]

    def test_species_flat_access(self, tmp_path):
        _setup_emdir(tmp_path, V2_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.inp.wp == [2.1, 0.049]
        assert data.inp.qm == [-1.0, 0.000545]

    def test_nspec(self, tmp_path):
        _setup_emdir(tmp_path, V2_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.inp.nspec == 2

    def test_non_species_params(self, tmp_path):
        _setup_emdir(tmp_path, V2_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.inp.nx == 64
        assert data.inp.cv == 1000.0
        assert data.inp.nodes == [4, 4, 32]


# ---------------------------------------------------------------------------
# 優先度テスト
# ---------------------------------------------------------------------------


class TestTomlPriority:
    def test_toml_over_inp(self, tmp_path):
        """plasma.toml と plasma.inp が共存する場合、toml が優先される。"""
        _setup_emdir(tmp_path, TOML_PRIORITY, "plasma.toml")
        _create_inp(tmp_path / "plasma.inp")
        data = emout.Emout(tmp_path)
        # TOML の値が使われること
        assert data.inp.nx == 128
        assert data.inp.dx == 1.0

    def test_inp_fallback(self, tmp_path):
        """plasma.toml がない場合、plasma.inp にフォールバックする。"""
        _setup_emdir(tmp_path)
        _create_inp(tmp_path / "plasma.inp")
        data = emout.Emout(tmp_path)
        assert data.inp.nx == 64
        assert data.inp.dx == 0.5


# ---------------------------------------------------------------------------
# 明示的ファイル名指定
# ---------------------------------------------------------------------------


class TestExplicitFilename:
    def test_explicit_toml(self, tmp_path):
        _setup_emdir(tmp_path, V1_TOML, "custom.toml")
        data = emout.Emout(tmp_path, inpfilename="custom.toml")
        assert data.inp.nx == 64

    def test_no_param_file(self, tmp_path):
        _setup_emdir(tmp_path)
        data = emout.Emout(tmp_path)
        assert data.inp is None


# ---------------------------------------------------------------------------
# data.toml アクセステスト
# ---------------------------------------------------------------------------


class TestTomlDataAccess:
    def test_v1_toml_direct_access(self, tmp_path):
        """V1 TOML の生データに直接アクセスできる。"""
        _setup_emdir(tmp_path, V1_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.toml is not None
        assert data.toml.tmgrid.nx == 64
        assert data.toml["tmgrid"]["nz"] == 512
        assert data.toml.meta.unit_conversion.dx == 0.5

    def test_v2_toml_species_access(self, tmp_path):
        """V2 TOML の [[species]] に直接アクセスできる。"""
        _setup_emdir(tmp_path, V2_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert data.toml is not None
        assert len(data.toml.species) == 2
        assert data.toml.species[0].wp == 2.1
        assert data.toml.species[0].qm == -1.0
        assert data.toml.species[1].wp == 0.049
        assert data.toml.species[1].qm == 0.000545

    def test_toml_none_for_inp(self, tmp_path):
        """plasma.inp のみの場合は data.toml は None。"""
        _setup_emdir(tmp_path)
        _create_inp(tmp_path / "plasma.inp")
        data = emout.Emout(tmp_path)
        assert data.toml is None

    def test_toml_contains(self, tmp_path):
        _setup_emdir(tmp_path, V1_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        assert "tmgrid" in data.toml
        assert "nonexistent" not in data.toml

    def test_toml_keys(self, tmp_path):
        _setup_emdir(tmp_path, V1_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        keys = list(data.toml.keys())
        assert "meta" in keys
        assert "tmgrid" in keys

    def test_toml_to_dict(self, tmp_path):
        _setup_emdir(tmp_path, V1_TOML, "plasma.toml")
        data = emout.Emout(tmp_path)
        d = data.toml.to_dict()
        assert isinstance(d, dict)
        assert d["tmgrid"]["nx"] == 64
