"""toml_converter モジュールの単体テスト."""

import pytest
import f90nml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from emout.utils.toml_converter import (
    load_toml_as_namelist,
    _detect_format_version,
    _extract_unit_conversion_key,
    _convert_v1,
    _convert_v2,
)


# ---------------------------------------------------------------------------
# テスト用 TOML データ
# ---------------------------------------------------------------------------

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

[plasma]
wp = [2.1, 0.049]
cv = 1000.0

[intp]
qm = [-1.0, 0.000545]
npin = [5242880, 5242880]
path = [44.24, 1.03]
"""

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
peth = 44.24
npbnd = [0, 0, 2]

[[species]]
wp = 0.049
qm = 0.000545
npin = 5242880
path = 1.03
peth = 1.03
npbnd = [0, 0, 2]

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

V2_TOML_CONDUCTORS = """\
[meta]
format_version = 2

[ptcond.conductor_groups.spacecraft]
mtd_vchg = 0
pfixed = 0.0

[ptcond.conductor_groups.probe]
mtd_vchg = -1
pfixed = -14.68

[[ptcond.objects]]
conductor = "spacecraft"
geotype = 3
bdyradius = 2.0
bdycoord = [64.0, 64.0, 63.0]

[[ptcond.objects]]
conductor = "probe"
geotype = 3
bdyradius = 1.0
bdycoord = [32.0, 32.0, 32.0]
"""

V2_TOML_BOUNDARIES = """\
[meta]
format_version = 2

[[ptcond.boundaries]]
type = "sphere"
sphere_origin = [64, 64, 63]
sphere_radius = 2.0
boundary_conductor_id = 1

[[ptcond.boundaries]]
type = "cuboid"
cuboid_shape = [1, 10, 1, 10, 1, 10]
"""

V2_TOML_DIPOLE = """\
[meta]
format_version = 2

[[dipole.sources]]
md = 0.001
mdx = 80.0
mdy = 80.0
mdz = 63.0
mddir = 3

[[dipole.sources]]
md = 0.002
mdx = 40.0
mdy = 40.0
mdz = 63.0
mddir = 1
"""

TOML_NO_META = """\
[tmgrid]
nx = 32
ny = 32
nz = 64
"""


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _parse(toml_str):
    return tomllib.loads(toml_str)


# ---------------------------------------------------------------------------
# format_version 検出
# ---------------------------------------------------------------------------


class TestDetectFormatVersion:
    def test_v1_implicit(self):
        assert _detect_format_version(_parse(V1_TOML)) == 1

    def test_v2_explicit(self):
        assert _detect_format_version(_parse(V2_TOML)) == 2

    def test_no_meta(self):
        assert _detect_format_version(_parse(TOML_NO_META)) == 1


# ---------------------------------------------------------------------------
# UnitConversionKey 抽出
# ---------------------------------------------------------------------------


class TestExtractUnitConversionKey:
    def test_present(self):
        key = _extract_unit_conversion_key(_parse(V1_TOML))
        assert key is not None
        assert key.dx == 0.5
        assert key.to_c == 10000.0

    def test_missing(self):
        key = _extract_unit_conversion_key(_parse(TOML_NO_META))
        assert key is None


# ---------------------------------------------------------------------------
# V1 変換
# ---------------------------------------------------------------------------


class TestConvertV1:
    def test_basic_groups(self):
        nml = _convert_v1(_parse(V1_TOML))
        assert nml["tmgrid"]["nx"] == 64
        assert nml["tmgrid"]["ny"] == 64
        assert nml["tmgrid"]["nz"] == 512
        assert nml["tmgrid"]["dt"] == pytest.approx(0.002)

    def test_list_params(self):
        nml = _convert_v1(_parse(V1_TOML))
        assert nml["mpi"]["nodes"] == [4, 4, 32]
        assert nml["plasma"]["wp"] == [2.1, 0.049]
        assert nml["intp"]["qm"] == [-1.0, 0.000545]

    def test_meta_excluded(self):
        nml = _convert_v1(_parse(V1_TOML))
        assert "meta" not in nml

    def test_scalar_param(self):
        nml = _convert_v1(_parse(V1_TOML))
        assert nml["plasma"]["cv"] == 1000.0


# ---------------------------------------------------------------------------
# V2 Species 展開
# ---------------------------------------------------------------------------


class TestConvertV2Species:
    def test_species_to_flat_arrays(self):
        nml = _convert_v2(_parse(V2_TOML))
        assert nml["plasma"]["wp"] == [2.1, 0.049]
        assert nml["intp"]["qm"] == [-1.0, 0.000545]
        assert nml["intp"]["npin"] == [5242880, 5242880]
        assert nml["intp"]["path"] == [44.24, 1.03]
        assert nml["intp"]["peth"] == [44.24, 1.03]

    def test_nspec_set(self):
        nml = _convert_v2(_parse(V2_TOML))
        assert nml["system"]["nspec"] == 2

    def test_2d_param_npbnd(self):
        nml = _convert_v2(_parse(V2_TOML))
        assert nml["system"]["npbnd"] == [0, 0, 2, 0, 0, 2]

    def test_non_species_tables_preserved(self):
        nml = _convert_v2(_parse(V2_TOML))
        assert nml["tmgrid"]["nx"] == 64
        assert nml["plasma"]["cv"] == 1000.0
        assert nml["mpi"]["nodes"] == [4, 4, 32]


# ---------------------------------------------------------------------------
# V2 Conductor 展開
# ---------------------------------------------------------------------------


class TestConvertV2Conductors:
    def test_conductor_groups(self):
        nml = _convert_v2(_parse(V2_TOML_CONDUCTORS))
        assert nml["ptcond"]["npcg"] == 2
        assert nml["ptcond"]["mtd_vchg"] == [0, -1]
        assert nml["ptcond"]["pfixed"] == [0.0, -14.68]

    def test_objects(self):
        nml = _convert_v2(_parse(V2_TOML_CONDUCTORS))
        assert nml["ptcond"]["npc"] == 2
        assert nml["ptcond"]["geotype"] == [3, 3]
        assert nml["ptcond"]["bdyradius"] == [2.0, 1.0]
        assert nml["ptcond"]["bdycoord"] == [64.0, 64.0, 63.0, 32.0, 32.0, 32.0]

    def test_pcgs(self):
        nml = _convert_v2(_parse(V2_TOML_CONDUCTORS))
        assert nml["ptcond"]["pcgs"] == [1, 2]


# ---------------------------------------------------------------------------
# V2 Boundary 展開
# ---------------------------------------------------------------------------


class TestConvertV2Boundaries:
    def test_boundary_types(self):
        nml = _convert_v2(_parse(V2_TOML_BOUNDARIES))
        assert nml["ptcond"]["boundary_type"] == "complex"
        assert nml["ptcond"]["boundary_types"] == [3, 2]  # sphere=3, cuboid=2

    def test_boundary_params(self):
        nml = _convert_v2(_parse(V2_TOML_BOUNDARIES))
        assert nml["ptcond"]["sphere_origin"] == [64, 64, 63, None, None, None]
        assert nml["ptcond"]["sphere_radius"] == [2.0, None]
        assert nml["ptcond"]["cuboid_shape"] == [None] * 6 + [1, 10, 1, 10, 1, 10]
        assert nml["ptcond"]["boundary_conductor_id"] == [1, None]


# ---------------------------------------------------------------------------
# V2 Dipole 展開
# ---------------------------------------------------------------------------


class TestConvertV2Dipole:
    def test_dipole_sources(self):
        nml = _convert_v2(_parse(V2_TOML_DIPOLE))
        assert nml["dipole"]["nmd"] == 2
        assert nml["dipole"]["md"] == [0.001, 0.002]
        assert nml["dipole"]["mdx"] == [80.0, 40.0]
        assert nml["dipole"]["mddir"] == [3, 1]


# ---------------------------------------------------------------------------
# load_toml_as_namelist (ファイル I/O テスト)
# ---------------------------------------------------------------------------


class TestLoadTomlAsNamelist:
    def test_v1_file(self, tmp_path):
        toml_file = tmp_path / "plasma.toml"
        toml_file.write_text(V1_TOML, encoding="utf-8")
        nml, convkey = load_toml_as_namelist(toml_file)
        assert nml["tmgrid"]["nx"] == 64
        assert convkey is not None
        assert convkey.dx == 0.5

    def test_v2_file(self, tmp_path):
        toml_file = tmp_path / "plasma.toml"
        toml_file.write_text(V2_TOML, encoding="utf-8")
        nml, convkey = load_toml_as_namelist(toml_file)
        assert nml["plasma"]["wp"] == [2.1, 0.049]
        assert nml["system"]["nspec"] == 2
        assert convkey is not None
        assert convkey.to_c == 10000.0

    def test_no_meta_file(self, tmp_path):
        toml_file = tmp_path / "plasma.toml"
        toml_file.write_text(TOML_NO_META, encoding="utf-8")
        nml, convkey = load_toml_as_namelist(toml_file)
        assert nml["tmgrid"]["nx"] == 32
        assert convkey is None
