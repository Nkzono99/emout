"""Unit tests for ``emout.utils.toml_converter``.

The module historically exposed TOML→namelist conversion helpers
(``_convert_v1``, ``_convert_v2``, ``load_toml_as_namelist``), but that
path has since been delegated to the external ``toml2inp`` command
bundled with MPIEMSES3D. Only the :class:`TomlData` attribute-access
wrapper and :func:`load_toml` remain in-process, so the tests here
cover just those.
"""

import pytest

from emout.utils.toml_converter import TomlData, load_toml


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

[tmgrid]
dt = 0.002
nx = 64
ny = 64
nz = 512
"""


# ---------------------------------------------------------------------------
# TomlData wrapper
# ---------------------------------------------------------------------------


class TestTomlData:
    def test_attribute_access(self):
        td = TomlData({"tmgrid": {"nx": 64, "ny": 32}})
        assert td.tmgrid.nx == 64
        assert td.tmgrid.ny == 32

    def test_dict_access(self):
        td = TomlData({"tmgrid": {"nx": 64}})
        assert td["tmgrid"]["nx"] == 64

    def test_list_of_dicts(self):
        td = TomlData({"species": [{"wp": 2.1}, {"wp": 0.049}]})
        assert td.species[0].wp == 2.1
        assert td.species[1].wp == 0.049

    def test_contains(self):
        td = TomlData({"a": 1, "b": 2})
        assert "a" in td
        assert "c" not in td

    def test_keys(self):
        td = TomlData({"a": 1, "b": 2})
        assert set(td.keys()) == {"a", "b"}

    def test_get_default(self):
        td = TomlData({"a": 1})
        assert td.get("a") == 1
        assert td.get("missing", 42) == 42

    def test_to_dict(self):
        d = {"a": 1, "b": {"c": 3}}
        td = TomlData(d)
        assert td.to_dict() is d

    def test_attribute_error(self):
        td = TomlData({"a": 1})
        with pytest.raises(AttributeError):
            td.nonexistent

    def test_repr(self):
        td = TomlData({"x": 1})
        assert "TomlData" in repr(td)

    def test_nested_deep(self):
        td = TomlData({"meta": {"unit_conversion": {"dx": 0.5, "to_c": 10000.0}}})
        assert td.meta.unit_conversion.dx == 0.5
        assert td.meta.unit_conversion.to_c == 10000.0


# ---------------------------------------------------------------------------
# load_toml
# ---------------------------------------------------------------------------


class TestLoadToml:
    def test_load_file(self, tmp_path):
        toml_file = tmp_path / "plasma.toml"
        toml_file.write_text(V2_TOML, encoding="utf-8")
        td = load_toml(toml_file)
        assert td.meta.format_version == 2
        assert td.species[0].wp == 2.1
        assert td.tmgrid.nx == 64

    def test_load_file_preserves_structure(self, tmp_path):
        toml_file = tmp_path / "plasma.toml"
        toml_file.write_text(V2_TOML, encoding="utf-8")
        td = load_toml(toml_file)
        # Dict-style access still works and nested lists stay as lists of TomlData.
        assert td["meta"]["unit_conversion"]["dx"] == 0.5
        assert len(td.species) == 2
        assert td.species[1].qm == 0.000545
