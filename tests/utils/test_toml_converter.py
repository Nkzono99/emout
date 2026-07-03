"""Unit tests for ``emout.utils.toml_converter``.

The module exposes the :class:`TomlData` attribute-access wrapper,
:func:`load_toml`, and the TOML-backed :class:`InpFile` compatibility
view used by ``data.inp``.
"""

import pytest

from emout.utils.toml_converter import TomlData, load_toml, load_toml_as_inp


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

    def test_transparent_access_collects_list_child_values(self):
        td = TomlData({"species": [{"wp": 2.1}, {"wp": 0.049}]})

        assert td.wp == [2.1, 0.049]

    def test_transparent_access_finds_nested_scalar(self):
        td = TomlData({"ptcond": {"zssurf": 60.0}})

        assert td.zssurf == 60.0

    def test_transparent_access_returns_scalar_for_single_list_child_value(self):
        td = TomlData({"ptcond": {"boundaries": [{"type": "flat-surface", "zssurf": 60.0}]}})

        assert td.zssurf == 60.0

    def test_transparent_access_prefers_direct_key(self):
        td = TomlData({"wp": 9.0, "species": [{"wp": 2.1}, {"wp": 0.049}]})

        assert td.wp == 9.0

    def test_transparent_access_prefers_shallow_nested_key(self):
        td = TomlData({"ptcond": {"zssurf": 60.0, "boundaries": [{"zssurf": 70.0}]}})

        assert td.zssurf == 60.0

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


# ---------------------------------------------------------------------------
# load_toml_as_inp
# ---------------------------------------------------------------------------


class TestLoadTomlAsInp:
    def test_loads_v2_species_as_legacy_arrays(self, tmp_path):
        toml_file = tmp_path / "plasma.toml"
        toml_file.write_text(V2_TOML, encoding="utf-8")

        inp = load_toml_as_inp(toml_file)

        assert inp.nx == 64
        assert inp.nspec == 2
        assert inp.wp == [2.1, 0.049]
        assert inp.qm == [-1.0, 0.000545]
        assert inp.path == [44.24, 1.03]
        assert inp.dx == 0.5

    def test_flattens_ptcond_boundary_entries(self, tmp_path):
        toml_file = tmp_path / "plasma.toml"
        toml_file.write_text(
            """\
[ptcond]
boundary_type = "complex"

[[ptcond.boundaries]]
type = "sphere"
sphere_origin = [1.0, 2.0, 3.0]
sphere_radius = 0.5

[[ptcond.boundaries]]
type = "cuboid"
cuboid_min = [0.0, 0.0, 0.0]
cuboid_max = [1.0, 1.0, 1.0]
""",
            encoding="utf-8",
        )

        inp = load_toml_as_inp(toml_file)

        assert inp.boundary_type == "complex"
        assert inp.boundary_types == ["sphere", "cuboid"]
        assert inp.sphere_origin == [[1.0, 2.0, 3.0], None]
        assert inp.nml["ptcond"].start_index["sphere_origin"] == [None, 1]
