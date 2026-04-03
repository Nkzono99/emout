"""Emout().toml での group 展開挙動テスト."""

import h5py
import numpy as np

import emout
from emout.utils.toml_converter import load_toml


GROUPED_TOML = """\
[meta]
format_version = 2

[meta.unit_conversion]
dx = 0.5
to_c = 10000.0

[species_groups.background]
qm = -1.0
path = 44.24

[[species]]
group_id = "background"
wp = 2.1
npin = 100

[meta.physical.species_groups.electron]
mass_ratio = 1.0
qm_sign = -1

[[meta.physical.species]]
group_id = "electron"
temperature_eV = 3.0

[ptcond.object_groups.probe]
conductor = "spacecraft"
geotype = 3

[ptcond.conductor_groups.spacecraft]
mtd_vchg = 0

[[ptcond.objects]]
group_id = "probe"
bdyradius = 2.0
"""


def _create_h5file(path, name, timesteps, shape):
    h5 = h5py.File(str(path), "w")
    group = h5.create_group(name)
    for i in range(timesteps):
        group.create_dataset(f"{i:04}", data=np.zeros(shape), dtype="f")
    h5.close()


def _setup_emdir(tmpdir):
    _create_h5file(tmpdir / "phisp00_0000.h5", "phisp", 3, (10, 10, 10))
    (tmpdir / "plasma.toml").write_text(GROUPED_TOML, encoding="utf-8")


class TestLoadTomlResolvedGroups:
    def test_default_load_preserves_groups(self, tmp_path):
        toml_file = tmp_path / "plasma.toml"
        toml_file.write_text(GROUPED_TOML, encoding="utf-8")

        data = load_toml(toml_file)

        assert data.species[0].group_id == "background"
        assert "species_groups" in data
        assert "object_groups" in data.ptcond

    def test_resolve_and_purge_groups(self, tmp_path):
        toml_file = tmp_path / "plasma.toml"
        toml_file.write_text(GROUPED_TOML, encoding="utf-8")

        data = load_toml(toml_file, resolve_groups=True, purge_groups=True)

        assert data.species[0].wp == 2.1
        assert data.species[0].qm == -1.0
        assert data.species[0].path == 44.24
        assert "species_groups" not in data

        assert data.meta.physical.species[0].mass_ratio == 1.0
        assert data.meta.physical.species[0].qm_sign == -1
        assert data.meta.physical.species[0].temperature_eV == 3.0
        assert "species_groups" not in data.meta.physical

        assert data.ptcond.objects[0].conductor == "spacecraft"
        assert data.ptcond.objects[0].geotype == 3
        assert data.ptcond.objects[0].bdyradius == 2.0
        assert "object_groups" not in data.ptcond


class TestEmoutTomlResolvedGroups:
    def test_emout_toml_resolves_groups(self, tmp_path):
        _setup_emdir(tmp_path)

        data = emout.Emout(tmp_path)

        assert data.toml is not None
        assert data.toml.species[0].qm == -1.0
        assert data.toml.species[0].path == 44.24
        assert "species_groups" not in data.toml
        assert data.toml.meta.physical.species[0].mass_ratio == 1.0
        assert data.toml.ptcond.objects[0].conductor == "spacecraft"
        assert "object_groups" not in data.toml.ptcond
