"""Extended tests for emout.utils.emsesinp covering uncovered lines.

Covers UnitConversionKey, AttrDict, and InpFile methods that are not
exercised by the existing test_inpfile.py suite.
"""

import f90nml
import pytest

from emout.utils.emsesinp import AttrDict, InpFile, UnitConversionKey
from emout.utils.units import Units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_inp(tmp_path, text, *, header=None):
    """Write a namelist string to a temporary .inp file and return its path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory (pytest fixture).
    text : str
        Fortran namelist content.
    header : str or None
        Optional ``!!key`` header line (written verbatim before the
        namelist).

    Returns
    -------
    pathlib.Path
        Path to the written file.
    """
    path = tmp_path / "plasma.inp"
    nml = f90nml.reads(text)
    with open(path, "w", encoding="utf-8") as f:
        if header is not None:
            f.write(header + "\n")
        f90nml.write(nml, f, force=True)
    return path


def _make_inp(tmp_path, text, *, header=None):
    """Create an InpFile from a namelist string.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory.
    text : str
        Fortran namelist content.
    header : str or None
        Optional ``!!key`` header.

    Returns
    -------
    InpFile
        Loaded InpFile instance.
    """
    path = _write_inp(tmp_path, text, header=header)
    return InpFile(str(path))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NML_BASIC = """\
&tmgrid
    dt = 0.002
    nx = 64
    ny = 64
    nz = 512
/
&mpi
    nodes(1:3) = 4, 4, 32
/
"""

NML_WITH_PLASMA = """\
&tmgrid
    dt = 0.002
    nx = 64
    ny = 64
    nz = 512
/
&mpi
    nodes(1:3) = 4, 4, 32
/
&plasma
    wp(1:2) = 1.0, 2.0
/
&intp
    path(1:2) = 100.0, 200.0
    peth(1:2) = 10.0, 20.0
    vdri(1:2) = 0.5, 1.0
/
&emissn
    curf(1:2) = 0.1, 0.2
    curfs(1:2) = 0.01, 0.02
/
"""

HEADER_KEY = "!!key dx=[0.5],to_c=[10000.0]"


@pytest.fixture
def basic_inp(tmp_path):
    """Return an InpFile with basic tmgrid + mpi groups.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory.

    Returns
    -------
    InpFile
        Loaded InpFile.
    """
    return _make_inp(tmp_path, NML_BASIC)


@pytest.fixture
def keyed_inp(tmp_path):
    """Return an InpFile with the !!key header.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory.

    Returns
    -------
    InpFile
        Loaded InpFile.
    """
    return _make_inp(tmp_path, NML_BASIC, header=HEADER_KEY)


@pytest.fixture
def full_inp(tmp_path):
    """Return an InpFile with plasma/intp/emissn groups and !!key header.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory.

    Returns
    -------
    InpFile
        Loaded InpFile.
    """
    return _make_inp(tmp_path, NML_WITH_PLASMA, header=HEADER_KEY)


# ===================================================================
# UnitConversionKey
# ===================================================================


class TestUnitConversionKey:
    """Tests for UnitConversionKey."""

    def test_keytext_property(self):
        """keytext returns the expected format string."""
        key = UnitConversionKey(dx=0.5, to_c=10000.0)
        assert key.keytext == "dx=[0.5],to_c=[10000.0]"

    def test_keytext_integer_values(self):
        """keytext works with integer-valued floats."""
        key = UnitConversionKey(dx=1, to_c=20000)
        text = key.keytext
        assert "dx=[1]" in text
        assert "to_c=[20000]" in text

    def test_load_with_header(self, tmp_path):
        """load() parses the !!key header correctly."""
        path = _write_inp(tmp_path, NML_BASIC, header=HEADER_KEY)
        key = UnitConversionKey.load(str(path))
        assert key is not None
        assert key.dx == 0.5
        assert key.to_c == 10000.0

    def test_load_without_header(self, tmp_path):
        """load() returns None when there is no !!key header."""
        path = _write_inp(tmp_path, NML_BASIC)
        key = UnitConversionKey.load(str(path))
        assert key is None

    def test_load_integer_format(self, tmp_path):
        """load() handles integer-format values like dx=[1],to_c=[20000]."""
        path = tmp_path / "plasma.inp"
        with open(path, "w", encoding="utf-8") as f:
            f.write("!!key dx=[1],to_c=[20000]\n")
            f.write("&tmgrid\n    nx = 1\n/\n")
        key = UnitConversionKey.load(str(path))
        assert key is not None
        assert key.dx == 1.0
        assert key.to_c == 20000.0

    def test_load_negative_values(self, tmp_path):
        """load() handles negative values in the header."""
        path = tmp_path / "plasma.inp"
        with open(path, "w", encoding="utf-8") as f:
            f.write("!!key dx=[-0.5],to_c=[-10000.0]\n")
            f.write("&tmgrid\n    nx = 1\n/\n")
        key = UnitConversionKey.load(str(path))
        assert key is not None
        assert key.dx == -0.5
        assert key.to_c == -10000.0

    def test_roundtrip_keytext_load(self, tmp_path):
        """keytext output can be loaded back by load()."""
        original = UnitConversionKey(dx=1.5, to_c=5000.0)
        path = tmp_path / "plasma.inp"
        with open(path, "w", encoding="utf-8") as f:
            f.write("!!key {}\n".format(original.keytext))
            f.write("&tmgrid\n    nx = 1\n/\n")
        loaded = UnitConversionKey.load(str(path))
        assert loaded.dx == original.dx
        assert loaded.to_c == original.to_c


# ===================================================================
# AttrDict
# ===================================================================


class TestAttrDict:
    """Tests for AttrDict."""

    def test_getattr_simple(self):
        """Attribute access returns dict values."""
        d = AttrDict({"a": 1, "b": "hello"})
        assert d.a == 1
        assert d.b == "hello"

    def test_getattr_missing_raises(self):
        """Accessing a missing attribute raises KeyError."""
        d = AttrDict({"a": 1})
        with pytest.raises(KeyError):
            _ = d.nonexistent

    def test_nested_access(self):
        """Nested AttrDict access works when inner value is also AttrDict."""
        inner = AttrDict({"x": 42})
        outer = AttrDict({"inner": inner})
        assert outer.inner.x == 42

    def test_dict_operations_preserved(self):
        """Standard dict operations still work."""
        d = AttrDict({"a": 1, "b": 2})
        assert len(d) == 2
        assert list(d.keys()) == ["a", "b"]
        d["c"] = 3
        assert d.c == 3


# ===================================================================
# InpFile.__contains__
# ===================================================================


class TestInpFileContains:
    """Tests for InpFile.__contains__."""

    def test_contains_group_name(self, basic_inp):
        """Group names are found by __contains__."""
        assert "tmgrid" in basic_inp
        assert "mpi" in basic_inp

    def test_contains_param_name(self, basic_inp):
        """Parameter names inside groups are found by __contains__."""
        assert "nx" in basic_inp
        assert "nodes" in basic_inp

    def test_not_contains_missing(self, basic_inp):
        """Missing keys return False."""
        assert "nonexistent" not in basic_inp
        assert "foobar" not in basic_inp

    def test_contains_searches_all_groups(self, basic_inp):
        """__contains__ searches across all groups."""
        assert "dt" in basic_inp  # in tmgrid
        assert "nodes" in basic_inp  # in mpi


# ===================================================================
# InpFile.__getitem__  (KeyError on missing)
# ===================================================================


class TestInpFileGetitem:
    """Tests for InpFile.__getitem__."""

    def test_getitem_group(self, basic_inp):
        """Accessing a group name returns the group dict."""
        grp = basic_inp["tmgrid"]
        assert "nx" in grp

    def test_getitem_param_across_groups(self, basic_inp):
        """Accessing a parameter name searches all groups."""
        assert basic_inp["dt"] == 0.002

    def test_getitem_missing_raises_keyerror(self, basic_inp):
        """Accessing a missing key raises KeyError."""
        with pytest.raises(KeyError):
            _ = basic_inp["totally_missing_key"]


# ===================================================================
# InpFile.__setitem__
# ===================================================================


class TestInpFileSetitem:
    """Tests for InpFile.__setitem__."""

    def test_setitem_param_in_group(self, basic_inp):
        """Setting a parameter value updates it inside its group."""
        basic_inp["nx"] = 128
        assert basic_inp["nx"] == 128

    def test_setitem_group_level(self, basic_inp):
        """Setting a group-level key replaces the group."""
        new_grp = f90nml.Namelist({"a": 1, "b": 2})
        basic_inp["tmgrid"] = new_grp
        assert basic_inp["tmgrid"]["a"] == 1

    def test_setitem_missing_raises_keyerror(self, basic_inp):
        """Setting a non-existent key raises KeyError."""
        with pytest.raises(KeyError):
            basic_inp["nonexistent_key"] = 999

    def test_setitem_updates_correct_group(self, basic_inp):
        """Setting a param updates only the group that contains it."""
        basic_inp["dt"] = 0.005
        assert basic_inp["tmgrid"]["dt"] == 0.005
        # mpi group should be unchanged
        assert basic_inp["mpi"]["nodes"] == [4, 4, 32]


# ===================================================================
# InpFile.remove()
# ===================================================================


class TestInpFileRemove:
    """Tests for InpFile.remove()."""

    def test_remove_group(self, basic_inp):
        """remove() can delete an entire group."""
        basic_inp.remove("mpi")
        assert "mpi" not in basic_inp.nml

    def test_remove_param(self, basic_inp):
        """remove() can delete a parameter within a group."""
        basic_inp.remove("dt")
        assert "dt" not in basic_inp.nml["tmgrid"]

    def test_remove_array_element_by_index(self, tmp_path):
        """remove(key, index=i) removes a single array element."""
        nml_text = """\
&grp
    arr(2:4) = 10, 20, 30
/
"""
        inp = _make_inp(tmp_path, nml_text)
        # Remove middle element (index 3 in Fortran 1-based)
        inp.remove("arr", index=3)
        arr = inp["arr"]
        assert 20 not in arr
        assert len(arr) == 2

    def test_remove_array_first_element_updates_start_index(self, tmp_path):
        """Removing the first element shifts start_index."""
        nml_text = """\
&grp
    arr(2:4) = 10, 20, 30
/
"""
        inp = _make_inp(tmp_path, nml_text)
        inp.remove("arr", index=2)
        # After removing first element, start_index should be 3
        assert inp.nml["grp"].start_index["arr"] == [3]
        assert inp["arr"] == [20, 30]

    def test_remove_all_elements_deletes_key(self, tmp_path):
        """Removing all elements removes the key entirely."""
        nml_text = """\
&grp
    arr(5:5) = 99
/
"""
        inp = _make_inp(tmp_path, nml_text)
        inp.remove("arr", index=5)
        assert "arr" not in inp.nml["grp"]


# ===================================================================
# InpFile.setlist()
# ===================================================================


class TestInpFileSetlist:
    """Tests for InpFile.setlist()."""

    def test_setlist_new_entry(self, tmp_path):
        """setlist() creates a new array parameter."""
        nml_text = """\
&grp
    scalar = 1
/
"""
        inp = _make_inp(tmp_path, nml_text)
        inp.setlist("grp", "arr", [10, 20, 30], start_index=1)
        assert inp["grp"]["arr"] == [10, 20, 30]
        assert inp.nml["grp"].start_index["arr"] == [1]

    def test_setlist_scalar_promoted_to_list(self, tmp_path):
        """setlist() wraps a scalar value in a list."""
        nml_text = """\
&grp
    scalar = 1
/
"""
        inp = _make_inp(tmp_path, nml_text)
        inp.setlist("grp", "single", 42, start_index=1)
        assert inp["grp"]["single"] == [42]

    def test_setlist_merge_overlapping(self, tmp_path):
        """setlist() merges when existing and new ranges overlap."""
        nml_text = """\
&grp
    arr(1:3) = 10, 20, 30
/
"""
        inp = _make_inp(tmp_path, nml_text)
        # Overwrite indices 2-4 with new values
        inp.setlist("grp", "arr", [200, 300, 400], start_index=2)
        arr = inp["grp"]["arr"]
        # Index 1 keeps old value, indices 2-4 get new values
        assert arr[0] == 10  # index 1 (old)
        assert arr[1] == 200  # index 2 (new, overwrites 20)
        assert arr[2] == 300  # index 3 (new, overwrites 30)
        assert arr[3] == 400  # index 4 (new, extends)
        assert inp.nml["grp"].start_index["arr"] == [1]

    def test_setlist_merge_non_overlapping_before(self, tmp_path):
        """setlist() merges when new range is entirely before existing."""
        nml_text = """\
&grp
    arr(5:6) = 50, 60
/
"""
        inp = _make_inp(tmp_path, nml_text)
        inp.setlist("grp", "arr", [10, 20], start_index=2)
        arr = inp["grp"]["arr"]
        # Indices 2-3: new values; 4: None gap; 5-6: old values
        assert arr[0] == 10  # index 2
        assert arr[1] == 20  # index 3
        assert arr[2] is None  # index 4 (gap)
        assert arr[3] == 50  # index 5
        assert arr[4] == 60  # index 6
        assert inp.nml["grp"].start_index["arr"] == [2]

    def test_setlist_merge_non_overlapping_after(self, tmp_path):
        """setlist() merges when new range is entirely after existing."""
        nml_text = """\
&grp
    arr(1:2) = 10, 20
/
"""
        inp = _make_inp(tmp_path, nml_text)
        inp.setlist("grp", "arr", [50, 60], start_index=5)
        arr = inp["grp"]["arr"]
        assert arr[0] == 10  # index 1
        assert arr[1] == 20  # index 2
        assert arr[2] is None  # index 3 (gap)
        assert arr[3] is None  # index 4 (gap)
        assert arr[4] == 50  # index 5
        assert arr[5] == 60  # index 6
        assert inp.nml["grp"].start_index["arr"] == [1]


# ===================================================================
# InpFile.save()
# ===================================================================


class TestInpFileSave:
    """Tests for InpFile.save()."""

    def test_save_with_convkey(self, keyed_inp, tmp_path):
        """save() writes the !!key header when convkey is present."""
        out = tmp_path / "out.inp"
        keyed_inp.save(str(out))
        with open(out, "r", encoding="utf-8") as f:
            first_line = f.readline()
        assert first_line.startswith("!!key")
        assert "dx=[0.5]" in first_line
        assert "to_c=[10000.0]" in first_line

    def test_save_without_convkey(self, basic_inp, tmp_path):
        """save() omits the header when convkey is None."""
        out = tmp_path / "out.inp"
        basic_inp.save(str(out))
        with open(out, "r", encoding="utf-8") as f:
            first_line = f.readline()
        assert not first_line.startswith("!!key")

    def test_save_roundtrip(self, keyed_inp, tmp_path):
        """save() then load produces the same data."""
        out = tmp_path / "out.inp"
        keyed_inp.save(str(out))
        reloaded = InpFile(str(out))
        assert reloaded["nx"] == keyed_inp["nx"]
        assert reloaded.convkey.dx == keyed_inp.convkey.dx
        assert reloaded.convkey.to_c == keyed_inp.convkey.to_c


# ===================================================================
# InpFile.__str__ / __repr__
# ===================================================================


class TestInpFileStr:
    """Tests for InpFile string representations."""

    def test_str_returns_string(self, basic_inp):
        """__str__ returns a non-empty string."""
        s = str(basic_inp)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_repr_equals_str(self, basic_inp):
        """__repr__ returns the same value as __str__."""
        assert repr(basic_inp) == str(basic_inp)


# ===================================================================
# InpFile.conversion()
# ===================================================================


class TestInpFileConversion:
    """Tests for InpFile.conversion()."""

    def test_conversion_updates_wp(self, full_inp):
        """conversion() transforms plasma wp values."""
        unit_from = Units(dx=0.5, to_c=10000.0)
        unit_to = Units(dx=1.0, to_c=20000.0)

        wp_before = list(full_inp["plasma"]["wp"])
        full_inp.conversion(unit_from, unit_to)
        wp_after = full_inp["plasma"]["wp"]

        # Values should change
        for before, after in zip(wp_before, wp_after):
            assert before != after

    def test_conversion_updates_intp_params(self, full_inp):
        """conversion() transforms intp path/peth/vdri values."""
        unit_from = Units(dx=0.5, to_c=10000.0)
        unit_to = Units(dx=1.0, to_c=20000.0)

        path_before = list(full_inp["intp"]["path"])
        peth_before = list(full_inp["intp"]["peth"])
        vdri_before = list(full_inp["intp"]["vdri"])

        full_inp.conversion(unit_from, unit_to)

        path_after = full_inp["intp"]["path"]
        peth_after = full_inp["intp"]["peth"]
        vdri_after = full_inp["intp"]["vdri"]

        for b, a in zip(path_before, path_after):
            assert b != a
        for b, a in zip(peth_before, peth_after):
            assert b != a
        for b, a in zip(vdri_before, vdri_after):
            assert b != a

    def test_conversion_updates_emissn_params(self, full_inp):
        """conversion() transforms emissn curf/curfs values."""
        unit_from = Units(dx=0.5, to_c=10000.0)
        unit_to = Units(dx=1.0, to_c=20000.0)

        curf_before = list(full_inp["emissn"]["curf"])
        curfs_before = list(full_inp["emissn"]["curfs"])

        full_inp.conversion(unit_from, unit_to)

        curf_after = full_inp["emissn"]["curf"]
        curfs_after = full_inp["emissn"]["curfs"]

        for b, a in zip(curf_before, curf_after):
            assert b != a
        for b, a in zip(curfs_before, curfs_after):
            assert b != a

    def test_conversion_updates_convkey(self, full_inp):
        """conversion() updates convkey to target units."""
        unit_from = Units(dx=0.5, to_c=10000.0)
        unit_to = Units(dx=1.0, to_c=20000.0)

        full_inp.conversion(unit_from, unit_to)

        assert full_inp.convkey.dx == 1.0
        assert full_inp.convkey.to_c == 20000.0

    def test_conversion_roundtrip(self, full_inp):
        """Converting there and back recovers original values."""
        unit_a = Units(dx=0.5, to_c=10000.0)
        unit_b = Units(dx=1.0, to_c=20000.0)

        wp_orig = list(full_inp["plasma"]["wp"])

        full_inp.conversion(unit_a, unit_b)
        full_inp.conversion(unit_b, unit_a)

        wp_final = full_inp["plasma"]["wp"]

        for orig, final in zip(wp_orig, wp_final):
            assert pytest.approx(orig, rel=1e-10) == final

    def test_conversion_skips_missing_groups(self, keyed_inp):
        """conversion() does not crash when plasma/intp/emissn are absent."""
        unit_from = Units(dx=0.5, to_c=10000.0)
        unit_to = Units(dx=1.0, to_c=20000.0)

        # Should not raise even though plasma/intp/emissn groups are missing
        keyed_inp.conversion(unit_from, unit_to)
        assert keyed_inp.convkey.dx == 1.0

    def test_conversion_handles_none_in_array(self, tmp_path):
        """conversion() skips None entries in sparse arrays."""
        nml_text = """\
&plasma
    wp(1:3) = 1.0, , 2.0
/
"""
        inp = _make_inp(tmp_path, nml_text, header=HEADER_KEY)
        unit_from = Units(dx=0.5, to_c=10000.0)
        unit_to = Units(dx=1.0, to_c=20000.0)

        inp.conversion(unit_from, unit_to)

        wp = inp["plasma"]["wp"]
        assert wp[1] is None  # None stays None


# ===================================================================
# InpFile.dx / .to_c properties
# ===================================================================


class TestInpFileProperties:
    """Tests for InpFile.dx and .to_c convenience properties."""

    def test_dx_property(self, keyed_inp):
        """dx property returns convkey.dx."""
        assert keyed_inp.dx == 0.5

    def test_to_c_property(self, keyed_inp):
        """to_c property returns convkey.to_c."""
        assert keyed_inp.to_c == 10000.0


# ===================================================================
# InpFile.__getattr__ / __setattr__ (AttrDict wrapping)
# ===================================================================


class TestInpFileAttrAccess:
    """Tests for InpFile attribute-style access."""

    def test_getattr_returns_attrdict_for_group(self, basic_inp):
        """Accessing a group name returns an AttrDict."""
        tmgrid = basic_inp.tmgrid
        assert isinstance(tmgrid, AttrDict)
        assert tmgrid.nx == 64

    def test_getattr_returns_value_for_param(self, basic_inp):
        """Accessing a parameter name directly returns the value."""
        assert basic_inp.nx == 64

    def test_setattr_updates_param(self, basic_inp):
        """Setting an attribute updates the namelist value."""
        basic_inp.nx = 128
        assert basic_inp.nx == 128

    def test_setattr_missing_raises_keyerror(self, basic_inp):
        """Setting a non-existent attribute raises KeyError."""
        with pytest.raises(KeyError):
            basic_inp.nonexistent = 42


# ===================================================================
# InpFile empty construction
# ===================================================================


class TestInpFileEmpty:
    """Tests for InpFile constructed without a file."""

    def test_empty_init(self):
        """InpFile() with no arguments creates empty namelist."""
        inp = InpFile()
        assert len(inp.nml) == 0
        assert inp.convkey is None

    def test_empty_init_with_convkey(self):
        """InpFile(convkey=...) stores the conversion key."""
        key = UnitConversionKey(dx=1.0, to_c=5000.0)
        inp = InpFile(convkey=key)
        assert inp.convkey is key
        assert inp.convkey.dx == 1.0
