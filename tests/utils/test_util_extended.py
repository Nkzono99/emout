"""Extended tests for emout/utils/util.py covering previously uncovered lines."""

from types import SimpleNamespace

import numpy as np
import pytest

from emout.utils.util import (
    DataFileInfo,
    RegexDict,
    apply_offset,
    hole_mask,
    interp2d,
    range_with_slice,
)


# ---------------------------------------------------------------------------
# interp2d
# ---------------------------------------------------------------------------


class TestInterp2d:
    def test_constant_mesh_early_return(self):
        """All elements equal -> early return with constant array."""
        mesh = np.full((4, 6), 3.5)
        result = interp2d(mesh, 2)
        assert result.shape == (8, 12)
        np.testing.assert_allclose(result, 3.5)

    def test_constant_mesh_zero(self):
        """Constant mesh of zeros."""
        mesh = np.zeros((3, 5))
        result = interp2d(mesh, 3)
        assert result.shape == (9, 15)
        np.testing.assert_allclose(result, 0.0)

    def test_varying_mesh(self):
        """Non-constant mesh triggers full interpolation path."""
        mesh = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = interp2d(mesh, 2, method="linear")
        assert result.shape == (4, 4)
        # Corners should be preserved
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[0, -1], 1.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, 0], 2.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, -1], 3.0, atol=1e-10)


# ---------------------------------------------------------------------------
# range_with_slice — negative indices
# ---------------------------------------------------------------------------


class TestRangeWithSlice:
    def test_negative_start(self):
        """Negative start is resolved against maxlen."""
        result = list(range_with_slice(slice(-3, None), 10))
        assert result == [7, 8, 9]

    def test_negative_stop(self):
        """Negative stop is resolved against maxlen."""
        result = list(range_with_slice(slice(0, -2), 10))
        assert result == list(range(0, 8))

    def test_negative_start_and_stop(self):
        result = list(range_with_slice(slice(-5, -2), 10))
        assert result == [5, 6, 7]

    def test_positive_slice(self):
        """Positive values pass through unchanged."""
        result = list(range_with_slice(slice(1, 4, 2), 10))
        assert result == [1, 3]

    def test_defaults(self):
        """None start/stop/step uses defaults 0/maxlen/1."""
        result = list(range_with_slice(slice(None, None, None), 5))
        assert result == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# apply_offset
# ---------------------------------------------------------------------------


class TestApplyOffset:
    def test_left(self):
        arr = np.array([10.0, 20.0, 30.0])
        result = apply_offset(arr.copy(), "left")
        np.testing.assert_allclose(result, [0.0, 10.0, 20.0])

    def test_center(self):
        arr = np.array([10.0, 20.0, 30.0])
        result = apply_offset(arr.copy(), "center")
        np.testing.assert_allclose(result, [-10.0, 0.0, 10.0])

    def test_right(self):
        arr = np.array([10.0, 20.0, 30.0])
        result = apply_offset(arr.copy(), "right")
        np.testing.assert_allclose(result, [-20.0, -10.0, 0.0])

    def test_numeric_offset(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = apply_offset(arr.copy(), 5.0)
        np.testing.assert_allclose(result, [6.0, 7.0, 8.0])

    def test_2d_left(self):
        """left uses ravel()[0], so works on multi-dim arrays too."""
        arr = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = apply_offset(arr.copy(), "left")
        np.testing.assert_allclose(result, [[0.0, 10.0], [20.0, 30.0]])


# ---------------------------------------------------------------------------
# RegexDict — KeyError and .get() default
# ---------------------------------------------------------------------------


class TestRegexDictExtended:
    def test_getitem_raises_keyerror(self):
        d = RegexDict({r"foo": 1})
        with pytest.raises(KeyError):
            d["bar"]

    def test_get_returns_default_on_miss(self):
        d = RegexDict({r"foo": 1})
        assert d.get("bar") is None

    def test_get_returns_custom_default(self):
        d = RegexDict({r"foo": 1})
        assert d.get("bar", 42) == 42

    def test_get_returns_value_on_hit(self):
        d = RegexDict({r"x[0-9]+": 99})
        assert d.get("x5") == 99


# ---------------------------------------------------------------------------
# DataFileInfo — None filename, Path filename, directory/abspath
# ---------------------------------------------------------------------------


class TestDataFileInfo:
    def test_none_filename(self):
        info = DataFileInfo(None)
        assert info.filename is None
        assert info.directory is None
        assert info.abspath is None
        assert str(info) == "None"

    def test_string_filename(self):
        info = DataFileInfo("/tmp/test/data.h5")
        from pathlib import Path

        assert isinstance(info.filename, Path)
        assert info.filename == Path("/tmp/test/data.h5")
        assert info.abspath == Path("/tmp/test/data.h5").resolve()

    def test_path_filename(self):
        from pathlib import Path

        p = Path("/tmp/test/data.h5")
        info = DataFileInfo(p)
        assert info.filename is p  # exact same object
        assert isinstance(info.filename, Path)

    def test_directory(self):
        from pathlib import Path

        info = DataFileInfo("/tmp/test/data.h5")
        assert info.directory == Path("/tmp/test").resolve()

    def test_str(self):
        info = DataFileInfo("/tmp/test/data.h5")
        assert str(info) == "/tmp/test/data.h5"


# ---------------------------------------------------------------------------
# hole_mask
# ---------------------------------------------------------------------------


class TestHoleMask:
    def _make_inp(self, nx=10, ny=10, nz=10):
        return SimpleNamespace(
            nx=nx,
            ny=ny,
            nz=nz,
            xlrechole=[2],
            xurechole=[5],
            ylrechole=[3],
            yurechole=[7],
            zssurf=6,
            zlrechole=[0, 4],  # zlrechole[1] == 4
        )

    def test_basic_shape(self):
        inp = self._make_inp()
        mask = hole_mask(inp)
        assert mask.shape == (11, 11, 11)  # nz+1, ny+1, nx+1
        assert mask.dtype == bool

    def test_hole_region_is_false(self):
        inp = self._make_inp()
        mask = hole_mask(inp)
        # Above zssurf (z >= 6): should be False in the non-reversed mask
        assert not mask[7, 0, 0]
        # Inside rectangular hole (z in [4,6), y in [3,8), x in [2,6))
        assert not mask[5, 4, 3]

    def test_outside_hole_is_true(self):
        inp = self._make_inp()
        mask = hole_mask(inp)
        # Below the hole region
        assert mask[0, 0, 0]

    def test_reverse(self):
        inp = self._make_inp()
        mask_normal = hole_mask(inp, reverse=False)
        mask_reversed = hole_mask(inp, reverse=True)
        # Reversed mask should be the inverse
        np.testing.assert_array_equal(mask_normal, ~mask_reversed)
