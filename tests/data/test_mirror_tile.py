"""Tests for Data.mirror() and Data.tile()."""

import numpy as np
import pytest

from emout.core.data import Data2d, Data3d


@pytest.fixture
def data2d():
    arr = np.arange(20).reshape(5, 4).astype(float)
    return Data2d(arr, name="test")


@pytest.fixture
def data3d():
    arr = np.arange(60).reshape(3, 4, 5).astype(float)
    return Data3d(arr, name="test3d")


class TestMirror:
    def test_mirror_axis0_shape(self, data2d):
        m = data2d.mirror(0)
        # 5 -> 5 + 4 = 9 (boundary not duplicated)
        assert m.shape == (9, 4)

    def test_mirror_axis1_shape(self, data2d):
        m = data2d.mirror(1)
        assert m.shape == (5, 7)

    def test_mirror_values_reflect(self, data2d):
        m = data2d.mirror(0)
        arr = np.asarray(m)
        # First half is original
        np.testing.assert_array_equal(arr[:5], np.asarray(data2d))
        # Second half is reverse of original (excluding boundary)
        np.testing.assert_array_equal(arr[5:], np.asarray(data2d)[-2::-1])

    def test_mirror_by_name(self, data2d):
        m = data2d.mirror("y")
        assert m.shape[0] == 9

    def test_mirror_preserves_type(self, data2d):
        assert type(data2d.mirror(0)) is Data2d

    def test_mirror_preserves_name(self, data2d):
        assert data2d.mirror(0).name == "test"

    def test_mirror_3d(self, data3d):
        m = data3d.mirror(0)
        assert m.shape == (5, 4, 5)
        assert type(m) is Data3d

    def test_mirror_coordinates_extend(self, data2d):
        m = data2d.mirror(0)
        # Original y: [0, 1, 2, 3, 4], mirrored: [0, 1, ..., 8]
        np.testing.assert_array_equal(m.y, np.arange(9))


class TestTile:
    def test_tile_default_drops_edge(self, data2d):
        """Default: [d0,...,d4, d1,...,d4] -> length 5 + 4 = 9."""
        t = data2d.tile(1, 0)
        assert t.shape == (9, 4)

    def test_tile_values_default(self, data2d):
        t = data2d.tile(1, 0)
        arr = np.asarray(t)
        orig = np.asarray(data2d)
        # First part: original
        np.testing.assert_array_equal(arr[:5], orig)
        # Second part: original without first row
        np.testing.assert_array_equal(arr[5:], orig[1:])

    def test_tile_include_edge(self, data2d):
        """include_edge=True: [d0,...,d4, d0,...,d4] -> length 10."""
        t = data2d.tile(1, 0, include_edge=True)
        assert t.shape == (10, 4)
        arr = np.asarray(t)
        orig = np.asarray(data2d)
        np.testing.assert_array_equal(arr[:5], orig)
        np.testing.assert_array_equal(arr[5:], orig)

    def test_tile_triples(self, data2d):
        t = data2d.tile(2, 0)
        # 5 + 4 + 4 = 13
        assert t.shape == (13, 4)

    def test_tile_by_name(self, data2d):
        t = data2d.tile(1, "x")
        # 4 + 3 = 7
        assert t.shape == (5, 7)

    def test_tile_preserves_type(self, data2d):
        assert type(data2d.tile(1, 0)) is Data2d


class TestChaining:
    def test_mirror_then_tile(self, data2d):
        """Reflect + periodic = ~4x domain."""
        d = data2d.mirror(0).tile(1, 0)
        # mirror: 5 -> 9, tile(1, drop_edge): 9 + 8 = 17
        assert d.shape == (17, 4)
        assert type(d) is Data2d

    def test_mirror_both_axes(self, data2d):
        d = data2d.mirror(0).mirror(1)
        assert d.shape == (9, 7)
