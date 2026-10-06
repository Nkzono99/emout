"""Index values and coordinate metadata must agree with NumPy slicing."""

import h5py
import numpy as np
import pytest

from emout.core.data import Data4d, GridDataSeries
from emout.core.data.selectors import normalize_selector, selector_positions
from emout.utils.util import range_with_slice


SLICES = [
    slice(None),
    slice(None, 0),
    slice(None, None, -1),
    slice(None, None, -2),
    slice(1, 99),
    slice(-99, 99),
    slice(-99, -100, -1),
    slice(4, 0, -2),
    slice(1, 5, 2),
]


@pytest.mark.parametrize("slc", SLICES)
@pytest.mark.parametrize("size", [0, 1, 6])
def test_range_with_slice_matches_python_sequence(slc, size):
    assert list(range_with_slice(slc, size)) == list(range(size))[slc]


@pytest.mark.parametrize("slc", SLICES)
@pytest.mark.parametrize("size", [0, 1, 6])
def test_selector_normalization_preserves_positions(slc, size):
    normalized = normalize_selector(slc, size)
    assert selector_positions(normalized, size) == tuple(range(size))[slc]
    assert normalize_selector(normalized, size) == normalized


@pytest.mark.parametrize("slc", SLICES)
def test_materialized_slices_keep_coordinates(slc):
    data = Data4d(np.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6))
    sliced = data[:, :, :, slc]
    np.testing.assert_array_equal(np.asarray(sliced), np.asarray(data)[:, :, :, slc])
    np.testing.assert_array_equal(sliced.x, np.arange(6)[slc])
    np.testing.assert_array_equal(sliced.axis(3), np.arange(6)[slc])


def test_chained_slices_keep_coordinates_and_time():
    data = Data4d(np.arange(7 * 3 * 4 * 6).reshape(7, 3, 4, 6))
    sliced = data[2::2, :, :, ::2][::-1, :, :, ::-1]
    np.testing.assert_array_equal(sliced.t, np.arange(7)[2::2][::-1])
    np.testing.assert_array_equal(sliced.x, np.arange(6)[::2][::-1])
    np.testing.assert_array_equal(sliced.axis(0), sliced.t)


def test_negative_integer_keeps_selected_coordinate():
    data = Data4d(np.zeros((2, 3, 4, 6)))
    sliced = data[-1, :, :, -1]
    np.testing.assert_array_equal(sliced.t, [1])
    np.testing.assert_array_equal(sliced.x, [5])
    assert sliced.slice_axes == [1, 2]


def test_materialized_ellipsis_matches_explicit_index():
    data = Data4d(np.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6))
    sliced = data[..., -1]
    np.testing.assert_array_equal(np.asarray(sliced), np.asarray(data)[..., -1])
    assert sliced.slice_axes == [0, 1, 2]
    np.testing.assert_array_equal(sliced.x, [5])


@pytest.mark.parametrize(
    "item",
    [
        (slice(0, 1), slice(None), slice(None), slice(1, 2)),
        (0, slice(None), slice(None), slice(None, None, -1)),
        (slice(None, None, -1), slice(None), slice(None), slice(None)),
        (0, slice(None), slice(None), slice(-99, -100, -1)),
    ],
)
def test_remote_recipe_reproduces_values_and_dimensions(item):
    data = Data4d(np.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6))
    sliced = data[item]
    restored = np.asarray(data)[sliced._to_recipe_index()]
    np.testing.assert_array_equal(restored, np.asarray(sliced))


@pytest.fixture
def grid_series(tmp_path):
    values = np.arange(3 * 2 * 3 * 6, dtype=np.float32).reshape(3, 2, 3, 6)
    path = tmp_path / "phi00_0000.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("phi")
        for index, array in enumerate(values):
            group.create_dataset(f"{index:04d}", data=array)
    with GridDataSeries(path, "phi") as series:
        yield series, values


@pytest.mark.parametrize("slc", SLICES)
def test_lazy_time_slices_match_numpy(grid_series, slc):
    series, values = grid_series
    selected = series[slc]
    result = selected.materialize()
    assert result.shape == values[slc].shape
    assert result.dtype == values.dtype
    np.testing.assert_array_equal(np.asarray(result), values[slc])
    np.testing.assert_array_equal(result.t, np.arange(len(values))[slc])


@pytest.mark.parametrize("slc", SLICES)
def test_lazy_spatial_slices_match_numpy(grid_series, slc):
    series, values = grid_series
    result = series[0, :, :, slc]
    np.testing.assert_array_equal(np.asarray(result), values[0, :, :, slc])
    np.testing.assert_array_equal(result.x, np.arange(6)[slc])


def test_duplicate_spatial_indexes_remain_valid_when_chained(grid_series):
    series, values = grid_series
    result = series.lazy[:, :, :, [2, 2, 2]][0, :, :, :]
    np.testing.assert_array_equal(np.asarray(result), values[0][:, :, [2, 2, 2]])


@pytest.mark.parametrize(
    "item",
    [
        np.eye(3, 4, dtype=bool),
        [2, 0, 2],
        np.array([True, False, True]),
        (slice(None), [3, 1, 1]),
        (None, Ellipsis, -1),
        (True, Ellipsis),
        (Ellipsis, None),
    ],
)
def test_advanced_indexing_returns_numpy_values(item):
    from emout.core.data import Data2d

    values = np.arange(12).reshape(3, 4)
    field = Data2d(values)
    result = field[item]
    assert type(result) is np.ndarray
    np.testing.assert_array_equal(result, values[item])
