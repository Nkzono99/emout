"""Restart extraction and spatial means must retain their grid meaning."""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

from emout.core.data import Data1d, Data2d, Data3d, Data4d, GridDataSeries
from emout.core.data.griddata_series import GridDataMean
from emout.utils import UnitTranslator


def _write_series(path, frames):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("phi")
        for index, frame in enumerate(frames):
            group.create_dataset(str(index), data=frame)


@pytest.mark.parametrize("segments", [2, 3])
@pytest.mark.parametrize("spatial", [(0, 0, 0), (slice(None, None, 2), slice(1, 3), slice(None))])
def test_restart_time_series_matches_indexing_and_iteration(tmp_path, segments, spatial):
    grid = np.arange(24.0).reshape(2, 3, 4)
    frame_values = ([0.0, 1.0, 10.0], [10.0, 3.0, 4.0], [4.0, 8.0])
    series = []
    for index, values in enumerate(frame_values[:segments]):
        path = tmp_path / f"segment{index}.h5"
        _write_series(path, [grid + value for value in values])
        series.append(GridDataSeries(path, "phi"))
    combined = series[0] + series[1]
    if segments == 3:
        combined = combined + series[2]
    x, y, z = spatial
    try:
        expected = np.asarray(combined[:, z, y, x])
        actual = combined.time_series(x, y, z, skip_overlap=True)
        iterated = np.array([np.asarray(frame)[z, y, x] for frame in combined])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, iterated)
        assert len(actual) == len(combined)
        assert actual.mean() == expected.mean()
        raw = np.concatenate([segment.time_series(x, y, z) for segment in series])
        np.testing.assert_array_equal(combined.time_series(x, y, z), raw)
        assert len(raw) == len(combined) + segments - 1
    finally:
        combined.close()


def test_single_frame_restart_segment_is_only_a_join(tmp_path):
    grid = np.ones((2, 3, 4))
    paths = [tmp_path / "first.h5", tmp_path / "join.h5"]
    _write_series(paths[0], [grid, grid * 2])
    _write_series(paths[1], [grid * 2])
    with GridDataSeries(paths[0], "phi") + GridDataSeries(paths[1], "phi") as combined:
        np.testing.assert_array_equal(combined.time_series(0, 0, 0, skip_overlap=True), [1, 2])


@pytest.fixture
def coordinate_series(tmp_path):
    values = np.arange(3 * 8 * 10 * 20, dtype=np.float32).reshape(3, 8, 10, 20)
    path = tmp_path / "phi.h5"
    _write_series(path, values)
    length = UnitTranslator(2, 1, unit="m")
    with GridDataSeries(path, "phi", axisunit=length, valunit=UnitTranslator(3, 1, unit="V")) as series:
        yield series, values


def test_single_series_accepts_overlap_keyword(coordinate_series):
    series, values = coordinate_series
    np.testing.assert_array_equal(series.time_series(0, 0, 0, skip_overlap=True), values[:, 0, 0, 0])


@pytest.mark.parametrize("axis,dims", [("x", (3,)), ("y", (2,)), ("z", (1,)), (-1, (3,))])
@pytest.mark.parametrize("use_si", [False, True])
def test_spatial_mean_keeps_axes_and_plot_extent(coordinate_series, axis, dims, use_si):
    series, values = coordinate_series
    item = (slice(None), slice(2, 8, 2), slice(1, 9, 2), slice(4, 16, 3))
    selection = series[item]
    mean = selection.mean(axis=axis, keep_metadata=True)
    assert type(mean) is Data3d
    assert mean.slice_axes == [dim for dim in range(4) if dim not in dims]
    assert mean.axisunits == selection.axisunits
    assert mean.valunit is series.valunit
    np.testing.assert_allclose(mean, values[item].mean(axis=dims))
    frame = mean[0]
    assert type(frame) is Data2d
    assert frame.slice_axes == mean.slice_axes[1:]
    image = frame.plot(use_si=use_si)
    horizontal, vertical = sorted(frame.use_axes)
    suffix = " [m]" if use_si else ""
    assert image.axes.get_xlabel() == horizontal + suffix
    assert image.axes.get_ylabel() == vertical + suffix
    h = getattr(frame, horizontal + ("_si" if use_si else ""))
    v = getattr(frame, vertical + ("_si" if use_si else ""))
    np.testing.assert_allclose(image.get_extent(), [h[0], h[-1], v[0], v[-1]])
    plt.close(image.figure)


@pytest.mark.parametrize(
    "axis,dims,cls,remaining",
    [
        (("x", "y"), (2, 3), Data2d, [0, 1]),
        (("t", "x"), (0, 3), Data2d, [1, 2]),
        (("z", "y", "x"), (1, 2, 3), Data1d, [0]),
        ((), (), Data4d, [0, 1, 2, 3]),
    ],
)
def test_multi_axis_mean_rebuilds_dimension_class(coordinate_series, axis, dims, cls, remaining):
    series, values = coordinate_series
    mean = series[:].mean(axis=axis, dtype=np.float64, keep_metadata=True)
    assert type(mean) is cls
    assert mean.slice_axes == remaining
    assert mean.dtype == np.float64
    np.testing.assert_allclose(mean, values.mean(axis=dims, dtype=np.float64))
    for dim, source_axis in enumerate(remaining):
        np.testing.assert_array_equal(mean.axis(dim), np.arange(values.shape[source_axis]))
    np.testing.assert_allclose(mean[-1], values.mean(axis=dims, dtype=np.float64)[-1])


def test_all_axes_mean_returns_scalar(coordinate_series):
    series, values = coordinate_series
    mean = series[:].mean(axis=("t", "z", "y", "x"), keep_metadata=True)
    assert np.isscalar(mean)
    assert mean == values.mean()


def test_mean_keeps_legacy_default_and_lazy_time_average(coordinate_series):
    series, values = coordinate_series
    legacy = series[:].mean(axis="x")
    assert type(legacy) is Data4d
    assert legacy.slice_axes == [0, 1, 2, 3]
    np.testing.assert_allclose(legacy, values.mean(axis=3))
    time_mean = series[:].mean(keep_metadata=True)
    assert isinstance(time_mean, GridDataMean)
    assert time_mean.slice_axes == [1, 2, 3]
    np.testing.assert_allclose(time_mean, values.mean(axis=0))


def test_derived_mean_does_not_refetch_unreduced_source(coordinate_series):
    series, _ = coordinate_series
    series._emout_dir = series.directory
    series._emout_open_kwargs = {"directory": str(series.directory)}
    series._emout_inp = object()
    series._emout_unit = object()
    mean = series[:].mean(axis="x", keep_metadata=True)
    assert mean.inp is series._emout_inp
    assert mean.unit is series._emout_unit
    assert mean._get_remote_open_kwargs() is None
    assert mean[0]._get_remote_open_kwargs() is None
