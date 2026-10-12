"""Factories for rebuilding dimensioned Data objects."""

from __future__ import annotations

from typing import Any

import numpy as np


_UNSET = object()


def _mean_with_metadata(data, axis, *, dtype=None):
    """Reduce selected dimensions while retaining the surviving grid axes.

    Derived arrays render locally: their source selectors describe the
    unreduced field and cannot be used to fetch or record the mean itself.
    Materializing the source already records the range required for replay.
    """
    mean = np.asarray(data).mean(axis=axis, dtype=dtype)
    dims = (axis,) if isinstance(axis, int) else axis
    return data_from_array(
        mean,
        filename=data.filename,
        name=data.name,
        tslice=data.tslice,
        zslice=data.zslice,
        yslice=data.yslice,
        xslice=data.xslice,
        slice_axes=[source_axis for dim, source_axis in enumerate(data.slice_axes) if dim not in dims],
        axisunits=data.axisunits,
        valunit=data.valunit,
        local_data_policy=getattr(data, "_local_data_policy", None),
        emout_dir=None,
        emout_open_kwargs=None,
        emout_inp=getattr(data, "_emout_inp", None),
        emout_unit=getattr(data, "_emout_unit", None),
        article_recorder=None,
        article_source_shape=getattr(data, "_article_source_shape", None),
    )


def data_from_array(
    array,
    *,
    filename=None,
    name=None,
    tslice=None,
    zslice=None,
    yslice=None,
    xslice=None,
    slice_axes=None,
    axisunits=None,
    valunit=None,
    local_data_policy=None,
    emout_dir: Any = _UNSET,
    emout_open_kwargs: Any = _UNSET,
    emout_inp: Any = _UNSET,
    emout_unit: Any = _UNSET,
    article_recorder: Any = _UNSET,
    article_source_shape: Any = _UNSET,
):
    """Build a Data object with metadata from an array."""
    array = np.asarray(array)
    if array.ndim == 0:
        return array.item()

    params = {
        "filename": filename,
        "name": name,
        "tslice": tslice,
        "zslice": zslice,
        "yslice": yslice,
        "xslice": xslice,
        "slice_axes": slice_axes,
        "axisunits": axisunits,
        "valunit": valunit,
        "local_data_policy": local_data_policy,
    }

    from .data import Data1d, Data2d, Data3d, Data4d

    if array.ndim == 1:
        data = Data1d(array, **params)
    elif array.ndim == 2:
        data = Data2d(array, **params)
    elif array.ndim == 3:
        data = Data3d(array, **params)
    elif array.ndim == 4:
        data = Data4d(array, **params)
    else:
        return array

    if emout_dir is not _UNSET:
        data._emout_dir = emout_dir
    if emout_open_kwargs is not _UNSET:
        data._emout_open_kwargs = emout_open_kwargs
    if emout_inp is not _UNSET:
        data._emout_inp = emout_inp
    if emout_unit is not _UNSET:
        data._emout_unit = emout_unit
    if article_recorder is not _UNSET:
        data._article_recorder = article_recorder
    if article_source_shape is not _UNSET:
        data._article_source_shape = article_source_shape
    return data


def data_from_payload(payload: dict[str, Any], *, local_data_policy: str | None = "allow"):
    """Rebuild a Data object from a remote ``fetch_field`` payload."""
    return data_from_array(
        payload["array"],
        name=payload["name"],
        tslice=payload["slices"][0],
        zslice=payload["slices"][1],
        yslice=payload["slices"][2],
        xslice=payload["slices"][3],
        slice_axes=payload["slice_axes"],
        axisunits=payload["axisunits"],
        valunit=payload["valunit"],
        local_data_policy=local_data_policy,
        emout_dir=None,
        emout_open_kwargs=None,
    )
