"""Tests for the Data2d.cmap() / Data2d.contour() shortcut methods.

These methods are thin wrappers around Data2d.plot(mode=...) that let
callers write ``data.contour()`` / ``data.cmap()`` directly. The tests
monkeypatch the underlying basic_plot helpers and check the dispatch.
"""

import numpy as np
import pytest

import emout.plot.basic_plot as basic_plot
from emout.core.data.data import Data2d


def _make_data2d():
    return Data2d(np.arange(20, dtype=float).reshape(4, 5), name="phi", filename="dummy.h5")


def test_cmap_delegates_to_plot_2dmap(monkeypatch):
    """data.cmap() は plot_2dmap を呼び、plot_2d_contour は呼ばない。"""
    calls = {"map": 0, "cont": 0}

    def _fake_2dmap(z, **kwargs):
        calls["map"] += 1
        return "map-image"

    def _fake_contour(z, **kwargs):
        calls["cont"] += 1
        return "contour-image"

    monkeypatch.setattr(basic_plot, "plot_2dmap", _fake_2dmap)
    monkeypatch.setattr(basic_plot, "plot_2d_contour", _fake_contour)

    data = _make_data2d()
    result = data.cmap()

    assert result == "map-image"
    assert calls == {"map": 1, "cont": 0}


def test_contour_delegates_to_plot_2d_contour(monkeypatch):
    """data.contour() は plot_2d_contour を呼び、plot_2dmap は呼ばない。"""
    calls = {"map": 0, "cont": 0}

    def _fake_2dmap(z, **kwargs):
        calls["map"] += 1
        return "map-image"

    def _fake_contour(z, **kwargs):
        calls["cont"] += 1
        return "contour-image"

    monkeypatch.setattr(basic_plot, "plot_2dmap", _fake_2dmap)
    monkeypatch.setattr(basic_plot, "plot_2d_contour", _fake_contour)

    data = _make_data2d()
    result = data.contour()

    assert result == "contour-image"
    assert calls == {"map": 0, "cont": 1}


def test_cmap_forwards_kwargs(monkeypatch):
    """data.cmap(vmin=..., vmax=..., cmap=...) がそのまま基本関数へ渡る。"""
    received = {}

    def _fake_2dmap(z, **kwargs):
        received.update(kwargs)
        return None

    monkeypatch.setattr(basic_plot, "plot_2dmap", _fake_2dmap)

    data = _make_data2d()
    data.cmap(vmin=-1.0, vmax=1.0, cmap="viridis")

    assert received["vmin"] == -1.0
    assert received["vmax"] == 1.0
    assert received["cmap"] == "viridis"


def test_cmap_rejects_mode_argument():
    """cmap() に mode= を渡すと TypeError になる。"""
    data = _make_data2d()
    with pytest.raises(TypeError, match="mode"):
        data.cmap(mode="cm")


def test_contour_rejects_mode_argument():
    """contour() に mode= を渡すと TypeError になる。"""
    data = _make_data2d()
    with pytest.raises(TypeError, match="mode"):
        data.contour(mode="cont")


def test_plot_mode_still_works(monkeypatch):
    """既存の Data2d.plot(mode='cm+cont') は互換のため引き続き動く。"""
    calls = {"map": 0, "cont": 0}

    def _fake_2dmap(z, **kwargs):
        calls["map"] += 1
        return "map-image"

    def _fake_contour(z, **kwargs):
        calls["cont"] += 1
        return "contour-image"

    monkeypatch.setattr(basic_plot, "plot_2dmap", _fake_2dmap)
    monkeypatch.setattr(basic_plot, "plot_2d_contour", _fake_contour)

    data = _make_data2d()
    result = data.plot(mode="cm+cont")

    assert result == ["map-image", "contour-image"]
    assert calls == {"map": 1, "cont": 1}
