"""Exercise real hole-surface drawing, including units and shared Axes."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from emout.core.data import Data3d
from emout.plot.extra_plot import plot_surface_with_hole, plot_surface_with_hole_half
from emout.utils import Units


@pytest.mark.parametrize("helper", [plot_surface_with_hole, plot_surface_with_hole_half])
@pytest.mark.parametrize("use_si", [None, False, True])
@pytest.mark.parametrize("explicit_axes", [False, True])
def test_consistent_hole_has_one_axes_and_matching_colour_units(helper, use_si, explicit_axes):
    units = Units(0.5, 10000)
    z, y, x = np.indices((9, 9, 9))
    field = Data3d(
        1.0 + x + y + z,
        name="phi",
        axisunits=[units.t, units.length, units.length, units.length],
        valunit=units.phi,
    )
    inp = SimpleNamespace(
        xlrechole=[0, 2],
        xurechole=[0, 6],
        ylrechole=[0, 2],
        yurechole=[0, 6],
        zlrechole=[0, 2],
        zurechole=[6, 0],
    )
    fig = plt.figure()
    try:
        options = {"consistent": True}
        if use_si is not None:
            options["use_si"] = use_si
        if explicit_axes:
            options["ax3d"] = fig.add_subplot(projection="3d")
        if helper is plot_surface_with_hole_half:
            options["off"] = 2
        helper(field, inp, **options)
        fig.canvas.draw()
        axes = [ax for ax in fig.axes if ax.name == "3d"]
        assert len(axes) == 1
        if explicit_axes:
            assert axes[0] is options["ax3d"]
        if helper is plot_surface_with_hole:
            assert len(axes[0].collections) == 9
        expected = field if use_si is False else field.val_si
        np.testing.assert_allclose(fig.axes[-1].get_ylim(), [expected.min(), expected.max()])
        colours = np.concatenate([artist.get_facecolors() for artist in axes[0].collections])
        assert len(np.unique(colours, axis=0)) > 2
    finally:
        plt.close(fig)


def test_consistent_hole_honours_zero_colour_limit_without_units():
    field = Data3d(np.ones((9, 9, 9)))
    inp = SimpleNamespace(
        xlrechole=[0, 2],
        xurechole=[0, 6],
        ylrechole=[0, 2],
        yurechole=[0, 6],
        zlrechole=[0, 2],
        zurechole=[6, 0],
    )
    fig = plt.figure()
    try:
        plot_surface_with_hole(field, inp, consistent=True, vmin=0, vmax=2)
        np.testing.assert_allclose(fig.axes[-1].get_ylim(), [0, 2])
    finally:
        plt.close(fig)
