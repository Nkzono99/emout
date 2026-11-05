"""``Data1d.plot`` / ``Data2d.plot`` draw into the Axes passed as ``ax``."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from emout.core.data.data import Data1d, Data2d


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def field():
    return Data2d(np.arange(12, dtype=float).reshape(3, 4), filename="dummy.h5", name="phisp")


@pytest.fixture
def profile():
    return Data1d(np.arange(5, dtype=float), filename="dummy.h5", name="phisp")


@pytest.fixture
def panels():
    """Two panels with the second one current, as ``plt.subplots`` leaves them."""
    _, axes = plt.subplots(1, 2)
    assert plt.gca() is axes[1]
    return axes


# ---------------------------------------------------------------------------
# Data2d
# ---------------------------------------------------------------------------


def test_colormap_is_drawn_into_the_given_axes(field, panels):
    image = field.plot(ax=panels[0])

    assert image.axes is panels[0]
    assert panels[0].get_title() == "phisp"
    assert (panels[0].get_xlabel(), panels[0].get_ylabel()) == ("x", "y")
    assert not panels[1].has_data()
    assert panels[1].get_title() == ""


def test_colorbar_is_attached_to_the_given_axes(field, panels):
    image = field.plot(ax=panels[0])

    colorbar_axes = image.colorbar.ax
    assert colorbar_axes.figure is panels[0].figure
    # The colorbar takes its room from the target panel, not from the current one.
    assert colorbar_axes.get_position().x0 < panels[1].get_position().x0


def test_current_axes_is_left_as_it_was(field, panels):
    field.plot(ax=panels[0])

    assert plt.gca() is panels[1]


def test_contour_is_drawn_into_the_given_axes(field, panels):
    contour = field.plot(mode="cont", ax=panels[0])

    assert contour.axes is panels[0]
    assert panels[0].get_title() == "phisp"
    assert not panels[1].has_data()


def test_colormap_and_contour_share_the_given_axes(field, panels):
    image, contour = field.plot(mode="cm+cont", ax=panels[0])

    assert image.axes is panels[0]
    assert contour.axes is panels[0]
    assert not panels[1].has_data()


def test_axes_of_another_figure_is_used_with_its_own_figure(field):
    target_figure, target = plt.subplots()
    current_figure, current = plt.subplots()

    image = field.plot(ax=target)

    assert image.axes is target
    assert image.colorbar.ax.figure is target_figure
    assert current_figure.axes == [current]
    assert not current.has_data()
    assert plt.gcf() is current_figure


def test_surface_is_drawn_into_the_given_3d_axes(field):
    figure = plt.figure()
    target = figure.add_subplot(1, 2, 1, projection="3d")
    other = figure.add_subplot(1, 2, 2, projection="3d")

    surface = field.plot(mode="surf", ax=target)

    assert surface.axes is target
    assert not other.has_data()


def test_saving_with_axes_saves_the_figure_that_holds_it(field, tmp_path):
    target_figure, target = plt.subplots()
    plt.figure()
    output = tmp_path / "field.png"

    assert field.plot(ax=target, savefilename=str(output)) is None

    assert output.stat().st_size > 0
    assert not plt.fignum_exists(target_figure.number)


def test_without_axes_the_current_axes_is_used(field, panels):
    image = field.plot()

    assert image.axes is panels[1]
    assert not panels[0].has_data()


# ---------------------------------------------------------------------------
# Data1d
# ---------------------------------------------------------------------------


def test_line_is_drawn_into_the_given_axes(profile, panels):
    (line,) = profile.plot(ax=panels[0], vmin=-1, vmax=6)

    assert line.axes is panels[0]
    assert (panels[0].get_xlabel(), panels[0].get_ylabel()) == ("x", "phisp")
    assert panels[0].get_ylim() == (-1, 6)
    assert not panels[1].has_data()
    assert plt.gca() is panels[1]


def test_saving_a_line_with_axes_saves_the_figure_that_holds_it(profile, tmp_path):
    target_figure, target = plt.subplots()
    figures = set(plt.get_fignums())
    output = tmp_path / "profile.png"

    assert profile.plot(ax=target, savefilename=str(output)) is None

    assert output.stat().st_size > 0
    # No extra figure was opened for the save; the target's figure was saved and closed.
    assert set(plt.get_fignums()) == figures - {target_figure.number}


def test_line_without_axes_uses_the_current_axes(profile, panels):
    (line,) = profile.plot()

    assert line.axes is panels[1]
    assert not panels[0].has_data()
