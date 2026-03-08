import matplotlib
import numpy as np
import pytest

from emout.plot.contour3d import (
    _format_level_value,
    _resolve_shared_exponent,
    contour3d,
)

matplotlib.use("Agg")


def test_format_level_value():
    assert _format_level_value(1.2345, fmt="%1.2f") == "1.23"
    assert _format_level_value(1.2345, fmt="{:.3g}") == "1.23"
    assert _format_level_value(1.2345, sigfigs=2) == "1.2"


def test_format_level_value_invalid_sigfigs():
    with pytest.raises(ValueError):
        _format_level_value(1.0, sigfigs=0)


def test_resolve_shared_exponent_auto():
    assert _resolve_shared_exponent([1.2e-20, -2.0e-20], "auto") == -20
    assert _resolve_shared_exponent([0.0, 0.0], "auto") == 0
    assert _resolve_shared_exponent([1.0], None) == 0
    assert _resolve_shared_exponent([1.0], -3) == -3


def test_contour3d_clabel_shared_exponent():
    z, y, x = np.mgrid[0:8, 0:8, 0:8]
    vol = (x + y + z).astype(np.float64) * 1e-20

    fig, ax = contour3d(
        vol,
        dx=1.0,
        levels=[1.23e-20],
        clabel=True,
        clabel_sigfigs=3,
        clabel_shared_exponent="auto",
        show=False,
    )

    texts = [txt.get_text() for txt in ax.texts]
    assert any(t == "1.23" for t in texts)
    assert any(r"\times 10^{-20}" in t for t in texts)

    matplotlib.pyplot.close(fig)
