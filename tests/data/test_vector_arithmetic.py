"""Vector operations must preserve grid alignment and physical component axes."""

import operator

import numpy as np
import pytest

from emout.core.data import Data2d, VectorData


def _vector(axes="xz", values=(2.0, 5.0), **kwargs):
    return VectorData(
        [Data2d(np.full((3, 4), value), name=f"e{axis}") for axis, value in zip(axes, values)],
        name=f"e{axes}",
        **kwargs,
    )


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, operator.truediv])
def test_vector_operations_align_component_axes(op):
    left = _vector("xz", (2.0, 5.0))
    right = _vector("zx", (10.0, 4.0))
    result = op(left, right)
    assert result.component_axes == ("x", "z")
    np.testing.assert_allclose(np.asarray(result.objs[0]), op(2.0, 4.0))
    np.testing.assert_allclose(np.asarray(result.objs[1]), op(5.0, 10.0))
    assert result.name == left.name


def test_vectors_with_different_component_axes_cannot_be_combined():
    with pytest.raises(ValueError, match="component axes"):
        _vector("xz") + _vector("xy")


@pytest.mark.parametrize("axes", [("x",), ("x", "x"), ("x", "q")])
def test_invalid_component_axes_are_rejected(axes):
    with pytest.raises(ValueError, match="component_axes"):
        _vector(component_axes=axes)


def test_vector_components_must_have_matching_shapes():
    with pytest.raises(ValueError, match="shape"):
        VectorData([Data2d(np.zeros((3, 4))), Data2d(np.zeros((1, 4)))])


def test_vector_components_must_have_matching_coordinates():
    first = Data2d(np.zeros((3, 4)))
    second = Data2d(np.zeros((3, 4)), xslice=slice(1, 5, 1))
    with pytest.raises(ValueError, match="grid"):
        VectorData([first, second])


@pytest.mark.parametrize(
    "operation", [lambda v: +v, lambda v: -v, lambda v: 10 + v, lambda v: 10 - v, lambda v: v[:, 1:]]
)
def test_vector_results_own_their_metadata(operation):
    source = _vector()
    result = operation(source)
    assert isinstance(result, VectorData)
    assert result.component_axes == source.component_axes
    assert result.attrs is not source.attrs
    result.attrs["name"] = "result"
    assert source.name == "exz"


def test_constructor_does_not_mutate_caller_attributes():
    attrs = {"name": "caller"}
    vector = _vector(attrs=attrs)
    assert attrs == {"name": "caller"}
    assert vector.attrs is not attrs


def test_filter_preserves_axes_of_selected_components():
    source = _vector("xyz", (1.0, 2.0, 3.0))
    result = source.filter(lambda comp: comp.name != "ey")
    assert result.component_axes == ("x", "z")
    np.testing.assert_array_equal(np.asarray(result.objs[0]), np.asarray(source.objs[0]))
    np.testing.assert_array_equal(np.asarray(result.objs[1]), np.asarray(source.objs[2]))
