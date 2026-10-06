"""Named samples and vector fields must retain their meaning through operations."""

import copy
import operator
import pickle

import h5py
import numpy as np
import pytest

import emout
from emout.core.data import ComponentValues, Data2d, GridDataSelection, GridDataSeries, VectorData
from emout.utils import Group, UnitTranslator


def vector(axes="xz"):
    fields = {"x": np.arange(12.0).reshape(3, 4) + 1, "z": np.arange(12.0).reshape(3, 4) + 20}
    return VectorData([Data2d(fields[axis], name=f"e{axis}") for axis in axes], name=f"e{axes}")


def test_existing_public_types_and_component_order():
    field = vector("zx")
    assert isinstance(field, Group)
    assert emout.VectorData2d is emout.VectorData3d is VectorData
    assert emout.ComponentValues is ComponentValues
    assert field.x_data is field.objs[0] is field.components["z"]
    assert field.y_data is field.components["x"]
    with pytest.raises(TypeError):
        field.components["x"] = 0


@pytest.mark.parametrize("op", [operator.pos, operator.neg, operator.abs, np.sqrt, np.sin, np.isfinite])
def test_unary_operations_return_grid_fields(op):
    field = vector()
    result = op(field)
    assert isinstance(result, VectorData)
    for axis in field.component_axes:
        np.testing.assert_allclose(result.components[axis].to_numpy(), op(field.components[axis].to_numpy()))
        np.testing.assert_array_equal(result.components[axis].x, field.components[axis].x)


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
    ],
)
@pytest.mark.parametrize("reflected", [False, True])
def test_array_operands_follow_numpy_broadcasting(op, reflected):
    field = vector()
    operand = np.array([1.0, 2.0, 3.0, 4.0])
    result = op(operand, field) if reflected else op(field, operand)
    assert isinstance(result, VectorData)
    for axis in field.component_axes:
        values = field.components[axis].to_numpy()
        expected = op(operand, values) if reflected else op(values, operand)
        np.testing.assert_allclose(result.components[axis].to_numpy(), expected)


@pytest.mark.parametrize(
    "operation, reference",
    [
        (lambda v: v[1, 2], lambda a: a[1, 2]),
        (lambda v: v.mean(), np.mean),
        (lambda v: v.sum(), np.sum),
        (lambda v: v.max(), np.max),
        (lambda v: v.mean(axis=0), lambda a: a.mean(axis=0)),
    ],
)
def test_samples_and_reductions_are_named_values(operation, reference):
    field = vector("zx")
    result = operation(field)
    assert type(result) is ComponentValues
    assert result.component_axes == ("z", "x")
    assert result.name == field.name
    expected = [reference(obj.to_numpy()) for obj in field.objs]
    np.testing.assert_allclose(result.to_numpy(), expected)
    assert result.attrs is not field.attrs
    assert result.x_data is result.objs[0]


def test_delegated_methods_that_return_fields_keep_plotting_api():
    field = vector()
    result = field.copy()
    assert isinstance(result, VectorData)
    np.testing.assert_array_equal(result.to_numpy(), field.to_numpy())
    assert not np.shares_memory(result.components["x"], field.components["x"])


def test_plain_array_attributes_do_not_claim_grid_metadata():
    field = vector()
    result = field.map(lambda obj: obj.to_numpy())
    assert type(result) is ComponentValues
    np.testing.assert_array_equal(result.to_numpy(), field.to_numpy())


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, np.add])
def test_named_samples_are_aligned_with_field_axes(op):
    field = vector()
    sample = ComponentValues([5.0, 2.0], component_axes=("z", "x"))
    result = op(field, sample)
    assert isinstance(result, VectorData)
    for axis, value in [("x", 2.0), ("z", 5.0)]:
        np.testing.assert_allclose(result.components[axis].to_numpy(), op(field.components[axis].to_numpy(), value))


def test_named_values_arithmetic_aligns_physical_axes():
    left = ComponentValues([2, 5], component_axes=("x", "z"))
    right = ComponentValues([10, 4], component_axes=("z", "x"))
    assert dict((left - right).components) == {"x": -2, "z": -5}


def test_named_assignment_aligns_physical_axes():
    field = vector()
    field[:, :] = ComponentValues([5.0, 2.0], component_axes=("z", "x"))
    np.testing.assert_array_equal(field.components["x"].to_numpy(), 2.0)
    np.testing.assert_array_equal(field.components["z"].to_numpy(), 5.0)


def test_multi_output_ufunc_preserves_axes():
    field = vector() / 3.0
    fractions, integers = np.modf(field)
    quotient, remainder = divmod(field, 2.0)
    for axis in field.component_axes:
        expected_fraction, expected_integer = np.modf(field.components[axis].to_numpy())
        np.testing.assert_allclose(fractions.components[axis].to_numpy(), expected_fraction)
        np.testing.assert_allclose(integers.components[axis].to_numpy(), expected_integer)
        np.testing.assert_allclose(quotient.components[axis].to_numpy(), field.components[axis].to_numpy() // 2.0)
        np.testing.assert_allclose(remainder.components[axis].to_numpy(), field.components[axis].to_numpy() % 2.0)


def test_ufunc_out_and_where_align_axes():
    field = vector()
    output = vector("zx") * 0
    mask = ComponentValues([np.ones((3, 4), bool), np.zeros((3, 4), bool)], component_axes=("z", "x"))
    assert np.add(field, 3, out=output, where=mask) is output
    np.testing.assert_array_equal(output.components["x"].to_numpy(), 0)
    np.testing.assert_array_equal(output.components["z"].to_numpy(), field.components["z"].to_numpy() + 3)


def test_augmented_assignment_preserves_existing_copy_behavior():
    field = vector()
    alias = field
    before = field.to_numpy().copy()
    field += 3
    assert field is not alias
    np.testing.assert_array_equal(alias.to_numpy(), before)
    np.testing.assert_array_equal(field.to_numpy(), before + 3)


def test_component_alias_and_storage_stay_consistent():
    field = vector()
    replacement = Data2d(np.zeros((3, 4)))
    field.x_data = replacement
    assert field.objs[0] is field.components["x"] is replacement
    field.objs[0] = field.objs[1]
    assert field.x_data is field.objs[0]


@pytest.mark.parametrize("keep", [0, 1, 2])
def test_filter_keeps_names_for_empty_and_single_component_results(keep):
    field = vector()
    result = field.filter(lambda obj: obj.name[-1] in "xz"[:keep])
    assert result.component_axes == tuple("xz"[:keep])
    assert isinstance(result, VectorData) == (keep == 2)


@pytest.mark.parametrize("operation", [np.negative, lambda v: v + 1, lambda v: np.add(v, 1, out=v)])
def test_local_data_policy_applies_to_vector_arithmetic(operation):
    field = vector()
    with emout.local_data_policy("remote_required"):
        with pytest.raises(emout.LocalDataAccessDisabledError):
            operation(field)


@pytest.mark.parametrize("operation", [copy.copy, copy.deepcopy, lambda v: pickle.loads(pickle.dumps(v))])
def test_named_values_copy_and_serialization(operation):
    source = ComponentValues([2.0, 5.0], name="sample", component_axes=("z", "x"))
    result = operation(source)
    assert type(result) is ComponentValues
    assert dict(result.components) == dict(source.components)
    assert result.name == "sample"


@pytest.mark.parametrize("protocol", [4, 5])
def test_vector_pickle_preserves_component_metadata(protocol):
    source = vector()[:, ::-1]
    result = pickle.loads(pickle.dumps(source, protocol=protocol))
    assert result.component_axes == source.component_axes
    np.testing.assert_array_equal(result.to_numpy(), source.to_numpy())
    for axis in source.component_axes:
        assert result.components[axis].name == source.components[axis].name
        np.testing.assert_array_equal(result.components[axis].x, source.components[axis].x)


def test_shared_axis_coordinates_and_vector_animation(monkeypatch):
    field = vector()[:, ::-1]
    np.testing.assert_array_equal(field.axis(1), [3, 2, 1, 0])
    titles = []
    monkeypatch.setattr(VectorData, "plot", lambda self, **kwargs: titles.append(kwargs["title"]))
    field.build_frame_updater(axis=1, use_si=False).update(1)
    assert "(2.0)" in titles[0]


def test_scalar_field_operand_cannot_mix_different_coordinates():
    field = vector()
    shifted = Data2d(np.zeros((3, 4)), xslice=slice(1, 5, 1))
    with pytest.raises(ValueError, match="grid coordinates"):
        field + shifted


@pytest.mark.parametrize("selection", [lambda v: v.lazy, lambda v: v[:], lambda v: v[:][:, :, 1:2, 1:]])
def test_vector_lazy_selection_does_not_read_fields(tmp_path, monkeypatch, selection):
    filename = tmp_path / "ex00_0000.h5"
    with h5py.File(filename, "w") as handle:
        group = handle.create_group("ex")
        for index in range(3):
            group.create_dataset(str(index), data=np.ones((2, 3, 4)))
    with GridDataSeries(filename, "ex") as series:
        field = VectorData([series, series], component_axes=("x", "z"))

        def no_reads(*args, **kwargs):
            pytest.fail("Vector metadata or selection eagerly read a grid field.")

        monkeypatch.setattr(series, "_read_selection", no_reads)
        result = selection(field)
        assert isinstance(result, VectorData)
        assert all(isinstance(component, GridDataSelection) for component in result.objs)


@pytest.mark.parametrize("protocol", [4, 5])
def test_pickle_restores_si_conversion_and_source_metadata(tmp_path, protocol):
    length = UnitTranslator(2.0, 1.0, unit="m")
    value = UnitTranslator(4.0, 1.0, unit="V/m")
    field = Data2d(
        np.arange(12.0).reshape(3, 4), filename=tmp_path / "ex.h5", name="ex", axisunits=[length] * 4, valunit=value
    )[:, ::-1]
    result = pickle.loads(pickle.dumps(field, protocol=protocol))
    assert result.filename == field.filename
    assert result.valunit.unit == "V/m"
    np.testing.assert_array_equal(result.val_si, field.val_si)
    np.testing.assert_array_equal(result.x_si, field.x_si)


def test_unnamed_group_operands_keep_positional_matching():
    field = vector()
    result = field + Group([2.0, 5.0])
    assert isinstance(result, VectorData)
    np.testing.assert_array_equal(result.components["x"].to_numpy(), field.components["x"].to_numpy() + 2)
    np.testing.assert_array_equal(result.components["z"].to_numpy(), field.components["z"].to_numpy() + 5)


def test_boolean_samples_are_values_without_grid_assumptions():
    field = vector()
    mask = np.eye(3, 4, dtype=bool)
    result = field[mask]
    assert type(result) is ComponentValues
    for axis in field.component_axes:
        np.testing.assert_array_equal(result.components[axis], field.components[axis].to_numpy()[mask])
