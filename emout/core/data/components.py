"""Named component values and NumPy arithmetic shared by vector containers."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from emout.utils import Group


def _infer_component_axes(objs, name=None) -> tuple[str, ...]:
    axes = tuple(str(getattr(component, "name", ""))[-1:] for component in objs)
    if len(set(axes)) == len(objs) and all(axis in ("x", "y", "z") for axis in axes):
        return axes
    suffix = str(name)[-len(objs) :] if name and objs else ""
    if len(suffix) == len(objs) and len(set(suffix)) == len(objs) and set(suffix) <= set("xyz"):
        return tuple(suffix)
    return tuple("xyz"[: len(objs)])


class ComponentValues(NDArrayOperatorsMixin, Group):
    """Values associated with physical vector axes, without grid assumptions.

    Scalar samples, reductions, and component attributes retain their axis
    names here. ``objs`` and element-wise delegation remain compatible with
    :class:`~emout.utils.group.Group`; ``components`` maps physical axes to values.
    Named operands are aligned by axis rather than their storage order.
    """

    __array_priority__ = 1000

    def __init__(self, objs, name=None, attrs=None, component_axes=None):
        objs = list(objs)
        attrs = dict(attrs) if attrs is not None else {}
        if name:
            attrs["name"] = name
        elif "name" not in attrs:
            attrs["name"] = getattr(objs[0], "name", "") if objs else ""
        axes = component_axes if component_axes is not None else attrs.get("component_axes")
        axes = tuple(axes) if axes is not None else _infer_component_axes(objs, attrs["name"])
        if len(axes) != len(objs) or len(set(axes)) != len(objs) or any(axis not in ("x", "y", "z") for axis in axes):
            raise ValueError("component_axes must contain one unique x, y, or z axis per component.")
        attrs["component_axes"] = axes
        super().__init__(objs, attrs=attrs)

    @property
    def name(self):
        """Return the display name inherited from the source vector."""
        return self.attrs["name"]

    @property
    def component_axes(self) -> tuple[str, ...]:
        """Return physical axes in component storage order."""
        return tuple(self.attrs["component_axes"])

    @property
    def components(self):
        """Return a read-only mapping from physical axes to component values."""
        return MappingProxyType(dict(zip(self.component_axes, self.objs)))

    @property
    def x_data(self):
        """Return the first component (the legacy positional alias)."""
        return self.objs[0]

    @property
    def y_data(self):
        """Return the second component (the legacy positional alias)."""
        if len(self.objs) < 2:
            raise AttributeError("This container has no second component.")
        return self.objs[1]

    @property
    def z_data(self):
        """Return the third component (the legacy positional alias)."""
        if len(self.objs) < 3:
            raise AttributeError("This container has no third component.")
        return self.objs[2]

    def __setattr__(self, key, value):
        if key in ("x_data", "y_data", "z_data"):
            self.objs[("x_data", "y_data", "z_data").index(key)] = value
            return
        super().__setattr__(key, value)

    def _component_for_axis(self, axis):
        if axis not in self.component_axes:
            raise ValueError(f'axes "{axis}" cannot be used because this vector has no {axis!r} component')
        return self.objs[self.component_axes.index(axis)]

    def _new_group(self, objs):
        return ComponentValues(objs, attrs=self.attrs, component_axes=self.component_axes)

    def filter(self, predicate):
        """Filter components while keeping each selected value's physical axis."""
        selected = [(axis, obj) for axis, obj in zip(self.component_axes, self.objs) if predicate(obj)]
        axes, objs = zip(*selected) if selected else ((), ())
        return self._result_with_axes(list(objs), axes)

    def _result_with_axes(self, objs, axes):
        return ComponentValues(objs, attrs=self.attrs, component_axes=axes)

    def _aligned_values(self, operand):
        if isinstance(operand, ComponentValues):
            if set(self.component_axes) != set(operand.component_axes):
                raise ValueError("Vector operands must have the same component axes.")
            return [operand._component_for_axis(axis) for axis in self.component_axes]
        if isinstance(operand, Group):
            if len(operand) != len(self):
                raise ValueError(f"group size mismatch: self has {len(self)} elements, arg has {len(operand)}")
            return operand.objs
        return [operand] * len(self)

    def _validate_operand(self, operand):
        """Hook for containers with additional grid invariants."""
        for component in operand.objs if isinstance(operand, Group) else (operand,):
            require = getattr(component, "_require_local_data_access", None)
            if callable(require):
                require("apply a NumPy ufunc to component data", getattr(component, "name", None))

    def _binary_operator(self, callable, other):
        self._validate_operand(other)
        return self._new_group([callable(obj, rhs) for obj, rhs in zip(self.objs, self._aligned_values(other))])

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Apply a NumPy ufunc independently to aligned physical components."""
        if method != "__call__":
            return NotImplemented
        for operand in inputs:
            self._validate_operand(operand)
        aligned_inputs = [self._aligned_values(operand) for operand in inputs]
        outputs = kwargs.pop("out", None)
        if outputs is not None:
            for output in outputs:
                if output is not None and not isinstance(output, ComponentValues):
                    raise TypeError("Vector ufunc outputs must be named component containers or None.")
                if output is not None:
                    self._validate_operand(output)
            aligned_outputs = [self._aligned_values(output) for output in outputs]
        if isinstance(kwargs.get("where"), Group):
            masks = self._aligned_values(kwargs.pop("where"))
        else:
            masks = None
        results = []
        for index in range(len(self)):
            component_kwargs = dict(kwargs)
            if outputs is not None:
                component_kwargs["out"] = tuple(output[index] for output in aligned_outputs)
            if masks is not None:
                component_kwargs["where"] = masks[index]
            results.append(ufunc(*(operand[index] for operand in aligned_inputs), **component_kwargs))
        if ufunc.nout > 1:
            return tuple(
                outputs[i]
                if outputs is not None and outputs[i] is not None
                else self._new_group([r[i] for r in results])
                for i in range(ufunc.nout)
            )
        return outputs[0] if outputs is not None and outputs[0] is not None else self._new_group(results)

    # Historically augmented assignment returned a new Group. Preserve that
    # behavior; explicit ufunc out= remains available for array mutation.
    __iadd__ = NDArrayOperatorsMixin.__add__
    __isub__ = NDArrayOperatorsMixin.__sub__
    __imul__ = NDArrayOperatorsMixin.__mul__
    __itruediv__ = NDArrayOperatorsMixin.__truediv__
    __ifloordiv__ = NDArrayOperatorsMixin.__floordiv__
    __imod__ = NDArrayOperatorsMixin.__mod__
    __ipow__ = NDArrayOperatorsMixin.__pow__
    __imatmul__ = NDArrayOperatorsMixin.__matmul__
    __ilshift__ = NDArrayOperatorsMixin.__lshift__
    __irshift__ = NDArrayOperatorsMixin.__rshift__
    __iand__ = NDArrayOperatorsMixin.__and__
    __ior__ = NDArrayOperatorsMixin.__or__
    __ixor__ = NDArrayOperatorsMixin.__xor__

    def __call__(self, *args, **kwargs):
        arguments = [self._aligned_values(arg) for arg in args]
        keywords = {key: self._aligned_values(value) for key, value in kwargs.items()}
        return self._new_group(
            [
                obj(*(arg[i] for arg in arguments), **{key: value[i] for key, value in keywords.items()})
                for i, obj in enumerate(self.objs)
            ]
        )

    def __setitem__(self, key, value):
        super().__setitem__(key, Group(self._aligned_values(value)))

    def to_numpy(self, stack_axis=0):
        """Explicitly stack component values in ``component_axes`` order."""
        arrays = [obj.to_numpy() if callable(getattr(obj, "to_numpy", None)) else np.asarray(obj) for obj in self.objs]
        return np.stack(arrays, axis=stack_axis) if arrays else np.array([])

    def __repr__(self):
        return f"<ComponentValues: name={self.name!r}, components={dict(self.components)!r}>"


class _ComponentMethods(ComponentValues):
    """Internal delegated callables whose results use the source's factory."""

    def __init__(self, objs, source):
        super().__init__(objs, attrs=source.attrs, component_axes=source.component_axes)
        object.__setattr__(self, "_source", source)

    def _new_group(self, objs):
        return self._source._new_group(objs)
