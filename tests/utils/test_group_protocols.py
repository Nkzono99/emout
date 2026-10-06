"""Copying and serialization must not invoke delegated object protocols."""

import copy
import pickle

import pytest

from emout.utils import Group


@pytest.mark.parametrize("values", [[], [1, 2]])
@pytest.mark.parametrize("operation", ["copy", "deepcopy", "pickle"])
def test_group_copy_and_serialization(values, operation):
    group = Group(values, attrs={"label": "original"})
    if operation == "pickle":
        result = pickle.loads(pickle.dumps(group))
    else:
        result = getattr(copy, operation)(group)
    assert type(result) is Group
    assert result is not group
    assert result.objs == values
    assert result.attrs == group.attrs
