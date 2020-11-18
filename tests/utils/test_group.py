import pytest
from emout.utils import Group
import numpy as np
def test_getitem():
    group = Group([
        [1, 2], 
        [3, 4, 5]
    ])
    idx_group = Group([1, 2])
    
    assert group[0].objs == [1, 3]
    assert group[1].objs == [2, 4]
    assert group[1:].objs == [[2], [4, 5]]
    assert group[idx_group].objs == [2, 5]
    assert group[:idx_group].objs == [[1], [3, 4]]


@pytest.mark.parametrize('key, value, expected',[
    (0, -1, [[-1, 2], [-1, 4, 5]]),
    (Group([1, 2]), -1, [[1, -1], [3, 4, -1]]),
])
def test_setitem(key, value, expected):
    group = Group([
        [1, 2],
        [3, 4, 5]
    ])

    group[key] = value
    assert group.objs == expected
