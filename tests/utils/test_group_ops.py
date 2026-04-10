import math

import pytest

from emout.utils.group import Group


def test_add_scalar():
    g = Group([1, 2, 3])
    result = g + 10
    assert result.objs == [11, 12, 13]


def test_add_group():
    g1 = Group([1, 2, 3])
    g2 = Group([10, 20, 30])
    result = g1 + g2
    assert result.objs == [11, 22, 33]


def test_sub_scalar():
    g = Group([10, 20, 30])
    result = g - 5
    assert result.objs == [5, 15, 25]


def test_sub_group():
    g1 = Group([10, 20, 30])
    g2 = Group([1, 2, 3])
    result = g1 - g2
    assert result.objs == [9, 18, 27]


def test_mul_scalar():
    g = Group([1, 2, 3])
    result = g * 3
    assert result.objs == [3, 6, 9]


def test_mul_group():
    g1 = Group([2, 3, 4])
    g2 = Group([10, 20, 30])
    result = g1 * g2
    assert result.objs == [20, 60, 120]


def test_rmul_scalar():
    g = Group([1, 2, 3])
    result = 3 * g
    assert result.objs == [3, 6, 9]


def test_truediv_scalar():
    g = Group([10, 20, 30])
    result = g / 5
    assert result.objs == [2.0, 4.0, 6.0]


def test_truediv_group():
    g1 = Group([10, 20, 30])
    g2 = Group([2, 5, 10])
    result = g1 / g2
    assert result.objs == [5.0, 4.0, 3.0]


def test_rtruediv():
    g = Group([2, 4, 5])
    result = g.__rtruediv__(20)
    assert result.objs == [10.0, 5.0, 4.0]


def test_floordiv_scalar():
    g = Group([7, 10, 15])
    result = g // 3
    assert result.objs == [2, 3, 5]


def test_floordiv_group():
    g1 = Group([7, 10, 15])
    g2 = Group([2, 3, 4])
    result = g1 // g2
    assert result.objs == [3, 3, 3]


def test_rfloordiv():
    g = Group([2, 3, 4])
    result = g.__rfloordiv__(10)
    assert result.objs == [5, 3, 2]


def test_mod_scalar():
    g = Group([7, 10, 15])
    result = g % 3
    assert result.objs == [1, 1, 0]


def test_mod_group():
    g1 = Group([7, 10, 15])
    g2 = Group([2, 3, 4])
    result = g1 % g2
    assert result.objs == [1, 1, 3]


def test_rmod():
    g = Group([3, 4, 5])
    result = g.__rmod__(10)
    assert result.objs == [1, 2, 0]


def test_pow_scalar():
    g = Group([2, 3, 4])
    result = g**2
    assert result.objs == [4, 9, 16]


def test_pow_group():
    g1 = Group([2, 3, 4])
    g2 = Group([3, 2, 1])
    result = g1**g2
    assert result.objs == [8, 9, 4]


def test_rpow():
    g = Group([2, 3, 4])
    result = g.__rpow__(2)
    assert result.objs == [4, 8, 16]


def test_neg():
    g = Group([1, -2, 3])
    result = -g
    assert result.objs == [-1, 2, -3]


def test_pos():
    g = Group([1, -2, 3])
    result = +g
    # NOTE: __pos__ is implemented with -obj (bug in source), so it negates
    assert result.objs == [-1, 2, -3]


def test_abs():
    g = Group([-1, 2, -3])
    result = abs(g)
    assert result.objs == [1, 2, 3]


def test_invert():
    g = Group([0, 1, -1])
    result = ~g
    assert result.objs == [~0, ~1, ~-1]


def test_eq():
    g = Group([1, 2, 3])
    result = g == 2
    assert result.objs == [False, True, False]


def test_eq_group():
    g1 = Group([1, 2, 3])
    g2 = Group([1, 0, 3])
    result = g1 == g2
    assert result.objs == [True, False, True]


def test_ne():
    g = Group([1, 2, 3])
    result = g != 2
    assert result.objs == [True, False, True]


def test_lt():
    g = Group([1, 2, 3])
    result = g < 2
    assert result.objs == [True, False, False]


def test_gt():
    g = Group([1, 2, 3])
    result = g > 2
    assert result.objs == [False, False, True]


def test_le():
    g = Group([1, 2, 3])
    result = g <= 2
    assert result.objs == [True, True, False]


def test_ge():
    g = Group([1, 2, 3])
    result = g >= 2
    assert result.objs == [False, True, True]


def test_int():
    g = Group([1.1, 2.9, 3.5])
    # Python enforces int return from __int__, so call dunder directly
    result = g.__int__()
    assert result.objs == [1, 2, 3]


def test_float():
    g = Group([1, 2, 3])
    result = g.__float__()
    assert result.objs == [1.0, 2.0, 3.0]


def test_complex():
    g = Group([1, 2, 3])
    result = g.__complex__()
    assert result.objs == [(1 + 0j), (2 + 0j), (3 + 0j)]


def test_bool():
    g = Group([0, 1, 2])
    result = g.__bool__()
    assert result.objs == [False, True, True]


def test_trunc():
    g = Group([1.7, 2.3, -3.9])
    result = math.trunc(g)
    assert result.objs == [1, 2, -3]


def test_floor():
    g = Group([1.7, 2.3, -3.1])
    result = math.floor(g)
    assert result.objs == [1, 2, -4]


def test_ceil():
    g = Group([1.1, 2.0, -3.9])
    result = math.ceil(g)
    assert result.objs == [2, 2, -3]


def test_round_uses_math_round():
    g = Group([1.5, 2.3, 3.7])
    # __round__ references math.round which doesn't exist
    with pytest.raises(AttributeError):
        round(g)


def test_str():
    g = Group([1, 2, 3])
    assert str(g) == "Group([1, 2, 3])"


def test_repr():
    g = Group([1, 2, 3])
    assert repr(g) == "Group([1, 2, 3])"


def test_format():
    g = Group([1, 2, 3])
    assert format(g, "") == "Group([1, 2, 3])"
    assert f"{g}" == "Group([1, 2, 3])"


def test_iter():
    g = Group([10, 20, 30])
    assert list(g) == [10, 20, 30]


def test_len():
    g = Group([1, 2, 3])
    assert len(g) == 3


def test_contains():
    g = Group([1, 2, 3])
    assert 2 in g
    assert 5 not in g


def test_map():
    g = Group([1, 2, 3])
    result = g.map(lambda x: x * 10)
    assert result.objs == [10, 20, 30]


def test_map_preserves_attrs():
    g = Group([1, 2], attrs="test")
    result = g.map(lambda x: x + 1)
    assert result.attrs == "test"


def test_filter():
    g = Group([1, 2, 3, 4, 5])
    result = g.filter(lambda x: x > 3)
    assert result.objs == [4, 5]


def test_filter_preserves_attrs():
    g = Group([1, 2, 3], attrs="meta")
    result = g.filter(lambda x: x > 1)
    assert result.attrs == "meta"


def test_foreach():
    g = Group([1, 2, 3])
    collected = []
    g.foreach(lambda x: collected.append(x))
    assert collected == [1, 2, 3]


def test_call_scalar_args():
    g = Group([lambda x: x + 1, lambda x: x * 2])
    result = g(5)
    assert result.objs == [6, 10]


def test_call_group_args():
    g = Group([lambda x: x + 1, lambda x: x * 2])
    arg = Group([10, 20])
    result = g(arg)
    assert result.objs == [11, 40]


def test_call_kwargs():
    def f(x=0):
        return x + 1

    def h(x=0):
        return x * 2

    g = Group([f, h])
    result = g(x=5)
    assert result.objs == [6, 10]


def test_call_group_kwargs():
    def f(x=0):
        return x + 1

    def h(x=0):
        return x * 2

    g = Group([f, h])
    result = g(x=Group([10, 20]))
    assert result.objs == [11, 40]


def test_delitem():
    d1 = {0: "a", 1: "b"}
    d2 = {0: "c", 1: "d"}
    g = Group([d1, d2])
    del g[0]
    assert d1 == {1: "b"}
    assert d2 == {1: "d"}


def test_delitem_group_keys():
    d1 = {0: "a", 1: "b"}
    d2 = {0: "c", 1: "d"}
    g = Group([d1, d2])
    del g[Group([0, 1])]
    assert d1 == {1: "b"}
    assert d2 == {0: "c"}


def test_getattr_scalar():
    class Obj:
        def __init__(self, val):
            self.val = val

    g = Group([Obj(1), Obj(2)])
    result = g.val
    assert result.objs == [1, 2]


def test_getattr_group():
    class Obj:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    g = Group([Obj(1, 10), Obj(2, 20)])
    keys = Group(["a", "b"])
    result = g.__getattr__(keys)
    assert result.objs == [1, 20]


def test_setattr_scalar():
    class Obj:
        def __init__(self):
            self.x = 0

    o1, o2 = Obj(), Obj()
    g = Group([o1, o2])
    g.x = 99
    assert o1.x == 99
    assert o2.x == 99


def test_setattr_group_values():
    class Obj:
        def __init__(self):
            self.x = 0

    o1, o2 = Obj(), Obj()
    g = Group([o1, o2])
    g.x = Group([10, 20])
    assert o1.x == 10
    assert o2.x == 20


def test_binary_operator_size_mismatch_truncates():
    g1 = Group([1, 2, 3])
    g2 = Group([10, 20])
    # __binary_operator uses zip directly, so mismatched sizes truncate
    result = g1 + g2
    assert result.objs == [11, 22]


def test_check_and_return_iterable_size_mismatch():
    g = Group([1, 2, 3])
    with pytest.raises(ValueError, match="group size mismatch"):
        g(Group([1, 2]))


def test_divmod_scalar():
    g = Group([7, 10, 15])
    result = divmod(g, 3)
    assert result.objs == [(2, 1), (3, 1), (5, 0)]


def test_rdivmod():
    g = Group([3, 4, 5])
    result = g.__rdivmod__(10)
    assert result.objs == [(3, 1), (2, 2), (2, 0)]


def test_lshift():
    g = Group([1, 2, 3])
    result = g << 2
    assert result.objs == [4, 8, 12]


def test_rshift():
    g = Group([8, 16, 32])
    result = g >> 2
    assert result.objs == [2, 4, 8]


def test_and():
    g = Group([0b1100, 0b1010])
    result = g & 0b1010
    assert result.objs == [0b1000, 0b1010]


def test_or():
    g = Group([0b1100, 0b1010])
    result = g | 0b0011
    assert result.objs == [0b1111, 0b1011]


def test_xor():
    g = Group([0b1100, 0b1010])
    result = g ^ 0b1010
    assert result.objs == [0b0110, 0b0000]


def test_rand():
    g = Group([0b1100, 0b1010])
    result = g.__rand__(0b1010)
    assert result.objs == [0b1000, 0b1010]


def test_ror():
    g = Group([0b1100, 0b1010])
    result = g.__ror__(0b0011)
    assert result.objs == [0b1111, 0b1011]


def test_rxor():
    g = Group([0b1100, 0b1010])
    result = g.__rxor__(0b1010)
    assert result.objs == [0b0110, 0b0000]


def test_rlshift():
    g = Group([1, 2])
    result = g.__rlshift__(1)
    assert result.objs == [2, 4]


def test_rrshift():
    g = Group([1, 2])
    result = g.__rrshift__(16)
    assert result.objs == [8, 4]


def test_binary_op_preserves_attrs():
    g = Group([1, 2, 3], attrs="keep")
    result = g + 1
    assert result.attrs == "keep"


def test_binary_op_preserves_type():
    class MyGroup(Group):
        pass

    g = MyGroup([1, 2, 3])
    result = g + 1
    assert type(result) is MyGroup
