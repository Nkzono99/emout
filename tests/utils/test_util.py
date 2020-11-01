from emout.utils import RegexDict


def test_regexdict():
    name2unit = RegexDict({
        r'a': 0,
        r'a[0-9]+': 5,
        r'b[0-9]': 10,
    })
    assert name2unit['a'] == 0
    assert name2unit['a1'] == 5
    assert name2unit['b1'] == 10
    assert name2unit['b8'] == 10
