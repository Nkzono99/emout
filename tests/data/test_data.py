import emout
import pytest


@pytest.fixture
def phisp(data):
    return data.phisp


def test_open_data(data):
    assert type(data.phisp) == emout.data.data.GridDataSeries
    assert type(data.ex) == emout.data.data.GridDataSeries

    with pytest.raises(AttributeError) as attr_error:
        data.no_attrs


def test_data_type(data):
    assert type(data.phisp) == emout.data.GridDataSeries

    assert type(data.phisp[:]) == emout.data.Data4d
    assert type(data.phisp[1]) == emout.data.Data3d
    assert type(data.phisp[1][1, :, :]) == emout.data.Data2d
    assert type(data.phisp[1][1, 1, :]) == emout.data.Data1d

    assert type(data.phisp[1, 1]) == emout.data.Data2d
    assert type(data.phisp[1, 1, 1]) == emout.data.Data1d


@pytest.mark.parametrize('tslice, expected', [
    (1, list('zyx')),
    (-1, list('zyx')),
    (slice(None), list('tzyx')),
    (slice(None, 1), list('tzyx')),
    (slice(1, None), list('tzyx')),
    ([1, 2, 4], list('tzyx'))
])
def test_tslice(phisp, tslice, expected):
    assert phisp[tslice].use_axes == expected

@pytest.mark.parametrize('tslice, expected', [
    (1, (100, 30, 30)),
    (slice(None), (5, 100, 30, 30)),
    (slice(1), (1, 100, 30, 30)),
    (slice(-1), (4, 100, 30, 30)),
    ([1, 2, 4], (3, 100, 30, 30))
])
def test_tslice_with_shape(phisp, tslice, expected):
    assert phisp[tslice].shape == expected


def test_2dslice(phisp):
    assert phisp[1][1, :, :].use_axes == list('yx')
    assert phisp[1][:, :, 1].use_axes == list('zy')
    assert phisp[1][:, 1, :].use_axes == list('zx')

    assert phisp[:][1, 1, :, :].use_axes == list('yx')
    assert phisp[:][1, :, :, 1].use_axes == list('zy')
    assert phisp[:][1, :, 1, :].use_axes == list('zx')

    assert phisp[:][:, 1, 1, :].use_axes == list('tx')
    assert phisp[:][:, 1, :, 1].use_axes == list('ty')
    assert phisp[:][:, :, 1, 1].use_axes == list('tz')

    assert phisp[1, 1].use_axes == list('yx')
    assert phisp[1, :, 1].use_axes == list('zx')
    assert phisp[1, 1, 1, :].use_axes == list('x')

    assert phisp[1, :, :, 1].use_axes == list('zy')
    assert phisp[1, :, :, 1][1].use_axes == list('y')
    assert phisp[1, :, :, 1][:, 1].use_axes == list('z')

    assert phisp[:, :, :, :].use_axes == list('tzyx')
    assert phisp[:][:, :, :, 1].use_axes == list('tzy')
    assert phisp[:, :, :, 1].use_axes == list('tzy')


@pytest.mark.parametrize('phisp, tslice, slices, expected', [
    (None, 1, (1, 1, slice(None)), list('x')),
    (None, 1, (1, slice(None), 1), list('y')),
    (None, 1, (slice(None), 1, 1), list('z')),
    (None, slice(None), (1, 1, 1, slice(None)), list('x')),
    (None, slice(None), (1, 1, slice(None), 1), list('y')),
    (None, slice(None), (1, slice(None), 1, 1), list('z')),
    (None, slice(None), (slice(None), 1, 1, 1), list('t')),
], indirect=['phisp'])
def test_1dslice(phisp, tslice, slices, expected):
    assert phisp[tslice][slices].use_axes == expected


def test_shape(phisp):
    assert phisp[1].shape == (100, 30, 30)
    assert phisp[1][:10, :, :].shape == (10, 30, 30)
    assert phisp[1][1, :, :].shape == (30, 30)
    assert phisp[1][1, :10, :].shape == (10, 30)
    assert phisp[1].shape == (100, 30, 30)


@pytest.mark.parametrize('name', [
    ('phisp'),
    ('nd1p'),
    ('nd12p'),
    ('jx'),
    ('jy'),
    ('jz'),
    ('bx'),
    ('by'),
    ('bz'),
    ('ex'),
    ('ey'),
    ('ez'),
    ('j1x'),
    ('j1y'),
    ('j1z'),
    ('rho'),
    ('rhobk')
])
def test_name2unit(name):
    name2unit = emout.Emout.name2unit
    assert name2unit.get(name) is not None
