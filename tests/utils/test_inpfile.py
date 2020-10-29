import pytest


@pytest.fixture
def inp(data):
    return data.inp


def test_getitem(inp):
    assert inp['tmgrid']['nx'] == 64
    assert inp['tmgrid']['ny'] == 64
    assert inp['tmgrid']['nz'] == 512
    assert inp['mpi']['nodes'] == [4, 4, 32]


def test_getitem_with_omitting(inp):
    assert inp['nx'] == 64
    assert inp['ny'] == 64
    assert inp['nz'] == 512
    assert inp['nodes'] == [4, 4, 32]


def test_getattr(inp):
    assert inp.tmgrid.nx == 64
    assert inp.tmgrid.ny == 64
    assert inp.tmgrid.nz == 512
    assert inp.mpi.nodes == [4, 4, 32]


def test_getattr_with_omitting(inp):
    assert inp.nx == 64
    assert inp.ny == 64
    assert inp.nz == 512
    assert inp.nodes == [4, 4, 32]
