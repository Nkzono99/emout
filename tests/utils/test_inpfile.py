import pytest


@pytest.fixture
def inp(data):
    """入力パラメータを返す。

    Parameters
    ----------
    data : object
        処理対象のデータ。

    Returns
    -------
    object
        処理結果です。
    """
    return data.inp


def test_getitem(inp):
    """getitem のテストを行う。

    Parameters
    ----------
    inp : object
        入力パラメータオブジェクトです。
    Returns
    -------
    None
        戻り値はありません。
    """
    assert inp["tmgrid"]["nx"] == 64
    assert inp["tmgrid"]["ny"] == 64
    assert inp["tmgrid"]["nz"] == 512
    assert inp["mpi"]["nodes"] == [4, 4, 32]


def test_getitem_with_omitting(inp):
    """getitem with omitting のテストを行う。

    Parameters
    ----------
    inp : object
        入力パラメータオブジェクトです。
    Returns
    -------
    None
        戻り値はありません。
    """
    assert inp["nx"] == 64
    assert inp["ny"] == 64
    assert inp["nz"] == 512
    assert inp["nodes"] == [4, 4, 32]


def test_getattr(inp):
    """getattr のテストを行う。

    Parameters
    ----------
    inp : object
        入力パラメータオブジェクトです。
    Returns
    -------
    None
        戻り値はありません。
    """
    assert inp.tmgrid.nx == 64
    assert inp.tmgrid.ny == 64
    assert inp.tmgrid.nz == 512
    assert inp.mpi.nodes == [4, 4, 32]


def test_getattr_with_omitting(inp):
    """getattr with omitting のテストを行う。

    Parameters
    ----------
    inp : object
        入力パラメータオブジェクトです。
    Returns
    -------
    None
        戻り値はありません。
    """
    assert inp.nx == 64
    assert inp.ny == 64
    assert inp.nz == 512
    assert inp.nodes == [4, 4, 32]
