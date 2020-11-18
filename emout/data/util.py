from emout.utils import UnitTranslator


def wpet_unit(out):
    """wpe * tの単位変換器を生成する.

    以下のコードを実行することで、データのt軸をwpe*tで規格化できる.

    >>> Emout.name2unit['t'] = wpet_unit

    Parameters
    ----------
    out : Emout
        Emoutオブジェクト

    Returns
    -------
    UnitTranslator
        wpe * tの単位変換器
    """
    return UnitTranslator(
        out.inp.wp[0] * out.inp.ifdiag * out.inp.dt,
        1,
        name='wpe * t',
        unit=''
    )


def wpit_unit(out):
    """wpi * tの単位変換器を生成する.

    以下のコードを実行することで、データのt軸をwpe*tで規格化できる.

    >>> Emout.name2unit['t'] = wpit_unit

    Parameters
    ----------
    out : Emout
        Emoutオブジェクト

    Returns
    -------
    UnitTranslator
        wpi * tの単位変換器
    """
    return UnitTranslator(
        out.inp.wp[1] * out.inp.ifdiag * out.inp.dt,
        1,
        name='wpi * t',
        unit=''
    )


def none_unit(out):
    return UnitTranslator(
        1,
        1,
        name='',
        unit=''
    )