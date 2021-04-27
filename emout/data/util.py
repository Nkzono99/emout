import scipy.constants as cn
from emout.utils import UnitTranslator


def t_unit(out):
    """tの単位変換器を生成する.

    Parameters
    ----------
    out : Emout
        Emoutオブジェクト

    Returns
    -------
    UnitTranslator
        tの単位変換器
    """
    return (out.unit.t * UnitTranslator(out.inp.ifdiag * out.inp.dt, 1)).set_name('t', unit='s')


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


def ndp_unit(out):
    wpe = out.unit.f.reverse(out.inp.wp[0])
    ne = wpe ** 2 * cn.m_e * cn.epsilon_0 / cn.e**2
    return UnitTranslator(
        ne * 1e-6,
        1.0,
        name='number density',
        unit='/cc'
    )


def nd3p_unit(out):
    wpp = out.unit.f.reverse(out.inp.wp[2])
    np = wpp ** 2 * cn.m_e * cn.epsilon_0 / cn.e**2
    return UnitTranslator(
        np * 1e-6,
        1.0,
        name='number density',
        unit='/cc'
    )

