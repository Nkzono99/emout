from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import scipy.constants as cn

from emout.utils import RegexDict, UnitTranslator

if TYPE_CHECKING:
    from emout.facade import Emout



def build_name2unit_mapping(max_ndp: int = 9) -> RegexDict:
    """必要なオブジェクトを構築する。
    
    Parameters
    ----------
    max_ndp : int, optional
        密度種別の最大番号です。
    Returns
    -------
    RegexDict
        処理結果です。
    """
    mapping: dict[str, Callable[..., UnitTranslator]] = {
        r"phisp": lambda out: out.unit.phi,
        r"rho": lambda out: out.unit.rho,
        r"rhobk": lambda out: out.unit.rho,
        r"j.*": lambda out: out.unit.J,
        r"b[xyz]": lambda out: out.unit.H,
        r"rb[xyz]": lambda out: out.unit.H,
        r"e[xyz]": lambda out: out.unit.E,
        r"re[xyz]": lambda out: out.unit.E,
        r"t": t_unit,
        r"axis": lambda out: out.unit.length,
        r"rhobksp[1-9]": lambda out: out.unit.rho,
        r"rhobkac[1-9]": lambda out: out.unit.rho,
        r"nd[1-9]\d*p": ndp_unit,
        r"p[1-9]v[xyz]": lambda out: out.unit.v,
        r"p[1-9][xyz]": lambda out: out.unit.length,
        r"p[1-9]tid": lambda _: UnitTranslator(from_unit=1, to_unit=1, name="trace id", unit="")        
    }

    return RegexDict(mapping)


def t_unit(out: Emout) -> UnitTranslator:
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
    return (out.unit.t * UnitTranslator(out.inp.ifdiag * out.inp.dt, 1)).set_name(
        "t", unit="s"
    )


def wpet_unit(out: Emout) -> UnitTranslator:
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
        out.inp.wp[0] * out.inp.ifdiag * out.inp.dt, 1, name="wpe * t", unit=""
    )


def wpit_unit(out: Emout) -> UnitTranslator:
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
        out.inp.wp[1] * out.inp.ifdiag * out.inp.dt, 1, name="wpi * t", unit=""
    )


def none_unit(out: Emout) -> UnitTranslator:
    """無次元の単位変換器を返す。
    
    Parameters
    ----------
    out : Emout
        単位変換の参照元 `Emout` オブジェクトです。
    Returns
    -------
    UnitTranslator
        処理結果です。
    """
    return UnitTranslator(1, 1, name="", unit="")


def ndp_unit(out: Emout) -> UnitTranslator:
    """`ndp` の単位変換器を返す。
    
    Parameters
    ----------
    out : Emout
        単位変換の参照元 `Emout` オブジェクトです。
    Returns
    -------
    UnitTranslator
        処理結果です。
    """
    wp = out.unit.f.reverse(out.inp.wp[0])
    mp = abs(cn.m_e / out.inp.qm[0])
    np = wp**2 * mp * cn.epsilon_0 / cn.e**2
    return UnitTranslator(np * 1e-6, 1.0, name="number density", unit="/cc")


def nd3p_unit(out: Emout) -> UnitTranslator:
    """`nd3p` の単位変換器を返す。
    
    Parameters
    ----------
    out : Emout
        単位変換の参照元 `Emout` オブジェクトです。
    Returns
    -------
    UnitTranslator
        処理結果です。
    """
    wpp = out.unit.f.reverse(out.inp.wp[2])
    np = wpp**2 * cn.m_e * cn.epsilon_0 / cn.e**2
    return UnitTranslator(np * 1e-6, 1.0, name="number density", unit="/cc")
