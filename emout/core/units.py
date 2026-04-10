"""Unit mapping from EMSES field names to physical-unit translators.

Defines :func:`build_name2unit_mapping` which creates a regex-keyed
dictionary that maps grid-data names (e.g. ``phisp``, ``bx``, ``nd1p``)
to factory functions producing :class:`~emout.utils.UnitTranslator` instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import scipy.constants as cn

from emout.utils import RegexDict, UnitTranslator

if TYPE_CHECKING:
    from emout.core.facade import Emout



def build_name2unit_mapping(max_ndp: int = 9) -> RegexDict:
    """Build a regex dictionary mapping field-name patterns to unit-translator factories.

    Parameters
    ----------
    max_ndp : int, optional
        Maximum species number for density fields (``nd{N}p``).

    Returns
    -------
    RegexDict
        Regex dictionary mapping field-name patterns to unit-translator
        factory functions.
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
    """Return an identity (dimensionless) unit translator.

    Parameters
    ----------
    out : Emout
        Source :class:`Emout` instance (unused, kept for API consistency).

    Returns
    -------
    UnitTranslator
        Identity translator (dimensionless).
    """
    return UnitTranslator(1, 1, name="", unit="")


def ndp_unit(out: Emout) -> UnitTranslator:
    """Return a unit translator for grid-density fields (``nd*p``).

    Parameters
    ----------
    out : Emout
        Source :class:`Emout` instance providing plasma parameters.

    Returns
    -------
    UnitTranslator
        Translator converting grid-density to number density in /cc.
    """
    wp = out.unit.f.reverse(out.inp.wp[0])
    mp = abs(cn.m_e / out.inp.qm[0])
    np = wp**2 * mp * cn.epsilon_0 / cn.e**2
    return UnitTranslator(np * 1e-6, 1.0, name="number density", unit="/cc")


def nd3p_unit(out: Emout) -> UnitTranslator:
    """Return a unit translator for species-3 grid-density (``nd3p``).

    Parameters
    ----------
    out : Emout
        Source :class:`Emout` instance providing plasma parameters.

    Returns
    -------
    UnitTranslator
        Translator converting grid-density to number density in /cc for
        species 3.
    """
    wpp = out.unit.f.reverse(out.inp.wp[2])
    np = wpp**2 * cn.m_e * cn.epsilon_0 / cn.e**2
    return UnitTranslator(np * 1e-6, 1.0, name="number density", unit="/cc")
