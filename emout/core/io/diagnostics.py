"""Readers for small EMSES diagnostic text files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DiagnosticDataFrame(pd.DataFrame):
    """DataFrame for EMSES diagnostic text files with SI conversion metadata."""

    _metadata = ["_emout_unit", "_emout_diagnostic"]

    def __init__(self, *args, unit=None, diagnostic: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._emout_unit = unit
        self._emout_diagnostic = diagnostic

    @property
    def _constructor(self):
        return DiagnosticDataFrame

    @property
    def val_si(self) -> "DiagnosticDataFrame":
        """Return diagnostic columns converted from EMSES units to SI units."""
        unit = self._emout_unit
        if unit is None:
            raise ValueError("unit is not set.")

        result = DiagnosticDataFrame(self.copy(deep=True), unit=unit, diagnostic=self._emout_diagnostic)
        for column in result.columns:
            translator = _diagnostic_column_unit(self._emout_diagnostic, str(column), unit)
            result[column] = translator.reverse(result[column])
        return result


def _diagnostic_column_unit(diagnostic: str | None, column: str, unit):
    if diagnostic in {"icur", "ocur"}:
        return unit.t if column == "step" or column.endswith("_step") else unit.i
    if diagnostic == "pbody":
        return unit.t if column == "step" else unit.phi
    raise ValueError(f"unsupported diagnostic type: {diagnostic!r}")


def icur_columns(inp) -> list[str]:
    """Return column names for an ``icur`` file."""
    names = []
    for ispec in range(inp.nspec):
        names.append(f"{ispec + 1}_step")
        for ipc in range(inp.npc):
            names.append(f"{ispec + 1}_body{ipc + 1}")
            names.append(f"{ispec + 1}_body{ipc + 1}_ema")
    return names


def ocur_columns(inp) -> list[str]:
    """Return column names for an ``ocur`` file."""
    return icur_columns(inp)


def pbody_columns(inp) -> list[str]:
    """Return column names for a ``pbody`` file."""
    return ["step"] + [f"body{i + 1}" for i in range(inp.npc + 1)]


def read_icur(path: str | Path, inp, unit=None) -> pd.DataFrame:
    """Read an ``icur`` diagnostic file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'icur' file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=icur_columns(inp))
    return DiagnosticDataFrame(df, unit=unit, diagnostic="icur")


def read_ocur(path: str | Path, inp, unit=None) -> pd.DataFrame:
    """Read an ``ocur`` diagnostic file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'ocur' file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ocur_columns(inp))
    return DiagnosticDataFrame(df, unit=unit, diagnostic="ocur")


def read_pbody(path: str | Path, inp, unit=None) -> pd.DataFrame:
    """Read a ``pbody`` diagnostic file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'pbody' file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", names=pbody_columns(inp))
    return DiagnosticDataFrame(df, unit=unit, diagnostic="pbody")
