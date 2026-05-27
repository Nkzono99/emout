"""Readers for small EMSES diagnostic text files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def icur_columns(inp) -> list[str]:
    """Return column names for an ``icur`` file."""
    names = []
    for ispec in range(inp.nspec):
        names.append(f"{ispec + 1}_step")
        for ipc in range(inp.npc):
            names.append(f"{ispec + 1}_body{ipc + 1}")
            names.append(f"{ispec + 1}_body{ipc + 1}_ema")
    return names


def pbody_columns(inp) -> list[str]:
    """Return column names for a ``pbody`` file."""
    return ["step"] + [f"body{i + 1}" for i in range(inp.npc + 1)]


def read_icur(path: str | Path, inp) -> pd.DataFrame:
    """Read an ``icur`` diagnostic file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'icur' file not found: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=icur_columns(inp))


def read_pbody(path: str | Path, inp) -> pd.DataFrame:
    """Read a ``pbody`` diagnostic file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'pbody' file not found: {path}")
    return pd.read_csv(path, sep=r"\s+", names=pbody_columns(inp))
