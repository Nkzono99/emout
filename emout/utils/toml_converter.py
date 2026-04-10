"""Attribute-access wrapper for raw TOML data via :class:`TomlData`.

The plasma.toml to plasma.inp conversion is handled by the MPIEMSES3D
``toml2inp`` command.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


_GROUP_TABLE_MAP = {
    (): {"species": "species_groups"},
    ("meta", "physical"): {
        "species": "species_groups",
        "conductors": "conductor_groups",
    },
    ("ptcond",): {
        "boundaries": "boundary_groups",
        "objects": "object_groups",
    },
    ("emissn",): {"planes": "plane_groups"},
    ("dipole",): {"sources": "source_groups"},
    ("jsrc",): {"sources": "source_groups"},
    ("testch",): {"charges": "charge_groups"},
}


# ---------------------------------------------------------------------------
# TomlData: attribute-access wrapper for TOML dictionaries
# ---------------------------------------------------------------------------


class TomlData:
    """Attribute-access wrapper for a TOML dictionary structure.

    Enables dot-access like ``data.species[0].wp`` for nested dicts and
    lists. Dictionary-style access (``data["tmgrid"]["nx"]``) is also
    supported.

    Parameters
    ----------
    data : dict
        Dictionary loaded from TOML.
    """

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", data)

    # --- dict-like access ---

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return (_wrap(v) for v in self._data.values())

    def items(self):
        return ((k, _wrap(v)) for k, v in self._data.items())

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        return _wrap(val) if val is not default else default

    # --- attribute access ---

    def __getattr__(self, key: str) -> Any:
        try:
            return _wrap(self._data[key])
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    # --- display ---

    def __repr__(self) -> str:
        return f"TomlData({self._data!r})"

    def __str__(self) -> str:
        return str(self._data)

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying dictionary."""
        return self._data


def _wrap(value: Any) -> Any:
    """Recursively wrap dicts as TomlData and lists of dicts as lists of TomlData."""
    if isinstance(value, dict):
        return TomlData(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with *overrides* recursively merged into *base*."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_group_entries(
    entries: list[dict],
    groups: Any,
) -> list[dict]:
    """Expand group defaults into each entry by resolving group_id references."""
    resolved: list[dict] = []
    groups_dict = groups if isinstance(groups, dict) else {}

    for entry in entries:
        group_id = entry.get("group_id")
        merged: Dict[str, Any] = {}

        if isinstance(group_id, str) and group_id in groups_dict:
            group_defaults = groups_dict[group_id]
            if isinstance(group_defaults, dict):
                merged = _deep_merge(merged, group_defaults)

        merged = _deep_merge(merged, entry)
        merged.pop("group_id", None)
        resolved.append(merged)

    return resolved


def _resolve_groups_in_data(
    data: Dict[str, Any],
    *,
    purge_groups: bool = False,
    path: tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Expand ``*_groups`` in the TOML data into each entry."""
    resolved: Dict[str, Any] = {}

    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_groups_in_data(
                value,
                purge_groups=purge_groups,
                path=path + (key,),
            )
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_groups_in_data(
                    item,
                    purge_groups=purge_groups,
                    path=path + (key,),
                )
                if isinstance(item, dict)
                else copy.deepcopy(item)
                for item in value
            ]
        else:
            resolved[key] = copy.deepcopy(value)

    group_table_map = _GROUP_TABLE_MAP.get(path, {})
    for entries_key, groups_key in group_table_map.items():
        entries = resolved.get(entries_key)
        groups = resolved.get(groups_key)
        if isinstance(entries, list) and entries and all(isinstance(entry, dict) for entry in entries):
            resolved[entries_key] = _resolve_group_entries(entries, groups)
            if purge_groups:
                resolved.pop(groups_key, None)

    return resolved


def load_toml(
    toml_path: Path,
    *,
    resolve_groups: bool = False,
    purge_groups: bool = False,
) -> TomlData:
    """Load plasma.toml and return it as a TomlData wrapper.

    Parameters
    ----------
    toml_path : Path
        Path to plasma.toml.
    resolve_groups : bool, optional
        Resolve ``group_id`` references and expand ``*_groups`` defaults
        into each entry.
    purge_groups : bool, optional
        When ``resolve_groups=True``, remove the source ``*_groups``
        tables from the returned data.

    Returns
    -------
    TomlData
        Attribute-access wrapper for the TOML dictionary.
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    if resolve_groups:
        data = _resolve_groups_in_data(data, purge_groups=purge_groups)
    return TomlData(data)
