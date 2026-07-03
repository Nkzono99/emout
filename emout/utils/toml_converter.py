"""TOML loading helpers for EMSES ``plasma.toml`` files.

The public ``data.toml`` interface preserves the native TOML structure via
:class:`TomlData`.  ``load_toml_as_inp`` also builds an :class:`InpFile`
compatible namelist view so existing ``data.inp`` code can use TOML-backed
parameters without relying on an external ``toml2inp`` command.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import f90nml

from emout.utils.emsesinp import InpFile, UnitConversionKey

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


_SPECIES_KEY_GROUPS = {
    "wp": "plasma",
    "qm": "intp",
    "npin": "intp",
    "path": "intp",
    "peth": "intp",
    "vdri": "intp",
}


# ---------------------------------------------------------------------------
# TomlData: attribute-access wrapper for TOML dictionaries
# ---------------------------------------------------------------------------


class TomlData:
    """Attribute-access wrapper for a TOML dictionary structure.

    Enables dot-access like ``data.species[0].wp`` for nested dicts and
    lists. If a requested attribute is not present at the current level,
    the shallowest nested TOML level is searched. Multiple matches are
    returned as a list, while a single match is returned as a scalar.
    Dictionary-style access (``data["tmgrid"]["nx"]``) remains direct-only.

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
            values = _find_transparent_values(self._data, key)
            if len(values) == 1:
                return _wrap(values[0])
            if values:
                return [_wrap(value) for value in values]
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


def _find_transparent_values(data: Dict[str, Any], key: str) -> list[Any]:
    """Return values for *key* from the shallowest nested TOML level."""
    matches: list[tuple[int, Any]] = []

    def walk(value: Any, depth: int) -> None:
        if isinstance(value, dict):
            if key in value:
                matches.append((depth, value[key]))
            for child in value.values():
                walk(child, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth)

    walk(data, 0)
    if not matches:
        return []

    shallowest = min(depth for depth, _ in matches)
    return [value for depth, value in matches if depth == shallowest]


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


def _load_toml_dict(toml_path: Path) -> Dict[str, Any]:
    """Load *toml_path* and return a plain dictionary."""
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def _unit_conversion_key(data: Dict[str, Any]) -> Optional[UnitConversionKey]:
    """Return a unit conversion key from ``[meta.unit_conversion]``."""
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return None
    unit_conversion = meta.get("unit_conversion")
    if not isinstance(unit_conversion, dict):
        return None
    if "dx" not in unit_conversion or "to_c" not in unit_conversion:
        return None
    return UnitConversionKey(
        float(unit_conversion["dx"]),
        float(unit_conversion["to_c"]),
    )


def _plain_value(value: Any) -> Any:
    """Return a deepcopy of TOML values with nested mappings normalized."""
    if isinstance(value, dict):
        return {key: _plain_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain_value(child) for child in value]
    return copy.deepcopy(value)


def _is_list_of_dicts(value: Any) -> bool:
    """Return whether *value* is a non-empty list of dictionaries."""
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)


def _set_group_value(group: f90nml.Namelist, key: str, value: Any) -> None:
    """Set a namelist value and attach Fortran-style start indexes for lists."""
    value = _plain_value(value)
    group[key] = value
    if isinstance(value, list):
        if value and all(isinstance(item, list) or item is None for item in value):
            group.start_index[key] = [None, 1]
        else:
            group.start_index[key] = [1]


def _ensure_group(nml: f90nml.Namelist, group_name: str) -> f90nml.Namelist:
    """Return an existing namelist group or create it."""
    if group_name not in nml:
        nml[group_name] = f90nml.Namelist()
    return nml[group_name]


def _entry_target_key(group_name: str, entries_key: str, entry_key: str) -> str:
    """Map TOML list-entry keys onto legacy namelist parameter names."""
    if group_name == "ptcond" and entries_key == "boundaries" and entry_key == "type":
        return "boundary_types"
    return entry_key


def _flatten_entry_list(group_name: str, entries_key: str, entries: list[dict]) -> Dict[str, list[Any]]:
    """Flatten ``[[group.entries]]`` into namelist-style arrays."""
    flattened: Dict[str, list[Any]] = {}
    for index, entry in enumerate(entries):
        for key, value in entry.items():
            target = _entry_target_key(group_name, entries_key, key)
            flattened.setdefault(target, [None] * len(entries))
            flattened[target][index] = _plain_value(value)
    return flattened


def _group_values_from_table(group_name: str, table: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a TOML table into values for a single namelist group."""
    values: Dict[str, Any] = {}
    for key, value in table.items():
        if key.endswith("_groups"):
            continue
        if isinstance(value, dict):
            continue
        if _is_list_of_dicts(value):
            values.update(_flatten_entry_list(group_name, key, value))
        else:
            values[key] = _plain_value(value)
    return values


def _merge_group_values(nml: f90nml.Namelist, group_name: str, values: Dict[str, Any]) -> None:
    """Merge *values* into *group_name* of a namelist."""
    if not values:
        return
    group = _ensure_group(nml, group_name)
    for key, value in values.items():
        _set_group_value(group, key, value)


def _merge_species(nml: f90nml.Namelist, species: Any) -> None:
    """Fold top-level ``[[species]]`` entries into legacy namelist arrays."""
    if not _is_list_of_dicts(species):
        return

    aggregated: Dict[str, list[Any]] = {}
    for index, entry in enumerate(species):
        for key, value in entry.items():
            if key == "group_id":
                continue
            aggregated.setdefault(key, [None] * len(species))
            aggregated[key][index] = _plain_value(value)

    for key, values in aggregated.items():
        group_name = _SPECIES_KEY_GROUPS.get(key, "plasma")
        _merge_group_values(nml, group_name, {key: values})

    emissn = _ensure_group(nml, "emissn")
    if "nspec" not in emissn:
        emissn["nspec"] = len(species)


def _toml_data_to_namelist(data: Dict[str, Any]) -> f90nml.Namelist:
    """Convert resolved TOML data into a legacy namelist view."""
    nml = f90nml.Namelist()

    for group_name, table in data.items():
        if group_name in {"meta", "species"} or group_name.endswith("_groups"):
            continue
        if not isinstance(table, dict):
            continue
        _merge_group_values(nml, group_name, _group_values_from_table(group_name, table))

    _merge_species(nml, data.get("species"))
    return nml


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
    data = _load_toml_dict(toml_path)
    if resolve_groups:
        data = _resolve_groups_in_data(data, purge_groups=purge_groups)
    return TomlData(data)


def load_toml_as_inp(
    toml_path: Path,
    *,
    resolve_groups: bool = True,
    purge_groups: bool = True,
) -> InpFile:
    """Load ``plasma.toml`` and return an :class:`InpFile` compatible view.

    Parameters
    ----------
    toml_path : Path
        Path to ``plasma.toml``.
    resolve_groups : bool, optional
        Resolve ``group_id`` references before building the namelist view.
    purge_groups : bool, optional
        Remove ``*_groups`` tables after group resolution.

    Returns
    -------
    InpFile
        TOML-backed namelist-compatible parameter object.
    """
    data = _load_toml_dict(toml_path)
    if resolve_groups:
        data = _resolve_groups_in_data(data, purge_groups=purge_groups)

    inp = InpFile(convkey=_unit_conversion_key(data))
    inp.nml = _toml_data_to_namelist(data)
    return inp
