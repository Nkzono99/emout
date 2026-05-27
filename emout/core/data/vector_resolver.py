"""Resolve vector field aliases such as ``exz`` and ``bxyz``."""

from __future__ import annotations

import re
from typing import Callable

from .vector_data import VectorData


def resolve_vector_name(name: str, load_component: Callable[[str], object], *, fallback_3d: bool = False):
    """Resolve a vector alias by loading its scalar components.

    Parameters
    ----------
    name : str
        Requested field name.
    load_component : callable
        Function that returns one scalar component by name.
    fallback_3d : bool, default False
        When True, a failed 3-component resolution returns ``None`` so the
        caller can try resolving the name as a scalar field.
    """
    m3 = re.match(r"^(.+?)([xyz])([xyz])([xyz])$", name)
    if m3:
        base, *axes = m3.groups()
        if len(set(axes)) == 3:
            try:
                return VectorData(
                    [load_component(f"{base}{axis}") for axis in axes],
                    name=name,
                    component_axes=tuple(axes),
                )
            except (AttributeError, KeyError, FileNotFoundError, OSError):
                if fallback_3d:
                    return None
                raise

    m2 = re.match(r"^(.+)([xyz])([xyz])$", name)
    if m2:
        base, axis1, axis2 = m2.groups()
        axes = (axis1, axis2)
        return VectorData(
            [load_component(f"{base}{axis}") for axis in axes],
            name=name,
            component_axes=axes,
        )

    return None
