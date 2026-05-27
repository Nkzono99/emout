"""Shared selector utilities for ``(t, z, y, x)`` grid indexing."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_index(index: int, size: int) -> int:
    """Normalize one integer index against an axis length."""
    index = int(index)
    if index < 0:
        index += size
    if not 0 <= index < size:
        raise IndexError(f"index {index} is out of bounds for axis with size {size}")
    return index


def normalize_selector(selector: Any, size: int) -> Any:
    """Normalize an integer, slice, or explicit selector for one axis."""
    if isinstance(selector, slice):
        rng = range(*selector.indices(size))
        return slice(rng.start, rng.stop, rng.step)
    if isinstance(selector, list):
        return tuple(normalize_index(index, size) for index in selector)
    if isinstance(selector, tuple):
        return tuple(normalize_index(index, size) for index in selector)
    if isinstance(selector, (int, np.integer)):
        return normalize_index(selector, size)
    raise TypeError(f"Unsupported selector type {type(selector).__name__}; expected int, slice, list, or tuple")


def selector_positions(selector: Any, size: int) -> tuple[int, ...]:
    """Enumerate concrete positions selected on one source axis."""
    if isinstance(selector, (int, np.integer)):
        return (int(selector),)
    if isinstance(selector, slice):
        return tuple(range(*selector.indices(size)))
    if isinstance(selector, list):
        return tuple(selector)
    if isinstance(selector, tuple):
        return selector
    raise TypeError(f"Cannot enumerate positions for selector type {type(selector).__name__}")


def selector_length(selector: Any, size: int) -> int:
    """Return the number of concrete positions selected on one axis."""
    return len(selector_positions(selector, size))


def selector_to_compact(values: tuple[int, ...]) -> Any:
    """Return a compact selector representing concrete positions."""
    if len(values) == 1:
        return slice(values[0], values[0] + 1, 1)
    if not values:
        return slice(0, 0, 1)
    step = values[1] - values[0]
    if all((right - left) == step for left, right in zip(values, values[1:])):
        return slice(values[0], values[-1] + step, step)
    return tuple(values)


def selector_to_metadata_slice(selector: Any, size: int) -> slice:
    """Represent a selector as slice metadata for Data objects."""
    if isinstance(selector, (int, np.integer)):
        selector = int(selector)
        return slice(selector, selector + 1, 1)
    if isinstance(selector, slice):
        rng = range(*selector.indices(size))
        return slice(rng.start, rng.stop, rng.step)

    positions = selector_positions(selector, size)
    compact = selector_to_compact(positions)
    if isinstance(compact, slice):
        return compact

    # Data metadata is slice-based; fall back to relative coordinates for
    # irregular explicit selections.
    return slice(0, len(positions), 1)


def compose_selector(base_selector: Any, sub_selector: Any, size: int) -> Any:
    """Compose a selector with a second selector applied to its result."""
    values = selector_positions(base_selector, size)
    if isinstance(sub_selector, (int, np.integer)):
        return values[normalize_index(int(sub_selector), len(values))]
    if isinstance(sub_selector, slice):
        return selector_to_compact(values[sub_selector])
    if isinstance(sub_selector, list):
        return tuple(values[normalize_index(index, len(values))] for index in sub_selector)
    if isinstance(sub_selector, tuple):
        return tuple(values[normalize_index(index, len(values))] for index in sub_selector)
    raise TypeError(f"Unsupported selector type {type(sub_selector).__name__}; expected int, slice, list, or tuple")


def normalize_item(item: Any, shape: tuple[int, ...], *, target: str = "data") -> tuple[Any, ...]:
    """Normalize an index expression against a shape."""
    if not isinstance(item, tuple):
        item = (item,)
    if item.count(Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if Ellipsis in item:
        idx = item.index(Ellipsis)
        fill = (slice(None),) * (len(shape) - (len(item) - 1))
        item = item[:idx] + fill + item[idx + 1 :]
    if len(item) > len(shape):
        raise IndexError(f"too many indices for {len(shape)}-dimensional {target}")
    item = item + (slice(None),) * (len(shape) - len(item))
    return tuple(normalize_selector(selector, size) for selector, size in zip(item, shape))
