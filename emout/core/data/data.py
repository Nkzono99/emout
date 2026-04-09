"""Dimensioned ndarray subclasses for EMSES grid data.

Re-exports all Data classes from their respective submodules for
backward compatibility.  New code should import from the submodules
directly (e.g. ``from emout.core.data._data3d import Data3d``).
"""

from ._base import Data, _REMOTE_PLOT_HANDLED
from ._data1d import Data1d
from ._data2d import Data2d
from ._data3d import Data3d
from ._data4d import Data4d

__all__ = ["Data", "Data1d", "Data2d", "Data3d", "Data4d", "_REMOTE_PLOT_HANDLED"]
