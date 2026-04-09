"""emout -- Python interface for EMSES simulation output.

Provides :class:`Emout` as the main entry point for reading grid data,
particle data, input parameters, and boundary definitions from EMSES
HDF5 and namelist files.

Quick start::

    import emout
    data = emout.Emout("/path/to/output")
    data.phisp[-1].plot()
"""

from .core import data
from .core.data import VectorData, VectorData2d, VectorData3d
from .core.facade import Emout
from .utils.emsesinp import InpFile, UnitConversionKey
from .utils.units import Units
