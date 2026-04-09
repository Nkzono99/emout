"""emout -- Python interface for EMSES simulation output.

Provides :class:`Emout` as the main entry point for reading grid data,
particle data, input parameters, and boundary definitions from EMSES
HDF5 and namelist files.

Quick start::

    import emout
    data = emout.Emout("/path/to/output")
    data.phisp[-1].plot()
"""

import logging as _logging

from .core import data
from .core.data import VectorData, VectorData2d, VectorData3d
from .core.facade import Emout
from .utils.emsesinp import InpFile, UnitConversionKey
from .utils.units import Units


def set_log_level(level: str = "WARNING") -> None:
    """Configure the log level for the entire ``emout`` package.

    Parameters
    ----------
    level : str, default ``"WARNING"``
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
        ``"CRITICAL"``.

    Examples
    --------
    >>> import emout
    >>> emout.set_log_level("DEBUG")   # verbose
    >>> emout.set_log_level("WARNING") # quiet (default)
    """
    logger = _logging.getLogger("emout")
    logger.setLevel(getattr(_logging, level.upper(), _logging.WARNING))
    if not logger.handlers:
        handler = _logging.StreamHandler()
        handler.setFormatter(
            _logging.Formatter("%(name)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
