"""Data classes for EMSES grid and particle output.

Re-exports
----------
Data, Data1d, Data2d, Data3d, Data4d
    Dimensioned numpy-subclass wrappers for grid data.
VectorData, VectorData2d, VectorData3d
    Multi-component vector field wrappers.
ComponentValues
    Named samples and per-component results without grid assumptions.
GridDataSeries, GridDataSelection
    Lazy time-series loader for grid HDF5 files.
ParticleData, ParticleDataSeries, MultiParticleDataSeries
    Particle output wrappers.
"""

from .data import Data, Data1d, Data2d, Data3d, Data4d
from emout.local_data_policy import LocalDataAccessDisabledError
from .vector_data import VectorData, VectorData2d, VectorData3d
from .components import ComponentValues
from .griddata_series import GridDataSelection, GridDataSeries
from .particle_data import ParticleData
from .particle_data_series import ParticleDataSeries, MultiParticleDataSeries
