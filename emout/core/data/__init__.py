"""Data classes for EMSES grid and particle output.

Re-exports
----------
Data, Data1d, Data2d, Data3d, Data4d
    Dimensioned numpy-subclass wrappers for grid data.
VectorData, VectorData2d, VectorData3d
    Multi-component vector field wrappers.
GridDataSeries
    Lazy time-series loader for grid HDF5 files.
ParticleData, ParticleDataSeries, MultiParticleDataSeries
    Particle output wrappers.
"""

from .data import Data, Data1d, Data2d, Data3d, Data4d
from .vector_data import VectorData, VectorData2d, VectorData3d
from .griddata_series import GridDataSeries
from .particle_data import ParticleData
from .particle_data_series import ParticleDataSeries, MultiParticleDataSeries
