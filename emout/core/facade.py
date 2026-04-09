"""Facade class that provides unified access to EMSES simulation outputs.

The :class:`Emout` class aggregates directory inspection, grid data loading,
particle data, boundary modelling, and backtrace analysis into a single
entry point.
"""

import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

from emout.utils import InpFile, Units

from .backtrace.solver_wrapper import BacktraceWrapper
from .data.griddata_series import GridDataSeries
from .data.vector_data import VectorData2d, VectorData3d
from .io.directory import DirectoryInspector
from .io.grid import GridDataLoader
from .units import build_name2unit_mapping

from .data.particle_data_series import ParticlesSeries

logger = logging.getLogger(__name__)


class Emout:
    """Facade class for unified access to EMSES simulation outputs."""

    # name-to-unit mapping (built once)
    name2unit = build_name2unit_mapping(max_ndp=9)

    def __init__(
        self,
        directory: Union[Path, str] = "./",
        append_directories: Union[List[Union[Path, str]], None] = None,
        ad: Union[List[Union[Path, str]], None] = None,
        inpfilename: Union[Path, str] = "plasma.inp",
        input_path: Union[Path, str, None] = None,
        output_directory: Union[Path, str, None] = None,
    ):
        """Initialize the Emout facade.

        Parameters
        ----------
        directory : Path or str, optional
            Base directory. When *input_path* and *output_directory* are
            not given, both input and output files are looked up here.
        append_directories : list of (Path or str) or None, optional
            Additional directories to chain for multi-segment runs.
        ad : list of (Path or str) or None, optional
            Short alias for *append_directories*. When both are given,
            *append_directories* takes precedence.
        inpfilename : Path or str, optional
            Input parameter file name. Ignored when *input_path* is set.
        input_path : Path or str or None, optional
            Full path to the input parameter file
            (e.g. ``/path/to/plasma.toml``).
        output_directory : Path or str or None, optional
            Directory containing simulation output files.
            Defaults to *directory* when not specified.
        """
        self._dir_inspector = DirectoryInspector(
            directory=directory,
            append_directories=append_directories or ad,
            inpfilename=inpfilename,
            input_path=input_path,
            output_directory=output_directory,
        )

        self._grid_loader = GridDataLoader(
            dir_inspector=self._dir_inspector,
            name2unit_map=Emout.name2unit,
        )

    @property
    def directory(self) -> Path:
        """Return the main output directory.

        Returns
        -------
        Path
            Base directory for EMSES output files.
        """
        return self._dir_inspector.main_directory

    @property
    def append_directories(self) -> List[Path]:
        """Return the list of chained append directories.

        Returns
        -------
        List[Path]
            Directories concatenated after the main directory during loading.
        """
        return self._dir_inspector.append_directories

    @property
    def inp(self) -> Union[InpFile, None]:
        """Return the parsed input parameter file.

        Returns
        -------
        InpFile or None
            Parsed ``plasma.inp`` parameters, or ``None`` if not loaded.
        """
        return self._dir_inspector.inp

    @property
    def toml(self):
        """Return the parsed TOML configuration.

        Only available when ``plasma.toml`` exists. Provides attribute
        access to the structured TOML data, e.g.
        ``data.toml.species[0].wp``.  Entries using ``group_id``
        (``*_groups``) are expanded at load time and the group tables
        are removed from the returned object.

        Returns
        -------
        TomlData or None
            Parsed TOML data, or ``None`` if unavailable.
        """
        return self._dir_inspector.toml

    @property
    def unit(self) -> Union[Units, None]:
        """Return the unit conversion object.

        Returns
        -------
        Units or None
            Unit translators derived from the conversion key, or ``None``
            if no conversion key is available.
        """
        return self._dir_inspector.unit

    def is_valid(self) -> bool:
        """Check whether the simulation completed successfully.

        Returns
        -------
        bool
            True if the last recorded step matches ``nstep`` in the input file.
        """
        return self._dir_inspector.is_valid()

    def available_fields(self) -> List[str]:
        """Return the names of available grid data fields.

        Scans the output directory for ``*00_0000.h5`` files and returns
        the field names (e.g. ``['ex', 'ey', 'ez', 'phisp']``).

        Returns
        -------
        list of str
            Sorted field names.
        """
        pattern = "*00_0000.h5"
        names = []
        for f in sorted(self._dir_inspector.main_directory.glob(pattern)):
            name = f.name.replace("00_0000.h5", "")
            if name and not name[0].isdigit():
                names.append(name)
        return names

    @property
    def icur(self) -> pd.DataFrame:
        """Return the ``icur`` diagnostic file as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Table with step numbers and per-species / per-body current
            columns parsed from the ``icur`` text file.
        """
        return self._dir_inspector.read_icur_as_dataframe()

    @property
    def pbody(self) -> pd.DataFrame:
        """Return the ``pbody`` diagnostic file as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Table with a ``step`` column and per-body particle-count
            columns parsed from the ``pbody`` text file.
        """
        return self._dir_inspector.read_pbody_as_dataframe()

    def particle(self, species: int):
        """Return a ParticlesSeries for the given species.

        Parameters
        ----------
        species : int
            Particle species number (1-based)

        Returns
        -------
        ParticlesSeries
            Particle data series for the specified species.
        """
        x_unit = self.unit.length
        v_unit = self.unit.v

        vunits = {
            "x": x_unit,
            "y": x_unit,
            "z": x_unit,
            "vx": v_unit,
            "vy": v_unit,
            "vz": v_unit,
        }

        return ParticlesSeries(self.directory, species=species, vunits=vunits)

    def __getattr__(self, name: str) -> Union[GridDataSeries, VectorData2d, VectorData3d]:
        """Dynamically resolve EMSES field names to data objects.

        The following name patterns are recognised:

        * ``p{N}`` -- particle species *N* (delegates to :meth:`particle`).
        * ``r[eb][xyz]`` -- relocated (cell-centred) electric / magnetic
          field component.
        * ``{base}{a1}{a2}{a3}`` (three distinct axis letters) --
          :class:`VectorData3d` combining three scalar components.
        * ``{base}{a1}{a2}`` -- :class:`VectorData2d` combining two
          scalar components.
        * anything else -- :class:`GridDataSeries` loaded from HDF5.

        Parameters
        ----------
        name : str
            Attribute name to resolve.

        Returns
        -------
        GridDataSeries or VectorData2d or VectorData3d
            The resolved data object.

        Raises
        ------
        AttributeError
            If *name* cannot be mapped to any known field or file.
        """
        import re

        m = re.match(r"^p([1-9])", name)
        if m:
            return self.particle(species=int(m.group(1)))

        try:
            return self._grid_loader.load(name)
        except (KeyError, FileNotFoundError, OSError) as e:
            raise AttributeError(f"Failed to load attribute '{name}': {e}") from e

    @property
    def boundaries(self):
        """Return the MPIEMSES finbound boundary collection.

        Builds :class:`~emout.core.boundaries.Boundary` instances from the
        ``boundary_types`` array in ``data.inp`` (when ``boundary_type =
        'complex'``) and returns them as a
        :class:`~emout.core.boundaries.BoundaryCollection`.

        Examples::

            data = Emout("output_dir")

            # Individual boundary mesh (grid units)
            mesh = data.boundaries[0].mesh()

            # SI units with resolution override
            mesh = data.boundaries[0].mesh(use_si=True, ntheta=96)

            # Composite mesh of all boundaries
            composite = data.boundaries.mesh(use_si=True)

            # Per-boundary keyword overrides
            composite = data.boundaries.mesh(
                use_si=True,
                per={0: {"ntheta": 64}, 1: {"nradial": 12}},
            )

        Returns
        -------
        BoundaryCollection
            Supported finbound boundaries. Returns an empty collection
            if ``data.inp`` is not loaded or is not in complex mode.
        """
        from .boundaries import BoundaryCollection

        return BoundaryCollection(
            self._dir_inspector.inp,
            self.unit,
            remote_open_kwargs=self._remote_open_kwargs,
        )

    @property
    def backtrace(self) -> BacktraceWrapper:
        """Return the backtrace solver wrapper.

        Returns
        -------
        BacktraceWrapper
            Backtrace API bound to this simulation's directory and input
            parameters.
        """
        return BacktraceWrapper(
            directory=self._dir_inspector.main_directory,
            inp=self._dir_inspector.inp,
            unit=self.unit,
            remote_open_kwargs=self._remote_open_kwargs,
        )

    @property
    def _emout_dir(self) -> str:
        return str(self._dir_inspector.main_directory)

    @property
    def _remote_open_kwargs(self) -> dict:
        kwargs = {
            "directory": str(self._dir_inspector._input_directory),
            "append_directories": [str(path) for path in self._dir_inspector.append_directories],
            "inpfilename": self._dir_inspector.inpfilename,
            "output_directory": str(self._dir_inspector.main_directory),
        }
        if self._dir_inspector.input_path is not None:
            kwargs["input_path"] = str(self._dir_inspector.input_path)
        return kwargs
