"""Grid data loading from EMSES HDF5 output files.

:class:`GridDataLoader` resolves field names (including relocated fields
and vector components) and returns :class:`~emout.core.data.data.GridDataSeries`
or :class:`~emout.core.data.vector_data.VectorData` instances.
"""

# emout/io/grid.py

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm

from ..data.griddata_series import GridDataSeries
from ..data.vector_data import VectorData2d, VectorData3d
from ..relocation.electric import relocated_electric_field
from ..relocation.magnetic import relocated_magnetic_field
from .directory import DirectoryInspector

logger = logging.getLogger(__name__)


def get_tqdm():
    """Return the appropriate tqdm variant for the current environment."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            logger.debug("Notebook environment detected")
            return notebook_tqdm
        else:
            logger.debug("IPython environment (non-notebook) detected")
            return tqdm
    except NameError:
        logger.debug("Standard Python environment detected")
        return tqdm


tqdm = get_tqdm()


class GridDataLoader:
    """Resolve EMSES field names and load grid data from HDF5 files.

    Supports plain scalars, relocated electric/magnetic fields (``re*``,
    ``rb*``), and multi-axis vector fields (``{name}xy``, ``{name}xyz``).
    """

    def __init__(
        self, dir_inspector: DirectoryInspector, name2unit_map: dict[str, Any]
    ):
        """Initialize the loader.

        Parameters
        ----------
        dir_inspector : DirectoryInspector
            Inspector that provides directory and parameter information.
        name2unit_map : dict[str, Any]
            Mapping from data names to unit translator factories.
        """
        self.dir_inspector = dir_inspector
        self.name2unit_map = name2unit_map

    def load(self, name: str) -> Any:
        """Load grid data by name, handling relocated and vector fields.

        - ``r[e/b][xyz]`` -- generate a relocated field.
        - ``(dname)(axis1)(axis2)(axis3)`` (xyz permutation) -- return VectorData3d.
        - ``(dname)(axis1)(axis2)`` -- return VectorData2d.
        - Otherwise -- return a chained GridDataSeries.
        """
        logger.debug(f"GridDataLoader.load: {name}")

        m = re.match(r"^r([eb][xyz])$", name)
        if m:
            fld = m.group(1)  # e.g. 'ex', 'by'
            logger.debug(f"Relocated field requested: {fld}")
            self._create_relocated_field_hdf5(fld)

        skip_2d_vector_parse = False
        m3 = re.match(r"^(.+?)([xyz])([xyz])([xyz])$", name)
        if m3:
            dname, axis1, axis2, axis3 = m3.groups()
            axes = (axis1, axis2, axis3)
            if len(set(axes)) == 3:
                skip_2d_vector_parse = True
                logger.debug(
                    f"Building VectorData3d: base={dname}, axes=({axis1},{axis2},{axis3})"
                )
                try:
                    arr1 = self.load(f"{dname}{axis1}")
                    arr2 = self.load(f"{dname}{axis2}")
                    arr3 = self.load(f"{dname}{axis3}")
                    vd = VectorData3d([arr1, arr2, arr3], name=name)
                    return vd
                except (KeyError, FileNotFoundError, OSError):
                    logger.debug(
                        "VectorData3d resolution failed; falling back to plain GridDataSeries.",
                        exc_info=True,
                    )

        if not skip_2d_vector_parse:
            m2 = re.match(r"(.+)([xyz])([xyz])$", name)
            if m2:
                dname, axis1, axis2 = m2.groups()
                logger.debug(
                    f"Building VectorData2d: base={dname}, axes=({axis1},{axis2})"
                )
                arr1 = self.load(
                    f"{dname}{axis1}"
                )  # recursively loads GridDataSeries or relocated field
                arr2 = self.load(f"{dname}{axis2}")
                vd = VectorData2d([arr1, arr2], name=name)
                return vd

        main_fp = self._find_h5file(self.dir_inspector.main_directory, name)
        logger.info(f"Loading grid data from: {main_fp.resolve()}")
        gd = self._load_griddata(main_fp)

        for ad in self.dir_inspector.append_directories:
            try:
                fp_append = self._find_h5file(ad, name)
            except FileNotFoundError:
                continue
            gd_append = self._load_griddata(fp_append)
            gd = gd.chain(gd_append)

        return gd

    def _find_h5file(self, directory: Path, name: str) -> Path:
        """Find exactly one HDF5 file for the given data name.

        Parameters
        ----------
        directory : Path
            Directory to search
        name : str
            Field name

        Returns
        -------
        Path
            Path to the matching HDF5 file.
        """
        pattern = f"{name}00_0000.h5"
        matches = list(directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"{pattern} not found in: {directory}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple matches for {pattern} in: {directory}")
        return matches[0]

    def _load_griddata(self, h5file_path: Path) -> GridDataSeries:
        # Pick unit translators from name2unit when available
        """Build a GridDataSeries from an HDF5 file.

        Parameters
        ----------
        h5file_path : Path
            Path to the ``{name}00_0000.h5`` file

        Returns
        -------
        GridDataSeries
            Grid time-series data initialised with unit information.
        """
        unit = self.dir_inspector.unit
        if unit is None:
            tunit = None
            axisunit = None
            valunit = None
        else:
            # "t" and "axis" are always expected to have keys
            tunit = self.name2unit_map.get("t", lambda out: None)(self.dir_inspector)
            axisunit = self.name2unit_map.get("axis", lambda out: None)(
                self.dir_inspector
            )
            # Actual value unit is extracted from the file name
            base_name = h5file_path.name.replace("00_0000.h5", "")
            valunit = self.name2unit_map.get(base_name, lambda out: None)(
                self.dir_inspector
            )

        series = GridDataSeries(
            h5file_path,
            h5file_path.name.replace("00_0000.h5", ""),
            tunit=tunit,
            axisunit=axisunit,
            valunit=valunit,
        )
        series._emout_dir = str(self.dir_inspector.main_directory)
        series._emout_open_kwargs = {
            "directory": str(self.dir_inspector._input_directory),
            "append_directories": [str(path) for path in self.dir_inspector.append_directories],
            "inpfilename": self.dir_inspector.inpfilename,
            "output_directory": str(self.dir_inspector.main_directory),
        }
        if self.dir_inspector.input_path is not None:
            series._emout_open_kwargs["input_path"] = str(self.dir_inspector.input_path)
        return series

    def _create_relocated_field_hdf5(self, field_name: str) -> None:
        """Generate relocated field HDF5 files in all target directories.

        Parameters
        ----------
        field_name : str
            Original field name (e.g. ``"ex"``, ``"by"``).
        """
        axis = "zyx".index(field_name[-1])

        main_dir = self.dir_inspector.main_directory
        self._create_one_relocated(main_dir, field_name, axis)

        for ad in self.dir_inspector.append_directories:
            self._create_one_relocated(ad, field_name, axis)

    def _create_one_relocated(
        self, directory: Path, name: str, axis: int
    ) -> None:
        """Generate a relocated HDF5 file in a single directory.

        Parameters
        ----------
        directory : Path
            Target directory.
        name : str
            Field name.
        axis : int
            Interpolation axis index (``x=2``, ``y=1``, ``z=0``).
        """
        input_fp = directory / f"{name}00_0000.h5"
        output_fp = directory / f"r{name}00_0000.h5"

        if output_fp.exists():
            logger.debug(f"Already exists: {output_fp.resolve()}")
            return

        logger.info(f"Creating relocated field: {output_fp.resolve()}")
        with h5py.File(input_fp, "r") as h5f_in:
            field = h5f_in[name]

            with h5py.File(output_fp, "w") as h5f_out:
                rgrp = h5f_out.create_group(f"r{name}")

                for key in tqdm(field.keys(), desc=f"Relocating {name}"):
                    arr = np.array(field[key])

                    if name.startswith("b"):
                        if name[-1] == "x":
                            axs = "yz"
                        elif name[-1] == "y":
                            axs = "zx"
                        else:
                            axs = "xy"
                        btypes = [self._get_btype(ax) for ax in axs]

                        rgrp[key] = relocated_magnetic_field(arr, axis=axis, btypes=btypes)
                    else:
                        rgrp[key] = relocated_electric_field(
                            arr, axis=axis, btype=self._get_btype(name)
                        )

    def _get_btype(self, name: str) -> str:
        """Return the boundary condition code for the given field.

        Parameters
        ----------
        name : str
            Field name.

        Returns
        -------
        str
            Boundary condition name (``"periodic"``, ``"dirichlet"``, or
            ``"neumann"``).
        """
        axis = "zyx".index(name[-1])
        btype_list = ["periodic", "dirichlet", "neumann"]

        return btype_list[self.dir_inspector.inp.mtd_vbnd[2 - axis]]
