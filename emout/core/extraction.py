"""Data extraction utilities for saving subsets of EMSES output.

Provides :class:`EmoutDataExtraction` which wraps an :class:`Emout` instance
and automatically persists accessed grid data as HDF5 snapshots.
"""

import logging
import re
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd

from .backtrace.solver_wrapper import BacktraceWrapper
from .data.griddata_series import GridDataSeries
from .data.vector_data import VectorData2d
from .facade import Emout

logger = logging.getLogger(__name__)


class EmoutDataExtraction:
    """Wrapper around :class:`Emout` that persists accessed data to disk.

    Every grid field accessed via attribute lookup is automatically saved
    as an HDF5 snapshot (last timestep only) into an extraction directory,
    enabling lightweight data subsets for post-processing or transfer.
    """
    def __init__(self, root: Union[Path, str], data: Emout, nparent=1):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        root : Union[Path, str]
            ファイルまたはディレクトリのパスです。
        data : Emout
            処理対象のデータ。
        nparent : int, optional
            再帰抽出時に遡る親階層数です。
        """
        self._root = Path(root)
        self._data = data
        self._nparent = nparent

        self.extract_dir.mkdir(parents=True, exist_ok=True)
        self._data.inp.save(
            self.extract_dir / "plasma.inp", convkey=self._data.inp.convkey
        )

    @property
    def directory(self):
        """Return the base simulation directory.

        Returns
        -------
        Path
            Root output directory of the wrapped :class:`Emout` instance.
        """
        return self._data.directory

    @property
    def inp(self):
        """Return the input parameter file.

        Returns
        -------
        InpFile or None
            Parsed ``plasma.inp`` parameters, or ``None`` if unavailable.
        """
        return self._data.inp

    @property
    def unit(self):
        """Return the unit conversion object.

        Returns
        -------
        Units or None
            Unit translators derived from the conversion key, or ``None``.
        """
        return self._data.unit

    def is_valid(self) -> bool:
        """データセットの妥当性を検証する。
        
        Returns
        -------
        bool
            条件判定結果です。
        """
        return self._data.is_valid()

    @property
    def icur(self) -> pd.DataFrame:
        """Return the ``icur`` diagnostic file as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Table with step numbers and per-species / per-body current
            columns parsed from the ``icur`` text file.
        """
        return self._data.icur

    @property
    def pbody(self) -> pd.DataFrame:
        """Return the ``pbody`` diagnostic file as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Table with a ``step`` column and per-body particle-count
            columns parsed from the ``pbody`` text file.
        """
        return self._data.pbody

    def __getattr__(self, name: str) -> Union[GridDataSeries, VectorData2d]:
        """Resolve a field name, persist the data, and return it.

        Delegates to the wrapped :class:`Emout` for name resolution, then
        saves the last-timestep snapshot as HDF5 into :attr:`extract_dir`.

        Parameters
        ----------
        name : str
            Attribute name to resolve (same patterns as :meth:`Emout.__getattr__`).

        Returns
        -------
        GridDataSeries or VectorData2d
            The resolved data object.
        """
        data = getattr(self._data, name)

        m2 = re.match(r"(.+)([xyz])([xyz])$", name)
        if m2:
            dname, axis1, axis2 = m2.groups()
            name1 = f"{dname}{axis1}"
            name2 = f"{dname}{axis2}"
            self.save_hdf5(name1)
            self.save_hdf5(name2)
        else:
            self.save_hdf5(name)

        return data

    def save_hdf5(self, name):
        """Save the last timestep of the named field to an HDF5 file.

        The file is written to :attr:`extract_dir` as
        ``{name}00_0000.h5``.  If the file already exists the write is
        skipped.

        Parameters
        ----------
        name : str
            Grid-data field name (e.g. ``"phisp"``, ``"ex"``).
        """
        path = self.extract_dir / f"{name}00_0000.h5"
        if path.exists():
            return
        with h5py.File(path, "w") as f:
            f.create_group(name)
            data = getattr(self._data, name)[-1, :, :, :]

            buf = np.empty_like(data)
            buf[:, :, :] = data[:, :, :]
            f[name].create_dataset("0000", data=buf)

    @property
    def backtrace(self) -> BacktraceWrapper:
        """Return the backtrace solver wrapper.

        Returns
        -------
        BacktraceWrapper
            Backtrace API bound to the underlying simulation data.
        """
        return self._data.backtrace

    @property
    def extract_dir(self):
        """Compute the extraction output directory.

        The path is derived from :attr:`directory` by taking the last
        *nparent* path components and appending them to *root*.

        Returns
        -------
        Path
            Directory where extracted HDF5 snapshots are written.
        """
        p = self.directory
        d = ""
        for _ in range(self._nparent):
            d = Path(p.name) if isinstance(d, str) else Path(f"{p.name}") / d
            p = p.parent

        if isinstance(d, Path):
            return self._root / d
        else:
            return self._root
