"""Time-series loaders for particle HDF5 output.

:class:`ParticleDataSeries` lazily loads per-component particle data
across timesteps, while :class:`MultiParticleDataSeries` (aliased as
``ParticlesSeries``) aggregates multiple components for a given species.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np

import emout.utils as utils
from emout.utils import DataFileInfo, UnitTranslator

from .particle_data import ParticleData


# ============================================================
# 1) One-file-per-component time series
# ============================================================
IndexLike = Union[int, slice, List[int]]
GetItemType = Union[int, slice, List[int], Tuple[IndexLike]]


class ParticleDataSeries:
    """Time-series data for a single particle component (x, vx, tid, etc.).

    Manages one HDF5 file with the expected structure:

    - Top-level group: ``self.h5[list(self.h5.keys())[0]]``
    - Time-step keys underneath (e.g. ``"0"``, ``"1"``, ``"2"``, ...)
    - Each key holds a dataset of shape ``(nparticle,)``
    """

    def __init__(
        self,
        filename: PathLike,
        name: str,
        tunit: UnitTranslator = None,
        valunit: UnitTranslator = None,
    ):
        """Initialize the series from a single HDF5 file.

        Parameters
        ----------
        filename : path-like
            Path to the HDF5 file
        name : str
            Component name (e.g. ``'x'``, ``'vx'``, ``'tid'``)
        tunit : UnitTranslator, optional
            Time-axis unit translator
        valunit : UnitTranslator, optional
            Value unit translator
        """
        self.datafile = DataFileInfo(filename)
        self.h5 = h5py.File(str(filename), "r")
        self.group = self.h5[list(self.h5.keys())[0]]

        # Assumes time-step keys are integer strings
        self._index2key = {int(key[-4:]): key for key in self.group.keys()}

        self.tunit = tunit
        self.valunit = valunit
        self.name = name

    def __repr__(self) -> str:
        return f"<ParticleDataSeries: name={self.name!r}, timesteps={len(self)}, file={self.filename.name}>"

    def close(self) -> None:
        """Close the underlying HDF5 file handle."""
        self.h5.close()

    @property
    def filename(self) -> Path:
        """Return the source HDF5 file path.

        Returns
        -------
        Path
            Source file path.
        """
        return self.datafile.filename

    @property
    def directory(self) -> Path:
        """Return the parent directory of the source file.

        Returns
        -------
        Path
            Parent directory path.
        """
        return self.datafile.directory

    def _sorted_timekeys(self) -> List[int]:
        """Return time-step keys sorted in ascending order.

        Returns
        -------
        List[int]
            Sorted time-step keys.
        """
        return sorted(self._index2key.keys())

    def _create_data_with_timekey(self, timekey: int) -> ParticleData:
        """Create a ParticleData for the given time-step key.

        Parameters
        ----------
        timekey : int
            Time-step key value

        Returns
        -------
        ParticleData
            Particle data at the specified time step.
        """
        if timekey not in self._index2key:
            raise IndexError(timekey)
        key = self._index2key[timekey]
        arr = np.array(self.group[key])  # (nparticle,)
        return ParticleData(arr, name=self.name, valunit=self.valunit)

    def __getitem__(self, item: Union[int, slice, List[int], Tuple[IndexLike]]):
        # Tuple indexing is not supported (kept for design compatibility)
        """Retrieve particle data by time index.

        Parameters
        ----------
        item : int, slice, or list of int
            Time-step index expression

        Returns
        -------
        ParticleData or list of ParticleData
            Data for the requested time step(s).
        """
        if isinstance(item, tuple):
            raise IndexError("Tuple indexing is not supported for ParticleDataSeries.")

        timekeys = self._sorted_timekeys()

        def pick_one(pos: int) -> ParticleData:
            """Return the ParticleData at the given position.

            Parameters
            ----------
            pos : int
                Time-series position index (negative indexing supported)

            Returns
            -------
            ParticleData
                Particle data at the specified position.
            """
            if pos < 0:
                pos = len(timekeys) + pos
            if pos < 0 or pos >= len(timekeys):
                raise IndexError(pos)
            return self._create_data_with_timekey(timekeys[pos])

        if isinstance(item, int):
            return pick_one(item)

        if isinstance(item, slice):
            positions = list(utils.range_with_slice(item, maxlen=len(timekeys)))
            return [pick_one(p) for p in positions]

        if isinstance(item, list):
            return [pick_one(p) for p in item]

        raise TypeError(type(item))

    def __iter__(self) -> Iterable[ParticleData]:
        """Iterate over all time steps.

        Returns
        -------
        Iterable[ParticleData]
            Iterator yielding ParticleData for each time step.
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """Return the number of time steps.

        Returns
        -------
        int
            Number of time steps.
        """
        return len(self._index2key)

    def chain(self, other: "ParticleDataSeries") -> "MultiParticleDataSeries":
        """Concatenate with another series and return a combined series.

        Parameters
        ----------
        other : ParticleDataSeries
            Series to concatenate

        Returns
        -------
        MultiParticleDataSeries
            Combined series.
        """
        return MultiParticleDataSeries(self, other)

    def __add__(self, other: "ParticleDataSeries") -> "MultiParticleDataSeries":
        """Concatenate two series using the ``+`` operator.

        Parameters
        ----------
        other : ParticleDataSeries
            Series to concatenate

        Returns
        -------
        MultiParticleDataSeries
            Combined series.
        """
        if not isinstance(other, ParticleDataSeries):
            raise TypeError(f"Cannot chain ParticleDataSeries with {type(other).__name__}")
        return self.chain(other)


class MultiParticleDataSeries(ParticleDataSeries):
    """Concatenation of multiple ParticleDataSeries.

    When ``drop_head_of_later=True``, the first time step of each
    subsequent series is assumed to duplicate the last step of the
    preceding series and is therefore skipped.
    """

    def __init__(self, *series: ParticleDataSeries, drop_head_of_later: bool = True):
        """Initialize from one or more particle data series.

        Parameters
        ----------
        *series : ParticleDataSeries
            Series to concatenate
        drop_head_of_later : bool, optional
            If True, drop the first time step of each series after the
            first, treating it as a duplicate of the previous series' last
            step.
        """
        self.series: List[ParticleDataSeries] = []
        for s in series:
            self.series += self._expand(s)

        if not self.series:
            raise ValueError("No series were provided.")

        self.datafile = self.series[0].datafile
        self.tunit = self.series[0].tunit
        self.valunit = self.series[0].valunit
        self.name = self.series[0].name

        self.drop_head_of_later = drop_head_of_later

    def _expand(self, s: Union["ParticleDataSeries", "MultiParticleDataSeries"]) -> List[ParticleDataSeries]:
        """Flatten a (possibly nested) series into a list of leaf series.

        Parameters
        ----------
        s : ParticleDataSeries or MultiParticleDataSeries
            Series to expand

        Returns
        -------
        List[ParticleDataSeries]
            Flattened list of leaf ParticleDataSeries objects.
        """
        if not isinstance(s, ParticleDataSeries):
            raise TypeError(f"Expected ParticleDataSeries, got {type(s).__name__}")
        if not isinstance(s, MultiParticleDataSeries):
            return [s]
        out: List[ParticleDataSeries] = []
        for ss in s.series:
            out += self._expand(ss)
        return out

    def close(self) -> None:
        """Close all underlying HDF5 file handles."""
        for s in self.series:
            s.close()

    def __len__(self) -> int:
        """Return the total number of time steps across all series.

        Returns
        -------
        int
            Total number of time steps.
        """
        if not self.drop_head_of_later:
            return int(np.sum([len(s) for s in self.series]))
        return len(self.series[0]) + int(np.sum([max(0, len(s) - 1) for s in self.series[1:]]))

    def _locate(self, index: int) -> Tuple[ParticleDataSeries, int]:
        """Map a global index to its owning series and local index.

        Parameters
        ----------
        index : int
            Global time-step index (negative indexing supported)

        Returns
        -------
        Tuple[ParticleDataSeries, int]
            ``(series, local_index)`` pair.
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)

        n0 = len(self.series[0])
        if index < n0:
            return self.series[0], index

        offset = n0
        for s in self.series[1:]:
            if self.drop_head_of_later:
                usable = max(0, len(s) - 1)
                if index < offset + usable:
                    local = (index - offset) + 1  # skip the first entry
                    return s, local
                offset += usable
            else:
                usable = len(s)
                if index < offset + usable:
                    local = index - offset
                    return s, local
                offset += usable

        raise IndexError(index)

    def __getitem__(self, item):
        """Retrieve particle data by time index.

        Parameters
        ----------
        item : int, slice, or list of int
            Time-step index expression

        Returns
        -------
        ParticleData or list of ParticleData
            Data for the requested time step(s).
        """
        if isinstance(item, tuple):
            raise IndexError("Tuple indexing is not supported for MultiParticleDataSeries.")

        if isinstance(item, int):
            s, local = self._locate(item)
            return s[local]

        if isinstance(item, slice):
            positions = list(utils.range_with_slice(item, maxlen=len(self)))
            return [self[i] for i in positions]

        if isinstance(item, list):
            return [self[i] for i in item]

        raise TypeError(type(item))

    def __iter__(self) -> Iterable[ParticleData]:
        """Iterate over all time steps.

        Returns
        -------
        Iterable[ParticleData]
            Iterator yielding ParticleData for each time step.
        """
        for i in range(len(self)):
            yield self[i]


# ============================================================
# 2) Snapshot (fixed t, bundling x, y, z, vx, vy, vz, tid)
# ============================================================
@dataclass(slots=True)
class ParticleSnapshot:
    """Single-timestep snapshot bundling all particle components."""

    fields: Dict[str, ParticleData]

    def __repr__(self) -> str:
        comps = list(self.fields.keys())
        n = len(next(iter(self.fields.values()))) if self.fields else 0
        return f"<ParticleSnapshot: components={comps}, particles={n}>"

    def __getattr__(self, name: str):
        """Resolve attribute access to a particle component or phase-space pair.

        Single component names (``'x'``, ``'vx'``, ``'tid'``) return the
        corresponding :class:`ParticleData`.  Two-key concatenations such
        as ``'xvx'`` or ``'yvz'`` return a callable that delegates to
        :meth:`plot_phase_space`.

        Parameters
        ----------
        name : str
            Component name or two-component shorthand

        Returns
        -------
        ParticleData or functools.partial
        """
        # Direct component lookup
        if name in self.fields:
            return self.fields[name]

        # Phase-space shorthand: e.g. "xvx" -> plot_phase_space("x", "vx")
        from functools import partial

        for key1 in self._PHASE_KEYS:
            if name.startswith(key1):
                rest = name[len(key1) :]
                if rest in self._PHASE_KEYS and rest != key1:
                    return partial(self.plot_phase_space, key1, rest)

        raise AttributeError(
            f"'{type(self).__name__}' has no component or pair '{name}'. Available: {list(self.fields.keys())}"
        )

    def keys(self) -> Iterable[str]:
        """Return the available component names.

        Returns
        -------
        Iterable[str]
            Component name keys.
        """
        return self.fields.keys()

    def as_dict(self) -> Dict[str, ParticleData]:
        """Return the snapshot contents as a plain dictionary.

        Returns
        -------
        Dict[str, ParticleData]
            Mapping from component name to ParticleData.
        """
        return dict(self.fields)

    def to_dataframe(self, use_si: bool = False):
        """Convert the snapshot to a :class:`pandas.DataFrame`.

        Parameters
        ----------
        use_si : bool, default False
            If True, convert values to SI units.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one column per loaded component.
        """
        import pandas as pd

        cols = {}
        for name, pdata in self.fields.items():
            if use_si and pdata.valunit is not None:
                cols[name] = pdata.val_si.values
            else:
                cols[name] = pdata.values
        return pd.DataFrame(cols)

    # ---- Phase-space plotting ----

    _PHASE_KEYS = ("x", "y", "z", "vx", "vy", "vz")

    def plot_phase_space(
        self,
        var1: str,
        var2: str,
        kind: str = "scatter",
        ax=None,
        use_si: bool = True,
        bins: int = 64,
        **kwargs,
    ):
        """Plot a 2-D phase-space diagram.

        Parameters
        ----------
        var1 : str
            Horizontal axis variable (``'x'``, ``'y'``, ``'z'``,
            ``'vx'``, ``'vy'``, ``'vz'``).
        var2 : str
            Vertical axis variable.
        kind : {'scatter', 'hist2d'}
            Plot type.
        ax : matplotlib.axes.Axes, optional
            Target axes. If ``None``, uses the current axes.
        use_si : bool, default True
            Convert to SI units when unit metadata is available.
        bins : int, default 64
            Number of bins per axis (only used when ``kind='hist2d'``).
        **kwargs
            Forwarded to :func:`matplotlib.axes.Axes.scatter` or
            :func:`matplotlib.axes.Axes.hist2d`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        for v in (var1, var2):
            if v not in self._PHASE_KEYS:
                raise KeyError(f"Unknown variable {v!r}; choose from {self._PHASE_KEYS}")
            if v not in self.fields:
                raise KeyError(f"Component {v!r} is not loaded in this snapshot. Available: {list(self.fields.keys())}")

        d1 = self.fields[var1]
        d2 = self.fields[var2]

        vals1 = d1.val_si.values if (use_si and d1.valunit) else d1.values
        vals2 = d2.val_si.values if (use_si and d2.valunit) else d2.values

        # Filter NaN particles
        mask = np.isfinite(vals1) & np.isfinite(vals2)
        vals1 = vals1[mask]
        vals2 = vals2[mask]

        if ax is None:
            ax = plt.gca()

        xlabel = f"{var1} [{d1.valunit.unit}]" if (use_si and d1.valunit) else var1
        ylabel = f"{var2} [{d2.valunit.unit}]" if (use_si and d2.valunit) else var2

        if kind == "scatter":
            kwargs.setdefault("s", 0.3)
            kwargs.setdefault("alpha", 0.5)
            ax.scatter(vals1, vals2, **kwargs)
        elif kind == "hist2d":
            ax.hist2d(vals1, vals2, bins=bins, **kwargs)
        else:
            raise ValueError(f"Unsupported kind {kind!r}; use 'scatter' or 'hist2d'")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{var1} vs {var2}")
        return ax


# ============================================================
# 3) Aggregator: scan directory and build per-component series
# ============================================================
class ParticlesSeries:
    """Aggregate per-component particle HDF5 files for a single species.

    Usage::

        p = ParticlesSeries(dir, species=1)
        p.x, p.vx, p.tid  # -> ParticleDataSeries or MultiParticleDataSeries
        p[t]               # -> ParticleSnapshot (x, y, z, vx, vy, vz, tid)

    Typical file names::

        p1xe00_0000.h5
        p1vxe00_0000.h5
        p1tid00_0000.h5
    """

    _DEFAULT_COMPONENTS = ("x", "y", "z", "vx", "vy", "vz", "tid")

    # Handle both cases: with and without "e" after the component name
    _FILE_RE = re.compile(
        r"^p(?P<sp>\d+)"
        r"(?P<comp>tid|vx|vy|vz|x|y|z)"
        r"(?:e)?(?P<seg>\d+)"
        r"_(?P<part>\d+)\.h5$"
    )

    def __init__(
        self,
        directory: PathLike,
        species: int,
        *,
        components: Tuple[str, ...] = _DEFAULT_COMPONENTS,
        tunit: UnitTranslator = None,
        vunits: Optional[Dict[str, UnitTranslator]] = None,
        drop_head_of_later: bool = True,
        strict_length: bool = True,
    ):
        """Initialize from a directory and species identifier.

        Parameters
        ----------
        directory : path-like
            Directory containing particle HDF5 files
        species : int
            Particle species number (1-based)
        components : tuple of str, optional
            Component names to load
        tunit : UnitTranslator, optional
            Time-axis unit translator
        vunits : dict of {str: UnitTranslator}, optional
            Per-component unit translator mapping
        drop_head_of_later : bool, optional
            If True, drop the first step of subsequent file series as a
            duplicate
        strict_length : bool, optional
            If True, verify that all component series have equal length
        """
        self.directory = Path(directory)
        self.species = int(species)
        self.components = tuple(components)
        self.tunit = tunit
        self.vunits = vunits or {}
        self.drop_head_of_later = drop_head_of_later
        self.strict_length = strict_length

        # comp -> ParticleDataSeries or MultiParticleDataSeries
        self._series: Dict[str, Union[ParticleDataSeries, MultiParticleDataSeries]] = {}
        self._build()

        if self.strict_length:
            self._validate_lengths()

    def _build(self) -> None:
        """Scan the directory and build per-component data series."""
        buckets: Dict[str, List[Tuple[Tuple[int, int], Path]]] = {c: [] for c in self.components}

        for fp in self.directory.glob("*.h5"):
            m = self._FILE_RE.match(fp.name)
            if not m:
                continue
            sp = int(m.group("sp"))
            if sp != self.species:
                continue

            comp = m.group("comp")
            if comp not in buckets:
                continue

            seg = int(m.group("seg"))
            part = int(m.group("part"))
            buckets[comp].append(((seg, part), fp))

        for comp, items in buckets.items():
            if not items:
                continue

            items.sort(key=lambda x: x[0])  # (seg, part)
            paths = [p for _, p in items]

            vunit = self.vunits.get(comp)
            series_list = [ParticleDataSeries(p, name=comp, tunit=self.tunit, valunit=vunit) for p in paths]

            if len(series_list) == 1:
                self._series[comp] = series_list[0]
            else:
                self._series[comp] = MultiParticleDataSeries(*series_list, drop_head_of_later=self.drop_head_of_later)

    def _validate_lengths(self) -> None:
        """Verify that all component series have equal length."""
        lengths = {k: len(v) for k, v in self._series.items()}
        if not lengths:
            raise ValueError("No particle component series were found.")
        uniq = sorted(set(lengths.values()))
        if len(uniq) > 1:
            raise ValueError(f"Component lengths are inconsistent: {lengths}")

    def close(self) -> None:
        # MultiParticleDataSeries.close() handles its own sub-series, but
        # we close each entry explicitly just in case.
        """Close all underlying HDF5 file handles."""
        for s in self._series.values():
            if isinstance(s, MultiParticleDataSeries):
                s.close()
            else:
                s.close()

    def __enter__(self) -> "ParticlesSeries":
        """Enter the context manager.

        Returns
        -------
        ParticlesSeries
            This instance.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the context manager and close resources.

        Parameters
        ----------
        exc_type : type or None
            Exception type, if any
        exc : BaseException or None
            Exception instance, if any
        tb : traceback or None
            Traceback object, if any
        """
        self.close()

    def __repr__(self) -> str:
        comps = self.available_components()
        return f"<ParticlesSeries: species={self.species}, components={list(comps)}, timesteps={len(self)}>"

    def __getattr__(self, name: str):
        # p.x / p.vx / p.tid returns the corresponding series
        """Resolve attribute access to a component series.

        Parameters
        ----------
        name : str
            Component name (e.g. ``'x'``, ``'vx'``, ``'tid'``)

        Returns
        -------
        ParticleDataSeries or MultiParticleDataSeries
            Data series for the requested component.
        """
        if name in self._series:
            return self._series[name]
        raise AttributeError(name)

    def available_components(self) -> Tuple[str, ...]:
        """Return the names of loaded particle components.

        Returns
        -------
        tuple of str
            Available component names.
        """
        return tuple(self._series.keys())

    def __len__(self) -> int:
        # Use a representative component (tid preferred)
        """Return the number of time steps.

        Returns
        -------
        int
            Number of time steps.
        """
        for key in ("tid", "x", "vx"):
            if key in self._series:
                return len(self._series[key])
        # fallback
        any_key = next(iter(self._series))
        return len(self._series[any_key])

    def __getitem__(self, tindex: int) -> ParticleSnapshot:
        """Return a snapshot bundling all components at a given time step.

        Parameters
        ----------
        tindex : int
            Time-step index

        Returns
        -------
        ParticleSnapshot
            Snapshot containing all loaded particle components.
        """
        fields: Dict[str, ParticleData] = {}
        for comp, s in self._series.items():
            fields[comp] = s[tindex]
        return ParticleSnapshot(fields)

    def __iter__(self) -> Iterable[ParticleSnapshot]:
        """Iterate over all time steps, yielding snapshots.

        Returns
        -------
        Iterable[ParticleSnapshot]
            Iterator yielding ParticleSnapshot for each time step.
        """
        for i in range(len(self)):
            yield self[i]
