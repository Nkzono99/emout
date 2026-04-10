"""Directory inspection and input-file discovery for EMSES runs.

:class:`DirectoryInspector` locates ``plasma.inp`` / ``plasma.toml``,
resolves append-directory chains, and provides lazy access to parsed
input parameters and unit conversion objects.
"""

# emout/io/directory.py

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from emout.utils import InpFile
from emout.utils import UnitConversionKey, Units

logger = logging.getLogger(__name__)


class DirectoryInspector:
    """Directory discovery and input-file loading helper for Emout.

    Provides ``main_directory``, ``append_directories``, ``inp``
    (InpFile), and ``unit`` (Units) to the facade layer.
    """

    def __init__(
        self,
        directory: Union[Path, str],
        append_directories: Union[List[Union[Path, str]], str, None] = None,
        inpfilename: Union[Path, str] = "plasma.inp",
        input_path: Union[Path, str, None] = None,
        output_directory: Union[Path, str, None] = None,
    ):
        # 1. Convert directory to Path
        """Initialize the DirectoryInspector.

        Parameters
        ----------
        directory : Path or str
            Base directory. When *input_path* and *output_directory* are
            not given, both input and output files are looked up here.
        append_directories : list of (Path or str), str, or None, optional
            Additional directories.  ``'auto'`` triggers numbered-suffix
            auto-discovery.
        inpfilename : Path or str, optional
            Input parameter file name.  ``None`` skips ``.inp`` loading.
            Ignored when *input_path* is set.
        input_path : Path or str or None, optional
            Full path to the input parameter file (e.g.
            ``/path/to/plasma.toml``).  Overrides *directory* /
            *inpfilename* when given.
        output_directory : Path or str or None, optional
            Directory containing simulation output files (h5, icur, pbody,
            etc.).  Defaults to *directory*.
        """
        if not isinstance(directory, Path):
            directory = Path(directory)

        # When input_path is given, derive input directory and filename from it
        if input_path is not None:
            input_path = Path(input_path)
            self._input_directory: Path = input_path.parent
            inpfilename = input_path.name
            self.input_path: Optional[Path] = input_path.resolve()
        else:
            self._input_directory = directory
            self.input_path = None
        self._input_directory = self._input_directory.resolve()
        self.inpfilename = None if inpfilename is None else str(inpfilename)

        # Output directory (h5, icur, pbody, etc.)
        if output_directory is not None:
            self.main_directory: Path = Path(output_directory)
        else:
            self.main_directory = directory
        self.main_directory = self.main_directory.resolve()

        logger.info(
            f"DirectoryInspector: input directory = {self._input_directory.resolve()}, "
            f"output directory = {self.main_directory.resolve()}"
        )

        # 2. Determine append directories
        self.append_directories: List[Path] = []
        if append_directories == "auto":
            append_directories_list = self._fetch_append_directories(self.main_directory)
        else:
            append_directories_list = append_directories or []

        for ad in append_directories_list:
            p = Path(ad) if not isinstance(ad, Path) else ad
            self.append_directories.append(p.resolve())

        # 3. Load inp + initialise Units
        self._inp: Optional[InpFile] = None
        self._unit: Optional[Units] = None
        self._toml_data = None  # TomlData (set when plasma.toml exists)
        self._load_inpfile(inpfilename)

    def _fetch_append_directories(self, directory: Path) -> List[Path]:
        """Discover numbered-suffix append directories.

        Check ``<directory>_2``, ``<directory>_3``, ... in order and stop
        when a candidate does not exist or fails ``is_valid()``.

        Parameters
        ----------
        directory : Path
            Main directory used as the discovery root

        Returns
        -------
        List[Path]
            Valid append directories found.
        """
        logger.info(f"Fetching append directories for: {directory}")
        result: List[Path] = []
        directory = directory.resolve()
        i = 2
        while True:
            candidate = directory.parent / f"{directory.name}_{i}"
            if not candidate.exists():
                logger.debug(f"Append directory not found: {candidate}")
                break

            # Recursively call DirectoryInspector for validity check
            helper = DirectoryInspector(candidate, append_directories=None, inpfilename=None)
            if not helper.is_valid():
                logger.warning(f"{candidate.resolve()} exists but is not valid; stopping discovery")
                break

            result.append(candidate)
            i += 1
        return result

    def _load_inpfile(self, inpfilename: Union[Path, str]) -> None:
        """Load the parameter file and initialise unit conversion.

        When ``plasma.toml`` exists, ``toml2inp`` is invoked to
        generate/update ``plasma.inp`` before loading.

        Parameters
        ----------
        inpfilename : Path or str
            File name relative to *_input_directory*.  ``None`` skips loading.
        """
        if inpfilename is None:
            return

        inpfilename_str = str(inpfilename)

        # Default case: "plasma.inp"
        if inpfilename_str == "plasma.inp":
            toml_path = self._input_directory / "plasma.toml"
            inp_path = self._input_directory / "plasma.inp"

            if toml_path.exists():
                self._run_toml2inp(toml_path, inp_path)
                self._store_toml_data(toml_path)

            if inp_path.exists():
                self._load_from_inp(inp_path)
            return

        # Explicit path: convert via toml2inp even when .toml is specified
        path = self._input_directory / inpfilename
        if not path.exists():
            return

        if path.suffix == ".toml":
            inp_path = path.with_suffix(".inp")
            self._run_toml2inp(path, inp_path)
            self._store_toml_data(path)
            if inp_path.exists():
                self._load_from_inp(inp_path)
        else:
            self._load_from_inp(path)

    def _load_from_inp(self, inp_path: Path) -> None:
        """Load a plasma.inp-format parameter file."""
        logger.info(f"Loading parameter file: {inp_path.resolve()}")
        self._inp = InpFile(inp_path)
        convkey = UnitConversionKey.load(inp_path)
        if convkey is not None:
            self._unit = Units(dx=convkey.dx, to_c=convkey.to_c)

    def _store_toml_data(self, toml_path: Path) -> None:
        """Store ``plasma.toml`` as a :class:`TomlData` instance.

        ``*_groups`` tables using ``group_id`` are expanded into each
        entry at this point, and group tables are removed from the
        returned ``data.toml``.
        """
        from emout.utils.toml_converter import load_toml

        self._toml_data = load_toml(
            toml_path,
            resolve_groups=True,
            purge_groups=True,
        )

    @staticmethod
    def _run_toml2inp(toml_path: Path, inp_path: Path) -> None:
        """Generate plasma.inp from plasma.toml via the ``toml2inp`` command."""
        toml2inp = shutil.which("toml2inp")
        if toml2inp is None:
            logger.warning(
                "toml2inp command not found; skipping conversion from %s",
                toml_path,
            )
            return

        logger.info("Running toml2inp: %s -> %s", toml_path, inp_path)
        try:
            subprocess.run(
                [toml2inp, str(toml_path), "-o", str(inp_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "toml2inp failed (returncode=%d): %s",
                exc.returncode,
                exc.stderr.strip(),
            )

    @property
    def inp(self) -> Optional[InpFile]:
        """Return the parsed input parameter file.

        Returns
        -------
        InpFile or None
            Parsed ``plasma.inp`` if loaded, otherwise ``None``.
        """
        return self._inp

    @property
    def toml(self):
        """Return the parsed TOML configuration.

        Only available when ``plasma.toml`` exists. Provides attribute
        access to the structured TOML, e.g. ``data.toml.species[0].wp``.
        Group defaults (``group_id``) are expanded into each entry.

        Returns
        -------
        TomlData or None
            Parsed TOML data, or ``None`` if unavailable.
        """
        return self._toml_data

    @property
    def unit(self) -> Optional[Units]:
        """Return the unit conversion object.

        Returns
        -------
        Units or None
            Unit translators if a conversion key was found, otherwise ``None``.
        """
        return self._unit

    def is_valid(self) -> bool:
        """Check whether the simulation completed successfully.

        Compare the last step in the ``icur`` file against ``nstep`` from
        the input parameters.
        """
        # Use the last append directory if available, otherwise main_directory
        dirpath = self.append_directories[-1] if self.append_directories else self.main_directory
        icur_file = dirpath / "icur"
        if not icur_file.exists():
            return False

        def read_last_line(fname: Path) -> str:
            """Read the last line of a file.

            Parameters
            ----------
            fname : Path
                Target file path

            Returns
            -------
            str
                Last line (UTF-8 decoded).
            """
            with open(fname, "rb") as f:
                f.seek(-2, 2)
                while f.read(1) != b"\n":
                    f.seek(-2, 1)
                return f.readline().decode("utf-8")

        try:
            last_line = read_last_line(icur_file)
        except OSError:
            return False

        if self._inp is None:
            toml_path = self._input_directory / "plasma.toml"
            inp_path = self._input_directory / "plasma.inp"
            if toml_path.exists():
                self._run_toml2inp(toml_path, inp_path)
            if inp_path.exists():
                self._inp = InpFile(inp_path)

        return int(last_line.split()[0]) == int(self._inp.nstep)

    def read_icur_as_dataframe(self) -> pd.DataFrame:
        """Read the ``icur`` diagnostic file as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Table with step and per-species/per-body current columns.
        """
        if self._inp is None:
            raise RuntimeError("read_icur: .inp has not been loaded")

        names = []
        for ispec in range(self._inp.nspec):
            names.append(f"{ispec + 1}_step")
            for ipc in range(self._inp.npc):
                names.append(f"{ispec + 1}_body{ipc + 1}")
                names.append(f"{ispec + 1}_body{ipc + 1}_ema")

        icur_path = self.main_directory / "icur"
        if not icur_path.exists():
            raise FileNotFoundError(f"'icur' file not found: {icur_path}")

        return pd.read_csv(icur_path, sep=r"\s+", header=None, names=names)

    def read_pbody_as_dataframe(self) -> pd.DataFrame:
        """Read the ``pbody`` diagnostic file as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Table with ``step`` and per-body particle-count columns.
        """
        if self._inp is None:
            raise RuntimeError("read_pbody: .inp has not been loaded")

        names = ["step"] + [f"body{i + 1}" for i in range(self._inp.npc + 1)]
        pbody_path = self.main_directory / "pbody"
        if not pbody_path.exists():
            raise FileNotFoundError(f"'pbody' file not found: {pbody_path}")

        return pd.read_csv(pbody_path, sep=r"\s+", names=names)
