import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

from emout.utils import InpFile, Units

from .backtrace.solver_wrapper import BacktraceWrapper
from .data.griddata_series import GridDataSeries
from .data.vector_data import VectorData2d
from .io.directory import DirectoryInspector
from .io.grid import GridDataLoader
from .units import build_name2unit_mapping

from .data.particle_data_series import ParticlesSeries

logger = logging.getLogger(__name__)


class Emout:
    """
    EMSES 出力／.inp ファイルをまとめて扱う Facade クラス。
    """

    # name2unit マッピング (一度だけビルド)
    name2unit = build_name2unit_mapping(max_ndp=9)

    def __init__(
        self,
        directory: Union[Path, str] = "./",
        append_directories: Union[List[Union[Path, str]], None] = None,
        ad: Union[List[Union[Path, str]], None] = None,
        inpfilename: Union[Path, str] = "plasma.inp",
    ):
        self._dir_inspector = DirectoryInspector(
            directory=directory,
            append_directories=append_directories or ad,
            inpfilename=inpfilename,
        )

        self._grid_loader = GridDataLoader(
            dir_inspector=self._dir_inspector,
            name2unit_map=Emout.name2unit,
        )

    @property
    def directory(self) -> Path:
        return self._dir_inspector.main_directory

    @property
    def append_directories(self) -> List[Path]:
        return self._dir_inspector.append_directories

    @property
    def inp(self) -> Union[InpFile, None]:
        return self._dir_inspector.inp

    @property
    def unit(self) -> Union[Units, None]:
        return self._dir_inspector.unit

    def is_valid(self) -> bool:
        return self._dir_inspector.is_valid()

    @property
    def icur(self) -> pd.DataFrame:
        """
        'icur' を DataFrame で返す
        """
        return self._dir_inspector.read_icur_as_dataframe()

    @property
    def pbody(self) -> pd.DataFrame:
        """
        'pbody' を DataFrame で返す
        """
        return self._dir_inspector.read_pbody_as_dataframe()

    def particle(self, species: int):
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

    def __getattr__(self, name: str) -> Union[GridDataSeries, VectorData2d]:
        """
        - r[e/b][xyz] → relocated field の生成
        - (dname)(axis1)(axis2) → VectorData2d
        - それ以外 → GridDataSeries
        """
        import re

        m = re.match(r"^p([1-9])", name)
        if m:
            return self.particle(species=int(m.group(1)))

        try:
            return self._grid_loader.load(name)
        except Exception as e:
            raise AttributeError(f"属性 '{name}' の読み込みに失敗しました: {e}")

    @property
    def backtrace(self) -> BacktraceWrapper:
        return BacktraceWrapper(
            directory=self._dir_inspector.main_directory,
            inp=self._dir_inspector.inp,
            unit=self.unit,
        )
