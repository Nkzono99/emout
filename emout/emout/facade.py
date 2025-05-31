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

    def __getattr__(self, name: str) -> Union[GridDataSeries, VectorData2d]:
        """
        - r[e/b][xyz] → relocated field の生成
        - (dname)(axis1)(axis2) → VectorData2d
        - それ以外 → GridDataSeries
        """
        try:
            return self._grid_loader.load(name)
        except Exception as e:
            raise AttributeError(f"属性 '{name}' の読み込みに失敗しました: {e}")

    @property
    def backtrace(self) -> BacktraceWrapper:
        return BacktraceWrapper(
            directory=self._dir_inspector.main_directory,
            inp=self._dir_inspector.inp,
        )
