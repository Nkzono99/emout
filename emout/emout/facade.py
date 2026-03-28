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
        """Emout ファサードを初期化する。

        Parameters
        ----------
        directory : Union[Path, str], optional
            処理対象ディレクトリのパスです。
        append_directories : Union[List[Union[Path, str]], None], optional
            追加で参照するディレクトリまたはそのリストです。
        ad : Union[List[Union[Path, str]], None], optional
            `append_directories` の短縮エイリアスです。両方指定した場合は `append_directories` が優先されます。
        inpfilename : Union[Path, str], optional
            入力パラメータファイル名です。通常は `plasma.inp` を指定します。
        """
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
        """メインディレクトリを返す。

        Returns
        -------
        Path
            EMSES 出力の基準ディレクトリです。
        """
        return self._dir_inspector.main_directory

    @property
    def append_directories(self) -> List[Path]:
        """追加チェーンディレクトリ一覧を返す。

        Returns
        -------
        List[Path]
            読み込み時に後段として連結されるディレクトリ一覧です。
        """
        return self._dir_inspector.append_directories

    @property
    def inp(self) -> Union[InpFile, None]:
        """読み込まれた `.inp` 情報を返す。

        Returns
        -------
        Union[InpFile, None]
            `.inp` の読み込み結果です。未読込時は `None`。
        """
        return self._dir_inspector.inp

    @property
    def toml(self):
        """TOML の生データを返す。

        ``plasma.toml`` から読み込んだ場合のみ有効。
        V2 の ``data.toml.species[0].wp`` のような
        TOML 本来の構造に直接アクセスできる。

        Returns
        -------
        TomlData or None
            TOML 読み込み時は ``TomlData``、それ以外は ``None``。
        """
        return self._dir_inspector.toml

    @property
    def unit(self) -> Union[Units, None]:
        """単位変換情報を返す。

        Returns
        -------
        Union[Units, None]
            `UnitConversionKey` が取得できた場合は `Units`、未設定なら `None`。
        """
        return self._dir_inspector.unit

    def is_valid(self) -> bool:
        """シミュレーション出力が正常終了しているか判定する。

        Returns
        -------
        bool
            条件判定結果です。
        """
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
        """粒子時系列データを読み込む `ParticlesSeries` を返す。

        Parameters
        ----------
        species : int
            粒子種別番号（1 始まり）です。

        Returns
        -------
        object
            指定種別の粒子データ系列オブジェクトです。
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
        """
        - r[e/b][xyz] → relocated field の生成
        - (dname)(axis1)(axis2)(axis3) → VectorData3d
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
        """バックトレース計算用ラッパーを返す。

        Returns
        -------
        BacktraceWrapper
            現在のディレクトリ/入力条件に紐づいたバックトレース API です。
        """
        return BacktraceWrapper(
            directory=self._dir_inspector.main_directory,
            inp=self._dir_inspector.inp,
            unit=self.unit,
        )
