# emout/io/directory.py

import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from emout.utils import InpFile
from emout.utils import UnitConversionKey, Units

logger = logging.getLogger(__name__)


class DirectoryInspector:
    """
    Emout 用のディレクトリ探索＆ .inp 読み込みヘルパークラス。
    Emout からはこのクラスを経由して 'main_directory', 'append_directories',
    'inp' (InpFile), 'unit' (Units) を参照できるようにする。
    """

    def __init__(
        self,
        directory: Union[Path, str],
        append_directories: Union[List[Union[Path, str]], str, None] = None,
        inpfilename: Union[Path, str] = "plasma.inp",
    ):
        # 1. メインディレクトリを Path に変換
        if not isinstance(directory, Path):
            directory = Path(directory)
        self.main_directory: Path = directory
        logger.info(
            f"DirectoryInspector: main directory = {self.main_directory.resolve()}"
        )

        # 2. append_dirs の決定
        self.append_directories: List[Path] = []
        if append_directories == "auto":
            append_directories_list = self._fetch_append_directories(directory)
        else:
            append_directories_list = append_directories or []

        for ad in append_directories_list:
            p = Path(ad) if not isinstance(ad, Path) else ad
            self.append_directories.append(p.resolve())

        # 3. inp 読み込み + Units 初期化
        self._inp: Optional[InpFile] = None
        self._unit: Optional[Units] = None
        self._load_inpfile(inpfilename)

    def _fetch_append_directories(self, directory: Path) -> List[Path]:
        logger.info(f"Fetching append directories for: {directory}")
        result: List[Path] = []
        i = 2
        while True:
            candidate = directory.parent / f"{directory.name}_{i}"
            if not candidate.exists():
                logger.debug(f"Append directory not found: {candidate}")
                break

            # 再帰的に DirectoryInspector を呼び出して妥当性チェック
            helper = DirectoryInspector(
                candidate, append_directories=None, inpfilename=None
            )
            if not helper.is_valid():
                logger.warning(
                    f"{candidate.resolve()} は存在するが有効ではないため終了"
                )
                break

            result.append(candidate)
            i += 1
        return result

    def _load_inpfile(self, inpfilename: Union[Path, str]) -> None:
        if inpfilename is None:
            return
        inp_path = self.main_directory / inpfilename
        if not inp_path.exists():
            return

        logger.info(f"Loading parameter file: {inp_path.resolve()}")
        self._inp = InpFile(inp_path)
        convkey = UnitConversionKey.load(inp_path)
        if convkey is not None:
            self._unit = Units(dx=convkey.dx, to_c=convkey.to_c)

    @property
    def inp(self) -> Optional[InpFile]:
        return self._inp

    @property
    def unit(self) -> Optional[Units]:
        return self._unit

    def is_valid(self) -> bool:
        """
        シミュレーションが正常終了しているかどうか判定する。
        最後に出力された 'icur' の最後のステップと .inp の nstep を比較する。
        """
        # append_directories があれば最後尾、そうでなければ main_directory
        dirpath = (
            self.append_directories[-1]
            if self.append_directories
            else self.main_directory
        )
        icur_file = dirpath / "icur"
        if not icur_file.exists():
            return False

        def read_last_line(fname: Path) -> str:
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
            self._inp = InpFile(dirpath / "plasma.inp")

        return int(last_line.split()[0]) == int(self._inp.nstep)

    def read_icur_as_dataframe(self) -> pd.DataFrame:
        """
        'icur' ファイルを DataFrame にして返す。
        Emout クラス時代のプロパティ icur に相当する機能をここへ移行。
        """
        if self._inp is None:
            raise RuntimeError("read_icur: .inp が読み込まれていません")

        names = []
        for ispec in range(self._inp.nspec):
            names.append(f"{ispec+1}_step")
            for ipc in range(self._inp.npc):
                names.append(f"{ispec+1}_body{ipc+1}")
                names.append(f"{ispec+1}_body{ipc+1}_ema")

        icur_path = self.main_directory / "icur"
        if not icur_path.exists():
            raise FileNotFoundError(f"'icur' ファイルが見つかりません: {icur_path}")

        return pd.read_csv(icur_path, sep=r"\s+", header=None, names=names)

    def read_pbody_as_dataframe(self) -> pd.DataFrame:
        """
        'pbody' ファイルを DataFrame にして返す。
        Emout クラス時代のプロパティ pbody に相当する機能をここへ移行。
        """
        if self._inp is None:
            raise RuntimeError("read_pbody: .inp が読み込まれていません")

        names = ["step"] + [f"body{i+1}" for i in range(self._inp.npc + 1)]
        pbody_path = self.main_directory / "pbody"
        if not pbody_path.exists():
            raise FileNotFoundError(f"'pbody' ファイルが見つかりません: {pbody_path}")

        return pd.read_csv(pbody_path, sep=r"\s+", names=names)
