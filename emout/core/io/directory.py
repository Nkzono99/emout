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
        input_path: Union[Path, str, None] = None,
        output_directory: Union[Path, str, None] = None,
    ):
        # 1. ディレクトリを Path に変換
        """DirectoryInspector を初期化する。

        Parameters
        ----------
        directory : Union[Path, str]
            基準ディレクトリです。`input_path` / `output_directory` 未指定時は
            入力ファイル・出力ファイルの両方をこのディレクトリから探索します。
        append_directories : Union[List[Union[Path, str]], str, None], optional
            追加で参照するディレクトリ群です。`'auto'` では連番サフィックス付きディレクトリを自動探索します。
        inpfilename : Union[Path, str], optional
            読み込む入力ファイル名です。`None` を指定すると `.inp` の読み込みをスキップします。
            `input_path` が指定されている場合は無視されます。
        input_path : Union[Path, str, None], optional
            入力パラメータファイルへのフルパスです（例: ``/path/to/plasma.toml``）。
            指定すると `directory` / `inpfilename` の代わりにこのパスが使われます。
        output_directory : Union[Path, str, None], optional
            シミュレーション出力ファイル（h5, icur, pbody 等）を格納したディレクトリです。
            未指定時は `directory` が使われます。
        """
        if not isinstance(directory, Path):
            directory = Path(directory)

        # input_path が指定された場合、そこから入力ディレクトリとファイル名を決定
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

        # 出力ディレクトリ (h5, icur, pbody 等)
        if output_directory is not None:
            self.main_directory: Path = Path(output_directory)
        else:
            self.main_directory = directory
        self.main_directory = self.main_directory.resolve()

        logger.info(
            f"DirectoryInspector: input directory = {self._input_directory.resolve()}, "
            f"output directory = {self.main_directory.resolve()}"
        )

        # 2. append_dirs の決定
        self.append_directories: List[Path] = []
        if append_directories == "auto":
            append_directories_list = self._fetch_append_directories(self.main_directory)
        else:
            append_directories_list = append_directories or []

        for ad in append_directories_list:
            p = Path(ad) if not isinstance(ad, Path) else ad
            self.append_directories.append(p.resolve())

        # 3. inp 読み込み + Units 初期化
        self._inp: Optional[InpFile] = None
        self._unit: Optional[Units] = None
        self._toml_data = None  # TomlData (plasma.toml が存在する場合に設定)
        self._load_inpfile(inpfilename)

    def _fetch_append_directories(self, directory: Path) -> List[Path]:
        """連番サフィックス付き追加ディレクトリを探索する。

        `<directory>_2`, `<directory>_3`, ... を順に確認し、
        `is_valid()` が `False` になった時点で探索を終了します。

        Parameters
        ----------
        directory : Path
            探索起点となるメインディレクトリです。

        Returns
        -------
        List[Path]
            有効と判定された追加ディレクトリ一覧です。
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
        """パラメータファイルを読み込み、単位変換情報を初期化する。

        ``plasma.toml`` が存在する場合は ``toml2inp`` コマンドで
        ``plasma.inp`` を生成・更新してから読み込む。

        Parameters
        ----------
        inpfilename : Union[Path, str]
            `_input_directory` からの相対ファイル名です。`None` の場合は読み込みません。

        Returns
        -------
        None
            戻り値はありません。
        """
        if inpfilename is None:
            return

        inpfilename_str = str(inpfilename)

        # デフォルト "plasma.inp" の場合
        if inpfilename_str == "plasma.inp":
            toml_path = self._input_directory / "plasma.toml"
            inp_path = self._input_directory / "plasma.inp"

            if toml_path.exists():
                self._run_toml2inp(toml_path, inp_path)
                self._store_toml_data(toml_path)

            if inp_path.exists():
                self._load_from_inp(inp_path)
            return

        # 明示指定: .toml が指定された場合も toml2inp で変換
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
        """plasma.inp 形式のファイルを読み込む。"""
        logger.info(f"Loading parameter file: {inp_path.resolve()}")
        self._inp = InpFile(inp_path)
        convkey = UnitConversionKey.load(inp_path)
        if convkey is not None:
            self._unit = Units(dx=convkey.dx, to_c=convkey.to_c)

    def _store_toml_data(self, toml_path: Path) -> None:
        """``plasma.toml`` を :class:`TomlData` として保持する。

        ``group_id`` を用いる ``*_groups`` は、この時点で各 entry に展開し、
        返却用の `data.toml` からは group table を除外する。
        """
        from emout.utils.toml_converter import load_toml

        self._toml_data = load_toml(
            toml_path,
            resolve_groups=True,
            purge_groups=True,
        )

    @staticmethod
    def _run_toml2inp(toml_path: Path, inp_path: Path) -> None:
        """``toml2inp`` コマンドで plasma.toml から plasma.inp を生成する。"""
        toml2inp = shutil.which("toml2inp")
        if toml2inp is None:
            logger.warning(
                "toml2inp command not found; "
                "skipping conversion from %s",
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
        """読み込まれた `InpFile` を返す。

        Returns
        -------
        Optional[InpFile]
            `.inp` 読み込み済みなら `InpFile`、未読込なら `None`。
        """
        return self._inp

    @property
    def toml(self):
        """TOML データを返す。

        ``plasma.toml`` が存在する場合のみ有効。
        ``data.toml.species[0].wp`` のように属性アクセスで
        構造化 TOML に直接アクセスできる。
        `group_id` ベースの group default は各 entry に展開済み。

        Returns
        -------
        TomlData or None
            TOML 読み込み時は `TomlData`、それ以外は `None`。
        """
        return self._toml_data

    @property
    def unit(self) -> Optional[Units]:
        """読み込まれた単位変換情報を返す。

        Returns
        -------
        Optional[Units]
            変換キーが取得できた場合は `Units`、未設定なら `None`。
        """
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
            """ファイル末尾行を読み込む。

            Parameters
            ----------
            fname : Path
                対象ファイルパス。

            Returns
            -------
            str
                末尾行（UTF-8 デコード済み文字列）。
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
        """`icur` ファイルを `DataFrame` として読み込む。

        Returns
        -------
        pandas.DataFrame
            ステップ列と各粒子種/ボディ電流列を持つテーブルです。
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
        """`pbody` ファイルを `DataFrame` として読み込む。

        Returns
        -------
        pandas.DataFrame
            `step` と各ボディ列からなるテーブルです。
        """
        if self._inp is None:
            raise RuntimeError("read_pbody: .inp が読み込まれていません")

        names = ["step"] + [f"body{i+1}" for i in range(self._inp.npc + 1)]
        pbody_path = self.main_directory / "pbody"
        if not pbody_path.exists():
            raise FileNotFoundError(f"'pbody' ファイルが見つかりません: {pbody_path}")

        return pd.read_csv(pbody_path, sep=r"\s+", names=names)
