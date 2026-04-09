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
    """EmoutDataExtraction クラス。
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
        """対象ディレクトリを返す。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self._data.directory

    @property
    def inp(self):
        """入力パラメータを返す。
        
        Returns
        -------
        object
            処理結果です。
        """
        return self._data.inp

    @property
    def unit(self):
        """単位変換オブジェクトを返す。
        
        Returns
        -------
        object
            処理結果です。
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
        """
        'icur' を DataFrame で返す
        """
        return self._data.icur

    @property
    def pbody(self) -> pd.DataFrame:
        """
        'pbody' を DataFrame で返す
        """
        return self._data.pbody

    def __getattr__(self, name: str) -> Union[GridDataSeries, VectorData2d]:
        """
        - r[e/b][xyz] → relocated field の生成
        - (dname)(axis1)(axis2) → VectorData2d
        - それ以外 → GridDataSeries
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
        """データを保存する。
        
        Parameters
        ----------
        name : object
            対象データ名またはキー名です。
        Returns
        -------
        None
            戻り値はありません。
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
        """バックトレースソルバを返す。
        
        Returns
        -------
        BacktraceWrapper
            処理結果です。
        """
        return self._data.backtrace

    @property
    def extract_dir(self):
        """dir を抽出する。
        
        Returns
        -------
        object
            処理結果です。
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
