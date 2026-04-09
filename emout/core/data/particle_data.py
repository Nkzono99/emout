from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ParticleData:
    """ParticleData クラス。
    """
    values: np.ndarray
    valunit: Optional[Any] = None
    name: str = "value"

    def __post_init__(self):
        """初期化後の検証と整形を行う。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        self.values = np.asarray(self.values, dtype=float)
        self.values[self.values == -9999] = np.nan
        if self.values.ndim != 1:
            raise ValueError("ParticleData must be a 1D float array.")

    @property
    def val_si(self) -> "ParticleData":
        """
        SI単位系に変換した ParticleData を返す
        """
        if self.valunit is None:
            raise ValueError("valunit is not set.")

        new_values = self.valunit.reverse(self.values)
        return ParticleData(new_values, valunit=self.valunit, name=self.name)

    # --- pandas bridge -----------------

    def to_series(self, index=None) -> pd.Series:
        """
        pandas.Series に変換（plotなどが使える）
        """

        return pd.Series(self.values, index=index, name=self.name)

    def plot(self, *args, **kwargs):
        """`pandas.Series.plot` を用いて粒子データを描画する。

        Parameters
        ----------
        *args : tuple
            `pandas.Series.plot` へ渡す位置引数です。
            例: `kind`, `x`, `y`。
        **kwargs : dict
            `pandas.Series.plot` へ渡すキーワード引数です。
            例: `ax`, `figsize`, `title`, `xlabel`, `ylabel`, `grid`,
            `legend`, `color`, `style`, `xlim`, `ylim`。

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.artist.Artist
            pandas が返す描画オブジェクトです（`kind` に依存）。
        """
        return self.to_series().plot(*args, **kwargs)

    def __len__(self):
        """要素数を返す。
        
        Returns
        -------
        int
            要素数。
        """
        return len(self.values)

    def __repr__(self):
        """文字列表現を返す。
        
        Returns
        -------
        str
            文字列表現。
        """
        return f"ParticleData(name={self.name}, unit={self.valunit}, values={self.values})"
