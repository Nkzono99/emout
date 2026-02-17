from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ParticleData:
    values: np.ndarray
    valunit: Optional[Any] = None
    name: str = "value"

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
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
        return ParticleData(new_values, valunit=self.valunit.si, name=self.name)

    # --- pandas bridge -----------------

    def to_series(self, index=None, replace_to_nan=True) -> pd.Series:
        """
        pandas.Series に変換（plotなどが使える）
        """
        if replace_to_nan:
            return pd.Series(self.values, index=index, name=self.name).replace(-9999, np.nan)

        return pd.Series(self.values, index=index, name=self.name)

    def plot(self, *args, **kwargs):
        """
        pandas の plot をそのまま使う
        """
        return self.to_series().plot(*args, **kwargs)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"ParticleData(name={self.name}, unit={self.valunit}, values={self.values})"
