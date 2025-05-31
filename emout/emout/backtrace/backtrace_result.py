from typing import Any, Iterator

import numpy as np

from .xy_data import XYData


class BacktraceResult:
    """
    get_backtrace の返り値をまとめて管理するクラス。
    - ts           : shape = (N_steps,)
    - probability  : shape = (N_steps,)
    - positions    : shape = (N_steps, 3)  # (x, y, z)
    - velocities   : shape = (N_steps, 3)  # (vx, vy, vz)

    以下の方法でアクセス・可視化できます：

      ts, prob, pos, vel = result
        → タプルアンパック

      result.pair("x", "y").plot()
        → x vs y

      result.pair("t", "x").plot()
        → t vs x

      result.tx.plot()
        → x軸に t, y軸に x (= pair("t","x") と同義)

      result.yvz.plot()
        → y vs vz (= pair("y","vz"))
    """

    # サポートするキー
    _VALID_KEYS = {"t", "x", "y", "z", "vx", "vy", "vz"}

    def __init__(
        self,
        ts: np.ndarray,
        probability: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        unit=None,
    ):
        """
        Parameters
        ----------
        ts : numpy.ndarray, shape = (N_steps,)
        probability : numpy.ndarray, shape = (N_steps,)
        positions : numpy.ndarray, shape = (N_steps, 3)
        velocities : numpy.ndarray, shape = (N_steps, 3)
        """
        N = ts.shape[0]
        if positions.ndim != 2 or positions.shape != (N, 3):
            raise ValueError(
                "positions は shape=(N_steps, 3) の配列である必要があります"
            )
        if velocities.ndim != 2 or velocities.shape != (N, 3):
            raise ValueError(
                "velocities は shape=(N_steps, 3) の配列である必要があります"
            )

        self.ts = ts
        self.probability = probability
        self.positions = positions
        self.velocities = velocities

        self.unit = unit

    def __iter__(self) -> Iterator[Any]:
        """
        タプルアンパック対応:
          ts, prob, pos, vel = result
        """
        yield self.ts
        yield self.probability
        yield self.positions
        yield self.velocities

    def __repr__(self) -> str:
        N = len(self.ts)
        return (
            f"<BacktraceResult: n_steps={N}, keys={list(BacktraceResult._VALID_KEYS)}> "
            f"(use result.pair(var1,var2) to get XYData)"
        )

    def pair(self, var1: str, var2: str) -> XYData:
        """
        任意 2 変数を取り出し、XYData を返す。
        var1, var2 は以下のいずれか：
          't', 'x', 'y', 'z', 'vx', 'vy', 'vz'

        例:
          result.pair("x", "y")     → x vs y
          result.pair("t", "x")     → t vs x
          result.pair("z", "vy")    → z vs vy
        """
        if (
            var1 not in BacktraceResult._VALID_KEYS
            or var2 not in BacktraceResult._VALID_KEYS
        ):
            raise KeyError(
                f"Allowed keys = {BacktraceResult._VALID_KEYS}, but got '{var1}', '{var2}'"
            )

        def _get_array(key: str) -> np.ndarray:
            if key == "t":
                u = self.unit.t if self.unit else None

                return self.ts, u

            elif key in {"x", "y", "z"}:
                idx = {"x": 0, "y": 1, "z": 2}[key]

                u = self.unit.length if self.unit else None

                return self.positions[:, idx], u

            elif key in {"vx", "vy", "vz"}:
                idx = {"vx": 0, "vy": 1, "vz": 2}[key]

                u = self.unit.v if self.unit else None

                return self.velocities[:, idx], u

            else:
                raise KeyError(f"Unexpected key: {key}")

        arr1, u1 = _get_array(var1)
        arr2, u2 = _get_array(var2)
        xlabel = var1
        ylabel = var2
        title = f"{var1} vs {var2}"

        return XYData(
            arr1,
            arr2,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            units=(u1, u2) if u1 else None,
        )

    def __getattr__(self, name: str) -> Any:
        """
        属性アクセスを「ペア名」として解釈。以下のルールで pair() を返す。

        1)「前方 → 後方」の順に _VALID_KEYS からマッチを探す

        例:
          - result.tx           → pair("t","x")
          - result.tz           → pair("t","z")
          - result.tvx          → pair("t","vx")
          - result.xvy          → pair("x","vy")
          - result.yz           → pair("y","z")
        """
        for key1 in BacktraceResult._VALID_KEYS:
            if name.startswith(key1):
                rest = name[len(key1) :]
                if rest in BacktraceResult._VALID_KEYS:
                    return self.pair(key1, rest)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
