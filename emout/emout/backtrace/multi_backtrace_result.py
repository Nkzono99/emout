from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np

from .xy_data import MultiXYData


class MultiBacktraceResult:
    """
    get_backtraces の返り値をまとめて管理するクラス。

    - ts_list         : shape = (N_traj, N_steps)
    - probabilities   : shape = (N_traj,)
    - positions_list  : shape = (N_traj, N_steps, 3)
    - velocities_list : shape = (N_traj, N_steps, 3)
    - last_indexes    : shape = (N_traj,)

    以下の方法でアクセス・可視化できます：

      ts_list, probs, pos_list, vel_list = result   # タプルアンパック

      # 任意の 2 変数ペア (var1, var2) を MultiXYData で取得
      result.pair("x", "y")    → 各 trajectory の x vs y
      result.pair("t", "x")    → 各 trajectory の t vs x
      result.tvx               → pair("t","vx") 省略形
      result.yvz               → pair("y","vz")

      # サンプリング: ランダムまたはインデックス指定
      result.sample(10)        → ランダムに 10 本だけ抽出
      result.sample([0,2,5])   → インデックス 0, 2, 5 の 3 本だけ抽出

      # プロット
      result.yvz.plot()
    """

    _VALID_KEYS = {"t", "x", "y", "z", "vx", "vy", "vz"}

    def __init__(
        self,
        ts_list: np.ndarray,
        probabilities: np.ndarray,
        positions_list: np.ndarray,
        velocities_list: np.ndarray,
        last_indexes: np.ndarray,
        unit=None,
    ):
        """
        Parameters
        ----------
        ts_list : numpy.ndarray, shape = (N_traj, N_steps)
        probabilities : numpy.ndarray, shape = (N_traj,)
        positions_list : numpy.ndarray, shape = (N_traj, N_steps, 3)
        velocities_list : numpy.ndarray, shape = (N_traj, N_steps, 3)
        last_indexes : numpy.ndarray, shape = (N_traj,)
        """
        if ts_list.ndim != 2:
            raise ValueError(
                "ts_list は 2D 配列 (N_traj, N_steps) である必要があります"
            )
        N, T = ts_list.shape

        if probabilities.shape != (N,):
            raise ValueError(
                "probabilities は shape=(N_traj, N_steps) である必要があります"
            )
        if positions_list.ndim != 3 or positions_list.shape != (N, T, 3):
            raise ValueError(
                "positions_list は shape=(N_traj, N_steps, 3) である必要があります"
            )
        if velocities_list.ndim != 3 or velocities_list.shape != (N, T, 3):
            raise ValueError(
                "velocities_list は shape=(N_traj, N_steps, 3) である必要があります"
            )

        self.ts_list = ts_list
        self.probabilities = probabilities
        self.positions_list = positions_list
        self.velocities_list = velocities_list
        self.last_indexes = last_indexes

        self.unit = unit

    def __iter__(self) -> Iterator[Any]:
        """
        タプルアンパック対応:
          ts_list, probabilities, positions_list, velocities_list, last_indexes = result
        """
        yield self.ts_list
        yield self.probabilities
        yield self.positions_list
        yield self.velocities_list
        yield self.last_indexes

    def __repr__(self) -> str:
        N, T = self.ts_list.shape
        return (
            f"<MultiBacktraceResult: n_traj={N}, n_steps={T}, keys={list(MultiBacktraceResult._VALID_KEYS)}> "
            f"(use result.pair(var1,var2) or result.sample(...) to inspect)"
        )

    def sample(
        self,
        indices: Union[int, Sequence[int], range, slice],
        random_state: Optional[int] = None,
    ) -> "MultiBacktraceResult":
        """
        ランダム／インデックス指定で一部トラジェクトリを抽出し、新しい MultiBacktraceResult を返す。

        Parameters
        ----------
        indices : int
            正の整数 k を渡すと「全 N_traj 本からランダムに k 本抽出」。
        indices : Sequence[int], range, slice
            int のリスト／range／slice を渡すと「指定されたインデックス群だけ抽出」。
        random_state : int, optional
            ランダムサンプリングのシード
        """
        N, T = self.ts_list.shape

        if isinstance(indices, int):
            k = indices
            if not (0 <= k <= N):
                raise ValueError(
                    "sample(int) の k は 0 ≤ k ≤ N_traj の範囲である必要があります"
                )
            rng = np.random.RandomState(random_state)
            chosen = rng.choice(N, size=k, replace=False)

        elif isinstance(indices, slice):
            chosen = list(range(N))[indices]

        elif isinstance(indices, range):
            chosen = list(indices)

        elif hasattr(indices, "__iter__"):
            chosen = list(indices)

        else:
            raise TypeError(
                "sample() の引数は int, slice, range, Sequence[int] のいずれかである必要があります"
            )

        if any((i < 0 or i >= N) for i in chosen):
            raise IndexError("sample() で指定されたインデックスが範囲外です")

        ts_sub = self.ts_list[chosen, :]
        prob_sub = self.probabilities[chosen]
        pos_sub = self.positions_list[chosen, :, :]
        vel_sub = self.velocities_list[chosen, :, :]
        last_indexes_sub = self.last_indexes[chosen]

        return MultiBacktraceResult(
            ts_sub, prob_sub, pos_sub, vel_sub, last_indexes_sub
        )

    def pair(self, var1: str, var2: str) -> MultiXYData:
        """
        任意の 2 変数を取り出し、MultiXYData を返す。
        var1, var2 は 't','x','y','z','vx','vy','vz' のいずれか。

        例:
          result.pair("x","y")     → 各 trajectory の x vs y
          result.pair("t","x")     → 各 trajectory の t vs x
          result.tvy               → pair("t","vy")
        """
        if (
            var1 not in MultiBacktraceResult._VALID_KEYS
            or var2 not in MultiBacktraceResult._VALID_KEYS
        ):
            raise KeyError(
                f"Allowed keys = {MultiBacktraceResult._VALID_KEYS}, but got '{var1}', '{var2}'"
            )

        def _get_array_list(key: str) -> np.ndarray:
            if key == "t":
                u = self.unit.t if self.unit else None

                return self.ts_list, u

            elif key in {"x", "y", "z"}:
                idx = {"x": 0, "y": 1, "z": 2}[key]

                u = self.unit.length if self.unit else None

                return self.positions_list[:, :, idx], u.unit

            elif key in {"vx", "vy", "vz"}:
                idx = {"vx": 0, "vy": 1, "vz": 2}[key]

                u = self.unit.v if self.unit else None

                return self.velocities_list[:, :, idx], u.unit

            else:
                raise KeyError(f"Unexpected key: {key}")

        arr1, u1 = _get_array_list(var1)  # shape = (N_traj, N_steps)
        arr2, u2 = _get_array_list(var2)

        xlabel = f"{var1} [m]" if self.unit else var1
        ylabel = f"{var2} [m]" if self.unit else var2
        title = f"{var1} vs {var2} (multiple trajectories)"

        return MultiXYData(
            arr1,
            arr2,
            self.last_indexes,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            unit=(u1, u2) if u1 else None,
        )

    def __getattr__(self, name: str) -> Any:
        """
        属性アクセスを「ペア名」と解釈。

        例:
          result.xvy   → pair("x","vy")
          result.tz    → pair("t","z")
          result.tvx   → pair("t","vx")
        """

        for key1 in MultiBacktraceResult._VALID_KEYS:
            if name.startswith(key1):
                rest = name[len(key1) :]
                if rest in MultiBacktraceResult._VALID_KEYS:
                    return self.pair(key1, rest)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
