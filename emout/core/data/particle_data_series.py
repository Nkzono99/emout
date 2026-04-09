from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np

import emout.utils as utils
from emout.utils import DataFileInfo, UnitTranslator

from .particle_data import ParticleData


# ============================================================
# 1) 1ファイル=1成分 の時系列
# ============================================================
IndexLike = Union[int, slice, List[int]]
GetItemType = Union[int, slice, List[int], Tuple[IndexLike]]


class ParticleDataSeries:
    """
    粒子1成分（x, vx, tid 等）の時系列データ（単一HDF5）を管理する。

    想定するHDF5構造：
      - 先頭グループ self.h5[list(self.h5.keys())[0]]
      - その配下に時刻キー（例: "0","1","2",...）が並ぶ
      - 各キーの dataset が shape=(nparticle,) の1D配列
    """

    def __init__(
        self,
        filename: PathLike,
        name: str,
        tunit: UnitTranslator = None,
        valunit: UnitTranslator = None,
    ):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        filename : PathLike
            保存先または読み込み対象のファイル名です。
        name : str
            対象データ名またはキー名です。
        tunit : UnitTranslator, optional
            時間軸の単位変換器です。
        valunit : UnitTranslator, optional
            値の単位変換器です。
        """
        self.datafile = DataFileInfo(filename)
        self.h5 = h5py.File(str(filename), "r")
        self.group = self.h5[list(self.h5.keys())[0]]

        # 時刻キーが整数文字列であることを想定
        self._index2key = {int(key[-4:]): key for key in self.group.keys()}

        self.tunit = tunit
        self.valunit = valunit
        self.name = name

    def close(self) -> None:
        """関連リソースをクローズする。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        self.h5.close()

    @property
    def filename(self) -> Path:
        """読み込み元ファイルパスを返す。
        
        Returns
        -------
        Path
            処理結果です。
        """
        return self.datafile.filename

    @property
    def directory(self) -> Path:
        """対象ディレクトリを返す。
        
        Returns
        -------
        Path
            処理結果です。
        """
        return self.datafile.directory

    def _sorted_timekeys(self) -> List[int]:
        """時刻キーを昇順で返す。
        
        Returns
        -------
        List[int]
            処理結果です。
        """
        return sorted(self._index2key.keys())

    def _create_data_with_timekey(self, timekey: int) -> ParticleData:
        """data with timekey を生成する。
        
        Parameters
        ----------
        timekey : int
            時間 index に対応するキー値です。
        Returns
        -------
        ParticleData
            処理結果です。
        """
        if timekey not in self._index2key:
            raise IndexError(timekey)
        key = self._index2key[timekey]
        arr = np.array(self.group[key])  # (nparticle,)
        return ParticleData(arr, name=self.name, valunit=self.valunit)

    def __getitem__(self, item: Union[int, slice, List[int], Tuple[IndexLike]]):
        # tuple は未サポート（元設計互換）
        """要素を取得する。
        
        Parameters
        ----------
        item : Union[int, slice, List[int], Tuple[IndexLike]]
            代入または更新する値です。
        Returns
        -------
        object
            処理結果です。
        """
        if isinstance(item, tuple):
            raise IndexError("Tuple indexing is not supported for ParticleDataSeries.")

        timekeys = self._sorted_timekeys()

        def pick_one(pos: int) -> ParticleData:
            """指定位置の `ParticleData` を返す。
            
            Parameters
            ----------
            pos : int
                取得したい時系列位置 index です（負 index 対応）。
            Returns
            -------
            ParticleData
                処理結果です。
            """
            if pos < 0:
                pos = len(timekeys) + pos
            if pos < 0 or pos >= len(timekeys):
                raise IndexError(pos)
            return self._create_data_with_timekey(timekeys[pos])

        if isinstance(item, int):
            return pick_one(item)

        if isinstance(item, slice):
            positions = list(utils.range_with_slice(item, maxlen=len(timekeys)))
            return [pick_one(p) for p in positions]

        if isinstance(item, list):
            return [pick_one(p) for p in item]

        raise TypeError(type(item))

    def __iter__(self) -> Iterable[ParticleData]:
        """イテレータを返す。
        
        Returns
        -------
        Iterable[ParticleData]
            処理結果です。
        """
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """要素数を返す。
        
        Returns
        -------
        int
            要素数。
        """
        return len(self._index2key)

    def chain(self, other: "ParticleDataSeries") -> "MultiParticleDataSeries":
        """別系列を連結した複合系列を返す。
        
        Parameters
        ----------
        other : "ParticleDataSeries"
            演算または比較の相手となる値です。
        Returns
        -------
        "MultiParticleDataSeries"
            処理結果です。
        """
        return MultiParticleDataSeries(self, other)

    def __add__(self, other: "ParticleDataSeries") -> "MultiParticleDataSeries":
        """加算演算を適用する。
        
        Parameters
        ----------
        other : "ParticleDataSeries"
            演算または比較の相手となる値です。
        Returns
        -------
        "MultiParticleDataSeries"
            処理結果です。
        """
        if not isinstance(other, ParticleDataSeries):
            raise TypeError()
        return self.chain(other)


class MultiParticleDataSeries(ParticleDataSeries):
    """
    連続する複数の ParticleDataSeries を結合して管理する。

    drop_head_of_later=True の場合：
      - 2本目以降の先頭1ステップは前ファイル末尾と重複するとみなし捨てる
    """

    def __init__(self, *series: ParticleDataSeries, drop_head_of_later: bool = True):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        *series : ParticleDataSeries
            追加の位置引数。内部で呼び出す関数へ渡されます。
        drop_head_of_later : bool, optional
            `True` の場合、2 本目以降の系列の先頭 1 ステップを
            重複データとみなして除外します。
        """
        self.series: List[ParticleDataSeries] = []
        for s in series:
            self.series += self._expand(s)

        if not self.series:
            raise ValueError("No series were provided.")

        self.datafile = self.series[0].datafile
        self.tunit = self.series[0].tunit
        self.valunit = self.series[0].valunit
        self.name = self.series[0].name

        self.drop_head_of_later = drop_head_of_later

    def _expand(
        self, s: Union["ParticleDataSeries", "MultiParticleDataSeries"]
    ) -> List[ParticleDataSeries]:
        """複合系列を `ParticleDataSeries` のリストへ展開する。
        
        Parameters
        ----------
        s : Union["ParticleDataSeries", "MultiParticleDataSeries"]
            展開対象の系列です。
        Returns
        -------
        List[ParticleDataSeries]
            フラット化した `ParticleDataSeries` のリストです。
        """
        if not isinstance(s, ParticleDataSeries):
            raise TypeError()
        if not isinstance(s, MultiParticleDataSeries):
            return [s]
        out: List[ParticleDataSeries] = []
        for ss in s.series:
            out += self._expand(ss)
        return out

    def close(self) -> None:
        """関連リソースをクローズする。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        for s in self.series:
            s.close()

    def __len__(self) -> int:
        """要素数を返す。
        
        Returns
        -------
        int
            要素数。
        """
        if not self.drop_head_of_later:
            return int(np.sum([len(s) for s in self.series]))
        return len(self.series[0]) + int(np.sum([max(0, len(s) - 1) for s in self.series[1:]]))

    def _locate(self, index: int) -> Tuple[ParticleDataSeries, int]:
        """全体 index に対応する系列とローカル index を特定する。
        
        Parameters
        ----------
        index : int
            参照する index 値です。
        Returns
        -------
        Tuple[ParticleDataSeries, int]
            処理結果です。
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)

        n0 = len(self.series[0])
        if index < n0:
            return self.series[0], index

        offset = n0
        for s in self.series[1:]:
            if self.drop_head_of_later:
                usable = max(0, len(s) - 1)
                if index < offset + usable:
                    local = (index - offset) + 1  # 先頭を飛ばす
                    return s, local
                offset += usable
            else:
                usable = len(s)
                if index < offset + usable:
                    local = index - offset
                    return s, local
                offset += usable

        raise IndexError(index)

    def __getitem__(self, item):
        """要素を取得する。
        
        Parameters
        ----------
        item : object
            代入または更新する値です。
        Returns
        -------
        object
            処理結果です。
        """
        if isinstance(item, tuple):
            raise IndexError("Tuple indexing is not supported for MultiParticleDataSeries.")

        if isinstance(item, int):
            s, local = self._locate(item)
            return s[local]

        if isinstance(item, slice):
            positions = list(utils.range_with_slice(item, maxlen=len(self)))
            return [self[i] for i in positions]

        if isinstance(item, list):
            return [self[i] for i in item]

        raise TypeError(type(item))

    def __iter__(self) -> Iterable[ParticleData]:
        """イテレータを返す。
        
        Returns
        -------
        Iterable[ParticleData]
            処理結果です。
        """
        for i in range(len(self)):
            yield self[i]


# ============================================================
# 2) スナップショット（t固定で x,y,z,vx,vy,vz,tid を束ねる）
# ============================================================
@dataclass(slots=True)
class ParticleSnapshot:
    """ParticleSnapshot クラス。
    """
    fields: Dict[str, ParticleData]

    def __getattr__(self, name: str) -> ParticleData:
        """属性アクセスを解決する。
        
        Parameters
        ----------
        name : str
            対象データ名またはキー名です。
        Returns
        -------
        ParticleData
            処理結果です。
        """
        try:
            return self.fields[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def keys(self) -> Iterable[str]:
        """利用可能なキー一覧を返す。
        
        Returns
        -------
        Iterable[str]
            処理結果です。
        """
        return self.fields.keys()

    def as_dict(self) -> Dict[str, ParticleData]:
        """内容を辞書形式で返す。
        
        Returns
        -------
        Dict[str, ParticleData]
            処理結果です。
        """
        return dict(self.fields)


# ============================================================
# 3) 束ねクラス：ディレクトリを走査して成分ごとにseriesを構築
# ============================================================
class ParticlesSeries:
    """
    成分ごとに分割された粒子HDF5群を束ねる管理クラス。

    API:
      p = ParticlesSeries(dir, species=1)
      p.x, p.vx, p.tid ... -> ParticleDataSeries or MultiParticleDataSeries
      p[t] -> ParticleSnapshot（x,y,z,vx,vy,vz,tid をまとめて取得）

    ファイル名例:
      p1xe00_0000.h5
      p1vxe00_0000.h5
      p1tid00_0000.h5
    """

    _DEFAULT_COMPONENTS = ("x", "y", "z", "vx", "vy", "vz", "tid")

    # compの後に "e" が入るケースと入らないケースを両対応
    _FILE_RE = re.compile(
        r"^p(?P<sp>\d+)"
        r"(?P<comp>tid|vx|vy|vz|x|y|z)"
        r"(?:e)?(?P<seg>\d+)"
        r"_(?P<part>\d+)\.h5$"
    )

    def __init__(
        self,
        directory: PathLike,
        species: int,
        *,
        components: Tuple[str, ...] = _DEFAULT_COMPONENTS,
        tunit: UnitTranslator = None,
        vunits: Optional[Dict[str, UnitTranslator]] = None,
        drop_head_of_later: bool = True,
        strict_length: bool = True,
    ):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        directory : PathLike
            処理対象ディレクトリのパスです。
        species : int
            粒子種別を表す文字列です。
        components : Tuple[str, ...], optional
            読み込む粒子成分名の一覧です。
        tunit : UnitTranslator, optional
            時間軸の単位変換器です。
        vunits : Optional[Dict[str, UnitTranslator]], optional
            粒子量ごとの単位変換器マップです。
        drop_head_of_later : bool, optional
            `True` の場合、後続ファイル系列の先頭 1 ステップを
            重複とみなして除外します。
        strict_length : bool, optional
            `True` の場合、全成分系列の長さが一致することを検証します。
        """
        self.directory = Path(directory)
        self.species = int(species)
        self.components = tuple(components)
        self.tunit = tunit
        self.vunits = vunits or {}
        self.drop_head_of_later = drop_head_of_later
        self.strict_length = strict_length

        # comp -> ParticleDataSeries or MultiParticleDataSeries
        self._series: Dict[str, Union[ParticleDataSeries, MultiParticleDataSeries]] = {}
        self._build()

        if self.strict_length:
            self._validate_lengths()

    def _build(self) -> None:
        """粒子データ系列を構築する。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        buckets: Dict[str, List[Tuple[Tuple[int, int], Path]]] = {c: [] for c in self.components}

        for fp in self.directory.glob("*.h5"):
            m = self._FILE_RE.match(fp.name)
            if not m:
                continue
            sp = int(m.group("sp"))
            if sp != self.species:
                continue

            comp = m.group("comp")
            if comp not in buckets:
                continue

            seg = int(m.group("seg"))
            part = int(m.group("part"))
            buckets[comp].append(((seg, part), fp))

        for comp, items in buckets.items():
            if not items:
                continue

            items.sort(key=lambda x: x[0])  # (seg, part)
            paths = [p for _, p in items]

            vunit = self.vunits.get(comp)
            series_list = [
                ParticleDataSeries(p, name=comp, tunit=self.tunit, valunit=vunit) for p in paths
            ]

            if len(series_list) == 1:
                self._series[comp] = series_list[0]
            else:
                self._series[comp] = MultiParticleDataSeries(
                    *series_list, drop_head_of_later=self.drop_head_of_later
                )

    def _validate_lengths(self) -> None:
        """成分系列の長さ整合性を検証する。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        lengths = {k: len(v) for k, v in self._series.items()}
        if not lengths:
            raise ValueError("No particle component series were found.")
        uniq = sorted(set(lengths.values()))
        if len(uniq) > 1:
            raise ValueError(f"Component lengths are inconsistent: {lengths}")

    def close(self) -> None:
        # MultiParticleDataSeries.close() も正しく実装してあるが、念のため分岐で閉じる
        """関連リソースをクローズする。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        for s in self._series.values():
            if isinstance(s, MultiParticleDataSeries):
                s.close()
            else:
                s.close()

    def __enter__(self) -> "ParticlesSeries":
        """コンテキストに入る。
        
        Returns
        -------
        "ParticlesSeries"
            処理結果です。
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """コンテキストを終了する。
        
        Parameters
        ----------
        exc_type : object
            例外型です。
        exc : object
            例外インスタンスです。
        tb : object
            トレースバック情報です。
        Returns
        -------
        None
            戻り値はありません。
        """
        self.close()

    def __getattr__(self, name: str):
        # p.x / p.vx / p.tid で series を返す
        """属性アクセスを解決する。
        
        Parameters
        ----------
        name : str
            対象データ名またはキー名です。
        Returns
        -------
        object
            処理結果です。
        """
        if name in self._series:
            return self._series[name]
        raise AttributeError(name)

    def available_components(self) -> Tuple[str, ...]:
        """利用可能な粒子成分名を返す。
        
        Returns
        -------
        Tuple[str, ...]
            処理結果です。
        """
        return tuple(self._series.keys())

    def __len__(self) -> int:
        # 代表成分で返す（tid優先）
        """要素数を返す。
        
        Returns
        -------
        int
            要素数。
        """
        for key in ("tid", "x", "vx"):
            if key in self._series:
                return len(self._series[key])
        # fallback
        any_key = next(iter(self._series))
        return len(self._series[any_key])

    def __getitem__(self, tindex: int) -> ParticleSnapshot:
        """要素を取得する。
        
        Parameters
        ----------
        tindex : int
            時間方向の index です。
        Returns
        -------
        ParticleSnapshot
            処理結果です。
        """
        fields: Dict[str, ParticleData] = {}
        for comp, s in self._series.items():
            fields[comp] = s[tindex]
        return ParticleSnapshot(fields)

    def __iter__(self) -> Iterable[ParticleSnapshot]:
        """イテレータを返す。
        
        Returns
        -------
        Iterable[ParticleSnapshot]
            処理結果です。
        """
        for i in range(len(self)):
            yield self[i]
