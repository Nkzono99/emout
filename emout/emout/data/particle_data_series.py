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
        self.datafile = DataFileInfo(filename)
        self.h5 = h5py.File(str(filename), "r")
        self.group = self.h5[list(self.h5.keys())[0]]

        # 時刻キーが整数文字列であることを想定
        self._index2key = {int(key[-4:]): key for key in self.group.keys()}

        self.tunit = tunit
        self.valunit = valunit
        self.name = name

    def close(self) -> None:
        self.h5.close()

    @property
    def filename(self) -> Path:
        return self.datafile.filename

    @property
    def directory(self) -> Path:
        return self.datafile.directory

    def _sorted_timekeys(self) -> List[int]:
        return sorted(self._index2key.keys())

    def _create_data_with_timekey(self, timekey: int) -> ParticleData:
        if timekey not in self._index2key:
            raise IndexError(timekey)
        key = self._index2key[timekey]
        arr = np.array(self.group[key])  # (nparticle,)
        return ParticleData(arr, name=self.name, valunit=self.valunit)

    def __getitem__(self, item: Union[int, slice, List[int], Tuple[IndexLike]]):
        # tuple は未サポート（元設計互換）
        if isinstance(item, tuple):
            raise IndexError("Tuple indexing is not supported for ParticleDataSeries.")

        timekeys = self._sorted_timekeys()

        def pick_one(pos: int) -> ParticleData:
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
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self._index2key)

    def chain(self, other: "ParticleDataSeries") -> "MultiParticleDataSeries":
        return MultiParticleDataSeries(self, other)

    def __add__(self, other: "ParticleDataSeries") -> "MultiParticleDataSeries":
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
        if not isinstance(s, ParticleDataSeries):
            raise TypeError()
        if not isinstance(s, MultiParticleDataSeries):
            return [s]
        out: List[ParticleDataSeries] = []
        for ss in s.series:
            out += self._expand(ss)
        return out

    def close(self) -> None:
        for s in self.series:
            s.close()

    def __len__(self) -> int:
        if not self.drop_head_of_later:
            return int(np.sum([len(s) for s in self.series]))
        return len(self.series[0]) + int(np.sum([max(0, len(s) - 1) for s in self.series[1:]]))

    def _locate(self, index: int) -> Tuple[ParticleDataSeries, int]:
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
        for i in range(len(self)):
            yield self[i]


# ============================================================
# 2) スナップショット（t固定で x,y,z,vx,vy,vz,tid を束ねる）
# ============================================================
@dataclass(slots=True)
class ParticleSnapshot:
    fields: Dict[str, ParticleData]

    def __getattr__(self, name: str) -> ParticleData:
        try:
            return self.fields[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def keys(self) -> Iterable[str]:
        return self.fields.keys()

    def as_dict(self) -> Dict[str, ParticleData]:
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
        lengths = {k: len(v) for k, v in self._series.items()}
        if not lengths:
            raise ValueError("No particle component series were found.")
        uniq = sorted(set(lengths.values()))
        if len(uniq) > 1:
            raise ValueError(f"Component lengths are inconsistent: {lengths}")

    def close(self) -> None:
        # MultiParticleDataSeries.close() も正しく実装してあるが、念のため分岐で閉じる
        for s in self._series.values():
            if isinstance(s, MultiParticleDataSeries):
                s.close()
            else:
                s.close()

    def __enter__(self) -> "ParticlesSeries":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __getattr__(self, name: str):
        # p.x / p.vx / p.tid で series を返す
        if name in self._series:
            return self._series[name]
        raise AttributeError(name)

    def available_components(self) -> Tuple[str, ...]:
        return tuple(self._series.keys())

    def __len__(self) -> int:
        # 代表成分で返す（tid優先）
        for key in ("tid", "x", "vx"):
            if key in self._series:
                return len(self._series[key])
        # fallback
        any_key = next(iter(self._series))
        return len(self._series[any_key])

    def __getitem__(self, tindex: int) -> ParticleSnapshot:
        fields: Dict[str, ParticleData] = {}
        for comp, s in self._series.items():
            fields[comp] = s[tindex]
        return ParticleSnapshot(fields)

    def __iter__(self) -> Iterable[ParticleSnapshot]:
        for i in range(len(self)):
            yield self[i]
