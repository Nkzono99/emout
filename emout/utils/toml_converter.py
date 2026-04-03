"""TOML の生データに属性アクセスで直接アクセスするための :class:`TomlData` ラッパー.

plasma.toml → plasma.inp の変換は MPIEMSES3D 側の ``toml2inp`` コマンドで行う。
"""

from pathlib import Path
from typing import Any, Dict

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# ---------------------------------------------------------------------------
# TomlData: TOML 辞書への属性アクセスラッパー
# ---------------------------------------------------------------------------


class TomlData:
    """TOML の辞書構造に属性アクセスできるラッパー.

    ``data.species[0].wp`` のようにネストした辞書・リストへ
    ドットアクセスで到達できる。辞書としてのアクセス
    (``data["tmgrid"]["nx"]``) も同時にサポートする。

    Parameters
    ----------
    data : dict
        TOML から読み込んだ辞書。
    """

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", data)

    # --- dict ライクアクセス ---

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return (_wrap(v) for v in self._data.values())

    def items(self):
        return ((k, _wrap(v)) for k, v in self._data.items())

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        return _wrap(val) if val is not default else default

    # --- 属性アクセス ---

    def __getattr__(self, key: str) -> Any:
        try:
            return _wrap(self._data[key])
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{key}'"
            )

    # --- 表示 ---

    def __repr__(self) -> str:
        return f"TomlData({self._data!r})"

    def __str__(self) -> str:
        return str(self._data)

    def to_dict(self) -> Dict[str, Any]:
        """元の辞書を返す。"""
        return self._data


def _wrap(value: Any) -> Any:
    """辞書を TomlData に、辞書リストを TomlData リストに再帰的にラップする。"""
    if isinstance(value, dict):
        return TomlData(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def load_toml(toml_path: Path) -> TomlData:
    """plasma.toml を読み込み TomlData として返す.

    Parameters
    ----------
    toml_path : Path
        plasma.toml のパス

    Returns
    -------
    TomlData
        TOML の辞書構造に属性アクセスできるラッパー
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return TomlData(data)
