"""plasma.toml を f90nml.Namelist に変換するモジュール.

TOML V1（フラット配列）と V2（構造化 [[species]] 等）の両方に対応する。
変換後は既存の InpFile クラスへそのまま注入できる。

また、TOML の生データに属性アクセスで直接アクセスするための
:class:`TomlData` ラッパーも提供する。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import f90nml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from emout.utils.emsesinp import UnitConversionKey


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
        TOML の辞書構造に属性アクセスできる��ッパー
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return TomlData(data)


# ---------------------------------------------------------------------------
# V2 [[species]] パラメ��タ → namelist グループ名の対応表
# ---------------------------------------------------------------------------
SPECIES_KEY_TO_GROUP: Dict[str, str] = {
    # &plasma
    "wp": "plasma",
    "denmod": "plasma",
    "denk": "plasma",
    "omegalw": "plasma",
    # &intp
    "qm": "intp",
    "npin": "intp",
    "np": "intp",
    "dnsf": "intp",
    "path": "intp",
    "peth": "intp",
    "vdri": "intp",
    "vdthz": "intp",
    "vdthxy": "intp",
    "vdx": "intp",
    "vdy": "intp",
    "vdz": "intp",
    "vpa": "intp",
    "vpb": "intp",
    "vpe": "intp",
    "spa": "intp",
    "spe": "intp",
    "speth": "intp",
    "f": "intp",
    "nphi": "intp",
    "ndst": "intp",
    "ioptd": "intp",
    "lcgamma": "intp",
    "lcbeta": "intp",
    # &inp
    "inpf": "inp",
    "inpb": "inp",
    "injct": "inp",
    "npr": "inp",
    # &emissn
    "nflag_emit": "emissn",
    "nepl": "emissn",
    "curf": "emissn",
    "curb": "emissn",
    "flpf": "emissn",
    "flpb": "emissn",
    "qp": "emissn",
    "qpr": "emissn",
    "abvdem": "emissn",
    "dnsb": "emissn",
    "ray_zenith_angle_deg": "emissn",
    "ray_azimuth_angle_deg": "emissn",
    "emission_time_mode": "emissn",
    "emission_start_step": "emissn",
    "emission_end_step": "emissn",
    "emission_time_sigma": "emissn",
    "emission_leave_anti_particle": "emissn",
    # &digcon
    "ipadig": "digcon",
    "ipahdf": "digcon",
    "ipaxyz": "digcon",
    "ildig": "digcon",
    "imdig": "digcon",
    "isort": "digcon",
    "irhsp": "digcon",
    # &system
    "npbnd": "system",
    "mfpath": "system",
    # &ptcond
    "pemax": "ptcond",
    "deltaemax": "ptcond",
}

# 2D 種パラメータ (各種がリストを持ち、フラット化時に extend する)
_SPECIES_2D_PARAMS = {"ipaxyz", "npbnd"}

# V2 ptcond.objects のスカラーパラメータ
_OBJECT_SCALAR_PARAMS = [
    "geotype",
    "xlpc",
    "xupc",
    "ylpc",
    "yupc",
    "zlpc",
    "zupc",
    "bdyradius",
    "bdyalign",
    "biasp",
    "dscaled",
    "nflag_subcell",
    "wirealign",
    "wirerradius",
    "wireeradius",
    "wirehlength",
]

# V2 ptcond.objects のマトリクスパラメータ (key -> first dim size)
_OBJECT_MATRIX_PARAMS = {
    "bdycoord": 3,
    "bdyedge": 2,
    "wireorigin": 3,
}

# V2 ptcond.conductor_groups のパラメータ
_CONDUCTOR_GROUP_PARAMS = ["mtd_vchg", "pfixed"]

# V2 ptcond.boundaries のスカラーパラメータ
_BOUNDARY_SCALAR_PARAMS = [
    "sphere_radius",
    "circle_radius",
    "cylinder_radius",
    "cylinder_height",
    "disk_radius",
    "disk_inner_radius",
    "disk_height",
    "plane_with_circle_radius",
    "boundary_conductor_id",
    "conductivity",
]

# V2 ptcond.boundaries のマトリクスパラメータ
_BOUNDARY_MATRIX_PARAMS = {
    "rectangle_shape": 6,
    "cuboid_shape": 6,
    "sphere_origin": 3,
    "circle_origin": 3,
    "cylinder_origin": 3,
    "disk_origin": 3,
    "plane_with_circle_origin": 3,
}

# V2 ptcond.boundaries の種パラメータ
_BOUNDARY_SPECIES_PARAMS = [
    "boundary_mirror_reflection_rate",
    "boundary_reversal_reflection_rate",
    "boundary_mirror_reflect_alpha",
    "boundary_reversal_reflect_alpha",
    "boundary_mirror_reflect_energy_loss_frac",
    "boundary_reversal_reflect_energy_loss_frac",
    "enable_secondary_electron_emission",
    "boundary_se_yield_max",
    "boundary_se_energy_max",
    "boundary_se_species_id",
    "boundary_se_model_type",
    "boundary_se_const_yield",
]

# boundary type 文字列 → 整数値
_BOUNDARY_TYPE_MAP = {
    "rectangle": 1,
    "cuboid": 2,
    "sphere": 3,
    "circlex": 4,
    "circley": 5,
    "circlez": 6,
    "cylinderx": 7,
    "cylindery": 8,
    "cylinderz": 9,
    "open-cylinderx": 10,
    "open-cylindery": 11,
    "open-cylinderz": 12,
    "diskx": 13,
    "disky": 14,
    "diskz": 15,
    "plane-with-circlex": 16,
    "plane-with-circley": 17,
    "plane-with-circlez": 18,
}

# emissn.planes パラメータ
_EMISSION_PLANE_PARAMS = [
    "nemd",
    "ipcpl",
    "xmine",
    "xmaxe",
    "ymine",
    "ymaxe",
    "zmine",
    "zmaxe",
    "curfs",
    "curbs",
    "dnsfs",
    "dnsbs",
    "flpfs",
    "flpbs",
    "remf",
    "remb",
]

# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def load_toml_as_namelist(
    toml_path: Path,
) -> Tuple[f90nml.Namelist, Optional[UnitConversionKey]]:
    """plasma.toml を読み込み f90nml.Namelist と UnitConversionKey を返す.

    Parameters
    ----------
    toml_path : Path
        plasma.toml のパス

    Returns
    -------
    tuple[f90nml.Namelist, Optional[UnitConversionKey]]
        変換後の Namelist と単位変換キー
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    convkey = _extract_unit_conversion_key(data)
    version = _detect_format_version(data)

    if version >= 2:
        nml = _convert_v2(data)
    else:
        nml = _convert_v1(data)

    return nml, convkey


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------


def _detect_format_version(data: Dict[str, Any]) -> int:
    meta = data.get("meta", {})
    return int(meta.get("format_version", 1))


def _extract_unit_conversion_key(
    data: Dict[str, Any],
) -> Optional[UnitConversionKey]:
    meta = data.get("meta", {})
    uc = meta.get("unit_conversion", {})
    if "dx" in uc and "to_c" in uc:
        return UnitConversionKey(dx=float(uc["dx"]), to_c=float(uc["to_c"]))
    return None


# ---------------------------------------------------------------------------
# V1 変換: テーブル → namelist グループの直接マッピング
# ---------------------------------------------------------------------------

# V1 で除外するトップレベルキー
_V1_SKIP_KEYS = {"meta", "$schema"}


def _convert_v1(data: Dict[str, Any]) -> f90nml.Namelist:
    nml = f90nml.Namelist()
    for group_name, group_data in data.items():
        if group_name in _V1_SKIP_KEYS:
            continue
        if isinstance(group_data, dict):
            nml[group_name] = f90nml.Namelist(group_data)
        # V1 ではリスト型のトップレベル (species 等) は想定しない
    return nml


# ---------------------------------------------------------------------------
# V2 変換
# ---------------------------------------------------------------------------

_V2_SKIP_KEYS = {"meta", "$schema", "species"}


def _convert_v2(data: Dict[str, Any]) -> f90nml.Namelist:
    nml = f90nml.Namelist()

    # 1. 通常テーブルをコピー (ptcond 等の構��化サブテーブルは後で処理)
    for group_name, group_data in data.items():
        if group_name in _V2_SKIP_KEYS:
            continue
        if not isinstance(group_data, dict):
            continue

        if group_name == "ptcond":
            _flatten_ptcond_v2(group_data, nml)
        elif group_name == "emissn":
            _flatten_emissn_v2(group_data, nml)
        elif group_name == "dipole":
            _flatten_array_of_tables(
                group_data,
                "sources",
                nml,
                "dipole",
                count_key="nmd",
                scalar_keys=["md", "mdx", "mdy", "mdz", "mddir"],
                matrix_keys={},
            )
        elif group_name == "testch":
            _flatten_array_of_tables(
                group_data,
                "charges",
                nml,
                "testch",
                count_key="ntch",
                scalar_keys=["qtch", "e1tch", "p1tch", "rcutoff"],
                matrix_keys={"rtch": 3},
            )
        elif group_name == "jsrc":
            _flatten_array_of_tables(
                group_data,
                "sources",
                nml,
                "jsrc",
                count_key="njs",
                scalar_keys=["wjs", "th0js"],
                matrix_keys={"rjs": 3, "ajs": 3},
            )
        else:
            nml[group_name] = f90nml.Namelist(group_data)

    # 2. [[species]] 展開
    if "species" in data:
        _flatten_species(data["species"], nml)

    return nml


def _ensure_group(nml: f90nml.Namelist, group: str) -> f90nml.Namelist:
    if group not in nml:
        nml[group] = f90nml.Namelist()
    return nml[group]


# ---------------------------------------------------------------------------
# Species 展開
# ---------------------------------------------------------------------------


def _flatten_species(
    species_list: List[Dict[str, Any]], nml: f90nml.Namelist
) -> None:
    # species 数を system.nspec に設定
    _ensure_group(nml, "system")["nspec"] = len(species_list)

    # すべての species で使われているキーを収集
    all_keys = set()
    for sp in species_list:
        all_keys.update(sp.keys())

    for key in all_keys:
        group_name = SPECIES_KEY_TO_GROUP.get(key)
        if group_name is None:
            continue

        values: List[Any] = []
        for sp in species_list:
            val = sp.get(key)
            if key in _SPECIES_2D_PARAMS:
                # 2D パラメータ: リストを extend
                if val is not None:
                    values.extend(val)
                else:
                    # パディング: 該当次元分の None
                    dim = {"ipaxyz": 7, "npbnd": 3}.get(key, 1)
                    values.extend([None] * dim)
            else:
                values.append(val)

        grp = _ensure_group(nml, group_name)
        grp[key] = values


# ---------------------------------------------------------------------------
# ptcond 展開
# ---------------------------------------------------------------------------


def _flatten_ptcond_v2(
    ptcond_data: Dict[str, Any], nml: f90nml.Namelist
) -> None:
    grp = _ensure_group(nml, "ptcond")

    # 通常パラメータ (objects, boundaries, conductor_groups 以外) をコピー
    for key, val in ptcond_data.items():
        if key not in ("objects", "boundaries", "conductor_groups"):
            grp[key] = val

    # conductor_groups
    cg_data = ptcond_data.get("conductor_groups", {})
    group_names = list(cg_data.keys())
    npcg = len(group_names)
    if npcg > 0:
        grp["npcg"] = npcg
        for param in _CONDUCTOR_GROUP_PARAMS:
            vals = [cg_data[gn].get(param) for gn in group_names]
            if any(v is not None for v in vals):
                grp[param] = vals

    # objects
    objects = ptcond_data.get("objects", [])
    npc = len(objects)
    if npc > 0:
        grp["npc"] = npc

        # pcgs: オブジェクトのグループ帰属
        if group_names:
            name_to_idx = {n: i + 1 for i, n in enumerate(group_names)}
            pcgs = []
            for obj in objects:
                cond = obj.get("conductor", "")
                pcgs.append(name_to_idx.get(cond, 1))
            grp["pcgs"] = pcgs

        for param in _OBJECT_SCALAR_PARAMS:
            vals = [obj.get(param) for obj in objects]
            if any(v is not None for v in vals):
                grp[param] = vals

        for param, dim in _OBJECT_MATRIX_PARAMS.items():
            flat: List[Any] = []
            has_any = False
            for obj in objects:
                val = obj.get(param)
                if val is not None:
                    has_any = True
                    flat.extend(val)
                else:
                    flat.extend([None] * dim)
            if has_any:
                grp[param] = flat

    # boundaries
    boundaries = ptcond_data.get("boundaries", [])
    if boundaries:
        grp["boundary_type"] = "complex"

        btypes = []
        for bnd in boundaries:
            btype_str = bnd.get("type", "")
            btypes.append(_BOUNDARY_TYPE_MAP.get(btype_str, 0))
        grp["boundary_types"] = btypes

        for param in _BOUNDARY_SCALAR_PARAMS:
            vals = [bnd.get(param) for bnd in boundaries]
            if any(v is not None for v in vals):
                grp[param] = vals

        for param, dim in _BOUNDARY_MATRIX_PARAMS.items():
            flat = []
            has_any = False
            for bnd in boundaries:
                val = bnd.get(param)
                if val is not None:
                    has_any = True
                    flat.extend(val)
                else:
                    flat.extend([None] * dim)
            if has_any:
                grp[param] = flat

        for param in _BOUNDARY_SPECIES_PARAMS:
            flat = []
            has_any = False
            for bnd in boundaries:
                val = bnd.get(param)
                if val is not None:
                    has_any = True
                    flat.extend(val)
            if has_any:
                grp[param] = flat


# ---------------------------------------------------------------------------
# emissn 展開
# ---------------------------------------------------------------------------


def _flatten_emissn_v2(
    emissn_data: Dict[str, Any], nml: f90nml.Namelist
) -> None:
    grp = _ensure_group(nml, "emissn")

    # 通常パラメータ
    for key, val in emissn_data.items():
        if key != "planes":
            grp[key] = val

    planes = emissn_data.get("planes", [])
    if planes:
        grp["nepl"] = len(planes)
        for param in _EMISSION_PLANE_PARAMS:
            vals = [p.get(param) for p in planes]
            if any(v is not None for v in vals):
                grp[param] = vals


# ---------------------------------------------------------------------------
# 汎用 array-of-tables 展開
# ---------------------------------------------------------------------------


def _flatten_array_of_tables(
    group_data: Dict[str, Any],
    table_key: str,
    nml: f90nml.Namelist,
    group_name: str,
    count_key: str,
    scalar_keys: List[str],
    matrix_keys: Dict[str, int],
) -> None:
    grp = _ensure_group(nml, group_name)

    # 通常パラメータ
    for key, val in group_data.items():
        if key != table_key:
            grp[key] = val

    entries = group_data.get(table_key, [])
    if not entries:
        return

    grp[count_key] = len(entries)

    for param in scalar_keys:
        vals = [e.get(param) for e in entries]
        if any(v is not None for v in vals):
            grp[param] = vals

    for param, dim in matrix_keys.items():
        flat: List[Any] = []
        has_any = False
        for e in entries:
            val = e.get(param)
            if val is not None:
                has_any = True
                flat.extend(val)
            else:
                flat.extend([None] * dim)
        if has_any:
            grp[param] = flat
