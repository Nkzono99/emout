# emout/io/grid.py

import logging
import re
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm

from ..data.griddata_series import GridDataSeries
from ..data.vector_data import VectorData2d
from ..relocation.electric import relocated_electric_field
from ..relocation.magnetic import relocated_magnetic_field
from .directory import DirectoryInspector

logger = logging.getLogger(__name__)


def get_tqdm():
    """適切な tqdm を返すユーティリティ関数."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            logger.debug("Notebook 環境検出")
            return notebook_tqdm
        else:
            logger.debug("IPython 環境（Notebook ではない）検出")
            return tqdm
    except NameError:
        logger.debug("標準 Python 環境検出")
        return tqdm


tqdm = get_tqdm()


class GridDataLoader:
    """
    Emout の __getattr__ 相当のロジックを担うクラス。
    - GridDataSeries / VectorData2d の生成
    - relocated field (.create) の呼び出し
    """

    def __init__(
        self, dir_inspector: DirectoryInspector, name2unit_map: dict[str, Any]
    ):
        self.dir_inspector = dir_inspector
        self.name2unit_map = name2unit_map

    def load(self, name: str) -> Any:
        """
        - r[e/b][xyz] の形式 → relocated field を生成
        - (dname)(axis1)(axis2) の形式 → VectorData2d を返す
        - それ以外 → GridDataSeries をチェーンして返す
        """
        logger.debug(f"GridDataLoader.load: {name}")

        m = re.match(r"^r([eb][xyz])$", name)
        if m:
            fld = m.group(1)  # ex: 'ebx', 'exy' など
            logger.debug(f"Relocated field requested: {fld}")
            self._create_relocated_field_hdf5(fld)

        m2 = re.match(r"(.+)([xyz])([xyz])$", name)
        if m2:
            dname, axis1, axis2 = m2.groups()
            logger.debug(f"VectorData2d を生成: base={dname}, axes=({axis1},{axis2})")
            arr1 = self.load(
                f"{dname}{axis1}"
            )  # 再帰的に GridDataSeries or relocated field
            arr2 = self.load(f"{dname}{axis2}")
            vd = VectorData2d([arr1, arr2], name=name)
            return vd

        main_fp = self._find_h5file(self.dir_inspector.main_directory, name)
        logger.info(f"Loading grid data from: {main_fp.resolve()}")
        gd = self._load_griddata(main_fp)

        for ad in self.dir_inspector.append_directories:
            try:
                fp_append = self._find_h5file(ad, name)
            except FileNotFoundError:
                continue
            gd_append = self._load_griddata(fp_append)
            gd = gd.chain(gd_append)

        return gd

    def _find_h5file(self, directory: Path, name: str) -> Path:
        pattern = f"{name}00_0000.h5"
        matches = list(directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"{pattern} が見つかりません: {directory}")
        if len(matches) > 1:
            raise RuntimeError(f"{pattern} が複数見つかりました: {directory}")
        return matches[0]

    def _load_griddata(self, h5file_path: Path) -> GridDataSeries:
        # unit が定義されていれば name2unit から拾う
        unit = self.dir_inspector.unit
        if unit is None:
            tunit = None
            axisunit = None
            valunit = None
        else:
            # "t", "axis" は常に key がある想定
            tunit = self.name2unit_map.get("t", lambda out: None)(self.dir_inspector)
            axisunit = self.name2unit_map.get("axis", lambda out: None)(
                self.dir_inspector
            )
            # 実際の値のユニットはファイル名から抽出
            base_name = h5file_path.name.replace("00_0000.h5", "")
            valunit = self.name2unit_map.get(base_name, lambda out: None)(
                self.dir_inspector
            )

        return GridDataSeries(
            h5file_path,
            h5file_path.name.replace("00_0000.h5", ""),
            tunit=tunit,
            axisunit=axisunit,
            valunit=valunit,
        )

    def _create_relocated_field_hdf5(self, field_name: str) -> None:
        axis = "zyx".index(field_name[-1])
        if field_name.startswith("b"):
            relocated_fn = relocated_magnetic_field
        else:
            relocated_fn = relocated_electric_field

        main_dir = self.dir_inspector.main_directory
        self._create_one_relocated(main_dir, field_name, axis, relocated_fn)

        for ad in self.dir_inspector.append_directories:
            self._create_one_relocated(ad, field_name, axis, relocated_fn)

    def _create_one_relocated(
        self, directory: Path, name: str, axis: int, relocated_fn: Callable
    ) -> None:
        input_fp = directory / f"{name}00_0000.h5"
        output_fp = directory / f"r{name}00_0000.h5"

        if output_fp.exists():
            logger.debug(f"既に存在: {output_fp.resolve()}")
            return

        logger.info(f"Relocated field 作成: {output_fp.resolve()}")
        with h5py.File(input_fp, "r") as h5f_in:
            field = h5f_in[name]

            with h5py.File(output_fp, "w") as h5f_out:
                rgrp = h5f_out.create_group(f"r{name}")

                for key in tqdm(field.keys(), desc=f"Relocating {name}"):
                    arr = np.array(field[key])

                    rgrp[key] = relocated_fn(
                        arr, axis=axis, btype=self._get_btype(name, directory)
                    )

    def _get_btype(self, name: str, directory: Path) -> str:
        axis = "zyx".index(name[-1])
        btype_list = ["periodic", "dirichlet", "neumann"]

        return btype_list[self.dir_inspector.inp.mtd_vbnd[2 - axis]]
