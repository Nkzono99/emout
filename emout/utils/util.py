"""Miscellaneous helpers: regex dict, file-info parsing, interpolation, and slicing."""

import re
from pathlib import Path
from typing import Union

import numpy as np
import scipy.interpolate as interp
from matplotlib.animation import PillowWriter, writers


def interp2d(mesh, n, **kwargs):
    """2 次元配列上で双線形補間を行う。
    
    Parameters
    ----------
    mesh : object
        描画メッシュ。
    n : object
        サンプル数または格子点数です。
    **kwargs : dict
        追加のキーワード引数。内部で呼び出す関数へ渡されます。
    
    Returns
    -------
    object
        処理結果です。
    """
    ny, nx = mesh.shape

    if (mesh == mesh[0, 0]).all():
        return np.zeros((int(ny * n), int(nx * n))) + mesh[0, 0]

    x_sparse = np.linspace(0, 1, nx)
    y_sparse = np.linspace(0, 1, ny)

    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)

    x_dense = np.linspace(0, 1, int(nx * n))
    y_dense = np.linspace(0, 1, int(ny * n))
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

    points = (X_sparse.flatten(), Y_sparse.flatten())
    value = mesh.flatten()
    points_dense = (X_dense.flatten(), Y_dense.flatten())

    mesh_dense = interp.griddata(points, value, points_dense, **kwargs)

    return mesh_dense.reshape(X_dense.shape)


def slice2tuple(slice_obj: slice):
    """スライスオブジェクトをタプルに変換する.

    Parameters
    ----------
    slice_obj : slice
        スライスオブジェクト

    Returns
    -------
    (start, stop, step) : int
        スライス情報をもつタプル
    """
    start = slice_obj.start
    stop = slice_obj.stop
    step = slice_obj.step
    return (start, stop, step)


def range_with_slice(slice_obj, maxlen):
    """スライスを引数とするrange関数.

    Parameters
    ----------
    slice_obj : slice
        スライスオブジェクト
    maxlen : int
        最大数(スライスの値が負である場合に用いる)

    Returns
    -------
    generator
        rangeジェネレータ
    """
    start = slice_obj.start or 0
    if start < 0:
        start = maxlen + start

    stop = slice_obj.stop or maxlen
    if stop < 0:
        stop = maxlen + stop

    step = slice_obj.step or 1
    return range(start, stop, step)


def apply_offset(
    line: "np.ndarray",
    offset: Union[float, str],
) -> "np.ndarray":
    """Apply a positional offset to a coordinate array.

    Parameters
    ----------
    line : numpy.ndarray
        Coordinate values to shift.
    offset : float or str
        ``"left"`` sets the first element to 0, ``"center"`` centres on
        the middle element, ``"right"`` sets the last element to 0.
        A numeric value is added directly.

    Returns
    -------
    numpy.ndarray
        Shifted array (modified in-place when possible).
    """
    flat = line.ravel()
    if offset == "left":
        line -= flat[0]
    elif offset == "center":
        line -= flat[len(flat) // 2]
    elif offset == "right":
        line -= flat[-1]
    else:
        line += offset
    return line


class RegexDict(dict):
    """正規表現をキーとする辞書クラス."""

    def __getitem__(self, key):
        """要素を取得する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        Returns
        -------
        object
            処理結果です。
        """
        if super().__contains__(key):
            return super().__getitem__(key)

        for regex in self:
            if re.fullmatch(regex, key):
                return self[regex]

        raise KeyError()

    def __contains__(self, key):
        """要素の包含判定を行う。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        Returns
        -------
        object
            処理結果です。
        """
        if super().__contains__(key):
            return True

        for regex in self:
            if re.fullmatch(regex, key):
                return True

        return False

    def get(self, key, default=None):
        """キーに対応する値を取得する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        default : object, optional
            キー未存在時に返す既定値です。
        Returns
        -------
        object
            処理結果です。
        """
        try:
            return self[key]
        except (KeyError, IndexError):
            return default


class DataFileInfo:
    """データファイル情報を管理するクラス."""

    def __init__(self, filename):
        """データファイル情報を管理するオブジェクトを生成する.

        Parameters
        ----------
        filename : str or Path
            ファイル名
        """
        if not isinstance(filename, Path):
            filename = Path(filename)
        self._filename = filename

    @property
    def filename(self):
        """ファイル名を返す.

        Returns
        -------
        Path
            ファイル名
        """
        return self._filename

    @property
    def directory(self):
        """ディレクトリの絶対パスを返す.

        Returns
        -------
        Path
            ディレクトリの絶対パス
        """
        return (self._filename / "../").resolve()

    @property
    def abspath(self):
        """ファイルの絶対パスを返す.

        Returns
        -------
        Path
            ファイルの絶対パス
        """
        return self._filename.resolve()

    def __str__(self):
        """文字列表現を返す。
        
        Returns
        -------
        str
            文字列表現です。
        """
        return str(self._filename)


@writers.register("quantized-pillow")
class QuantizedPillowWriter(PillowWriter):
    """色数を256としたPillowWriterラッパークラス."""

    def grab_frame(self, **savefig_kwargs):
        """フレームを取得して 256 色へ量子化する。
        
        Parameters
        ----------
        **savefig_kwargs : dict
            追加のキーワード引数。内部で呼び出す関数へ渡されます。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        super().grab_frame(**savefig_kwargs)
        self._frames[-1] = self._frames[-1].convert("RGB").quantize()


def hole_mask(inp, reverse=False):
    """矩形ホール領域のマスク配列を生成する。
    
    Parameters
    ----------
    inp : object
        入力パラメータオブジェクトです。
    reverse : bool, optional
        `True` の場合、生成したマスクを反転して返します。
    Returns
    -------
    object
        処理結果です。
    """
    shape = (inp.nz + 1, inp.ny + 1, inp.nx + 1)
    xl = int(inp.xlrechole[0])
    xu = int(inp.xurechole[0])
    yl = int(inp.ylrechole[0])
    yu = int(inp.yurechole[0])
    zu = int(inp.zssurf)
    zl = int(inp.zlrechole[1])

    mask = np.ones(shape, dtype=bool)
    mask[zu:, :, :] = False
    mask[zl:zu, yl : yu + 1, xl : xu + 1] = False
    return (not reverse) == mask
