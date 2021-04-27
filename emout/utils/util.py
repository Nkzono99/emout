import re
from pathlib import Path

import numpy as np
import scipy.interpolate as interp
from matplotlib.animation import PillowWriter, writers


def interp2d(mesh, n, **kwargs):
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


class RegexDict(dict):
    """正規表現をキーとする辞書クラス.
    """

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)

        for regex in self:
            if re.fullmatch(regex, key):
                return self[regex]

        raise KeyError()

    def __contains__(self, key):
        if super().__contains__(key):
            return True

        for regex in self:
            if re.fullmatch(regex, key):
                return True

        return False

    def get(self, key, default=None):
        try:
            return self[key]
        except Exception:
            return default


class DataFileInfo:
    """データファイル情報を管理するクラス.
    """

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
        return (self._filename / '../').resolve()

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
        return str(self._filename)


@writers.register('quantized-pillow')
class QuantizedPillowWriter(PillowWriter):
    """ 色数を256としたPillowWriterラッパークラス.
    """

    def grab_frame(self, **savefig_kwargs):
        super().grab_frame(**savefig_kwargs)
        self._frames[-1] = self._frames[-1].convert('RGB').quantize()
