from pathlib import Path
import re


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
    start = slice_obj.start or 0
    if start < 0:
        start = maxlen + start

    stop = slice_obj.stop or maxlen
    if stop < 0:
        stop = maxlen + stop

    step = slice_obj.step or 1
    return range(start, stop, step)


class RegexDict(dict):
    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)

        for regex in self:
            if re.match(regex, key):
                return self[regex]

        raise KeyError()

    def __contains__(self, key):
        if super().__contains__(key):
            return True

        for regex in self:
            if re.match(regex, key):
                return True

        return False


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
