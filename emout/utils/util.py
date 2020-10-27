from pathlib import Path


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
