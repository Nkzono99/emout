"""Parser for EMSES ``plasma.inp`` namelist files.

:class:`InpFile` reads Fortran namelist parameters via :mod:`f90nml` and
exposes them as attributes, handling sparse-array indexing and the
``!!key`` unit-conversion header.
"""

import re

import f90nml

from emout.utils.units import Units


class UnitConversionKey:
    """単位変換キー.

    Attributes
    ----------
    dx : float
        グリッド幅[m]
    to_c : float
        EMSES単位系での光速の値
    """

    def __init__(self, dx, to_c):
        """単位変換キーを生成する.

        Parameters
        ----------
        dx : float
            グリッド幅[m]
        to_c : float
            EMSES単位系での光速の値
        """
        self.dx = dx
        self.to_c = to_c

    @classmethod
    def load(cls, filename):
        """ファイルから単位変換キーをロードする.

        ファイルの一行目に以下のような文字列が書かれている場合dx, to_cを読み取る.
        !!key dx=[1.0],to_c=[10000.0]

        Parameters
        ----------
        filename : str or Path
            単位変換キーを含むファイル.

        Returns
        -------
        UnitConversionKey or None
            単位変換キー
        """
        with open(filename, "r", encoding="utf-8") as f:
            line = f.readline()

        if not line.startswith("!!key"):
            return None

        # "!!key dx=[1.0],to_c=[10000.0]"
        text = line[6:].strip()
        pattern = r"dx=\[([+-]?\d+(?:\.\d+)?)\],to_c=\[([+-]?\d+(?:\.\d+)?)\]"
        m = re.match(pattern, text)
        dx = float(m.group(1))
        to_c = float(m.group(2))
        return UnitConversionKey(dx, to_c)

    @property
    def keytext(self):
        """単位変換キーの文字列を返す.

        Returns
        -------
        str
            単位変換キーの文字列
        """
        return "dx=[{}],to_c=[{}]".format(self.dx, self.to_c)


class AttrDict(dict):
    """AttrDict クラス。
    """
    def __init__(self, *args, **kwargs):
        """インスタンスを初期化します。
        
        Parameters
        ----------
        *args : tuple
            追加の位置引数です。委譲先の関数へそのまま渡されます。
        **kwargs : dict
            追加のキーワード引数です。委譲先の関数へそのまま渡されます。
        """
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        """属性アクセスを解決する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        Returns
        -------
        object
            処理結果です。
        """
        return self[key]


class InpFile:
    """パラメータファイルを管理する.

    nml : f90nml.Namelist
        Namelistオブジェクト
    """

    def __init__(self, filename=None, convkey=None):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        filename : object, optional
            保存先または読み込み対象のファイル名です。
        convkey : object, optional
            単位変換に使う `UnitConversionKey` です。
            `filename` から `!!key` を読み取れた場合はそちらを優先します。
        """
        if filename:
            self.nml = f90nml.read(filename)
            self.convkey = UnitConversionKey.load(filename) or convkey
        else:
            self.nml = f90nml.Namelist()
            self.convkey = convkey

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
        if key in self.nml.keys():
            return True
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    return True
        return False

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
        if key in self.nml.keys():
            return self.nml[key]
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    return self.nml[group][key]
        raise KeyError()

    def __setitem__(self, key, item):
        """要素を設定する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        item : object
            代入または更新する値です。
        Returns
        -------
        None
            戻り値はありません。
        """
        if key in self.nml.keys():
            self.nml[key] = item
            return
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    self.nml[group][key] = item
                    return
        raise KeyError()

    def __getattr__(self, key):
        """属性アクセスを解決する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        Returns
        -------
        object
            処理結果です。
        """
        item = self[key]
        if isinstance(item, dict):
            return AttrDict(item)
        return item

    def __setattr__(self, key, item):
        """属性を設定する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        item : object
            代入または更新する値です。
        Returns
        -------
        None
            戻り値はありません。
        """
        if key in ["nml", "convkey"]:
            super().__setattr__(key, item)
            return

        if key not in self:
            raise KeyError()

        self[key] = item

    def remove(self, key, index=None):
        """パラメータを削除する.

        Parameters
        ----------
        key : str
            グループ名(&groupname)またはパラメータ名(parameter)
        index : int, optional
            特定のインデックスのみ削除する場合指定する, by default None
        """
        if key in self.nml.keys():
            del self.nml[key]
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    try:
                        if index is None:
                            del self.nml[group][key]
                            del self.nml[group].start_index[key]
                        else:
                            (start_index,) = self.nml[group].start_index[key]
                            del self.nml[group][key][index - start_index]

                            if index == start_index:
                                self.nml[group].start_index[key] = [start_index + 1]

                            if len(list(self.nml[group][key])) == 0:
                                del self.nml[group][key]
                                del self.nml[group].start_index[key]
                    except (KeyError, IndexError):
                        pass
                    return

    def setlist(self, group, name, value, start_index=1):
        """リスト型のパラメータを設定する.

        Parameters
        ----------
        group : str
            グループ名(&groupname)
        name : str
            パラメータ名(parameter)
        value : type or list(type)
            設定する値
        start_index : int, optional
            設定するインデックス, by default 1
        """
        if not isinstance(value, list):
            value = [value]

        if name in self.nml[group].start_index:
            end_index = start_index + len(value)

            (start_index_init,) = self.nml[group].start_index[name]
            end_index_init = start_index_init + len(self.nml[group][name])

            min_start_index = min(start_index, start_index_init)
            max_end_index = max(end_index, end_index_init)

            new_list = [None] * (max_end_index - min_start_index)
            for i, index in enumerate(
                range(
                    start_index_init - min_start_index, end_index_init - min_start_index
                )
            ):
                new_list[index] = self.nml[group][name][i]
            for i, index in enumerate(
                range(start_index - min_start_index, end_index - min_start_index)
            ):
                new_list[index] = value[i]

            self.nml[group].start_index[name] = [min_start_index]
            self.nml[group][name] = new_list
        else:
            self.nml[group].start_index[name] = [start_index]
            self.nml[group][name] = value

    def save(self, filename, convkey=None):
        """パラメータをファイルに保存する.

        Parameters
        ----------
        filename : str or Path
            保存するファイル名
        convkey : UnitConversionKey, optional
            単位変換キー, by default None
        """
        convkey = convkey or self.convkey
        with open(filename, "wt", encoding="utf-8") as f:
            if convkey is not None:
                f.write("!!key {}\n".format(convkey.keytext))
            f90nml.write(self.nml, f, force=True)

    def __str__(self):
        """文字列表現を返す。
        
        Returns
        -------
        str
            文字列表現です。
        """
        return str(self.nml)

    def __repr__(self):
        """文字列表現を返す。
        
        Returns
        -------
        str
            文字列表現。
        """
        return str(self)

    def conversion(self, unit_from: Units, unit_to: Units):
        """単位変換を適用する。
        
        Parameters
        ----------
        unit_from : Units
            変換元単位です。
        unit_to : Units
            変換先単位です。
        Returns
        -------
        None
            戻り値はありません。
        """
        def conv(group, name, unit_name):
            """指定パラメータを単位変換して更新する。
            
            Parameters
            ----------
            group : object
                対象の namelist グループ名です。
            name : object
                対象データ名またはキー名です。
            unit_name : object
                参照する単位名です。
            Returns
            -------
            None
                戻り値はありません。
            """
            value_from = self[group][name]

            value = getattr(unit_from, unit_name).reverse(value_from)
            value_to = getattr(unit_to, unit_name).trans(value)

            self[group][name] = value_to

        def conv1d(group, name, unit_name):
            """1 次元配列パラメータを単位変換して更新する。
            
            Parameters
            ----------
            group : object
                対象の namelist グループ名です。
            name : object
                対象データ名またはキー名です。
            unit_name : object
                参照する単位名です。
            Returns
            -------
            None
                戻り値はありません。
            """
            if group not in self:
                return
            if name not in self[group]:
                return

            values_from = self[group][name]

            values_to = []
            for value_from in values_from:
                if value_from is None:
                    value_to = None
                else:
                    value = getattr(unit_from, unit_name).reverse(value_from)
                    value_to = getattr(unit_to, unit_name).trans(value)
                values_to.append(value_to)

            self[group][name] = values_to

        conv1d("plasma", "wp", "f")

        conv1d("intp", "path", "v")
        conv1d("intp", "peth", "v")
        conv1d("intp", "vdri", "v")

        conv1d("emissn", "curf", "J")
        conv1d("emissn", "curfs", "J")

        self.convkey = UnitConversionKey(unit_to.dx, unit_to.to_c)

    @property
    def dx(self) -> float:
        """変換キー `dx` を返す。
        
        Returns
        -------
        float
            処理結果です。
        """
        return self.convkey.dx

    @property
    def to_c(self) -> float:
        """変換キー `to_c` を返す。
        
        Returns
        -------
        float
            処理結果です。
        """
        return self.convkey.to_c
