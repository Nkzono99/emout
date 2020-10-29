import re

import f90nml


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
        with open(filename, 'r', encoding='utf-8') as f:
            line = f.readline()

        if not line.startswith('!!key'):
            return None

        # !!key dx=[1.0],to_c=[10000.0]
        text = line[6:].strip()
        pattern = r'dx=\[([+-]?\d+(?:\.\d+)?)\],to_c=\[([+-]?\d+(?:\.\d+)?)\]'
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
        return 'dx=[{}],to_c=[{}]'.format(self.dx, self.to_c)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        return self[key]


class InpFile:
    """パラメータファイルを管理する.

    nml : f90nml.Namelist
        Namelistオブジェクト
    """

    def __init__(self, filename):
        self.nml = f90nml.read(filename)

    def __getitem__(self, key):
        if key in self.nml.keys():
            return self.nml[key]
        else:
            for group in self.nml.keys():
                if key in self.nml[group].keys():
                    return self.nml[group][key]
        raise KeyError()

    def __getattr__(self, key):
        item = self[key]
        if isinstance(item, dict):
            return AttrDict(item)
        return item

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
                            start_index, = self.nml[group].start_index[key]
                            del self.nml[group][key][index - start_index]

                            if index == start_index:
                                self.nml[group].start_index[key] = [
                                    start_index + 1]

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

            start_index_init, = self.nml[group].start_index[name]
            end_index_init = start_index_init + len(self.nml[group][name])

            min_start_index = min(start_index, start_index_init)
            max_end_index = max(end_index, end_index_init)

            new_list = [None] * (max_end_index - min_start_index)
            for i, index in enumerate(range(start_index_init-min_start_index, end_index_init-min_start_index)):
                new_list[index] = self.nml[group][name][i]
            for i, index in enumerate(range(start_index-min_start_index, end_index-min_start_index)):
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
        with open(filename, 'wt', encoding='utf-8') as f:
            if convkey is not None:
                f.write('!!key {}\n'.format(convkey.keytext))
            f90nml.write(self.nml, f, force=True)

    def __str__(self):
        return str(self.nml)

    def __repr__(self):
        return str(self)
