import re
from pathlib import Path
from typing import Callable, Union

import pandas as pd
import scipy.constants as cn

from emout.utils import InpFile, RegexDict, UnitConversionKey, Units, UnitTranslator

from .vector_data import VectorData2d
from .griddata_series import GridDataSeries


def t_unit(out: "Emout") -> UnitTranslator:
    """tの単位変換器を生成する.

    Parameters
    ----------
    out : Emout
        Emoutオブジェクト

    Returns
    -------
    UnitTranslator
        tの単位変換器
    """
    return (out.unit.t * UnitTranslator(out.inp.ifdiag * out.inp.dt, 1)).set_name(
        "t", unit="s"
    )


def wpet_unit(out: "Emout") -> UnitTranslator:
    """wpe * tの単位変換器を生成する.

    以下のコードを実行することで、データのt軸をwpe*tで規格化できる.

    >>> Emout.name2unit['t'] = wpet_unit

    Parameters
    ----------
    out : Emout
        Emoutオブジェクト

    Returns
    -------
    UnitTranslator
        wpe * tの単位変換器
    """
    return UnitTranslator(
        out.inp.wp[0] * out.inp.ifdiag * out.inp.dt, 1, name="wpe * t", unit=""
    )


def wpit_unit(out: "Emout") -> UnitTranslator:
    """wpi * tの単位変換器を生成する.

    以下のコードを実行することで、データのt軸をwpe*tで規格化できる.

    >>> Emout.name2unit['t'] = wpit_unit

    Parameters
    ----------
    out : Emout
        Emoutオブジェクト

    Returns
    -------
    UnitTranslator
        wpi * tの単位変換器
    """
    return UnitTranslator(
        out.inp.wp[1] * out.inp.ifdiag * out.inp.dt, 1, name="wpi * t", unit=""
    )


def none_unit(out: "Emout") -> UnitTranslator:
    return UnitTranslator(1, 1, name="", unit="")


def ndp_unit(ispec: int) -> Callable[["Emout"], UnitTranslator]:
    def ndp_unit(out: "Emout") -> UnitTranslator:
        wp = out.unit.f.reverse(out.inp.wp[ispec])
        mp = abs(cn.m_e / out.inp.qm[ispec])
        np = wp**2 * mp * cn.epsilon_0 / cn.e**2
        return UnitTranslator(np * 1e-6, 1.0, name="number density", unit="/cc")

    return ndp_unit


def nd3p_unit(out: "Emout") -> UnitTranslator:
    wpp = out.unit.f.reverse(out.inp.wp[2])
    np = wpp**2 * cn.m_e * cn.epsilon_0 / cn.e**2
    return UnitTranslator(np * 1e-6, 1.0, name="number density", unit="/cc")


class Emout:
    """EMSES出力・inpファイルを管理する.

    Attributes
    ----------
    directory : Path
        管理するディレクトリ
    dataname : GridData
        3次元データ(datanameは"phisp"などのhdf5ファイルの先頭の名前)
    """

    name2unit = RegexDict(
        {
            r"phisp": lambda self: self.unit.phi,
            # r'nd[12]p': ndp_unit,
            r"nd1p": ndp_unit(0),
            r"nd2p": ndp_unit(1),
            r"nd3p": ndp_unit(2),
            r"nd4p": ndp_unit(3),
            r"nd5p": ndp_unit(4),
            r"nd6p": ndp_unit(5),
            r"nd7p": ndp_unit(6),
            r"nd8p": ndp_unit(7),
            r"nd9p": ndp_unit(8),
            r"rho": lambda self: self.unit.rho,
            r"rhobk": lambda self: self.unit.rho,
            r"j.*": lambda self: self.unit.J,
            r"b[xyz]": lambda self: self.unit.H,
            r"e[xyz]": lambda self: self.unit.E,
            r"t": t_unit,
            r"axis": lambda self: self.unit.length,
            r"rhobksp[1-9]": lambda self: self.unit.rho,
        }
    )

    def __init__(self, directory="./", append_directories=[], inpfilename="plasma.inp"):
        """EMSES出力・inpファイルを管理するオブジェクトを生成する.

        Parameters
        ----------
        directory : str or Path
            管理するディレクトリ, by default './'
        append_directories : list(str) or list(Path) or "auto"
            管理する継続ディレクトリのリスト, by default []
        inpfilename : str, optional
            パラメータファイルの名前, by default 'plasma.inp'
        """
        if not isinstance(directory, Path):
            directory = Path(directory)
        self.directory = directory

        if append_directories == "auto":
            append_directories = self.__fetch_append_directories(directory)

        self.append_directories = []

        for append_directory in append_directories:
            if not isinstance(append_directory, Path):
                append_directory = Path(append_directory)
            self.append_directories.append(append_directory)

        # パラメータファイルの読み取りと単位変換器の初期化
        self._inp = None
        self._unit = None
        if inpfilename is not None and (directory / inpfilename).exists():
            self._inp = InpFile(directory / inpfilename)
            convkey = UnitConversionKey.load(directory / inpfilename)
            if convkey is not None:
                self._unit = Units(dx=convkey.dx, to_c=convkey.to_c)
    
    def __fetch_append_directories(self, directory: Path):
        append_directories = []

        i = 2
        while True:
            path_next = f"{str(directory.resolve())}_{i}"
            directory_next = Path(path_next)

            if not directory_next.exists():
                break

            append_directories.append(directory_next)

            i += 1

        return append_directories

    def __fetch_filepath(self, directory: Path, pattern: str) -> Path:
        filepathes = list(directory.glob(pattern))
        if len(filepathes) == 0:
            raise Exception(f"{pattern} is not found.")
        if len(filepathes) >= 2:
            raise Exception(
                f"There are multiple files that satisfy {pattern}.  Please specify so that just one is specified."
            )

        filepath = filepathes[0]

        return filepath

    def __load_griddata(self, h5file_path: Path) -> "GridDataSeries":
        if self.unit is None:
            tunit = None
            axisunit = None
        else:
            tunit = Emout.name2unit.get("t", lambda self: None)(self)
            axisunit = Emout.name2unit.get("axis", lambda self: None)(self)

        name = str(h5file_path.name).replace("00_0000.h5", "")

        if self.unit is None:
            valunit = None
        else:
            valunit = Emout.name2unit.get(name, lambda self: None)(self)

        data = GridDataSeries(
            h5file_path, name, tunit=tunit, axisunit=axisunit, valunit=valunit
        )

        return data

    def __getattr__(self, __name: str) -> "GridDataSeries":
        m = re.match("(.+)([xyz])([xyz])$", __name)
        if m:
            dname = m.group(1)
            axis1 = m.group(2)
            axis2 = m.group(3)
            vector_data = VectorData2d(
                [getattr(self, f"{dname}{axis1}"), getattr(self, f"{dname}{axis2}")],
                name=__name,
            )

            setattr(self, __name, vector_data)

            return vector_data

        filepath = self.__fetch_filepath(self.directory, f"{__name}00_0000.h5")
        griddata = self.__load_griddata(filepath)

        for append_directory in self.append_directories:
            filepath = self.__fetch_filepath(append_directory, f"{__name}00_0000.h5")
            griddata_append = self.__load_griddata(filepath)

            griddata = griddata.chain(griddata_append)

        setattr(self, __name, griddata)

        return griddata

    @property
    def inp(self) -> Union[InpFile, None]:
        """パラメータの辞書(Namelist)を返す.

        Returns
        -------
        InpFile or None
            パラメータの辞書(Namelist)
        """
        return self._inp

    @property
    def unit(self) -> Union[Units, None]:
        """単位変換オブジェクトを返す.

        Returns
        -------
        Units or None
            単位変換オブジェクト
        """
        return self._unit

    @property
    def icur(self) -> pd.DataFrame:

        names = []
        for ispec in range(self.inp.nspec):
            names.append(f"{ispec+1}_step")
            for ipc in range(self.inp.npc):
                names.append(f"{ispec+1}_body{ipc+1}")
                names.append(f"{ispec+1}_body{ipc+1}_ema")

        df = pd.read_csv(self.directory / "icur", sep="\s+", header=None, names=names)

        return df

    @property
    def pbody(self) -> pd.DataFrame:
        names = ["step"] + [f"body{i+1}" for i in range(self.inp.npc + 1)]

        df = pd.read_csv(self.directory / "pbody", sep="\s+", names=names)

        return df
