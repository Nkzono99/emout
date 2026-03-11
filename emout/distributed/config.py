# emout/distributed/config.py
import os
from pathlib import Path

import psutil


def _get_local_ip():
    """local ip を取得する。
    
    Returns
    -------
    object
        処理結果です。
    """
    try:
        return psutil.net_if_addrs()["ib0"][0].address
    except Exception:
        return "127.0.0.1"


class DaskConfig:
    """DaskConfig クラス。
    """
    @property
    def scheduler_ip(self) -> str:
        """Dask スケジューラの IP 設定を返す。
        
        Returns
        -------
        str
            文字列表現です。
        """
        return os.environ.get("EMOUT_DASK_SCHED_IP", _get_local_ip())

    @property
    def scheduler_port(self) -> int:
        """Dask スケジューラのポート設定を返す。
        
        Returns
        -------
        int
            件数または index を返します。
        """
        return int(os.environ.get("EMOUT_DASK_SCHED_PORT", "8786"))

    @property
    def partition(self) -> str:
        """ジョブ投入先パーティション設定を返す。
        
        Returns
        -------
        str
            文字列表現です。
        """
        return os.environ.get("EMOUT_DASK_PARTITION", "gr20001a")

    @property
    def processes(self) -> int:
        """ワーカーのプロセス数設定を返す。
        
        Returns
        -------
        int
            件数または index を返します。
        """
        return int(os.environ.get("EMOUT_DASK_PROCESSES", "1"))

    @property
    def threads(self) -> int:
        """ワーカーのスレッド数設定を返す。
        
        Returns
        -------
        int
            件数または index を返します。
        """
        return int(os.environ.get("EMOUT_DASK_THREADS", "1"))

    @property
    def cores(self) -> int:
        """ワーカーの総コア数設定を返す。
        
        Returns
        -------
        int
            件数または index を返します。
        """
        return int(os.environ.get("EMOUT_DASK_CORES", "60"))

    @property
    def memory(self) -> str:
        """ワーカーのメモリ割り当て設定を返す。
        
        Returns
        -------
        str
            文字列表現です。
        """
        return os.environ.get("EMOUT_DASK_MEMORY", "60G")

    @property
    def walltime(self) -> str:
        """ジョブの実行時間上限設定を返す。
        
        Returns
        -------
        str
            文字列表現です。
        """
        return os.environ.get("EMOUT_DASK_WALLTIME", "03:00:00")

    @property
    def env_mods(self) -> list[str]:
        """起動時に読み込む環境モジュール設定を返す。
        
        Returns
        -------
        list[str]
            処理結果です。
        """
        s = os.environ.get("EMOUT_DASK_ENV_MODS", "")
        if not s:
            return []
        return [cmd.strip() for cmd in s.split(";") if cmd.strip()]

    @property
    def logdir(self) -> Path:
        """ログ出力ディレクトリ設定を返す。
        
        Returns
        -------
        Path
            処理結果です。
        """
        p = os.environ.get("EMOUT_DASK_LOGDIR", "")
        if p:
            return Path(p)
        else:
            return Path.cwd() / "logs" / "dask_logs"
