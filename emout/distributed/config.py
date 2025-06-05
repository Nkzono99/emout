# emout/distributed/config.py
import os
from pathlib import Path

import psutil


def _get_local_ip():
    try:
        return psutil.net_if_addrs()["ib0"][0].address
    except Exception:
        return "127.0.0.1"


class DaskConfig:
    @property
    def scheduler_ip(self) -> str:
        return os.environ.get("EMOUT_DASK_SCHED_IP", _get_local_ip())

    @property
    def scheduler_port(self) -> int:
        return int(os.environ.get("EMOUT_DASK_SCHED_PORT", "8786"))

    @property
    def partition(self) -> str:
        return os.environ.get("EMOUT_DASK_PARTITION", "gr20001a")

    @property
    def processes(self) -> int:
        return int(os.environ.get("EMOUT_DASK_PROCESSES", "1"))

    @property
    def threads(self) -> int:
        return int(os.environ.get("EMOUT_DASK_THREADS", "1"))

    @property
    def cores(self) -> int:
        return int(os.environ.get("EMOUT_DASK_CORES", "60"))

    @property
    def memory(self) -> str:
        return os.environ.get("EMOUT_DASK_MEMORY", "60G")

    @property
    def walltime(self) -> str:
        return os.environ.get("EMOUT_DASK_WALLTIME", "03:00:00")

    @property
    def env_mods(self) -> list[str]:
        s = os.environ.get("EMOUT_DASK_ENV_MODS", "")
        if not s:
            return []
        return [cmd.strip() for cmd in s.split(";") if cmd.strip()]

    @property
    def logdir(self) -> Path:
        p = os.environ.get("EMOUT_DASK_LOGDIR", "")
        if p:
            return Path(p)
        else:
            return Path.cwd() / "logs" / "dask_logs"
