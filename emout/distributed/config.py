# emout/distributed/config.py
import json
import logging
import os
import socket
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_DIR = Path.home() / ".emout"
_STATE_FILE = _STATE_DIR / "server.json"


def _get_local_ip() -> str:
    """Dask スケジューラに適した IP を自動選択する。

    優先順:
      1. 環境変数 EMOUT_DASK_SCHED_IP が設定されていればそれを使う
      2. InfiniBand (ib0) — HPC の高速ネットワーク
      3. その他の非ローカルインタフェース (10.x, 192.168.x 等)
      4. 127.0.0.1 (フォールバック)
    """
    env_ip = os.environ.get("EMOUT_DASK_SCHED_IP")
    if env_ip:
        return env_ip

    try:
        import psutil
    except ImportError:
        logger.debug("psutil not available, falling back to 127.0.0.1")
        return "127.0.0.1"

    candidates = []
    try:
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family.name != "AF_INET":
                    continue
                if a.address == "127.0.0.1":
                    continue
                priority = 0 if name.startswith("ib") else 1
                candidates.append((priority, name, a.address))
    except Exception:
        pass

    if not candidates:
        return "127.0.0.1"

    candidates.sort()
    best_name, best_ip = candidates[0][1], candidates[0][2]
    logger.info("Auto-detected scheduler IP: %s (%s)", best_ip, best_name)
    return best_ip


def _is_port_open(ip: str, port: int, timeout: float = 1.0) -> bool:
    """指定 IP:port に TCP 接続できるか簡易チェック。"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


class DaskConfig:
    """環境変数ベースの Dask クラスタ設定。"""

    @property
    def scheduler_ip(self) -> str:
        return _get_local_ip()

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
        return Path.cwd() / "logs" / "dask_logs"
