"""Network and cluster configuration helpers.

Resolves local IP addresses, checks port availability, and manages
the ``~/.emout/server.json`` state file used by the CLI.
"""

# emout/distributed/config.py
from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_DIR = Path.home() / ".emout"
_STATE_FILE = _STATE_DIR / "server.json"


# Linux kernel ARPHRD_* constants (include/uapi/linux/if_arp.h)
_ARPHRD_INFINIBAND = 32
_ARPHRD_ETHER = 1


def _interface_type(name: str) -> int:
    """Read the kernel interface type from ``/sys/class/net/<name>/type``."""
    try:
        return int(Path(f"/sys/class/net/{name}/type").read_text().strip())
    except (OSError, ValueError):
        return -1


def _get_local_ip() -> str:
    """Auto-select an IP address suitable for the Dask scheduler.

    Priority:
      1. Environment variable ``EMOUT_DASK_SCHED_IP`` if set
      2. InfiniBand (kernel type=32) -- HPC high-speed network
      3. Ethernet (kernel type=1) -- management / general-purpose network
      4. 127.0.0.1 (fallback)

    Interface detection uses the kernel ``ARPHRD_*`` constants rather
    than interface names (``ib0``, ``ens2f0``, etc.), so it works
    regardless of the local naming convention.
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
                iftype = _interface_type(name)
                if iftype == _ARPHRD_INFINIBAND:
                    priority = 0  # InfiniBand -- highest priority
                elif iftype == _ARPHRD_ETHER:
                    priority = 1  # Ethernet
                else:
                    priority = 2  # Unknown
                candidates.append((priority, name, a.address))
    except Exception:
        pass

    if not candidates:
        return "127.0.0.1"

    candidates.sort()
    best_name, best_ip = candidates[0][1], candidates[0][2]
    logger.info("Auto-detected scheduler IP: %s (%s, type=%d)", best_ip, best_name, candidates[0][0])
    return best_ip


def _pick_port(ip: str, max_retries: int = 20) -> int:
    """Choose a free port derived from the user's UID.

    Starts with ``10000 + (UID % 50000)`` and increments until a port
    that is not already in use is found.
    """
    base = 10000 + (os.getuid() % 50000)
    for i in range(max_retries):
        port = base + i
        if port > 65535:
            break
        if not _is_port_open(ip, port, timeout=0.3):
            return port
    return base  # fallback: return the original even if probe failed


def _is_port_open(ip: str, port: int, timeout: float = 1.0) -> bool:
    """Check whether a TCP connection to the given IP:port can be established."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


class DaskConfig:
    """Environment-variable-based Dask cluster configuration."""

    @property
    def scheduler_ip(self) -> str:
        return _get_local_ip()

    @property
    def scheduler_port(self) -> int:
        env_port = os.environ.get("EMOUT_DASK_SCHED_PORT")
        if env_port:
            return int(env_port)
        return _pick_port(self.scheduler_ip)

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
