"""Dask cluster lifecycle management (start / stop / connect)."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any, Mapping

from .clusters import SimpleDaskCluster
from .config import DaskConfig
from .remote_render import clear_sessions
from .security import ensure_cluster_security, load_client_security_from_state
from .server_state import (
    DEFAULT_SERVER_NAME,
    clear_server_state,
    load_server_state,
    normalize_server_name,
)

_global_cluster = None
_WORKER_READY_TIMEOUT = 5.0
_WORKER_READY_POLL = 0.5
_SERVER_STARTUP_GRACE = 30.0


def _client_kwargs_from_state(state: Mapping[str, Any] | None, timeout: str | float) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"timeout": timeout}
    security = load_client_security_from_state(state)
    if security is not None:
        kwargs["security"] = security
    return kwargs


def get_cluster_info(state: Mapping[str, Any], timeout: str | float = "3s") -> dict[str, Any]:
    """Fetch scheduler information for a saved server state."""
    from dask.distributed import Client

    client = Client(state["address"], **_client_kwargs_from_state(state, timeout))
    try:
        return client.scheduler_info()
    finally:
        client.close()


def _pid_is_alive(pid: Any) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _state_started_recently(state: Mapping[str, Any] | None, grace_seconds: float = _SERVER_STARTUP_GRACE) -> bool:
    if state is None:
        return False
    started_at = state.get("started_at")
    if not isinstance(started_at, (int, float)):
        return False
    return (time.time() - float(started_at)) < grace_seconds


def _worker_job_ids(state: Mapping[str, Any] | None) -> list[int]:
    if state is None:
        return []
    values = state.get("worker_job_ids")
    if not isinstance(values, list):
        return []
    job_ids: list[int] = []
    for value in values:
        try:
            job_ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return job_ids


def query_worker_job_states(state: Mapping[str, Any] | None, timeout: float = 3.0) -> dict[int, str] | None:
    """Query SLURM job states for tracked worker jobs.

    Returns ``None`` when job IDs are unavailable or ``squeue`` cannot be used.
    """
    job_ids = _worker_job_ids(state)
    if not job_ids:
        return None

    cmd = ["squeue", "-h", "-j", ",".join(str(job_id) for job_id in job_ids), "-o", "%i|%T"]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    states: dict[int, str] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        job_id_text, _, job_state = line.partition("|")
        try:
            job_id = int(job_id_text.strip())
        except ValueError:
            continue
        states[job_id] = job_state.strip()
    return states


def worker_jobs_active(state: Mapping[str, Any] | None) -> bool | None:
    """Return whether tracked worker jobs still exist in SLURM."""
    states = query_worker_job_states(state)
    if states is None:
        return None
    return bool(states)


def state_lost_workers(state: Mapping[str, Any] | None, info: Mapping[str, Any] | None = None) -> bool:
    """Return ``True`` if the saved server likely lost all worker jobs."""
    if state is None:
        return False
    if info is not None and info.get("workers"):
        return False
    jobs_active = worker_jobs_active(state)
    if jobs_active is False and not _state_started_recently(state):
        return True
    return False


def no_worker_reason(state: Mapping[str, Any] | None, info: Mapping[str, Any] | None = None) -> str:
    """Describe why a scheduler currently has no usable workers."""
    if state_lost_workers(state, info):
        return "Worker jobs are no longer running. They were likely cancelled or timed out."

    jobs_active = worker_jobs_active(state)
    if jobs_active is True:
        return "Scheduler is reachable, but workers have not registered yet."
    if _state_started_recently(state):
        return "Scheduler is reachable, but workers are still starting up."
    return "Scheduler is reachable, but no worker is connected."


def cleanup_saved_server_state(state: Mapping[str, Any], client=None) -> None:
    """Best-effort cleanup for a stale saved server session."""
    try:
        if client is not None:
            client.shutdown()
    except Exception:
        pass

    pid = state.get("pid")
    if isinstance(pid, int) and pid != os.getpid():
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    clear_server_state(state.get("name"))
    clear_sessions()


def ensure_client_has_workers(
    client,
    *,
    state: Mapping[str, Any] | None = None,
    timeout: float = _WORKER_READY_TIMEOUT,
    poll: float = _WORKER_READY_POLL,
) -> dict[str, Any]:
    """Wait briefly for at least one worker, then fail fast if none arrive."""
    deadline = time.monotonic() + timeout
    last_info: dict[str, Any] | None = None

    while True:
        last_info = client.scheduler_info()
        if last_info.get("workers"):
            return last_info
        if time.monotonic() >= deadline:
            break
        time.sleep(poll)

    if state is not None and state_lost_workers(state, last_info):
        cleanup_saved_server_state(state, client=client)
        raise RuntimeError(
            "emout server has no active workers. Worker jobs were likely cancelled or timed out, "
            "so the saved server state was cleared."
        )

    raise RuntimeError(f"emout server has no active workers. {no_worker_reason(state, last_info)}")


def start_cluster(
    scheduler_ip: str | None = None,
    scheduler_port: int | None = None,
    partition: str | None = None,
    processes: int | None = None,
    threads: int | None = None,
    cores: int | None = None,
    memory: str | None = None,
    walltime: str | None = None,
    env_mods: list[str] | None = None,
    logdir: str | None = None,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
    protocol: str | None = None,
    security_files: dict[str, str] | None = None,
):
    """Start a Dask cluster and return a client."""
    global _global_cluster
    if _global_cluster is not None:
        return _global_cluster.get_client()

    cfg = DaskConfig()

    ip = scheduler_ip if scheduler_ip is not None else cfg.scheduler_ip
    port = scheduler_port if scheduler_port is not None else cfg.scheduler_port
    part = partition if partition is not None else cfg.partition
    p = processes if processes is not None else cfg.processes
    t = threads if threads is not None else cfg.threads
    c = cores if cores is not None else cfg.cores
    m = memory if memory is not None else cfg.memory
    wt = walltime if walltime is not None else cfg.walltime
    emods = env_mods if env_mods is not None else cfg.env_mods
    ld = logdir if logdir is not None else str(cfg.logdir)
    server_name = normalize_server_name(server_name)
    cluster_protocol = protocol if protocol is not None else cfg.protocol

    if cluster_protocol == "tls" and security_files is None:
        security = ensure_cluster_security(server_name=server_name, scheduler_host=ip)
        security_files = security.cluster_kwargs()

    cluster = SimpleDaskCluster(
        scheduler_ip=ip,
        scheduler_port=port,
        partition=part,
        processes=p,
        threads=t,
        cores=c,
        memory=m,
        walltime=wt,
        env_mods=emods,
        logdir=ld,
        protocol=cluster_protocol,
        security_files=security_files,
    )
    cluster.start_scheduler()
    job_ids = cluster.submit_worker(jobs=1)
    print("Submitted worker job IDs:", job_ids)

    _global_cluster = cluster

    client = _global_cluster.get_client()
    setattr(client, "_emout_worker_job_ids", list(job_ids))
    return client


def connect(
    address: str | None = None,
    *,
    name: str | None = None,
    timeout: str | float = "5s",
    security=None,
    require_workers: bool = False,
    worker_timeout: float = _WORKER_READY_TIMEOUT,
):
    """Connect to a running emout server.

    If *address* is omitted, auto-detect from the saved server state.
    """
    from dask.distributed import Client

    state = None
    if address is None:
        lookup_name = normalize_server_name(name) if name is not None else None
        state = load_server_state(lookup_name)
        if state is None:
            raise RuntimeError(
                "emout server is not running. Run 'emout server start' or specify the address explicitly."
            )
        address = state["address"]
    elif security is None and (name is not None or str(address).startswith("tls://")):
        lookup_name = normalize_server_name(name) if name is not None else None
        state = load_server_state(lookup_name)

    if security is None:
        security = load_client_security_from_state(state)

    try:
        if security is None:
            client = Client(address)
        else:
            client = Client(address, timeout=timeout, security=security)
    except Exception as exc:
        if state is not None and (state_lost_workers(state) or not _pid_is_alive(state.get("pid"))):
            cleanup_saved_server_state(state)
            raise RuntimeError(
                "Saved emout server state was stale and has been cleared. Start the server again."
            ) from exc
        raise

    if not require_workers:
        return client

    try:
        ensure_client_has_workers(client, state=state, timeout=worker_timeout)
    except Exception:
        client.close()
        raise
    return client


def stop_cluster(
    address: str | None = None,
    *,
    name: str | None = None,
    state: Mapping[str, Any] | None = None,
    timeout: str | float = "5s",
):
    """Stop a running Dask cluster."""
    global _global_cluster
    if _global_cluster is not None:
        _global_cluster.close_client()
        _global_cluster.stop_scheduler()
        _global_cluster = None
        clear_sessions()
        return

    if state is None and (name is not None or address is None or str(address).startswith("tls://")):
        lookup_name = normalize_server_name(name) if name is not None else None
        state = load_server_state(lookup_name)

    if address is None:
        if state is None:
            raise RuntimeError("No active local cluster and no scheduler address was provided.")
        address = state["address"]

    client = connect(address, name=name, timeout=timeout, security=load_client_security_from_state(state))
    try:
        client.shutdown()
    finally:
        client.close()
        clear_sessions()
