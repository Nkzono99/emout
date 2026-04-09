"""Dask cluster lifecycle management (start / stop / connect)."""

from .clusters import SimpleDaskCluster
from .config import DaskConfig
from .remote_render import clear_sessions

_global_cluster = None

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
):
    """Start a Dask cluster and return a client.

    Parameters
    ----------
    scheduler_ip : str | None, optional
        IP address of the Dask scheduler.
    scheduler_port : int | None, optional
        Port number of the Dask scheduler.
    partition : str | None, optional
        SLURM partition name for job submission.
    processes : int | None, optional
        Number of processes per worker job.
    threads : int | None, optional
        Number of threads per process.
    cores : int | None, optional
        Total number of cores allocated to a job.
    memory : str | None, optional
        Amount of memory allocated to a job.
    walltime : str | None, optional
        Maximum wall-clock time for a job.
    env_mods : list[str] | None, optional
        Environment modules to load at job start.
    logdir : str | None, optional
        Directory for log output.
    Returns
    -------
    object
        Connected Dask client.
    """
    global _global_cluster
    if _global_cluster is not None:
        return _global_cluster.get_client()

    cfg = DaskConfig()

    # ── Retrieve config values; override with explicit arguments ──
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
    )
    cluster.start_scheduler()
    job_ids = cluster.submit_worker(jobs=1)
    print("Submitted worker job IDs:", job_ids)

    _global_cluster = cluster

    return _global_cluster.get_client()


def connect(address: str | None = None):
    """Connect to a running emout server.

    If *address* is omitted, auto-detect from ``~/.emout/server.json``.

    Parameters
    ----------
    address : str | None, optional
        Dask scheduler address (e.g. ``"tcp://10.0.0.1:32332"``).
        If ``None``, read from the state file written by
        ``emout server start``.

    Returns
    -------
    dask.distributed.Client
        Connected Dask client.
    """
    from dask.distributed import Client
    from pathlib import Path
    import json

    if address is None:
        state_file = Path.home() / ".emout" / "server.json"
        if not state_file.exists():
            raise RuntimeError(
                "emout server is not running. "
                "Run 'emout server start' or specify the address explicitly."
            )
        state = json.loads(state_file.read_text())
        address = state["address"]

    return Client(address)


def stop_cluster(address: str | None = None):
    """Stop a running Dask cluster.

    Parameters
    ----------
    address : str | None, optional
        Scheduler address for a cluster started in another process.

    Returns
    -------
    None
        No return value.
    """
    global _global_cluster
    if _global_cluster is not None:
        _global_cluster.close_client()
        _global_cluster.stop_scheduler()
        _global_cluster = None
        clear_sessions()
        return

    if address is None:
        raise RuntimeError("No active local cluster and no scheduler address was provided.")

    from dask.distributed import Client

    client = Client(address, timeout="5s")
    try:
        client.shutdown()
    finally:
        client.close()
        clear_sessions()
