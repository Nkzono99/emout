from .clusters import SimpleDaskCluster
from .config import DaskConfig


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
    cfg = DaskConfig()
  
    # ── config の内容を取得。引数が None でなければ上書きする ──
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

    return cluster


def stop_cluster(cluster):
    cluster.close_client()
    cluster.stop_scheduler()
