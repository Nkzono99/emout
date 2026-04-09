from .clusters import SimpleDaskCluster
from .config import DaskConfig

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
    """Dask クラスタを起動してクライアントを返す。
    
    Parameters
    ----------
    scheduler_ip : str | None, optional
        Dask スケジューラの IP アドレスです。
    scheduler_port : int | None, optional
        Dask スケジューラのポート番号です。
    partition : str | None, optional
        投入先ジョブパーティション名です。
    processes : int | None, optional
        ワーカージョブごとのプロセス数です。
    threads : int | None, optional
        1 プロセスあたりのスレッド数です。
    cores : int | None, optional
        ジョブに割り当てる総コア数です。
    memory : str | None, optional
        ジョブに割り当てるメモリ量です。
    walltime : str | None, optional
        ジョブの実行時間上限です。
    env_mods : list[str] | None, optional
        ジョブ開始時に読み込む環境モジュールです。
    logdir : str | None, optional
        ログ出力先ディレクトリです。
    Returns
    -------
    object
        処理結果です。
    """
    global _global_cluster
    if _global_cluster is not None:
        return _global_cluster.get_client()

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
    
    _global_cluster = cluster

    return _global_cluster.get_client()


def connect(address: str | None = None):
    """起動済みの emout server に接続する。

    ``address`` を省略すると ``~/.emout/server.json`` から自動検出する。

    Parameters
    ----------
    address : str | None, optional
        Dask スケジューラのアドレス (例: ``"tcp://10.0.0.1:32332"``)。
        ``None`` の場合は ``emout server start`` が書き出した状態ファイルから読む。

    Returns
    -------
    dask.distributed.Client
        接続済みの Dask クライアント。
    """
    from dask.distributed import Client
    from pathlib import Path
    import json

    if address is None:
        state_file = Path.home() / ".emout" / "server.json"
        if not state_file.exists():
            raise RuntimeError(
                "emout server が起動していません。"
                "'emout server start' を実行するか、address を明示してください。"
            )
        state = json.loads(state_file.read_text())
        address = state["address"]

    return Client(address)


def stop_cluster():
    """起動済み Dask クラスタを停止する。
    
    Returns
    -------
    None
        戻り値はありません。
    """
    global _global_cluster
    _global_cluster.close_client()
    _global_cluster.stop_scheduler()
