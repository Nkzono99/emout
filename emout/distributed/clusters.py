#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_dask_cluster.py

京大スパコン(SLURM)向けに、Python コードから
- Dask Scheduler の起動
- Dask Worker の sbatch 投入
- Client 接続
をすべて自動的に行えるようにするミニマルなラッパーライブラリです。

使い方(例):

.. code-block:: python

    from simple_dask_cluster import SimpleDaskCluster

    # クラスを生成
    cluster = SimpleDaskCluster(
        scheduler_ip="10.10.64.1",
        scheduler_port=8786,
        partition="gr20001a",
        processes=1,
        threads=1,
        cores=1,
        memory="4G",
        walltime="01:00:00",
        env_mods=["module load Anaconda3", "conda activate dask_env"],
        logdir="/home/b/b36291/large0/exp_dipole/logs",
        sbatch_extra=None,  # 追加 sbatch オプションがあればリストで渡す
    )

    # Scheduler をバックグラウンドで起動
    cluster.start_scheduler()
    # Worker を複数投入(ここでは 2 台)
    cluster.submit_worker(jobs=2)
    # Client を取得して分散計算を実行
    client = cluster.get_client()
    # ⇒ たとえば dask.array を使った処理を client.compute() で呼べる

    # 最後に後始末
    client.close()
    cluster.stop_scheduler()
    # (SLURM ジョブ自体は SLURM の期限(walltime)が来るか、scancel で落とす)
"""

import os
import subprocess
import time
from pathlib import Path

from dask.distributed import Client


class SimpleDaskCluster:
    """
    Dask Scheduler と Worker(sbatch)を Python でまとめて管理するクラス。
    """

    def __init__(
        self,
        scheduler_ip: str,
        scheduler_port: int = 8786,
        partition: str = "gr20001a",
        processes: int = 1,
        threads: int = 1,
        cores: int = 1,
        memory: str = "4G",
        walltime: str = "01:00:00",
        env_mods: list[str] | None = None,
        logdir: str | Path | None = None,
        sbatch_extra: list[str] | None = None,
    ):
        """
        Parameters
        ----------
        scheduler_ip : str
            Dask Scheduler を bind する計算ノードの IP アドレス
        scheduler_port : int, default=8786
            Dask Scheduler が待ち受ける TCP ポート番号
        partition : str, default="gr20001a"
            SLURM のパーティション名(例: "gr20001a" など)
        processes : int, default=1
            sbatch で投げる dask-worker の「プロセス数」(p=… に対応)
        threads : int, default=1
            dask-worker の --nthreads に対応
        cores : int, default=1
            sbatch のリソース指定 c=… に対応
        memory : str, default="4G"
            sbatch のリソース指定 m=… に対応 (例: "4G", "8000M" など)
        walltime : str, default="01:00:00"
            sbatch の実行時間 (hh:mm:ss)
        env_mods : list[str] | None, default=None
            ジョブ実行時に実行するシェルコマンド(例: ["module load Anaconda3", "conda activate dask_env"])
        logdir : str | Path | None, default=None
            SLURM ジョブの標準出力・標準エラーを置くディレクトリ。
            None の場合はカレントディレクトリ
        sbatch_extra : list[str] | None, default=None
            sbatch に追加で渡したいオプション (例: ["--mem-per-cpu=2000M"])
        """
        self.scheduler_ip = scheduler_ip
        self.scheduler_port = scheduler_port
        self.partition = partition
        self.processes = processes
        self.threads = threads
        self.cores = cores
        self.memory = memory
        self.walltime = walltime
        self.env_mods = env_mods or []
        self.sbatch_extra = sbatch_extra or []

        # ログディレクトリ
        if logdir is None:
            logdir = Path.cwd() / "dask_logs"
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        # Scheduler プロセス (subprocess.Popen)
        self._sched_proc: subprocess.Popen | None = None
        # 現在投入した Worker ジョブの JOB ID を保持
        self.worker_job_ids: list[int] = []

        # Client オブジェクト (後で生成)
        self._client: Client | None = None

    def start_scheduler(self, no_dashboard: bool = True):
        """
        バックグラウンドで dask-scheduler を立ち上げる。
        `self._sched_proc` に Popen オブジェクトを保持する。
        """

        if self._sched_proc is not None and self._sched_proc.poll() is None:
            # すでに起動中
            print("[SimpleDaskCluster] Scheduler is already running.")
            return

        cmd = [
            "dask",
            "scheduler",
            "--host",
            self.scheduler_ip,
            "--port",
            str(self.scheduler_port),
        ]
        if no_dashboard:
            cmd.append("--no-dashboard")

        # 出力をログファイルに書き出す
        sched_out = self.logdir / "scheduler.out"
        sched_err = self.logdir / "scheduler.err"

        print(f"[SimpleDaskCluster] Starting scheduler: {' '.join(cmd)}")
        with open(sched_out, "a") as fo, open(sched_err, "a") as fe:
            # Popen でバックグラウンド起動
            self._sched_proc = subprocess.Popen(
                cmd,
                stdout=fo,
                stderr=fe,
                text=True,
                bufsize=1,
            )

        # 少し待って、起動できたかチェック
        time.sleep(1.0)
        if self._sched_proc.poll() is not None:
            raise RuntimeError(
                f"Scheduler failed to start. See {sched_err} for details."
            )
        print("[SimpleDaskCluster] Scheduler started successfully.")

    def stop_scheduler(self):
        """
        起動中の Scheduler を停止する (kill)。
        """
        if self._sched_proc is None:
            print("[SimpleDaskCluster] No scheduler process to stop.")
            return
        if self._sched_proc.poll() is not None:
            print("[SimpleDaskCluster] Scheduler is not running.")
            return

        print(
            f"[SimpleDaskCluster] Terminating scheduler (pid={self._sched_proc.pid}) ..."
        )
        self._sched_proc.terminate()
        try:
            self._sched_proc.wait(timeout=5)
            print("[SimpleDaskCluster] Scheduler terminated.")
        except subprocess.TimeoutExpired:
            print("[SimpleDaskCluster] Scheduler did not exit; killing ...")
            self._sched_proc.kill()
            self._sched_proc.wait()
            print("[SimpleDaskCluster] Scheduler killed.")
        finally:
            self._sched_proc = None

    def submit_worker(self, jobs: int = 1):
        """
        Worker (dask-worker) を sbatch で投げる。
        `jobs` の数だけ SLURM ジョブを投入し、それぞれの JOBID を返す。
        """
        if self._sched_proc is None:
            raise RuntimeError(
                "Scheduler is not running. Call start_scheduler() first."
            )

        new_job_ids: list[int] = []
        for _ in range(jobs):
            # sbatch 用スクリプトを一時ディレクトリに書き出す
            sbatch_script = self._generate_worker_script()
            job_submit_cmd = ["sbatch", str(sbatch_script)]
            # 環境変数 DASK_SCHED_IP を与えて sbatch する
            env = os.environ.copy()
            env["DASK_SCHED_IP"] = self.scheduler_ip

            print(
                f"[SimpleDaskCluster] Submitting worker job: {' '.join(job_submit_cmd)}"
            )
            completed = subprocess.run(
                job_submit_cmd, capture_output=True, text=True, env=env
            )
            if completed.returncode != 0:
                raise RuntimeError(f"sbatch failed: {completed.stderr.strip()}")

            # sbatch から返ってくる標準出力例: "Submitted batch job 123456"
            stdout = completed.stdout.strip()
            parts = stdout.split()
            try:
                job_id = int(parts[-1])
            except Exception:
                raise RuntimeError(
                    f"Could not parse job ID from sbatch output: {stdout}"
                )
            print(f"[SimpleDaskCluster] Worker job submitted; JOBID={job_id}")
            new_job_ids.append(job_id)

        self.worker_job_ids.extend(new_job_ids)
        return new_job_ids

    def _generate_worker_script(self) -> Path:
        """
        Worker 用の sbatch スクリプトをテンポラリファイルとして書き出す。
        返り値はスクリプトファイルのパス (Path)。
        """
        script_lines: list[str] = []

        # ヘッダー
        script_lines.append("#!/bin/bash")
        script_lines.append(f"#SBATCH -p {self.partition}")
        script_lines.append(
            f"#SBATCH --rsc p={self.processes}:t={self.threads}:c={self.cores}:m={self.memory}"
        )
        script_lines.append(f"#SBATCH -t {self.walltime}")
        script_lines.append(f"#SBATCH -J dask-worker")
        script_lines.append(f"#SBATCH -o {self.logdir}/worker_%J.out")
        script_lines.append(f"#SBATCH -e {self.logdir}/worker_%J.err")

        # 追加 sbatch オプションがあれば
        for extra in self.sbatch_extra:
            script_lines.append(extra)

        script_lines.append("")  # 空行

        # 環境モジュールや conda activate
        for cmd in self.env_mods:
            script_lines.append(cmd)
        script_lines.append("")  # 空行

        # Scheduler の IP は環境変数 DASK_SCHED_IP から読む
        script_lines.append("HOST=${DASK_SCHED_IP}")
        script_lines.append(f"PORT={self.scheduler_port}")
        script_lines.append("")

        # dask-worker コマンド本体
        script_lines.append("dask worker tcp://${HOST}:${PORT} \\")
        script_lines.append(f"    --nthreads {self.threads} \\")
        script_lines.append(f"    --no-dashboard --memory-limit {self.memory}")
        script_lines.append("")

        # 最後に日付表示しておく
        script_lines.append("date")

        # ファイルに書き出し
        tmpdir = Path("/tmp/simple_dask_workers")
        tmpdir.mkdir(parents=True, exist_ok=True)

        # 一意のファイル名を作成
        script_path = tmpdir / f"worker_{int(time.time()*1000)}.sh"
        with open(script_path, "w") as f:
            f.write("\n".join(script_lines))

        # sbatch で実行できるように実行権限を付与
        script_path.chmod(0o744)
        return script_path

    def get_client(self, timeout: float = 30.0) -> Client:
        """
        dask.distributed.Client を返す。初回呼び出し時に実際に接続を試みる。
        Scheduler にワーカーがまだつながっていなくても、
        Client が自動的にリトライしてくれる設計です。
        """
        if self._sched_proc is None:
            raise RuntimeError("Scheduler is not running.")

        if self._client is not None:
            return self._client

        sched_addr = f"tcp://{self.scheduler_ip}:{self.scheduler_port}"
        print(f"[SimpleDaskCluster] Connecting Dask Client to {sched_addr} ...")

        t0 = time.time()
        while True:
            try:
                self._client = Client(sched_addr, timeout=timeout)
                break
            except Exception as e:
                elapsed = time.time() - t0
                if elapsed > timeout:
                    raise RuntimeError(
                        f"Could not connect to scheduler at {sched_addr}: {e}"
                    )
                print(
                    "[SimpleDaskCluster] Waiting for scheduler to accept connections ..."
                )
                time.sleep(1)

        print("[SimpleDaskCluster] Dask Client connected.")
        return self._client

    def close_client(self):
        """
        もし Client が生きていれば close する。
        """
        if self._client:
            self._client.close()
            self._client = None
