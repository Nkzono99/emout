#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_dask_cluster.py

Minimal wrapper library for automatically performing the following from
Python code, targeting SLURM-based supercomputers:
- Start a Dask Scheduler
- Submit Dask Workers via sbatch
- Connect a Client

Usage example:

.. code-block:: python

    from simple_dask_cluster import SimpleDaskCluster

    # Create the cluster object
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
        sbatch_extra=None,  # Pass additional sbatch options as a list
    )

    # Start the scheduler in the background
    cluster.start_scheduler()
    # Submit multiple workers (here 2)
    cluster.submit_worker(jobs=2)
    # Get a client and run distributed computations
    client = cluster.get_client()
    # e.g. call client.compute() with dask.array operations

    # Clean up
    client.close()
    cluster.stop_scheduler()
    # (SLURM jobs expire at walltime or can be cancelled with scancel)
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from dask.distributed import Client


class SimpleDaskCluster:
    """
    Manage a Dask Scheduler and Workers (sbatch) together from Python.
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
            IP address of the compute node to bind the Dask Scheduler.
        scheduler_port : int, default=8786
            TCP port for the Dask Scheduler.
        partition : str, default="gr20001a"
            SLURM partition name (e.g. "gr20001a").
        processes : int, default=1
            Number of processes per dask-worker sbatch job (p=...).
        threads : int, default=1
            Corresponds to dask-worker --nthreads.
        cores : int, default=1
            SLURM resource specification c=... .
        memory : str, default="4G"
            SLURM resource specification m=... (e.g. "4G", "8000M").
        walltime : str, default="01:00:00"
            SLURM walltime (hh:mm:ss).
        env_mods : list[str] | None, default=None
            Shell commands to run at job start (e.g. ["module load Anaconda3", "conda activate dask_env"]).
        logdir : str | Path | None, default=None
            Directory for SLURM job stdout/stderr.
            Uses the current directory if None.
        sbatch_extra : list[str] | None, default=None
            Additional sbatch options (e.g. ["--mem-per-cpu=2000M"]).
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

        # Log directory
        if logdir is None:
            logdir = Path.cwd() / "dask_logs"
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        # Scheduler process (subprocess.Popen)
        self._sched_proc: subprocess.Popen | None = None
        # Track submitted worker JOB IDs
        self.worker_job_ids: list[int] = []

        # Client object (created later)
        self._client: Client | None = None

    def start_scheduler(self, no_dashboard: bool = True):
        """
        Start dask-scheduler in the background.
        Store the Popen object in ``self._sched_proc``.
        """

        if self._sched_proc is not None and self._sched_proc.poll() is None:
            # Already running
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

        # Write output to log files
        sched_out = self.logdir / "scheduler.out"
        sched_err = self.logdir / "scheduler.err"

        print(f"[SimpleDaskCluster] Starting scheduler: {' '.join(cmd)}")
        with open(sched_out, "a") as fo, open(sched_err, "a") as fe:
            # Start in the background with Popen
            self._sched_proc = subprocess.Popen(
                cmd,
                stdout=fo,
                stderr=fe,
                text=True,
                bufsize=1,
            )

        # Wait briefly and check whether it started successfully
        time.sleep(1.0)
        if self._sched_proc.poll() is not None:
            raise RuntimeError(f"Scheduler failed to start. See {sched_err} for details.")
        print("[SimpleDaskCluster] Scheduler started successfully.")

    def stop_scheduler(self):
        """
        Stop the running Scheduler (kill).
        """
        if self._sched_proc is None:
            print("[SimpleDaskCluster] No scheduler process to stop.")
            return
        if self._sched_proc.poll() is not None:
            print("[SimpleDaskCluster] Scheduler is not running.")
            return

        print(f"[SimpleDaskCluster] Terminating scheduler (pid={self._sched_proc.pid}) ...")
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
        Submit Workers (dask-worker) via sbatch.
        Submit *jobs* SLURM jobs and return their JOBIDs.
        """
        if self._sched_proc is None:
            raise RuntimeError("Scheduler is not running. Call start_scheduler() first.")

        new_job_ids: list[int] = []
        for _ in range(jobs):
            # Write the sbatch script to a temporary directory
            sbatch_script = self._generate_worker_script()
            job_submit_cmd = ["sbatch", str(sbatch_script)]
            # Pass the scheduler IP via environment variable
            env = os.environ.copy()
            env["DASK_SCHED_IP"] = self.scheduler_ip

            print(f"[SimpleDaskCluster] Submitting worker job: {' '.join(job_submit_cmd)}")
            completed = subprocess.run(job_submit_cmd, capture_output=True, text=True, env=env)
            if completed.returncode != 0:
                raise RuntimeError(f"sbatch failed: {completed.stderr.strip()}")

            # Parse JOBID from sbatch stdout, e.g. "Submitted batch job 123456"
            stdout = completed.stdout.strip()
            parts = stdout.split()
            try:
                job_id = int(parts[-1])
            except Exception:
                raise RuntimeError(f"Could not parse job ID from sbatch output: {stdout}")
            print(f"[SimpleDaskCluster] Worker job submitted; JOBID={job_id}")
            new_job_ids.append(job_id)

        self.worker_job_ids.extend(new_job_ids)
        return new_job_ids

    def _generate_worker_script(self) -> Path:
        """
        Write the Worker sbatch script to a temporary file.
        Return the path (Path) to the script file.
        """
        script_lines: list[str] = []

        # Header
        script_lines.append("#!/bin/bash")
        script_lines.append(f"#SBATCH -p {self.partition}")
        script_lines.append(f"#SBATCH --rsc p={self.processes}:t={self.threads}:c={self.cores}:m={self.memory}")
        script_lines.append(f"#SBATCH -t {self.walltime}")
        script_lines.append("#SBATCH -J dask-worker")
        script_lines.append(f"#SBATCH -o {self.logdir}/worker_%J.out")
        script_lines.append(f"#SBATCH -e {self.logdir}/worker_%J.err")

        # Additional sbatch options
        for extra in self.sbatch_extra:
            script_lines.append(extra)

        script_lines.append("")  # blank line

        # Environment modules and conda activate
        for cmd in self.env_mods:
            script_lines.append(cmd)
        script_lines.append("")  # blank line

        # Read scheduler IP from environment variable DASK_SCHED_IP
        script_lines.append("HOST=${DASK_SCHED_IP}")
        script_lines.append(f"PORT={self.scheduler_port}")
        script_lines.append("")

        # dask-worker command
        script_lines.append("dask worker tcp://${HOST}:${PORT} \\")
        script_lines.append(f"    --nthreads {self.threads} \\")
        script_lines.append(f"    --no-dashboard --memory-limit {self.memory}")
        script_lines.append("")

        # Print timestamp
        script_lines.append("date")

        # Write to file
        tmpdir = Path("/tmp/simple_dask_workers")
        tmpdir.mkdir(parents=True, exist_ok=True)

        # Create a unique file name
        script_path = tmpdir / f"worker_{int(time.time() * 1000)}.sh"
        with open(script_path, "w") as f:
            f.write("\n".join(script_lines))

        # Make the script executable
        script_path.chmod(0o744)
        return script_path

    def get_client(self, timeout: float = 30.0) -> Client:
        """
        Return a dask.distributed.Client. On first call, attempt to connect.
        The Client will automatically retry even if no workers have connected
        to the scheduler yet.
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
                    raise RuntimeError(f"Could not connect to scheduler at {sched_addr}: {e}")
                print("[SimpleDaskCluster] Waiting for scheduler to accept connections ...")
                time.sleep(1)

        print("[SimpleDaskCluster] Dask Client connected.")
        return self._client

    def close_client(self):
        """
        Close the Client if it is alive.
        """
        if self._client:
            self._client.close()
            self._client = None
