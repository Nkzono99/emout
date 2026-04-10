"""Tests for emout.distributed.clusters (SimpleDaskCluster).

All Dask/distributed imports and subprocess calls are mocked so no real
clusters are started.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10)
    or importlib.util.find_spec("dask") is None
    or importlib.util.find_spec("distributed") is None,
    reason="distributed runtime requires Python >= 3.10 with dask/distributed",
)


def _make_cluster(tmp_path, **overrides):
    """Create a SimpleDaskCluster with sensible defaults pointing at tmp_path."""
    from emout.distributed.clusters import SimpleDaskCluster

    defaults = dict(
        scheduler_ip="10.0.0.1",
        scheduler_port=8786,
        partition="test_part",
        processes=2,
        threads=4,
        cores=8,
        memory="8G",
        walltime="02:00:00",
        env_mods=["module load Python", "conda activate myenv"],
        logdir=str(tmp_path / "logs"),
        sbatch_extra=["--exclusive"],
    )
    defaults.update(overrides)
    return SimpleDaskCluster(**defaults)


# ===================================================================
# Initialization
# ===================================================================


class TestInit:
    """Test SimpleDaskCluster constructor."""

    def test_default_values(self, tmp_path):
        from emout.distributed.clusters import SimpleDaskCluster

        cluster = SimpleDaskCluster(scheduler_ip="1.2.3.4", logdir=str(tmp_path))
        assert cluster.scheduler_ip == "1.2.3.4"
        assert cluster.scheduler_port == 8786
        assert cluster.partition == "gr20001a"
        assert cluster.processes == 1
        assert cluster.threads == 1
        assert cluster.cores == 1
        assert cluster.memory == "4G"
        assert cluster.walltime == "01:00:00"
        assert cluster.env_mods == []
        assert cluster.sbatch_extra == []
        assert cluster._sched_proc is None
        assert cluster.worker_job_ids == []
        assert cluster._client is None

    def test_custom_values(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        assert cluster.scheduler_ip == "10.0.0.1"
        assert cluster.scheduler_port == 8786
        assert cluster.partition == "test_part"
        assert cluster.processes == 2
        assert cluster.threads == 4
        assert cluster.cores == 8
        assert cluster.memory == "8G"
        assert cluster.walltime == "02:00:00"
        assert cluster.env_mods == ["module load Python", "conda activate myenv"]
        assert cluster.sbatch_extra == ["--exclusive"]

    def test_logdir_created(self, tmp_path):
        logdir = tmp_path / "nested" / "logs"
        assert not logdir.exists()
        _make_cluster(tmp_path, logdir=str(logdir))
        assert logdir.exists()

    def test_logdir_default_cwd(self, monkeypatch, tmp_path):
        """When logdir is None, defaults to cwd/dask_logs."""
        from emout.distributed.clusters import SimpleDaskCluster

        monkeypatch.chdir(tmp_path)
        cluster = SimpleDaskCluster(scheduler_ip="1.2.3.4", logdir=None)
        assert cluster.logdir == tmp_path / "dask_logs"
        assert cluster.logdir.exists()

    def test_logdir_as_path(self, tmp_path):
        logdir = tmp_path / "pathobj"
        cluster = _make_cluster(tmp_path, logdir=logdir)
        assert cluster.logdir == logdir
        assert logdir.exists()

    def test_env_mods_none_becomes_empty_list(self, tmp_path):
        from emout.distributed.clusters import SimpleDaskCluster

        cluster = SimpleDaskCluster(scheduler_ip="1.2.3.4", env_mods=None, logdir=str(tmp_path))
        assert cluster.env_mods == []

    def test_sbatch_extra_none_becomes_empty_list(self, tmp_path):
        from emout.distributed.clusters import SimpleDaskCluster

        cluster = SimpleDaskCluster(scheduler_ip="1.2.3.4", sbatch_extra=None, logdir=str(tmp_path))
        assert cluster.sbatch_extra == []

    def test_initial_worker_job_ids_empty(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        assert cluster.worker_job_ids == []


# ===================================================================
# start_scheduler
# ===================================================================


class TestStartScheduler:
    """Test start_scheduler method."""

    def test_start_scheduler_success(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_proc.pid = 12345

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, patch("time.sleep"):
            cluster.start_scheduler()

        assert cluster._sched_proc is mock_proc
        popen_call = mock_popen.call_args
        cmd = popen_call[0][0]
        assert cmd[0] == "dask"
        assert cmd[1] == "scheduler"
        assert "--host" in cmd
        assert "10.0.0.1" in cmd
        assert "--port" in cmd
        assert "8786" in cmd
        assert "--no-dashboard" in cmd

    def test_start_scheduler_no_dashboard_false(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, patch("time.sleep"):
            cluster.start_scheduler(no_dashboard=False)

        cmd = mock_popen.call_args[0][0]
        assert "--no-dashboard" not in cmd

    def test_start_scheduler_already_running(self, tmp_path, capsys):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        cluster._sched_proc = mock_proc

        cluster.start_scheduler()
        captured = capsys.readouterr()
        assert "already running" in captured.out.lower()

    def test_start_scheduler_fails(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited immediately

        with patch("subprocess.Popen", return_value=mock_proc), patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Scheduler failed to start"):
                cluster.start_scheduler()

    def test_start_scheduler_log_files(self, tmp_path):
        """Scheduler output goes to logdir/scheduler.out and .err."""
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        open_calls = []
        real_open = open

        def tracking_open(path, *args, **kwargs):
            open_calls.append(str(path))
            return MagicMock()  # mock file handle

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("time.sleep"),
            patch("builtins.open", tracking_open),
        ):
            cluster.start_scheduler()

        log_dir = str(cluster.logdir)
        assert any("scheduler.out" in c for c in open_calls)
        assert any("scheduler.err" in c for c in open_calls)


# ===================================================================
# stop_scheduler
# ===================================================================


class TestStopScheduler:
    """Test stop_scheduler method."""

    def test_stop_no_process(self, tmp_path, capsys):
        cluster = _make_cluster(tmp_path)
        cluster._sched_proc = None
        cluster.stop_scheduler()
        captured = capsys.readouterr()
        assert "no scheduler" in captured.out.lower()

    def test_stop_already_exited(self, tmp_path, capsys):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already exited
        cluster._sched_proc = mock_proc

        cluster.stop_scheduler()
        captured = capsys.readouterr()
        assert "not running" in captured.out.lower()

    def test_stop_terminate_success(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # running
        mock_proc.pid = 9999
        mock_proc.wait.return_value = 0
        cluster._sched_proc = mock_proc

        cluster.stop_scheduler()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert cluster._sched_proc is None

    def test_stop_terminate_timeout_then_kill(self, tmp_path, capsys):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # running
        mock_proc.pid = 9999
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
        cluster._sched_proc = mock_proc

        cluster.stop_scheduler()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert cluster._sched_proc is None
        captured = capsys.readouterr()
        assert "killing" in captured.out.lower()

    def test_stop_sets_proc_to_none(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 123
        cluster._sched_proc = mock_proc

        cluster.stop_scheduler()
        assert cluster._sched_proc is None


# ===================================================================
# submit_worker
# ===================================================================


class TestSubmitWorker:
    """Test submit_worker method."""

    def test_submit_requires_scheduler(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        cluster._sched_proc = None
        with pytest.raises(RuntimeError, match="Scheduler is not running"):
            cluster.submit_worker()

    def test_submit_single_worker(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = "Submitted batch job 123456"

        with (
            patch("subprocess.run", return_value=completed),
            patch.object(cluster, "_generate_worker_script", return_value=Path("/tmp/worker.sh")),
        ):
            job_ids = cluster.submit_worker(jobs=1)

        assert job_ids == [123456]
        assert cluster.worker_job_ids == [123456]

    def test_submit_multiple_workers(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        call_count = 0

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.returncode = 0
            result.stdout = f"Submitted batch job {100 + call_count}"
            return result

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch.object(cluster, "_generate_worker_script", return_value=Path("/tmp/worker.sh")),
        ):
            job_ids = cluster.submit_worker(jobs=3)

        assert len(job_ids) == 3
        assert job_ids == [101, 102, 103]
        assert cluster.worker_job_ids == [101, 102, 103]

    def test_submit_accumulates_job_ids(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        counter = [0]

        def fake_run(*args, **kwargs):
            counter[0] += 1
            result = MagicMock()
            result.returncode = 0
            result.stdout = f"Submitted batch job {counter[0]}"
            return result

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch.object(cluster, "_generate_worker_script", return_value=Path("/tmp/worker.sh")),
        ):
            cluster.submit_worker(jobs=2)
            cluster.submit_worker(jobs=1)

        assert cluster.worker_job_ids == [1, 2, 3]

    def test_submit_sbatch_failure(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        completed = MagicMock()
        completed.returncode = 1
        completed.stderr = "sbatch: error: allocation failure"

        with (
            patch("subprocess.run", return_value=completed),
            patch.object(cluster, "_generate_worker_script", return_value=Path("/tmp/worker.sh")),
        ):
            with pytest.raises(RuntimeError, match="sbatch failed"):
                cluster.submit_worker()

    def test_submit_parse_job_id_failure(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = "Unexpected output with no number"

        with (
            patch("subprocess.run", return_value=completed),
            patch.object(cluster, "_generate_worker_script", return_value=Path("/tmp/worker.sh")),
        ):
            with pytest.raises(RuntimeError, match="Could not parse job ID"):
                cluster.submit_worker()

    def test_submit_passes_env_with_dask_sched_ip(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = "Submitted batch job 999"

        with (
            patch("subprocess.run", return_value=completed) as mock_run,
            patch.object(cluster, "_generate_worker_script", return_value=Path("/tmp/worker.sh")),
        ):
            cluster.submit_worker()

        call_kwargs = mock_run.call_args[1]
        assert "env" in call_kwargs
        assert call_kwargs["env"]["DASK_SCHED_IP"] == "10.0.0.1"


# ===================================================================
# _generate_worker_script
# ===================================================================


class TestGenerateWorkerScript:
    """Test _generate_worker_script method."""

    def test_script_is_created(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()

        assert script_path.exists()
        content = script_path.read_text()
        assert "#!/bin/bash" in content

    def test_script_contains_partition(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "#SBATCH -p test_part" in content

    def test_script_contains_resource_spec(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "p=2:t=4:c=8:m=8G" in content

    def test_script_contains_walltime(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "#SBATCH -t 02:00:00" in content

    def test_script_contains_job_name(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "#SBATCH -J dask-worker" in content

    def test_script_contains_log_paths(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "worker_%J.out" in content
        assert "worker_%J.err" in content

    def test_script_contains_sbatch_extra(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "--exclusive" in content

    def test_script_contains_env_mods(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "module load Python" in content
        assert "conda activate myenv" in content

    def test_script_contains_dask_worker_cmd(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "dask worker tcp://${HOST}:${PORT}" in content
        assert "--nthreads 4" in content
        assert "--memory-limit 8G" in content

    def test_script_contains_scheduler_port(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "PORT=8786" in content

    def test_script_contains_host_env_var(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "HOST=${DASK_SCHED_IP}" in content

    def test_script_ends_with_date(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        lines = content.strip().split("\n")
        assert lines[-1].strip() == "date"

    def test_script_is_executable(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        script_path = cluster._generate_worker_script()
        mode = script_path.stat().st_mode
        assert mode & 0o744 == 0o744

    def test_script_no_env_mods(self, tmp_path):
        cluster = _make_cluster(tmp_path, env_mods=[])
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "module load" not in content
        assert "conda activate" not in content

    def test_script_no_sbatch_extra(self, tmp_path):
        cluster = _make_cluster(tmp_path, sbatch_extra=[])
        script_path = cluster._generate_worker_script()
        content = script_path.read_text()
        assert "--exclusive" not in content

    def test_unique_script_names(self, tmp_path):
        """Each call generates a unique script path."""
        cluster = _make_cluster(tmp_path)
        path1 = cluster._generate_worker_script()
        # Tiny sleep to ensure timestamp differs
        time.sleep(0.002)
        path2 = cluster._generate_worker_script()
        assert path1 != path2


# ===================================================================
# get_client
# ===================================================================


class TestGetClient:
    """Test get_client method."""

    def test_get_client_no_scheduler(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        cluster._sched_proc = None
        with pytest.raises(RuntimeError, match="Scheduler is not running"):
            cluster.get_client()

    def test_get_client_returns_cached(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc
        mock_client = MagicMock()
        cluster._client = mock_client

        result = cluster.get_client()
        assert result is mock_client

    def test_get_client_creates_new(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        mock_client = MagicMock()

        with patch("dask.distributed.Client", return_value=mock_client):
            result = cluster.get_client()

        assert result is mock_client
        assert cluster._client is mock_client

    def test_get_client_retries_on_failure(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        mock_client = MagicMock()
        attempts = [0]

        def fake_client(*args, **kwargs):
            attempts[0] += 1
            if attempts[0] < 3:
                raise ConnectionError("not ready")
            return mock_client

        with (
            patch("dask.distributed.Client", side_effect=fake_client),
            patch("time.sleep"),
            patch("time.time", side_effect=[0, 0.1, 0.2, 0.3]),
        ):
            result = cluster.get_client(timeout=10.0)

        assert result is mock_client
        assert attempts[0] == 3

    def test_get_client_timeout(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        call_count = [0]

        def fake_client(*args, **kwargs):
            call_count[0] += 1
            raise ConnectionError("not ready")

        # time.time returns increasing values that exceed timeout
        with (
            patch("dask.distributed.Client", side_effect=fake_client),
            patch("time.sleep"),
            patch("time.time", side_effect=[0, 100]),
        ):
            with pytest.raises(RuntimeError, match="Could not connect"):
                cluster.get_client(timeout=5.0)

    def test_get_client_address_format(self, tmp_path):
        cluster = _make_cluster(tmp_path, scheduler_ip="192.168.1.5", scheduler_port=9999)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        cluster._sched_proc = mock_proc

        with patch("dask.distributed.Client") as MockClient:
            cluster.get_client()

        MockClient.assert_called_once()
        addr = MockClient.call_args[0][0]
        assert addr == "tcp://192.168.1.5:9999"


# ===================================================================
# close_client
# ===================================================================


class TestCloseClient:
    """Test close_client method."""

    def test_close_client_when_none(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        cluster._client = None
        # Should not raise
        cluster.close_client()
        assert cluster._client is None

    def test_close_client_calls_close(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_client = MagicMock()
        cluster._client = mock_client

        cluster.close_client()

        mock_client.close.assert_called_once()
        assert cluster._client is None

    def test_close_client_sets_none(self, tmp_path):
        cluster = _make_cluster(tmp_path)
        mock_client = MagicMock()
        cluster._client = mock_client

        cluster.close_client()
        assert cluster._client is None


# ===================================================================
# Full lifecycle
# ===================================================================


class TestLifecycle:
    """Integration-style tests for the full cluster lifecycle."""

    def test_full_lifecycle(self, tmp_path):
        cluster = _make_cluster(tmp_path)

        # Start scheduler
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 42

        with patch("subprocess.Popen", return_value=mock_proc), patch("time.sleep"):
            cluster.start_scheduler()

        assert cluster._sched_proc is not None

        # Submit workers
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = "Submitted batch job 555"

        with (
            patch("subprocess.run", return_value=completed),
            patch.object(cluster, "_generate_worker_script", return_value=Path("/tmp/w.sh")),
        ):
            ids = cluster.submit_worker(jobs=1)

        assert ids == [555]

        # Get client
        mock_client = MagicMock()
        with patch("dask.distributed.Client", return_value=mock_client):
            client = cluster.get_client()

        assert client is mock_client

        # Close client
        cluster.close_client()
        assert cluster._client is None
        mock_client.close.assert_called_once()

        # Stop scheduler
        cluster.stop_scheduler()
        mock_proc.terminate.assert_called_once()
        assert cluster._sched_proc is None
