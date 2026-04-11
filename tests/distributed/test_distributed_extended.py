"""Extended tests for the distributed subsystem.

Covers config.py, client.py, utils.py, and remote_figure.py without
starting any real Dask cluster.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The __init__ of emout.distributed re-exports the *function*
# ``remote_figure`` which shadows the module of the same name.
# Use sys.modules to get the actual module object.
import emout.distributed  # noqa: F401 — ensure the package is loaded
import importlib as _importlib

_rf_mod = _importlib.import_module("emout.distributed.remote_figure")

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10)
    or importlib.util.find_spec("dask") is None
    or importlib.util.find_spec("distributed") is None,
    reason="distributed runtime requires Python >= 3.10 with dask/distributed",
)


# ===================================================================
# Helpers
# ===================================================================


class FakeFuture:
    """Simulate a Dask Future that resolves synchronously."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


# ===================================================================
# config.py -- DaskConfig
# ===================================================================


class TestDaskConfig:
    """Test DaskConfig property resolution from environment variables."""

    def test_scheduler_ip_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_SCHED_IP", "192.168.1.100")
        cfg = DaskConfig()
        assert cfg.scheduler_ip == "192.168.1.100"

    def test_scheduler_port_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_SCHED_PORT", "9999")
        cfg = DaskConfig()
        assert cfg.scheduler_port == 9999

    def test_partition_default(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.delenv("EMOUT_DASK_PARTITION", raising=False)
        cfg = DaskConfig()
        assert cfg.partition == "gr20001a"

    def test_partition_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_PARTITION", "gpuq")
        cfg = DaskConfig()
        assert cfg.partition == "gpuq"

    def test_processes_default(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.delenv("EMOUT_DASK_PROCESSES", raising=False)
        cfg = DaskConfig()
        assert cfg.processes == 1

    def test_processes_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_PROCESSES", "8")
        cfg = DaskConfig()
        assert cfg.processes == 8

    def test_threads_default(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.delenv("EMOUT_DASK_THREADS", raising=False)
        cfg = DaskConfig()
        assert cfg.threads == 1

    def test_threads_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_THREADS", "4")
        cfg = DaskConfig()
        assert cfg.threads == 4

    def test_cores_default(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.delenv("EMOUT_DASK_CORES", raising=False)
        cfg = DaskConfig()
        assert cfg.cores == 60

    def test_cores_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_CORES", "16")
        cfg = DaskConfig()
        assert cfg.cores == 16

    def test_memory_default(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.delenv("EMOUT_DASK_MEMORY", raising=False)
        cfg = DaskConfig()
        assert cfg.memory == "60G"

    def test_memory_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_MEMORY", "128G")
        cfg = DaskConfig()
        assert cfg.memory == "128G"

    def test_walltime_default(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.delenv("EMOUT_DASK_WALLTIME", raising=False)
        cfg = DaskConfig()
        assert cfg.walltime == "03:00:00"

    def test_walltime_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_WALLTIME", "12:00:00")
        cfg = DaskConfig()
        assert cfg.walltime == "12:00:00"

    def test_env_mods_empty(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.delenv("EMOUT_DASK_ENV_MODS", raising=False)
        cfg = DaskConfig()
        assert cfg.env_mods == []

    def test_env_mods_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_ENV_MODS", "module load gcc; conda activate myenv")
        cfg = DaskConfig()
        assert cfg.env_mods == ["module load gcc", "conda activate myenv"]

    def test_env_mods_strips_whitespace(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_ENV_MODS", "  cmd1  ;  cmd2  ; ; ")
        cfg = DaskConfig()
        assert cfg.env_mods == ["cmd1", "cmd2"]

    def test_logdir_default(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.delenv("EMOUT_DASK_LOGDIR", raising=False)
        cfg = DaskConfig()
        assert cfg.logdir == Path.cwd() / "logs" / "dask_logs"

    def test_logdir_from_env(self, monkeypatch):
        from emout.distributed.config import DaskConfig

        monkeypatch.setenv("EMOUT_DASK_LOGDIR", "/tmp/my_logs")
        cfg = DaskConfig()
        assert cfg.logdir == Path("/tmp/my_logs")


# ===================================================================
# config.py -- _get_local_ip
# ===================================================================


class TestGetLocalIp:
    """Test IP auto-detection."""

    def test_env_override(self, monkeypatch):
        from emout.distributed.config import _get_local_ip

        monkeypatch.setenv("EMOUT_DASK_SCHED_IP", "172.16.0.42")
        assert _get_local_ip() == "172.16.0.42"

    def test_fallback_without_psutil(self, monkeypatch):
        from emout.distributed import config as config_mod

        monkeypatch.delenv("EMOUT_DASK_SCHED_IP", raising=False)

        orig_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def fake_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        monkeypatch.delitem(sys.modules, "psutil", raising=False)

        result = config_mod._get_local_ip()
        assert result == "127.0.0.1"

    def test_psutil_with_infiniband(self, monkeypatch):
        """InfiniBand interface should be preferred."""
        from emout.distributed import config as config_mod

        monkeypatch.delenv("EMOUT_DASK_SCHED_IP", raising=False)

        fake_addr_ib = SimpleNamespace(family=SimpleNamespace(name="AF_INET"), address="10.10.0.1")
        fake_addr_eth = SimpleNamespace(family=SimpleNamespace(name="AF_INET"), address="192.168.0.1")

        fake_psutil = MagicMock()
        fake_psutil.net_if_addrs.return_value = {
            "ib0": [fake_addr_ib],
            "eth0": [fake_addr_eth],
        }

        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        def fake_interface_type(name):
            return 32 if name == "ib0" else 1

        monkeypatch.setattr(config_mod, "_interface_type", fake_interface_type)

        result = config_mod._get_local_ip()
        assert result == "10.10.0.1"

    def test_psutil_filters_loopback(self, monkeypatch):
        """127.0.0.1 should be skipped."""
        from emout.distributed import config as config_mod

        monkeypatch.delenv("EMOUT_DASK_SCHED_IP", raising=False)

        fake_lo = SimpleNamespace(family=SimpleNamespace(name="AF_INET"), address="127.0.0.1")
        fake_eth = SimpleNamespace(family=SimpleNamespace(name="AF_INET"), address="10.0.0.5")

        fake_psutil = MagicMock()
        fake_psutil.net_if_addrs.return_value = {
            "lo": [fake_lo],
            "eth0": [fake_eth],
        }

        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        monkeypatch.setattr(config_mod, "_interface_type", lambda name: 1)

        result = config_mod._get_local_ip()
        assert result == "10.0.0.5"

    def test_no_candidates_returns_fallback(self, monkeypatch):
        """If psutil finds no AF_INET interfaces, return 127.0.0.1."""
        from emout.distributed import config as config_mod

        monkeypatch.delenv("EMOUT_DASK_SCHED_IP", raising=False)

        fake_ipv6 = SimpleNamespace(family=SimpleNamespace(name="AF_INET6"), address="::1")

        fake_psutil = MagicMock()
        fake_psutil.net_if_addrs.return_value = {"lo": [fake_ipv6]}

        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

        result = config_mod._get_local_ip()
        assert result == "127.0.0.1"


# ===================================================================
# config.py -- _pick_port / _is_port_open
# ===================================================================


class TestPickPort:
    """Test port selection logic."""

    def test_is_port_open_on_closed_port(self):
        from emout.distributed.config import _is_port_open

        assert _is_port_open("127.0.0.1", 1, timeout=0.1) is False

    def test_pick_port_returns_int(self, monkeypatch):
        from emout.distributed.config import _pick_port

        monkeypatch.setattr("emout.distributed.config._is_port_open", lambda ip, port, timeout=0.3: False)
        port = _pick_port("127.0.0.1")
        assert isinstance(port, int)
        assert port >= 10000

    def test_pick_port_skips_open_ports(self, monkeypatch):
        from emout.distributed.config import _pick_port

        base = 10000 + (os.getuid() % 50000)

        def fake_open(ip, port, timeout=0.3):
            return port == base

        monkeypatch.setattr("emout.distributed.config._is_port_open", fake_open)
        port = _pick_port("127.0.0.1")
        assert port == base + 1

    def test_pick_port_fallback_when_all_taken(self, monkeypatch):
        from emout.distributed.config import _pick_port

        monkeypatch.setattr("emout.distributed.config._is_port_open", lambda ip, port, timeout=0.3: True)
        base = 10000 + (os.getuid() % 50000)
        port = _pick_port("127.0.0.1", max_retries=5)
        assert port == base


# ===================================================================
# config.py -- _interface_type
# ===================================================================


class TestInterfaceType:
    """Test the kernel interface type reader."""

    def test_returns_negative_for_missing_interface(self):
        from emout.distributed.config import _interface_type

        result = _interface_type("nonexistent_iface_xyz_42")
        assert result == -1


# ===================================================================
# client.py -- start_cluster
# ===================================================================


class TestStartCluster:
    """Test start_cluster with mocked SimpleDaskCluster."""

    def test_start_cluster_creates_and_starts(self, monkeypatch):
        from emout.distributed import client as client_mod

        monkeypatch.setattr(client_mod, "_global_cluster", None)
        monkeypatch.setattr(
            client_mod,
            "ensure_cluster_security",
            lambda **kwargs: SimpleNamespace(
                cluster_kwargs=lambda: {
                    "ca_file": "/tmp/ca.pem",
                    "scheduler_cert": "/tmp/scheduler-cert.pem",
                    "scheduler_key": "/tmp/scheduler-key.pem",
                    "worker_cert": "/tmp/worker-cert.pem",
                    "worker_key": "/tmp/worker-key.pem",
                    "client_cert": "/tmp/client-cert.pem",
                    "client_key": "/tmp/client-key.pem",
                }
            ),
        )

        calls = []
        fake_client = MagicMock()

        class FakeCluster:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))

            def start_scheduler(self):
                calls.append("start_scheduler")

            def submit_worker(self, jobs=1):
                calls.append(("submit_worker", jobs))
                return [12345]

            def get_client(self):
                calls.append("get_client")
                return fake_client

        monkeypatch.setattr(client_mod, "SimpleDaskCluster", FakeCluster)

        result = client_mod.start_cluster(
            scheduler_ip="10.0.0.1",
            scheduler_port=8786,
            partition="test",
            processes=2,
            threads=4,
            cores=8,
            memory="16G",
            walltime="01:00:00",
        )

        assert result is fake_client
        assert calls[0][0] == "init"
        assert calls[0][1]["scheduler_ip"] == "10.0.0.1"
        assert calls[0][1]["protocol"] == "tls"
        assert "start_scheduler" in calls
        assert ("submit_worker", 1) in calls
        assert getattr(fake_client, "_emout_worker_job_ids") == [12345]

        monkeypatch.setattr(client_mod, "_global_cluster", None)

    def test_start_cluster_returns_existing(self, monkeypatch):
        from emout.distributed import client as client_mod

        fake_client = MagicMock()
        fake_cluster = MagicMock()
        fake_cluster.get_client.return_value = fake_client

        monkeypatch.setattr(client_mod, "_global_cluster", fake_cluster)

        result = client_mod.start_cluster()
        assert result is fake_client
        fake_cluster.get_client.assert_called_once()

        monkeypatch.setattr(client_mod, "_global_cluster", None)

    def test_start_cluster_uses_config_defaults(self, monkeypatch):
        from emout.distributed import client as client_mod

        monkeypatch.setattr(client_mod, "_global_cluster", None)
        monkeypatch.setattr(
            client_mod,
            "ensure_cluster_security",
            lambda **kwargs: SimpleNamespace(
                cluster_kwargs=lambda: {
                    "ca_file": "/tmp/ca.pem",
                    "scheduler_cert": "/tmp/scheduler-cert.pem",
                    "scheduler_key": "/tmp/scheduler-key.pem",
                    "worker_cert": "/tmp/worker-cert.pem",
                    "worker_key": "/tmp/worker-key.pem",
                    "client_cert": "/tmp/client-cert.pem",
                    "client_key": "/tmp/client-key.pem",
                }
            ),
        )

        init_kwargs = {}

        class FakeCluster:
            def __init__(self, **kwargs):
                init_kwargs.update(kwargs)

            def start_scheduler(self):
                pass

            def submit_worker(self, jobs=1):
                return [1]

            def get_client(self):
                return MagicMock()

        monkeypatch.setattr(client_mod, "SimpleDaskCluster", FakeCluster)
        monkeypatch.setenv("EMOUT_DASK_SCHED_IP", "1.2.3.4")
        monkeypatch.setenv("EMOUT_DASK_PARTITION", "gpu_partition")

        client_mod.start_cluster()

        assert init_kwargs["scheduler_ip"] == "1.2.3.4"
        assert init_kwargs["partition"] == "gpu_partition"
        assert init_kwargs["protocol"] == "tls"

        monkeypatch.setattr(client_mod, "_global_cluster", None)


# ===================================================================
# client.py -- stop_cluster
# ===================================================================


class TestStopCluster:
    """Test stop_cluster."""

    def test_stop_local_cluster(self, monkeypatch):
        from emout.distributed import client as client_mod

        events = []
        fake_cluster = MagicMock()
        fake_cluster.close_client = lambda: events.append("close_client")
        fake_cluster.stop_scheduler = lambda: events.append("stop_scheduler")
        monkeypatch.setattr(client_mod, "_global_cluster", fake_cluster)
        monkeypatch.setattr(client_mod, "clear_sessions", lambda: events.append("clear"))

        client_mod.stop_cluster()

        assert events == ["close_client", "stop_scheduler", "clear"]
        assert client_mod._global_cluster is None

    def test_stop_no_cluster_no_address_raises(self, monkeypatch):
        from emout.distributed import client as client_mod

        monkeypatch.setattr(client_mod, "_global_cluster", None)

        with pytest.raises(RuntimeError, match="No active local cluster"):
            client_mod.stop_cluster()


# ===================================================================
# client.py -- connect
# ===================================================================


class TestConnect:
    """Test connect() with mocked Dask Client."""

    def test_connect_with_address(self, monkeypatch):
        import dask.distributed
        from emout.distributed import client as client_mod

        calls = []

        class FakeClient:
            def __init__(self, address):
                calls.append(address)

        monkeypatch.setattr(dask.distributed, "Client", FakeClient)

        result = client_mod.connect("tcp://10.0.0.1:9999")
        assert isinstance(result, FakeClient)
        assert calls == ["tcp://10.0.0.1:9999"]

    def test_connect_reads_state_file(self, monkeypatch, tmp_path):
        import dask.distributed
        from emout.distributed import client as client_mod

        state_dir = tmp_path / ".emout"
        state_dir.mkdir()
        state_file = state_dir / "server.json"
        state_file.write_text(json.dumps({"address": "tcp://auto:5555"}))

        calls = []

        class FakeClient:
            def __init__(self, address):
                calls.append(address)

        monkeypatch.setattr(dask.distributed, "Client", FakeClient)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = client_mod.connect()
        assert calls == ["tcp://auto:5555"]

    def test_connect_raises_without_state(self, monkeypatch, tmp_path):
        from emout.distributed import client as client_mod

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with pytest.raises(RuntimeError, match="emout server is not running"):
            client_mod.connect()

    def test_connect_require_workers_clears_stale_state(self, monkeypatch, tmp_path):
        import dask.distributed
        from emout.distributed import client as client_mod

        state_dir = tmp_path / ".emout"
        state_dir.mkdir()
        state_file = state_dir / "server.json"
        state_file.write_text(
            json.dumps(
                {
                    "name": "default",
                    "address": "tcp://auto:5555",
                    "pid": 123456,
                    "started_at": 1.0,
                    "worker_job_ids": [999],
                }
            )
        )

        events = []

        class FakeClient:
            def __init__(self, address):
                self.address = address

            def scheduler_info(self):
                return {"workers": {}}

            def shutdown(self):
                events.append("shutdown")

            def close(self):
                events.append("close")

        monkeypatch.setattr(dask.distributed, "Client", FakeClient)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(client_mod, "_pid_is_alive", lambda pid: False)
        monkeypatch.setattr(client_mod, "query_worker_job_states", lambda state, timeout=3.0: {})
        monkeypatch.setattr(client_mod.time, "monotonic", lambda: 10.0)
        monkeypatch.setattr(client_mod, "clear_server_state", lambda name=None: events.append(("clear", name)))
        monkeypatch.setattr(client_mod, "clear_sessions", lambda: events.append("clear_sessions"))
        monkeypatch.setattr(client_mod.os, "kill", lambda pid, sig: events.append(("kill", pid, sig)))

        with pytest.raises(RuntimeError, match="saved server state was cleared"):
            client_mod.connect(require_workers=True, worker_timeout=0.0)

        assert "shutdown" in events
        assert ("clear", "default") in events
        assert "clear_sessions" in events


# ===================================================================
# utils.py -- run_backend
# ===================================================================


class TestRunBackend:
    """Test run_backend dispatching."""

    def test_direct_call_without_dask_client(self, monkeypatch):
        from emout.distributed.utils import run_backend
        import dask.distributed

        def _raise_value_error():
            raise ValueError("no client")

        monkeypatch.setattr(dask.distributed, "default_client", _raise_value_error)

        result = run_backend(lambda x, y: x + y, 3, 7)
        assert result == 10

    def test_direct_call_with_dask_client(self, monkeypatch):
        from emout.distributed.utils import run_backend
        import dask.distributed

        class FakeClient:
            def compute(self, task):
                result = task.compute()
                return FakeFuture(result)

        monkeypatch.setattr(dask.distributed, "default_client", lambda: FakeClient())

        result = run_backend(lambda x: x * 2, 5)
        assert result == 10

    def test_local_fallback_when_no_client(self, monkeypatch):
        from emout.distributed.utils import run_backend
        import dask.distributed

        def no_client():
            raise ValueError("no client")

        monkeypatch.setattr(dask.distributed, "default_client", no_client)

        def my_sum(lst):
            return sum(lst)

        result = run_backend(my_sum, [1, 2, 3])
        assert result == 6

    def test_with_kwargs(self, monkeypatch):
        from emout.distributed.utils import run_backend
        import dask.distributed

        def _raise():
            raise ValueError("no client")

        monkeypatch.setattr(dask.distributed, "default_client", _raise)

        result = run_backend(lambda a, b=10: a + b, 5, b=20)
        assert result == 25


# ===================================================================
# remote_figure.py -- recording state
# ===================================================================


class TestRecordingState:
    """Test the global recording functions."""

    def test_initial_state(self):
        _rf_mod._reset_recording_state()
        assert _rf_mod.is_recording() is False

    def test_record_field_plot(self):
        _rf_mod._reset_recording_state()
        _rf_mod._recording = True
        _rf_mod._commands = []

        _rf_mod.record_field_plot("phisp", (0, slice(None)), {"cmap": "jet"}, {"directory": "/tmp"})

        assert len(_rf_mod._commands) == 1
        cmd = _rf_mod._commands[0]
        assert cmd[0] == "field_plot"
        assert cmd[1] == "phisp"
        assert cmd[2] == (0, slice(None))
        assert cmd[3] == {"cmap": "jet"}
        assert cmd[4] == {"directory": "/tmp"}

        _rf_mod._reset_recording_state()

    def test_record_plot_surfaces(self):
        _rf_mod._reset_recording_state()
        _rf_mod._commands = []

        _rf_mod.record_plot_surfaces(
            "phisp",
            (0,),
            ["surface"],
            {"use_si": True, "vmin": -1.0},
            {"directory": "/tmp"},
        )

        assert len(_rf_mod._commands) == 1
        cmd = _rf_mod._commands[0]
        assert cmd[0] == "plot_surfaces"
        assert cmd[1] == "phisp"
        assert cmd[2] == (0,)
        assert cmd[3] == ["surface"]
        assert cmd[4] == {"use_si": True, "vmin": -1.0}
        assert cmd[5] == {"directory": "/tmp"}

        _rf_mod._reset_recording_state()

    def test_record_plt_call(self):
        _rf_mod._reset_recording_state()
        _rf_mod._commands = []

        _rf_mod.record_plt_call("xlabel", ("X [m]",), {})
        _rf_mod.record_plt_call("ylabel", ("Y [m]",), {"fontsize": 14})

        assert len(_rf_mod._commands) == 2
        assert _rf_mod._commands[0] == ("plt", "xlabel", ("X [m]",), {})
        assert _rf_mod._commands[1] == ("plt", "ylabel", ("Y [m]",), {"fontsize": 14})

        _rf_mod._reset_recording_state()

    def test_record_boundary_plot(self):
        _rf_mod._reset_recording_state()
        _rf_mod._commands = []

        _rf_mod.record_boundary_plot({"use_si": True}, {"directory": "/sim"})

        assert len(_rf_mod._commands) == 1
        assert _rf_mod._commands[0] == ("boundary_plot", {"use_si": True}, {"directory": "/sim"})

        _rf_mod._reset_recording_state()

    def test_record_backtrace_render(self):
        _rf_mod._reset_recording_state()
        _rf_mod._commands = []

        _rf_mod.record_backtrace_render("cache_0", "vx", "vz", {"cmap": "hot"})

        assert _rf_mod._commands[0] == ("backtrace_render", "cache_0", "vx", "vz", {"cmap": "hot"})

        _rf_mod._reset_recording_state()

    def test_record_energy_spectrum(self):
        _rf_mod._reset_recording_state()
        _rf_mod._commands = []

        _rf_mod.record_energy_spectrum("key_0", {"scale": "log"})

        assert _rf_mod._commands[0] == ("energy_spectrum", "key_0", {"scale": "log"})

        _rf_mod._reset_recording_state()

    def test_bind_session(self):
        _rf_mod._reset_recording_state()
        sentinel = object()

        _rf_mod.bind_session(sentinel)
        assert _rf_mod._session is sentinel

        _rf_mod.bind_session(None)  # should not overwrite
        assert _rf_mod._session is sentinel

        _rf_mod._reset_recording_state()

    def test_request_session_with_none_kwargs(self):
        """request_session(None) should be a no-op."""
        _rf_mod._reset_recording_state()
        _rf_mod.request_session(None)
        assert _rf_mod._session is None

        _rf_mod._reset_recording_state()


# ===================================================================
# remote_figure.py -- RemoteFigure class
# ===================================================================


class TestRemoteFigure:
    """Test RemoteFigure open/close and context manager protocol."""

    def test_initial_state(self):
        from emout.distributed.remote_figure import RemoteFigure

        rf = RemoteFigure()
        assert rf.is_open is False
        assert rf.fmt == "png"
        assert rf.dpi == 150
        assert rf.figsize is None

    def test_custom_parameters(self):
        from emout.distributed.remote_figure import RemoteFigure

        rf = RemoteFigure(fmt="svg", dpi=300, figsize=(12, 8))
        assert rf.fmt == "svg"
        assert rf.dpi == 300
        assert rf.figsize == (12, 8)

    def test_open_sets_recording(self, monkeypatch):
        from emout.distributed.remote_figure import RemoteFigure
        import emout.distributed.remote_render as rr_mod

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        rf = RemoteFigure()
        rf.open()

        try:
            assert rf.is_open is True
            assert _rf_mod.is_recording() is True
        finally:
            rf.close()

        assert rf.is_open is False
        assert _rf_mod.is_recording() is False

    def test_double_open_raises(self, monkeypatch):
        from emout.distributed.remote_figure import RemoteFigure
        import emout.distributed.remote_render as rr_mod

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        rf = RemoteFigure()
        rf.open()
        try:
            with pytest.raises(RuntimeError, match="already open"):
                rf.open()
        finally:
            rf.close()

    def test_close_without_open_is_noop(self):
        from emout.distributed.remote_figure import RemoteFigure

        rf = RemoteFigure()
        rf.close()  # Should not raise

    def test_context_manager_protocol(self, monkeypatch):
        from emout.distributed.remote_figure import RemoteFigure
        import emout.distributed.remote_render as rr_mod

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        with RemoteFigure() as rf:
            assert rf.is_open is True
            assert _rf_mod.is_recording() is True

        assert rf.is_open is False
        assert _rf_mod.is_recording() is False

    def test_monkey_patches_plt(self, monkeypatch):
        """RemoteFigure.open() should monkey-patch plt methods."""
        from emout.distributed.remote_figure import RemoteFigure
        import emout.distributed.remote_render as rr_mod
        import matplotlib.pyplot as plt

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        orig_xlabel = plt.xlabel

        rf = RemoteFigure()
        rf.open()

        try:
            # plt.xlabel should now be a recorder
            assert plt.xlabel is not orig_xlabel
            # Call it and check it records
            plt.xlabel("test label")
            assert any(c[1] == "xlabel" for c in _rf_mod._commands)
        finally:
            rf.close()

        # Should be restored
        assert plt.xlabel is orig_xlabel

    def test_close_replays_on_session(self, monkeypatch):
        """Commands should be replayed on the session when close() is called."""
        from emout.distributed.remote_figure import RemoteFigure
        import emout.distributed.remote_render as rr_mod

        replayed = []
        displayed = []

        class FakeSession:
            def replay_figure(self, commands, fmt="png", dpi=150):
                replayed.append((commands, fmt, dpi))
                return FakeFuture(b"fake-png")

        fake_session = FakeSession()
        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: fake_session)
        monkeypatch.setattr(rr_mod, "display_image", lambda img_bytes: displayed.append(img_bytes))

        rf = RemoteFigure(session=fake_session)
        rf.open()
        _rf_mod.record_plt_call("xlabel", ("test",), {})
        rf.close()

        assert len(replayed) == 1
        commands, fmt, dpi = replayed[0]
        assert any(c[1] == "xlabel" for c in commands)
        assert fmt == "png"
        assert displayed == [b"fake-png"]

    def test_figsize_adds_figure_command(self, monkeypatch):
        """When figsize is set, a plt.figure command should be prepended."""
        from emout.distributed.remote_figure import RemoteFigure
        import emout.distributed.remote_render as rr_mod

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        rf = RemoteFigure(figsize=(10, 6))
        rf.open()

        figure_cmds = [c for c in _rf_mod._commands if c[0] == "figure_call" and c[2] == "figure"]
        assert len(figure_cmds) == 1
        assert figure_cmds[0][4] == {"figsize": (10, 6)}

        rf.close()

    def test_subplot_returns_axes_proxy_and_records_axes_call(self, monkeypatch):
        from emout.distributed.remote_figure import AxesProxy, RemoteFigure
        import emout.distributed.remote_render as rr_mod
        import matplotlib.pyplot as plt

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        rf = RemoteFigure()
        rf.open()
        try:
            ax = plt.subplot(111)
            ax.axhline(0.5, color="red")
            commands = list(_rf_mod._commands)
        finally:
            rf.close()

        assert isinstance(ax, AxesProxy)
        assert any(cmd[0] == "figure_call" and cmd[2] == "subplot" for cmd in commands)
        assert any(cmd[0] == "axes_call" and cmd[2] == "axhline" for cmd in commands)

    def test_figure_proxy_records_add_axes_axis_and_spine_calls(self, monkeypatch):
        from emout.distributed.remote_figure import RemoteFigure
        import emout.distributed.remote_render as rr_mod
        import matplotlib.pyplot as plt

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        rf = RemoteFigure()
        rf.open()
        try:
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="3d")
            ax.xaxis.set_tick_params(pad=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
            commands = list(_rf_mod._commands)
        finally:
            rf.close()

        assert any(cmd[0] == "figure_call" and cmd[1] is not None and cmd[2] == "add_axes" for cmd in commands)
        assert any(cmd[0] == "axis_call" and cmd[2] == "xaxis" and cmd[3] == "set_tick_params" for cmd in commands)
        assert any(cmd[0] == "spine_call" and cmd[2] == "left" and cmd[3] == "set_visible" for cmd in commands)

    def test_pyplot_colorbar_returns_proxy(self, monkeypatch):
        from emout.distributed.remote_figure import ColorbarProxy, RemoteFigure
        import emout.distributed.remote_render as rr_mod
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        rf = RemoteFigure()
        rf.open()
        try:
            fig = plt.figure()
            cax = fig.add_axes([0.82, 0.15, 0.04, 0.7])
            mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="viridis")
            cbar = plt.colorbar(mappable, cax=cax)
            cbar.set_label("phi")
            cbar.ax.tick_params(labelsize=8)
            commands = list(_rf_mod._commands)
        finally:
            rf.close()

        assert isinstance(cbar, ColorbarProxy)
        assert any(cmd[0] == "figure_call" and cmd[2] == "colorbar" for cmd in commands)
        assert any(cmd[0] == "colorbar_call" and cmd[2] == "set_label" for cmd in commands)

    def test_del_warns_if_not_closed(self, monkeypatch):
        from emout.distributed.remote_figure import RemoteFigure
        import emout.distributed.remote_render as rr_mod

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        rf = RemoteFigure()
        rf.open()

        with pytest.warns(ResourceWarning, match="not closed"):
            rf.__del__()

        assert rf.is_open is False


# ===================================================================
# remote_figure.py -- remote_figure context manager function
# ===================================================================


class TestRemoteFigureContextManager:
    """Test the convenience remote_figure() context manager."""

    def test_basic_usage(self, monkeypatch):
        from emout.distributed.remote_figure import (
            remote_figure as rf_ctx,
            is_recording,
        )
        import emout.distributed.remote_render as rr_mod

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        assert is_recording() is False

        with rf_ctx():
            assert is_recording() is True

        assert is_recording() is False

    def test_cleanup_on_exception(self, monkeypatch):
        from emout.distributed.remote_figure import (
            remote_figure as rf_ctx,
            is_recording,
        )
        import emout.distributed.remote_render as rr_mod

        monkeypatch.setattr(rr_mod, "get_or_create_session", lambda **kw: None)

        with pytest.raises(ValueError):
            with rf_ctx():
                assert is_recording() is True
                raise ValueError("test error")

        assert is_recording() is False


# ===================================================================
# remote_figure.py -- _parse_magic_line
# ===================================================================


class TestParseMagicLine:
    """Test the IPython cell magic argument parser."""

    def test_empty_line(self):
        from emout.distributed.remote_figure import _parse_magic_line

        assert _parse_magic_line("") == {}
        assert _parse_magic_line("   ") == {}

    def test_dpi(self):
        from emout.distributed.remote_figure import _parse_magic_line

        result = _parse_magic_line("--dpi 300")
        assert result == {"dpi": 300}

    def test_dpi_short_flag(self):
        from emout.distributed.remote_figure import _parse_magic_line

        result = _parse_magic_line("-d 200")
        assert result == {"dpi": 200}

    def test_fmt(self):
        from emout.distributed.remote_figure import _parse_magic_line

        result = _parse_magic_line("--fmt svg")
        assert result == {"fmt": "svg"}

    def test_fmt_short_flag(self):
        from emout.distributed.remote_figure import _parse_magic_line

        result = _parse_magic_line("-f pdf")
        assert result == {"fmt": "pdf"}

    def test_figsize(self):
        from emout.distributed.remote_figure import _parse_magic_line

        result = _parse_magic_line("--figsize 12,8")
        assert result == {"figsize": (12.0, 8.0)}

    def test_emout_dir(self):
        from emout.distributed.remote_figure import _parse_magic_line

        result = _parse_magic_line("--emout-dir /tmp/output")
        assert result == {"emout_dir": "/tmp/output"}

    def test_combined(self):
        from emout.distributed.remote_figure import _parse_magic_line

        result = _parse_magic_line("--dpi 300 --fmt svg --figsize 10,5 --emout-dir /sim")
        assert result == {"dpi": 300, "fmt": "svg", "figsize": (10.0, 5.0), "emout_dir": "/sim"}

    def test_unknown_tokens_ignored(self):
        from emout.distributed.remote_figure import _parse_magic_line

        result = _parse_magic_line("--unknown foo --dpi 100")
        assert result == {"dpi": 100}


# ===================================================================
# remote_figure.py -- PLT_METHODS constant
# ===================================================================


class TestPltMethods:
    """Verify the PLT_METHODS list covers expected methods."""

    def test_contains_essential_methods(self):
        from emout.distributed.remote_figure import _PLT_METHODS

        for name in ("xlabel", "ylabel", "title", "xlim", "ylim", "legend", "colorbar"):
            assert name in _PLT_METHODS

    def test_all_strings(self):
        from emout.distributed.remote_figure import _PLT_METHODS

        assert all(isinstance(m, str) for m in _PLT_METHODS)


# ===================================================================
# remote_render.py -- _normalize_emout_kwargs
# ===================================================================


class TestNormalizeEmoutKwargs:
    """Test emout_kwargs normalization."""

    def test_with_emout_dir(self):
        from emout.distributed.remote_render import _normalize_emout_kwargs

        result = _normalize_emout_kwargs(emout_dir="/tmp/sim")
        assert "directory" in result
        assert Path(result["directory"]).is_absolute()

    def test_with_emout_dir_and_input_path(self):
        from emout.distributed.remote_render import _normalize_emout_kwargs

        result = _normalize_emout_kwargs(
            emout_dir="/tmp/sim",
            input_path="/tmp/sim/plasma.toml",
        )
        assert "directory" in result
        assert "input_path" in result
        assert "output_directory" in result

    def test_with_emout_kwargs(self):
        from emout.distributed.remote_render import _normalize_emout_kwargs

        kwargs = {
            "directory": "/tmp/sim",
            "input_path": "/tmp/sim/plasma.toml",
            "output_directory": "/tmp/sim",
            "append_directories": ["/tmp/extra1", "/tmp/extra2"],
        }
        result = _normalize_emout_kwargs(emout_kwargs=kwargs)
        assert Path(result["directory"]).is_absolute()
        assert len(result["append_directories"]) == 2

    def test_raises_without_arguments(self):
        from emout.distributed.remote_render import _normalize_emout_kwargs

        with pytest.raises(ValueError, match="emout_dir or emout_kwargs"):
            _normalize_emout_kwargs()

    def test_resolves_relative_paths(self, tmp_path, monkeypatch):
        from emout.distributed.remote_render import _normalize_emout_kwargs

        monkeypatch.chdir(tmp_path)
        result = _normalize_emout_kwargs(emout_dir="relative/path")
        assert Path(result["directory"]).is_absolute()


# ===================================================================
# remote_render.py -- clear_sessions / get_or_create_session
# ===================================================================


class TestSessionManagement:
    """Test session lifecycle without a real Dask cluster."""

    def test_clear_sessions(self):
        from emout.distributed import remote_render

        remote_render._shared_session = "something"
        remote_render.clear_sessions()
        assert remote_render._shared_session is None

    def test_get_or_create_returns_none_without_client(self, monkeypatch):
        from emout.distributed import remote_render
        import dask.distributed

        remote_render.clear_sessions()

        def _raise():
            raise ValueError("no client")

        monkeypatch.setattr(dask.distributed, "default_client", _raise)
        monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent"))

        with pytest.warns(UserWarning, match="emout server is not running"):
            result = remote_render.get_or_create_session(emout_dir="/tmp")
        assert result is None

    def test_get_or_create_reuses_session(self, monkeypatch):
        """Subsequent calls return the same shared session."""
        import dask.distributed
        from emout.distributed import remote_render

        remote_render.clear_sessions()

        class FakeClient:
            def scheduler_info(self):
                return {"workers": {"w1": {}}}

            def submit(self, cls, *args, **kwargs):
                return FakeFuture(object())

        monkeypatch.setattr(dask.distributed, "default_client", lambda: FakeClient())

        s1 = remote_render.get_or_create_session(emout_dir="/tmp/a")
        s2 = remote_render.get_or_create_session(emout_dir="/tmp/b")

        assert s1 is s2

        remote_render.clear_sessions()

    def test_get_or_create_returns_none_when_worker_is_missing(self, monkeypatch):
        import dask.distributed
        from emout.distributed import remote_render

        remote_render.clear_sessions()

        class FakeClient:
            def scheduler_info(self):
                return {"workers": {}}

        monkeypatch.setattr(dask.distributed, "default_client", lambda: FakeClient())

        from emout.distributed import client as client_mod

        monkeypatch.setattr(
            client_mod,
            "ensure_client_has_workers",
            lambda client, **kwargs: (_ for _ in ()).throw(RuntimeError("workers are gone")),
        )

        with pytest.warns(UserWarning, match="workers are gone"):
            result = remote_render.get_or_create_session(emout_dir="/tmp/a")

        assert result is None


# ===================================================================
# remote_render.py -- RemoteProbabilityResult
# ===================================================================


class TestRemoteProbabilityResult:
    """Test the remote proxy for ProbabilityResult."""

    def test_pair_returns_remote_heatmap(self):
        from emout.distributed.remote_render import RemoteProbabilityResult, RemoteHeatmap

        session = MagicMock()
        rpr = RemoteProbabilityResult(session, "key_0")
        hm = rpr.pair("vx", "vz")
        assert isinstance(hm, RemoteHeatmap)

    def test_dynamic_attr_access(self):
        from emout.distributed.remote_render import RemoteProbabilityResult, RemoteHeatmap

        session = MagicMock()
        rpr = RemoteProbabilityResult(session, "key_0")
        hm = rpr.vxvz
        assert isinstance(hm, RemoteHeatmap)

    def test_invalid_attr_raises(self):
        from emout.distributed.remote_render import RemoteProbabilityResult

        session = MagicMock()
        rpr = RemoteProbabilityResult(session, "key_0")
        with pytest.raises(AttributeError):
            rpr.invalid_attr

    def test_same_axis_raises(self):
        """result.vxvx should raise AttributeError (same axis)."""
        from emout.distributed.remote_render import RemoteProbabilityResult

        session = MagicMock()
        rpr = RemoteProbabilityResult(session, "key_0")
        with pytest.raises(AttributeError):
            rpr.vxvx

    def test_repr(self):
        from emout.distributed.remote_render import RemoteProbabilityResult

        rpr = RemoteProbabilityResult(MagicMock(), "my_key")
        assert "my_key" in repr(rpr)

    def test_drop(self):
        from emout.distributed.remote_render import RemoteProbabilityResult

        session = MagicMock()
        rpr = RemoteProbabilityResult(session, "key_0")
        rpr.drop()
        session.drop.assert_called_once_with("key_0")

    def test_fetch(self):
        from emout.distributed.remote_render import RemoteProbabilityResult

        session = MagicMock()
        session.fetch_object.return_value.result.return_value = {"kind": "probability"}
        rpr = RemoteProbabilityResult(session, "key_0")
        assert rpr.fetch() == {"kind": "probability"}


# ===================================================================
# remote_render.py -- RemoteBacktraceResult
# ===================================================================


class TestRemoteBacktraceResult:
    """Test the remote proxy for BacktraceResult."""

    def test_pair_returns_remote_xy_data(self):
        from emout.distributed.remote_render import RemoteBacktraceResult, RemoteXYData

        session = MagicMock()
        rbr = RemoteBacktraceResult(session, "key_0")
        xy = rbr.pair("x", "vz")
        assert isinstance(xy, RemoteXYData)

    def test_dynamic_attr_access(self):
        from emout.distributed.remote_render import RemoteBacktraceResult, RemoteXYData

        session = MagicMock()
        rbr = RemoteBacktraceResult(session, "key_0")
        xy = rbr.xvz
        assert isinstance(xy, RemoteXYData)

    def test_t_axis_shorthand_is_supported(self):
        from emout.distributed.remote_render import RemoteBacktraceResult, RemoteXYData

        session = MagicMock()
        rbr = RemoteBacktraceResult(session, "key_0")
        xy = rbr.tx
        assert isinstance(xy, RemoteXYData)

    def test_invalid_attr_raises(self):
        from emout.distributed.remote_render import RemoteBacktraceResult

        session = MagicMock()
        rbr = RemoteBacktraceResult(session, "key_0")
        with pytest.raises(AttributeError):
            rbr.nonexistent

    def test_repr(self):
        from emout.distributed.remote_render import RemoteBacktraceResult

        rbr = RemoteBacktraceResult(MagicMock(), "bt_key")
        assert "bt_key" in repr(rbr)

    def test_drop(self):
        from emout.distributed.remote_render import RemoteBacktraceResult

        session = MagicMock()
        rbr = RemoteBacktraceResult(session, "key_0")
        rbr.drop()
        session.drop.assert_called_once_with("key_0")

    def test_fetch(self):
        from emout.distributed.remote_render import RemoteBacktraceResult

        session = MagicMock()
        session.fetch_object.return_value.result.return_value = {"kind": "backtrace"}
        rbr = RemoteBacktraceResult(session, "key_0")
        assert rbr.fetch() == {"kind": "backtrace"}

    def test_sample_returns_remote_backtrace_result(self):
        from emout.distributed.remote_render import RemoteBacktraceResult

        session = MagicMock()
        session.call_method.return_value.result.return_value = True
        rbr = RemoteBacktraceResult(session, "key_0")
        sampled = rbr.sample(2, random_state=1)
        assert isinstance(sampled, RemoteBacktraceResult)


# ===================================================================
# remote_render.py -- RemoteHeatmap / RemoteXYData repr
# ===================================================================


class TestRemoteProxyRepr:
    """Test repr strings for remote proxies."""

    def test_remote_heatmap_repr(self):
        from emout.distributed.remote_render import RemoteHeatmap

        hm = RemoteHeatmap(MagicMock(), "k", "vx", "vz")
        s = repr(hm)
        assert "vx" in s
        assert "vz" in s

    def test_remote_xy_data_repr(self):
        from emout.distributed.remote_render import RemoteXYData

        xy = RemoteXYData(MagicMock(), "k", "x", "vy")
        s = repr(xy)
        assert "x" in s
        assert "vy" in s


# ===================================================================
# remote_render.py -- display_image
# ===================================================================


class TestDisplayImage:
    """Test the display_image helper."""

    @staticmethod
    def _make_png():
        """Create minimal valid PNG bytes using matplotlib Agg backend.

        Uses Figure() directly to avoid issues with plt.subplots being
        monkey-patched by RemoteFigure tests.
        """
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from io import BytesIO

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        buf = BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()

    def test_display_on_axes(self):
        """display_image with an axes argument should render onto it."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from emout.distributed.remote_render import display_image

        png_bytes = self._make_png()

        fig = Figure()
        ax_dst = fig.add_subplot(111)
        result = display_image(png_bytes, ax=ax_dst)
        assert result is ax_dst

    def test_display_without_ipython_creates_axes(self, monkeypatch):
        """Without IPython, display_image(bytes, ax=None) should create a new axes."""
        import matplotlib

        matplotlib.use("Agg")
        from emout.distributed.remote_render import display_image

        # Ensure IPython import fails
        import builtins

        _real_import = builtins.__import__

        def _no_ipython(name, *args, **kwargs):
            if "IPython" in name:
                raise ImportError("no IPython")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_ipython)
        for key in list(sys.modules):
            if key.startswith("IPython"):
                monkeypatch.delitem(sys.modules, key, raising=False)

        png_bytes = self._make_png()
        result = display_image(png_bytes)
        assert result is not None  # an axes was created

        import matplotlib.pyplot as plt

        plt.close("all")


# ===================================================================
# remote_render.py -- _next_key
# ===================================================================


class TestNextKey:
    """Test the key counter."""

    def test_unique_keys(self):
        from emout.distributed.remote_render import _next_key

        k1 = _next_key("test")
        k2 = _next_key("test")
        assert k1 != k2
        assert k1.startswith("test_")
        assert k2.startswith("test_")

    def test_default_prefix(self):
        from emout.distributed.remote_render import _next_key

        k = _next_key()
        assert k.startswith("result_")
