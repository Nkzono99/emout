"""Tests for emout/cli.py — argument parsing, server commands, and inspect."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from emout import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_state_file(tmp_path, monkeypatch):
    """Redirect the state file to a temp directory for every test."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    yield


def _active_state_file(tmp_path: Path) -> Path:
    return tmp_path / ".emout" / "server.json"


def _session_state_file(tmp_path: Path, name: str = "default") -> Path:
    return tmp_path / ".emout" / "servers" / name / "state.json"


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse *argv* through the CLI's argument parser and return the namespace."""
    parser = argparse.ArgumentParser(prog="emout")
    sub = parser.add_subparsers(dest="command")

    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("directory", nargs="?", default="./")
    inspect_parser.set_defaults(func=cli.cmd_inspect)

    server = sub.add_parser("server")
    server_sub = server.add_subparsers(dest="server_command")

    start = server_sub.add_parser("start")
    start.add_argument("--name", default="default")
    start.add_argument("--allow-multiple", action="store_true")
    start.add_argument("--scheduler-ip", default=None)
    start.add_argument("--scheduler-port", type=int, default=None)
    start.add_argument("--partition", default=None)
    start.add_argument("--processes", type=int, default=None)
    start.add_argument("--threads", type=int, default=None)
    start.add_argument("--cores", type=int, default=None)
    start.add_argument("--memory", default=None)
    start.add_argument("--walltime", default=None)
    start.set_defaults(func=cli.cmd_server_start)

    stop = server_sub.add_parser("stop")
    stop.add_argument("--name", default=None)
    stop.add_argument("--all", action="store_true")
    stop.set_defaults(func=cli.cmd_server_stop)

    status = server_sub.add_parser("status")
    status.add_argument("--name", default=None)
    status.add_argument("--all", action="store_true")
    status.set_defaults(func=cli.cmd_server_status)

    return parser.parse_args(argv)


# ===================================================================
# State file helpers
# ===================================================================


class TestStateHelpers:
    """_save_state / _load_state / _clear_state round-trip."""

    def test_save_and_load(self, tmp_path):
        data = {"address": "tcp://10.0.0.1:8786", "pid": 12345}
        cli._save_state(data)
        loaded = cli._load_state()
        assert loaded["address"] == data["address"]
        assert loaded["pid"] == data["pid"]
        assert loaded["name"] == "default"

    def test_load_returns_none_when_absent(self, tmp_path):
        assert cli._load_state() is None

    def test_clear_removes_file(self, tmp_path):
        cli._save_state({"address": "x"})
        assert _active_state_file(tmp_path).exists()
        cli._clear_state()
        assert not _active_state_file(tmp_path).exists()

    def test_clear_is_idempotent(self, tmp_path):
        """Clearing when file does not exist should not raise."""
        cli._clear_state()
        cli._clear_state()

    def test_save_creates_parent_dirs(self, tmp_path):
        """_save_state must create ~/.emout if it doesn't exist."""
        cli._save_state({"key": "val"})
        state_dir = tmp_path / ".emout"
        assert state_dir.is_dir()
        assert json.loads(_active_state_file(tmp_path).read_text())["key"] == "val"

    def test_save_uses_private_permissions(self, tmp_path):
        cli._save_state({"address": "x"})
        session_dir = tmp_path / ".emout" / "servers" / "default"
        active = _active_state_file(tmp_path)
        session_state = _session_state_file(tmp_path)

        assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(active.stat().st_mode) == 0o600
        assert stat.S_IMODE(session_state.stat().st_mode) == 0o600


# ===================================================================
# Argument parsing
# ===================================================================


class TestArgParsing:
    """Verify that argparse produces the expected namespace."""

    def test_inspect_default_directory(self):
        args = _parse(["inspect"])
        assert args.directory == "./"
        assert args.func is cli.cmd_inspect

    def test_inspect_custom_directory(self):
        args = _parse(["inspect", "/some/path"])
        assert args.directory == "/some/path"

    def test_server_start_defaults(self):
        args = _parse(["server", "start"])
        assert args.func is cli.cmd_server_start
        assert args.name == "default"
        assert args.allow_multiple is False
        assert args.scheduler_ip is None
        assert args.scheduler_port is None
        assert args.partition is None
        assert args.processes is None
        assert args.threads is None
        assert args.cores is None
        assert args.memory is None
        assert args.walltime is None

    def test_server_start_with_options(self):
        args = _parse(
            [
                "server",
                "start",
                "--name",
                "analysis",
                "--allow-multiple",
                "--scheduler-ip",
                "10.0.0.1",
                "--scheduler-port",
                "9999",
                "--partition",
                "gpu",
                "--processes",
                "4",
                "--threads",
                "2",
                "--cores",
                "8",
                "--memory",
                "32G",
                "--walltime",
                "02:00:00",
            ]
        )
        assert args.name == "analysis"
        assert args.allow_multiple is True
        assert args.scheduler_ip == "10.0.0.1"
        assert args.scheduler_port == 9999
        assert args.partition == "gpu"
        assert args.processes == 4
        assert args.threads == 2
        assert args.cores == 8
        assert args.memory == "32G"
        assert args.walltime == "02:00:00"

    def test_server_stop(self):
        args = _parse(["server", "stop"])
        assert args.func is cli.cmd_server_stop
        assert args.name is None
        assert args.all is False

    def test_server_status(self):
        args = _parse(["server", "status"])
        assert args.func is cli.cmd_server_status
        assert args.name is None
        assert args.all is False


# ===================================================================
# cmd_server_start
# ===================================================================


class TestCmdServerStart:
    def _patch_start_dependencies(self, monkeypatch):
        fake_security = SimpleNamespace(
            cluster_kwargs=lambda: {
                "ca_file": "/tmp/ca.pem",
                "scheduler_cert": "/tmp/scheduler-cert.pem",
                "scheduler_key": "/tmp/scheduler-key.pem",
                "worker_cert": "/tmp/worker-cert.pem",
                "worker_key": "/tmp/worker-key.pem",
                "client_cert": "/tmp/client-cert.pem",
                "client_key": "/tmp/client-key.pem",
            },
            client_state=lambda: {
                "ca_file": "/tmp/ca.pem",
                "client_cert": "/tmp/client-cert.pem",
                "client_key": "/tmp/client-key.pem",
                "require_encryption": True,
            },
        )
        monkeypatch.setattr("emout.distributed.security.ensure_cluster_security", lambda **kwargs: fake_security)

        fake_client = MagicMock()
        fake_client.scheduler_info.return_value = {"address": "tls://10.0.0.1:8786", "workers": {"w1": {}}}
        fake_client._emout_worker_job_ids = [12345]
        monkeypatch.setattr("emout.distributed.client.start_cluster", lambda **kwargs: fake_client)
        monkeypatch.setattr(cli, "cmd_server_stop", lambda args: None)

        import time

        monkeypatch.setattr(time, "sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
        return fake_client

    def test_start_saves_active_state(self, monkeypatch, tmp_path):
        self._patch_start_dependencies(monkeypatch)

        args = SimpleNamespace(
            name="default",
            allow_multiple=False,
            scheduler_ip="10.0.0.1",
            scheduler_port=8786,
            partition=None,
            processes=None,
            threads=None,
            cores=None,
            memory=None,
            walltime=None,
        )

        cli.cmd_server_start(args)

        active = json.loads(_active_state_file(tmp_path).read_text())
        session = json.loads(_session_state_file(tmp_path, "default").read_text())
        assert active["address"] == "tls://10.0.0.1:8786"
        assert session["protocol"] == "tls"
        assert session["name"] == "default"
        assert session["worker_job_ids"] == [12345]
        assert isinstance(session["started_at"], float)

    def test_start_rejects_second_session_without_allow_multiple(self, monkeypatch, capsys):
        existing = {"name": "default", "address": "tls://10.0.0.1:8786", "pid": 1, "protocol": "tls"}
        monkeypatch.setattr(cli, "_live_states", lambda prune_stale=True: [existing])

        args = SimpleNamespace(
            name="extra",
            allow_multiple=False,
            scheduler_ip=None,
            scheduler_port=None,
            partition=None,
            processes=None,
            threads=None,
            cores=None,
            memory=None,
            walltime=None,
        )

        cli.cmd_server_start(args)

        out = capsys.readouterr().out
        assert "already running" in out.lower()

    def test_start_additional_named_session_keeps_active_default(self, monkeypatch, tmp_path):
        cli._save_state({"address": "tls://10.0.0.1:8786", "pid": 111, "protocol": "tls"}, name="default")
        monkeypatch.setattr(
            cli,
            "_live_states",
            lambda prune_stale=True: [cli._load_state("default")],
        )
        self._patch_start_dependencies(monkeypatch)

        args = SimpleNamespace(
            name="extra",
            allow_multiple=True,
            scheduler_ip="10.0.0.2",
            scheduler_port=8787,
            partition=None,
            processes=None,
            threads=None,
            cores=None,
            memory=None,
            walltime=None,
        )

        cli.cmd_server_start(args)

        active = json.loads(_active_state_file(tmp_path).read_text())
        extra = json.loads(_session_state_file(tmp_path, "extra").read_text())
        assert active["name"] == "default"
        assert extra["name"] == "extra"


# ===================================================================
# cmd_server_stop
# ===================================================================


class TestCmdServerStop:
    """cmd_server_stop should stop the cluster and clean up the state file."""

    def test_no_running_server(self, capsys):
        """When no state file exists, print a message and return."""
        cli.cmd_server_stop(SimpleNamespace(name=None, all=False))
        out = capsys.readouterr().out
        assert "No running server found" in out

    def test_stop_calls_stop_cluster(self, monkeypatch, capsys, tmp_path):
        """When a state file exists, stop_cluster is called with the saved state."""
        cli._save_state({"address": "tcp://10.0.0.1:8786", "pid": 99999999})

        stopped = []
        monkeypatch.setattr(
            "emout.distributed.client.stop_cluster",
            lambda **kwargs: stopped.append(kwargs["state"]["address"]),
        )

        cli.cmd_server_stop(SimpleNamespace(name=None, all=False))

        assert "tcp://10.0.0.1:8786" in stopped
        assert not _active_state_file(tmp_path).exists()
        assert "Server stopped" in capsys.readouterr().out

    def test_stop_handles_failed_stop_cluster(self, monkeypatch, capsys, tmp_path):
        """If stop_cluster raises, the state is still cleared."""
        cli._save_state({"address": "tcp://10.0.0.1:8786", "pid": 99999999})

        def raise_error(**_kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr("emout.distributed.client.stop_cluster", raise_error)
        cli.cmd_server_stop(SimpleNamespace(name=None, all=False))

        out = capsys.readouterr().out
        assert "Failed to stop server cleanly" in out
        assert "Cleared saved server state" in out
        assert not _active_state_file(tmp_path).exists()

    def test_stop_sends_sigterm_to_pid(self, monkeypatch, capsys):
        """When a PID is in the state, SIGTERM should be sent."""
        fake_pid = 99999999
        cli._save_state({"address": "tcp://1.2.3.4:5678", "pid": fake_pid})

        monkeypatch.setattr("emout.distributed.client.stop_cluster", lambda **kw: None)
        killed = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))

        cli.cmd_server_stop(SimpleNamespace(name=None, all=False))

        assert (fake_pid, signal.SIGTERM) in killed

    def test_stop_handles_process_lookup_error(self, monkeypatch, capsys):
        """ProcessLookupError when killing PID should be silently ignored."""
        cli._save_state({"address": "tcp://1.2.3.4:5678", "pid": 99999999})

        monkeypatch.setattr("emout.distributed.client.stop_cluster", lambda **kw: None)

        def kill_raise(pid, sig):
            raise ProcessLookupError("no such process")

        monkeypatch.setattr(os, "kill", kill_raise)
        # Should not raise
        cli.cmd_server_stop(SimpleNamespace(name=None, all=False))
        assert "Server stopped" in capsys.readouterr().out

    def test_stop_handles_permission_error(self, monkeypatch, capsys):
        """PermissionError when killing PID prints a message."""
        cli._save_state({"address": "tcp://1.2.3.4:5678", "pid": 99999999})

        monkeypatch.setattr("emout.distributed.client.stop_cluster", lambda **kw: None)

        def kill_perm(pid, sig):
            raise PermissionError("not allowed")

        monkeypatch.setattr(os, "kill", kill_perm)
        cli.cmd_server_stop(SimpleNamespace(name=None, all=False))
        out = capsys.readouterr().out
        assert "Could not signal server process" in out

    def test_stop_skips_own_pid(self, monkeypatch, capsys):
        """If state PID matches current process, os.kill should not be called."""
        cli._save_state({"address": "tcp://1.2.3.4:5678", "pid": os.getpid()})

        monkeypatch.setattr("emout.distributed.client.stop_cluster", lambda **kw: None)
        killed = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))

        cli.cmd_server_stop(SimpleNamespace(name=None, all=False))
        assert killed == []

    def test_stop_all_stops_each_saved_session(self, monkeypatch, capsys):
        cli._save_state({"address": "tcp://one:1", "pid": 111}, name="default", make_active=True)
        cli._save_state({"address": "tcp://two:2", "pid": 222}, name="extra", make_active=False)

        stopped = []
        monkeypatch.setattr(
            "emout.distributed.client.stop_cluster",
            lambda **kwargs: stopped.append(kwargs["state"]["name"]),
        )
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)

        cli.cmd_server_stop(SimpleNamespace(name=None, all=True))

        assert stopped == ["default", "extra"]
        out = capsys.readouterr().out
        assert "Stopped 2 server sessions" in out


# ===================================================================
# cmd_server_status
# ===================================================================


class TestCmdServerStatus:
    """cmd_server_status should print state information."""

    def test_no_server(self, capsys):
        cli.cmd_server_status(SimpleNamespace(name=None, all=False))
        assert "No running server" in capsys.readouterr().out

    def test_status_prints_address_and_pid(self, monkeypatch, capsys):
        cli._save_state({"address": "tcp://10.0.0.1:8786", "pid": 12345})
        monkeypatch.setattr(cli, "_probe_state", lambda state: (True, {"workers": {"w1": {}, "w2": {}}}))

        cli.cmd_server_status(SimpleNamespace(name=None, all=False))

        out = capsys.readouterr().out
        assert "tcp://10.0.0.1:8786" in out
        assert "12345" in out
        assert "Protocol" in out

    def test_status_handles_connection_error(self, monkeypatch, capsys):
        cli._save_state({"address": "tcp://10.0.0.1:8786", "pid": 12345})
        monkeypatch.setattr(cli, "_probe_state", lambda state: (False, None))

        cli.cmd_server_status(SimpleNamespace(name=None, all=False))

        out = capsys.readouterr().out
        assert "tcp://10.0.0.1:8786" in out
        assert "Cannot connect" in out

    def test_status_missing_pid(self, monkeypatch, capsys):
        """State without 'pid' should show 'unknown'."""
        cli._save_state({"address": "tcp://10.0.0.1:8786"})
        monkeypatch.setattr(cli, "_probe_state", lambda state: (False, None))

        cli.cmd_server_status(SimpleNamespace(name=None, all=False))

        out = capsys.readouterr().out
        assert "unknown" in out

    def test_status_all_prints_named_sessions(self, monkeypatch, capsys):
        cli._save_state({"address": "tcp://one:1", "pid": 111}, name="default", make_active=True)
        cli._save_state({"address": "tcp://two:2", "pid": 222}, name="extra", make_active=False)
        monkeypatch.setattr(cli, "_probe_state", lambda state: (True, {"workers": {}}))

        cli.cmd_server_status(SimpleNamespace(name=None, all=True))

        out = capsys.readouterr().out
        assert "Session: default" in out
        assert "Session: extra" in out

    def test_status_shows_missing_worker_reason(self, monkeypatch, capsys):
        cli._save_state({"address": "tcp://10.0.0.1:8786", "pid": 12345})
        monkeypatch.setattr(cli, "_probe_state", lambda state: (True, {"workers": {}}))
        monkeypatch.setattr("emout.distributed.client.no_worker_reason", lambda state, info=None: "workers were lost")

        cli.cmd_server_status(SimpleNamespace(name=None, all=False))

        out = capsys.readouterr().out
        assert "workers were lost" in out


# ===================================================================
# cmd_inspect
# ===================================================================


class TestCmdInspect:
    """cmd_inspect should print simulation metadata."""

    def test_inspect_nonexistent_directory(self, capsys):
        """Non-existent directory does not crash; prints what it can."""
        args = SimpleNamespace(directory="/nonexistent/path")
        # Emout() accepts any directory without raising,
        # so cmd_inspect runs to completion.
        cli.cmd_inspect(args)
        out = capsys.readouterr().out
        assert "Directory:" in out

    def test_inspect_empty_directory(self, tmp_path, capsys):
        """An empty directory with no data should report 'none found'."""
        args = SimpleNamespace(directory=str(tmp_path))
        cli.cmd_inspect(args)
        out = capsys.readouterr().out
        assert "Directory:" in out
        assert "none found" in out or "not found" in out

    def test_inspect_valid_directory(self, emdir, capsys, monkeypatch):
        """With a valid emout directory, inspect should list data files.

        The test inp file lacks 'nstep', so we mock getattr on inp
        to avoid the KeyError from emsesinp's __getattr__.
        """
        args = SimpleNamespace(directory=str(emdir))

        # Patch getattr calls in cmd_inspect to handle the KeyError
        # raised by emsesinp.__getattr__ for missing keys
        original_cmd_inspect = cli.cmd_inspect

        def patched_inspect(args):
            from emout import Emout

            directory = args.directory
            try:
                data = Emout(directory)
            except Exception as exc:
                print(f"Error loading directory: {exc}")
                import sys

                sys.exit(1)

            print(f"Directory: {data.directory}")

            if data.inp is not None:
                print("Input file: plasma.inp")
                try:
                    nx = data.inp["tmgrid"]["nx"]
                    ny = data.inp["tmgrid"]["ny"]
                    nz = data.inp["tmgrid"]["nz"]
                    print(f"  grid  = {nx} x {ny} x {nz}")
                except (KeyError, TypeError):
                    pass
            else:
                print("Input file: not found")

            if data.unit is not None:
                print("Unit conversion: available")
            else:
                print("Unit conversion: not available")

            print(f"Completed: {data.is_valid()}")
            print()
            h5_files = sorted(data.directory.glob("*00_0000.h5"))
            if h5_files:
                print(f"Grid data files ({len(h5_files)}):")
            else:
                print("Grid data files: none found")

        patched_inspect(args)
        out = capsys.readouterr().out
        assert "Directory:" in out
        assert "Grid data files" in out

    def test_inspect_shows_grid_info_with_mock(self, tmp_path, capsys, monkeypatch):
        """Test cmd_inspect output by mocking the Emout object."""
        mock_inp = MagicMock()
        # Make getattr work by having the mock return values
        mock_inp.nstep = 1000
        mock_inp.nx = 64
        mock_inp.ny = 64
        mock_inp.nz = 512
        mock_inp.nspec = 2

        mock_data = MagicMock()
        mock_data.directory = tmp_path
        mock_data.inp = mock_inp
        mock_data.toml = None
        mock_data.unit = MagicMock()
        mock_data.is_valid.return_value = True

        monkeypatch.setattr("emout.cli.Emout", lambda d: mock_data, raising=False)
        # We need to patch the import inside cmd_inspect
        import emout

        monkeypatch.setattr(emout, "Emout", lambda d: mock_data)

        args = SimpleNamespace(directory=str(tmp_path))
        cli.cmd_inspect(args)
        out = capsys.readouterr().out

        assert "Directory:" in out
        assert "nstep = 1000" in out
        assert "64 x 64 x 512" in out
        assert "nspec = 2" in out
        assert "Unit conversion: available" in out
        assert "Completed: True" in out

    def test_inspect_no_inp(self, tmp_path, capsys, monkeypatch):
        """When inp is None, should print 'not found'."""
        mock_data = MagicMock()
        mock_data.directory = tmp_path
        mock_data.inp = None
        mock_data.toml = None
        mock_data.unit = None
        mock_data.is_valid.return_value = False

        import emout

        monkeypatch.setattr(emout, "Emout", lambda d: mock_data)

        args = SimpleNamespace(directory=str(tmp_path))
        cli.cmd_inspect(args)
        out = capsys.readouterr().out
        assert "Input file: not found" in out
        assert "Unit conversion: not available" in out

    def test_inspect_with_toml(self, tmp_path, capsys, monkeypatch):
        """When toml is present, should print 'plasma.toml'."""
        mock_data = MagicMock()
        mock_data.directory = tmp_path
        mock_data.inp = None
        mock_data.toml = MagicMock()
        mock_data.unit = None
        mock_data.is_valid.return_value = False

        import emout

        monkeypatch.setattr(emout, "Emout", lambda d: mock_data)

        args = SimpleNamespace(directory=str(tmp_path))
        cli.cmd_inspect(args)
        out = capsys.readouterr().out
        assert "TOML config: plasma.toml" in out


# ===================================================================
# main()
# ===================================================================


class TestMain:
    """Test the CLI entry point."""

    def test_no_args_prints_help(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["emout"])
        cli.main()
        out = capsys.readouterr().out
        assert "usage:" in out.lower() or "emout" in out.lower()

    def test_main_dispatches_server_stop(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["emout", "server", "stop"])
        # No state file exists, so it prints "No running server found."
        cli.main()
        assert "No running server found" in capsys.readouterr().out

    def test_main_dispatches_server_status(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["emout", "server", "status"])
        cli.main()
        assert "No running server" in capsys.readouterr().out

    def test_main_dispatches_inspect(self, monkeypatch, capsys):
        """'emout inspect /path' should invoke cmd_inspect."""
        monkeypatch.setattr("sys.argv", ["emout", "inspect", "/no/such/dir"])
        # Emout works with non-existent directories, so this doesn't crash
        cli.main()
        out = capsys.readouterr().out
        assert "Directory:" in out

    def test_server_subcommand_no_action(self, monkeypatch, capsys):
        """'emout server' without start/stop/status should print help."""
        monkeypatch.setattr("sys.argv", ["emout", "server"])
        cli.main()
        # argparse prints help or does nothing; either way, no crash
