"""Probe load failures in a child process and handle native failure reports."""

from types import SimpleNamespace
import subprocess
import signal

import pytest

from emout.core.backtrace.diagnostics import backend_info
from emout.core.backtrace.trace_wrapper import TraceWrapper


def test_missing_library_reports_actionable_loader_error(tmp_path):
    path = tmp_path / "missing-library.so"
    result = backend_info(library_path=path, check_load=True)
    assert result["library_path"] == str(path)
    assert result["exists"] is False
    assert result["loadable"] is False
    assert "missing-library.so" in result["load_error"]
    assert result["solver_compatibility"] == "not_checked"


@pytest.mark.parametrize("returncode", [0, -signal.SIGILL, -signal.SIGSEGV])
def test_probe_reports_child_status_without_loading_in_parent(tmp_path, monkeypatch, returncode):
    import emout.core.backtrace.diagnostics as diagnostics

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    result = backend_info(library_path=tmp_path / "backend.so", check_load=True)
    assert result["loadable"] is (returncode == 0)
    assert result["returncode"] == returncode
    assert "ctypes.CDLL" in calls[0][2] or "ctypes.WinDLL" in calls[0][2]
    if returncode < 0:
        assert signal.Signals(-returncode).name in result["load_error"]


def test_backend_probe_timeout_is_reported(tmp_path, monkeypatch):
    import emout.core.backtrace.diagnostics as diagnostics

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("loader", 1)

    monkeypatch.setattr(diagnostics.subprocess, "run", timeout)
    result = backend_info(library_path=tmp_path / "backend.so", check_load=True, timeout=1)
    assert result["loadable"] is False
    assert "timed out" in result["load_error"]


def test_trace_backend_info_is_explicit_and_does_not_load_by_default(tmp_path, monkeypatch):
    import emout.core.backtrace.diagnostics as diagnostics

    def unexpected(*args, **kwargs):
        raise AssertionError("loading was not requested")

    monkeypatch.setattr(diagnostics.subprocess, "run", unexpected)
    trace = TraceWrapper(tmp_path, SimpleNamespace(), None)
    result = trace.backend_info(library_path=tmp_path / "backend.so")
    assert result["loadable"] is None
