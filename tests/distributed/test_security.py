"""Tests for emout.distributed.security."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10)
    or importlib.util.find_spec("distributed") is None
    or importlib.util.find_spec("cryptography") is None,
    reason="security helpers require Python >= 3.10 with distributed and cryptography",
)


def test_ensure_cluster_security_creates_private_files(monkeypatch, tmp_path):
    from emout.distributed.security import ensure_cluster_security, load_client_security_from_state

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    files = ensure_cluster_security("secure", scheduler_host="127.0.0.1")

    assert stat.S_IMODE(files.certs_dir.stat().st_mode) == 0o700
    for path in files.all_files():
        assert path.exists()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    security = load_client_security_from_state({"protocol": "tls", "tls": files.client_state()})
    assert security is not None


def test_ensure_cluster_security_reuses_existing_files(monkeypatch, tmp_path):
    from emout.distributed.security import ensure_cluster_security

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    files = ensure_cluster_security("secure")
    files.client_key.chmod(0o644)

    again = ensure_cluster_security("secure")

    assert again.client_key == files.client_key
    assert stat.S_IMODE(files.client_key.stat().st_mode) == 0o600
