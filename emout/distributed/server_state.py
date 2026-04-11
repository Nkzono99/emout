"""Persistent state helpers for ``emout server`` sessions.

The active/default session remains mirrored at ``~/.emout/server.json``
for backward compatibility, while named sessions live under
``~/.emout/servers/<name>/state.json``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

DEFAULT_SERVER_NAME = "default"

_VALID_SERVER_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def normalize_server_name(name: str | None) -> str:
    """Return a validated server/session name."""
    server_name = (name or DEFAULT_SERVER_NAME).strip()
    if not server_name:
        raise ValueError("Server name must not be empty.")
    if not _VALID_SERVER_NAME.fullmatch(server_name):
        raise ValueError("Server name may contain only ASCII letters, digits, '.', '_', and '-'.")
    return server_name


def state_root_dir() -> Path:
    """Return ``~/.emout``."""
    return Path.home() / ".emout"


def active_state_file() -> Path:
    """Return the backward-compatible active state file path."""
    return state_root_dir() / "server.json"


def servers_dir() -> Path:
    """Return the directory holding named server states."""
    return state_root_dir() / "servers"


def server_session_dir(name: str | None = None) -> Path:
    """Return the per-session directory path."""
    return servers_dir() / normalize_server_name(name)


def server_state_file(name: str | None = None) -> Path:
    """Return the per-session state file path."""
    return server_session_dir(name) / "state.json"


def server_certs_dir(name: str | None = None) -> Path:
    """Return the per-session certificate directory."""
    return server_session_dir(name) / "certs"


def ensure_private_dir(path: str | Path) -> Path:
    """Create *path* with user-only permissions."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def write_private_bytes(path: str | Path, data: bytes, mode: int = 0o600) -> Path:
    """Write *data* to *path* with explicit user-only permissions."""
    filepath = Path(path)
    ensure_private_dir(filepath.parent)
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(filepath, mode)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return filepath


def write_private_text(path: str | Path, text: str, mode: int = 0o600) -> Path:
    """Write UTF-8 text with explicit user-only permissions."""
    return write_private_bytes(path, text.encode("utf-8"), mode=mode)


def write_private_json(path: str | Path, data: dict[str, Any], mode: int = 0o600) -> Path:
    """Write JSON with explicit user-only permissions."""
    return write_private_text(path, json.dumps(data, indent=2, sort_keys=True), mode=mode)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_server_state(name: str | None = None) -> dict[str, Any] | None:
    """Load a saved server state.

    If *name* is omitted, the active/default state is loaded first and
    ``servers/default/state.json`` is used as a fallback.
    """
    if name is None:
        active = active_state_file()
        if active.exists():
            state = _read_json(active)
            state.setdefault("name", DEFAULT_SERVER_NAME)
            return state

        default_state = server_state_file(DEFAULT_SERVER_NAME)
        if default_state.exists():
            state = _read_json(default_state)
            state.setdefault("name", DEFAULT_SERVER_NAME)
            return state
        return None

    state_path = server_state_file(name)
    if not state_path.exists():
        return None
    state = _read_json(state_path)
    state.setdefault("name", normalize_server_name(name))
    return state


def list_server_states() -> list[dict[str, Any]]:
    """Return all known server states."""
    states: dict[str, dict[str, Any]] = {}

    active = load_server_state()
    if active is not None:
        active_name = normalize_server_name(active.get("name"))
        active["name"] = active_name
        states[active_name] = active

    states_root = servers_dir()
    if not states_root.exists():
        return [states[name] for name in sorted(states)]

    for state_path in sorted(states_root.glob("*/state.json")):
        state = _read_json(state_path)
        state_name = normalize_server_name(state.get("name") or state_path.parent.name)
        state["name"] = state_name
        states[state_name] = state

    return [states[name] for name in sorted(states)]


def save_server_state(
    data: dict[str, Any],
    *,
    name: str | None = None,
    make_active: bool = True,
) -> dict[str, Any]:
    """Persist a per-session state and optionally mirror it as active."""
    server_name = normalize_server_name(name or data.get("name"))
    state = dict(data)
    state["name"] = server_name

    ensure_private_dir(state_root_dir())
    ensure_private_dir(servers_dir())
    ensure_private_dir(server_session_dir(server_name))
    write_private_json(server_state_file(server_name), state)

    if make_active:
        write_private_json(active_state_file(), state)

    return state


def clear_server_state(name: str | None = None) -> None:
    """Remove a server state and its session directory."""
    target_name: str | None
    if name is None:
        state = load_server_state()
        target_name = None if state is None else normalize_server_name(state.get("name"))
    else:
        target_name = normalize_server_name(name)

    if target_name is not None:
        shutil.rmtree(server_session_dir(target_name), ignore_errors=True)

    active = load_server_state()
    if active is not None:
        active_name = normalize_server_name(active.get("name"))
        if target_name is None or active_name == target_name:
            active_state_file().unlink(missing_ok=True)
