"""Process-wide policy for local field-data access."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

LOCAL_DATA_POLICY_ALLOW = "allow"
LOCAL_DATA_POLICY_REMOTE_REQUIRED = "remote_required"
VALID_LOCAL_DATA_POLICIES = (
    LOCAL_DATA_POLICY_ALLOW,
    LOCAL_DATA_POLICY_REMOTE_REQUIRED,
)
LOCAL_DATA_POLICY_ENV = "EMOUT_LOCAL_DATA_POLICY"

_global_local_data_policy: str | None = None
_context_local_data_policy: ContextVar[str | None] = ContextVar(
    "emout_local_data_policy",
    default=None,
)


class LocalDataAccessDisabledError(RuntimeError):
    """Raised when local field-data access is disabled by policy."""


def normalize_local_data_policy(policy: str | None) -> str | None:
    """Return a canonical local-data policy name."""
    if policy is None:
        return None
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized in {"disabled", "disable", "remote", "strict"}:
        normalized = LOCAL_DATA_POLICY_REMOTE_REQUIRED
    if normalized in {"enabled", "enable", "local"}:
        normalized = LOCAL_DATA_POLICY_ALLOW
    if normalized not in VALID_LOCAL_DATA_POLICIES:
        allowed = ", ".join(repr(name) for name in VALID_LOCAL_DATA_POLICIES)
        raise ValueError(f"Unknown local data policy {policy!r}; expected one of: {allowed}")
    return normalized


def get_local_data_policy(override: str | None = None) -> str:
    """Return the effective local-data access policy."""
    normalized_override = normalize_local_data_policy(override)
    if normalized_override is not None:
        return normalized_override

    context_policy = _context_local_data_policy.get()
    if context_policy is not None:
        return context_policy

    if _global_local_data_policy is not None:
        return _global_local_data_policy

    env_policy = os.environ.get(LOCAL_DATA_POLICY_ENV)
    normalized_env_policy = normalize_local_data_policy(env_policy)
    if normalized_env_policy is not None:
        return normalized_env_policy

    return LOCAL_DATA_POLICY_ALLOW


def set_local_data_policy(policy: str) -> None:
    """Set the process-wide default local-data access policy."""
    global _global_local_data_policy
    _global_local_data_policy = normalize_local_data_policy(policy)


def reset_local_data_policy() -> None:
    """Clear the process-wide override and fall back to context/env/defaults."""
    global _global_local_data_policy
    _global_local_data_policy = None


def disable_local_data_access() -> None:
    """Require field data to be read through remote execution."""
    set_local_data_policy(LOCAL_DATA_POLICY_REMOTE_REQUIRED)


def enable_local_data_access() -> None:
    """Allow local field-data reads."""
    set_local_data_policy(LOCAL_DATA_POLICY_ALLOW)


def is_local_data_access_disabled(override: str | None = None) -> bool:
    """Return whether local field-data reads are disabled."""
    return get_local_data_policy(override) == LOCAL_DATA_POLICY_REMOTE_REQUIRED


@contextmanager
def local_data_policy(policy: str) -> Iterator[None]:
    """Temporarily override local-data access policy in the current context."""
    token = _context_local_data_policy.set(normalize_local_data_policy(policy))
    try:
        yield
    finally:
        _context_local_data_policy.reset(token)


def local_data_access_error(operation: str, target: str | None = None) -> LocalDataAccessDisabledError:
    """Build a consistent policy error for local field-data access."""
    where = f" for {target}" if target else ""
    return LocalDataAccessDisabledError(
        "Local field data access is disabled by local_data_policy='remote_required'. "
        f"Refused to {operation}{where}. "
        "Use Emout.remote() with remote_scope()/remote_figure(), or temporarily allow local access with "
        "emout.local_data_policy('allow') for small data."
    )


def require_local_data_access(policy: str | None, operation: str, target: str | None = None) -> None:
    """Raise if the effective policy disallows local field-data access."""
    if is_local_data_access_disabled(policy):
        raise local_data_access_error(operation, target)
