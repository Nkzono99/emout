"""emout CLI — Dask サーバーの永続管理.

Usage::

    emout server start [OPTIONS]    # スケジューラ + ワーカーを起動
    emout server stop                # 停止
    emout server status              # 起動中のアドレスを表示

起動中のサーバーにはスクリプトから ``emout.distributed.connect()`` で接続する。
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Any

from emout.distributed.server_state import (
    DEFAULT_SERVER_NAME,
    active_state_file,
    clear_server_state,
    list_server_states,
    load_server_state,
    normalize_server_name,
    save_server_state,
    server_session_dir,
)


def _save_state(data: dict, *, name: str | None = None, make_active: bool = True) -> dict:
    """Persist server state."""
    return save_server_state(data, name=name, make_active=make_active)


def _load_state(name: str | None = None) -> dict[str, Any] | None:
    """Load persisted server state."""
    return load_server_state(name)


def _clear_state(name: str | None = None) -> None:
    """Remove persisted server state."""
    clear_server_state(name)


def _pid_is_alive(pid: int | None) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _probe_state(state: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    from emout.distributed.client import get_cluster_info, state_lost_workers

    try:
        info = get_cluster_info(state, timeout="3s")
        if state_lost_workers(state, info):
            return False, info
        return True, info
    except Exception:
        if state_lost_workers(state):
            return False, None
        return _pid_is_alive(state.get("pid")), None


def _live_states(prune_stale: bool = True) -> list[dict[str, Any]]:
    live: list[dict[str, Any]] = []
    for state in list_server_states():
        is_live, _info = _probe_state(state)
        if is_live:
            live.append(state)
        elif prune_stale:
            _clear_state(state["name"])
    return live


# ---------------------------------------------------------------------------
# server start
# ---------------------------------------------------------------------------


def cmd_server_start(args):
    """Start a Dask scheduler and worker, then block until Ctrl-C."""
    from emout.distributed.client import start_cluster
    from emout.distributed.config import DaskConfig
    from emout.distributed.security import ensure_cluster_security

    cfg = DaskConfig()
    server_name = normalize_server_name(args.name)
    live_states = _live_states(prune_stale=True)

    current = next((state for state in live_states if state["name"] == server_name), None)
    if current is not None:
        print(f"Server session '{server_name}' is already running at {current['address']}.")
        return

    if live_states and not args.allow_multiple:
        running = ", ".join(state["name"] for state in live_states)
        print("Another emout server session is already running.")
        print(f"Running sessions: {running}")
        print("Use '--allow-multiple --name <session>' to start an additional session.")
        return

    scheduler_ip = args.scheduler_ip or cfg.scheduler_ip
    security = ensure_cluster_security(server_name=server_name, scheduler_host=scheduler_ip)

    client = start_cluster(
        scheduler_ip=scheduler_ip,
        scheduler_port=args.scheduler_port or cfg.scheduler_port,
        partition=args.partition,
        processes=args.processes,
        threads=args.threads,
        cores=args.cores,
        memory=args.memory,
        walltime=args.walltime,
        server_name=server_name,
        protocol="tls",
        security_files=security.cluster_kwargs(),
    )

    addr = client.scheduler_info()["address"]
    n_workers = len(client.scheduler_info().get("workers", {}))
    make_active = (not live_states) or (not args.allow_multiple)
    state = _save_state(
        {
            "address": addr,
            "pid": os.getpid(),
            "protocol": "tls",
            "session_dir": str(server_session_dir(server_name)),
            "started_at": __import__("time").time(),
            "worker_job_ids": list(getattr(client, "_emout_worker_job_ids", [])),
            "tls": security.client_state(),
        },
        name=server_name,
        make_active=make_active,
    )

    print(f"Session: {server_name}")
    print(f"Scheduler running at {addr}")
    print(f"Detected IP: {scheduler_ip}")
    print(f"Workers: {n_workers}")
    print(f"State saved to {active_state_file() if make_active else Path(state['session_dir']) / 'state.json'}")
    if not make_active:
        print()
        print(f"Connect explicitly with: from emout.distributed import connect; connect(name='{server_name}')")
    print()
    print("Scripts will auto-connect — just use emout normally:")
    print("  data = emout.Emout('output_dir')")
    print("  data.phisp[-1, :, 100, :].plot()  # auto-remote")
    print()
    print("Or explicitly: from emout.distributed import connect; connect()")
    print()
    print("Press Ctrl-C to stop the server.")

    try:
        client.scheduler_info()
        import time

        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cmd_server_stop(argparse.Namespace(name=server_name, all=False))


# ---------------------------------------------------------------------------
# server stop
# ---------------------------------------------------------------------------


def _stop_one_server(state: dict[str, Any]) -> bool:
    from emout.distributed.client import stop_cluster

    stopped_cleanly = True
    try:
        stop_cluster(state=state)
    except Exception as exc:
        stopped_cleanly = False
        print(f"Failed to stop server cleanly: {exc}")

    pid = state.get("pid")
    if isinstance(pid, int) and pid != os.getpid():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Could not signal server process PID {pid}.")

    _clear_state(state["name"])
    return stopped_cleanly


def cmd_server_stop(args):
    """Stop the running Dask server and clean up the saved state."""
    if getattr(args, "all", False):
        states = list_server_states()
    else:
        target_name = normalize_server_name(args.name) if getattr(args, "name", None) else None
        state = _load_state(target_name)
        states = [] if state is None else [state]

    if not states:
        print("No running server found.")
        return

    all_clean = True
    for state in states:
        print(f"Stopping session '{state['name']}'...")
        all_clean = _stop_one_server(state) and all_clean

    if all_clean:
        if len(states) == 1:
            print("Server stopped.")
        else:
            print(f"Stopped {len(states)} server sessions.")
    else:
        print("Cleared saved server state.")


# ---------------------------------------------------------------------------
# server status
# ---------------------------------------------------------------------------


def _print_state_status(state: dict[str, Any]) -> None:
    from emout.distributed.client import no_worker_reason

    print(f"Session: {state['name']}")
    print(f"Server address: {state['address']}")
    print(f"PID: {state.get('pid', 'unknown')}")
    print(f"Protocol: {state.get('protocol', 'tcp')}")

    is_live, info = _probe_state(state)
    if info is not None:
        n_workers = len(info.get("workers", {}))
        print(f"Workers: {n_workers}")
        if n_workers == 0:
            print(no_worker_reason(state, info))
    elif is_live:
        print("Workers: unknown (scheduler starting or not reachable yet)")
    else:
        print("Cannot connect: scheduler is not reachable")


def cmd_server_status(args):
    """Print the current Dask server status."""
    if getattr(args, "all", False):
        states = list_server_states()
        if not states:
            print("No running server.")
            return
        for index, state in enumerate(states):
            if index:
                print()
            _print_state_status(state)
        return

    target_name = normalize_server_name(args.name) if getattr(args, "name", None) else None
    state = _load_state(target_name)
    if state is None:
        print("No running server.")
        return
    _print_state_status(state)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def cmd_inspect(args):
    """Print a summary of available simulation data in a directory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments including directory and optional flags.
    """
    from emout import Emout

    directory = args.directory

    try:
        data = Emout(directory)
    except Exception as exc:
        print(f"Error loading directory: {exc}")
        sys.exit(1)

    print(f"Directory: {data.directory}")

    if data.inp is not None:
        inp = data.inp
        print("Input file: plasma.inp")
        nstep = getattr(inp, "nstep", None)
        if nstep is not None:
            print(f"  nstep = {nstep}")
        nx = getattr(inp, "nx", None)
        ny = getattr(inp, "ny", None)
        nz = getattr(inp, "nz", None)
        if nx is not None:
            print(f"  grid  = {nx} x {ny} x {nz}")
        nspec = getattr(inp, "nspec", None)
        if nspec is not None:
            print(f"  nspec = {nspec}")
    else:
        print("Input file: not found")

    if data.toml is not None:
        print("TOML config: plasma.toml")

    if data.unit is not None:
        print("Unit conversion: available")
    else:
        print("Unit conversion: not available (no conversion key in inp)")

    print(f"Completed: {data.is_valid()}")

    print()
    h5_files = sorted(data.directory.glob("*00_0000.h5"))
    if h5_files:
        print(f"Grid data files ({len(h5_files)}):")
        for f in h5_files:
            name = f.name.replace("00_0000.h5", "")
            try:
                import h5py

                with h5py.File(str(f), "r") as hf:
                    grp = hf[list(hf.keys())[0]]
                    n_steps = len(grp.keys())
                    first_key = sorted(grp.keys())[0]
                    shape = grp[first_key].shape
                    print(f"  {name:12s}  steps={n_steps:4d}  shape={shape}")
            except Exception:
                print(f"  {name:12s}  (unreadable)")
    else:
        print("Grid data files: none found")

    p_files = sorted(data.directory.glob("p*00_0000.h5"))
    if p_files:
        species = sorted(set(f.name[1] for f in p_files if f.name[1].isdigit()))
        print(f"\nParticle species: {', '.join(species)}")

    diag_files = []
    for name in ("icur", "pbody"):
        candidate = data.directory / name
        if candidate.exists():
            diag_files.append(name)
    if diag_files:
        print(f"Diagnostic files: {', '.join(diag_files)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    """Entry point for the ``emout`` CLI."""
    parser = argparse.ArgumentParser(prog="emout", description="emout CLI")
    sub = parser.add_subparsers(dest="command")

    inspect_parser = sub.add_parser("inspect", help="Show simulation metadata")
    inspect_parser.add_argument(
        "directory",
        nargs="?",
        default="./",
        help="Simulation output directory (default: current directory)",
    )
    inspect_parser.set_defaults(func=cmd_inspect)

    server = sub.add_parser("server", help="Manage the Dask render server")
    server_sub = server.add_subparsers(dest="server_command")

    start = server_sub.add_parser("start", help="Start scheduler + workers")
    start.add_argument("--name", default=DEFAULT_SERVER_NAME, help="Server session name (default: default)")
    start.add_argument(
        "--allow-multiple",
        action="store_true",
        help="Allow additional named server sessions for the same user",
    )
    start.add_argument("--scheduler-ip", default=None)
    start.add_argument("--scheduler-port", type=int, default=None)
    start.add_argument("--partition", default=None)
    start.add_argument("--processes", type=int, default=None)
    start.add_argument("--threads", type=int, default=None)
    start.add_argument("--cores", type=int, default=None)
    start.add_argument("--memory", default=None)
    start.add_argument("--walltime", default=None)
    start.set_defaults(func=cmd_server_start)

    stop = server_sub.add_parser("stop", help="Stop the running server")
    stop.add_argument("--name", default=None, help="Named server session to stop")
    stop.add_argument("--all", action="store_true", help="Stop all saved server sessions")
    stop.set_defaults(func=cmd_server_stop)

    status = server_sub.add_parser("status", help="Show server status")
    status.add_argument("--name", default=None, help="Named server session to inspect")
    status.add_argument("--all", action="store_true", help="Show all saved server sessions")
    status.set_defaults(func=cmd_server_status)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
