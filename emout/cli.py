"""emout CLI — Dask サーバーの永続管理.

Usage::

    emout server start [OPTIONS]    # スケジューラ + ワーカーを起動
    emout server stop                # 停止
    emout server status              # 起動中のアドレスを表示

起動中のサーバーにはスクリプトから ``emout.distributed.connect()`` で接続する。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

_STATE_DIR = Path.home() / ".emout"
_STATE_FILE = _STATE_DIR / "server.json"


def _save_state(data: dict) -> None:
    """Persist server state (address, PID) to ``~/.emout/server.json``."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(data, indent=2))


def _load_state() -> dict | None:
    """Load server state from ``~/.emout/server.json``, or ``None`` if absent."""
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return None


def _clear_state() -> None:
    """Remove the ``~/.emout/server.json`` state file if it exists."""
    _STATE_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# server start
# ---------------------------------------------------------------------------


def cmd_server_start(args):
    """Start a Dask scheduler and worker, then block until Ctrl-C.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments including scheduler-ip, scheduler-port,
        partition, processes, threads, cores, memory, and walltime.
    """
    from emout.distributed.client import start_cluster
    from emout.distributed.config import DaskConfig

    cfg = DaskConfig()

    client = start_cluster(
        scheduler_ip=args.scheduler_ip,
        scheduler_port=args.scheduler_port or cfg.scheduler_port,
        partition=args.partition,
        processes=args.processes,
        threads=args.threads,
        cores=args.cores,
        memory=args.memory,
        walltime=args.walltime,
    )

    addr = client.scheduler_info()["address"]
    n_workers = len(client.scheduler_info().get("workers", {}))
    _save_state({"address": addr, "pid": os.getpid()})

    from emout.distributed.config import _get_local_ip
    detected_ip = _get_local_ip()

    print(f"Scheduler running at {addr}")
    print(f"Detected IP: {detected_ip}")
    print(f"Workers: {n_workers}")
    print(f"State saved to {_STATE_FILE}")
    print()
    print("Scripts will auto-connect — just use emout normally:")
    print("  data = emout.Emout('output_dir')")
    print("  data.phisp[-1, :, 100, :].plot()  # auto-remote")
    print()
    print("Or explicitly: from emout.distributed import connect; connect()")
    print()
    print("Press Ctrl-C to stop the server.")

    try:
        client.scheduler_info()  # keep the process alive
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cmd_server_stop(args)


# ---------------------------------------------------------------------------
# server stop
# ---------------------------------------------------------------------------


def cmd_server_stop(_args):
    """Stop the running Dask server and clean up the state file.

    Parameters
    ----------
    _args : argparse.Namespace
        Parsed CLI arguments (unused).
    """
    from emout.distributed.client import stop_cluster

    state = _load_state()
    if state is None:
        print("No running server found.")
        return

    stopped_cleanly = True
    try:
        stop_cluster(address=state.get("address"))
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

    _clear_state()
    if stopped_cleanly:
        print("Server stopped.")
    else:
        print("Cleared saved server state.")


# ---------------------------------------------------------------------------
# server status
# ---------------------------------------------------------------------------


def cmd_server_status(_args):
    """Print the current Dask server address, PID, and worker count.

    Parameters
    ----------
    _args : argparse.Namespace
        Parsed CLI arguments (unused).
    """
    state = _load_state()
    if state is None:
        print("No running server.")
        return
    print(f"Server address: {state['address']}")
    print(f"PID: {state.get('pid', 'unknown')}")

    try:
        from dask.distributed import Client
        c = Client(state["address"], timeout="3s")
        info = c.scheduler_info()
        n_workers = len(info.get("workers", {}))
        print(f"Workers: {n_workers}")
        c.close()
    except Exception as e:
        print(f"Cannot connect: {e}")


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

    # Input file
    if data.inp is not None:
        inp = data.inp
        print(f"Input file: plasma.inp")
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

    # Unit info
    if data.unit is not None:
        print(f"Unit conversion: available")
    else:
        print("Unit conversion: not available (no conversion key in inp)")

    # Simulation validity
    print(f"Completed: {data.is_valid()}")

    # Scan HDF5 files
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

    # Particle files
    p_files = sorted(data.directory.glob("p*00_0000.h5"))
    if p_files:
        species = sorted(set(
            f.name[1] for f in p_files if f.name[1].isdigit()
        ))
        print(f"\nParticle species: {', '.join(species)}")

    # Diagnostic files
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

    # inspect
    inspect_parser = sub.add_parser("inspect", help="Show simulation metadata")
    inspect_parser.add_argument(
        "directory", nargs="?", default="./",
        help="Simulation output directory (default: current directory)",
    )
    inspect_parser.set_defaults(func=cmd_inspect)

    # server
    server = sub.add_parser("server", help="Manage the Dask render server")
    server_sub = server.add_subparsers(dest="server_command")

    # server start
    start = server_sub.add_parser("start", help="Start scheduler + workers")
    start.add_argument("--scheduler-ip", default=None)
    start.add_argument("--scheduler-port", type=int, default=None)
    start.add_argument("--partition", default=None)
    start.add_argument("--processes", type=int, default=None)
    start.add_argument("--threads", type=int, default=None)
    start.add_argument("--cores", type=int, default=None)
    start.add_argument("--memory", default=None)
    start.add_argument("--walltime", default=None)
    start.set_defaults(func=cmd_server_start)

    # server stop
    stop = server_sub.add_parser("stop", help="Stop the running server")
    stop.set_defaults(func=cmd_server_stop)

    # server status
    status = server_sub.add_parser("status", help="Show server status")
    status.set_defaults(func=cmd_server_status)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
