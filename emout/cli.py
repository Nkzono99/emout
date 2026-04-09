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
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(data, indent=2))


def _load_state() -> dict | None:
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return None


def _clear_state() -> None:
    _STATE_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# server start
# ---------------------------------------------------------------------------


def cmd_server_start(args):
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
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(prog="emout", description="emout CLI")
    sub = parser.add_subparsers(dest="command")

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
