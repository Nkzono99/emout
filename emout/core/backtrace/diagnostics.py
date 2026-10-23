"""Inspect solver setup without executing a trace in the current process."""

import importlib.metadata
import importlib.util
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys


def backend_info(*, library_path=None, system="auto", check_load=False, timeout=10.0):
    """Return backend metadata and optionally probe loading in a child process.

    Loading does not run the solver and cannot establish instruction-set or
    OpenMP compatibility of its computation paths. A crash in a library's
    load-time initializers is isolated from the calling Python process.
    """
    selected_system = platform.system().lower() if system == "auto" else system
    suffixes = {"linux": "so", "darwin": "dylib", "windows": "dll"}
    if selected_system not in suffixes:
        raise ValueError("system must be 'auto', 'linux', 'darwin', or 'windows'")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive")
    info = {
        "system": selected_system,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "backend_version": None,
        "library_path": None,
        "exists": False,
        "loadable": None,
        "load_error": None,
        "returncode": None,
        "cpu_features": [],
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "solver_compatibility": "not_checked",
    }
    try:
        info["backend_version"] = importlib.metadata.version("vdist-solver-fortran")
    except importlib.metadata.PackageNotFoundError:
        pass
    if library_path is None:
        try:
            spec = importlib.util.find_spec("vdsolverf")
        except (ImportError, ValueError):
            spec = None
        if spec is None or spec.origin is None:
            info["load_error"] = "vdsolverf is not installed; install vdist-solver-fortran or supply library_path."
            return info
        library_path = Path(spec.origin).parent / f"libvdist-solver-fortran.{suffixes[selected_system]}"
    library_path = Path(library_path).expanduser().resolve()
    info["library_path"] = str(library_path)
    info["exists"] = library_path.is_file()
    if platform.system().lower() == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as source:
                for line in source:
                    if line.startswith(("flags", "Features")):
                        info["cpu_features"] = sorted(line.split(":", 1)[1].split())
                        break
        except OSError:
            pass
    if check_load:
        loader = "WinDLL" if selected_system == "windows" else "CDLL"
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import ctypes, sys; ctypes.{loader}(sys.argv[1])", str(library_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            info["loadable"] = False
            info["load_error"] = f"Library load timed out after {timeout} seconds."
        else:
            info["returncode"] = result.returncode
            info["loadable"] = result.returncode == 0
            if result.returncode < 0:
                info["load_error"] = f"Library loader terminated by {signal.Signals(-result.returncode).name}."
            elif result.returncode:
                info["load_error"] = result.stderr.strip()[-2000:]
    return info
