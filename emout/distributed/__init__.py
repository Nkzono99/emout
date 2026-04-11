"""Optional Dask-based distributed computing subsystem.

Available only on Python >= 3.10 with ``dask`` and ``distributed``
installed.  Provides cluster management, remote figure recording,
and remote backtrace / field rendering.
"""

import sys

if sys.version_info >= (3, 10):
    from .client import start_cluster, stop_cluster, connect
    from .remote_figure import remote_figure, RemoteFigure, register_magics
    from .remote_render import (
        RemoteEmout,
        RemoteBacktraceWrapper,
        RemoteRef,
        RemoteScope,
        RemoteSession,
        RemoteProbabilityResult,
        RemoteBacktraceResult,
        RemoteHeatmap,
        RemoteXYData,
        display_image,
        get_or_create_session,
        clear_sessions,
        remote_scope,
    )
