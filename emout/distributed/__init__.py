import sys

if sys.version_info.minor >= 10:
    from .client import start_cluster, stop_cluster, connect
    from .remote_figure import remote_figure, RemoteFigure, register_magics
    from .remote_render import (
        RemoteSession,
        RemoteProbabilityResult,
        RemoteBacktraceResult,
        RemoteHeatmap,
        RemoteXYData,
        display_image,
        get_or_create_session,
        clear_sessions,
    )
