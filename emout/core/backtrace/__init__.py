"""Backtrace analysis subpackage.

Provides solvers and result containers for computing particle
backtraces and energy-flux probability distributions from EMSES output.
"""

from .trace_result import TraceResult
from .trace_wrapper import TraceWrapper

__all__ = ["TraceResult", "TraceWrapper"]
