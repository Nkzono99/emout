"""Backward-compatibility shim: ``emout.emout`` -> ``emout.core``.

All imports from ``emout.emout.*`` are transparently redirected to
``emout.core.*``.  New code should import from ``emout.core.*`` directly.

.. deprecated:: 2.9.0
    The ``emout.emout`` namespace will be removed in a future release.
    Use ``emout.core`` instead.
"""
import importlib
import sys
import warnings


class _EmoutShimFinder:
    """Redirect ``emout.emout.X`` imports to ``emout.core.X``."""

    _warned = set()

    def find_module(self, fullname, path=None):
        if fullname.startswith("emout.emout."):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        real_name = "emout.core" + fullname[len("emout.emout"):]
        if fullname not in self._warned:
            self._warned.add(fullname)
            warnings.warn(
                f"Importing from '{fullname}' is deprecated. "
                f"Use '{real_name}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        mod = importlib.import_module(real_name)
        sys.modules[fullname] = mod
        return mod


if not any(isinstance(f, _EmoutShimFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _EmoutShimFinder())

warnings.warn(
    "The 'emout.emout' namespace is deprecated. "
    "Use 'emout.core' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export top-level names so ``from emout.emout import Emout`` works.
from emout.core import *  # noqa: F401,F403
