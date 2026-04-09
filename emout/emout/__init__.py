"""Backward-compatibility shim: ``emout.emout`` -> ``emout.core``.

All imports from ``emout.emout.*`` are transparently redirected to
``emout.core.*``.  New code should import from ``emout.core.*`` directly.
"""
import importlib
import sys


class _EmoutShimFinder:
    """Redirect ``emout.emout.X`` imports to ``emout.core.X``."""

    def find_module(self, fullname, path=None):
        if fullname.startswith("emout.emout."):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        real_name = "emout.core" + fullname[len("emout.emout"):]
        mod = importlib.import_module(real_name)
        sys.modules[fullname] = mod
        return mod


if not any(isinstance(f, _EmoutShimFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _EmoutShimFinder())

# Re-export top-level names so ``from emout.emout import Emout`` works.
from emout.core import *  # noqa: F401,F403
