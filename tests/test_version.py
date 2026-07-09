from importlib.metadata import version

import emout


def test_package_exposes_distribution_version():
    assert hasattr(emout, "__version__")
    assert emout.__version__ == version("emout")
