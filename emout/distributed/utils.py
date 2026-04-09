"""Utility for dispatching computation to a local or Dask backend."""

import sys


def run_backend(func, *args, **kwargs):
    """Execute *func* locally or on the connected Dask cluster.

    If a Dask client is active, submits *func* via ``client.submit``;
    otherwise falls back to a direct local call.

    Parameters
    ----------
    func : callable
        The function to execute.
    *args
        Positional arguments forwarded to *func*.
    **kwargs
        Keyword arguments forwarded to *func*.

    Returns
    -------
    object
        The return value of *func*.
    """
    if sys.version_info < (3, 10):
        return func(*args, **kwargs)

    from dask import delayed
    from dask.distributed import default_client

    try:
        client = default_client()
    except ValueError:
        client = None

    if client is None:
        # Dask Client が存在しなければ同期実行
        return func(*args, **kwargs)

    task = delayed(func)(*args, **kwargs)
    future = client.compute(task)
    return future.result()
