def run_backend(func, *args, **kwargs):
    """
    • compute=True: 即 execute(func) して結果を返す
    • compute=False: Dask Future を返す (Client が無ければ即時実行して返す)
    """
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
