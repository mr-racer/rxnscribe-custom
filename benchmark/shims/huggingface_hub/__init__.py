import os


def hf_hub_download(repo_id, filename, **kwargs):
    path = os.environ.get("BENCH_MOLSCRIBE_CKPT")
    if not path:
        raise RuntimeError("benchmark shim: set BENCH_MOLSCRIBE_CKPT")
    return path
