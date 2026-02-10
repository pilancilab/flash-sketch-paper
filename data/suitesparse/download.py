from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import tarfile
from pathlib import Path

import requests

import globals as g
from io_utils import ensure_dir


SUITESPARSE_BASE_URL = "https://suitesparse-collection-website.herokuapp.com/MM"


def _download_url(group, name):
    """Return the SuiteSparse download URL for a given group/name."""
    return f"{SUITESPARSE_BASE_URL}/{group}/{name}.tar.gz"


def download_matrix(group, name, cache_dir=None):
    """Download and extract a SuiteSparse matrix, returning the extract path."""
    if cache_dir is None:
        cache_dir = g.SUITESPARSE_CACHE_DIR()

    cache_dir = Path(cache_dir)
    target_dir = cache_dir / group / name
    archive_path = target_dir / f"{name}.tar.gz"

    if target_dir.exists() and any(target_dir.glob("*.mtx")):
        return target_dir

    ensure_dir(target_dir)

    if not archive_path.exists():
        url = _download_url(group, name)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(archive_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    with open(archive_path, "rb") as handle:
        signature = handle.read(2)
    if signature != b"\x1f\x8b":
        archive_path.unlink(missing_ok=True)
        url = _download_url(group, name)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(archive_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        with open(archive_path, "rb") as handle:
            signature = handle.read(2)
        if signature != b"\x1f\x8b":
            archive_path.unlink(missing_ok=True)
            raise ValueError(f"Downloaded file is not a gzip archive: {archive_path}")

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    return target_dir
