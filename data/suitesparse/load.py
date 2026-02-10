from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math
import numpy as np
import scipy.io
import scipy.sparse
import torch

import globals as g
from data.suitesparse.download import download_matrix
from torch_utils import manual_seed, resolve_dtype


@dataclass(frozen=True)
class SuiteSparseDatasetConfig:
    """Config for loading a SuiteSparse matrix as a dense tensor."""

    dataset: str = g.DATASET_SUITESPARSE
    group: str = "HB"
    name: str = "1138_bus"
    densify_max_elems: int = 10_000_000
    submatrix_rows: int = 2048
    submatrix_cols: int = 2048
    max_dense_elems: Optional[int] = None
    max_dense_fraction: Optional[float] = None
    auto_submatrix: bool = False
    enforce_max_elems: bool = True
    transpose: bool = False
    seed: int = 0
    dtype: str = g.DTYPE_FP32


def _find_mtx_file(root_dir, preferred_stem=None):
    """Locate a .mtx file under the extracted directory."""
    root_dir = Path(root_dir)
    matches = list(root_dir.rglob("*.mtx"))
    if not matches:
        raise FileNotFoundError(f"No .mtx file found under {root_dir}")
    if preferred_stem:
        for candidate in matches:
            if candidate.stem == preferred_stem:
                return candidate
    return matches[0]


def _resolve_max_dense_elems(cfg, device, dtype):
    """Return the maximum dense element count allowed for the dataset."""
    if cfg.max_dense_elems is not None and cfg.max_dense_fraction is not None:
        raise ValueError("Set only one of max_dense_elems or max_dense_fraction.")

    if cfg.max_dense_elems is not None:
        max_elems = int(cfg.max_dense_elems)
    elif cfg.max_dense_fraction is not None:
        if device.type != "cuda":
            raise RuntimeError("max_dense_fraction requires a CUDA device.")
        if not (0 < cfg.max_dense_fraction <= 1):
            raise ValueError("max_dense_fraction must be in (0, 1].")
        total_mem = torch.cuda.get_device_properties(device).total_memory
        elem_size = torch.tensor([], dtype=dtype).element_size()
        max_elems = int(total_mem * cfg.max_dense_fraction // elem_size)
    else:
        max_elems = int(cfg.densify_max_elems)

    if max_elems <= 0:
        raise ValueError("Maximum dense elements must be positive.")
    return max_elems


def _compute_submatrix_shape(m, n, max_elems):
    """Return submatrix shape that fits within max_elems while keeping aspect ratio."""
    if max_elems <= 0:
        raise ValueError("max_elems must be positive.")
    aspect = m / n
    rows = min(m, max(1, int(math.sqrt(max_elems * aspect))))
    cols = min(n, max(1, int(max_elems // rows)))
    return rows, cols


def _resolve_submatrix_shape(m, n, cfg, max_elems):
    """Return the submatrix shape and whether a submatrix is used."""
    if m * n <= max_elems:
        return m, n, False
    if cfg.auto_submatrix:
        rows, cols = _compute_submatrix_shape(m, n, max_elems)
        return rows, cols, True
    if cfg.submatrix_rows is None or cfg.submatrix_cols is None:
        raise ValueError("submatrix_rows/submatrix_cols must be set when auto_submatrix=False.")
    rows = min(cfg.submatrix_rows, m)
    cols = min(cfg.submatrix_cols, n)
    if cfg.enforce_max_elems and rows * cols > max_elems:
        raise ValueError(
            "Requested submatrix exceeds max_elems; reduce submatrix_rows/submatrix_cols or increase max."
        )
    return rows, cols, True


def _densify_matrix(matrix, cfg, max_elems):
    """Return a dense numpy array from a sparse matrix with optional submatrixing."""
    m, n = matrix.shape
    rows, cols, used_submatrix = _resolve_submatrix_shape(m, n, cfg, max_elems)

    if not used_submatrix:
        return matrix.toarray(), {"used_submatrix": False, "rows": m, "cols": n}

    if not scipy.sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()

    rng = np.random.default_rng(cfg.seed)
    row_start = int(rng.integers(0, m - rows + 1)) if m > rows else 0
    col_start = int(rng.integers(0, n - cols + 1)) if n > cols else 0
    submatrix = matrix[row_start : row_start + rows, col_start : col_start + cols]
    return (
        submatrix.toarray(),
        {
            "used_submatrix": True,
            "rows": rows,
            "cols": cols,
            "row_start": row_start,
            "col_start": col_start,
        },
    )


def load_suitesparse_matrix(cfg, device):
    """Download, load, and densify a SuiteSparse matrix into a torch tensor."""
    manual_seed(cfg.seed)

    extract_dir = download_matrix(cfg.group, cfg.name)
    mtx_path = _find_mtx_file(extract_dir, cfg.name)

    matrix = scipy.io.mmread(str(mtx_path))
    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.coo_matrix(matrix)
    if cfg.transpose:
        matrix = matrix.transpose()

    dtype = resolve_dtype(cfg.dtype)
    if dtype != torch.float32:
        raise ValueError("Only fp32 is supported for SuiteSparse datasets.")
    device = torch.device(device)
    max_elems = _resolve_max_dense_elems(cfg, device, dtype)
    dense, subinfo = _densify_matrix(matrix, cfg, max_elems)

    A = torch.tensor(dense, device=device, dtype=dtype)
    metadata = {
        "dataset": cfg.dataset,
        "group": cfg.group,
        "name": cfg.name,
        "shape": list(A.shape),
        "seed": cfg.seed,
        "dtype": cfg.dtype,
        "densify_max_elems": cfg.densify_max_elems,
        "submatrix_rows": cfg.submatrix_rows,
        "submatrix_cols": cfg.submatrix_cols,
        "max_dense_elems": cfg.max_dense_elems,
        "max_dense_fraction": cfg.max_dense_fraction,
        "auto_submatrix": cfg.auto_submatrix,
        "enforce_max_elems": cfg.enforce_max_elems,
        "transpose": cfg.transpose,
        "resolved_max_elems": max_elems,
        "used_submatrix": subinfo["used_submatrix"],
        "selected_rows": subinfo["rows"],
        "selected_cols": subinfo["cols"],
        "row_start": subinfo.get("row_start"),
        "col_start": subinfo.get("col_start"),
    }
    return A, metadata
