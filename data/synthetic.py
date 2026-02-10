from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass

import torch

import globals as g
from torch_utils import manual_seed, resolve_dtype


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Config for generating a synthetic dense matrix."""

    dataset: str = g.DATASET_SYNTHETIC
    name: str = "synthetic_gaussian"
    d: int = 4096
    n: int = 1024
    distribution: str = g.DIST_GAUSSIAN
    rank: int = 64
    noise: float = 0.01
    seed: int = 0
    dtype: str = g.DTYPE_FP32


def make_synthetic_matrix(cfg, device):
    """Generate a synthetic dense matrix A and metadata."""
    manual_seed(cfg.seed)

    dtype = resolve_dtype(cfg.dtype)
    d, n = cfg.d, cfg.n

    if dtype != torch.float32:
        raise ValueError("Only fp32 is supported for synthetic datasets.")

    if cfg.distribution == g.DIST_GAUSSIAN:
        A = torch.randn((d, n), device=device, dtype=dtype)
    elif cfg.distribution == g.DIST_RADEMACHER:
        A = torch.randint(0, 2, (d, n), device=device, dtype=torch.int32)
        A = (A * 2 - 1).to(dtype=dtype)
    elif cfg.distribution == g.DIST_LOW_RANK:
        U = torch.randn((d, cfg.rank), device=device, dtype=dtype)
        V = torch.randn((cfg.rank, n), device=device, dtype=dtype)
        A = U @ V
        if cfg.noise > 0:
            A = A + cfg.noise * torch.randn((d, n), device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown distribution: {cfg.distribution}")

    metadata = {
        "dataset": cfg.dataset,
        "name": cfg.name,
        "shape": [d, n],
        "distribution": cfg.distribution,
        "seed": cfg.seed,
        "dtype": cfg.dtype,
    }
    return A, metadata
