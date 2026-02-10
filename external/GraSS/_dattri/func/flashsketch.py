"""FlashSketch BlockPerm projection adapter for GraSS."""

from __future__ import annotations

import os
import sys
from typing import Optional, Union

import torch
from torch import Tensor

from .utils import _vectorize as vectorize

ENV_GRASS_FLASH_SKIP_ZEROS = "GRASS_FLASH_SKIP_ZEROS"


def _resolve_flashsketch_root() -> str:
    """Return the FlashSketch repo root from environment."""
    root = os.environ.get("FLASH_SKETCH_ROOT")
    if not root:
        raise ValueError("FLASH_SKETCH_ROOT must point to the FlashSketch repo root.")
    return root


def _load_flashsketch_kernel(kernel_mode: str):
    """Import the FlashSketch BlockPerm kernel and config."""
    root = _resolve_flashsketch_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    if kernel_mode == "latest":
        from kernels.flashsketch.flashsketch import flashsketch_cuda_apply
        from sketches.flashsketch import FlashSketchConfig

        return flashsketch_cuda_apply, FlashSketchConfig, True
    raise ValueError(f"Unknown FlashSketch kernel_mode: {kernel_mode}")


def _env_flag(name: str, default: bool = False) -> bool:
    """Return True if an environment variable should be treated as truthy."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


class FlashSketchProjector:
    """FlashSketch projector wrapper for GraSS projection API."""

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int,
        seed: int,
        device: torch.device,
        *,
        kernel_mode: str = "latest",
        kappa: int = 2,
        s: int = 2,
        block_rows: int = 128,
        max_batch_size: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        skip_zeros: Optional[bool] = None,
    ) -> None:
        """Initialize FlashSketch projector settings."""
        if device.type != "cuda":
            raise ValueError("FlashSketch projector requires CUDA device.")
        if proj_dim <= 0:
            raise ValueError("proj_dim must be positive.")
        self.feature_dim = int(feature_dim)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.device = device
        self.kappa = int(kappa)
        self.s = int(s)
        self.block_rows = int(block_rows)
        if max_batch_size is not None:
            max_batch_size = int(max_batch_size)
            if max_batch_size <= 0:
                raise ValueError("max_batch_size must be positive.")
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.skip_zeros = (
            bool(skip_zeros)
            if skip_zeros is not None
            else _env_flag(ENV_GRASS_FLASH_SKIP_ZEROS, default=False)
        )
        self.kernel_mode = str(kernel_mode)
        self._kernel, self._config_cls, self._transpose_input = _load_flashsketch_kernel(
            self.kernel_mode
        )

    def project(
        self,
        features: Union[dict, Tensor],
        ensemble_id: int = 0,
    ) -> Tensor:
        """Project a batch of features with FlashSketch."""
        if isinstance(features, dict):
            features = vectorize(features, device=self.device)
        elif not isinstance(features, torch.Tensor):
            raise ValueError("features must be a Tensor or dict of Tensors.")

        if features.device != self.device:
            features = features.to(self.device)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        elif features.ndim > 2:
            features = features.flatten(start_dim=1)

        orig_dtype = features.dtype
        if orig_dtype != torch.float32:
            features = features.to(torch.float32)

        seed = self.seed + int(1e5) * int(ensemble_id)
        cfg = self._config_cls(
            k=self.proj_dim,
            kappa=self.kappa,
            s=self.s,
            seed=seed,
            dtype="fp32",
            block_rows=self.block_rows,
            skip_zeros=self.skip_zeros,
        )

        def _apply_kernel(batch: Tensor) -> Tensor:
            if self._transpose_input:
                x_t = batch.t().contiguous()
                y_t = self._kernel(x_t, cfg)
                return y_t.t().contiguous()
            return self._kernel(batch.contiguous(), cfg)

        batch_size = features.size(0)
        feature_dim = features.size(1)
        max_safe_batch = int((2**31 - 1) // max(1, feature_dim))
        if max_safe_batch < 1:
            raise ValueError(
                "FlashSketch feature_dim too large for int32 indexing; "
                "reduce feature_dim or use a different projector."
            )
        chunk_size = batch_size
        if self.max_batch_size is not None:
            chunk_size = min(chunk_size, self.max_batch_size)
        chunk_size = min(chunk_size, max_safe_batch)
        if chunk_size <= 0:
            raise ValueError("FlashSketch chunk size must be positive.")

        if batch_size <= chunk_size:
            out = _apply_kernel(features)
        else:
            out = torch.empty(
                (batch_size, self.proj_dim),
                device=self.device,
                dtype=torch.float32,
            )
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                out[start:end] = _apply_kernel(features[start:end])

        if orig_dtype != torch.float32:
            out = out.to(orig_dtype)
        return out

    def free_memory(self) -> None:
        """No-op for FlashSketch projector."""
        return None
