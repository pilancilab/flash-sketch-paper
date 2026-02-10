from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()

from typing import Optional, Union

import torch
from torch import Tensor

from _dattri.func.utils import _vectorize as vectorize
from kernels.grass_sjlt.grass_sjlt import sjlt_projection_cuda


class SjltKernelProjector:
    """Internal GraSS SJLT kernel projector."""

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int,
        seed: int,
        device: torch.device,
        *,
        s: int,
        max_batch_size: Optional[int] = None,
        threads: int = 1024,
        fixed_blocks: int = 84,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("SjltKernelProjector requires CUDA device.")
        if proj_dim <= 0:
            raise ValueError("proj_dim must be positive.")
        if s <= 0:
            raise ValueError("s must be positive.")
        if max_batch_size is not None:
            max_batch_size = int(max_batch_size)
            if max_batch_size <= 0:
                raise ValueError("max_batch_size must be positive.")
        self.feature_dim = int(feature_dim)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.device = device
        self.s = int(s)
        self.max_batch_size = max_batch_size
        self.threads = int(threads)
        self.fixed_blocks = int(fixed_blocks)
        self.dtype = dtype
        self._rand_indices: Optional[Tensor] = None
        self._rand_signs: Optional[Tensor] = None
        self._ensemble_id: Optional[int] = None
        self._ensure_randomness(0)

    def _ensure_randomness(self, ensemble_id: int) -> None:
        if self._ensemble_id == ensemble_id:
            return
        generator = torch.Generator(device=self.device)
        seed = self.seed + int(1e5) * int(ensemble_id)
        generator.manual_seed(int(seed))
        self._rand_indices = torch.randint(
            0,
            self.proj_dim,
            (self.feature_dim, self.s),
            generator=generator,
            device=self.device,
            dtype=torch.int64,
        )
        rand_signs = torch.randint(
            0,
            2,
            (self.feature_dim, self.s),
            generator=generator,
            device=self.device,
            dtype=torch.int8,
        )
        self._rand_signs = rand_signs * 2 - 1
        self._ensemble_id = ensemble_id

    def _project_batch(self, features: Tensor) -> Tensor:
        output = sjlt_projection_cuda(
            features,
            self._rand_indices,
            self._rand_signs,
            self.proj_dim,
            self.s,
            self.threads,
            self.fixed_blocks,
        )[0]
        return output

    def project(self, features: Union[dict, Tensor], ensemble_id: int = 0) -> Tensor:
        """Project a batch of features with the internal SJLT kernel."""
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

        self._ensure_randomness(ensemble_id)

        batch_size = features.size(0)
        feature_dim = features.size(1)
        max_safe_batch = int((2**31 - 1) // max(1, feature_dim))
        if max_safe_batch < 1:
            raise ValueError(
                "SjltKernelProjector feature_dim too large for int32 indexing; "
                "reduce feature_dim or use a different projector."
            )
        chunk_size = batch_size
        if self.max_batch_size is not None:
            chunk_size = min(chunk_size, self.max_batch_size)
        chunk_size = min(chunk_size, max_safe_batch)
        if chunk_size <= 0:
            raise ValueError("SjltKernelProjector chunk size must be positive.")

        if batch_size <= chunk_size:
            out = self._project_batch(features.contiguous())
        else:
            out = torch.empty(
                (batch_size, self.proj_dim),
                device=self.device,
                dtype=torch.float32,
            )
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                out[start:end] = self._project_batch(features[start:end].contiguous())

        if orig_dtype != torch.float32:
            out = out.to(orig_dtype)
        return out

    def free_memory(self) -> None:
        """No-op for SJLT kernel projector."""
        return None
