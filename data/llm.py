from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from dataclasses import dataclass
import hashlib
import json

import torch

import globals as g
from torch_utils import manual_seed, resolve_dtype


@dataclass(frozen=True)
class LlmWeightsDatasetConfig:
    """Config for loading an LLM weight matrix as a dense tensor."""

    dataset: str = g.DATASET_LLM
    name: str = "llm_weights"
    model_name: str = g.LLM_MODEL_GPT2
    param_name: str | None = g.LLM_PARAM_GPT2_MLP_CFC
    submatrix_rows: int | None = None
    submatrix_cols: int | None = None
    seed: int = 0
    dtype: str = g.DTYPE_FP32
    cache: bool = True


@dataclass(frozen=True)
class LlmGradientsDatasetConfig:
    """Config for loading an LLM gradient matrix as a dense tensor."""

    dataset: str = g.DATASET_LLM
    name: str = "llm_gradients"
    model_name: str = g.LLM_MODEL_GPT2
    param_name: str | None = g.LLM_PARAM_GPT2_MLP_CFC
    seq_len: int = g.LLM_DEFAULT_SEQ_LEN
    batch_size: int = g.LLM_DEFAULT_BATCH_SIZE
    submatrix_rows: int | None = None
    submatrix_cols: int | None = None
    seed: int = 0
    dtype: str = g.DTYPE_FP32
    cache: bool = True


def _load_model(model_name, device, dtype):
    """Return a causal LM model on the requested device."""
    if dtype != torch.float32:
        raise ValueError("Only fp32 is supported for LLM datasets.")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model


def _sanitize_name(value):
    """Return a filesystem-safe slug for cache names."""
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def _llm_cache_key(cfg):
    """Return a stable cache key and payload for an LLM dataset config."""
    payload = {
        "dataset": cfg.dataset,
        "name": cfg.name,
        "model_name": cfg.model_name,
        "param_name": cfg.param_name,
        "submatrix_rows": cfg.submatrix_rows,
        "submatrix_cols": cfg.submatrix_cols,
        "seed": cfg.seed,
        "dtype": cfg.dtype,
    }
    if isinstance(cfg, LlmGradientsDatasetConfig):
        payload["seq_len"] = cfg.seq_len
        payload["batch_size"] = cfg.batch_size
        payload["source"] = g.LLM_SOURCE_GRADIENTS
    else:
        payload["source"] = g.LLM_SOURCE_WEIGHTS
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:12]
    return key, payload


def _llm_cache_path(cfg):
    """Return the cache path and payload for an LLM dataset config."""
    key, payload = _llm_cache_key(cfg)
    model_slug = _sanitize_name(cfg.model_name)
    name_slug = _sanitize_name(cfg.name)
    cache_dir = g.LLM_CACHE_DIR() / model_slug
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name_slug}_{key}.pt", payload


def _load_llm_cache(cfg, device, dtype):
    """Return cached weights/metadata if available."""
    if not getattr(cfg, "cache", False):
        return None
    cache_path, payload = _llm_cache_path(cfg)
    if not cache_path.exists():
        return None
    cached = torch.load(cache_path, map_location="cpu")
    matrix = cached["matrix"].to(device=device, dtype=dtype)
    metadata = cached.get("metadata", {})
    metadata.update(
        {
            "cache_hit": True,
            "cache_path": str(cache_path),
            "cache_key": payload,
        }
    )
    return matrix, metadata


def _save_llm_cache(cfg, matrix, metadata, payload):
    """Persist an LLM matrix and metadata to the cache."""
    if not getattr(cfg, "cache", False):
        return
    cache_path, _ = _llm_cache_path(cfg)
    metadata = dict(metadata)
    metadata.update(
        {
            "cache_hit": False,
            "cache_path": str(cache_path),
            "cache_key": payload,
        }
    )
    torch.save(
        {"matrix": matrix.detach().cpu(), "metadata": metadata},
        cache_path,
    )


def _select_param(model, param_name):
    """Return a named parameter tensor from the model."""
    params = list(model.named_parameters())
    if param_name:
        for name, param in params:
            if name == param_name:
                return name, param
        raise ValueError(f"Parameter '{param_name}' not found in model.")

    candidates = [(name, param) for name, param in params if param.dim() == 2]
    if not candidates:
        raise ValueError("No 2D parameters available for LLM dataset.")
    name, param = max(candidates, key=lambda item: item[1].numel())
    return name, param


def _concat_param_matrices(matrices, names):
    """Concatenate 2D tensors along rows (dim=0)."""
    if not matrices:
        raise ValueError("No parameters available for concatenation.")
    base_shape = matrices[0].shape[1]
    for tensor in matrices[1:]:
        if tensor.shape[1] != base_shape:
            raise ValueError("Concatenated parameters must share the same column size.")
    return torch.cat(matrices, dim=0), names


def _concat_templates(model, templates, layer_indices):
    """Return named tensors for template-based concatenation."""
    params = dict(model.named_parameters())
    named_tensors = []
    for idx in layer_indices:
        for template, transpose in templates:
            name = template.format(layer=idx)
            if name not in params:
                raise ValueError(f"Parameter '{name}' not found in model.")
            tensor = params[name]
            if tensor.dim() != 2:
                raise ValueError(f"Parameter '{name}' is not 2D.")
            named_tensors.append((name, tensor, transpose))
    return named_tensors


def _concat_layer_matrices(named_tensors, use_grad=False):
    """Concatenate tensors (or grads) from named template tensors."""
    matrices = []
    names = []
    for name, tensor, transpose in named_tensors:
        source = tensor.grad if use_grad else tensor
        if source is None:
            raise ValueError(f"No gradient available for parameter '{name}'.")
        mat = source.detach()
        if transpose:
            mat = mat.t()
        matrices.append(mat)
        names.append(f"{name}^T" if transpose else name)
    return _concat_param_matrices(matrices, names)


def _slice_matrix(matrix, cfg):
    """Return a sliced matrix and slice metadata."""
    rows, cols = matrix.shape
    sub_rows = cfg.submatrix_rows or rows
    sub_cols = cfg.submatrix_cols or cols
    if sub_rows > rows or sub_cols > cols:
        raise ValueError(
            "Requested submatrix exceeds parameter shape; "
            "reduce submatrix_rows/submatrix_cols."
        )
    if sub_rows == rows and sub_cols == cols:
        return matrix, {"rows": rows, "cols": cols, "row_start": 0, "col_start": 0}

    manual_seed(cfg.seed)
    row_start = int(torch.randint(0, rows - sub_rows + 1, (1,), device="cpu").item())
    col_start = int(torch.randint(0, cols - sub_cols + 1, (1,), device="cpu").item())
    sub = matrix[row_start : row_start + sub_rows, col_start : col_start + sub_cols]
    return (
        sub.contiguous(),
        {"rows": sub_rows, "cols": sub_cols, "row_start": row_start, "col_start": col_start},
    )


def _concat_templates_for_model(model_name, full):
    """Return concat templates for a given model name."""
    if model_name in {g.LLM_MODEL_GPT2, g.LLM_MODEL_GPT2_MEDIUM, g.LLM_MODEL_DISTILGPT2}:
        if full:
            return (
                (g.LLM_PARAM_GPT2_ATTN_CATTN_TEMPLATE, True),
                (g.LLM_PARAM_GPT2_ATTN_CPROJ_TEMPLATE, False),
                (g.LLM_PARAM_GPT2_MLP_CFC_TEMPLATE, True),
                (g.LLM_PARAM_GPT2_MLP_CPROJ_TEMPLATE, False),
            )
        return (
            (g.LLM_PARAM_GPT2_MLP_CFC_TEMPLATE, False),
            (g.LLM_PARAM_GPT2_MLP_CPROJ_TEMPLATE, True),
        )
    if model_name == g.LLM_MODEL_QWEN2_1P5B:
        if full:
            return (
                (g.LLM_PARAM_QWEN2_ATTN_Q_TEMPLATE, False),
                (g.LLM_PARAM_QWEN2_ATTN_K_TEMPLATE, False),
                (g.LLM_PARAM_QWEN2_ATTN_V_TEMPLATE, False),
                (g.LLM_PARAM_QWEN2_ATTN_O_TEMPLATE, False),
                (g.LLM_PARAM_QWEN2_MLP_GATE_TEMPLATE, False),
                (g.LLM_PARAM_QWEN2_MLP_UP_TEMPLATE, False),
                (g.LLM_PARAM_QWEN2_MLP_DOWN_TEMPLATE, True),
            )
        return (
            (g.LLM_PARAM_QWEN2_MLP_GATE_TEMPLATE, False),
            (g.LLM_PARAM_QWEN2_MLP_DOWN_TEMPLATE, True),
        )
    raise ValueError(f"Unsupported model for concat templates: {model_name}")


def _get_layer_count(model):
    """Return the number of transformer layers for a model."""
    config = model.config
    if hasattr(config, "n_layer"):
        return int(config.n_layer)
    if hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    raise ValueError("Model config is missing layer count attribute.")


def load_llm_weights_matrix(cfg, device):
    """Load an LLM weight matrix into a torch tensor."""
    manual_seed(cfg.seed)
    dtype = resolve_dtype(cfg.dtype)
    if dtype != torch.float32:
        raise ValueError("Only fp32 is supported for LLM datasets.")
    device = torch.device(device)

    cached = _load_llm_cache(cfg, device, dtype)
    if cached is not None:
        return cached

    model = _load_model(cfg.model_name, device, dtype)
    concat_meta = {}
    if cfg.param_name == g.LLM_PARAM_CONCAT_LAYERS:
        layer_count = _get_layer_count(model)
        layer_indices = list(range(layer_count))
        templates = _concat_templates_for_model(cfg.model_name, full=False)
        params = _concat_templates(model, templates, layer_indices)
        matrix, _ = _concat_layer_matrices(params, use_grad=False)
        concat_meta = {
            "concat_layers": layer_count,
            "concat_templates": [tpl for tpl, _ in templates],
        }
        param_name = g.LLM_PARAM_CONCAT_LAYERS
    elif cfg.param_name == g.LLM_PARAM_CONCAT_LAYERS_FULL:
        layer_count = _get_layer_count(model)
        layer_indices = list(range(layer_count))
        templates = _concat_templates_for_model(cfg.model_name, full=True)
        params = _concat_templates(model, templates, layer_indices)
        matrix, _ = _concat_layer_matrices(params, use_grad=False)
        concat_meta = {
            "concat_layers": layer_count,
            "concat_templates": [tpl for tpl, _ in templates],
            "concat_variant": g.LLM_PARAM_CONCAT_LAYERS_FULL,
        }
        param_name = g.LLM_PARAM_CONCAT_LAYERS_FULL
    else:
        param_name, param = _select_param(model, cfg.param_name)
        matrix = param.detach()
    matrix, slice_info = _slice_matrix(matrix, cfg)
    matrix = matrix.to(device=device, dtype=dtype)

    metadata = {
        "dataset": cfg.dataset,
        "name": cfg.name,
        "group": cfg.model_name,
        "shape": list(matrix.shape),
        "seed": cfg.seed,
        "dtype": cfg.dtype,
        "variant": g.LLM_SOURCE_WEIGHTS,
        "param_name": param_name,
        "submatrix_rows": cfg.submatrix_rows,
        "submatrix_cols": cfg.submatrix_cols,
        "row_start": slice_info["row_start"],
        "col_start": slice_info["col_start"],
    }
    metadata.update(concat_meta)
    _save_llm_cache(cfg, matrix, metadata, _llm_cache_key(cfg)[1])
    return matrix, metadata


def _make_random_batch(model, cfg, device):
    """Return random token ids and labels for gradient computation."""
    vocab = int(model.config.vocab_size)
    batch = int(cfg.batch_size)
    seq_len = int(cfg.seq_len)
    if batch <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive for LLM gradients.")
    inputs = torch.randint(0, vocab, (batch, seq_len), device=device)
    labels = inputs.clone()
    return inputs, labels


def load_llm_gradients_matrix(cfg, device):
    """Load an LLM gradient matrix into a torch tensor."""
    manual_seed(cfg.seed)
    dtype = resolve_dtype(cfg.dtype)
    if dtype != torch.float32:
        raise ValueError("Only fp32 is supported for LLM datasets.")
    device = torch.device(device)

    cached = _load_llm_cache(cfg, device, dtype)
    if cached is not None:
        return cached

    model = _load_model(cfg.model_name, device, dtype)
    concat_meta = {}
    if cfg.param_name == g.LLM_PARAM_CONCAT_LAYERS:
        layer_count = int(model.config.n_layer)
        layer_indices = list(range(layer_count))
        templates = (
            (g.LLM_PARAM_GPT2_MLP_CFC_TEMPLATE, False),
            (g.LLM_PARAM_GPT2_MLP_CPROJ_TEMPLATE, True),
        )
        params = _concat_templates(model, templates, layer_indices)
        param = None
        param_name = g.LLM_PARAM_CONCAT_LAYERS
    elif cfg.param_name == g.LLM_PARAM_CONCAT_LAYERS_FULL:
        layer_count = int(model.config.n_layer)
        layer_indices = list(range(layer_count))
        templates = (
            (g.LLM_PARAM_GPT2_ATTN_CATTN_TEMPLATE, True),
            (g.LLM_PARAM_GPT2_ATTN_CPROJ_TEMPLATE, False),
            (g.LLM_PARAM_GPT2_MLP_CFC_TEMPLATE, True),
            (g.LLM_PARAM_GPT2_MLP_CPROJ_TEMPLATE, False),
        )
        params = _concat_templates(model, templates, layer_indices)
        param = None
        param_name = g.LLM_PARAM_CONCAT_LAYERS_FULL
    else:
        param_name, param = _select_param(model, cfg.param_name)
    model.zero_grad(set_to_none=True)

    inputs, labels = _make_random_batch(model, cfg, device)
    outputs = model(input_ids=inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

    if param is not None:
        if param.grad is None:
            raise ValueError(f"No gradient available for parameter '{param_name}'.")
        matrix = param.grad.detach()
    else:
        matrix, _ = _concat_layer_matrices(params, use_grad=True)
        concat_meta = {
            "concat_layers": layer_count,
            "concat_templates": [tpl for tpl, _ in templates],
            "concat_variant": param_name,
        }

    matrix, slice_info = _slice_matrix(matrix, cfg)
    matrix = matrix.to(device=device, dtype=dtype)

    metadata = {
        "dataset": cfg.dataset,
        "name": cfg.name,
        "group": cfg.model_name,
        "shape": list(matrix.shape),
        "seed": cfg.seed,
        "dtype": cfg.dtype,
        "variant": g.LLM_SOURCE_GRADIENTS,
        "param_name": param_name,
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "submatrix_rows": cfg.submatrix_rows,
        "submatrix_cols": cfg.submatrix_cols,
        "row_start": slice_info["row_start"],
        "col_start": slice_info["col_start"],
    }
    metadata.update(concat_meta)
    _save_llm_cache(cfg, matrix, metadata, _llm_cache_key(cfg)[1])
    return matrix, metadata
