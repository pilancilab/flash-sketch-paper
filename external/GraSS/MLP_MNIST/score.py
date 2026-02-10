"""
MLP + MNIST experiment for TRAK attribution.

Usage:
    python score.py --proj_type normal --proj_dim 1024
    python score.py --proj_type sjlt --proj_dim 4096
    python score.py --proj_type random_mask --proj_dim 512

Projection types: normal, rademacher, sjlt, sjlt_kernel, sjlt_cusparse,
gaussian_dense_cublas, srht_fwht, random_mask, grass, identity, flashsketch,
flashsketch_trans, sjlt_kernel_grass, flashsketch_grass, flashsketch_trans_grass
"""

import argparse
import os
import sys
import numpy as np
import time

# Add parent directory to path for _dattri import
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import torch
from torch import nn

from _dattri.algorithm.trak import TRAKAttributor
from _dattri.benchmark.load import load_benchmark
from _dattri.benchmark.utils import SubsetSampler
from _dattri.metric import lds
from _dattri.task import AttributionTask

def create_validation_split(
    test_dataset,
    test_sampler,
    groundtruth,
    val_ratio=0.1,
    seed=0,
    batch_size=64,
):
    """
    Split the test set into validation and test sets, and reconstruct groundtruth.

    Args:
        test_dataset: The test dataset
        test_sampler: The test sampler
        groundtruth: Tuple of (ground_truth_values, subset_indices)
        val_ratio: Ratio of test data to use for validation
        seed: Random seed for reproducibility
        batch_size: Batch size for validation/test dataloaders

    Returns:
        val_dataloader: DataLoader for validation set
        test_dataloader: DataLoader for test set
        val_gt: Groundtruth for validation set
        test_gt: Groundtruth for test set
    """
    # Get test indices from the sampler
    test_indices = list(test_sampler)
    num_test = len(test_indices)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Shuffle indices and split
    np.random.shuffle(test_indices)
    num_val = int(val_ratio * num_test)
    val_indices = test_indices[:num_val]
    new_test_indices = test_indices[num_val:]

    # Create validation and test samplers
    val_sampler = SubsetSampler(val_indices)
    new_test_sampler = SubsetSampler(new_test_indices)

    # Create dataloaders
    val_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
    )

    new_test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=new_test_sampler,
    )

    # Reconstruct groundtruth
    original_gt_values, subset_indices = groundtruth

    # Map original test indices to positions in the groundtruth tensor
    test_indices_dict = {idx: pos for pos, idx in enumerate(test_sampler)}

    # Extract validation groundtruth
    val_gt_indices = [test_indices_dict[idx] for idx in val_indices]
    val_gt_values = original_gt_values[:, val_gt_indices]

    # Extract test groundtruth
    test_gt_indices = [test_indices_dict[idx] for idx in new_test_indices]
    test_gt_values = original_gt_values[:, test_gt_indices]

    # Return validation and test sets with reconstructed groundtruth
    return (
        val_dataloader,
        new_test_dataloader,
        (val_gt_values, subset_indices),
        (test_gt_values, subset_indices)
    )


def _env_override(name, cast):
    """Return an override value from env if present."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid env override for {name}: {raw}") from exc


def _summary_stats(values):
    """Return mean/std for a list of floats (std is population)."""
    if not values:
        return None, None
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = tensor.mean().item()
    std = tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0
    return float(mean), float(std)


def _apply_env_overrides(args):
    """Apply environment overrides to parsed args."""
    proj_type = _env_override("GRASS_PROJ_TYPE", str)
    if proj_type is not None:
        args.proj_type = proj_type

    proj_dim = _env_override("GRASS_PROJ_DIM", int)
    if proj_dim is not None:
        args.proj_dim = proj_dim

    device = _env_override("GRASS_DEVICE", str)
    if device is not None:
        args.device = device

    seed = _env_override("GRASS_SEED", int)
    if seed is not None:
        args.seed = seed

    val_ratio = _env_override("GRASS_VAL_RATIO", float)
    if val_ratio is not None:
        args.val_ratio = val_ratio

    sjlt_c = _env_override("GRASS_SJLT_C", int)
    if sjlt_c is not None:
        args.sjlt_c = sjlt_c

    batch_size = _env_override("GRASS_BATCH_SIZE", int)
    if batch_size is not None:
        args.batch_size = batch_size

    proj_max_batch_size = _env_override("GRASS_PROJ_MAX_BATCH_SIZE", int)
    if proj_max_batch_size is not None:
        args.proj_max_batch_size = proj_max_batch_size

    flash_kappa = _env_override("GRASS_FLASH_KAPPA", int)
    if flash_kappa is not None:
        args.flashsketch_kappa = flash_kappa

    flash_s = _env_override("GRASS_FLASH_S", int)
    if flash_s is not None:
        args.flashsketch_s = flash_s

    flash_block_rows = _env_override("GRASS_FLASH_BLOCK_ROWS", int)
    if flash_block_rows is not None:
        args.flashsketch_block_rows = flash_block_rows

    flash_seed = _env_override("GRASS_FLASH_SEED", int)
    if flash_seed is not None:
        args.flashsketch_seed = flash_seed

def main():
    parser = argparse.ArgumentParser(description="TRAK attribution for MLP on MNIST")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--proj_type",
        type=str,
        default="normal",
        help=(
            "Projection type: normal, rademacher, sjlt, sjlt_kernel, sjlt_cusparse, "
            "gaussian_dense_cublas, srht_fwht, fjlt, random_mask, selective_mask, "
            "grass, grass_N, selective_grass, selective_grass_N, flashsketch, "
            "flashsketch_trans, sjlt_kernel_grass, flashsketch_grass, "
            "flashsketch_trans_grass, sjlt_cusparse_grass, gaussian_dense_cublas_grass, "
            "srht_fwht_grass, identity"
        ),
    )
    parser.add_argument("--proj_dim", type=int, default=1024, help="Projection dimension")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of test data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sjlt_c", type=int, default=1, help="SJLT column sparsity")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loaders")
    parser.add_argument("--proj_max_batch_size", type=int, default=64, help="Max batch size for projection")
    parser.add_argument("--flashsketch_kappa", type=int, default=2, help="FlashSketch kappa (nnz multiplier)")
    parser.add_argument("--flashsketch_s", type=int, default=2, help="FlashSketch s (nnz per block)")
    parser.add_argument("--flashsketch_block_rows", type=int, default=128, help="FlashSketch block row size")
    parser.add_argument("--flashsketch_seed", type=int, default=None, help="FlashSketch seed override")
    args = parser.parse_args()
    _apply_env_overrides(args)

    # Define the grid of damping values to search
    damping_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    # Print the settings
    print("Settings: MLP + MNIST")
    print("Projection Type:", args.proj_type)
    print("Projection Dimension:", args.proj_dim)
    print("Validation Split Ratio:", args.val_ratio)
    print("Random Seed:", args.seed)
    print("Damping Grid Search Values:", damping_values)

    # Create MNIST dataset
    model_details, groundtruth = load_benchmark(model="mlp", dataset="mnist", metric="lds")

    print("Original groundtruth shapes:")
    print("Ground truth values shape:", groundtruth[0].shape)
    print("Subset indices shape:", groundtruth[1].shape)

    model = model_details["model"].to(args.device)
    model = model.eval()

    # Define loss functions
    def f(params, data_target_pair):
        image, label = data_target_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        logp = -loss(yhat, label_t)
        return logp - torch.log(1 - torch.exp(logp))

    def m(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        p = torch.exp(-loss(yhat, label_t))
        return p

    # Create train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        model_details["train_dataset"],
        batch_size=args.batch_size,
        sampler=model_details["train_sampler"],
    )

    # Split test data into validation and test sets
    val_dataloader, test_dataloader, val_gt, test_gt = create_validation_split(
        model_details["test_dataset"],
        model_details["test_sampler"],
        groundtruth,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print("\nAfter splitting:")
    print("Validation groundtruth values shape:", val_gt[0].shape)
    print("Test groundtruth values shape:", test_gt[0].shape)

    # Create task
    task = AttributionTask(model=model, loss_func=f, checkpoints=model_details["models_half"][:10])

    # Setup projector kwargs
    projector_kwargs = {
        "device": args.device,
        "proj_type": args.proj_type,
        "proj_seed": args.seed,
        "proj_dim": args.proj_dim,
        "proj_max_batch_size": args.proj_max_batch_size,
        "proj_sjlt_c": args.sjlt_c,
    }
    if args.proj_type in (
        "flashsketch",
        "flashsketch_trans",
        "flashsketch_grass",
        "flashsketch_trans_grass",
    ):
        projector_kwargs["proj_kappa"] = args.flashsketch_kappa
        projector_kwargs["proj_s"] = args.flashsketch_s
        projector_kwargs["proj_block_rows"] = args.flashsketch_block_rows
        if args.flashsketch_seed is not None:
            projector_kwargs["proj_seed"] = args.flashsketch_seed

    # Grid search over damping values
    best_damping = None
    best_lds_score = float('-inf')
    validation_results = {}

    # Prepare attributor and cache once (reused for different damping values)
    base_attributor = TRAKAttributor(
        task=task,
        correct_probability_func=m,
        device=args.device,
        projector_kwargs=projector_kwargs,
    )

    # Grid search through damping values
    print("\nPerforming grid search over damping values:")
    for damping in damping_values:
        print(f"\nEvaluating damping = {damping}")

        # Update the regularization parameter
        base_attributor.regularization = damping
        base_attributor.cache(train_dataloader)
        # Evaluate on validation set
        with torch.no_grad():
            val_score = base_attributor.attribute(test_dataloader=val_dataloader)

        val_lds_score = lds(val_score, val_gt)[0]
        mean_val_lds = torch.mean(val_lds_score[~torch.isnan(val_lds_score)]).item()
        validation_results[damping] = mean_val_lds

        print(f"Damping: {damping}, Validation LDS: {mean_val_lds}")

        # Track best damping value
        if mean_val_lds > best_lds_score:
            best_lds_score = mean_val_lds
            best_damping = damping

    print("\nValidation Results:")
    for damping, score in validation_results.items():
        print(f"Damping: {damping}, LDS: {score}")

    print(f"\nBest damping value: {best_damping} (Validation LDS: {best_lds_score})")

    # Evaluate the best damping value on the test set
    print("\nEvaluating best damping value on test set...")
    base_attributor.regularization = best_damping

    if hasattr(base_attributor, "reset_projection_timer"):
        base_attributor.track_projection_time = True
        base_attributor.reset_projection_timer()

    proj_start = time.perf_counter()
    base_attributor.cache(train_dataloader)
    proj_time_s = time.perf_counter() - proj_start
    with torch.no_grad():
        test_score = base_attributor.attribute(test_dataloader=test_dataloader)

    test_lds_score = lds(test_score, test_gt)[0]
    test_lds_valid = test_lds_score[~torch.isnan(test_lds_score)]
    mean_test_lds = torch.mean(test_lds_valid).item()
    std_test_lds = (
        torch.std(test_lds_valid, unbiased=False).item()
        if test_lds_valid.numel() > 1
        else 0.0
    )

    print(f"Final Results:")
    print(f"Best Damping: {best_damping}")
    print(f"Validation LDS: {best_lds_score}")
    print(f"Test LDS: {mean_test_lds}")

    proj_only_time_s = getattr(base_attributor, "projection_only_time_s", None)
    proj_build_time_s = getattr(base_attributor, "projection_build_time_s", None)
    proj_only_batches = getattr(base_attributor, "projection_only_batches", None)
    proj_time_mean_s, proj_time_std_s = _summary_stats([float(proj_time_s)])
    proj_only_time_s_list = getattr(base_attributor, "projection_only_time_s_list", [])
    proj_build_time_s_list = getattr(base_attributor, "projection_build_time_s_list", [])
    proj_only_time_mean_s, proj_only_time_std_s = _summary_stats(proj_only_time_s_list)
    proj_build_time_mean_s, proj_build_time_std_s = _summary_stats(proj_build_time_s_list)
    result = {
        "best_damping": best_damping,
        "lds": mean_test_lds,
        "lds_mean": mean_test_lds,
        "lds_std": std_test_lds,
        "lds_samples": int(test_lds_valid.numel()),
        "proj_time_s": proj_time_s,
        "proj_time_ms": proj_time_s * 1000.0,
        "proj_time_mean_ms": (
            float(proj_time_mean_s) * 1000.0 if proj_time_mean_s is not None else None
        ),
        "proj_time_std_ms": (
            float(proj_time_std_s) * 1000.0 if proj_time_std_s is not None else None
        ),
        "proj_time_samples": 1,
        "proj_only_time_s": proj_only_time_s,
        "proj_only_time_ms": (
            proj_only_time_s * 1000.0 if proj_only_time_s is not None else None
        ),
        "proj_only_time_mean_ms": (
            float(proj_only_time_mean_s) * 1000.0
            if proj_only_time_mean_s is not None
            else None
        ),
        "proj_only_time_std_ms": (
            float(proj_only_time_std_s) * 1000.0
            if proj_only_time_std_s is not None
            else None
        ),
        "proj_only_time_samples": int(len(proj_only_time_s_list)),
        "proj_build_time_s": proj_build_time_s,
        "proj_build_time_ms": (
            proj_build_time_s * 1000.0 if proj_build_time_s is not None else None
        ),
        "proj_build_time_mean_ms": (
            float(proj_build_time_mean_s) * 1000.0
            if proj_build_time_mean_s is not None
            else None
        ),
        "proj_build_time_std_ms": (
            float(proj_build_time_std_s) * 1000.0
            if proj_build_time_std_s is not None
            else None
        ),
        "proj_build_time_samples": int(len(proj_build_time_s_list)),
        "proj_only_batches": proj_only_batches,
    }

    os.makedirs("./results", exist_ok=True)
    torch.save(result, f"./results/{args.proj_type}-{args.proj_dim}.pt")

if __name__ == "__main__":
    main()
