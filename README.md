# FlashSketch: Sketch-Kernel Co-Design for Fast Sparse Sketching on GPUs

This repo contains the **code and experiments** for the FlashSketch paper,
**"FlashSketch: Sketch-Kernel Co-Design for Fast Sparse Sketching on GPUs."**

[Arxiv paper link.](https://arxiv.org/abs/2602.06071)

It includes:
1) **FlashSketch kernels** and supporting CUDA extensions.
2) **End-to-end RNLA benchmarks** used in the paper (camera-ready and ablations).
3) Benchmarks for an end-to-end ML pipeline for data attribution ([GraSS](https://github.com/TRAIS-Lab/GraSS)).

## Quickstart

```bash
# SSH (recommended if you have keys)
git clone --recurse-submodules git@github.com:pilancilab/flash-sketch-paper.git
cd flash-sketch-paper
```

```bash
# HTTPS
git clone --recurse-submodules https://github.com/pilancilab/flash-sketch-paper.git
cd flash-sketch-paper
```

Setup the Python environment:

```bash
make setup
```

Build all kernels:

```bash
make -j kernels.build
```

Run tests:

```bash
make test.small  # fast, small test suite
make test.full   # exhaustive test suite
```

Run the camera-ready benchmark and generate figures:

```bash
make bench.camera-ready
make fig.camera-ready
```

Figures are written to `file_storage/figures/` by default.

To run the GraSS pipeline and generate its figures:

```bash
make bench.grass.external
make fig.grass.camera_ready
```

To run all the ablations in the Appendix and generate their figures:

```bash
make bench.ablation
make fig.ablation.nolegend
make fig.ablation.legend
```

---

## Overview

This README focuses on **repo content** and build/run entry points.

## Build and run

Kernel extensions are **prebuilt** by the Makefile and then loaded as shared objects.  
If a `.so` is missing, the loader errors out (no silent fallback).

Common targets:

```bash
make setup            # create .venv + install deps
make kernels.build    # build all kernel .so files
make kernels.clean    # remove file_storage/build (force rebuild)
make bench.camera-ready # camera-ready benchmarks (paper main)
make bench.ablation     # ablation benchmarks (paper appendix)
make fig.camera-ready   # regenerate paper-ready main figures
make fig.ablation.nolegend # regenerate appendix figures (nolegend)
make fig.grass.camera_ready # regenerate GraSS camera-ready figure
make fig.ablation.summary-table # regenerate randnla speedup table
make test.small         # quick validation tests
make test.full          # exhaustive grid tests
make paper            # build the PDF
```

Useful environment overrides (auto-detects GPU archs via nvidia-smi if unset):

```bash
CUDA_ARCHS="80 86" make kernels.build
CUDA_ARCH=86 make kernels.build
```

---

## The one interface that everything uses

All benchmarks interact with sketching through one primitive:

```python
def sketch(A, cfg):
    '''
    A: dense CUDA tensor, shape (d, n)  (d=input dimension, n=#vectors/columns)
    cfg: method-specific config dataclass (flat fields)
    returns:
      SA: dense CUDA tensor, shape (k, n)
    '''
    ...
    return SA
```

A “sketch method” is **plug-and-play** across benchmarks: implement `sketch(A, cfg) -> SA`, add it to the registry, and it becomes benchmarkable everywhere.

### Configs are flat

Each sketch method defines its own dataclass config (flat fields, no inheritance).  
Method configs include `method`, `k`, `seed`, `dtype`, plus method-specific knobs.

---

**Sketch Registry**
1. `flashsketch` - FlashSketch block-permutation sparse JL kernel (Ours, Algorithm 1 in the paper).
2. `flashblockrow` - FlashBlockRow block-row sampling sketch (Ours, Algorithm 2 in the paper).
3. `gaussian_dense_cublas` - Dense Gaussian JL transform via cuBLAS GEMM (baseline).
4. `sjlt_cusparse` - Sparse JL transform using cuSPARSE SpMM (baseline).
5. `srht_fwht` - Subsampled randomized Hadamard transform using the FWHT kernel.
6. `sjlt_grass_kernel` - GraSS SJLT CUDA kernel baseline.

---

## Folder structure

Target layout (you can adjust names slightly, but keep the separation):

```
globals.py                         # global string constants + common paths
tests/                            # correctness + regression tests

sketches/                          # user-facing sketch methods (wrappers)
  registry.py                      # SKETCH_REGISTRY: method_name -> (sketch_fn, config_cls)
  gaussian_dense_cublas.py         # baseline
  sjlt_cusparse.py                 # baseline
  flashblockrow.py                 # FlashBlockRow kernel
  flashsketch.py                   # FlashSketch kernel
  sjlt_grass_kernel.py             # GraSS SJLT baseline
  srht_fwht.py                     # SRHT baseline (fast FWHT backend by default)

kernels/                           # low-level kernels (no benchmark imports)
  flashblockrow/                   # CUDA + C++ kernel + bindings
  flashsketch/                     # CUDA + C++ kernel + bindings
  grass_sjlt/                      # CUDA + C++ kernel + bindings

data/                              # dataset loaders + generators
  synthetic.py                     # dense synthetic matrices
  suitesparse/                     # SuiteSparse tooling
    catalog.py
    download.py
    load.py

bench/e2e/                         # task implementations + metrics
  metrics.py
  report.py
  tasks/
    gram_approx.py
    ridge_regression.py
    sketch_and_solve_ls.py
    ose_error.py

bench/ablation/                    # ablation sweeps for core metrics
bench/camera_ready/                # camera-ready benchmark configs
bench/grass_external/              # GraSS integration experiments

analysis/figures_src/              # plot generators (config-driven)
paper/                             # LaTeX paper (figures live in paper/figures)
file_storage/                      # all outputs (runs, datasets cache, figures)
```

---

# Output format (so results are easy to analyze)

Each run should write:
- one machine-readable table (Parquet preferred)
- one small JSON manifest (provenance + config snapshot)
- optional: a small “summary.json” with best points for easy grepping

A single row in the table includes:
- dataset id + shape
- method name
- method config fields (k, s, etc.)
- timing metrics
- seed(s)
- environment metadata (GPU name, dtype, etc.)

# Citation

If you use FlashSketch in your research, please cite the paper:


```
@article{dwaraknath2026flashsketch,
      title={FlashSketch: Sketch-Kernel Co-Design for Fast Sparse Sketching on GPUs}, 
      author={Rajat Vadiraj Dwaraknath and Sungyoon Kim and Mert Pilanci},
      year={2026},
      eprint={2602.06071},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2602.06071}, 
}
```
