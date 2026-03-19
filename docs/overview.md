# Repository Overview

This is a neuroscience encoding analysis pipeline that studies how GPT-2-XL language model embeddings correlate with intracranial EEG (iEEG) neural activity from human subjects listening to a podcast. The pipeline trains ridge regression encoding models, performs permutation-based significance testing, and conducts SVD analysis to identify which directions in embedding space best predict neural responses.

## Pipeline Stages

1. **Significance testing** (`make sig`) — identifies electrode-lag pairs with statistically significant encoding
2. **Encoding model training** (`make train`) — trains ridge regression models on significant electrodes
3. **SVD analysis** (`make svd`) — decomposes model weights to find dominant directions in embedding space
4. **Visualization** (`make plots`) — generates correlation plots with optional significance overlays

---

## File Reference

### Root

| File | Purpose |
|------|---------|
| `Makefile` | Build targets for all pipeline stages; supports SLURM or local execution |
| `submit.sh` | SLURM job submission script (1 CPU, 1 GPU, 32GB RAM, 2hr); activates `align_env` virtualenv |
| `pytest.ini` | Pytest config with custom markers and test path |

### `configs/`

YAML config files passed to scripts via `--config`:

| File | Purpose |
|------|---------|
| `encoding_mid.yml` | Encoding config for GPT-2-XL layer 24 (mid-layer) |
| `encoding_final.yml` | Encoding config for GPT-2-XL layer 48 (final layer) |
| `significance_default.yml` | Default significance testing (5000 permutations, FDR q=0.01) |
| `significance_fast.yml` | Reduced permutations for quick iteration |
| `svd_default.yml` | SVD on mid-layer (layer 24) with k values [1,2,5,…,1600] and 0.95 retention threshold |

### `scripts/`

Main entry points for the pipeline:

| File | Purpose |
|------|---------|
| `run_significance.py` | Permutation-test significance per electrode-lag pair; outputs `significance_detailed.csv` and `all_subject_sig.csv` |
| `train_encoding_models.py` | Trains ridge regression models on significant electrodes; outputs coefficients, correlations, and plots |
| `generate_plots.py` | Generates or regenerates correlation plots from existing run results; supports backfilling SVD plots for cross_subject, per_subject, and per_region modes (auto-detected or via `--svd-modes`) |
| `run_svd_analysis.py` | SVD decomposition of encoding weights; evaluates truncation at various k values and selects optimal k |

### `utils/`

Core library modules:

| File | Purpose |
|------|---------|
| `config.py` | Dataclass definitions for `EncodingConfig`, `SignificanceConfig`, `SVDConfig`; `get_results_dir()` for canonical output paths |
| `config_utils.py` | Argument parsing and YAML loading with CLI override support (dot-notation, e.g. `--encoding.layer=48`) |
| `data_loaders.py` | Loads iEEG BIDS data via MNE, GPT-2 embeddings from HDF5, builds word-aligned epochs, handles time offsets |
| `embeddings.py` | Loads contextual GPT-2-XL token embeddings from HDF5 and averages tokens per word |
| `encoding.py` | K-fold cross-validation ridge regression (Himalaya backend), coefficient serialization, cross-subject aggregation |
| `significance.py` | GPU-accelerated permutation testing, vectorized Pearson correlation, Benjamini-Hochberg FDR correction |
| `analysis.py` | SVD decomposition of weights, truncated correlation evaluation, best-k selection, residualization utilities |
| `plotting.py` | Matplotlib helpers for correlation vs lag plots, significance overlays, SVD truncation analysis figures |
| `examples.py` | Generates context windows and markdown tables of example words for qualitative analysis |
| `__init__.py` | Public API — re-exports all key functions for convenient imports |

### `tests/`

Unit tests with full pipeline coverage:

| File | Purpose |
|------|---------|
| `conftest.py` | Shared fixtures: synthetic (X, Y) data, CSV inputs |
| `test_config_utils.py` | Config loading, CLI override parsing, nested attribute access |
| `test_data_loaders.py` | Subject mapping, electrode file parsing |
| `test_embeddings.py` | Embedding loading and per-word averaging |
| `test_encoding.py` | Ridge regression, k-fold CV, result serialization |
| `test_analysis.py` | SVD decomposition, truncation evaluation, best-k selection |
| `test_significance.py` | Permutation testing, FDR correction |
| `test_examples.py` | Context building and table generation |

### `notebooks/`

Exploratory Jupyter notebooks (not part of the main pipeline):

| File | Purpose |
|------|---------|
| `cluster-extraction.py` | Cluster analysis on neural/embedding data |
| `dictionary.py` | Word dictionary and vocabulary analysis |
| `word-analysis.py` | Word-level qualitative analysis |
| `svd-interpretability.py` | Interactive marimo notebook for SVD direction interpretability, fold consistency, cross-source comparison, and subspace analyses |

### `processed_data/`

Pre-computed embeddings and intermediate results cached to disk.
