# Repository Overview

This is a neuroscience encoding analysis pipeline that studies how GPT-2-XL language model embeddings correlate with intracranial EEG (iEEG) neural activity from human subjects listening to a podcast. The pipeline trains ridge regression encoding models, performs permutation-based significance testing, and conducts SVD analysis to identify which directions in embedding space best predict neural responses.

## Pipeline Stages

1. **Significance testing** (`make sig`) — identifies electrode-lag pairs with statistically significant encoding
2. **Encoding model training** (`make train`) — trains ridge regression models on significant electrodes
3. **SVD analysis** (`make svd`) — decomposes model weights to find dominant directions in embedding space
4. **Partial SVD analysis** (`make partial-svd`) — removes contiguous windows of SVD directions from word embeddings and retrains ridge regression to measure how much each group contributes to encoding performance
5. **SRRR analysis** (`make srrr`) — finds the minimal subset of SVD directions that explains all significant encoding variance, via stability selection and a statistical sufficiency test
6. **RRR analysis** (`make rrr`) — raw-input RRR baseline: sweeps over ranks to find the minimal rank that captures full-ridge variance, directly in embedding space (no SVD projection)
7. **Partial RRR analysis** (`make partial-rrr`) — partials out RRR-learned subspaces from word embeddings and retrains ridge regression to measure how much each rank-k subspace contributes to encoding performance
8. **SAE analysis** (`make sae`) — projects embeddings into an RRR-learned k-dimensional subspace and trains a sparse autoencoder (overcomplete hidden layer with ReLU + L1 penalty) followed by regression to predict neural targets; tests whether superposition in the RRR subspace can be decoded with more linear dimensions
9. **Visualization** (`make plots`) — generates correlation plots with optional significance overlays

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
| `partial_svd_default.yml` | Partial SVD analysis: single-direction sweep up to direction 50 |
| `partial_svd_per_region.yml` | Partial SVD variant with Destrieux atlas region plots |
| `partial_rrr_default.yml` | Partial RRR analysis: cross-subject subspace importance (rank_grid [1,2,5,10,20,50]) |
| `rrr_default.yml` | RRR cross-subject baseline (rank_grid [1,2,5,10,20,50]) |
| `rrr_per_subject.yml` | Per-subject RRR variant |
| `rrr_per_region.yml` | Per-region RRR variant with Destrieux atlas regions |
| `sae_default.yml` | SAE cross-subject default (k=10, hidden_dim=50, end-to-end, OLS, L1=0.001) |

### `scripts/`

Main entry points for the pipeline:

| File | Purpose |
|------|---------|
| `run_significance.py` | Permutation-test significance per electrode-lag pair; outputs `significance_detailed.csv` and `all_subject_sig.csv` |
| `train_encoding_models.py` | Trains ridge regression models on significant electrodes; outputs coefficients, correlations, and plots |
| `generate_plots.py` | Generates or regenerates correlation plots from existing run results; supports backfilling SVD plots for cross_subject, per_subject, and per_region modes (auto-detected or via `--svd-modes`) |
| `run_svd_analysis.py` | SVD decomposition of encoding weights; evaluates truncation at various k values and selects optimal k |
| `run_srrr.py` | SRRR analysis; reads SVD decompositions and finds minimal direction subset via stability selection and sufficiency testing; supports cross_subject, per_subject, and per_region modes |
| `run_partial_svd.py` | Partial SVD analysis; removes windows of SVD directions from embeddings, retrains ridge regression, and measures correlation drop; generates cross-subject, per-subject, and per-region sweep plots |
| `run_rrr.py` | RRR baseline; fits RRR directly from raw embeddings to neural targets (no SVD projection); sweeps over ranks with residual R², CCA, and Procrustes metrics; supports cross_subject, per_subject, and per_region modes |
| `run_partial_rrr.py` | Partial RRR analysis; loads RRR-learned A matrices, partials out their subspaces from embeddings, retrains ridge, and measures correlation drop; supports cross_subject, per_subject, and per_region modes |
| `run_sae.py` | SAE analysis; projects embeddings into RRR k-dim subspace, trains sparse autoencoder + regression (end-to-end or alternating), evaluates with correlation/R²/recovery metrics; supports cross_subject, per_subject, and per_region modes |

### `utils/`

Core library modules:

| File | Purpose |
|------|---------|
| `config.py` | Dataclass definitions for `EncodingConfig`, `SignificanceConfig`, `SVDConfig`, `PartialSVDConfig`, `PartialRRRConfig`, `SRRRConfig`, `RRRConfig`, `SAEConfig`; `get_results_dir()` for canonical output paths |
| `config_utils.py` | Argument parsing and YAML loading with CLI override support (dot-notation, e.g. `--encoding.layer=48`) |
| `data_loaders.py` | Loads iEEG BIDS data via MNE, GPT-2 embeddings from HDF5, builds word-aligned epochs, handles time offsets; `build_XY_from_subjects()` for stacking multi-subject neural data; `load_sig_electrodes()` and `load_fold_indices()` for shared data-loading across scripts |
| `embeddings.py` | Loads contextual GPT-2-XL token embeddings from HDF5 and averages tokens per word |
| `encoding.py` | K-fold cross-validation ridge regression (Himalaya backend), coefficient serialization, cross-subject aggregation |
| `significance.py` | GPU-accelerated permutation testing, vectorized Pearson correlation, Benjamini-Hochberg FDR correction |
| `analysis.py` | SVD decomposition of weights, truncated correlation evaluation, best-k selection, residualization utilities |
| `srrr.py` | Core SRRR/RRR functions: ProxGD optimizer, stability selection, `fit_dense_rrr`, sufficiency test, alignment metrics (CCA, Procrustes), I/O helpers for both SRRR and RRR outputs |
| `sae.py` | SAE model (`SAEEncoder`, `SAERegressorEndToEnd`), training functions (`train_sae_end_to_end`, `train_sae_alternating`), dead feature resampling (`_resample_dead_neurons`), and SAE-specific metrics (L0 sparsity, dead neurons, activation frequency). Encoder bias is initialized to zero to prevent dead-on-init features. |
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
| `test_srrr.py` | SRRR core functions: ProxGD, stability selection, sufficiency test, alignment metrics, I/O |
| `test_partial_svd.py` | Partial SVD: `partial_out_svd_directions` orthogonality/idempotency, config round-trip, sweep window generation |
| `test_rrr.py` | RRR utilities: `save_rrr_result`/`load_rrr_result` roundtrip, `_run_rrr_for_unit` integration, `build_XY_from_subjects` |
| `test_partial_rrr.py` | Partial RRR: `partial_out_subspace` orthogonality/idempotency/non-orthonormal handling, sweep integration, save results, config round-trip |
| `test_sae.py` | SAE: encoder shape/ReLU, zero-bias init, dead feature resampling, end-to-end and alternating training (loss decrease, OLS/ridge, resampling integration), metrics computation, config round-trip |
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
