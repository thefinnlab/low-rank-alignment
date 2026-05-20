# Repository Overview

This is a neuroscience encoding analysis pipeline that studies how language-model embeddings correlate with intracranial EEG (iEEG) neural activity. The original/default dataset is a podcast stimulus, and the codebase now also supports Brain Treebank movie stimuli through a dataset registry that centralizes transcript loading, neural-data/cache paths, and stimulus-scoped result directories. The pipeline trains ridge regression encoding models, performs permutation-based significance testing, and conducts SVD analysis to identify which directions in embedding space best predict neural responses.

## Pipeline Stages

1. **Cluster-based lag significance** (`make sig`) — trains per-subject encoding models, builds reliable electrode-lag clusters from out-of-fold predictions, scores clusters by summed Fisher-z mean/SEM across fold-local blocks, scores a fold-local circular-shift null distribution, applies a global max-statistic null across subjects/electrodes with FDR correction, and writes downstream-compatible significance CSVs. `make heuristic-sig` uses the same observed cluster-generation path but skips null scoring/FDR and marks clusters significant only when they meet `min_cluster_size`.
1b. **Phase-randomization significance** — more robust electrode-level significance via phase randomization of continuous neural signal: `generate_null_distribution.py` (batchable, run in parallel) followed by `compute_significance.py` (reads null distribution shards, FDR correction). See details below.
2. **Encoding model training** (`make train`) — trains ridge regression models on significant electrodes; when explicit `ridge_alphas` are omitted, ridge fits use a shared dimension-aware alpha grid that adapts to the current training design matrix and can cache refined grids after diagnostics
3. **SVD analysis** (`make svd`) — decomposes model weights to find dominant directions in embedding space
4. **Partial SVD analysis** (`make partial-svd`) — removes contiguous windows of SVD directions from word embeddings and retrains ridge regression to measure how much each group contributes to encoding performance
5. **SRRR analysis** (`make srrr`) — finds the minimal subset of SVD directions that explains all significant encoding variance, via stability selection and a statistical sufficiency test
6. **RRR analysis** (`make rrr`) — raw-input RRR baseline: fits or loads fold-level RRR factors, sweeps rank-k projections from the learned basis, and compares rank-k neural-relevance ridge predictions with full-ridge and raw RRR diagnostics; rank sweeps resume from completed fold/rank outputs by default, skipping complete units before data/regression setup and building baselines only for missing folds; `refresh: true` forces a full recompute
6a. **RRR/PCA sufficiency bootstrap** (`run_rrr_sufficiency.py`) — paired block bootstrap for deciding whether rank-k RRR or PCA subspace ridge predictions are statistically close enough to a full-basis ridge baseline (`X @ A_full` for RRR, full fold PCA scores for PCA); supports cross-subject and leave-one-subject-out runs, reuses saved real RRR fold bases, recomputes PCA fold bases from the same training splits, streams LOSO held-out subject/unit runs to limit peak RAM, can restrict LOSO held-out subjects, can optionally cache out-of-fold predictions before bootstrapping, and can emit per-target sufficiency summaries
6b. **PCA analysis** (`make pca`) — PCA-based subspace baseline: computes PCA on word embeddings to find a low-dimensional input subspace, trains ridge regression from projected embeddings to neural targets, and sweeps over a rank grid; plots optionally overlay corresponding RRR results for comparison
7. **Partial RRR analysis** (`make partial-rrr`) — partials out RRR-learned subspaces from word embeddings and retrains ridge regression to measure how much each rank-k subspace contributes to encoding performance
8. **SAE analysis** (`make sae`) — projects embeddings into an RRR-learned k-dimensional subspace and trains a sparse autoencoder (overcomplete hidden layer with ReLU + L1 penalty) followed by regression to predict neural targets; tests whether superposition in the RRR subspace can be decoded with more linear dimensions
9. **Ablation analysis** (`run_ablation.py`) — loads pre-computed RRR A/B matrices, projects embeddings into the latent space, applies quantile-based masks (tail-only, center-only, random) to the projected activations, trains fresh ridge regressions from masked projections to neural targets, and evaluates reconstruction performance; determines whether encoding information is concentrated in distributional tails or centers; supports cross_subject, per_subject, and per_region modes
10. **Visualization** (`make plots`) — generates correlation plots with optional significance overlays

The Makefile also provides batch targets for releasing multiple config runs:
`make generate-embeddings-batch EMBED_CONFIGS="..."`, `make sig-batch
SIG_CONFIGS="..."`, and `make rrr-batch RRR_CONFIGS="..."`. Random-init
embedding runs can use `make generate-random-embeddings`, which wraps
`scripts/utilities/generate_random_embeddings.py`.

### Generating embeddings for new models

`make generate-embeddings EMBED_CONFIG=configs/embeddings/generate_<model>.yml` runs
`scripts/utilities/generate_embeddings.py`, which loads any HuggingFace model via
`transformers`, loads words through the active dataset adapter (or the legacy
podcast `transcript_path`), and produces per-token contextual embeddings.
Generic HuggingFace generation tokenizes with `is_split_into_words=True` and
uses a sliding window of configurable `context_window` and `stride`. Causal
models (GPT-2, Pythia, GPT-J, Llama) use native causal attention; bidirectional
models (RoBERTa) use `attention_mode=bidirectional_strict` (stride=1) so each
target sees only preceding tokens. Set `random_init_model: true` to use a
HuggingFace model id only as the tokenizer and architecture source, with fresh
random weights controlled by `random_init_seed`. GPT-2-XL configs default to
`attention_mode=podcast_per_token`, which matches the legacy podcast generator:
each word is tokenized as `tokenizer.tokenize(" " + word)`, each target token
gets its own left-padded context row of `context_window + 1` tokens, causal LM
hidden states are read from the target position, and next-token metrics
(`rank`, `true_prob`, `top_pred`, `entropy`) are written to `transcript.tsv` and
`performance.json`. Podcast outputs go to `data/stimuli/{output_name}/`; Brain
Treebank outputs go to `data/stimuli/{output_name}/movie-{movie}/`.
Both layouts contain files matching the existing GPT-2-XL layout:
`features.hdf5` (per-layer hidden states), `transcript.tsv` (per-token
word/token metadata), and `metadata.json` (generation parameters), with
`performance.json` present for podcast-compatible causal LM runs. The
`transformer` embedding getter in `src/utils/embeddings.py` loads any of these
directories by `dataset`, `model_dir`, and `layer`. Current transformer
generation configs save three explicit layers per model: embedding/input,
middle, and final hidden-state indices. Downstream encoding, significance, and
RRR configs store these as list-valued `encoding.per_model_config.layer` entries
and the entrypoints expand them into one scalar-layer run at a time before
deriving results directories or loading embeddings. Offline GPU hosts should
pre-populate the HF cache (e.g. `huggingface-cli download meta-llama/Meta-Llama-3-8B`)
and run with the default `local_files_only: true`.

---

## File Reference

### Root

| File | Purpose |
|------|---------|
| `Makefile` | Build targets for all pipeline stages; supports SLURM or local execution, including heuristic significance, config-list batch targets for embeddings, significance, and RRR, plus a random-init embedding wrapper target |
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
| `pca_default.yml` | PCA cross-subject baseline (rank_grid [1,2,5,...,1600]) |
| `pca_per_subject.yml` | Per-subject PCA variant |
| `pca_per_region.yml` | Per-region PCA variant with Destrieux atlas regions |
| `rrr_per_subject.yml` | Per-subject RRR variant |
| `rrr_per_region.yml` | Per-region RRR variant with Destrieux atlas regions |
| `sae_default.yml` | SAE cross-subject default (k=10, hidden_dim=50, end-to-end, OLS, L1=0.001) |
| `ablation_default.yml` | Ablation cross-subject default (rank_grid [1,2,5,...,1600], center_pcts [10,30,50,70,90], masks [tail,center,random], 5 random seeds) |

Model-specific transformer releases live under `configs/significance/<model>/`
and `configs/rrr/<model>/`, with separate significance, cross-subject RRR, and
leave-one-subject-out RRR configs for each generated embedding directory.

### `scripts/`

Main entry points for the pipeline:

| File | Purpose |
|------|---------|
| `run_significance.py` | Cluster-based lag significance entrypoint; trains per-subject encoding models with out-of-fold predictions, filters reliable lag clusters across fold-local blocks, scores a fold-local circular-shift null distribution, compares observed clusters to a global max-statistic null across subjects/electrodes, and writes `significance_detailed.csv`, `all_subject_sig.csv`, and `cluster_null_distribution.npz` |
| `run_heuristic_significance.py` | Heuristic lag-cluster significance entrypoint; reuses per-subject training and observed cluster generation from `run_significance.py`, skips null scoring and FDR correction, writes `significance_detailed.csv` and `all_subject_sig.csv`, and leaves p/q-value fields as `NaN` |
| `generate_null_distribution.py` | Batchable phase-randomization null distribution generator; each invocation runs N iterations per subject and writes a uniquely-named `.npy` shard to `{null_dir}/sub_{id}/`; multiple jobs can run in parallel |
| `compute_significance.py` | Reads null distribution shards, re-runs encoding on real data, computes per-electrode p-values via max-statistic procedure, applies FDR correction, outputs `all_subject_sig.csv` and `significance_detailed.csv` |
| `train_encoding_models.py` | Trains ridge regression models on significant electrodes; outputs coefficients, correlations, and plots. Uses the shared dimension-aware ridge alpha policy unless `ridge_alphas` is explicit. When `log_alphas=True`, saves alpha diagnostic plots, metadata, and reusable refined grids. When `refine_alphas=True`, uses a two-pass coarse→refined alpha sweep |
| `generate_plots.py` | Generates or regenerates correlation plots from existing run results; supports backfilling SVD plots for cross_subject, per_subject, and per_region modes (auto-detected or via `--svd-modes`) |
| `electrode_region_distribution.py` | Dataset-aware utility for per-subject electrode region distributions. It supports podcast FIF headers via Destrieux atlas lookup and Brain Treebank electrode-label/localization metadata, defaults Brain Treebank summaries to all subjects unless `--movie`/`--subjects` are supplied, and writes detailed/summary CSVs plus stacked region-distribution, 2D/3D location, and nilearn glass-brain projection plots. |
| `run_svd_analysis.py` | SVD decomposition of encoding weights; evaluates truncation at various k values and selects optimal k |
| `run_srrr.py` | SRRR analysis; reads SVD decompositions and finds minimal direction subset via stability selection and sufficiency testing; supports cross_subject, per_subject, and per_region modes |
| `run_partial_svd.py` | Partial SVD analysis; removes windows of SVD directions from embeddings, retrains ridge regression, and measures correlation drop; generates cross-subject, per-subject, and per-region sweep plots |
| `run_rrr.py` | RRR baseline; fits RRR directly from raw embeddings to neural targets (no SVD projection), saves fold-level A/B factors, and sweeps rank-k ridge predictions from the learned A basis alongside full-ridge and raw RRR diagnostics; standardizes subspace ridge inputs by default and resolves rank-specific alpha grids via the shared dimension-aware policy; resumes completed fold/rank outputs unless `refresh: true`, including pre-input unit skips and per-fold baseline training for partial resumes; supports cross_subject, per_subject, per_region, and leave_one_subject_out modes |
| `run_rrr_sufficiency.py` | RRR/PCA sufficiency bootstrap; loads saved RRR `{unit}/fold_{f}/A.npy` bases or recomputes PCA fold bases from the same training splits, builds full-basis (`X @ A_full`) and rank-k out-of-fold ridge predictions, then runs a paired contiguous-block bootstrap over Fisher-z correlation deltas. In `cross_subject` mode outputs under `{rrr_source_dir}/sufficiency_bootstrap/` or the mirrored PCA directory; in `leave_one_subject_out` mode streams `train_subjects`, `transfer`, and `baseline` units for each selected held-out subject. Supports `held_out_subjects` to subset LOSO runs, `cache_oof_predictions`/`refresh_oof_predictions` to save and reuse OOF predictions per sufficiency output directory, and `track_per_target_sufficiency` to write `per_target_sufficiency.csv` plus softer target-fraction summary metrics. |
| `run_pca.py` | PCA baseline; computes PCA on word embeddings to find low-dimensional input subspaces, trains ridge regression from projected embeddings to neural targets, and reports full-ridge plus PCA rank metrics; plots overlay corresponding RRR results when available; supports cross_subject, per_subject, and per_region modes |
| `precompute_epochs.py` | Builds dataset-scoped 32 Hz epoch caches by saving continuous high-gamma/resampled data plus per-word slice indices; Brain Treebank precompute first maps movie-time transcript rows to each subject's neural sample clock using trigger timings; duplicate onset samples are dropped and full MNE Epochs arrays are not materialized during precompute |
| `run_partial_rrr.py` | Partial RRR analysis; loads RRR-learned A matrices, partials out their subspaces from embeddings, retrains ridge, and measures correlation drop; supports cross_subject, per_subject, and per_region modes |
| `run_sae.py` | SAE analysis; projects embeddings into RRR k-dim subspace, trains sparse autoencoder + regression (end-to-end or alternating), evaluates with correlation/R²/recovery metrics; supports cross_subject, per_subject, and per_region modes. **Hyperparameter sweeps**: set any scalar config param as a list (e.g. `hidden_dim: [5, 10, 30]`, `l1_penalty: [0.005, 0.01]`) to run all combinations. Results are organized hierarchically under k directories (e.g. `sae/k5/hidden_dim=5/l1_penalty=0.005/`). Summary plots at each k directory show how metrics vary per swept parameter. |
| `run_ablation.py` | RRR latent space ablation; loads pre-computed RRR A/B matrices, applies quantile-based masks (tail/center/random) to projected activations, trains fresh ridge regressions from masked projections to neural targets, sweeps over rank and center-zone percentile grids; outputs per-k pct_zeroed plots and a vs-rank summary figure; supports cross_subject, per_subject, and per_region modes |

### `utils/`

Core library modules:

| File | Purpose |
|------|---------|

| `constants.py` | Project-wide constants; `RIDGE_ALPHAS` — legacy alpha grid for RidgeCV; `refine_alpha_grid()` — builds a refined grid from observed best alphas; `check_alpha_grid_boundaries()` — detects when selected alphas cluster at grid edges |
| `alpha_grids.py` | Shared ridge alpha policy helpers: dimension-aware grid construction from `X_train`, explicit-alpha override handling, compatible refined-grid cache loading, and alpha diagnostics/metadata saving |
| `config.py` | Dataclass definitions for `EncodingConfig`, `SignificanceConfig`, `NullDistConfig`, `NewSignificanceConfig`, `SVDConfig`, `PartialSVDConfig`, `PartialRRRConfig`, `SRRRConfig`, `RRRConfig`, `RRRSufficiencyConfig`, `PCAConfig`, `SAEConfig`, `AblationConfig`; `get_results_dir()` for canonical output paths. `EncodingConfig` now carries additive `dataset` / `per_dataset_config` fields so runs can stay model-specific while also being stimulus-scoped. Ridge fits can optionally configure inner alpha-selection folds with `ridge_cv_folds` and Himalaya memory controls with `ridge_solver_params`, `ridge_y_in_cpu`, and `ridge_force_cpu`. Most downstream configs support `sig_dir` to override where significance CSVs are read from (defaults to `output_dir`). |
| `config_utils.py` | Argument parsing and YAML loading with CLI override support (dot-notation, e.g. `--encoding.per_model_config.layer=48`), plus downstream layer-list fanout helpers |
| `registry.py` | Dual registries for embedding getters and dataset adapters. Dataset adapters resolve dataset-specific transcript tables, raw/cache locations, significance loading, and stimulus context. Built-in adapters cover `podcast` and movie-scoped `brain_treebank`; Brain Treebank HDF5 sEEG recordings are converted from microvolts to MNE `RawArray` objects in volts, subject word tables are aligned from movie time to neural sample indices via `subject_timings`, and optional high-gamma preprocessing now mirrors the podcast path with 512 Hz resampling, average reference, 60 Hz harmonic notch filtering, then 70-200 Hz Butterworth/Hilbert envelope extraction before shared epoching/resampling. |
| `data_loaders.py` | Loads iEEG data and precomputed epoch caches, now via dataset-dispatched helpers so transcript/cache/significance behavior can vary by dataset while keeping existing utility entrypoints stable. `compute_epoch_cache_indices()` derives cache slice indices without materializing full MNE Epochs arrays, uses Brain Treebank estimated sample indices when available, and drops duplicate onset samples. `build_XY_from_subjects()` still stacks multi-subject neural data, but rejects mixed-stimulus or mismatched word-selection cache combinations. |
| `embeddings.py` | Loads contextual token embeddings from HDF5 and pools tokens per word. Transcript and HDF5 directory selection are dataset-aware: podcast uses `data/stimuli/{model_dir}/`, while Brain Treebank uses `data/stimuli/{model_dir}/movie-{movie}/` and prefers the generated per-token `transcript.tsv` for contextual pooling. Registered getters: `gpt2xl`, `transformer`, `glove`, `arbitrary` |
| `encoding.py` | K-fold cross-validation ridge regression (Himalaya backend), coefficient serialization, cross-subject aggregation; `extract_best_alphas()` — extracts per-target selected alphas from fitted RidgeCV; `train_encoding_two_pass()` — coarse-then-refined alpha sweep |
| `significance.py` | GPU-accelerated permutation testing, vectorized Pearson correlation, Benjamini-Hochberg FDR correction, fold/block lag-cluster scoring, heuristic minimum-size cluster marking, fold-local circular-shift null scoring, and global max-null aggregation; `phase_randomize_raw()` for signal phase randomization preserving autocorrelation; `compute_observed_max_per_electrode()` for max-across-lags statistics |
| `analysis.py` | SVD decomposition of weights, truncated correlation evaluation, best-k selection, residualization utilities |
| `srrr.py` | Core SRRR/RRR functions: ProxGD optimizer, stability selection, `fit_dense_rrr`, sufficiency test, alignment metrics (CCA, Procrustes), I/O helpers for both SRRR and RRR outputs |
| `sae.py` | SAE model (`SAEEncoder`, `SAERegressorEndToEnd`), training functions (`train_sae_end_to_end`, `train_sae_alternating`), dead feature resampling (`_resample_dead_neurons`), and SAE-specific metrics (L0 sparsity, dead neurons, activation frequency). Encoder bias is initialized to zero to prevent dead-on-init features. |
| `plotting.py` | Matplotlib helpers for correlation vs lag plots, significance overlays, SVD truncation analysis figures, RRR/PCA rank plots, ablation pct-zeroed and vs-rank plots; `plot_alpha_diagnostics()` — 3-panel alpha selection summary; `plot_alpha_vs_rank()` — how optimal alpha shifts with RRR rank |
| `rrr_utils.py` | Shared helpers for RRR-based scripts: `add_sig_nonsig_corrs`, `apply_sig_nonsig_splits_and_strip`, `make_sig_split_df`, `resolve_rrr_source_dir` |
| `rrr_sufficiency.py` | Bootstrap statistics for RRR/PCA sufficiency: Himalaya-backed per-target correlations, Fisher-z transforms, contiguous block bootstrap indices, paired full-vs-rank correlation delta bootstraps, full-model uncertainty estimates, and sufficiency decisions. |
| `subspace_utils.py` | Shared utilities for subspace evaluation: `evaluate_input_space` (train ridge on X, return correlation/R2/Y_hat), `evaluate_complement` (predict from complement of k-dim subspace), `evaluate_arbitrary` (predict from arbitrary embeddings), `compute_stacked_joint_metrics` (2-model OLS stacking via inner CV), and `compute_stacked_joint_ridge_means` (batched fold-based ridge-stacked scalar check) |
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
| `test_rrr_sufficiency.py` | RRR/PCA sufficiency utilities: Fisher-z correlation deltas, NaN/constant handling, block bootstrap reproducibility, strict sufficiency decision, synthetic RRR/PCA out-of-fold script paths, and a synthetic LOSO output-layout check. |
| `test_pca.py` | PCA utilities: `evaluate_input_space`, `evaluate_complement`, `evaluate_arbitrary` unit tests; `_run_pca_for_unit` integration (CSV output, pca_corr/pca_r2 columns, monotonicity, rank clamping, plot generation) |
| `test_partial_rrr.py` | Partial RRR: `partial_out_subspace` orthogonality/idempotency/non-orthonormal handling, sweep integration, save results, config round-trip |
| `test_sae.py` | SAE: encoder shape/ReLU, zero-bias init, dead feature resampling, end-to-end and alternating training (loss decrease, OLS/ridge, resampling integration), metrics computation, config round-trip, sweep detection helpers, sweep summary plot |
| `test_ablation.py` | Ablation: quantile mask construction (center/tail/random fractions, complementarity, median centering), `_run_ablation_for_unit` integration (grid_results.csv, summary.csv, baseline inclusion, per-k plots, performance monotonicity) |
| `test_alpha_diagnostics.py` | Alpha grid utilities: `refine_alpha_grid`, `check_alpha_grid_boundaries`, `extract_best_alphas`, `solve_regression` `return_alphas` parameter |
| `test_significance.py` | Permutation testing, FDR correction |
| `test_examples.py` | Context building and table generation |

### `notebooks/`

Exploratory Jupyter notebooks (not part of the main pipeline):

| File | Purpose |
|------|---------|
| `cluster-extraction.py` | Cluster analysis on neural/embedding data |
| `dictionary.py` | Word dictionary and vocabulary analysis |
| `word-analysis.py` | Word-level qualitative analysis |
| `rank-correlation-distributions.py` | Interactive marimo notebook for comparing low-rank RRR/PCA per-target correlation distributions against a max-rank reference, including rank/reference ratios and deltas |
| `rrr-srrr-review.py` | Interactive marimo notebook for RRR interpretability: direction word activations, B-matrix temporal/spatial visualization (nilearn brain maps), cross-fold consistency, cross-source comparison, cross-region electrode similarity, and activation statistics |
| `svd-interpretability.py` | Interactive marimo notebook for SVD direction interpretability, fold consistency, cross-source comparison, and subspace analyses |

### `processed_data/`

Pre-computed embeddings and intermediate results cached to disk.
