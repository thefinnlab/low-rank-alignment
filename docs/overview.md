# Repository Overview

This repository is a neuroscience encoding analysis pipeline for studying how language-model embeddings predict intracranial EEG (iEEG) neural activity. The default dataset is the podcast stimulus, and the dataset registry also supports Brain Treebank movie stimuli by centralizing transcript loading, neural-data/cache paths, and stimulus-scoped result directories.

Neural preprocessing uses 70–200 Hz Hilbert amplitude. Brain Treebank attaches available MNI electrode coordinates to each MNE recording, excludes the contacts listed in `corrupted_elec.json`, applies the Podcast despiking procedure, and computes a global common-average reference over the remaining contacts before notch and high-gamma filtering. Epoch precomputation extracts word-locked windows at 512 Hz and only then resamples each aligned window to 32 Hz, avoiding quantization onto one continuous 32 Hz time grid. The resulting epoch caches are HDF5-backed so downstream analyses can load only the requested significant electrodes from disk.

The retained pipeline has three main analysis entrypoints:

1. **Lag-cluster significance** (`make sig`) — trains per-subject encoding models with out-of-fold predictions, builds reliable electrode-lag clusters, and scores 10,000 reproducible null draws using independent fold-local circular shifts sampled with replacement. It applies global BH-FDR either across one winning cluster per electrode (the default) or across all candidate clusters, then writes downstream-compatible significance CSVs.
2. **Encoding model training** (`make train`) — trains ridge regression encoding models on significant electrodes. New `encoding_results.joblib` files are intentionally slim: they keep correlations, fold indices, target scalers, electrode names, run metadata, optional shared word-selection provenance, and optional arbitrary-baseline correlations, while avoiding fitted model and coefficient payloads.
3. **RRR analysis** (`make rrr`) — fits raw-input reduced-rank regression from embeddings to neural targets, sweeps rank-k projections, and compares rank-k neural-relevance ridge predictions against fresh full-ridge fold baselines. The retained RRR script can also reuse shared word selections saved by significance/encoding runs, run a PCA subspace baseline when `RRRConfig.run_pca_baseline` is enabled, evaluate a `cross_dataset` mode from rank-specific source-run averaged RRR/PCA subspaces, and run `multi_view` analyses that project several saved RRR views at configured ranks, union their target-dataset significant electrodes, fit a fold-local Himalaya banded-ridge full model, and sweep the rank of the resulting joint neural-relevance space. An optional circular-shift null-augmented rank check remains available through `RRRConfig.run_null_augmented`. RRR and PCA baselines save fold-level `A.npy` subspace factors for downstream comparison.

Downstream pairwise subspace reproducibility analysis is available through `scripts/rrr/compare_subspaces.py`, which computes ordered CCA/RSA comparisons over configured saved RRR/PCA unit directories and writes reusable CSV/NPZ summaries plus quick-look plots under each base unit directory. Config-driven partial-support MAXVAR GCCA is available through `scripts/gcca/run_gcca.py`. It uses an explicitly configured anchor and independently ranked views, selects component-specific support and ridge values within the anchor folds, and saves calibrated per-view ambient maps, word-level scores, reliability sufficient statistics, and fold-stable axis/block diagnostics under a dataset-scoped directory. `scripts/gcca/run_encoding.py` evaluates calibrated fold-specific GCCA embeddings against the union of significant electrodes from anchor-dataset member runs. Its optional `run_rrr` mode fits fold-local RRR to each shared-dimension GCCA encoder, saves the full rank-k input factor `A`, transforms the GCCA embeddings through it, and uses a fresh ridge encoder for the final GCCA metrics. It does not refit raw-embedding baselines; instead it loads existing RRR per-target results, matches `(subject, electrode, lag)` identities, and writes direct correlation-versus-rank comparisons. The companion comparison notebook provides the interactive component explorer.

The Makefile also provides batch targets for releasing multiple config runs:
`make generate-embeddings-batch`, `make sig-batch`, and `make rrr-batch`. Random-init embedding runs can use `make generate-random-embeddings`, which wraps `scripts/utilities/generate_random_embeddings.py`.

## Generating Embeddings

`make generate-embeddings EMBED_CONFIG=configs/embeddings/generate_<model>.yml` runs `scripts/utilities/generate_embeddings.py`, which loads HuggingFace models through `transformers`, loads words through the active dataset adapter, and writes per-token contextual embeddings.

HuggingFace generation tokenizes each transcript word independently with the same space-prefixed convention as `data/code/llm.py` for all model types, then uses a sliding window controlled by `context_window` and `stride`. Causal models use native causal attention; bidirectional models use `attention_mode=bidirectional_strict` so each target sees only preceding tokens. Set `random_init_model: true` to use a model id as the tokenizer and architecture source while initializing fresh random weights.

Outputs use the existing layout:

- Podcast: `data/stimuli/{output_name}/`
- Brain Treebank: `data/stimuli/{output_name}/movie-{movie}/`
- Files: `features.hdf5`, `transcript.tsv`, `metadata.json`, and, for podcast-compatible causal LM runs, `performance.json`

Downstream configs can store list-valued `encoding.per_model_config.layer` entries. The entrypoints fan those out into one scalar-layer run at a time before deriving result directories or loading embeddings.

## File Reference

### Root

| File | Purpose |
|------|---------|
| `Makefile` | Targets for retained pipeline stages and utilities: significance, encoding training, RRR, plots, embedding generation, random embeddings, epoch precompute, atlas printing, and tests |
| `submit.sh` | SLURM job submission script |
| `pytest.ini` | Pytest config |

### `configs/`

| Path | Purpose |
|------|---------|
| `configs/encoding/` | Encoding training configs |
| `configs/significance/` | Lag-cluster significance configs |
| `configs/rrr/` | RRR configs, including optional embedded PCA settings and an example saved-view banded RRR analysis in `multi_view.yml` |
| `configs/rrr/compare_subspaces.yml` | Example config for ordered pairwise CCA/RSA comparisons over saved RRR/PCA subspace factors |
| `configs/gcca/maxvar_gcca.yml` | Example anchor/view configuration for partial-support MAXVAR GCCA and downstream score artifacts |
| `configs/gcca/encoding.yml` | Downstream calibrated GCCA encoding and saved-RRR comparison configuration |
| `configs/embeddings/` | HuggingFace and random-init embedding generation configs |
| `configs/precompute_epochs/` | Epoch-cache precompute configs |
| `configs/brain_treebank/` | Brain Treebank scoped embedding, epoch, significance, training, and RRR configs |

### `scripts/`

| File | Purpose |
|------|---------|
| `scripts/sig_lags/run_significance.py` | Cluster-based lag significance entrypoint with configurable null count, seed, and electrode- or cluster-level global FDR; writes significance CSVs, the null distribution, and per-subject test-fold index artifacts for RRR |
| `scripts/encoding/train_encoding_models.py` | Ridge encoding training entrypoint; writes slim per-subject `encoding_results.joblib`, `corrs.npy`, optional `arbitrary_corrs.npy`, plots, and alpha diagnostics |
| `scripts/rrr/run_rrr.py` | Raw-input RRR rank sweep with full-ridge baselines, cross-dataset evaluation, and multiview banded-ridge analysis over saved RRR subspaces |
| `scripts/rrr/compare_subspaces.py` | Pairwise saved-subspace reproducibility analysis; computes ordered CCA/RSA over configured RRR/PCA `A.npy` factors and saves tables, component artifacts, and plots |
| `scripts/gcca/run_gcca.py` | Config-driven partial-support MAXVAR GCCA; saves fold models, calibrated word representations, support/reliability tables, and stability blocks |
| `scripts/gcca/run_encoding.py` | Evaluate calibrated GCCA ridge encodings without rebuilding baselines and compare matched targets with saved RRR runs |
| `scripts/utilities/generate_embeddings.py` | Generate contextual HuggingFace embeddings |
| `scripts/utilities/export_transcript.py` | Export a dataset adapter's normalized transcript word table for review |
| `scripts/utilities/generate_random_embeddings.py` | Generate random-init contextual embeddings |
| `scripts/utilities/precompute_epochs.py` | Build dataset-scoped, word-aligned 32 Hz epoch-tensor caches |
| `scripts/utilities/generate_plots.py` | Regenerate correlation plots from existing encoding results |
| `scripts/utilities/electrode_region_distribution.py` | Dataset-aware electrode region summaries and plots |
| `scripts/utilities/print_atlas_regions.py` | Print atlas region labels for electrodes |
| `scripts/utilities/download_model.py`, `download_tokenizer.py` | Cache HuggingFace model/tokenizer assets |

### `notebooks/`

| File | Purpose |
|------|---------|
| `notebooks/rank-correlation-distributions.py` | Marimo app for comparing rank-k, reference, and null-augmented correlation distributions across electrode, lag, and cluster groupings |
| `notebooks/fisher-ecog.py` | Marimo app for generating word-level Fisher participation-ratio features from causal LMs or Whisper ASR encoder states and optionally comparing saved features with ECoG responses |
| `notebooks/compare-subspaces.py` | Marimo app for discovering saved RRR/PCA `A.npy` factors, comparing selected or fold-averaged RRR subspaces with principal angles and Grassmann/spectral/Fubini-Study distances, cross-rank Grassmann summaries, optional intrinsic Grassmann means, random-subspace nulls, automatically resolved cross-side PCA sibling baselines, RSA over projected transcript word embeddings, held-out Procrustes/CCA alignment within or across datasets, partial-support multiview MAXVAR GCCA with union-rank capacity, per-view ambient maps, and fold-stable axis/block diagnostics, and subplot summaries of batch CCA/RSA outputs |
| `notebooks/rrr-srrr-review.py` | Marimo review app for RRR/SRRR outputs, including A-direction activation examples, hypothesis-feature regressions against A-direction activations, B-matrix temporal/brain-map views, and cross-fold/source comparisons |

### `src/`

| File | Purpose |
|------|---------|
| `src/core/config.py` | Dataclasses for retained entrypoints and utilities: `EncodingConfig`, `SignificanceConfig`, `RRRConfig`, subspace comparison, embedding generation, epoch precompute, and electrode-region configs |
| `src/core/registry.py` | Embedding getter and dataset adapter registries |
| `src/utils/config_utils.py` | YAML loading, CLI overrides, and layer fanout |
| `src/utils/data_loaders.py` | Dataset-dispatched iEEG loading, aligned epoch extraction/cache I/O, significance loading, and X/Y assembly |
| `src/utils/embeddings.py` | Contextual and static embedding loaders |
| `src/utils/encoding.py` | Cross-validation ridge training, slim encoding result serialization, fold baseline training for RRR, offsets, and correlation loaders |
| `src/utils/significance.py` | Correlation, FDR, lag-cluster scoring, and circular-shift null helpers |
| `src/utils/alpha_grids.py`, `constants.py` | Shared ridge alpha-grid policy and diagnostics |
| `src/utils/srrr.py`, `rrr_utils.py`, `subspace_utils.py`, `subspace_compare.py`, `subspace_io.py` | RRR factor fitting, RRR result I/O, rank evaluation, saved-subspace discovery, comparison-run loading, and subspace CCA/RSA helpers |
| `src/utils/gcca.py`, `multiview_rrr.py` | Config-driven partial-support GCCA orchestration and shared saved-view preparation/banded-ridge utilities for multiview RRR |
| `src/utils/decomposition.py` | Shared multi-embedding alignment, raw neural-target, fold, and target-manifest preparation used by RRR and GCCA encoding |
| `src/utils/plotting.py` | Correlation, alpha, RRR/PCA rank, and utility plots |
| `src/utils/atlas.py` | Atlas lookup and electrode-region grouping |

### `tests/`

Focused retained-behavior tests cover:

- `test_encoding.py`
- `test_significance.py`
- `test_rrr.py`
- `test_config_utils.py`
- `test_data_loaders.py`
- `test_embeddings.py`
- `test_dataset_registry.py`
- `test_alpha_diagnostics.py`

Additional tests cover live shared utility modules such as correlations, atlas lookup, examples, layer fanout, and embedding-generation helpers.
