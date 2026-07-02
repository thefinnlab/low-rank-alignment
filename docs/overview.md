# Repository Overview

This repository is a neuroscience encoding analysis pipeline for studying how language-model embeddings predict intracranial EEG (iEEG) neural activity. The default dataset is the podcast stimulus, and the dataset registry also supports Brain Treebank movie stimuli by centralizing transcript loading, neural-data/cache paths, and stimulus-scoped result directories.

Neural preprocessing uses 70–200 Hz Hilbert amplitude. Brain Treebank attaches available MNI electrode coordinates to each MNE recording, excludes the contacts listed in `corrupted_elec.json`, applies the Podcast despiking procedure, and computes a global common-average reference over the remaining contacts before notch and high-gamma filtering. Epoch precomputation extracts word-locked windows at 512 Hz and only then resamples each aligned window to 32 Hz, avoiding quantization onto one continuous 32 Hz time grid. The resulting epoch caches are HDF5-backed so downstream analyses can load only the requested significant electrodes from disk.

The retained pipeline has three main analysis entrypoints:

1. **Lag-cluster significance** (`make sig`) — trains per-subject encoding models with out-of-fold predictions, builds reliable electrode-lag clusters, and scores 10,000 reproducible null draws using independent fold-local circular shifts sampled with replacement. It applies global BH-FDR either across one winning cluster per electrode (the default) or across all candidate clusters, then writes downstream-compatible significance CSVs.
2. **Encoding model training** (`make train`) — trains ridge regression encoding models on significant electrodes. New `encoding_results.joblib` files are intentionally slim: they keep correlations, fold indices, target scalers, electrode names, run metadata, optional shared word-selection provenance, and optional arbitrary-baseline correlations, while avoiding fitted model and coefficient payloads.
3. **RRR analysis** (`make rrr`) — fits raw-input reduced-rank regression from embeddings to neural targets, sweeps rank-k projections, and compares rank-k neural-relevance ridge predictions against fresh full-ridge fold baselines. The retained RRR script can also reuse shared word selections saved by significance/encoding runs, run a PCA subspace baseline when `RRRConfig.run_pca_baseline` is enabled, evaluate a `cross_dataset` mode from rank-specific source-run averaged RRR/PCA subspaces, plus an optional circular-shift null-augmented rank check when `RRRConfig.run_null_augmented` is enabled. RRR and PCA baselines save fold-level `A.npy` subspace factors for downstream comparison.

The Makefile also provides batch targets for releasing multiple config runs:
`make generate-embeddings-batch`, `make sig-batch`, and `make rrr-batch`. Random-init embedding runs can use `make generate-random-embeddings`, which wraps `scripts/utilities/generate_random_embeddings.py`.

## Generating Embeddings

`make generate-embeddings EMBED_CONFIG=configs/embeddings/generate_<model>.yml` runs `scripts/utilities/generate_embeddings.py`, which loads HuggingFace models through `transformers`, loads words through the active dataset adapter, and writes per-token contextual embeddings.

Generic HuggingFace generation tokenizes with `is_split_into_words=True` and uses a sliding window controlled by `context_window` and `stride`. Causal models use native causal attention; bidirectional models use `attention_mode=bidirectional_strict` so each target sees only preceding tokens. Set `random_init_model: true` to use a model id as the tokenizer and architecture source while initializing fresh random weights.

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
| `configs/rrr/` | RRR configs, including optional embedded PCA baseline settings |
| `configs/embeddings/` | HuggingFace and random-init embedding generation configs |
| `configs/precompute_epochs/` | Epoch-cache precompute configs |
| `configs/brain_treebank/` | Brain Treebank scoped embedding, epoch, significance, training, and RRR configs |

### `scripts/`

| File | Purpose |
|------|---------|
| `scripts/sig_lags/run_significance.py` | Cluster-based lag significance entrypoint with configurable null count, seed, and electrode- or cluster-level global FDR; writes significance CSVs, the null distribution, and per-subject test-fold index artifacts for RRR |
| `scripts/encoding/train_encoding_models.py` | Ridge encoding training entrypoint; writes slim per-subject `encoding_results.joblib`, `corrs.npy`, optional `arbitrary_corrs.npy`, plots, and alpha diagnostics |
| `scripts/rrr/run_rrr.py` | Raw-input RRR rank sweep with fresh full-ridge fold baselines, optional PCA baseline via `run_pca_baseline`, and cross-dataset evaluation from averaged source subspaces |
| `scripts/utilities/generate_embeddings.py` | Generate contextual HuggingFace embeddings |
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
| `notebooks/compare-subspaces.py` | Marimo app for discovering saved RRR/PCA `A.npy` factors, comparing selected subspaces with principal angles and Grassmann/spectral/Fubini-Study distances, and optionally plotting random-subspace nulls |
| `notebooks/rrr-srrr-review.py` | Marimo review app for RRR/SRRR outputs, including A-direction activation examples, hypothesis-feature regressions against A-direction activations, B-matrix temporal/brain-map views, and cross-fold/source comparisons |

### `src/`

| File | Purpose |
|------|---------|
| `src/core/config.py` | Dataclasses for retained entrypoints and utilities: `EncodingConfig`, `SignificanceConfig`, `RRRConfig`, embedding generation, epoch precompute, and electrode-region configs |
| `src/core/registry.py` | Embedding getter and dataset adapter registries |
| `src/utils/config_utils.py` | YAML loading, CLI overrides, and layer fanout |
| `src/utils/data_loaders.py` | Dataset-dispatched iEEG loading, aligned epoch extraction/cache I/O, significance loading, and X/Y assembly |
| `src/utils/embeddings.py` | Contextual and static embedding loaders |
| `src/utils/encoding.py` | Cross-validation ridge training, slim encoding result serialization, fold baseline training for RRR, offsets, and correlation loaders |
| `src/utils/significance.py` | Correlation, FDR, lag-cluster scoring, and circular-shift null helpers |
| `src/utils/alpha_grids.py`, `constants.py` | Shared ridge alpha-grid policy and diagnostics |
| `src/utils/srrr.py`, `rrr_utils.py`, `subspace_utils.py`, `subspace_compare.py` | RRR factor fitting, RRR result I/O, rank evaluation, saved-subspace discovery, and subspace comparison helpers |
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
