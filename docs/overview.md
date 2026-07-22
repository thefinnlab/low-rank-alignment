# Repository Overview

This repository is a neuroscience encoding analysis pipeline for studying how language-model embeddings predict intracranial EEG (iEEG) neural activity. The default dataset is the podcast stimulus, and the dataset registry also supports Brain Treebank movie stimuli by centralizing transcript loading, neural-data/cache paths, and stimulus-scoped result directories.

Neural preprocessing uses 70–200 Hz Hilbert amplitude. Brain Treebank attaches available MNI electrode coordinates to each MNE recording, excludes the contacts listed in `corrupted_elec.json`, applies the Podcast despiking procedure, and computes a global common-average reference over the remaining contacts before notch and high-gamma filtering. Epoch precomputation extracts word-locked windows at 512 Hz and only then resamples each aligned window to 32 Hz, avoiding quantization onto one continuous 32 Hz time grid. The resulting epoch caches are HDF5-backed so downstream analyses can load only the requested significant electrodes from disk.

The retained pipeline has three main analysis entrypoints:

1. **Lag-cluster significance** (`make sig`) — trains per-subject encoding models with out-of-fold predictions, builds reliable electrode-lag clusters, and scores 10,000 reproducible null draws using independent fold-local circular shifts sampled with replacement. It applies global BH-FDR either across one winning cluster per electrode (the default) or across all candidate clusters, then writes downstream-compatible significance CSVs.
2. **Encoding model training** (`make train`) — trains ridge regression encoding models on significant electrodes. New `encoding_results.joblib` files are intentionally slim: they keep correlations, fold indices, target scalers, electrode names, run metadata, optional shared word-selection provenance, and optional arbitrary-baseline correlations, while avoiding fitted model and coefficient payloads.
3. **RRR analysis** (`make rrr`) — fits raw-input reduced-rank regression from embeddings to neural targets, sweeps rank-k projections, and compares rank-k neural-relevance ridge predictions against fresh full-ridge fold baselines. Per-target NR correlations are saved by default. `RRRConfig.save_b_matrices` optionally saves target factors and enables raw-RRR prediction metrics; its per-target correlations are separately controlled by `save_rrr_corr_per_target`. The retained RRR script can also reuse shared word selections saved by significance/encoding runs, run a PCA subspace baseline when `RRRConfig.run_pca_baseline` is enabled, evaluate a `cross_dataset` mode from rank-specific source-run averaged RRR/PCA subspaces, and run `multi_view` analyses that project several saved RRR views at configured ranks, union their target-dataset significant electrodes, fit a fold-local Himalaya banded-ridge full model, and sweep the rank of the resulting joint neural-relevance space. An optional circular-shift null-augmented rank check remains available through `RRRConfig.run_null_augmented`. RRR and PCA baselines save fold-level `A.npy` subspace factors for downstream comparison.

A separate deep-fMRI-dataset workflow lives under `scripts/fmri/` and deliberately does not generalize the iEEG dataset adapters. The external dataset repository is stored at `data/deep-fMRI-dataset/` by default. The shared story resolver temporarily and explicitly holds out `life` and `fromboyhoodtofatherhood` for every subject because their response files are missing for UTS04 and UTS09, respectively; the centralized `TEMPORARILY_HELD_OUT_FMRI_STORIES` constant is the intended place to restore them after fixing the data. `make fmri-generate-embeddings` writes one contextual-embedding artifact set per selected story. `make fmri-sig` filters each subject to voxels with finite responses at every TR of every selected training story, fits full-embedding ridge models with one scan session held out at a time, and applies per-subject BH-FDR to a deterministic within-story circular-shift null. Significance CSVs retain original voxel indices and mark filtered voxels explicitly. `make fmri-rrr` projects word embeddings through rank-specific extrinsic means of saved iEEG RRR factors, automatically evaluates sibling PCA factors when a complete set is present, and fits fMRI ridge encoders against all selected subjects' significant voxels. It reports Pearson correlation and R² versus rank alongside a fresh full-embedding ridge ceiling.

The fMRI preprocessing follows `data/deep-fMRI-dataset/encoding/encoding.py`: word vectors are Lanczos-resampled to TR times, stimulus columns are z-scored within each story, and delays 1–4 are concatenated by default. Unlike upstream, FIR delays are applied separately per story so story boundaries are zero-padded rather than carrying delayed features across the inter-story silence. The slice is exactly `[5 + trim : -trim]`; with the default `trim=5`, the fixed five-TR leading alignment adjustment makes the effective removal 10 TRs at the start and 5 at the end, while the configurable analysis trim itself is five TRs on both ends.

Downstream pairwise subspace reproducibility analysis is available through `scripts/rrr/compare_subspaces.py`, which computes ordered CCA/RSA comparisons over configured saved RRR/PCA unit directories and writes reusable CSV/NPZ summaries plus quick-look plots under each base unit directory. Config-driven partial-support MAXVAR GCCA is available through `scripts/gcca/run_gcca.py`. It uses an explicitly configured anchor and independently ranked views, selects component-specific support and ridge values within the anchor folds, and saves calibrated per-view ambient maps, word-level scores, reliability sufficient statistics, and fold-stable axis/block diagnostics under a dataset-scoped directory. `scripts/gcca/run_encoding.py` evaluates calibrated fold-specific GCCA embeddings against the union of significant electrodes from anchor-dataset member runs. Its optional `run_rrr` mode fits fold-local RRR to each shared-dimension GCCA encoder, saves the full rank-k input factor `A`, transforms the GCCA embeddings through it, and uses a fresh ridge encoder for the final GCCA metrics. It does not refit raw-embedding baselines; instead it loads existing RRR per-target results, matches `(subject, electrode, lag)` identities, and writes direct correlation-versus-rank comparisons. The companion comparison notebook provides the interactive component explorer.

Saved RRR input subspaces can also be probed with `scripts/rrr/run_language_decoding.py`. The config-driven podcast analysis projects word embeddings through within-dataset fold-matched or cross-dataset fold-averaged `A` factors, then fits fold-safe linear decoders for lexical, syntactic, predictability, sentence, and intonation-unit targets. It saves reduced and ambient embedding-space decoder weights at every task/rank/fold, full-embedding references, alignment provenance, reusable tables, and a multi-panel performance-versus-rank plot. New targets are registered as word-aligned getter functions in `src/utils/language_tasks.py`.

The Makefile exposes the language probe as `make language-decoding` and also provides batch targets for releasing multiple config runs:
`make generate-embeddings-batch`, `make sig-batch`, and `make rrr-batch`. `make sig-rrr-batch` submits paired significance and cross-subject RRR jobs for each model in `PIPELINE_MODELS`; `DATASET` selects `podcast` or `cars-2`, and each RRR job receives an `afterok` dependency on its matching significance job. Random-init embedding runs can use `make generate-random-embeddings`, which wraps `scripts/utilities/generate_random_embeddings.py`.

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
| `Makefile` | Targets for retained pipeline stages and utilities: significance, encoding training, RRR, language decoding, plots, embedding generation, random embeddings, epoch precompute, atlas printing, and tests |
| `submit.sh` | SLURM job submission script |
| `pytest.ini` | Pytest config |

### `configs/`

| Path | Purpose |
|------|---------|
| `configs/encoding/` | Encoding training configs |
| `configs/significance/` | Lag-cluster significance configs |
| `configs/rrr/` | RRR configs, including optional embedded PCA settings and an example saved-view banded RRR analysis in `multi_view.yml` |
| `configs/rrr/language_decoding.yml` | Example podcast language-feature decoding sweep over saved RRR input factors |
| `configs/rrr/compare_subspaces.yml` | Example config for ordered pairwise CCA/RSA comparisons over saved RRR/PCA subspace factors |
| `configs/gcca/maxvar_gcca.yml` | Example anchor/view configuration for partial-support MAXVAR GCCA and downstream score artifacts |
| `configs/gcca/encoding.yml` | Downstream calibrated GCCA encoding and saved-RRR comparison configuration |
| `configs/embeddings/` | HuggingFace and random-init embedding generation configs |
| `configs/precompute_epochs/` | Epoch-cache precompute configs |
| `configs/fmri/` | Story embedding, voxel significance, and cross-dataset RRR/PCA configs for deep-fMRI-dataset |
| `configs/brain_treebank/` | Brain Treebank scoped embedding, epoch, significance, training, and RRR configs; Cars 2 mirrors the podcast model-folder layout for significance and cross-subject/leave-one-subject-out RRR configs |

### `scripts/`

| File | Purpose |
|------|---------|
| `scripts/sig_lags/run_significance.py` | Cluster-based lag significance entrypoint with configurable null count, seed, and electrode- or cluster-level global FDR; writes significance CSVs, the null distribution, and per-subject test-fold index artifacts for RRR |
| `scripts/encoding/train_encoding_models.py` | Ridge encoding training entrypoint; writes slim per-subject `encoding_results.joblib`, `corrs.npy`, optional `arbitrary_corrs.npy`, plots, and alpha diagnostics |
| `scripts/rrr/run_rrr.py` | Raw-input RRR rank sweep with full-ridge baselines, cross-dataset evaluation, and multiview banded-ridge analysis over saved RRR subspaces |
| `scripts/rrr/compare_subspaces.py` | Pairwise saved-subspace reproducibility analysis; computes ordered CCA/RSA over configured RRR/PCA `A.npy` factors and saves tables, component artifacts, and plots |
| `scripts/rrr/run_language_decoding.py` | Decode registered word-level language targets across a saved RRR rank sweep with leakage-safe within- or cross-dataset folds |
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
| `scripts/fmri/generate_embeddings.py` | Generate contextual embeddings independently for each configured fMRI story |
| `scripts/fmri/run_significance.py` | Session-held-out voxel significance with within-story circular-shift nulls and per-subject FDR |
| `scripts/fmri/run_cross_dataset_rrr.py` | Evaluate saved fold-averaged iEEG RRR and optional PCA axes on significant fMRI voxels |

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
| `src/utils/language_tasks.py`, `language_decoding.py` | Extensible podcast language-target registry plus saved-RRR projection, linear decoding, artifact, and plot orchestration |
| `src/utils/plotting.py` | Correlation, alpha, RRR/PCA rank, and utility plots |
| `src/utils/atlas.py` | Atlas lookup and electrode-region grouping |
| `src/utils/fmri.py` | deep-fMRI story resolution, TextGrid loading, Lanczos/TR alignment, exact trimming/FIR preprocessing, response assembly, session folds, and null helpers |

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
