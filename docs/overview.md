# Repository Overview

This repository contains script-driven neuroscience analyses of how language-model
embeddings predict iEEG and fMRI responses. The supported executable surface is
`scripts/`; utilities exist to serve those workflows rather than as a standalone API.

## Retained workflows

- `scripts/run_significance.py` dispatches significance by dataset to the iEEG
  single-lag correlation or fMRI held-out-session implementation.
- `scripts/run_rrr.py` dispatches reduced-rank regression by dataset while exposing
  one Config and Artifact contract.
- `scripts/run_cross_stimulus_rrr.py` runs standalone within-subject Brain Treebank
  transfer across every directed movie pair using common Laplacian contacts, no
  significance filtering, and orthogonalized RRR input spaces; see
  [cross_stimulus_rrr.md](cross_stimulus_rrr.md).
- `scripts/run_target_dimensionality_rrr.py` reruns internal RRR over fold-safe
  target-PCA dimensions, with one Artifact variant per dimension and compiled
  correlation/R² sweep outputs.
- `scripts/rrr/compare_subspaces.py` performs configured leakage-safe comparisons
  between saved RRR/PCA spaces and publishes one source-organized Artifact per
  generating dataset/model/layer/subspace.
- `scripts/fmri/` contains fMRI embedding generation workflows.
- `scripts/language-interpret/` contains language decoding and reconstruction.
  Language decoding includes word-aligned and TR-aligned pre-FIR modes for fMRI,
  all published through the same Artifact client.
- `scripts/utilities/` contains data preparation, embedding generation, language-feature
  generation, plotting, atlas-reporting entrypoints, and a read-only iEEG utility for
  comparing max-lag correlation with electrode-specific or subject-wide shared-max
  nulls, max-lag correlation/SEM, and lag-cluster significance on matched encoding
  fits and circular-shift null samples. A companion
  refit utility parses that comparison's console report and recomputes all-lag
  correlations for only the union of selected electrodes, without permutation tests.

The Makefile exposes significance and RRR through the unified entry points. Each
retained analysis workflow owns one client-facing `Artifact` class. Their canonical
identity, variants, selector fields, resume behavior, and legacy migration are
described in [artifacts.md](artifacts.md). Language decoding and reconstruction load
RRR/PCA factors through RRR Artifact identities and publish task-level results through
their own Artifacts.
The `cross-temporal-pipeline` Make target submits per-model temporal-window RRR to
cross-temporal-window transfer chains for podcast and Cars 2, using Slurm `afterok`
dependencies between those artifact-producing stages. It consumes existing
significance artifacts rather than submitting significance jobs.
The `rrr-loso-batch` Make target submits one model-named podcast
leave-one-subject-out RRR job per non-GloVe pipeline model and consumes the
cross-model-layer significance artifacts already generated for those models.

## Source layout

- `src/core/config.py` defines the modality implementation dataclasses; the two
  public facade scripts define the strict client Configs.
- `src/core/registry.py` defines embedding and dataset adapter registries.
- `src/utils/shared/` contains configuration, embedding, iEEG/fMRI loading, ridge,
  correlation, alpha-grid, and tensor helpers used by multiple workflows.
- `src/utils/significance/` contains the modality-specific significance runners and
  their reusable statistical helpers.
- `src/utils/rrr/` contains the modality-specific RRR runners, dense fitting, input
  preparation, evaluation, multiview handling, cross-temporal-window transfer, and
  result helpers, including the standalone cross-stimulus mechanics. iEEG RRR mode implementations are organized under
  `src/utils/rrr/ieeg_rrr_mode/`, with `rrr_ieeg.py` retained as the compatibility
  facade.
- `src/utils/subspace_comparison/` contains saved-space metrics and directed comparison
  orchestration.
- `src/utils/language_interpret/` contains decoding, reconstruction, and task registry
  utilities shared by iEEG and fMRI scripts.
- `src/utils/language_features/` contains the reusable annotation pipeline and built-in
  providers.
- `src/utils/atlas/` and `src/utils/plotting/` contain atlas and retained plotting code.

Brain Treebank high-gamma preprocessing supports common-average and shaft-Laplacian
rereferencing. Common average remains the default. Laplacian caches and downstream
analyses are selected with `per_dataset_config.rereference: laplacian`; see
[brain_treebank.md](brain_treebank.md) for the cache naming, contact-retention, and
configuration contract. The cross-stimulus setup helpers download the eligible
subject trials, generate missing movie embeddings, and precompute missing
Laplacian epoch caches through the existing preprocessing implementation.
The checked-in Cars 2 and Megamind pipeline configs use Laplacian rereferencing;
Megamind covers subjects 4, 6, and 7 and mirrors the Cars 2 significance and RRR
model/mode matrix.

Imports use these explicit package paths. The former flat `src.utils.<module>` paths
are intentionally unsupported.

Target-dimensionality RRR is intentionally limited to ordinary internal iEEG and
fMRI analyses. Target PCA is fitted only on each evaluation fold's training targets
(or the fMRI training stories), and metrics are evaluated in the retained component
space. Power-of-two outer and inner grids include exact endpoints; the outer endpoint
is capped by the smallest fold-estimable PCA rank, and the inner RRR endpoint is also
capped by the embedding dimension.

Significance and RRR YAMLs use a top-level dataset discriminator, shared settings,
and exactly one `ieeg` or `fmri` block. RRR uses modality-neutral `internal` and
`cross_dataset` modes; specialized modes remain iEEG-only. RRR significance selection
is configured uniformly across modalities as `within-layer` or as an explicit
`cross-model-layer` union; union artifact entries may compactly list several layers
for one model. Cross-model-layer batch RRR uses per-model configs so model-specific
rank and ridge-memory settings can be retained for both iEEG and fMRI.
Selected ridge alphas are optional output: iEEG RRR uses `ieeg.log_alphas`, fMRI
RRR uses `fmri.ridge.log_alphas`, and the standalone cross-stimulus runner uses its
top-level `log_alphas`; all default to false while alpha selection itself still runs.
Cross-dataset RRR source identities include their own significance selection, which
deterministically resolves and validates the saved source variant.
Source and target selections are required to be corresponding within-layer selections
for the same model and layer over their respective datasets.
Cross-dataset transfer from an iEEG source refits one ridge/RRR model on all source
stimulus words and significant electrodes, then transfers the reduced-QR input basis
at each rank. Its optional PCA basis is likewise fit on all source words. Saved iEEG
fold factors are not loaded or averaged for transfer. fMRI sources expose saved
whole-training-stimulus rank bases; aggregation remains only where the fMRI FIR RRR
factor is combined across delay blocks, using the configured equal-subspace
`extrinsic` method, raw-coefficient `factor_weighted` method, or normalized
prediction-energy `prediction_weighted` method. The prediction-weighted option uses
squared singular-value energy from the SVD of the full ridge model's training
predictions and is available only for within-fMRI FIR aggregation. It is the default
for internal fMRI FIR fitting and for loading an fMRI source into cross-dataset RRR;
unspecified EEG aggregation continues to compute with the extrinsic method.
RRR modes that evaluate directly fitted or fold-matched raw input factors report both
the original projected-ridge metrics and a companion `*_orth` family fitted after a
reduced QR decomposition. PCA and aggregated cross-dataset bases are already
orthonormal and do not run the companion fit.
The iEEG `leave_one_subject_out` and `leave_one_region_out` modes share one
train/transfer/within-held-out implementation. LORO trains on significant electrodes
in the other configured atlas groups, transfers the learned factor to the held-out
region, and also produces the within-region baseline that replaces the former
`per_region` mode. Leave-one-out partitions are validated to have disjoint
`(subject, electrode)` target identities before fitting.

## Tests

Tests cover behavior reachable from retained scripts: configuration and layer fanout,
data/epoch loading, embeddings, significance, RRR and multiview analysis, fMRI,
subspace comparison, language interpretation/features, atlas lookup, and Makefile batch
submission behavior.

See [language_feature_generation.md](language_feature_generation.md) for the reusable
language-feature artifact contract and provider details.
