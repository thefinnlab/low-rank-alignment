# Repository Overview

This repository contains script-driven neuroscience analyses of how language-model
embeddings predict iEEG and fMRI responses. The supported executable surface is
`scripts/`; utilities exist to serve those workflows rather than as a standalone API.

## Retained workflows

- `scripts/run_significance.py` dispatches significance by dataset to the iEEG
  lag-cluster or fMRI held-out-session implementation.
- `scripts/run_rrr.py` dispatches reduced-rank regression by dataset while exposing
  one Config and Artifact contract.
- `scripts/rrr/compare_subspaces.py` performs configured leakage-safe comparisons
  between saved RRR/PCA spaces and publishes one source-organized Artifact per
  generating dataset/model/layer/subspace.
- `scripts/fmri/` contains fMRI embedding generation and held-out
  language-decoding workflows.
- `scripts/language-interpret/` contains language decoding and reconstruction.
- `scripts/utilities/` contains data preparation, embedding generation, language-feature
  generation, plotting, and atlas-reporting entrypoints.

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
  result helpers. iEEG RRR mode implementations are organized under
  `src/utils/rrr/ieeg_rrr_mode/`, with `rrr_ieeg.py` retained as the compatibility
  facade.
- `src/utils/subspace_comparison/` contains saved-space metrics and directed comparison
  orchestration.
- `src/utils/language_interpret/` contains decoding, reconstruction, and task registry
  utilities shared by iEEG and fMRI scripts.
- `src/utils/language_features/` contains the reusable annotation pipeline and built-in
  providers.
- `src/utils/atlas/` and `src/utils/plotting/` contain atlas and retained plotting code.

Imports use these explicit package paths. The former flat `src.utils.<module>` paths
are intentionally unsupported.

Significance and RRR YAMLs use a top-level dataset discriminator, shared settings,
and exactly one `ieeg` or `fmri` block. RRR uses modality-neutral `internal` and
`cross_dataset` modes; specialized modes remain iEEG-only. RRR significance selection
is configured uniformly across modalities as `within-layer` or as an explicit
`cross-model-layer` union; union artifact entries may compactly list several layers
for one model. Cross-model-layer batch RRR uses per-model configs so model-specific
rank and ridge-memory settings can be retained for both iEEG and fMRI.
Cross-dataset RRR source identities include their own significance selection, which
deterministically resolves and validates the saved source variant.
Source and target selections are required to be corresponding within-layer selections
for the same model and layer over their respective datasets.

## Tests

Tests cover behavior reachable from retained scripts: configuration and layer fanout,
data/epoch loading, embeddings, significance, RRR and multiview analysis, fMRI,
subspace comparison, language interpretation/features, atlas lookup, and Makefile batch
submission behavior.

See [language_feature_generation.md](language_feature_generation.md) for the reusable
language-feature artifact contract and provider details.
