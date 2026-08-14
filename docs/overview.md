# Repository Overview

This repository contains script-driven neuroscience analyses of how language-model
embeddings predict iEEG and fMRI responses. The supported executable surface is
`scripts/`; utilities exist to serve those workflows rather than as a standalone API.

## Retained workflows

- `scripts/sig_lags/run_significance.py` performs iEEG lag-cluster significance.
- `scripts/rrr/run_rrr.py` performs iEEG reduced-rank regression, PCA baselines,
  cross-dataset transfer, temporal-window analyses, and multiview analyses.
- `scripts/rrr/compare_subspaces.py` performs configured leakage-safe comparisons
  between saved RRR/PCA spaces.
- `scripts/fmri/` contains fMRI embedding generation, significance, RRR, and held-out
  language-decoding workflows.
- `scripts/language-interpret/` contains language decoding and reconstruction.
- `scripts/utilities/` contains data preparation, embedding generation, language-feature
  generation, plotting, and atlas-reporting entrypoints.

The Makefile exposes the retained workflows. Core significance and RRR producers
own result persistence through script-specific `Artifact` classes. Their canonical
identity, variants, selector fields, resume behavior, and legacy migration are
described in [artifacts.md](artifacts.md). Language decoding and reconstruction are
temporarily deferred from that storage refactor.

## Source layout

- `src/core/config.py` defines configuration dataclasses for retained entrypoints.
- `src/core/registry.py` defines embedding and dataset adapter registries.
- `src/utils/shared/` contains configuration, embedding, iEEG/fMRI loading, ridge,
  correlation, alpha-grid, and tensor helpers used by multiple workflows.
- `src/utils/significance/` contains iEEG lag-cluster significance helpers.
- `src/utils/rrr/` contains dense RRR fitting, input preparation, evaluation,
  multiview handling, result helpers, and saved-subspace I/O.
- `src/utils/subspace_comparison/` contains saved-space metrics and directed comparison
  orchestration.
- `src/utils/language_interpret/` contains decoding, reconstruction, and task registry
  utilities shared by iEEG and fMRI scripts.
- `src/utils/language_features/` contains the reusable annotation pipeline and built-in
  providers.
- `src/utils/atlas/` and `src/utils/plotting/` contain atlas and retained plotting code.

Imports use these explicit package paths. The former flat `src.utils.<module>` paths
are intentionally unsupported.

## Tests

Tests cover behavior reachable from retained scripts: configuration and layer fanout,
data/epoch loading, embeddings, significance, RRR and multiview analysis, fMRI,
subspace comparison, language interpretation/features, atlas lookup, and Makefile batch
submission behavior.

See [language_feature_generation.md](language_feature_generation.md) for the reusable
language-feature artifact contract and provider details.
