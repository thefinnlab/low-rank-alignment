# Analysis Artifacts

Core analysis results are owned by an `Artifact` class defined in the script that
produces them. Consumers import that concrete class; there is no generic artifact
reference in configuration or public APIs.

The canonical identity is:

```text
results/<script>/<mode>/<dataset>/offset-<offset>/<model>/layer-<layer>/
```

The current script components are filename stems: `run_significance`, `run_rrr`,
and `run_cross_dataset_rrr`. Dataset components are `podcast`,
`brain-treebank-<stimulus>`, and `deep-fmri`. fMRI always uses `offset-0`.
GloVe dimensions are part of the model (`glove-300`) and its layer is
`layer-none`. Configured model directory names are otherwise preserved.

Each canonical leaf indexes deterministic variants. A variant hashes normalized
effective configuration and upstream Artifact identities while excluding storage
roots, refresh, and plotting switches. Moving a results or data root therefore
does not change run identity. Per-region analyses additionally fingerprint an
explicit atlas file's contents. Loading without a variant is permitted only when
exactly one variant exists.

## Contract

Producer classes declare required and optional fields. Fields may be scalar or
selector collections keyed by logical values such as subject, method, rank, fold,
region, temporal window, held-out subject, or LOSO phase. Selectors contain no
paths. `load_artifacts(fields=..., selectors=...)` loads only requested values and
returns the concrete Artifact instance.

Writes publish payloads atomically and then update the manifest, with a checksum
and physical codec recorded for every selector entry. Partial runs are valid.
Required fields and selector combinations are checked only by `mark_complete()`.
Plot fields are optional, and `plot: false` never prevents completion.

RRR resume planning is based exclusively on manifest entries. A rank/fold is
complete only when its metrics and required fold factors exist, including target
factors when configured. `refresh: false` reuses those checkpoints;
`refresh: true` starts a fresh manifest generation for the deterministic variant.

## Dependencies

iEEG RRR constructs `scripts.sig_lags.run_significance.Artifact` from the explicit
`significance_artifact` config block. Cross-dataset RRR sources use mutually
exclusive `ieeg_rrr_source` or `fmri_rrr_source` blocks containing the selected
producer's constructor arguments. fMRI RRR similarly constructs the fMRI
significance Artifact and, for cross-dataset runs, the iEEG RRR Artifact.

Language decoding and reconstruction remain on their legacy interfaces and are
explicitly deferred from this refactor.

## Legacy migration

`scripts/utilities/migrate_results.py` is dry-run by default. It inventories legacy
trees, emits a YAML mapping template for identities it cannot prove, and copies
only mappings supplied by the operator when `--apply` is given. Copies are
checksum-verified, differing destinations are rejected, identical destinations
are skipped, and repeated application is idempotent. Canonical script roots are
always excluded from legacy discovery.
