# Analysis Artifacts

Core analysis results are owned by the unified `Artifact` classes in
`scripts/run_significance.py` and `scripts/run_rrr.py`. Consumers use generic
artifact identity blocks; the dataset selects the underlying implementation.

The canonical identity is:

```text
results/<script>/<mode>/<dataset>/offset-<offset>/<model>/layer-<layer>/
```

The script components are `run_significance` and `run_rrr`. Dataset components are `podcast`,
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

RRR constructs the unified significance Artifact from `significance_artifact`.
Cross-dataset and multiview inputs use generic `rrr_source`/`source` identities;
there are no modality-named dependency fields. `internal` identifies the ordinary
within-dataset RRR producer for both modalities.

Significance exposes complete subject-keyed `target_significance` tables with
`target_id`, `target_kind`, and `significant` columns. RRR exposes common factor,
metric, target-manifest, and mean-result fields. `load_input_factors()` always
returns a list: one matrix per iEEG fold or a one-element fMRI rank-basis set.
`load_mean_results()` preserves additional modality-specific metric columns.

Language decoding and reconstruction remain on their legacy interfaces and are
explicitly deferred from this refactor.

## Legacy migration

`scripts/utilities/migrate_results.py` is dry-run by default. It inventories legacy
trees, emits a YAML mapping template for identities it cannot prove, and converts
recognized files into loadable unified Artifact fields when `--apply` is given.
Source checksums are recorded, differing reruns are rejected, identical reruns are
skipped, and incomplete legacy runs remain partial manifests. Canonical script
roots are excluded from legacy discovery.
