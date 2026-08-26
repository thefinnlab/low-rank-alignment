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

Each canonical leaf contains readable variant subdirectories. RRR variants include
`significance-<selection-mode>-<digest>`, where the stable digest identifies the
reference and expanded selection-source set. Cross-dataset runs also describe their
RRR upstream identity under
`source/<mode>/<dataset>/offset-<offset>/<model>/layer-<layer>`, and specialized
runs use names such as `orthonormalized` or `atlas-<name>`. A configuration that
would collide at the same readable location is rejected and must supply an
explicit `variant_name`. Loading without a variant is permitted only when exactly
one variant exists.

RRR consumers may pass `significance_selection` directly to the `Artifact`
constructor in place of `variant`. The Artifact computes the same stable digest,
selects that variant, and verifies the selection recorded in its manifest. Explicit
`variant` and `significance_selection` arguments are mutually exclusive. Cross-dataset
RRR source blocks use this form so the source factors are constrained to the intended
significant-target set rather than relying on single-variant discovery.

## Contract

Producer classes declare required and optional fields. Fields may be scalar or
selector collections keyed by logical values such as subject, method, rank, fold,
region, temporal window, held-out subject, or LOSO phase. Selectors contain no
paths. Each producer declares its own preferred selector order; undeclared selector
keys sort alphabetically after those preferred keys. `load_artifacts(fields=...,
selectors=...)` loads only requested values and returns the concrete Artifact
instance.

Writes publish payloads atomically and then update the manifest, with the physical
codec and file size recorded for every selector entry. Tables are CSV, arrays are
NPY, array mappings are NPZ, plots are PNG, JSON-compatible values are JSON, and
fitted encoding objects are joblib files. Partial runs are valid.
Required fields and selector combinations are checked only by `mark_complete()`.
Plot fields are optional, and `plot: false` never prevents completion.

RRR resume planning is based exclusively on manifest entries. A rank/fold is
complete only when its metrics and required fold factors exist, including target
factors when configured. `refresh: false` reuses those checkpoints;
`refresh: true` starts a fresh manifest generation for the deterministic variant.

RRR stores per-fold result tables separately from mean results. Unit-scoped metadata
and named diagnostics use the same method/subject/rank/fold selectors as factors and
metrics. fMRI source factors may additionally identify a delay and basis role.

Significance diagnostics and plots may use a name in addition to their subject
selector, allowing distinct control correlations, alpha diagnostics, and diagnostic
figures to coexist for one subject.

## Dependencies

RRR resolves significance through `significance_selection`. `within-layer` uses its
single `reference_artifact`; `cross-model-layer` selects the OR-union of its explicit
`artifacts`, while the reference continues to provide iEEG folds and encoding
metadata. A union artifact may set `layer` to a list, which expands to one concrete
Artifact identity per layer. The reference and every expanded selection Artifact are
recorded as upstreams.
Cross-dataset and multiview inputs use generic `rrr_source`/`source` identities;
there are no modality-named dependency fields. `internal` identifies the ordinary
within-dataset RRR producer for both modalities.
Cross-dataset RRR currently requires corresponding `within-layer` source and target
significance selections: their model and layer must match, while each reference must
match its own dataset-specific RRR identity. The concrete source Artifact is recorded
as an upstream dependency; rank and fold field selectors therefore need only the RRR
or PCA method rather than a serialized source identity.

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
Source paths and sizes are recorded, every rerun is semantically compared with the
decoded Artifact values, differing reruns are rejected, identical reruns are
reported as skipped, and incomplete legacy runs remain partial manifests. Legacy dependency
directories are replaced by path-free `rrr_source` identities before configuration
storage. Canonical script roots are excluded from legacy discovery;
rank-saturation, subspace-comparison, multiview, and config-less nested
cross-dataset outputs are intentionally deferred.

For older iEEG significance artifacts that lack the saved stimulus-row `selection`,
`scripts/utilities/migrate_ieeg_selection.py` performs a metadata-only repair from
the existing epoch caches. It does not retrain significance models or recompute
statistics.
