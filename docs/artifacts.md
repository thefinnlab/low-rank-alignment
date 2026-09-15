# Analysis Artifacts

Core analysis results are owned by the `Artifact` classes in the producing scripts.
Significance and RRR use unified modality-neutral contracts; language decoding,
language reconstruction, and subspace comparison define result-specific contracts.
Consumers use generic artifact identity blocks rather than result paths.

The canonical identity is:

```text
results/<script>/<mode>/<dataset>/offset-<offset>/<model>/layer-<layer>/
```

Script components include `run_significance`, `run_rrr`,
`run_target_dimensionality_rrr`,
`run_cross_condition_rrr`,
`run_language_decoding`, `run_language_reconstruction`, and `compare_subspaces`.
Dataset components are `podcast`,
`brain-treebank-<stimulus>`, and `deep-fmri`. fMRI always uses `offset-0`.
GloVe dimensions are part of the model (`glove-300`) and its layer is
`layer-none`. Configured model directory names are otherwise preserved.

Each canonical leaf contains readable variant subdirectories. RRR variants include
`significance-<selection-mode>-<digest>`, where the stable digest identifies the
reference and expanded selection-source set. Every configured subspace aggregation
adds an `aggregation-<method>` component, including `aggregation-extrinsic` and
`aggregation-prediction_weighted`. An unspecified (`null`) aggregation adds no path
component; this is the usual EEG default and computes extrinsic aggregation wherever
aggregation is needed. When such an Artifact is used as an RRR source, its tagged
method propagates as `source-aggregation-<method>` so downstream results
also remain distinct. Cross-dataset runs additionally describe their
RRR upstream identity under
`source/<mode>/<dataset>/offset-<offset>/<model>/layer-<layer>`, and specialized
runs use names such as `atlas-<name>`. A configuration that
would collide at the same readable location is rejected and must supply an
explicit `variant_name`. Loading without a variant is permitted only when exactly
one variant exists.

Brain Treebank Laplacian analyses are separated from legacy common-average results.
Significance uses the `rereference-laplacian` variant, and RRR includes the same
rereference label alongside its other readable variant components. RRR dependency
blocks select it with `rereference: laplacian` in addition to selecting the matching
Laplacian significance variant. A cross-dataset result carries the source label as
`source-rereference-laplacian`, keeping common-average and Laplacian transfers distinct.

RRR consumers may pass `significance_selection` directly to the `Artifact`
constructor in place of `variant`, together with `subspace_aggregation` when selecting
an explicitly named aggregation result and `rereference` for a Brain Treebank source.
The Artifact computes the same readable variant, selects it, and verifies the requested
values against its manifest. Explicit `variant` and derived selection arguments are
mutually exclusive. Cross-dataset RRR source blocks use this form so the source factors
are constrained to the intended significant-target set rather than relying on
single-variant discovery.

The `full_fit` RRR mode is a factor-only producer. An iEEG variant stores one
`method: rrr` factor and whole-dataset `input_scalers` entry, plus the matching PCA
pair when `run_pca_baseline` is enabled. Full-fit Artifacts deliberately do not
publish held-out rank metrics. fMRI uses normal internal RRR Artifacts, which save
rank-specific factors alongside held-out metrics.

## Contract

Producer classes declare required and optional fields. Fields may be scalar or
selector collections keyed by logical values such as subject, method, rank, fold,
region, temporal window, held-out subject, or LOSO phase. Selectors contain no paths.
Each producer declares its own preferred selector order; undeclared selector
keys sort alphabetically after those preferred keys. `load_artifacts(fields=...,
selectors=...)` loads only requested values and returns the concrete Artifact
instance.

Writes publish payloads and then update the manifest, with the physical codec and
uncompressed size recorded for every selector entry. `run_rrr` and
`run_cross_condition_rrr` store all non-plot payloads in one compressed
`artifact.h5` container per variant; their plots remain ordinary PNG files in the
existing selector-based subdirectories. The Artifact API still exposes each field
and selector independently. Other producers use individual files: tables are CSV,
arrays are NPY, array mappings are NPZ, plots are PNG, JSON-compatible values are
JSON, and fitted encoding/reconstruction objects are joblib files. Partial runs are valid.
Required fields and selector combinations are checked only by `mark_complete()`.
Plot fields are optional, and `plot: false` never prevents completion.

The cross-condition producer uses generic `source_item` and `target_item` selectors
for pair metrics and item-scoped target manifests. Its configuration includes a
producer contract version so legacy stimulus-named variants are not silently resumed.
Its dataset identity is `brain-treebank` for stimulus items and `podcast` for region
items. Region items intentionally have different target sets, so only input-basis
transfer metrics—not frozen full-map transfer metrics—are produced between regions.

RRR resume planning is based exclusively on manifest entries. A rank/fold is
complete only when its metrics and required fold factors exist, including target
factors when configured. `refresh: false` reuses those checkpoints;
`refresh: true` starts a fresh manifest generation for the deterministic variant.

RRR stores per-fold result tables separately from mean results. Unit-scoped metadata
and named diagnostics use the same method/subject/rank/fold selectors as factors and
metrics. fMRI source factors may additionally identify a delay and basis role.
Leave-one-subject-out fields additionally use `held_out_subject` and `loso_phase`.
For fMRI, `train_subjects` owns the non-held-out rank bases, while `transfer` and
`baseline` own evaluation metrics and summaries; PCA uses `method: pca` with the
`baseline` phase. The held-out full-embedding ceiling is stored once with only its
`held_out_subject` selector and is shared by both phase summaries. Unlike fMRI
internal rank bases, fMRI LOSO factors are partition-scoped and are not implicit
cross-dataset sources.

Target-dimensionality RRR reuses the RRR field and selector contract under its own
script identity. Each fold-safe target-PCA dimension is a readable
`target_pca_dim-<n>` variant combined with the usual significance and dataset
variant components. Component manifests use synthetic `pc-<n>` targets because the
fold-specific bases do not map to individual electrodes or voxels, and PCA state is
not stored. The maximum-dimension variant additionally owns `sweep_results` and the
correlation/R² target-dimension heatmaps compiled across the requested variants.

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
Cross-dataset RRR requires corresponding source and target significance selections.
Their selection modes and reference model/layer must match, while each reference must
match its own dataset-specific RRR identity. For `cross-model-layer`, the complete
expanded model/layer union must also be identical across source and target. The
concrete source Artifact is recorded as an upstream dependency; rank and fold field
selectors therefore need only the RRR or PCA method rather than a serialized source
identity.

Significance exposes complete subject-keyed `target_significance` tables with
`target_id`, `target_kind`, and `significant` columns. RRR exposes common factor,
metric, target-manifest, and mean-result fields. `load_input_factors()` always
returns a list: one matrix per iEEG fold or a one-element fMRI rank-basis set.
`load_mean_results()` preserves additional modality-specific metric columns.
Current iEEG fold factors use the `standardized-v1` coordinate contract and each
`input_factors` entry has a matching `input_scalers` entry containing the training-fold
mean and scale. Consumers must either standardize embeddings with that state before
applying the factor or explicitly convert the factor to the algebraically equivalent
raw-input representation. Resume checks require this paired scaler, so legacy
raw-coordinate factor entries are recomputed rather than reused by current iEEG runs.

## Language interpretation

Language decoding and reconstruction accept one `rrr_source` identity. The selected
RRR Artifact supplies both `method: rrr` and `method: pca` factors; optional
`rrr_selectors` narrow unit-scoped factors such as a subject, region, or temporal
window. The RRR source is recorded as an upstream dependency. Result identities use
the decoded/reconstructed dataset, offset, embedding model, and layer, with the
language mode as the Artifact mode.

Decoding exposes modality-neutral sample selection, backward-compatible word
selection, run metadata, per-task/fold fit metrics, decoder weights, held-out
predictions, the complete grid and summary, optional prediction agreement tables,
and plots. Its `fmri_word` and `fmri_tr_pre_fir` modes take their story split,
embeddings, and rank factors from a deep-fMRI RRR Artifact. The former decodes raw
word-aligned features from locally story-standardized pooled-word embeddings; the
latter uses the source timing and trimming to decode Lanczos-aligned current-TR targets from
projected pre-FIR inputs. Both modes require a normal internal fMRI RRR Artifact.
Categorical TR targets also expose per-channel metrics, availability diagnostics,
smoothed-channel ridge weights, and dominant-label classifier weights.
Reconstruction exposes corresponding provenance,
per-fit metrics, optional fitted models, fold and summary tables, RRR/PCA contrasts,
predictor contrasts, and plots.
In `cross_dataset` language modes, a single full-fit factor is applied directly;
fold-factor aggregation remains only as compatibility behavior for older sources.

The importable downstream facades are `scripts.run_language_decoding.Artifact` and
`scripts.run_language_reconstruction.Artifact`; the existing executable files under
`scripts/language-interpret/` retain their command-line roles.

## Subspace comparisons

Subspace comparison configuration uses an ordered list whose entries contain
`model`, `layers`, and `datasets`; `offsets` defaults to `[0]`. Each configured source
dataset/model/layer/subspace owns a `compare_subspaces` Artifact. Its fields contain
the source-to-target cell cache, fold metrics, summary, row matrices, row heatmaps,
source metadata, validation metadata, and all RRR upstream identities.

A downstream collection loader requires the same explicit selection shape and keeps
only comparisons whose source and target both belong to it:

```python
Artifact(
    analysis_name="all_spaces_over_podcast_saturation_rrr",
    evaluation_dataset="podcast",
    models=[
        {"model": "gpt2", "layers": [0, 6, 12],
         "datasets": ["podcast", "cars-2"]},
        {"model": "gpt2-xl-batch", "layers": [0, 24, 48],
         "datasets": ["podcast"]},
    ],
    results_root="results/subspace_comparisons",
).load_artifacts(["fold_metrics", "summary"])
```

The loaded object also exposes `available_comparisons`, containing the distinct
source/target endpoints actually found. Missing, never-computed pairs are omitted.

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

Legacy manifests with a missing or null `upstreams` field are read as having no
upstream dependencies. They remain incompatible with runs that declare an upstream.

For older iEEG significance artifacts that lack the saved stimulus-row `selection`,
`scripts/utilities/migrate_ieeg_selection.py` performs a metadata-only repair from
the existing epoch caches. It does not retrain significance models or recompute
statistics.
