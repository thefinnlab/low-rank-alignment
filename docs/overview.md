# Repository Overview

This repository contains script-driven neuroscience analyses of how language-model
embeddings predict iEEG and fMRI responses. The supported executable surface is
`scripts/`; utilities exist to serve those workflows rather than as a standalone API.

## Retained workflows

- `scripts/run_significance.py` dispatches significance by dataset to the iEEG
  single-lag correlation or fMRI held-out-session implementation.
- `scripts/run_rrr.py` dispatches reduced-rank regression by dataset while exposing
  one Config and Artifact contract. Its `full_fit` mode fits a single reusable
  factor set on every selected row and significant target, without a held-out rank
  evaluation, for iEEG sources.
- `scripts/run_cross_condition_rrr.py` runs standalone directed transfer between
  provider-generated X/Y items with no significance filtering and orthogonalized RRR
  input spaces. Its selectable providers compare Brain Treebank movies within subject
  using common Laplacian contacts or podcast atlas regions using all mapped contacts;
  see
  [cross_condition_rrr.md](cross_condition_rrr.md).
- `scripts/run_target_dimensionality_rrr.py` reruns internal RRR over fold-safe
  target-PCA dimensions, with one Artifact variant per dimension and compiled
  correlation/R² sweep outputs.
- `scripts/rrr/compare_subspaces.py` performs configured leakage-safe comparisons
  between saved RRR/PCA spaces and publishes one source-organized Artifact per
  generating dataset/model/layer/subspace.
- `scripts/fmri/` contains fMRI embedding generation workflows.
- `scripts/language-interpret/` contains language decoding and reconstruction.
  Language decoding includes word-aligned and TR-aligned pre-FIR methods for fMRI,
  all published through the same Artifact client.
  Decoders write result fields directly into a compressed HDF5 Artifact container,
  with plots retained as separate PNGs.
  TR/pre-FIR decoding also supports speaking rate: word onsets per acquisition
  interval divided by its duration, including silent intervals.
  `scripts/run_language_decoding_batch.py` drives checked-in internal and all six
  directed Podcast/Cars 2/fMRI manifests, running all configured layers in one job
  per model by default. Pass `--layer` (or `make language-decoding LAYER=<layer>`)
  to isolate one configured layer in its own job, or use
  `make language-decoding PER_LAYER=true` to submit every configured model/layer
  combination separately. `LANGUAGE_DECODING_CPUS=<n>` controls each Slurm
  job's CPU allocation (default 1). `FMRI_TR_PRE_FIR=true` selects TR decoding
  for fMRI targets in both ordinary and missing-only Make targets; categorical decoder cross-validation uses
  every CPU available to the process. Directed decoding applies the source dataset's saved internal basis to
  the target transcript; iEEG sources use their single `full_fit` basis.
  For directed decoding into fMRI, `fmri_target_artifact` supplies the target data
  configuration, while `rrr_source` supplies the transferred RRR/PCA bases.
  Its optional `--layer` filter supports small exploratory runs, and output-root
  overrides leave upstream RRR lookups at their separately configured roots.
  Word decoding uses the label offsets declared by each manifest and the same
  upstream basis; TR decoding stays at label offset 0. fMRI targets decode words
  by default for both internal and directed cross-dataset decoding. The
  `fmri_tr_pre_fir` flag selects TR decoding and adds speaking rate. All fMRI
  decoding uses `within_dataset` or `cross_dataset` Artifact modes according to
  the source/target relationship; TR results occupy an additional
  `fmri_tr_pre_fir` folder within that mode's path.
  Word-aligned speaker decoding retains individual speakers only when present in
  every outer fold's training and test sets; remaining speakers share an `other`
  class. Speaker runs use a separate Artifact variant and save the class mapping.
- `scripts/utilities/` contains data preparation, embedding generation, language-feature
  generation, plotting, atlas-reporting entrypoints, and a read-only iEEG utility for
  comparing max-lag correlation with electrode-specific or subject-wide shared-max
  nulls, max-lag correlation/SEM, and lag-cluster significance on matched encoding
  fits and circular-shift null samples. A companion
  refit utility parses that comparison's console report and recomputes all-lag
  correlations for only the union of selected electrodes, without permutation tests.
  `generate_recurrence_features.py` updates existing word-recurrence labels across
  datasets without rerunning unrelated annotation providers.
  `generate_audio_intensity.py` adds Brain Treebank-style 500 ms RMS intensity
  labels from Podcast and fMRI WAV files to their word-feature artifacts.
  `run_audio_intensity_decoding_sweep.py` runs the intensity-only internal decoder
  over the configured Podcast and fMRI model/layer and word-offset grids.
  Intensity RMS decoding is an optional task stored in the standard language
  decoding Artifact. `migrate_intensity_decoding.py` moves newer rank-only
  intensity results into matching standard Artifacts, including their aggregate
  tables and plots; it defaults to a dry run and requires `--apply` to write.
  `check_rrr_results.py` checks the complete internal, leave-one-subject-out, and
  directed cross-dataset matrix for podcast, Cars 2, and fMRI, reporting artifacts
  that are absent or not marked complete by the standard manifest contract.
  `check_language_decoding_results.py` similarly checks every configured model,
  layer, and label offset for both internal decoding and all directed
  Podcast/Cars 2/fMRI decoding pairs. The `language-decoding-missing` Make target
  submits each missing or incomplete model/layer as a separate job with refresh
  disabled.

The Makefile has six individual analysis targets: `significance`, `rrr`,
`multi-view`, `target-dimensionality`, `cross-condition`, and `language-decoding`.
They share `DATASET`, `CMD`, and `OVERRIDES` controls; all except `multi-view`
also use `MODELS`. RRR and target-dimensionality additionally use
`RRR_MODE`, while RRR uses `SOURCE_DATASET` for cross-dataset transfer and
cross-condition uses `CROSS_MODE=region|stimulus`. Target-dimensionality accepts
the ordinary internal `cross_subject` and `cross_model_layer` configurations and
submits one job per selected model; the script expands configured model layers and
PCA dimensions within each job. For fMRI RRR,
`RRR_MODE=cross_model_layer_per_layer` reuses the cross-model-layer configs but
submits each configured model layer as a separate process or Slurm job, providing
a hard memory-release boundary between layers; the ordinary `cross_model_layer`
mode continues to submit one job per model. Slurm submission is the default; cross-condition
submits one job per configured model layer (and one layerless job for GloVe), as do
iEEG and fMRI leave-one-subject-out RRR. `CMD="uv run python"` runs the same jobs locally.
The `multi-view` target submits one combined-model job for Podcast, Cars 2, or fMRI
using the corresponding checked-in configuration. The default model set
is GPT-2, GPT-2 XL, Llama 3 8B, GPT-J 6B, RoBERTa Large, randomly initialized
GPT-2, and GloVe 300D. fMRI supports this complete set and supports its internal,
cross-model-layer, leave-one-subject-out, and cross-dataset RRR configurations;
cross-condition remains iEEG-only.
The `final-results` target checks that every significance Artifact in the configured
cross-model/layer union is complete, submits internal RRR for podcast, Cars 2, and
fMRI, leave-one-subject-out RRR for both iEEG datasets, and both cross-condition
families. Its fMRI-to-podcast and fMRI-to-Cars-2 jobs use Slurm `afterok`
dependencies on the complete internal-fMRI batch.
The `significance-missing` (also available as `missing-significance`) target checks
the same model/layer matrix and submits only missing or incomplete significance
Artifacts, grouping all missing layers into one job per dataset/model. With the
default `refresh: false`, the client skips already-present layer artifacts.

Each retained analysis workflow owns one client-facing `Artifact` class. Their canonical
identity, variants, selector fields, resume behavior, and legacy migration are
described in [artifacts.md](artifacts.md). Language decoding and reconstruction load
RRR/PCA factors through RRR Artifact identities and publish task-level results through
their own Artifacts.
RRR, cross-condition RRR, and language-decoding variants compact their non-plot fields into one compressed
HDF5 container while keeping plots as separately browsable PNG files.

## Source layout

- `notebooks/final_plots.py` is the curated paper-figure notebook. It loads the
  completed RRR, significance, and decoding results once at startup, reports
  missing directed results, and draws the aggregate, encoding, significance,
  quantile, and paired RRR/PCA saturation-sensitivity figures from compact
  in-memory data. Its decoding aggregate includes a volume feature that aligns
  Podcast and fMRI audio RMS with Cars 2 native intensity RMS.
  Its leave-one-subject-out aggregate compares transfer and PCA with a
  subject-matched RRR control from the full internal training run.
  A fingerprinted cache under `results/` makes later starts fast while artifact
  manifests and metric code are unchanged.
  Shared artifact-independent rank calculations live in
  `src/utils/rrr/plot_metrics.py`; reusable axis and figure functions live in
  `src/utils/plotting/`.
- `notebooks/plots.py` provides interactive RRR/PCA comparisons, including aggregate
  landscape figures for internal, leave-one-subject-out, and directed cross-dataset
  RRR. Internal and leave-one-subject-out views stack selected model/layer plots
  with dashed red full-model references (including full-embedding regression for LOSO)
  and normalized convergence line plots by dataset, with shared
  x-axes and saturation distributions in columns on the right that share a y-axis.
  Saturation controls default to 99% of each eligible target quantile's maximum
  correlation. They use the 5th, 10th, 25th, 50th, 75th, 90th, and 95th
  percentiles, including only those whose peak correlation exceeds 0.005.
  They average each target across folds, treat quantile progress as monotonic,
  and linearly interpolate the first rank where every eligible quantile has
  crossed its threshold. RRR is blue and PCA
  is green; transfer curves use darker shades and dashed lines.
  Cross-dataset shows
  all six directed normalized convergence panels alongside paired distributions
  of normalized AUC differences (internal RRR minus transfer RRR, transfer RRR minus
  transfer PCA, and internal PCA minus transfer PCA) for each directed pair. A
  separate scatter figure compares the PCA and RRR internal-minus-transfer AUC gaps
  for every directed model/layer result. A linear-model figure compares the PCA
  gap alone with the PCA gap plus offset-zero cross-dataset correlation of decoded
  feature advantage, showing fitted and leave-one-model-layer-out R². A
  language-decoding aggregate shows
  internal RRR–PCA relative-difference curves for concreteness, sentence position,
  and entropy beside offset-zero correlations of the same advantage across datasets.
  Interactive plots also
  include a shared top-level model multiselect that filters internal RRR, leave-one-subject-out,
  and directed cross-dataset RRR results before plotting (defaulting to all available
  models),
  electrode/voxel's performance relative to its best tested rank within RRR or PCA, summarized
  across targets as a mean with ±1 SD after averaging evaluation folds. Its normalized
  convergence plots compare RRR and PCA across model layers by interpolating monotonic
  curves over zero-based relative rank on a symmetric-log axis and weighting each model
  layer equally. Internal RRR/PCA, leave-one-subject-out, and directed cross-dataset
  views each provide paired AUC and saturation-rank distributions, with an independent
  percentage-of-maximum slider for each view. Correlations are averaged in Fisher-z
  space and converted back before normalization and thresholding. Saturation is the
  first linearly interpolated rank reaching the selected percentage; LOSO computes it per subject
  before averaging ranks per model/layer, using the same baseline selector as AUC.
  Dedicated loading cells retain compact per-rank tables in memory: internal means,
  LOSO subject means (including both baseline choices), and cross-dataset fold means.
  Convergence filters, saturation controls, and LOSO/cross-dataset model-layer
  selectors reuse these tables without artifact reads. Compact loader caches can be
  cleared before rerunning loading cells to refresh results; target-level violin
  plots still load individual target values for the selected artifact.
  The leave-one-subject-out convergence view applies the same comparison to the RRR
  baseline, transferred RRR, and PCA baseline after averaging subjects within each model
  layer and rank. Directed cross-dataset panels compare within-dataset RRR/PCA with their
  transferred counterparts after averaging folds within each model layer, retaining only
  model layers with all four curves for that direction; each directed dataset pair also
  has a paired model-layer distribution of normalized-curve AUCs. All AUC summaries
  integrate normalized monotonic performance over normalized log rank from rank 1 to the
  curve's maximum rank, independently of the symmetric-log display scale. It also
  provides a reusable paired-distribution plot combining violins, adjacent-pair
  lines, distribution-colored points, and box plots. Its language-decoding view
  discovers currently saved internal and directed cross-dataset summaries,
  filters them by target dataset, model/layer, source space, and label offset,
  and tabulates selectable continuous and categorical metrics for RRR, PCA, and
  their raw and epsilon-stabilized relative differences at each available
  rank, clamping RRR and PCA scores independently to zero for the relative
  calculation while retaining their raw displayed values. A companion table
  aggregates every saved dataset, model/layer, source
  space, offset, and target across ranks with positive RRR performance, retains
  results where RRR beats PCA at two or more ranks, and orders them by the
  relative-difference curve's AUC over normalized log-rank. The selected-artifact
  and averaged-summary tables use the same AUC ordering, while the
  transfer-correlation table remains correlation-ranked. Selecting a row in this summary
  reactively plots its relative difference across retained positive-RRR ranks,
  distinguishing ranks where RRR does and does not beat PCA. A reusable
  averaged-summary view groups curves by normalized model depth, model family,
  internal versus cross-dataset condition, or causal versus bidirectional
  architecture, excluding random GPT-2 and layerless GloVe. It can either
  average member-relative curves or average clamped RRR/PCA scores before
  computing relative difference. Selected summaries show translucent member
  curves behind the mean and report incomplete per-rank member coverage. A
  transfer-consistency view integrates each clamped relative curve over
  normalized log-rank, matches each directed cross-dataset result to either its
  source- or target-internal result by model, layer, task, and label offset, and
  ranks directed dataset pairs by Pearson correlation. Its internal-fMRI and
  fMRI-target cross-dataset RRR views explicitly select the
  `input_scaling-pooled-training-stories-v1` variants. Offsets are selectable;
  selecting a pair displays its AUC scatter, fitted line, and identity line.
- `src/core/config.py` defines the modality implementation dataclasses; the two
  public facade scripts define the strict client Configs.
- `src/core/registry.py` defines embedding and dataset adapter registries.
- `src/utils/shared/` contains configuration, embedding, iEEG/fMRI loading, ridge,
  correlation, alpha-grid, and tensor helpers used by multiple workflows.
- `src/utils/significance/` contains the modality-specific significance runners and
  their reusable statistical helpers.
- `src/utils/rrr/` contains the modality-specific RRR runners, dense fitting, input
  preparation, evaluation, multiview handling, cross-temporal-window transfer, and
  result helpers, including the standalone cross-condition mechanics.
  `src/utils/rrr/cross_condition_items/` keeps each condition-specific item provider
  in its own module: `stimulus.py` provides Brain Treebank movie items and `region.py`
  provides podcast atlas-region items. Mode implementations are organized under
  `src/utils/rrr/ieeg_rrr_mode/` and `src/utils/rrr/fmri_rrr_mode/`, with
  `rrr_ieeg.py` and `rrr_fmri.py` retained as compatibility facades. The fMRI package
  separates shared loading, fitting, evaluation, and Artifact helpers from its
  internal, cross-dataset, and leave-one-subject-out modes.
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
configuration contract. The cross-condition setup helpers download the eligible
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
RRR uses `fmri.ridge.log_alphas`, and the standalone cross-condition runner uses its
top-level `log_alphas`; all default to false while alpha selection itself still runs.
Cross-dataset RRR source identities include their own significance selection, which
deterministically resolves and validates the saved source variant. Brain Treebank
sources additionally declare `rrr_source.rereference`, so Laplacian source variants
are selected and remain distinct in downstream result identities.
Source and target selections must use the same selection mode, reference model/layer,
and, for cross-model-layer selection, the same model/layer union over their respective
datasets.
The reusable `full_fit` mode fits one ridge/RRR model on all iEEG source stimulus
words and significant electrodes and saves its RRR factor plus the whole-dataset
embedding scaler. Its optional PCA factor is fit and saved from the same standardized
rows. Cross-dataset RRR consumes this saved factor when the selected source is a
`full_fit` Artifact, while retaining the previous on-demand whole-stimulus refit for
older source modes. Normal internal fMRI RRR saves whole-training-stimulus,
rank-specific RRR and PCA bases alongside its held-out evaluation. Aggregation remains
only where the fMRI FIR RRR factor is combined across delay blocks, using the
configured equal-subspace
`extrinsic` method, raw-coefficient `factor_weighted` method, or normalized
prediction-energy `prediction_weighted` method. The prediction-weighted option uses
squared singular-value energy from the SVD of the full ridge model's training
predictions and is available only for within-fMRI FIR aggregation. It is the default
for internal fMRI FIR fitting and for loading an fMRI source into cross-dataset RRR;
unspecified EEG aggregation continues to compute with the extrinsic method.
Checked-in `full_fit` configs cover every layer of the seven primary models for
Podcast and Cars 2. They retain the final cross-model/layer significance union
and can be launched with `make rrr DATASET=<podcast|cars-2> RRR_MODE=full_fit`.
Directed Podcast/Cars 2 transfers, including those targeting fMRI, select these
`full_fit` artifacts as their iEEG sources.
`make regenerate-transfer-results` submits these source fits and the per-layer fMRI
source fits first, submits all directed cross-dataset RRR and decoding consumers with
Slurm `afterok` dependencies, regenerates internal decoding, and waits until the whole
submitted pipeline has completed. Its Slurm requests use 128G/GPU80 for iEEG RRR,
200G/GPU80 for fMRI RRR, and 32G GPU nodes without the GPU80 constraint for language
decoding.
Cross-dataset RRR applies every transferred RRR/PCA basis in its source dataset's
saved standardized coordinates. It applies the source mean and scale to target
embeddings, projects through a QR-orthonormalized source basis, and passes those
reduced vectors directly to ridge without target-dataset normalization. fMRI applies
the same source-only transform before FIR expansion, while full-embedding ceilings
retain the target dataset's own training-only scaler.
Native iEEG factors retain their fold or whole-dataset input scaler. fMRI RRR fits one
input scaler over all aligned TRs from the pooled training stories, applies it unchanged
to held-out stories, and saves it once at artifact scope for all RRR/PCA ranks.
Corrected fMRI source outputs use a versioned input-scaling variant so they cannot
silently resume results produced under the former per-story contract. fMRI
significance remains on its separate per-story normalization workflow.
Internal iEEG RRR and PCA fit and save their factors in embeddings standardized with
that fold's training statistics. Each fold factor is paired with its saved input mean
and scale. Direct rank evaluation, fold-matched subspace comparison, and language
workflows standardize inputs before projection; consumers operating on raw embeddings
convert the factor with the saved scale. Consequently, reduced QR at the complete
embedding rank is an orthogonal rotation of the same standardized design used by the
full ridge model and reproduces that model's ceiling.
The iEEG and fMRI `multi_view` modes select every component through a complete RRR
Artifact identity, apply each component's saved input scaler to target embeddings,
and project through a QR-orthonormalized basis at that view's requested rank in
standardized coordinates. They publish the combined fit through the normal RRR
Artifact fields. A required `multi_view_name` supplies its synthetic model identity;
the component Artifacts remain explicit upstreams. Matching-dataset iEEG sources use
fold-matched factors, while cross-dataset iEEG sources use a single `full_fit` basis
trained on the complete source dataset and fMRI sources use their requested
whole-training rank-specific basis. fMRI targets apply each component scaler before
FIR expansion, fit one band per view and delay with training-session CV, and aggregate
the fitted full-FIR RRR delay blocks into a reusable pre-FIR basis. The checked-in
Podcast, Cars 2, and fMRI configurations use five layers from each of six transformer
families plus GloVe, retaining 64 components per view; four-delay fMRI runs therefore
use 124 bands. Cars 2 selects subjects 3, 7, and 10 with Laplacian rereferencing.
Cross-dataset iEEG language decoding consumes a single `full_fit` factor and its scaler
directly rather than computing a Grassmann mean across source folds. fMRI language
decoding consumes rank factor/scaler pairs from a normal internal fMRI RRR Artifact.
Artifact-backed language decoding reads multi-view provenance from the RRR Artifact
rather than requiring a legacy materialized `config.yml`. All decoding modes apply the
saved source scaler before projection. fMRI word and TR/pre-FIR decoding reconstruct
fMRI multi-view component spaces from that provenance, and directed word decoding can
transfer a saved fMRI multi-view source into an iEEG target. Directed
cross-dataset decoding uses a QR-orthonormalized source basis and passes its projected
vectors directly to the decoder without target-dataset normalization; internal
decoding retains its ordinary training-fold decoder scaler.
RRR modes that evaluate directly fitted or fold-matched input factors report both
the original projected-ridge metrics and a companion `*_orth` family fitted after a
reduced QR decomposition. PCA and aggregated cross-dataset bases are evaluated in
standardized coordinates and are already orthonormal, so these cases do not run the
companion fit.
Automatically generated ridge-alpha grids use 25 log-spaced values, including
cached boundary-refined grids; explicitly configured alpha lists are used unchanged.
Word-aligned iEEG analyses default to the shared GloVe-covered vocabulary through
`EncodingConfig.limit_to_glove`. Cross-condition configs expose the matching
`limit_to_glove` and `glove_dim` controls and enable the same 300D vocabulary filter.
Contextual embeddings are still generated for the complete transcript; filtering is
applied when embeddings and neural epochs are aligned for analysis.
`load_embeddings(..., limit_to_glove=True, glove_dim=300)` owns the auxiliary GloVe
lookup and adds the `in_glove` column to its returned word table, so analysis clients
do not load GloVe separately.
fMRI significance and RRR expose the same controls through `FMRIDataConfig`. Because
fMRI embeddings are interpolated onto TRs, GloVe-OOV word rows retain their timestamps
and are zeroed for every model before Lanczos interpolation rather than being removed.
The iEEG `leave_one_subject_out` mode fits only the fold-specific RRR input factors on
the non-held-out subjects, without evaluating that training cohort or running its PCA
baseline. For the held-out subject it evaluates ridge models only after reduced-QR
orthonormalization of the transferred A matrix, plus the matching within-subject RRR
baseline (and its optional PCA baseline). Partitions are validated to have disjoint
`(subject, electrode)` target identities before fitting.
The fMRI `leave_one_subject_out` mode uses the shared training stories and fixed test
story for every partition. It fits FIR RRR factors on the selected voxels from all
non-held-out subjects without evaluating that cohort, aggregates delay blocks with
the configured fMRI method, and evaluates reduced-QR transferred factors with fresh
ridge models on the held-out subject. Each held-out subject also receives a matching
within-subject RRR baseline and, by default, a pre-FIR PCA baseline. One held-out
full-embedding ridge ceiling is shared by both evaluation phases. Response loading is
phase-local, and completed partitions resume from their saved Artifact fields.

## Tests

Tests cover behavior reachable from retained scripts: configuration and layer fanout,
data/epoch loading, embeddings, significance, RRR and multiview analysis, fMRI,
subspace comparison, language interpretation/features, atlas lookup, and Makefile
analysis submission behavior.

See [language_feature_generation.md](language_feature_generation.md) for the reusable
language-feature artifact contract and provider details.
