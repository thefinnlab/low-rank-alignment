# Cross-condition orthogonal RRR

`scripts/run_cross_condition_rrr.py` is a standalone workflow for measuring transfer
of an embedding-space RRR subspace between named conditions. It supports Brain
Treebank movies and podcast atlas regions. It deliberately does not dispatch through
`scripts/run_rrr.py`: all-electrode selection, source fitting, directed pairing, and
its Artifact selectors are different from ordinary significance-backed RRR.

The fitting and directed-pairing code is item-based rather than stimulus-based.
`ComparisonItem` requires a unique `name`, a two-dimensional `X`, and an aligned
two-dimensional `Y`; it may also carry a compact `target_set` name. An `ItemProvider`
returns one `ComparisonGroup` at a time, including context selectors and optional
provider-specific manifests and preflight tables. `get_stimulus_items` produces one
group per Brain Treebank subject, while `get_region_items` produces one podcast group
containing one item per configured atlas region.
Condition-specific providers live under `src/utils/rrr/cross_condition_items/`, one
provider module per item-construction approach. Brain Treebank loading helpers are in
`stimulus.py`; podcast atlas grouping and cache loading are in `region.py`. Select the
provider with the top-level `item_getter: stimulus` or `item_getter: region` config
field. `dataset` must respectively be `brain_treebank` or `podcast`.
Because every movie in a subject group uses the same ordered common contacts and lag
axis, the stimulus provider assigns all of them `subject-<id>-targets`.
Every podcast region uses the same aligned embedding rows but contains all mapped
electrodes from that region across configured subjects. Region items have distinct
`target_set` values, so they never attempt frozen full-mapping transfer.

## Estimand

For every Brain Treebank subject with at least two loadable movies, the stimulus
provider uses the ordered intersection of usable Laplacian contacts and the complete
-2 to +2 second lag axis. The podcast provider instead groups all electrodes assigned
to each configured atlas region, pooling targets across subjects while retaining the
shared podcast embeddings as X. It fits a source RRR model on all words in one item
and evaluates every directed source-to-target pair over contiguous target folds. No
significance Artifact is loaded and no significance filtering is applied.

For each requested rank, the ordered raw factor is first truncated to `A[:, :k]` and
that rank-specific matrix is independently decomposed as `A_k = Q_k R_k`, with
deterministic QR signs. The workflow never obtains a rank-k basis by slicing a Q
matrix fitted at a larger rank. Only the orthonormal `Q_k` is used for projected-ridge evaluation. This makes
the transferred representation a subspace projection rather than a weighted set of
coordinates whose scales interact with ridge. Raw-A projected-ridge results are not
produced.

Frozen full-mapping transfer remains exactly equivalent to the fitted source RRR:
the output factor is changed to `B'_k = B_k R_k.T`, and the reduced source intercept
is retained, so `A_k B_k.T == Q_k B'_k.T`. Target responses are standardized from
the training portion of each target fold before either basis or frozen-map scoring.

The saved methods are:

- `within_rrr`: leakage-safe target RRR `Q` plus a target readout (the oracle).
- `transfer_rrr_basis`: all-source `Q` plus a target-fold readout.
- `transfer_rrr_full`: the frozen, mapping-equivalent source RRR, produced only when
  source and target declare the same non-empty `target_set`.
- `within_pca` and `transfer_pca_basis`: embedding-only orthogonal controls.
- `target_full_ridge`: the unreduced target encoding baseline.

The summary includes an initial transferred-minus-oracle gap, but fold and per-target
correlation and R2 are retained separately so downstream summaries can be changed.
Sample-level predictions and observations are not saved.
Pairs with different outputs still receive both basis-transfer comparisons. A
skipped frozen-map comparison is recorded once in diagnostics with reason
`target_set_missing` or `target_set_mismatch`.

## Running

To prepare a remote host from an empty Brain Treebank data directory, run:

```bash
scripts/download_brain_treebank.sh --data-root data/brain-treebank
uv run python scripts/generate_brain_treebank_embeddings.py \
  --config configs/brain_treebank/cars_2/generate_gpt2.yml
uv run python scripts/utilities/precompute_epochs.py \
  --config configs/cross_condition/stimulus/gpt2.yml \
  --cross-condition
```

Each preparation step is resumable. Embeddings and epoch caches are skipped
when their complete artifacts already exist.

Checked-in region and stimulus configs cover the full default model set, including
randomly initialized GPT-2. Transformer configs use layers at 0%, 25%, 50%, 75%, and
100% depth; GloVe uses 300 dimensions and no layer selector. All configs restrict
analysis rows to words covered by GloVe.
For example:

```bash
uv run python scripts/run_cross_condition_rrr.py \
  --config configs/cross_condition/stimulus/gpt2.yml
```

`per_model_config.layer` may be a scalar or list. Subjects are discovered from
Brain Treebank metadata unless `subjects` is set; an optional `movies` mapping can
restrict individual subjects. Every configured rank must be feasible for every
included subject/movie. The default low-rank SVD computes only the required leading
RRR target directions and has a fixed configurable seed; set `use_low_rank_svd: false`
only when an exact decomposition is computationally practical.

For podcast regions, set `dataset: podcast`, `item_getter: region`, and the podcast
`data_root`. `atlas_region_groups` defaults to `REGION_GROUPS` from
`src/utils/atlas/core.py` and can still be overridden in a config. `subjects: null`
uses subjects 1–9, and
`atlas_path: null` uses nilearn's cached/downloaded Destrieux 2009 atlas. Unlike the
stimulus provider, the region provider neither uses nor validates `rereference`, and
`num_folds` may be any value of at least two.

Alpha selection always runs because it is part of fitting the ridge readouts, but
its selected-alpha arrays are not saved by default. Set `log_alphas: true` to retain
them in the metric payloads; leave it false to keep only the resulting scores.

The Makefile runs either comparison family for any selected models:

```bash
make cross-condition DATASET=podcast CROSS_MODE=region
make cross-condition DATASET=brain-treebank CROSS_MODE=stimulus
make cross-condition DATASET=podcast CROSS_MODE=region CMD="uv run python"
```

Preflight records empty contact selections and groups with fewer than two items.
Provider loading, alignment, and shape errors are raised immediately. A run with no
analyzable comparison saves its exclusion diagnostics but is not marked complete.

## Artifact contract

The producer identity is `run_cross_condition_rrr/cross_condition/<dataset>`, where
the dataset is `brain-treebank` for stimulus items or `podcast` for region items.
Rank metrics are selected by provider context (subject for stimulus items), source
item, target item, method, rank (except the unreduced baseline), and target fold.
Factor storage is not rank-expanded: each group/source item has one full raw RRR `A`
matrix, selected by provider context, `source_item`, and `method: rrr`. Target
manifests are selected by provider context and `target_item`. Rank-specific `Q_k` and
`R_k` can be regenerated
by applying deterministic QR to `A[:, :k]`. No Q, R, B, intercept, PCA basis, or
rank-specific factor copies are persisted. Eligibility, exclusions, target manifests,
fold results, pair summaries, and skipped-comparison diagnostics are stored as
separate fields. The contract version participates in the variant digest, so legacy
`source_stimulus`/`target_stimulus` variants remain untouched and are not resumed.
