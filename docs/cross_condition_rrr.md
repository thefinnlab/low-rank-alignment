# Cross-condition orthogonal RRR

`scripts/run_cross_condition_rrr.py` is a standalone Brain Treebank workflow for
measuring within-subject transfer of an embedding-space RRR subspace between movies.
It deliberately does not dispatch through `scripts/run_rrr.py`: stimulus discovery,
all-electrode selection, source fitting, directed pairing, and its Artifact selectors
are different from ordinary significance-backed RRR.

The fitting and directed-pairing code is item-based rather than stimulus-based.
`ComparisonItem` requires a unique `name`, a two-dimensional `X`, and an aligned
two-dimensional `Y`; it may also carry a compact `target_set` name. An `ItemProvider`
returns one `ComparisonGroup` at a time, including context selectors and optional
provider-specific manifests and preflight tables. `get_stimulus_items` is the
current provider and produces one group per subject. A future provider can therefore
construct items from conditions such as brain regions without duplicating the
pairwise analysis.
Condition-specific providers live under `src/utils/rrr/cross_condition_items/`, one
provider module per item-construction approach. The current getter and its Brain
Treebank loading helpers are in `stimulus.py`; a region getter can be added alongside
it without adding dataset-specific code to the generic comparison module.
Because every movie in a subject group uses the same ordered common contacts and lag
axis, the stimulus provider assigns all of them `subject-<id>-targets`.

## Estimand

For every subject with at least two loadable movies, the workflow uses the ordered
intersection of usable Laplacian contacts and the complete -2 to +2 second lag axis.
It fits a source RRR model on all words in one movie and evaluates every directed
source-to-target pair over five contiguous target folds. No significance Artifact is
loaded and no significance filtering is applied.

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
  --config configs/brain_treebank/cross_condition/gpt2.yml \
  --cross-condition
```

Each preparation step is resumable. Embeddings and epoch caches are skipped
when their complete artifacts already exist. The equivalent Make targets are
`download-brain-treebank`, `generate-brain-treebank-embeddings`, and
`precompute-cross-condition-epochs`.

The checked-in example runs one GPT-2 layer:

```bash
uv run python scripts/run_cross_condition_rrr.py \
  --config configs/brain_treebank/cross_condition/gpt2.yml
```

`per_model_config.layer` may be a scalar or list. Subjects are discovered from
Brain Treebank metadata unless `subjects` is set; an optional `movies` mapping can
restrict individual subjects. Every configured rank must be feasible for every
included subject/movie. The default low-rank SVD computes only the required leading
RRR target directions and has a fixed configurable seed; set `use_low_rank_svd: false`
only when an exact decomposition is computationally practical.

Alpha selection always runs because it is part of fitting the ridge readouts, but
its selected-alpha arrays are not saved by default. Set `log_alphas: true` to retain
them in the metric payloads; leave it false to keep only the resulting scores.

Preflight records missing caches, embeddings, transcripts, empty contact
intersections, and single-movie exclusions. A run with no analyzable subject saves
its exclusion diagnostics but is not marked complete. The current checkout contains
only legacy common-average Cars 2 caches, so the example requires Laplacian caches
and the remaining recordings in the execution environment.

## Artifact contract

The producer identity is `run_cross_condition_rrr/cross_condition/brain-treebank`.
Rank metrics are selected by subject, source item, target item, method, rank
(except the unreduced baseline), and target fold. Factor storage is not rank-expanded:
each subject/source movie has one full raw RRR `A` matrix, selected only by subject,
`source_item`, and `method: rrr`. Target manifests are selected by subject and
`target_item`. Rank-specific `Q_k` and `R_k` can be regenerated
by applying deterministic QR to `A[:, :k]`. No Q, R, B, intercept, PCA basis, or
rank-specific factor copies are persisted. Eligibility, exclusions, target manifests,
fold results, pair summaries, and skipped-comparison diagnostics are stored as
separate fields. The contract version participates in the variant digest, so legacy
`source_stimulus`/`target_stimulus` variants remain untouched and are not resumed.
