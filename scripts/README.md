# Retained scripts

Start with [the complete reproduction steps](../docs/reproduction.md) for the
ordered setup-to-figures workflow and required external inputs.

Run Python scripts from the repository root with `uv run python`. Install the
locked environment with `uv sync`. Configuration locations are listed in
[the config guide](../configs/README.md).

| Purpose | Scripts |
| --- | --- |
| Brain Treebank data and embeddings | `download_brain_treebank.sh`, `generate_brain_treebank_embeddings.py` |
| Local model snapshots | `utilities/cache_pipeline_models.py` |
| Podcast embeddings | `utilities/generate_embeddings.py`, `utilities/generate_random_embeddings.py` |
| fMRI embeddings | `fmri/generate_pipeline_embeddings.py`, `fmri/generate_embeddings.py` |
| Neural epoch caches | `utilities/precompute_epochs.py` |
| Shared language labels and resources | `utilities/generate_language_features.py` |
| fMRI sentence and intonation-unit boundaries | `utilities/generate_fmri_language_boundaries.py` |
| fMRI native, grammatical, lexical, and IU labels | `utilities/generate_fmri_interpretable_features.py`, `utilities/generate_fmri_iu_features.py` |
| Additional labels used by decoding | `utilities/generate_surprisal.py`, `utilities/generate_recurrence_features.py`, `utilities/generate_audio_intensity.py`, `utilities/generate_cars2_concreteness.py` |
| Significance filtering | `run_significance.py` |
| Internal, cross-dataset, LOSO, and transfer-source RRR | `run_rrr.py` |
| Language decoding | `run_language_decoding_batch.py`, `run_language_decoding.py`, `language-interpret/run_language_decoding.py` |
| Missing/incomplete result checks | `utilities/check_final_significance.py`, `utilities/check_rrr_results.py`, `utilities/check_language_decoding_results.py` |
| Source fits and directed transfer jobs with dependencies | `utilities/run_transfer_regeneration.py` |

Paths in the table are relative to `scripts/`. The decoding facade and its
implementation are both required by the batch runner and downstream result readers.

Prepare dataset files and local model/annotation resources before generating
embeddings, epoch caches, and labels. See [Brain Treebank setup](../docs/brain_treebank.md)
and [language-feature generation](../docs/language_feature_generation.md) for
preparation details. Complete significance filtering before internal RRR and
source fits. Complete source fits before directed RRR and decoding.

The Makefile exposes `significance`, `rrr`, and `language-decoding` with
`DATASET=podcast|cars-2|fmri` and the seven retained model keys. Commands submit
Slurm jobs by default; `CMD="uv run python"` executes locally. Paper figure
export lives in `notebooks/final_plots.py`:

```sh
uv run python notebooks/final_plots.py --output-dir figures
```

Exploratory analyses, legacy config generators and plot backfills, self-surprisal
experiments, and migration scripts are excluded from this submission surface.
Shared modules in `src/` retain their existing implementations.
