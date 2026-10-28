# Paper configurations

The checked-in configurations cover Podcast, Brain Treebank **Cars 2**, and
fMRI, using GPT-2, GPT-2 XL, GPT-J 6B, Llama 3 8B, RoBERTa Large,
randomly initialized GPT-2, and GloVe 300D.

Internal RRR uses the union of significant targets across the selected models
and layers (`significance_selection.mode: cross-model-layer`). The retained
YAML settings are unchanged from the result configurations.

| Workflow | Podcast | Cars 2 | fMRI |
| --- | --- | --- | --- |
| Significance | `significance/<model>/significance.yml` | `brain_treebank/cars_2/significance/<model>/significance.yml` | `cross_dataset/fmri/significance/<model>.yml` |
| Internal RRR | `rrr/<model>/cross_model_layer.yml` | `brain_treebank/cars_2/rrr/<model>/cross_model_layer.yml` | `fmri/within_fmri_cross_model_layer/<model>.yml` |
| Leave one subject out | `rrr/<model>/leave_one_subject_out.yml` | `brain_treebank/cars_2/rrr/<model>/leave_one_subject_out.yml` | `fmri/leave_one_subject_out/<model>.yml` |
| Transfer source fit | `rrr/<model>/full_fit.yml` | `brain_treebank/cars_2/rrr/<model>/full_fit.yml` | Use internal fMRI RRR |

Paths in this table are relative to `configs/`. Directed cross-dataset RRR
configs live in `cross_dataset/<target>/from_<source>/<model>.yml`, covering
all six directions. Cross-dataset iEEG sources require the `full_fit` artifacts;
fMRI sources require the internal fMRI artifacts.

Language decoding configs live in `language_decoding/internal/<dataset>.yml`
and `language_decoding/cross_dataset/<target>/from_<source>.yml`. These are
shared across models; `scripts/run_language_decoding_batch.py` selects the
model and layers and resolves the corresponding RRR artifacts.

Preprocessing configs are retained in `embeddings/`,
`brain_treebank/cars_2/generate_*.yml`, `precompute_epochs/`,
`brain_treebank/cars_2/precompute_epochs.yml`, and `language_features/`.
`fmri/generate_gpt2_xl.yml` also supplies the fMRI data template for
`scripts/fmri/generate_pipeline_embeddings.py`; pass the flat `embeddings/`
configs to that script to generate each model's selected layers.
GloVe loading is handled by the analysis data loaders.

Model keys used in filenames differ from some artifact identities: GPT-2 XL
uses `gpt2-xl-batch` in iEEG artifacts and GloVe uses `glove-300`.
The retained configs encode these identities explicitly.

The Makefile defaults to these internal RRR configs:

```sh
make significance DATASET=podcast MODELS="gpt2 glove"
make rrr DATASET=podcast MODELS="gpt2 glove"
make rrr DATASET=cars-2 RRR_MODE=leave_one_subject_out MODELS=gpt2
make rrr DATASET=fmri RRR_MODE=cross_dataset SOURCE_DATASET=podcast MODELS=gpt2
make language-decoding DATASET=podcast MODELS=gpt2
```

These commands submit Slurm jobs by default; add `CMD="uv run python"` to
run locally. Generate embeddings, neural epoch caches, and language features
first; complete significance before internal/source RRR, and source RRR before
cross-dataset RRR or decoding. The removed experiments have no Make targets.

Configs for Megamind, additional model families, within-layer-only significance
selection, temporal-window experiments, cross-condition comparisons, multi-view
RRR, subspace comparisons, reconstruction, and legacy single-layer analysis
examples have been removed.
