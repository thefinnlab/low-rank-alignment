# Brain Treebank preprocessing

## Cross-stimulus setup

The cross-stimulus workflow needs the source HDF5 recordings, movie metadata,
timings, labels, localization, and transcripts. Download the shared archives and
only the trials for subjects who watched at least two movies with:

```bash
scripts/download_brain_treebank.sh --data-root data/brain-treebank
```

Use `--dry-run` to inspect the selected subjects and movies before downloading
the trial recordings. The downloader resumes partial ZIP files and reuses valid
downloads and extracted HDF5 files.

Generate the missing contextual embeddings for every distinct movie in that
cohort with:

```bash
uv run python scripts/generate_brain_treebank_embeddings.py \
  --config configs/brain_treebank/cars_2/generate_gpt2.yml
```

The command preserves the normal per-model/per-movie embedding layout and skips
movies whose features, transcript, metadata, and requested layers are complete.

Finally, precompute all missing Laplacian epoch caches using the existing
preprocessing implementation:

```bash
uv run python scripts/utilities/precompute_epochs.py \
  --config configs/brain_treebank/cross_stimulus/gpt2.yml \
  --cross-stimulus
```

This mode discovers eligible subjects, gathers every exact subject/trial pair,
and invokes the same per-trial preprocessing used by the standard command. A
single trial can be run with `--subject SUBJECT --trial TRIAL`.

Brain Treebank word-aligned epoch preprocessing computes a 70–200 Hz Hilbert envelope
from the source HDF5 voltage traces. It drops contacts listed in `corrupted_elec.json`,
resamples the continuous recording to 512 Hz, despikes each contact, rereferences the
voltage, applies the notch and band-pass filters, computes the envelope, and finally
extracts word-aligned epochs at 32 Hz.

## Rereferencing

Set the rereference mode alongside the movie in `per_dataset_config`:

```yaml
per_dataset_config:
  movie: cars-2
  rereference: laplacian
```

Supported values are:

- `common_average` (default): subtract the mean of all remaining sEEG contacts.
- `laplacian`: for contact `n` on a shaft, compute
  `V[n] - (V[n-1] + V[n+1]) / 2` before filtering and envelope extraction.

Laplacian preprocessing retains a contact only when both consecutively numbered
neighbors on the same shaft remain after corrupted-contact removal. Shaft endpoints,
contacts beside a missing or corrupted contact, and channels without a numbered-contact
label are excluded. `#` and `*` markers in Brain Treebank labels are ignored when
identifying the shaft and contact number, while the original channel name is retained
in the cache.

Common-average caches retain the legacy filename. Laplacian caches add
`_ref-laplacian` before `_ieeg.h5`, so both sets can coexist. Each new cache also
records its rereference mode, and the loader rejects a cache whose mode differs from
the requested configuration. Legacy Brain Treebank caches without this metadata are
treated as `common_average`. Precomputation provenance is stimulus-scoped within each
subject directory: common-average uses `config_movie-<movie>.yml`, and Laplacian uses
`config_movie-<movie>_ref-laplacian.yml`. This prevents two movies watched by the same
subject from attempting to reuse one provenance file.

Legacy provenance files can be migrated without changing epoch data. For example,
preview and then migrate the Cars 2 Laplacian files with:

```bash
uv run python scripts/utilities/migrate_brain_treebank_epoch_provenance.py \
  --movie cars-2 --rereference laplacian --dry-run
uv run python scripts/utilities/migrate_brain_treebank_epoch_provenance.py \
  --movie cars-2 --rereference laplacian
```

Use the same `per_dataset_config.rereference` value in precomputation and every
downstream significance, RRR, comparison, or language-analysis encoding configuration.
Laplacian significance Artifacts use the `rereference-laplacian` variant. RRR Artifacts
also include `rereference-laplacian` in their readable variant path; an RRR significance
selection must therefore reference the corresponding Laplacian significance variant.
