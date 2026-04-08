"""
precompute_epochs.py — One-time preprocessing: resample each subject's high-gamma
signal to 32 Hz and save the result to disk.

After running this script, all other scripts can replace the slow
  load_raws(...) + build_word_epochs(...)
sequence with the fast
  load_precomputed_epochs(...) + epochs_from_cache(...)
which avoids re-running MNE's FFT resample on every script invocation.

Saves per subject to:
  {data_root}/derivatives/ecogprep/sub-{id:02d}/ieeg/
      sub-{id:02d}_task-podcast_desc-epochs32hz_ieeg.npz

The .npz contains:
  raw_data     — (n_elecs, n_total_samples) float32 at 32 Hz
  selection    — (n_words_kept,) int, word indices kept by MNE epoching
  epoch_starts — (n_words_kept,) int, start sample in raw_data for each epoch
  n_lags       — scalar int, samples per epoch window
  ch_names     — (n_elecs,) electrode names
"""

import argparse
import gc

import numpy as np
from tqdm import tqdm

from src.utils import (
    load_raws,
    load_embeddings,
    build_word_epochs,
    save_precomputed_epochs,
)
from src.core.config import EncodingConfig
from src.utils.config_utils import parse_args, load_config

_TARGET_SFREQ = 32.0
_TMIN = -2.0
_TMAX = 2.0


def precompute_subject(sub_id, raw, df_word):
    """Precompute and return cache components for one subject.

    Runs build_word_epochs once on the original raw to get the canonical
    selection and n_lags, then resamples the full continuous signal to
    32 Hz and computes per-epoch start indices by direct time-to-sample
    conversion on the resampled object.

    Returns:
        raw_data     — (n_elecs, n_samples) float32
        selection    — int array
        epoch_starts — int array
        ch_names     — list of str
        n_lags       — int
    """
    import mne

    print(f"  Building reference epochs (run once to get selection / n_lags)...")
    epochs_ref, data_ref = build_word_epochs(raw, df_word, _TMIN, _TMAX, _TARGET_SFREQ)
    selection = epochs_ref.selection
    n_lags = data_ref.shape[2]
    print(f"  n_words_kept={len(selection)}, n_lags={n_lags}")

    print(f"  Resampling full continuous signal to {_TARGET_SFREQ} Hz...")
    raw_resampled = raw.copy().resample(
        _TARGET_SFREQ, npad="auto", method="fft", window="hamming"
    )
    raw_data = raw_resampled.get_data().astype(np.float32)
    print(f"  raw_data shape: {raw_data.shape}")

    tmin_samp = int(round(_TMIN * _TARGET_SFREQ))
    onset_times = df_word.iloc[selection]["start"].values
    onset_samples = raw_resampled.time_as_index(onset_times)
    epoch_starts = onset_samples + tmin_samp

    # Sanity check: all windows must be within bounds
    out_of_bounds = np.where(
        (epoch_starts < 0) | (epoch_starts + n_lags > raw_data.shape[1])
    )[0]
    if len(out_of_bounds) > 0:
        raise RuntimeError(
            f"Subject {sub_id}: {len(out_of_bounds)} epochs would fall outside "
            "the resampled raw array bounds.  This should not happen if selection "
            "matches the original build_word_epochs output."
        )

    del raw_resampled, epochs_ref, data_ref
    gc.collect()

    return raw_data, selection, epoch_starts, raw.ch_names, n_lags


def main():
    args, overrides = parse_args("Precompute 32 Hz epochs cache")
    config: EncodingConfig = load_config(args.config, overrides, EncodingConfig)

    subject_ids = config.subjects or list(range(1, 10))

    print(f"Precomputing epochs for subjects: {subject_ids}")
    print(f"  data_root: {config.data_root}")

    _, df_word = load_embeddings(
        config.data_root, config.model, **config.per_model_config
    )

    for sub_id in tqdm(subject_ids, desc="Subjects"):
        print(f"\n{'='*60}")
        print(f"Subject {sub_id}")
        print(f"{'='*60}")

        raws = load_raws(
            [sub_id], config.data_root, use_high_gamma=True, do_drop_bads=True
        )
        raw = raws[0]
        print(f"  Loaded raw: {len(raw.ch_names)} electrodes, {raw.info['sfreq']} Hz")

        raw_data, selection, epoch_starts, ch_names, n_lags = precompute_subject(
            sub_id, raw, df_word
        )
        save_precomputed_epochs(
            sub_id,
            config.data_root,
            raw_data,
            selection,
            epoch_starts,
            ch_names,
            n_lags,
        )

        del raws, raw, raw_data
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
