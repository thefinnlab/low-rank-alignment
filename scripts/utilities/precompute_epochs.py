"""
precompute_epochs.py — One-time preprocessing: resample each subject's high-gamma
signal to 32 Hz and save the result to disk.

After running this script, all other scripts can replace the slow
  load_raws(...) + build_word_epochs(...)
sequence with the fast
  load_precomputed_epochs(...) + epochs_from_cache(...)
which avoids re-running high-gamma preprocessing/resampling on every script
invocation and avoids materializing a full MNE Epochs array during precompute.

Saves per subject to the active dataset adapter's precomputed-epochs path.

The .npz contains:
  raw_data     — (n_elecs, n_total_samples) float32 at 32 Hz
  selection    — (n_words_kept,) int, word indices kept after duplicate/bounds filtering
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
    compute_epoch_cache_indices,
    save_precomputed_epochs,
)
from src.core.config import EncodingConfig
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import parse_args, load_config

_TARGET_SFREQ = 32.0
_TMIN = -2.0
_TMAX = 2.0


def precompute_subject(sub_id, raw, df_word):
    """Precompute and return cache components for one subject.

    Resamples the full continuous signal to 32 Hz, then computes valid
    word-epoch windows by direct time-to-sample conversion. This avoids
    materializing an n_words × n_electrodes × n_lags MNE Epochs array.

    Returns:
        raw_data     — (n_elecs, n_samples) float32
        selection    — int array
        epoch_starts — int array
        ch_names     — list of str
        n_lags       — int
    """
    print(f"  Resampling full continuous signal to {_TARGET_SFREQ} Hz...")
    raw_resampled = raw.copy().resample(
        _TARGET_SFREQ, npad="auto", method="fft", window="hamming"
    )
    raw_data = raw_resampled.get_data().astype(np.float32)
    print(f"  raw_data shape: {raw_data.shape}")

    selection, epoch_starts, n_lags = compute_epoch_cache_indices(
        raw_resampled, df_word, _TMIN, _TMAX, _TARGET_SFREQ
    )
    print(f"  n_words_kept={len(selection)}, n_lags={n_lags}")

    del raw_resampled
    gc.collect()

    return raw_data, selection, epoch_starts, raw.ch_names, n_lags


def main():
    args, overrides = parse_args("Precompute 32 Hz epochs cache")
    config: EncodingConfig = load_config(args.config, overrides, EncodingConfig)

    subject_ids = config.subjects or list(range(1, 10))
    dataset_context = resolve_dataset_context(
        dataset=config.dataset,
        subject_ids=subject_ids,
        per_dataset_config=config.per_dataset_config,
        data_root=config.data_root,
    )
    get_dataset_adapter(config.dataset).validate_subjects_share_stimulus(
        subject_ids, config.per_dataset_config, config.data_root
    )

    print(f"Precomputing epochs for subjects: {subject_ids}")
    print(f"  data_root: {config.data_root}")

    _, df_word = load_embeddings(
        config.data_root,
        config.model,
        dataset=config.dataset,
        per_dataset_config=config.per_dataset_config,
        dataset_context=dataset_context,
        **config.per_model_config,
    )
    dataset_adapter = get_dataset_adapter(config.dataset)

    for sub_id in tqdm(subject_ids, desc="Subjects"):
        print(f"\n{'='*60}")
        print(f"Subject {sub_id}")
        print(f"{'='*60}")

        raws = load_raws(
            [sub_id],
            config.data_root,
            use_high_gamma=True,
            do_drop_bads=True,
            dataset=config.dataset,
            per_dataset_config=config.per_dataset_config,
            dataset_context=dataset_context,
        )
        raw = raws[0]
        print(f"  Loaded raw: {len(raw.ch_names)} electrodes, {raw.info['sfreq']} Hz")

        subject_df_word = df_word
        if hasattr(dataset_adapter, "load_subject_word_table"):
            subject_df_word = dataset_adapter.load_subject_word_table(
                config.data_root,
                dataset_context,
                sub_id,
                df_word,
            )
            print(f"  Subject-aligned words: {len(subject_df_word)}")

        raw_data, selection, epoch_starts, ch_names, n_lags = precompute_subject(
            sub_id, raw, subject_df_word
        )
        save_precomputed_epochs(
            sub_id,
            config.data_root,
            raw_data,
            selection,
            epoch_starts,
            ch_names,
            n_lags,
            dataset=config.dataset,
            per_dataset_config=config.per_dataset_config,
            dataset_context=dataset_context,
        )

        del raws, raw, raw_data
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
