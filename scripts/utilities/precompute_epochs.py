"""
precompute_epochs.py — One-time preprocessing: build word-aligned 32 Hz epochs.

After running this script, all other scripts can replace the slow
  load_raws(...) + build_word_epochs(...)
sequence with the fast
  load_precomputed_epochs(...) + epochs_from_cache(...)
which avoids re-running high-gamma preprocessing/resampling on every invocation.

Saves per subject to the active dataset adapter's precomputed-epochs path.

The .h5 contains:
  epoch_data   — (n_words_kept, n_elecs, 128) float32 at 32 Hz
  selection    — (n_words_kept,) int, word indices kept after duplicate/bounds filtering
  n_lags       — scalar int, samples per epoch window
  ch_names     — (n_elecs,) electrode names
"""

import argparse
import gc

from tqdm import tqdm

from src.utils import (
    load_raws,
    load_embeddings,
    build_aligned_epoch_data,
    save_precomputed_epochs,
)
from src.core.config import EncodingConfig
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import parse_args, load_config

_TARGET_SFREQ = 32.0
_TMIN = -2.0
_TMAX = 2.0


def precompute_subject(raw, df_word):
    """Precompute and return cache components for one subject.

    Extracts word epochs on the source-rate grid, then resamples aligned
    batches to 32 Hz.

    Returns:
        epoch_data   — (n_words_kept, n_elecs, n_lags) float32
        selection    — int array
        ch_names     — list of str
        n_lags       — int
    """
    print(
        f"  Extracting epochs at {raw.info['sfreq']} Hz and resampling aligned "
        f"batches to {_TARGET_SFREQ} Hz..."
    )
    epoch_data, selection = build_aligned_epoch_data(
        raw,
        df_word,
        tmin=_TMIN,
        tmax=_TMAX,
        target_sfreq=_TARGET_SFREQ,
        batch_size=64,
    )
    n_lags = epoch_data.shape[-1]
    print(f"  epoch_data shape: {epoch_data.shape}")
    print(f"  n_words_kept={len(selection)}, n_lags={n_lags}")

    gc.collect()

    return epoch_data, selection, raw.ch_names, n_lags


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

    per_model_config = dict(config.per_model_config)
    if isinstance(per_model_config.get("layer"), list):
        per_model_config["layer"] = per_model_config["layer"][0]

    _, df_word = load_embeddings(
        config.data_root,
        config.model,
        dataset=config.dataset,
        per_dataset_config=config.per_dataset_config,
        dataset_context=dataset_context,
        **per_model_config,
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
                drop_untimed_words=True,
            )
            print(f"  Subject-aligned words: {len(subject_df_word)}")

        epoch_data, selection, ch_names, n_lags = precompute_subject(
            raw, subject_df_word
        )
        save_precomputed_epochs(
            sub_id,
            config.data_root,
            epoch_data,
            selection,
            ch_names,
            sfreq=_TARGET_SFREQ,
            tmin=_TMIN,
            tmax=_TMAX,
            dataset=config.dataset,
            per_dataset_config=config.per_dataset_config,
            dataset_context=dataset_context,
        )

        del raws, raw, epoch_data
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
