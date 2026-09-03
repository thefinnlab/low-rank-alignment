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

import gc
import argparse
import h5py
import os
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from src.utils.shared.ieeg import (
    load_raws,
    build_aligned_epoch_data,
    save_precomputed_epochs,
    get_precomputed_epochs_path,
)
from src.utils.shared.embeddings import load_embeddings
from src.core.config import EncodingConfig
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.rrr.cross_stimulus import discover_subject_movies
from src.utils.shared.config import (
    load_config,
    parse_override_args,
    save_effective_config,
)

_TARGET_SFREQ = 32.0
_TMIN = -2.0
_TMAX = 2.0


def provenance_config_filename(dataset_context) -> str:
    """Return a stimulus-scoped filename for an epoch cache's provenance."""
    rereference = getattr(dataset_context, "rereference", None)
    if getattr(dataset_context, "dataset", None) != "brain_treebank":
        return (
            "config.yml"
            if rereference in (None, "common_average")
            else f"config_ref-{rereference}.yml"
        )

    stimulus = dataset_context.stimulus_id
    suffix = (
        "" if rereference in (None, "common_average") else f"_ref-{rereference}"
    )
    return f"config_movie-{stimulus}{suffix}.yml"


def precompute_subject(raw, df_word, *, batch_size: int = 64):
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
        batch_size=batch_size,
    )
    n_lags = epoch_data.shape[-1]
    print(f"  epoch_data shape: {epoch_data.shape}")
    print(f"  n_words_kept={len(selection)}, n_lags={n_lags}")

    gc.collect()

    return epoch_data, selection, raw.ch_names, n_lags


def _cache_is_complete(path, *, dataset: str, stimulus: str, rereference: str) -> bool:
    """Return whether an epoch cache is structurally usable for this context."""
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as cache:
            required = {"epoch_data", "selection", "ch_names"}
            if not required.issubset(cache):
                return False
            data = cache["epoch_data"]
            if data.ndim != 3 or data.shape[0] != len(cache["selection"]):
                return False
            if data.shape[1] != len(cache["ch_names"]):
                return False
            attrs = cache.attrs
            if int(attrs.get("n_lags", -1)) != data.shape[2]:
                return False
            if float(attrs.get("sfreq", -1)) != _TARGET_SFREQ:
                return False
            if float(attrs.get("tmin", 999)) != _TMIN:
                return False
            if float(attrs.get("tmax", 999)) != _TMAX:
                return False
            cached_dataset = attrs.get("dataset", "")
            cached_stimulus = attrs.get("stimulus_id", "")
            cached_reference = attrs.get("rereference", "")
            values = (cached_dataset, cached_stimulus, cached_reference)
            values = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in values)
            return values == (dataset, stimulus, rereference)
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _trial_record(data_root: str, subject: int, trial: int) -> dict:
    adapter = get_dataset_adapter("brain_treebank")
    records = [
        record
        for record in adapter._metadata_records(data_root, subject)
        if int(record["trial_id"]) == int(trial)
    ]
    if not records:
        raise ValueError(f"No Brain Treebank metadata found for subject {subject}, trial {trial}.")
    return records[0]


def _per_trial_context(config: EncodingConfig, subject: int, trial: int):
    record = _trial_record(config.data_root, subject, trial)
    per_dataset_config = dict(config.per_dataset_config)
    per_dataset_config["movie"] = record["movie"]
    context = resolve_dataset_context(
        dataset=config.dataset,
        subject_ids=[subject],
        per_dataset_config=per_dataset_config,
        data_root=config.data_root,
    )
    # The adapter normally selects a movie record by stimulus. Replace that
    # subject entry with the exact requested trial so the raw loader cannot
    # accidentally use another recording if a movie was repeated.
    return record, per_dataset_config, replace(
        context, subject_metadata={subject: record}
    )


def precompute_trial(
    config: EncodingConfig,
    subject: int,
    trial: int,
    *,
    skip_existing=True,
    epoch_batch_size: int = 64,
):
    """Precompute one Brain Treebank subject/trial using the normal pipeline."""
    if config.dataset != "brain_treebank":
        raise ValueError("Trial-level epoch precomputation requires dataset=brain_treebank.")

    record, per_dataset_config, dataset_context = _per_trial_context(
        config, subject, trial
    )
    cache_path = get_precomputed_epochs_path(
        subject,
        config.data_root,
        dataset=config.dataset,
        per_dataset_config=per_dataset_config,
        dataset_context=dataset_context,
    )
    rereference = dataset_context.rereference
    if skip_existing and _cache_is_complete(
        cache_path,
        dataset=config.dataset,
        stimulus=record["movie"],
        rereference=rereference,
    ):
        print(
            f"Skipping subject {subject}, trial {trial}: cache exists ({cache_path})"
        )
        return False

    save_effective_config(
        config,
        os.path.dirname(cache_path),
        filename=provenance_config_filename(dataset_context),
    )
    per_model_config = dict(config.per_model_config)
    if isinstance(per_model_config.get("layer"), list):
        per_model_config["layer"] = per_model_config["layer"][0]

    _, df_word = load_embeddings(
        config.data_root,
        config.model,
        dataset=config.dataset,
        per_dataset_config=per_dataset_config,
        dataset_context=dataset_context,
        **per_model_config,
    )
    dataset_adapter = get_dataset_adapter(config.dataset)
    print(f"\n{'=' * 60}")
    print(f"Subject {subject}, trial {trial}, movie {record['movie']}")
    print(f"{'=' * 60}")
    raws = load_raws(
        [subject],
        config.data_root,
        use_high_gamma=True,
        do_drop_bads=True,
        dataset=config.dataset,
        per_dataset_config=per_dataset_config,
        dataset_context=dataset_context,
    )
    raw = raws[0]
    print(f"  Loaded raw: {len(raw.ch_names)} electrodes, {raw.info['sfreq']} Hz")
    subject_df_word = df_word
    if hasattr(dataset_adapter, "load_subject_word_table"):
        subject_df_word = dataset_adapter.load_subject_word_table(
            config.data_root,
            dataset_context,
            subject,
            df_word,
            drop_untimed_words=True,
        )
        print(f"  Subject-aligned words: {len(subject_df_word)}")
    epoch_data, selection, ch_names, _ = precompute_subject(
        raw, subject_df_word, batch_size=epoch_batch_size
    )
    save_precomputed_epochs(
        subject,
        config.data_root,
        epoch_data,
        selection,
        ch_names,
        sfreq=_TARGET_SFREQ,
        tmin=_TMIN,
        tmax=_TMAX,
        dataset=config.dataset,
        per_dataset_config=per_dataset_config,
        dataset_context=dataset_context,
    )
    del raws, raw, epoch_data
    gc.collect()
    return True


def _run_standard(config: EncodingConfig, epoch_batch_size: int):
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
    for sub_id in tqdm(subject_ids, desc="Subjects"):
        cache_path = get_precomputed_epochs_path(
            sub_id,
            config.data_root,
            dataset=config.dataset,
            per_dataset_config=config.per_dataset_config,
            dataset_context=dataset_context,
        )
        save_effective_config(
            config,
            os.path.dirname(cache_path),
            filename=provenance_config_filename(dataset_context),
        )
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
        dataset_adapter = get_dataset_adapter(config.dataset)
        if hasattr(dataset_adapter, "load_subject_word_table"):
            subject_df_word = dataset_adapter.load_subject_word_table(
                config.data_root,
                dataset_context,
                sub_id,
                df_word,
                drop_untimed_words=True,
            )
            print(f"  Subject-aligned words: {len(subject_df_word)}")

        epoch_data, selection, ch_names, _ = precompute_subject(
            raw, subject_df_word, batch_size=epoch_batch_size
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


def _run_cross_stimulus(config: EncodingConfig, epoch_batch_size: int):
    # The standalone cross-stimulus RRR config intentionally omits ``dataset``
    # because its own Config is Brain Treebank-specific. When that config is
    # reused here as an EncodingConfig, the dataclass default would otherwise
    # be ``podcast``.
    config.dataset = "brain_treebank"
    per_dataset_config = dict(config.per_dataset_config)
    per_dataset_config.setdefault("rereference", "laplacian")
    config.per_dataset_config = per_dataset_config
    trials = cross_stimulus_trials(config.data_root, config.subjects)
    subjects = sorted({subject for subject, _, _ in trials})
    print(f"Cross-stimulus epoch plan: {len(trials)} trials across subjects {subjects}")
    for subject, trial, movie in trials:
        print(f"  sub-{subject} trial-{trial:03d}: {movie}")
    completed = sum(
        precompute_trial(
            config,
            subject,
            trial,
            epoch_batch_size=epoch_batch_size,
        )
        for subject, trial, _ in trials
    )
    print(f"\nDone. Precomputed {completed}/{len(trials)} missing trial caches.")


def cross_stimulus_trials(
    data_root: str, subjects: list[int] | None = None, *, min_movies: int = 2
):
    """Return exact ``(subject, trial, movie)`` records for the cohort."""
    eligible = discover_subject_movies(data_root, subjects, min_movies=min_movies)
    adapter = get_dataset_adapter("brain_treebank")
    trials = []
    for subject in sorted(eligible):
        for record in adapter._metadata_records(data_root, subject):
            trials.append((subject, int(record["trial_id"]), record["movie"]))
    return trials


def main():
    parser = argparse.ArgumentParser(description="Precompute 32 Hz epochs cache")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--cross-stimulus", action="store_true")
    parser.add_argument("--subject", type=int)
    parser.add_argument("--trial", type=int)
    parser.add_argument(
        "--epoch-batch-size",
        type=int,
        default=64,
        help="Number of word epochs held in memory during extraction (default: 64).",
    )
    args, unknown = parser.parse_known_args()
    overrides = parse_override_args(unknown)
    config: EncodingConfig = load_config(args.config, overrides, EncodingConfig)

    if args.cross_stimulus and (args.subject is not None or args.trial is not None):
        parser.error("--cross-stimulus cannot be combined with --subject or --trial")
    if (args.subject is None) != (args.trial is None):
        parser.error("--subject and --trial must be provided together")
    if args.epoch_batch_size < 1:
        parser.error("--epoch-batch-size must be positive")
    if args.cross_stimulus:
        _run_cross_stimulus(config, args.epoch_batch_size)
    elif args.subject is not None:
        # Subject/trial selection is a Brain Treebank-specific interface.
        config.dataset = "brain_treebank"
        precompute_trial(
            config,
            args.subject,
            args.trial,
            epoch_batch_size=args.epoch_batch_size,
        )
    else:
        _run_standard(config, args.epoch_batch_size)


if __name__ == "__main__":
    main()
