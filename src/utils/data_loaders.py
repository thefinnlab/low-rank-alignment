import os
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from src.core.registry import get_dataset_adapter, resolve_dataset_context


def _maybe_resolve_dataset_context(
    dataset: str,
    subject_ids: list[int] | None,
    per_dataset_config: dict | None,
    data_root: str,
    dataset_context,
):
    if dataset_context is not None:
        return dataset_context
    if dataset == "podcast" and not per_dataset_config:
        return None
    return resolve_dataset_context(
        dataset=dataset,
        subject_ids=subject_ids,
        per_dataset_config=per_dataset_config,
        data_root=data_root,
    )


def load_raws(
    subject_ids,
    data_root,
    use_high_gamma=False,
    do_drop_bads=False,
    signal_unit=None,
    per_subject_electrodes=None,
    channel_reg_ex=None,
    no_preload=False,
    dataset="podcast",
    per_dataset_config=None,
    dataset_context=None,
):
    """
    Loads raw iEEG data with YAML-configurable preprocessing.
    Dataset preprocessing, bad-channel removal, and unit scaling happen here.
    Analysis-rate resampling is intentionally deferred until after epoch alignment.
    """
    context = _maybe_resolve_dataset_context(
        dataset, subject_ids, per_dataset_config, data_root, dataset_context
    )
    adapter = get_dataset_adapter(dataset if context is None else context.dataset)
    if context is not None:
        adapter.validate_subjects_share_stimulus(
            subject_ids, per_dataset_config or {}, data_root
        )

    load_options = {
        "use_high_gamma": use_high_gamma,
        "do_drop_bads": do_drop_bads,
        "signal_unit": signal_unit,
        "per_subject_electrodes": per_subject_electrodes,
        "channel_reg_ex": channel_reg_ex,
        "no_preload": no_preload,
    }

    raws = []
    for sub_id in subject_ids:
        print(f"Loading raw data for subject {sub_id}...")
        raw = adapter.load_raw_recording(
            sub_id,
            data_root,
            context or adapter.resolve_analysis_context([], {}, data_root),
            load_options,
        )
        raws.append(raw)
    return raws


def read_subject_mapping(participant_map_file, delimiter="\t"):
    """
    Read and parse a participant mapping file to create subject ID mappings.

    This function reads a tab-delimited or custom-delimited file containing participant
    information and creates a mapping from NYU subject IDs to converted participant IDs.
    The participant IDs are expected to be in the format "sub-XX" where XX is extracted
    as an integer.

    Args:
        participant_map_file (str): Path to the participant mapping CSV/TSV file.
                                   Expected to contain columns 'nyu_id' and 'participant_id'.
        delimiter (str, optional): Delimiter used in the file. Defaults to "\\t" (tab).

    Returns:
        dict[int, int]: A dictionary mapping NYU subject IDs to converted participant IDs.
                       Keys are nyu_id values, values are integers extracted from
                       participant_id (e.g., "sub-05" -> 5).

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If required columns 'nyu_id' or 'participant_id' are missing.
        ValueError: If participant_id format is invalid (not "sub-XX" format).

    Example:
        >>> # File contains:
        >>> # nyu_id    participant_id
        >>> # 661       sub-05
        >>> # 717       sub-12
        >>> mapping = read_subject_mapping('participants.tsv')
        >>> print(mapping)
        {661: 5, 717: 12}
    """
    participant_info_df = pd.read_csv(participant_map_file, delimiter=delimiter)

    def participant_id_to_int(participant_id):
        parts = participant_id.split("-")
        if len(parts) != 2 or not parts[1]:
            raise ValueError(
                f"Invalid participant_id format: '{participant_id}'. Expected 'sub-XX'."
            )
        try:
            return int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid participant_id format: '{participant_id}'. Expected 'sub-XX' where XX is an integer."
            )

    subject_id_map = {
        x[1]["nyu_id"]: participant_id_to_int(x[1]["participant_id"])
        for x in participant_info_df.iterrows()
    }
    return subject_id_map


def read_electrode_file(file_path, subject_mapping: Optional[dict] = None):
    """
    Parse an electrode mapping CSV file to create a subject-to-electrodes mapping.

    This function reads a CSV file containing electrode information organized by subject
    and returns a dictionary mapping each subject ID to their list of electrode names.
    Each subject can have multiple electrodes, and the electrode order is preserved
    as it appears in the CSV file.

    Args:
        file_path (str): Path to the CSV file containing electrode data.
                        The CSV must have columns 'subject' (int) and 'elec' (str).
        subject_mapping (dict[int, int]): Optional mapping to go from the subject ID in the
                        file to the subject ID of our dataset (i.e. 798 -> 9)

    Returns:
        dict: A dictionary where keys are subject IDs (int) and values are lists
              of electrode names (str) for that subject. For example:
              {1: ['A1', 'A2', 'B1'], 2: ['C1', 'C2']}

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If required columns 'subject' or 'elec' are missing from the CSV.

    Example:
        >>> # CSV file contains:
        >>> # subject,elec
        >>> # 1,A1
        >>> # 1,A2
        >>> # 2,C1
        >>> result = read_electrode_file('electrodes.csv')
        >>> print(result)
        {1: ['A1', 'A2'], 2: ['C1']}
    """
    file_data = pd.read_csv(file_path)
    subjects, electrodes = file_data.subject, file_data.elec

    sub_elec_mapping = {}
    for subject, electrode in zip(subjects, electrodes):
        subject = int(subject)
        if subject_mapping is not None:
            subject = subject_mapping[subject]

        if subject not in sub_elec_mapping.keys():
            sub_elec_mapping[subject] = []

        sub_elec_mapping[subject].append(electrode)

    return sub_elec_mapping


def build_word_epochs(
    raw,
    df_word: pd.DataFrame,
    tmin: float = -2.0,
    tmax: float = 2.0,
    target_sfreq: float = 32.0,
) -> tuple:
    """
    Build and resample word-locked MNE epochs from a raw recording.

    Args:
        raw: MNE Raw object.
        df_word: DataFrame with a 'start' column (word onset times in seconds),
                 one row per word, indexed by word_idx.
        tmin: Epoch start relative to word onset (seconds).
        tmax: Epoch end relative to word onset (seconds).
        target_sfreq: Resample frequency after epoching (Hz).

    Returns:
        epochs: MNE Epochs object (resampled).
        data: np.ndarray of shape (n_words_kept, n_electrodes, n_lags).
    """
    import mne

    events = np.zeros((len(df_word), 3), dtype=int)
    events[:, 0] = (df_word["start"].values * raw.info["sfreq"]).astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        proj=False,
        preload=True,
        event_repeated="merge",
        verbose=False,
    )
    epochs = epochs.resample(
        sfreq=target_sfreq, npad="auto", method="fft", window="hamming"
    )
    return epochs, epochs.get_data(copy=False)


def build_aligned_epoch_data(
    raw,
    df_word: pd.DataFrame,
    tmin: float = -2.0,
    tmax: float = 2.0,
    target_sfreq: float = 32.0,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract source-rate word epochs, then resample each aligned epoch.

    The returned data uses a word-relative sampling grid instead of slicing all
    words from one globally downsampled continuous signal.
    """
    import mne

    if "start" not in df_word:
        raise ValueError("df_word must contain a 'start' column.")
    if tmax <= tmin:
        raise ValueError(f"tmax must be greater than tmin, got {tmin=} {tmax=}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    source_sfreq = float(raw.info["sfreq"])
    if "word_idx" in df_word.columns:
        original_indices = pd.to_numeric(
            df_word["word_idx"], errors="coerce"
        ).to_numpy()
    else:
        original_indices = np.arange(len(df_word), dtype=int)

    onset_times = pd.to_numeric(df_word["start"], errors="coerce").to_numpy()

    finite = np.isfinite(onset_times) & np.isfinite(original_indices)
    source_onsets = np.rint(onset_times[finite] * source_sfreq).astype(np.int64)
    finite_indices = original_indices[finite].astype(int)

    _, first_positions = np.unique(source_onsets, return_index=True)
    first_positions = np.sort(first_positions)
    source_onsets = source_onsets[first_positions]
    selection = finite_indices[first_positions]

    source_n_lags = int(round((tmax - tmin) * source_sfreq))
    target_n_lags = int(round((tmax - tmin) * target_sfreq))
    tmin_samples = int(round(tmin * source_sfreq))
    epoch_starts = source_onsets + tmin_samples
    in_bounds = (epoch_starts >= 0) & (
        epoch_starts + source_n_lags <= raw.n_times
    )
    epoch_starts = epoch_starts[in_bounds]
    selection = selection[in_bounds]

    epoch_data = np.empty(
        (len(selection), len(raw.ch_names), target_n_lags), dtype=np.float32
    )
    for batch_start in range(0, len(selection), batch_size):
        batch_stop = min(batch_start + batch_size, len(selection))
        source_batch = np.stack(
            [
                raw.get_data(start=int(start), stop=int(start + source_n_lags))
                for start in epoch_starts[batch_start:batch_stop]
            ],
            axis=0,
        )
        resampled = mne.filter.resample(
            source_batch,
            up=target_sfreq,
            down=source_sfreq,
            axis=-1,
            npad="auto",
            method="fft",
            window="hamming",
            verbose=False,
        )
        if resampled.shape[-1] != target_n_lags:
            raise RuntimeError(
                "Aligned epoch resampling produced an unexpected number of samples: "
                f"expected {target_n_lags}, got {resampled.shape[-1]}."
            )
        epoch_data[batch_start:batch_stop] = resampled.astype(
            np.float32, copy=False
        )

    return epoch_data, selection.astype(int)


def compute_epoch_cache_indices(
    raw,
    df_word: pd.DataFrame,
    tmin: float = -2.0,
    tmax: float = 2.0,
    target_sfreq: float = 32.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute cache indices for word-locked epochs without materializing epochs.

    Returns the word selection, start sample for each epoch in the resampled raw
    array, and the number of lag samples. Duplicate onset samples are dropped,
    keeping the first word at each sample.
    """
    if "start" not in df_word:
        raise ValueError("df_word must contain a 'start' column.")
    if tmax <= tmin:
        raise ValueError(f"tmax must be greater than tmin, got {tmin=} {tmax=}.")

    if "word_idx" in df_word.columns:
        original_indices = pd.to_numeric(df_word["word_idx"], errors="coerce").to_numpy()
    else:
        original_indices = np.arange(len(df_word), dtype=int)

    source_sfreq = float(raw.info["sfreq"])
    starts = pd.to_numeric(df_word["start"], errors="coerce").to_numpy()

    finite = np.isfinite(starts) & np.isfinite(original_indices)
    source_onsets = np.floor(starts[finite] * source_sfreq).astype(np.int64)
    finite_indices = original_indices[finite].astype(int)
    finite_starts = starts[finite]

    _, first_positions = np.unique(source_onsets, return_index=True)
    first_positions = np.sort(first_positions)
    selection = finite_indices[first_positions]
    onset_times = finite_starts[first_positions]

    n_lags = int(round((tmax - tmin) * target_sfreq))
    if n_lags <= 0:
        raise ValueError(f"Computed non-positive n_lags={n_lags}.")

    tmin_samp = int(round(tmin * target_sfreq))
    onset_samples = raw.time_as_index(onset_times)
    epoch_starts = onset_samples + tmin_samp

    in_bounds = (epoch_starts >= 0) & (epoch_starts + n_lags <= raw.n_times)
    return (
        selection[in_bounds].astype(int),
        epoch_starts[in_bounds].astype(int),
        n_lags,
    )


def load_data_from_cache(
    subs_to_load,
    data_root,
    offset,
    embeddings,
    sig_electrodes=None,
    valid_rows=None,
    dataset="podcast",
    per_dataset_config=None,
    dataset_context=None,
    shared_selection=None,
):
    import gc
    from src.utils.encoding import apply_offset

    context = _maybe_resolve_dataset_context(
        dataset, subs_to_load, per_dataset_config, data_root, dataset_context
    )
    Y_cols = []
    X_ref = None
    ref_selection = None
    elec_names = []
    ref_stimulus_id = getattr(context, "stimulus_id", None)
    for sub_id in subs_to_load:
        if context is None:
            cache = load_precomputed_epochs(sub_id, data_root)
        else:
            cache = load_precomputed_epochs(
                sub_id,
                data_root,
                dataset=context.dataset,
                per_dataset_config=per_dataset_config,
                dataset_context=context,
            )
        cache_stimulus_id = cache.get("stimulus_id")
        if ref_stimulus_id is None:
            ref_stimulus_id = cache_stimulus_id
        elif cache_stimulus_id != ref_stimulus_id:
            raise ValueError(
                f"Cached subjects must share one stimulus. "
                f"Expected '{ref_stimulus_id}', got '{cache_stimulus_id}' for subject {sub_id}."
            )
        if shared_selection is not None:
            cache = filter_epoch_cache_to_selection(cache, shared_selection)
        selection = cache["selection"]
        if ref_selection is None:
            ref_selection = selection
        elif not np.array_equal(ref_selection, selection):
            raise ValueError(
                "Cached subjects must have identical word selections before joint loading. "
                f"Subject {sub_id} differs from the first loaded subject; regenerate caches "
                "with the same transcript/alignment policy or train subjects separately."
            )
        selected_elec_names = sig_electrodes[sub_id] if sig_electrodes else None
        epoch_data = cache.get("epoch_data")
        if selected_elec_names is not None and hasattr(epoch_data, "read_flat"):
            ch_names = cache["ch_names"]
            data = epoch_data.read_flat(
                electrode_indices=[ch_names.index(e) for e in selected_elec_names]
            )
        else:
            data = epochs_from_cache(cache, elec_names=selected_elec_names)
        if sig_electrodes:
            elec_names.append(selected_elec_names)
        else:
            elec_names.append(cache["ch_names"])
        X_arr, Y_sub = apply_offset(
            embeddings[selection],
            data,
            offset,
            valid_rows=valid_rows[selection] if valid_rows is not None else None,
        )
        if X_ref is None:
            X_ref = X_arr

        Y_cols.append(Y_sub)
        del cache, data, Y_sub
    gc.collect()
    return (
        X_ref,
        np.concatenate(Y_cols, axis=1).astype(np.float32, copy=False),
        ref_selection,
        elec_names,
    )


def build_XY_from_subjects(
    subject_ids: list,
    enc,
    sig_electrodes: dict,
    aligned_embeddings: "np.ndarray",
    valid_rows: Optional[np.ndarray] = None,
    shared_selection: Optional[np.ndarray] = None,
) -> tuple:
    """Load neural data for multiple subjects and stack into X and Y arrays.

    Uses precomputed epoch caches (see scripts/precompute_epochs.py). For each
    subject with significant electrodes, slices epoch windows from the cache and
    concatenates neural responses column-wise to form Y. X is taken from the first
    subject (all subjects share the same embedding sequence after alignment).

    Args:
        subject_ids: Ordered list of subject IDs to load.
        enc: EncodingConfig with data_root, offset.
        sig_electrodes: Dict mapping subject_id → list of significant electrode names.
            Subjects absent from this dict are skipped.
        aligned_embeddings: (n_words_total, n_embed) embedding array.
        df_word: Unused; retained for call-site compatibility.

    Returns:
        X: (n_words, n_embed) embedding matrix (from first subject with data).
        Y: (n_words, n_targets) concatenated neural response matrix, float32.
        selection: (n_words_pre_offset,) integer array of word indices into
            df_word (from the first subject's epochs).

    Raises:
        ValueError: If no subjects have significant electrodes.
    """

    subs_to_load = [s for s in subject_ids if s in sig_electrodes and sig_electrodes[s]]
    if not subs_to_load:
        raise ValueError(
            f"No subjects with significant electrodes found in subject_ids={subject_ids}."
        )

    dataset = getattr(enc, "dataset", "podcast")
    per_dataset_config = getattr(enc, "per_dataset_config", {})
    dataset_context = _maybe_resolve_dataset_context(
        dataset, subs_to_load, per_dataset_config, enc.data_root, None
    )

    return load_data_from_cache(
        subs_to_load,
        enc.data_root,
        enc.offset,
        aligned_embeddings,
        sig_electrodes,
        valid_rows=valid_rows,
        dataset=dataset,
        per_dataset_config=per_dataset_config,
        dataset_context=dataset_context,
        shared_selection=shared_selection,
    )


def load_sig_electrodes(
    output_dir,
    subject_ids,
    sig_dir=None,
    dataset="podcast",
    per_dataset_config=None,
    dataset_context=None,
    data_root="data/",
):
    """Load significance CSV and build a subject → electrode list mapping.

    Args:
        output_dir: Root results directory (used as default location for the CSV).
        subject_ids: Subject IDs to include.
        sig_dir: Optional override directory to read ``all_subject_sig.csv`` from.
            When provided, the CSV is loaded from ``sig_dir`` instead of ``output_dir``.

    Returns:
        sig_electrodes: dict mapping sub_id → list of significant electrode names.
            Only subjects with at least one significant electrode are included.
        sig_df: The loaded ``all_subject_sig.csv`` DataFrame.

    Raises:
        FileNotFoundError: If the significance CSV is missing.
        ValueError: If no significant electrodes are found for any subject.
    """
    context = _maybe_resolve_dataset_context(
        dataset, subject_ids, per_dataset_config, data_root, dataset_context
    )
    adapter = get_dataset_adapter(dataset if context is None else context.dataset)
    resolved_context = context or adapter.resolve_analysis_context([], {}, data_root)
    csv_dir = sig_dir if sig_dir is not None else output_dir
    sig_csv = os.path.join(csv_dir, "all_subject_sig.csv")
    if not os.path.exists(sig_csv):
        raise FileNotFoundError(
            f"all_subject_sig.csv not found at {sig_csv}. "
            "Run run_significance.py first."
        )
    sig_electrodes, sig_df = adapter.load_significance(
        output_dir,
        subject_ids,
        resolved_context,
        sig_dir=sig_dir,
    )
    print(f"Loaded significance CSV: {len(sig_df):,} significant electrodes")

    if not sig_electrodes:
        raise ValueError(
            f"No significant electrodes found for subjects {subject_ids} in {sig_csv}."
        )

    return sig_electrodes, sig_df


def load_fold_indices(output_dir, subject_ids, sig_electrodes):
    """Load cross-validation fold indices and lags from the first subject's encoding results.

    Args:
        output_dir: Root results directory.
        subject_ids: Ordered subject IDs.
        sig_electrodes: dict mapping sub_id → electrode list (used to find first
            subject with data).

    Returns:
        provided_idxs: list of (train_idx, test_idx) tuples.
        lags: lag array from encoding results, or None.
    """
    import os
    from src.utils.encoding import load_encoding_results

    subs_with_sig = [s for s in subject_ids if s in sig_electrodes]
    first_sub = subs_with_sig[0]
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{first_sub}", "encoding_results.joblib"
        )
    )
    ref_test_idxs = enc_results["test_idxs"]
    lags = enc_results.get("lags", None)

    # We need n_words to build train indices, but we don't have X yet.
    # Use the encoding results' own data shape to infer it.
    # The test_idxs partition all words, so we can infer n_words from them.
    n_words = max(idx.max() for idx in ref_test_idxs) + 1

    provided_idxs = []
    for test_idx in ref_test_idxs:
        train_mask = np.ones(n_words, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        provided_idxs.append((train_idx, test_idx))

    return provided_idxs, lags


def get_precomputed_epochs_path(
    sub_id,
    data_root,
    dataset="podcast",
    per_dataset_config=None,
    dataset_context=None,
):
    """Return the canonical path for a subject's precomputed 32 Hz epoch cache."""
    context = _maybe_resolve_dataset_context(
        dataset, [sub_id], per_dataset_config, data_root, dataset_context
    )
    adapter = get_dataset_adapter(dataset if context is None else context.dataset)
    return adapter.get_precomputed_epochs_path(
        sub_id,
        data_root,
        context or adapter.resolve_analysis_context([], {}, data_root),
    )


def save_precomputed_epochs(
    sub_id,
    data_root,
    epoch_data,
    selection,
    ch_names,
    sfreq=32.0,
    tmin=-2.0,
    tmax=2.0,
    dataset="podcast",
    per_dataset_config=None,
    dataset_context=None,
):
    """Save precomputed epoch cache to disk.

    Args:
        sub_id: Integer subject ID.
        data_root: BIDS dataset root directory.
        epoch_data: np.ndarray shaped (n_words, n_elecs, n_lags), float32.
        selection: np.ndarray of word indices kept by MNE epoching.
        ch_names: List of electrode name strings.
        sfreq: Epoch sampling frequency.
        tmin: Epoch start relative to word onset, in seconds.
        tmax: Exclusive epoch end relative to word onset, in seconds.
    """
    epoch_data = np.asarray(epoch_data, dtype=np.float32)
    selection = np.asarray(selection, dtype=int)
    if epoch_data.ndim != 3:
        raise ValueError(
            f"epoch_data must have shape (words, electrodes, lags), got {epoch_data.shape}."
        )
    if epoch_data.shape[0] != len(selection):
        raise ValueError(
            "epoch_data word count must match selection length, got "
            f"{epoch_data.shape[0]} and {len(selection)}."
        )
    if epoch_data.shape[1] != len(ch_names):
        raise ValueError(
            "epoch_data electrode count must match ch_names length, got "
            f"{epoch_data.shape[1]} and {len(ch_names)}."
        )
    context = _maybe_resolve_dataset_context(
        dataset, [sub_id], per_dataset_config, data_root, dataset_context
    )
    path = get_precomputed_epochs_path(
        sub_id,
        data_root,
        dataset=dataset,
        per_dataset_config=per_dataset_config,
        dataset_context=context,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_words, n_elecs, n_lags = epoch_data.shape
    chunks = (min(n_words, 128), min(n_elecs, 16), n_lags)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset(
            "epoch_data",
            data=epoch_data,
            dtype="float32",
            chunks=chunks,
            compression="lzf",
        )
        h5f.create_dataset("selection", data=selection)
        h5f.create_dataset(
            "ch_names",
            data=np.asarray(ch_names, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        h5f.attrs["n_lags"] = int(n_lags)
        h5f.attrs["sfreq"] = float(sfreq)
        h5f.attrs["tmin"] = float(tmin)
        h5f.attrs["tmax"] = float(tmax)
        h5f.attrs["dataset"] = context.dataset if context is not None else dataset
        h5f.attrs["stimulus_id"] = getattr(
            context, "stimulus_id", "podcast" if dataset == "podcast" else ""
        )
        h5f.attrs["stimulus_label"] = getattr(
            context, "stimulus_label", "podcast" if dataset == "podcast" else ""
        )
    print(f"  Saved precomputed epochs: {path}")


class _LazyH5EpochData:
    """Array-like handle for epoch data stored in an HDF5 cache."""

    def __init__(self, path: str, row_indices=None):
        self.path = path
        self.row_indices = (
            None if row_indices is None else np.asarray(row_indices, dtype=int)
        )
        with h5py.File(self.path, "r") as h5f:
            dset = h5f["epoch_data"]
            self._shape = tuple(dset.shape)
            self.dtype = dset.dtype

    @property
    def shape(self):
        if self.row_indices is None:
            return self._shape
        return (len(self.row_indices), self._shape[1], self._shape[2])

    def __array__(self, dtype=None, copy=None):
        data = self.read()
        if copy is None:
            return np.asarray(data, dtype=dtype)
        return np.array(data, dtype=dtype, copy=copy)

    @staticmethod
    def _sorted_indexer(indices):
        indices = np.asarray(indices, dtype=int)
        order = np.argsort(indices)
        return indices[order], np.argsort(order)

    def _read_rows_for_electrode(self, dset, elec_idx: int):
        if self.row_indices is None:
            return dset[:, elec_idx, :]
        if len(self.row_indices) == 0:
            return np.empty((0, dset.shape[2]), dtype=dset.dtype)
        sorted_rows, restore = self._sorted_indexer(self.row_indices)
        return dset[sorted_rows, elec_idx, :][restore]

    def read(self, electrode_indices=None):
        with h5py.File(self.path, "r") as h5f:
            dset = h5f["epoch_data"]
            if electrode_indices is None:
                if self.row_indices is None:
                    return dset[:].astype(np.float32, copy=False)
                if len(self.row_indices) == 0:
                    return np.empty((0, dset.shape[1], dset.shape[2]), dtype=np.float32)
                sorted_rows, restore = self._sorted_indexer(self.row_indices)
                return dset[sorted_rows, :, :][restore].astype(np.float32, copy=False)

            electrode_indices = np.asarray(electrode_indices, dtype=int)
            out = np.empty(
                (self.shape[0], len(electrode_indices), dset.shape[2]),
                dtype=np.float32,
            )
            for out_idx, elec_idx in enumerate(electrode_indices):
                out[:, out_idx, :] = self._read_rows_for_electrode(dset, int(elec_idx))
            return out

    def read_flat(self, electrode_indices=None):
        with h5py.File(self.path, "r") as h5f:
            dset = h5f["epoch_data"]
            if electrode_indices is None:
                electrode_indices = np.arange(dset.shape[1])
            else:
                electrode_indices = np.asarray(electrode_indices, dtype=int)

            n_rows = self.shape[0]
            n_lags = dset.shape[2]
            out = np.empty((n_rows, len(electrode_indices) * n_lags), dtype=np.float32)
            for out_idx, elec_idx in enumerate(electrode_indices):
                out[:, out_idx * n_lags : (out_idx + 1) * n_lags] = (
                    self._read_rows_for_electrode(dset, int(elec_idx))
                )
            return out

    def with_rows(self, row_indices):
        return _LazyH5EpochData(self.path, row_indices=row_indices)


def load_precomputed_epochs(
    sub_id,
    data_root,
    dataset="podcast",
    per_dataset_config=None,
    dataset_context=None,
):
    """Load precomputed epoch cache from disk.

    Args:
        sub_id: Integer subject ID.
        data_root: BIDS dataset root directory.

    Returns:
        dict with keys:
            epoch_data — lazy HDF5 handle or legacy float32 array
            selection — (n_words_kept,) int array of kept word indices
            n_lags    — int, samples per epoch window
            ch_names  — list of electrode name strings

    Raises:
        FileNotFoundError if cache file does not exist.
    """
    context = _maybe_resolve_dataset_context(
        dataset, [sub_id], per_dataset_config, data_root, dataset_context
    )
    path = get_precomputed_epochs_path(
        sub_id,
        data_root,
        dataset=dataset,
        per_dataset_config=per_dataset_config,
        dataset_context=context,
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Precomputed HDF5 epochs not found: {path}. "
            "Run scripts/utilities/precompute_epochs.py first."
        )

    with h5py.File(path, "r") as cache:
        if "epoch_data" not in cache:
            raise ValueError(
                f"Precomputed epoch cache missing 'epoch_data': {path}. Regenerate it "
                "with scripts/utilities/precompute_epochs.py before running analyses."
            )
        attrs = cache.attrs
        ch_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in cache["ch_names"][:]
        ]
        return {
            "epoch_data": _LazyH5EpochData(path),
            "selection": cache["selection"][:],
            "n_lags": int(attrs["n_lags"]),
            "sfreq": float(attrs["sfreq"]),
            "tmin": float(attrs["tmin"]),
            "tmax": float(attrs["tmax"]),
            "ch_names": ch_names,
            "dataset": str(attrs.get("dataset", dataset)),
            "stimulus_id": str(
                attrs.get("stimulus_id", getattr(context, "stimulus_id", "podcast"))
            ),
            "stimulus_label": str(
                attrs.get(
                    "stimulus_label", getattr(context, "stimulus_label", "podcast")
                )
            ),
            "path": path,
        }


def compute_shared_word_selection(
    subject_ids,
    data_root,
    dataset="podcast",
    per_dataset_config=None,
    dataset_context=None,
):
    """Return the sorted intersection of cached word selections across subjects."""
    selections = []
    for sub_id in subject_ids:
        cache = load_precomputed_epochs(
            sub_id,
            data_root,
            dataset=dataset,
            per_dataset_config=per_dataset_config,
            dataset_context=dataset_context,
        )
        selections.append(np.asarray(cache["selection"], dtype=int))
    if not selections:
        raise ValueError("Cannot compute a shared word selection for no subjects.")
    shared = selections[0]
    for selection in selections[1:]:
        shared = np.intersect1d(shared, selection, assume_unique=False)
    if shared.size == 0:
        raise ValueError(
            f"No shared cached word selections found across subjects {list(subject_ids)}."
        )
    return shared.astype(int, copy=False)


def filter_epoch_cache_to_selection(cache: dict, shared_selection) -> dict:
    """Return a cache copy restricted to ``shared_selection`` in that order."""
    shared_selection = np.asarray(shared_selection, dtype=int)
    cache_selection = np.asarray(cache["selection"], dtype=int)
    row_by_word = {int(word_idx): row_i for row_i, word_idx in enumerate(cache_selection)}
    missing = [
        int(word_idx)
        for word_idx in shared_selection
        if int(word_idx) not in row_by_word
    ]
    if missing:
        preview = ", ".join(str(x) for x in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            "Requested shared word selection contains word indices absent from "
            f"subject cache: {preview}{suffix}"
        )
    row_idxs = np.array([row_by_word[int(word_idx)] for word_idx in shared_selection])
    filtered = dict(cache)
    epoch_data = cache["epoch_data"]
    if hasattr(epoch_data, "with_rows"):
        filtered["epoch_data"] = epoch_data.with_rows(row_idxs)
    else:
        filtered["epoch_data"] = np.asarray(epoch_data)[row_idxs]
    filtered["selection"] = shared_selection
    return filtered


def selection_after_offset(epoch_selection, offset: int, valid_rows=None) -> np.ndarray:
    """Return word indices represented by model rows after offset and filtering."""
    selection = np.asarray(epoch_selection, dtype=int)
    if offset < 0:
        selection = selection[:offset]
        if valid_rows is not None:
            valid_rows = np.asarray(valid_rows)[:offset]
    elif offset > 0:
        selection = selection[offset:]
        if valid_rows is not None:
            valid_rows = np.asarray(valid_rows)[offset:]
    if valid_rows is not None:
        selection = selection[np.asarray(valid_rows, dtype=bool)]
    return selection.astype(int, copy=False)


def build_word_selection_metadata(
    mode: str,
    subjects,
    epoch_selection,
    offset: int,
    valid_rows=None,
) -> dict:
    """Build row-provenance metadata stored in encoding_results.joblib."""
    epoch_selection = np.asarray(epoch_selection, dtype=int)
    analysis_selection = selection_after_offset(
        epoch_selection,
        int(offset),
        valid_rows=valid_rows,
    )
    return {
        "mode": mode,
        "subjects": [int(s) for s in subjects],
        "epoch_selection": epoch_selection,
        "analysis_selection": analysis_selection,
        "offset": int(offset),
        "valid_rows_applied": valid_rows is not None,
    }


def epochs_from_cache(cache, elec_names=None):
    """Extract epoch data array from a precomputed cache.

    Returns the aligned epoch tensor stored in the cache.

    Args:
        cache: dict returned by load_precomputed_epochs.
        elec_names: Optional list of electrode names to select.  If None all
            electrodes are returned.

    Returns:
        np.ndarray of shape (n_words_kept, n_elecs, n_lags), float32.
    """
    if elec_names is not None:
        ch_names = cache["ch_names"]
        idx = [ch_names.index(e) for e in elec_names]
    else:
        idx = None

    epoch_data = cache["epoch_data"]
    if hasattr(epoch_data, "read"):
        return epoch_data.read(electrode_indices=idx)
    if idx is not None:
        epoch_data = epoch_data[:, idx, :]

    return np.asarray(epoch_data, dtype=np.float32)


def get_arbitrary_embeddings(X, epochs_or_selection, df_word, seed=None):
    """
    Assign unique random embeddings per word type, preserving per-word consistency.

    For each unique word in the selection, randomly picks one of the n_words rows
    of X (without replacement) to use as the arbitrary embedding for that word type.
    All occurrences of the same word receive the same row of X.

    Args:
        X (np.ndarray): Embedding array of shape (n_words, n_features).
        epochs_or_selection: Either an MNE Epochs-like object with a `.selection`
            attribute, or a numpy integer array of word indices into df_word.
        df_word (pd.DataFrame): DataFrame with a 'word' column; rows indexed by position.
        seed (int | None): Random seed for reproducibility. When None, uses the
            global numpy random state.

    Returns:
        np.ndarray: Array of shape (n_words, n_features) with arbitrary embeddings
                    assigned consistently per word type.
    """
    if isinstance(epochs_or_selection, np.ndarray):
        selection = epochs_or_selection
    else:
        selection = epochs_or_selection.selection
    words = df_word.iloc[selection].reset_index()["word"].apply(lambda x: x.lower())
    unique_words = words.unique()
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    X_arb = np.zeros_like(X)
    for word_idx, word in enumerate(unique_words):
        # Use line for reusing actual embeddings
        X_arb[words[words == word].index] = rng.uniform(size=(1, X.shape[1]))

    return X_arb
