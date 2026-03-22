from typing import Optional

import numpy as np
import pandas as pd


def load_raws(
    subject_ids,
    data_root,
    use_high_gamma=False,
    target_sr=None,
    do_drop_bads=False,
    signal_unit=None,
    per_subject_electrodes=None,
    channel_reg_ex=None,
):
    """
    Loads raw iEEG data with YAML-configurable preprocessing.
    All transformations (Resampling, Drop Bad, Scaling) happen here directly on the Raw object.
    """
    import mne
    import os
    from mne_bids import BIDSPath

    raws = []

    for sub_id in subject_ids:
        if use_high_gamma:
            root_path = os.path.join(data_root, "derivatives/ecogprep")
            description = "highgamma"
            print(f"Loading High Gamma data for subject {sub_id}...")
        else:
            root_path = os.path.join(data_root, "derivatives/ecogprep")
            description = None
            print(f"Loading Raw data for subject {sub_id}...")

        file_path = BIDSPath(
            root=root_path,
            subject=f"{sub_id:02}",
            task="podcast",
            datatype="ieeg",
            description=description,
            suffix="ieeg",
            extension=".fif",
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

        if target_sr is not None:
            if int(raw.info["sfreq"]) != int(target_sr):
                print(
                    f"  - [Config] Resampling {raw.info['sfreq']}Hz -> {target_sr}Hz"
                )
                raw.resample(target_sr)

        if do_drop_bads:
            try:
                ch_path = file_path.copy()
                ch_path.update(suffix="channels", extension=".tsv")
                tsv_path = str(ch_path)

                if not os.path.exists(tsv_path) and getattr(
                    file_path, "description", None
                ):
                    ch_path2 = file_path.copy()
                    ch_path2.update(
                        description=None, suffix="channels", extension=".tsv"
                    )
                    tsv_path = str(ch_path2)

                if not os.path.exists(tsv_path) and "derivatives" in str(file_path):
                    raw_bids_path = BIDSPath(
                        root=data_root,
                        subject=f"{sub_id:02}",
                        task="podcast",
                        datatype="ieeg",
                        suffix="channels",
                        extension=".tsv",
                    )
                    tsv_path = str(raw_bids_path)

                if os.path.exists(tsv_path):
                    df_ch = pd.read_csv(tsv_path, sep="\t")
                    if "status" in df_ch.columns and "name" in df_ch.columns:
                        bad_channels = df_ch[df_ch["status"] == "bad"]["name"].tolist()
                        if bad_channels:
                            bad_channels = [
                                ch for ch in bad_channels if ch in raw.ch_names
                            ]
                            raw.info["bads"].extend(bad_channels)
                            print(
                                f"  - [Config] Dropping {len(bad_channels)} bad channels"
                            )
                            raw.drop_channels(bad_channels)
            except Exception as e:
                print(f"Warning: Failed to process bad channels: {e}")

        if signal_unit == "uV":
            print(f"  - [Config] Scaling data: Volt -> MicroVolt (x 1e6)")
            raw.apply_function(lambda x: x * 1e6, channel_wise=False)

        if per_subject_electrodes:
            subject_electrode_names = per_subject_electrodes[sub_id]
            picks = [ch for ch in subject_electrode_names if ch in raw.ch_names]
            raw = raw.pick(picks)
        elif channel_reg_ex:
            picks = mne.pick_channels_regexp(raw.ch_names, channel_reg_ex)
            raw = raw.pick(picks)

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


def read_electrode_file(
    file_path, subject_mapping: Optional[dict] = None
):
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


def build_XY_from_subjects(
    subject_ids: list,
    enc,
    sig_electrodes: dict,
    aligned_embeddings: "np.ndarray",
    df_word: "pd.DataFrame",
) -> tuple:
    """Load neural data for multiple subjects and stack into X and Y arrays.

    For each subject with significant electrodes, builds word-locked epochs and
    concatenates neural responses column-wise to form Y. X is taken from the first
    subject (all subjects share the same embedding sequence after alignment).

    Args:
        subject_ids: Ordered list of subject IDs to load.
        enc: EncodingConfig with data_root, offset.
        sig_electrodes: Dict mapping subject_id → list of significant electrode names.
            Subjects absent from this dict are skipped.
        aligned_embeddings: (n_words_total, n_embed) embedding array.
        df_word: DataFrame with word onset times ('start' column).

    Returns:
        X: (n_words, n_embed) embedding matrix (from first subject with data).
        Y: (n_words, n_targets) concatenated neural response matrix, float32.

    Raises:
        ValueError: If no subjects have significant electrodes.
    """
    import gc
    from utils.encoding import apply_offset

    subs_to_load = [s for s in subject_ids if s in sig_electrodes and sig_electrodes[s]]
    if not subs_to_load:
        raise ValueError(
            f"No subjects with significant electrodes found in subject_ids={subject_ids}."
        )

    raws = load_raws(
        subs_to_load, enc.data_root, use_high_gamma=True,
        per_subject_electrodes={s: sig_electrodes[s] for s in subs_to_load},
    )

    Y_cols = []
    X_ref = None
    for raw, sub_id in zip(raws, subs_to_load):
        epochs, data = build_word_epochs(raw, df_word)
        X_arr, Y_sub = apply_offset(aligned_embeddings[epochs.selection], data, enc.offset)
        if X_ref is None:
            X_ref = X_arr
        Y_cols.append(Y_sub)
    del raws
    gc.collect()

    return X_ref, np.concatenate(Y_cols, axis=1).astype(np.float32)


def load_sig_electrodes(output_dir, subject_ids):
    """Load significance CSV and build a subject → electrode list mapping.

    Args:
        output_dir: Root results directory containing ``all_subject_sig.csv``.
        subject_ids: Subject IDs to include.

    Returns:
        sig_electrodes: dict mapping sub_id → list of significant electrode names.
            Only subjects with at least one significant electrode are included.
        sig_df: The loaded ``all_subject_sig.csv`` DataFrame.

    Raises:
        FileNotFoundError: If the significance CSV is missing.
        ValueError: If no significant electrodes are found for any subject.
    """
    import os

    sig_csv = os.path.join(output_dir, "all_subject_sig.csv")
    if not os.path.exists(sig_csv):
        raise FileNotFoundError(
            f"all_subject_sig.csv not found at {sig_csv}. "
            "Run run_significance.py first."
        )
    sig_df = pd.read_csv(sig_csv)
    print(f"Loaded significance CSV: {len(sig_df):,} significant electrodes")

    sig_electrodes = {
        sub_id: sig_df[sig_df["subject"] == sub_id]["elec"].tolist()
        for sub_id in subject_ids
    }
    sig_electrodes = {s: e for s, e in sig_electrodes.items() if e}

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
    from utils.encoding import load_encoding_results

    subs_with_sig = [s for s in subject_ids if s in sig_electrodes]
    first_sub = subs_with_sig[0]
    enc_results = load_encoding_results(
        os.path.join(output_dir, "subjects", f"sub_{first_sub}", "encoding_results.joblib")
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


def get_arbitrary_embeddings(X, epochs, df_word):
    """
    Assign unique random embeddings per word type, preserving per-word consistency.

    For each unique word in the selection, randomly picks one of the n_words rows
    of X (without replacement) to use as the arbitrary embedding for that word type.
    All occurrences of the same word receive the same row of X.

    Args:
        X (np.ndarray): Embedding array of shape (n_words, n_features).
        epochs: MNE Epochs-like object with a `.selection` attribute (integer indices
                into df_word).
        df_word (pd.DataFrame): DataFrame with a 'word' column; rows indexed by position.

    Returns:
        np.ndarray: Array of shape (n_words, n_features) with arbitrary embeddings
                    assigned consistently per word type.
    """
    words = df_word.iloc[epochs.selection].reset_index()["word"].apply(lambda x: x.lower())
    unique_words = words.unique()
    emb_idxs = np.random.choice(np.arange(X.shape[0]), size=len(unique_words), replace=False)
    arb_word_embs = X[emb_idxs]

    X_arb = np.zeros_like(X)
    for word_idx, word in enumerate(unique_words):
        X_arb[words[words == word].index] = arb_word_embs[word_idx]

    return X_arb
