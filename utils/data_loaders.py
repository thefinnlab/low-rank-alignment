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
