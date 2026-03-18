import marimo

__generated_with = "0.20.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from typing import Optional
    from pathlib import Path

    import mne
    import h5py
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy import stats
    from scipy.stats import pearsonr, entropy

    from nilearn.plotting import plot_markers
    from mne_bids import BIDSPath

    from himalaya.backend import set_backend, get_backend
    from himalaya.ridge import RidgeCV
    from himalaya.scoring import correlation_score

    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.manifold import TSNE
    from scipy.spatial.distance import cosine
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans
    import umap
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import pdist
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from statsmodels.stats.multitest import multipletests
    from scipy.spatial.distance import squareform

    import urllib.request
    import os

    import matplotlib.animation as animation
    from IPython.display import HTML
    import io
    import base64
    import tempfile

    return (
        BIDSPath,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        KFold,
        KMeans,
        LinearRegression,
        Optional,
        PCA,
        Path,
        RidgeCV,
        StandardScaler,
        TSNE,
        animation,
        base64,
        correlation_score,
        cosine,
        dendrogram,
        entropy,
        fcluster,
        get_backend,
        h5py,
        linkage,
        make_pipeline,
        mne,
        mo,
        multipletests,
        np,
        os,
        pd,
        plot_markers,
        plt,
        set_backend,
        squareform,
        stats,
        tempfile,
        torch,
        umap,
        urllib,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Setup
    """)
    return


@app.cell
def _(set_backend, torch):
    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using cuda!")
    return


@app.cell
def _(os, urllib):
    bids_root = "data/"  # if using a local dataset, set this variable accordingly

    embedding_path = f"{bids_root}stimuli/gpt2-xl/features.hdf5"

    if not len(bids_root):
        url = f"https://s3.amazonaws.com/openneuro.org/ds005574/{embedding_path}"
        local_path = "features.hdf5"

        if not os.path.exists(local_path):
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, local_path)
            print("Download complete.")
        else:
            print("File already exists, skipping download.")

        embedding_path = local_path

    print(f"Using embedding file path: {embedding_path}")
    return bids_root, embedding_path


@app.cell
def _(embedding_path, h5py):
    modelname, layer = 'gpt2-xl', 24
    with h5py.File(embedding_path, "r") as f:
        contextual_embeddings = f[f"layer-{layer}"][...]
        final_layer_contextual_embeddings = f[f"layer-48"][...]
    print(f"LLM embedding matrix has shape: {contextual_embeddings.shape}")
    return contextual_embeddings, final_layer_contextual_embeddings


@app.cell
def _(bids_root, os, pd, urllib):
    # Download the transcript, if required
    def _():
        transcript_path = f"{bids_root}stimuli/gpt2-xl/transcript.tsv"
        if not len(bids_root):
            url = f"https://s3.amazonaws.com/openneuro.org/ds005574/{transcript_path}"
            local_path = "transcript.tsv"

            if not os.path.exists(local_path):
                print(f"Downloading from {url}...")
                urllib.request.urlretrieve(url, local_path)
                print("Download complete.")
            else:
                print("File already exists, skipping download.")

            transcript_path = local_path
        return transcript_path

    transcript_path = _()

    # Load transcript
    df_contextual = pd.read_csv(transcript_path, sep="\t", index_col=0)
    if "rank" in df_contextual.columns:
        model_acc = (df_contextual["rank"] == 0).mean()
        print(f"Model accuracy: {model_acc*100:.3f}%")
    df_contextual.head()
    return (df_contextual,)


@app.cell
def _(
    contextual_embeddings,
    df_contextual,
    end,
    epsilon,
    final_layer_contextual_embeddings,
    np,
    start,
    token_probs,
):
    def word_entropy(
        span_entropies: np.ndarray,
        aggregation: str = "sum",      # "sum" | "mean" | "joint_sample"
    ):
        if aggregation == "sum":
            # Joint entropy under independence assumption
            return float(np.sum(span_entropies))

        elif aggregation == "mean":
            # Normalise by word length so short/long words are comparable
            return float(np.mean(span_entropies))

        elif aggregation == "joint_sample":
            # Monte Carlo estimate of the true joint entropy.
            # Sample T word-sequences, estimate p(word) via product of token probs.
            n_samples = 1000
            span_probs = token_probs[start:end]  # [k, vocab_size]
            # Sample token indices for each position
            samples = np.array([
                np.random.choice(span_probs.shape[1], size=n_samples, p=p)
                for p in span_probs
            ])  # [k, n_samples]
            # Compute log-probability of each sampled word sequence
            sample_log_probs = np.sum(
                [np.log(span_probs[i, samples[i]] + epsilon) for i in range(len(span_probs))],
                axis=0,
            )  # [n_samples]
            # H ≈ -E[log p(word)]
            return float(-np.mean(sample_log_probs))

        return None

    def get_aligned_embeds_and_word_stats(contextual_embeddings):
        aligned_embeddings = []
        word_probs = []
        word_entropies = []
        for _, group in df_contextual.groupby("word_idx"): # group by word index
            indices = group.index.to_numpy()
            average_emb = contextual_embeddings[indices].mean(0) # average features
            aligned_embeddings.append(average_emb)
            word_probs.append(np.prod(df_contextual["true_prob"].iloc[indices]))
            word_entropies.append(word_entropy(df_contextual["entropy"].iloc[indices]))

        aligned_embeddings = np.stack(aligned_embeddings)
        word_probs = np.stack(word_probs)
        word_entropies = np.stack(word_entropies)

        return aligned_embeddings, word_probs, word_entropies


    aligned_embeddings, word_probs, word_entropies = get_aligned_embeds_and_word_stats(contextual_embeddings)
    final_layer_aligned_embeddings, _, _ = get_aligned_embeds_and_word_stats(final_layer_contextual_embeddings)

    print(f"LLM embeddings matrix has shape: {aligned_embeddings.shape}")
    return (
        aligned_embeddings,
        final_layer_aligned_embeddings,
        word_entropies,
        word_probs,
    )


@app.cell
def _(word_entropies):
    word_entropies
    return


@app.cell
def _(df_contextual, word_entropies, word_probs):
    df_word = df_contextual.groupby("word_idx").agg(dict(word="first", start="first", end="last"))
    df_word["true_prob"] = word_probs
    df_word["entropy"] = word_entropies
    df_word.head()
    return (df_word,)


@app.cell(hide_code=True)
def _(GPT2LMHeadModel, GPT2TokenizerFast, np, torch):
    def get_token_distributions(text: str, model, max_ctx, stride) -> dict:
        """
        Run GPT-2 XL over a transcript and return the full output probability
        distribution for every token.

        Args:
            text: The transcript as a plain string.

        Returns:
            A dict with:
              - "token_ids": int32 numpy array of shape (T,)
              - "tokens":    object numpy array of shape (T,) with decoded strings
              - "probs":     float32 numpy array of shape (T, V)
                             probs[i] = P(· | token_0 ... token_{i-1})
                             probs[0] = model's distribution given only BOS
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = GPT2TokenizerFast.from_pretrained(model, cache_dir="model_cache/", local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(model, cache_dir="model_cache/", local_files_only=True).eval().to(device)

        input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]  # (T,)
        T = len(input_ids)
        V = model.config.vocab_size

        probs = np.zeros((T + 1, V), dtype=np.float32)

        with torch.no_grad():
            # Get BOS prior for position 0
            bos_id = tokenizer.bos_token_id
            bos_logits = model(torch.tensor([[bos_id]], device=device)).logits[0, 0]
            probs[0] = torch.softmax(bos_logits, dim=-1).cpu().numpy()

            # Sliding window for the rest
            start = 0
            while start < T:
                print(f"{start}/{T} = {start / T:.2f}")
                end = min(start + max_ctx, T + 1)
                window = input_ids[start:end].unsqueeze(0).to(device)
                logits = model(window).logits[0]
                p = torch.softmax(logits, dim=-1).cpu().numpy()

                rec_start = max(1, start + stride) if start > 0 else 1
                for j in range(rec_start, end):
                    probs[j] = p[j - start - 1]

                if end == T:
                    break
                start += stride

        tokens = np.array([tokenizer.decode([tid]) for tid in input_ids.tolist()], dtype=object)

        return {
            "token_ids": input_ids.numpy().astype(np.int32),
            "tokens": tokens,
            "probs": probs,
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
        }

    def save_distributions(result: dict, output_path: str) -> None:
        """Save the output of get_token_distributions to a compressed .npz file."""
        np.savez_compressed(
            output_path,
            token_ids=result["token_ids"],
            tokens=result["tokens"],
            probs=result["probs"],
        )

    return get_token_distributions, save_distributions


@app.cell
def _(df_word, get_token_distributions, np, os, save_distributions):
    MODEL = "gpt2-xl"
    MAX_CTX = 1024
    STRIDE = 512

    if os.path.exists("gpt2_prob_output.npz"):
        gpt2_probs_data = np.load("gpt2_prob_output.npz", allow_pickle=True)
    else:
        podcast_transcript = " ".join(df_word.word.tolist())
        gpt2_probs_data = get_token_distributions(podcast_transcript, MODEL, MAX_CTX, STRIDE)
        save_distributions(gpt2_probs_data, "gpt2_prob_output.npz")

    gpt2_probs  = gpt2_probs_data["probs"]   # (T, 50257)
    gpt2_tokens = gpt2_probs_data["tokens"]
    gpt2_probs
    return (gpt2_probs,)


@app.cell
def _(df_contextual):
    df_contextual
    return


@app.cell(hide_code=True)
def _(df_contextual, gpt2_probs, np):
    def get_aligned_probs(probs):
        # Make sure <BOS> prediction is included
        pred_probs = [probs[0]]
        # TODO: Need to account for punctuation marks alone. "it." maybe should be based just on it? Not sure
        for _, group in df_contextual.groupby("word_idx"): # group by word index
            indices = group.index.to_numpy()
            final_pred_prob = probs[indices[-1] + 1] # take the prediction at the end of the word
            pred_probs.append(final_pred_prob)
        aligned_pred_probs = np.stack(pred_probs)

        return aligned_pred_probs

    aligned_gpt2_probs = get_aligned_probs(gpt2_probs)
    aligned_gpt2_probs
    return (aligned_gpt2_probs,)


@app.cell(hide_code=True)
def _(aligned_gpt2_probs, entropy, np):
    def kl_divergence(probs, eps=1e-10) -> np.ndarray:
        P = probs[:-1] + eps
        Q = probs[1:] + eps
        kl = entropy(P.T, Q.T)  # transpose so shape is (V, T-1), returns (T-1,)
        return kl.astype(np.float32)

    gpt2_word_kl_divergence = kl_divergence(aligned_gpt2_probs)
    gpt2_word_kl_divergence
    return (gpt2_word_kl_divergence,)


@app.cell(hide_code=True)
def _(BIDSPath, Optional, mne, os, pd):
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


    def read_subject_mapping(participant_map_file: str, delimiter="\t"):
        """
        Read and parse a participant mapping file to create subject ID mappings.

        This function reads a tab-delimited or custom-delimited file containing participant
        information and creates a mapping from NYU subject IDs to converted participant IDs.
        The participant IDs are expected to be in the format "sub-XX" where XX is extracted
        as an integer.

        Args:
            participant_map_file (str): Path to the participant mapping CSV/TSV file.
                                       Expected to contain columns 'nyu_id' and 'participant_id'.
            delimiter (str, optional): Delimiter used in the file. Defaults to "\t" (tab).

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
        # Read the subject mapping from data.
        participant_info_df = pd.read_csv(participant_map_file, delimiter=delimiter)

        def participant_id_to_int(participant_id):
            return int(participant_id.split("-")[1])

        subject_id_map = {
            x[1]["nyu_id"]: participant_id_to_int(x[1]["participant_id"])
            for x in participant_info_df.iterrows()
        }
        return subject_id_map


    def read_electrode_file(
        file_path: str, subject_mapping: Optional[dict[int, int]] = None
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

    return load_raws, read_electrode_file, read_subject_mapping


@app.cell
def _(load_raws, read_electrode_file, read_subject_mapping):
    subject_mapping = read_subject_mapping("data/participants.tsv")
    sig_elecs = read_electrode_file("all_subject_sig.csv", subject_mapping)
    raws = load_raws([i for i in sig_elecs.keys()], "data/", per_subject_electrodes=sig_elecs)
    raws
    return raws, sig_elecs


@app.cell
def _():
    # def _():
    #     file_path = BIDSPath(root=f"{bids_root}derivatives/ecogprep",
    #                         subject="03", task="podcast", datatype="ieeg", description="highgamma",
    #                         suffix="ieeg", extension=".fif")
    #     print(f"File path within the dataset: {file_path}")

    #     # You only need to run this if using Colab (i.e. if you did not set bids_root to a local directory)
    #     if not len(bids_root):
    #         url = f"https://s3.amazonaws.com/openneuro.org/ds005574/{file_path}"
    #         if not os.path.exists(file_path.basename):
    #             print(f"Downloading from {url}...")
    #             urllib.request.urlretrieve(url, file_path.basename)
    #             print("Download complete.")
    #         else:
    #             print("File already exists, skipping download.")
    #         file_path = file_path.basename
    #     return file_path

    # file_path = _()

    # raw = mne.io.read_raw_fif(file_path, verbose=False)
    # picks = mne.pick_channels_regexp(raw.ch_names, "LG[AB]*")
    # raw = raw.pick(picks)
    # raw
    return


@app.cell
def _(df_word, np, raws):
    events = np.zeros((len(df_word), 3), dtype=int)
    events[:, 0] = (df_word.start * raws[0].info['sfreq']).astype(int)
    events.shape
    return (events,)


@app.cell
def _(events, mne, np, raws):
    datas = []

    for raw in raws:
        epochs = mne.Epochs(
            raw,
            events,
            tmin=-2.0,
            tmax=2.0,
            baseline=None,
            proj=False,
            event_id=None,
            preload=True,
            event_repeated="merge",
        )

        epochs = epochs.resample(sfreq=32, npad='auto', method='fft', window='hamming')
        data = epochs.get_data(copy=False)

        datas.append(data)

    datas = np.concatenate(datas, axis=1)
    print(f"Datas array has a shape of: {datas.shape}")
    return datas, epochs


@app.cell
def _(raws, sig_elecs):
    electrode_names = [f"S/{list(sig_elecs.keys())[i]}/E/{e_name}" for i, r in enumerate(raws) for e_name in r.ch_names]
    return (electrode_names,)


@app.cell
def _(datas, epochs):
    epochs_data = datas.reshape(len(epochs), -1)
    print(f"ECoG data matrix shape: {epochs_data.shape}")
    return (epochs_data,)


@app.cell
def _(aligned_embeddings, df_word, epochs, final_layer_aligned_embeddings):
    selected_df = df_word.iloc[epochs.selection]
    averaged_embeddings = aligned_embeddings[epochs.selection]
    final_layer_averaged_embeddings = final_layer_aligned_embeddings[epochs.selection]
    print(averaged_embeddings.shape, final_layer_averaged_embeddings.shape)
    return averaged_embeddings, final_layer_averaged_embeddings, selected_df


@app.cell
def _(mo):
    mo.md("""
    ## Probes
    """)
    return


@app.cell
def _(
    averaged_embeddings,
    epochs_data,
    final_layer_averaged_embeddings,
    get_backend,
    np,
):
    X = averaged_embeddings
    Y = epochs_data

    X_final_layer = final_layer_averaged_embeddings

    # Shift to see how well prior word encodes into local area
    # X = X[:-1]
    # Y = Y[1:]

    if "torch" in get_backend().__name__:
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        X_final_layer = X_final_layer.astype(np.float32)

    X.shape, Y.shape, X_final_layer.shape
    return X, X_final_layer, Y


@app.cell
def _(KFold, RidgeCV, StandardScaler, make_pipeline, np):
    alphas = np.logspace(1, 10, 10) # specify alpha values
    inner_cv = KFold(n_splits=5, shuffle=False) # inner 5-fold cross-validation setup
    def create_model(_):
        return make_pipeline(
            StandardScaler(), RidgeCV(alphas, fit_intercept=True, cv=inner_cv) # pipeline
        )

    return alphas, create_model, inner_cv


@app.cell
def _(
    KFold,
    StandardScaler,
    X,
    Y,
    correlation_score,
    create_model,
    datas,
    get_backend,
    np,
):
    epochs_shape = datas.shape[1:]  # number of electrodes * number of lags

    def train_encoding(X, Y, model_constructor_fn, modify_inputs_fn=None, modify_targets_fn=None, epochs_shape=epochs_shape, fix_model=False, provided_idxs=None, provided_scalers=[]):

        models = []
        corrs = [] # empty array to store correlation results
        scalers = []
        tr_idxs, test_idxs  = [], []
        if provided_idxs is None:
            kfold = KFold(2, shuffle=False) # outer 2-fold cross-validation setup
            folds = list(kfold.split(X))
        else:
            folds = provided_idxs

        for i, (train_index, test_index) in enumerate(folds): # loop through folds

            # Split train and test datasets
            X1_train, X1_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            if modify_inputs_fn:
                X1_train = modify_inputs_fn(X1_train, i, idx=train_index)
                X1_test = modify_inputs_fn(X1_test, i, idx=test_index)

            if modify_targets_fn:
                Y_train = modify_targets_fn(Y_train, i, idx=train_index)
                Y_test = modify_targets_fn(Y_test, i, idx=test_index)

            # Standardize Y
            if not provided_scalers:
                scaler = StandardScaler().fit(Y_train)
            else:
                scaler = provided_scalers[i]

            Y_train = scaler.transform(Y_train)
            Y_test = scaler.transform(Y_test)
            scalers.append(scaler)

            model = model_constructor_fn(i)

            if not fix_model:
                model.fit(X1_train, Y_train) # Fit pipeline with transforms and ridge estimator
            Y_preds = model.predict(X1_test) # Use trained model to predict on test set
            corr = correlation_score(Y_test, Y_preds)
            if epochs_shape:
                corr = corr.reshape(epochs_shape) # Compute correlation score

            if "torch" in get_backend().__name__: # if using gpu, transform tensor back to numpy
                corr = corr.numpy(force=True)

            corrs.append(corr) # append fold correlation results to final results
            models.append(model)
            tr_idxs.append(train_index)
            test_idxs.append(test_index)
        return np.stack(corrs), models, tr_idxs, test_idxs, scalers

    # set_backend("torch") # resort to torch or numpy if cuda out of memory
    corrs_embedding, encoding_models, encoding_tr_idxs, encoding_test_idxs, encoding_scalers = train_encoding(X, Y, create_model)
    print(f"Encoding performance correlating matrix shape: {corrs_embedding.shape}")
    return (
        corrs_embedding,
        encoding_models,
        encoding_scalers,
        encoding_test_idxs,
        encoding_tr_idxs,
        train_encoding,
    )


@app.cell
def _(X, Y, create_model, df_word, epochs, np, train_encoding):
    # Setup arbitrary embeddings for each word by shuffling associated word embeddings.
    words = df_word.iloc[epochs.selection].reset_index()["word"].apply(lambda x: x.lower())
    unique_words = words.unique()
    emb_idxs = np.random.choice(np.arange(X.shape[0]), size=len(unique_words), replace=False)
    arb_word_embs = X[emb_idxs]

    X_arb = np.zeros_like(X)
    for word_idx, word in enumerate(unique_words):
        X_arb[words[words == word].index] = arb_word_embs[word_idx]

    corrs_arb, arb_encoding_models, _, _, _ = train_encoding(X_arb, Y, create_model)
    print(f"Encoding performance correlating matrix shape: {corrs_arb.shape}")
    return arb_encoding_models, corrs_arb, unique_words, words


@app.cell
def _(X, Y, create_model, np, train_encoding, unique_words, words):
    # Setup random embeddings for each word by shuffling associated word embeddings.
    random_word_embs = np.random.normal(size=len(unique_words))

    X_rand = np.zeros_like(X)
    for rand_word_idx, rand_word in enumerate(unique_words):
        X_rand[words[words == rand_word].index] = random_word_embs[rand_word_idx]

    corrs_rand, rand_encoding_models, _, _, _ = train_encoding(X_rand, Y, create_model)
    print(f"Encoding performance correlating matrix shape: {corrs_rand.shape}")
    return corrs_rand, rand_encoding_models


@app.cell
def _(X, Y, create_model, np, train_encoding):
    # Setup a single embedding for all words so it can just learn the mean response.
    single_emb = np.random.normal(size=X.shape[1])

    X_single = np.tile(single_emb, (X.shape[0], 1))
    # Add some random bit of noise to each drawing.
    X_single += np.random.normal(scale=1e-5, size=X_single.shape)

    corrs_single, single_encoding_models, _, _, _ = train_encoding(X_single, Y, create_model)
    print(f"Encoding performance correlating matrix shape: {corrs_single.shape}")
    return (corrs_single,)


@app.cell
def _(X_final_layer, Y, create_model, train_encoding):
    corrs_final_layer_embedding, _, _, _, _ = train_encoding(X_final_layer, Y, create_model)
    print(f"Encoding performance correlating matrix shape: {corrs_final_layer_embedding.shape}")
    return (corrs_final_layer_embedding,)


@app.cell
def _(LinearRegression, Y, np, selected_df, train_encoding):
    X_surprisal = selected_df["true_prob"].to_numpy()

    def create_ols_model(_):
        return LinearRegression(fit_intercept=True)
    X_design = np.column_stack([np.ones(X_surprisal.shape[0]), X_surprisal])  # shape [n, 2]

    corrs_surprisal, _, _, _, _ = train_encoding(X_design, Y, create_ols_model)
    print(f"Encoding performance correlating matrix shape: {corrs_surprisal.shape}")
    return X_surprisal, corrs_surprisal, create_ols_model


@app.cell
def _(
    Y,
    create_ols_model,
    epochs,
    gpt2_word_kl_divergence,
    np,
    train_encoding,
):
    X_kl_divergence = gpt2_word_kl_divergence[epochs.selection]
    X_kl_divergence = np.column_stack([np.ones(X_kl_divergence.shape[0]), X_kl_divergence])  # shape [n, 2]

    corrs_kl, _, _, _, _ = train_encoding(X_kl_divergence, Y, create_ols_model)
    print(f"Encoding performance correlating matrix shape: {corrs_kl.shape}")
    return X_kl_divergence, corrs_kl


@app.cell
def _(X_kl_divergence, X_surprisal, Y, create_ols_model, np, train_encoding):
    X_combined = np.column_stack([np.ones(X_kl_divergence.shape[0]), X_kl_divergence, X_surprisal])  # shape [n, 3]

    corrs_combined, _, _, _, _ = train_encoding(X_combined, Y, create_ols_model)
    print(f"Encoding performance correlating matrix shape: {corrs_combined.shape}")
    return (corrs_combined,)


@app.cell
def _(
    corrs_arb,
    corrs_combined,
    corrs_embedding,
    corrs_final_layer_embedding,
    corrs_kl,
    corrs_rand,
    corrs_single,
    corrs_surprisal,
    np,
    plt,
):
    lags = np.arange(-2 * 512, 2 * 512, 16) / 512 # specify the lags
    def add_corrs(corrs, electrode=None, label=None, color="black", ax=None):
        if electrode:
            mean = corrs.mean((0))[electrode]
            err = corrs.std((0)) / np.sqrt(np.prod(corrs.shape[:1]))
            err = err[electrode]
        else:
            mean = corrs.mean((0,1))
            err = corrs.std((0,1)) / np.sqrt(np.prod(corrs.shape[:2]))
        if ax:
            ax.plot(lags, mean, color=color, label=label)
            ax.fill_between(lags, mean - err, mean + err, alpha=0.1, color=color)
        else:
            plt.plot(lags, mean, color=color, label=label)
            plt.fill_between(lags, mean - err, mean + err, alpha=0.1, color=color)

    def plot_embedding_correlations(electrode=None, plot_corrs=[]):
        fig, ax = plt.subplots()

        for corrs in plot_corrs:
            add_corrs(corrs["corrs"], electrode=electrode, ax=ax, color=corrs["color"], label=corrs["label"])
        plt.legend()
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("encoding performance (r ± sem)")
        ax.axvline(0, c=(.9, .9, .9), ls="--")
        ax.axhline(0, c=(.9, .9, .9), ls="--")
        plt.show()

    plot_embedding_correlations(plot_corrs=[
        {"corrs": corrs_embedding, "color": "black", "label": "Middle Layer"},
        {"corrs": corrs_arb, "color": "red", "label": "Arbitrary"},
        {"corrs": corrs_rand, "color": "blue", "label": "Random"},
        {"corrs": corrs_single, "color": "orange", "label": "Single"},
        {"corrs": corrs_final_layer_embedding, "color": "green", "label": "Final Layer"},
        {"corrs": corrs_surprisal, "color": "teal", "label": "Surprisal"},
        {"corrs": corrs_kl, "color": "purple", "label": "KL"},
        {"corrs": corrs_combined, "color": "pink", "label": "KL + Surprisal"},
    ])
    return add_corrs, lags, plot_embedding_correlations


@app.cell
def _(corrs_embedding, np, plot_markers, plt, raws):
    values = corrs_embedding.mean(0).max(-1)

    ch2loc = {ch['ch_name']: ch['loc'][:3] for raw in raws for ch in raw.info['chs']}
    coords = np.vstack([ch2loc[ch] for raw in raws for ch in raw.info['ch_names']])
    coords *= 1000  # nilearn likes to plot in meters, not mm
    print("Coordinate matrix shape: ", coords.shape)

    order = values.argsort()
    plot_markers(values[order], coords[order],
                 node_size=30, display_mode='lzr',
                 node_vmin=0, node_cmap='inferno_r', colorbar=True)
    plt.show()
    return (values,)


@app.cell
def _(np, values):
    electrode_corrs, electrodes_ordered = np.flip(np.sort(values)), np.flip(np.argsort(values))
    electrode_corrs, electrodes_ordered
    return electrode_corrs, electrodes_ordered


@app.cell
def _(corrs_arb, np):
    arb_values = corrs_arb.mean(0).max(-1)
    arb_electrode_corrs, arb_electrodes_ordered = np.flip(np.sort(arb_values)), np.flip(np.argsort(arb_values))
    arb_electrode_corrs, arb_electrodes_ordered
    return (arb_electrode_corrs,)


@app.cell(hide_code=True)
def _(arb_electrode_corrs, electrode_corrs, np, plt, stats):
    def _():
        # --- Compute correlation ---
        pearson_r, p_value = stats.pearsonr(electrode_corrs, arb_electrode_corrs)
        print(f"Pearson r:  {pearson_r:.4f}")
        print(f"P-value:    {p_value:.4e}")

        # --- Fit a regression line ---
        slope, intercept = np.polyfit(electrode_corrs, arb_electrode_corrs, 1)
        line_x = np.linspace(electrode_corrs.min(), electrode_corrs.max(), 200)
        line_y = slope * line_x + intercept

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter + regression line
        axes[0].scatter(electrode_corrs, arb_electrode_corrs, alpha=0.6, edgecolors="white", linewidths=0.5, color="steelblue")
        axes[0].plot(line_x, line_y, color="tomato", linewidth=2, label=f"y = {slope:.2f}x + {intercept:.2f}")
        axes[0].set_title(f"Scatter Plot\nPearson r = {pearson_r:.4f},  p = {p_value:.2e}", fontsize=13)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Residuals
        residuals = arb_electrode_corrs - (slope * electrode_corrs + intercept)
        axes[1].scatter(electrode_corrs, residuals, alpha=0.6, edgecolors="white", linewidths=0.5, color="mediumpurple")
        axes[1].axhline(0, color="tomato", linewidth=2, linestyle="--")
        axes[1].set_title("Residuals", fontsize=13)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("Residual")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("correlation.png", dpi=150)
        return plt.show()


    _()
    return


@app.cell(hide_code=True)
def _(electrode_names, electrodes_ordered, mo):
    electrode_labels = {electrode_names[i]: i for i in electrodes_ordered}
    electrode_corr_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    electrode_corr_dropdown
    return electrode_corr_dropdown, electrode_labels


@app.cell
def _(
    corrs_arb,
    corrs_combined,
    corrs_embedding,
    corrs_final_layer_embedding,
    corrs_kl,
    corrs_rand,
    corrs_single,
    corrs_surprisal,
    electrode_corr_dropdown,
    plot_embedding_correlations,
):
    plot_embedding_correlations(electrode=electrode_corr_dropdown.value, plot_corrs=[
        {"corrs": corrs_embedding, "color": "black", "label": "Middle Layer"},
        {"corrs": corrs_arb, "color": "red", "label": "Arbitrary"},
        {"corrs": corrs_rand, "color": "blue", "label": "Random"},
        {"corrs": corrs_single, "color": "orange", "label": "Single"},
        {"corrs": corrs_final_layer_embedding, "color": "green", "label": "Final Layer"},
        {"corrs": corrs_surprisal, "color": "teal", "label": "Surprisal"},
        {"corrs": corrs_kl, "color": "purple", "label": "KL"},
        {"corrs": corrs_combined, "color": "pink", "label": "KL + Surprisal"},
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Trading Coefficients
    """)
    return


@app.cell
def _(X, Y, datas, encoding_models, encoding_test_idxs, np, torch):
    def get_all_predictions_and_targets_test_set(encoding_models, test_idxs, X, Y, num_electrodes, num_timepoints):
        all_predictions = []
        all_targets = []
        for i, trained_model in enumerate(encoding_models):
            all_predictions.append(trained_model.predict(X[test_idxs[i]]))
            all_targets.append(Y[test_idxs[i]])
    
        all_predictions = torch.vstack(all_predictions).numpy().reshape((X.shape[0], num_electrodes, num_timepoints))
        all_targets = np.vstack(all_targets).reshape((X.shape[0], num_electrodes, num_timepoints))
        return all_predictions, all_targets

    all_predictions, all_targets = get_all_predictions_and_targets_test_set(encoding_models, encoding_test_idxs, X, Y, datas.shape[1], datas.shape[2])
    return all_predictions, all_targets


@app.cell(hide_code=True)
def _(add_corrs, lags, np, plt):
    def compute_correlations(base, comparator, timepoint):
        """
        Vectorized correlation computation.

        base, comparator: [num_words, num_elecs, num_timepoints]
        Returns corrs: [1, num_elecs, num_timepoints]
        """
        # base_slice: [num_words, num_elecs]
        base_slice = base[:, :, timepoint]

        # Center both arrays along the words axis
        b = base_slice - base_slice.mean(axis=0)           # [num_words, num_elecs]
        c = comparator - comparator.mean(axis=0)           # [num_words, num_elecs, num_timepoints]

        # Compute numerator: sum over words -> [num_elecs, num_timepoints]
        numerator = np.einsum('we,wet->et', b, c)

        # Compute denominators
        b_std = np.sqrt((b**2).sum(axis=0))                # [num_elecs]
        c_std = np.sqrt((c**2).sum(axis=0))                # [num_elecs, num_timepoints]

        denom = b_std[:, np.newaxis] * c_std               # [num_elecs, num_timepoints]

        corrs = numerator / denom                          # [num_elecs, num_timepoints]
        return corrs[np.newaxis, :, :]                     # [1, num_elecs, num_timepoints]

    def plot_timepoint_correlation(base, comparator, timepoint, electrode=None):
        """
        Plot correlation between a fixed timepoint in base and all timepoints in comparator.

        Parameters:
            base:        array of shape [num_words, num_elecs, num_timepoints]
            comparator:  array of shape [num_words, num_elecs, num_timepoints]
            timepoint:   index into the last axis of base to use as the reference
            electrode:   optional electrode index; if None, mean over electrodes
        """
        num_words, num_elecs, num_timepoints = base.shape

        corrs = compute_correlations(base, comparator, timepoint)
        # baseline_corrs = compute_correlations(base, )

        x = np.arange(num_timepoints)

        fig, ax = plt.subplots()
        add_corrs(corrs, electrode=electrode, ax=ax, label=f"Base t={lags[timepoint]}")
        ax.set_xlabel("Comparator timepoint")
        ax.set_ylabel("Correlation (r ± sem)")
        ax.axvline(lags[timepoint], c=(0.9, 0.9, 0.9), ls="--", label="Reference timepoint")
        ax.axhline(0, c=(0.9, 0.9, 0.9), ls="--")
        plt.legend()
        plt.show()

    def plot_electrode_correlation(base, comparator, base_electrode, comparator_electrode, plot_baseline=False):
        """
        Correlate two electrodes across words at each timepoint.

        base, comparator:     [num_words, num_elecs, num_timepoints]
        base_electrode:       electrode index into base
        comparator_electrode: electrode index into comparator
        """
        # Extract electrode traces: [num_words, num_timepoints]
        b = base[:, base_electrode, :]
        c = comparator[:, comparator_electrode, :]

        def get_correlation(b, c):
            # Vectorized Pearson correlation across words for each timepoint
            b = b - b.mean(axis=0)
            c = c - c.mean(axis=0)

            numerator = (b * c).sum(axis=0)
            denom = np.sqrt((b**2).sum(axis=0) * (c**2).sum(axis=0))
            corrs = numerator / denom  # [num_timepoints]
            return corrs

        checked_corrs = get_correlation(b, c)
        baseline_corr = get_correlation(base[:, base_electrode, :], comparator[:, base_electrode, :])

        fig, ax = plt.subplots()
        ax.plot(lags, checked_corrs)
        if plot_baseline:
            ax.plot(lags, baseline_corr, color="red", label="Baseline")
        ax.axhline(0, c=(0.9, 0.9, 0.9), ls="--")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Correlation (r)")
        ax.set_title(f"Base electrode {base_electrode} vs Comparator electrode {comparator_electrode}")
        plt.legend()
        plt.show()

    def plot_electrode_similarity_heatmap(base, comparator):
        """
        For each pair of electrodes (one from base, one from comparator),
        compute the correlation of their responses across words AND timepoints,
        giving a single similarity value per pair.

        base, comparator: [num_words, num_elecs, num_timepoints]
        Output heatmap:   [num_elecs_base, num_elecs_comparator]
        """
        num_words, num_elecs_base, num_timepoints = base.shape
        num_elecs_comp = comparator.shape[1]

        # Flatten words and timepoints -> [num_elecs, num_words * num_timepoints]
        b = base.transpose(1, 0, 2).reshape(num_elecs_base, -1)
        c = comparator.transpose(1, 0, 2).reshape(num_elecs_comp, -1)

        # Normalize each electrode trace
        b = b - b.mean(axis=1, keepdims=True)
        c = c - c.mean(axis=1, keepdims=True)
        b = b / np.sqrt((b**2).sum(axis=1, keepdims=True))
        c = c / np.sqrt((c**2).sum(axis=1, keepdims=True))

        # Matrix multiply to get all pairwise correlations at once
        # [num_elecs_base, num_elecs_comp]
        corrs = b @ c.T

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corrs, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Correlation (r)")
        ax.set_xlabel("Comparator electrode")
        ax.set_ylabel("Base electrode")
        ax.set_title("Electrode pairwise similarity")
        plt.tight_layout()
        plt.show()

        return corrs

    return (
        plot_electrode_correlation,
        plot_electrode_similarity_heatmap,
        plot_timepoint_correlation,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Raw data
    """)
    return


@app.cell(hide_code=True)
def _(electrode_corr_dropdown, electrode_labels, lags, mo):
    electrode_raw_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    timepoint_raw_dropdown = mo.ui.dropdown(
        options={str(lags[i]): i for i in range(len(lags))},
        value=str(lags[0]),
        label="timepoint",
    )

    mo.hstack([electrode_corr_dropdown, timepoint_raw_dropdown])
    return (timepoint_raw_dropdown,)


@app.cell
def _(all_targets, plot_timepoint_correlation, timepoint_raw_dropdown):
    plot_timepoint_correlation(all_targets, all_targets, timepoint_raw_dropdown.value)
    return


@app.cell
def _(
    all_targets,
    electrode_corr_dropdown,
    plot_timepoint_correlation,
    timepoint_raw_dropdown,
):
    plot_timepoint_correlation(all_targets, all_targets, timepoint_raw_dropdown.value, electrode=electrode_corr_dropdown.value)
    return


@app.cell(hide_code=True)
def _(electrode_labels, mo):
    electrode_raw_base_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    electrode_raw_comparator_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    mo.hstack([electrode_raw_base_dropdown, electrode_raw_comparator_dropdown])
    return electrode_raw_base_dropdown, electrode_raw_comparator_dropdown


@app.cell
def _(
    all_targets,
    electrode_raw_base_dropdown,
    electrode_raw_comparator_dropdown,
    plot_electrode_correlation,
):
    plot_electrode_correlation(all_targets, all_targets, electrode_raw_base_dropdown.value, electrode_raw_comparator_dropdown.value)
    return


@app.cell
def _(all_targets, plot_electrode_similarity_heatmap):
    plot_electrode_similarity_heatmap(all_targets, all_targets)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Predictions
    """)
    return


@app.cell(hide_code=True)
def _(electrode_labels, lags, mo):
    electrode_prediction_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    timepoint_prediction_dropdown = mo.ui.dropdown(
        options={str(lags[i]): i for i in range(len(lags))},
        value=str(lags[0]),
        label="timepoint",
    )

    mo.hstack([electrode_prediction_dropdown, timepoint_prediction_dropdown])
    return electrode_prediction_dropdown, timepoint_prediction_dropdown


@app.cell
def _(
    all_predictions,
    all_targets,
    plot_timepoint_correlation,
    timepoint_prediction_dropdown,
):
    plot_timepoint_correlation(all_targets, all_predictions, timepoint_prediction_dropdown.value)
    return


@app.cell
def _(
    all_predictions,
    all_targets,
    electrode_prediction_dropdown,
    plot_timepoint_correlation,
    timepoint_prediction_dropdown,
):
    plot_timepoint_correlation(all_targets, all_predictions, timepoint_prediction_dropdown.value, electrode=electrode_prediction_dropdown.value)
    return


@app.cell(hide_code=True)
def _(electrode_labels, mo):
    electrode_pred_base_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Real Electrode",
    )

    electrode_pred_comparator_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Prediction Electrode",
    )

    mo.hstack([electrode_pred_base_dropdown, electrode_pred_comparator_dropdown])
    return electrode_pred_base_dropdown, electrode_pred_comparator_dropdown


@app.cell
def _(
    all_predictions,
    all_targets,
    electrode_pred_base_dropdown,
    electrode_pred_comparator_dropdown,
    plot_electrode_correlation,
):
    plot_electrode_correlation(all_targets, all_predictions, electrode_pred_base_dropdown.value, electrode_pred_comparator_dropdown.value, plot_baseline=True)
    return


@app.cell
def _(all_predictions, all_targets, plot_electrode_similarity_heatmap):
    plot_electrode_similarity_heatmap(all_targets, all_predictions)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Significance Tests
    """)
    return


@app.cell(hide_code=True)
def _(Path, multipletests, np, os, torch):
    from tqdm.notebook import tqdm

    def significant_lags_all_electrodes_gpu(
        predicted, actual,
        n_permutations=5000,
        fdr_q=0.01,
        seed=42,
        device=None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on: {device}")

        pred = torch.tensor(predicted, dtype=torch.float32, device=device)
        act  = torch.tensor(actual,    dtype=torch.float32, device=device)
        n_electrodes, n_lags, n_words = pred.shape

        def batch_corrcoef(x, y):
            x = x - x.mean(dim=-1, keepdim=True)
            y = y - y.mean(dim=-1, keepdim=True)
            num   = (x * y).sum(dim=-1)
            denom = (x**2).sum(dim=-1).sqrt() * (y**2).sum(dim=-1).sqrt()
            return num / denom

        # Step 1: observed correlations
        obs_corrs = batch_corrcoef(pred, act)  # (n_electrodes, n_lags)

        # Step 2: null distribution — one permutation at a time (1.44 GB peak)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        null_ge_count = torch.zeros(n_electrodes, n_lags, device=device)

        for _ in tqdm(range(n_permutations), desc="Permutations"):
            idx = torch.randperm(n_words, device=device, generator=rng)
            null_corr = batch_corrcoef(pred, act[:, :, idx])
            null_ge_count += (null_corr >= obs_corrs)

        # Steps 3 & 4: p-values + FDR (unchanged)
        p_values    = (null_ge_count / n_permutations).cpu().numpy()
        obs_corrs_np = obs_corrs.cpu().numpy()
        flat_p       = p_values.ravel()
        reject_flat, q_flat, _, _ = multipletests(flat_p, alpha=fdr_q, method='fdr_bh')
        sig      = reject_flat.reshape(n_electrodes, n_lags)
        q_values = q_flat.reshape(n_electrodes, n_lags)

        return sig, p_values, q_values, obs_corrs_np

    def save_significance_results(
        filepath,
        sig,
        p_values,
        q_values,
        obs_corrs
    ):
        """
        Save outputs of significant_lags_all_electrodes to a .npz file.

        Parameters
        ----------
        filepath  : str, path to save to (e.g. 'results/significance.npz')
                    .npz extension will be added automatically if not present
        sig       : np.ndarray of bool, shape (n_electrodes, n_lags)
        p_values  : np.ndarray, shape (n_electrodes, n_lags)
        q_values  : np.ndarray, shape (n_electrodes, n_lags)
        obs_corrs : np.ndarray, shape (n_electrodes, n_lags)
        """
        os.makedirs(Path(filepath).parent.absolute(), exist_ok=True)
        np.savez(
            filepath,
            sig=sig,
            p_values=p_values,
            q_values=q_values,
            obs_corrs=obs_corrs
        )


    def load_significance_results(filepath):
        """
        Load outputs saved by save_significance_results.

        Parameters
        ----------
        filepath : str, path to .npz file

        Returns
        -------
        sig, p_values, q_values, obs_corrs
        """
        data = np.load(filepath)
        return data['sig'], data['p_values'], data['q_values'], data['obs_corrs']


    def get_sig_filepath(partial_clusters, sig_dir="sigs", sig_filename="sig_results.npz"):
        """
        partial_clusters: List of tuples of the clusters on that run which were partialed out before generating this data
        sig_dir: Root directory of significance data.
        """
        end_path_directory = [sig_dir]
        for curr_cluster in partial_clusters:
            end_path_directory.append("_".join(curr_cluster))

        end_path_directory.append(sig_filename)
        return os.path.join(*end_path_directory)

    def get_sig_results(partial_clusters, all_predictions, all_targets, sig_dir="sigs", sig_filename="sig_results.npz"):
        """
        partial_clusters: List of tuples of the clusters on that run which were partialed out before generating this data
        sig_dir: Root directory of significance data.
        """
        sig_filepath = get_sig_filepath(partial_clusters)

        if not os.path.exists(sig_filepath):
            electrodes_predictions = all_predictions.transpose(1, 2, 0)
            electrodes_targets = all_targets.transpose(1, 2, 0)
            sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(electrodes_predictions, electrodes_targets)
            os.makedirs("sigs/test.npz", exist_ok=True)
            save_significance_results(sig_filepath, sig, p_values, q_values, obs_corrs)
        else:
            sig, p_values, q_values, obs_corrs = load_significance_results(sig_filepath)

        return sig, p_values, q_values

    return (get_sig_results,)


@app.cell
def _(all_predictions, all_targets, get_sig_results):
    sig, p_values, q_values = get_sig_results([], all_predictions, all_targets)
    sig, p_values, q_values
    return p_values, sig


@app.cell
def _(plt, sig):
    plt.imshow(sig)
    return


@app.cell
def _(p_values, plt):
    plt.hist(p_values.flatten(), bins=40)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Data Vis
    """)
    return


@app.cell
def _(Y, datas):
    reshaped_Y = Y.reshape((Y.shape[0], datas.shape[1], datas.shape[2]))
    mean_responses = reshaped_Y.mean(0)
    return (reshaped_Y,)


@app.cell(hide_code=True)
def _(Y, electrode_labels, mo):
    word_labels = [f"word_{i}" for i in range(Y.shape[0])]


    word_dropdown = mo.ui.dropdown(
        options={label: i for i, label in enumerate(word_labels)},
        value=word_labels[0],
        label="Word",
    )
    electrode_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )
    mo.hstack([word_dropdown, electrode_dropdown], gap=2)
    return electrode_dropdown, word_dropdown


@app.cell(hide_code=True)
def _(
    X,
    datas,
    electrode_dropdown,
    encoding_models,
    encoding_test_idxs,
    lags,
    mo,
    np,
    plt,
    reshaped_Y,
    word_dropdown,
):
    def _():
        word_idx = word_dropdown.value
        elec_idx = electrode_dropdown.value

        signal = reshaped_Y[word_idx, elec_idx, :]
        signal = signal - np.mean(signal)
        signal = signal / abs(np.max(signal))

        # Get model where this word is in the test.
        for i, test_idxs in enumerate(encoding_test_idxs):
            if word_idx in test_idxs:
                curr_model = encoding_models[i]
        predictions = curr_model.predict(np.expand_dims(X[word_idx], 0)).reshape(datas.shape[1:])[elec_idx]
        predictions = predictions - predictions.mean()
        predictions = predictions / predictions.max().abs()

        # if sample_rate is not None:
        #     time = (np.arange(n_samples) / sample_rate) * 1000  # ms
        #     xlabel = "Time (ms)"
        # else:
        #     time = np.arange(n_samples)
        #     xlabel = "Sample"

        xlabel = "Time (ms)"

        # import numpy as np  # needed inside cell

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(lags, signal, linewidth=1.2, color="steelblue")

        ax.plot(lags, predictions, color="red")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"Response — {word_dropdown._value!r} | {electrode_dropdown._value!r}"
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return mo.mpl.interactive(fig)


    _()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Mean Responses
    """)
    return


@app.cell(hide_code=True)
def _(electrode_labels, mo):
    electrode_mean_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    electrode_mean_dropdown
    return (electrode_mean_dropdown,)


@app.cell(hide_code=True)
def _(
    all_predictions,
    electrode_dropdown,
    electrode_mean_dropdown,
    lags,
    mo,
    plt,
    reshaped_Y,
    word_dropdown,
):
    def _():
        elec_idx = electrode_mean_dropdown.value

        signal = reshaped_Y.mean(0)[elec_idx, :]
        signal -= signal.mean()
        signal /= abs(signal.max())

        mean_prediction = all_predictions.mean(0)[elec_idx, :]
        mean_prediction -= mean_prediction.mean()
        mean_prediction /= abs(mean_prediction.max())


        # if sample_rate is not None:
        #     time = (np.arange(n_samples) / sample_rate) * 1000  # ms
        #     xlabel = "Time (ms)"
        # else:
        #     time = np.arange(n_samples)
        #     xlabel = "Sample"

        xlabel = "Time (ms)"

        # import numpy as np  # needed inside cell

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lags, signal, linewidth=1.2, color="steelblue")
        # ax.plot(lags, mean_prediction, linewidth=1.2, color="red")
        # ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"Response — {word_dropdown._value!r} | {electrode_dropdown._value!r}"
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        return mo.mpl.interactive(fig)


    _()
    return


@app.cell
def _(np):
    class MeanResponse:
        def __init__(self):
            self.means = None

        def fit(self, _X, Y):
            self.means = Y.mean(0)

        def predict(self, X):
            return np.tile(self.means, (X.shape[0], 1)) + np.random.normal(scale=1e-6, size=(X.shape[0], self.means.shape[0]))

    return (MeanResponse,)


@app.cell
def _(MeanResponse, X, Y, train_encoding):
    corrs_mean, _, _, _, _ = train_encoding(X, Y, lambda x: MeanResponse())
    return (corrs_mean,)


@app.cell
def _(corrs_embedding, corrs_mean, np, plt):
    def _():
        lags = np.arange(-2 * 512, 2 * 512, 16) / 512 # specify the lags
        def add_corrs(corrs, label=None, color="black", ax=None):
            mean = corrs.mean((0,1))
            err = corrs.std((0,1)) / np.sqrt(np.prod(corrs.shape[:2]))
            if ax:
                ax.plot(lags, mean, color=color, label=label)
                ax.fill_between(lags, mean - err, mean + err, alpha=0.1, color=color)
            else:
                plt.plot(lags, mean, color=color, label=label)
                plt.fill_between(lags, mean - err, mean + err, alpha=0.1, color=color)

        fig, ax = plt.subplots()
        add_corrs(corrs_embedding, ax=ax)
        add_corrs(corrs_mean, ax=ax, color="red")
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("encoding performance (r ± sem)")
        ax.axvline(0, c=(.9, .9, .9), ls="--")
        ax.axhline(0, c=(.9, .9, .9), ls="--")
        plt.show()

    _()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Find best words
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def correlate_predictions(predicted, actual):
        """
        predicted, actual: arrays of shape [n_words, n_electrodes, n_timepoints]
        returns: correlations of shape [n_words, n_electrodes]
        """
        # Center along the timepoints axis
        pred_centered = predicted - predicted.mean(axis=-1, keepdims=True)
        actual_centered = actual - actual.mean(axis=-1, keepdims=True)

        # Compute numerator: dot product along timepoints
        numerator = (pred_centered * actual_centered).sum(axis=-1)

        # Compute denominator: product of norms along timepoints
        pred_norm = np.sqrt((pred_centered ** 2).sum(axis=-1))
        actual_norm = np.sqrt((actual_centered ** 2).sum(axis=-1))
        denominator = pred_norm * actual_norm

        # Avoid division by zero
        correlations = np.where(denominator > 0, numerator / denominator, 0.0)

        return correlations  # shape: [n_words, n_electrodes]

    return (correlate_predictions,)


@app.cell
def _(all_predictions, all_targets, correlate_predictions):
    per_word_elec_correlations = correlate_predictions(all_predictions, all_targets)
    per_word_elec_correlations.shape
    return (per_word_elec_correlations,)


@app.cell
def _(per_word_elec_correlations):
    per_word_correlations = per_word_elec_correlations.mean(1)
    return (per_word_correlations,)


@app.cell
def _(per_word_elec_correlations, plt):
    plt.hist(per_word_elec_correlations.flatten())
    plt.show()
    return


@app.cell
def _(per_word_elec_correlations, plt):
    plt.hist(per_word_elec_correlations.mean(1).flatten())
    plt.show()
    return


@app.cell
def _(
    RidgeCV,
    StandardScaler,
    X,
    alphas,
    inner_cv,
    make_pipeline,
    np,
    per_word_correlations,
    train_encoding,
):
    performance_pred_model = make_pipeline(
        StandardScaler(), RidgeCV(alphas, fit_intercept=True, cv=inner_cv) # pipeline
    )
    corrs_performance, performance_pred_models, performance_pred_tr, performance_pred_tests, _ = train_encoding(X, np.expand_dims(per_word_correlations, 1), lambda _: performance_pred_model, epochs_shape=None)
    print(f"Encoding performance correlating matrix shape: {corrs_performance.shape}")
    return performance_pred_models, performance_pred_tests


@app.cell(hide_code=True)
def _(np, plt, stats):
    def get_predicted_corrs(models, input_embeds, word_correlations, test_idxs):
        predicted_corrs = []
        reordered_groundtruth_correlations = []
        for i, model in enumerate(models):
            predicted_corrs.append(model.predict(input_embeds[test_idxs[i]]))
            reordered_groundtruth_correlations.append(word_correlations[test_idxs[i]])

        return np.vstack(predicted_corrs).flatten(), np.vstack(reordered_groundtruth_correlations).flatten()


    def plot_correlations(x, y):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value**2

        # Line of best fit
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(x, y, alpha=0.3, s=10, color='steelblue', label='Data')
        ax.plot(x_line, y_line, color='red', linewidth=2, label='Line of Best Fit')

        # Annotation box
        stats_text = (
            f"r = {r_value:.4f}\n"
            f"R² = {r_squared:.4f}\n"
            f"p = {p_value:.2e}"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Predicted Correlation')
        ax.set_ylabel('Actual Correlation')
        ax.set_title('Correlation between Array 1 and Array 2')
        ax.legend()
        plt.tight_layout()
        plt.show()

    return get_predicted_corrs, plot_correlations


@app.cell
def _(
    X,
    get_predicted_corrs,
    per_word_correlations,
    performance_pred_models,
    performance_pred_tests,
    plot_correlations,
):
    predicted_corrs, reordered_groundtruth_correlations = get_predicted_corrs(performance_pred_models, X, per_word_correlations, performance_pred_tests)

    plot_correlations(predicted_corrs, reordered_groundtruth_correlations)
    return


@app.cell
def _(np, performance_pred_models):
    def model_dir_similarity(models):
        model_1_dir = models[0].named_steps['ridgecv'].coef_.flatten()
        model_2_dir = models[1].named_steps['ridgecv'].coef_.flatten()
        norm1 = model_1_dir / np.linalg.norm(model_1_dir, axis=0, keepdims=True)
        norm2 = model_2_dir / np.linalg.norm(model_2_dir, axis=0, keepdims=True)

        # Compute cosine similarity matrix [n, m]
        return norm1 @ norm2.T

    model_dir_similarity(performance_pred_models)
    return (model_dir_similarity,)


@app.cell(hide_code=True)
def _(electrode_labels, mo):
    # Electrode specific response
    electrode_direction_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    electrode_direction_dropdown
    return (electrode_direction_dropdown,)


@app.cell
def _(
    RidgeCV,
    StandardScaler,
    X,
    alphas,
    electrode_direction_dropdown,
    get_predicted_corrs,
    inner_cv,
    make_pipeline,
    np,
    per_word_elec_correlations,
    plot_correlations,
    train_encoding,
):
    performance_elec_pred_model = make_pipeline(
        StandardScaler(), RidgeCV(alphas, fit_intercept=True, cv=inner_cv) # pipeline
    )

    curr_correlations = per_word_elec_correlations[:, electrode_direction_dropdown.value]
    corrs_elec_performance, performance_elec_pred_models, performance_elec_pred_tr, performance_elec_pred_tests, _ = train_encoding(X, np.expand_dims(curr_correlations, 1), lambda _: performance_elec_pred_model, epochs_shape=None)

    predicted_elec_corrs, elec_groundtruth_correlations = get_predicted_corrs(performance_elec_pred_models, X, curr_correlations, performance_elec_pred_tests)

    plot_correlations(predicted_elec_corrs, elec_groundtruth_correlations)
    return (performance_elec_pred_models,)


@app.cell
def _(model_dir_similarity, performance_elec_pred_models):
    model_dir_similarity(performance_elec_pred_models)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Model Coefficients
    """)
    return


@app.cell
def _(X, datas, encoding_models):
    # model_coefs = model.named_steps['ridgecv'].coef_.view(1600, num_elecs, 128).permute(1, 2, 0)
    model_1_coefs = encoding_models[0].named_steps['ridgecv'].coef_.view(X.shape[1], datas.shape[1], datas.shape[2]).permute(1, 2, 0)
    model_2_coefs = encoding_models[1].named_steps['ridgecv'].coef_.view(X.shape[1], datas.shape[1], datas.shape[2]).permute(1, 2, 0)
    model_1_bias = encoding_models[0].named_steps['ridgecv'].intercept_
    model_2_bias = encoding_models[1].named_steps['ridgecv'].intercept_
    mean_model_coefs = (model_1_coefs + model_2_coefs) / 2
    mean_model_bias = (model_1_bias + model_2_bias) / 2
    model_1_coefs.shape, model_2_coefs.shape
    return (
        mean_model_bias,
        mean_model_coefs,
        model_1_bias,
        model_1_coefs,
        model_2_bias,
        model_2_coefs,
    )


@app.cell
def _(X, datas, encoding_models, np):
    # Try again but remove scaling and operate in raw space this time
    def absorb_scaler(encoding_model):
        """Convert Ridge coefficients from standardized space back to raw input space."""
        scaler = encoding_model.named_steps["standardscaler"]
        ridge = encoding_model.named_steps["ridgecv"]

        # ridge.coef_ shape: (n_targets, n_features)
        # In standardized space: y = coef @ x_scaled + intercept
        # where x_scaled = (x - mean) / scale
        # So: y = (coef / scale) @ x + (intercept - coef @ mean / scale)

        # coef_ shape: (n_features, n_targets) in your case
        raw_coefs = ridge.coef_ / scaler.scale_[:, np.newaxis]
        raw_intercept = ridge.intercept_ - (scaler.mean_ @ raw_coefs.numpy())

        return raw_coefs.numpy(), raw_intercept

    raw_coefs_1, raw_intercept_1 = absorb_scaler(encoding_models[0])
    raw_coefs_2, raw_intercept_2 = absorb_scaler(encoding_models[1])

    raw_coefs_1 = raw_coefs_1.reshape(X.shape[1], datas.shape[1], datas.shape[2]).transpose(1, 2, 0)
    raw_coefs_2 = raw_coefs_2.reshape(X.shape[1], datas.shape[1], datas.shape[2]).transpose(1, 2, 0)

    mean_raw_coefs = (raw_coefs_1 + raw_coefs_2) / 2
    mean_raw_intercept = (raw_intercept_1 + raw_intercept_2) / 2
    return (
        mean_raw_coefs,
        mean_raw_intercept,
        raw_coefs_1,
        raw_coefs_2,
        raw_intercept_1,
        raw_intercept_2,
    )


@app.cell(hide_code=True)
def _(electrode_labels, mo):
    electrode_coef_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    electrode_coef_dropdown
    return (electrode_coef_dropdown,)


@app.cell(hide_code=True)
def _(PCA, cosine, lags, np, plt):
    def plot_pairwise_cosine_similarity(arr1: np.ndarray, arr2: np.ndarray, title: str = "Pairwise Cosine Similarity") -> None:
        """
        Plots the pairwise cosine similarity between each pair of embeddings.

        Args:
            arr1: Array of shape [n, n_dim]
            arr2: Array of shape [m, n_dim]
            title: Title for the plot
        """
        # Normalize rows to unit vectors
        norm1 = arr1 / np.linalg.norm(arr1, axis=1, keepdims=True)
        norm2 = arr2 / np.linalg.norm(arr2, axis=1, keepdims=True)

        # Compute cosine similarity matrix [n, m]
        similarity = norm1 @ norm2.T

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(similarity, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        plt.colorbar(im, ax=ax, label="Cosine Similarity")

        ax.set_title(title)
        ax.set_xlabel("Array 2 index")
        ax.set_ylabel("Array 1 index")

        # Annotate cells if the matrix is small enough
        if similarity.shape[0] <= 20 and similarity.shape[1] <= 20:
            for i in range(similarity.shape[0]):
                for j in range(similarity.shape[1]):
                    ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        plt.show()

        return similarity

    def high_d_plots(model_coefs, electrode_idx):

        # --- Replace this with your actual data ---
        # Shape: (128, 1600)
        vectors = model_coefs[electrode_idx]
        timepoints = lags

        # Normalize to unit vectors (isolates direction from magnitude)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors = vectors / norms

        # ── 1. PCA Trajectory ─────────────────────────────────────────────────────────
        # Projects onto the 2 directions of maximum variance across all timepoints.
        # A smooth curve = gradual directional drift; loops/reversals = cyclical behavior.

        pca = PCA(n_components=3)
        pca_coords = pca.fit_transform(unit_vectors)
        explained = pca.explained_variance_ratio_

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Directional Evolution of Vectors Over Time", fontsize=14)

        ax = axes[0]
        sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=timepoints, cmap='plasma', s=40, zorder=3)
        ax.plot(pca_coords[:, 0], pca_coords[:, 1], alpha=0.3, color='gray', linewidth=0.8)
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
        ax.set_title("PCA Trajectory (PC1 vs PC2)")
        plt.colorbar(sc, ax=ax, label="Timepoint")

        ax = axes[1]
        sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 2], c=timepoints, cmap='plasma', s=40, zorder=3)
        ax.plot(pca_coords[:, 0], pca_coords[:, 2], alpha=0.3, color='gray', linewidth=0.8)
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
        ax.set_ylabel(f"PC3 ({explained[2]:.1%} var)")
        ax.set_title("PCA Trajectory (PC1 vs PC3)")
        plt.colorbar(sc, ax=ax, label="Timepoint")

        # ── 2. Cosine Similarity to First Vector ──────────────────────────────────────
        # Shows how much the direction drifts away from the starting point over time.
        # A declining trend = persistent drift; oscillations = reversals or cycles.

        ax = axes[2]
        cos_sim_to_first = [1 - cosine(unit_vectors[0], unit_vectors[t]) for t in range(128)]
        ax.plot(timepoints, cos_sim_to_first, color='steelblue', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Similarity to t=0 Direction")
        ax.set_ylim(-1.05, 1.05)

        plt.tight_layout()
        plt.savefig("direction_evolution_row1.png", dpi=150)
        plt.show()

        # ── 3. Cosine Similarity Heatmap ──────────────────────────────────────────────
        # Full pairwise directional similarity matrix. Blocks of high similarity = stable
        # phases; sharp transitions between blocks = abrupt directional changes.

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sim_matrix = unit_vectors @ unit_vectors.T
        im = axes[0].imshow(sim_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, origin='upper')
        axes[0].set_title("Pairwise Cosine Similarity Heatmap")
        axes[0].set_xlabel("Timepoint")
        axes[0].set_ylabel("Timepoint")
        plt.colorbar(im, ax=axes[0])

        # ── 4. Step-wise Angular Change ───────────────────────────────────────────────
        # Angle (in degrees) between consecutive timepoints.
        # Spikes = sudden directional jumps; flat regions = stable direction.

        angles = []
        for t in range(1, 128):
            sim = np.clip(1 - cosine(unit_vectors[t-1], unit_vectors[t]), -1, 1)
            angles.append(np.degrees(np.arccos(sim)))

        axes[1].plot(timepoints[1:], angles, color='crimson', linewidth=1.2)
        axes[1].fill_between(timepoints[1:], angles, alpha=0.2, color='crimson')
        axes[1].set_xlabel("Timepoint")
        axes[1].set_ylabel("Angle (degrees)")
        axes[1].set_title("Step-wise Angular Change Between Consecutive Timepoints")

        plt.tight_layout()
        plt.savefig("direction_evolution_row2.png", dpi=150)
        plt.show()

        # ── 5. Top PCA Component Loadings Over Time ───────────────────────────────────
        # Projects each timepoint onto each of the top 3 PCs individually.
        # Reveals the independent "modes" of directional variation.

        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['steelblue', 'darkorange', 'green']
        for i in range(3):
            ax.plot(timepoints, pca_coords[:, i], label=f"PC{i+1} ({explained[i]:.1%})", 
                    color=colors[i], linewidth=1.4)
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Projection")
        ax.set_title("Projection onto Top 3 PCA Directions Over Time")
        ax.legend()
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)

        plt.tight_layout()
        plt.savefig("direction_evolution_row3.png", dpi=150)
        return plt.show()

    return high_d_plots, plot_pairwise_cosine_similarity


@app.cell
def _(electrode_coef_dropdown, high_d_plots, raw_coefs_1):
    high_d_plots(raw_coefs_1, electrode_coef_dropdown.value)
    return


@app.cell
def _(electrode_coef_dropdown, high_d_plots, raw_coefs_2):
    high_d_plots(raw_coefs_2, electrode_coef_dropdown.value)
    return


@app.cell
def _(electrode_coef_dropdown, high_d_plots, mean_raw_coefs):
    high_d_plots(mean_raw_coefs, electrode_coef_dropdown.value)
    return


@app.cell
def _(
    electrode_coef_dropdown,
    model_1_coefs,
    model_2_coefs,
    plot_pairwise_cosine_similarity,
):
    plot_pairwise_cosine_similarity(model_1_coefs[electrode_coef_dropdown.value], model_2_coefs[electrode_coef_dropdown.value])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Arbitrary Model
    """)
    return


@app.cell
def _(X, arb_encoding_models, datas, electrode_coef_dropdown, high_d_plots):
    # model_coefs = model.named_steps['ridgecv'].coef_.view(1600, 127, 128).permute(1, 2, 0)
    arb_model_1_coefs = arb_encoding_models[0].named_steps['ridgecv'].coef_.view(X.shape[1], datas.shape[1], datas.shape[2]).permute(1, 2, 0)
    arb_model_2_coefs = arb_encoding_models[1].named_steps['ridgecv'].coef_.view(X.shape[1], datas.shape[1], datas.shape[2]).permute(1, 2, 0)

    high_d_plots(arb_model_1_coefs, electrode_coef_dropdown.value)
    return arb_model_1_coefs, arb_model_2_coefs


@app.cell
def _(arb_model_2_coefs, electrode_coef_dropdown, high_d_plots):
    high_d_plots(arb_model_2_coefs, electrode_coef_dropdown.value)
    return


@app.cell
def _(
    arb_model_1_coefs,
    arb_model_2_coefs,
    electrode_coef_dropdown,
    plot_pairwise_cosine_similarity,
):
    plot_pairwise_cosine_similarity(arb_model_1_coefs[electrode_coef_dropdown.value], arb_model_2_coefs[electrode_coef_dropdown.value])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Random Embeddings
    """)
    return


@app.cell
def _(X, datas, electrode_coef_dropdown, high_d_plots, rand_encoding_models):
    # model_coefs = model.named_steps['ridgecv'].coef_.view(1600, 127, 128).permute(1, 2, 0)
    rand_model_1_coefs = rand_encoding_models[0].named_steps['ridgecv'].coef_.view(X.shape[1], datas.shape[1], datas.shape[2]).permute(1, 2, 0)
    rand_model_2_coefs = rand_encoding_models[1].named_steps['ridgecv'].coef_.view(X.shape[1], datas.shape[1], datas.shape[2]).permute(1, 2, 0)

    high_d_plots(rand_model_1_coefs, electrode_coef_dropdown.value)
    return rand_model_1_coefs, rand_model_2_coefs


@app.cell
def _(electrode_coef_dropdown, high_d_plots, rand_model_2_coefs):
    high_d_plots(rand_model_2_coefs, electrode_coef_dropdown.value)
    return


@app.cell
def _(
    electrode_coef_dropdown,
    plot_pairwise_cosine_similarity,
    rand_model_1_coefs,
    rand_model_2_coefs,
):
    plot_pairwise_cosine_similarity(rand_model_1_coefs[electrode_coef_dropdown.value], rand_model_2_coefs[electrode_coef_dropdown.value])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Electrode vs Electrode
    """)
    return


@app.cell
def _(mean_raw_coefs, plot_pairwise_cosine_similarity):
    plot_pairwise_cosine_similarity(mean_raw_coefs.mean(1), mean_raw_coefs.mean(1))
    return


@app.cell
def _(mean_raw_coefs, plot_pairwise_cosine_similarity):
    plot_pairwise_cosine_similarity(mean_raw_coefs[:, 75], mean_raw_coefs[:, 75])
    return


@app.cell
def _(model_2_coefs, plot_pairwise_cosine_similarity):
    plot_pairwise_cosine_similarity(model_2_coefs[:, 75], model_2_coefs[:, 75])
    return


@app.cell(hide_code=True)
def _(lags, mo):
    timepoint_center_dropdown = mo.ui.dropdown(
        options={str(lags[i]): i for i in range(len(lags))},
        value=str(lags[0]),
        label="timepoint",
    )

    timepoint_width_slider = mo.ui.slider(
        start=1,
        stop=9,
        step=2,
        show_value=True,
    )

    mo.hstack([timepoint_center_dropdown, timepoint_width_slider])
    return timepoint_center_dropdown, timepoint_width_slider


@app.cell
def _(
    mean_raw_coefs,
    plot_pairwise_cosine_similarity,
    timepoint_center_dropdown,
    timepoint_width_slider,
):
    def plot_time_averaged_model_coefficient_similarity(model_coefs, mid_timepoint, timepoint_width):
        lower_timepoint = mid_timepoint - timepoint_width // 2
        upper_timepoint = mid_timepoint + timepoint_width // 2 + 1
        final_coefs = model_coefs[:, lower_timepoint:upper_timepoint].mean(1)
        plot_pairwise_cosine_similarity(final_coefs, final_coefs)

    plot_time_averaged_model_coefficient_similarity(mean_raw_coefs, timepoint_center_dropdown.value, timepoint_width_slider.value)
    return


@app.cell(hide_code=True)
def _(
    animation,
    base64,
    mean_raw_coefs,
    mo,
    np,
    os,
    plt,
    tempfile,
    timepoint_width_slider,
):
    def create_heatmap_animation(model_coefs, timepoint_width, step=3, fps=8, figsize=(4, 3)):
        min_mid = timepoint_width // 2
        max_mid = model_coefs.shape[1] - timepoint_width // 2 - 1
        timepoints = range(min_mid, max_mid + 1, step)  # skip frames with step

        fig, ax = plt.subplots(figsize=figsize)  # smaller figure = smaller file

        def get_similarity_matrix(mid_timepoint):
            lower = mid_timepoint - timepoint_width // 2
            upper = mid_timepoint + timepoint_width // 2 + 1
            final_coefs = model_coefs[:, lower:upper].mean(1)
            norms = np.linalg.norm(final_coefs, axis=1, keepdims=True)
            normalized = final_coefs / (norms + 1e-8)
            return normalized @ normalized.T

        sim_matrix = get_similarity_matrix(min_mid)
        im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax)
        title = ax.set_title(f"Mid timepoint: {min_mid}")
        plt.tight_layout()

        def update(mid_timepoint):
            sim_matrix = get_similarity_matrix(mid_timepoint)
            im.set_data(sim_matrix)
            title.set_text(f"Mid timepoint: {mid_timepoint}")
            return [im, title]

        anim = animation.FuncAnimation(
            fig, update, frames=timepoints, interval=1000//fps, blit=True
        )
        plt.close(fig)
        return anim

    anim = create_heatmap_animation(mean_raw_coefs, timepoint_width_slider.value)

    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as fp:
        tmp_path = fp.name

    anim.save(tmp_path, writer='pillow', fps=10)

    with open(tmp_path, 'rb') as fp:
        b64 = base64.b64encode(fp.read()).decode()

    os.unlink(tmp_path)

    mo.Html(f'<img src="data:image/gif;base64,{b64}" style="width:100%"/>')
    return


@app.cell
def _(mo):
    mo.md("""
    ## Word Embeddings
    """)
    return


@app.cell
def _(X, plot_pairwise_cosine_similarity):
    cos_sim_embeds = plot_pairwise_cosine_similarity(X, X)
    return (cos_sim_embeds,)


@app.cell
def _(cos_sim_embeds, np, plt):
    cosine_sims = np.tril(cos_sim_embeds, -1)[np.tril(np.ones_like(cos_sim_embeds, dtype=bool), -1)]
    plt.hist(cosine_sims, bins=40)
    plt.show()
    return (cosine_sims,)


@app.cell
def _(cosine_sims, np):
    np.mean(cosine_sims), np.std(cosine_sims)
    return


@app.cell
def _(cosine_sims, np, plt):
    deg_differences = np.acos(cosine_sims) * 180 / np.pi
    plt.hist(deg_differences, bins=40)
    plt.show()
    return (deg_differences,)


@app.cell
def _(deg_differences, np):
    np.mean(deg_differences), np.std(deg_differences)
    return


@app.cell
def _(deg_differences):
    min(deg_differences), max(deg_differences)
    return


@app.cell
def _(X, np):
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    centroid = X_norm.mean(0)
    centroid_mag = np.linalg.norm(centroid)
    normalized_centroid = centroid / centroid_mag
    centroid_mag
    return X_norm, normalized_centroid


@app.cell
def _(X, normalized_centroid, np, plot_pairwise_cosine_similarity):
    relative_to_centroid = plot_pairwise_cosine_similarity(X, np.tile(normalized_centroid, (X.shape[0], 1)))
    return (relative_to_centroid,)


@app.cell
def _(np, plt, relative_to_centroid):
    deg_centroid_differences = np.acos(relative_to_centroid[:, 0]) * 180 / np.pi
    plt.hist(deg_centroid_differences, bins=40)
    plt.show()
    return


@app.cell
def _(PCA, X_norm, plt):
    pca = PCA(n_components=10)
    pca_coords = pca.fit_transform(X_norm)
    explained = pca.explained_variance_ratio_
    plt.plot(explained)
    plt.scatter(range(len(explained)), explained)
    plt.show()
    explained.sum()
    return


@app.cell
def _(X, mean_raw_coefs, plot_pairwise_cosine_similarity):
    plot_pairwise_cosine_similarity(X, mean_raw_coefs[:, 75])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Combine Coefficients and Word Embeddings
    """)
    return


@app.cell(hide_code=True)
def _(electrode_labels, mo):
    electrode_pca_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    electrode_pca_dropdown
    return (electrode_pca_dropdown,)


@app.cell
def _(
    electrode_pca_dropdown,
    model_1_coefs,
    model_2_coefs,
    normalized_centroid,
    np,
    plot_pairwise_cosine_similarity,
    raw_coefs_1,
    raw_coefs_2,
):
    model_coefs_to_centroid_sim_1 = plot_pairwise_cosine_similarity(raw_coefs_1[electrode_pca_dropdown.value], np.tile(normalized_centroid, (model_1_coefs.shape[1], 1)))
    model_coefs_to_centroid_sim_2 = plot_pairwise_cosine_similarity(raw_coefs_2[electrode_pca_dropdown.value], np.tile(normalized_centroid, (model_2_coefs.shape[1], 1)))
    model_coefs_to_model_coefs = plot_pairwise_cosine_similarity(raw_coefs_1[electrode_pca_dropdown.value], raw_coefs_2[electrode_pca_dropdown.value])
    return (
        model_coefs_to_centroid_sim_1,
        model_coefs_to_centroid_sim_2,
        model_coefs_to_model_coefs,
    )


@app.cell
def _(
    lags,
    model_coefs_to_centroid_sim_1,
    model_coefs_to_centroid_sim_2,
    model_coefs_to_model_coefs,
    np,
    plt,
):
    plt.plot(lags, model_coefs_to_centroid_sim_1[:, 0], label="model 1")
    plt.plot(lags, model_coefs_to_centroid_sim_2[:, 0], label="model 2")
    plt.plot(lags, np.diag(model_coefs_to_model_coefs), label = "model to model")
    plt.legend()
    plt.show()
    return


@app.cell
def _(
    lags,
    model_coefs_to_centroid_sim_1,
    model_coefs_to_centroid_sim_2,
    model_coefs_to_model_coefs,
    np,
    plt,
):
    plt.plot(lags, np.acos(model_coefs_to_centroid_sim_1[:, 0]) * 180 / np.pi, label="model 1")
    plt.plot(lags, np.acos(model_coefs_to_centroid_sim_2[:, 0]) * 180 / np.pi, label="model 2")
    plt.plot(lags, np.acos(np.diag(model_coefs_to_model_coefs)) * 180 / np.pi, label = "model to model")
    plt.legend()
    plt.show()
    return


@app.cell
def _(PCA, electrode_pca_dropdown, mean_raw_coefs, np, plt):
    pca_model = PCA(n_components=2)
    norm_model_coefs = mean_raw_coefs / np.linalg.norm(mean_raw_coefs, axis=2, keepdims=True)
    pca_coords_model = pca_model.fit_transform(norm_model_coefs[electrode_pca_dropdown.value])
    explained_model = pca_model.explained_variance_ratio_

    plt.plot(explained_model)
    plt.scatter(range(len(explained_model)), explained_model)
    plt.show()
    explained_model.sum()
    return explained_model, norm_model_coefs, pca_coords_model, pca_model


@app.cell
def _(
    explained_model,
    lags,
    normalized_centroid,
    np,
    pca_coords_model,
    pca_model,
    plt,
):
    def _():
        sc = plt.scatter(pca_coords_model[:, 0], pca_coords_model[:, 1], c=lags, cmap='plasma', s=40, zorder=3)
        plt.plot(pca_coords_model[:, 0], pca_coords_model[:, 1], alpha=0.3, color='gray', linewidth=0.8)
        pca_centroid = pca_model.transform(np.expand_dims(normalized_centroid, 0))
        plt.scatter(pca_centroid[:, 0], pca_centroid[:, 1])
        plt.xlabel(f"PC1 ({explained_model[0]:.1%} var)")
        plt.ylabel(f"PC2 ({explained_model[1]:.1%} var)")
        plt.title("PCA Trajectory (PC1 vs PC2)")
        plt.colorbar(sc, label="Timepoint")
        plt.show()

    _()
    return


@app.cell
def _(
    PCA,
    X_norm,
    electrode_pca_dropdown,
    lags,
    norm_model_coefs,
    normalized_centroid,
    np,
    plt,
):
    pca_model_3d = PCA(n_components=3)
    pca_coords_model_3d = pca_model_3d.fit_transform(norm_model_coefs[electrode_pca_dropdown.value])
    explained_model_3d = pca_model_3d.explained_variance_ratio_

    def _():
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Sample and project word embeddings into the same PCA space
        n_samples = 500  # adjust as needed
        sample_idx = np.random.choice(X_norm.shape[0], size=n_samples, replace=False)
        X_sample = X_norm[sample_idx]
        X_pca = pca_model_3d.transform(X_sample)  # uses already-fitted PCA

        ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
            alpha=0.15, s=10, color='steelblue', label='Word embeddings'
        )

        # Original trajectory plot
        sc = ax.scatter(
            pca_coords_model_3d[:, 0],
            pca_coords_model_3d[:, 1],
            pca_coords_model_3d[:, 2],
            c=lags, cmap='plasma', s=40
        )
        ax.plot(
            pca_coords_model_3d[:, 0],
            pca_coords_model_3d[:, 1],
            pca_coords_model_3d[:, 2],
            alpha=0.3, color='gray', linewidth=0.8
        )

        pca_centroid_3d = pca_model_3d.transform(np.expand_dims(normalized_centroid, 0))
        ax.scatter(
            pca_centroid_3d[:, 0], pca_centroid_3d[:, 1], pca_centroid_3d[:, 2],
            s=80, marker='*', color='red', label='Centroid'
        )

        ax.set_xlabel(f"PC1 ({explained_model_3d[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({explained_model_3d[1]:.1%} var)")
        ax.set_zlabel(f"PC3 ({explained_model_3d[2]:.1%} var)")
        ax.set_title("PCA Trajectory (PC1 vs PC2 vs PC3)")
        ax.legend()
        plt.colorbar(sc, ax=ax, label="Timepoint", shrink=0.6)
        plt.tight_layout()
        plt.show()
    _()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Dataset Examples
    """)
    return


@app.cell(hide_code=True)
def _(electrode_labels, lags, mo):
    electrode_examples_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    timepoint_examples_dropdown = mo.ui.dropdown(
        options={str(lags[i]): i for i in range(len(lags))},
        value=str(lags[0]),
        label="timepoint",
    )

    mo.hstack([electrode_examples_dropdown, timepoint_examples_dropdown])
    return electrode_examples_dropdown, timepoint_examples_dropdown


@app.cell
def _(
    all_predictions,
    electrode_examples_dropdown,
    plt,
    timepoint_examples_dropdown,
):
    curr_activations = all_predictions[:, electrode_examples_dropdown.value, timepoint_examples_dropdown.value]
    plt.hist(curr_activations, bins=40)
    plt.show()
    return (curr_activations,)


@app.cell
def _(
    all_targets,
    curr_activations,
    electrode_examples_dropdown,
    plot_correlations,
    timepoint_examples_dropdown,
):
    curr_targets = all_targets[:, electrode_examples_dropdown.value, timepoint_examples_dropdown.value]
    plot_correlations(curr_activations, curr_targets)
    return


@app.cell
def _(curr_activations, plot_correlations, selected_df):
    plot_correlations(curr_activations, selected_df["true_prob"])
    return


@app.cell
def _(X_kl_divergence, curr_activations, plot_correlations):
    plot_correlations(curr_activations, X_kl_divergence[:, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    num_bins_slider = mo.ui.slider(
        start=3, stop=20, value=8, step=1, label="Number of bins"
    )
    samples_per_bin_slider = mo.ui.slider(
        start=1, stop=30, value=5, step=1, label="Samples per bin"
    )
    context_slider = mo.ui.slider(
        start=1, stop=30, value=8, step=1, label="Context window (words each side)"
    )
    seed_input = mo.ui.number(value=0, label="Random seed")

    controls = mo.hstack(
        [num_bins_slider, samples_per_bin_slider, context_slider, seed_input],
        gap="2rem"
    )
    controls
    return context_slider, num_bins_slider, samples_per_bin_slider, seed_input


@app.cell(hide_code=True)
def _(curr_activations, mo, np, num_bins_slider, pd):
    import altair as alt

    # Fine-grained histogram fixed at 100 bins
    def plot_activations(curr_activations):
        _fine_counts, _fine_edges = np.histogram(curr_activations, bins=100)
        _hist_df = pd.DataFrame({
            "x1": _fine_edges[:-1],
            "x2": _fine_edges[1:],
            "count": _fine_counts,
        })

        _bars = alt.Chart(_hist_df).mark_bar(color="#93c5fd", stroke="#bfdbfe", strokeWidth=0.5).encode(
            x=alt.X("x1:Q", title="Activation",
                    scale=alt.Scale(domain=[float(_fine_edges[0]), float(_fine_edges[-1])])),
            x2="x2:Q",
            y=alt.Y("count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("x1:Q", title="From", format=".3f"),
                alt.Tooltip("x2:Q", title="To", format=".3f"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )

        # Bin dividers from the slider
        _n = num_bins_slider.value
        _edges = np.linspace(curr_activations.min(), curr_activations.max(), _n + 1)
        _rule_df = pd.DataFrame({
            "x": list(_edges[1:-1]),
            "label": [f"Bin {i+1} / {i+2}" for i in range(_n - 1)],
        })
        _rules = alt.Chart(_rule_df).mark_rule(
            color="#1e3a5f", strokeWidth=2
        ).encode(
            x="x:Q",
            tooltip=[alt.Tooltip("x:Q", title="Boundary", format=".3f"), "label:N"],
        )

        # Bin number labels at top of chart
        _label_df = pd.DataFrame({
            "x": [(_edges[i] + _edges[i+1]) / 2 for i in range(_n)],
            "label": [str(i+1) for i in range(_n)],
        })
        _labels = alt.Chart(_label_df).mark_text(
            align="center", baseline="top", dy=4, fontSize=11,
            color="#1e3a5f", fontWeight="bold"
        ).encode(
            x="x:Q",
            text="label:N",
            y=alt.value(0),
        )

        _chart = (_bars + _rules + _labels).properties(
            width="container",
            height=220,
            title="Activation distribution with bin boundaries",
        ).configure_view(strokeWidth=0).configure_axis(grid=False)

        return _chart

    mo.ui.altair_chart(plot_activations(curr_activations))
    return (plot_activations,)


@app.cell(hide_code=True)
def _(
    context_slider,
    curr_activations,
    df_word,
    epochs,
    mo,
    np,
    num_bins_slider,
    pd,
    samples_per_bin_slider,
    seed_input,
):
    def _make_context(words_arr, idx, window):
        lo = max(0, idx - window)
        hi = min(len(words_arr), idx + window + 1)
        parts = []
        for i in range(lo, hi):
            w = words_arr[i]
            if i == idx:
                parts.append(f"**{w}**")   # bold the target word
            else:
                parts.append(w)
        return " ".join(parts)

    def _sample(activations, words, num_bins, samples_per_bin, context_window, seed):
        rng = np.random.default_rng(seed)
        words_arr = np.array(words)
        bin_edges = np.linspace(activations.min(), activations.max(), num_bins + 1)
        bin_edges[-1] += 1e-9

        rows = []
        for i in range(num_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (activations >= lo) & (activations < hi)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            chosen = rng.choice(
                indices, size=min(samples_per_bin, len(indices)), replace=False
            )
            for idx in chosen:
                rows.append({
                    "bin": i + 1,
                    "range": f"[{lo:.3f}, {hi:.3f})",
                    "activation": float(activations[idx]),
                    "word": words_arr[idx],
                    "context": _make_context(words_arr, idx, context_window),
                    "count_in_bin": int(mask.sum()),
                })
        return pd.DataFrame(rows)

    def generate_example_table(curr_activations):
        df = _sample(
            curr_activations,
            df_word.iloc[epochs.selection]["word"],
            num_bins=num_bins_slider.value,
            samples_per_bin=samples_per_bin_slider.value,
            context_window=context_slider.value,
            seed=int(seed_input.value),
        )

        if df.empty:
            output_data = mo.callout(mo.md("No data — check your inputs."), kind="warn")
        else:
            bin_tables = []
            for bin_idx, bin_group in df.groupby("bin"):
                rng_label = bin_group["range"].iloc[0]
                count = bin_group["count_in_bin"].iloc[0]
                pct = 100 * count / len(curr_activations)
                header = mo.md(
                    f"### Bin {bin_idx} &nbsp; `{rng_label}` "
                    f"&nbsp; <small>({count} words, {pct:.1f}%)</small>"
                )
                # Build an HTML table so columns are clearly separated
                row_html = ""
                for _, row in bin_group.sort_values("activation").iterrows():
                    # Highlight the target word in the context with a coloured span
                    context_html = row["context"].replace(
                        f"**{row['word']}**",
                        f'<mark style="background:#fde68a;padding:0 2px;border-radius:3px;">{row["word"]}</mark>'
                    )
                    row_html += f"""
                    <tr>
                      <td style="font-family:monospace;white-space:nowrap;padding:6px 12px;border-bottom:1px solid #e5e7eb;color:#6b7280;">{row['activation']:.4f}</td>
                      <td style="font-family:monospace;white-space:nowrap;padding:6px 12px;border-bottom:1px solid #e5e7eb;font-weight:600;">{row['word']}</td>
                      <td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;line-height:1.6;">{context_html}</td>
                    </tr>"""
                table_html = mo.Html(f"""
                <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
                  <thead>
                    <tr style="background:#f3f4f6;text-align:left;">
                      <th style="padding:8px 12px;border-bottom:2px solid #d1d5db;white-space:nowrap;">Activation</th>
                      <th style="padding:8px 12px;border-bottom:2px solid #d1d5db;white-space:nowrap;">Word</th>
                      <th style="padding:8px 12px;border-bottom:2px solid #d1d5db;">Context</th>
                    </tr>
                  </thead>
                  <tbody>{row_html}</tbody>
                </table>""")
                bin_tables.append(mo.vstack([header, table_html]))

            output_data = mo.vstack(bin_tables, gap="1.5rem")
            return output_data

    mo.vstack([
        mo.md("# Activation Distribution Explorer"),
        mo.md(
            "Samples words from across the activation distribution. "
            "The target word is **bolded** in its context window."
        ),
        # mo.divider(),
        generate_example_table(curr_activations),
    ])
    return (generate_example_table,)


@app.cell(hide_code=True)
def _(lags, mo):
    # Take the mean over activations over a wider window of timepoints to try and reduce the effects of noise.
    lower_timepoint_slider = mo.ui.slider(
        steps=lags, label="Lower timepoint", show_value=True
    )
    upper_timepoint_slider = mo.ui.slider(
        steps=lags, label="Lower timepoint", show_value=True
    )

    mo.hstack([lower_timepoint_slider, upper_timepoint_slider])
    return lower_timepoint_slider, upper_timepoint_slider


@app.cell
def _(
    all_predictions,
    electrode_examples_dropdown,
    lags,
    lower_timepoint_slider,
    np,
    plot_activations,
    upper_timepoint_slider,
):
    lower_bound = np.where(lags == lower_timepoint_slider.value)[0][0]
    upper_bound = np.where(lags == upper_timepoint_slider.value)[0][0]
    mean_activations = all_predictions[:, electrode_examples_dropdown.value, lower_bound:upper_bound].mean(1)

    plot_activations(mean_activations)
    return (mean_activations,)


@app.cell(hide_code=True)
def _(generate_example_table, mean_activations, mo):
    mo.vstack([
        mo.md("# Activation Distribution Explorer"),
        mo.md(
            "Samples words from across the activation distribution. "
            "The target word is **bolded** in its context window."
        ),
        # mo.divider(),
        generate_example_table(mean_activations),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Dataset Activations
    """)
    return


@app.cell(hide_code=True)
def _(electrode_labels, lags, mo):
    electrode_main_examples_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode main",
    )

    timepoint_main_examples_dropdown = mo.ui.dropdown(
        options={str(lags[i]): i for i in range(len(lags))},
        value=str(lags[0]),
        label="timepoint main",
    )

    electrode_alternative_examples_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode alt",
    )

    timepoint_alternative_examples_dropdown = mo.ui.dropdown(
        options={str(lags[i]): i for i in range(len(lags))},
        value=str(lags[0]),
        label="timepoint alt",
    )

    mo.hstack([electrode_main_examples_dropdown, timepoint_main_examples_dropdown, electrode_alternative_examples_dropdown, timepoint_alternative_examples_dropdown])
    return (
        electrode_alternative_examples_dropdown,
        electrode_main_examples_dropdown,
        timepoint_alternative_examples_dropdown,
        timepoint_main_examples_dropdown,
    )


@app.cell
def _(
    all_predictions,
    electrode_alternative_examples_dropdown,
    electrode_main_examples_dropdown,
    plot_correlations,
    timepoint_alternative_examples_dropdown,
    timepoint_main_examples_dropdown,
):
    plot_correlations(all_predictions[:, electrode_main_examples_dropdown.value, timepoint_main_examples_dropdown.value], all_predictions[:, electrode_alternative_examples_dropdown.value, timepoint_alternative_examples_dropdown.value])
    return


@app.cell(hide_code=True)
def _(plt, torch):
    import seaborn as sns

    def plot_correlation_heatmap(array_a, array_b):
        def standardize(x):
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            std[std == 0] = 1
            return (x - mean) / std

        a = standardize(array_a)  # [5230, 16000]
        b = standardize(array_b)  # [5230, 16000]

        corr_matrix = (a.T @ b) / array_a.shape[0]  # [16000, 16000]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Pearson Correlation")

        ax.set_xticks(range(array_b.shape[1]))
        ax.set_yticks(range(array_a.shape[1]))
        ax.set_xticklabels([f"B_{j}" for j in range(array_b.shape[1])])
        ax.set_yticklabels([f"A_{i}" for i in range(array_a.shape[1])])
        ax.set_xlabel("Array B Samples")
        ax.set_ylabel("Array A Samples")
        ax.set_title("Cross-Correlation Heatmap (Array A vs Array B)")

        plt.tight_layout()
        plt.show()

        return corr_matrix

    def get_correlation_gpu(array_a, array_b):
        a = torch.tensor(array_a, device='cuda', dtype=torch.float32)
        b = torch.tensor(array_b, device='cuda', dtype=torch.float32)

        # Standardize on GPU
        a = (a - a.mean(dim=0)) / a.std(dim=0)
        b = (b - b.mean(dim=0)) / b.std(dim=0)

        corr_matrix = (a.T @ b) / array_a.shape[0]
        return corr_matrix.cpu().numpy()

    return get_correlation_gpu, plot_correlation_heatmap


@app.cell
def _(all_predictions, plot_correlation_heatmap):
    elec_vs_self_corrs = plot_correlation_heatmap(all_predictions[:, 2], all_predictions[:, 2])
    return (elec_vs_self_corrs,)


@app.cell
def _(elec_vs_self_corrs, lags, plt, timepoint_main_examples_dropdown):
    plt.plot(lags, elec_vs_self_corrs[timepoint_main_examples_dropdown.value])
    plt.axvline(lags[timepoint_main_examples_dropdown.value], c=(0.9, 0.9, 0.9), ls="--", label="Reference timepoint")
    return


@app.cell
def _(all_predictions, plot_correlation_heatmap):
    plot_correlation_heatmap(all_predictions[:, 8], all_predictions[:, 77])
    return


@app.cell
def _(all_predictions, get_correlation_gpu):
    # Compute correlations between activations across all times and electrodes.
    all_preds_elec = all_predictions
    all_activations = all_preds_elec.reshape((all_preds_elec.shape[0], -1))
    all_predictions_corrs = get_correlation_gpu(all_activations, all_activations)
    return (all_predictions_corrs,)


@app.cell
def _(all_predictions_corrs):
    all_predictions_corrs.shape
    return


@app.cell
def _(all_predictions_corrs, np, plt):
    tril_all_prediction_corrs = np.tril(all_predictions_corrs, -1)
    plt.hist(1 - tril_all_prediction_corrs[tril_all_prediction_corrs != 0])
    return


@app.cell
def _(all_predictions_corrs, linkage, squareform):
    dist_matrix = 1 - all_predictions_corrs  # values in [0, 2]
    # or: dist_matrix = 1 - np.abs(corr_matrix)  # if negative correlations should also cluster

    # squareform converts square matrix to condensed upper-triangle form
    condensed = squareform(dist_matrix, checks=False)

    # Linkage methods: 'average' or 'ward' work well for correlation data
    Z_corr = linkage(condensed, method='average')
    return (Z_corr,)


@app.cell
def _(Z_corr, fcluster):
    # Cut the tree — tune t based on your expectations
    labels_corr = fcluster(Z_corr, t=0.85, criterion='distance')  # t is the distance threshold
    return (labels_corr,)


@app.cell
def _(datas, labels_corr, np, sig):
    electrode_hier_labels = labels_corr.reshape(datas.shape[1:])
    significant_labels = - np.ones_like(electrode_hier_labels)
    significant_labels[sig] = electrode_hier_labels[sig]
    # # Filter down to just the classes that were found in significant lags
    unique_labels = {x: i for i, x in enumerate(np.unique(significant_labels))}
    assert unique_labels[-1] == 0
    significant_labels = np.array([unique_labels[x] - 1  for elec in significant_labels for x in elec]).reshape(electrode_hier_labels.shape)
    significant_labels
    return (significant_labels,)


@app.cell
def _(lags, np, plt, significant_labels):
    plt.imshow(significant_labels)
    plt.colorbar()
    plt.xticks(ticks=np.arange(significant_labels.shape[1], step=20), labels=lags[::20])

    plt.show()
    return


@app.cell
def _(Counter, plt, significant_labels):
    class_counts = Counter(significant_labels[significant_labels != -1].flatten())
    plt.bar(class_counts.keys(), class_counts.values())
    plt.show()
    return (class_counts,)


@app.cell(hide_code=True)
def _(class_counts):
    sorted_class_counts = sorted(class_counts.items(), key=lambda item: -item[1])
    sorted_class_counts
    return (sorted_class_counts,)


@app.cell(hide_code=True)
def _(Z_corr, dendrogram, plt):
    plt.figure(figsize=(14, 6))
    dendrogram(
        Z_corr,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=30,                   # show last 30 merges
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index (or cluster size)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(electrode_labels, mo):
    electrode_clusters_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    electrode_clusters_dropdown
    return (electrode_clusters_dropdown,)


@app.cell(hide_code=True)
def _(electrode_clusters_dropdown, lags, plt, significant_labels):
    plt.plot(lags, significant_labels[electrode_clusters_dropdown.value])
    return


@app.cell(hide_code=True)
def _(all_predictions, np, significant_labels):
    all_class_mean_activations = []
    for cluster_id in np.unique(significant_labels):
        if cluster_id == -1:
            continue
        all_class_mean_activations.append(all_predictions[:, significant_labels == cluster_id].mean(1))
    all_class_mean_activations = np.column_stack(all_class_mean_activations).transpose(1, 0)
    all_class_mean_activations
    return (all_class_mean_activations,)


@app.cell
def _(
    all_class_mean_activations,
    plot_correlation_heatmap,
    sorted_class_counts,
):
    sorted_classes = [x[0] for x in sorted_class_counts]
    plot_correlation_heatmap(all_class_mean_activations.T[sorted_classes], all_class_mean_activations.T[sorted_classes])
    return


@app.cell(hide_code=True)
def _(mo, sorted_class_counts):
    class_dropdown = mo.ui.dropdown(
        options={f"{class_id}:{class_count}": class_id for class_id, class_count in sorted_class_counts},
        value=":".join([str(x) for x in list(sorted_class_counts)[0]]),
        label="Class",
    )

    class_dropdown
    return (class_dropdown,)


@app.cell(hide_code=True)
def _(np, plt):
    def plot_class_distributions(
        data: np.ndarray,
        target_class: int,
        elecs_per_subject: list[int] | None = None,
    ) -> None:
        """
        Plot the distribution of a target class across electrodes, timepoints,
        and (optionally) subjects.

        Args:
            data:               Array of shape [n_elecs, n_timepoints] of class labels
            target_class:       The class integer to analyze
            elecs_per_subject:  List where each entry is the number of electrodes
                                belonging to that subject, in order. Must sum to n_elecs.
                                Subjects with 0 electrodes are skipped.
                                e.g. [3, 4, 0, 1] → subjects 0-3 own elecs [0:3, 3:7, -, 7:8]
        """
        n_elecs, n_timepoints = data.shape

        class_mask = (data == target_class)
        total_occurrences = class_mask.sum()
        if total_occurrences == 0:
            print(f"Class {target_class} not found in data.")
            return

        # ── Per-electrode probabilities ───────────────────────────────────────────
        elec_counts = class_mask.sum(axis=1)
        elec_probs  = elec_counts / total_occurrences

        # ── Per-timepoint probabilities ───────────────────────────────────────────
        time_counts = class_mask.sum(axis=0)
        time_probs  = time_counts / total_occurrences

        # ── Parse subjects ────────────────────────────────────────────────────────
        # Build a list of (label, electrode_slice) for subjects with >0 electrodes
        subjects = []   # list of (label, slice)
        if elecs_per_subject is not None:
            if sum(elecs_per_subject) != n_elecs:
                raise ValueError(
                    f"elecs_per_subject sums to {sum(elecs_per_subject)}, "
                    f"but data has {n_elecs} electrodes."
                )
            cursor = 0
            for subj_idx, n in enumerate(elecs_per_subject):
                if n > 0:
                    subjects.append((f"S{subj_idx + 1}", slice(cursor, cursor + n)))
                cursor += n

        # ── Per-subject-per-timepoint probabilities ───────────────────────────────
        # shape: [n_active_subjects, n_timepoints]
        if subjects:
            subj_time_probs = np.array([
                class_mask[sl].sum(axis=0) / total_occurrences
                for _, sl in subjects
            ])
            subj_labels = [lbl for lbl, _ in subjects]

            # Per-subject totals (for the subject bar chart)
            subj_probs = subj_time_probs.sum(axis=1)

            colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))

        # ── Layout ────────────────────────────────────────────────────────────────
        n_plots = 3 if subjects else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        fig.suptitle(f"Class {target_class} — probability distributions", fontsize=14)

        pct = plt.FuncFormatter(lambda y, _: f"{y:.1%}")
        x_elec = np.arange(n_elecs)
        x_time = np.arange(n_timepoints)

        # ── Electrode distribution ─────────────────────────────────────────────────
        ax = axes[0]
        ax.bar(x_elec, elec_probs, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Electrode index")
        ax.set_ylabel("Probability")
        ax.set_title("Distribution across electrodes")
        ax.set_xlim(-0.5, n_elecs - 0.5)
        ax.yaxis.set_major_formatter(pct)

        # ── Timepoint distribution (stacked by subject if available) ──────────────
        ax = axes[1]
        if subjects:
            bottoms = np.zeros(n_timepoints)
            for i, (label, color) in enumerate(zip(subj_labels, colors)):
                ax.bar(
                    x_time, subj_time_probs[i],
                    bottom=bottoms,
                    color=color, edgecolor="none", linewidth=0,
                    label=label, width=1.0,
                )
                bottoms += subj_time_probs[i]
            ax.legend(title="Subject", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        else:
            ax.bar(x_time, time_probs, color="coral", edgecolor="white", linewidth=0.4)

        ax.set_xlabel("Timepoint index")
        ax.set_ylabel("Probability")
        ax.set_title("Distribution across timepoints")
        ax.set_xlim(-0.5, n_timepoints - 0.5)
        ax.yaxis.set_major_formatter(pct)

        # ── Subject distribution ───────────────────────────────────────────────────
        if subjects:
            ax = axes[2]
            x_subj = np.arange(len(subjects))
            bars = ax.bar(
                x_subj, subj_probs,
                color=colors, edgecolor="white", linewidth=0.4,
            )
            ax.set_xlabel("Subject")
            ax.set_ylabel("Probability")
            ax.set_title("Distribution across subjects")
            ax.set_xticks(x_subj)
            ax.set_xticklabels(subj_labels)
            ax.yaxis.set_major_formatter(pct)

            for bar, prob in zip(bars, subj_probs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(subj_probs) * 0.01,
                    f"{prob:.1%}",
                    ha="center", va="bottom", fontsize=8,
                )

        plt.tight_layout()
        plt.show()

    return (plot_class_distributions,)


@app.cell(hide_code=True)
def _(raws):
    subject_electrode_splits = [len(r.ch_names) for r in raws]
    subject_electrode_splits.insert(1, 0)
    return (subject_electrode_splits,)


@app.cell
def _(
    class_dropdown,
    plot_class_distributions,
    significant_labels,
    subject_electrode_splits,
):
    plot_class_distributions(significant_labels, class_dropdown.value, elecs_per_subject=subject_electrode_splits)
    return


@app.cell(hide_code=True)
def _(all_class_mean_activations, class_dropdown, plot_activations):
    class_mean_activations = all_class_mean_activations[class_dropdown.value]
    plot_activations(class_mean_activations)
    return (class_mean_activations,)


@app.cell(hide_code=True)
def _(class_mean_activations, generate_example_table, mo):
    mo.vstack([
        mo.md("# Activation Distribution Explorer"),
        mo.md(
            "Samples words from across the activation distribution. "
            "The target word is **bolded** in its context window."
        ),
        # mo.divider(),
        generate_example_table(class_mean_activations),
    ])
    return


@app.cell
def _(class_mean_activations, plot_correlations, selected_df):
    # Identify which clusters best align with explainable variables.
    plot_correlations(class_mean_activations, selected_df["true_prob"])
    return


@app.cell
def _(X_kl_divergence, class_mean_activations, plot_correlations):
    plot_correlations(class_mean_activations, X_kl_divergence[:, 1])
    return


@app.cell
def _(class_mean_activations, plot_correlations, selected_df):
    plot_correlations(class_mean_activations, selected_df["entropy"])
    return


@app.cell
def _():
    # TODO: Add way to measure how much variance a given cluster explains in the real data as a way of measuring that clusters performance
    return


@app.cell
def _():
    # TODO: add a way to measure how much a cluster's performance is driven by different segments of its distribution
    return


@app.cell(hide_code=True)
def _(all_class_mean_activations, np, selected_df, stats):
    def correlate_classes_with_corollary(
            class_activations: np.ndarray,
            corollary: np.ndarray,
            class_names: list[str] | None = None,
            method: str = "pearson",
        ) -> list[dict]:
            """
            Correlate each row of class_activations (shape [n_classes, n_words])
            with corollary (shape [n_words]).

            Parameters
            ----------
            class_activations : np.ndarray  shape (n_classes, n_words)
            corollary         : np.ndarray  shape (n_words,)
            class_names       : optional list of n_classes strings
            method            : "pearson" | "spearman" | "kendall"

            Returns
            -------
            List of dicts sorted by |r|, each with keys:
                rank, name, r, p_value, abs_r, direction
            """
            if class_names is None:
                class_names = [f"Class {i}" for i in range(class_activations.shape[0])]

            corr_fn = {
                "pearson":  stats.pearsonr,
                "spearman": stats.spearmanr,
                "kendall":  stats.kendalltau,
            }[method]

            results = []
            n_classes = class_activations.shape[0]
            for i in range(n_classes):
                r, p = corr_fn(class_activations[i], corollary)
                results.append(
                    dict(
                        index=i,
                        name=class_names[i],
                        r=float(r),
                        p_value=float(p),
                        abs_r=abs(float(r)),
                        direction="positive" if r >= 0 else "negative",
                    )
                )

            results.sort(key=lambda d: d["abs_r"], reverse=True)
            for rank, d in enumerate(results, 1):
                d["rank"] = rank

            return results

    results = correlate_classes_with_corollary(
            all_class_mean_activations, selected_df["true_prob"], method="pearson"
        )
    return correlate_classes_with_corollary, results


@app.cell(hide_code=True)
def _(mo, results):
    top_n_slider = mo.ui.slider(
        start=5, stop=len(results), step=1, value=min(15, len(results)),
        label="Top N classes to display",
    )
    method_dropdown = mo.ui.dropdown(
        options=["pearson", "spearman", "kendall"],
        value="pearson",
        label="Correlation method",
    )
    sig_threshold = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.05,
        label="p-value significance threshold",
    )
    mo.hstack([top_n_slider, method_dropdown, sig_threshold], gap=2)
    return method_dropdown, sig_threshold, top_n_slider


@app.cell(hide_code=True)
def _(
    all_class_mean_activations,
    correlate_classes_with_corollary,
    method_dropdown,
    mo,
    np,
    plt,
    sig_threshold,
    top_n_slider,
):
    import matplotlib.cm as cm
    # Recompute if method changes
    def plot_class_activation_with_corrolary(corollary):
        live_results = correlate_classes_with_corollary(
            all_class_mean_activations, corollary, method=method_dropdown.value
        )
        top = live_results[: top_n_slider.value]

        names  = [d["name"]  for d in top]
        rs     = [d["r"]     for d in top]
        pvals  = [d["p_value"] for d in top]
        abs_rs = [d["abs_r"] for d in top]

        # Colour: blue = positive, red = negative; alpha encodes significance
        cmap_pos = cm.get_cmap("Blues")
        cmap_neg = cm.get_cmap("Reds")
        colors = []
        for r_val, p_val in zip(rs, pvals):
            intensity = 0.35 + 0.65 * abs(r_val)   # never fully white
            base = cmap_pos(intensity) if r_val >= 0 else cmap_neg(intensity)
            alpha = 1.0 if p_val <= sig_threshold.value else 0.35
            colors.append((*base[:3], alpha))

        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(top) * 0.42)),
                                 gridspec_kw={"width_ratios": [3, 1]})
        fig.patch.set_facecolor("#0f1117")
        for ax in axes:
            ax.set_facecolor("#0f1117")

        # ── Left: horizontal bar chart ──
        ax = axes[0]
        y_pos = np.arange(len(top))[::-1]   # rank 1 at the top
        bars = ax.barh(y_pos, rs, color=colors, edgecolor="none", height=0.7)
        ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9, color="#e0e0e0")
        ax.set_xlabel("Correlation coefficient (r)", color="#aaa", fontsize=10)
        ax.set_title(
            f"Top {len(top)} classes by |r| with corollary  [{method_dropdown.value.title()}]",
            color="white", fontsize=12, pad=10,
        )
        ax.set_xlim(-1.05, 1.05)
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        # Annotate bars with r and p
        for bar, r_val, p_val in zip(bars, rs, pvals):
            sig_star = "*" if p_val <= sig_threshold.value else ""
            label = f" {r_val:+.3f}{sig_star}"
            ax.text(
                r_val + (0.02 if r_val >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left" if r_val >= 0 else "right",
                fontsize=8,
                color="white",
            )

        # ── Right: p-value lollipop ──
        ax2 = axes[1]
        ax2.scatter(pvals, y_pos, color=["#4fc3f7" if p <= sig_threshold.value else "#555"
                                          for p in pvals], zorder=3, s=50)
        for yi, pv in zip(y_pos, pvals):
            ax2.plot([0, pv], [yi, yi], color="#333", linewidth=1, zorder=2)
        ax2.axvline(sig_threshold.value, color="#f4a261", linewidth=1,
                    linestyle="--", label=f"p={sig_threshold.value}")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([], visible=False)
        ax2.set_xlabel("p-value", color="#aaa", fontsize=10)
        ax2.set_title("Significance", color="white", fontsize=12, pad=10)
        ax2.tick_params(colors="#aaa")
        ax2.legend(fontsize=8, facecolor="#222", labelcolor="#aaa", edgecolor="#333")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")

        plt.tight_layout(pad=1.5)
        return mo.mpl.interactive(fig)

    return (plot_class_activation_with_corrolary,)


@app.cell
def _(plot_class_activation_with_corrolary, selected_df):
    plot_class_activation_with_corrolary(selected_df["true_prob"])
    return


@app.cell
def _(plot_class_activation_with_corrolary, selected_df):
    plot_class_activation_with_corrolary(selected_df["entropy"])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Partialing out Clusters
    """)
    return


@app.cell(hide_code=True)
def _(mo, sorted_class_counts):
    cluster_partial_dropdown = mo.ui.dropdown(
        options={f"{class_id}:{class_count}": class_id for class_id, class_count in sorted_class_counts},
        value=":".join([str(x) for x in list(sorted_class_counts)[0]]),
        label="Class",
    )

    use_bias_checkbox = mo.ui.checkbox(label="Use Bias")

    mo.hstack([cluster_partial_dropdown, use_bias_checkbox])
    return cluster_partial_dropdown, use_bias_checkbox


@app.cell(hide_code=True)
def _(
    cluster_partial_dropdown,
    mean_model_bias,
    mean_model_coefs,
    model_1_bias,
    model_1_coefs,
    model_2_bias,
    model_2_coefs,
    np,
    significant_labels,
    use_bias_checkbox,
):
    # Get the mean encoding embedding for a cluster. Its direction should align with the mean activations
    def get_cluster_mean_direction(model_coefs, cluster_id, bias=None):
        cluster_coefs = model_coefs[significant_labels == cluster_id]
        if bias is not None:
            cluster_bias = bias[(significant_labels == cluster_id).flatten()]
            cluster_coefs = np.column_stack([cluster_coefs, cluster_bias])
        return cluster_coefs.mean(0)

    cluster_mean_1 = get_cluster_mean_direction(model_1_coefs.numpy(), cluster_partial_dropdown.value, bias=model_1_bias if use_bias_checkbox.value else None)
    cluster_mean_2 = get_cluster_mean_direction(model_2_coefs.numpy(), cluster_partial_dropdown.value, bias=model_2_bias if use_bias_checkbox.value else None)
    cluster_mean = get_cluster_mean_direction(mean_model_coefs.numpy(), cluster_partial_dropdown.value, bias=mean_model_bias if use_bias_checkbox.value else None)
    # cluster_mean = (cluster_mean_1 + cluster_mean_2) / 2
    return (
        cluster_mean,
        cluster_mean_1,
        cluster_mean_2,
        get_cluster_mean_direction,
    )


@app.cell(hide_code=True)
def _(
    X,
    all_class_mean_activations,
    cluster_mean_1,
    cluster_mean_2,
    cluster_partial_dropdown,
    encoding_models,
    encoding_test_idxs,
    np,
    plot_correlations,
    use_bias_checkbox,
):
    # Ensure that the cluster mean correlates exactly with the averaged activations for that cluster
    X_test_1 = encoding_models[0].named_steps["standardscaler"].transform(X[encoding_test_idxs[0]])
    X_test_2 = encoding_models[1].named_steps["standardscaler"].transform(X[encoding_test_idxs[1]])
    X_bias_1 = np.column_stack([X_test_1, np.ones(X_test_1.shape[0])]) if use_bias_checkbox.value else X_test_1
    X_bias_2 = np.column_stack([X_test_2, np.ones(X_test_2.shape[0])]) if use_bias_checkbox.value else X_test_2

    first_set_means = X_bias_1 @ cluster_mean_1
    second_set_means = X_bias_2 @ cluster_mean_2
    mean_dir_cluster_preds = np.concat([first_set_means, second_set_means])
    # if use_bias_checkbox.value:
    #     mean_dir_cluster_preds = mean_dir_cluster_preds.mean(1)
    curr_class_mean_activations = all_class_mean_activations[cluster_partial_dropdown.value]
    plot_correlations(mean_dir_cluster_preds, curr_class_mean_activations)
    return (
        X_bias_1,
        X_bias_2,
        curr_class_mean_activations,
        mean_dir_cluster_preds,
    )


@app.cell(hide_code=True)
def _(
    X_bias_1,
    X_bias_2,
    cluster_mean,
    curr_class_mean_activations,
    np,
    plot_correlations,
    use_bias_checkbox,
):
    X_bias = np.concat([X_bias_1, X_bias_2])
    curr_mean_model_dir = cluster_mean if use_bias_checkbox.value else cluster_mean
    mean_coefs_cluster_preds = X_bias @ curr_mean_model_dir
    # if use_bias_checkbox.value:
    #     mean_coefs_cluster_preds = mean_coefs_cluster_preds.mean(1)
    plot_correlations(mean_coefs_cluster_preds, curr_class_mean_activations)
    return


@app.cell(hide_code=True)
def _(
    cluster_partial_dropdown,
    get_cluster_mean_direction,
    mean_raw_coefs,
    mean_raw_intercept,
    raw_coefs_1,
    raw_coefs_2,
    raw_intercept_1,
    raw_intercept_2,
    use_bias_checkbox,
):
    raw_cluster_means_1 = get_cluster_mean_direction(raw_coefs_1, cluster_partial_dropdown.value, bias=raw_intercept_1 if use_bias_checkbox.value else None)
    raw_cluster_means_2 = get_cluster_mean_direction(raw_coefs_2, cluster_partial_dropdown.value, bias=raw_intercept_2 if use_bias_checkbox.value else None)
    raw_mean_cluster_mean = get_cluster_mean_direction(mean_raw_coefs, cluster_partial_dropdown.value, bias=mean_raw_intercept if use_bias_checkbox.value else None)
    return raw_cluster_means_1, raw_cluster_means_2, raw_mean_cluster_mean


@app.cell(hide_code=True)
def _(
    X,
    curr_class_mean_activations,
    encoding_test_idxs,
    mean_dir_cluster_preds,
    np,
    plot_correlations,
    raw_cluster_means_1,
    raw_cluster_means_2,
    use_bias_checkbox,
):
    # Ensure that the cluster mean correlates exactly with the averaged activations for that cluster
    X_raw_test_1 = X[encoding_test_idxs[0]]
    X_raw_test_2 = X[encoding_test_idxs[1]]
    X_raw_bias_1 = np.column_stack([X_raw_test_1, np.ones(X_raw_test_1.shape[0])]) if use_bias_checkbox.value else X_raw_test_1
    X_raw_bias_2 = np.column_stack([X_raw_test_2, np.ones(X_raw_test_2.shape[0])]) if use_bias_checkbox.value else X_raw_test_2

    first_set_raw_means = X_raw_bias_1 @ raw_cluster_means_1
    second_set_raw_means = X_raw_bias_2 @ raw_cluster_means_2
    mean_raw_dir_cluster_preds = np.concat([first_set_raw_means, second_set_raw_means])
    # if use_bias_checkbox.value:
    #     mean_dir_cluster_preds = mean_dir_cluster_preds.mean(1)
    plot_correlations(mean_dir_cluster_preds, curr_class_mean_activations)
    return X_raw_bias_1, X_raw_bias_2


@app.cell(hide_code=True)
def _(
    X_raw_bias_1,
    X_raw_bias_2,
    curr_class_mean_activations,
    np,
    plot_correlations,
    raw_mean_cluster_mean,
):
    X_raw_bias = np.concat([X_raw_bias_1, X_raw_bias_2])
    raw_mean_coefs_cluster_preds = X_raw_bias @ raw_mean_cluster_mean
    plot_correlations(raw_mean_coefs_cluster_preds, curr_class_mean_activations)
    return


@app.cell(hide_code=True)
def _(np):
    def partial_out_direction(X, Y, directions, predictions, fold_indices, scalers):
        """
        Remove variance explained by a direction in embedding space from X and Y.

        Parameters
        ----------
        X : array of shape [n_words, n_dim]
            Word embeddings.
        Y : array of shape [n_words, n_elecs * n_timepoints]
            Neural data (raw, not z-scored).
        directions : list of array, each of shape [n_dim]
            Direction to partial out, one per fold.
        predictions : array [n_words, n_elecs * n_timepoints]
        fold_indices : list of array
            Test indices for each fold.
        scalers : list of StandardScaler
            The fitted scaler from each fold's training set.

        Returns
        -------
        X_residuals : list of array, each of shape [n_words, n_dim]
            X with the component along the direction removed, one per fold.
        Y_residual : array of shape [n_words, n_elecs * n_timepoints]
            Y with the out-of-sample predicted variance subtracted in raw space.
        """
        # Inverse-transform each fold's predictions back to raw space, then stitch
        Y_pred_full = np.zeros_like(Y)
        for idx, scaler in zip(fold_indices, scalers):
            Y_pred_full[idx] = scaler.inverse_transform(predictions[idx])

        Y_residual = Y - Y_pred_full

        # Remove the direction from X per fold (since direction differs per fold)
        X_residuals = []
        for direction in directions:
            d = direction / np.linalg.norm(direction)
            projections = X @ d
            X_res = X - np.outer(projections, d)
            X_residuals.append(X_res)

        return X_residuals, Y_residual

    return (partial_out_direction,)


@app.cell
def _():
    # def encode_and_cluster_encoders(X, Y, num_electrodes, num_timepoints, partial_clusters=[], folds=None):
    #     # TODO: Should I reuse scalers?
    #     corrs_embedding, encoding_models, encoding_tr_idxs, encoding_test_idxs, encoding_scalers = train_encoding(X, Y, create_model, provided_idxs=folds)
    #     all_predictions, all_targets = get_all_predictions_and_targets_test_set(encoding_models, encoding_test_idxs, X, Y, num_electrodes, num_timepoints)
    #     sig, p_values, q_values = get_sig_results(partial_clusters, all_predictions, all_targets)
    
    #     all_activations = all_predictions.reshape((all_predictions.shape[0], -1))
    #     all_predictions_corrs = get_correlation_gpu(all_activations, all_activations)

    #     dist_matrix = 1 - all_predictions_corrs  # values in [0, 2]
    
    #     # squareform converts square matrix to condensed upper-triangle form
    #     condensed = squareform(dist_matrix, checks=False)
    
    #     # Linkage methods: 'average' or 'ward' work well for correlation data
    #     Z_corr = linkage(condensed, method='average')

    #     return {
    #         "Z_cluster": Z_corr,
    #         "encoding_corrs": corrs_embedding,
    #         "encoding_models": encoding_models,
    #         "train_idxs": encoding_tr_idxs,
    #         "test_idxs": encoding_test_idxs,
    #     }

    # def 
    return


@app.cell
def _(
    X,
    Y,
    all_predictions,
    encoding_scalers,
    encoding_test_idxs,
    partial_out_direction,
    raw_cluster_means_1,
    raw_cluster_means_2,
):
    # TODO: Fix this
    X_residuals, Y_residuals = partial_out_direction(X, Y, [raw_cluster_means_1, raw_cluster_means_2], all_predictions.reshape(all_predictions.shape[0], -1), encoding_test_idxs, encoding_scalers)
    X_residuals, Y_residuals
    return (X_residuals,)


@app.cell
def _(
    X,
    X_residuals,
    Y,
    create_model,
    encoding_test_idxs,
    encoding_tr_idxs,
    train_encoding,
):
    def make_modify_inputs_fn(X_residuals):
        """
        Returns a modify_inputs_fn that swaps in the pre-computed
        X residual for each fold, indexed into the right rows.
        """
        def modify_inputs_fn(X_subset, fold_idx, idx=None):
            return X_residuals[fold_idx][idx]
        return modify_inputs_fn

    corrs_partialed_dir, partialed_encoding_models, _, _, _ = train_encoding(X, Y, create_model, modify_inputs_fn=make_modify_inputs_fn(X_residuals), provided_idxs=list(zip(encoding_tr_idxs, encoding_test_idxs)))
    print(f"Encoding performance correlating matrix shape: {corrs_partialed_dir.shape}")
    return (corrs_partialed_dir,)


@app.cell
def _(
    corrs_arb,
    corrs_embedding,
    corrs_final_layer_embedding,
    corrs_partialed_dir,
    plot_embedding_correlations,
):
    plot_embedding_correlations(plot_corrs=[
        {"corrs": corrs_embedding, "color": "black", "label": "Middle Layer"},
        {"corrs": corrs_partialed_dir, "label": "Partialed Out", "color": "red"},
        {"corrs": corrs_arb, "color": "teal", "label": "Arbitrary"},
        {"corrs": corrs_final_layer_embedding, "color": "green", "label": "Final Layer"},
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cluster Analysis
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(electrode_labels, electrodes_ordered, lags, mo):
    electrode_time_cluster_dropdown = mo.ui.dropdown(
        options={label: i for label, i in electrode_labels.items()},
        value=list(electrode_labels.keys())[0],
        label="Electrode",
    )

    electrode_or_timepoint_dropdown = mo.ui.dropdown(
        options={"electrode": "elec", "timepoint": "time"},
        value="electrode",
        label="Electrode or timepoint",
    )

    timepoint_cluster_slider = mo.ui.slider(
        steps=lags,
        label="Timepoint",
        show_value=True
    )

    top_n_elecs_cluster_slider = mo.ui.slider(
        steps=list(range(len(electrodes_ordered))),
        label="Top N Electrodes",
        show_value=True
    )

    mo.hstack([electrode_or_timepoint_dropdown, electrode_time_cluster_dropdown, timepoint_cluster_slider, top_n_elecs_cluster_slider])
    return (
        electrode_or_timepoint_dropdown,
        electrode_time_cluster_dropdown,
        timepoint_cluster_slider,
        top_n_elecs_cluster_slider,
    )


@app.cell
def _(
    KMeans,
    electrode_or_timepoint_dropdown,
    electrode_time_cluster_dropdown,
    electrodes_ordered,
    lags,
    model_1_coefs,
    model_2_coefs,
    np,
    plt,
    timepoint_cluster_slider,
    top_n_elecs_cluster_slider,
):
    from sklearn.metrics import silhouette_score

    curr_model_coefs_cluster = (model_1_coefs + model_2_coefs) / 2
    # curr_model_coefs_cluster = curr_model_coefs_cluster.numpy()[electrodes_ordered[:30]]
    # curr_model_coefs_cluster = curr_model_coefs_cluster.reshape((-1, curr_model_coefs_cluster.shape[-1]))

    if electrode_or_timepoint_dropdown.value == "elec":
        curr_model_coefs_cluster = curr_model_coefs_cluster[electrode_time_cluster_dropdown.value]
    else:
        curr_lag = np.where(lags == timepoint_cluster_slider.value)[0][0]
        curr_model_coefs_cluster = curr_model_coefs_cluster[:, curr_lag]
        curr_model_coefs_cluster = curr_model_coefs_cluster.numpy()[electrodes_ordered[:top_n_elecs_cluster_slider.value]]

    norms = np.linalg.norm(curr_model_coefs_cluster, axis=1, keepdims=True)
    curr_model_coefs_cluster /= norms

    scores = []
    inertias = []
    ks = list(range(2, min(curr_model_coefs_cluster.shape[0], 100)))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42).fit(curr_model_coefs_cluster)
        curr_labels = km.predict(curr_model_coefs_cluster)
        scores.append(silhouette_score(curr_model_coefs_cluster, curr_labels, metric='cosine'))
        inertias.append(km.inertia_)

    plt.plot(ks, scores)
    plt.axvline(np.argmax(scores) + min(ks), label=f"k={np.argmax(scores) + min(ks)}",  c=(0.9, 0.9, 0.9), ls="--")
    plt.legend()
    plt.show()

    plt.plot(ks, inertias, marker='o')
    plt.legend()
    plt.show()
    return curr_model_coefs_cluster, ks, scores, silhouette_score


@app.cell
def _(KMeans, curr_model_coefs_cluster, ks, np, scores):
    kmeans = KMeans(n_clusters=np.argmax(scores) + min(ks), random_state=42)
    labels = kmeans.fit_predict(curr_model_coefs_cluster)

    # plt.plot(lags, labels)
    # plt.show()
    return (labels,)


@app.cell
def _(curr_model_coefs_cluster, labels, plt, umap):
    reducer = umap.UMAP(
        n_components=2,      # output dimensions (2 for visualization)
        n_neighbors=15,      # controls local vs global structure (typical range: 5–50)
        min_dist=0.1,        # how tightly points cluster (0.0–1.0)
        metric='cosine',  # or 'cosine', 'manhattan', etc.
        random_state=42
    )

    embedding = reducer.fit_transform(curr_model_coefs_cluster)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)
    plt.title("UMAP Projection")
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(TSNE, curr_model_coefs_cluster, labels, plt):
    tsne_reducer = TSNE(
        n_components=2,      # almost always 2 (or 3 for 3D plots)
        perplexity=min(curr_model_coefs_cluster.shape[0] - 1, 30),       # roughly "effective number of neighbors" (typical: 5–50)
        learning_rate='auto',
        # n_iter=1000,         # more iterations = more stable result
        metric='cosine',  # or 'cosine', etc.
        random_state=42
    )

    tsne_embedding = tsne_reducer.fit_transform(curr_model_coefs_cluster)

    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=labels, s=5, alpha=0.7)
    plt.title("t-SNE Projection")
    plt.show()
    return


@app.cell
def _(curr_model_coefs_cluster, dendrogram, linkage, plt):
    Z = linkage(curr_model_coefs_cluster, method='average', metric='cosine')
    plt.figure(figsize=(14, 6))
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=30,                   # show last 30 merges
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index (or cluster size)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    return (Z,)


@app.cell
def _(Z, curr_model_coefs_cluster, fcluster, np, silhouette_score):
    from collections import Counter

    def print_intra_cluster_stats(labels, X=curr_model_coefs_cluster):
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_embs = X[mask]
            centroid = cluster_embs.mean(axis=0)

            # Intra-cluster cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(cluster_embs)
            np.fill_diagonal(sim_matrix, np.nan)
            avg_sim = np.nanmean(sim_matrix)

            print(f"Cluster {cluster_id}: size={mask.sum()}, avg_intra_cosine={avg_sim:.3f}")

    def print_inter_cluster_stats(labels, X=curr_model_coefs_cluster):
        from sklearn.metrics import davies_bouldin_score

        sil = silhouette_score(X, labels, metric='cosine')
        db  = davies_bouldin_score(X, labels)
        print(f"Silhouette: {sil:.3f} (higher is better, max 1.0)")
        print(f"Davies-Bouldin: {db:.3f} (lower is better)")

    # By number of clusters
    hier_labels_num = fcluster(Z, 12, criterion='maxclust')

    # By distance threshold (cut the dendrogram at a specific height)
    hier_labels_dist = fcluster(Z, t=0.5, criterion='distance')

    print(hier_labels_num.shape)       # (n,)
    print(np.unique(hier_labels_num)) 
    print(Counter(hier_labels_num))
    print_intra_cluster_stats(hier_labels_num)
    print_inter_cluster_stats(hier_labels_num)

    print(hier_labels_dist.shape)       # (n,)
    print(np.unique(hier_labels_dist)) 
    print(Counter(hier_labels_dist))
    print_intra_cluster_stats(hier_labels_dist)
    print_inter_cluster_stats(hier_labels_dist)
    return (Counter,)


@app.cell
def _(Z, plt):
    last_merges = Z[-20:, 2]  # last 20 merge distances
    plt.plot(range(1, 21), last_merges[::-1], marker='o')
    plt.xlabel("Number of clusters (from right)")
    plt.ylabel("Merge distance")
    plt.title("Elbow / Scree plot")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shared information
    """)
    return


@app.cell
def _(X, encoding_test_idxs, encoding_tr_idxs):
    prev_words, curr_words = X[:-1], X[1:]

    # Set folds so that we ignore the final entry.
    def drop_value(list_of_lists, n):
        return [[x for x in sublist if x != n] for sublist in list_of_lists]

    drop_last_tr_idxs = drop_value(encoding_tr_idxs, X.shape[0] - 1)
    drop_last_test_idxs = drop_value(encoding_test_idxs, X.shape[0] - 1)

    # TODO: these likely leak an example but idrc rn
    drop_last_folds = [(i, j) for i, j in zip(drop_last_tr_idxs, drop_last_test_idxs)]
    return curr_words, drop_last_folds, prev_words


@app.cell
def _(create_model, curr_words, drop_last_folds, prev_words, train_encoding):
    # set_backend("torch") # resort to torch or numpy if cuda out of memory
    corrs_word_preds, next_word_models, tr_idx, test_idx, _ = train_encoding(prev_words, curr_words, create_model, epochs_shape=None, provided_idxs=drop_last_folds)
    print(f"Encoding performance correlating matrix shape: {corrs_word_preds.shape}")
    return (next_word_models,)


@app.cell
def _(Y, drop_last_folds, encoding_models, prev_words, train_encoding):
    # def predict_next_word(x, i, idx=None):
    #     return next_word_models[i].predict(x)

    def return_existing_prediction_model(i):
        return encoding_models[i]

    corrs_next_pred_fix_model, prev_word_pred_models, _, _,  _ = train_encoding(prev_words, Y[1:], return_existing_prediction_model, fix_model=False, provided_idxs=drop_last_folds)
    print(f"Encoding performance correlating matrix shape: {corrs_next_pred_fix_model.shape}")
    return (
        corrs_next_pred_fix_model,
        prev_word_pred_models,
        return_existing_prediction_model,
    )


@app.cell
def _(
    Y,
    curr_words,
    drop_last_folds,
    next_word_models,
    prev_word_pred_models,
    prev_words,
    return_existing_prediction_model,
    train_encoding,
):
    def partial_next_word(x, i, idx):
        return x - next_word_models[i].predict(prev_words[idx]).numpy()

    def partial_target(x, y, i, idx):
        return y - prev_word_pred_models[i].predict(prev_words[idx]).numpy()

    corrs_partialed_pred, _, _, _, _ = train_encoding(curr_words, Y[1:], return_existing_prediction_model, modify_inputs_fn=partial_next_word, modify_targets_fn=partial_target, fix_model=True, provided_idxs=drop_last_folds)
    print(f"Encoding performance correlating matrix shape: {corrs_partialed_pred.shape}")
    return


@app.cell
def _(
    Y,
    drop_last_folds,
    prev_words,
    return_existing_prediction_model,
    train_encoding,
):
    # TODO: See if model parameters are consistent across the two
    corrs_partialed, _, _, _, _ = train_encoding(prev_words, Y[1:], return_existing_prediction_model, fix_model=False, provided_idxs=drop_last_folds)
    print(f"Encoding performance correlating matrix shape: {corrs_partialed.shape}")
    return


@app.cell
def _(
    add_corrs,
    corrs_embedding,
    corrs_mean,
    corrs_next_pred_fix_model,
    np,
    plt,
):
    def _():
        lags = np.arange(-2 * 512, 2 * 512, 16) / 512 # specify the lags

        fig, ax = plt.subplots()
        add_corrs(corrs_next_pred_fix_model, ax=ax, color="red", label="Next Pred")
        add_corrs(corrs_embedding, ax=ax, label="Baseline")
        # add_corrs(corrs_partialed_pred, ax=ax, label="Partialed", color="green")
        # add_corrs(corrs_partialed, ax=ax, label="Model Free", color="orange")
        add_corrs(corrs_mean, ax=ax, label="Mean", color="orange")
        plt.legend()
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("encoding performance (r ± sem)")
        ax.axvline(0, c=(.9, .9, .9), ls="--")
        ax.axhline(0, c=(.9, .9, .9), ls="--")
        plt.show()
    _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(X, plot_pairwise_cosine_similarity, prev_word_model):
    plot_pairwise_cosine_similarity(prev_word_model.predict(X[:-1])[:20], X[1:][:20])
    return


@app.cell
def _(X, plot_pairwise_cosine_similarity):
    plot_pairwise_cosine_similarity(X[:20], X[:20])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
