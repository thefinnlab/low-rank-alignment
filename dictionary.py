import marimo

__generated_with = "0.20.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import altair as alt
    import joblib
    from typing import Optional
    from pathlib import Path

    from collections import Counter
    import mne
    import h5py
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy import stats
    from scipy.stats import pearsonr, entropy
    from tqdm.notebook import tqdm

    from nilearn.plotting import plot_markers
    from mne_bids import BIDSPath

    from himalaya.backend import set_backend, get_backend
    from himalaya.ridge import RidgeCV
    from himalaya.scoring import correlation_score

    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression, CCA
    from sklearn.decomposition import PCA, FastICA, DictionaryLearning
    from sklearn.linear_model import LinearRegression
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import cosine
    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans
    import umap
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, inconsistent
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
        DictionaryLearning,
        FastICA,
        KFold,
        LinearRegression,
        Optional,
        PLSRegression,
        Path,
        RidgeCV,
        StandardScaler,
        correlation_score,
        cosine_similarity,
        dendrogram,
        entropy,
        fcluster,
        get_backend,
        h5py,
        inconsistent,
        joblib,
        make_pipeline,
        mne,
        mo,
        multipletests,
        np,
        os,
        pd,
        plt,
        set_backend,
        stats,
        torch,
        tqdm,
        urllib,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Helpers
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Data Loaders
    """)
    return


@app.cell(hide_code=True)
def _(BIDSPath, Optional, mne, np, os, pd):
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


    def get_arbitrary_embeddings(X, epochs, df_word):
        words = df_word.iloc[epochs.selection].reset_index()["word"].apply(lambda x: x.lower())
        unique_words = words.unique()
        emb_idxs = np.random.choice(np.arange(X[0].shape[0]), size=len(unique_words), replace=False)
        arb_word_embs = X[emb_idxs]

        X_arb = np.zeros_like(X)
        for word_idx, word in enumerate(unique_words):
            X_arb[words[words == word].index] = arb_word_embs[word_idx]

        return X_arb

    return (
        get_arbitrary_embeddings,
        load_raws,
        read_electrode_file,
        read_subject_mapping,
    )


@app.cell
def _(mo):
    mo.md("""
    ### Encoding Helpers
    """)
    return


@app.cell(hide_code=True)
def _(KFold, StandardScaler, correlation_score, get_backend, np):
    def train_encoding(X, Y, model_constructor_fn, modify_inputs_fn=None, modify_targets_fn=None, epochs_shape=None, fix_model=False, provided_idxs=None, provided_scalers=[], num_folds=2):
        num_folds = len(provided_idxs) if provided_idxs is not None else num_folds
        # Convert to having X copied per-fold to allow for inputting a unique X per fold if necessary.
        if not isinstance(X, list):
            X = [X for _ in range(num_folds)]

        models = []
        corrs = [] # empty array to store correlation results
        scalers = []
        tr_idxs, test_idxs  = [], []
        if provided_idxs is None:
            kfold = KFold(num_folds, shuffle=False) # outer 2-fold cross-validation setup
            folds = list(kfold.split(X[0]))
        else:
            folds = provided_idxs

        for i, (train_index, test_index) in enumerate(folds): # loop through folds
            # Split train and test datasets
            X_curr = X[i]
            X1_train, X1_test = X_curr[train_index], X_curr[test_index]
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

    return (train_encoding,)


@app.cell
def _(mo):
    mo.md("""
    ### Trained Model Helpers
    """)
    return


@app.cell(hide_code=True)
def _(get_model_coefficients, np, torch):
    def get_scaled_model_coefficients(encoding_model, num_electrodes, num_lags):
        ridge_coefficients = encoding_model.named_steps['ridgecv'].coef_
        ridge_coefficients = ridge_coefficients.view(ridge_coefficients.shape[0], num_electrodes, num_lags).permute(1, 2, 0)
        ridge_intercept = encoding_model.named_steps['ridgecv'].intercept_
        return ridge_coefficients, ridge_intercept


    def absorb_scaler(encoding_model, num_electrodes, num_lags):
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
        raw_coefs = raw_coefs.view(raw_coefs.shape[0], num_electrodes, num_lags).permute(1, 2, 0)

        return raw_coefs.numpy(), raw_intercept.numpy()

    def get_models_coefficients(encoding_models, num_electrodes, num_lags, scaled_model=False):
        coefficients, intercepts = [], []
        for encoding_model in encoding_models:
            if scaled_model:
                coefs = get_model_coefficients(encoding_model, num_electrodes, num_lags)
            else:
                coefs = absorb_scaler(encoding_model, num_electrodes, num_lags)

            coefficients.append(coefs[0])
            intercepts.append(coefs[1])

        return coefficients, intercepts


    def get_all_targets_test_set(test_idxs, Y, num_electrodes, num_lags):
        # Align Y with the test idxs
        all_targets = []
        for test_idx in test_idxs:
            all_targets.append(Y[test_idx])
        all_targets = np.vstack(all_targets).reshape((Y.shape[0], num_electrodes, num_lags))
        return all_targets


    def get_all_predictions_and_targets_test_set(encoding_models, test_idxs, X, Y, num_electrodes, num_timepoints):
        all_predictions = []

        for i, trained_model in enumerate(encoding_models):
            all_predictions.append(trained_model.predict(X[i][test_idxs[i]]))

        all_predictions = torch.vstack(all_predictions).numpy().reshape((X[0].shape[0], num_electrodes, num_timepoints))

        return all_predictions, get_all_targets_test_set(test_idxs, Y, num_electrodes, num_timepoints)

    return (
        absorb_scaler,
        get_all_predictions_and_targets_test_set,
        get_all_targets_test_set,
    )


@app.cell
def _(mo):
    mo.md("""
    ### Plotting Helpers
    """)
    return


@app.cell(hide_code=True)
def _(
    cosine_similarity,
    dendrogram,
    fcluster,
    inconsistent,
    lags,
    np,
    plt,
    stats,
):
    def add_null_corrs(null_corrs_per_iter, electrode=None, label=None, color="black", ax=None, linestyle="--", percentile=95):
        """
        Parameters
        ----------
        null_corrs_per_iter : ndarray, shape (n_null, n_folds, num_electrodes, num_lags)
            Raw per-iteration null correlations for a single k value.
        percentile : float
            Width of the null envelope to shade (e.g. 95 -> 2.5th to 97.5th percentile across null iterations).
        """
        ax = ax or plt.gca()

        if electrode is not None:
            # (n_null, n_folds, num_electrodes, num_lags) -> mean over folds -> (n_null, num_lags)
            per_iter = null_corrs_per_iter[:, :, electrode, :].mean(axis=1)
        else:
            # mean over folds and electrodes -> (n_null, num_lags)
            per_iter = null_corrs_per_iter.mean(axis=(1, 2))

        # Central estimate: mean across null iterations
        mean = per_iter.mean(axis=0)

        # SEM across null iterations (same logic as add_corrs but over n_null)
        sem = per_iter.std(axis=0) / np.sqrt(per_iter.shape[0])

        # Percentile envelope: spread of null distribution across iterations
        lo = np.percentile(per_iter, (100 - percentile) / 2, axis=0)
        hi = np.percentile(per_iter, 100 - (100 - percentile) / 2, axis=0)

        ax.plot(lags, mean, color=color, label=label, linestyle=linestyle)
        ax.fill_between(lags, mean - sem, mean + sem, alpha=0.2, color=color)   # SEM band
        ax.fill_between(lags, lo, hi, alpha=0.08, color=color)                  # percentile envelope


    def add_corrs(corrs, electrode=None, label=None, color="black", ax=None, linestyle=None):
        if electrode:
            mean = corrs.mean((0))[electrode]
            err = corrs.std((0)) / np.sqrt(np.prod(corrs.shape[:1]))
            err = err[electrode]
        else:
            mean = corrs.mean((0,1))
            err = corrs.std((0,1)) / np.sqrt(np.prod(corrs.shape[:2]))
        if ax:
            ax.plot(lags, mean, color=color, label=label, linestyle=linestyle)
            ax.fill_between(lags, mean - err, mean + err, alpha=0.1, color=color)
        else:
            plt.plot(lags, mean, color=color, label=label, linestyle=linestyle)
            plt.fill_between(lags, mean - err, mean + err, alpha=0.1, color=color)

    def plot_embedding_correlations(electrode=None, plot_corrs=[], plot_nulls=[]):
        fig, ax = plt.subplots()
        for corrs in plot_corrs:
            add_corrs(corrs["corrs"], electrode=electrode, ax=ax, color=corrs["color"], label=corrs["label"], linestyle=corrs["linestyle"])
        for null in plot_nulls:
            add_null_corrs(null["corrs"], electrode=electrode, ax=ax, color=null["color"], label=null["label"], linestyle=null.get("linestyle", "--"), percentile=null.get("percentile", 95))
        plt.legend()
        ax.set_xlabel("lag (s)")
        ax.set_ylabel("encoding performance (r ± sem)")
        ax.axvline(0, c=(.9, .9, .9), ls="--")
        ax.axhline(0, c=(.9, .9, .9), ls="--")


    def plot_class_distributions(
        data: np.ndarray,
        target_class: int,
        signs_per_fold: list[np.ndarray] = None,
        elecs_per_subject: list[int] | None = None,
    ) -> None:
        n_elecs, n_timepoints = data.shape

        class_mask = (data == target_class)
        total_occurrences = class_mask.sum()
        if total_occurrences == 0:
            print(f"Class {target_class} not found in data.")
            return

        elec_counts = class_mask.sum(axis=1)
        elec_probs  = elec_counts / total_occurrences

        time_counts = class_mask.sum(axis=0)
        time_probs  = time_counts / total_occurrences

        subjects = []
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

        if subjects:
            subj_time_probs = np.array([
                class_mask[sl].sum(axis=0) / total_occurrences
                for _, sl in subjects
            ])
            subj_labels = [lbl for lbl, _ in subjects]
            subj_probs = subj_time_probs.sum(axis=1)
            colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))

        # ── Aggregate signs across folds ──────────────────────────────────────────
        has_signs = signs_per_fold is not None
        if has_signs:
            # Each fold has its own signs for the cluster members.
            # Since cluster membership is the same (same electrode-lag pairs),
            # but each fold's model may flip differently, we average the signs
            # to get a consensus direction per electrode-lag pair.
            # signs_per_fold: list of arrays, each shape (n_members_in_cluster,)
            avg_signs = np.mean(signs_per_fold, axis=0)  # values in [-1, 1]

            sign_map = np.zeros_like(class_mask, dtype=float)
            sign_map[class_mask] = avg_signs

            pos_mask = (sign_map > 0) & class_mask
            neg_mask = (sign_map < 0) & class_mask
            # For "mixed" (avg_sign == 0), neither mask fires — shown as neutral

            elec_pos = pos_mask.sum(axis=1)
            elec_neg = neg_mask.sum(axis=1)
            elec_total = elec_pos + elec_neg
            elec_pos_frac = np.divide(elec_pos, elec_total,
                                      out=np.full_like(elec_pos, 0.5, dtype=float),
                                      where=elec_total > 0)

            time_pos = pos_mask.sum(axis=0)
            time_neg = neg_mask.sum(axis=0)
            time_total = time_pos + time_neg
            time_pos_frac = np.divide(time_pos, time_total,
                                      out=np.full_like(time_pos, 0.5, dtype=float),
                                      where=time_total > 0)

        # ── Layout ────────────────────────────────────────────────────────────────
        n_cols = 3 if subjects else 2
        n_rows = 2 if has_signs else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(f"Class {target_class} — probability distributions", fontsize=14)

        pct = plt.FuncFormatter(lambda y, _: f"{y:.1%}")
        x_elec = np.arange(n_elecs)
        x_time = np.arange(n_timepoints)

        # ── Row 1: Probability distributions ──────────────────────────────────────
        ax = axes[0, 0]
        ax.bar(x_elec, elec_probs, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Electrode index")
        ax.set_ylabel("Probability")
        ax.set_title("Distribution across electrodes")
        ax.set_xlim(-0.5, n_elecs - 0.5)
        ax.yaxis.set_major_formatter(pct)

        ax = axes[0, 1]
        if subjects:
            bottoms = np.zeros(n_timepoints)
            for i, (label, color) in enumerate(zip(subj_labels, colors)):
                ax.bar(x_time, subj_time_probs[i], bottom=bottoms,
                       color=color, edgecolor="none", linewidth=0, label=label, width=1.0)
                bottoms += subj_time_probs[i]
            ax.legend(title="Subject", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        else:
            ax.bar(x_time, time_probs, color="coral", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Timepoint index")
        ax.set_ylabel("Probability")
        ax.set_title("Distribution across timepoints")
        ax.set_xlim(-0.5, n_timepoints - 0.5)
        ax.yaxis.set_major_formatter(pct)

        if subjects:
            ax = axes[0, 2]
            x_subj = np.arange(len(subjects))
            bars = ax.bar(x_subj, subj_probs, color=colors, edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Subject")
            ax.set_ylabel("Probability")
            ax.set_title("Distribution across subjects")
            ax.set_xticks(x_subj)
            ax.set_xticklabels(subj_labels)
            ax.yaxis.set_major_formatter(pct)
            for bar, prob in zip(bars, subj_probs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(subj_probs) * 0.01,
                        f"{prob:.1%}", ha="center", va="bottom", fontsize=8)

        # ── Row 2: Sign distributions ─────────────────────────────────────────────
        if has_signs:
            from matplotlib.colors import TwoSlopeNorm
            cmap = plt.cm.RdBu
            norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

            ax = axes[1, 0]
            bar_colors = cmap(norm(elec_pos_frac))
            ax.bar(x_elec, elec_counts, color=bar_colors, edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Electrode index")
            ax.set_ylabel("Count")
            ax.set_title("Sign distribution across electrodes")
            ax.set_xlim(-0.5, n_elecs - 0.5)

            ax = axes[1, 1]
            bar_colors = cmap(norm(time_pos_frac))
            ax.bar(x_time, time_counts, color=bar_colors, edgecolor="white", linewidth=0.4, width=1.0)
            ax.set_xlabel("Timepoint index")
            ax.set_ylabel("Count")
            ax.set_title("Sign distribution across timepoints")
            ax.set_xlim(-0.5, n_timepoints - 0.5)

            if subjects:
                ax = axes[1, 2]
                subj_pos_fracs = []
                subj_counts_list = []
                for _, sl in subjects:
                    s_pos = pos_mask[sl].sum()
                    s_neg = neg_mask[sl].sum()
                    s_total = s_pos + s_neg
                    subj_pos_fracs.append(s_pos / s_total if s_total > 0 else 0.5)
                    subj_counts_list.append(s_total)
                subj_pos_fracs = np.array(subj_pos_fracs)
                subj_counts_arr = np.array(subj_counts_list)

                bar_colors = cmap(norm(subj_pos_fracs))
                bars = ax.bar(x_subj, subj_counts_arr, color=bar_colors, edgecolor="white", linewidth=0.4)
                ax.set_xlabel("Subject")
                ax.set_ylabel("Count")
                ax.set_title("Sign distribution across subjects")
                ax.set_xticks(x_subj)
                ax.set_xticklabels(subj_labels)
                for bar, frac in zip(bars, subj_pos_fracs):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(subj_counts_arr) * 0.01,
                            f"+{frac:.0%}", ha="center", va="bottom", fontsize=8)

            # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            # sm.set_array([])
            # cbar = fig.colorbar(sm, ax=axes[1, :].tolist(), shrink=0.3, pad=0)
            # cbar.set_label("Fraction positive (blue=+1, red=−1)")

        plt.tight_layout()
        plt.show()


    def plot_correlations(x, y, x_name="Array 1", y_name="Array 2"):
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

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"Relationship between {x_name} and {y_name}")
        ax.legend()
        plt.tight_layout()
        plt.show()


    def plot_atom_distributions(
        atom_maps: np.ndarray,      # [num_atoms, n_elecs, n_lags]
        target_atom: int,           # Which atom index to visualize
        elecs_per_subject: list[int] | None = None,
    ) -> None:
        """
        Visualizes the spatial and temporal distribution of a specific 
        Dictionary Learning atom.
        """
        # Extract the map for the specific atom
        # Shape: [n_elecs, n_lags]
        data = atom_maps[target_atom]
        n_elecs, n_lags = data.shape

        # In dictionary learning, 'loadings' replace 'occurrences'
        total_loading = data.sum()
        if total_loading == 0:
            print(f"Atom {target_atom} has no loading in the significant models.")
            return

        # Normalize to get a 'probability distribution' of the atom across the brain
        elec_probs = data.sum(axis=1) / total_loading
        time_probs = data.sum(axis=0) / total_loading

        # --- Subject Logic (Reused from your code) ---
        subjects = []
        if elecs_per_subject is not None:
            cursor = 0
            for subj_idx, n in enumerate(elecs_per_subject):
                if n > 0:
                    subjects.append((f"S{subj_idx + 1}", slice(cursor, cursor + n)))
                cursor += n

        # Layout Setup
        n_plots = 3 if subjects else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        fig.suptitle(f"Atom {target_atom} — Spatial/Temporal Loading", fontsize=14)
        pct = plt.FuncFormatter(lambda y, _: f"{y:.1%}")

        # 1. Electrode Distribution
        axes[0].bar(np.arange(n_elecs), elec_probs, color="indigo")
        axes[0].set_title("Loading per Electrode")
        axes[0].yaxis.set_major_formatter(pct)

        # 2. Timepoint (Lag) Distribution
        if subjects:
            bottoms = np.zeros(n_lags)
            colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
            for i, (label, sl) in enumerate(subjects):
                subj_map = data[sl].sum(axis=0) / total_loading
                axes[1].bar(np.arange(n_lags), subj_map, bottom=bottoms, 
                            label=label, color=colors[i], width=1.0)
                bottoms += subj_map
            axes[1].legend(title="Subject", loc="upper right", fontsize=8)
        else:
            axes[1].bar(np.arange(n_lags), time_probs, color="mediumseagreen")

        axes[1].set_title("Loading per Lag (ms)")
        axes[1].yaxis.set_major_formatter(pct)

        # 3. Subject Totals
        if subjects:
            subj_probs = [data[sl].sum() / total_loading for _, sl in subjects]
            axes[2].bar([s[0] for s in subjects], subj_probs, color=colors)
            axes[2].set_title("Loading per Subject")
            axes[2].yaxis.set_major_formatter(pct)

        plt.tight_layout()
        plt.show()


    # def plot_activations(curr_activations):
    #     _fine_counts, _fine_edges = np.histogram(curr_activations, bins=100)
    #     _hist_df = pd.DataFrame({
    #         "x1": _fine_edges[:-1],
    #         "x2": _fine_edges[1:],
    #         "count": _fine_counts,
    #     })

    #     _bars = alt.Chart(_hist_df).mark_bar(color="#93c5fd", stroke="#bfdbfe", strokeWidth=0.5).encode(
    #         x=alt.X("x1:Q", title="Activation",
    #                 scale=alt.Scale(domain=[float(_fine_edges[0]), float(_fine_edges[-1])])),
    #         x2="x2:Q",
    #         y=alt.Y("count:Q", title="Count"),
    #         tooltip=[
    #             alt.Tooltip("x1:Q", title="From", format=".3f"),
    #             alt.Tooltip("x2:Q", title="To", format=".3f"),
    #             alt.Tooltip("count:Q", title="Count"),
    #         ],
    #     )

    #     # Bin dividers from the slider
    #     _n = num_bins_slider.value
    #     _edges = np.linspace(curr_activations.min(), curr_activations.max(), _n + 1)
    #     _rule_df = pd.DataFrame({
    #         "x": list(_edges[1:-1]),
    #         "label": [f"Bin {i+1} / {i+2}" for i in range(_n - 1)],
    #     })
    #     _rules = alt.Chart(_rule_df).mark_rule(
    #         color="#1e3a5f", strokeWidth=2
    #     ).encode(
    #         x="x:Q",
    #         tooltip=[alt.Tooltip("x:Q", title="Boundary", format=".3f"), "label:N"],
    #     )

    #     # Bin number labels at top of chart
    #     _label_df = pd.DataFrame({
    #         "x": [(_edges[i] + _edges[i+1]) / 2 for i in range(_n)],
    #         "label": [str(i+1) for i in range(_n)],
    #     })
    #     _labels = alt.Chart(_label_df).mark_text(
    #         align="center", baseline="top", dy=4, fontSize=11,
    #         color="#1e3a5f", fontWeight="bold"
    #     ).encode(
    #         x="x:Q",
    #         text="label:N",
    #         y=alt.value(0),
    #     )

    #     _chart = (_bars + _rules + _labels).properties(
    #         width="container",
    #         height=220,
    #         title="Activation distribution with bin boundaries",
    #     ).configure_view(strokeWidth=0).configure_axis(grid=False)

    #     return _chart

    def plot_cluster_diagnostics(Z, labels=None, figsize=(18, 20), tail_fraction=0.05):
        """
        Diagnostic plots for choosing a dendrogram distance cutoff.

        Parameters
        ----------
        Z : ndarray
            Linkage matrix from scipy.cluster.hierarchy.linkage.
        labels : list, optional
            Labels for leaf nodes in the dendrogram.
        figsize : tuple
            Figure size.
        tail_fraction : float
            Fraction of top merges to focus on for detailed plots (default 5%).
        """
        merge_dists = Z[:, 2]
        n_merges = len(merge_dists)
        n_samples = n_merges + 1
        n_tail = max(int(n_merges * tail_fraction), 50)

        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # 1. Dendrogram (truncated for large datasets)
        ax = axes[0, 0]
        truncate = n_samples > 80
        dendrogram(
            Z,
            ax=ax,
            truncate_mode='lastp' if truncate else None,
            p=40 if truncate else n_samples,
            leaf_rotation=90,
            leaf_font_size=8,
            labels=labels if not truncate else None,
            color_threshold=0,
        )
        ax.set_title(f'Dendrogram{" (truncated)" if truncate else ""}')
        ax.set_ylabel('Merge distance')

        # 2. Sorted merge distances — zoomed to tail
        ax = axes[0, 1]
        sorted_dists = np.sort(merge_dists)
        tail = sorted_dists[-n_tail:]
        ax.plot(range(n_tail), tail, marker='.', ms=4)
        gaps = np.diff(sorted_dists)
        top_gap_idxs = np.argsort(gaps)[-min(5, len(gaps)):]
        for idx in top_gap_idxs:
            cut_val = (sorted_dists[idx] + sorted_dists[idx + 1]) / 2
            if cut_val >= tail[0]:
                ax.axhline(cut_val, color='red', alpha=0.5, ls='--', lw=1)
        ax.set_title(f'Last {n_tail} merge distances (red = largest gaps)')
        ax.set_xlabel(f'Merge index (of top {n_tail})')
        ax.set_ylabel('Distance')

        # 3. Number of clusters vs. cut threshold — focused on interesting range
        ax = axes[1, 0]
        t_lo = np.percentile(merge_dists, max(80, 100 - tail_fraction * 200))
        t_hi = merge_dists.max() * 1.05
        t_range = np.linspace(t_lo, t_hi, 300)
        n_clusters = np.array([
            len(np.unique(fcluster(Z, t=t, criterion='distance'))) for t in t_range
        ])
        ax.plot(t_range, n_clusters)
        ax.set_title('Number of clusters vs. distance threshold')
        ax.set_xlabel('Distance threshold t')
        ax.set_ylabel('Number of clusters')
        ax.set_yscale('log')
        for nc in np.unique(n_clusters):
            mask = n_clusters == nc
            t_vals = t_range[mask]
            t_width = t_vals.max() - t_vals.min()
            if t_width / (t_range[-1] - t_range[0]) > 0.03:
                ax.axhspan(nc - 0.3, nc + 0.3, alpha=0.08, color='green')

        # 4. Top merge distance gaps
        ax = axes[1, 1]
        n_gaps_show = min(40, len(gaps))
        top_gaps = np.sort(gaps)[::-1][:n_gaps_show]
        ax.bar(range(n_gaps_show), top_gaps, width=0.8)
        ax.set_title(f'Top {n_gaps_show} merge distance gaps')
        ax.set_xlabel('Rank')
        ax.set_ylabel('Gap size')

        # 5. Inconsistency coefficients
        ax = axes[2, 0]
        incons = inconsistent(Z, d=2)
        incons_vals = incons[:, 3]
        ax.hist(incons_vals, bins=80, edgecolor='black', linewidth=0.3)
        # zoom into the right tail via a twin axis
        threshold_99 = np.percentile(incons_vals, 99)
        ax.axvline(threshold_99, color='red', ls='--', lw=1,
                   label=f'99th percentile ({threshold_99:.2f})')
        ax.legend()
        ax.set_title('Inconsistency coefficient distribution (depth=2)')
        ax.set_xlabel('Inconsistency coefficient')
        ax.set_ylabel('Count')

        # 6. Cluster size distribution at candidate thresholds
        ax = axes[2, 1]
        candidates = []
        if len(gaps) > 0:
            for idx in np.argsort(gaps)[-min(4, len(gaps)):]:
                candidates.append((sorted_dists[idx] + sorted_dists[idx + 1]) / 2)
        candidates = sorted(set(candidates))

        if candidates:
            bar_width = 0.8 / len(candidates)
            for i, t in enumerate(candidates):
                cl = fcluster(Z, t=t, criterion='distance')
                sizes = np.sort(np.bincount(cl)[1:])[::-1]
                # only show top clusters to keep readable
                n_show = min(30, len(sizes))
                ax.bar(
                    np.arange(n_show) + i * bar_width,
                    sizes[:n_show],
                    width=bar_width,
                    label=f't={t:.3f} ({len(sizes)} cl)',
                    alpha=0.7,
                )
            ax.legend(fontsize=8)
        ax.set_title('Cluster size distribution at candidate thresholds')
        ax.set_xlabel('Cluster rank (largest first)')
        ax.set_ylabel('Cluster size')

        plt.tight_layout()
        plt.savefig('cluster_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Summary table
        print(f"\n=== Candidate cut thresholds ({n_samples} samples, {n_merges} merges) ===")
        if candidates:
            for t in candidates:
                cl = fcluster(Z, t=t, criterion='distance')
                sizes = np.sort(np.bincount(cl)[1:])[::-1]
                print(f"  t={t:.4f} → {len(sizes):4d} clusters | "
                      f"largest: {sizes[0]:5d} | median: {np.median(sizes):5.0f} | "
                      f"singletons: {np.sum(sizes == 1):4d}")


    def plot_cosine_similarity_heatmap(embeddings, labels=None, cmap="RdBu_r", figsize=(10, 8)):
        """Plot cosine similarity heatmap between all pairs of embeddings.

        Args:
            embeddings: array of shape [num_clusters, embed_dim]
            labels: optional list of labels for each cluster
            cmap: colormap name
            figsize: figure size tuple
        """
        sim_matrix = cosine_similarity(embeddings)
        num_clusters = sim_matrix.shape[0]

        if labels is None:
            labels = [str(i) for i in range(num_clusters)]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(sim_matrix, cmap=cmap, vmin=-1, vmax=1)

        ax.set_xticks(range(num_clusters))
        ax.set_yticks(range(num_clusters))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        # Annotate cells with values
        for i in range(num_clusters):
            for j in range(num_clusters):
                color = "white" if abs(sim_matrix[i, j]) > 0.7 else "black"
                ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, label="Cosine Similarity")
        ax.set_title("Pairwise Cosine Similarity")
        plt.tight_layout()
        return fig, sim_matrix

    return (plot_embedding_correlations,)


@app.cell
def _(mo):
    mo.md("""
    ### Significance Helpers
    """)
    return


@app.cell(hide_code=True)
def _(Path, multipletests, np, os, torch, tqdm):
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


    def get_sig_results(all_predictions, all_targets):
        """
        partial_clusters: List of tuples of the clusters on that run which were partialed out before generating this data
        sig_dir: Root directory of significance data.
        """
        electrodes_predictions = all_predictions.transpose(1, 2, 0)
        electrodes_targets = all_targets.transpose(1, 2, 0)
        sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(electrodes_predictions, electrodes_targets)

        return sig, p_values, q_values

    return (get_sig_results,)


@app.cell
def _(mo):
    mo.md("""
    ### General Analysis
    """)
    return


@app.cell(hide_code=True)
def _(os, torch):
    def get_correlation_gpu(array_a, array_b):
        a = torch.tensor(array_a, device='cuda', dtype=torch.float32)
        b = torch.tensor(array_b, device='cuda', dtype=torch.float32)

        # Standardize on GPU
        a = (a - a.mean(dim=0)) / a.std(dim=0)
        b = (b - b.mean(dim=0)) / b.std(dim=0)

        corr_matrix = (a.T @ b) / array_a.shape[0]
        return corr_matrix.cpu().numpy()


    def get_partial_cluster_filepath(partial_clusters):
        """
        partial_clusters: List of tuples of the clusters on that run which were partialed out before generating this data. Of form [[num_clusters, [partialed_clusters]]]
        sig_dir: Root directory of significance data.
        """
        end_path_directory = []
        for num_clusters, curr_cluster in partial_clusters:
            end_path_directory.append(f"{num_clusters}-" + "_".join([str(x) for x in sorted(curr_cluster)]))

        target_dir = ""
        if end_path_directory:
            target_dir = os.path.join(*end_path_directory)
        return target_dir

    return (get_partial_cluster_filepath,)


@app.cell
def _(mo):
    mo.md("""
    ### Comparative Helpers
    """)
    return


@app.cell(hide_code=True)
def _(DictionaryLearning, FastICA, PLSRegression, absorb_scaler, np):
    def run_comparative_analysis_on_folds(
        encoding_models, # List of trained models (one per fold)
        sig,
        X,               # [num_words, embed_dim]
        Y,               # [num_words, num_elecs, num_timepoints]
        train_folds,
        num_electrodes, 
        num_lags,
        n_components=50
    ):
        # --- 2. Extract Significant Coefficients ---
        # We'll use the mean coefficients across folds for the significant lags
        all_coefs = []
        for model in encoding_models:
            coef, _ = absorb_scaler(model, num_electrodes, num_lags)
            all_coefs.append(coef)

        dls = []
        icas = []
        plss = []

        for i, train_idx in enumerate(train_folds):
            # Filter: Only keep weights where sig == True
            # W_sig shape: [num_significant_lags, embed_dim]
            W_sig = all_coefs[i][sig]

            print(f"Number of significant electrode-lags found: {W_sig.shape[0]}")

            if W_sig.shape[0] < n_components:
                print("Warning: More components requested than significant signals available.")
                n_components = W_sig.shape[0]

            # --- 3. Run the Models ---

            # A. Dictionary Learning (Superposition-friendly)
            dl = DictionaryLearning(n_components=n_components, transform_alpha=1.0, random_state=42)
            dls.append(dl)

            # B. ICA (Independence-friendly)
            ica = FastICA(n_components=n_components, random_state=42)
            icas.append(ica)

            # C. PLS (Covariance-friendly)
            # Note: PLS needs the raw X and Y, but we'll filter Y to only significant channels
            Y_reshaped = Y.reshape(X[0].shape[0], -1)
            sig_flat = sig.flatten()
            Y_sig_only = Y_reshaped[:, sig_flat]

            pls = PLSRegression(n_components=min(n_components, Y_sig_only.shape[1]))
            pls.fit(X[train_idx], Y_sig_only[train_idx])
            plss.append(pls)

        return {
            "dict_learning": dls,
            "icas": icas,
            "plss": plss,
        }


    def get_method_mapping(fitted_model, W_sig, sig_mask, method='dl'):
        """
        Extracts spatial-temporal maps for different dimensionality reduction methods.

        Returns: [n_components, n_elecs, n_lags]
        """
        n_elecs, n_lags = sig_mask.shape

        if method == 'dl':
            # Sparse codes: How much each atom is used by each significant model
            loadings = fitted_model.transform(W_sig) 
        elif method == 'ica':
            # Independent sources
            loadings = fitted_model.transform(W_sig)
        elif method == 'pls':
            # PLS y_weights_ represent the contribution of each Y-variable (elec-lag)
            # to each latent component.
            loadings = fitted_model.y_weights_
        else:
            raise ValueError("Method must be 'dl', 'ica', or 'pls'")

        n_components = loadings.shape[1]
        maps = np.zeros((n_components, n_elecs, n_lags))

        for i in range(n_components):
            # We use absolute value because directionality (pos/neg) in these 
            # methods is often arbitrary, but 'importance' is magnitude.
            maps[i][sig_mask] = np.abs(loadings[:, i])

        return maps

    return


@app.cell
def _(mo):
    mo.md("""
    ### Example Helpers
    """)
    return


@app.cell(hide_code=True)
def _(df_word, epochs, mo, np, pd):
    def get_example_display_controls():
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

        return num_bins_slider, samples_per_bin_slider, context_slider, seed_input


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

    def generate_example_table(curr_activations, num_bins_slider, samples_per_bin_slider, context_slider, seed_input):
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


    def get_mo_display_activations(curr_activations, num_bins_slider, samples_per_bin_slider, context_slider, seed_input):
        return mo.vstack([
            mo.md("# Activation Distribution Explorer"),
            mo.md(
                "Samples words from across the activation distribution. "
                "The target word is **bolded** in its context window."
            ),
            # mo.divider(),
            generate_example_table(curr_activations, num_bins_slider, samples_per_bin_slider, context_slider, seed_input),
        ])

    return generate_example_table, get_example_display_controls


@app.cell
def _(mo):
    mo.md("""
    ### Partialing Helpers
    """)
    return


@app.cell(hide_code=True)
def _(get_partial_cluster_filepath, joblib, np, os):
    def partial_out_direction(X, Y, directions, predictions_list, fold_indices, scalers):
        """
        Remove variance explained by a direction in embedding space from X and Y.

        Parameters
        ----------
        Xs : list of array of shape [n_words, n_dim]
            Word embeddings.
        Y : array of shape [n_words, n_elecs * n_timepoints]
            Neural data (raw, not z-scored).
        directions : list of list of array, each of shape [n_dim]
            Direction to partial out, one per partial direction, one per fold. 
        predictions_list : list of array, each [n_words]
            One prediction vector per direction to partial out.
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
        # Stack all predictions and tile to match Y shape
        # predictions_stack: [n_words, n_directions]
        predictions_stack = np.column_stack(predictions_list)

        # For each fold, inverse-transform each direction's predictions,
        # then combine via OLS to avoid double-subtracting shared variance
        Y_pred_full = np.zeros_like(Y)

        for idx, scaler in zip(fold_indices, scalers):
            # Build the design matrix of all predictions for this fold's samples
            # Each column is one direction's predictions, tiled across Y columns
            fold_preds_raw = []
            for d_i in range(predictions_stack.shape[1]):
                pred_tiled = np.tile(
                    predictions_stack[idx, d_i:d_i+1], (1, Y.shape[1])
                )
                fold_preds_raw.append(scaler.inverse_transform(pred_tiled))

            # Stack into [n_fold_samples, n_directions, n_columns] then
            # for each Y column, regress on the n_directions predictors
            # to get the joint prediction
            preds_matrix = np.stack(fold_preds_raw, axis=-1)  # [n_samples, n_cols, n_dirs]

            # A is the same across columns after inverse_transform
            # So solve once, apply to all columns
            A = preds_matrix[:, 0, :]  # [n_samples, n_dirs] — same for every col
            betas, _, _, _ = np.linalg.lstsq(A, Y[idx], rcond=None)  # [n_dirs, n_cols]
            Y_pred_full[idx] = A @ betas

        Y_residual = Y - Y_pred_full

        # Remove the direction from X per fold (since direction differs per fold)
        X_residuals = [X[i].copy() for i in range(len(directions[0]))]
        for fold_dir in directions:
            for i, direction in enumerate(fold_dir):
                d = direction / np.linalg.norm(direction)
                projections = X_residuals[i] @ d
                X_residuals[i] -= np.outer(projections, d)

        return X_residuals, Y_residual


    def save_partial_results(partial_clusters, X, Y, all_mean_cluster_dirs, partialed_dirs, scalers=None, filename="cluster_results.npz"):
        partial_cluster_dir = get_partial_cluster_filepath(partial_clusters)
        os.makedirs(partial_cluster_dir, exist_ok=True)
        print(f"Saving results to {partial_cluster_dir}")
        np.savez(
            os.path.join(partial_cluster_dir, filename),
            *X,
            Y=Y,
            all_mean_cluster_dirs=all_mean_cluster_dirs,
            partialed_dirs=np.array(partialed_dirs),
            num_X=np.array(len(X)),
        )

        if scalers is not None:
            joblib.dump(scalers, os.path.join(partial_cluster_dir, "scalers.joblib"))


    def load_partial_results(partial_clusters, filename="cluster_results.npz"):
        partial_cluster_dir = get_partial_cluster_filepath(partial_clusters)
        data = np.load(os.path.join(partial_cluster_dir, filename), allow_pickle=False)
        num_X = int(data["num_X"])
        X = [data[f"arr_{i}"] for i in range(num_X)]
        Y = data["Y"]
        all_mean_cluster_dirs = data["all_mean_cluster_dirs"]
        partialed_dirs = data["partialed_dirs"].tolist()

        scalers_path = os.path.join(partial_cluster_dir, "scalers.joblib")
        scalers = joblib.load(scalers_path) if os.path.exists(scalers_path) else None

        return X, Y, all_mean_cluster_dirs, partialed_dirs, scalers

    return


@app.cell
def _(mo):
    mo.md("""
    ### Client Interface
    """)
    return


@app.cell(hide_code=True)
def _(
    get_all_predictions_and_targets_test_set,
    get_sig_results,
    joblib,
    os,
    train_encoding,
):
    def encode_ecog(X, Y, num_electrodes, num_lags, model_constructor, result_dir="", folds=None, epochs_shape=None, skip_sigs=False):
        # TODO: Should I reuse scalers?
        corrs_embedding, encoding_models, encoding_tr_idxs, encoding_test_idxs, encoding_scalers = train_encoding(X, Y, model_constructor, provided_idxs=folds, epochs_shape=epochs_shape)
        all_predictions, all_targets = get_all_predictions_and_targets_test_set(encoding_models, encoding_test_idxs, X, Y, num_electrodes, num_lags)
        if not skip_sigs:
            sig, p_values, q_values = get_sig_results(all_predictions, all_targets)

        # all_activations = all_predictions.reshape((all_predictions.shape[0], -1))
        # all_predictions_corrs = get_correlation_gpu(all_activations, all_activations)

        results = {
            "encoding_corrs": corrs_embedding,
            "encoding_models": encoding_models,
            "model_predictions": all_predictions,
            # "prediction_corrs": all_predictions_corrs,
            "Y_scalers": encoding_scalers,
            "train_idxs": encoding_tr_idxs,
            "test_idxs": encoding_test_idxs,
        }
        if not skip_sigs:
            results["sig"] = {
                "sig_bool": sig,
                "p_values": p_values,
                "q_values": q_values,
            }
        return results


    def save_encoding_results(results, filepath):
        joblib.dump(results, filepath)


    def load_encoding_results(filepath):
        return joblib.load(filepath)

    def get_encoding_result_filepath(run_name, file_name="encoding_results.joblib"):
        return os.path.join(run_name, file_name)

    return (
        encode_ecog,
        get_encoding_result_filepath,
        load_encoding_results,
        save_encoding_results,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Setup
    """)
    return


@app.cell
def _(np):
    bids_root = "data/"  # if using a local dataset, set this variable accordingly
    modelname, layer = 'gpt2-xl', 24

    # For generating predictions if necessary
    MODEL = "gpt2-xl"
    MAX_CTX = 1024
    STRIDE = 512

    lags = np.arange(-2 * 512, 2 * 512, 16) / 512 # specify the lags
    return MAX_CTX, MODEL, STRIDE, bids_root, lags, layer


@app.cell(hide_code=True)
def _(
    MAX_CTX,
    MODEL,
    STRIDE,
    bids_root,
    end,
    entropy,
    epsilon,
    get_token_distributions,
    h5py,
    layer,
    load_raws,
    mne,
    np,
    os,
    pd,
    read_electrode_file,
    read_subject_mapping,
    save_distributions,
    set_backend,
    start,
    token_probs,
    torch,
    urllib,
):
    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using cuda!")

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

    with h5py.File(embedding_path, "r") as f:
        contextual_embeddings = f[f"layer-{layer}"][...]
        final_layer_contextual_embeddings = f[f"layer-48"][...]
    print(f"LLM embedding matrix has shape: {contextual_embeddings.shape}")

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

    df_word = df_contextual.groupby("word_idx").agg(dict(word="first", start="first", end="last"))
    df_word["true_prob"] = word_probs
    df_word["entropy"] = word_entropies

    if os.path.exists("gpt2_prob_output.npz"):
        gpt2_probs_data = np.load("gpt2_prob_output.npz", allow_pickle=True)
    else:
        podcast_transcript = " ".join(df_word.word.tolist())
        gpt2_probs_data = get_token_distributions(podcast_transcript, MODEL, MAX_CTX, STRIDE)
        save_distributions(gpt2_probs_data, "gpt2_prob_output.npz")

    gpt2_probs  = gpt2_probs_data["probs"]   # (T, 50257)
    gpt2_tokens = gpt2_probs_data["tokens"]

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

    def kl_divergence(probs, eps=1e-10) -> np.ndarray:
        P = probs[:-1] + eps
        Q = probs[1:] + eps
        kl = entropy(P.T, Q.T)  # transpose so shape is (V, T-1), returns (T-1,)
        return kl.astype(np.float32)

    gpt2_word_kl_divergence = kl_divergence(aligned_gpt2_probs)

    subject_mapping = read_subject_mapping("data/participants.tsv")
    sig_elecs = read_electrode_file("all_subject_sig.csv", subject_mapping)
    # raws = load_raws([i for i in sig_elecs.keys()], "data/", per_subject_electrodes=sig_elecs, use_high_gamma=True)

    raws = load_raws(
        list(range(1, 10)),
        "data/",
        use_high_gamma=True,
        target_sr=None,
        do_drop_bads=False,
        signal_unit=None,
        per_subject_electrodes=None,
        channel_reg_ex=None,
    )

    events = np.zeros((len(df_word), 3), dtype=int)
    events[:, 0] = (df_word.start * raws[0].info['sfreq']).astype(int)
    print("Events shape:", events.shape)

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

        # TODO: inspect whether the method of resampling matters as well
        epochs = epochs.resample(sfreq=32, npad='auto', method='fft', window='hamming')
        data = epochs.get_data(copy=False)

        datas.append(data)

    datas = np.concatenate(datas, axis=1)
    print(f"Datas array has a shape of: {datas.shape}")

    electrode_names = [f"S/{list(sig_elecs.keys())[i]}/E/{e_name}" for i, r in enumerate(raws) for e_name in r.ch_names]

    epochs_data = datas.reshape(len(epochs), -1)
    print(f"ECoG data matrix shape: {epochs_data.shape}")

    selected_df = df_word.iloc[epochs.selection]
    averaged_embeddings = aligned_embeddings[epochs.selection]
    final_layer_averaged_embeddings = final_layer_aligned_embeddings[epochs.selection]
    print("Averaged embeddings:", averaged_embeddings.shape, "Final layer embeddings:", final_layer_averaged_embeddings.shape)
    return averaged_embeddings, datas, df_word, epochs, epochs_data, raws


@app.cell
def _(mo):
    mo.md("""
    ## Encoding
    """)
    return


@app.cell
def _():
    word_offsets_to_train = [0] # list(range(-1, 2))
    run_name = "mid_gpt2"
    skip_sigs = False
    return run_name, skip_sigs, word_offsets_to_train


@app.cell(hide_code=True)
def _(
    KFold,
    RidgeCV,
    StandardScaler,
    averaged_embeddings,
    datas,
    encode_ecog,
    epochs_data,
    get_all_targets_test_set,
    get_backend,
    get_encoding_result_filepath,
    get_sig_results,
    load_encoding_results,
    make_pipeline,
    np,
    os,
    run_name,
    save_encoding_results,
    skip_sigs,
    word_offsets_to_train,
):
    def get_ridge_model_constructor():
            alphas = np.logspace(1, 10, 10) # specify alpha values
            inner_cv = KFold(n_splits=5, shuffle=False) # inner 5-fold cross-validation setup
            def create_model(_):
                return make_pipeline(
                    StandardScaler(), RidgeCV(alphas, fit_intercept=True, cv=inner_cv) # pipeline
                )
            return create_model

    embed_dim, num_electrodes, num_lags = averaged_embeddings.shape[1], datas.shape[1], datas.shape[2]

    per_offset_results = {}
    for word_offset in word_offsets_to_train:
        if word_offset < 0:
            X, Y = averaged_embeddings[:word_offset], epochs_data[-word_offset:]
        elif word_offset > 0:
            X, Y = averaged_embeddings[word_offset:], epochs_data[:-word_offset]
        else:
            X, Y = averaged_embeddings, epochs_data
        # Make them into a list partially because we may want different lists per fold later but also an annoying holdover
        X = [X, X]

        if "torch" in get_backend().__name__:
            X = [t.astype(np.float32) for t in X]
            Y = Y.astype(np.float32)


        result_dir = f"{run_name}-{word_offset}"
        os.makedirs(result_dir, exist_ok=True)
        results_file = get_encoding_result_filepath(result_dir)
        if os.path.exists(results_file):
            result = load_encoding_results(results_file)
            # Generate significant results if they're not already saved
            if "sig" not in result and not skip_sigs:
                sig, p_values, q_values = get_sig_results(result["model_predictions"], get_all_targets_test_set(result["test_idxs"], Y, num_electrodes, num_lags))
                result["sig"] = {
                    "sig_bool": sig,
                    "p_values": p_values,
                    "q_values": q_values,
                }
                save_encoding_results(result, results_file)

            per_offset_results[word_offset] = result
            continue

        encode_cluster_results = encode_ecog(X, Y, num_electrodes, num_lags, get_ridge_model_constructor(), result_dir=result_dir, folds=None, epochs_shape=(num_electrodes, num_lags), skip_sigs=skip_sigs)
        save_encoding_results(encode_cluster_results, results_file)
        per_offset_results[word_offset] = encode_cluster_results
    return (
        X,
        Y,
        get_ridge_model_constructor,
        num_electrodes,
        num_lags,
        per_offset_results,
    )


@app.cell(hide_code=True)
def _(
    averaged_embeddings,
    df_word,
    epochs,
    epochs_data,
    get_arbitrary_embeddings,
    get_ridge_model_constructor,
    np,
    num_electrodes,
    num_lags,
    per_offset_results,
    train_encoding,
):
    # Add in controls
    # Arbitrary models
    original_folds = list(zip(per_offset_results[0]["train_idxs"], per_offset_results[0]["test_idxs"]))
    arbitrary_embeddings = get_arbitrary_embeddings(averaged_embeddings, epochs, df_word).astype(np.float32)
    corrs_arb, _, _, _, _ = train_encoding(arbitrary_embeddings, epochs_data, get_ridge_model_constructor(), epochs_shape=(num_electrodes, num_lags), provided_idxs=original_folds)
    return corrs_arb, original_folds


@app.cell
def _():
    plotting_run_names_to_include = ["mid_gpt2"]
    plotting_offsets_to_include = [0]
    return plotting_offsets_to_include, plotting_run_names_to_include


@app.cell(hide_code=True)
def _(
    Path,
    corrs_arb,
    get_encoding_result_filepath,
    load_encoding_results,
    plot_embedding_correlations,
    plotting_offsets_to_include,
    plotting_run_names_to_include,
    plt,
):
    def plot_multiple_offset_results(all_offset_results):
        offset_plots = []
        linestyles = ["-", "--"]

        for i, (run_name, offset_results) in enumerate(all_offset_results.items()):
            colors = ["green", "teal", "purple", "orange", "brown", "gray", "olive", "cyan"]

            for j, (offset, offset_result) in enumerate(sorted(offset_results.items(), key=lambda x: x[0])):
                offset_plots.append({"corrs": offset_result["encoding_corrs"], "color": colors[j % len(colors)], "label": f"{run_name}-{offset}", "linestyle": linestyles[i % len(linestyles)]})

        plot_embedding_correlations(plot_corrs= offset_plots + [
            {"corrs": corrs_arb, "color": "red", "label": "Arbitrary", "linestyle": "-"},
        ])


    def get_all_offset_results(run_names, offsets_to_include):
        all_offset_results = {}
        for curr_name in run_names:
            dirs = [d for d in Path(".").iterdir() if d.is_dir() and d.name.startswith(curr_name)]
            all_offset_results[curr_name] = {}
            for dir in dirs:
                curr_word_offset = int(dir.name.removeprefix(f"{curr_name}-"))
                if offsets_to_include:
                    if curr_word_offset not in offsets_to_include:
                        continue
                filepath = get_encoding_result_filepath(dir)
                all_offset_results[curr_name][curr_word_offset] = load_encoding_results(filepath)

        return all_offset_results



    plot_multiple_offset_results(get_all_offset_results(plotting_run_names_to_include, plotting_offsets_to_include))
    plt.show()
    return (get_all_offset_results,)


@app.cell
def _(mo):
    mo.md("""
    ## Decomposition
    """)
    return


@app.cell
def _():
    # Want a path to generate various decomposition of our weights and compare the results
    # Paths:
    # PCA over model weights, PLS over the data itself, dictionary learning on raw weights, dictionary learning on SVD vectors of weights, dictionary learning on SVD decomposition with some number of components, dictionary learning on word embeddings after applying top k singular vectors
    # Metrics:
    # K-directions prediction encoding
    # Self-guided measures of interpretability (maybe later LLM help)
    # variance captured
    return


@app.cell
def _():
    decom_run_names = ["mid_gpt2"]
    decom_offsets = [0]
    return decom_offsets, decom_run_names


@app.cell(hide_code=True)
def _(
    absorb_scaler,
    decom_offsets,
    decom_run_names,
    get_all_offset_results,
    np,
    num_electrodes,
    num_lags,
):
    def get_offset_model_coeffients(offset_results):
        # Get model coefficients from all results
        all_coefficients, all_intercepts = {}, {}
        for curr_run_name, run_offset_results in offset_results.items():
            run_coefficients = {}
            run_intercepts = {}
            for curr_offset, offset_result in run_offset_results.items():
                fold_coefficients = []
                fold_intercepts = []
                encoding_models = offset_result["encoding_models"]
                for fold_model in encoding_models:
                    model_coefficients, model_intercepts = absorb_scaler(fold_model, num_electrodes, num_lags)
                    fold_coefficients.append(model_coefficients)
                    fold_intercepts.append(model_intercepts)
                run_coefficients[curr_offset] = np.array(fold_coefficients)
                run_intercepts[curr_offset] = np.array(fold_intercepts)
            all_coefficients[curr_run_name] = run_coefficients
            all_intercepts[curr_run_name] = run_intercepts

        return all_coefficients, all_intercepts


    def merge_result_coefficients(coefficients_dict, run_names, offsets):
        coefficients = []
        for run_name in run_names:
            run_coefficients = []
            for offset in offsets:
                run_coefficients.append(coefficients_dict[run_name][offset])
            coefficients.append(np.stack(run_coefficients))
        return np.stack(coefficients)

    decom_offset_results = get_all_offset_results(decom_run_names, decom_offsets)
    decom_coefficients_dict, decom_intercepts_dict = get_offset_model_coeffients(decom_offset_results)
    decom_coefficients = merge_result_coefficients(decom_coefficients_dict, decom_run_names, decom_offsets)
    print("\n".join([f"{name}: {shape}" for name, shape in zip(["Runs", "Offsets", "Folds", "Elecs", "Lags", "Embed Dim"], decom_coefficients.shape)]))
    return decom_coefficients, decom_offset_results


@app.cell
def _(torch):
    print(torch.version.cuda)
    return


@app.cell
def _():
    from sklearn.decomposition import MiniBatchDictionaryLearning
    from joblib import Parallel, delayed

    return MiniBatchDictionaryLearning, Parallel, delayed


@app.cell(hide_code=True)
def _(DictionaryLearning, MiniBatchDictionaryLearning, np):
    def pca_on_weights(weights, n_components=None):
        """
        Run PCA on a weight matrix to find axes of maximal variance across models.
        Parameters
        ----------
        weights : array-like, shape (n_folds, num_models, embed_dim)
            Each row is one model's projection vector in embedding space.
            PCA is run independently per fold.
        n_components : int or None
            Number of principal components to return. Defaults to all.
        Returns
        -------
        components : ndarray, shape (n_folds, n_components, embed_dim)
        explained_variance : ndarray, shape (n_folds, n_components)
        explained_variance_ratio : ndarray, shape (n_folds, n_components)
        projected : ndarray, shape (n_folds, num_models, n_components)
        """
        W = np.asarray(weights, dtype=np.float64)
        n_folds, n_models, embed_dim = W.shape
        if n_components is None:
            n_components = min(n_models, embed_dim)

        all_components = np.empty((n_folds, n_components, embed_dim))
        all_var = np.empty((n_folds, n_components))
        all_ratio = np.empty((n_folds, n_components))
        all_proj = np.empty((n_folds, n_models, n_components))

        for f in range(n_folds):
            mean = W[f].mean(axis=0)
            W_centered = W[f] - mean

            U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)

            all_components[f] = Vt[:n_components]
            singular_values = S[:n_components]
            all_var[f] = singular_values ** 2 / (n_models - 1)
            total_variance = np.sum(S ** 2) / (n_models - 1)
            all_ratio[f] = all_var[f] / total_variance
            all_proj[f] = W_centered @ all_components[f].T

        return all_components, all_var, all_ratio, all_proj


    def dictionary_learning_on_weights(weights, n_components=None, alpha=1.0, **kwargs):
        """
        Run Dictionary Learning on a weight matrix to find sparse components across models.
        Parameters
        ----------
        weights : array-like, shape (n_folds, num_models, embed_dim)
            Each row is one model's projection vector in embedding space.
            Dictionary learning is run independently per fold.
        n_components : int or None
            Number of dictionary atoms to learn. Defaults to num_models.
        alpha : float
            Sparsity controlling parameter (regularization).
        **kwargs
            Additional keyword arguments passed to sklearn DictionaryLearning.
        Returns
        -------
        components : ndarray, shape (n_folds, n_components, embed_dim)
            Learned dictionary atoms.
        codes : ndarray, shape (n_folds, num_models, n_components)
            Sparse codes (each model expressed as a sparse combination of atoms).
        errors : ndarray, shape (n_folds,)
            Reconstruction error per fold.
        """
        W = np.asarray(weights, dtype=np.float64)
        n_folds, n_models, embed_dim = W.shape
        if n_components is None:
            n_components = n_models

        all_components = np.empty((n_folds, n_components, embed_dim))
        all_codes = np.empty((n_folds, n_models, n_components))
        all_errors = np.empty((n_folds,))

        for f in range(n_folds):
            dl = DictionaryLearning(
                n_components=n_components,
                alpha=alpha,
                fit_algorithm="lars",
                transform_algorithm="lasso_lars",
                **kwargs,
            )
            codes = dl.fit_transform(W[f])  # (n_models, n_components)

            all_components[f] = dl.components_
            all_codes[f] = codes
            all_errors[f] = np.mean(np.sum((W[f] - codes @ dl.components_) ** 2, axis=1))

        return all_components, all_codes, all_errors


    def mini_batch_dictionary_learning_on_weights(weights, n_components=None, alpha=1.0, batch_size=None, **kwargs):
        """
        Run MiniBatch Dictionary Learning on a weight matrix.

        Parameters
        ----------
        weights : array-like, shape (n_folds, num_models, embed_dim)
        n_components : int or None
            Number of dictionary atoms. Defaults to num_models.
        alpha : float
            Sparsity controlling parameter.
        batch_size : int or None
            Mini-batch size. Defaults to num_models // 4.
        **kwargs
            Additional keyword arguments passed to MiniBatchDictionaryLearning.

        Returns
        -------
        components : ndarray, shape (n_folds, n_components, embed_dim)
        codes : ndarray, shape (n_folds, num_models, n_components)
        errors : ndarray, shape (n_folds,)
        """
        W = np.asarray(weights, dtype=np.float64)
        n_folds, n_models, embed_dim = W.shape
        if n_components is None:
            n_components = n_models
        if batch_size is None:
            batch_size = max(1, n_models // 4)

        all_components = np.empty((n_folds, n_components, embed_dim))
        all_codes = np.empty((n_folds, n_models, n_components))
        all_errors = np.empty((n_folds,))

        for f in range(n_folds):
            dl = MiniBatchDictionaryLearning(
                n_components=n_components,
                alpha=alpha,
                batch_size=batch_size,
                **kwargs,
            )
            codes = dl.fit_transform(W[f])
            recon = codes @ dl.components_
            all_components[f] = dl.components_
            all_codes[f] = codes
            all_errors[f] = np.mean(np.sum((W[f] - recon) ** 2, axis=1))

        return all_components, all_codes, all_errors


    return (
        dictionary_learning_on_weights,
        mini_batch_dictionary_learning_on_weights,
        pca_on_weights,
    )


@app.cell(hide_code=True)
def _(np):
    # Take fold 0 models and identify top PCA components
    # fold_coefficients = decom_coefficients[:, :, fold_selection.value].reshape((-1, decom_coefficients.shape[-1]))
    # Limit to just those that are significant
    def get_significant_coefficients(coefficients, offset_results):
        significant_coefficients = []
        # all_significant_bools = []
        for i, (curr_run_name, run_results) in enumerate(offset_results.items()):
            for j, (offset, offset_result) in enumerate(run_results.items()):
                # Change so that folds are the first array
                significant_coefficients.append(coefficients[i, j, :, offset_result["sig"]["sig_bool"]].transpose(1, 0, 2))
                # all_significant_bools.append(offset_result["sig"]["sig_bool"])
        num_runs = len(offset_results)
        num_offsets = len(list(offset_results.items())[0])
        def unflatten_model_data(model_data):
            """Expand out the flattened array from gathering only significant models back into the original form
            of [num_runs, num_offsets, num_folds (if passed), num_elecs, num_lags, embed_dim (or whatever else is here)]

                model_data must be generated by this function or be of the same form.
            """
            # 
            significant_results_counter = 0
            # Exclude folds
            if model_data.ndim == 2:
                expanded_data = np.zeros((coefficients.shape[0], coefficients.shape[1], coefficients.shape[3], coefficients.shape[4], model_data.shape[-1]))
            elif model_data.ndim == 3:
                expanded_data = np.zeros((*coefficients.shape[:-1], model_data.shape[-1]))
            else:
                raise RuntimeError(f"Illegal model data. Shape: {model_data.shape}")

            for i, (curr_run_name, run_results) in enumerate(offset_results.items()):
                for j, (offset, offset_result) in enumerate(run_results.items()):
                    num_significant_results = np.sum(offset_result["sig"]["sig_bool"])
                    if model_data.ndim == 2:
                        expanded_data[i, j, offset_result["sig"]["sig_bool"]] = model_data[significant_results_counter:significant_results_counter + num_significant_results]
                    elif model_data.ndim == 3:
                        # print(expanded_data[i, j, :, offset_result["sig"]["sig_bool"]].shape)
                        # print(model_data[:, significant_results_counter:significant_results_counter + num_significant_results].shape)
                        expanded_data[i, j, :, offset_result["sig"]["sig_bool"]] = model_data[:, significant_results_counter:significant_results_counter + num_significant_results].transpose(1, 0, 2)
                    significant_results_counter += num_significant_results

            return expanded_data

        return np.concatenate(significant_coefficients, axis=1), unflatten_model_data

    return (get_significant_coefficients,)


@app.cell
def _(decom_coefficients, decom_offset_results, get_significant_coefficients):
    significant_coefficients, unflatten_coefficients_fn = get_significant_coefficients(decom_coefficients, decom_offset_results)
    print(f"Significant Coefficients Shape: {significant_coefficients.shape}")
    return significant_coefficients, unflatten_coefficients_fn


@app.cell(hide_code=True)
def _(generate_example_table, get_example_display_controls, mo, np, plt):
    from matplotlib.colors import TwoSlopeNorm

    def get_pca_explorer_controls(n_components, n_folds):
        """Create marimo UI controls for PCA exploration."""
        fold_selector = mo.ui.dropdown(
            options={f"Fold {f}": f for f in range(n_folds)},
            value=f"Fold 0",
            label="Fold",
        )
        component_slider = mo.ui.slider(
            start=0, stop=n_components - 1, value=0, step=1, label="Component"
        )
        return fold_selector, component_slider


    def get_component_activations(components, fold, component_idx, embeddings):
        """
        Project word embeddings onto a single PCA component to get per-word activations.

        Parameters
        ----------
        components : ndarray, shape (n_folds, n_components, embed_dim)
        fold : int
        component_idx : int
        embeddings : ndarray, shape (n_words, embed_dim)

        Returns
        -------
        activations : ndarray, shape (n_words,)
        """
        direction = components[fold, component_idx]  # (embed_dim,)
        return embeddings @ direction


    def plot_component_loadings(
        proj,
        fold: int,
        component_idx: int,
        n_electrodes: int,
        n_lags: int,
        elecs_per_subject: list[int] | None = None,
        lag_offset: int = 0,
    ):
        """
        Show how a PCA component loads onto electrodes, lags, and subjects.

        Parameters
        ----------
        proj : ndarray, shape (n_folds, n_models, n_components)
            Projected weights from pca_on_weights.
        fold : int
        component_idx : int
        n_electrodes : int
        n_lags : int
        elecs_per_subject : list[int] or None
            Number of electrodes per subject for subject-level aggregation.
        lag_offset : int
            Offset to add to lag indices for display (e.g., if lags start at -200ms).
        """
        loadings = proj[fold, :, component_idx]  # (n_models,)
        loading_matrix = loadings.reshape(n_electrodes, n_lags)

        # Aggregate across axes
        elec_loadings = loading_matrix.mean(axis=1)   # avg over lags
        lag_loadings = loading_matrix.mean(axis=0)     # avg over electrodes

        # Subject info
        subjects = []
        if elecs_per_subject is not None:
            cursor = 0
            for subj_idx, n in enumerate(elecs_per_subject):
                if n > 0:
                    subjects.append((f"S{subj_idx + 1}", slice(cursor, cursor + n)))
                cursor += n

        n_cols = 3 if subjects else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9))
        fig.suptitle(
            f"Component {component_idx} loadings (Fold {fold})",
            fontsize=14,
        )

        cmap = plt.cm.RdBu_r
        vmax = max(abs(elec_loadings.min()), abs(elec_loadings.max()), 1e-12)
        norm_elec = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        vmax_lag = max(abs(lag_loadings.min()), abs(lag_loadings.max()), 1e-12)
        norm_lag = TwoSlopeNorm(vmin=-vmax_lag, vcenter=0, vmax=vmax_lag)

        x_elec = np.arange(n_electrodes)
        x_lag = np.arange(n_lags) + lag_offset

        # ── Row 1: Mean loadings ──────────────────────────────────────────────────
        ax = axes[0, 0]
        colors = cmap(norm_elec(elec_loadings))
        ax.bar(x_elec, elec_loadings, color=colors, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Electrode index")
        ax.set_ylabel("Mean loading")
        ax.set_title("Mean loading per electrode")
        ax.set_xlim(-0.5, n_electrodes - 0.5)

        ax = axes[0, 1]
        colors = cmap(norm_lag(lag_loadings))
        ax.bar(x_lag, lag_loadings, color=colors, edgecolor="white", linewidth=0.3, width=1.0)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Mean loading")
        ax.set_title("Mean loading per lag")
        ax.set_xlim(x_lag[0] - 0.5, x_lag[-1] + 0.5)

        if subjects:
            ax = axes[0, 2]
            subj_labels = [lbl for lbl, _ in subjects]
            subj_loadings = np.array([
                loading_matrix[sl].mean() for _, sl in subjects
            ])
            vmax_subj = max(abs(subj_loadings.min()), abs(subj_loadings.max()), 1e-12)
            norm_subj = TwoSlopeNorm(vmin=-vmax_subj, vcenter=0, vmax=vmax_subj)
            colors = cmap(norm_subj(subj_loadings))
            x_subj = np.arange(len(subjects))
            bars = ax.bar(x_subj, subj_loadings, color=colors, edgecolor="white", linewidth=0.3)
            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_xlabel("Subject")
            ax.set_ylabel("Mean loading")
            ax.set_title("Mean loading per subject")
            ax.set_xticks(x_subj)
            ax.set_xticklabels(subj_labels)
            for bar, val in zip(bars, subj_loadings):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + (vmax_subj * 0.02 * np.sign(val)),
                    f"{val:.4f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=7,
                )

        # ── Row 2: Heatmap and variance ───────────────────────────────────────────
        ax = axes[1, 0]
        vmax_mat = max(abs(loading_matrix.min()), abs(loading_matrix.max()), 1e-12)
        im = ax.imshow(
            loading_matrix,
            aspect="auto",
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-vmax_mat, vcenter=0, vmax=vmax_mat),
            origin="lower",
            extent=[x_lag[0] - 0.5, x_lag[-1] + 0.5, -0.5, n_electrodes - 0.5],
        )
        ax.set_xlabel("Lag")
        ax.set_ylabel("Electrode index")
        ax.set_title("Loading matrix (electrode × lag)")
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Std across lags per electrode (how variable is this component over time)
        ax = axes[1, 1]
        elec_std = loading_matrix.std(axis=1)
        ax.bar(x_elec, elec_std, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Electrode index")
        ax.set_ylabel("Std of loading across lags")
        ax.set_title("Temporal variability per electrode")
        ax.set_xlim(-0.5, n_electrodes - 0.5)

        if subjects:
            # Per-subject stacked temporal profiles
            ax = axes[1, 2]
            from matplotlib.cm import tab10
            subj_colors = tab10(np.linspace(0, 1, len(subjects)))
            for i, ((label, sl), color) in enumerate(zip(subjects, subj_colors)):
                subj_temporal = loading_matrix[sl].mean(axis=0)
                ax.plot(x_lag, subj_temporal, color=color, label=label, linewidth=1.2)
            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_xlabel("Lag")
            ax.set_ylabel("Mean loading")
            ax.set_title("Temporal profile per subject")
            ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")

        plt.tight_layout()
        return fig


    def pca_explorer(
        components,
        proj,
        var_ratio,
        embeddings,
        words,
        df_word,
        epochs,
        n_electrodes,
        n_lags,
        elecs_per_subject=None,
        lag_offset=0,
    ):
        """
        Build a full marimo PCA explorer.

        Parameters
        ----------
        components : ndarray, shape (n_folds, n_components, embed_dim)
        proj : ndarray, shape (n_folds, n_models, n_components)
        var_ratio : ndarray, shape (n_folds, n_components)
        embeddings : ndarray, shape (n_words, embed_dim)
        words : list of str
        df_word : DataFrame with 'word' column
        epochs : marimo selection object
        n_electrodes : int
        n_lags : int
        elecs_per_subject : list[int] or None
        lag_offset : int
        """
        n_folds, n_comp, _ = components.shape

        fold_selector, component_slider = get_pca_explorer_controls(n_comp, n_folds)
        num_bins_slider, samples_per_bin_slider, context_slider, seed_input = get_example_display_controls()

        def render():
            fold = fold_selector.value
            comp_idx = component_slider.value
            pct = var_ratio[fold, comp_idx] * 100

            activations = get_component_activations(components, fold, comp_idx, embeddings)

            loading_fig = plot_component_loadings(
                proj, fold, comp_idx,
                n_electrodes, n_lags,
                elecs_per_subject=elecs_per_subject,
                lag_offset=lag_offset,
            )

            example_table = generate_example_table(
                activations,
                num_bins_slider, samples_per_bin_slider, context_slider, seed_input,
            )

            return mo.vstack([
                mo.md(f"# PCA Component Explorer"),
                mo.hstack([fold_selector, component_slider], justify="start", gap="1rem"),
                mo.md(f"**Component {comp_idx}** — explains {pct:.2f}% of variance"),
                mo.md("## Component Loadings (electrode × lag × subject)"),
                mo.as_html(loading_fig),
                mo.md("## Word Activations"),
                mo.hstack(
                    [num_bins_slider, samples_per_bin_slider, context_slider, seed_input],
                    justify="start", gap="1rem",
                ),
                example_table,
            ])

        return render

    return (
        get_component_activations,
        get_pca_explorer_controls,
        plot_component_loadings,
    )


@app.cell
def _(
    get_example_display_controls,
    get_pca_explorer_controls,
    mo,
    pca_on_weights,
    significant_coefficients,
):
    n_components = 1600
    components, var, var_ratio, proj = pca_on_weights(significant_coefficients, n_components=n_components)
    print(f"Top {n_components} PCs explain fold 1: {var_ratio[0].sum():.1%}, fold 2: {var_ratio[1].sum():.1%} of variance")

    n_folds, n_comp, _ = components.shape

    fold_selector, component_slider = get_pca_explorer_controls(n_comp, n_folds)
    num_bins_slider, samples_per_bin_slider, context_slider, seed_input = get_example_display_controls()

    mo.hstack([
        fold_selector, component_slider,
        num_bins_slider, samples_per_bin_slider, context_slider, seed_input,
    ], justify="start", gap="1rem")
    return (
        component_slider,
        components,
        context_slider,
        fold_selector,
        n_folds,
        num_bins_slider,
        proj,
        samples_per_bin_slider,
        seed_input,
        var_ratio,
    )


@app.cell
def _(
    averaged_embeddings,
    component_slider,
    components,
    fold_selector,
    get_component_activations,
    mo,
    var_ratio,
):
    fold = fold_selector.value
    comp_idx = component_slider.value
    pct = var_ratio[fold, comp_idx] * 100

    activations = get_component_activations(components, fold, comp_idx, averaged_embeddings)

    mo.md(f"**Component {comp_idx}** — explains {pct:.2f}% of variance")
    return activations, comp_idx, fold


@app.cell
def _(
    comp_idx,
    fold,
    mo,
    plot_component_loadings,
    proj,
    raws,
    unflatten_coefficients_fn,
):
    subject_electrode_splits = [len(r.ch_names) for r in raws]
    subject_electrode_splits.insert(1, 0)

    loading_fig = plot_component_loadings(
        unflatten_coefficients_fn(proj)[0][0], fold, comp_idx,
        n_electrodes=183,
        n_lags=128,
        elecs_per_subject=subject_electrode_splits,
        lag_offset=-200,
    )
    mo.as_html(loading_fig)
    return


@app.cell
def _(
    activations,
    context_slider,
    generate_example_table,
    num_bins_slider,
    samples_per_bin_slider,
    seed_input,
):
    generate_example_table(
        activations,
        num_bins_slider, samples_per_bin_slider, context_slider, seed_input,
    )
    return


@app.cell
def _(np):
    def find_dominant_directions(weights, n_directions=50):
        """
        Find dominant directions in embedding space that models attend to.

        Parameters
        ----------
        weights : ndarray, shape (n_folds, n_models, embed_dim)
        n_directions : int

        Returns
        -------
        directions : ndarray, shape (n_folds, n_directions, embed_dim)
            Unit-norm dominant directions.
        alignments : ndarray, shape (n_folds, n_models, n_directions)
            Dot product of each model's weight with each direction (signed).
        """
        W = np.asarray(weights, dtype=np.float64)
        n_folds, n_models, embed_dim = W.shape

        # Subtract out mean
        # W = W - W.mean(0)

        all_Vt = np.empty((n_folds, n_directions, embed_dim))
        all_alignments = np.empty((n_folds, n_models, n_directions))
        all_S = np.empty((n_folds, n_directions))
        all_U = np.empty((n_folds, n_models, n_directions))

        for f in range(n_folds):
            # Normalize each model's weights to unit length
            norms = np.linalg.norm(W[f], axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            W_unit = W[f] / norms

            # SVD on the UN-centered matrix
            # Left singular vectors weighted by singular values give
            # the directions that best explain the actual weight vectors
            # (not deviations from the mean)
            U, S, Vt = np.linalg.svd(W_unit, full_matrices=False)

            all_Vt[f] = Vt[:n_directions]
            # Alignment: how much does each model point along each direction
            all_alignments[f] = W[f] @ Vt[:n_directions].T
            all_S[f] = S[:n_directions]
            all_U[f] = U.T[:n_directions].T

        return all_U, all_S, all_Vt, all_alignments

    return (find_dominant_directions,)


@app.cell
def _(find_dominant_directions, significant_coefficients):
    all_Us, all_S, all_Vt, all_alignments = find_dominant_directions(significant_coefficients, n_directions=1600)
    return all_S, all_Us, all_Vt


@app.cell
def _():
    # W_random = np.random.randn(*significant_coefficients.shape)
    # rand_Us, rand_S, rand_Vt, rand_alignments = find_dominant_directions(W_random, n_directions=1600)
    return


@app.cell
def _(all_S, np):
    cumulative = np.cumsum(all_S ** 2, axis=1) / np.sum(all_S ** 2, axis=1, keepdims=True)
    return (cumulative,)


@app.cell
def _(cumulative, plt):
    plt.plot(cumulative[0])
    plt.plot(cumulative[1])
    plt.show()
    return


@app.cell
def _(all_S, plt):
    plt.plot(all_S[0])
    plt.plot(all_S[1])
    plt.show()
    return


@app.cell(hide_code=True)
def _(absorb_scaler, np, plt, torch):
    # TODO: Make this work with multiple models

    def _correlation_per_output(Y_true, Y_pred):
        """Pearson correlation per output column."""
        Y_true = Y_true - Y_true.mean(axis=0)
        Y_pred = Y_pred - Y_pred.mean(axis=0)
        num = np.sum(Y_true * Y_pred, axis=0)
        denom = np.sqrt(np.sum(Y_true ** 2, axis=0) * np.sum(Y_pred ** 2, axis=0))
        return num / np.maximum(denom, 1e-12)


    # def evaluate_random_basis_null(
    #     encoding_models,
    #     test_idxs,
    #     X,
    #     Y,
    #     num_electrodes,
    #     num_lags,
    #     k_values,
    #     sig_mask,
    #     W_mean=None,
    #     scalers=None,
    #     n_null=100,
    #     embed_dim=None,
    #     seed=42,
    # ):
    #     """
    #     Null hypothesis test: replace SVD basis with random orthonormal directions.

    #     Parameters
    #     ----------
    #     n_null : int
    #         Number of random bases to sample per k.
    #     embed_dim : int or None
    #         Embedding dimension. If None, inferred from model weights.
    #     seed : int
    #         Random seed for reproducibility.

    #     Returns
    #     -------
    #     dict with keys:
    #         'k_values': array of k values tested
    #         'full_corrs': mean full encoding performance (num_electrodes, num_lags)
    #         'null_corrs_mean': mean over null iterations (len(k_values), num_electrodes, num_lags)
    #         'null_corrs_std': std over null iterations (len(k_values), num_electrodes, num_lags)
    #         'null_corrs_per_iter': all null correlations (len(k_values), n_null, num_electrodes, num_lags)
    #         'sig_mask': sig_mask
    #         'full_corrs_per_fold': (n_folds, num_electrodes, num_lags)
    #     """
    #     rng = np.random.default_rng(seed)
    #     n_folds = len(encoding_models)
    #     k_values = np.array(k_values)

    #     full_corrs = np.zeros((n_folds, num_electrodes, num_lags))
    #     # (n_k, n_null, n_folds, num_electrodes, num_lags)
    #     null_corrs = np.zeros((len(k_values), n_null, n_folds, num_electrodes, num_lags))

    #     for f, model in enumerate(encoding_models):
    #         X_test = X[f][test_idxs[f]]
    #         Y_test = Y[test_idxs[f]]
    #         if scalers is not None:
    #             Y_test = scalers[f].transform(Y_test)

    #         W, intercept = absorb_scaler(model, num_electrodes, num_lags)
    #         W = W.reshape((-1, W.shape[-1]))

    #         if embed_dim is None:
    #             embed_dim = W.shape[-1]

    #         # Full performance (same as original function)
    #         Y_pred_full = X_test @ W.T + intercept
    #         corr_full = _correlation_per_output(Y_test, Y_pred_full)
    #         full_corrs[f] = corr_full.reshape(num_electrodes, num_lags)

    #         # Random basis null — reuse same truncation logic as evaluate_svd_truncation
    #         for ki, k in enumerate(k_values):
    #             print(k)
    #             for ni in range(n_null):
    #                 print(ni)
    #                 rand_basis, _ = np.linalg.qr(rng.standard_normal((embed_dim, k)))
    #                 rand_basis = rand_basis.T  # (k, embed_dim), same shape as Vt_sig[:k]

    #                 if W_mean is not None:
    #                     W_trunc = ((W - W_mean[f]) @ rand_basis.T) @ rand_basis
    #                     W_trunc += W_mean[f]
    #                 else:
    #                     W_trunc = (W @ rand_basis.T) @ rand_basis

    #                 Y_pred_trunc = X_test @ W_trunc.T + intercept
    #                 print("corr")
    #                 corr_trunc = _correlation_per_output(Y_test, Y_pred_trunc)
    #                 print("done")
    #                 null_corrs[ki, ni, f] = corr_trunc.reshape(num_electrodes, num_lags)

    #     full_mean = full_corrs.mean(axis=0)
    #     # Average over folds first, then summarize over null iterations
    #     null_corrs_per_iter = null_corrs.mean(axis=2)  # (n_k, n_null, num_electrodes, num_lags)

    #     return {
    #         'k_values': k_values,
    #         'full_corrs': full_mean,
    #         'null_corrs_mean': null_corrs_per_iter.mean(axis=1),
    #         'null_corrs_std': null_corrs_per_iter.std(axis=1),
    #         'null_corrs_per_iter': null_corrs_per_iter,
    #         'sig_mask': sig_mask,
    #         'full_corrs_per_fold': full_corrs,
    #     }


    def evaluate_random_basis_null_cpu(
        encoding_models,
        test_idxs,
        X,
        Y,
        num_electrodes,
        num_lags,
        k_values,
        sig_mask,
        W_mean=None,
        scalers=None,
        n_null=100,
        null_chunk_size=10,
        embed_dim=None,
        seed=42,
    ):
        """
        Parameters
        ----------
        encoding_models : list
        test_idxs : list
        X : list of ndarrays
        Y : ndarray
        num_electrodes : int
        num_lags : int
        k_values : list of int
        sig_mask : ndarray, shape (n_electrodes, n_lags), dtype bool
        W_mean : list of ndarrays or None
        scalers : list or None
        n_null : int
            Number of random bases to sample per k.
        null_chunk_size : int
            Number of null iterations to process at once. Reduce if OOM.
        embed_dim : int or None
            Embedding dimension. If None, inferred from model weights.
        seed : int
        """
        rng = np.random.default_rng(seed)
        n_folds = len(encoding_models)
        k_values = np.array(k_values)

        full_corrs = np.zeros((n_folds, num_electrodes, num_lags))
        null_corrs = np.zeros((len(k_values), n_null, n_folds, num_electrodes, num_lags))

        for f, model in enumerate(encoding_models):
            X_test = X[f][test_idxs[f]].astype(np.float32)
            Y_test = Y[test_idxs[f]]
            if scalers is not None:
                Y_test = scalers[f].transform(Y_test)
            Y_test = Y_test.astype(np.float32)

            W, intercept = absorb_scaler(model, num_electrodes, num_lags)
            W = W.reshape((-1, W.shape[-1])).astype(np.float32)

            if embed_dim is None:
                embed_dim = W.shape[-1]

            # Full performance
            Y_pred_full = X_test @ W.T + intercept
            corr_full = _correlation_per_output(Y_test, Y_pred_full)
            full_corrs[f] = corr_full.reshape(num_electrodes, num_lags)

            # Precompute centered Y for correlation
            Y_c = Y_test - Y_test.mean(0)
            Y_norm = np.linalg.norm(Y_c, axis=0)

            for ki, k in enumerate(k_values):
                for chunk_start in range(0, n_null, null_chunk_size):
                    chunk_end = min(chunk_start + null_chunk_size, n_null)
                    chunk_size = chunk_end - chunk_start

                    # Random orthonormal bases: (chunk_size, k, embed_dim)
                    rand_matrices = rng.standard_normal((chunk_size, embed_dim, k)).astype(np.float32)
                    Q, _ = np.linalg.qr(rand_matrices)          # (chunk_size, embed_dim, k)
                    bases = Q.transpose(0, 2, 1)                 # (chunk_size, k, embed_dim)

                    # Batched projection
                    if W_mean is not None:
                        W_centered = (W - W_mean[f]).astype(np.float32)
                        W_proj = np.einsum('ij,nkj->nik', W_centered, bases)   # (chunk_size, n_models, k)
                        W_trunc = np.einsum('nik,nkj->nij', W_proj, bases)     # (chunk_size, n_models, embed_dim)
                        W_trunc += W_mean[f].astype(np.float32)
                    else:
                        W_proj = np.einsum('ij,nkj->nik', W, bases)            # (chunk_size, n_models, k)
                        W_trunc = np.einsum('nik,nkj->nij', W_proj, bases)     # (chunk_size, n_models, embed_dim)

                    # Batched prediction: (chunk_size, n_time, n_models)
                    Y_pred_trunc = np.einsum('ti,nji->ntj', X_test, W_trunc) + intercept

                    # Batched correlation
                    Y_pred_c = Y_pred_trunc - Y_pred_trunc.mean(1, keepdims=True)
                    num = np.einsum('tj,ntj->nj', Y_c, Y_pred_c)
                    denom = Y_norm * np.linalg.norm(Y_pred_c, axis=1)
                    corr_trunc = num / np.maximum(denom, 1e-8)

                    null_corrs[ki, chunk_start:chunk_end, f] = corr_trunc.reshape(
                        chunk_size, num_electrodes, num_lags
                    )

        null_corrs_per_iter = null_corrs.mean(axis=2)  # (n_k, n_null, num_electrodes, num_lags)

        return {
            'k_values': k_values,
            'full_corrs': full_corrs.mean(axis=0),
            'null_corrs_mean': null_corrs_per_iter.mean(axis=1),
            'null_corrs_std': null_corrs_per_iter.std(axis=1),
            'null_corrs_per_iter': null_corrs_per_iter,
            'sig_mask': sig_mask,
            'full_corrs_per_fold': full_corrs,
        }



    def _correlation_per_output_torch(Y_true, Y_pred):
        """(n_time, n_models) -> (n_models,)"""
        Y_true = Y_true - Y_true.mean(0)
        Y_pred = Y_pred - Y_pred.mean(0)
        num = (Y_true * Y_pred).sum(0)
        denom = Y_true.norm(dim=0) * Y_pred.norm(dim=0)
        return num / denom.clamp(min=1e-8)


    def _correlation_per_output_torch_batched(Y_true, Y_pred_batch):
        """
        Y_true:      (n_time, n_models)
        Y_pred_batch: (n_null, n_time, n_models)
        returns:     (n_null, n_models)
        """
        Y_true = Y_true - Y_true.mean(0)
        Y_pred_batch = Y_pred_batch - Y_pred_batch.mean(1, keepdim=True)
        num = (Y_true.unsqueeze(0) * Y_pred_batch).sum(1)                             # (n_null, n_models)
        denom = Y_true.norm(dim=0) * Y_pred_batch.norm(dim=1)                         # (n_null, n_models)
        return num / denom.clamp(min=1e-8)


    def evaluate_random_basis_null_gpu(
        encoding_models,
        test_idxs,
        X,
        Y,
        num_electrodes,
        num_lags,
        k_values,
        sig_mask,
        W_mean=None,
        scalers=None,
        n_null=100,
        null_chunk_size=10,  # tune this down if still OOM
        embed_dim=None,
        seed=42,
        device="cuda",
    ):
        torch.manual_seed(seed)
        n_folds = len(encoding_models)
        k_values = np.array(k_values)
        dtype = torch.float32

        full_corrs = np.zeros((n_folds, num_electrodes, num_lags))
        null_corrs = np.zeros((len(k_values), n_null, n_folds, num_electrodes, num_lags))

        for f, model in enumerate(encoding_models):
            X_test = X[f][test_idxs[f]]
            Y_test = Y[test_idxs[f]]
            if scalers is not None:
                Y_test = scalers[f].transform(Y_test)

            W, intercept = absorb_scaler(model, num_electrodes, num_lags)
            W = W.reshape((-1, W.shape[-1]))

            if embed_dim is None:
                embed_dim = W.shape[-1]

            # Move to GPU once per fold
            W_gpu = torch.tensor(W, dtype=dtype, device=device)                        # (n_models, embed_dim)
            X_gpu = torch.tensor(X_test, dtype=dtype, device=device)                   # (n_time, embed_dim)
            Y_gpu = torch.tensor(Y_test, dtype=dtype, device=device)                   # (n_time, n_models)
            intercept_gpu = torch.tensor(intercept, dtype=dtype, device=device)        # (n_models,)

            if W_mean is not None:
                W_mean_gpu = torch.tensor(W_mean[f], dtype=dtype, device=device)

            # Full performance
            Y_pred_full = X_gpu @ W_gpu.T + intercept_gpu
            full_corrs[f] = _correlation_per_output_torch(Y_gpu, Y_pred_full).cpu().numpy().reshape(num_electrodes, num_lags)

            for ki, k in enumerate(k_values):
                print(ki)
                chunks = range(0, n_null, null_chunk_size)
                for chunk_start in chunks:
                    print(chunk_start, n_null)
                    chunk_end = min(chunk_start + null_chunk_size, n_null)
                    chunk_size = chunk_end - chunk_start

                    rand_matrices = torch.randn(chunk_size, embed_dim, k, dtype=dtype, device=device)
                    Q, _ = torch.linalg.qr(rand_matrices)
                    bases = Q.permute(0, 2, 1)  # (chunk_size, k, embed_dim)

                    if W_mean is not None:
                        W_centered = W_gpu - W_mean_gpu
                        W_trunc = torch.bmm(
                            W_centered.unsqueeze(0).expand(chunk_size, -1, -1) @ bases.permute(0, 2, 1),
                            bases
                        ) + W_mean_gpu
                    else:
                        W_trunc = torch.bmm(
                            W_gpu.unsqueeze(0).expand(chunk_size, -1, -1) @ bases.permute(0, 2, 1),
                            bases
                        )

                    Y_pred_trunc = torch.bmm(
                        X_gpu.unsqueeze(0).expand(chunk_size, -1, -1),
                        W_trunc.permute(0, 2, 1)
                    ) + intercept_gpu

                    corr_trunc = _correlation_per_output_torch_batched(Y_gpu, Y_pred_trunc)
                    null_corrs[ki, chunk_start:chunk_end, f] = (
                        corr_trunc.cpu().numpy().reshape(chunk_size, num_electrodes, num_lags)
                    )

        null_corrs_per_iter = null_corrs.mean(axis=2)

        return {
            'k_values': k_values,
            'full_corrs': full_corrs.mean(axis=0),
            'null_corrs_mean': null_corrs_per_iter.mean(axis=1),
            'null_corrs_std': null_corrs_per_iter.std(axis=1),
            'null_corrs_per_iter': null_corrs_per_iter,
            'sig_mask': sig_mask,
            'full_corrs_per_fold': full_corrs,
        }


    def evaluate_svd_truncation(
        encoding_models,
        test_idxs,
        X,
        Y,
        num_electrodes,
        num_lags,
        k_values,
        sig_mask,
        W_mean=None,
        scalers=None,
        svd=None,
    ):
        """
        Parameters
        ----------
        sig_mask : ndarray, shape (n_electrodes, n_lags), dtype bool
            Which electrode/lag pairs are significant.
        svd : tuple (U, S, Vt) or None
            Pre-computed SVD of the significant weights.
            U: (n_folds, n_sig, n_components)
            S: (n_folds, n_components)
            Vt: (n_folds, n_components, embed_dim)
            If None, SVD is computed from the model weights.
        """
        n_folds = len(encoding_models)
        k_values = np.array(k_values)
        sig_flat = sig_mask.ravel()

        full_corrs = np.zeros((n_folds, num_electrodes, num_lags))
        truncated_corrs = np.zeros((len(k_values), n_folds, num_electrodes, num_lags))

        for f, model in enumerate(encoding_models):
            X_test = X[f][test_idxs[f]]
            Y_test = Y[test_idxs[f]]
            if scalers is not None:
                Y_test = scalers[f].transform(Y_test)

            W, intercept = absorb_scaler(model, num_electrodes, num_lags)
            W = W.reshape((-1, W.shape[-1]))

            Y_pred_full = X_test @ W.T + intercept
            corr_full = _correlation_per_output(Y_test, Y_pred_full)
            full_corrs[f] = corr_full.reshape(num_electrodes, num_lags)

            if svd is not None:
                Vt_sig = svd[2][f]
            else:
                W_sig = W[sig_flat]
                _, _, Vt_sig = np.linalg.svd(W_sig, full_matrices=False)

            for ki, k in enumerate(k_values):
                basis = Vt_sig[:k]

                if W_mean is not None:
                    W_trunc = ((W - W_mean[f]) @ basis.T) @ basis
                    W_trunc += W_mean[f]
                else:
                    W_trunc = (W @ basis.T) @ basis

                Y_pred_trunc = X_test @ W_trunc.T + intercept
                corr_trunc = _correlation_per_output(Y_test, Y_pred_trunc)
                truncated_corrs[ki, f] = corr_trunc.reshape(num_electrodes, num_lags)

        full_mean = full_corrs.mean(axis=0)
        trunc_mean = truncated_corrs.mean(axis=1)

        return {
            'k_values': k_values,
            'full_corrs': full_mean,
            'truncated_corrs': trunc_mean,
            'sig_mask': sig_mask,
            'full_corrs_per_fold': full_corrs,
            'truncated_corrs_per_fold': truncated_corrs,
        }


    def plot_svd_truncation_results(results, threshold=0.95):
        k_values = results['k_values']
        full = results['full_corrs']
        trunc = results['truncated_corrs']
        sig_mask = results['sig_mask']

        full_global = full.mean()
        full_sig = full[sig_mask].mean()
        full_nonsig = full[~sig_mask].mean() if (~sig_mask).any() else 0

        trunc_global = trunc.mean(axis=(1, 2))
        trunc_sig = np.array([t[sig_mask].mean() for t in trunc])
        trunc_nonsig = np.array([t[~sig_mask].mean() for t in trunc]) if (~sig_mask).any() else np.zeros(len(k_values))

        best_k, _ = find_best_k(results, threshold)
        best_k_idx = np.where(k_values == best_k)[0][0]

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # ── Panel 1: Performance vs k, split by significance ──────────────────────
        ax = axes[0, 0]
        ax.plot(k_values, trunc_sig, 'o-', color='steelblue', markersize=5, label='Significant')
        if (~sig_mask).any():
            ax.plot(k_values, trunc_nonsig, 's-', color='gray', markersize=4,
                    alpha=0.6, label='Non-significant')
        ax.axhline(full_sig, color='steelblue', linestyle='--', alpha=0.5)
        if (~sig_mask).any():
            ax.axhline(full_nonsig, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(best_k, color='green', linestyle='--', alpha=0.5, label=f'k={best_k}')
        ax.set_xlabel("Number of SVD components (k)")
        ax.set_ylabel("Mean correlation")
        ax.set_title("Performance by significance group")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── Panel 2: Retention ratio ──────────────────────────────────────────────
        ax = axes[0, 1]
        ratio_sig = trunc_sig / full_sig
        ax.plot(k_values, ratio_sig, 'o-', color='steelblue', markersize=5,
                label='Significant')
        if (~sig_mask).any() and full_nonsig > 1e-6:
            ratio_nonsig = trunc_nonsig / full_nonsig
            ax.plot(k_values, ratio_nonsig, 's-', color='gray', markersize=4,
                    alpha=0.6, label='Non-significant')
        ax.axhline(threshold, color='red', linestyle='--', label=f'{threshold:.0%} threshold')
        ax.axvline(best_k, color='green', linestyle='--', alpha=0.5, label=f'k={best_k}')
        ax.set_xlabel("Number of SVD components (k)")
        ax.set_ylabel("Fraction of full performance")
        ax.set_title("Relative performance retention")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # ── Panel 3: Per-electrode drop at best k ─────────────────────────────────
        ax = axes[1, 0]
        elec_full = full.mean(axis=1)
        elec_trunc = trunc[best_k_idx].mean(axis=1)
        drop = 1 - elec_trunc / np.maximum(elec_full, 1e-12)

        sig_frac_per_elec = sig_mask.mean(axis=1)
        colors = plt.cm.viridis(sig_frac_per_elec)
        ax.bar(np.arange(len(drop)), drop, color=colors, edgecolor='white', linewidth=0.3)
        ax.set_xlabel("Electrode index")
        ax.set_ylabel("Fractional performance drop")
        ax.set_title(f"Per-electrode drop at k={best_k} (color=frac sig lags)")
        ax.set_xlim(-0.5, len(drop) - 0.5)
        ax.grid(True, alpha=0.3)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=ax, shrink=0.8, label='Fraction significant lags')

        # ── Panel 4: Per-lag drop at best k ───────────────────────────────────────
        ax = axes[1, 1]
        lag_full = full.mean(axis=0)
        lag_trunc = trunc[best_k_idx].mean(axis=0)
        drop = 1 - lag_trunc / np.maximum(lag_full, 1e-12)

        sig_frac_per_lag = sig_mask.mean(axis=0)
        ax.plot(drop, color='steelblue', linewidth=1.2, label='Performance drop')
        ax2 = ax.twinx()
        ax2.fill_between(np.arange(len(sig_frac_per_lag)), sig_frac_per_lag,
                         alpha=0.15, color='coral', label='Frac sig electrodes')
        ax2.set_ylabel("Fraction significant electrodes", color='coral')
        ax.set_xlabel("Lag index")
        ax.set_ylabel("Fractional performance drop")
        ax.set_title(f"Per-lag drop at k={best_k}")
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2)

        fig.suptitle("SVD Truncation Analysis (fold-averaged)", fontsize=14, y=1.01)
        fig.tight_layout()
        return fig


    def find_best_k(results, threshold=0.95):
        """Find smallest k that retains `threshold` fraction of full performance on significant outputs."""
        sig_mask = results['sig_mask']
        full_sig = results['full_corrs'][sig_mask].mean()
        trunc_sig = np.array([t[sig_mask].mean() for t in results['truncated_corrs']])
        ratios = trunc_sig / full_sig

        passing = np.where(ratios >= threshold)[0]
        if len(passing) == 0:
            print(f"No k reaches {threshold:.0%} of full performance")
            return results['k_values'][-1], ratios

        best_idx = passing[0]
        return results['k_values'][best_idx], ratios

    return (
        evaluate_random_basis_null_gpu,
        evaluate_svd_truncation,
        plot_svd_truncation_results,
    )


@app.cell
def _(
    X,
    Y,
    all_S,
    all_Us,
    all_Vt,
    decom_offset_results,
    evaluate_svd_truncation,
    num_electrodes,
    num_lags,
):
    target_result = decom_offset_results["mid_gpt2"][0]
    svd_results = evaluate_svd_truncation(
        encoding_models=target_result["encoding_models"],
        test_idxs=target_result["test_idxs"],
        X=X,
        Y=Y,
        num_electrodes=num_electrodes,
        num_lags=num_lags,
        k_values= list(range(1, 11)), # [1, 10, 50, 100, 200, 400, 800, 1600],
        # W_mean=significant_coefficients.mean(1),
        sig_mask=target_result["sig"]["sig_bool"],
        scalers=target_result["Y_scalers"],
        svd=(all_Us, all_S, all_Vt),
    )
    return svd_results, target_result


@app.cell
def _(svd_results):
    svd_results
    return


@app.cell
def _(plot_svd_truncation_results, svd_results):
    plot_svd_truncation_results(svd_results)
    return


@app.cell
def _():
    # plot_svd_truncation_results(rand_)
    return


@app.cell
def _(X, all_Vt, df_word, np, significant_coefficients):
    def get_top_words_for_vector(vector, vocab_embeddings, word_list, top_n=15):
        """
        Finds the words that have the highest cosine similarity to a given 1600D vector.
        """
        # Ensure vector is a 1D array
        vector = vector.flatten()

        # Calculate dot products (cosine similarity if both are normalized)
        # If vocab_embeddings isn't normalized, these are just projection scores
        scores = vocab_embeddings @ vector

        # Get indices of top scores
        top_indices = np.argsort(scores)[::-1][:top_n]
        bottom_indices = np.argsort(scores)[:top_n]

        return [word_list[i] for i in top_indices], [word_list[i] for i in bottom_indices]

    # --- 1. Interpret the Mean (The Baseline) ---
    mean_top, mean_bottom = get_top_words_for_vector(significant_coefficients.mean(1)[0], X[0], df_word.word)
    print(f"Mean Vector Top Words: {mean_top}")
    print(f"Mean Vector Bottom Words: {mean_bottom}")
    print("-" * 30)

    # --- 2. Interpret the Variations (SVD Components) ---
    # Vt is the basis from your SVD [k, 1600]
    for i in range(30):
        pc_top, pc_bottom = get_top_words_for_vector(all_Vt[0][i], X[0], df_word.word)
        print(f"PC{i+1} Positive Direction: {pc_top}")
        print(f"PC{i+1} Negative Direction: {pc_bottom}")
        print("-" * 30)
    return


@app.cell
def _(torch):
    torch.cuda.empty_cache()
    return


@app.cell
def _(
    X,
    Y,
    evaluate_random_basis_null_gpu,
    num_electrodes,
    num_lags,
    target_result,
):
    rand_results = evaluate_random_basis_null_gpu(
        encoding_models=target_result["encoding_models"],
        test_idxs=target_result["test_idxs"],
        X=X,
        Y=Y,
        null_chunk_size=2,
        num_electrodes=num_electrodes,
        num_lags=num_lags,
        k_values=[1, 5],
        # W_mean=significant_coefficients.mean(1),
        sig_mask=target_result["sig"]["sig_bool"],
        scalers=target_result["Y_scalers"],
    )
    return (rand_results,)


@app.cell
def _(plot_svd_truncation_results, rand_results):
    rand_results["truncated_corrs"] = rand_results["null_corrs_mean"]
    plot_svd_truncation_results(rand_results)
    return


@app.cell
def _():
    decom_run_names_to_include = ["mid_gpt2"]
    decom_offsets_to_include = [0]
    return


@app.cell
def _(rand_results):
    rand_results["null_corrs_per_iter"][1]
    return


@app.cell
def _(
    LinearRegression,
    StandardScaler,
    X,
    all_Vt,
    epochs_data,
    make_pipeline,
    n_folds,
    num_electrodes,
    num_lags,
    original_folds,
    train_encoding,
):
    # Add in controls
    # Arbitrary models
    # original_folds = list(zip(per_offset_results[0]["train_idxs"], per_offset_results[0]["test_idxs"]))
    # arbitrary_embeddings = get_arbitrary_embeddings(averaged_embeddings, epochs, df_word).astype(np.float32)


    def get_ols_model_constructor():
        def create_model(_):
            return make_pipeline(
                StandardScaler(),
                LinearRegression(fit_intercept=True)
            )
        return create_model


    projected_X_list = [X[f] @ all_Vt[f, :7].T for f in range(n_folds)]

    # project_x(X, all_Vt[:, :3], target_result["test_idxs"])

    # svd_transformed_xs = project_x(projected_X_list, all_Vt[:, :3], target_result["test_idxs"])
    corrs_svd, _, _, _, _ = train_encoding(projected_X_list, epochs_data, get_ols_model_constructor(), epochs_shape=(num_electrodes, num_lags), provided_idxs=original_folds)
    return (corrs_svd,)


@app.cell
def _(
    corrs_svd,
    np,
    plot_embedding_correlations,
    plt,
    rand_results,
    svd_results,
):
    # plot_multiple_offset_results(get_all_offset_results(decom_run_names_to_include, decom_offsets_to_include))
    plot_embedding_correlations(plot_corrs=[{"corrs": svd_results["full_corrs_per_fold"], "color": "white", "label": "Full Embedding", "linestyle": "-"}, {"corrs": svd_results["truncated_corrs_per_fold"][6], "color": "yellow", "label": "7D SVD Decomposition", "linestyle": "-"}, {"corrs": svd_results["truncated_corrs_per_fold"][2], "color": "green", "label": "3D SVD Decomposition", "linestyle": "-"}, {"corrs": corrs_svd, "color": "red", "label": "Project X", "linestyle": "-"}
                                           ], plot_nulls=[{"corrs": np.expand_dims(rand_results["null_corrs_per_iter"][1], 0), "color": "cyan", "label": "10d Random Direction ", "linestyle": "-"}])
    plt.show()
    return


@app.cell(hide_code=True)
def _(
    Parallel,
    delayed,
    dictionary_learning_on_weights,
    mini_batch_dictionary_learning_on_weights,
    np,
    plt,
    tqdm,
):
    def sweep_alpha_dictionary_learning(weights, alphas=None, n_components=None, **kwargs):
        """
        Sweep over alpha values for DictionaryLearning and plot diagnostics.

        Parameters
        ----------
        weights : array-like, shape (n_folds, num_models, embed_dim)
        alphas : array-like of floats, or None
            Alpha values to try. Defaults to a log-spaced range based on data scale.
        n_components : int or None
            Number of dictionary atoms. Defaults to num_models.
        **kwargs
            Additional keyword arguments passed to DictionaryLearning.

        Returns
        -------
        results : dict
            Keys: 'alphas', and per-metric arrays of shape (n_alphas, n_folds).
        fig : matplotlib Figure
        """
        W = np.asarray(weights, dtype=np.float64)
        n_folds = W.shape[0]

        if alphas is None:
            scale = np.mean(np.abs(W))
            alphas = np.logspace(np.log10(scale * 1e-4), np.log10(scale * 10), 20)

        n_alphas = len(alphas)
        recon_errors = np.zeros((n_alphas, n_folds))
        sparsity = np.zeros((n_alphas, n_folds))
        mean_abs_code = np.zeros((n_alphas, n_folds))
        rel_errors = np.zeros((n_alphas, n_folds))

        for i, alpha in enumerate(alphas):
            print(f"Alpha {alpha} of {alphas}")
            components, codes, errors = dictionary_learning_on_weights(
                W, n_components=n_components, alpha=alpha, **kwargs
            )
            for f in range(n_folds):
                recon_errors[i, f] = errors[f]
                rel_errors[i, f] = errors[f] / np.mean(np.sum(W[f] ** 2, axis=1))
                sparsity[i, f] = np.mean(codes[f] == 0)
                mean_abs_code[i, f] = np.mean(np.abs(codes[f]))

        results = {
            "alphas": alphas,
            "recon_errors": recon_errors,
            "rel_errors": rel_errors,
            "sparsity": sparsity,
            "mean_abs_code": mean_abs_code,
        }

        # --- Plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fold_colors = plt.cm.tab10(np.linspace(0, 1, n_folds))

        panels = [
            ("recon_errors", "Reconstruction MSE", "MSE"),
            ("rel_errors", "Relative Reconstruction Error", "Relative Error"),
            ("sparsity", "Code Sparsity (fraction zero)", "Fraction Zero"),
            ("mean_abs_code", "Mean |code|", "Mean Absolute Code"),
        ]

        for ax, (key, title, ylabel) in zip(axes.flat, panels):
            for f in range(n_folds):
                ax.plot(alphas, results[key][:, f], "o-", color=fold_colors[f],
                        label=f"Fold {f}", markersize=4)
            ax.set_xscale("log")
            ax.set_xlabel("alpha")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle("Dictionary Learning Alpha Sweep", fontsize=14, y=1.01)
        fig.tight_layout()

        return results, fig


    def sweep_alpha_mini_batch_dictionary_learning(weights, alphas=None, n_components=None, n_jobs=-1, **kwargs):
        """
        Sweep over alpha values for MiniBatch DictionaryLearning in parallel and plot diagnostics.

        Parameters
        ----------
        weights : array-like, shape (n_folds, num_models, embed_dim)
        alphas : array-like of floats, or None
            Alpha values to try. Defaults to a log-spaced range based on data scale.
        n_components : int or None
            Number of dictionary atoms. Defaults to num_models.
        n_jobs : int
            Number of parallel jobs. -1 uses all cores.
        **kwargs
            Additional keyword arguments passed to MiniBatchDictionaryLearning.

        Returns
        -------
        results : dict
            Keys: 'alphas', and per-metric arrays of shape (n_alphas, n_folds).
        fig : matplotlib Figure
        """
        W = np.asarray(weights, dtype=np.float64)
        n_folds = W.shape[0]

        if alphas is None:
            scale = np.mean(np.abs(W))
            alphas = np.logspace(np.log10(scale * 1e-4), np.log10(scale * 10), 20)

        def _run_one(i, alpha):
            components, codes, errors = mini_batch_dictionary_learning_on_weights(
                W, n_components=n_components, alpha=alpha, **kwargs
            )
            per_fold = {}
            for f in range(n_folds):
                per_fold[f] = {
                    "recon_error": errors[f],
                    "rel_error": errors[f] / np.mean(np.sum(W[f] ** 2, axis=1)),
                    "sparsity": np.mean(codes[f] == 0),
                    "mean_abs_code": np.mean(np.abs(codes[f])),
                }
            return i, per_fold

        all_results_gen = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(_run_one)(i, a) for i, a in enumerate(alphas)
        )

        n_alphas = len(alphas)
        recon_errors = np.zeros((n_alphas, n_folds))
        rel_errors = np.zeros((n_alphas, n_folds))
        sparsity = np.zeros((n_alphas, n_folds))
        mean_abs_code = np.zeros((n_alphas, n_folds))

        for i, per_fold in tqdm(all_results_gen, total=n_alphas, desc="Alpha sweep"):
            for f in range(n_folds):
                recon_errors[i, f] = per_fold[f]["recon_error"]
                rel_errors[i, f] = per_fold[f]["rel_error"]
                sparsity[i, f] = per_fold[f]["sparsity"]
                mean_abs_code[i, f] = per_fold[f]["mean_abs_code"]

        results = {
            "alphas": alphas,
            "recon_errors": recon_errors,
            "rel_errors": rel_errors,
            "sparsity": sparsity,
            "mean_abs_code": mean_abs_code,
        }

        # --- Plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fold_colors = plt.cm.tab10(np.linspace(0, 1, n_folds))

        panels = [
            ("recon_errors", "Reconstruction MSE", "MSE"),
            ("rel_errors", "Relative Reconstruction Error", "Relative Error"),
            ("sparsity", "Code Sparsity (fraction zero)", "Fraction Zero"),
            ("mean_abs_code", "Mean |code|", "Mean Absolute Code"),
        ]

        for ax, (key, title, ylabel) in zip(axes.flat, panels):
            for f in range(n_folds):
                ax.plot(alphas, results[key][:, f], "o-", color=fold_colors[f],
                        label=f"Fold {f}", markersize=4)
            ax.set_xscale("log")
            ax.set_xlabel("alpha")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle("Dictionary Learning Alpha Sweep", fontsize=14, y=1.01)
        fig.tight_layout()

        return results, fig

    return (sweep_alpha_mini_batch_dictionary_learning,)


@app.cell
def _(all_Vt, np, significant_coefficients):
    transformed_coefficients = []
    for i_coef, sig_coef in enumerate(significant_coefficients):
        transformed_coefficients.append(sig_coef @ all_Vt[i_coef][:400].T)
    transformed_coefficients = np.stack(transformed_coefficients)
    transformed_coefficients
    return (transformed_coefficients,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run Alpha Sweep")
    run_button
    return (run_button,)


@app.cell
def _(
    mo,
    np,
    plt,
    run_button,
    sweep_alpha_mini_batch_dictionary_learning,
    transformed_coefficients,
):
    mo.stop(not run_button.value, mo.md("Click the button to run."))

    alpha_results, fig = sweep_alpha_mini_batch_dictionary_learning(transformed_coefficients, 
                                                                    alphas=np.logspace(-5, 0, 8),  # fewer alpha values
        n_components=800,
        max_iter=100,          # cap iterations
        # batch_size=512,        # larger batches = fewer updates per epoch
        random_state=42,
        n_jobs=-1,              # explicit rather than -1
    )
    plt.show()
    return


@app.cell
def _(dictionary_learning_on_weights, significant_coefficients):
    all_components, all_codes, all_errors = dictionary_learning_on_weights(significant_coefficients, n_components=50, alpha=1.0)
    return (all_components,)


@app.cell
def _(all_components):
    all_components
    return


@app.cell
def _(np, weights):
    np.mean(np.abs(weights))
    return


if __name__ == "__main__":
    app.run()
