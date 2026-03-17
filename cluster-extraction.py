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
        Counter,
        DictionaryLearning,
        FastICA,
        KFold,
        Optional,
        PLSRegression,
        Path,
        RidgeCV,
        StandardScaler,
        alt,
        correlation_score,
        cosine_similarity,
        dendrogram,
        entropy,
        fcluster,
        get_backend,
        h5py,
        inconsistent,
        joblib,
        linkage,
        make_pipeline,
        mne,
        mo,
        multipletests,
        np,
        os,
        pd,
        plt,
        set_backend,
        squareform,
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


@app.cell
def _():
    isinstance([2, 3], list)
    return


@app.cell(hide_code=True)
def _(KFold, StandardScaler, correlation_score, get_backend, np):
    def train_encoding(X, Y, model_constructor_fn, modify_inputs_fn=None, modify_targets_fn=None, epochs_shape=None, fix_model=False, provided_idxs=None, provided_scalers=[], num_folds=2):
        num_folds = len(provided_idxs) if provided_idxs is not None else num_folds
        print(num_folds)
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

        return raw_coefs.numpy(), raw_intercept

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


    def get_all_predictions_and_targets_test_set(encoding_models, test_idxs, X, Y, num_electrodes, num_timepoints):
        all_predictions = []
        all_targets = []
        for i, trained_model in enumerate(encoding_models):
            all_predictions.append(trained_model.predict(X[i][test_idxs[i]]))
            all_targets.append(Y[test_idxs[i]])

        all_predictions = torch.vstack(all_predictions).numpy().reshape((X[0].shape[0], num_electrodes, num_timepoints))
        all_targets = np.vstack(all_targets).reshape((X[0].shape[0], num_electrodes, num_timepoints))
        return all_predictions, all_targets

    return (
        absorb_scaler,
        get_all_predictions_and_targets_test_set,
        get_models_coefficients,
    )


@app.cell
def _(mo):
    mo.md("""
    ### Plotting Helpers
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    cosine_similarity,
    dendrogram,
    fcluster,
    inconsistent,
    lags,
    np,
    num_bins_slider,
    pd,
    plt,
    stats,
):
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

    return (
        plot_activations,
        plot_class_distributions,
        plot_cluster_diagnostics,
        plot_correlations,
        plot_cosine_similarity_heatmap,
        plot_embedding_correlations,
    )


@app.cell
def _(mo):
    mo.md("""
    ### Significance Helpers
    """)
    return


@app.cell(hide_code=True)
def _(Path, get_partial_cluster_filepath, multipletests, np, os, torch, tqdm):
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
        return os.path.join(get_partial_cluster_filepath(partial_clusters), sig_dir, sig_filename)

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

    return get_correlation_gpu, get_partial_cluster_filepath


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

    return get_method_mapping, run_comparative_analysis_on_folds


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

    return get_example_display_controls, get_mo_display_activations


@app.cell
def _(mo):
    mo.md("""
    ### Cluster Helpers
    """)
    return


@app.cell(hide_code=True)
def _(datas, get_models_coefficients, np):
    def get_aligned_cluster_mean_with_signs(cluster_vectors):
        if len(cluster_vectors) <= 1:
            return cluster_vectors[0] if len(cluster_vectors) == 1 else None, np.ones(len(cluster_vectors))

        corrs = np.corrcoef(cluster_vectors)
        ref_idx = np.argmax(np.abs(corrs).sum(axis=1))
        reference = cluster_vectors[ref_idx]

        signs = np.array([
            1.0 if np.dot(v, reference) >= 0 else -1.0
            for v in cluster_vectors
        ])
        aligned = cluster_vectors * signs[:, np.newaxis]
        return aligned.mean(axis=0), signs


    def get_cluster_alignment_signs(cluster_vectors):
        """Returns an array of +1/-1 indicating how to align each vector."""
        if len(cluster_vectors) <= 1:
            return np.ones(len(cluster_vectors))

        corrs = np.corrcoef(cluster_vectors)
        ref_idx = np.argmax(np.abs(corrs).sum(axis=1))
        reference = cluster_vectors[ref_idx]

        signs = np.array([
            1.0 if np.dot(v, reference) >= 0 else -1.0
            for v in cluster_vectors
        ])
        return signs

    def get_significant_clusters(cluster_labels, sig):
        electrode_hier_labels = cluster_labels.reshape(datas.shape[1:])
        significant_labels = - np.ones_like(electrode_hier_labels)
        significant_labels[sig] = electrode_hier_labels[sig]
        # # Filter down to just the classes that were found in significant lags
        unique_labels = {x: i for i, x in enumerate(np.unique(significant_labels))}
        assert unique_labels[-1] == 0
        significant_labels = np.array([unique_labels[x] - 1  for elec in significant_labels for x in elec]).reshape(electrode_hier_labels.shape)
        return significant_labels


    # def get_cluster_mean_activations(cluster_labels, all_predictions):
    #     all_class_mean_activations = []
    #     for cluster_id in np.unique(cluster_labels):
    #         if cluster_id == -1:
    #             continue
    #         all_class_mean_activations.append(all_predictions[:, cluster_labels == cluster_id].mean(1))
    #     return np.column_stack(all_class_mean_activations).transpose(1, 0)


    # def get_model_cluster_mean_direction(model_coefs, cluster_id, significant_labels, bias=None):
    #     cluster_coefs = model_coefs[significant_labels == cluster_id]
    #     if bias is not None:
    #         cluster_bias = bias[(significant_labels == cluster_id).flatten()]
    #         cluster_coefs = np.column_stack([cluster_coefs, cluster_bias])
    #     return cluster_coefs.mean(0)


    def get_model_cluster_mean_direction_with_signs(model_coefs, cluster_id, significant_labels, bias=None):
        mask = (significant_labels == cluster_id)
        cluster_coefs = model_coefs[mask]

        if bias is not None:
            cluster_bias = bias[mask.flatten()]
            cluster_coefs = np.column_stack([cluster_coefs, cluster_bias])

        mean_dir, signs = get_aligned_cluster_mean_with_signs(cluster_coefs)
        return mean_dir, signs


    def get_encoding_model_cluster_mean_direction(encoding_models, cluster_ids, num_electrodes, num_lags, significant_labels, bias=False, scaled_model=False):
        cluster_means = []
        cluster_signs = []

        coefficients, intercepts = get_models_coefficients(encoding_models, num_electrodes, num_lags, scaled_model=scaled_model)

        for cluster_id in cluster_ids:
            curr_cluster_means = []
            curr_cluster_signs = []

            for i in range(len(encoding_models)):
                mean_dir, signs = get_model_cluster_mean_direction_with_signs(
                    coefficients[i], cluster_id, significant_labels,
                    bias=intercepts[i] if bias else None
                )
                curr_cluster_means.append(mean_dir)
                curr_cluster_signs.append(signs)

            cluster_means.append(curr_cluster_means)
            cluster_signs.append(curr_cluster_signs)

        return np.array(cluster_means), cluster_signs

    def get_cluster_mean_activations(cluster_labels, all_predictions, alignment_signs_per_cluster, test_idxs):
        """
        cluster_labels: (n_models_in_cluster,) array of cluster IDs
        all_predictions: (n_samples, n_models_in_cluster) array
        alignment_signs_per_cluster: dict mapping cluster_id -> list of sign arrays (one per fold)
        test_idxs: list of index arrays (one per fold), indicating which samples belong to each fold
        """
        all_class_mean_activations = []
        unique_ids = np.unique(cluster_labels)

        for cluster_id in unique_ids:
            if cluster_id == -1:
                continue

            cluster_acts = all_predictions[:, cluster_labels == cluster_id]
            aligned_acts = np.empty_like(cluster_acts)

            for fold_idx, fold_sample_idxs in enumerate(test_idxs):
                signs = alignment_signs_per_cluster[cluster_id - 1][fold_idx]
                aligned_acts[fold_sample_idxs] = cluster_acts[fold_sample_idxs] * signs[np.newaxis, :]

            all_class_mean_activations.append(aligned_acts.mean(axis=1))

        return np.column_stack(all_class_mean_activations).transpose(1, 0)

    # def get_encoding_model_cluster_mean_direction(encoding_models, cluster_id, num_electrodes, num_lags, significant_labels, bias=False, scaled_model=False):
    #     cluster_means = []

    #     coefficients, intercepts = get_models_coefficients(encoding_models, num_electrodes, num_lags, scaled_model=scaled_model)
    #     for i in range(len(encoding_models)):
    #         mean_dir = get_model_cluster_mean_direction(coefficients[i], cluster_id, significant_labels, bias=intercepts[i] if bias else None)
    #         cluster_means.append(mean_dir)

    #     return cluster_means
    return (
        get_cluster_mean_activations,
        get_encoding_model_cluster_mean_direction,
    )


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

    return load_partial_results, partial_out_direction, save_partial_results


@app.cell
def _(mo):
    mo.md("""
    ### Client Interface
    """)
    return


@app.cell(hide_code=True)
def _(
    get_all_predictions_and_targets_test_set,
    get_correlation_gpu,
    get_sig_results,
    linkage,
    np,
    squareform,
    train_encoding,
):
    def encode_and_cluster_encoders(X, Y, num_electrodes, num_lags, model_constructor, partial_clusters=[], folds=None, epochs_shape=None):
        # TODO: Should I reuse scalers?
        corrs_embedding, encoding_models, encoding_tr_idxs, encoding_test_idxs, encoding_scalers = train_encoding(X, Y, model_constructor, provided_idxs=folds, epochs_shape=epochs_shape)
        all_predictions, all_targets = get_all_predictions_and_targets_test_set(encoding_models, encoding_test_idxs, X, Y, num_electrodes, num_lags)
        sig, p_values, q_values = get_sig_results(partial_clusters, all_predictions, all_targets)

        # Limit it to only the significant results.
        all_predictions = all_predictions[:, sig]
        all_activations = all_predictions.reshape((all_predictions.shape[0], -1))
        all_predictions_corrs = get_correlation_gpu(all_activations, all_activations)

        dist_matrix = np.arccos(np.abs(all_predictions_corrs))

        # squareform converts square matrix to condensed upper-triangle form
        condensed = squareform(dist_matrix, checks=False)

        Z_corr = linkage(condensed, method='ward')

        return {
            "Z_cluster": Z_corr,
            "encoding_corrs": corrs_embedding,
            "encoding_models": encoding_models,
            "model_predictions": all_predictions,
            "prediction_corrs": all_predictions_corrs,
            "Y_scalers": encoding_scalers,
            "sig": {
                "sig_bool": sig,
                "p_values": p_values,
                "q_values": q_values,
            },
            "train_idxs": encoding_tr_idxs,
            "test_idxs": encoding_test_idxs,
        }

    return (encode_and_cluster_encoders,)


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


@app.cell
def _():
    # Determine which path of partialing to base the current set of directions on
    curr_partial_clusters = []
    return (curr_partial_clusters,)


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
    raws = load_raws([i for i in sig_elecs.keys()], "data/", per_subject_electrodes=sig_elecs)

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
    ## Encoding and Clustering
    """)
    return


@app.cell(hide_code=True)
def _(
    KFold,
    RidgeCV,
    StandardScaler,
    averaged_embeddings,
    curr_partial_clusters,
    datas,
    encode_and_cluster_encoders,
    epochs_data,
    get_backend,
    load_partial_results,
    make_pipeline,
    np,
):
    if not curr_partial_clusters:
        X = averaged_embeddings
        Y = epochs_data
        # Make them into a list
        X = [X, X]
    else:
        X, Y, _, _ = load_partial_results(curr_partial_clusters)

    if "torch" in get_backend().__name__:
        X = [x.astype(np.float32) for x in X]
        Y = Y.astype(np.float32)

    def get_ridge_model_constructor():
        alphas = np.logspace(1, 10, 10) # specify alpha values
        inner_cv = KFold(n_splits=5, shuffle=False) # inner 5-fold cross-validation setup
        def create_model(_):
            return make_pipeline(
                StandardScaler(), RidgeCV(alphas, fit_intercept=True, cv=inner_cv) # pipeline
            )
        return create_model

    embed_dim, num_electrodes, num_lags = X[0].shape[1], datas.shape[1], datas.shape[2]

    encode_cluster_results = encode_and_cluster_encoders(X, Y, num_electrodes, num_lags, get_ridge_model_constructor(), partial_clusters=curr_partial_clusters, folds=None, epochs_shape=(num_electrodes, num_lags))

    original_folds = list(zip(encode_cluster_results["train_idxs"], encode_cluster_results["test_idxs"]))
    encode_cluster_results
    return (
        X,
        Y,
        encode_cluster_results,
        get_ridge_model_constructor,
        num_electrodes,
        num_lags,
        original_folds,
    )


@app.cell(hide_code=True)
def _(
    X,
    Y,
    df_word,
    epochs,
    get_arbitrary_embeddings,
    get_ridge_model_constructor,
    num_electrodes,
    num_lags,
    original_folds,
    train_encoding,
):
    # Add in controls
    # Arbitrary models
    corrs_arb, _, _, _, _ = train_encoding([get_arbitrary_embeddings(X[0], epochs, df_word), get_arbitrary_embeddings(X[1], epochs, df_word)], Y, get_ridge_model_constructor(), epochs_shape=(num_electrodes, num_lags), provided_idxs=original_folds)
    return (corrs_arb,)


@app.cell(hide_code=True)
def _(corrs_arb, encode_cluster_results, plot_embedding_correlations):
    plot_embedding_correlations(plot_corrs=[
        {"corrs": encode_cluster_results["encoding_corrs"], "color": "black", "label": "Middle Layer"},
        {"corrs": corrs_arb, "color": "red", "label": "Arbitrary"},
    ])
    return


@app.cell
def _(encode_cluster_results, plot_cluster_diagnostics):
    plot_cluster_diagnostics(encode_cluster_results["Z_cluster"])
    return


@app.cell(hide_code=True)
def _(encode_cluster_results, fcluster, np, num_electrodes, num_lags):
    cluster_labels = fcluster(encode_cluster_results["Z_cluster"], t=3, criterion='maxclust')
    for cl in np.unique(cluster_labels):
        mask = cluster_labels == cl
        within_corrs = encode_cluster_results["prediction_corrs"][np.ix_(mask, mask)]
        triu = within_corrs[np.triu_indices(mask.sum(), k=1)]
        print(f"Cluster {cl} (n={mask.sum()}): "
              f"mean |corr|={np.mean(np.abs(triu)):.3f}, "
              f"median |corr|={np.median(np.abs(triu)):.3f}")

    expanded_clusters = - np.ones((num_electrodes, num_lags), dtype=int)
    expanded_clusters[encode_cluster_results["sig"]["sig_bool"]] = cluster_labels


    # significant_labels = get_significant_clusters(cluster_labels, encode_cluster_results["sig"]["sig_bool"])
    return cluster_labels, expanded_clusters


@app.cell(hide_code=True)
def _(Counter, cluster_labels, mo):
    class_counts = Counter(cluster_labels[cluster_labels != -1].flatten())
    sorted_class_counts = sorted(class_counts.items(), key=lambda item: -item[1])

    class_dropdown = mo.ui.dropdown(
        options={f"{class_id}:{class_count}": class_id for class_id, class_count in sorted_class_counts},
        value=":".join([str(x) for x in list(sorted_class_counts)[0]]),
        label="Class",
    )

    class_dropdown
    return (class_dropdown,)


@app.cell(hide_code=True)
def _(
    class_dropdown,
    cluster_labels,
    encode_cluster_results,
    expanded_clusters,
    get_cluster_mean_activations,
    get_encoding_model_cluster_mean_direction,
    np,
    num_electrodes,
    num_lags,
):
    all_mean_cluster_dirs, cluster_signs = get_encoding_model_cluster_mean_direction(encode_cluster_results["encoding_models"], np.unique(cluster_labels), num_electrodes, num_lags, expanded_clusters, bias=True, scaled_model=False)
    mean_cluster_dirs = all_mean_cluster_dirs[class_dropdown.value - 1]

    all_cluster_activations = get_cluster_mean_activations(cluster_labels, encode_cluster_results["model_predictions"], cluster_signs, encode_cluster_results["test_idxs"])
    curr_cluster_activations = all_cluster_activations[class_dropdown.value - 1]
    return (
        all_cluster_activations,
        all_mean_cluster_dirs,
        cluster_signs,
        curr_cluster_activations,
        mean_cluster_dirs,
    )


@app.cell(hide_code=True)
def _(
    class_dropdown,
    cluster_signs,
    expanded_clusters,
    plot_class_distributions,
    raws,
):
    subject_electrode_splits = [len(r.ch_names) for r in raws]
    subject_electrode_splits.insert(1, 0)

    plot_class_distributions(expanded_clusters, class_dropdown.value, signs_per_fold=cluster_signs[class_dropdown.value - 1], elecs_per_subject=subject_electrode_splits)
    return


@app.cell
def _(get_example_display_controls, mo):
    num_bins_slider, samples_per_bin_slider, context_slider, seed_input = get_example_display_controls()
    mo.hstack([num_bins_slider, samples_per_bin_slider, context_slider, seed_input])
    return context_slider, num_bins_slider, samples_per_bin_slider, seed_input


@app.cell
def _(curr_cluster_activations, mo, plot_activations):
    mo.ui.altair_chart(plot_activations(curr_cluster_activations))
    return


@app.cell
def _(
    context_slider,
    curr_cluster_activations,
    get_mo_display_activations,
    num_bins_slider,
    samples_per_bin_slider,
    seed_input,
):
    get_mo_display_activations(curr_cluster_activations, num_bins_slider, samples_per_bin_slider, context_slider, seed_input)
    return


@app.cell(hide_code=True)
def _(
    X,
    curr_cluster_activations,
    encode_cluster_results,
    mean_cluster_dirs,
    np,
    plot_correlations,
):
    X_test_1 = X[0][encode_cluster_results["test_idxs"][0]]
    X_test_2 = X[1][encode_cluster_results["test_idxs"][1]]
    X_bias_1 = np.column_stack([X_test_1, np.ones(X_test_1.shape[0])])
    X_bias_2 = np.column_stack([X_test_2, np.ones(X_test_2.shape[0])])

    first_set_raw_means = X_bias_1 @ mean_cluster_dirs[0]
    second_set_raw_means = X_bias_2 @ mean_cluster_dirs[1]
    mean_raw_dir_cluster_preds = np.concat([first_set_raw_means, second_set_raw_means])
    plot_correlations(mean_raw_dir_cluster_preds, curr_cluster_activations, x_name="Mean Cluster Pred", y_name="Mean Activation")
    return X_bias_1, X_bias_2


@app.cell(hide_code=True)
def _(
    X_bias_1,
    X_bias_2,
    curr_cluster_activations,
    mean_cluster_dirs,
    np,
    plot_correlations,
):
    X_bias = np.concat([X_bias_1, X_bias_2])
    curr_mean_model_dir = (mean_cluster_dirs[0] + mean_cluster_dirs[1]) / 2
    mean_coefs_cluster_preds = X_bias @ curr_mean_model_dir
    plot_correlations(mean_coefs_cluster_preds, curr_cluster_activations)
    return


@app.cell
def _(X, all_mean_cluster_dirs, np, plot_cosine_similarity_heatmap):
    plot_cosine_similarity_heatmap(np.array(all_mean_cluster_dirs).mean(1)[:, :X[0].shape[1]])
    return


@app.cell(hide_code=True)
def _(cluster_labels, mo, np):
    partial_clusters_multiselect = mo.ui.multiselect(options={str(x): int(x - 1) for x in np.unique(cluster_labels)})

    run_button = mo.ui.run_button(label="Run Partialing")
    mo.hstack([partial_clusters_multiselect, run_button])
    return partial_clusters_multiselect, run_button


@app.cell(hide_code=True)
def _(
    X,
    Y,
    all_cluster_activations,
    all_mean_cluster_dirs,
    corrs_arb,
    encode_cluster_results,
    get_ridge_model_constructor,
    mo,
    num_electrodes,
    num_lags,
    original_folds,
    partial_clusters_multiselect,
    partial_out_direction,
    plot_embedding_correlations,
    run_button,
    train_encoding,
):
    mo.stop(not run_button.value, mo.md("Click the button to run."))

    print(f"Partialing out {partial_clusters_multiselect.value}")

    # Drop bias if included
    partial_dirs = all_mean_cluster_dirs[partial_clusters_multiselect.value]
    partial_dirs = partial_dirs[:, :, :X[0].shape[1]]

    X_residuals, Y_residuals = partial_out_direction([X, X], Y, partial_dirs, all_cluster_activations[partial_clusters_multiselect.value], encode_cluster_results["test_idxs"], encode_cluster_results["Y_scalers"])

    corrs_partialed_dir, partialed_encoding_models, _, _, _ = train_encoding(X_residuals, Y, get_ridge_model_constructor(), epochs_shape=(num_electrodes, num_lags), provided_idxs=original_folds)
    print(f"Encoding performance correlating matrix shape: {corrs_partialed_dir.shape}")

    plot_embedding_correlations(plot_corrs=[
        {"corrs": encode_cluster_results["encoding_corrs"], "color": "black", "label": "Middle Layer"},
        {"corrs": corrs_partialed_dir, "label": "Partialed Out", "color": "red"},
        {"corrs": corrs_arb, "color": "teal", "label": "Arbitrary"},
    ])
    return X_residuals, Y_residuals


@app.cell(hide_code=True)
def _(
    X_residuals,
    Y_residuals,
    df_word,
    epochs,
    get_arbitrary_embeddings,
    get_ridge_model_constructor,
    num_electrodes,
    num_lags,
    original_folds,
    plot_embedding_correlations,
    train_encoding,
):
    corrs_partial_encoding, _, _, _, _ = train_encoding(X_residuals, Y_residuals, get_ridge_model_constructor(), epochs_shape=(num_electrodes, num_lags), provided_idxs=original_folds)


    X_arb_residuals = [get_arbitrary_embeddings(X_curr, epochs, df_word) for X_curr in X_residuals]
    corrs_partial_arb, _, _, _, _ = train_encoding(X_arb_residuals, Y_residuals, get_ridge_model_constructor(), epochs_shape=(num_electrodes, num_lags), provided_idxs=original_folds)

    plot_embedding_correlations(plot_corrs=[
        {"corrs": corrs_partial_encoding, "label": "Partialed Encoding", "color": "black"},
        {"corrs": corrs_partial_arb, "color": "teal", "label": "Partialed Arbitrary"},
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    save_button = mo.ui.run_button(label="Save Results")
    save_button
    return (save_button,)


@app.cell(hide_code=True)
def _(
    X_residuals,
    Y_residuals,
    all_mean_cluster_dirs,
    curr_partial_clusters,
    encode_cluster_results,
    mo,
    partial_clusters_multiselect,
    save_button,
    save_partial_results,
):
    mo.stop(not save_button.value, mo.md("Click the button to save results."))

    partialed_dirs = partial_clusters_multiselect.value

    save_partial_results(curr_partial_clusters + [all_mean_cluster_dirs.shape[0], partialed_dirs], X_residuals, Y_residuals, all_mean_cluster_dirs, partialed_dirs, X_scalers=encode_cluster_results, filename="cluster_results.npz")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Baselines
    """)
    return


@app.cell
def _(mo):
    compare_results_button = mo.ui.run_button(label="Run Comparative Analysis")
    compare_results_button
    return (compare_results_button,)


@app.cell
def _(
    X,
    Y,
    compare_results_button,
    encode_cluster_results,
    mo,
    num_electrodes,
    num_lags,
    run_comparative_analysis_on_folds,
):
    mo.stop(not compare_results_button.value, mo.md("Click the button to run."))

    comparative_results = run_comparative_analysis_on_folds(encode_cluster_results["encoding_models"], # List of trained models (one per fold)
        encode_cluster_results["sig"]["sig_bool"],                                                      
        X,               # [num_words, embed_dim]
        Y.reshape(Y.shape[0], num_electrodes, num_lags),  # [num_words, num_elecs, num_timepoints]
        encode_cluster_results["train_idxs"],
        num_electrodes, 
        num_lags,
        n_components=50)
    return (comparative_results,)


@app.cell
def _(comparative_results):
    comparative_results
    return


@app.cell
def _(comparative_results, get_method_mapping):
    get_method_mapping(comparative_results["dict_learning"])
    return


app._unparsable_cell(
    r"""
    plot_atom_distributions(
        atom_maps: np.ndarray,      # [num_atoms, n_elecs, n_lags]
        target_atom: int,           # Which atom index to visualize
        elecs_per_subject: list[int] | None = None,
    )
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
