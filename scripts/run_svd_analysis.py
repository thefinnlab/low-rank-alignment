"""
run_svd_analysis.py — SVD decomposition of encoding model weights.

Identifies dominant directions in weight space, evaluates how many SVD components
are sufficient to reconstruct full test-set performance, and generates summary
visualizations.

Writes to {output_dir}/svd/:
  svd_decomposition.npz         — SVD factors (U, S, Vt, alignments) per fold
  truncation_results.npz        — correlations at each k + best_k
  truncation_plot.png           — 4-panel summary figure
  subjects/sub_{id}/
    corrs_k{best_k}.npy
    corrs_plot_k{best_k}.png
    sig_corrs_plot_k{best_k}.png
  cross_subject_corrs_plot_k{best_k}.png
  cross_subject_sig_corrs_plot_k{best_k}.png
"""

import gc
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    load_raws,
    load_embeddings,
    build_word_epochs,
    apply_offset,
    load_all_subject_coefficients,
    load_all_subject_correlations,
    find_dominant_directions,
    compute_truncated_correlations,
    find_best_k,
    add_corrs,
    plot_corrs_by_significance,
    plot_svd_truncation_results,
)
from utils.config import SVDConfig, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import load_encoding_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_global_sig_mask(sig_df, metadata_df, n_lags):
    """Build (n_total_elecs, n_lags) bool mask aligned to metadata_df row order.

    Args:
        sig_df: DataFrame with columns [subject, electrode, lag_idx, sig].
        metadata_df: DataFrame with columns [subject, electrode].
        n_lags: number of lag positions.

    Returns:
        np.ndarray of shape (n_total_elecs, n_lags), dtype bool.
    """
    n_elecs = len(metadata_df)
    mask = np.zeros((n_elecs, n_lags), dtype=bool)

    for row_i, row in metadata_df.iterrows():
        matches = sig_df[
            (sig_df["subject"] == row["subject"]) &
            (sig_df["electrode"] == row["electrode"])
        ]
        for _, sig_row in matches.iterrows():
            lag_i = int(sig_row["lag_idx"])
            assert 0 <= lag_i < n_lags, (
                f"lag_idx {lag_i} from significance_detailed.csv is out of range "
                f"[0, {n_lags}) for electrode '{row['electrode']}' "
                f"(subject {row['subject']}). The CSV may have been produced with "
                "different lag settings."
            )
            mask[row_i, lag_i] = bool(sig_row["sig"])

    return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args, overrides = parse_args("SVD decomposition of encoding model weights")
    config: SVDConfig = load_config(args.config, overrides, SVDConfig)
    enc = config.encoding

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)
    svd_dir = config.svd_output_dir or os.path.join(output_dir, "svd")

    os.makedirs(svd_dir, exist_ok=True)
    print(f"Output dir:  {output_dir}")
    print(f"SVD dir:     {svd_dir}")

    # ------------------------------------------------------------------
    # C. Load significance CSV + build global sig_mask
    # ------------------------------------------------------------------
    sig_csv = os.path.join(output_dir, "significance_detailed.csv")
    if not os.path.exists(sig_csv):
        raise FileNotFoundError(
            f"significance_detailed.csv not found at {sig_csv}. "
            "Run run_significance.py first."
        )
    sig_df = pd.read_csv(sig_csv)
    print(f"Loaded significance CSV: {len(sig_df):,} rows")

    # ------------------------------------------------------------------
    # D. Load all subject coefficients
    # ------------------------------------------------------------------
    print("Loading all subject coefficients...")
    coefficients_all, intercepts_all, metadata_df = load_all_subject_coefficients(
        output_dir, subjects=subject_ids
    )
    # coefficients_all: (n_folds, n_total_elecs, n_lags, n_features)
    n_folds, n_total_elecs, n_lags, n_features = coefficients_all.shape
    print(f"  Shape: {coefficients_all.shape}")

    # ------------------------------------------------------------------
    # C (cont). Build global sig_mask now that we know n_lags
    # ------------------------------------------------------------------
    sig_mask_global = _build_global_sig_mask(sig_df, metadata_df, n_lags)
    print(f"  Significant pairs: {sig_mask_global.sum()} / {sig_mask_global.size}")

    # ------------------------------------------------------------------
    # E. Filter to significant weights: W_sig (n_folds, n_sig, n_features)
    # ------------------------------------------------------------------
    sig_flat = sig_mask_global.flatten()  # (n_total_elecs * n_lags,)
    W_sig = np.stack([
        coefficients_all[f].reshape(-1, n_features)[sig_flat]
        for f in range(n_folds)
    ])  # (n_folds, n_sig, n_features)
    print(f"  Significant weight matrix: {W_sig.shape}")

    # ------------------------------------------------------------------
    # F. SVD
    # ------------------------------------------------------------------
    n_directions = config.n_directions
    print(f"Computing SVD (n_directions={n_directions})...")
    all_U, all_S, all_Vt, all_alignments = find_dominant_directions(W_sig, n_directions)

    # Stack Vt: all folds have the same rank after truncation
    Vt_arr = np.stack(all_Vt)  # (n_folds, actual_n_dirs, n_features)
    actual_n_dirs = Vt_arr.shape[1]

    # ------------------------------------------------------------------
    # G. Save SVD decomposition
    # ------------------------------------------------------------------
    svd_path = os.path.join(svd_dir, "svd_decomposition.npz")
    np.savez(
        svd_path,
        **{f"U_fold{f}": all_U[f] for f in range(n_folds)},
        **{f"S_fold{f}": all_S[f] for f in range(n_folds)},
        **{f"Vt_fold{f}": all_Vt[f] for f in range(n_folds)},
        **{f"alignments_fold{f}": all_alignments[f] for f in range(n_folds)},
    )
    print(f"Saved SVD decomposition → {svd_path}")

    # ------------------------------------------------------------------
    # H. Load embeddings once
    # ------------------------------------------------------------------
    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    # Clamp k_values to the number of directions actually computed
    k_values = [k for k in config.k_values if k <= actual_n_dirs]
    print(f"k_values: {k_values}")

    # ------------------------------------------------------------------
    # I. Per-subject loop
    # ------------------------------------------------------------------
    per_subject_results = {}
    per_subject_full_corrs_orig = {}  # original corrs from training, for plotting
    lags = None
    ref_test_idxs = None  # test_idxs from the first subject, used for cross-subject consistency check

    for sub_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Subject {sub_id}")

        # Load encoding results to reuse the exact test_idxs from training,
        # avoiding any risk of evaluating on training data.
        sub_enc_dir = os.path.join(output_dir, "subjects", f"sub_{sub_id}")
        enc_results = load_encoding_results(
            os.path.join(sub_enc_dir, "encoding_results.joblib")
        )
        test_idxs = enc_results["test_idxs"]
        assert len(test_idxs) == n_folds, (
            f"Subject {sub_id} has {len(test_idxs)} folds in encoding_results but "
            f"coefficients_all has {n_folds} folds. Results may be from different runs."
        )
        if ref_test_idxs is None:
            ref_test_idxs = test_idxs
        else:
            for fold_i, (ref_idx, sub_idx) in enumerate(zip(ref_test_idxs, test_idxs)):
                assert set(ref_idx) == set(sub_idx), (
                    f"Subject {sub_id} fold {fold_i} test indices differ from subject "
                    f"{subject_ids[0]}. All subjects must use the same word-level train/test "
                    "split to ensure comparable evaluation."
                )
        if lags is None:
            lags = enc_results["lags"]
        per_subject_full_corrs_orig[sub_id] = enc_results["corrs"]  # (n_folds, n_elecs, n_lags)

        # Load raw neural data
        raws = load_raws([sub_id], enc.data_root, use_high_gamma=True)
        raw = raws[0]
        n_elecs_sub = len(raw.ch_names)

        epochs, data = build_word_epochs(raw, df_word)
        X_arr, Y = apply_offset(aligned_embeddings[epochs.selection], data, enc.offset)
        # Y shape: (n_words, n_elecs_sub * n_lags)

        # Slice subject coefficients
        sub_rows = metadata_df.index[metadata_df["subject"] == sub_id].tolist()
        coefs_sub = coefficients_all[:, sub_rows, :, :]  # (n_folds, n_elecs_sub, n_lags, n_features)
        ints_sub = [
            intercepts_all[f].reshape(n_total_elecs, n_lags)[sub_rows].ravel()
            for f in range(n_folds)
        ]

        sig_mask_sub = sig_mask_global[sub_rows]  # (n_elecs_sub, n_lags)

        print(f"  Evaluating truncation (n_elecs={n_elecs_sub})...")
        sub_results = compute_truncated_correlations(
            coefficients=coefs_sub,
            intercepts=ints_sub,
            test_idxs=test_idxs,
            X=X_arr,
            Y=Y,
            n_elecs=n_elecs_sub,
            n_lags=n_lags,
            Vt=Vt_arr,
            k_values=k_values,
            sig_mask=sig_mask_sub,
        )
        per_subject_results[sub_id] = sub_results

        del raws, raw, epochs, data, X_arr, Y
        gc.collect()

    # ------------------------------------------------------------------
    # J. Merge across subjects
    # ------------------------------------------------------------------
    print("\nMerging results across subjects...")
    full_corrs_per_fold = np.concatenate(
        [per_subject_results[s]["full_corrs_per_fold"] for s in subject_ids], axis=1
    )  # (n_folds, n_total_elecs, n_lags)

    truncated_corrs_all = np.concatenate(
        [per_subject_results[s]["truncated_corrs_per_fold"] for s in subject_ids], axis=2
    )  # (n_k, n_folds, n_total_elecs, n_lags)

    merged_results = {
        "k_values": np.array(k_values),
        "full_corrs": full_corrs_per_fold.mean(axis=0),
        "truncated_corrs": truncated_corrs_all.mean(axis=1),
        "sig_mask": sig_mask_global,
        "full_corrs_per_fold": full_corrs_per_fold,
        "truncated_corrs_per_fold": truncated_corrs_all,
    }

    # ------------------------------------------------------------------
    # K. find_best_k
    # ------------------------------------------------------------------
    best_k = find_best_k(merged_results, threshold=config.threshold)
    print(f"best_k = {best_k}")

    # ------------------------------------------------------------------
    # L. Save truncation results
    # ------------------------------------------------------------------
    k_vals = np.array(k_values)
    trunc_npz_path = os.path.join(svd_dir, "truncation_results.npz")
    np.savez(
        trunc_npz_path,
        k_values=k_vals,
        full_corrs=full_corrs_per_fold.mean(0),
        full_corrs_per_fold=full_corrs_per_fold,
        sig_mask=sig_mask_global,
        best_k=np.array(best_k),
        **{f"trunc_corrs_k{k}": truncated_corrs_all[i].mean(0) for i, k in enumerate(k_vals)},
        **{f"trunc_corrs_per_fold_k{k}": truncated_corrs_all[i] for i, k in enumerate(k_vals)},
    )
    print(f"Saved truncation results → {trunc_npz_path}")

    # ------------------------------------------------------------------
    # M. Plot truncation summary
    # ------------------------------------------------------------------
    plot_svd_truncation_results(
        merged_results,
        threshold=config.threshold,
        save_path=os.path.join(svd_dir, "truncation_plot.png"),
    )
    print(f"Saved truncation plot → {svd_dir}/truncation_plot.png")

    # ------------------------------------------------------------------
    # N. Correlation plots at best_k
    # ------------------------------------------------------------------
    best_ki = int(np.where(k_vals == best_k)[0][0])

    def _save_corrs_plot(corrs_trunc, corrs_full, lags, title, save_path):
        """Plot SVD-truncated vs full-model correlations on the same axes."""
        fig, ax = plt.subplots(figsize=(8, 4))
        add_corrs(corrs_full, lags, label="Full model", color="blue", ax=ax)
        add_corrs(corrs_trunc, lags, label=f"SVD k={best_k}", color="steelblue",
                  linestyle="--", ax=ax)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Correlation")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    # Per-subject
    for sub_id in subject_ids:
        sub_dir = os.path.join(svd_dir, "subjects", f"sub_{sub_id}")
        os.makedirs(sub_dir, exist_ok=True)

        sub_rows = metadata_df.index[metadata_df["subject"] == sub_id].tolist()
        corrs_sub = truncated_corrs_all[best_ki, :, sub_rows, :]  # (n_folds, n_elecs_sub, n_lags)
        np.save(os.path.join(sub_dir, f"corrs_k{best_k}.npy"), corrs_sub)

        if lags is not None:
            full_corrs_sub = per_subject_full_corrs_orig[sub_id]
            _save_corrs_plot(
                corrs_sub, full_corrs_sub, lags,
                title=f"Subject {sub_id}",
                save_path=os.path.join(sub_dir, f"corrs_plot_k{best_k}.png"),
            )

            sig_mask_sub = sig_mask_global[sub_rows]
            plot_corrs_by_significance(
                corrs_sub, sig_mask_sub, lags,
                save_path=os.path.join(sub_dir, f"sig_corrs_plot_k{best_k}.png"),
            )

        print(f"  sub_{sub_id}: saved to {sub_dir}")

    # Cross-subject
    if lags is not None:
        corrs_cross_trunc = truncated_corrs_all[best_ki]  # (n_folds, n_total_elecs, n_lags)
        corrs_cross_full, _ = load_all_subject_correlations(output_dir, subjects=subject_ids)

        _save_corrs_plot(
            corrs_cross_trunc, corrs_cross_full, lags,
            title="Cross-subject",
            save_path=os.path.join(svd_dir, f"cross_subject_corrs_plot_k{best_k}.png"),
        )
        plot_corrs_by_significance(
            corrs_cross_trunc, sig_mask_global, lags,
            save_path=os.path.join(svd_dir, f"cross_subject_sig_corrs_plot_k{best_k}.png"),
        )
        print(f"Saved cross-subject plots → {svd_dir}/cross_subject_*_k{best_k}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
