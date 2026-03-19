"""
run_svd_analysis.py — SVD decomposition of encoding model weights.

Two modes controlled by SVDConfig.per_subject_svd:

  False (default, cross-subject):
    Compute a single SVD across all subjects' significant weights. Evaluate
    truncation per-subject using that global basis, then aggregate.

    Writes to {output_dir}/svd/:
      svd_decomposition.npz, truncation_results.npz, truncation_plot.png
      subjects/sub_{id}/corrs_k{best_k}.npy, corrs_plot_*.png
      cross_subject_corrs_plot_k{best_k}.png
      cross_subject_sig_corrs_plot_k{best_k}.png

  True (per-subject):
    Compute an independent SVD per subject using only that subject's weights.
    Each subject is fully self-contained; there is no cross-subject aggregation.

    Writes to {output_dir}/per_subject_svd/subjects/sub_{id}/:
      svd_decomposition.npz, truncation_results.npz, truncation_plot.png
      corrs_k{best_k}.npy, corrs_plot_k{best_k}.png, sig_corrs_plot_k{best_k}.png
"""

import gc
import os

import matplotlib
matplotlib.use("Agg")
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
    plot_corrs_by_significance,
    plot_svd_truncation_results,
    plot_svd_corrs,
)
from utils.atlas import build_electrode_region_map
from utils.config import SVDConfig, SVDMode, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import load_encoding_results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_global_sig_mask(sig_df, metadata_df, n_lags):
    """Build (n_total_elecs, n_lags) bool mask aligned to metadata_df row order."""
    mask = np.zeros((len(metadata_df), n_lags), dtype=bool)
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


def _load_subject_encoding_info(sub_id, output_dir, n_folds, ref_test_idxs=None):
    """Load encoding results for one subject; validate fold count and test indices.

    Args:
        sub_id: Subject ID.
        output_dir: Root results directory.
        n_folds: Expected number of folds.
        ref_test_idxs: If given, assert per-fold set equality with this reference.

    Returns:
        test_idxs: list of test index arrays, one per fold.
        lags: 1-D lag array.
        full_corrs_orig: (n_folds, n_elecs, n_lags) correlations from training.
    """
    sub_enc_dir = os.path.join(output_dir, "subjects", f"sub_{sub_id}")
    enc_results = load_encoding_results(os.path.join(sub_enc_dir, "encoding_results.joblib"))
    test_idxs = enc_results["test_idxs"]
    assert len(test_idxs) == n_folds, (
        f"Subject {sub_id} has {len(test_idxs)} folds in encoding_results but "
        f"coefficients_all has {n_folds} folds. Results may be from different runs."
    )
    if ref_test_idxs is not None:
        for fold_i, (ref_idx, sub_idx) in enumerate(zip(ref_test_idxs, test_idxs)):
            assert set(ref_idx) == set(sub_idx), (
                f"Subject {sub_id} fold {fold_i} test indices differ from reference subject. "
                "All subjects must use the same word-level train/test split."
            )
    return test_idxs, enc_results["lags"], enc_results["corrs"]


def _filter_sig_and_svd(coefficients, sig_mask, n_directions):
    """Filter coefficients to significant (electrode, lag) pairs and compute SVD.

    Args:
        coefficients: (n_folds, n_elecs, n_lags, n_features).
        sig_mask: (n_elecs, n_lags) bool.
        n_directions: int or None — passed to find_dominant_directions.

    Returns:
        Vt_arr: (n_folds, actual_n_dirs, n_features) stacked right singular vectors.
        all_U, all_S, all_Vt, all_alignments: per-fold SVD outputs from find_dominant_directions.
    """
    n_folds, _, _, n_features = coefficients.shape
    sig_flat = sig_mask.flatten()
    W_sig = np.stack([
        coefficients[f].reshape(-1, n_features)[sig_flat]
        for f in range(n_folds)
    ])
    print(f"  Significant weight matrix: {W_sig.shape}")
    all_U, all_S, all_Vt, all_alignments = find_dominant_directions(W_sig, n_directions)
    Vt_arr = np.stack(all_Vt)  # (n_folds, actual_n_dirs, n_features)
    return Vt_arr, all_U, all_S, all_Vt, all_alignments


def _save_svd_decomposition(path, all_U, all_S, all_Vt, all_alignments):
    """Save per-fold SVD factors to a .npz file."""
    n_folds = len(all_U)
    np.savez(
        path,
        **{f"U_fold{f}": all_U[f] for f in range(n_folds)},
        **{f"S_fold{f}": all_S[f] for f in range(n_folds)},
        **{f"Vt_fold{f}": all_Vt[f] for f in range(n_folds)},
        **{f"alignments_fold{f}": all_alignments[f] for f in range(n_folds)},
    )


def _save_truncation_results(path, results, best_k):
    """Save truncation evaluation results to a .npz file."""
    k_vals = results["k_values"]
    np.savez(
        path,
        k_values=k_vals,
        full_corrs=results["full_corrs"],
        full_corrs_per_fold=results["full_corrs_per_fold"],
        sig_mask=results["sig_mask"],
        best_k=np.array(best_k),
        **{f"trunc_corrs_k{k}": results["truncated_corrs_per_fold"][i].mean(0)
           for i, k in enumerate(k_vals)},
        **{f"trunc_corrs_per_fold_k{k}": results["truncated_corrs_per_fold"][i]
           for i, k in enumerate(k_vals)},
    )


def _plot_subject_results(sub_id, subject_dir, corrs_trunc, full_corrs_orig,
                          sig_mask_sub, lags, best_k):
    """Save per-subject corrs.npy and generate correlation plots."""
    np.save(os.path.join(subject_dir, f"corrs_k{best_k}.npy"), corrs_trunc)
    plot_svd_corrs(
        corrs_trunc, full_corrs_orig, lags,
        title=f"Subject {sub_id}",
        save_path=os.path.join(subject_dir, f"corrs_plot_k{best_k}.png"),
        best_k=best_k,
    )
    plot_corrs_by_significance(
        corrs_trunc, sig_mask_sub, lags,
        save_path=os.path.join(subject_dir, f"sig_corrs_plot_k{best_k}.png"),
    )


def _evaluate_subject_truncation(
    sub_id, metadata_df, coefficients_all, intercepts_all, sig_mask_global,
    n_total_elecs, n_lags, n_folds, enc, aligned_embeddings, df_word,
    test_idxs, Vt, k_values, electrode_subset=None,
):
    """Slice per-subject arrays, load neural data, and compute truncated correlations.

    Returns:
        results: dict from compute_truncated_correlations.
        n_elecs: number of electrodes loaded for this subject.
        sig_mask_sub: (n_elecs, n_lags) bool mask for this subject.
    """
    sub_rows = metadata_df.index[metadata_df["subject"] == sub_id].tolist()
    if electrode_subset is not None:
        sub_rows = [r for r in sub_rows if metadata_df.loc[r, "electrode"] in set(electrode_subset)]
    sub_elec_names = metadata_df.loc[sub_rows, "electrode"].tolist()
    coefs_sub = coefficients_all[:, sub_rows, :, :]
    ints_sub = [
        intercepts_all[f].reshape(n_total_elecs, n_lags)[sub_rows].ravel()
        for f in range(n_folds)
    ]
    sig_mask_sub = sig_mask_global[sub_rows]

    raws = load_raws([sub_id], enc.data_root, use_high_gamma=True,
                     per_subject_electrodes={sub_id: sub_elec_names})
    raw = raws[0]
    n_elecs = len(raw.ch_names)
    epochs, data = build_word_epochs(raw, df_word)
    X_arr, Y = apply_offset(aligned_embeddings[epochs.selection], data, enc.offset)

    results = compute_truncated_correlations(
        coefficients=coefs_sub, intercepts=ints_sub, test_idxs=test_idxs,
        X=X_arr, Y=Y, n_elecs=n_elecs, n_lags=n_lags,
        Vt=Vt, k_values=k_values, sig_mask=sig_mask_sub,
    )
    del raws, raw, epochs, data, X_arr, Y
    gc.collect()

    return results, n_elecs, sig_mask_sub


# ---------------------------------------------------------------------------
# Branch: cross-subject SVD (default)
# ---------------------------------------------------------------------------

def _run_cross_subject_svd(
    config, enc, output_dir, svd_dir, subject_ids,
    sig_mask_global, coefficients_all, intercepts_all,
    metadata_df, n_folds, n_total_elecs, n_lags, n_features,
    aligned_embeddings, df_word,
):
    """Compute a single SVD across all subjects, evaluate per-subject, then aggregate."""

    # -- Global SVD --
    print(f"Computing cross-subject SVD (n_directions={config.n_directions})...")
    Vt_arr, all_U, all_S, all_Vt, all_alignments = _filter_sig_and_svd(
        coefficients_all, sig_mask_global, config.n_directions
    )
    actual_n_dirs = Vt_arr.shape[1]
    _save_svd_decomposition(
        os.path.join(svd_dir, "svd_decomposition.npz"),
        all_U, all_S, all_Vt, all_alignments,
    )

    k_values = [k for k in config.k_values if k <= actual_n_dirs]
    print(f"k_values: {k_values}")

    # -- Per-subject evaluation with global Vt --
    per_subject_results = {}
    per_subject_full_corrs_orig = {}
    lags = None
    ref_test_idxs = None

    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        test_idxs, sub_lags, full_corrs_orig = _load_subject_encoding_info(
            sub_id, output_dir, n_folds, ref_test_idxs
        )
        if ref_test_idxs is None:
            ref_test_idxs = test_idxs
        if lags is None:
            lags = sub_lags
        per_subject_full_corrs_orig[sub_id] = full_corrs_orig

        print(f"  Evaluating truncation...")
        per_subject_results[sub_id], n_elecs, _ = _evaluate_subject_truncation(
            sub_id, metadata_df, coefficients_all, intercepts_all, sig_mask_global,
            n_total_elecs, n_lags, n_folds, enc, aligned_embeddings, df_word,
            test_idxs, Vt=Vt_arr, k_values=k_values,
        )
        print(f"  n_elecs={n_elecs}")

    # -- Merge across subjects --
    full_corrs_per_fold = np.concatenate(
        [per_subject_results[s]["full_corrs_per_fold"] for s in subject_ids], axis=1
    )  # (n_folds, n_total_elecs, n_lags)
    truncated_corrs_all = np.concatenate(
        [per_subject_results[s]["truncated_corrs_per_fold"] for s in subject_ids], axis=2
    )  # (n_k, n_folds, n_total_elecs, n_lags)
    merged_results = {
        "k_values": np.array(k_values),
        "full_corrs": full_corrs_per_fold.mean(0),
        "truncated_corrs": truncated_corrs_all.mean(1),
        "sig_mask": sig_mask_global,
        "full_corrs_per_fold": full_corrs_per_fold,
        "truncated_corrs_per_fold": truncated_corrs_all,
    }

    best_k = find_best_k(merged_results, threshold=config.threshold)
    print(f"\nbest_k = {best_k}")

    _save_truncation_results(
        os.path.join(svd_dir, "truncation_results.npz"), merged_results, best_k
    )
    plot_svd_truncation_results(
        merged_results, threshold=config.threshold,
        save_path=os.path.join(svd_dir, "truncation_plot.png"),
    )

    # -- Per-subject output plots at best_k --
    best_ki = int(np.where(np.array(k_values) == best_k)[0][0])
    for sub_id in subject_ids:
        sub_dir = os.path.join(svd_dir, "subjects", f"sub_{sub_id}")
        os.makedirs(sub_dir, exist_ok=True)
        sub_rows = metadata_df.index[metadata_df["subject"] == sub_id].tolist()
        corrs_sub = truncated_corrs_all[best_ki][:, sub_rows, :]
        if lags is not None:
            _plot_subject_results(
                sub_id, sub_dir, corrs_sub,
                per_subject_full_corrs_orig[sub_id],
                sig_mask_global[sub_rows], lags, best_k,
            )
        print(f"  sub_{sub_id}: saved to {sub_dir}")

    # -- Cross-subject output plots --
    if lags is not None:
        corrs_cross_full, _ = load_all_subject_correlations(output_dir, subjects=subject_ids)
        plot_svd_corrs(
            truncated_corrs_all[best_ki], corrs_cross_full, lags,
            title="Cross-subject",
            save_path=os.path.join(svd_dir, f"cross_subject_corrs_plot_k{best_k}.png"),
            best_k=best_k,
        )
        plot_corrs_by_significance(
            truncated_corrs_all[best_ki], sig_mask_global, lags,
            save_path=os.path.join(svd_dir, f"cross_subject_sig_corrs_plot_k{best_k}.png"),
        )


# ---------------------------------------------------------------------------
# Branch: per-subject SVD
# ---------------------------------------------------------------------------

def _run_per_subject_svd(
    config, enc, output_dir, svd_dir, subject_ids,
    sig_mask_global, coefficients_all, intercepts_all,
    metadata_df, n_folds, n_total_elecs, n_lags, n_features,
    aligned_embeddings, df_word,
):
    """Compute an independent SVD per subject; evaluate and report only on that subject."""
    lags = None
    ref_test_idxs = None

    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        test_idxs, sub_lags, full_corrs_orig = _load_subject_encoding_info(
            sub_id, output_dir, n_folds, ref_test_idxs
        )
        if ref_test_idxs is None:
            ref_test_idxs = test_idxs
        if lags is None:
            lags = sub_lags

        sub_rows = metadata_df.index[metadata_df["subject"] == sub_id].tolist()
        sig_mask_sub = sig_mask_global[sub_rows]

        if sig_mask_sub.sum() == 0:
            print(f"  Subject {sub_id}: no significant electrode-lag pairs, skipping.")
            continue

        # -- Per-subject SVD --
        coefs_sub = coefficients_all[:, sub_rows, :, :]
        print(f"  Computing per-subject SVD (n_directions={config.n_directions})...")
        Vt_sub, all_U, all_S, all_Vt, all_alignments = _filter_sig_and_svd(
            coefs_sub, sig_mask_sub, config.n_directions
        )
        actual_n_dirs = Vt_sub.shape[1]
        k_values = [k for k in config.k_values if k <= actual_n_dirs]

        sub_dir = os.path.join(svd_dir, "subjects", f"sub_{sub_id}")
        os.makedirs(sub_dir, exist_ok=True)
        _save_svd_decomposition(
            os.path.join(sub_dir, "svd_decomposition.npz"),
            all_U, all_S, all_Vt, all_alignments,
        )

        # -- Evaluation --
        print(f"  Evaluating truncation (k_values={k_values})...")
        results, n_elecs, sig_mask_sub = _evaluate_subject_truncation(
            sub_id, metadata_df, coefficients_all, intercepts_all, sig_mask_global,
            n_total_elecs, n_lags, n_folds, enc, aligned_embeddings, df_word,
            test_idxs, Vt=Vt_sub, k_values=k_values,
        )
        print(f"  n_elecs={n_elecs}")

        best_k = find_best_k(results, threshold=config.threshold)
        print(f"  best_k = {best_k}")

        _save_truncation_results(
            os.path.join(sub_dir, "truncation_results.npz"), results, best_k
        )
        plot_svd_truncation_results(
            results, threshold=config.threshold,
            save_path=os.path.join(sub_dir, "truncation_plot.png"),
        )

        if lags is not None:
            best_ki = int(np.where(np.array(k_values) == best_k)[0][0])
            corrs_trunc = results["truncated_corrs_per_fold"][best_ki]  # (n_folds, n_elecs, n_lags)
            _plot_subject_results(
                sub_id, sub_dir, corrs_trunc, full_corrs_orig, sig_mask_sub, lags, best_k
            )

        print(f"  Saved to {sub_dir}")


# ---------------------------------------------------------------------------
# Branch: per-region SVD
# ---------------------------------------------------------------------------

def _run_per_region_svd(
    config, enc, output_dir, svd_dir, subject_ids,
    sig_mask_global, coefficients_all, intercepts_all,
    metadata_df, n_folds, n_total_elecs, n_lags, n_features,
    aligned_embeddings, df_word,
):
    """Compute an independent SVD per atlas region; evaluate per-subject within each region."""

    # -- Pre-load encoding info for all subjects --
    per_subject_enc = {}
    lags = None
    ref_test_idxs = None
    for sub_id in subject_ids:
        test_idxs, sub_lags, full_corrs_orig = _load_subject_encoding_info(
            sub_id, output_dir, n_folds, ref_test_idxs
        )
        if ref_test_idxs is None:
            ref_test_idxs = test_idxs
        if lags is None:
            lags = sub_lags
        per_subject_enc[sub_id] = (test_idxs, sub_lags, full_corrs_orig)

    # -- Build electrode region map --
    print("Building electrode region map...")
    electrode_region_map = build_electrode_region_map(
        subject_ids, enc.data_root, config.atlas_region_groups, config.atlas_path
    )
    print(f"  Regions found: {list(electrode_region_map.keys())}")

    # -- Per-region SVD --
    for region_name, sub_elec_map in electrode_region_map.items():
        print(f"\n{'='*60}\nRegion: {region_name}")

        region_dir = os.path.join(svd_dir, "regions", region_name)
        os.makedirs(region_dir, exist_ok=True)

        # Filter sub_elec_map to only electrodes present in metadata_df (encoding model output).
        # build_electrode_region_map reads all raw channels, but some may have been excluded
        # from the encoding model (e.g. too few words), so we restrict to the intersection.
        sub_elec_map = {
            sub_id: [e for e in elecs
                     if e in set(metadata_df.loc[metadata_df["subject"] == sub_id, "electrode"])]
            for sub_id, elecs in sub_elec_map.items()
        }
        sub_elec_map = {sub_id: elecs for sub_id, elecs in sub_elec_map.items() if elecs}

        # Build region_rows in metadata_df order (same order as _evaluate_subject_truncation)
        region_rows = []
        for sub_id, elec_list in sub_elec_map.items():
            elec_set = set(elec_list)
            for row_i in metadata_df.index[metadata_df["subject"] == sub_id].tolist():
                if metadata_df.loc[row_i, "electrode"] in elec_set:
                    region_rows.append(row_i)

        coefs_region = coefficients_all[:, region_rows, :, :]
        sig_mask_region = sig_mask_global[region_rows]

        n_sig = sig_mask_region.sum()
        print(f"  Electrodes: {len(region_rows)}, significant pairs: {n_sig}")
        if n_sig == 0:
            print(f"  No significant electrode-lag pairs, skipping.")
            continue

        # -- Region SVD --
        print(f"  Computing SVD (n_directions={config.n_directions})...")
        Vt_region, all_U, all_S, all_Vt, all_alignments = _filter_sig_and_svd(
            coefs_region, sig_mask_region, config.n_directions
        )
        actual_n_dirs = Vt_region.shape[1]
        k_values = [k for k in config.k_values if k <= actual_n_dirs]
        if not k_values:
            print(f"  No valid k_values after filtering, skipping.")
            continue

        _save_svd_decomposition(
            os.path.join(region_dir, "svd_decomposition.npz"),
            all_U, all_S, all_Vt, all_alignments,
        )

        # -- Per-subject evaluation --
        per_subject_results = {}
        subs_in_region = list(sub_elec_map.keys())
        for sub_id in subs_in_region:
            elec_list = sub_elec_map[sub_id]
            test_idxs, _, _ = per_subject_enc[sub_id]
            print(f"  Evaluating subject {sub_id} ({len(elec_list)} electrodes)...")
            per_subject_results[sub_id], n_elecs, _ = _evaluate_subject_truncation(
                sub_id, metadata_df, coefficients_all, intercepts_all, sig_mask_global,
                n_total_elecs, n_lags, n_folds, enc, aligned_embeddings, df_word,
                test_idxs, Vt=Vt_region, k_values=k_values,
                electrode_subset=elec_list,
            )
            print(f"    n_elecs={n_elecs}")

        # -- Merge across subjects --
        full_corrs_per_fold = np.concatenate(
            [per_subject_results[s]["full_corrs_per_fold"] for s in subs_in_region], axis=1
        )  # (n_folds, n_region_elecs, n_lags)
        truncated_corrs_all = np.concatenate(
            [per_subject_results[s]["truncated_corrs_per_fold"] for s in subs_in_region], axis=2
        )  # (n_k, n_folds, n_region_elecs, n_lags)
        merged_results = {
            "k_values": np.array(k_values),
            "full_corrs": full_corrs_per_fold.mean(0),
            "truncated_corrs": truncated_corrs_all.mean(1),
            "sig_mask": sig_mask_region,
            "full_corrs_per_fold": full_corrs_per_fold,
            "truncated_corrs_per_fold": truncated_corrs_all,
        }

        best_k = find_best_k(merged_results, threshold=config.threshold)
        print(f"  best_k = {best_k}")

        _save_truncation_results(
            os.path.join(region_dir, "truncation_results.npz"), merged_results, best_k
        )
        plot_svd_truncation_results(
            merged_results, threshold=config.threshold,
            save_path=os.path.join(region_dir, "truncation_plot.png"),
        )

        # -- Per-subject plots at best_k --
        best_ki = int(np.where(np.array(k_values) == best_k)[0][0])
        offset = 0
        for sub_id in subs_in_region:
            sub_dir = os.path.join(region_dir, "subjects", f"sub_{sub_id}")
            os.makedirs(sub_dir, exist_ok=True)

            n_sub_elecs = per_subject_results[sub_id]["full_corrs_per_fold"].shape[1]
            corrs_sub = truncated_corrs_all[best_ki, :, offset:offset + n_sub_elecs, :]
            sub_full_corrs = per_subject_results[sub_id]["full_corrs_per_fold"]

            sub_region_rows = [
                r for r in metadata_df.index[metadata_df["subject"] == sub_id].tolist()
                if metadata_df.loc[r, "electrode"] in set(sub_elec_map[sub_id])
            ]
            sig_mask_sub = sig_mask_global[sub_region_rows]

            if lags is not None:
                _plot_subject_results(
                    sub_id, sub_dir, corrs_sub, sub_full_corrs, sig_mask_sub, lags, best_k
                )
            print(f"  sub_{sub_id}: saved to {sub_dir}")
            offset += n_sub_elecs

        # -- Region-level aggregate plots --
        if lags is not None:
            plot_svd_corrs(
                truncated_corrs_all[best_ki], full_corrs_per_fold, lags,
                title=f"Region: {region_name}",
                save_path=os.path.join(region_dir, f"region_corrs_plot_k{best_k}.png"),
                best_k=best_k,
            )
            plot_corrs_by_significance(
                truncated_corrs_all[best_ki], sig_mask_region, lags,
                save_path=os.path.join(region_dir, f"region_sig_corrs_plot_k{best_k}.png"),
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args, overrides = parse_args("SVD decomposition of encoding model weights")
    config: SVDConfig = load_config(args.config, overrides, SVDConfig)
    enc = config.encoding

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)

    if config.svd_output_dir:
        svd_dir = config.svd_output_dir
    elif config.mode == SVDMode.PER_REGION:
        svd_dir = os.path.join(output_dir, "per_region_svd")
    elif config.mode == SVDMode.PER_SUBJECT:
        svd_dir = os.path.join(output_dir, "per_subject_svd")
    else:
        svd_dir = os.path.join(output_dir, "svd")

    os.makedirs(svd_dir, exist_ok=True)
    print(f"Output dir:  {output_dir}")
    print(f"SVD dir:     {svd_dir}")
    print(f"Mode:        {config.mode.value}")

    # -- Load significance CSV --
    sig_csv = os.path.join(output_dir, "significance_detailed.csv")
    if not os.path.exists(sig_csv):
        raise FileNotFoundError(
            f"significance_detailed.csv not found at {sig_csv}. "
            "Run run_significance.py first."
        )
    sig_df = pd.read_csv(sig_csv)
    print(f"Loaded significance CSV: {len(sig_df):,} rows")

    # -- Load all subject coefficients --
    print("Loading all subject coefficients...")
    coefficients_all, intercepts_all, metadata_df = load_all_subject_coefficients(
        output_dir, subjects=subject_ids
    )
    n_folds, n_total_elecs, n_lags, n_features = coefficients_all.shape
    print(f"  Shape: {coefficients_all.shape}")

    sig_mask_global = _build_global_sig_mask(sig_df, metadata_df, n_lags)
    print(f"  Significant pairs: {sig_mask_global.sum()} / {sig_mask_global.size}")

    # -- Load embeddings --
    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    # -- Dispatch --
    branch_kwargs = dict(
        config=config, enc=enc, output_dir=output_dir, svd_dir=svd_dir,
        subject_ids=subject_ids,
        sig_mask_global=sig_mask_global,
        coefficients_all=coefficients_all, intercepts_all=intercepts_all,
        metadata_df=metadata_df,
        n_folds=n_folds, n_total_elecs=n_total_elecs, n_lags=n_lags, n_features=n_features,
        aligned_embeddings=aligned_embeddings, df_word=df_word,
    )

    if config.mode == SVDMode.PER_REGION:
        _run_per_region_svd(**branch_kwargs)
    elif config.mode == SVDMode.PER_SUBJECT:
        _run_per_subject_svd(**branch_kwargs)
    else:
        _run_cross_subject_svd(**branch_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
