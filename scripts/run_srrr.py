"""
run_srrr.py — Sparse Reduced Rank Regression on SVD-projected neural encoding weights.

Reads SVD decompositions written by run_svd_analysis.py and, for each fold of
cross-validation, finds the *minimal* subset of SVD directions that explains all
significant variance captured by the full encoding model.

Three modes controlled by SRRRConfig.mode (mirrors SVDConfig.mode):

  cross_subject (default):
    Reads SVD from {output_dir}/svd/.
    Builds global Y across all subjects (significant electrodes).
    Writes to {output_dir}/srrr/.

  per_subject:
    Reads SVD from {output_dir}/per_subject_svd/subjects/sub_{id}/.
    Runs SRRR independently per subject.
    Writes to {output_dir}/per_subject_srrr/subjects/sub_{id}/.

  per_region:
    Reads SVD from {output_dir}/per_region_svd/regions/{region}/.
    Runs SRRR independently per atlas region.
    Writes to {output_dir}/per_region_srrr/regions/{region}/.

Output layout per unit (subject or region):
    {srrr_unit_dir}/
        grid_results.csv
        summary.csv
        knee_selection.csv
        null_dist_fold_{f}_k{k}.npy
        k{k}_lam{l:.2e}/fold_{f}/
            selection_probs.npy, selected_idx.npy, A.npy, B.npy, metrics.npz
"""

import gc
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import (
    load_raws,
    load_embeddings,
    build_word_epochs,
    apply_offset,
    load_all_subject_coefficients,
)
from utils.atlas import build_electrode_region_map
from utils.config import SRRRConfig, SVDMode, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import load_encoding_results
from utils.significance import load_significance_results
from utils.srrr import (
    build_null_distribution,
    compute_cca_similarity,
    compute_procrustes_similarity,
    fit_dense_rrr,
    find_knee_lambda,
    load_svd_decomposition,
    predict_from_factors,
    project_to_svd_space,
    run_stability_selection,
    run_sufficiency_test,
    save_srrr_result,
    select_directions,
)


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
            if 0 <= lag_i < n_lags:
                mask[row_i, lag_i] = bool(sig_row["sig"])
    return mask


def _load_subject_neural_data(sub_id, sub_elec_names, enc, aligned_embeddings, df_word):
    """Load neural data Y and aligned embeddings X for one subject.

    Returns:
        X_arr: (n_words, n_embed) embeddings aligned to epochs.
        Y: (n_words * n_lags, n_elecs) — raw neural data before any offset.
        n_elecs: int
    """
    raws = load_raws(
        [sub_id], enc.data_root, use_high_gamma=True,
        per_subject_electrodes={sub_id: sub_elec_names},
    )
    raw = raws[0]
    n_elecs = len(raw.ch_names)
    epochs, data = build_word_epochs(raw, df_word)
    X_arr, Y = apply_offset(aligned_embeddings[epochs.selection], data, enc.offset)
    del raws, raw, epochs, data
    gc.collect()
    return X_arr, Y, n_elecs


def _run_srrr_for_unit(
    svd_dir: str,
    srrr_unit_dir: str,
    X_raw: np.ndarray,
    Y_full: np.ndarray,
    test_idxs: list,
    n_words: int,
    config: SRRRConfig,
) -> pd.DataFrame:
    """Core SRRR loop for one analysis unit (cross-subject, per-subject, or per-region).

    Args:
        svd_dir: Directory containing svd_decomposition.npz.
        srrr_unit_dir: Root output directory for this unit.
        X_raw: (n_words, n_embed) raw embedding matrix.
        Y_full: (n_words, n_targets) neural response matrix (already z-scored or raw).
        test_idxs: List of test index arrays, one per fold.
        n_words: Total number of words (n_words == X_raw.shape[0]).
        config: SRRRConfig.

    Returns:
        DataFrame of per-(fold, k, lam) metrics for grid_results.csv.
    """
    os.makedirs(srrr_unit_dir, exist_ok=True)

    n_folds = len(test_idxs)
    Vt_arr = load_svd_decomposition(svd_dir, n_folds)  # (n_folds, n_dirs, n_embed)
    n_dirs = Vt_arr.shape[1]

    all_rows = []

    for fold_f in range(n_folds):
        test_idx = test_idxs[fold_f]
        train_mask = np.ones(n_words, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        Vt_f = Vt_arr[fold_f]  # (n_dirs, n_embed)
        X_proj_full = project_to_svd_space(X_raw, Vt_f)  # (n_words, n_dirs)
        X_proj_train = X_proj_full[train_idx]
        X_proj_test = X_proj_full[test_idx]

        # Z-score Y on training set
        scaler = StandardScaler()
        Y_train_raw = Y_full[train_idx]
        Y_test_raw = Y_full[test_idx]
        Y_train = scaler.fit_transform(Y_train_raw)
        Y_test = scaler.transform(Y_test_raw)

        for k in config.rank_grid:
            null_cache = os.path.join(
                srrr_unit_dir, f"null_dist_fold_{fold_f}_k{k}.npy"
            ) if config.cache_null_dist else None

            null_r2 = build_null_distribution(
                X_proj_train, Y_train,
                X_proj_test, Y_test,
                k=k,
                n_permutations=config.n_permutations,
                cache_path=null_cache,
                seed=fold_f * 1000 + k,
            )

            # Fit full model (all directions) for reference
            full_result = fit_dense_rrr(
                X_proj_train, Y_train, k=k,
                selected_idx=np.arange(n_dirs),
            )
            Y_hat_full = predict_from_factors(
                X_proj_test,
                full_result["A_dense"],
                full_result["B_dense"],
                selected_idx=np.arange(n_dirs),
            )

            prev_A = None
            prev_B = None

            for lam in sorted(config.lambda_grid):
                # Stability selection
                sel_probs = run_stability_selection(
                    X_proj_train, Y_train,
                    k=k, lam=lam,
                    n_iters=config.n_stability_iters,
                    subsample_frac=config.subsample_fraction,
                    max_iter=config.max_iter,
                    tol=config.tol,
                    lr=config.learning_rate,
                    seed=fold_f * 100000 + k * 1000 + int(lam * 1000),
                )
                selected_idx = select_directions(sel_probs, threshold=config.selection_threshold)

                # Consensus model on selected directions
                if len(selected_idx) == 0:
                    n_targets = Y_train.shape[1]
                    A_cons = np.zeros((0, k), dtype=np.float32)
                    B_cons = np.zeros((n_targets, k), dtype=np.float32)
                    Y_hat_min = np.zeros_like(Y_test)
                else:
                    # Warm start: if warm_start and previous A exists, slice to selected rows
                    A_init = None
                    B_init = None
                    if config.warm_start and prev_A is not None:
                        pass  # Stability selection uses fresh starts; warm start for consensus only
                        A_init = None  # Could slice prev_A[selected_idx] but indices may differ

                    cons_result = fit_dense_rrr(
                        X_proj_train, Y_train, k=k,
                        selected_idx=selected_idx,
                    )
                    A_cons = cons_result["A_dense"]
                    B_cons = cons_result["B_dense"]
                    Y_hat_min = predict_from_factors(
                        X_proj_test, A_cons, B_cons, selected_idx=selected_idx
                    )

                    prev_A = A_cons
                    prev_B = B_cons

                # Sufficiency test
                suf = run_sufficiency_test(
                    X_proj_test, Y_test,
                    Y_hat_full=Y_hat_full,
                    Y_hat_min=Y_hat_min,
                    null_r2=null_r2,
                    p_value_threshold=config.p_value_threshold,
                )

                # Alignment metrics
                cca_sim = compute_cca_similarity(Y_hat_full, Y_hat_min)
                proc_sim = compute_procrustes_similarity(Y_hat_full, Y_hat_min)

                metrics = {
                    **suf,
                    "cca_similarity": cca_sim,
                    "procrustes_similarity": proc_sim,
                    "n_selected": len(selected_idx),
                    "fold": fold_f,
                    "k": k,
                    "lam": lam,
                }

                save_srrr_result(
                    srrr_unit_dir, k=k, lam=lam, fold=fold_f,
                    selection_probs=sel_probs,
                    selected_idx=selected_idx,
                    A=A_cons, B=B_cons,
                    metrics=metrics,
                )

                all_rows.append(metrics)
                print(
                    f"    fold={fold_f} k={k} lam={lam:.2e} "
                    f"n_sel={len(selected_idx)} "
                    f"min_r2={suf['minimal_r2']:.3f} "
                    f"full_r2={suf['full_r2']:.3f} "
                    f"sufficient={suf['sufficient']}"
                )

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(srrr_unit_dir, "grid_results.csv"), index=False)

    # Summary: average over folds
    summary = (
        df.groupby(["k", "lam"])
        .agg(
            minimal_r2=("minimal_r2", "mean"),
            full_r2=("full_r2", "mean"),
            recovery_r2=("recovery_r2", "mean"),
            sufficient_frac=("sufficient", "mean"),
            n_selected=("n_selected", "mean"),
            cca_similarity=("cca_similarity", "mean"),
            procrustes_similarity=("procrustes_similarity", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(srrr_unit_dir, "summary.csv"), index=False)

    # Knee selection: for each k, find knee lambda by minimal_r2 vs lambda
    knee_rows = []
    for k in config.rank_grid:
        sub = summary[summary["k"] == k].sort_values("lam")
        if len(sub) == 0:
            continue
        knee_lam = find_knee_lambda(
            sub["lam"].tolist(),
            sub["minimal_r2"].values,
            sensitivity=config.knee_sensitivity,
        )
        knee_row = sub[sub["lam"] == knee_lam].iloc[0].to_dict()
        knee_row["knee_lam"] = knee_lam
        knee_rows.append(knee_row)

    pd.DataFrame(knee_rows).to_csv(
        os.path.join(srrr_unit_dir, "knee_selection.csv"), index=False
    )

    return df


# ---------------------------------------------------------------------------
# Branch: cross-subject SRRR
# ---------------------------------------------------------------------------

def _run_cross_subject_srrr(
    config, enc, output_dir, srrr_dir, subject_ids,
    sig_mask_global, metadata_df, n_folds, n_total_elecs, n_lags,
    aligned_embeddings, df_word,
):
    """SRRR using the cross-subject SVD basis."""
    svd_dir = os.path.join(output_dir, "svd")

    # Load reference test_idxs from first subject
    ref_test_idxs = None
    for sub_id in subject_ids:
        sub_enc_dir = os.path.join(output_dir, "subjects", f"sub_{sub_id}")
        enc_results = load_encoding_results(
            os.path.join(sub_enc_dir, "encoding_results.joblib")
        )
        if ref_test_idxs is None:
            ref_test_idxs = enc_results["test_idxs"]
        break

    # Build global Y: significant (electrode, lag) targets across all subjects
    # For SRRR we use the raw word-level embeddings and neural responses.
    # Build Y as a flat concatenation of significant electrode-lag responses.
    # Collect all significant electrode-lag column indices (flattened mask).
    sig_flat = sig_mask_global.flatten()  # (n_total_elecs * n_lags,)
    n_sig_targets = int(sig_flat.sum())
    if n_sig_targets == 0:
        print("  No significant electrode-lag pairs; skipping cross-subject SRRR.")
        return

    # We need X and Y at the word level. Load first subject to get word count.
    print("  Building global Y from significant electrode-lag pairs...")
    # Use one subject to get embeddings dimension and word alignment
    first_sub = subject_ids[0]
    sub_rows = metadata_df.index[metadata_df["subject"] == first_sub].tolist()
    sub_elec_names = metadata_df.loc[sub_rows, "electrode"].tolist()

    X_arr, Y_first, n_elecs_first = _load_subject_neural_data(
        first_sub, sub_elec_names, enc, aligned_embeddings, df_word
    )
    n_words = X_arr.shape[0]

    # Collect Y columns from all subjects for significant (electrode, lag) pairs
    # This mirrors the weight matrix construction in SVD analysis
    # We use per-subject neural data and slice significant lags
    Y_cols = []
    for sub_id in subject_ids:
        sub_rows_sub = metadata_df.index[metadata_df["subject"] == sub_id].tolist()
        sub_elec_names_sub = metadata_df.loc[sub_rows_sub, "electrode"].tolist()
        sig_mask_sub = sig_mask_global[sub_rows_sub]  # (n_sub_elecs, n_lags)

        if sig_mask_sub.sum() == 0:
            continue

        if sub_id == first_sub:
            X_sub = X_arr
            Y_sub_raw = Y_first
        else:
            X_sub, Y_sub_raw, _ = _load_subject_neural_data(
                sub_id, sub_elec_names_sub, enc, aligned_embeddings, df_word
            )

        # Y_sub_raw shape: (n_words, n_sub_elecs * n_lags) or (n_words, n_sub_elecs)?
        # apply_offset returns Y as (n_words, n_elecs * n_lags) typically.
        # We need to select columns corresponding to significant (elec, lag) pairs.
        # Reshape to (n_words, n_sub_elecs, n_lags) and flatten sig mask
        n_sub_elecs = len(sub_elec_names_sub)
        expected_cols = n_sub_elecs * n_lags
        if Y_sub_raw.shape[1] == expected_cols:
            Y_sub_3d = Y_sub_raw.reshape(n_words, n_sub_elecs, n_lags)
            sig_mask_flat_sub = sig_mask_sub.flatten()
            # Select significant columns
            Y_sub_sig = Y_sub_raw[:, sig_mask_flat_sub]
        else:
            # Y is (n_words, n_elecs) — no lag dimension; use all electrodes if any sig
            Y_sub_sig = Y_sub_raw[:, np.any(sig_mask_sub, axis=1)]

        if Y_sub_sig.shape[1] > 0:
            Y_cols.append(Y_sub_sig)

    if not Y_cols:
        print("  No valid Y columns found; skipping.")
        return

    Y_full = np.concatenate(Y_cols, axis=1).astype(np.float32)
    print(f"  X shape: {X_arr.shape}, Y shape: {Y_full.shape}")

    _run_srrr_for_unit(
        svd_dir=svd_dir,
        srrr_unit_dir=srrr_dir,
        X_raw=X_arr,
        Y_full=Y_full,
        test_idxs=ref_test_idxs,
        n_words=n_words,
        config=config,
    )


# ---------------------------------------------------------------------------
# Branch: per-subject SRRR
# ---------------------------------------------------------------------------

def _run_per_subject_srrr(
    config, enc, output_dir, srrr_dir, subject_ids,
    sig_mask_global, metadata_df, n_folds, n_total_elecs, n_lags,
    aligned_embeddings, df_word,
):
    """SRRR using per-subject SVD bases."""
    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        svd_sub_dir = os.path.join(output_dir, "per_subject_svd", "subjects", f"sub_{sub_id}")
        srrr_sub_dir = os.path.join(srrr_dir, "subjects", f"sub_{sub_id}")

        if not os.path.exists(os.path.join(svd_sub_dir, "svd_decomposition.npz")):
            print(f"  SVD decomposition not found at {svd_sub_dir}; skipping.")
            continue

        sub_rows = metadata_df.index[metadata_df["subject"] == sub_id].tolist()
        sub_elec_names = metadata_df.loc[sub_rows, "electrode"].tolist()
        sig_mask_sub = sig_mask_global[sub_rows]  # (n_sub_elecs, n_lags)

        if sig_mask_sub.sum() == 0:
            print(f"  No significant electrode-lag pairs; skipping.")
            continue

        # Load test_idxs for this subject
        sub_enc_dir = os.path.join(output_dir, "subjects", f"sub_{sub_id}")
        enc_results = load_encoding_results(
            os.path.join(sub_enc_dir, "encoding_results.joblib")
        )
        test_idxs = enc_results["test_idxs"]

        # Load neural data
        X_arr, Y_raw, n_elecs = _load_subject_neural_data(
            sub_id, sub_elec_names, enc, aligned_embeddings, df_word
        )
        n_words = X_arr.shape[0]

        # Select significant columns
        n_sub_elecs = len(sub_elec_names)
        if Y_raw.shape[1] == n_sub_elecs * n_lags:
            sig_flat = sig_mask_sub.flatten()
            Y_sig = Y_raw[:, sig_flat]
        else:
            Y_sig = Y_raw[:, np.any(sig_mask_sub, axis=1)]

        if Y_sig.shape[1] == 0:
            print(f"  No significant targets after column selection; skipping.")
            continue

        print(f"  X: {X_arr.shape}, Y: {Y_sig.shape}")
        _run_srrr_for_unit(
            svd_dir=svd_sub_dir,
            srrr_unit_dir=srrr_sub_dir,
            X_raw=X_arr,
            Y_full=Y_sig.astype(np.float32),
            test_idxs=test_idxs,
            n_words=n_words,
            config=config,
        )


# ---------------------------------------------------------------------------
# Branch: per-region SRRR
# ---------------------------------------------------------------------------

def _run_per_region_srrr(
    config, enc, output_dir, srrr_dir, subject_ids,
    sig_mask_global, metadata_df, n_folds, n_total_elecs, n_lags,
    aligned_embeddings, df_word,
):
    """SRRR using per-region SVD bases."""

    # Pre-load test_idxs and X from first subject (word order shared)
    first_sub = subject_ids[0]
    sub_enc_dir = os.path.join(output_dir, "subjects", f"sub_{first_sub}")
    enc_results = load_encoding_results(
        os.path.join(sub_enc_dir, "encoding_results.joblib")
    )
    ref_test_idxs = enc_results["test_idxs"]

    # Build electrode region map
    print("Building electrode region map...")
    electrode_region_map = build_electrode_region_map(
        subject_ids, enc.data_root,
        getattr(config, "atlas_region_groups", None),
        getattr(config, "atlas_path", None),
    )
    print(f"  Regions found: {list(electrode_region_map.keys())}")

    for region_name, sub_elec_map in electrode_region_map.items():
        print(f"\n{'='*60}\nRegion: {region_name}")

        svd_region_dir = os.path.join(output_dir, "per_region_svd", "regions", region_name)
        srrr_region_dir = os.path.join(srrr_dir, "regions", region_name)

        if not os.path.exists(os.path.join(svd_region_dir, "svd_decomposition.npz")):
            print(f"  SVD decomposition not found at {svd_region_dir}; skipping.")
            continue

        # Filter to electrodes present in metadata_df
        sub_elec_map = {
            sub_id: [e for e in elecs
                     if e in set(metadata_df.loc[metadata_df["subject"] == sub_id, "electrode"])]
            for sub_id, elecs in sub_elec_map.items()
        }
        sub_elec_map = {s: e for s, e in sub_elec_map.items() if e}

        # Collect Y columns across subjects in this region
        X_ref = None
        Y_cols = []

        for sub_id, elec_list in sub_elec_map.items():
            sub_rows_sub = [
                r for r in metadata_df.index[metadata_df["subject"] == sub_id].tolist()
                if metadata_df.loc[r, "electrode"] in set(elec_list)
            ]
            if not sub_rows_sub:
                continue

            sig_mask_sub = sig_mask_global[sub_rows_sub]
            if sig_mask_sub.sum() == 0:
                continue

            X_arr, Y_raw, n_elecs = _load_subject_neural_data(
                sub_id, elec_list, enc, aligned_embeddings, df_word
            )
            if X_ref is None:
                X_ref = X_arr

            n_sub_elecs = len(elec_list)
            if Y_raw.shape[1] == n_sub_elecs * n_lags:
                sig_flat = sig_mask_sub.flatten()
                Y_sig = Y_raw[:, sig_flat]
            else:
                Y_sig = Y_raw[:, np.any(sig_mask_sub, axis=1)]

            if Y_sig.shape[1] > 0:
                Y_cols.append(Y_sig)

        if X_ref is None or not Y_cols:
            print("  No valid data for region; skipping.")
            continue

        Y_full = np.concatenate(Y_cols, axis=1).astype(np.float32)
        n_words = X_ref.shape[0]
        print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

        _run_srrr_for_unit(
            svd_dir=svd_region_dir,
            srrr_unit_dir=srrr_region_dir,
            X_raw=X_ref,
            Y_full=Y_full,
            test_idxs=ref_test_idxs,
            n_words=n_words,
            config=config,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args, overrides = parse_args("SRRR: find minimal SVD directions for encoding model")
    config: SRRRConfig = load_config(args.config, overrides, SRRRConfig)
    enc = config.encoding

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)

    # Resolve SRRR output directory
    if config.srrr_output_dir:
        srrr_dir = config.srrr_output_dir
    elif config.mode == SVDMode.PER_REGION:
        srrr_dir = os.path.join(output_dir, "per_region_srrr")
    elif config.mode == SVDMode.PER_SUBJECT:
        srrr_dir = os.path.join(output_dir, "per_subject_srrr")
    else:
        srrr_dir = os.path.join(output_dir, "srrr")

    os.makedirs(srrr_dir, exist_ok=True)
    print(f"Output dir:  {output_dir}")
    print(f"SRRR dir:    {srrr_dir}")
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

    # -- Load all subject coefficients (for metadata and n_folds) --
    print("Loading all subject coefficients...")
    coefficients_all, intercepts_all, metadata_df = load_all_subject_coefficients(
        output_dir, subjects=subject_ids
    )
    n_folds, n_total_elecs, n_lags, n_features = coefficients_all.shape
    print(f"  Shape: {coefficients_all.shape}")
    del coefficients_all, intercepts_all  # not needed for SRRR

    sig_mask_global = _build_global_sig_mask(sig_df, metadata_df, n_lags)
    print(f"  Significant pairs: {sig_mask_global.sum()} / {sig_mask_global.size}")

    # -- Load embeddings --
    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    # -- Dispatch --
    branch_kwargs = dict(
        config=config, enc=enc, output_dir=output_dir, srrr_dir=srrr_dir,
        subject_ids=subject_ids,
        sig_mask_global=sig_mask_global,
        metadata_df=metadata_df,
        n_folds=n_folds, n_total_elecs=n_total_elecs, n_lags=n_lags,
        aligned_embeddings=aligned_embeddings, df_word=df_word,
    )

    if config.mode == SVDMode.PER_REGION:
        _run_per_region_srrr(**branch_kwargs)
    elif config.mode == SVDMode.PER_SUBJECT:
        _run_per_subject_srrr(**branch_kwargs)
    else:
        _run_cross_subject_srrr(**branch_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
