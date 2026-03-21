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
        null_dist_fold_{f}.npy
        k{k}_lam{l:.2e}/fold_{f}/
            selection_probs.npy, selected_idx.npy, A.npy, B.npy, metrics.npz
"""

import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import (
    load_raws,
    load_embeddings,
    build_word_epochs,
    apply_offset,
)
from utils.atlas import build_electrode_region_map
from utils.config import SRRRConfig, SVDMode, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import load_encoding_results, get_ridge_model_constructor
from utils.srrr import (
    build_null_distribution,
    compute_cca_similarity,
    compute_procrustes_similarity,
    fit_dense_rrr,
    find_knee_lambda,
    load_svd_decomposition,
    predict_from_factors,
    run_stability_selection,
    run_sufficiency_test,
    save_srrr_result,
    select_directions,
)


# ---------------------------------------------------------------------------
# Core per-unit loop
# ---------------------------------------------------------------------------

def _run_srrr_for_unit(
    svd_dir: str,
    srrr_unit_dir: str,
    X_raw: np.ndarray,
    Y_full: np.ndarray,
    test_idxs: list,
    config: SRRRConfig,
) -> pd.DataFrame:
    """Core SRRR loop for one analysis unit (cross-subject, per-subject, or per-region).

    Args:
        svd_dir: Directory containing svd_decomposition.npz.
        srrr_unit_dir: Root output directory for this unit.
        X_raw: (n_words, n_embed) raw embedding matrix.
        Y_full: (n_words, n_targets) neural response matrix.
        test_idxs: List of test index arrays, one per fold.
        config: SRRRConfig.

    Returns:
        DataFrame of per-(fold, k, lam) metrics for grid_results.csv.
    """
    os.makedirs(srrr_unit_dir, exist_ok=True)

    n_folds = len(test_idxs)
    n_words = X_raw.shape[0]
    Vt_arr = load_svd_decomposition(svd_dir, n_folds)  # (n_folds, n_dirs, n_embed)

    all_rows = []

    for fold_f in range(n_folds):
        fold_start = time.time()
        print(f"\n[Fold {fold_f + 1}/{n_folds}]")

        test_idx = test_idxs[fold_f]
        train_mask = np.ones(n_words, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        Vt_f = Vt_arr[fold_f]  # (n_dirs, n_embed)
        X_proj = X_raw @ Vt_f.T  # (n_words, n_dirs)
        X_proj_train = X_proj[train_idx]
        X_proj_test = X_proj[test_idx]

        # Z-score Y on training set, apply to test
        scaler = StandardScaler()
        Y_train = scaler.fit_transform(Y_full[train_idx])
        Y_test = scaler.transform(Y_full[test_idx])

        # Full ridge baseline — fitted once per fold, shared across all k/lam
        print("  Fitting full ridge baseline...")
        ridge_model = get_ridge_model_constructor()(fold_f)
        ridge_model.fit(X_proj_train, Y_train)
        Y_hat_full = ridge_model.predict(X_proj_test)

        # Null distribution via ridge permutations — once per fold (no rank dependence)
        null_cache = (
            os.path.join(srrr_unit_dir, f"null_dist_fold_{fold_f}.npy")
            if config.cache_null_dist else None
        )
        cached = null_cache and os.path.exists(null_cache)
        print(f"  Building null distribution ({config.n_permutations} permutations){' [cached]' if cached else ''}...")
        null_r2 = build_null_distribution(
            X_proj_train, Y_train, X_proj_test, Y_test,
            model_constructor_fn=get_ridge_model_constructor,
            n_permutations=config.n_permutations,
            cache_path=null_cache,
            seed=fold_f,
        )
        print(f"  Null distribution ready. Running {len(config.rank_grid) * len(config.lambda_grid)} (k, lam) combos...")

        grid = [(k, lam) for k in config.rank_grid for lam in sorted(config.lambda_grid)]
        for k, lam in tqdm(grid, desc=f"  fold {fold_f}", leave=True, unit="combo"):
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

            if len(selected_idx) == 0:
                warnings.warn(
                    f"No directions selected at fold={fold_f}, k={k}, lam={lam:.2e}. "
                    "Consider lowering selection_threshold or lambda_grid values.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                n_targets = Y_train.shape[1]
                A_cons = np.zeros((0, k), dtype=np.float32)
                B_cons = np.zeros((n_targets, k), dtype=np.float32)
                Y_hat_min = np.zeros_like(Y_test)
                Y_hat_min_train = np.zeros_like(Y_train)
            else:
                cons_result = fit_dense_rrr(
                    X_proj_train, Y_train, k=k, selected_idx=selected_idx,
                )
                A_cons = cons_result["A_dense"]
                B_cons = cons_result["B_dense"]
                Y_hat_min = predict_from_factors(
                    X_proj_test, A_cons, B_cons, selected_idx=selected_idx,
                )
                Y_hat_min_train = predict_from_factors(
                    X_proj_train, A_cons, B_cons, selected_idx=selected_idx,
                )

            suf = run_sufficiency_test(
                X_proj_train, Y_train - Y_hat_min_train,
                X_proj_test, Y_test,
                Y_hat_full=Y_hat_full,
                Y_hat_min=Y_hat_min,
                null_r2=null_r2,
                p_value_threshold=config.p_value_threshold,
                model_constructor_fn=get_ridge_model_constructor,
            )
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
            f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s"
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

    # Knee selection: for each k, find the knee lambda by minimal_r2 vs lambda
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
    sig_electrodes, n_folds, aligned_embeddings, df_word,
):
    """SRRR using the cross-subject SVD basis."""
    svd_dir = os.path.join(output_dir, "svd")

    enc_results = load_encoding_results(
        os.path.join(output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib")
    )
    ref_test_idxs = enc_results["test_idxs"]

    # Load all subjects' significant electrodes at once
    raws = load_raws(
        subject_ids, enc.data_root, use_high_gamma=True,
        per_subject_electrodes=sig_electrodes,
    )
    Y_cols = []
    X_ref = None
    for raw, sub_id in zip(raws, subject_ids):
        if sub_id not in sig_electrodes:
            continue
        epochs, data = build_word_epochs(raw, df_word)
        X_arr, Y_sub = apply_offset(aligned_embeddings[epochs.selection], data, enc.offset)
        if X_ref is None:
            X_ref = X_arr
        Y_cols.append(Y_sub)  # (n_words, n_elecs * n_lags)
    del raws
    gc.collect()

    if not Y_cols:
        print("  No significant electrodes found; skipping cross-subject SRRR.")
        return

    Y_full = np.concatenate(Y_cols, axis=1).astype(np.float32)
    print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

    _run_srrr_for_unit(
        svd_dir=svd_dir,
        srrr_unit_dir=srrr_dir,
        X_raw=X_ref,
        Y_full=Y_full,
        test_idxs=ref_test_idxs,
        config=config,
    )


# ---------------------------------------------------------------------------
# Branch: per-subject SRRR
# ---------------------------------------------------------------------------

def _run_per_subject_srrr(
    config, enc, output_dir, srrr_dir, subject_ids,
    sig_electrodes, n_folds, aligned_embeddings, df_word,
):
    """SRRR using per-subject SVD bases."""
    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        svd_sub_dir = os.path.join(output_dir, "per_subject_svd", "subjects", f"sub_{sub_id}")
        srrr_sub_dir = os.path.join(srrr_dir, "subjects", f"sub_{sub_id}")

        if not os.path.exists(os.path.join(svd_sub_dir, "svd_decomposition.npz")):
            print(f"  SVD decomposition not found at {svd_sub_dir}; skipping.")
            continue

        sub_sig_elecs = sig_electrodes.get(sub_id, [])
        if not sub_sig_elecs:
            print(f"  No significant electrodes; skipping.")
            continue

        enc_results = load_encoding_results(
            os.path.join(output_dir, "subjects", f"sub_{sub_id}", "encoding_results.joblib")
        )
        test_idxs = enc_results["test_idxs"]

        raws = load_raws(
            [sub_id], enc.data_root, use_high_gamma=True,
            per_subject_electrodes={sub_id: sub_sig_elecs},
        )
        raw = raws[0]
        epochs, data = build_word_epochs(raw, df_word)
        X_arr, Y_sub = apply_offset(aligned_embeddings[epochs.selection], data, enc.offset)
        del raws, raw, epochs, data
        gc.collect()

        # Y_sub shape: (n_words, n_sig_elecs * n_lags)
        print(f"  X: {X_arr.shape}, Y: {Y_sub.shape}")
        _run_srrr_for_unit(
            svd_dir=svd_sub_dir,
            srrr_unit_dir=srrr_sub_dir,
            X_raw=X_arr,
            Y_full=Y_sub.astype(np.float32),
            test_idxs=test_idxs,
            config=config,
        )


# ---------------------------------------------------------------------------
# Branch: per-region SRRR
# ---------------------------------------------------------------------------

def _run_per_region_srrr(
    config, enc, output_dir, srrr_dir, subject_ids,
    sig_electrodes, n_folds, aligned_embeddings, df_word,
):
    """SRRR using per-region SVD bases."""
    enc_results = load_encoding_results(
        os.path.join(output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib")
    )
    ref_test_idxs = enc_results["test_idxs"]

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

        # Intersect region electrodes with significant electrodes
        per_subject_region_elecs = {
            sub_id: [e for e in elecs if e in set(sig_electrodes.get(sub_id, []))]
            for sub_id, elecs in sub_elec_map.items()
        }
        per_subject_region_elecs = {s: e for s, e in per_subject_region_elecs.items() if e}

        if not per_subject_region_elecs:
            print(f"  No significant electrodes in region; skipping.")
            continue

        # Load all subjects in region at once
        subs_in_region = list(per_subject_region_elecs.keys())
        raws = load_raws(
            subs_in_region, enc.data_root, use_high_gamma=True,
            per_subject_electrodes=per_subject_region_elecs,
        )
        Y_cols = []
        X_ref = None
        for raw, sub_id in zip(raws, subs_in_region):
            epochs, data = build_word_epochs(raw, df_word)
            X_arr, Y_sub = apply_offset(aligned_embeddings[epochs.selection], data, enc.offset)
            if X_ref is None:
                X_ref = X_arr
            Y_cols.append(Y_sub)  # (n_words, n_elecs * n_lags)
        del raws
        gc.collect()

        Y_full = np.concatenate(Y_cols, axis=1).astype(np.float32)
        print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

        _run_srrr_for_unit(
            svd_dir=svd_region_dir,
            srrr_unit_dir=srrr_region_dir,
            X_raw=X_ref,
            Y_full=Y_full,
            test_idxs=ref_test_idxs,
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

    # Load all_subject_sig.csv — one row per significant (subject, electrode)
    sig_csv = os.path.join(output_dir, "all_subject_sig.csv")
    if not os.path.exists(sig_csv):
        raise FileNotFoundError(
            f"all_subject_sig.csv not found at {sig_csv}. "
            "Run run_significance.py first."
        )
    sig_df = pd.read_csv(sig_csv)
    print(f"Loaded significance CSV: {len(sig_df):,} significant electrodes")

    # Build subject → significant electrode list mapping
    sig_electrodes = {
        sub_id: sig_df[sig_df["subject"] == sub_id]["elec"].tolist()
        for sub_id in subject_ids
    }
    sig_electrodes = {s: e for s, e in sig_electrodes.items() if e}

    if not sig_electrodes:
        raise ValueError(
            f"No significant electrodes found for subjects {subject_ids} in {sig_csv}."
        )

    # Determine n_folds from first subject's encoding results
    first_sub = next(iter(sig_electrodes))
    enc_results = load_encoding_results(
        os.path.join(output_dir, "subjects", f"sub_{first_sub}", "encoding_results.joblib")
    )
    n_folds = len(enc_results["test_idxs"])
    print(f"n_folds: {n_folds}")

    # Load embeddings
    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    branch_kwargs = dict(
        config=config, enc=enc, output_dir=output_dir, srrr_dir=srrr_dir,
        subject_ids=subject_ids,
        sig_electrodes=sig_electrodes,
        n_folds=n_folds,
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
