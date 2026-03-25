"""
run_rrr.py — Raw-Input Reduced Rank Regression baseline.

Performs RRR directly from raw embeddings to neural targets — no SVD projection.
Sweeps over a configured rank grid to find the minimal rank that captures all
variance explained by an unconstrained ridge baseline.

Three modes controlled by RRRConfig.mode (mirrors SVDConfig.mode):

  cross_subject (default):
    Builds global Y across all subjects (significant electrodes).
    Writes to {output_dir}/rrr/.

  per_subject:
    Runs RRR independently per subject.
    Writes to {output_dir}/per_subject_rrr/subjects/sub_{id}/.

  per_region:
    Runs RRR independently per atlas region.
    Writes to {output_dir}/per_region_rrr/regions/{region}/.

Output layout per unit (subject or region):
    {rrr_unit_dir}/
        grid_results.csv
        summary.csv
        fold_{f}/
            A.npy, B.npy          (full-rank factor matrices, shared across k)
        k{k}/fold_{f}/
            metrics.npz
"""

import gc
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
    build_global_sig_mask,
    load_embeddings,
    build_word_epochs,
    apply_offset,
    build_XY_from_subjects,
    load_raws,
)
from utils.analysis import build_sig_mask_flat_for_elecs
from utils.data_loaders import load_sig_electrodes
from utils.atlas import build_electrode_region_map
from utils.config import RRRConfig, SVDMode, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import (
    load_encoding_results,
    get_ridge_model_constructor,
    build_enc_subjects_info,
    build_per_fold_baselines,
)
from utils.sae import solve_regression
from utils.srrr import (
    compute_cca_similarity,
    compute_procrustes_similarity,
    compute_r2,
    fit_dense_rrr,
    predict_from_factors,
    run_sufficiency_test,
    save_rrr_factors,
    save_rrr_result,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_prediction_metrics(
    Y_hat_min: np.ndarray,
    Y_hat_min_train: np.ndarray,
    Y_train: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_full: np.ndarray,
    config: RRRConfig,
) -> tuple[dict, float | None, float | None]:
    """Run sufficiency test and optional alignment metrics for a minimal model.

    Returns:
        (suf, cca_sim, proc_sim) where suf is the run_sufficiency_test result dict.
    """
    suf = run_sufficiency_test(
        X_train,
        Y_train - Y_hat_min_train,
        X_test,
        Y_test,
        Y_hat_full=Y_hat_full,
        Y_hat_min=Y_hat_min,
        null_r2=None,
        model_constructor_fn=get_ridge_model_constructor,
    )
    cca_sim = (
        compute_cca_similarity(Y_hat_full, Y_hat_min) if config.compute_cca else None
    )
    proc_sim = (
        compute_procrustes_similarity(Y_hat_full, Y_hat_min)
        if config.compute_procrustes
        else None
    )
    return suf, cca_sim, proc_sim


def _add_sig_nonsig_corrs(
    metrics: dict,
    suf: dict,
    sig_mask_flat: np.ndarray,
    pairs: list[tuple[str, str]],
) -> None:
    """Add sig/nonsig split correlations to metrics dict in-place.

    Args:
        metrics: Dict to update.
        suf: Sufficiency test result dict containing per-target arrays.
        sig_mask_flat: Boolean mask over targets (True = significant).
        pairs: List of (metric_col_name, suf_per_target_key) tuples.
    """
    for col_name, suf_key in pairs:
        arr = suf[suf_key]
        metrics[f"{col_name}_sig"] = (
            float(arr[sig_mask_flat].mean()) if sig_mask_flat.any() else float("nan")
        )
        metrics[f"{col_name}_nonsig"] = (
            float(arr[~sig_mask_flat].mean())
            if (~sig_mask_flat).any()
            else float("nan")
        )


# ---------------------------------------------------------------------------
# Core per-unit loop
# ---------------------------------------------------------------------------


def _run_rrr_for_unit(
    rrr_unit_dir: str,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    config: RRRConfig,
    sig_mask_flat: np.ndarray | None = None,
) -> pd.DataFrame:
    """Core RRR loop for one analysis unit (cross-subject, per-subject, or per-region).

    Args:
        rrr_unit_dir: Root output directory for this unit.
        X_raw: (n_words, n_embed) raw embedding matrix.
        test_idxs: List of test index arrays, one per fold.
        per_fold_baselines: Output of build_per_fold_baselines — pre-computed scaled Y
            arrays and baseline predictions from the saved encoding model.
        config: RRRConfig.

    Returns:
        DataFrame of per-(fold, k) metrics for grid_results.csv.
    """
    os.makedirs(rrr_unit_dir, exist_ok=True)

    n_folds = len(test_idxs)
    n_embed = X_raw.shape[1]
    selected_idx = np.arange(n_embed)

    all_rows = []

    for fold_f in range(n_folds):
        fold_start = time.time()
        print(f"\n[Fold {fold_f + 1}/{n_folds}]")

        baseline = per_fold_baselines[fold_f]
        train_idx = baseline["train_idx"]
        test_idx = baseline["test_idx"]
        X_train = X_raw[train_idx]
        X_test = X_raw[test_idx]
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]
        Y_hat_full = baseline["Y_hat_full_test"]
        Y_hat_ols_train = baseline["Y_hat_full_train"]

        # Fit RRR once at max rank — the SVD is identical for all k values;
        # only the truncation point differs.
        max_k = max(config.rank_grid)
        rrr_result = fit_dense_rrr(
            X_train,
            Y_train,
            k=max_k,
            selected_idx=selected_idx,
            precomputed_Y_hat_ols=Y_hat_ols_train,
            precomputed_beta_full=baseline["beta_full_raw"],
        )
        A_full = rrr_result["A_dense"]
        B_full = rrr_result["B_dense"]
        save_rrr_factors(rrr_unit_dir, fold=fold_f, A=A_full, B=B_full)

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  Running {len(config.rank_grid)} rank(s): {config.rank_grid}")
        for k in tqdm(
            config.rank_grid, desc=f"  fold {fold_f}", leave=True, unit="rank"
        ):
            A = A_full[:, :k]
            B = B_full[:, :k]

            # Effective rank of A (may be < k if A is rank-deficient)
            sv = np.linalg.svd(A, compute_uv=False)
            effective_rank = (
                int(np.sum(sv > 1e-6 * sv[0])) if len(sv) > 0 and sv[0] > 0 else 0
            )

            Y_hat_rrr = Y_hat_full @ B @ B.T
            Y_hat_rrr_train = Y_hat_ols_train @ B @ B.T

            suf, cca_sim, proc_sim = _run_prediction_metrics(
                Y_hat_rrr, Y_hat_rrr_train, Y_train, X_train, X_test, Y_test,
                Y_hat_full, config,
            )

            # Non-reduced-rank ridge from k-dim subspace
            X_proj_train = X_train @ A  # (n_train, k)
            X_proj_test = X_test @ A  # (n_test, k)
            nr_W, nr_b = solve_regression(
                torch.as_tensor(X_proj_train, dtype=torch.float32, device=device),
                torch.as_tensor(Y_train, dtype=torch.float32, device=device),
                "ridge",
                device,
                standardize=False,
                alphas=np.logspace(-2, 10, 13),
            )
            X_proj_train_t = torch.as_tensor(
                X_proj_train, dtype=torch.float32, device=device
            )
            X_proj_test_t = torch.as_tensor(
                X_proj_test, dtype=torch.float32, device=device
            )
            Y_hat_nr = (X_proj_test_t @ nr_W + nr_b).cpu().numpy()
            Y_hat_nr_train = (X_proj_train_t @ nr_W + nr_b).cpu().numpy()

            nr_suf, nr_cca_sim, nr_proc_sim = _run_prediction_metrics(
                Y_hat_nr, Y_hat_nr_train, Y_train, X_train, X_test, Y_test,
                Y_hat_full, config,
            )

            metrics = {
                "full_r2": suf["full_r2"],
                "minimal_r2": suf["minimal_r2"],
                "recovery_r2": suf["recovery_r2"],
                "pred_r2": suf["pred_r2"],
                "full_corr": suf["full_corr"],
                "minimal_corr": suf["minimal_corr"],
                "recovery_corr": suf["recovery_corr"],
                "effective_rank": effective_rank,
                "fold": fold_f,
                "k": k,
            }
            if sig_mask_flat is not None:
                _add_sig_nonsig_corrs(metrics, suf, sig_mask_flat, [
                    ("full_corr", "full_corr_per_target"),
                    ("minimal_corr", "minimal_corr_per_target"),
                    ("recovery_corr", "recovery_corr_per_target"),
                ])
            if cca_sim is not None:
                metrics["cca_similarity"] = cca_sim
            if proc_sim is not None:
                metrics["procrustes_similarity"] = proc_sim

            metrics.update({
                "nr_r2": nr_suf["minimal_r2"],
                "nr_corr": nr_suf["minimal_corr"],
                "nr_recovery_r2": nr_suf["recovery_r2"],
                "nr_recovery_corr": nr_suf["recovery_corr"],
                "nr_pred_r2": nr_suf["pred_r2"],
                "nr_corr_per_target": nr_suf["minimal_corr_per_target"],
            })
            if sig_mask_flat is not None:
                _add_sig_nonsig_corrs(metrics, nr_suf, sig_mask_flat, [
                    ("nr_corr", "minimal_corr_per_target"),
                    ("nr_recovery_corr", "recovery_corr_per_target"),
                ])
            if nr_cca_sim is not None:
                metrics["nr_cca_similarity"] = nr_cca_sim
            if nr_proc_sim is not None:
                metrics["nr_procrustes_similarity"] = nr_proc_sim
            save_rrr_result(rrr_unit_dir, k=k, fold=fold_f, metrics=metrics)
            all_rows.append(metrics)

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(rrr_unit_dir, "grid_results.csv"), index=False)

    # Summary: average over folds (only include columns that were computed)
    agg_cols = [
        "full_r2",
        "minimal_r2",
        "recovery_r2",
        "pred_r2",
        "full_corr",
        "minimal_corr",
        "recovery_corr",
        "effective_rank",
        "nr_r2",
        "nr_corr",
        "nr_recovery_r2",
        "nr_recovery_corr",
        "nr_pred_r2",
    ]
    if sig_mask_flat is not None:
        agg_cols.extend(
            [
                "full_corr_sig",
                "full_corr_nonsig",
                "minimal_corr_sig",
                "minimal_corr_nonsig",
                "recovery_corr_sig",
                "recovery_corr_nonsig",
                "nr_corr_sig",
                "nr_corr_nonsig",
                "nr_recovery_corr_sig",
                "nr_recovery_corr_nonsig",
            ]
        )
    if config.compute_cca:
        agg_cols.extend(["cca_similarity", "nr_cca_similarity"])
    if config.compute_procrustes:
        agg_cols.extend(["procrustes_similarity", "nr_procrustes_similarity"])
    summary = df.groupby("k")[agg_cols].mean().reset_index()
    summary.to_csv(os.path.join(rrr_unit_dir, "summary.csv"), index=False)

    from utils.plotting import plot_rrr_rank_results

    plot_rrr_rank_results(
        summary,
        grid_df=df,
        title=f"RRR: perf vs rank ({os.path.basename(rrr_unit_dir)})",
        save_path=os.path.join(rrr_unit_dir, "rank_analysis.png"),
    )

    # Non-reduced-rank plot: rename nr_ columns to standard names and reuse plotter
    nr_col_map = {
        "nr_r2": "minimal_r2",
        "nr_corr": "minimal_corr",
        "nr_recovery_r2": "recovery_r2",
        "nr_recovery_corr": "recovery_corr",
        "nr_pred_r2": "pred_r2",
        "nr_corr_sig": "minimal_corr_sig",
        "nr_corr_nonsig": "minimal_corr_nonsig",
        "nr_recovery_corr_sig": "recovery_corr_sig",
        "nr_recovery_corr_nonsig": "recovery_corr_nonsig",
        "nr_cca_similarity": "cca_similarity",
        "nr_procrustes_similarity": "procrustes_similarity",
    }
    # Drop original reduced-rank columns, then rename nr_ columns into their place
    drop_cols = [v for v in nr_col_map.values() if v in df.columns]
    nr_grid = df.drop(columns=drop_cols).rename(columns=nr_col_map)
    nr_summary = summary.drop(columns=drop_cols).rename(columns=nr_col_map)
    plot_rrr_rank_results(
        nr_summary,
        grid_df=nr_grid,
        title=f"RRR (non-reduced ridge): perf vs rank ({os.path.basename(rrr_unit_dir)})",
        save_path=os.path.join(rrr_unit_dir, "rank_analysis_nr.png"),
    )

    return df


# ---------------------------------------------------------------------------
# Significance mask helper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Branch: cross-subject RRR
# ---------------------------------------------------------------------------


def _run_cross_subject_rrr(
    config,
    enc,
    output_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """RRR on raw embeddings using all subjects' significant electrodes."""
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib"
        )
    )
    ref_test_idxs = enc_results["test_idxs"]

    subs_with_sig = [
        s for s in subject_ids if s in sig_electrodes and sig_electrodes[s]
    ]
    if not subs_with_sig:
        print("  No significant electrodes found; skipping cross-subject RRR.")
        return

    X_ref, Y_full = build_XY_from_subjects(
        subject_ids,
        enc,
        sig_electrodes,
        aligned_embeddings,
        df_word,
    )
    print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

    enc_subjects_info = build_enc_subjects_info(
        output_dir, subs_with_sig, sig_electrodes
    )
    per_fold_baselines = build_per_fold_baselines(
        X_ref, Y_full, ref_test_idxs, enc_subjects_info
    )

    # Build significance mask for sig/non-sig correlation split
    sig_mask_flat = build_sig_mask_flat_for_elecs(
        output_dir,
        {s: sig_electrodes[s] for s in subs_with_sig},
        enc_results,
    )

    _run_rrr_for_unit(
        rrr_unit_dir=rrr_dir,
        X_raw=X_ref,
        test_idxs=ref_test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
    )


# ---------------------------------------------------------------------------
# Branch: per-subject RRR
# ---------------------------------------------------------------------------


def _run_per_subject_rrr(
    config,
    enc,
    output_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """RRR on raw embeddings independently per subject."""
    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        rrr_sub_dir = os.path.join(rrr_dir, "subjects", f"sub_{sub_id}")
        sub_sig_elecs = sig_electrodes.get(sub_id, [])
        if not sub_sig_elecs:
            print(f"  No significant electrodes; skipping.")
            continue

        enc_results = load_encoding_results(
            os.path.join(
                output_dir, "subjects", f"sub_{sub_id}", "encoding_results.joblib"
            )
        )
        test_idxs = enc_results["test_idxs"]

        raws = load_raws(
            [sub_id],
            enc.data_root,
            use_high_gamma=True,
            per_subject_electrodes={sub_id: sub_sig_elecs},
        )
        raw = raws[0]
        epochs, data = build_word_epochs(raw, df_word)
        X_arr, Y_sub = apply_offset(
            aligned_embeddings[epochs.selection], data, enc.offset
        )
        del raws, raw, epochs, data
        gc.collect()

        Y_sub = Y_sub.astype(np.float32)
        print(f"  X: {X_arr.shape}, Y: {Y_sub.shape}")

        enc_subjects_info = build_enc_subjects_info(
            output_dir, [sub_id], {sub_id: sub_sig_elecs}
        )
        per_fold_baselines = build_per_fold_baselines(
            X_arr, Y_sub, test_idxs, enc_subjects_info
        )

        sig_mask_flat = build_sig_mask_flat_for_elecs(
            output_dir, {sub_id: sub_sig_elecs}, enc_results
        )

        _run_rrr_for_unit(
            rrr_unit_dir=rrr_sub_dir,
            X_raw=X_arr,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
        )


# ---------------------------------------------------------------------------
# Branch: per-region RRR
# ---------------------------------------------------------------------------


def _run_per_region_rrr(
    config,
    enc,
    output_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    df_word,
):
    """RRR on raw embeddings independently per atlas region."""
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib"
        )
    )
    ref_test_idxs = enc_results["test_idxs"]

    print("Building electrode region map...")
    electrode_region_map = build_electrode_region_map(
        subject_ids,
        enc.data_root,
        getattr(config, "atlas_region_groups", None),
        getattr(config, "atlas_path", None),
    )
    print(f"  Regions found: {list(electrode_region_map.keys())}")

    for region_name, sub_elec_map in electrode_region_map.items():
        print(f"\n{'='*60}\nRegion: {region_name}")

        rrr_region_dir = os.path.join(rrr_dir, "regions", region_name)

        # Intersect region electrodes with significant electrodes
        per_subject_region_elecs = {
            sub_id: [e for e in elecs if e in set(sig_electrodes.get(sub_id, []))]
            for sub_id, elecs in sub_elec_map.items()
        }
        per_subject_region_elecs = {
            s: e for s, e in per_subject_region_elecs.items() if e
        }

        if not per_subject_region_elecs:
            print(f"  No significant electrodes in region; skipping.")
            continue

        subs_in_region = list(per_subject_region_elecs.keys())
        X_ref, Y_full = build_XY_from_subjects(
            subs_in_region,
            enc,
            per_subject_region_elecs,
            aligned_embeddings,
            df_word,
        )
        print(f"  X: {X_ref.shape}, Y: {Y_full.shape}")

        enc_subjects_info = build_enc_subjects_info(
            output_dir, subs_in_region, per_subject_region_elecs
        )
        per_fold_baselines = build_per_fold_baselines(
            X_ref, Y_full, ref_test_idxs, enc_subjects_info
        )

        # Build sig mask for this region's electrodes
        sig_mask_flat = build_sig_mask_flat_for_elecs(
            output_dir, per_subject_region_elecs, enc_results
        )

        _run_rrr_for_unit(
            rrr_unit_dir=rrr_region_dir,
            X_raw=X_ref,
            test_idxs=ref_test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args, overrides = parse_args("RRR: raw-input reduced rank regression baseline")
    config: RRRConfig = load_config(args.config, overrides, RRRConfig)
    enc = config.encoding

    import torch
    from himalaya.backend import set_backend

    if torch.cuda.is_available():
        set_backend("torch_cuda")

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)

    # Resolve RRR output directory
    if config.rrr_output_dir:
        rrr_dir = config.rrr_output_dir
    elif config.mode == SVDMode.PER_REGION:
        rrr_dir = os.path.join(output_dir, "per_region_rrr")
    elif config.mode == SVDMode.PER_SUBJECT:
        rrr_dir = os.path.join(output_dir, "per_subject_rrr")
    else:
        rrr_dir = os.path.join(output_dir, "rrr")

    os.makedirs(rrr_dir, exist_ok=True)
    print(f"Output dir:  {output_dir}")
    print(f"RRR dir:     {rrr_dir}")
    print(f"Mode:        {config.mode.value}")

    # Load significance CSV and build subject → electrode mapping
    sig_electrodes, _ = load_sig_electrodes(output_dir, subject_ids)

    # Load embeddings
    print("Loading embeddings...")
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    branch_kwargs = dict(
        config=config,
        enc=enc,
        output_dir=output_dir,
        rrr_dir=rrr_dir,
        subject_ids=subject_ids,
        sig_electrodes=sig_electrodes,
        aligned_embeddings=aligned_embeddings,
        df_word=df_word,
    )

    if config.mode == SVDMode.PER_REGION:
        _run_per_region_rrr(**branch_kwargs)
    elif config.mode == SVDMode.PER_SUBJECT:
        _run_per_subject_rrr(**branch_kwargs)
    else:
        _run_cross_subject_rrr(**branch_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
