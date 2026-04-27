"""
run_rrr.py — Raw-Input Reduced Rank Regression + optional PCA baseline.

Performs RRR directly from raw embeddings to neural targets — no SVD projection.
Sweeps over a configured rank grid. Optionally also runs PCA subspace analysis
(when RRRConfig.run_pca_baseline=True) using the same rank grid and data splits.

Four modes controlled by RRRConfig.mode:

  cross_subject (default):
    Builds global Y across all subjects (significant electrodes).
    Writes to {output_dir}/rrr/.

  per_subject:
    Runs independently per subject.
    Writes to {output_dir}/per_subject_rrr/subjects/sub_{id}/.

  per_region:
    Runs independently per atlas region.
    Writes to {output_dir}/per_region_rrr/regions/{region}/.

  leave_one_subject_out:
    For each held-out subject, trains RRR on the remaining n-1 subjects, then
    evaluates transfer of the learned A matrix to the held-out subject alongside
    a within-subject baseline.
    Writes to {output_dir}/loso_rrr/held_out_sub_{id}/{train_subjects,transfer,baseline}/.

Output layout per unit:
    {unit_dir}/
        grid_results.csv
        summary.csv
        rank_analysis.png
        fold_{f}/
            A.npy, B.npy              (RRR only; full-rank factor matrices)
            full_model_metrics.npz    (full ridge baseline, saved once per fold)
        k{k}/fold_{f}/
            metrics.npz               (per-rank metrics including *_per_target arrays)

Metrics (per k, per fold in grid_results.csv):
    RRR model  : rrr_corr, rrr_r2
    NR ridge   : nr_corr, nr_r2
    Complement : complement_corr, complement_r2
    Joint (NR + complement)  : nr_complement_joint_corr, nr_complement_joint_r2
    Joint (NR + arbitrary)   : nr_arbitrary_joint_corr, nr_arbitrary_joint_r2  [if arbitrary]
    Effective rank: effective_rank

Full model (saved once per fold, not duplicated across k):
    full_corr, full_r2, full_corr_per_target  → fold_{f}/full_model_metrics.npz

Per-target arrays (only in metrics.npz, not in grid_results.csv):
    rrr_corr_per_target, nr_corr_per_target, complement_corr_per_target,
    nr_complement_joint_corr_per_target, nr_arbitrary_joint_corr_per_target
"""

import gc
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import (
    load_embeddings,
    build_XY_from_subjects,
)
from src.utils.data_loaders import load_sig_electrodes
from src.utils.atlas import build_electrode_region_map
from src.core.config import RRRConfig, ScriptMode, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import parse_args, load_config
from src.utils.embeddings import get_glove_embeddings
from src.utils.encoding import (
    load_encoding_results,
    build_per_fold_baselines,
    get_and_offset_arbitrary_embeddings,
)
from src.utils.srrr import (
    fit_dense_rrr,
    save_rrr_factors,
    save_rrr_result,
)
from src.utils.analysis import build_sig_mask_flat_for_elecs
from src.utils.rrr_utils import (
    add_sig_nonsig_corrs,
    make_sig_split_df,
)
from src.utils.subspace_utils import (
    evaluate_input_space,
    evaluate_complement,
    evaluate_arbitrary,
    compute_stacked_joint_metrics,
)
from himalaya.scoring import correlation_score, r2_score


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save_full_model_metrics_once(
    unit_dir: str,
    fold_f: int,
    Y_test: np.ndarray,
    Y_hat_full: np.ndarray,
) -> None:
    """Save full-model metrics to fold_{f}/full_model_metrics.npz (once, not per-k)."""
    fold_dir = os.path.join(unit_dir, f"fold_{fold_f}")
    os.makedirs(fold_dir, exist_ok=True)
    path = os.path.join(fold_dir, "full_model_metrics.npz")
    if os.path.exists(path):
        return
    corr_arr = correlation_score(Y_test, Y_hat_full)
    r2_arr = r2_score(Y_test, Y_hat_full)
    np.savez(
        path,
        full_corr=float(corr_arr.mean()),
        full_r2=float(r2_arr.mean()),
        full_corr_per_target=corr_arr,
    )


def _resolve_pca_sibling_dir(rrr_unit_dir: str) -> str:
    """Return the PCA output dir that mirrors a given RRR unit dir."""
    for rrr_part, pca_part in [
        ("per_subject_rrr", "per_subject_pca"),
        ("per_region_rrr", "per_region_pca"),
        ("/rrr", "/pca"),
    ]:
        if rrr_part in rrr_unit_dir:
            return rrr_unit_dir.replace(rrr_part, pca_part, 1)
    return rrr_unit_dir + "_pca"


def _strip_per_target_and_add_sig_nonsig(
    metrics: dict,
    sig_mask_flat: "np.ndarray | None",
) -> None:
    """Add sig/nonsig split means for *_per_target arrays, then delete them in-place."""
    per_target_keys = [k for k in list(metrics) if k.endswith("_per_target")]
    if sig_mask_flat is not None:
        for pt_key in per_target_keys:
            base = pt_key[: -len("_per_target")]
            arr = metrics[pt_key]
            metrics[f"{base}_sig"] = (
                float(arr[sig_mask_flat].mean()) if sig_mask_flat.any() else float("nan")
            )
            metrics[f"{base}_nonsig"] = (
                float(arr[~sig_mask_flat].mean())
                if (~sig_mask_flat).any()
                else float("nan")
            )
    for pt_key in per_target_keys:
        del metrics[pt_key]


def _run_common_subspace_fold_k(
    A_full: np.ndarray,
    k: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    X_arbitrary: "np.ndarray | None",
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sig_mask_flat: "np.ndarray | None",
    fold_f: int,
    alphas: "np.ndarray | None",
    device: str,
    unit_dir: str,
    oof_spill_dir: str,
    primary_label: str,
) -> dict:
    """Shared evaluation for one (fold, k): NR + complement + optional arbitrary.

    Called by both _run_rrr_for_unit (primary_label='nr') and _run_pca_for_unit
    (primary_label='pca') after the reduction matrix A_full is known.

    Saves per-rank metrics (including *_per_target arrays) to
    {unit_dir}/k{k}/fold_{fold_f}/metrics.npz.

    Spills OOF predictions to {oof_spill_dir} for post-loop joint computation.

    Returns a metrics dict with per-target arrays stripped (for grid_results.csv).
    """
    A_k = A_full[:, :k]

    # NR: project X into k-dim subspace, train ridge
    nr_alphas = (
        np.logspace(-2, 10, 13) if k < 50 else np.logspace(1, 10, 13)
    )
    X_proj_train = X_train @ A_k
    X_proj_test = X_test @ A_k
    nr_result = evaluate_input_space(
        X_proj_train,
        X_proj_test,
        Y_train,
        Y_test,
        subspace_name=primary_label,
        alphas=nr_alphas,
        device=device,
    )

    # Complement evaluation
    comp_alphas = (
        np.logspace(1, 10, 13) if k < 50 else np.logspace(-2, 10, 13)
    )
    complement_result = evaluate_complement(
        A_full,
        k,
        X_train,
        X_test,
        Y_train,
        Y_test,
        alphas=comp_alphas,
        device=device,
    )

    # Arbitrary evaluation (null baseline)
    arbitrary_result = None
    if X_arbitrary is not None:
        arbitrary_result = evaluate_arbitrary(
            X_arbitrary[train_idx],
            X_arbitrary[test_idx],
            Y_train,
            Y_test,
            alphas=alphas,
            device=device,
        )

    # Build metrics dict (with per-target arrays for metrics.npz save)
    metrics = {
        f"{primary_label}_corr": nr_result[f"{primary_label}_corr"],
        f"{primary_label}_r2": nr_result[f"{primary_label}_r2"],
        f"{primary_label}_corr_per_target": nr_result[f"{primary_label}_corr_per_target"],
        "complement_corr": complement_result["complement_corr"],
        "complement_r2": complement_result["complement_r2"],
        "complement_corr_per_target": complement_result["complement_corr_per_target"],
    }
    if arbitrary_result is not None:
        metrics["arbitrary_corr"] = arbitrary_result["arbitrary_corr"]
        metrics["arbitrary_r2"] = arbitrary_result["arbitrary_r2"]
        metrics["arbitrary_corr_per_target"] = arbitrary_result["arbitrary_corr_per_target"]

    # Save to metrics.npz (includes per-target)
    save_rrr_result(unit_dir, k=k, fold=fold_f, metrics=metrics)

    # Spill OOF predictions to disk for post-loop joint computation
    np.save(
        os.path.join(oof_spill_dir, f"{k}_{fold_f}_{primary_label}.npy"),
        nr_result[f"{primary_label}_Y_hat"],
    )
    np.save(
        os.path.join(oof_spill_dir, f"{k}_{fold_f}_complement.npy"),
        complement_result["complement_Y_hat"],
    )
    np.save(os.path.join(oof_spill_dir, f"{k}_{fold_f}_Y_test.npy"), Y_test)
    if arbitrary_result is not None:
        np.save(
            os.path.join(oof_spill_dir, f"{k}_{fold_f}_arbitrary.npy"),
            arbitrary_result["arbitrary_Y_hat"],
        )

    # Add sig/nonsig splits and strip per-target arrays (for grid_results.csv row)
    _strip_per_target_and_add_sig_nonsig(metrics, sig_mask_flat)

    return metrics


def _compute_and_save_joint_metrics(
    oof_spill_dir: str,
    rank_grid: list,
    n_folds: int,
    unit_dir: str,
    all_rows: list,
    primary_label: str,
    sig_mask_flat: "np.ndarray | None",
) -> None:
    """Post-loop: compute stacked joint metrics, update rows and metrics.npz files.

    Computes joint metrics for:
        - {primary_label} + complement
        - {primary_label} + arbitrary  (if arbitrary OOF files exist)

    Updates all_rows in-place with joint scalar metrics.
    Appends joint *_per_target arrays to each k/fold metrics.npz.
    Cleans up the OOF spill directory when done.
    """
    import shutil

    for k in rank_grid:
        all_primary, all_complement, all_y_test = [], [], []
        all_arbitrary = []
        has_arbitrary = True

        for fold_f in range(n_folds):
            all_primary.append(
                np.load(os.path.join(oof_spill_dir, f"{k}_{fold_f}_{primary_label}.npy"))
            )
            all_complement.append(
                np.load(os.path.join(oof_spill_dir, f"{k}_{fold_f}_complement.npy"))
            )
            all_y_test.append(
                np.load(os.path.join(oof_spill_dir, f"{k}_{fold_f}_Y_test.npy"))
            )
            arb_path = os.path.join(oof_spill_dir, f"{k}_{fold_f}_arbitrary.npy")
            if has_arbitrary and os.path.exists(arb_path):
                all_arbitrary.append(np.load(arb_path))
            else:
                has_arbitrary = False

        # Joint: primary + complement
        jm_comp = compute_stacked_joint_metrics(
            all_primary, all_complement, all_y_test, n_splits=n_folds
        )
        comp_joint_key = f"{primary_label}_complement_joint"

        # Joint: primary + arbitrary
        jm_arb = None
        arb_joint_key = f"{primary_label}_arbitrary_joint"
        if has_arbitrary and all_arbitrary:
            jm_arb = compute_stacked_joint_metrics(
                all_primary, all_arbitrary, all_y_test, n_splits=n_folds
            )

        # Update in-memory rows
        for row in all_rows:
            if row["k"] != k:
                continue
            row[f"{comp_joint_key}_corr"] = float(jm_comp["corr_per_target"].mean())
            row[f"{comp_joint_key}_r2"] = float(jm_comp["r2_per_target"].mean())
            if sig_mask_flat is not None:
                arr = jm_comp["corr_per_target"]
                row[f"{comp_joint_key}_corr_sig"] = (
                    float(arr[sig_mask_flat].mean()) if sig_mask_flat.any() else float("nan")
                )
                row[f"{comp_joint_key}_corr_nonsig"] = (
                    float(arr[~sig_mask_flat].mean())
                    if (~sig_mask_flat).any()
                    else float("nan")
                )
            if jm_arb is not None:
                row[f"{arb_joint_key}_corr"] = float(jm_arb["corr_per_target"].mean())
                row[f"{arb_joint_key}_r2"] = float(jm_arb["r2_per_target"].mean())
                if sig_mask_flat is not None:
                    arr = jm_arb["corr_per_target"]
                    row[f"{arb_joint_key}_corr_sig"] = (
                        float(arr[sig_mask_flat].mean()) if sig_mask_flat.any() else float("nan")
                    )
                    row[f"{arb_joint_key}_corr_nonsig"] = (
                        float(arr[~sig_mask_flat].mean())
                        if (~sig_mask_flat).any()
                        else float("nan")
                    )

        # Append per-target arrays to each fold's metrics.npz
        for fold_f in range(n_folds):
            metrics_path = os.path.join(unit_dir, f"k{k}", f"fold_{fold_f}", "metrics.npz")
            if not os.path.exists(metrics_path):
                continue
            existing = dict(np.load(metrics_path, allow_pickle=True))
            existing[f"{comp_joint_key}_corr_per_target"] = jm_comp["corr_per_target"]
            existing[f"{comp_joint_key}_r2_per_target"] = jm_comp["r2_per_target"]
            if jm_arb is not None:
                existing[f"{arb_joint_key}_corr_per_target"] = jm_arb["corr_per_target"]
                existing[f"{arb_joint_key}_r2_per_target"] = jm_arb["r2_per_target"]
            np.savez(metrics_path, **existing)

    shutil.rmtree(oof_spill_dir, ignore_errors=True)


def _build_results(
    df: pd.DataFrame,
    sig_mask_flat: "np.ndarray | None",
    unit_dir: str,
    primary_label: str,
) -> None:
    """Build summary CSV and rank analysis plots from grid results DataFrame."""
    # Core agg columns (filter to what's actually present)
    candidate_cols = [
        "full_corr",
        "full_r2",
        f"{primary_label}_corr",
        f"{primary_label}_r2",
        "complement_corr",
        "complement_r2",
        f"{primary_label}_complement_joint_corr",
        f"{primary_label}_complement_joint_r2",
        f"{primary_label}_arbitrary_joint_corr",
        f"{primary_label}_arbitrary_joint_r2",
        "arbitrary_corr",
        "arbitrary_r2",
        "effective_rank",
        # RRR-specific (only in RRR runs)
        "rrr_corr",
        "rrr_r2",
    ]
    if sig_mask_flat is not None:
        sig_candidates = []
        for col in candidate_cols:
            if col.endswith("_corr"):
                sig_candidates.extend([f"{col}_sig", f"{col}_nonsig"])
        candidate_cols.extend(sig_candidates)

    agg_cols = [c for c in candidate_cols if c in df.columns]
    summary = df.groupby("k")[agg_cols].mean().reset_index()
    summary.to_csv(os.path.join(unit_dir, "summary.csv"), index=False)

    from src.utils.plotting import plot_rrr_rank_results

    title_base = f"{primary_label.upper()}: perf vs rank ({os.path.basename(unit_dir)})"
    plot_rrr_rank_results(
        summary,
        grid_df=df,
        title=title_base,
        save_path=os.path.join(unit_dir, "rank_analysis.png"),
        primary_label=primary_label,
    )
    if sig_mask_flat is not None:
        for split in ("sig", "nonsig"):
            plot_rrr_rank_results(
                make_sig_split_df(summary, split),
                grid_df=make_sig_split_df(df, split),
                title=f"{title_base} ({split} targets)",
                save_path=os.path.join(unit_dir, f"rank_analysis_{split}.png"),
                primary_label=primary_label,
            )


# ---------------------------------------------------------------------------
# Core per-unit loops
# ---------------------------------------------------------------------------


def _run_rrr_for_unit(
    rrr_unit_dir: str,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    config: RRRConfig,
    sig_mask_flat: "np.ndarray | None" = None,
    X_arbitrary: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """Core RRR loop for one analysis unit (cross-subject, per-subject, or per-region).

    Fits RRR once at max rank per fold, sweeps the rank grid, and evaluates:
        - Raw RRR predictions (via B matrix projection)
        - NR ridge from k-dim RRR subspace (via A matrix projection)
        - Complement of the RRR subspace
        - Arbitrary embeddings baseline
        - Joint models (NR + complement, NR + arbitrary)

    Args:
        rrr_unit_dir: Root output directory for this unit.
        X_raw: (n_words, n_embed) raw embedding matrix.
        test_idxs: List of test index arrays, one per fold.
        per_fold_baselines: Output of build_per_fold_baselines.
        config: RRRConfig.
        sig_mask_flat: Boolean mask over targets (True = significant).
        X_arbitrary: (n_words, n_embed) arbitrary embeddings for null baseline.

    Returns:
        DataFrame of per-(fold, k) metrics for grid_results.csv.
    """
    os.makedirs(rrr_unit_dir, exist_ok=True)

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None

    n_folds = len(test_idxs)
    n_embed = X_raw.shape[1]
    selected_idx = np.arange(n_embed)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_rows = []
    oof_spill_dir = os.path.join(rrr_unit_dir, "_oof_spill")
    os.makedirs(oof_spill_dir, exist_ok=True)

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

        # Save full model metrics once per fold (not per-k)
        _save_full_model_metrics_once(rrr_unit_dir, fold_f, Y_test, Y_hat_full)
        full_corr = float(correlation_score(Y_test, Y_hat_full).mean())
        full_r2 = float(r2_score(Y_test, Y_hat_full).mean())

        # Fit RRR once at max rank; truncate to each k below
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

        print(f"  Running {len(config.rank_grid)} rank(s): {config.rank_grid}")
        for k in tqdm(config.rank_grid, desc=f"  fold {fold_f}", leave=True, unit="rank"):
            A_k = A_full[:, :k]
            B_k = B_full[:, :k]

            # Effective rank of A_k
            sv = np.linalg.svd(A_k, compute_uv=False)
            effective_rank = (
                int(np.sum(sv > 1e-6 * sv[0])) if len(sv) > 0 and sv[0] > 0 else 0
            )

            # Raw RRR predictions (project Y through B)
            Y_hat_rrr = Y_hat_full @ B_k @ B_k.T
            rrr_corr_arr = correlation_score(Y_test, Y_hat_rrr)
            rrr_r2_arr = r2_score(Y_test, Y_hat_rrr)

            # Shared NR + complement + arbitrary evaluation
            common = _run_common_subspace_fold_k(
                A_full,
                k,
                X_train,
                X_test,
                Y_train,
                Y_test,
                X_arbitrary,
                train_idx,
                test_idx,
                sig_mask_flat,
                fold_f,
                alphas,
                device,
                rrr_unit_dir,
                oof_spill_dir,
                primary_label="nr",
            )

            # Add RRR-specific metrics and update metrics.npz with rrr per-target
            rrr_metrics = {
                "rrr_corr": float(rrr_corr_arr.mean()),
                "rrr_r2": float(rrr_r2_arr.mean()),
                "rrr_corr_per_target": rrr_corr_arr,
            }
            # Load existing metrics.npz and append rrr per-target
            metrics_path = os.path.join(
                rrr_unit_dir, f"k{k}", f"fold_{fold_f}", "metrics.npz"
            )
            existing = dict(np.load(metrics_path, allow_pickle=True))
            existing["rrr_corr_per_target"] = rrr_corr_arr
            existing["rrr_r2_per_target"] = rrr_r2_arr
            np.savez(metrics_path, **existing)

            # Build row (no per-target)
            row = {
                "fold": fold_f,
                "k": k,
                "effective_rank": effective_rank,
                "full_corr": full_corr,
                "full_r2": full_r2,
                "rrr_corr": float(rrr_corr_arr.mean()),
                "rrr_r2": float(rrr_r2_arr.mean()),
                **common,
            }
            if sig_mask_flat is not None:
                row["rrr_corr_sig"] = (
                    float(rrr_corr_arr[sig_mask_flat].mean())
                    if sig_mask_flat.any()
                    else float("nan")
                )
                row["rrr_corr_nonsig"] = (
                    float(rrr_corr_arr[~sig_mask_flat].mean())
                    if (~sig_mask_flat).any()
                    else float("nan")
                )
            all_rows.append(row)

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    # Compute stacked OLS joint correlations (requires held-out OOF predictions)
    print("\nComputing stacked OLS joint correlations...")
    _compute_and_save_joint_metrics(
        oof_spill_dir,
        config.rank_grid,
        n_folds,
        rrr_unit_dir,
        all_rows,
        primary_label="nr",
        sig_mask_flat=sig_mask_flat,
    )

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(rrr_unit_dir, "grid_results.csv"), index=False)
    _build_results(df, sig_mask_flat, rrr_unit_dir, primary_label="rrr")

    # Optionally run PCA baseline using the same data splits
    if config.run_pca_baseline:
        pca_unit_dir = _resolve_pca_sibling_dir(rrr_unit_dir)
        print(f"\nRunning PCA baseline → {pca_unit_dir}")
        _run_pca_for_unit(
            pca_unit_dir=pca_unit_dir,
            X_raw=X_raw,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            rank_grid=config.rank_grid,
            sig_mask_flat=sig_mask_flat,
            X_arbitrary=X_arbitrary,
            alphas=alphas,
            device=device,
        )

    return df


def _run_pca_for_unit(
    pca_unit_dir: str,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    rank_grid: list,
    sig_mask_flat: "np.ndarray | None" = None,
    X_arbitrary: "np.ndarray | None" = None,
    alphas: "np.ndarray | None" = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """PCA subspace analysis for one unit — structurally parallel to _run_rrr_for_unit.

    Computes PCA on training embeddings (per fold, proper CV), then evaluates:
        - Ridge from k-dim PCA subspace (primary model)
        - Complement of the PCA subspace
        - Arbitrary embeddings baseline
        - Joint models (PCA + complement, PCA + arbitrary)

    The only difference from _run_rrr_for_unit is how A_full is computed: here it
    comes from SVD of X_train rather than from RRR. All downstream evaluation is
    handled by the same shared helpers.

    Args:
        pca_unit_dir: Root output directory for this unit.
        X_raw: (n_words, n_embed) raw embedding matrix.
        test_idxs: List of test index arrays, one per fold.
        per_fold_baselines: Output of build_per_fold_baselines.
        rank_grid: List of rank values to evaluate.
        sig_mask_flat: Boolean mask over targets (True = significant).
        X_arbitrary: (n_words, n_embed) arbitrary embeddings for null baseline.
        alphas: Ridge alpha grid (passed to arbitrary evaluation); None → solver default.
        device: Torch device string.

    Returns:
        DataFrame of per-(fold, k) metrics for grid_results.csv.
    """
    os.makedirs(pca_unit_dir, exist_ok=True)

    n_folds = len(test_idxs)
    all_rows = []
    oof_spill_dir = os.path.join(pca_unit_dir, "_oof_spill")
    os.makedirs(oof_spill_dir, exist_ok=True)

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

        # Save full model metrics once per fold (not per-k)
        _save_full_model_metrics_once(pca_unit_dir, fold_f, Y_test, Y_hat_full)

        # Compute PCA on training data only (proper CV)
        X_mean = X_train.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_train - X_mean, full_matrices=False)
        A_full = Vt.T  # (n_embed, min(n_train, n_embed))
        max_components = A_full.shape[1]

        valid_ranks = [k for k in rank_grid if k <= max_components]
        if len(valid_ranks) < len(rank_grid):
            skipped = [k for k in rank_grid if k > max_components]
            print(f"  Skipping ranks {skipped} (> {max_components} PCA components)")

        print(f"  Running {len(valid_ranks)} rank(s): {valid_ranks}")
        for k in tqdm(valid_ranks, desc=f"  fold {fold_f}", leave=True, unit="rank"):
            common = _run_common_subspace_fold_k(
                A_full,
                k,
                X_train,
                X_test,
                Y_train,
                Y_test,
                X_arbitrary,
                train_idx,
                test_idx,
                sig_mask_flat,
                fold_f,
                alphas,
                device,
                pca_unit_dir,
                oof_spill_dir,
                primary_label="pca",
            )
            row = {"fold": fold_f, "k": k, **common}
            all_rows.append(row)

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    # Compute stacked OLS joint correlations
    print("\nComputing stacked OLS joint correlations...")
    _compute_and_save_joint_metrics(
        oof_spill_dir,
        rank_grid,
        n_folds,
        pca_unit_dir,
        all_rows,
        primary_label="pca",
        sig_mask_flat=sig_mask_flat,
    )

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(pca_unit_dir, "grid_results.csv"), index=False)
    _build_results(df, sig_mask_flat, pca_unit_dir, primary_label="pca")

    return df


# ---------------------------------------------------------------------------
# Data preparation helper
# ---------------------------------------------------------------------------


def _get_decomposition_inputs(
    subject_ids,
    output_dir,
    sig_electrodes,
    encoding_config,
    embeddings,
    word_df,
    compute_sig_nonsig=False,
    sig_dir=None,
    word_in_glove=None,
):
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib"
        )
    )
    test_idxs = enc_results["test_idxs"]

    subs_with_sig = [
        s for s in subject_ids if s in sig_electrodes and sig_electrodes[s]
    ]
    if not subs_with_sig:
        print("  No significant electrodes found; skipping.")
        return None

    X_raw, Y_full, selection, _ = build_XY_from_subjects(
        subject_ids,
        encoding_config,
        sig_electrodes,
        embeddings,
        valid_rows=word_in_glove,
    )
    print(f"  X: {X_raw.shape}, Y: {Y_full.shape}")

    per_fold_baselines = build_per_fold_baselines(X_raw, Y_full, test_idxs)

    X_arbitrary = get_and_offset_arbitrary_embeddings(
        embeddings,
        selection,
        word_df,
        encoding_config.offset,
        encoding_config.arbitrary_seed,
        valid_rows=word_in_glove,
    )

    sig_mask_flat = None
    if compute_sig_nonsig:
        sig_mask_flat = build_sig_mask_flat_for_elecs(
            output_dir,
            {s: sig_electrodes[s] for s in subs_with_sig},
            enc_results,
            sig_dir=sig_dir,
        )

    return X_raw, X_arbitrary, per_fold_baselines, test_idxs, sig_mask_flat


# ---------------------------------------------------------------------------
# Branch: cross-subject RRR
# ---------------------------------------------------------------------------


def _run_cross_subject_rrr(
    config: RRRConfig,
    enc,
    output_dir,
    sig_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
):
    """RRR on raw embeddings using all subjects' significant electrodes."""
    results = _get_decomposition_inputs(
        subject_ids,
        output_dir,
        sig_electrodes,
        enc,
        aligned_embeddings,
        word_df,
        compute_sig_nonsig=config.compute_sig_nonsig,
        sig_dir=sig_dir,
        word_in_glove=word_in_glove,
    )
    if results is None:
        return
    X_raw, X_arbitrary, per_fold_baselines, test_idxs, sig_mask_flat = results
    _run_rrr_for_unit(
        rrr_unit_dir=rrr_dir,
        X_raw=X_raw,
        test_idxs=test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
        X_arbitrary=X_arbitrary,
    )


# ---------------------------------------------------------------------------
# Branch: per-subject RRR
# ---------------------------------------------------------------------------


def _run_per_subject_rrr(
    config: RRRConfig,
    enc,
    output_dir,
    sig_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
):
    """RRR on raw embeddings independently per subject."""
    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        rrr_sub_dir = os.path.join(rrr_dir, "subjects", f"sub_{sub_id}")
        sub_sig_elecs = sig_electrodes.get(sub_id, [])
        if not sub_sig_elecs:
            print("  No significant electrodes; skipping.")
            continue

        results = _get_decomposition_inputs(
            [sub_id],
            output_dir,
            {sub_id: sub_sig_elecs},
            enc,
            aligned_embeddings,
            word_df,
            compute_sig_nonsig=config.compute_sig_nonsig,
            sig_dir=sig_dir,
            word_in_glove=word_in_glove,
        )
        if results is None:
            continue
        X_raw, X_arbitrary, per_fold_baselines, test_idxs, sig_mask_flat = results
        _run_rrr_for_unit(
            rrr_unit_dir=rrr_sub_dir,
            X_raw=X_raw,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
            X_arbitrary=X_arbitrary,
        )


# ---------------------------------------------------------------------------
# Branch: per-region RRR
# ---------------------------------------------------------------------------


def _run_per_region_rrr(
    config: RRRConfig,
    enc,
    output_dir,
    sig_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
):
    """RRR on raw embeddings independently per atlas region."""
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

        per_subject_region_elecs = {
            sub_id: [e for e in elecs if e in set(sig_electrodes.get(sub_id, []))]
            for sub_id, elecs in sub_elec_map.items()
        }
        per_subject_region_elecs = {s: e for s, e in per_subject_region_elecs.items() if e}

        if not per_subject_region_elecs:
            print("  No significant electrodes in region; skipping.")
            continue

        subs_in_region = list(per_subject_region_elecs.keys())
        results = _get_decomposition_inputs(
            subs_in_region,
            output_dir,
            per_subject_region_elecs,
            enc,
            aligned_embeddings,
            word_df,
            compute_sig_nonsig=config.compute_sig_nonsig,
            sig_dir=sig_dir,
            word_in_glove=word_in_glove,
        )
        if results is None:
            continue
        X_raw, X_arbitrary, per_fold_baselines, test_idxs, sig_mask_flat = results
        _run_rrr_for_unit(
            rrr_unit_dir=rrr_region_dir,
            X_raw=X_raw,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
            X_arbitrary=X_arbitrary,
        )


# ---------------------------------------------------------------------------
# Branch: leave-one-subject-out RRR
# ---------------------------------------------------------------------------


def _run_loso_transfer_for_unit(
    transfer_dir: str,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    train_rrr_dir: str,
    config: "RRRConfig",
    sig_mask_flat: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """Evaluate transfer of the A matrix learned on n-1 subjects to the held-out subject.

    For each fold and rank k, loads the A matrix saved by _run_rrr_for_unit on the
    training subjects, projects X through A_k, trains a fresh ridge on the held-out
    subject's training data, and evaluates on the held-out fold. Also evaluates the
    complement of the transferred subspace and computes stacked joint metrics for
    transfer + complement across folds.

    Args:
        transfer_dir: Output directory for transfer metrics.
        X_raw: (n_words, n_embed) embedding matrix (shared across all subjects).
        test_idxs: List of test index arrays (one per fold); used only for n_folds.
        per_fold_baselines: Fold baselines built from the held-out subject's Y matrix.
        train_rrr_dir: Directory where _run_rrr_for_unit saved A.npy files per fold.
        config: RRRConfig with rank_grid and encoding settings.
        sig_mask_flat: Optional boolean mask for significant targets in held-out subject.

    Returns:
        DataFrame with one row per (fold, k) containing transfer metrics.
    """
    os.makedirs(transfer_dir, exist_ok=True)

    import torch

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_folds = len(test_idxs)
    all_rows = []
    oof_spill_dir = os.path.join(transfer_dir, "_oof_spill")
    os.makedirs(oof_spill_dir, exist_ok=True)

    for fold_f in range(n_folds):
        fold_start = time.time()
        print(f"\n[Transfer fold {fold_f + 1}/{n_folds}]")

        # Load A matrix from training subjects (saved by _run_rrr_for_unit)
        a_path = os.path.join(train_rrr_dir, f"fold_{fold_f}", "A.npy")
        A_full = np.load(a_path)

        baseline = per_fold_baselines[fold_f]
        train_idx = baseline["train_idx"]
        test_idx = baseline["test_idx"]
        X_train = X_raw[train_idx]
        X_test = X_raw[test_idx]
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]

        for k in tqdm(config.rank_grid, desc=f"  transfer fold {fold_f}", leave=True, unit="rank"):
            A_k = A_full[:, :k]
            X_train_proj = X_train @ A_k
            X_test_proj = X_test @ A_k

            result = evaluate_input_space(
                X_train_proj,
                X_test_proj,
                Y_train,
                Y_test,
                device=device,
                alphas=alphas,
                subspace_name="transfer",
            )
            complement_result = evaluate_complement(
                A_full,
                k,
                X_train,
                X_test,
                Y_train,
                Y_test,
                alphas=alphas,
                device=device,
            )

            transfer_corr_arr = result["transfer_corr_per_target"]
            complement_corr_arr = complement_result["complement_corr_per_target"]
            metrics = {
                "transfer_corr": result["transfer_corr"],
                "transfer_r2": result["transfer_r2"],
                "transfer_corr_per_target": transfer_corr_arr,
                "complement_corr": complement_result["complement_corr"],
                "complement_r2": complement_result["complement_r2"],
                "complement_corr_per_target": complement_corr_arr,
            }
            save_rrr_result(transfer_dir, k=k, fold=fold_f, metrics=metrics)

            np.save(
                os.path.join(oof_spill_dir, f"{k}_{fold_f}_transfer.npy"),
                result["transfer_Y_hat"],
            )
            np.save(
                os.path.join(oof_spill_dir, f"{k}_{fold_f}_complement.npy"),
                complement_result["complement_Y_hat"],
            )
            np.save(os.path.join(oof_spill_dir, f"{k}_{fold_f}_Y_test.npy"), Y_test)

            row = {
                "fold": fold_f,
                "k": k,
                "transfer_corr": result["transfer_corr"],
                "transfer_r2": result["transfer_r2"],
                "complement_corr": complement_result["complement_corr"],
                "complement_r2": complement_result["complement_r2"],
            }
            if sig_mask_flat is not None:
                row["transfer_corr_sig"] = (
                    float(transfer_corr_arr[sig_mask_flat].mean())
                    if sig_mask_flat.any()
                    else float("nan")
                )
                row["transfer_corr_nonsig"] = (
                    float(transfer_corr_arr[~sig_mask_flat].mean())
                    if (~sig_mask_flat).any()
                    else float("nan")
                )
                row["complement_corr_sig"] = (
                    float(complement_corr_arr[sig_mask_flat].mean())
                    if sig_mask_flat.any()
                    else float("nan")
                )
                row["complement_corr_nonsig"] = (
                    float(complement_corr_arr[~sig_mask_flat].mean())
                    if (~sig_mask_flat).any()
                    else float("nan")
                )
            all_rows.append(row)

        print(f"  Transfer fold {fold_f} done in {time.time() - fold_start:.1f}s")

    print("\nComputing stacked OLS joint correlations...")
    _compute_and_save_joint_metrics(
        oof_spill_dir,
        config.rank_grid,
        n_folds,
        transfer_dir,
        all_rows,
        primary_label="transfer",
        sig_mask_flat=sig_mask_flat,
    )

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(transfer_dir, "grid_results.csv"), index=False)
    _build_results(df, sig_mask_flat, transfer_dir, primary_label="transfer")
    return df


def _run_leave_one_subject_out_rrr(
    config: "RRRConfig",
    enc,
    output_dir,
    sig_dir,
    rrr_dir,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
):
    """Leave-one-subject-out RRR: train on n-1 subjects, transfer A to held-out subject.

    For each held-out subject:
      1. Train RRR on the remaining n-1 subjects (writes to train_subjects/).
      2. Evaluate transfer of the learned A matrix to the held-out subject (writes to transfer/).
      3. Run a within-subject RRR baseline on the held-out subject alone (writes to baseline/).
    """
    for held_out_sub in subject_ids:
        print(f"\n{'='*60}\nHeld-out subject: {held_out_sub}")

        loso_sub_dir = os.path.join(rrr_dir, f"held_out_sub_{held_out_sub}")

        # --- Step A: train RRR on n-1 subjects ---
        train_subs = [s for s in subject_ids if s != held_out_sub]
        train_sig_elecs = {s: sig_electrodes.get(s, []) for s in train_subs}

        print(f"  Training subjects: {train_subs}")
        results_train = _get_decomposition_inputs(
            train_subs,
            output_dir,
            train_sig_elecs,
            enc,
            aligned_embeddings,
            word_df,
            compute_sig_nonsig=config.compute_sig_nonsig,
            sig_dir=sig_dir,
            word_in_glove=word_in_glove,
        )
        if results_train is None:
            print("  No training data; skipping this held-out subject.")
            continue
        X_raw, X_arbitrary, per_fold_baselines_train, test_idxs, sig_mask_train = results_train

        train_dir = os.path.join(loso_sub_dir, "train_subjects")
        print(f"  Running train-subjects RRR → {train_dir}")
        _run_rrr_for_unit(
            rrr_unit_dir=train_dir,
            X_raw=X_raw,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines_train,
            config=config,
            sig_mask_flat=sig_mask_train,
            X_arbitrary=X_arbitrary,
        )

        # --- Step B: build held-out subject data ---
        held_sig_elecs = sig_electrodes.get(held_out_sub, [])
        if not held_sig_elecs:
            print(f"  Subject {held_out_sub} has no significant electrodes; skipping transfer/baseline.")
            continue

        print(f"  Building held-out subject data...")
        results_held = _get_decomposition_inputs(
            [held_out_sub],
            output_dir,
            {held_out_sub: held_sig_elecs},
            enc,
            aligned_embeddings,
            word_df,
            compute_sig_nonsig=config.compute_sig_nonsig,
            sig_dir=sig_dir,
            word_in_glove=word_in_glove,
        )
        if results_held is None:
            print(f"  No held-out subject data; skipping transfer/baseline.")
            continue
        X_raw_held, X_arbitrary_held, per_fold_baselines_held, _, sig_mask_held = results_held

        # --- Step C: transfer A from train subjects to held-out subject ---
        transfer_dir = os.path.join(loso_sub_dir, "transfer")
        print(f"  Running transfer evaluation → {transfer_dir}")
        _run_loso_transfer_for_unit(
            transfer_dir=transfer_dir,
            X_raw=X_raw_held,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines_held,
            train_rrr_dir=train_dir,
            config=config,
            sig_mask_flat=sig_mask_held,
        )

        # --- Step D: within-subject baseline RRR on held-out subject alone ---
        baseline_dir = os.path.join(loso_sub_dir, "baseline")
        print(f"  Running within-subject baseline RRR → {baseline_dir}")
        _run_rrr_for_unit(
            rrr_unit_dir=baseline_dir,
            X_raw=X_raw_held,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines_held,
            config=config,
            sig_mask_flat=sig_mask_held,
            X_arbitrary=X_arbitrary_held,
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
    dataset_context = resolve_dataset_context(
        dataset=enc.dataset,
        subject_ids=subject_ids,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    get_dataset_adapter(enc.dataset).validate_subjects_share_stimulus(
        subject_ids, enc.per_dataset_config, enc.data_root
    )
    output_dir = config.output_dir or get_results_dir(
        enc.model,
        enc.offset,
        enc.per_model_config,
        dataset_context=dataset_context,
    )

    # Resolve RRR output directory
    if config.rrr_output_dir:
        rrr_dir = config.rrr_output_dir
    elif config.mode == ScriptMode.PER_REGION:
        rrr_dir = os.path.join(output_dir, "per_region_rrr")
    elif config.mode == ScriptMode.PER_SUBJECT:
        rrr_dir = os.path.join(output_dir, "per_subject_rrr")
    elif config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
        rrr_dir = os.path.join(output_dir, "loso_rrr")
    else:
        rrr_dir = os.path.join(output_dir, "rrr")

    os.makedirs(rrr_dir, exist_ok=True)
    print(f"Output dir:  {output_dir}")
    print(f"RRR dir:     {rrr_dir}")
    print(f"Mode:        {config.mode.value}")
    if config.run_pca_baseline:
        print(f"PCA baseline: enabled (results will mirror RRR dir with 'pca' in path)")

    # Load significance CSV and build subject → electrode mapping
    sig_dir = config.sig_dir or output_dir
    sig_electrodes, _ = load_sig_electrodes(
        output_dir,
        subject_ids,
        sig_dir=sig_dir,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        data_root=enc.data_root,
    )

    # Load embeddings
    print("Loading embeddings...")
    aligned_embeddings, word_df = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        **enc.per_model_config,
    )

    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = get_glove_embeddings(
            enc.data_root, word_df, **enc.per_model_config
        )[1]["in_glove"]

    branch_kwargs = dict(
        config=config,
        enc=enc,
        output_dir=output_dir,
        sig_dir=sig_dir,
        rrr_dir=rrr_dir,
        subject_ids=subject_ids,
        sig_electrodes=sig_electrodes,
        aligned_embeddings=aligned_embeddings,
        word_df=word_df,
        word_in_glove=word_in_glove,
    )

    if config.mode == ScriptMode.PER_REGION:
        _run_per_region_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.PER_SUBJECT:
        _run_per_subject_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
        _run_leave_one_subject_out_rrr(**branch_kwargs)
    else:
        _run_cross_subject_rrr(**branch_kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
