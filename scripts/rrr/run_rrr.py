"""
run_rrr.py — Raw-Input Reduced Rank Regression + optional PCA baseline.

Performs RRR directly from raw embeddings to neural targets — no SVD projection.
Sweeps over a configured rank grid. Optionally also runs PCA subspace analysis
(when RRRConfig.run_pca_baseline=True) using the same rank grid and data splits.
By default, completed rank outputs in an existing run directory are reused;
set RRRConfig.refresh=True to recompute the full configured rank grid.

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
    Ridge joint check        : nr_complement_joint_ridge_corr, nr_complement_joint_ridge_r2  [if enabled]
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
from src.utils.tensor_utils import to_numpy
from src.utils.data_loaders import load_sig_electrodes
from src.utils.atlas import build_electrode_region_map
from src.core.config import RRRConfig, ScriptMode, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import (
    parse_args,
    load_config,
    expand_per_model_layer_configs,
)
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
    compute_stacked_joint_ridge_means,
)
from src.utils.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
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
    corr_arr = to_numpy(correlation_score(Y_test, Y_hat_full))
    r2_arr = to_numpy(r2_score(Y_test, Y_hat_full))
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
                float(arr[sig_mask_flat].mean())
                if sig_mask_flat.any()
                else float("nan")
            )
            metrics[f"{base}_nonsig"] = (
                float(arr[~sig_mask_flat].mean())
                if (~sig_mask_flat).any()
                else float("nan")
            )
    for array_key in per_target_keys + [
        k for k in list(metrics) if k.endswith("_best_alphas")
    ]:
        del metrics[array_key]


def _load_existing_grid(unit_dir: str) -> pd.DataFrame:
    """Load prior grid results if present; otherwise return an empty frame."""
    path = os.path.join(unit_dir, "grid_results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_reference_test_idxs(output_dir: str, subject_ids) -> list:
    enc_results = load_encoding_results(
        os.path.join(
            output_dir, "subjects", f"sub_{subject_ids[0]}", "encoding_results.joblib"
        )
    )
    return enc_results["test_idxs"]


def _metrics_path(unit_dir: str, k: int, fold_f: int) -> str:
    return os.path.join(unit_dir, f"k{k}", f"fold_{fold_f}", "metrics.npz")


def _fold_factors_exist(unit_dir: str, fold_f: int, require_b: bool = True) -> bool:
    fold_dir = os.path.join(unit_dir, f"fold_{fold_f}")
    paths = [os.path.join(fold_dir, "A.npy")]
    if require_b:
        paths.append(os.path.join(fold_dir, "B.npy"))
    return all(os.path.exists(path) for path in paths)


def _oof_paths_exist(
    oof_spill_dir: str,
    k: int,
    fold_f: int,
    primary_label: str,
    require_arbitrary: bool = False,
) -> bool:
    paths = [
        os.path.join(oof_spill_dir, f"{k}_{fold_f}_{primary_label}.npy"),
        os.path.join(oof_spill_dir, f"{k}_{fold_f}_complement.npy"),
        os.path.join(oof_spill_dir, f"{k}_{fold_f}_Y_test.npy"),
    ]
    if require_arbitrary:
        paths.append(os.path.join(oof_spill_dir, f"{k}_{fold_f}_arbitrary.npy"))
    return all(os.path.exists(path) for path in paths)


def _rank_complete_from_artifacts(
    unit_dir: str,
    k: int,
    n_folds: int,
    require_factors: bool = False,
) -> bool:
    if any(
        not os.path.exists(_metrics_path(unit_dir, k, fold_f))
        for fold_f in range(n_folds)
    ):
        return False
    if require_factors and any(
        not _fold_factors_exist(unit_dir, fold_f, require_b=True)
        for fold_f in range(n_folds)
    ):
        return False
    return True


def _completed_pairs_from_artifacts(
    unit_dir: str,
    rank_grid: list[int],
    n_folds: int,
    require_factors: bool = False,
    oof_spill_dir: "str | None" = None,
    primary_label: "str | None" = None,
    require_arbitrary_oof: bool = False,
) -> set[tuple[int, int]]:
    """Completed (k, fold) pairs based on artifact files, not grid_results.csv.

    If a rank is only partially complete, OOF spill files are also required so
    stacked joint metrics can be rebuilt when the missing pairs finish. Fully
    complete ranks do not need spill files to be considered completed, because
    their per-fold metric files can rebuild grid rows.
    """
    completed: set[tuple[int, int]] = set()
    for k in rank_grid:
        complete_rank = _rank_complete_from_artifacts(
            unit_dir, k, n_folds, require_factors=require_factors
        )
        for fold_f in range(n_folds):
            if not os.path.exists(_metrics_path(unit_dir, k, fold_f)):
                continue
            if require_factors and not _fold_factors_exist(
                unit_dir, fold_f, require_b=True
            ):
                continue
            if (
                not complete_rank
                and oof_spill_dir is not None
                and primary_label is not None
                and not _oof_paths_exist(
                    oof_spill_dir,
                    k,
                    fold_f,
                    primary_label,
                    require_arbitrary=require_arbitrary_oof,
                )
            ):
                continue
            completed.add((k, fold_f))
    return completed


def _ranks_by_fold_to_run(
    unit_dir: str,
    rank_grid: list[int],
    n_folds: int,
    refresh: bool,
    require_factors: bool = False,
    oof_spill_dir: "str | None" = None,
    primary_label: "str | None" = None,
    require_arbitrary_oof: bool = False,
) -> tuple[dict[int, list[int]], set[tuple[int, int]]]:
    """Return per-fold ranks to compute plus (k, fold) pairs reusable from disk."""
    if refresh:
        return {fold_f: list(rank_grid) for fold_f in range(n_folds)}, set()
    completed = _completed_pairs_from_artifacts(
        unit_dir,
        rank_grid,
        n_folds,
        require_factors=require_factors,
        oof_spill_dir=oof_spill_dir,
        primary_label=primary_label,
        require_arbitrary_oof=require_arbitrary_oof,
    )
    ranks_by_fold = {
        fold_f: [k for k in rank_grid if (k, fold_f) not in completed]
        for fold_f in range(n_folds)
    }
    return ranks_by_fold, completed


def _folds_to_build_from_ranks(ranks_by_fold: dict[int, list[int]]) -> set[int]:
    return {fold_f for fold_f, ranks in ranks_by_fold.items() if ranks}


def _unit_ranks_by_fold_to_run(
    unit_dir: str,
    rank_grid: list[int],
    n_folds: int,
    refresh: bool,
    require_factors: bool,
    primary_label: str,
    require_arbitrary_oof: bool = True,
) -> tuple[dict[int, list[int]], set[tuple[int, int]]]:
    oof_spill_dir = os.path.join(unit_dir, "_oof_spill")
    return _ranks_by_fold_to_run(
        unit_dir,
        rank_grid,
        n_folds,
        refresh=refresh,
        require_factors=require_factors,
        oof_spill_dir=oof_spill_dir,
        primary_label=primary_label,
        require_arbitrary_oof=require_arbitrary_oof,
    )


def _rrr_output_resume_plan(
    rrr_unit_dir: str,
    config: RRRConfig,
    n_folds: int,
) -> dict:
    rrr_ranks_by_fold, rrr_completed_pairs = _unit_ranks_by_fold_to_run(
        rrr_unit_dir,
        config.rank_grid,
        n_folds,
        refresh=config.refresh,
        require_factors=True,
        primary_label="nr",
        require_arbitrary_oof=True,
    )
    folds_needed = _folds_to_build_from_ranks(rrr_ranks_by_fold)
    pca_ranks_by_fold = None
    pca_completed_pairs = None

    if config.run_pca_baseline:
        pca_unit_dir = _resolve_pca_sibling_dir(rrr_unit_dir)
        pca_ranks_by_fold, pca_completed_pairs = _unit_ranks_by_fold_to_run(
            pca_unit_dir,
            config.rank_grid,
            n_folds,
            refresh=config.refresh,
            require_factors=False,
            primary_label="pca",
            require_arbitrary_oof=True,
        )
        folds_needed |= _folds_to_build_from_ranks(pca_ranks_by_fold)

    return {
        "folds_to_build": folds_needed,
        "rrr_ranks_by_fold": rrr_ranks_by_fold,
        "rrr_completed_pairs": rrr_completed_pairs,
        "pca_ranks_by_fold": pca_ranks_by_fold,
        "pca_completed_pairs": pca_completed_pairs,
    }


def _transfer_output_resume_plan(
    transfer_dir: str,
    config: RRRConfig,
    n_folds: int,
) -> dict:
    ranks_by_fold, completed_pairs = _unit_ranks_by_fold_to_run(
        transfer_dir,
        config.rank_grid,
        n_folds,
        refresh=config.refresh,
        require_factors=False,
        primary_label="transfer",
        require_arbitrary_oof=False,
    )
    return {
        "folds_to_build": _folds_to_build_from_ranks(ranks_by_fold),
        "ranks_by_fold": ranks_by_fold,
        "completed_pairs": completed_pairs,
    }


def _scalar_from_metric(metrics: dict, key: str) -> "float | None":
    if key in metrics:
        value = metrics[key]
        arr = np.asarray(value)
        return float(arr.mean()) if arr.shape else float(arr)
    per_target_key = f"{key}_per_target"
    if per_target_key in metrics:
        return float(np.asarray(metrics[per_target_key]).mean())
    return None


def _row_from_metrics_file(
    unit_dir: str,
    k: int,
    fold_f: int,
    primary_label: str,
    sig_mask_flat: "np.ndarray | None",
    include_effective_rank: bool = False,
) -> dict:
    metrics = dict(np.load(_metrics_path(unit_dir, k, fold_f), allow_pickle=True))
    row = {"fold": fold_f, "k": k}

    if include_effective_rank and _fold_factors_exist(unit_dir, fold_f, require_b=True):
        A_full = np.load(os.path.join(unit_dir, f"fold_{fold_f}", "A.npy"))
        A_k = A_full[:, :k]
        sv = np.linalg.svd(A_k, compute_uv=False)
        row["effective_rank"] = (
            int(np.sum(sv > 1e-6 * sv[0])) if len(sv) > 0 and sv[0] > 0 else 0
        )

    base_keys = [
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
    ]
    if primary_label == "nr":
        base_keys.extend(["rrr_corr", "rrr_r2"])

    for key in base_keys:
        value = _scalar_from_metric(metrics, key)
        if value is not None:
            row[key] = value

    if sig_mask_flat is not None:
        for key in base_keys:
            per_target_key = f"{key}_per_target"
            if per_target_key not in metrics:
                continue
            arr = np.asarray(metrics[per_target_key])
            if key.endswith("_corr"):
                row[f"{key}_sig"] = (
                    float(arr[sig_mask_flat].mean())
                    if sig_mask_flat.any()
                    else float("nan")
                )
                row[f"{key}_nonsig"] = (
                    float(arr[~sig_mask_flat].mean())
                    if (~sig_mask_flat).any()
                    else float("nan")
                )
    return row


def _merge_grid_rows(
    existing_df: pd.DataFrame,
    new_rows: list[dict],
    completed_pairs: set[tuple[int, int]],
    rank_grid: list[int],
    unit_dir: "str | None" = None,
    primary_label: "str | None" = None,
    sig_mask_flat: "np.ndarray | None" = None,
    include_effective_rank: bool = False,
) -> pd.DataFrame:
    """Merge trusted prior (k, fold) rows with rows generated in this invocation."""
    parts = []
    completed_index = pd.MultiIndex.from_tuples(completed_pairs, names=["k", "fold"])
    grid_pairs = set()
    if (
        completed_pairs
        and not existing_df.empty
        and {"k", "fold"}.issubset(existing_df.columns)
    ):
        prior = existing_df.copy()
        prior["k"] = prior["k"].astype(int)
        prior["fold"] = prior["fold"].astype(int)
        prior = prior.set_index(["k", "fold"])
        prior = prior[prior.index.isin(completed_index)].reset_index()
        grid_pairs = set(zip(prior["k"].astype(int), prior["fold"].astype(int)))
        parts.append(prior)
    missing_grid_pairs = completed_pairs - grid_pairs
    if missing_grid_pairs and unit_dir is not None and primary_label is not None:
        parts.append(
            pd.DataFrame(
                [
                    _row_from_metrics_file(
                        unit_dir,
                        k,
                        fold_f,
                        primary_label,
                        sig_mask_flat,
                        include_effective_rank=include_effective_rank,
                    )
                    for k, fold_f in sorted(missing_grid_pairs)
                ]
            )
        )
    if new_rows:
        parts.append(pd.DataFrame(new_rows))
    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True, sort=False)
    df = df[df["k"].isin(rank_grid)].copy()
    df["k"] = df["k"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df = df.drop_duplicates(subset=["k", "fold"], keep="last")
    leading = [col for col in ("fold", "k") if col in df.columns]
    df = df[leading + [col for col in df.columns if col not in leading]]
    return df.sort_values(["k", "fold"]).reset_index(drop=True)


def _joint_ranks_ready_to_compute(
    df: pd.DataFrame,
    rank_grid: list[int],
    n_folds: int,
    oof_spill_dir: str,
    primary_label: str,
    require_arbitrary_oof: bool = False,
    compute_joint_ridge: bool = False,
) -> set[int]:
    """Ranks with complete rows/spills but missing joint scalar columns."""
    if df.empty or "k" not in df.columns or "fold" not in df.columns:
        return set()

    required_cols = [
        f"{primary_label}_complement_joint_corr",
        f"{primary_label}_complement_joint_r2",
    ]
    if compute_joint_ridge:
        required_cols.extend(
            [
                f"{primary_label}_complement_joint_ridge_corr",
                f"{primary_label}_complement_joint_ridge_r2",
            ]
        )

    ready = set()
    expected_folds = set(range(n_folds))
    for k in rank_grid:
        rows = df[df["k"].astype(int) == k]
        if set(rows["fold"].astype(int).tolist()) != expected_folds:
            continue
        missing_joint = any(
            col not in rows.columns or rows[col].isna().any() for col in required_cols
        )
        if not missing_joint:
            continue
        if all(
            _oof_paths_exist(
                oof_spill_dir,
                k,
                fold_f,
                primary_label,
                require_arbitrary=require_arbitrary_oof,
            )
            for fold_f in expected_folds
        ):
            ready.add(k)
    return ready


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
    alpha_policy: dict | None,
    standardize_subspace_inputs: bool,
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

    # NR/PCA: project X into k-dim subspace, train ridge
    X_proj_train = X_train @ A_k
    X_proj_test = X_test @ A_k
    rank_dir = os.path.join(unit_dir, f"k{k}", f"fold_{fold_f}")
    nr_context = {
        "script": "run_rrr",
        "role": primary_label,
        "fold": int(fold_f),
        "k": int(k),
        "n_features": int(X_proj_train.shape[1]),
        "standardize": bool(standardize_subspace_inputs),
    }
    nr_alphas = resolve_ridge_alphas(
        alphas,
        X_proj_train,
        standardize=standardize_subspace_inputs,
        context=nr_context,
        cache_dir=os.path.join(rank_dir, f"{primary_label}_alpha"),
        policy=alpha_policy,
    )
    nr_result = evaluate_input_space(
        X_proj_train,
        X_proj_test,
        Y_train,
        Y_test,
        subspace_name=primary_label,
        alphas=nr_alphas,
        standardize=standardize_subspace_inputs,
        return_alphas=True,
        device=device,
    )

    # Complement evaluation
    if k == A_full.shape[1]:
        X_comp_train = np.ones((X_train.shape[0], 1))
    else:
        X_comp_train = (X_train @ A_full)[:, k:]
    comp_context = {
        "script": "run_rrr",
        "role": "complement",
        "fold": int(fold_f),
        "k": int(k),
        "n_features": int(X_comp_train.shape[1]),
        "standardize": bool(standardize_subspace_inputs),
    }
    comp_alphas = resolve_ridge_alphas(
        alphas,
        X_comp_train,
        standardize=standardize_subspace_inputs,
        context=comp_context,
        cache_dir=os.path.join(rank_dir, "complement_alpha"),
        policy=alpha_policy,
    )
    complement_result = evaluate_complement(
        A_full,
        k,
        X_train,
        X_test,
        Y_train,
        Y_test,
        alphas=comp_alphas,
        standardize=standardize_subspace_inputs,
        return_alphas=True,
        device=device,
    )

    # Arbitrary evaluation (null baseline)
    arbitrary_result = None
    if X_arbitrary is not None:
        arb_context = {
            "script": "run_rrr",
            "role": "arbitrary",
            "fold": int(fold_f),
            "k": int(k),
            "n_features": int(X_arbitrary[train_idx].shape[1]),
            "standardize": True,
        }
        arbitrary_alphas = resolve_ridge_alphas(
            alphas,
            X_arbitrary[train_idx],
            standardize=True,
            context=arb_context,
            cache_dir=os.path.join(rank_dir, "arbitrary_alpha"),
            policy=alpha_policy,
        )
        arbitrary_result = evaluate_arbitrary(
            X_arbitrary[train_idx],
            X_arbitrary[test_idx],
            Y_train,
            Y_test,
            alphas=arbitrary_alphas,
            standardize=True,
            return_alphas=True,
            device=device,
        )
    save_alpha_diagnostics(
        os.path.join(rank_dir, f"{primary_label}_alpha"),
        nr_alphas,
        best_alphas=nr_result[f"{primary_label}_best_alphas"],
        context=nr_context,
        policy=alpha_policy,
    )
    save_alpha_diagnostics(
        os.path.join(rank_dir, "complement_alpha"),
        comp_alphas,
        best_alphas=complement_result["complement_best_alphas"],
        context=comp_context,
        policy=alpha_policy,
    )
    if arbitrary_result is not None:
        save_alpha_diagnostics(
            os.path.join(rank_dir, "arbitrary_alpha"),
            arbitrary_alphas,
            best_alphas=arbitrary_result["arbitrary_best_alphas"],
            context=arb_context,
            policy=alpha_policy,
        )

    # Build metrics dict (with per-target arrays for metrics.npz save)
    metrics = {
        f"{primary_label}_corr": nr_result[f"{primary_label}_corr"],
        f"{primary_label}_r2": nr_result[f"{primary_label}_r2"],
        f"{primary_label}_corr_per_target": nr_result[
            f"{primary_label}_corr_per_target"
        ],
        f"{primary_label}_best_alphas": nr_result[f"{primary_label}_best_alphas"],
        "complement_corr": complement_result["complement_corr"],
        "complement_r2": complement_result["complement_r2"],
        "complement_corr_per_target": complement_result["complement_corr_per_target"],
        "complement_best_alphas": complement_result["complement_best_alphas"],
    }
    if arbitrary_result is not None:
        metrics["arbitrary_corr"] = arbitrary_result["arbitrary_corr"]
        metrics["arbitrary_r2"] = arbitrary_result["arbitrary_r2"]
        metrics["arbitrary_corr_per_target"] = arbitrary_result[
            "arbitrary_corr_per_target"
        ]
        metrics["arbitrary_best_alphas"] = arbitrary_result["arbitrary_best_alphas"]

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
    compute_joint_ridge: bool = False,
    device: str = "cpu",
) -> None:
    """Post-loop: compute stacked joint metrics, update rows and metrics.npz files.

    Computes joint metrics for:
        - {primary_label} + complement
        - {primary_label} + arbitrary  (if arbitrary OOF files exist)

    Updates all_rows in-place with joint scalar metrics.
    Appends OLS joint *_per_target arrays to each k/fold metrics.npz.
    When compute_joint_ridge=True, also stores ridge-stacked scalar means in
    grid_results.csv/summary.csv only.
    Cleans up the OOF spill directory when done.
    """
    import shutil

    evaluated_ranks = {row["k"] for row in all_rows}
    for k in [k for k in rank_grid if k in evaluated_ranks]:
        all_primary, all_complement, all_y_test = [], [], []
        all_arbitrary = []
        has_arbitrary = True

        for fold_f in range(n_folds):
            all_primary.append(
                np.load(
                    os.path.join(oof_spill_dir, f"{k}_{fold_f}_{primary_label}.npy")
                )
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
        jm_comp_ridge = None
        comp_ridge_key = f"{primary_label}_complement_joint_ridge"
        if compute_joint_ridge:
            jm_comp_ridge = compute_stacked_joint_ridge_means(
                all_primary,
                all_complement,
                all_y_test,
                n_splits=n_folds,
                device=device,
            )

        # Joint: primary + arbitrary
        jm_arb = None
        arb_joint_key = f"{primary_label}_arbitrary_joint"
        jm_arb_ridge = None
        arb_ridge_key = f"{primary_label}_arbitrary_joint_ridge"
        if has_arbitrary and all_arbitrary:
            jm_arb = compute_stacked_joint_metrics(
                all_primary, all_arbitrary, all_y_test, n_splits=n_folds
            )
            if compute_joint_ridge:
                jm_arb_ridge = compute_stacked_joint_ridge_means(
                    all_primary,
                    all_arbitrary,
                    all_y_test,
                    n_splits=n_folds,
                    device=device,
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
                    float(arr[sig_mask_flat].mean())
                    if sig_mask_flat.any()
                    else float("nan")
                )
                row[f"{comp_joint_key}_corr_nonsig"] = (
                    float(arr[~sig_mask_flat].mean())
                    if (~sig_mask_flat).any()
                    else float("nan")
                )
            if jm_comp_ridge is not None:
                row[f"{comp_ridge_key}_corr"] = float(jm_comp_ridge["corr"])
                row[f"{comp_ridge_key}_r2"] = float(jm_comp_ridge["r2"])
            if jm_arb is not None:
                row[f"{arb_joint_key}_corr"] = float(jm_arb["corr_per_target"].mean())
                row[f"{arb_joint_key}_r2"] = float(jm_arb["r2_per_target"].mean())
                if sig_mask_flat is not None:
                    arr = jm_arb["corr_per_target"]
                    row[f"{arb_joint_key}_corr_sig"] = (
                        float(arr[sig_mask_flat].mean())
                        if sig_mask_flat.any()
                        else float("nan")
                    )
                    row[f"{arb_joint_key}_corr_nonsig"] = (
                        float(arr[~sig_mask_flat].mean())
                        if (~sig_mask_flat).any()
                        else float("nan")
                    )
            if jm_arb_ridge is not None:
                row[f"{arb_ridge_key}_corr"] = float(jm_arb_ridge["corr"])
                row[f"{arb_ridge_key}_r2"] = float(jm_arb_ridge["r2"])

        # Append per-target arrays to each fold's metrics.npz
        for fold_f in range(n_folds):
            metrics_path = os.path.join(
                unit_dir, f"k{k}", f"fold_{fold_f}", "metrics.npz"
            )
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
        f"{primary_label}_complement_joint_ridge_corr",
        f"{primary_label}_complement_joint_ridge_r2",
        f"{primary_label}_arbitrary_joint_corr",
        f"{primary_label}_arbitrary_joint_r2",
        f"{primary_label}_arbitrary_joint_ridge_corr",
        f"{primary_label}_arbitrary_joint_ridge_r2",
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

    display_label = (
        "RRR subspace (NR)"
        if primary_label == "nr" and "rrr_corr" in df.columns
        else primary_label.upper()
    )
    title_base = f"{display_label}: perf vs rank ({os.path.basename(unit_dir)})"
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
    ranks_by_fold: "dict[int, list[int]] | None" = None,
    completed_pairs: "set[tuple[int, int]] | None" = None,
    pca_ranks_by_fold: "dict[int, list[int]] | None" = None,
    pca_completed_pairs: "set[tuple[int, int]] | None" = None,
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

    oof_spill_dir = os.path.join(rrr_unit_dir, "_oof_spill")
    os.makedirs(oof_spill_dir, exist_ok=True)

    existing_grid = _load_existing_grid(rrr_unit_dir)
    if ranks_by_fold is None or completed_pairs is None:
        ranks_by_fold, completed_pairs = _unit_ranks_by_fold_to_run(
            rrr_unit_dir,
            config.rank_grid,
            n_folds,
            refresh=config.refresh,
            require_factors=True,
            primary_label="nr",
            require_arbitrary_oof=X_arbitrary is not None,
        )
    total_to_run = sum(len(ranks) for ranks in ranks_by_fold.values())
    if completed_pairs:
        print(f"  Reusing {len(completed_pairs)} completed fold/rank pair(s)")
    if total_to_run == 0:
        print(
            "  All requested fold/rank pairs complete; rebuilding summary/plots from existing artifacts."
        )
        df = _merge_grid_rows(
            existing_grid,
            [],
            completed_pairs,
            config.rank_grid,
            unit_dir=rrr_unit_dir,
            primary_label="nr",
            sig_mask_flat=sig_mask_flat,
            include_effective_rank=True,
        )
        joint_ranks = _joint_ranks_ready_to_compute(
            df,
            config.rank_grid,
            n_folds,
            oof_spill_dir,
            primary_label="nr",
            require_arbitrary_oof=X_arbitrary is not None,
            compute_joint_ridge=config.compute_joint_ridge,
        )
        if joint_ranks:
            joint_label = "OLS + ridge" if config.compute_joint_ridge else "OLS"
            print(f"  Completing stacked {joint_label} joint metrics")
            merged_rows = df.to_dict("records")
            _compute_and_save_joint_metrics(
                oof_spill_dir,
                sorted(joint_ranks),
                n_folds,
                rrr_unit_dir,
                merged_rows,
                primary_label="nr",
                sig_mask_flat=sig_mask_flat,
                compute_joint_ridge=config.compute_joint_ridge,
                device=device,
            )
            df = pd.DataFrame(merged_rows)
            if not df.empty:
                df = df.sort_values(["k", "fold"]).reset_index(drop=True)
        df.to_csv(os.path.join(rrr_unit_dir, "grid_results.csv"), index=False)
        _build_results(df, sig_mask_flat, rrr_unit_dir, primary_label="nr")
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
                alpha_policy=config.encoding.alpha_grid,
                standardize_subspace_inputs=config.standardize_subspace_inputs,
                device=device,
                refresh=config.refresh,
                compute_joint_ridge=config.compute_joint_ridge,
                ranks_by_fold=pca_ranks_by_fold,
                completed_pairs=pca_completed_pairs,
            )
        return df

    all_rows = []
    affected_ranks = set()

    for fold_f in range(n_folds):
        ranks_to_run = ranks_by_fold[fold_f]
        if not ranks_to_run:
            print(f"\n[Fold {fold_f + 1}/{n_folds}] complete; skipping")
            continue

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
        full_corr = float(to_numpy(correlation_score(Y_test, Y_hat_full)).mean())
        full_r2 = float(to_numpy(r2_score(Y_test, Y_hat_full)).mean())

        # Fit or load RRR once per fold; truncate to each k below.
        max_k = max(config.rank_grid)
        fold_dir = os.path.join(rrr_unit_dir, f"fold_{fold_f}")
        a_path = os.path.join(fold_dir, "A.npy")
        b_path = os.path.join(fold_dir, "B.npy")
        if not config.refresh and _fold_factors_exist(
            rrr_unit_dir, fold_f, require_b=True
        ):
            A_full = np.load(a_path)
            B_full = np.load(b_path)
            if A_full.shape[1] < max_k or B_full.shape[1] < max_k:
                print("  Existing factors do not cover max rank; refitting")
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
            else:
                print("  Loaded existing fold factors")
        else:
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

        print(f"  Running {len(ranks_to_run)} rank(s): {ranks_to_run}")
        for k in tqdm(ranks_to_run, desc=f"  fold {fold_f}", leave=True, unit="rank"):
            affected_ranks.add(k)
            A_k = A_full[:, :k]
            B_k = B_full[:, :k]

            # Effective rank of A_k
            sv = np.linalg.svd(A_k, compute_uv=False)
            effective_rank = (
                int(np.sum(sv > 1e-6 * sv[0])) if len(sv) > 0 and sv[0] > 0 else 0
            )

            # Raw RRR predictions (project Y through B)
            Y_hat_rrr = Y_hat_full @ B_k @ B_k.T
            rrr_corr_arr = to_numpy(correlation_score(Y_test, Y_hat_rrr))
            rrr_r2_arr = to_numpy(r2_score(Y_test, Y_hat_rrr))

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
                config.encoding.alpha_grid,
                config.standardize_subspace_inputs,
                device,
                rrr_unit_dir,
                oof_spill_dir,
                primary_label="nr",
            )

            # Load existing metrics.npz and append RRR per-target metrics.
            metrics_path = os.path.join(
                rrr_unit_dir, f"k{k}", f"fold_{fold_f}", "metrics.npz"
            )
            existing = dict(np.load(metrics_path, allow_pickle=True))
            existing["full_corr"] = full_corr
            existing["full_r2"] = full_r2
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

    # Compute stacked joint correlations (requires held-out OOF predictions)
    joint_label = "OLS + ridge" if config.compute_joint_ridge else "OLS"
    print(f"\nComputing stacked {joint_label} joint correlations...")
    merged_rows = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        config.rank_grid,
        unit_dir=rrr_unit_dir,
        primary_label="nr",
        sig_mask_flat=sig_mask_flat,
        include_effective_rank=True,
    ).to_dict("records")
    joint_ranks = set(affected_ranks) | _joint_ranks_ready_to_compute(
        pd.DataFrame(merged_rows),
        config.rank_grid,
        n_folds,
        oof_spill_dir,
        primary_label="nr",
        require_arbitrary_oof=X_arbitrary is not None,
        compute_joint_ridge=config.compute_joint_ridge,
    )
    _compute_and_save_joint_metrics(
        oof_spill_dir,
        sorted(joint_ranks),
        n_folds,
        rrr_unit_dir,
        merged_rows,
        primary_label="nr",
        sig_mask_flat=sig_mask_flat,
        compute_joint_ridge=config.compute_joint_ridge,
        device=device,
    )

    df = pd.DataFrame(merged_rows)
    if not df.empty:
        df = df.sort_values(["k", "fold"]).reset_index(drop=True)
    df.to_csv(os.path.join(rrr_unit_dir, "grid_results.csv"), index=False)
    _build_results(df, sig_mask_flat, rrr_unit_dir, primary_label="nr")

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
            alpha_policy=config.encoding.alpha_grid,
            standardize_subspace_inputs=config.standardize_subspace_inputs,
            device=device,
            refresh=config.refresh,
            compute_joint_ridge=config.compute_joint_ridge,
            ranks_by_fold=pca_ranks_by_fold,
            completed_pairs=pca_completed_pairs,
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
    alpha_policy: dict | None = None,
    standardize_subspace_inputs: bool = True,
    device: str = "cpu",
    refresh: bool = False,
    compute_joint_ridge: bool = False,
    ranks_by_fold: "dict[int, list[int]] | None" = None,
    completed_pairs: "set[tuple[int, int]] | None" = None,
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
    oof_spill_dir = os.path.join(pca_unit_dir, "_oof_spill")
    os.makedirs(oof_spill_dir, exist_ok=True)

    existing_grid = _load_existing_grid(pca_unit_dir)
    if ranks_by_fold is None or completed_pairs is None:
        ranks_by_fold, completed_pairs = _unit_ranks_by_fold_to_run(
            pca_unit_dir,
            rank_grid,
            n_folds,
            refresh=refresh,
            require_factors=False,
            primary_label="pca",
            require_arbitrary_oof=X_arbitrary is not None,
        )
    total_to_run = sum(len(ranks) for ranks in ranks_by_fold.values())
    if completed_pairs:
        print(f"  Reusing {len(completed_pairs)} completed PCA fold/rank pair(s)")
    if total_to_run == 0:
        print(
            "  All requested PCA fold/rank pairs complete; rebuilding summary/plots from existing artifacts."
        )
        df = _merge_grid_rows(
            existing_grid,
            [],
            completed_pairs,
            rank_grid,
            unit_dir=pca_unit_dir,
            primary_label="pca",
            sig_mask_flat=sig_mask_flat,
        )
        joint_ranks = _joint_ranks_ready_to_compute(
            df,
            rank_grid,
            n_folds,
            oof_spill_dir,
            primary_label="pca",
            require_arbitrary_oof=X_arbitrary is not None,
            compute_joint_ridge=compute_joint_ridge,
        )
        if joint_ranks:
            joint_label = "OLS + ridge" if compute_joint_ridge else "OLS"
            print(f"  Completing stacked {joint_label} joint metrics")
            merged_rows = df.to_dict("records")
            _compute_and_save_joint_metrics(
                oof_spill_dir,
                sorted(joint_ranks),
                n_folds,
                pca_unit_dir,
                merged_rows,
                primary_label="pca",
                sig_mask_flat=sig_mask_flat,
                compute_joint_ridge=compute_joint_ridge,
                device=device,
            )
            df = pd.DataFrame(merged_rows)
            if not df.empty:
                df = df.sort_values(["k", "fold"]).reset_index(drop=True)
        df.to_csv(os.path.join(pca_unit_dir, "grid_results.csv"), index=False)
        _build_results(df, sig_mask_flat, pca_unit_dir, primary_label="pca")
        return df

    all_rows = []
    affected_ranks = set()

    for fold_f in range(n_folds):
        ranks_to_run = ranks_by_fold[fold_f]
        if not ranks_to_run:
            print(f"\n[Fold {fold_f + 1}/{n_folds}] complete; skipping")
            continue

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

        valid_ranks = [k for k in ranks_to_run if k <= max_components]
        if len(valid_ranks) < len(ranks_to_run):
            skipped = [k for k in ranks_to_run if k > max_components]
            print(f"  Skipping ranks {skipped} (> {max_components} PCA components)")

        print(f"  Running {len(valid_ranks)} rank(s): {valid_ranks}")
        for k in tqdm(valid_ranks, desc=f"  fold {fold_f}", leave=True, unit="rank"):
            affected_ranks.add(k)
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
                alpha_policy,
                standardize_subspace_inputs,
                device,
                pca_unit_dir,
                oof_spill_dir,
                primary_label="pca",
            )
            row = {"fold": fold_f, "k": k, **common}
            all_rows.append(row)

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    # Compute stacked joint correlations
    joint_label = "OLS + ridge" if compute_joint_ridge else "OLS"
    print(f"\nComputing stacked {joint_label} joint correlations...")
    merged_rows = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        rank_grid,
        unit_dir=pca_unit_dir,
        primary_label="pca",
        sig_mask_flat=sig_mask_flat,
    ).to_dict("records")
    joint_ranks = set(affected_ranks) | _joint_ranks_ready_to_compute(
        pd.DataFrame(merged_rows),
        rank_grid,
        n_folds,
        oof_spill_dir,
        primary_label="pca",
        require_arbitrary_oof=X_arbitrary is not None,
        compute_joint_ridge=compute_joint_ridge,
    )
    _compute_and_save_joint_metrics(
        oof_spill_dir,
        sorted(joint_ranks),
        n_folds,
        pca_unit_dir,
        merged_rows,
        primary_label="pca",
        sig_mask_flat=sig_mask_flat,
        compute_joint_ridge=compute_joint_ridge,
        device=device,
    )

    df = pd.DataFrame(merged_rows)
    if not df.empty:
        df = df.sort_values(["k", "fold"]).reset_index(drop=True)
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
    folds_to_build: "set[int] | None" = None,
    alpha_cache_dir: str | None = None,
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

    per_fold_baselines = build_per_fold_baselines(
        X_raw,
        Y_full,
        test_idxs,
        alphas=(
            np.array(encoding_config.ridge_alphas, dtype=float)
            if encoding_config.ridge_alphas is not None
            else None
        ),
        alpha_policy=encoding_config.alpha_grid,
        alpha_cache_dir=alpha_cache_dir,
        alpha_context={
            "script": "run_rrr",
            "model": encoding_config.model,
            "per_model_config": encoding_config.per_model_config,
            "offset": encoding_config.offset,
        },
        folds_to_build=folds_to_build,
    )

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
    n_folds = len(_load_reference_test_idxs(output_dir, subject_ids))
    resume_plan = _rrr_output_resume_plan(rrr_dir, config, n_folds)
    folds_to_build = resume_plan["folds_to_build"]
    if not folds_to_build:
        print(
            "  Cross-subject RRR/PCA outputs complete; skipping data/regression setup."
        )
        return

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
        folds_to_build=folds_to_build,
        alpha_cache_dir=rrr_dir,
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
        ranks_by_fold=resume_plan["rrr_ranks_by_fold"],
        completed_pairs=resume_plan["rrr_completed_pairs"],
        pca_ranks_by_fold=resume_plan["pca_ranks_by_fold"],
        pca_completed_pairs=resume_plan["pca_completed_pairs"],
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

        n_folds = len(_load_reference_test_idxs(output_dir, [sub_id]))
        resume_plan = _rrr_output_resume_plan(rrr_sub_dir, config, n_folds)
        folds_to_build = resume_plan["folds_to_build"]
        if not folds_to_build:
            print("  Subject RRR/PCA outputs complete; skipping data/regression setup.")
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
            folds_to_build=folds_to_build,
            alpha_cache_dir=rrr_sub_dir,
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
            ranks_by_fold=resume_plan["rrr_ranks_by_fold"],
            completed_pairs=resume_plan["rrr_completed_pairs"],
            pca_ranks_by_fold=resume_plan["pca_ranks_by_fold"],
            pca_completed_pairs=resume_plan["pca_completed_pairs"],
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
        per_subject_region_elecs = {
            s: e for s, e in per_subject_region_elecs.items() if e
        }

        if not per_subject_region_elecs:
            print("  No significant electrodes in region; skipping.")
            continue

        subs_in_region = list(per_subject_region_elecs.keys())
        n_folds = len(_load_reference_test_idxs(output_dir, subs_in_region))
        resume_plan = _rrr_output_resume_plan(rrr_region_dir, config, n_folds)
        folds_to_build = resume_plan["folds_to_build"]
        if not folds_to_build:
            print("  Region RRR/PCA outputs complete; skipping data/regression setup.")
            continue

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
            folds_to_build=folds_to_build,
            alpha_cache_dir=rrr_region_dir,
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
            ranks_by_fold=resume_plan["rrr_ranks_by_fold"],
            completed_pairs=resume_plan["rrr_completed_pairs"],
            pca_ranks_by_fold=resume_plan["pca_ranks_by_fold"],
            pca_completed_pairs=resume_plan["pca_completed_pairs"],
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
    ranks_by_fold: "dict[int, list[int]] | None" = None,
    completed_pairs: "set[tuple[int, int]] | None" = None,
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
    oof_spill_dir = os.path.join(transfer_dir, "_oof_spill")
    os.makedirs(oof_spill_dir, exist_ok=True)

    existing_grid = _load_existing_grid(transfer_dir)
    if ranks_by_fold is None or completed_pairs is None:
        ranks_by_fold, completed_pairs = _unit_ranks_by_fold_to_run(
            transfer_dir,
            config.rank_grid,
            n_folds,
            refresh=config.refresh,
            require_factors=False,
            primary_label="transfer",
        )
    missing_a = [
        os.path.join(train_rrr_dir, f"fold_{fold_f}", "A.npy")
        for fold_f in range(n_folds)
        if not os.path.exists(os.path.join(train_rrr_dir, f"fold_{fold_f}", "A.npy"))
    ]
    if missing_a:
        raise FileNotFoundError(
            "Missing train-subject RRR factor(s) needed for transfer: "
            + ", ".join(missing_a)
        )
    total_to_run = sum(len(ranks) for ranks in ranks_by_fold.values())
    if completed_pairs:
        print(f"  Reusing {len(completed_pairs)} completed transfer fold/rank pair(s)")
    if total_to_run == 0:
        print(
            "  All requested transfer fold/rank pairs complete; rebuilding summary/plots from existing artifacts."
        )
        df = _merge_grid_rows(
            existing_grid,
            [],
            completed_pairs,
            config.rank_grid,
            unit_dir=transfer_dir,
            primary_label="transfer",
            sig_mask_flat=sig_mask_flat,
        )
        joint_ranks = _joint_ranks_ready_to_compute(
            df,
            config.rank_grid,
            n_folds,
            oof_spill_dir,
            primary_label="transfer",
            compute_joint_ridge=config.compute_joint_ridge,
        )
        if joint_ranks:
            joint_label = "OLS + ridge" if config.compute_joint_ridge else "OLS"
            print(f"  Completing stacked {joint_label} joint metrics")
            merged_rows = df.to_dict("records")
            _compute_and_save_joint_metrics(
                oof_spill_dir,
                sorted(joint_ranks),
                n_folds,
                transfer_dir,
                merged_rows,
                primary_label="transfer",
                sig_mask_flat=sig_mask_flat,
                compute_joint_ridge=config.compute_joint_ridge,
                device=device,
            )
            df = pd.DataFrame(merged_rows)
            if not df.empty:
                df = df.sort_values(["k", "fold"]).reset_index(drop=True)
        df.to_csv(os.path.join(transfer_dir, "grid_results.csv"), index=False)
        _build_results(df, sig_mask_flat, transfer_dir, primary_label="transfer")
        return df

    all_rows = []
    affected_ranks = set()

    for fold_f in range(n_folds):
        ranks_to_run = ranks_by_fold[fold_f]
        if not ranks_to_run:
            print(f"\n[Transfer fold {fold_f + 1}/{n_folds}] complete; skipping")
            continue

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

        for k in tqdm(
            ranks_to_run, desc=f"  transfer fold {fold_f}", leave=True, unit="rank"
        ):
            affected_ranks.add(k)
            common = _run_common_subspace_fold_k(
                A_full,
                k,
                X_train,
                X_test,
                Y_train,
                Y_test,
                None,
                train_idx,
                test_idx,
                sig_mask_flat,
                fold_f,
                alphas=alphas,
                alpha_policy=config.encoding.alpha_grid,
                standardize_subspace_inputs=config.standardize_subspace_inputs,
                device=device,
                unit_dir=transfer_dir,
                oof_spill_dir=oof_spill_dir,
                primary_label="transfer",
            )

            row = {
                "fold": fold_f,
                "k": k,
                **common,
            }
            all_rows.append(row)

        print(f"  Transfer fold {fold_f} done in {time.time() - fold_start:.1f}s")

    joint_label = "OLS + ridge" if config.compute_joint_ridge else "OLS"
    print(f"\nComputing stacked {joint_label} joint correlations...")
    merged_rows = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        config.rank_grid,
        unit_dir=transfer_dir,
        primary_label="transfer",
        sig_mask_flat=sig_mask_flat,
    ).to_dict("records")
    joint_ranks = set(affected_ranks) | _joint_ranks_ready_to_compute(
        pd.DataFrame(merged_rows),
        config.rank_grid,
        n_folds,
        oof_spill_dir,
        primary_label="transfer",
        compute_joint_ridge=config.compute_joint_ridge,
    )
    _compute_and_save_joint_metrics(
        oof_spill_dir,
        sorted(joint_ranks),
        n_folds,
        transfer_dir,
        merged_rows,
        primary_label="transfer",
        sig_mask_flat=sig_mask_flat,
        compute_joint_ridge=config.compute_joint_ridge,
        device=device,
    )

    df = pd.DataFrame(merged_rows)
    if not df.empty:
        df = df.sort_values(["k", "fold"]).reset_index(drop=True)
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
        test_idxs = _load_reference_test_idxs(output_dir, train_subs)
        n_folds = len(test_idxs)
        train_dir = os.path.join(loso_sub_dir, "train_subjects")

        print(f"  Training subjects: {train_subs}")
        train_resume_plan = _rrr_output_resume_plan(train_dir, config, n_folds)
        train_folds_to_build = train_resume_plan["folds_to_build"]
        if train_folds_to_build:
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
                folds_to_build=train_folds_to_build,
                alpha_cache_dir=train_dir,
            )
            if results_train is None:
                print("  No training data; skipping this held-out subject.")
                continue
            X_raw, X_arbitrary, per_fold_baselines_train, test_idxs, sig_mask_train = (
                results_train
            )

            print(f"  Running train-subjects RRR → {train_dir}")
            _run_rrr_for_unit(
                rrr_unit_dir=train_dir,
                X_raw=X_raw,
                test_idxs=test_idxs,
                per_fold_baselines=per_fold_baselines_train,
                config=config,
                sig_mask_flat=sig_mask_train,
                X_arbitrary=X_arbitrary,
                ranks_by_fold=train_resume_plan["rrr_ranks_by_fold"],
                completed_pairs=train_resume_plan["rrr_completed_pairs"],
                pca_ranks_by_fold=train_resume_plan["pca_ranks_by_fold"],
                pca_completed_pairs=train_resume_plan["pca_completed_pairs"],
            )
        else:
            print("  Train-subject RRR/PCA outputs complete; skipping setup.")

        # --- Step B: build held-out subject data ---
        held_sig_elecs = sig_electrodes.get(held_out_sub, [])
        if not held_sig_elecs:
            print(
                f"  Subject {held_out_sub} has no significant electrodes; skipping transfer/baseline."
            )
            continue

        transfer_dir = os.path.join(loso_sub_dir, "transfer")
        baseline_dir = os.path.join(loso_sub_dir, "baseline")
        transfer_resume_plan = _transfer_output_resume_plan(
            transfer_dir, config, n_folds
        )
        baseline_resume_plan = _rrr_output_resume_plan(baseline_dir, config, n_folds)
        held_folds_to_build = (
            transfer_resume_plan["folds_to_build"]
            | baseline_resume_plan["folds_to_build"]
        )
        if not held_folds_to_build:
            print("  Transfer/baseline outputs complete; skipping held-out setup.")
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
            folds_to_build=held_folds_to_build,
            alpha_cache_dir=os.path.join(loso_sub_dir, "held_out_alpha"),
        )
        if results_held is None:
            print(f"  No held-out subject data; skipping transfer/baseline.")
            continue
        X_raw_held, X_arbitrary_held, per_fold_baselines_held, _, sig_mask_held = (
            results_held
        )

        # --- Step C: transfer A from train subjects to held-out subject ---
        print(f"  Running transfer evaluation → {transfer_dir}")
        _run_loso_transfer_for_unit(
            transfer_dir=transfer_dir,
            X_raw=X_raw_held,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines_held,
            train_rrr_dir=train_dir,
            config=config,
            sig_mask_flat=sig_mask_held,
            ranks_by_fold=transfer_resume_plan["ranks_by_fold"],
            completed_pairs=transfer_resume_plan["completed_pairs"],
        )

        # --- Step D: within-subject baseline RRR on held-out subject alone ---
        print(f"  Running within-subject baseline RRR → {baseline_dir}")
        _run_rrr_for_unit(
            rrr_unit_dir=baseline_dir,
            X_raw=X_raw_held,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines_held,
            config=config,
            sig_mask_flat=sig_mask_held,
            X_arbitrary=X_arbitrary_held,
            ranks_by_fold=baseline_resume_plan["rrr_ranks_by_fold"],
            completed_pairs=baseline_resume_plan["rrr_completed_pairs"],
            pca_ranks_by_fold=baseline_resume_plan["pca_ranks_by_fold"],
            pca_completed_pairs=baseline_resume_plan["pca_completed_pairs"],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_config(config: RRRConfig):
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


def main():
    args, overrides = parse_args("RRR: raw-input reduced rank regression baseline")
    config: RRRConfig = load_config(args.config, overrides, RRRConfig)
    configs = expand_per_model_layer_configs(config)
    for layer_config in configs:
        if len(configs) > 1:
            print(
                f"\nRunning RRR for layer "
                f"{layer_config.encoding.per_model_config['layer']}"
            )
        run_config(layer_config)


if __name__ == "__main__":
    main()
