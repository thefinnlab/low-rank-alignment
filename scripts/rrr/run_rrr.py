"""
run_rrr.py — Raw-Input Reduced Rank Regression + optional PCA baseline.

Performs RRR directly from raw embeddings to neural targets — no SVD projection.
Sweeps over a configured rank grid. Optionally also runs PCA subspace analysis
(when RRRConfig.run_pca_baseline=True) using the same rank grid and data splits.
By default, completed rank outputs in an existing run directory are reused;
set RRRConfig.refresh=True to recompute the full configured rank grid.

Six modes controlled by RRRConfig.mode:

  cross_subject (default):
    Builds global Y across all subjects (significant electrodes).
    Writes to {output_dir}/rrr/.

  cross_dataset:
    Loads fold-level A matrices from RRRConfig.cross_result_dir, averages them
    into one source-dataset subspace, and evaluates rank-k NR ridge models on
    the configured target dataset folds.
    Writes to {output_dir}/cross_dataset_rrr/{source_name}/.

  multi_view:
    Projects configured saved RRR views at fixed ranks, using raw fold-matched
    A matrices within the target dataset and one fold-aggregated SVD basis for
    cross-dataset views. Fits a banded-ridge full model before the usual RRR
    rank sweep.
    Writes to {output_dir}/multi_view_rrr/.

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
            A.npy                     (RRR only; full-rank input factors)
            B.npy                     (optional full-rank target factors)
            full_model_metrics.npz    (full ridge baseline, saved once per fold)
        k{k}/fold_{f}/
            metrics.npz               (per-rank metrics including *_per_target arrays)

Metrics (per k, per fold in grid_results.csv):
    RRR model  : rrr_corr, rrr_r2
    NR ridge   : nr_corr, nr_r2
    Effective rank: effective_rank

Full model (saved once per fold, not duplicated across k):
    full_corr, full_r2, full_corr_per_target  → fold_{f}/full_model_metrics.npz

Optional per-target correlation arrays (only in metrics.npz):
    nr_corr_per_target  (RRRConfig.save_nr_corr_per_target=True by default)
    rrr_corr_per_target (RRRConfig.save_rrr_corr_per_target=False by default)
"""

import json
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import (
    load_embeddings,
)
from src.utils.tensor_utils import to_numpy
from src.utils.correlation import (
    fisher_z_mean,
)
from src.utils.data_loaders import load_sig_electrodes
from src.utils.atlas import build_electrode_region_map
from src.core.config import RRRConfig, ScriptMode, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import (
    parse_args,
    load_config,
    expand_per_model_layer_configs,
    save_effective_config,
)
from src.utils.embeddings import get_glove_embeddings
from src.utils.encoding import (
    load_encoding_results,
    build_per_fold_baselines,
)
from src.utils.srrr import (
    fit_dense_rrr,
    save_rrr_factors,
    save_rrr_result,
)
from src.utils.rrr_utils import (
    add_sig_nonsig_corrs,
    make_sig_split_df,
)
from src.utils.subspace_utils import (
    evaluate_input_space,
)
from src.utils.subspace_compare import (
    aggregate_subspace_from_unit_dir,
    validate_aggregation_cache,
)
from src.utils.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
from src.utils.decomposition import _get_decomposition_inputs
from src.utils.multiview_rrr import (
    fit_banded_ridge_baseline,
    load_view_bases,
    load_view_embeddings,
    project_view_bands,
    resolve_rrr_views,
    target_dataset_key,
    union_sig_mask,
    union_view_significance,
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
    corr_arr = to_numpy(correlation_score(Y_test, Y_hat_full))
    r2_arr = to_numpy(r2_score(Y_test, Y_hat_full))
    np.savez(
        path,
        full_corr=float(fisher_z_mean(corr_arr)),
        full_r2=float(r2_arr.mean()),
        full_corr_per_target=corr_arr,
    )


def _resolve_pca_sibling_dir(rrr_unit_dir: str) -> str:
    """Return the PCA output dir that mirrors a given RRR unit dir."""
    for rrr_part, pca_part in [
        ("multi_view_rrr", "multi_view_pca"),
        ("cross_dataset_rrr", "cross_dataset_pca"),
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
            if arr.ndim > 1 and arr.shape[-1] == sig_mask_flat.shape[0]:
                sig_arr = arr[..., sig_mask_flat]
                nonsig_arr = arr[..., ~sig_mask_flat]
            else:
                sig_arr = arr[sig_mask_flat]
                nonsig_arr = arr[~sig_mask_flat]
            reducer = fisher_z_mean if base.endswith("_corr") else np.nanmean
            metrics[f"{base}_sig"] = (
                float(reducer(sig_arr)) if sig_mask_flat.any() else float("nan")
            )
            metrics[f"{base}_nonsig"] = (
                float(reducer(nonsig_arr)) if (~sig_mask_flat).any() else float("nan")
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


def _load_shared_word_selection_metadata(output_dir: str, subject_ids: list[int]) -> dict:
    """Load and validate shared word-selection metadata from encoding results."""
    reference = None
    for sub_id in subject_ids:
        path = os.path.join(
            output_dir, "subjects", f"sub_{sub_id}", "encoding_results.joblib"
        )
        enc_results = load_encoding_results(path)
        metadata = enc_results.get("word_selection")
        if metadata is None:
            raise ValueError(
                "RRRConfig.encoding.shared_word_selection=True requires "
                f"word_selection metadata in {path}. Re-run significance or encoding "
                "with shared_word_selection enabled."
            )
        stored_subjects = [int(s) for s in metadata.get("subjects", [])]
        if stored_subjects != [int(s) for s in subject_ids]:
            raise ValueError(
                "Shared word-selection metadata subjects do not match current RRR "
                f"subjects. Stored={stored_subjects}, current={list(subject_ids)}."
            )
        if metadata.get("mode") != "shared_intersection":
            raise ValueError(
                f"Expected shared_intersection word selection in {path}, "
                f"got {metadata.get('mode')!r}."
            )
        if reference is None:
            reference = metadata
            continue
        for key in ("epoch_selection", "analysis_selection"):
            if not np.array_equal(reference[key], metadata[key]):
                raise ValueError(
                    "Shared word-selection metadata differs across subject encoding "
                    f"results for key {key!r}."
                )
    if reference is None:
        raise ValueError("Cannot load shared word-selection metadata for no subjects.")
    return reference


def _metrics_path(unit_dir: str, k: int, fold_f: int) -> str:
    return os.path.join(unit_dir, f"k{k}", f"fold_{fold_f}", "metrics.npz")


def _fold_factors_exist(unit_dir: str, fold_f: int, require_b: bool = True) -> bool:
    fold_dir = os.path.join(unit_dir, f"fold_{fold_f}")
    paths = [os.path.join(fold_dir, "A.npy")]
    if require_b:
        paths.append(os.path.join(fold_dir, "B.npy"))
    return all(os.path.exists(path) for path in paths)


def _completed_pairs_from_artifacts(
    unit_dir: str,
    rank_grid: list[int],
    n_folds: int,
    require_factors: bool = False,
    require_b: bool = True,
    required_metric_keys: "set[str] | None" = None,
) -> set[tuple[int, int]]:
    """Completed (k, fold) pairs based on artifact files, not grid_results.csv."""
    completed: set[tuple[int, int]] = set()
    for k in rank_grid:
        for fold_f in range(n_folds):
            metrics_path = _metrics_path(unit_dir, k, fold_f)
            if not os.path.exists(metrics_path):
                continue
            if require_factors and not _fold_factors_exist(
                unit_dir, fold_f, require_b=require_b
            ):
                continue
            if required_metric_keys:
                try:
                    with np.load(metrics_path, allow_pickle=True) as metrics:
                        if not required_metric_keys.issubset(set(metrics.files)):
                            continue
                except Exception:
                    continue
            completed.add((k, fold_f))
    return completed


def _ranks_by_fold_to_run(
    unit_dir: str,
    rank_grid: list[int],
    n_folds: int,
    refresh: bool,
    require_factors: bool = False,
    require_b: bool = True,
    required_metric_keys: "set[str] | None" = None,
) -> tuple[dict[int, list[int]], set[tuple[int, int]]]:
    """Return per-fold ranks to compute plus (k, fold) pairs reusable from disk."""
    if refresh:
        return {fold_f: list(rank_grid) for fold_f in range(n_folds)}, set()
    completed = _completed_pairs_from_artifacts(
        unit_dir,
        rank_grid,
        n_folds,
        require_factors=require_factors,
        require_b=require_b,
        required_metric_keys=required_metric_keys,
    )
    ranks_by_fold = {
        fold_f: [k for k in rank_grid if (k, fold_f) not in completed]
        for fold_f in range(n_folds)
    }
    return ranks_by_fold, completed


def _folds_to_build_from_ranks(ranks_by_fold: dict[int, list[int]]) -> set[int]:
    return {fold_f for fold_f, ranks in ranks_by_fold.items() if ranks}


def _X_for_fold(X_raw, fold_f: int) -> np.ndarray:
    """Return one fold's design from a shared matrix or per-fold matrix list."""
    return np.asarray(X_raw[fold_f] if isinstance(X_raw, list) else X_raw)


def _unit_ranks_by_fold_to_run(
    unit_dir: str,
    rank_grid: list[int],
    n_folds: int,
    refresh: bool,
    require_factors: bool,
    require_b: bool = True,
    required_metric_keys: "set[str] | None" = None,
) -> tuple[dict[int, list[int]], set[tuple[int, int]]]:
    return _ranks_by_fold_to_run(
        unit_dir,
        rank_grid,
        n_folds,
        refresh=refresh,
        require_factors=require_factors,
        require_b=require_b,
        required_metric_keys=required_metric_keys,
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
        require_b=config.save_b_matrices,
        required_metric_keys=(
            {"null_aug_corr_per_target", "null_aug_r2_per_target"}
            if config.run_null_augmented
            else None
        ),
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
        if not arr.shape:
            return float(arr)
        return float(fisher_z_mean(arr) if key.endswith("_corr") else np.nanmean(arr))
    per_target_key = f"{key}_per_target"
    if per_target_key in metrics:
        arr = np.asarray(metrics[per_target_key])
        return float(fisher_z_mean(arr) if key.endswith("_corr") else np.nanmean(arr))
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

    if include_effective_rank and _fold_factors_exist(unit_dir, fold_f, require_b=False):
        A_full = np.load(os.path.join(unit_dir, f"fold_{fold_f}", "A.npy"))
        A_k = A_full[:, :k]
        sv = np.linalg.svd(A_k, compute_uv=False)
        row["effective_rank"] = (
            int(np.sum(sv > 1e-6 * sv[0])) if len(sv) > 0 and sv[0] > 0 else 0
        )
    elif include_effective_rank and "effective_rank" in metrics:
        row["effective_rank"] = int(np.asarray(metrics["effective_rank"]).item())

    base_keys = [
        "full_corr",
        "full_r2",
        f"{primary_label}_corr",
        f"{primary_label}_r2",
        "null_aug_corr",
        "null_aug_r2",
        "null_aug_corr_q05",
        "null_aug_corr_q50",
        "null_aug_corr_q95",
        "null_aug_full_advantage_within_q95",
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
                if arr.ndim > 1 and arr.shape[-1] == sig_mask_flat.shape[0]:
                    sig_arr = arr[..., sig_mask_flat]
                    nonsig_arr = arr[..., ~sig_mask_flat]
                else:
                    sig_arr = arr[sig_mask_flat]
                    nonsig_arr = arr[~sig_mask_flat]
                row[f"{key}_sig"] = (
                    float(fisher_z_mean(sig_arr))
                    if sig_mask_flat.any()
                    else float("nan")
                )
                row[f"{key}_nonsig"] = (
                    float(fisher_z_mean(nonsig_arr))
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


def _run_common_subspace_fold_k(
    A_full: np.ndarray,
    k: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    sig_mask_flat: "np.ndarray | None",
    fold_f: int,
    alphas: "np.ndarray | None",
    alpha_policy: dict | None,
    standardize_subspace_inputs: bool,
    ridge_cv_folds: int,
    ridge_solver_params: dict | None,
    ridge_y_in_cpu: bool | None,
    ridge_force_cpu: bool | None,
    device: str,
    unit_dir: str,
    primary_label: str,
    full_corr_per_target: "np.ndarray | None" = None,
    null_augmented_config: "RRRConfig | None" = None,
    save_primary_corr_per_target: bool = True,
) -> dict:
    """Shared evaluation for one (fold, k) projected subspace.

    Called by both _run_rrr_for_unit (primary_label='nr') and _run_pca_for_unit
    (primary_label='pca') after the reduction matrix A_full is known.

    Saves per-rank metrics (including *_per_target arrays) to
    {unit_dir}/k{k}/fold_{fold_f}/metrics.npz.

    Returns a metrics dict with per-target arrays stripped (for grid_results.csv).
    """
    A_k = A_full[:, :k]

    X_proj_train = X_train @ A_k
    X_proj_test = X_test @ A_k
    rank_dir = os.path.join(unit_dir, f"k{k}", f"fold_{fold_f}")
    context = {
        "script": "run_rrr",
        "role": primary_label,
        "fold": int(fold_f),
        "k": int(k),
        "n_features": int(X_proj_train.shape[1]),
        "standardize": bool(standardize_subspace_inputs),
    }
    resolved_alphas = resolve_ridge_alphas(
        alphas,
        X_proj_train,
        standardize=standardize_subspace_inputs,
        context=context,
        cache_dir=os.path.join(rank_dir, f"{primary_label}_alpha"),
        policy=alpha_policy,
    )
    result = evaluate_input_space(
        X_proj_train,
        X_proj_test,
        Y_train,
        Y_test,
        subspace_name=primary_label,
        alphas=resolved_alphas,
        ridge_cv_folds=ridge_cv_folds,
        standardize=standardize_subspace_inputs,
        ridge_solver_params=ridge_solver_params,
        ridge_y_in_cpu=ridge_y_in_cpu,
        ridge_force_cpu=ridge_force_cpu,
        return_alphas=True,
        device=device,
    )
    save_alpha_diagnostics(
        os.path.join(rank_dir, f"{primary_label}_alpha"),
        resolved_alphas,
        best_alphas=result[f"{primary_label}_best_alphas"],
        context=context,
        policy=alpha_policy,
    )

    metrics = {
        f"{primary_label}_corr": result[f"{primary_label}_corr"],
        f"{primary_label}_r2": result[f"{primary_label}_r2"],
        f"{primary_label}_r2_per_target": result[f"{primary_label}_r2_per_target"],
        f"{primary_label}_best_alphas": result[f"{primary_label}_best_alphas"],
    }
    if save_primary_corr_per_target:
        metrics[f"{primary_label}_corr_per_target"] = result[
            f"{primary_label}_corr_per_target"
        ]
    if null_augmented_config is not None and null_augmented_config.run_null_augmented:
        if full_corr_per_target is None:
            raise ValueError(
                "full_corr_per_target is required when run_null_augmented=True"
            )
        metrics.update(
            _run_null_augmented_fold_k(
                A_full,
                k,
                X_train,
                X_test,
                Y_train,
                Y_test,
                full_corr_per_target=full_corr_per_target,
                primary_corr_per_target=result[f"{primary_label}_corr_per_target"],
                fold_f=fold_f,
                alphas=alphas,
                alpha_policy=alpha_policy,
                standardize_subspace_inputs=standardize_subspace_inputs,
                ridge_cv_folds=ridge_cv_folds,
                ridge_solver_params=ridge_solver_params,
                ridge_y_in_cpu=ridge_y_in_cpu,
                ridge_force_cpu=ridge_force_cpu,
                device=device,
                unit_dir=unit_dir,
                config=null_augmented_config,
            )
        )

    save_rrr_result(unit_dir, k=k, fold=fold_f, metrics=metrics)
    _strip_per_target_and_add_sig_nonsig(metrics, sig_mask_flat)
    for artifact_key in (
        "null_aug_shifts_train",
        "null_aug_shifts_test",
        "null_aug_reference_rank",
    ):
        metrics.pop(artifact_key, None)
    return metrics


def _run_subspace_rank_loop_for_fold(
    *,
    unit_dir: str,
    A_full: np.ndarray,
    ranks_to_run: list[int],
    X_raw: np.ndarray,
    baseline: dict,
    sig_mask_flat: "np.ndarray | None",
    fold_f: int,
    alphas: "np.ndarray | None",
    alpha_policy: dict | None,
    standardize_subspace_inputs: bool,
    ridge_cv_folds: int,
    ridge_solver_params: dict | None,
    ridge_y_in_cpu: bool | None,
    ridge_force_cpu: bool | None,
    device: str,
    primary_label: str,
    progress_desc: str,
    B_full: "np.ndarray | None" = None,
    null_augmented_config: "RRRConfig | None" = None,
    save_primary_corr_per_target: bool = True,
    save_rrr_corr_per_target: bool = False,
) -> list[dict]:
    """Evaluate rank-k projected ridge models for one fold from a known A matrix.

    ``B_full`` is provided only for native RRR runs, where raw RRR predictions are
    also evaluated alongside NR.
    """
    train_idx = baseline["train_idx"]
    test_idx = baseline["test_idx"]
    X_train = X_raw[train_idx]
    X_test = X_raw[test_idx]
    Y_train = baseline["Y_train"]
    Y_test = baseline["Y_test"]
    Y_hat_full = baseline["Y_hat_full_test"]

    _save_full_model_metrics_once(unit_dir, fold_f, Y_test, Y_hat_full)
    full_corr_per_target = to_numpy(correlation_score(Y_test, Y_hat_full))
    full_corr = float(fisher_z_mean(full_corr_per_target))
    full_r2 = float(to_numpy(r2_score(Y_test, Y_hat_full)).mean())

    valid_ranks = [k for k in ranks_to_run if k <= A_full.shape[1]]
    if len(valid_ranks) < len(ranks_to_run):
        skipped = [k for k in ranks_to_run if k > A_full.shape[1]]
        print(f"  Skipping ranks {skipped} (> {A_full.shape[1]} source directions)")

    print(f"  Running {len(valid_ranks)} rank(s): {valid_ranks}")
    rows = []
    for k in tqdm(valid_ranks, desc=progress_desc, leave=True, unit="rank"):
        A_k = A_full[:, :k]
        sv = np.linalg.svd(A_k, compute_uv=False)
        effective_rank = (
            int(np.sum(sv > 1e-6 * sv[0])) if len(sv) > 0 and sv[0] > 0 else 0
        )

        common = _run_common_subspace_fold_k(
            A_full,
            k,
            X_train,
            X_test,
            Y_train,
            Y_test,
            sig_mask_flat,
            fold_f,
            alphas,
            alpha_policy,
            standardize_subspace_inputs,
            ridge_cv_folds,
            ridge_solver_params,
            ridge_y_in_cpu,
            ridge_force_cpu,
            device,
            unit_dir,
            primary_label=primary_label,
            full_corr_per_target=full_corr_per_target,
            null_augmented_config=null_augmented_config,
            save_primary_corr_per_target=save_primary_corr_per_target,
        )

        metrics_path = os.path.join(unit_dir, f"k{k}", f"fold_{fold_f}", "metrics.npz")
        existing = dict(np.load(metrics_path, allow_pickle=True))
        existing["full_corr"] = full_corr
        existing["full_r2"] = full_r2
        existing["effective_rank"] = np.array(effective_rank, dtype=np.int64)

        row = {
            "fold": fold_f,
            "k": k,
            "effective_rank": effective_rank,
            "full_corr": full_corr,
            "full_r2": full_r2,
            **common,
        }

        if B_full is not None:
            B_k = B_full[:, :k]
            Y_hat_rrr = Y_hat_full @ B_k @ B_k.T
            rrr_corr_arr = to_numpy(correlation_score(Y_test, Y_hat_rrr))
            rrr_r2_arr = to_numpy(r2_score(Y_test, Y_hat_rrr))
            rrr_corr = float(fisher_z_mean(rrr_corr_arr))
            existing["rrr_corr"] = rrr_corr
            if save_rrr_corr_per_target:
                existing["rrr_corr_per_target"] = rrr_corr_arr
            existing["rrr_r2_per_target"] = rrr_r2_arr
            existing["rrr_r2"] = float(rrr_r2_arr.mean())
            row["rrr_corr"] = rrr_corr
            row["rrr_r2"] = float(existing["rrr_r2"])
            if sig_mask_flat is not None:
                rrr_corr_sig = (
                    float(fisher_z_mean(rrr_corr_arr[sig_mask_flat]))
                    if sig_mask_flat.any()
                    else float("nan")
                )
                rrr_corr_nonsig = (
                    float(fisher_z_mean(rrr_corr_arr[~sig_mask_flat]))
                    if (~sig_mask_flat).any()
                    else float("nan")
                )
                existing["rrr_corr_sig"] = rrr_corr_sig
                existing["rrr_corr_nonsig"] = rrr_corr_nonsig
                row["rrr_corr_sig"] = rrr_corr_sig
                row["rrr_corr_nonsig"] = rrr_corr_nonsig

        np.savez(metrics_path, **existing)
        rows.append(row)

    return rows


def _sample_valid_circular_shift(
    split_len: int,
    min_shift: int,
    rng: np.random.Generator,
) -> int:
    """Sample a non-trivial circular shift respecting a minimum distance."""
    min_shift = int(min_shift)
    if min_shift < 0:
        raise ValueError(
            f"null_augmented_min_shift must be non-negative, got {min_shift}"
        )
    if split_len <= 1:
        raise ValueError("Cannot circularly shift a split with fewer than 2 rows.")
    if min_shift == 0:
        low = 1
        high = split_len
    else:
        low = min_shift
        high = split_len - min_shift
    if high <= low:
        raise ValueError(
            "Split is too short for null-augmented circular shifts: "
            f"split_len={split_len}, min_shift={min_shift}. "
            "Need split_len > 2 * min_shift."
        )
    return int(rng.integers(low, high))


def _null_augmented_reference_rank(A_full: np.ndarray, config: RRRConfig) -> int:
    """Resolve and validate the rank used as the low+tail augmented width."""
    requested = (
        max(config.rank_grid)
        if config.null_augmented_reference_rank is None
        else int(config.null_augmented_reference_rank)
    )
    if requested <= 0:
        raise ValueError(
            f"null_augmented_reference_rank must be positive, got {requested}"
        )
    if requested > A_full.shape[1]:
        raise ValueError(
            "null_augmented_reference_rank exceeds fitted RRR factor width: "
            f"{requested} > {A_full.shape[1]}"
        )
    return requested


def _run_null_augmented_fold_k(
    A_full: np.ndarray,
    k: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    full_corr_per_target: np.ndarray,
    primary_corr_per_target: np.ndarray,
    fold_f: int,
    alphas: "np.ndarray | None",
    alpha_policy: dict | None,
    standardize_subspace_inputs: bool,
    ridge_cv_folds: int,
    ridge_solver_params: dict | None,
    ridge_y_in_cpu: bool | None,
    ridge_force_cpu: bool | None,
    device: str,
    unit_dir: str,
    config: RRRConfig,
) -> dict:
    """Evaluate null-augmented models by appending shifted omitted coordinates."""
    reference_rank = _null_augmented_reference_rank(A_full, config)
    if k > reference_rank:
        raise ValueError(
            f"Rank k={k} exceeds null_augmented_reference_rank={reference_rank}"
        )

    n_repeats = int(config.null_augmented_repeats)
    if n_repeats <= 0:
        raise ValueError(f"null_augmented_repeats must be positive, got {n_repeats}")

    X_low_train = X_train @ A_full[:, :k]
    X_low_test = X_test @ A_full[:, :k]
    X_tail_train = X_train @ A_full[:, k:reference_rank]
    X_tail_test = X_test @ A_full[:, k:reference_rank]
    has_tail = X_tail_train.shape[1] > 0
    if not has_tail:
        n_targets = Y_test.shape[1]
        empty_target = np.empty((0, n_targets), dtype=np.float32)
        empty_repeat = np.empty((0,), dtype=np.int64)
        return {
            "null_aug_corr_per_target": empty_target,
            "null_aug_r2_per_target": empty_target.copy(),
            "null_aug_shifts_train": empty_repeat,
            "null_aug_shifts_test": empty_repeat.copy(),
            "null_aug_reference_rank": np.array(reference_rank, dtype=np.int64),
            "null_aug_corr": float("nan"),
            "null_aug_r2": float("nan"),
            "null_aug_corr_q05": float("nan"),
            "null_aug_corr_q50": float("nan"),
            "null_aug_corr_q95": float("nan"),
            "null_aug_full_advantage_within_q95": float("nan"),
        }

    corr_repeats = []
    r2_repeats = []
    train_shifts = np.zeros(n_repeats, dtype=np.int64)
    test_shifts = np.zeros(n_repeats, dtype=np.int64)

    seed = int(config.null_augmented_seed) + 1000003 * int(fold_f) + 9176 * int(k)
    rng = np.random.default_rng(seed)

    for repeat_i in range(n_repeats):
        train_shift = _sample_valid_circular_shift(
            X_tail_train.shape[0], config.null_augmented_min_shift, rng
        )
        test_shift = _sample_valid_circular_shift(
            X_tail_test.shape[0], config.null_augmented_min_shift, rng
        )
        tail_train = np.roll(X_tail_train, shift=train_shift, axis=0)
        tail_test = np.roll(X_tail_test, shift=test_shift, axis=0)
        X_aug_train = np.concatenate([X_low_train, tail_train], axis=1)
        X_aug_test = np.concatenate([X_low_test, tail_test], axis=1)
        train_shifts[repeat_i] = train_shift
        test_shifts[repeat_i] = test_shift

        rank_dir = os.path.join(unit_dir, f"k{k}", f"fold_{fold_f}")
        context = {
            "script": "run_rrr",
            "role": "null_aug",
            "fold": int(fold_f),
            "k": int(k),
            "repeat": int(repeat_i),
            "reference_rank": int(reference_rank),
            "n_features": int(X_aug_train.shape[1]),
            "standardize": bool(standardize_subspace_inputs),
        }
        resolved_alphas = resolve_ridge_alphas(
            alphas,
            X_aug_train,
            standardize=standardize_subspace_inputs,
            context=context,
            policy=alpha_policy,
        )
        result = evaluate_input_space(
            X_aug_train,
            X_aug_test,
            Y_train,
            Y_test,
            subspace_name="null_aug",
            alphas=resolved_alphas,
            ridge_cv_folds=ridge_cv_folds,
            standardize=standardize_subspace_inputs,
            ridge_solver_params=ridge_solver_params,
            ridge_y_in_cpu=ridge_y_in_cpu,
            ridge_force_cpu=ridge_force_cpu,
            return_alphas=True,
            device=device,
        )
        corr_repeats.append(result["null_aug_corr_per_target"])
        r2_repeats.append(result["null_aug_r2_per_target"])

    corr_arr = np.stack(corr_repeats, axis=0)
    r2_arr = np.stack(r2_repeats, axis=0)
    full_advantage = full_corr_per_target[np.newaxis, :] - corr_arr
    observed_advantage = full_corr_per_target - primary_corr_per_target
    upper = np.nanpercentile(full_advantage, 95, axis=0)
    finite = (
        np.isfinite(observed_advantage) & np.isfinite(upper) & (observed_advantage > 0)
    )
    frac_within = (
        float(np.nanmean(observed_advantage[finite] <= upper[finite]))
        if finite.any()
        else float("nan")
    )

    return {
        "null_aug_corr_per_target": corr_arr,
        "null_aug_r2_per_target": r2_arr,
        "null_aug_shifts_train": train_shifts,
        "null_aug_shifts_test": test_shifts,
        "null_aug_reference_rank": np.array(reference_rank, dtype=np.int64),
        "null_aug_corr": float(fisher_z_mean(corr_arr.reshape(-1))),
        "null_aug_r2": float(np.nanmean(r2_arr)),
        "null_aug_corr_q05": float(np.nanpercentile(corr_arr, 5)),
        "null_aug_corr_q50": float(np.nanpercentile(corr_arr, 50)),
        "null_aug_corr_q95": float(np.nanpercentile(corr_arr, 95)),
        "null_aug_full_advantage_within_q95": frac_within,
    }


def _build_results(
    df: pd.DataFrame,
    sig_mask_flat: "np.ndarray | None",
    unit_dir: str,
    primary_label: str,
) -> None:
    """Build summary CSV and rank analysis plots from grid results DataFrame."""

    def _is_corr_mean_col(col: str) -> bool:
        return (
            col.endswith("_corr")
            or col.endswith("_corr_sig")
            or col.endswith("_corr_nonsig")
        )

    # Core agg columns (filter to what's actually present)
    candidate_cols = [
        "full_corr",
        "full_r2",
        f"{primary_label}_corr",
        f"{primary_label}_r2",
        "effective_rank",
        "null_aug_corr",
        "null_aug_r2",
        "null_aug_corr_q05",
        "null_aug_corr_q50",
        "null_aug_corr_q95",
        "null_aug_full_advantage_within_q95",
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
    summary_rows = []
    for k, group in df.groupby("k", sort=True):
        row = {"k": k}
        for col in agg_cols:
            values = group[col].to_numpy()
            row[col] = (
                float(fisher_z_mean(values))
                if _is_corr_mean_col(col)
                else (
                    float(np.nanmean(values))
                    if np.isfinite(values).any()
                    else float("nan")
                )
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
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
    X_raw: "np.ndarray | list[np.ndarray]",
    test_idxs: list,
    per_fold_baselines: list,
    config: RRRConfig,
    sig_mask_flat: "np.ndarray | None" = None,
    ranks_by_fold: "dict[int, list[int]] | None" = None,
    completed_pairs: "set[tuple[int, int]] | None" = None,
    pca_ranks_by_fold: "dict[int, list[int]] | None" = None,
    pca_completed_pairs: "set[tuple[int, int]] | None" = None,
) -> pd.DataFrame:
    """Core RRR loop for one analysis unit (cross-subject, per-subject, or per-region).

    Fits RRR once at max rank per fold, sweeps the rank grid, and evaluates:
        - Raw RRR predictions (via B matrix projection)
        - NR ridge from k-dim RRR subspace (via A matrix projection)

    Args:
        rrr_unit_dir: Root output directory for this unit.
        X_raw: (n_words, n_embed) raw embedding matrix.
        test_idxs: List of test index arrays, one per fold.
        per_fold_baselines: Output of build_per_fold_baselines.
        config: RRRConfig.
        sig_mask_flat: Boolean mask over targets (True = significant).

    Returns:
        DataFrame of per-(fold, k) metrics for grid_results.csv.
    """
    os.makedirs(rrr_unit_dir, exist_ok=True)

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None

    n_folds = len(test_idxs)
    n_embed = _X_for_fold(X_raw, 0).shape[1]
    if any(_X_for_fold(X_raw, fold).shape[1] != n_embed for fold in range(n_folds)):
        raise ValueError("All fold-specific RRR designs must share feature width")
    selected_idx = np.arange(n_embed)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    existing_grid = _load_existing_grid(rrr_unit_dir)
    if ranks_by_fold is None or completed_pairs is None:
        ranks_by_fold, completed_pairs = _unit_ranks_by_fold_to_run(
            rrr_unit_dir,
            config.rank_grid,
            n_folds,
            refresh=config.refresh,
            require_factors=True,
            require_b=config.save_b_matrices,
            required_metric_keys=(
                {"null_aug_corr_per_target", "null_aug_r2_per_target"}
                if config.run_null_augmented
                else None
            ),
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
                alphas=alphas,
                alpha_policy=config.encoding.alpha_grid,
                standardize_subspace_inputs=config.standardize_subspace_inputs,
                ridge_cv_folds=config.encoding.ridge_cv_folds,
                ridge_solver_params=config.encoding.ridge_solver_params,
                ridge_y_in_cpu=config.encoding.ridge_y_in_cpu,
                ridge_force_cpu=config.encoding.ridge_force_cpu,
                device=device,
                refresh=config.refresh,
                ranks_by_fold=pca_ranks_by_fold,
                completed_pairs=pca_completed_pairs,
            )
        return df

    all_rows = []
    for fold_f in range(n_folds):
        ranks_to_run = ranks_by_fold[fold_f]
        if not ranks_to_run:
            print(f"\n[Fold {fold_f + 1}/{n_folds}] complete; skipping")
            continue

        fold_start = time.time()
        print(f"\n[Fold {fold_f + 1}/{n_folds}]")

        baseline = per_fold_baselines[fold_f]
        train_idx = baseline["train_idx"]
        X_fold = _X_for_fold(X_raw, fold_f)
        X_train = X_fold[train_idx]
        Y_train = baseline["Y_train"]
        Y_hat_ols_train = baseline["Y_hat_full_train"]

        # Fit or load RRR once per fold; truncate to each k below.
        max_k = min(X_train.shape[1], Y_train.shape[1])
        fold_dir = os.path.join(rrr_unit_dir, f"fold_{fold_f}")
        a_path = os.path.join(fold_dir, "A.npy")
        b_path = os.path.join(fold_dir, "B.npy")
        required_factors_exist = _fold_factors_exist(
            rrr_unit_dir, fold_f, require_b=config.save_b_matrices
        )
        if not config.refresh and required_factors_exist:
            A_full = np.load(a_path)
            B_full = np.load(b_path) if config.save_b_matrices else None
            factors_too_small = A_full.shape[1] < max_k or (
                B_full is not None and B_full.shape[1] < max_k
            )
            if factors_too_small:
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
                B_full = (
                    rrr_result["B_dense"] if config.save_b_matrices else None
                )
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
            B_full = rrr_result["B_dense"] if config.save_b_matrices else None
            if config.refresh and not config.save_b_matrices and os.path.exists(b_path):
                os.remove(b_path)
            save_rrr_factors(rrr_unit_dir, fold=fold_f, A=A_full, B=B_full)

        all_rows.extend(
            _run_subspace_rank_loop_for_fold(
                unit_dir=rrr_unit_dir,
                A_full=A_full,
                B_full=B_full,
                ranks_to_run=ranks_to_run,
                X_raw=X_fold,
                baseline=baseline,
                sig_mask_flat=sig_mask_flat,
                fold_f=fold_f,
                alphas=alphas,
                alpha_policy=config.encoding.alpha_grid,
                standardize_subspace_inputs=config.standardize_subspace_inputs,
                ridge_cv_folds=config.encoding.ridge_cv_folds,
                ridge_solver_params=config.encoding.ridge_solver_params,
                ridge_y_in_cpu=config.encoding.ridge_y_in_cpu,
                ridge_force_cpu=config.encoding.ridge_force_cpu,
                device=device,
                primary_label="nr",
                progress_desc=f"  fold {fold_f}",
                null_augmented_config=config,
                save_primary_corr_per_target=config.save_nr_corr_per_target,
                save_rrr_corr_per_target=config.save_rrr_corr_per_target,
            )
        )

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        config.rank_grid,
        unit_dir=rrr_unit_dir,
        primary_label="nr",
        sig_mask_flat=sig_mask_flat,
        include_effective_rank=True,
    )
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
            alphas=alphas,
            alpha_policy=config.encoding.alpha_grid,
            standardize_subspace_inputs=config.standardize_subspace_inputs,
            ridge_cv_folds=config.encoding.ridge_cv_folds,
            ridge_solver_params=config.encoding.ridge_solver_params,
            ridge_y_in_cpu=config.encoding.ridge_y_in_cpu,
            ridge_force_cpu=config.encoding.ridge_force_cpu,
            device=device,
            refresh=config.refresh,
            ranks_by_fold=pca_ranks_by_fold,
            completed_pairs=pca_completed_pairs,
        )

    return df


def _run_pca_for_unit(
    pca_unit_dir: str,
    X_raw: "np.ndarray | list[np.ndarray]",
    test_idxs: list,
    per_fold_baselines: list,
    rank_grid: list,
    sig_mask_flat: "np.ndarray | None" = None,
    alphas: "np.ndarray | None" = None,
    alpha_policy: dict | None = None,
    standardize_subspace_inputs: bool = False,
    ridge_cv_folds: int = 5,
    ridge_solver_params: dict | None = None,
    ridge_y_in_cpu: bool | None = None,
    ridge_force_cpu: bool | None = None,
    device: str = "cpu",
    refresh: bool = False,
    ranks_by_fold: "dict[int, list[int]] | None" = None,
    completed_pairs: "set[tuple[int, int]] | None" = None,
) -> pd.DataFrame:
    """PCA subspace analysis for one unit — structurally parallel to _run_rrr_for_unit.

    Computes PCA on training embeddings (per fold, proper CV), then evaluates:
        - Ridge from k-dim PCA subspace (primary model)

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
        alphas: Ridge alpha grid; None means solver default.
        device: Torch device string.

    Returns:
        DataFrame of per-(fold, k) metrics for grid_results.csv.
    """
    os.makedirs(pca_unit_dir, exist_ok=True)

    n_folds = len(test_idxs)
    existing_grid = _load_existing_grid(pca_unit_dir)
    if ranks_by_fold is None or completed_pairs is None:
        ranks_by_fold, completed_pairs = _unit_ranks_by_fold_to_run(
            pca_unit_dir,
            rank_grid,
            n_folds,
            refresh=refresh,
            require_factors=False,
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
        df.to_csv(os.path.join(pca_unit_dir, "grid_results.csv"), index=False)
        _build_results(df, sig_mask_flat, pca_unit_dir, primary_label="pca")
        return df

    all_rows = []
    for fold_f in range(n_folds):
        ranks_to_run = ranks_by_fold[fold_f]
        if not ranks_to_run:
            print(f"\n[Fold {fold_f + 1}/{n_folds}] complete; skipping")
            continue

        fold_start = time.time()
        print(f"\n[Fold {fold_f + 1}/{n_folds}]")

        baseline = per_fold_baselines[fold_f]
        X_fold = _X_for_fold(X_raw, fold_f)
        train_idx = baseline["train_idx"]
        test_idx = baseline["test_idx"]
        X_train = X_fold[train_idx]
        X_test = X_fold[test_idx]
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]
        Y_hat_full = baseline["Y_hat_full_test"]

        # Save full model metrics once per fold (not per-k)
        _save_full_model_metrics_once(pca_unit_dir, fold_f, Y_test, Y_hat_full)
        full_corr = float(
            fisher_z_mean(to_numpy(correlation_score(Y_test, Y_hat_full)))
        )
        full_r2 = float(to_numpy(r2_score(Y_test, Y_hat_full)).mean())

        # Compute PCA on training data only (proper CV)
        X_mean = X_train.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_train - X_mean, full_matrices=False)
        A_full = Vt.T  # (n_embed, min(n_train, n_embed))
        fold_dir = os.path.join(pca_unit_dir, f"fold_{fold_f}")
        os.makedirs(fold_dir, exist_ok=True)
        np.save(os.path.join(fold_dir, "A.npy"), A_full)
        max_components = A_full.shape[1]

        valid_ranks = [k for k in ranks_to_run if k <= max_components]
        if len(valid_ranks) < len(ranks_to_run):
            skipped = [k for k in ranks_to_run if k > max_components]
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
                sig_mask_flat,
                fold_f,
                alphas,
                alpha_policy,
                standardize_subspace_inputs,
                ridge_cv_folds,
                ridge_solver_params,
                ridge_y_in_cpu,
                ridge_force_cpu,
                device,
                pca_unit_dir,
                primary_label="pca",
            )
            metrics_path = os.path.join(
                pca_unit_dir, f"k{k}", f"fold_{fold_f}", "metrics.npz"
            )
            existing = dict(np.load(metrics_path, allow_pickle=True))
            existing["full_corr"] = full_corr
            existing["full_r2"] = full_r2
            np.savez(metrics_path, **existing)

            row = {
                "fold": fold_f,
                "k": k,
                "full_corr": full_corr,
                "full_r2": full_r2,
                **common,
            }
            all_rows.append(row)

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        rank_grid,
        unit_dir=pca_unit_dir,
        primary_label="pca",
        sig_mask_flat=sig_mask_flat,
    )
    df.to_csv(os.path.join(pca_unit_dir, "grid_results.csv"), index=False)
    _build_results(df, sig_mask_flat, pca_unit_dir, primary_label="pca")

    return df


# ---------------------------------------------------------------------------
# Data preparation helper
# ---------------------------------------------------------------------------


def _build_rrr_decomposition_inputs(
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
    shared_selection_metadata: "dict | None" = None,
):
    prepared = _get_decomposition_inputs(
        subject_ids,
        output_dir,
        sig_electrodes,
        encoding_config,
        embeddings,
        compute_sig_nonsig=compute_sig_nonsig,
        sig_dir=sig_dir,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
    )
    if prepared is None:
        return None
    X_raw = next(iter(prepared.embeddings.values()))
    print(f"  X: {X_raw.shape}, Y: {prepared.targets.shape}")
    per_fold_baselines = build_per_fold_baselines(
        X_raw,
        prepared.targets,
        prepared.test_idxs,
        alphas=(
            np.array(encoding_config.ridge_alphas, dtype=float)
            if encoding_config.ridge_alphas is not None
            else None
        ),
        ridge_cv_folds=encoding_config.ridge_cv_folds,
        ridge_solver_params=encoding_config.ridge_solver_params,
        ridge_y_in_cpu=encoding_config.ridge_y_in_cpu,
        ridge_force_cpu=encoding_config.ridge_force_cpu,
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

    return X_raw, per_fold_baselines, prepared.test_idxs, prepared.sig_mask_flat


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
    shared_selection_metadata=None,
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

    results = _build_rrr_decomposition_inputs(
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
        shared_selection_metadata=shared_selection_metadata,
    )
    if results is None:
        return
    X_raw, per_fold_baselines, test_idxs, sig_mask_flat = results
    _run_rrr_for_unit(
        rrr_unit_dir=rrr_dir,
        X_raw=X_raw,
        test_idxs=test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
        ranks_by_fold=resume_plan["rrr_ranks_by_fold"],
        completed_pairs=resume_plan["rrr_completed_pairs"],
        pca_ranks_by_fold=resume_plan["pca_ranks_by_fold"],
        pca_completed_pairs=resume_plan["pca_completed_pairs"],
    )


# ---------------------------------------------------------------------------
# Branch: multiview banded RRR
# ---------------------------------------------------------------------------


def _run_multi_view_rrr(
    config: RRRConfig,
    output_dir: str,
    sig_dir: str,
    rrr_dir: str,
    subject_ids: list[int],
) -> None:
    """Fit banded ridge over saved RRR views, then run the standard RRR sweep."""
    from sklearn.preprocessing import StandardScaler

    validate_aggregation_cache(
        rrr_dir, config.subspace_aggregation, refresh=config.refresh
    )
    resolved = resolve_rrr_views(config)
    target_key = target_dataset_key(config, subject_ids)
    union, detailed_frames, significance_sources = union_view_significance(
        resolved, target_key, subject_ids
    )
    embeddings, word_df = load_view_embeddings(config, resolved, subject_ids)

    enc = config.encoding
    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = get_glove_embeddings(
            enc.data_root, word_df, **enc.per_model_config
        )[1]["in_glove"]
    shared_selection_metadata = None
    if enc.shared_word_selection:
        shared_selection_metadata = _load_shared_word_selection_metadata(
            sig_dir, subject_ids
        )

    prepared = _get_decomposition_inputs(
        subject_ids,
        output_dir,
        union,
        enc,
        embeddings,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
    )
    if prepared is None:
        return
    sig_mask_flat = (
        union_sig_mask(prepared.target_manifest, detailed_frames)
        if config.compute_sig_nonsig
        else None
    )
    bases, basis_modes = load_view_bases(
        config,
        resolved,
        target_key,
        prepared.selection,
        prepared.test_idxs,
    )
    view_names = [view.name for view in config.views]
    n_words = prepared.targets.shape[0]
    all_rows = np.arange(n_words, dtype=int)
    widths = [int(view.rank) for view in config.views]
    stops = np.cumsum(widths)
    band_boundaries = list(zip([0, *stops[:-1].tolist()], stops.tolist()))
    empty_design = np.empty((n_words, int(stops[-1])), dtype=np.float32)
    fold_designs: list[np.ndarray] = [empty_design] * len(prepared.test_idxs)
    baselines: list[dict | None] = [None] * len(prepared.test_idxs)
    resume_plan = _rrr_output_resume_plan(
        rrr_dir, config, len(prepared.test_idxs)
    )
    folds_to_build = resume_plan["folds_to_build"]
    explicit_banded_alphas = (
        np.asarray(config.banded_ridge_alphas, dtype=float)
        if config.banded_ridge_alphas is not None
        else None
    )

    for fold_f, test_idx in enumerate(prepared.test_idxs):
        if fold_f not in folds_to_build:
            continue
        test_idx = np.asarray(test_idx, dtype=int)
        train_idx = np.setdiff1d(all_rows, test_idx, assume_unique=False)
        bands, boundaries = project_view_bands(
            prepared.embeddings, bases, view_names, fold_f
        )
        if band_boundaries != boundaries:
            raise ValueError("Multiview band boundaries differ across folds")
        X_full = np.concatenate(bands, axis=1).astype(np.float32, copy=False)
        fold_designs[fold_f] = X_full

        target_scaler = StandardScaler().fit(prepared.targets[train_idx])
        Y_train = target_scaler.transform(prepared.targets[train_idx]).astype(
            np.float32
        )
        Y_test = target_scaler.transform(prepared.targets[test_idx]).astype(np.float32)
        alpha_dir = os.path.join(rrr_dir, f"fold_{fold_f}", "banded_full_alpha")
        alpha_context = {
            "script": "run_rrr",
            "role": "multi_view_banded_full",
            "fold": int(fold_f),
            "n_features": int(X_full.shape[1]),
            "bands": view_names,
            "standardize": False,
        }
        banded_alphas = resolve_ridge_alphas(
            explicit_banded_alphas,
            X_full[train_idx],
            standardize=False,
            context=alpha_context,
            cache_dir=alpha_dir,
            policy=enc.alpha_grid,
        )
        fitted = fit_banded_ridge_baseline(
            [band[train_idx] for band in bands],
            [band[test_idx] for band in bands],
            Y_train,
            alphas=banded_alphas,
            ridge_cv_folds=enc.ridge_cv_folds,
            solver_params=config.banded_ridge_solver_params,
            y_in_cpu=enc.ridge_y_in_cpu,
            force_cpu=enc.ridge_force_cpu,
        )
        save_alpha_diagnostics(
            alpha_dir,
            banded_alphas,
            best_alphas=fitted.best_alphas,
            context=alpha_context,
            policy=enc.alpha_grid,
        )
        np.savez_compressed(
            os.path.join(rrr_dir, f"fold_{fold_f}", "banded_ridge_diagnostics.npz"),
            view_names=np.asarray(view_names),
            band_boundaries=np.asarray(boundaries, dtype=int),
            deltas=fitted.deltas,
            kernel_weights=np.exp(fitted.deltas),
            best_alphas=fitted.best_alphas,
            intercept=fitted.intercept,
        )
        baselines[fold_f] = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "Y_train": Y_train,
            "Y_test": Y_test,
            "Y_hat_full_train": fitted.Y_hat_train,
            "Y_hat_full_test": fitted.Y_hat_test,
            "beta_full_raw": fitted.beta_full_raw,
        }

    metadata = {
        "mode": "multi_view",
        "target_dataset_key": list(target_key),
        "views": view_names,
        "view_ranks": {view.name: int(view.rank) for view in config.views},
        "basis_modes": basis_modes,
        "subspace_aggregation": config.subspace_aggregation,
        "band_boundaries": band_boundaries,
        "significance_sources": significance_sources,
    }
    with open(os.path.join(rrr_dir, "multi_view_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"  Multiview X: {(n_words, int(stops[-1]))}, Y: {prepared.targets.shape}, "
        f"bands: {dict(zip(view_names, [b - a for a, b in band_boundaries]))}"
    )
    _run_rrr_for_unit(
        rrr_unit_dir=rrr_dir,
        X_raw=fold_designs,
        test_idxs=prepared.test_idxs,
        per_fold_baselines=baselines,
        config=config,
        sig_mask_flat=sig_mask_flat,
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
    shared_selection_metadata=None,
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

        results = _build_rrr_decomposition_inputs(
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
            shared_selection_metadata=shared_selection_metadata,
        )
        if results is None:
            continue
        X_raw, per_fold_baselines, test_idxs, sig_mask_flat = results
        _run_rrr_for_unit(
            rrr_unit_dir=rrr_sub_dir,
            X_raw=X_raw,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
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
    shared_selection_metadata=None,
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

        results = _build_rrr_decomposition_inputs(
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
            shared_selection_metadata=shared_selection_metadata,
        )
        if results is None:
            continue
        X_raw, per_fold_baselines, test_idxs, sig_mask_flat = results
        _run_rrr_for_unit(
            rrr_unit_dir=rrr_region_dir,
            X_raw=X_raw,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask_flat,
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
    subject's training data, and evaluates on the held-out fold.

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

    existing_grid = _load_existing_grid(transfer_dir)
    if ranks_by_fold is None or completed_pairs is None:
        ranks_by_fold, completed_pairs = _unit_ranks_by_fold_to_run(
            transfer_dir,
            config.rank_grid,
            n_folds,
            refresh=config.refresh,
            require_factors=False,
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
        df.to_csv(os.path.join(transfer_dir, "grid_results.csv"), index=False)
        _build_results(df, sig_mask_flat, transfer_dir, primary_label="transfer")
        return df

    all_rows = []
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
            common = _run_common_subspace_fold_k(
                A_full,
                k,
                X_train,
                X_test,
                Y_train,
                Y_test,
                sig_mask_flat,
                fold_f,
                alphas=alphas,
                alpha_policy=config.encoding.alpha_grid,
                standardize_subspace_inputs=config.standardize_subspace_inputs,
                ridge_cv_folds=config.encoding.ridge_cv_folds,
                ridge_solver_params=config.encoding.ridge_solver_params,
                ridge_y_in_cpu=config.encoding.ridge_y_in_cpu,
                ridge_force_cpu=config.encoding.ridge_force_cpu,
                device=device,
                unit_dir=transfer_dir,
                primary_label="transfer",
            )

            row = {
                "fold": fold_f,
                "k": k,
                **common,
            }
            all_rows.append(row)

        print(f"  Transfer fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        config.rank_grid,
        unit_dir=transfer_dir,
        primary_label="transfer",
        sig_mask_flat=sig_mask_flat,
    )
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
    shared_selection_metadata=None,
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
            results_train = _build_rrr_decomposition_inputs(
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
                shared_selection_metadata=shared_selection_metadata,
            )
            if results_train is None:
                print("  No training data; skipping this held-out subject.")
                continue
            X_raw, per_fold_baselines_train, test_idxs, sig_mask_train = results_train

            print(f"  Running train-subjects RRR → {train_dir}")
            _run_rrr_for_unit(
                rrr_unit_dir=train_dir,
                X_raw=X_raw,
                test_idxs=test_idxs,
                per_fold_baselines=per_fold_baselines_train,
                config=config,
                sig_mask_flat=sig_mask_train,
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
        results_held = _build_rrr_decomposition_inputs(
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
            shared_selection_metadata=shared_selection_metadata,
        )
        if results_held is None:
            print(f"  No held-out subject data; skipping transfer/baseline.")
            continue
        X_raw_held, per_fold_baselines_held, _, sig_mask_held = results_held

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
            ranks_by_fold=baseline_resume_plan["rrr_ranks_by_fold"],
            completed_pairs=baseline_resume_plan["rrr_completed_pairs"],
            pca_ranks_by_fold=baseline_resume_plan["pca_ranks_by_fold"],
            pca_completed_pairs=baseline_resume_plan["pca_completed_pairs"],
        )


# ---------------------------------------------------------------------------
# Branch: cross-dataset RRR
# ---------------------------------------------------------------------------


def _sanitize_cross_source_name(cross_result_dir: str) -> str:
    """Return a filesystem-safe short name for a source results directory."""
    normalized = os.path.normpath(str(cross_result_dir))
    name = os.path.basename(normalized) or os.path.basename(os.path.dirname(normalized))
    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in name)
    return safe or "source"


def _cross_dataset_source_unit_dir(cross_result_dir: str, kind: str) -> str:
    """Resolve a source run root to its RRR/PCA unit directory."""
    if kind not in {"rrr", "pca"}:
        raise ValueError(f"Unsupported cross-dataset source kind: {kind}")
    return os.path.join(cross_result_dir, kind)


def _load_cross_dataset_A_by_rank(
    cross_result_dir: str,
    kind: str,
    ranks: list[int],
    aggregation: str = "extrinsic",
) -> dict[int, np.ndarray]:
    """Load rank-specific averaged source fold A matrices for cross-dataset evaluation."""
    unit_dir = _cross_dataset_source_unit_dir(cross_result_dir, kind)
    averaged = {}
    for rank in sorted(set(int(k) for k in ranks)):
        try:
            averaged[rank] = aggregate_subspace_from_unit_dir(
                unit_dir, rank=rank, method=aggregation
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not load cross-dataset {kind.upper()} source factors from "
                f"{unit_dir}. Expected fold_*/A.npy under a {kind}/ unit directory."
            ) from exc
    return averaged


def _run_cross_dataset_subspace_for_unit(
    unit_dir: str,
    A_by_rank: dict[int, np.ndarray],
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    config: RRRConfig,
    primary_label: str = "nr",
    sig_mask_flat: "np.ndarray | None" = None,
    ranks_by_fold: "dict[int, list[int]] | None" = None,
    completed_pairs: "set[tuple[int, int]] | None" = None,
) -> pd.DataFrame:
    """Evaluate rank-specific source-dataset subspaces on target dataset folds."""
    os.makedirs(unit_dir, exist_ok=True)

    for rank, A_rank in A_by_rank.items():
        if A_rank.ndim != 2:
            raise ValueError(f"Expected 2D A matrix for rank {rank}, got {A_rank.shape}")
        if A_rank.shape[0] != X_raw.shape[1]:
            raise ValueError(
                "Cross-dataset source A ambient dimension does not match target embeddings: "
                f"{A_rank.shape[0]} vs {X_raw.shape[1]} for rank {rank}"
            )
        if A_rank.shape[1] < rank:
            raise ValueError(
                f"Cross-dataset source A for rank {rank} has only "
                f"{A_rank.shape[1]} direction(s)."
            )

    import torch

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_folds = len(test_idxs)

    existing_grid = _load_existing_grid(unit_dir)
    if ranks_by_fold is None or completed_pairs is None:
        ranks_by_fold, completed_pairs = _unit_ranks_by_fold_to_run(
            unit_dir,
            config.rank_grid,
            n_folds,
            refresh=config.refresh,
            require_factors=False,
        )
    missing_ranks = sorted(
        {
            int(rank)
            for ranks in ranks_by_fold.values()
            for rank in ranks
            if int(rank) not in A_by_rank
        }
    )
    if missing_ranks:
        raise ValueError(
            "Missing rank-specific cross-dataset source A matrix/matrices for "
            f"rank(s): {missing_ranks}"
        )

    total_to_run = sum(len(ranks) for ranks in ranks_by_fold.values())
    if completed_pairs:
        print(
            f"  Reusing {len(completed_pairs)} completed cross-dataset fold/rank pair(s)"
        )
    if total_to_run == 0:
        print(
            "  All requested cross-dataset fold/rank pairs complete; rebuilding summary/plots from existing artifacts."
        )
        df = _merge_grid_rows(
            existing_grid,
            [],
            completed_pairs,
            config.rank_grid,
            unit_dir=unit_dir,
            primary_label=primary_label,
            sig_mask_flat=sig_mask_flat,
            include_effective_rank=True,
        )
        df.to_csv(os.path.join(unit_dir, "grid_results.csv"), index=False)
        _build_results(df, sig_mask_flat, unit_dir, primary_label=primary_label)
        return df

    all_rows = []
    for fold_f in range(n_folds):
        ranks_to_run = ranks_by_fold[fold_f]
        if not ranks_to_run:
            print(f"\n[Cross-dataset fold {fold_f + 1}/{n_folds}] complete; skipping")
            continue

        fold_start = time.time()
        print(f"\n[Cross-dataset fold {fold_f + 1}/{n_folds}]")

        for rank in ranks_to_run:
            all_rows.extend(
                _run_subspace_rank_loop_for_fold(
                    unit_dir=unit_dir,
                    A_full=A_by_rank[int(rank)],
                    ranks_to_run=[int(rank)],
                    X_raw=X_raw,
                    baseline=per_fold_baselines[fold_f],
                    sig_mask_flat=sig_mask_flat,
                    fold_f=fold_f,
                    alphas=alphas,
                    alpha_policy=config.encoding.alpha_grid,
                    standardize_subspace_inputs=config.standardize_subspace_inputs,
                    ridge_cv_folds=config.encoding.ridge_cv_folds,
                    ridge_solver_params=config.encoding.ridge_solver_params,
                    ridge_y_in_cpu=config.encoding.ridge_y_in_cpu,
                    ridge_force_cpu=config.encoding.ridge_force_cpu,
                    device=device,
                    primary_label=primary_label,
                    progress_desc=f"  cross-dataset fold {fold_f}",
                )
            )

        print(f"  Cross-dataset fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        config.rank_grid,
        unit_dir=unit_dir,
        primary_label=primary_label,
        sig_mask_flat=sig_mask_flat,
        include_effective_rank=True,
    )
    df.to_csv(os.path.join(unit_dir, "grid_results.csv"), index=False)
    _build_results(df, sig_mask_flat, unit_dir, primary_label=primary_label)
    return df


def _run_cross_dataset_rrr(
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
    shared_selection_metadata=None,
):
    """Evaluate target-dataset NR models using rank-specific source A averages."""
    if not config.cross_result_dir:
        raise ValueError("RRRConfig.cross_result_dir is required for mode=cross_dataset")

    validate_aggregation_cache(
        rrr_dir, config.subspace_aggregation, refresh=config.refresh
    )

    print(
        "Loading rank-specific cross-dataset source RRR subspaces from "
        f"{config.cross_result_dir}"
    )
    A_rrr_by_rank = _load_cross_dataset_A_by_rank(
        config.cross_result_dir,
        kind="rrr",
        ranks=config.rank_grid,
        aggregation=config.subspace_aggregation,
    )

    A_pca_by_rank = None
    pca_dir = None
    pca_resume_plan = None
    if config.run_pca_baseline:
        print(
            "Loading rank-specific cross-dataset source PCA subspaces from "
            f"{config.cross_result_dir}"
        )
        A_pca_by_rank = _load_cross_dataset_A_by_rank(
            config.cross_result_dir,
            kind="pca",
            ranks=config.rank_grid,
            aggregation=config.subspace_aggregation,
        )
        pca_dir = _resolve_pca_sibling_dir(rrr_dir)
        validate_aggregation_cache(
            pca_dir, config.subspace_aggregation, refresh=config.refresh
        )

    n_folds = len(_load_reference_test_idxs(output_dir, subject_ids))
    rrr_resume_plan = _transfer_output_resume_plan(rrr_dir, config, n_folds)
    folds_to_build = set(rrr_resume_plan["folds_to_build"])
    if config.run_pca_baseline and pca_dir is not None:
        pca_resume_plan = _transfer_output_resume_plan(pca_dir, config, n_folds)
        folds_to_build |= pca_resume_plan["folds_to_build"]

    if not folds_to_build:
        print(
            "  Cross-dataset RRR/PCA outputs complete; skipping data/regression setup."
        )
        return

    results = _build_rrr_decomposition_inputs(
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
        shared_selection_metadata=shared_selection_metadata,
    )
    if results is None:
        return
    X_raw, per_fold_baselines, test_idxs, sig_mask_flat = results

    print(f"  Running cross-dataset RRR-source evaluation → {rrr_dir}")
    _run_cross_dataset_subspace_for_unit(
        unit_dir=rrr_dir,
        A_by_rank=A_rrr_by_rank,
        X_raw=X_raw,
        test_idxs=test_idxs,
        per_fold_baselines=per_fold_baselines,
        config=config,
        primary_label="nr",
        sig_mask_flat=sig_mask_flat,
        ranks_by_fold=rrr_resume_plan["ranks_by_fold"],
        completed_pairs=rrr_resume_plan["completed_pairs"],
    )

    if config.run_pca_baseline and A_pca_by_rank is not None and pca_dir is not None:
        print(f"  Running cross-dataset PCA-source evaluation → {pca_dir}")
        _run_cross_dataset_subspace_for_unit(
            unit_dir=pca_dir,
            A_by_rank=A_pca_by_rank,
            X_raw=X_raw,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            primary_label="pca",
            sig_mask_flat=sig_mask_flat,
            ranks_by_fold=pca_resume_plan["ranks_by_fold"],
            completed_pairs=pca_resume_plan["completed_pairs"],
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
        rrr_base_dir = config.rrr_output_dir
    elif config.mode == ScriptMode.PER_REGION:
        rrr_base_dir = os.path.join(output_dir, "per_region_rrr")
    elif config.mode == ScriptMode.PER_SUBJECT:
        rrr_base_dir = os.path.join(output_dir, "per_subject_rrr")
    elif config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
        rrr_base_dir = os.path.join(output_dir, "loso_rrr")
    elif config.mode == ScriptMode.CROSS_DATASET:
        rrr_base_dir = os.path.join(output_dir, "cross_dataset_rrr")
    elif config.mode == ScriptMode.MULTI_VIEW:
        rrr_base_dir = os.path.join(output_dir, "multi_view_rrr")
    else:
        rrr_base_dir = os.path.join(output_dir, "rrr")

    if config.mode == ScriptMode.CROSS_DATASET:
        if not config.cross_result_dir:
            raise ValueError(
                "RRRConfig.cross_result_dir is required for mode=cross_dataset"
            )
        rrr_dir = os.path.join(
            rrr_base_dir, _sanitize_cross_source_name(config.cross_result_dir)
        )
    else:
        rrr_dir = rrr_base_dir

    os.makedirs(rrr_dir, exist_ok=True)
    save_effective_config(config, rrr_dir)
    if config.run_pca_baseline:
        save_effective_config(config, _resolve_pca_sibling_dir(rrr_dir))
    print(f"Output dir:  {output_dir}")
    print(f"RRR dir:     {rrr_dir}")
    print(f"Mode:        {config.mode.value}")
    if config.run_pca_baseline:
        print(f"PCA baseline: enabled (results will mirror RRR dir with 'pca' in path)")

    sig_dir = config.sig_dir or output_dir
    if config.mode == ScriptMode.MULTI_VIEW:
        _run_multi_view_rrr(
            config=config,
            output_dir=output_dir,
            sig_dir=sig_dir,
            rrr_dir=rrr_dir,
            subject_ids=subject_ids,
        )
        print("\nDone.")
        return

    # Load significance CSV and build subject → electrode mapping
    sig_electrodes, _ = load_sig_electrodes(
        output_dir,
        subject_ids,
        sig_dir=sig_dir,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        data_root=enc.data_root,
    )
    shared_selection_metadata = None
    if enc.shared_word_selection:
        shared_selection_metadata = _load_shared_word_selection_metadata(
            sig_dir, subject_ids
        )
        print(
            "Using saved shared word-selection metadata: "
            f"{len(shared_selection_metadata['epoch_selection']):,} cached words, "
            f"{len(shared_selection_metadata['analysis_selection']):,} analysis rows"
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
        shared_selection_metadata=shared_selection_metadata,
    )

    if config.mode == ScriptMode.PER_REGION:
        _run_per_region_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.PER_SUBJECT:
        _run_per_subject_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
        _run_leave_one_subject_out_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.CROSS_DATASET:
        _run_cross_dataset_rrr(**branch_kwargs)
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
