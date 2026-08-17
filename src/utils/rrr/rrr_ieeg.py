"""iEEG reduced-rank regression implementation for the unified client."""

import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.tensors import to_numpy
from src.utils.shared.correlation import (
    fisher_z_mean,
)
from src.utils.atlas.core import build_electrode_region_map
from src.core.config import RRRConfig, ScriptMode
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.shared.config import (
    parse_args,
    load_config,
    expand_per_model_layer_configs,
)
from src.utils.shared.embeddings import get_glove_embeddings
from src.utils.shared.encoding import build_per_fold_baselines
from src.utils.rrr.core import (
    fit_dense_rrr,
)
from src.utils.rrr.results import (
    add_sig_nonsig_corrs,
    make_sig_split_df,
)
from src.utils.rrr.evaluation import (
    evaluate_input_space,
)
from src.utils.subspace_comparison.metrics import (
    aggregate_subspace_basis,
)
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.rrr.inputs import _get_decomposition_inputs
from src.utils.rrr.multiview import (
    fit_banded_ridge_baseline,
    project_view_bands,
)
from src.utils.rrr.significance import PreparedSignificance, load_significance_selection
from himalaya.scoring import correlation_score, r2_score
from scripts.run_rrr import Artifact, UnitSelector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _save_full_model_metrics_once(
    artifact: Artifact,
    unit: UnitSelector,
    fold_f: int,
    Y_test: np.ndarray,
    Y_hat_full: np.ndarray,
    *,
    refresh: bool = False,
) -> None:
    """Save full-model metrics once per fold."""
    selector = artifact._selector(unit, fold=int(fold_f))
    if artifact.has_field("full_metrics", selector=selector) and not refresh:
        return
    corr_arr = to_numpy(correlation_score(Y_test, Y_hat_full))
    r2_arr = to_numpy(r2_score(Y_test, Y_hat_full))
    artifact.save_full_metrics(
        unit=unit, fold=fold_f,
        metrics={"full_corr": float(fisher_z_mean(corr_arr)),
                 "full_r2": float(r2_arr.mean()),
                 "full_corr_per_target": corr_arr},
        overwrite=refresh,
    )


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


def _load_existing_grid(artifact: Artifact, unit: UnitSelector) -> pd.DataFrame:
    """Build prior grid rows from manifest-recorded rank metrics."""
    entries = artifact.load_collection("rank_metrics", unit.mapping())
    rows = []
    for selector_items, metrics in entries.items():
        selector = dict(selector_items)
        scalar_metrics = {}
        for key, value in metrics.items():
            array = np.asarray(value)
            if array.ndim == 0:
                scalar_metrics[key] = array.item()
        rows.append({"k": int(selector["rank"]), "fold": int(selector["fold"]),
                     **scalar_metrics})
    return pd.DataFrame(rows)


def _reference_encoding_results(significance, subject_ids) -> dict:
    try:
        return significance.encoding_results[int(subject_ids[0])]
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            f"SignificanceArtifact lacks encoding results for subject {subject_ids[0]}"
        ) from exc


def _load_shared_word_selection_metadata(significance, subject_ids: list[int]) -> dict:
    """Load and validate shared word-selection metadata from encoding results."""
    reference = None
    for sub_id in subject_ids:
        enc_results = significance.encoding_results[int(sub_id)]
        metadata = enc_results.get("word_selection")
        if metadata is None:
            raise ValueError(
                "RRRConfig.encoding.shared_word_selection=True requires "
                f"word_selection metadata for subject {sub_id}. Re-run significance "
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
                f"Expected shared_intersection word selection for subject {sub_id}, "
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


def _folds_to_build_from_ranks(ranks_by_fold: dict[int, list[int]]) -> set[int]:
    return {fold_f for fold_f, ranks in ranks_by_fold.items() if ranks}


def _X_for_fold(X_raw, fold_f: int) -> np.ndarray:
    """Return one fold's design from a shared matrix or per-fold matrix list."""
    return np.asarray(X_raw[fold_f] if isinstance(X_raw, list) else X_raw)


def _rrr_output_resume_plan(
    artifact: Artifact,
    unit: UnitSelector,
    config: RRRConfig,
    n_folds: int,
) -> dict:
    rrr_plan = artifact.resume_plan(
        unit=unit,
        ranks=config.rank_grid,
        folds=range(n_folds),
        refresh=config.refresh,
        require_target_factors=config.save_b_matrices,
        required_metric_keys=(
            {"null_aug_corr_per_target", "null_aug_r2_per_target"}
            if config.run_null_augmented
            else None
        ),
    )
    folds_needed = set(rrr_plan["folds_to_build"])
    pca_ranks_by_fold = None
    pca_completed_pairs = None

    if config.run_pca_baseline:
        pca_plan = artifact.resume_plan(
            unit=UnitSelector(**{**unit.mapping(), "method": "pca"}),
            ranks=config.rank_grid,
            folds=range(n_folds),
            refresh=config.refresh,
        )
        pca_ranks_by_fold = pca_plan["ranks_by_fold"]
        pca_completed_pairs = pca_plan["completed_pairs"]
        folds_needed |= pca_plan["folds_to_build"]

    return {
        "folds_to_build": folds_needed,
        "rrr_ranks_by_fold": rrr_plan["ranks_by_fold"],
        "rrr_completed_pairs": rrr_plan["completed_pairs"],
        "pca_ranks_by_fold": pca_ranks_by_fold,
        "pca_completed_pairs": pca_completed_pairs,
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


def _row_from_metrics(
    artifact: Artifact,
    unit: UnitSelector,
    k: int,
    fold_f: int,
    primary_label: str,
    sig_mask_flat: "np.ndarray | None",
    include_effective_rank: bool = False,
) -> dict:
    metrics = artifact.load_rank_metrics(unit=unit, rank=k, fold=fold_f)
    row = {"fold": fold_f, "k": k}

    factor_selector = artifact._selector(unit, fold=int(fold_f))
    if include_effective_rank and artifact.has_field("input_factors", selector=factor_selector):
        A_full = artifact.load_factors(unit=unit, fold=fold_f)["input_factors"]
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
    artifact: "Artifact | None" = None,
    unit: "UnitSelector | None" = None,
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
    if missing_grid_pairs and artifact is not None and unit is not None and primary_label is not None:
        parts.append(
            pd.DataFrame(
                [
                    _row_from_metrics(
                        artifact,
                        unit,
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
    artifact: Artifact,
    unit: UnitSelector,
    primary_label: str,
    full_corr_per_target: "np.ndarray | None" = None,
    null_augmented_config: "RRRConfig | None" = None,
    save_primary_corr_per_target: bool = True,
    orthonormalize_a_per_rank: bool = False,
) -> dict:
    """Shared evaluation for one (fold, k) projected subspace.

    Called by both _run_rrr_for_unit (primary_label='nr') and _run_pca_for_unit
    (primary_label='pca') after the reduction matrix A_full is known.

    Saves per-rank metrics (including *_per_target arrays) to
    the logical rank/fold selector.

    Returns a metrics dict with per-target arrays stripped (for grid_results.csv).
    """
    A_k = A_full[:, :k]
    if orthonormalize_a_per_rank:
        A_k = np.linalg.qr(A_k, mode="reduced")[0]

    X_proj_train = X_train @ A_k
    X_proj_test = X_test @ A_k
    context = {
        "script": "run_rrr",
        "role": primary_label,
        "fold": int(fold_f),
        "k": int(k),
        "n_features": int(X_proj_train.shape[1]),
        "standardize": bool(standardize_subspace_inputs),
    }
    diagnostic_selector = artifact._selector(unit, rank=int(k), fold=int(fold_f))
    if artifact.has_field("diagnostics", selector=diagnostic_selector):
        resolved_alphas = np.asarray(
            artifact.load_field("diagnostics", selector=diagnostic_selector)["alpha_grid"]
        )
    else:
        resolved_alphas = resolve_ridge_alphas(
            alphas, X_proj_train, standardize=standardize_subspace_inputs,
            context=context, cache_dir=None, policy=alpha_policy,
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
    artifact.save_diagnostics(
        unit=unit, rank=k, fold=fold_f,
        diagnostics={"alpha_grid": resolved_alphas,
                     "best_alphas": result[f"{primary_label}_best_alphas"],
                     "context": context, "policy": alpha_policy or {}},
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
                artifact=artifact,
                unit=unit,
                config=null_augmented_config,
            )
        )

    artifact.save_rank_metrics(unit=unit, rank=k, fold=fold_f, metrics=metrics)
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
    artifact: Artifact,
    unit: UnitSelector,
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
    orthonormalize_a_per_rank: bool = False,
    refresh: bool = False,
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

    _save_full_model_metrics_once(
        artifact, unit, fold_f, Y_test, Y_hat_full, refresh=refresh
    )
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
            artifact,
            unit,
            primary_label=primary_label,
            full_corr_per_target=full_corr_per_target,
            null_augmented_config=null_augmented_config,
            save_primary_corr_per_target=save_primary_corr_per_target,
            orthonormalize_a_per_rank=orthonormalize_a_per_rank,
        )

        existing = artifact.load_rank_metrics(unit=unit, rank=k, fold=fold_f)
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

        artifact.save_rank_metrics(
            unit=unit, rank=k, fold=fold_f, metrics=existing, overwrite=True
        )
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
    artifact: Artifact,
    unit: UnitSelector,
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
    artifact: Artifact,
    unit: UnitSelector,
    primary_label: str,
    *,
    plot: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
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
    artifact.save_summary(unit=unit, summary=summary, overwrite=True)

    from src.utils.plotting.encoding import plot_rrr_rank_results

    display_label = (
        "RRR subspace (NR)"
        if primary_label == "nr" and "rrr_corr" in df.columns
        else primary_label.upper()
    )
    title_base = f"{display_label}: perf vs rank"
    if plot:
        figure = plot_rrr_rank_results(
            summary, grid_df=df, title=title_base, primary_label=primary_label,
        )
        artifact.save_plot(unit=unit, name="rank_analysis", figure=figure,
                           overwrite=True)
    if plot and sig_mask_flat is not None:
        for split in ("sig", "nonsig"):
            figure = plot_rrr_rank_results(
                make_sig_split_df(summary, split),
                grid_df=make_sig_split_df(df, split),
                title=f"{title_base} ({split} targets)",
                primary_label=primary_label,
            )
            artifact.save_plot(unit=unit, name=f"rank_analysis_{split}",
                               figure=figure, overwrite=True)
    return summary


# ---------------------------------------------------------------------------
# Core per-unit loops
# ---------------------------------------------------------------------------


def _run_rrr_for_unit(
    artifact: Artifact,
    unit: UnitSelector,
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
        artifact: Producing script's result abstraction.
        X_raw: (n_words, n_embed) raw embedding matrix.
        test_idxs: List of test index arrays, one per fold.
        per_fold_baselines: Output of build_per_fold_baselines.
        config: RRRConfig.
        sig_mask_flat: Boolean mask over targets (True = significant).

    Returns:
        DataFrame of per-(fold, k) metrics for grid_results.csv.
    """
    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None

    n_folds = len(test_idxs)
    n_embed = _X_for_fold(X_raw, 0).shape[1]
    if any(_X_for_fold(X_raw, fold).shape[1] != n_embed for fold in range(n_folds)):
        raise ValueError("All fold-specific RRR designs must share feature width")
    selected_idx = np.arange(n_embed)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    existing_grid = _load_existing_grid(artifact, unit)
    if ranks_by_fold is None or completed_pairs is None:
        plan = artifact.resume_plan(
            unit=unit, ranks=config.rank_grid, folds=range(n_folds),
            refresh=config.refresh, require_target_factors=config.save_b_matrices,
            required_metric_keys=(
                {"null_aug_corr_per_target", "null_aug_r2_per_target"}
                if config.run_null_augmented
                else None
            ),
        )
        ranks_by_fold, completed_pairs = plan["ranks_by_fold"], plan["completed_pairs"]
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
            artifact=artifact,
            unit=unit,
            primary_label="nr",
            sig_mask_flat=sig_mask_flat,
            include_effective_rank=True,
        )
        _build_results(df, sig_mask_flat, artifact, unit, primary_label="nr",
                       plot=config.plot, overwrite=config.refresh)
        if config.run_pca_baseline:
            pca_unit = UnitSelector(**{**unit.mapping(), "method": "pca"})
            print("\nRunning PCA baseline")
            _run_pca_for_unit(
                artifact=artifact,
                unit=pca_unit,
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
                plot=config.plot,
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
        factor_selector = artifact._selector(unit, fold=int(fold_f))
        required_factors_exist = artifact.has_field("input_factors", selector=factor_selector)
        if config.save_b_matrices:
            required_factors_exist &= artifact.has_field("target_factors", selector=factor_selector)
        if not config.refresh and required_factors_exist:
            factors = artifact.load_factors(unit=unit, fold=fold_f)
            A_full = factors["input_factors"]
            B_full = factors.get("target_factors")
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
                    return_b=config.save_b_matrices,
                )
                A_full = rrr_result["A_dense"]
                B_full = (
                    rrr_result["B_dense"] if config.save_b_matrices else None
                )
                artifact.save_factors(unit=unit, fold=fold_f,
                                      input_factors=A_full, target_factors=B_full,
                                      overwrite=True)
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
                return_b=config.save_b_matrices,
            )
            A_full = rrr_result["A_dense"]
            B_full = rrr_result["B_dense"] if config.save_b_matrices else None
            artifact.save_factors(unit=unit, fold=fold_f, input_factors=A_full,
                                  target_factors=B_full, overwrite=config.refresh)

        all_rows.extend(
            _run_subspace_rank_loop_for_fold(
                artifact=artifact,
                unit=unit,
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
                orthonormalize_a_per_rank=config.orthonormalize_a_per_rank,
                refresh=config.refresh,
            )
        )

        print(f"  Fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        config.rank_grid,
        artifact=artifact,
        unit=unit,
        primary_label="nr",
        sig_mask_flat=sig_mask_flat,
        include_effective_rank=True,
    )
    _build_results(df, sig_mask_flat, artifact, unit, primary_label="nr",
                   plot=config.plot, overwrite=config.refresh)

    # Optionally run PCA baseline using the same data splits
    if config.run_pca_baseline:
        pca_unit = UnitSelector(**{**unit.mapping(), "method": "pca"})
        print("\nRunning PCA baseline")
        _run_pca_for_unit(
            artifact=artifact,
            unit=pca_unit,
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
            plot=config.plot,
        )

    return df


def _run_pca_for_unit(
    artifact: Artifact,
    unit: UnitSelector,
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
    plot: bool = True,
) -> pd.DataFrame:
    """PCA subspace analysis for one unit — structurally parallel to _run_rrr_for_unit.

    Computes PCA on training embeddings (per fold, proper CV), then evaluates:
        - Ridge from k-dim PCA subspace (primary model)

    The only difference from _run_rrr_for_unit is how A_full is computed: here it
    comes from SVD of X_train rather than from RRR. All downstream evaluation is
    handled by the same shared helpers.

    Args:
        artifact: Producing script's result abstraction.
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
    n_folds = len(test_idxs)
    existing_grid = _load_existing_grid(artifact, unit)
    if ranks_by_fold is None or completed_pairs is None:
        plan = artifact.resume_plan(unit=unit, ranks=rank_grid,
                                    folds=range(n_folds), refresh=refresh)
        ranks_by_fold, completed_pairs = plan["ranks_by_fold"], plan["completed_pairs"]
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
            artifact=artifact,
            unit=unit,
            primary_label="pca",
            sig_mask_flat=sig_mask_flat,
        )
        _build_results(df, sig_mask_flat, artifact, unit, primary_label="pca",
                       plot=plot, overwrite=refresh)
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
        _save_full_model_metrics_once(
            artifact, unit, fold_f, Y_test, Y_hat_full, refresh=refresh
        )
        full_corr = float(
            fisher_z_mean(to_numpy(correlation_score(Y_test, Y_hat_full)))
        )
        full_r2 = float(to_numpy(r2_score(Y_test, Y_hat_full)).mean())

        # Compute PCA on training data only (proper CV)
        X_mean = X_train.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_train - X_mean, full_matrices=False)
        A_full = Vt.T  # (n_embed, min(n_train, n_embed))
        artifact.save_factors(unit=unit, fold=fold_f, input_factors=A_full,
                              overwrite=refresh)
        max_components = A_full.shape[1]

        valid_ranks = [k for k in ranks_to_run if k <= max_components]
        if len(valid_ranks) < len(ranks_to_run):
            skipped = [k for k in ranks_to_run if k > max_components]
            print(f"  Skipping ranks {skipped} (> {max_components} PCA components)")
            for k in skipped:
                metrics = {"skipped": True, "available_components": max_components}
                artifact.save_rank_metrics(unit=unit, rank=k, fold=fold_f,
                                           metrics=metrics, overwrite=True)
                all_rows.append({"fold": fold_f, "k": k, **metrics})

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
                artifact,
                unit,
                primary_label="pca",
            )
            existing = artifact.load_rank_metrics(unit=unit, rank=k, fold=fold_f)
            existing["full_corr"] = full_corr
            existing["full_r2"] = full_r2
            artifact.save_rank_metrics(unit=unit, rank=k, fold=fold_f,
                                       metrics=existing, overwrite=True)

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
        artifact=artifact,
        unit=unit,
        primary_label="pca",
        sig_mask_flat=sig_mask_flat,
    )
    _build_results(df, sig_mask_flat, artifact, unit, primary_label="pca",
                   plot=plot, overwrite=refresh)

    return df


# ---------------------------------------------------------------------------
# Data preparation helper
# ---------------------------------------------------------------------------


def _build_rrr_decomposition_inputs(
    subject_ids,
    sig_electrodes,
    encoding_results,
    detailed_significance,
    encoding_config,
    embeddings,
    word_df,
    compute_sig_nonsig=False,
    word_in_glove=None,
    folds_to_build: "set[int] | None" = None,
    shared_selection_metadata: "dict | None" = None,
):
    prepared = _get_decomposition_inputs(
        subject_ids,
        sig_electrodes,
        encoding_config,
        embeddings,
        compute_sig_nonsig=compute_sig_nonsig,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
        encoding_results=encoding_results,
        significance=detailed_significance,
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
        alpha_cache_dir=None,
        alpha_context={
            "script": "run_rrr",
            "model": encoding_config.model,
            "per_model_config": encoding_config.per_model_config,
            "offset": encoding_config.offset,
        },
        folds_to_build=folds_to_build,
    )

    return (X_raw, per_fold_baselines, prepared.test_idxs,
            prepared.sig_mask_flat, prepared.target_manifest)


def _temporal_window_target_mask(target_manifest, window) -> np.ndarray:
    """Return target columns whose lag lies in ``[start_sec, end_sec)``."""
    lags = target_manifest["lag"].to_numpy(dtype=float)
    mask = (lags >= float(window.start_sec)) & (lags < float(window.end_sec))
    if not mask.any():
        available = np.unique(lags)
        extent = (
            f"[{available.min():g}, {available.max():g}]"
            if len(available)
            else "<none>"
        )
        raise ValueError(
            f"Temporal window {window.name!r} [{window.start_sec}, "
            f"{window.end_sec}) selects no sampled lag; available lag extent is "
            f"{extent} seconds"
        )
    return mask


# ---------------------------------------------------------------------------
# Branch: cross-subject RRR
# ---------------------------------------------------------------------------


def _run_cross_subject_rrr(
    config: RRRConfig,
    enc,
    artifact: Artifact,
    significance,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
    shared_selection_metadata=None,
):
    """RRR on raw embeddings using all subjects' significant electrodes."""
    unit = UnitSelector(method="rrr")
    reference = _reference_encoding_results(significance, subject_ids)
    n_folds = len(reference["test_idxs"])
    resume_plan = _rrr_output_resume_plan(artifact, unit, config, n_folds)
    folds_to_build = resume_plan["folds_to_build"]
    if not folds_to_build:
        print(
            "  Cross-subject RRR/PCA outputs complete; skipping data/regression setup."
        )
        return [(unit, n_folds)]

    results = _build_rrr_decomposition_inputs(
        subject_ids,
        sig_electrodes,
        reference,
        significance.detailed_significance,
        enc,
        aligned_embeddings,
        word_df,
        compute_sig_nonsig=config.compute_sig_nonsig,
        word_in_glove=word_in_glove,
        folds_to_build=folds_to_build,
        shared_selection_metadata=shared_selection_metadata,
    )
    if results is None:
        return []
    X_raw, per_fold_baselines, test_idxs, sig_mask_flat, target_manifest = results
    artifact.save_target_manifest(unit=unit, target_manifest=target_manifest,
                                  overwrite=config.refresh)
    _run_rrr_for_unit(
        artifact=artifact,
        unit=unit,
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
    units = [(unit, n_folds)]
    if config.run_pca_baseline:
        units.append((UnitSelector(method="pca"), n_folds))
    return units


def _run_temporal_windows_rrr(
    config: RRRConfig,
    enc,
    artifact: Artifact,
    significance,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
    shared_selection_metadata=None,
):
    """Run independent cross-subject RRR analyses for configured lag windows."""
    del word_df  # Retained for the common branch-call interface.
    reference = _reference_encoding_results(significance, subject_ids)
    n_folds = len(reference["test_idxs"])
    resume_plans = {}
    for window in config.temporal_windows:
        unit = UnitSelector(method="rrr", temporal_window=window.name)
        plan = _rrr_output_resume_plan(artifact, unit, config, n_folds)
        resume_plans[window.name] = plan

    prepared = _get_decomposition_inputs(
        subject_ids,
        sig_electrodes,
        reference,
        significance.detailed_significance,
        enc,
        aligned_embeddings,
        compute_sig_nonsig=config.compute_sig_nonsig,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
    )
    if prepared is None:
        return

    X_raw = next(iter(prepared.embeddings.values()))
    window_records = []
    for window in config.temporal_windows:
        print(
            f"\n{'=' * 60}\nTemporal window: {window.name} "
            f"[{window.start_sec}, {window.end_sec}) seconds"
        )
        target_mask = _temporal_window_target_mask(prepared.target_manifest, window)
        target_indices = np.flatnonzero(target_mask)
        manifest = (
            prepared.target_manifest.loc[target_mask].copy().reset_index(drop=True)
        )
        manifest["original_target_index"] = target_indices
        unique_lags = np.sort(manifest["lag"].unique())
        unit = UnitSelector(method="rrr", temporal_window=window.name)
        artifact.save_target_manifest(unit=unit, target_manifest=manifest,
                                      overwrite=config.refresh)

        plan = resume_plans[window.name]
        Y_window = prepared.targets[:, target_mask]
        sig_mask = (
            prepared.sig_mask_flat[target_mask]
            if prepared.sig_mask_flat is not None
            else None
        )
        per_fold_baselines = build_per_fold_baselines(
            X_raw,
            Y_window,
            prepared.test_idxs,
            alphas=(
                np.array(enc.ridge_alphas, dtype=float)
                if enc.ridge_alphas is not None
                else None
            ),
            ridge_cv_folds=enc.ridge_cv_folds,
            ridge_solver_params=enc.ridge_solver_params,
            ridge_y_in_cpu=enc.ridge_y_in_cpu,
            ridge_force_cpu=enc.ridge_force_cpu,
            alpha_policy=enc.alpha_grid,
            alpha_cache_dir=None,
            alpha_context={
                "script": "run_rrr",
                "mode": "temporal_windows",
                "window": window.name,
                "model": enc.model,
                "per_model_config": enc.per_model_config,
                "offset": enc.offset,
            },
            folds_to_build=plan["folds_to_build"],
        )
        _run_rrr_for_unit(
            artifact=artifact,
            unit=unit,
            X_raw=X_raw,
            test_idxs=prepared.test_idxs,
            per_fold_baselines=per_fold_baselines,
            config=config,
            sig_mask_flat=sig_mask,
            ranks_by_fold=plan["rrr_ranks_by_fold"],
            completed_pairs=plan["rrr_completed_pairs"],
            pca_ranks_by_fold=plan["pca_ranks_by_fold"],
            pca_completed_pairs=plan["pca_completed_pairs"],
        )
        units = {"rrr": unit}
        if config.run_pca_baseline:
            units["pca"] = UnitSelector(method="pca", temporal_window=window.name)
        window_records.append(
            {
                "metadata": {
                    "window": window.name,
                    "start_sec": float(window.start_sec),
                    "end_sec": float(window.end_sec),
                    "actual_lag_min_sec": float(unique_lags.min()),
                    "actual_lag_max_sec": float(unique_lags.max()),
                    "n_lags": int(len(unique_lags)),
                    "n_targets": int(target_mask.sum()),
                },
                "units": units,
            }
        )

    combined = []
    returned_units = []
    for record in window_records:
        for method, unit in record["units"].items():
            frame = artifact.load_summary(unit=unit).copy()
            for key, value in record["metadata"].items():
                frame[key] = value
            frame["method"] = method
            combined.append(frame)
            returned_units.append((unit, n_folds))
    artifact.save_field("mode_metadata", pd.concat(combined, ignore_index=True),
                        selector={"kind": "temporal_window_summary"},
                        overwrite=config.refresh)
    return returned_units


# ---------------------------------------------------------------------------
# Branch: multiview banded RRR
# ---------------------------------------------------------------------------


def _run_multi_view_rrr(
    config: RRRConfig,
    artifact: Artifact,
    significance: PreparedSignificance,
    subject_ids: list[int],
) -> list[tuple[UnitSelector, int]]:
    """Fit banded ridge over saved RRR views, then run the standard RRR sweep."""
    from sklearn.preprocessing import StandardScaler

    sources = {}
    source_configs = {}
    for view in config.views:
        if view.ieeg_rrr_source:
            cls = Artifact
            kwargs = dict(view.ieeg_rrr_source)
        else:
            from scripts.fmri.run_cross_dataset_rrr import Artifact as cls
            kwargs = dict(view.fmri_rrr_source or {})
        kwargs.setdefault("results_root", config.results_root)
        source = cls(**kwargs)
        loaded = source.load_artifacts(
            fields=["effective_config", "upstream_identities", "target_manifest"]
        )
        sources[view.name] = loaded
        source_configs[view.name] = loaded.effective_config

    sig_frame = significance.significant_electrodes
    union = {
        int(subject): group["elec"].astype(str).tolist()
        for subject, group in sig_frame.groupby("subject")
    }
    significance_sources = [significance.reference.identity_record()]
    for identity in significance.source_identities:
        if identity not in significance_sources:
            significance_sources.append(identity)
    target_context = resolve_dataset_context(
        dataset=config.encoding.dataset, subject_ids=subject_ids,
        per_dataset_config=config.encoding.per_dataset_config,
        data_root=config.encoding.data_root,
    )
    embeddings = {}
    word_df = None
    for view in config.views:
        view_enc = source_configs[view.name]["encoding"]
        if int(view_enc.get("offset", 0)) != int(config.encoding.offset):
            raise ValueError(
                f"View {view.name} uses offset {view_enc.get('offset')}; "
                f"target uses {config.encoding.offset}"
            )
        values, words = load_embeddings(
            config.encoding.data_root, view_enc["model"],
            dataset=config.encoding.dataset,
            per_dataset_config=config.encoding.per_dataset_config,
            dataset_context=target_context, **view_enc.get("per_model_config", {}),
        )
        if word_df is not None and not np.array_equal(
            word_df["word"].astype(str).to_numpy(), words["word"].astype(str).to_numpy()
        ):
            raise ValueError(f"Embedding transcript differs for view {view.name}")
        word_df = words
        embeddings[view.name] = np.asarray(values)

    enc = config.encoding
    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = get_glove_embeddings(
            enc.data_root, word_df, **enc.per_model_config
        )[1]["in_glove"]
    shared_selection_metadata = None
    if enc.shared_word_selection:
        shared_selection_metadata = _load_shared_word_selection_metadata(
            significance, subject_ids
        )

    reference = _reference_encoding_results(significance, subject_ids)
    prepared_values = _build_rrr_decomposition_inputs(
        subject_ids,
        union,
        reference,
        significance.detailed_significance,
        enc,
        embeddings[config.views[0].name],
        word_df,
        compute_sig_nonsig=config.compute_sig_nonsig,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
    )
    if prepared_values is None:
        return []
    _, prepared_baselines, test_idxs, sig_mask_flat, target_manifest = prepared_values
    selection = np.asarray(reference["selection"])
    bases, basis_modes = {}, {}
    for view in config.views:
        source = sources[view.name]
        rank = int(view.rank)
        factor_items = source.load_collection("input_factors", {"method": "rrr"})
        fold_factors = {int(dict(key)["fold"]): value for key, value in factor_items.items()
                        if "fold" in dict(key)}
        if source.dataset == artifact.dataset:
            if len(fold_factors) != len(test_idxs):
                raise ValueError(f"View {view.name} does not provide all target folds")
            bases[view.name] = {fold: np.asarray(value)[:, :rank]
                                for fold, value in fold_factors.items()}
            basis_modes[view.name] = "matching_target_fold_raw"
        else:
            mean_basis = aggregate_subspace_basis(
                list(fold_factors.values()), rank=rank,
                method=config.subspace_aggregation,
            )
            bases[view.name] = {fold: mean_basis for fold in range(len(test_idxs))}
            basis_modes[view.name] = f"cross_dataset_fold_{config.subspace_aggregation}"
    view_names = [view.name for view in config.views]
    n_words = len(selection)
    all_rows = np.arange(n_words, dtype=int)
    widths = [int(view.rank) for view in config.views]
    stops = np.cumsum(widths)
    band_boundaries = list(zip([0, *stops[:-1].tolist()], stops.tolist()))
    empty_design = np.empty((n_words, int(stops[-1])), dtype=np.float32)
    fold_designs: list[np.ndarray] = [empty_design] * len(test_idxs)
    baselines: list[dict | None] = [None] * len(test_idxs)
    unit = UnitSelector(method="rrr")
    artifact.save_target_manifest(unit=unit, target_manifest=target_manifest,
                                  overwrite=config.refresh)
    resume_plan = _rrr_output_resume_plan(
        artifact, unit, config, len(test_idxs)
    )
    folds_to_build = resume_plan["folds_to_build"]
    explicit_banded_alphas = (
        np.asarray(config.banded_ridge_alphas, dtype=float)
        if config.banded_ridge_alphas is not None
        else None
    )

    for fold_f, test_idx in enumerate(test_idxs):
        if fold_f not in folds_to_build:
            continue
        test_idx = np.asarray(test_idx, dtype=int)
        train_idx = np.setdiff1d(all_rows, test_idx, assume_unique=False)
        bands, boundaries = project_view_bands(
            embeddings, bases, view_names, fold_f
        )
        if band_boundaries != boundaries:
            raise ValueError("Multiview band boundaries differ across folds")
        X_full = np.concatenate(bands, axis=1).astype(np.float32, copy=False)
        fold_designs[fold_f] = X_full

        raw_targets = prepared_baselines[fold_f]["Y_train_raw"] if "Y_train_raw" in prepared_baselines[fold_f] else None
        baseline = prepared_baselines[fold_f]
        Y_train = baseline["Y_train"]
        Y_test = baseline["Y_test"]
        if raw_targets is not None:
            target_scaler = StandardScaler().fit(raw_targets)
            Y_train = target_scaler.transform(raw_targets).astype(
            np.float32
            )
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
            cache_dir=None,
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
        artifact.save_diagnostics(unit=unit, fold=fold_f, diagnostics={
            "view_names": view_names, "band_boundaries": boundaries,
            "alphas": banded_alphas, "deltas": fitted.deltas,
            "kernel_weights": np.exp(fitted.deltas),
            "best_alphas": fitted.best_alphas, "intercept": fitted.intercept,
        }, overwrite=config.refresh)
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
        "target_dataset": artifact.dataset,
        "views": view_names,
        "view_ranks": {view.name: int(view.rank) for view in config.views},
        "basis_modes": basis_modes,
        "subspace_aggregation": config.subspace_aggregation,
        "band_boundaries": band_boundaries,
        "significance_sources": significance_sources,
    }
    artifact.save_field("mode_metadata", metadata, selector={"kind": "multi_view"},
                        overwrite=config.refresh)

    print(
        f"  Multiview X: {(n_words, int(stops[-1]))}, "
        f"bands: {dict(zip(view_names, [b - a for a, b in band_boundaries]))}"
    )
    _run_rrr_for_unit(
        artifact=artifact,
        unit=unit,
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
    return [(unit, len(test_idxs))]


# ---------------------------------------------------------------------------
# Branch: per-subject RRR
# ---------------------------------------------------------------------------


def _run_per_subject_rrr(
    config: RRRConfig,
    enc,
    artifact: Artifact,
    significance,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
    shared_selection_metadata=None,
):
    """RRR on raw embeddings independently per subject."""
    units = []
    for sub_id in subject_ids:
        print(f"\n{'='*60}\nSubject {sub_id}")

        unit = UnitSelector(method="rrr", subject=int(sub_id))
        sub_sig_elecs = sig_electrodes.get(sub_id, [])
        if not sub_sig_elecs:
            print("  No significant electrodes; skipping.")
            continue

        reference = _reference_encoding_results(significance, [sub_id])
        n_folds = len(reference["test_idxs"])
        resume_plan = _rrr_output_resume_plan(artifact, unit, config, n_folds)
        folds_to_build = resume_plan["folds_to_build"]
        if not folds_to_build:
            print("  Subject RRR/PCA outputs complete; skipping data/regression setup.")
            continue

        results = _build_rrr_decomposition_inputs(
            [sub_id],
            {sub_id: sub_sig_elecs},
            reference,
            significance.detailed_significance,
            enc,
            aligned_embeddings,
            word_df,
            compute_sig_nonsig=config.compute_sig_nonsig,
            word_in_glove=word_in_glove,
            folds_to_build=folds_to_build,
            shared_selection_metadata=shared_selection_metadata,
        )
        if results is None:
            continue
        X_raw, per_fold_baselines, test_idxs, sig_mask_flat, target_manifest = results
        artifact.save_target_manifest(unit=unit, target_manifest=target_manifest,
                                      overwrite=config.refresh)
        _run_rrr_for_unit(
            artifact=artifact,
            unit=unit,
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
        units.append((unit, n_folds))
        if config.run_pca_baseline:
            units.append((UnitSelector(method="pca", subject=int(sub_id)), n_folds))
    return units


# ---------------------------------------------------------------------------
# Branch: per-region RRR
# ---------------------------------------------------------------------------


def _run_per_region_rrr(
    config: RRRConfig,
    enc,
    artifact: Artifact,
    significance,
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

    units = []
    for region_name, sub_elec_map in electrode_region_map.items():
        print(f"\n{'='*60}\nRegion: {region_name}")

        unit = UnitSelector(method="rrr", region=region_name)

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
        reference = _reference_encoding_results(significance, subs_in_region)
        n_folds = len(reference["test_idxs"])
        resume_plan = _rrr_output_resume_plan(artifact, unit, config, n_folds)
        folds_to_build = resume_plan["folds_to_build"]
        if not folds_to_build:
            print("  Region RRR/PCA outputs complete; skipping data/regression setup.")
            continue

        results = _build_rrr_decomposition_inputs(
            subs_in_region,
            per_subject_region_elecs,
            reference,
            significance.detailed_significance,
            enc,
            aligned_embeddings,
            word_df,
            compute_sig_nonsig=config.compute_sig_nonsig,
            word_in_glove=word_in_glove,
            folds_to_build=folds_to_build,
            shared_selection_metadata=shared_selection_metadata,
        )
        if results is None:
            continue
        X_raw, per_fold_baselines, test_idxs, sig_mask_flat, target_manifest = results
        artifact.save_target_manifest(unit=unit, target_manifest=target_manifest,
                                      overwrite=config.refresh)
        _run_rrr_for_unit(
            artifact=artifact,
            unit=unit,
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
        units.append((unit, n_folds))
        if config.run_pca_baseline:
            units.append((UnitSelector(method="pca", region=region_name), n_folds))
    return units


# ---------------------------------------------------------------------------
# Branch: leave-one-subject-out RRR
# ---------------------------------------------------------------------------


def _run_loso_transfer_for_unit(
    artifact: Artifact,
    unit: UnitSelector,
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    train_unit: UnitSelector,
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
        train_unit: Logical selector for the trained RRR factors.
        config: RRRConfig with rank_grid and encoding settings.
        sig_mask_flat: Optional boolean mask for significant targets in held-out subject.

    Returns:
        DataFrame with one row per (fold, k) containing transfer metrics.
    """
    import torch

    enc = config.encoding
    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_folds = len(test_idxs)

    existing_grid = _load_existing_grid(artifact, unit)
    if ranks_by_fold is None or completed_pairs is None:
        plan = artifact.resume_plan(unit=unit, ranks=config.rank_grid,
                                    folds=range(n_folds), refresh=config.refresh,
                                    require_input_factors=False)
        ranks_by_fold, completed_pairs = plan["ranks_by_fold"], plan["completed_pairs"]
    for fold_f in range(n_folds):
        if not artifact.has_field(
            "input_factors", selector=artifact._selector(train_unit, fold=fold_f)
        ):
            raise ValueError(f"Missing train-subject RRR factors for fold {fold_f}")
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
            artifact=artifact,
            unit=unit,
            primary_label="transfer",
            sig_mask_flat=sig_mask_flat,
        )
        _build_results(df, sig_mask_flat, artifact, unit, primary_label="transfer",
                       plot=config.plot, overwrite=config.refresh)
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
        A_full = artifact.load_factors(unit=train_unit, fold=fold_f)["input_factors"]

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
                artifact=artifact,
                unit=unit,
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
        artifact=artifact,
        unit=unit,
        primary_label="transfer",
        sig_mask_flat=sig_mask_flat,
    )
    _build_results(df, sig_mask_flat, artifact, unit, primary_label="transfer",
                   plot=config.plot, overwrite=config.refresh)
    return df


def _run_leave_one_subject_out_rrr(
    config: "RRRConfig",
    enc,
    artifact: Artifact,
    significance,
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
    units = []
    for held_out_sub in subject_ids:
        print(f"\n{'='*60}\nHeld-out subject: {held_out_sub}")

        # --- Step A: train RRR on n-1 subjects ---
        train_subs = [s for s in subject_ids if s != held_out_sub]
        train_sig_elecs = {s: sig_electrodes.get(s, []) for s in train_subs}
        train_reference = _reference_encoding_results(significance, train_subs)
        test_idxs = train_reference["test_idxs"]
        n_folds = len(test_idxs)
        train_unit = UnitSelector(method="rrr", held_out_subject=int(held_out_sub),
                                  loso_phase="train_subjects")

        print(f"  Training subjects: {train_subs}")
        train_resume_plan = _rrr_output_resume_plan(artifact, train_unit, config, n_folds)
        train_folds_to_build = train_resume_plan["folds_to_build"]
        if train_folds_to_build:
            results_train = _build_rrr_decomposition_inputs(
                train_subs,
                train_sig_elecs,
                train_reference,
                significance.detailed_significance,
                enc,
                aligned_embeddings,
                word_df,
                compute_sig_nonsig=config.compute_sig_nonsig,
                word_in_glove=word_in_glove,
                folds_to_build=train_folds_to_build,
                shared_selection_metadata=shared_selection_metadata,
            )
            if results_train is None:
                print("  No training data; skipping this held-out subject.")
                continue
            (X_raw, per_fold_baselines_train, test_idxs, sig_mask_train,
             train_manifest) = results_train
            artifact.save_target_manifest(unit=train_unit, target_manifest=train_manifest,
                                          overwrite=config.refresh)

            print("  Running train-subjects RRR")
            _run_rrr_for_unit(
                artifact=artifact,
                unit=train_unit,
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

        transfer_unit = UnitSelector(method="rrr", held_out_subject=int(held_out_sub),
                                     loso_phase="transfer")
        baseline_unit = UnitSelector(method="rrr", held_out_subject=int(held_out_sub),
                                     loso_phase="baseline")
        transfer_plan = artifact.resume_plan(
            unit=transfer_unit, ranks=config.rank_grid, folds=range(n_folds),
            refresh=config.refresh, require_input_factors=False,
        )
        transfer_resume_plan = {
            "folds_to_build": transfer_plan["folds_to_build"],
            "ranks_by_fold": transfer_plan["ranks_by_fold"],
            "completed_pairs": transfer_plan["completed_pairs"],
        }
        baseline_resume_plan = _rrr_output_resume_plan(
            artifact, baseline_unit, config, n_folds
        )
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
            {held_out_sub: held_sig_elecs},
            _reference_encoding_results(significance, [held_out_sub]),
            significance.detailed_significance,
            enc,
            aligned_embeddings,
            word_df,
            compute_sig_nonsig=config.compute_sig_nonsig,
            word_in_glove=word_in_glove,
            folds_to_build=held_folds_to_build,
            shared_selection_metadata=shared_selection_metadata,
        )
        if results_held is None:
            print(f"  No held-out subject data; skipping transfer/baseline.")
            continue
        (X_raw_held, per_fold_baselines_held, _, sig_mask_held,
         held_manifest) = results_held
        artifact.save_target_manifest(unit=transfer_unit, target_manifest=held_manifest,
                                      overwrite=config.refresh)
        artifact.save_target_manifest(unit=baseline_unit, target_manifest=held_manifest,
                                      overwrite=config.refresh)

        # --- Step C: transfer A from train subjects to held-out subject ---
        print("  Running transfer evaluation")
        _run_loso_transfer_for_unit(
            artifact=artifact,
            unit=transfer_unit,
            X_raw=X_raw_held,
            test_idxs=test_idxs,
            per_fold_baselines=per_fold_baselines_held,
            train_unit=train_unit,
            config=config,
            sig_mask_flat=sig_mask_held,
            ranks_by_fold=transfer_resume_plan["ranks_by_fold"],
            completed_pairs=transfer_resume_plan["completed_pairs"],
        )

        # --- Step D: within-subject baseline RRR on held-out subject alone ---
        print("  Running within-subject baseline RRR")
        _run_rrr_for_unit(
            artifact=artifact,
            unit=baseline_unit,
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
        units.extend([(train_unit, n_folds), (baseline_unit, n_folds),
                      (transfer_unit, n_folds, False, False)])
        if config.run_pca_baseline:
            units.extend([
                (UnitSelector(method="pca", held_out_subject=int(held_out_sub),
                              loso_phase="train_subjects"), n_folds),
                (UnitSelector(method="pca", held_out_subject=int(held_out_sub),
                              loso_phase="baseline"), n_folds),
            ])
    return units


# ---------------------------------------------------------------------------
# Branch: cross-dataset RRR
# ---------------------------------------------------------------------------


def _load_cross_dataset_A_by_rank(
    source,
    kind: str,
    ranks: list[int],
    aggregation: str = "extrinsic",
    source_format: str = "fold_factors",
) -> dict[int, np.ndarray]:
    """Load rank-specific source factors through its producing Artifact."""
    averaged = {}
    for rank in sorted(set(int(k) for k in ranks)):
        if source_format == "rank_bases":
            selector = {"method": kind, "rank": rank}
            if not source.has_field("input_factors", selector=selector):
                raise ValueError(f"Source Artifact has no {kind} factors for rank {rank}")
            averaged[rank] = source.load_field("input_factors", selector=selector)
            continue
        if source_format != "fold_factors":
            raise ValueError(f"Unsupported cross-dataset source format: {source_format}")
        collection = source.load_collection("input_factors", {"method": kind})
        matrices = [np.asarray(matrix)[:, :rank] for matrix in collection.values()]
        if not matrices:
            raise ValueError(f"Source Artifact has no {kind} fold factors")
        averaged[rank] = aggregate_subspace_basis(matrices, rank, aggregation)
    return averaged


def _run_cross_dataset_subspace_for_unit(
    artifact: Artifact,
    unit: UnitSelector,
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

    existing_grid = _load_existing_grid(artifact, unit)
    if ranks_by_fold is None or completed_pairs is None:
        plan = artifact.resume_plan(unit=unit, ranks=config.rank_grid,
                                    folds=range(n_folds), refresh=config.refresh,
                                    require_input_factors=False)
        ranks_by_fold, completed_pairs = plan["ranks_by_fold"], plan["completed_pairs"]
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
            artifact=artifact,
            unit=unit,
            primary_label=primary_label,
            sig_mask_flat=sig_mask_flat,
            include_effective_rank=True,
        )
        _build_results(df, sig_mask_flat, artifact, unit, primary_label=primary_label,
                       plot=config.plot, overwrite=config.refresh)
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
                    artifact=artifact,
                    unit=unit,
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
                    orthonormalize_a_per_rank=config.orthonormalize_a_per_rank,
                    refresh=config.refresh and rank == ranks_to_run[0],
                )
            )

        print(f"  Cross-dataset fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = _merge_grid_rows(
        existing_grid,
        all_rows,
        completed_pairs,
        config.rank_grid,
        artifact=artifact,
        unit=unit,
        primary_label=primary_label,
        sig_mask_flat=sig_mask_flat,
        include_effective_rank=True,
    )
    _build_results(df, sig_mask_flat, artifact, unit, primary_label=primary_label,
                   plot=config.plot, overwrite=config.refresh)
    return df


def _run_cross_dataset_rrr(
    config: RRRConfig,
    enc,
    artifact: Artifact,
    significance,
    subject_ids,
    sig_electrodes,
    aligned_embeddings,
    word_df,
    word_in_glove=None,
    shared_selection_metadata=None,
):
    """Evaluate target-dataset NR models using rank-specific source A averages."""
    if config.ieeg_rrr_source:
        source_kwargs = dict(config.ieeg_rrr_source)
        source_kwargs.setdefault("results_root", config.results_root)
        source = Artifact(**source_kwargs)
    elif config.fmri_rrr_source:
        from scripts.fmri.run_cross_dataset_rrr import Artifact as FMRIRRRArtifact
        source_kwargs = dict(config.fmri_rrr_source)
        source_kwargs.setdefault("results_root", config.results_root)
        source = FMRIRRRArtifact(**source_kwargs)
    else:
        raise ValueError("cross_dataset requires an explicit RRR source Artifact")
    if source.variant is None:
        source._resolve_variant()
    source_identity = json.dumps(source.identity_record(), sort_keys=True)
    unit = UnitSelector(method="rrr", cross_dataset_source=source_identity)
    print(f"Loading source RRR subspaces from {source.identity_record()}")
    A_rrr_by_rank = _load_cross_dataset_A_by_rank(
        source,
        kind="rrr",
        ranks=config.rank_grid,
        aggregation=config.subspace_aggregation,
        source_format=config.cross_source_format,
    )

    A_pca_by_rank = None
    pca_unit = None
    pca_resume_plan = None
    if config.run_pca_baseline:
        print(
            "Loading rank-specific cross-dataset source PCA subspaces from "
            f"{source.identity_record()}"
        )
        A_pca_by_rank = _load_cross_dataset_A_by_rank(
            source,
            kind="pca",
            ranks=config.rank_grid,
            aggregation=config.subspace_aggregation,
            source_format=config.cross_source_format,
        )
        pca_unit = UnitSelector(method="pca", cross_dataset_source=source_identity)

    reference = _reference_encoding_results(significance, subject_ids)
    n_folds = len(reference["test_idxs"])
    raw_plan = artifact.resume_plan(unit=unit, ranks=config.rank_grid,
                                    folds=range(n_folds), refresh=config.refresh,
                                    require_input_factors=False)
    rrr_resume_plan = {"folds_to_build": raw_plan["folds_to_build"],
                       "ranks_by_fold": raw_plan["ranks_by_fold"],
                       "completed_pairs": raw_plan["completed_pairs"]}
    folds_to_build = set(rrr_resume_plan["folds_to_build"])
    if config.run_pca_baseline and pca_unit is not None:
        raw_pca_plan = artifact.resume_plan(
            unit=pca_unit, ranks=config.rank_grid, folds=range(n_folds),
            refresh=config.refresh, require_input_factors=False,
        )
        pca_resume_plan = {"folds_to_build": raw_pca_plan["folds_to_build"],
                           "ranks_by_fold": raw_pca_plan["ranks_by_fold"],
                           "completed_pairs": raw_pca_plan["completed_pairs"]}
        folds_to_build |= pca_resume_plan["folds_to_build"]

    if not folds_to_build:
        print(
            "  Cross-dataset RRR/PCA outputs complete; skipping data/regression setup."
        )
        return

    results = _build_rrr_decomposition_inputs(
        subject_ids,
        sig_electrodes,
        reference,
        significance.detailed_significance,
        enc,
        aligned_embeddings,
        word_df,
        compute_sig_nonsig=config.compute_sig_nonsig,
        word_in_glove=word_in_glove,
        folds_to_build=folds_to_build,
        shared_selection_metadata=shared_selection_metadata,
    )
    if results is None:
        return
    X_raw, per_fold_baselines, test_idxs, sig_mask_flat, target_manifest = results
    artifact.save_target_manifest(unit=unit, target_manifest=target_manifest,
                                  overwrite=config.refresh)

    print("  Running cross-dataset RRR-source evaluation")
    _run_cross_dataset_subspace_for_unit(
        artifact=artifact,
        unit=unit,
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

    units = [(unit, n_folds, False, True)]
    if config.run_pca_baseline and A_pca_by_rank is not None and pca_unit is not None:
        artifact.save_target_manifest(unit=pca_unit, target_manifest=target_manifest,
                                      overwrite=config.refresh)
        print("  Running cross-dataset PCA-source evaluation")
        _run_cross_dataset_subspace_for_unit(
            artifact=artifact,
            unit=pca_unit,
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
        units.append((pca_unit, n_folds, False, True))
    return units


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_ieeg(config: RRRConfig):
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
    significance = load_significance_selection(
        config.significance_selection,
        results_root=config.results_root,
        subjects=subject_ids,
        modality="ieeg",
    )
    target_frames = list(significance.target_significance.values())
    target_frame = pd.concat(target_frames, ignore_index=True)
    significant = target_frame[target_frame["significant"].astype(bool)]
    significance.significant_electrodes = significant.rename(
        columns={"target_id": "elec"}
    )[["subject", "elec"]]
    significance.detailed_significance = pd.concat(
        list(significance.significance_details.values()), ignore_index=True
    )
    significance.encoding_results.update({
        int(subject): value
        for subject, value in list(significance.encoding_results.items())
    })
    upstreams = [significance.reference.identity_record()]
    for identity in significance.source_identities:
        if identity not in upstreams:
            upstreams.append(identity)
    for source_kwargs, source_cls in (
        (config.ieeg_rrr_source, Artifact),
        (config.fmri_rrr_source, None),
    ):
        if not source_kwargs:
            continue
        if source_cls is None:
            source_cls = Artifact
        kwargs = dict(source_kwargs)
        kwargs.setdefault("results_root", config.results_root)
        upstream = source_cls(**kwargs)
        if upstream.variant is None:
            upstream._resolve_variant()
        upstreams.append(upstream.identity_record())
    for view in config.views:
        source_kwargs = view.ieeg_rrr_source or view.fmri_rrr_source
        source_cls = Artifact
        if view.fmri_rrr_source:
            source_cls = Artifact
        kwargs = dict(source_kwargs or {})
        kwargs.setdefault("results_root", config.results_root)
        upstream = source_cls(**kwargs)
        if upstream.variant is None:
            upstream._resolve_variant()
        upstreams.append(upstream.identity_record())
    artifact = Artifact.from_ieeg_config(config, dataset_context)
    artifact.begin(
        config=config, upstreams=upstreams, refresh=config.refresh,
        variant_extras=getattr(artifact, "variant_extras", {}),
    )
    artifact.save_artifacts(effective_config=asdict(config),
                            upstream_identities=upstreams,
                            plot=config.plot, overwrite=config.refresh)
    print(f"Artifact:    {artifact.identity_record()}")
    print(f"Mode:        {config.mode.value}")
    if config.run_pca_baseline:
        print(f"PCA baseline: enabled (results will mirror RRR dir with 'pca' in path)")

    if config.mode == ScriptMode.MULTI_VIEW:
        units = _run_multi_view_rrr(
            config=config,
            artifact=artifact,
            significance=significance,
            subject_ids=subject_ids,
        )
        required = {"target_manifest": [], "input_factors": [], "full_metrics": [],
                    "rank_metrics": [], "mean_results": []}
        if config.save_b_matrices:
            required["target_factors"] = []
        for unit, n_folds in units:
            required["target_manifest"].append(unit.mapping())
            required["mean_results"].append(unit.mapping())
            for fold in range(n_folds):
                selector = artifact._selector(unit, fold=fold)
                required["input_factors"].append(selector)
                required["full_metrics"].append(selector)
                if config.save_b_matrices:
                    required["target_factors"].append(selector)
                for rank in config.rank_grid:
                    required["rank_metrics"].append(
                        artifact._selector(unit, fold=fold, rank=int(rank))
                    )
        artifact.mark_complete(required)
        print("\nDone.")
        return artifact

    # Load significance CSV and build subject → electrode mapping
    sig_frame = significance.significant_electrodes
    sig_electrodes = {
        int(subject): group["elec"].astype(str).tolist()
        for subject, group in sig_frame.groupby("subject")
    }
    shared_selection_metadata = None
    if enc.shared_word_selection:
        shared_selection_metadata = _load_shared_word_selection_metadata(
            significance, subject_ids
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
        artifact=artifact,
        significance=significance,
        subject_ids=subject_ids,
        sig_electrodes=sig_electrodes,
        aligned_embeddings=aligned_embeddings,
        word_df=word_df,
        word_in_glove=word_in_glove,
        shared_selection_metadata=shared_selection_metadata,
    )

    if config.mode == ScriptMode.PER_REGION:
        units = _run_per_region_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.PER_SUBJECT:
        units = _run_per_subject_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
        units = _run_leave_one_subject_out_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.CROSS_DATASET:
        units = _run_cross_dataset_rrr(**branch_kwargs)
    elif config.mode == ScriptMode.TEMPORAL_WINDOWS:
        units = _run_temporal_windows_rrr(**branch_kwargs)
    else:
        units = _run_cross_subject_rrr(**branch_kwargs)

    required = {field: [] for field in (
        "target_manifest", "input_factors", "full_metrics", "rank_metrics", "mean_results"
    )}
    if config.save_b_matrices:
        required["target_factors"] = []
    for item in units or []:
        unit, n_folds = item[:2]
        require_factors = item[2] if len(item) > 2 else True
        require_full = item[3] if len(item) > 3 else True
        required["target_manifest"].append(unit.mapping())
        required["mean_results"].append(unit.mapping())
        for fold in range(n_folds):
            fold_selector = artifact._selector(unit, fold=fold)
            if require_factors:
                required["input_factors"].append(fold_selector)
            if require_full:
                required["full_metrics"].append(fold_selector)
            if require_factors and config.save_b_matrices and unit.method == "rrr":
                required["target_factors"].append(fold_selector)
            for rank in config.rank_grid:
                required["rank_metrics"].append(
                    artifact._selector(unit, fold=fold, rank=int(rank))
                )
    if units:
        artifact.mark_complete(required)
    print("\nDone.")
    return artifact
