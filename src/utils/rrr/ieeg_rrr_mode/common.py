"""iEEG reduced-rank regression implementation for the unified client."""

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
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.rrr.inputs import _get_decomposition_inputs
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
        f"{primary_label}_orth_corr",
        f"{primary_label}_orth_r2",
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
    evaluate_orthonormalized: bool = False,
) -> dict:
    """Shared evaluation for one (fold, k) projected subspace.

    Called by both _run_rrr_for_unit (primary_label='nr') and _run_pca_for_unit
    (primary_label='pca') after the reduction matrix A_full is known.

    Saves per-rank metrics (including *_per_target arrays) to
    the logical rank/fold selector.

    Returns a metrics dict with per-target arrays stripped (for grid_results.csv).
    """
    A_k = A_full[:, :k]
    evaluations = [(primary_label, A_k, None)]
    if evaluate_orthonormalized:
        evaluations.append(
            (f"{primary_label}_orth", np.linalg.qr(A_k, mode="reduced")[0],
             f"{primary_label}_orth")
        )

    metrics = {}
    results = {}
    for label, factor, diagnostic_name in evaluations:
        X_proj_train = X_train @ factor
        X_proj_test = X_test @ factor
        context = {
            "script": "run_rrr",
            "role": label,
            "fold": int(fold_f),
            "k": int(k),
            "n_features": int(X_proj_train.shape[1]),
            "standardize": bool(standardize_subspace_inputs),
        }
        diagnostic_selector = artifact._selector(
            unit, rank=int(k), fold=int(fold_f), name=diagnostic_name
        )
        if artifact.has_field("diagnostics", selector=diagnostic_selector):
            resolved_alphas = np.asarray(
                artifact.load_field(
                    "diagnostics", selector=diagnostic_selector
                )["alpha_grid"]
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
            subspace_name=label,
            alphas=resolved_alphas,
            ridge_cv_folds=ridge_cv_folds,
            standardize=standardize_subspace_inputs,
            ridge_solver_params=ridge_solver_params,
            ridge_y_in_cpu=ridge_y_in_cpu,
            ridge_force_cpu=ridge_force_cpu,
            return_alphas=True,
            device=device,
        )
        results[label] = result
        artifact.save_diagnostics(
            unit=unit, rank=k, fold=fold_f, name=diagnostic_name,
            diagnostics={"alpha_grid": resolved_alphas,
                         "best_alphas": result[f"{label}_best_alphas"],
                         "context": context, "policy": alpha_policy or {}},
        )
        metrics.update({
            f"{label}_corr": result[f"{label}_corr"],
            f"{label}_r2": result[f"{label}_r2"],
            f"{label}_r2_per_target": result[f"{label}_r2_per_target"],
            f"{label}_best_alphas": result[f"{label}_best_alphas"],
        })
        if save_primary_corr_per_target:
            metrics[f"{label}_corr_per_target"] = result[
                f"{label}_corr_per_target"
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
                primary_corr_per_target=results[primary_label][
                    f"{primary_label}_corr_per_target"
                ],
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

    artifact.save_rank_metrics(
        unit=unit, rank=k, fold=fold_f, metrics=metrics, overwrite=True
    )
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
    evaluate_orthonormalized: bool = False,
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
            evaluate_orthonormalized=evaluate_orthonormalized,
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
        f"{primary_label}_orth_corr",
        f"{primary_label}_orth_r2",
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
            required_metric_keys={
                "nr_orth_corr",
                "nr_orth_r2_per_target",
                "nr_orth_best_alphas",
            } | (
                {"null_aug_corr_per_target", "null_aug_r2_per_target"}
                if config.run_null_augmented else set()
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
            artifact.save_target_manifest(
                unit=pca_unit,
                target_manifest=artifact.load_field(
                    "target_manifest", selector=unit.mapping()
                ),
                overwrite=config.refresh,
            )
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
                evaluate_orthonormalized=True,
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
        artifact.save_target_manifest(
            unit=pca_unit,
            target_manifest=artifact.load_field(
                "target_manifest", selector=unit.mapping()
            ),
            overwrite=config.refresh,
        )
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
    target_pca_dim: "int | None" = None,
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
        target_pca_dim=target_pca_dim,
    )

    target_manifest = prepared.target_manifest
    sig_mask_flat = prepared.sig_mask_flat
    if target_pca_dim is not None:
        target_manifest = pd.DataFrame({
            "target_id": [f"pc-{index + 1}" for index in range(int(target_pca_dim))],
            "target_kind": "target_pca_component",
            "component": np.arange(1, int(target_pca_dim) + 1, dtype=int),
        })
        sig_mask_flat = None

    return (X_raw, per_fold_baselines, prepared.test_idxs,
            sig_mask_flat, target_manifest)


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
# Main
# ---------------------------------------------------------------------------


def run_ieeg(config: RRRConfig, target_pca_dim: int | None = None):
    enc = config.encoding
    if target_pca_dim is not None and config.mode != ScriptMode.CROSS_SUBJECT:
        raise ValueError("target_pca_dim is supported only for internal iEEG RRR")

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
    artifact_cls = Artifact
    if target_pca_dim is not None:
        from scripts.run_target_dimensionality_rrr import Artifact as TargetDimArtifact
        artifact_cls = TargetDimArtifact
    artifact = artifact_cls.from_ieeg_config(config, dataset_context)
    if target_pca_dim is not None:
        artifact.target_pca_dim = int(target_pca_dim)
        artifact.variant_extras["target_pca_dim"] = int(target_pca_dim)
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
        from .multi_view import run_multi_view

        units = run_multi_view(
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
    if target_pca_dim is not None:
        branch_kwargs["target_pca_dim"] = int(target_pca_dim)

    if config.mode == ScriptMode.PER_SUBJECT:
        from .per_subject import run_per_subject
        units = run_per_subject(**branch_kwargs)
    elif config.mode == ScriptMode.LEAVE_ONE_SUBJECT_OUT:
        from .leave_one_subject_out import run_leave_one_subject_out
        units = run_leave_one_subject_out(**branch_kwargs)
    elif config.mode == ScriptMode.LEAVE_ONE_REGION_OUT:
        from .leave_one_region_out import run_leave_one_region_out
        units = run_leave_one_region_out(**branch_kwargs)
    elif config.mode == ScriptMode.CROSS_DATASET:
        from .cross_dataset import run_cross_dataset
        units = run_cross_dataset(**branch_kwargs)
    elif config.mode == ScriptMode.CROSS_TEMPORAL_WINDOWS:
        from .cross_temporal_windows import run_cross_temporal_windows
        units = run_cross_temporal_windows(**branch_kwargs)
    elif config.mode == ScriptMode.TEMPORAL_WINDOWS:
        from .temporal_windows import run_temporal_windows
        units = run_temporal_windows(**branch_kwargs)
    else:
        from .cross_subject import run_cross_subject
        units = run_cross_subject(**branch_kwargs)

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
