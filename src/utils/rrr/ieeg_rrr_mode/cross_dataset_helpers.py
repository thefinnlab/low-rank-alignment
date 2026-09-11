"""Subspace-loading and evaluation helpers shared by transfer modes."""

import time

import numpy as np
import pandas as pd

from scripts.run_rrr import Artifact, UnitSelector
from src.core.config import RRRConfig
from src.utils.subspace_comparison.metrics import aggregate_subspace_basis

from .common import (_X_for_fold, _build_results, _load_existing_grid,
                     _merge_grid_rows, _run_subspace_rank_loop_for_fold)


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
    A_by_rank: dict[int, "np.ndarray | dict[int, np.ndarray]"],
    X_raw: np.ndarray,
    test_idxs: list,
    per_fold_baselines: list,
    config: RRRConfig,
    primary_label: str = "nr",
    sig_mask_flat: "np.ndarray | None" = None,
    ranks_by_fold: "dict[int, list[int]] | None" = None,
    completed_pairs: "set[tuple[int, int]] | None" = None,
    evaluate_orthonormalized: bool = False,
) -> pd.DataFrame:
    """Evaluate rank-specific source subspaces on target dataset folds."""
    target_ambient_dim = _X_for_fold(X_raw, 0).shape[1]
    if any(
        _X_for_fold(X_raw, fold).shape[1] != target_ambient_dim
        for fold in range(len(test_idxs))
    ):
        raise ValueError("All fold-specific target designs must share feature width")
    for rank, A_value in A_by_rank.items():
        matrices = A_value.values() if isinstance(A_value, dict) else [A_value]
        for A_rank in matrices:
            if A_rank.ndim != 2:
                raise ValueError(
                    f"Expected 2D A matrix for rank {rank}, got {A_rank.shape}"
                )
            if A_rank.shape[0] != target_ambient_dim:
                raise ValueError(
                    "Cross-dataset source A ambient dimension does not match target "
                    f"embeddings: {A_rank.shape[0]} vs {target_ambient_dim} "
                    f"for rank {rank}"
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
        required_metric_keys = (
            {
                f"{primary_label}_orth_corr",
                f"{primary_label}_orth_r2_per_target",
            } | ({f"{primary_label}_orth_best_alphas"}
                 if config.log_alphas else set())
            if evaluate_orthonormalized
            else None
        )
        plan = artifact.resume_plan(
            unit=unit, ranks=config.rank_grid, folds=range(n_folds),
            refresh=config.refresh, require_input_factors=False,
            required_metric_keys=required_metric_keys,
        )
        ranks_by_fold, completed_pairs = plan["ranks_by_fold"], plan["completed_pairs"]
    missing_ranks = sorted({
        int(rank) for ranks in ranks_by_fold.values() for rank in ranks
        if int(rank) not in A_by_rank
    })
    if missing_ranks:
        raise ValueError(
            "Missing rank-specific cross-dataset source A matrix/matrices for "
            f"rank(s): {missing_ranks}"
        )

    total_to_run = sum(len(ranks) for ranks in ranks_by_fold.values())
    if completed_pairs:
        print(f"  Reusing {len(completed_pairs)} completed cross-dataset fold/rank pair(s)")
    if total_to_run == 0:
        print(
            "  All requested cross-dataset fold/rank pairs complete; rebuilding "
            "summary/plots from existing artifacts."
        )
        df = _merge_grid_rows(
            existing_grid, [], completed_pairs, config.rank_grid,
            artifact=artifact, unit=unit, primary_label=primary_label,
            sig_mask_flat=sig_mask_flat, include_effective_rank=True,
        )
        _build_results(
            df, sig_mask_flat, artifact, unit, primary_label=primary_label,
            plot=config.plot, overwrite=config.refresh,
        )
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
            all_rows.extend(_run_subspace_rank_loop_for_fold(
                artifact=artifact,
                unit=unit,
                A_full=(A_by_rank[int(rank)][fold_f]
                        if isinstance(A_by_rank[int(rank)], dict)
                        else A_by_rank[int(rank)]),
                ranks_to_run=[int(rank)],
                X_raw=_X_for_fold(X_raw, fold_f),
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
                evaluate_orthonormalized=evaluate_orthonormalized,
                log_alphas=config.log_alphas,
                refresh=config.refresh and rank == ranks_to_run[0],
            ))
        print(f"  Cross-dataset fold {fold_f} done in {time.time() - fold_start:.1f}s")

    df = _merge_grid_rows(
        existing_grid, all_rows, completed_pairs, config.rank_grid,
        artifact=artifact, unit=unit, primary_label=primary_label,
        sig_mask_flat=sig_mask_flat, include_effective_rank=True,
    )
    _build_results(
        df, sig_mask_flat, artifact, unit, primary_label=primary_label,
        plot=config.plot, overwrite=config.refresh,
    )
    return df


def _load_cross_temporal_A_by_fold(source, kind, source_window, ranks, n_folds):
    """Load leakage-safe, fold-matched factors for one source window."""
    result = {}
    for rank in sorted(set(int(k) for k in ranks)):
        by_fold = {}
        for fold in range(n_folds):
            selector = {"method": kind, "temporal_window": source_window, "fold": fold}
            if not source.has_field("input_factors", selector=selector):
                raise ValueError(
                    f"Source Artifact has no {kind} factors for temporal window "
                    f"{source_window!r}, fold {fold}"
                )
            if not source.has_field("input_scalers", selector=selector):
                raise ValueError(
                    f"Source {kind} factors for temporal window {source_window!r}, "
                    f"fold {fold} use the legacy raw-coordinate contract; rerun "
                    "the source RRR analysis first"
                )
            matrix = source.load_field("input_factors", selector=selector)
            if matrix.ndim != 2 or matrix.shape[1] < rank:
                raise ValueError(
                    f"Source {kind} factors for temporal window {source_window!r}, "
                    f"fold {fold} cannot provide rank {rank}; shape={matrix.shape}"
                )
            by_fold[fold] = matrix[:, :rank]
        result[rank] = by_fold
    return result


__all__ = ["_load_cross_dataset_A_by_rank", "_load_cross_temporal_A_by_fold",
           "_run_cross_dataset_subspace_for_unit"]
