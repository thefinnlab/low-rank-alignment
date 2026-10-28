"""Reusable mechanics for item-wise cross-condition RRR comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.utils.rrr.evaluation import evaluate_projected_input_space


@dataclass
class ComparisonItem:
    """One named, aligned input/target pair used in pairwise comparisons."""

    name: str
    X: np.ndarray
    Y: np.ndarray
    target_set: str | None = None


@dataclass
class ComparisonGroup:
    """One provider-produced set of mutually comparable items."""

    items: list[ComparisonItem]
    context: dict = field(default_factory=dict)
    target_manifests: dict[str, pd.DataFrame] = field(default_factory=dict)
    eligibility: list[dict] = field(default_factory=list)
    exclusions: list[dict] = field(default_factory=list)


ItemProvider = Callable[[], ComparisonGroup]


def validate_comparison_items(
    items: list[ComparisonItem], rank_grid: list[int], num_folds: int,
) -> None:
    """Validate the shape and identity contract required by pairwise RRR."""
    if len(items) < 2:
        raise ValueError("A comparison group requires at least two items")
    if any(not isinstance(item.name, str) or not item.name.strip() for item in items):
        raise ValueError("Comparison item names must be non-empty strings")
    names = [item.name for item in items]
    if len(set(names)) != len(names):
        raise ValueError("Comparison item names must be unique within a group")
    feature_width = None
    max_rank = max(map(int, rank_grid))
    for item in items:
        item.X = np.asarray(item.X)
        item.Y = np.asarray(item.Y)
        if item.X.ndim != 2 or item.Y.ndim != 2:
            raise ValueError(
                f"Item {item.name!r} requires two-dimensional X and Y arrays"
            )
        if len(item.X) != len(item.Y):
            raise ValueError(
                f"Item {item.name!r} has different X/Y row counts: "
                f"{len(item.X)} != {len(item.Y)}"
            )
        if len(item.X) < int(num_folds):
            raise ValueError(
                f"Item {item.name!r} has {len(item.X)} rows, fewer than the "
                f"{num_folds} requested folds"
            )
        if feature_width is None:
            feature_width = item.X.shape[1]
        elif item.X.shape[1] != feature_width:
            raise ValueError("All comparison items must share the same X feature width")
        if item.target_set is not None and (
            not isinstance(item.target_set, str) or not item.target_set.strip()
        ):
            raise ValueError(
                f"Item {item.name!r} target_set must be a non-empty string or None"
            )
        feasible = min(item.X.shape[0], item.X.shape[1], item.Y.shape[1])
        if max_rank > feasible:
            raise ValueError(
                f"Item {item.name!r} cannot estimate rank {max_rank}; maximum is "
                f"{feasible} from X={item.X.shape}, Y={item.Y.shape}"
            )


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_projected_ridge(
    Q: np.ndarray, rank: int, X_train: np.ndarray, X_test: np.ndarray,
    Y_train: np.ndarray, Y_test: np.ndarray, *, alphas=None,
    alpha_policy=None, ridge_cv_folds=5, ridge_solver_params=None,
    ridge_y_in_cpu=None, ridge_force_cpu=None, use_low_rank_svd=True,
    low_rank_svd_seed=0,
) -> dict:
    """Fit a target readout on orthogonal projected coordinates."""
    result, _ = evaluate_projected_input_space(
        Q[:, :rank], X_train, X_test, Y_train, Y_test,
        alphas=(None if alphas is None else np.asarray(alphas, dtype=float)),
        alpha_policy=alpha_policy,
        alpha_context={
            "script": "run_cross_condition_rrr", "role": "orthogonal_projection",
            "rank": int(rank),
        }, standardize=False, ridge_cv_folds=ridge_cv_folds,
        ridge_solver_params=ridge_solver_params,
        ridge_y_in_cpu=ridge_y_in_cpu, ridge_force_cpu=ridge_force_cpu,
        device=_device(),
    )
    return result
