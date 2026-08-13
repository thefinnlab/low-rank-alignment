"""Reusable ridge alpha-grid selection and diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from src.utils.shared.constants import check_alpha_grid_boundaries, refine_alpha_grid

ALPHA_POLICY_VERSION = 1

DEFAULT_ALPHA_GRID_POLICY = {
    "strategy": "auto",
    "points_per_decade": 4,
    "min_points": 21,
    "max_points": 49,
    "cache": True,
    "floor": 1e-4,
    "ceiling": 1e12,
    "upper_decades": 6.0,
    "low_dim_lower_decades": 6.0,
    "mid_dim_lower_decades": 5.0,
    "high_dim_lower_decades": 4.0,
    "low_dim_threshold": 50,
    "high_dim_threshold": 500,
    "boundary_fraction_threshold": 0.05,
    "refined_points": 31,
    "refined_margin_decades": 1.0,
}


def _policy_dict(policy: Any | None) -> dict:
    values = dict(DEFAULT_ALPHA_GRID_POLICY)
    if policy is None:
        return values
    if is_dataclass(policy):
        policy = asdict(policy)
    values.update({k: v for k, v in dict(policy).items() if v is not None})
    return values


def _context_key(context: dict | str | None, policy: dict) -> str:
    payload = {
        "context": context or {},
        "policy": {k: policy[k] for k in sorted(policy) if k != "cache"},
        "version": ALPHA_POLICY_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_path(cache_dir: str | None) -> str | None:
    if cache_dir is None:
        return None
    return os.path.join(cache_dir, "alpha_grid_cache.npz")


def _standardized_design_for_scale(
    X_train: np.ndarray, standardize: bool
) -> np.ndarray:
    X = np.asarray(X_train, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    if standardize:
        scale = X.std(axis=0, ddof=0, keepdims=True)
        scale[scale == 0] = 1.0
        X = X / scale
    return X


def _lower_decades_for_dim(n_features: int, policy: dict) -> float:
    if n_features < int(policy["low_dim_threshold"]):
        return float(policy["low_dim_lower_decades"])
    if n_features >= int(policy["high_dim_threshold"]):
        return float(policy["high_dim_lower_decades"])
    return float(policy["mid_dim_lower_decades"])


def build_dimension_aware_alpha_grid(
    X_train: np.ndarray,
    standardize: bool = True,
    policy: Any | None = None,
) -> np.ndarray:
    """Build a scale- and dimension-aware logspace grid for RidgeCV.

    The grid is centered on the average diagonal scale of the design Gram
    matrix after applying the same centering/scaling convention used by the
    downstream ridge solve. Lower-end coverage is narrower for high-dimensional
    designs to avoid spending grid points on under-regularized values.
    """
    cfg = _policy_dict(policy)
    X_scaled = _standardized_design_for_scale(X_train, standardize=standardize)
    n_features = X_scaled.shape[1]
    gram_scale = float(np.sum(X_scaled * X_scaled) / max(n_features, 1))
    if not np.isfinite(gram_scale) or gram_scale <= 0:
        gram_scale = 1.0

    lower_decades = _lower_decades_for_dim(n_features, cfg)
    lo = np.log10(gram_scale) - lower_decades
    hi = np.log10(gram_scale) + float(cfg["upper_decades"])
    lo = max(lo, np.log10(float(cfg["floor"])))
    hi = min(hi, np.log10(float(cfg["ceiling"])))
    if hi <= lo:
        hi = lo + 1.0

    span = hi - lo
    n_points = int(np.ceil(span * float(cfg["points_per_decade"]))) + 1
    n_points = max(int(cfg["min_points"]), min(int(cfg["max_points"]), n_points))
    return np.logspace(lo, hi, n_points)


def load_cached_alpha_grid(
    cache_dir: str | None,
    context: dict | str | None,
    policy: Any | None = None,
) -> np.ndarray | None:
    """Load a compatible cached refined alpha grid, if present."""
    path = _cache_path(cache_dir)
    cfg = _policy_dict(policy)
    if path is None or not os.path.exists(path) or not bool(cfg["cache"]):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if str(data["context_key"]) != _context_key(context, cfg):
            return None
        return np.asarray(data["refined_grid"], dtype=float)
    except Exception:
        return None


def resolve_ridge_alphas(
    explicit_alphas: np.ndarray | list[float] | None,
    X_train: np.ndarray,
    standardize: bool = True,
    context: dict | str | None = None,
    cache_dir: str | None = None,
    policy: Any | None = None,
) -> np.ndarray:
    """Resolve the alpha grid to use for a ridge fit.

    Explicit alphas are always returned unchanged. Otherwise a compatible
    cached refined grid is preferred, falling back to a fresh auto grid.
    """
    if explicit_alphas is not None:
        return np.asarray(explicit_alphas, dtype=float)

    cfg = _policy_dict(policy)
    if cfg.get("strategy", "auto") == "legacy":
        from src.utils.shared.constants import RIDGE_ALPHAS, LARGE_RIDGE_ALPHAS

        if X_train.shape[1] < int(cfg["low_dim_threshold"]):
            return np.asarray(RIDGE_ALPHAS, dtype=float)
        else:
            return np.asarray(LARGE_RIDGE_ALPHAS, dtype=float)

    cached = load_cached_alpha_grid(cache_dir, context, cfg)
    if cached is not None:
        return cached
    return build_dimension_aware_alpha_grid(
        X_train, standardize=standardize, policy=cfg
    )


def maybe_refine_alpha_grid(
    best_alphas: np.ndarray,
    alpha_grid: np.ndarray,
    policy: Any | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Return a refined reusable grid when alpha selections hit boundaries."""
    cfg = _policy_dict(policy)
    boundary = check_alpha_grid_boundaries(
        np.asarray(best_alphas, dtype=float),
        np.asarray(alpha_grid, dtype=float),
        boundary_fraction_threshold=float(cfg["boundary_fraction_threshold"]),
    )
    if not (boundary["hit_lower"] or boundary["hit_upper"]):
        return None, boundary
    refined = refine_alpha_grid(
        np.asarray(best_alphas, dtype=float),
        n_points=int(cfg["refined_points"]),
        margin_decades=float(cfg["refined_margin_decades"]),
        floor=float(cfg["floor"]),
        ceiling=float(cfg["ceiling"]),
    )
    return refined, boundary


def save_alpha_diagnostics(
    output_dir: str,
    alpha_grid: np.ndarray,
    best_alphas: np.ndarray | list[np.ndarray] | None = None,
    context: dict | str | None = None,
    policy: Any | None = None,
) -> dict:
    """Save alpha diagnostics and cache a refined grid when warranted."""
    os.makedirs(output_dir, exist_ok=True)
    cfg = _policy_dict(policy)
    alpha_grid = np.asarray(alpha_grid, dtype=float)

    arrays = {"alpha_grid": alpha_grid}
    flat_best = None
    if best_alphas is not None:
        if isinstance(best_alphas, list):
            for i, arr in enumerate(best_alphas):
                arrays[f"fold_{i}"] = np.asarray(arr, dtype=float)
            flat_best = np.concatenate(
                [np.asarray(a, dtype=float) for a in best_alphas]
            )
        else:
            flat_best = np.asarray(best_alphas, dtype=float)
            arrays["best_alphas"] = flat_best

    boundary = None
    refined_grid = None
    if flat_best is not None and len(flat_best) > 0:
        refined_grid, boundary = maybe_refine_alpha_grid(flat_best, alpha_grid, cfg)
        if refined_grid is not None:
            arrays["refined_grid"] = refined_grid

    np.savez(os.path.join(output_dir, "alpha_diagnostics.npz"), **arrays)

    metadata = {
        "policy_version": ALPHA_POLICY_VERSION,
        "policy": cfg,
        "context": context or {},
        "context_key": _context_key(context, cfg),
        "alpha_min": float(alpha_grid.min()),
        "alpha_max": float(alpha_grid.max()),
        "n_alphas": int(len(alpha_grid)),
        "boundary": boundary,
        "cached_refined_grid": refined_grid is not None,
    }
    with open(os.path.join(output_dir, "alpha_diagnostics.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    if refined_grid is not None and bool(cfg["cache"]):
        np.savez(
            os.path.join(output_dir, "alpha_grid_cache.npz"),
            context_key=np.array(_context_key(context, cfg)),
            refined_grid=refined_grid,
        )

    return metadata
