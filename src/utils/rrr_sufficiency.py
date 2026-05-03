"""Bootstrap utilities for RRR rank sufficiency analysis."""

from __future__ import annotations

import numpy as np

from himalaya.scoring import correlation_score

from src.utils.tensor_utils import to_numpy


def fisher_z(r: np.ndarray, clip: float = 0.999999) -> np.ndarray:
    """Return Fisher-z transformed correlations with finite clipping."""
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -clip, clip))


def corr_per_target(Y: np.ndarray, Yhat: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation independently for each target column.

    Non-finite samples are ignored per target. Targets with fewer than two valid
    samples return NaN. The correlation itself is delegated to Himalaya's
    ``correlation_score`` so scoring stays consistent with the encoding/RRR
    pipeline.
    """
    Y = np.asarray(Y, dtype=float)
    Yhat = np.asarray(Yhat, dtype=float)
    if Y.shape != Yhat.shape:
        raise ValueError(
            f"Y and Yhat must have matching shapes, got {Y.shape} and {Yhat.shape}"
        )
    if Y.ndim != 2:
        raise ValueError("Y and Yhat must be 2-D arrays")

    if np.isfinite(Y).all() and np.isfinite(Yhat).all():
        return to_numpy(correlation_score(Y, Yhat))

    out = np.full(Y.shape[1], np.nan, dtype=float)
    for t in range(Y.shape[1]):
        y = Y[:, t]
        yh = Yhat[:, t]
        valid = np.isfinite(y) & np.isfinite(yh)
        if valid.sum() < 2:
            continue
        out[t] = float(correlation_score(y[valid, np.newaxis], yh[valid, np.newaxis])[0])
    return out


def block_bootstrap_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` indices by concatenating contiguous blocks with replacement."""
    if n <= 0:
        raise ValueError("n must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
    blocks = [(start + np.arange(block_size)) % n for start in starts]
    return np.concatenate(blocks)[:n].astype(int, copy=False)


def paired_corr_delta(
    Y: np.ndarray,
    Yhat_full: np.ndarray,
    Yhat_rank: np.ndarray,
) -> dict:
    """Compute per-target full/rank correlations and Fisher-z deltas."""
    r_full = corr_per_target(Y, Yhat_full)
    r_rank = corr_per_target(Y, Yhat_rank)
    z_full = fisher_z(r_full)
    z_rank = fisher_z(r_rank)
    delta = z_full - z_rank
    return {
        "r_full": r_full,
        "r_rank": r_rank,
        "z_full": z_full,
        "z_rank": z_rank,
        "delta": delta,
        "max_delta": _nanmax_or_nan(delta),
    }


def paired_corr_delta_bootstrap(
    Y: np.ndarray,
    Yhat_full: np.ndarray,
    Yhat_rank: np.ndarray,
    n_bootstrap: int = 5000,
    block_size: int = 50,
    rng: np.random.Generator | None = None,
    save_per_target: bool = True,
) -> dict:
    """Block bootstrap paired full-vs-rank Fisher-z correlation deltas."""
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")
    rng = np.random.default_rng() if rng is None else rng
    observed = paired_corr_delta(Y, Yhat_full, Yhat_rank)
    n, n_targets = Y.shape
    max_delta = np.full(n_bootstrap, np.nan, dtype=float)
    target_delta = (
        np.full((n_bootstrap, n_targets), np.nan, dtype=float)
        if save_per_target
        else None
    )

    for b in range(n_bootstrap):
        idx = block_bootstrap_indices(n, block_size, rng)
        boot = paired_corr_delta(Y[idx], Yhat_full[idx], Yhat_rank[idx])
        max_delta[b] = boot["max_delta"]
        if save_per_target:
            target_delta[b] = boot["delta"]

    return {
        "observed": observed,
        "max_delta": max_delta,
        "target_delta": target_delta,
    }


def bootstrap_full_model_ci_widths(
    Y: np.ndarray,
    Yhat_full: np.ndarray,
    n_bootstrap: int = 5000,
    block_size: int = 50,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """Estimate per-target Fisher-z CI widths for the full model alone."""
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")
    rng = np.random.default_rng() if rng is None else rng
    n, n_targets = Y.shape
    z_boot = np.full((n_bootstrap, n_targets), np.nan, dtype=float)
    for b in range(n_bootstrap):
        idx = block_bootstrap_indices(n, block_size, rng)
        z_boot[b] = fisher_z(corr_per_target(Y[idx], Yhat_full[idx]))
    alpha = (1.0 - ci) / 2.0
    lo = np.nanquantile(z_boot, alpha, axis=0)
    hi = np.nanquantile(z_boot, 1.0 - alpha, axis=0)
    widths = hi - lo
    return {
        "z_boot": z_boot,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_width": widths,
        "median_ci_width": float(np.nanmedian(widths)),
        "mean_ci_width": float(np.nanmean(widths)),
    }


def ci_bounds(samples: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    """Return percentile CI bounds for a bootstrap sample."""
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.nanquantile(samples, alpha)),
        float(np.nanquantile(samples, 1.0 - alpha)),
    )


def is_sufficient(upper_ci: float, epsilon: float) -> bool:
    """Rank is sufficient when the upper confidence bound is below epsilon."""
    return bool(upper_ci < epsilon)


def _nanmax_or_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return float("nan")
    return float(np.nanmax(x))
