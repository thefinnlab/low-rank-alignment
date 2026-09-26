"""Small, artifact-independent calculations for RRR result figures."""

from __future__ import annotations

import numpy as np


QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
PEAK_FLOOR = 0.005


def fisher_mean(values, axis=None):
    """Average correlations in Fisher-z space."""
    values = np.asarray(values, dtype=float)
    z = np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7))
    return np.tanh(np.nanmean(z, axis=axis))


def monotonic_normalized_curve(ranks, correlations):
    """Return zero-based relative rank and a monotonic curve in [0, 1]."""
    ranks = np.asarray(ranks, dtype=float)
    correlations = np.asarray(correlations, dtype=float)
    valid = np.isfinite(ranks) & np.isfinite(correlations)
    ranks, correlations = ranks[valid], correlations[valid]
    if len(ranks) < 2:
        return np.array([]), np.array([])
    order = np.argsort(ranks)
    ranks, correlations = ranks[order], correlations[order]
    unique, inverse = np.unique(ranks, return_inverse=True)
    if len(unique) != len(ranks):
        correlations = np.bincount(inverse, weights=correlations) / np.bincount(inverse)
        ranks = unique
    if len(ranks) < 2 or ranks[-1] == ranks[0]:
        return np.array([]), np.array([])
    curve = np.maximum.accumulate(np.maximum(correlations, 0))
    peak = curve[-1]
    if peak <= 0:
        return np.array([]), np.array([])
    return (ranks - ranks[0]) / (ranks[-1] - ranks[0]), curve / peak


def normalized_log_rank_auc(ranks, correlations):
    """Integrate normalized monotonic performance over normalized log rank."""
    ranks = np.asarray(ranks, dtype=float)
    correlations = np.asarray(correlations, dtype=float)
    valid = np.isfinite(ranks) & np.isfinite(correlations) & (ranks > 0)
    ranks, correlations = ranks[valid], correlations[valid]
    if len(ranks) < 2:
        return np.nan
    order = np.argsort(ranks)
    ranks, correlations = ranks[order], correlations[order]
    unique, inverse = np.unique(ranks, return_inverse=True)
    if len(unique) != len(ranks):
        correlations = np.bincount(inverse, weights=correlations) / np.bincount(inverse)
        ranks = unique
    if len(ranks) < 2 or not np.isclose(ranks[0], 1):
        raise ValueError(f"Log-rank AUC requires rank 1; got {ranks[0]:g}")
    curve = np.maximum.accumulate(np.maximum(correlations, 0))
    if curve[-1] <= 0:
        return np.nan
    return float(np.trapezoid(curve / curve[-1], x=np.log(ranks) / np.log(ranks[-1])))


def rank_quantiles(ranks, target_correlations, levels=QUANTILES):
    """Calculate quantiles across targets for each row of a rank-by-target array."""
    ranks = np.asarray(ranks, dtype=float)
    values = np.asarray(target_correlations, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(ranks):
        raise ValueError("Expected one rank-by-target row for each rank")
    return np.nanquantile(values, levels, axis=1).T


def normalized_quantile_curve(values, min_peak=PEAK_FLOOR):
    """Normalize one quantile's monotonic curve when its peak passes the floor."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    curve = np.maximum.accumulate(np.maximum(values[valid], 0))
    peak = float(curve[-1])
    if peak <= min_peak or peak <= 0:
        return np.full(values.shape, np.nan)
    result = np.full(values.shape, np.nan)
    result[valid] = curve / peak
    return result


def quantile_saturation_rank(ranks, quantiles, fraction=0.99, min_peak=PEAK_FLOOR):
    """Latest first crossing among eligible quantiles, interpolated in rank."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ranks = np.asarray(ranks, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim != 2 or quantiles.shape[0] != len(ranks):
        raise ValueError("quantiles must have one row per rank")
    crossings = []
    for column in quantiles.T:
        valid = np.isfinite(ranks) & np.isfinite(column)
        q_ranks, values = ranks[valid], column[valid]
        if not len(values):
            continue
        normalized = normalized_quantile_curve(values, min_peak)
        if not np.isfinite(normalized).any():
            continue
        index = int(np.searchsorted(normalized, fraction, side="left"))
        if index >= len(q_ranks):
            continue
        if index == 0:
            crossings.append(float(q_ranks[0]))
        else:
            left, right = normalized[index - 1:index + 1]
            weight = (fraction - left) / (right - left) if right > left else 1.0
            crossings.append(float(q_ranks[index - 1] + weight * (q_ranks[index] - q_ranks[index - 1])))
    return max(crossings) if crossings else np.nan


def average_normalized_quantiles(curves, grid=None, min_peak=PEAK_FLOOR):
    """Average eligible quantiles equally across model layers on relative rank."""
    if grid is None:
        grid = np.r_[0.0, np.geomspace(1e-4, 1.0, 199)]
    grid = np.asarray(grid, dtype=float)
    if not curves:
        return grid, np.empty((len(grid), 0)), np.array([], dtype=int)
    n_quantiles = curves[0][1].shape[1]
    sums = np.zeros((len(grid), n_quantiles))
    counts = np.zeros(n_quantiles, dtype=int)
    for ranks, matrix in curves:
        for index, column in enumerate(np.asarray(matrix).T):
            normalized = normalized_quantile_curve(column, min_peak)
            x, y = monotonic_normalized_curve(ranks, normalized)
            if len(x):
                sums[:, index] += np.interp(grid, x, y)
                counts[index] += 1
    result = np.full_like(sums, np.nan)
    present = counts > 0
    result[:, present] = sums[:, present] / counts[present]
    return grid, result, counts


def paired_sign_flip(first, second, *, alternative="less", permutations=10_000, seed=0):
    """One-sided paired sign-flip p-value with the notebook's fixed seed."""
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    difference = difference[np.isfinite(difference)]
    if not len(difference):
        return np.nan
    observed = float(np.mean(difference))
    rng = np.random.default_rng(seed)
    shuffled = np.asarray([
        np.mean(difference * rng.choice((-1.0, 1.0), size=len(difference)))
        for _ in range(permutations)
    ])
    exceed = np.count_nonzero(shuffled <= observed if alternative == "less"
                              else shuffled >= observed)
    return (exceed + 1) / (permutations + 1)
