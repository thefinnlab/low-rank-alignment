"""Correlation helpers, including Fisher-z averaging for Pearson r values."""

import warnings

import numpy as np

_FISHER_EPS = np.finfo(np.float64).eps


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _fisher_z(values) -> np.ndarray:
    arr = _as_float_array(values)
    return np.arctanh(np.clip(arr, -1.0 + _FISHER_EPS, 1.0 - _FISHER_EPS))


def fisher_z_mean(corrs, axis=None):
    """Average Pearson correlations in Fisher-z space and return r-space values."""
    z = _fisher_z(corrs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_z = np.nanmean(z, axis=axis)
    return np.tanh(mean_z)


def fisher_z_mean_sem(corrs, axis=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Fisher-z mean r and asymmetric SEM distances in r space.

    SEM is computed in Fisher-z space over the requested axis, then transformed
    back to r space as lower/upper distances from the Fisher-z mean.
    """
    z = _fisher_z(corrs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_z = np.nanmean(z, axis=axis)
        n = np.sum(np.isfinite(z), axis=axis)
        std_z = np.nanstd(z, axis=axis, ddof=1)
        sem_z = np.where(n > 1, std_z / np.sqrt(n), 0.0)
    mean_r = np.tanh(mean_z)
    lower_r = np.tanh(mean_z - sem_z)
    upper_r = np.tanh(mean_z + sem_z)
    return mean_r, mean_r - lower_r, upper_r - mean_r


def targetwise_correlation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute Pearson r independently for matching columns of two matrices."""
    A = _as_float_array(A)
    B = _as_float_array(B)
    if A.shape != B.shape:
        raise ValueError(
            f"A and B must have identical shapes, got {A.shape} and {B.shape}"
        )
    if A.ndim != 2:
        raise ValueError(f"A and B must be 2D, got ndim={A.ndim}")

    A_centered = A - A.mean(axis=0)
    B_centered = B - B.mean(axis=0)
    numerator = np.sum(A_centered * B_centered, axis=0)
    denom = np.sqrt(
        np.sum(A_centered * A_centered, axis=0)
        * np.sum(B_centered * B_centered, axis=0)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = numerator / denom
    corr[denom == 0] = np.nan
    return corr
