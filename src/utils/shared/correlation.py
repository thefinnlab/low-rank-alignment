"""Utilities used by retained script workflows."""

import warnings
import numpy as np
_FISHER_EPS = np.finfo(np.float64).eps


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _fisher_z(values) -> np.ndarray:
    arr = _as_float_array(values)
    return np.arctanh(np.clip(arr, -1.0 + _FISHER_EPS, 1.0 - _FISHER_EPS))


def fisher_z_mean(corrs, axis=None, return_z=False):
    """Average Pearson correlations in Fisher-z space and return r-space values."""
    z = _fisher_z(corrs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_z = np.nanmean(z, axis=axis)
    if return_z:
        return mean_z
    return np.tanh(mean_z)


def fisher_z_mean_sem(
    corrs, axis=None, return_z=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if return_z:
        return mean_z, sem_z
    mean_r = np.tanh(mean_z)
    lower_r = np.tanh(mean_z - sem_z)
    upper_r = np.tanh(mean_z + sem_z)
    return mean_r, mean_r - lower_r, upper_r - mean_r
