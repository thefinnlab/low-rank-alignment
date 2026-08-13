"""fMRI-specific significance helpers."""

import numpy as np
from statsmodels.stats.multitest import multipletests

from src.utils.shared.fmri import _zscore_columns


def circular_shift_pvalues(
    true: np.ndarray,
    predicted: np.ndarray,
    story_slices: dict[str, slice],
    *,
    num_shifts: int,
    minimal_shift: int,
    seed: int,
    batch_size: int = 1024,
) -> tuple[np.ndarray, dict]:
    """One-sided empirical p-values using FFT cross-correlations in target batches."""
    true_z = _zscore_columns(np.asarray(true, dtype=np.float64))
    pred_z = _zscore_columns(np.asarray(predicted, dtype=np.float64))
    observed = np.nan_to_num(np.mean(true_z * pred_z, axis=0))
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(observed.shape, dtype=np.int64)
    story_items = list(story_slices.items())
    sampled_shifts = np.empty((num_shifts, len(story_items)), dtype=np.int32)
    for story_index, (story, part) in enumerate(story_items):
        length = part.stop - part.start
        allowed = np.arange(minimal_shift, length - minimal_shift + 1)
        if not len(allowed):
            raise ValueError(
                f"Story {story} has {length} rows, too short for minimal_shift={minimal_shift}"
            )
        sampled_shifts[:, story_index] = rng.choice(
            allowed, size=num_shifts, replace=True
        )

    n_rows, n_targets = true_z.shape
    for target_start in range(0, n_targets, batch_size):
        target_stop = min(target_start + batch_size, n_targets)
        null_dot = np.zeros((num_shifts, target_stop - target_start), dtype=np.float64)
        for story_index, (_, part) in enumerate(story_items):
            story_true = true_z[part, target_start:target_stop]
            story_pred = pred_z[part, target_start:target_stop]
            cross = np.fft.irfft(
                np.conj(np.fft.rfft(story_true, axis=0))
                * np.fft.rfft(story_pred, axis=0),
                n=len(story_true),
                axis=0,
            )
            lookup = (-sampled_shifts[:, story_index]) % len(story_true)
            null_dot += cross[lookup]
        null_corr = null_dot / n_rows
        exceedances[target_start:target_stop] = np.sum(
            null_corr >= observed[target_start:target_stop], axis=0
        )
    pvalues = (exceedances + 1.0) / (num_shifts + 1.0)
    shifts_used = [
        {
            story: int(sampled_shifts[draw, story_index])
            for story_index, (story, _) in enumerate(story_items)
        }
        for draw in range(min(10, num_shifts))
    ]
    return pvalues, {"observed": observed, "representative_shifts": shifts_used}


def bh_fdr(pvalues: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    rejected, adjusted, _, _ = multipletests(pvalues, alpha=q, method="fdr_bh")
    return rejected.astype(bool), adjusted
