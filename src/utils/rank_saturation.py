"""Target-level rank curves and correlation-aware saturation summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CLIP = 1.0 - 1e-7


def fisher_z(values: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(values, dtype=float), -CLIP, CLIP))


def fisher_r(values: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(values, dtype=float))


def mean_folds(correlations: np.ndarray) -> np.ndarray:
    """Fisher-average an ``(fold, rank, target)`` correlation array."""
    values = fisher_z(correlations)
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=0)


def electrode_units(target_z: np.ndarray, manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Average target z values over lags, yielding equally weighted electrodes."""
    required = {"subject", "electrode"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"iEEG manifest requires columns {sorted(required)}")
    keys = manifest["subject"].astype(str) + ":" + manifest["electrode"].astype(str)
    codes, labels = pd.factorize(keys, sort=False)
    units = np.stack(
        [np.nanmean(target_z[:, codes == code], axis=1) for code in range(len(labels))],
        axis=1,
    )
    return units, np.asarray(labels, dtype=str)


def fmri_spatial_blocks(
    manifest: pd.DataFrame, fmri_repo_root: str | Path, block_mm: float
) -> np.ndarray:
    """Map saved response columns to subject-specific physical-space cubes."""
    if block_mm <= 0:
        raise ValueError("block_mm must be positive")
    import nibabel as nib

    manifest = manifest.reset_index(drop=True)
    root = Path(fmri_repo_root) / "data" / "ds003020" / "derivatives"
    result = np.empty(len(manifest), dtype=object)
    for subject, rows in manifest.groupby("subject", sort=False):
        mask_path = (
            root / "pycortex-db" / str(subject) / "transforms"
            / f"{subject}_auto" / "mask_thick.nii.gz"
        )
        image = nib.load(mask_path)
        mask = np.asanyarray(image.dataobj).astype(bool)
        coordinates = np.column_stack(np.nonzero(mask))
        indices = rows["voxel_index"].to_numpy(dtype=int)
        if indices.size and (indices.min() < 0 or indices.max() >= len(coordinates)):
            raise ValueError(f"Voxel index outside {mask_path} for {subject}")
        homogeneous = np.column_stack([coordinates[indices], np.ones(len(indices))])
        xyz = (image.affine @ homogeneous.T).T[:, :3]
        cubes = np.floor(xyz / float(block_mm)).astype(int)
        result[rows.index] = [f"{subject}:{x}:{y}:{z}" for x, y, z in cubes]
    return result.astype(str)


def _cluster_bootstrap_means(
    target_z: np.ndarray, cluster_ids: np.ndarray, repeats: int, seed: int
) -> np.ndarray:
    """Target-weighted cluster bootstrap means for every rank."""
    codes, _ = pd.factorize(cluster_ids, sort=False)
    n_clusters = int(codes.max()) + 1
    sums = np.stack([np.nansum(target_z[:, codes == i], axis=1) for i in range(n_clusters)])
    counts = np.stack([np.isfinite(target_z[:, codes == i]).sum(axis=1) for i in range(n_clusters)])
    rng = np.random.default_rng(seed)
    output = np.empty((repeats, target_z.shape[0]), dtype=float)
    for repeat in range(repeats):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        output[repeat] = sums[chosen].sum(axis=0) / counts[chosen].sum(axis=0)
    return output


def _cluster_bootstrap_medians(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    repeats: int,
    seed: int,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Target-weighted cluster bootstrap medians for every rank."""
    codes, _ = pd.factorize(cluster_ids, sort=False)
    n_clusters = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=repeats
    )
    output = np.empty((repeats, values.shape[0]), dtype=float)
    for rank_index, row in enumerate(values):
        finite = np.isfinite(row)
        ordered = np.flatnonzero(finite)[np.argsort(row[finite], kind="stable")]
        sorted_values = row[ordered]
        sorted_codes = codes[ordered]
        valid_cluster_sizes = np.bincount(sorted_codes, minlength=n_clusters)
        for start in range(0, repeats, batch_size):
            stop = min(start + batch_size, repeats)
            batch_counts = counts[start:stop]
            total_weights = batch_counts @ valid_cluster_sizes
            cumulative = np.cumsum(batch_counts[:, sorted_codes], axis=1)
            lower_positions = np.argmax(
                cumulative >= ((total_weights + 1) // 2)[:, None], axis=1
            )
            upper_positions = np.argmax(
                cumulative >= (total_weights // 2 + 1)[:, None], axis=1
            )
            output[start:stop, rank_index] = (
                sorted_values[lower_positions] + sorted_values[upper_positions]
            ) / 2.0
    return output


@dataclass
class SaturationSummary:
    curve: pd.DataFrame
    retention: pd.DataFrame
    bootstrap_saturation: np.ndarray


def normalized_log_auc(
    ranks: np.ndarray, performance_r: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return monotonic performance, peak-normalized performance, and log-rank AUC.

    The AUC is divided by the covered log-rank interval, so it represents mean
    normalized performance over that interval rather than depending on its width.
    """
    ranks = np.asarray(ranks, dtype=float)
    performance_r = np.asarray(performance_r, dtype=float)
    if ranks.ndim != 1 or performance_r.shape != ranks.shape:
        raise ValueError("ranks and performance_r must be matching one-dimensional arrays")
    if len(ranks) < 2:
        raise ValueError("log-rank AUC requires at least two ranks")
    if np.any(~np.isfinite(ranks)) or np.any(ranks <= 0) or np.any(np.diff(ranks) <= 0):
        raise ValueError("ranks must be finite, positive, and strictly increasing")
    if np.any(~np.isfinite(performance_r)):
        raise ValueError("performance_r must contain only finite values")
    monotonic = np.maximum.accumulate(performance_r)
    peak = float(monotonic[-1])
    if peak <= 0:
        raise ValueError("normalized AUC requires a positive maximum performance")
    normalized = monotonic / peak
    log_ranks = np.log(ranks)
    widths = np.diff(log_ranks)
    auc = float(np.sum(widths * (normalized[:-1] + normalized[1:]) / 2.0))
    auc /= float(log_ranks[-1] - log_ranks[0])
    return monotonic, normalized, auc


def summarize_rank_curve(
    ranks: np.ndarray,
    target_z: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    bootstrap_repeats: int = 2000,
    seed: int = 0,
    performance_fraction: float = 0.985,
) -> SaturationSummary:
    """Summarize a target-weighted curve and its one-SE saturation point."""
    ranks = np.asarray(ranks, dtype=int)
    target_z = np.asarray(target_z, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if target_z.shape[0] != len(ranks) or target_z.shape[1] != len(cluster_ids):
        raise ValueError("rank/target dimensions do not match ranks and cluster_ids")
    if not 0 < performance_fraction <= 1:
        raise ValueError("performance_fraction must be in (0, 1]")
    boot = _cluster_bootstrap_means(target_z, cluster_ids, bootstrap_repeats, seed)
    mean_z = np.nanmean(target_z, axis=1)
    sem_z = np.nanstd(boot, axis=0, ddof=1)
    peak_index = int(np.nanargmax(mean_z))
    threshold_z = mean_z[peak_index] - sem_z[peak_index]
    eligible = np.flatnonzero(mean_z >= threshold_z)
    saturation_index = int(eligible[0])
    probability_worse = np.nanmean(
        boot < boot[:, peak_index, None], axis=0
    )
    probability_eligible = np.flatnonzero(probability_worse < 0.95)
    probability_saturation_index = int(probability_eligible[0])
    if mean_z[peak_index] > 0:
        percent_eligible = np.flatnonzero(mean_z >= 0.99 * mean_z[peak_index])
        percent_saturation_rank = ranks[int(percent_eligible[0])]
        fraction_eligible = np.flatnonzero(
            mean_z >= performance_fraction * mean_z[peak_index]
        )
        fraction_saturation_index = int(fraction_eligible[0])
    else:
        percent_saturation_rank = np.nan
        fraction_saturation_index = peak_index

    unit_peak = np.nanmax(target_z, axis=0)
    losses = unit_peak[None, :] - target_z
    margin = sem_z[peak_index]
    median_loss = np.nanmedian(losses, axis=1)
    best_loss_index = int(np.nanargmin(median_loss))
    loss_boot = _cluster_bootstrap_medians(
        losses, cluster_ids, bootstrap_repeats, seed + 1
    )
    fixed_best_loss_boot = loss_boot[:, best_loss_index]
    loss_bound = float(np.nanquantile(fixed_best_loss_boot, 0.841))
    loss_eligible = np.flatnonzero(median_loss <= loss_bound)
    loss_saturation_index = int(loss_eligible[0])
    bootstrap_saturation = np.asarray([
        ranks[np.flatnonzero(sample <= loss_bound)[0]]
        if np.any(sample <= loss_bound)
        else ranks[int(np.nanargmin(sample))]
        for sample in loss_boot
    ])
    curve = pd.DataFrame({
        "rank": ranks,
        "mean_z": mean_z,
        "mean_r": fisher_r(mean_z),
        "sem_z": sem_z,
        "lower_r": fisher_r(mean_z - sem_z),
        "upper_r": fisher_r(mean_z + sem_z),
        "peak_rank": ranks[peak_index],
        "saturation_rank": ranks[fraction_saturation_index],
        "performance_fraction": performance_fraction,
        "performance_fraction_rank": ranks[fraction_saturation_index],
        "performance_saturation_rank": ranks[saturation_index],
        "probability_saturation_rank": ranks[probability_saturation_index],
        "performance_99pct_rank": percent_saturation_rank,
        "probability_worse_than_mean_best": probability_worse,
        "best_loss_rank": ranks[best_loss_index],
        "median_loss_saturation_rank": ranks[loss_saturation_index],
        "loss_bound_z": loss_bound,
        "threshold_z": threshold_z,
    })
    monotonic_r, normalized_r, log_auc = normalized_log_auc(
        ranks, curve["mean_r"].to_numpy()
    )
    curve["monotonic_mean_r"] = monotonic_r
    curve["normalized_monotonic_mean_r"] = normalized_r
    curve["normalized_log_auc"] = log_auc
    retention = pd.DataFrame({
        "rank": ranks,
        "median_loss_z": median_loss,
        "q25_loss_z": np.nanquantile(losses, 0.25, axis=1),
        "q75_loss_z": np.nanquantile(losses, 0.75, axis=1),
        "fraction_within_peak_sem": np.nanmean(losses <= margin, axis=1),
        "best_loss_rank": ranks[best_loss_index],
        "loss_bound_z": loss_bound,
        "saturation_rank": ranks[fraction_saturation_index],
        "median_loss_saturation_rank": ranks[loss_saturation_index],
    })
    return SaturationSummary(curve, retention, bootstrap_saturation)
