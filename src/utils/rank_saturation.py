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
    target_z: np.ndarray,
    cluster_ids: np.ndarray,
    repeats: int,
    seed: int,
    *,
    equal_cluster_weight: bool = False,
) -> np.ndarray:
    """Cluster-bootstrap means for every rank."""
    codes, _ = pd.factorize(cluster_ids, sort=False)
    n_clusters = int(codes.max()) + 1
    sums = np.stack([np.nansum(target_z[:, codes == i], axis=1) for i in range(n_clusters)])
    counts = np.stack([np.isfinite(target_z[:, codes == i]).sum(axis=1) for i in range(n_clusters)])
    rng = np.random.default_rng(seed)
    output = np.empty((repeats, target_z.shape[0]), dtype=float)
    for repeat in range(repeats):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        if equal_cluster_weight:
            output[repeat] = np.nanmean(sums[chosen] / counts[chosen], axis=0)
        else:
            output[repeat] = sums[chosen].sum(axis=0) / counts[chosen].sum(axis=0)
    return output


@dataclass
class SaturationSummary:
    curve: pd.DataFrame
    retention: pd.DataFrame
    bootstrap_saturation: np.ndarray


def summarize_rank_curve(
    ranks: np.ndarray,
    target_z: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    bootstrap_repeats: int = 2000,
    seed: int = 0,
    equal_cluster_weight: bool = False,
) -> SaturationSummary:
    """Summarize a target-weighted curve and its one-SE saturation point."""
    ranks = np.asarray(ranks, dtype=int)
    target_z = np.asarray(target_z, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if target_z.shape[0] != len(ranks) or target_z.shape[1] != len(cluster_ids):
        raise ValueError("rank/target dimensions do not match ranks and cluster_ids")
    boot = _cluster_bootstrap_means(
        target_z,
        cluster_ids,
        bootstrap_repeats,
        seed,
        equal_cluster_weight=equal_cluster_weight,
    )
    if equal_cluster_weight:
        codes, _ = pd.factorize(cluster_ids, sort=False)
        cluster_means = np.stack([
            np.nanmean(target_z[:, codes == code], axis=1)
            for code in range(int(codes.max()) + 1)
        ])
        mean_z = np.nanmean(cluster_means, axis=0)
    else:
        mean_z = np.nanmean(target_z, axis=1)
    sem_z = np.nanstd(boot, axis=0, ddof=1)
    peak_index = int(np.nanargmax(mean_z))
    threshold_z = mean_z[peak_index] - sem_z[peak_index]
    eligible = np.flatnonzero(mean_z >= threshold_z)
    saturation_index = int(eligible[0])

    bootstrap_saturation = np.empty(bootstrap_repeats, dtype=int)
    for index, sample in enumerate(boot):
        sample_peak = int(np.nanargmax(sample))
        sample_threshold = sample[sample_peak] - sem_z[sample_peak]
        bootstrap_saturation[index] = ranks[np.flatnonzero(sample >= sample_threshold)[0]]

    unit_peak = np.nanmax(target_z, axis=0)
    losses = unit_peak[None, :] - target_z
    margin = sem_z[peak_index]
    curve = pd.DataFrame({
        "rank": ranks,
        "mean_z": mean_z,
        "mean_r": fisher_r(mean_z),
        "sem_z": sem_z,
        "lower_r": fisher_r(mean_z - sem_z),
        "upper_r": fisher_r(mean_z + sem_z),
        "peak_rank": ranks[peak_index],
        "saturation_rank": ranks[saturation_index],
        "threshold_z": threshold_z,
    })
    retention = pd.DataFrame({
        "rank": ranks,
        "median_loss_z": np.nanmedian(losses, axis=1),
        "q25_loss_z": np.nanquantile(losses, 0.25, axis=1),
        "q75_loss_z": np.nanquantile(losses, 0.75, axis=1),
        "fraction_within_peak_sem": np.nanmean(losses <= margin, axis=1),
    })
    return SaturationSummary(curve, retention, bootstrap_saturation)
