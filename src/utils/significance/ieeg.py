"""Lag-cluster significance helpers for iEEG workflows."""

import hashlib
import os
from pathlib import Path
import warnings
import numpy as np
import torch
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from src.utils.shared.tensors import to_numpy

_FISHER_EPS = np.finfo(np.float64).eps


def fisher_z(values):
    """Return Fisher-z transformed correlations with finite clipping."""
    arr = np.asarray(values, dtype=np.float64)
    return np.arctanh(np.clip(arr, -1.0 + _FISHER_EPS, 1.0 - _FISHER_EPS))


def batch_corrcoef(x, y):
    """
    Compute Pearson correlation coefficients in batch over the last dimension.

    Both x and y have shape (n_electrodes, n_lags, n_words). For each
    (electrode, lag) pair the correlation between the n_words samples is
    returned, giving an output of shape (n_electrodes, n_lags).

    Args:
        x (torch.Tensor): Shape (n_electrodes, n_lags, n_words).
        y (torch.Tensor): Shape (n_electrodes, n_lags, n_words).

    Returns:
        torch.Tensor: Shape (n_electrodes, n_lags) of Pearson r values.
    """
    return _batch_corrcoef_last_dim(x, y)


def _batch_corrcoef_last_dim(x, y):
    """Compute Pearson correlations over the last dimension with broadcasting."""
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    num = (x * y).sum(dim=-1)
    denom = (x**2).sum(dim=-1).sqrt() * (y**2).sum(dim=-1).sqrt()
    with torch.no_grad():
        out = num / denom
        out = torch.where(denom == 0, torch.full_like(out, float("nan")), out)
    return out


def split_fold_into_blocks(fold_length, blocks_per_fold):
    """Split one fold length into non-empty block index arrays.

    Uses np.array_split so fold lengths do not need to be divisible by the
    requested number of blocks. Empty blocks are dropped when blocks_per_fold is
    larger than fold_length.
    """
    if fold_length <= 0:
        raise ValueError(f"fold_length must be positive, got {fold_length}")
    if blocks_per_fold <= 0:
        raise ValueError(f"blocks_per_fold must be positive, got {blocks_per_fold}")
    return [
        block.astype(int, copy=False)
        for block in np.array_split(np.arange(fold_length), blocks_per_fold)
        if len(block) > 0
    ]


def reshape_fold_predictions_and_targets(
    fold_predictions, test_idxs, Y, n_electrodes, n_lags
):
    """Preserve fold grouping while reshaping predictions and targets.

    Args:
        fold_predictions: List of held-out prediction matrices from train_encoding
            with return_preds=True, one array per fold.
        test_idxs: List of held-out row indices, one array per fold.
        Y: Full target matrix of shape (n_words, n_electrodes * n_lags).
        n_electrodes: Number of electrodes.
        n_lags: Number of lags.

    Returns:
        Tuple of (fold_predictions_3d, fold_targets_3d), where each item is a
        list of arrays shaped (fold_len, n_electrodes, n_lags).
    """
    if len(fold_predictions) != len(test_idxs):
        raise ValueError(
            "fold_predictions and test_idxs must have the same length, got "
            f"{len(fold_predictions)} and {len(test_idxs)}"
        )
    fold_predictions_3d = []
    fold_targets_3d = []
    for fold_idx, (pred, test_idx) in enumerate(zip(fold_predictions, test_idxs)):
        pred = to_numpy(pred)
        test_idx = np.asarray(test_idx)
        fold_len = len(test_idx)
        expected = (fold_len, n_electrodes * n_lags)
        if pred.shape != expected:
            raise ValueError(
                f"Fold {fold_idx} predictions have shape {pred.shape}; "
                f"expected {expected}."
            )
        target = Y[test_idx]
        fold_predictions_3d.append(pred.reshape(fold_len, n_electrodes, n_lags))
        fold_targets_3d.append(target.reshape(fold_len, n_electrodes, n_lags))
    return fold_predictions_3d, fold_targets_3d


def _resolve_torch_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_fold_arrays(fold_predictions, fold_targets):
    if len(fold_predictions) != len(fold_targets):
        raise ValueError(
            "fold_predictions and fold_targets must have the same length, got "
            f"{len(fold_predictions)} and {len(fold_targets)}"
        )
    if not fold_predictions:
        raise ValueError("At least one fold is required.")
    for fold_idx, (pred, target) in enumerate(zip(fold_predictions, fold_targets)):
        pred = np.asarray(pred)
        target = np.asarray(target)
        if pred.shape != target.shape or pred.ndim != 3:
            raise ValueError(
                "Each fold prediction/target pair must have shape "
                "(fold_len, n_electrodes, n_lags); "
                f"fold {fold_idx} has {pred.shape} and {target.shape}"
            )


def compute_fold_block_correlations(
    fold_predictions, fold_targets, blocks_per_fold, device=None
):
    """Compute blockwise electrode-lag correlations for fold-local data."""
    _validate_fold_arrays(fold_predictions, fold_targets)
    torch_device = _resolve_torch_device(device)
    block_corrs = []
    block_rows = []
    for fold_idx, (pred, target) in enumerate(zip(fold_predictions, fold_targets)):
        pred = np.asarray(pred)
        target = np.asarray(target)
        for block_idx, local_idx in enumerate(
            split_fold_into_blocks(len(pred), blocks_per_fold)
        ):
            pred_t = torch.tensor(
                pred[local_idx].transpose(1, 2, 0),
                dtype=torch.float32,
                device=torch_device,
            )
            target_t = torch.tensor(
                target[local_idx].transpose(1, 2, 0),
                dtype=torch.float32,
                device=torch_device,
            )
            block_corrs.append(to_numpy(batch_corrcoef(pred_t, target_t)))
            block_rows.append(
                {
                    "fold_idx": int(fold_idx),
                    "block_idx": int(block_idx),
                    "start": int(local_idx[0]),
                    "stop": int(local_idx[-1] + 1),
                    "n_samples": int(len(local_idx)),
                }
            )
    if not block_corrs:
        raise ValueError("No fold blocks were available for correlation scoring.")
    return np.stack(block_corrs), block_rows


def _compute_shifted_block_correlations_batch(
    fold_predictions, fold_targets, blocks_per_fold, batch_shift_matrix, device=None
):
    """Compute block correlations for a batch of fold-local circular shifts.

    Returns an array with shape
    ``(batch_size, n_blocks_total, n_electrodes, n_lags)``.
    """
    _validate_fold_arrays(fold_predictions, fold_targets)
    batch_shift_matrix = np.asarray(batch_shift_matrix, dtype=int)
    if batch_shift_matrix.ndim != 2:
        raise ValueError(
            "batch_shift_matrix must have shape (batch_size, n_folds), got "
            f"{batch_shift_matrix.shape}"
        )
    if batch_shift_matrix.shape[1] != len(fold_predictions):
        raise ValueError(
            "batch_shift_matrix fold count does not match fold data: "
            f"{batch_shift_matrix.shape[1]} and {len(fold_predictions)}"
        )
    if batch_shift_matrix.shape[0] == 0:
        raise ValueError("batch_shift_matrix must contain at least one shift row.")

    block_specs = []
    for fold_idx, pred in enumerate(fold_predictions):
        for local_idx in split_fold_into_blocks(len(pred), blocks_per_fold):
            block_specs.append((fold_idx, local_idx))
    if not block_specs:
        raise ValueError("No fold blocks were available for correlation scoring.")

    torch_device = _resolve_torch_device(device)
    target_tensors = [
        torch.tensor(np.asarray(target), dtype=torch.float32, device=torch_device)
        for target in fold_targets
    ]
    block_corrs = []
    for fold_idx, local_idx in block_specs:
        pred = torch.tensor(
            np.asarray(fold_predictions[fold_idx])[local_idx].transpose(1, 2, 0),
            dtype=torch.float32,
            device=torch_device,
        )
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        pred_norm = (pred_centered**2).sum(dim=-1).sqrt()
        fold_shifts = batch_shift_matrix[:, fold_idx]
        fold_len = len(fold_targets[fold_idx])
        source_idx = (local_idx[None, :] - fold_shifts[:, None]) % fold_len
        source_idx_t = torch.tensor(source_idx, dtype=torch.long, device=torch_device)
        target = target_tensors[fold_idx][source_idx_t].permute(0, 2, 3, 1)
        target_centered = target - target.mean(dim=-1, keepdim=True)
        target_norm = (target_centered**2).sum(dim=-1).sqrt()
        numerator = (pred_centered.unsqueeze(0) * target_centered).sum(dim=-1)
        denominator = pred_norm.unsqueeze(0) * target_norm
        corr = numerator / denominator
        corr = torch.where(denominator == 0, torch.full_like(corr, float("nan")), corr)
        block_corrs.append(to_numpy(corr))
    return np.stack(block_corrs, axis=1)


def compute_fold_correlations(fold_predictions, fold_targets, device=None):
    """Compute correlations after concatenating fold-local predictions/targets."""
    _validate_fold_arrays(fold_predictions, fold_targets)
    pred = np.concatenate([np.asarray(pred) for pred in fold_predictions], axis=0)
    target = np.concatenate([np.asarray(target) for target in fold_targets], axis=0)
    torch_device = _resolve_torch_device(device)
    pred_t = torch.tensor(
        pred.transpose(1, 2, 0), dtype=torch.float32, device=torch_device
    )
    target_t = torch.tensor(
        target.transpose(1, 2, 0), dtype=torch.float32, device=torch_device
    )
    return to_numpy(batch_corrcoef(pred_t, target_t))


def build_cluster_filter(block_corrs, positive_fraction=0.8, mean_corr_threshold=0.0):
    """Return reliable electrode-lag mask from blockwise correlations.

    Block correlations are averaged in Fisher-z space, transformed back to
    correlation space, then thresholded there so the cutoff is interpretable as
    a Pearson r value.
    """
    block_corrs = np.asarray(block_corrs, dtype=np.float64)
    if block_corrs.ndim != 3:
        raise ValueError(
            f"block_corrs must have shape (n_blocks, n_electrodes, n_lags), got {block_corrs.shape}"
        )
    finite = np.isfinite(block_corrs)
    positive = (block_corrs > 0) & finite
    valid_counts = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        positive_frac = positive.sum(axis=0) / valid_counts
    z = fisher_z(block_corrs)
    mean_corr = np.tanh(np.nanmean(z, axis=0))
    mask = (
        (valid_counts > 0)
        & (positive_frac >= positive_fraction)
        & (mean_corr >= mean_corr_threshold)
    )
    return mask, positive_frac, mean_corr


def label_lag_clusters(cluster_mask, max_matching_distance=1):
    """Label lag clusters independently for each electrode.

    Candidate lags are joined when the gap between neighboring candidate lag
    indices is at most ``max_matching_distance``. For example, distance 1
    requires directly adjacent candidates, while distance 2 allows one missing
    lag between candidates.
    """
    cluster_mask = np.asarray(cluster_mask, dtype=bool)
    if cluster_mask.ndim != 2:
        raise ValueError(
            f"cluster_mask must have shape (n_electrodes, n_lags), got {cluster_mask.shape}"
        )
    if max_matching_distance < 1:
        raise ValueError("max_matching_distance must be a positive integer")
    labels = np.full(cluster_mask.shape, -1, dtype=int)
    clusters = []
    cluster_id = 0
    for elec_idx in range(cluster_mask.shape[0]):
        lag_idxs = np.flatnonzero(cluster_mask[elec_idx])
        if len(lag_idxs) == 0:
            continue
        current = [int(lag_idxs[0])]
        for lag_idx in lag_idxs[1:]:
            lag_idx = int(lag_idx)
            if lag_idx - current[-1] <= max_matching_distance:
                current.append(lag_idx)
            else:
                labels[elec_idx, current] = cluster_id
                clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "electrode_idx": elec_idx,
                        "lag_indices": np.array(current, dtype=int),
                    }
                )
                cluster_id += 1
                current = [lag_idx]
        labels[elec_idx, current] = cluster_id
        clusters.append(
            {
                "cluster_id": cluster_id,
                "electrode_idx": elec_idx,
                "lag_indices": np.array(current, dtype=int),
            }
        )
        cluster_id += 1
    return labels, clusters


def compute_sem_normalized_target_scores(block_corrs):
    """Return per-target Fisher-z mean divided by Fisher-z SEM across blocks."""
    block_corrs = np.asarray(block_corrs, dtype=np.float64)
    if block_corrs.ndim != 3:
        raise ValueError(
            f"block_corrs must have shape (n_blocks, n_electrodes, n_lags), got {block_corrs.shape}"
        )
    z = fisher_z(block_corrs)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_z = np.nanmean(z, axis=0)
        n = np.sum(np.isfinite(z), axis=0)
        std_z = np.nanstd(z, axis=0, ddof=1)
        sem_z = np.where(n > 1, std_z / np.sqrt(n), 0.0)
        target_scores = mean_z / sem_z
    return target_scores, mean_z, sem_z


def score_lag_clusters(target_scores, clusters):
    """Score clusters by summed SEM-normalized blockwise Fisher-z mass."""
    target_scores = np.asarray(target_scores, dtype=np.float64)
    scored = []
    for cluster in clusters:
        elec_idx = int(cluster["electrode_idx"])
        lag_indices = np.asarray(cluster["lag_indices"], dtype=int)
        lag_scores = target_scores[elec_idx, lag_indices]
        score = float(np.nansum(lag_scores))
        scored.append(
            {
                **cluster,
                "cluster_score": score,
                "cluster_size": int(len(lag_indices)),
                "peak_target_score": float(np.nanmax(lag_scores)),
            }
        )
    return scored


def score_fold_lag_clusters(
    fold_predictions,
    fold_targets,
    blocks_per_fold,
    max_matching_distance=1,
    positive_fraction=0.8,
    mean_corr_threshold=0.0,
    device=None,
):
    """Compute fold-local correlations, build lag clusters, and score them.

    This is the shared scoring path for observed data and fold-local shifted
    null samples.
    """
    block_corrs, block_rows = compute_fold_block_correlations(
        fold_predictions, fold_targets, blocks_per_fold, device=device
    )
    corr = compute_fold_correlations(fold_predictions, fold_targets, device=device)
    cluster_mask, positive_fraction_observed, mean_corr = build_cluster_filter(
        block_corrs,
        positive_fraction=positive_fraction,
        mean_corr_threshold=mean_corr_threshold,
    )
    cluster_labels, clusters = label_lag_clusters(
        cluster_mask, max_matching_distance=max_matching_distance
    )
    target_scores, mean_z, sem_z = compute_sem_normalized_target_scores(block_corrs)
    scored_clusters = score_lag_clusters(target_scores, clusters)
    for cluster in scored_clusters:
        elec_idx = int(cluster["electrode_idx"])
        lag_indices = np.asarray(cluster["lag_indices"], dtype=int)
        cluster["peak_corr"] = float(np.nanmax(corr[elec_idx, lag_indices]))
    return {
        "corr": corr,
        "block_corrs": block_corrs,
        "target_scores": target_scores,
        "mean_z": mean_z,
        "sem_z": sem_z,
        "block_rows": block_rows,
        "cluster_mask": cluster_mask,
        "cluster_labels": cluster_labels,
        "clusters": clusters,
        "scored_clusters": scored_clusters,
        "positive_fraction": positive_fraction_observed,
        "mean_corr": mean_corr,
    }


def valid_fold_shifts(fold_length, minimal_shift=500, shift_step=1):
    """Return valid circular shifts for one held-out fold."""
    if shift_step <= 0:
        raise ValueError(f"shift_step must be positive, got {shift_step}")
    stop = fold_length - minimal_shift
    if minimal_shift >= stop:
        raise ValueError(
            "Held-out fold is too short for fold-local circular shifts: "
            f"fold_len={fold_length}, minimal_shift={minimal_shift}. "
            "Need fold_len > 2 * minimal_shift."
        )
    return np.arange(minimal_shift, stop, shift_step, dtype=int)


def derive_subject_null_seed(null_seed, subject_id):
    """Derive a stable RNG seed from a base seed and subject identifier."""
    if isinstance(null_seed, bool) or not isinstance(null_seed, int):
        raise ValueError("null_seed must be an integer")
    digest = hashlib.blake2b(
        str(subject_id).encode("utf-8"), digest_size=8, person=b"sig-null"
    ).digest()
    subject_component = int.from_bytes(digest, byteorder="little", signed=False)
    base_component = null_seed % (2**64)
    return int(
        np.random.SeedSequence([base_component, subject_component]).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def fold_local_circular_shift_null_scores(
    fold_predictions,
    fold_targets,
    blocks_per_fold=5,
    max_matching_distance=1,
    positive_fraction=0.8,
    mean_corr_threshold=0.0,
    minimal_shift=500,
    shift_step=1,
    num_null_shifts=10000,
    seed=42,
    device=None,
    null_batch_size=16,
):
    """Score a fold-local circular-shift cluster null distribution.

    Each null sample independently draws one allowed shift per held-out fold,
    with replacement. Fold predictions are never paired with targets from
    another fold.
    """
    _validate_fold_arrays(fold_predictions, fold_targets)
    if isinstance(num_null_shifts, bool) or not isinstance(num_null_shifts, int):
        raise ValueError("num_null_shifts must be a positive integer")
    if num_null_shifts <= 0:
        raise ValueError("num_null_shifts must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(null_batch_size, bool) or not isinstance(
        null_batch_size, (int, np.integer)
    ):
        raise ValueError("null_batch_size must be a positive integer")
    null_batch_size = int(null_batch_size)
    if null_batch_size <= 0:
        raise ValueError("null_batch_size must be a positive integer")
    shift_lists = [
        valid_fold_shifts(len(target), minimal_shift, shift_step)
        for target in fold_targets
    ]
    rng = np.random.default_rng(seed % (2**64))
    n_null = num_null_shifts
    n_electrodes = np.asarray(fold_predictions[0]).shape[1]
    max_per_electrode = np.zeros((n_null, n_electrodes), dtype=np.float32)
    shift_matrix = np.column_stack(
        [rng.choice(shifts, size=n_null, replace=True) for shifts in shift_lists]
    )

    for batch_start in tqdm(
        range(0, n_null, null_batch_size), desc="Fold-local circular shifts"
    ):
        batch_stop = min(batch_start + null_batch_size, n_null)
        batch_block_corrs = _compute_shifted_block_correlations_batch(
            fold_predictions,
            fold_targets,
            blocks_per_fold=blocks_per_fold,
            batch_shift_matrix=shift_matrix[batch_start:batch_stop],
            device=device,
        )
        for batch_offset, block_corrs in enumerate(batch_block_corrs):
            sample_idx = batch_start + batch_offset
            cluster_mask, _, _ = build_cluster_filter(
                block_corrs,
                positive_fraction=positive_fraction,
                mean_corr_threshold=mean_corr_threshold,
            )
            _, clusters = label_lag_clusters(
                cluster_mask, max_matching_distance=max_matching_distance
            )
            target_scores, _, _ = compute_sem_normalized_target_scores(block_corrs)
            for cluster in score_lag_clusters(target_scores, clusters):
                elec_idx = int(cluster["electrode_idx"])
                max_per_electrode[sample_idx, elec_idx] = max(
                    max_per_electrode[sample_idx, elec_idx],
                    float(cluster["cluster_score"]),
                )
    return max_per_electrode, {
        "shift_matrix": shift_matrix,
        "n_null": int(n_null),
        "minimal_shift": int(minimal_shift),
        "shift_step": int(shift_step),
        "fold_lengths": np.array([len(target) for target in fold_targets], dtype=int),
        "seed": int(seed),
        "sampling_mode": "iid_with_replacement_per_fold",
    }


def compute_per_electrode_cluster_p_values(cluster_records, max_per_electrode):
    """Compute plus-one corrected p-values from subject-local electrode nulls."""
    max_per_electrode = np.asarray(max_per_electrode, dtype=np.float64)
    if max_per_electrode.ndim != 2:
        raise ValueError(
            "max_per_electrode must have shape (n_null, n_electrodes), got "
            f"{max_per_electrode.shape}"
        )
    if max_per_electrode.shape[0] == 0:
        raise ValueError("max_per_electrode must contain at least one null sample.")
    p_values = []
    for rec in cluster_records:
        elec_idx = int(rec["subject_electrode_index"])
        if elec_idx < 0 or elec_idx >= max_per_electrode.shape[1]:
            raise ValueError(
                "Missing null distribution for "
                f"subject={rec.get('subject')}, subject_electrode_index={elec_idx}."
            )
        null = max_per_electrode[:, elec_idx]
        score = float(rec["cluster_score"])
        p_values.append((np.sum(null >= score) + 1) / (len(null) + 1))
    return np.asarray(p_values, dtype=float)


def apply_cluster_fdr(cluster_records, fdr_q=0.01, fdr_mode="electrode"):
    """Assign cluster- and electrode-level BH results to candidate records."""
    if fdr_mode not in {"electrode", "cluster"}:
        raise ValueError(
            f"fdr_mode must be either 'electrode' or 'cluster', got {fdr_mode!r}"
        )
    if not cluster_records:
        return cluster_records

    cluster_p = np.asarray(
        [record["cluster_p_value"] for record in cluster_records], dtype=float
    )
    cluster_reject, cluster_q, _, _ = multipletests(
        cluster_p, alpha=fdr_q, method="fdr_bh"
    )

    best_by_electrode = {}
    for index, record in enumerate(cluster_records):
        key = (record["subject"], record["electrode"])
        candidate = (
            -float(record["cluster_score"]),
            int(record["cluster_id"]),
            index,
        )
        if key not in best_by_electrode or candidate < best_by_electrode[key]:
            best_by_electrode[key] = candidate

    best_indices = [candidate[2] for candidate in best_by_electrode.values()]
    electrode_p = cluster_p[best_indices]
    electrode_reject, electrode_q, _, _ = multipletests(
        electrode_p, alpha=fdr_q, method="fdr_bh"
    )
    electrode_results = {
        (cluster_records[index]["subject"], cluster_records[index]["electrode"]): (
            float(electrode_p[result_idx]),
            float(electrode_q[result_idx]),
            bool(electrode_reject[result_idx]),
            index,
        )
        for result_idx, index in enumerate(best_indices)
    }

    for index, record in enumerate(cluster_records):
        key = (record["subject"], record["electrode"])
        elec_p, elec_q, elec_sig, best_index = electrode_results[key]
        is_best = index == best_index
        record["cluster_q_value"] = float(cluster_q[index])
        record["electrode_p_value"] = elec_p
        record["electrode_q_value"] = elec_q
        record["is_electrode_best_cluster"] = is_best
        record["p_value"] = float(cluster_p[index])
        record["q_value"] = float(cluster_q[index])
        if fdr_mode == "cluster":
            record["sig"] = bool(cluster_reject[index])
        else:
            record["sig"] = bool(is_best and elec_sig)
    return cluster_records
