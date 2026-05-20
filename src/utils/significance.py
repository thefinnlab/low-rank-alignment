import os
from pathlib import Path
import warnings

import numpy as np
import torch
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from src.utils.tensor_utils import to_numpy

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


def validate_min_cluster_size(min_cluster_size):
    """Return a validated positive integer minimum cluster size."""
    if isinstance(min_cluster_size, bool) or not isinstance(
        min_cluster_size, (int, np.integer)
    ):
        raise ValueError(
            f"min_cluster_size must be a positive integer, got {min_cluster_size!r}"
        )
    min_cluster_size = int(min_cluster_size)
    if min_cluster_size < 1:
        raise ValueError(
            f"min_cluster_size must be a positive integer, got {min_cluster_size}"
        )
    return min_cluster_size


def mark_clusters_by_min_size(scored_clusters, min_cluster_size=1):
    """Mark scored lag clusters significant when their size reaches a threshold."""
    min_cluster_size = validate_min_cluster_size(min_cluster_size)
    marked = []
    for cluster in scored_clusters:
        cluster_size = int(
            cluster.get("cluster_size", len(np.asarray(cluster["lag_indices"])))
        )
        marked.append(
            {
                **cluster,
                "cluster_size": cluster_size,
                "sig": cluster_size >= min_cluster_size,
            }
        )
    return marked


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


def fold_local_circular_shift_null_scores(
    fold_predictions,
    fold_targets,
    blocks_per_fold=5,
    max_matching_distance=1,
    positive_fraction=0.8,
    mean_corr_threshold=0.0,
    minimal_shift=500,
    shift_step=1,
    device=None,
    null_batch_size=16,
):
    """Score a fold-local circular-shift cluster null distribution.

    Each null sample shifts targets independently inside each held-out fold.
    Fold predictions are never paired with targets from another fold. When folds
    have different numbers of valid shifts, shorter shift lists cycle.
    """
    del null_batch_size  # Reserved for future batching; looped scoring keeps RAM bounded.
    _validate_fold_arrays(fold_predictions, fold_targets)
    shift_lists = [
        valid_fold_shifts(len(target), minimal_shift, shift_step)
        for target in fold_targets
    ]
    n_null = max(len(shifts) for shifts in shift_lists)
    n_electrodes = np.asarray(fold_predictions[0]).shape[1]
    max_per_electrode = np.zeros((n_null, n_electrodes), dtype=np.float32)
    shift_matrix = np.zeros((n_null, len(fold_predictions)), dtype=int)

    for sample_idx in tqdm(range(n_null), desc="Fold-local circular shifts"):
        shifted_targets = []
        for fold_idx, (target, shifts) in enumerate(zip(fold_targets, shift_lists)):
            shift = int(shifts[sample_idx % len(shifts)])
            shift_matrix[sample_idx, fold_idx] = shift
            shifted_targets.append(np.roll(target, shift=shift, axis=0))
        result = score_fold_lag_clusters(
            fold_predictions,
            shifted_targets,
            blocks_per_fold,
            max_matching_distance=max_matching_distance,
            positive_fraction=positive_fraction,
            mean_corr_threshold=mean_corr_threshold,
            device=device,
        )
        for cluster in result["scored_clusters"]:
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
    }


def aggregate_global_null(max_per_electrode_by_subject):
    """Aggregate subject/electrode null scores into a global max-statistic null."""
    arrays = [np.asarray(arr, dtype=np.float64) for arr in max_per_electrode_by_subject]
    if not arrays:
        raise ValueError("At least one null array is required.")
    for arr in arrays:
        if arr.ndim != 2:
            raise ValueError(
                f"Each null array must have shape (n_null, n_electrodes), got {arr.shape}"
            )
    n_nulls = {arr.shape[0] for arr in arrays}
    if len(n_nulls) != 1:
        raise ValueError(
            "All subject null arrays must have the same number of samples, got "
            f"{sorted(n_nulls)}."
        )
    n_null = next(iter(n_nulls))
    if n_null == 0:
        raise ValueError("Null arrays must contain at least one sample.")
    return np.nanmax(np.concatenate(arrays, axis=1), axis=1)


def significant_lags_all_electrodes_gpu(
    predicted,
    actual,
    fdr_q=0.01,
    minimal_shift=500,
    device=None,
):
    """
    Identify statistically significant (electrode, lag) pairs via permutation test + FDR.

    Computes observed Pearson correlations between predicted and actual neural
    responses, then builds a null distribution by permuting word order, and
    applies Benjamini-Hochberg FDR correction.

    Args:
        predicted (np.ndarray): Shape (n_electrodes, n_lags, n_words).
        actual (np.ndarray): Shape (n_electrodes, n_lags, n_words).
        fdr_q (float): FDR threshold (alpha) for Benjamini-Hochberg correction.
        seed (int): Random seed for reproducibility.
        device (torch.device, optional): Device to run on. Defaults to CUDA if
            available, else CPU.

    Returns:
        Tuple of (sig, p_values, q_values, obs_corrs), each of shape
        (n_electrodes, n_lags). sig is bool dtype.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    pred = torch.tensor(predicted, dtype=torch.float32, device=device)
    act = torch.tensor(actual, dtype=torch.float32, device=device)
    n_electrodes, n_lags, n_words = pred.shape

    obs_corrs = batch_corrcoef(pred, act)  # (n_electrodes, n_lags)

    null_ge_count = torch.zeros(n_electrodes, n_lags, device=device)
    valid_shifts = range(minimal_shift, n_words - minimal_shift)

    for shift in tqdm(valid_shifts, desc="Exhaustive Circular Shifts"):
        # Roll 'act' by the current shift
        act_shifted = torch.roll(act, shifts=shift, dims=2)
        null_corr = batch_corrcoef(pred, act_shifted)
        null_ge_count += null_corr >= obs_corrs

    n_permutations = len(valid_shifts)
    p_values = to_numpy(null_ge_count / n_permutations)
    obs_corrs_np = to_numpy(obs_corrs)
    flat_p = p_values.ravel()
    reject_flat, q_flat, _, _ = multipletests(flat_p, alpha=fdr_q, method="fdr_bh")
    sig = reject_flat.reshape(n_electrodes, n_lags)
    q_values = q_flat.reshape(n_electrodes, n_lags)

    return sig, p_values, q_values, obs_corrs_np


def save_significance_results(filepath, sig, p_values, q_values, obs_corrs):
    """
    Save outputs of significant_lags_all_electrodes_gpu to a .npz file.

    Parameters
    ----------
    filepath  : str, path to save to (e.g. 'results/significance.npz')
                .npz extension will be added automatically if not present
    sig       : np.ndarray of bool, shape (n_electrodes, n_lags)
    p_values  : np.ndarray, shape (n_electrodes, n_lags)
    q_values  : np.ndarray, shape (n_electrodes, n_lags)
    obs_corrs : np.ndarray, shape (n_electrodes, n_lags)
    """
    os.makedirs(Path(filepath).parent.absolute(), exist_ok=True)
    np.savez(
        filepath,
        sig=sig,
        p_values=p_values,
        q_values=q_values,
        obs_corrs=obs_corrs,
    )


def load_significance_results(filepath):
    """
    Load outputs saved by save_significance_results.

    Parameters
    ----------
    filepath : str, path to .npz file

    Returns
    -------
    sig, p_values, q_values, obs_corrs
    """
    data = np.load(filepath)
    return data["sig"], data["p_values"], data["q_values"], data["obs_corrs"]


def phase_randomize_data(data, rng=None):
    """Phase-randomize a (n_channels, n_samples) numpy array.

    Identical logic to phase_randomize_raw but operates directly on a numpy
    array to avoid MNE object-creation overhead.  Use this in tight loops
    (e.g. null-distribution generation) where the signal is already a
    precomputed numpy array.

    Args:
        data: np.ndarray of shape (n_channels, n_samples).
        rng (numpy.random.Generator, optional): RNG for reproducibility.

    Returns:
        np.ndarray of shape (n_channels, n_samples) with phase-randomized data.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_channels, n_samples = data.shape
    fft_vals = np.fft.rfft(data, axis=1)
    n_freqs = fft_vals.shape[1]

    random_phases = rng.uniform(0, 2 * np.pi, (n_channels, n_freqs))
    random_phases[:, 0] = 0
    if n_samples % 2 == 0:
        random_phases[:, -1] = 0

    fft_randomized = np.abs(fft_vals) * np.exp(1j * random_phases)
    return np.fft.irfft(fft_randomized, n=n_samples, axis=1)


def phase_randomize_raw(raw, rng=None):
    """Phase-randomize each channel of an MNE Raw object.

    For each channel, the Fourier transform is computed, the phase spectrum
    is replaced with random uniform values in [0, 2*pi), and the signal is
    reconstructed via the inverse FFT.  This preserves the power spectrum
    (and therefore the autocorrelation) of each channel while destroying
    any systematic relationship between the neural signal and external events
    (e.g. word onsets).

    Args:
        raw: MNE Raw object (must be preloaded).
        rng (numpy.random.Generator, optional): Random number generator for
            reproducibility.  If None, a new default generator is used.

    Returns:
        mne.io.RawArray: A new Raw object with phase-randomized data and
        the same info structure as the input.
    """
    import mne

    if rng is None:
        rng = np.random.default_rng()

    data = raw.get_data()  # (n_channels, n_samples)
    n_channels, n_samples = data.shape

    # Vectorised FFT across all channels at once
    fft_vals = np.fft.rfft(data, axis=1)  # (n_channels, n_freqs)
    n_freqs = fft_vals.shape[1]

    # Random phases for all channels × frequency bins
    random_phases = rng.uniform(0, 2 * np.pi, (n_channels, n_freqs))
    # DC component must stay real
    random_phases[:, 0] = 0
    # Nyquist component must stay real for even-length signals
    if n_samples % 2 == 0:
        random_phases[:, -1] = 0

    fft_randomized = np.abs(fft_vals) * np.exp(1j * random_phases)
    randomized_data = np.fft.irfft(fft_randomized, n=n_samples, axis=1)

    return mne.io.RawArray(randomized_data, raw.info, verbose=False)


def compute_observed_max_per_electrode(predictions, targets):
    """Compute max-across-lags Pearson correlation per electrode.

    Args:
        predictions (np.ndarray): Shape (n_words, n_electrodes, n_lags).
        targets (np.ndarray): Shape (n_words, n_electrodes, n_lags).

    Returns:
        max_per_electrode (np.ndarray): Shape (n_electrodes,) — max correlation
            across lags for each electrode.
        corrs (np.ndarray): Shape (n_electrodes, n_lags) — full correlation
            matrix.
    """
    pred = torch.tensor(
        predictions.transpose(1, 2, 0), dtype=torch.float32
    )  # (n_elec, n_lags, n_words)
    tgt = torch.tensor(
        targets.transpose(1, 2, 0), dtype=torch.float32
    )  # (n_elec, n_lags, n_words)
    corrs = batch_corrcoef(pred, tgt).numpy()  # (n_elec, n_lags)
    max_per_electrode = corrs.max(axis=1)  # (n_elec,)
    return max_per_electrode, corrs


def get_sig_results(all_predictions, all_targets):
    """
    Convenience wrapper: transpose arrays and run significance testing.

    Parameters
    ----------
    all_predictions : np.ndarray, shape (n_words, n_electrodes, n_lags)
    all_targets     : np.ndarray, shape (n_words, n_electrodes, n_lags)

    Returns
    -------
    sig, p_values, q_values
    """
    electrodes_predictions = all_predictions.transpose(1, 2, 0)
    electrodes_targets = all_targets.transpose(1, 2, 0)
    sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(
        electrodes_predictions, electrodes_targets
    )
    return sig, p_values, q_values
