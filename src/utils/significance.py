import os
from pathlib import Path

import numpy as np
import torch
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


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
    return num / denom


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
    p_values = (null_ge_count / n_permutations).cpu().numpy()
    obs_corrs_np = obs_corrs.cpu().numpy()
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
