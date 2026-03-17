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
    denom = (x ** 2).sum(dim=-1).sqrt() * (y ** 2).sum(dim=-1).sqrt()
    return num / denom


def significant_lags_all_electrodes_gpu(
    predicted,
    actual,
    n_permutations=5000,
    fdr_q=0.01,
    seed=42,
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
        n_permutations (int): Number of permutations for the null distribution.
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

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    null_ge_count = torch.zeros(n_electrodes, n_lags, device=device)

    for _ in tqdm(range(n_permutations), desc="Permutations"):
        idx = torch.randperm(n_words, device=device, generator=rng)
        null_corr = batch_corrcoef(pred, act[:, :, idx])
        null_ge_count += (null_corr >= obs_corrs)

    p_values = (null_ge_count / n_permutations).cpu().numpy()
    obs_corrs_np = obs_corrs.cpu().numpy()
    flat_p = p_values.ravel()
    reject_flat, q_flat, _, _ = multipletests(flat_p, alpha=fdr_q, method='fdr_bh')
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
    return data['sig'], data['p_values'], data['q_values'], data['obs_corrs']


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
