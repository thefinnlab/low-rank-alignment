"""Shared utilities for subspace-based evaluation.

Used by both run_rrr.py (non-reduced-rank evaluation from RRR subspace) and
run_pca.py (evaluation from PCA subspace).  Extracting this logic ensures
changes to the evaluation pipeline are reflected in both scripts.
"""

import numpy as np
import torch

from utils.analysis import partial_out_subspace
from utils.rrr_utils import add_sig_nonsig_corrs, run_prediction_metrics
from utils.sae import solve_regression
from utils.srrr import compute_correlation


def evaluate_subspace_ols(
    A: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_full: np.ndarray,
    compute_cca: bool = False,
    compute_procrustes: bool = False,
    device: str = "cpu",
) -> dict:
    """Project X into a k-dim subspace via *A*, train OLS, and compute metrics.

    Args:
        A: (n_embed, k) projection matrix defining the subspace.
        X_train: (n_train, n_embed) training embeddings.
        X_test: (n_test, n_embed) test embeddings.
        Y_train: (n_train, n_targets) scaled training neural targets.
        Y_test: (n_test, n_targets) scaled test neural targets.
        Y_hat_full: (n_test, n_targets) full-model (unconstrained ridge) predictions.
        compute_cca: Compute CCA similarity between Y_hat_full and subspace predictions.
        compute_procrustes: Compute Procrustes similarity.
        device: Torch device string.

    Returns:
        Dict with keys:
            suf        – sufficiency test result dict (from run_prediction_metrics)
            cca_sim    – float | None
            proc_sim   – float | None
            Y_hat      – (n_test, n_targets) subspace OLS predictions
            Y_hat_train – (n_train, n_targets) subspace OLS predictions on train
    """
    X_proj_train = X_train @ A  # (n_train, k)
    X_proj_test = X_test @ A    # (n_test, k)

    W, b = solve_regression(
        torch.as_tensor(X_proj_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_train, dtype=torch.float32, device=device),
        "ols",
        device,
        standardize=False,
    )

    X_proj_train_t = torch.as_tensor(X_proj_train, dtype=torch.float32, device=device)
    X_proj_test_t = torch.as_tensor(X_proj_test, dtype=torch.float32, device=device)
    Y_hat = (X_proj_test_t @ W + b).cpu().numpy()
    Y_hat_train = (X_proj_train_t @ W + b).cpu().numpy()

    suf, cca_sim, proc_sim = run_prediction_metrics(
        Y_hat, Y_hat_train, Y_train, X_train, X_test, Y_test,
        Y_hat_full, regression_type="ols",
        compute_cca=compute_cca,
        compute_procrustes=compute_procrustes,
    )

    return {
        "suf": suf,
        "cca_sim": cca_sim,
        "proc_sim": proc_sim,
        "Y_hat": Y_hat,
        "Y_hat_train": Y_hat_train,
    }


def evaluate_complement(
    A: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_subspace_train: np.ndarray,
    Y_hat_subspace_test: np.ndarray,
    alphas: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """Evaluate how well the complement of a subspace predicts neural data.

    Projects embeddings orthogonal to the subspace defined by A, then trains
    ridge regression to predict both raw neural data and residuals left by the
    subspace model.

    Args:
        A: (n_embed, k) projection matrix defining the subspace.
        X_train: (n_train, n_embed) training embeddings.
        X_test: (n_test, n_embed) test embeddings.
        Y_train: (n_train, n_targets) training neural targets.
        Y_test: (n_test, n_targets) test neural targets.
        Y_hat_subspace_train: (n_train, n_targets) subspace model predictions on train.
        Y_hat_subspace_test: (n_test, n_targets) subspace model predictions on test.
        alphas: Ridge alpha grid; None → solver default.
        device: Torch device string.

    Returns:
        Dict with keys:
            complement_corr             – float, mean correlation predicting Y_test
            complement_recovery_corr    – float, mean correlation predicting residuals
            complement_corr_per_target  – (n_targets,) per-target correlations
            complement_recovery_corr_per_target – (n_targets,) per-target correlations
    """
    X_comp_train = partial_out_subspace(X_train, A.T)
    X_comp_test = partial_out_subspace(X_test, A.T)

    # Predict raw neural data from complement
    W_raw, b_raw = solve_regression(
        torch.as_tensor(X_comp_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_train, dtype=torch.float32, device=device),
        "ridge", device, standardize=False, alphas=alphas,
    )
    X_comp_test_t = torch.as_tensor(X_comp_test, dtype=torch.float32, device=device)
    Y_hat_comp = (X_comp_test_t @ W_raw + b_raw).cpu().numpy()
    comp_corr_arr = compute_correlation(Y_test, Y_hat_comp)

    # Predict residuals from complement
    Y_resid_train = Y_train - Y_hat_subspace_train
    Y_resid_test = Y_test - Y_hat_subspace_test

    W_resid, b_resid = solve_regression(
        torch.as_tensor(X_comp_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_resid_train, dtype=torch.float32, device=device),
        "ridge", device, standardize=False, alphas=alphas,
    )
    Y_hat_resid = (X_comp_test_t @ W_resid + b_resid).cpu().numpy()
    comp_recovery_corr_arr = compute_correlation(Y_resid_test, Y_hat_resid)

    return {
        "complement_corr": float(comp_corr_arr.mean()),
        "complement_recovery_corr": float(comp_recovery_corr_arr.mean()),
        "complement_corr_per_target": comp_corr_arr,
        "complement_recovery_corr_per_target": comp_recovery_corr_arr,
    }


def evaluate_arbitrary(
    X_arbitrary_train: np.ndarray,
    X_arbitrary_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_subspace_train: np.ndarray,
    Y_hat_subspace_test: np.ndarray,
    alphas: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """Evaluate how well arbitrary (shuffled) embeddings predict neural data.

    Trains ridge regression from pre-computed arbitrary embeddings to predict
    both raw neural data and residuals left by the subspace model.

    Args:
        X_arbitrary_train: (n_train, n_embed) arbitrary embeddings (train split).
        X_arbitrary_test: (n_test, n_embed) arbitrary embeddings (test split).
        Y_train: (n_train, n_targets) training neural targets.
        Y_test: (n_test, n_targets) test neural targets.
        Y_hat_subspace_train: (n_train, n_targets) subspace model predictions on train.
        Y_hat_subspace_test: (n_test, n_targets) subspace model predictions on test.
        alphas: Ridge alpha grid; None → solver default.
        device: Torch device string.

    Returns:
        Dict with keys:
            arbitrary_corr             – float, mean correlation predicting Y_test
            arbitrary_recovery_corr    – float, mean correlation predicting residuals
            arbitrary_corr_per_target  – (n_targets,) per-target correlations
            arbitrary_recovery_corr_per_target – (n_targets,) per-target correlations
    """
    # Predict raw neural data from arbitrary embeddings
    W_raw, b_raw = solve_regression(
        torch.as_tensor(X_arbitrary_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_train, dtype=torch.float32, device=device),
        "ridge", device, standardize=False, alphas=alphas,
    )
    X_arb_test_t = torch.as_tensor(X_arbitrary_test, dtype=torch.float32, device=device)
    Y_hat_arb = (X_arb_test_t @ W_raw + b_raw).cpu().numpy()
    arb_corr_arr = compute_correlation(Y_test, Y_hat_arb)

    # Predict residuals from arbitrary embeddings
    Y_resid_train = Y_train - Y_hat_subspace_train
    Y_resid_test = Y_test - Y_hat_subspace_test

    W_resid, b_resid = solve_regression(
        torch.as_tensor(X_arbitrary_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_resid_train, dtype=torch.float32, device=device),
        "ridge", device, standardize=False, alphas=alphas,
    )
    Y_hat_resid = (X_arb_test_t @ W_resid + b_resid).cpu().numpy()
    arb_recovery_corr_arr = compute_correlation(Y_resid_test, Y_hat_resid)

    return {
        "arbitrary_corr": float(arb_corr_arr.mean()),
        "arbitrary_recovery_corr": float(arb_recovery_corr_arr.mean()),
        "arbitrary_corr_per_target": arb_corr_arr,
        "arbitrary_recovery_corr_per_target": arb_recovery_corr_arr,
    }


def build_nr_metrics_dict(
    nr_result: dict,
    sig_mask_flat: np.ndarray | None = None,
    complement_result: dict | None = None,
    arbitrary_result: dict | None = None,
) -> dict:
    """Convert :func:`evaluate_subspace_ols` output into a flat metrics dict.

    All keys are prefixed with ``nr_`` so they can be merged into the
    caller's metrics dict without colliding with reduced-rank columns.

    Args:
        nr_result: Return value of :func:`evaluate_subspace_ols`.
        sig_mask_flat: Boolean mask over targets (True = significant).
        complement_result: Return value of :func:`evaluate_complement`.
        arbitrary_result: Return value of :func:`evaluate_arbitrary`.

    Returns:
        Flat dict of NR metrics ready to ``metrics.update(…)``.
    """
    suf = nr_result["suf"]
    out: dict = {
        "nr_r2": suf["minimal_r2"],
        "nr_corr": suf["minimal_corr"],
        "nr_recovery_r2": suf["recovery_r2"],
        "nr_recovery_corr": suf["recovery_corr"],
        "nr_pred_r2": suf["pred_r2"],
        "nr_corr_per_target": suf["minimal_corr_per_target"],
    }
    if sig_mask_flat is not None:
        add_sig_nonsig_corrs(out, suf, sig_mask_flat, [
            ("nr_corr", "minimal_corr_per_target"),
            ("nr_recovery_corr", "recovery_corr_per_target"),
        ])
    if nr_result["cca_sim"] is not None:
        out["nr_cca_similarity"] = nr_result["cca_sim"]
    if nr_result["proc_sim"] is not None:
        out["nr_procrustes_similarity"] = nr_result["proc_sim"]

    if complement_result is not None:
        out["nr_complement_corr"] = complement_result["complement_corr"]
        out["nr_complement_recovery_corr"] = complement_result["complement_recovery_corr"]
        if sig_mask_flat is not None:
            _add_complement_sig_nonsig(out, complement_result, sig_mask_flat, "nr_")

    if arbitrary_result is not None:
        out["nr_arbitrary_corr"] = arbitrary_result["arbitrary_corr"]
        out["nr_arbitrary_recovery_corr"] = arbitrary_result["arbitrary_recovery_corr"]
        if sig_mask_flat is not None:
            _add_arbitrary_sig_nonsig(out, arbitrary_result, sig_mask_flat, "nr_")

    return out


def _add_complement_sig_nonsig(
    out: dict, complement_result: dict, sig_mask_flat: np.ndarray, prefix: str = "",
) -> None:
    """Add sig/nonsig splits for complement metrics."""
    for key, per_target_key in [
        ("complement_corr", "complement_corr_per_target"),
        ("complement_recovery_corr", "complement_recovery_corr_per_target"),
    ]:
        arr = complement_result[per_target_key]
        out[f"{prefix}{key}_sig"] = (
            float(arr[sig_mask_flat].mean()) if sig_mask_flat.any() else float("nan")
        )
        out[f"{prefix}{key}_nonsig"] = (
            float(arr[~sig_mask_flat].mean()) if (~sig_mask_flat).any() else float("nan")
        )


def _add_arbitrary_sig_nonsig(
    out: dict, arbitrary_result: dict, sig_mask_flat: np.ndarray, prefix: str = "",
) -> None:
    """Add sig/nonsig splits for arbitrary metrics."""
    for key, per_target_key in [
        ("arbitrary_corr", "arbitrary_corr_per_target"),
        ("arbitrary_recovery_corr", "arbitrary_recovery_corr_per_target"),
    ]:
        arr = arbitrary_result[per_target_key]
        out[f"{prefix}{key}_sig"] = (
            float(arr[sig_mask_flat].mean()) if sig_mask_flat.any() else float("nan")
        )
        out[f"{prefix}{key}_nonsig"] = (
            float(arr[~sig_mask_flat].mean()) if (~sig_mask_flat).any() else float("nan")
        )
