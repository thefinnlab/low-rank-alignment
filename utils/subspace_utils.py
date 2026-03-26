"""Shared utilities for subspace-based ridge evaluation.

Used by both run_rrr.py (non-reduced-rank ridge from RRR subspace) and
run_pca.py (ridge from PCA subspace).  Extracting this logic ensures
changes to the evaluation pipeline are reflected in both scripts.
"""

import numpy as np
import torch

from utils.rrr_utils import add_sig_nonsig_corrs, run_prediction_metrics
from utils.sae import solve_regression


def evaluate_subspace_ridge(
    A: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_full: np.ndarray,
    alphas: np.ndarray | None = None,
    compute_cca: bool = False,
    compute_procrustes: bool = False,
    log_alphas: bool = False,
    device: str = "cpu",
) -> dict:
    """Project X into a k-dim subspace via *A*, train ridge, and compute metrics.

    Args:
        A: (n_embed, k) projection matrix defining the subspace.
        X_train: (n_train, n_embed) training embeddings.
        X_test: (n_test, n_embed) test embeddings.
        Y_train: (n_train, n_targets) scaled training neural targets.
        Y_test: (n_test, n_targets) scaled test neural targets.
        Y_hat_full: (n_test, n_targets) full-model (unconstrained ridge) predictions.
        alphas: Ridge alpha grid; None → solver default.
        compute_cca: Compute CCA similarity between Y_hat_full and subspace predictions.
        compute_procrustes: Compute Procrustes similarity.
        log_alphas: Return per-target best alphas from the ridge solve.
        device: Torch device string.

    Returns:
        Dict with keys:
            suf        – sufficiency test result dict (from run_prediction_metrics)
            cca_sim    – float | None
            proc_sim   – float | None
            Y_hat      – (n_test, n_targets) subspace ridge predictions
            Y_hat_train – (n_train, n_targets) subspace ridge predictions on train
            best_alphas – np.ndarray | None (populated only when log_alphas=True)
    """
    X_proj_train = X_train @ A  # (n_train, k)
    X_proj_test = X_test @ A    # (n_test, k)

    _result = solve_regression(
        torch.as_tensor(X_proj_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_train, dtype=torch.float32, device=device),
        "ridge",
        device,
        standardize=False,
        alphas=alphas,
        return_alphas=log_alphas,
    )
    if log_alphas:
        W, b, best_alphas = _result
    else:
        W, b = _result
        best_alphas = None

    X_proj_train_t = torch.as_tensor(X_proj_train, dtype=torch.float32, device=device)
    X_proj_test_t = torch.as_tensor(X_proj_test, dtype=torch.float32, device=device)
    Y_hat = (X_proj_test_t @ W + b).cpu().numpy()
    Y_hat_train = (X_proj_train_t @ W + b).cpu().numpy()

    suf, cca_sim, proc_sim = run_prediction_metrics(
        Y_hat, Y_hat_train, Y_train, X_train, X_test, Y_test,
        Y_hat_full, alphas=alphas,
        compute_cca=compute_cca,
        compute_procrustes=compute_procrustes,
    )

    return {
        "suf": suf,
        "cca_sim": cca_sim,
        "proc_sim": proc_sim,
        "Y_hat": Y_hat,
        "Y_hat_train": Y_hat_train,
        "best_alphas": best_alphas,
    }


def build_nr_metrics_dict(
    nr_result: dict,
    sig_mask_flat: np.ndarray | None = None,
) -> dict:
    """Convert :func:`evaluate_subspace_ridge` output into a flat metrics dict.

    All keys are prefixed with ``nr_`` so they can be merged into the
    caller's metrics dict without colliding with reduced-rank columns.

    Args:
        nr_result: Return value of :func:`evaluate_subspace_ridge`.
        sig_mask_flat: Boolean mask over targets (True = significant).

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
    return out
