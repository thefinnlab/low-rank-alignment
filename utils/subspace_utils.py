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
    X_proj_test = X_test @ A  # (n_test, k)

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
        Y_hat,
        Y_hat_train,
        Y_train,
        X_train,
        X_test,
        Y_test,
        Y_hat_full,
        regression_type="ols",
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


def evaluate_input_space(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_subspace_train: np.ndarray,
    Y_hat_subspace_test: np.ndarray,
    device: str = "cpu",
    alphas: np.ndarray | None = None,
    subspace_name: str = "",
) -> dict:
    # Predict raw neural data from complement
    W_raw, b_raw = solve_regression(
        torch.as_tensor(X_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_train, dtype=torch.float32, device=device),
        "ridge",
        device,
        standardize=False,
        alphas=alphas,
    )
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    Y_hat = (X_test_t @ W_raw + b_raw).cpu().numpy()
    comp_corr_arr = compute_correlation(Y_test, Y_hat)

    # Predict residuals from complement
    Y_resid_train = Y_train - Y_hat_subspace_train
    Y_resid_test = Y_test - Y_hat_subspace_test

    W_resid, b_resid = solve_regression(
        torch.as_tensor(X_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_resid_train, dtype=torch.float32, device=device),
        "ridge",
        device,
        standardize=False,
        alphas=alphas,
    )
    Y_hat_resid = (X_test_t @ W_resid + b_resid).cpu().numpy()
    comp_recovery_corr_arr = compute_correlation(Y_resid_test, Y_hat_resid)

    prefix = f"{subspace_name}_" if subspace_name else ""
    return {
        f"{prefix}corr": float(comp_corr_arr.mean()),
        f"{prefix}recovery_corr": float(comp_recovery_corr_arr.mean()),
        f"{prefix}corr_per_target": comp_corr_arr,
        f"{prefix}recovery_corr_per_target": comp_recovery_corr_arr,
        f"{prefix}Y_hat": Y_hat,
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

    return evaluate_input_space(
        X_comp_train,
        X_comp_test,
        Y_train,
        Y_test,
        Y_hat_subspace_train,
        Y_hat_subspace_test,
        alphas=alphas,
        device=device,
        subspace_name="complement",
    )


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
    return evaluate_input_space(
        X_arbitrary_train,
        X_arbitrary_test,
        Y_train,
        Y_test,
        Y_hat_subspace_train,
        Y_hat_subspace_test,
        alphas=alphas,
        device=device,
        subspace_name="arbitrary",
    )


def build_nr_metrics_dict(
    nr_result: dict,
    sig_mask_flat: np.ndarray | None = None,
    complement_result: dict | None = None,
    arbitrary_result: dict | None = None,
    prefix: str = "nr_",
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
        f"{prefix}r2": suf["minimal_r2"],
        f"{prefix}corr": suf["minimal_corr"],
        f"{prefix}recovery_r2": suf["recovery_r2"],
        f"{prefix}recovery_corr": suf["recovery_corr"],
        f"{prefix}pred_r2": suf["pred_r2"],
    }
    if sig_mask_flat is not None:
        add_sig_nonsig_corrs(
            out,
            suf,
            sig_mask_flat,
            [
                (f"{prefix}corr", f"{prefix}minimal_corr_per_target"),
                (f"{prefix}recovery_corr", f"{prefix}recovery_corr_per_target"),
            ],
        )
    if nr_result["cca_sim"] is not None:
        out[f"{prefix}cca_similarity"] = nr_result["cca_sim"]
    if nr_result["proc_sim"] is not None:
        out[f"{prefix}procrustes_similarity"] = nr_result["proc_sim"]

    if complement_result is not None:
        out[f"{prefix}complement_corr"] = complement_result["complement_corr"]
        out[f"{prefix}complement_recovery_corr"] = complement_result[
            "complement_recovery_corr"
        ]
        if sig_mask_flat is not None:
            _add_complement_sig_nonsig(out, complement_result, sig_mask_flat, prefix)

    if arbitrary_result is not None:
        out[f"{prefix}arbitrary_corr"] = arbitrary_result["arbitrary_corr"]
        out[f"{prefix}arbitrary_recovery_corr"] = arbitrary_result[
            "arbitrary_recovery_corr"
        ]
        if sig_mask_flat is not None:
            _add_arbitrary_sig_nonsig(out, arbitrary_result, sig_mask_flat, prefix)

    return out


def _add_complement_sig_nonsig(
    out: dict,
    complement_result: dict,
    sig_mask_flat: np.ndarray,
    prefix: str = "",
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
            float(arr[~sig_mask_flat].mean())
            if (~sig_mask_flat).any()
            else float("nan")
        )


def _add_arbitrary_sig_nonsig(
    out: dict,
    arbitrary_result: dict,
    sig_mask_flat: np.ndarray,
    prefix: str = "",
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
            float(arr[~sig_mask_flat].mean())
            if (~sig_mask_flat).any()
            else float("nan")
        )


def compute_cross_model_metrics(
    Y_hat_A: np.ndarray,
    Y_hat_B: np.ndarray,
    Y_test: np.ndarray,
    label_A: str,
    label_B: str,
) -> dict:
    """Compute prediction correlation and joint model metrics for a pair of models.

    Args:
        Y_hat_A: (n_test, n_targets) predictions from model A.
        Y_hat_B: (n_test, n_targets) predictions from model B.
        Y_test: (n_test, n_targets) true neural targets.
        label_A: Short label for model A (e.g. "rrr", "pca", "complement").
        label_B: Short label for model B (e.g. "full", "complement", "arbitrary").

    Returns:
        Dict with two keys:
            {label_A}_{label_B}_pred_corr  – mean Pearson r between Y_hat_A and Y_hat_B
            {label_A}_{label_B}_joint_corr – mean Pearson r of an OLS model that takes
                [Y_hat_A, Y_hat_B] as inputs and predicts Y_test (unique-variance measure)
    """
    key_prefix = f"{label_A}_{label_B}"

    # Prediction correlation: how similar are the two models' predictions?
    pred_corr_per_target = compute_correlation(Y_hat_A, Y_hat_B)

    # Joint model: OLS on [Y_hat_A[:, t], Y_hat_B[:, t], 1] → Y_test[:, t] per target
    n_test, n_targets = Y_test.shape
    joint_corr_per_target = np.empty(n_targets)
    for t in range(n_targets):
        X_joint = np.column_stack([Y_hat_A[:, t], Y_hat_B[:, t], np.ones(n_test)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_joint, Y_test[:, t], rcond=None)
            Y_pred_joint = X_joint @ coeffs
            c = np.corrcoef(Y_test[:, t], Y_pred_joint)[0, 1]
            joint_corr_per_target[t] = float(c) if not np.isnan(c) else 0.0
        except np.linalg.LinAlgError:
            joint_corr_per_target[t] = 0.0

    return {
        f"{key_prefix}_pred_corr": float(pred_corr_per_target.mean()),
        f"{key_prefix}_joint_corr": float(joint_corr_per_target.mean()),
        f"{key_prefix}_pred_corr_per_target": pred_corr_per_target,
        f"{key_prefix}_joint_corr_per_target": joint_corr_per_target,
    }
