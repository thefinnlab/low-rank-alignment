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
        "ridge",
        device,
        standardize=False,
        alphas=np.logspace(-2, 10, 13) if A.shape[1] < 50 else np.logspace(1, 10, 13),
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
        regression_type="ridge",
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
    k: int,
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
        A: (n_embed, n_embed) projection matrix defining the subspace.
        k: int, dimension of the subspace to be complemented out.
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
    if k == A.shape[1]:
        X_comp_train = np.ones(
            (X_train.shape[0], 1)
        )  # No complement when subspace is full-dim
        X_comp_test = np.ones((X_test.shape[0], 1))
    else:
        X_comp_train = (X_train @ A)[:, k:]  # Project train embeddings into complement
        X_comp_test = (X_test @ A)[:, k:]  # Project test embeddings into complement

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


def evaluate_complement_and_arbitrary(
    A: np.ndarray,
    k: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_arbitrary: "np.ndarray | None",
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_subspace_train: np.ndarray,
    Y_hat_subspace_test: np.ndarray,
    alphas: "np.ndarray | None" = None,
    device: str = "cpu",
) -> "tuple[dict, dict | None]":
    """Run complement and (optionally) arbitrary evaluations in one call.

    Args:
        A: (n_embed, k) projection matrix defining the subspace.
        X_train: (n_train, n_embed) training embeddings.
        X_test: (n_test, n_embed) test embeddings.
        X_arbitrary: (n_words, n_embed) arbitrary embeddings, or None to skip.
        train_idx: Integer index array selecting training rows from X_arbitrary.
        test_idx: Integer index array selecting test rows from X_arbitrary.
        Y_train: (n_train, n_targets) training neural targets.
        Y_test: (n_test, n_targets) test neural targets.
        Y_hat_subspace_train: (n_train, n_targets) subspace model predictions on train.
        Y_hat_subspace_test: (n_test, n_targets) subspace model predictions on test.
        alphas: Ridge alpha grid; None → solver default.
        device: Torch device string.

    Returns:
        (complement_result, arbitrary_result) where arbitrary_result is None when
        X_arbitrary is None.
    """
    complement_result = evaluate_complement(
        A,
        k,
        X_train,
        X_test,
        Y_train,
        Y_test,
        Y_hat_subspace_train,
        Y_hat_subspace_test,
        alphas=alphas,
        device=device,
    )
    arbitrary_result = None
    if X_arbitrary is not None:
        arbitrary_result = evaluate_arbitrary(
            X_arbitrary[train_idx],
            X_arbitrary[test_idx],
            Y_train,
            Y_test,
            Y_hat_subspace_train,
            Y_hat_subspace_test,
            alphas=alphas,
            device=device,
        )
    return complement_result, arbitrary_result


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
        f"{prefix}corr_per_target": suf["minimal_corr_per_target"],
    }
    if sig_mask_flat is not None:
        add_sig_nonsig_corrs(
            out,
            suf,
            sig_mask_flat,
            [
                (f"{prefix}corr", "minimal_corr_per_target"),
                (f"{prefix}recovery_corr", "recovery_corr_per_target"),
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
        Dict with keys:
            {label_A}_{label_B}_pred_corr         – mean Pearson r between Y_hat_A and Y_hat_B
            {label_A}_{label_B}_pred_corr_per_target – per-target version of the above

        Note: joint_corr (stacked OLS unique-variance metric) is computed separately via
        compute_stacked_joint_corr after the full CV loop to avoid data leakage.
    """
    key_prefix = f"{label_A}_{label_B}"

    # Prediction correlation: how similar are the two models' predictions?
    pred_corr_per_target = compute_correlation(Y_hat_A, Y_hat_B)

    return {
        f"{key_prefix}_pred_corr": float(pred_corr_per_target.mean()),
        f"{key_prefix}_pred_corr_per_target": pred_corr_per_target,
    }


def compute_stacked_joint_corr(
    all_Y_hat_A: list,
    all_Y_hat_B: list,
    all_Y_test: list,
    n_splits: int = 5,
) -> np.ndarray:
    """Compute joint OLS stacking correlation via a proper second cross-validation loop.

    Concatenates out-of-fold predictions from all outer folds into an N-sample dataset,
    then runs a K-fold inner CV to train and evaluate the 2-input stacking OLS without
    data leakage. Uses normal equations (batched across targets) for efficiency with
    large target counts.

    Args:
        all_Y_hat_A: Per-fold predictions from model A, each (n_test_f, n_targets).
        all_Y_hat_B: Per-fold predictions from model B, each (n_test_f, n_targets).
        all_Y_test: Per-fold true neural targets, each (n_test_f, n_targets).
        n_splits: Number of folds for the inner CV.

    Returns:
        (n_targets,) array of mean per-target joint correlation across inner folds.
    """
    from sklearn.model_selection import KFold

    Y_hat_A = np.concatenate(all_Y_hat_A, axis=0)  # (N, n_targets)
    Y_hat_B = np.concatenate(all_Y_hat_B, axis=0)
    Y = np.concatenate(all_Y_test, axis=0)

    n_targets = Y.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=False)
    eye3 = np.eye(3) * 1e-8  # small ridge for numerical stability

    fold_corrs = []
    for train_idx, test_idx in kf.split(Y):
        A_tr = Y_hat_A[train_idx]  # (n_train, n_targets)
        B_tr = Y_hat_B[train_idx]
        Y_tr = Y[train_idx]
        n_train = len(train_idx)

        # Build XtX (n_targets, 3, 3) and XtY (n_targets, 3) via normal equations.
        # Features per target t: [Y_hat_A[:, t], Y_hat_B[:, t], 1].
        # We compute the dot products directly rather than materialising the full
        # (n_train, n_targets, 3) design matrix, which would be ~1 GB at 100k targets.
        AA = np.einsum("it,it->t", A_tr, A_tr)
        AB = np.einsum("it,it->t", A_tr, B_tr)
        BB = np.einsum("it,it->t", B_tr, B_tr)
        AO = A_tr.sum(axis=0)
        BO = B_tr.sum(axis=0)

        XtX = np.empty((n_targets, 3, 3))
        XtX[:, 0, 0] = AA
        XtX[:, 0, 1] = XtX[:, 1, 0] = AB
        XtX[:, 0, 2] = XtX[:, 2, 0] = AO
        XtX[:, 1, 1] = BB
        XtX[:, 1, 2] = XtX[:, 2, 1] = BO
        XtX[:, 2, 2] = n_train
        XtX += eye3

        AY = np.einsum("it,it->t", A_tr, Y_tr)
        BY = np.einsum("it,it->t", B_tr, Y_tr)
        # Stack into (n_targets, 3, 1) to ensure explicit batching
        XtY = np.stack([AY, BY, Y_tr.sum(axis=0)], axis=-1)[:, :, np.newaxis]

        # beta will result in (n_targets, 3, 1)
        beta = np.linalg.solve(XtX, XtY).squeeze(-1)  # Squeeze back to (n_targets, 3)

        A_te = Y_hat_A[test_idx]
        B_te = Y_hat_B[test_idx]
        Y_te = Y[test_idx]
        Y_pred = A_te * beta[:, 0] + B_te * beta[:, 1] + beta[:, 2]

        fold_corrs.append(compute_correlation(Y_te, Y_pred))

    return np.mean(fold_corrs, axis=0)  # (n_targets,)


def add_primary_cross_model_metrics(
    metrics: dict,
    Y_hat_primary: np.ndarray,
    Y_hat_full: np.ndarray,
    Y_hat_complement: np.ndarray,
    Y_hat_arbitrary: "np.ndarray | None",
    Y_test: np.ndarray,
    primary_label: str,
) -> None:
    """Add cross-model metrics for a primary prediction vs full, complement, and arbitrary.

    Adds the following metric groups to *metrics* in-place:
      - {primary_label}_full_*
      - {primary_label}_complement_*
      - complement_full_*
      - {primary_label}_arbitrary_* and complement_arbitrary_* (when Y_hat_arbitrary is not None)

    Args:
        metrics: Dict to update in-place.
        Y_hat_primary: (n_test, n_targets) predictions from the primary model.
        Y_hat_full: (n_test, n_targets) predictions from the full ridge baseline.
        Y_hat_complement: (n_test, n_targets) predictions from the complement model.
        Y_hat_arbitrary: (n_test, n_targets) predictions from the arbitrary baseline, or None.
        Y_test: (n_test, n_targets) true neural targets.
        primary_label: Short label for the primary model (e.g. "rrr", "pca", "nr").
    """
    metrics.update(
        compute_cross_model_metrics(
            Y_hat_primary, Y_hat_full, Y_test, primary_label, "full"
        )
    )
    metrics.update(
        compute_cross_model_metrics(
            Y_hat_primary, Y_hat_complement, Y_test, primary_label, "complement"
        )
    )
    metrics.update(
        compute_cross_model_metrics(
            Y_hat_complement, Y_hat_full, Y_test, "complement", "full"
        )
    )
    if Y_hat_arbitrary is not None:
        metrics.update(
            compute_cross_model_metrics(
                Y_hat_primary, Y_hat_arbitrary, Y_test, primary_label, "arbitrary"
            )
        )
        metrics.update(
            compute_cross_model_metrics(
                Y_hat_complement, Y_hat_arbitrary, Y_test, "complement", "arbitrary"
            )
        )
