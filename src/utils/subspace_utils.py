"""Shared utilities for subspace-based evaluation.

Used by both run_rrr.py (non-reduced-rank evaluation from RRR subspace) and
PCA analysis (evaluation from PCA subspace).  Extracting this logic ensures
changes to the evaluation pipeline are reflected in both cases.
"""

import numpy as np
import torch

from src.utils.analysis import partial_out_subspace
from src.utils.sae import solve_regression
from src.utils.tensor_utils import to_numpy
from himalaya.scoring import r2_score
from himalaya.scoring import correlation_score


def evaluate_input_space(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    device: str = "cpu",
    alphas: np.ndarray | None = None,
    ridge_cv_folds: int = 5,
    standardize: bool = True,
    ridge_solver_params: dict | None = None,
    ridge_y_in_cpu: bool | None = None,
    ridge_force_cpu: bool | None = None,
    return_alphas: bool = False,
    subspace_name: str = "",
) -> dict:
    """Train ridge on X and evaluate correlation and R² against Y_test.

    Args:
        X_train: (n_train, n_features) training inputs.
        X_test: (n_test, n_features) test inputs.
        Y_train: (n_train, n_targets) training neural targets.
        Y_test: (n_test, n_targets) test neural targets.
        device: Torch device string.
        alphas: Ridge alpha grid; None → solver default.
        ridge_cv_folds: Number of inner CV folds used by RidgeCV alpha
            selection.
        standardize: Whether to standardize inputs before ridge. The fitted
            scaler is absorbed into the returned linear prediction.
        ridge_solver_params: Optional Himalaya RidgeCV solver_params.
        ridge_y_in_cpu: Optional Himalaya RidgeCV Y_in_cpu override.
        ridge_force_cpu: Optional Himalaya RidgeCV force_cpu override.
        return_alphas: Include per-target selected alphas in the result.
        subspace_name: Prefix for all returned keys (e.g. "nr", "complement", "pca").

    Returns:
        Dict with keys (prefixed by ``{subspace_name}_`` when non-empty):
            corr             – float, mean Pearson r vs Y_test
            r2               – float, mean R² vs Y_test
            corr_per_target  – (n_targets,) per-target Pearson r
            r2_per_target    – (n_targets,) per-target R²
            Y_hat            – (n_test, n_targets) predictions
    """
    solved = solve_regression(
        torch.as_tensor(X_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_train, dtype=torch.float32, device=device),
        "ridge",
        device,
        standardize=standardize,
        alphas=alphas,
        ridge_cv_folds=ridge_cv_folds,
        solver_params=ridge_solver_params,
        y_in_cpu=ridge_y_in_cpu,
        force_cpu=ridge_force_cpu,
        return_alphas=return_alphas,
    )
    if return_alphas:
        W, b, best_alphas = solved
    else:
        W, b = solved
        best_alphas = None
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    Y_hat = to_numpy(X_test_t @ W + b)

    corr_arr = to_numpy(correlation_score(Y_test, Y_hat))
    r2_arr = to_numpy(r2_score(Y_test, Y_hat))

    prefix = f"{subspace_name}_" if subspace_name else ""
    result = {
        f"{prefix}corr": float(corr_arr.mean()),
        f"{prefix}r2": float(r2_arr.mean()),
        f"{prefix}corr_per_target": corr_arr,
        f"{prefix}r2_per_target": r2_arr,
        f"{prefix}Y_hat": Y_hat,
    }
    if return_alphas:
        result[f"{prefix}best_alphas"] = best_alphas
    return result


def evaluate_complement(
    A: np.ndarray,
    k: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    alphas: np.ndarray | None = None,
    ridge_cv_folds: int = 5,
    standardize: bool = True,
    ridge_solver_params: dict | None = None,
    ridge_y_in_cpu: bool | None = None,
    ridge_force_cpu: bool | None = None,
    return_alphas: bool = False,
    device: str = "cpu",
) -> dict:
    """Evaluate how well the complement of a subspace predicts neural data.

    Projects embeddings orthogonal to the k-dim subspace defined by the first k
    columns of A, then trains ridge regression to predict raw neural data.

    Args:
        A: (n_embed, n_embed) full projection matrix.
        k: Dimension of the subspace to complement out.
        X_train: (n_train, n_embed) training embeddings.
        X_test: (n_test, n_embed) test embeddings.
        Y_train: (n_train, n_targets) training neural targets.
        Y_test: (n_test, n_targets) test neural targets.
        alphas: Ridge alpha grid; None → solver default.
        ridge_cv_folds: Number of inner CV folds used by RidgeCV alpha
            selection.
        standardize: Whether to standardize complement inputs before ridge.
        ridge_solver_params: Optional Himalaya RidgeCV solver_params.
        ridge_y_in_cpu: Optional Himalaya RidgeCV Y_in_cpu override.
        ridge_force_cpu: Optional Himalaya RidgeCV force_cpu override.
        return_alphas: Include per-target selected alphas in the result.
        device: Torch device string.

    Returns:
        Dict with keys complement_corr, complement_r2, complement_corr_per_target,
        complement_r2_per_target, complement_Y_hat.
    """
    if k == A.shape[1]:
        X_comp_train = np.ones((X_train.shape[0], 1))
        X_comp_test = np.ones((X_test.shape[0], 1))
    else:
        X_comp_train = (X_train @ A)[:, k:]
        X_comp_test = (X_test @ A)[:, k:]

    return evaluate_input_space(
        X_comp_train,
        X_comp_test,
        Y_train,
        Y_test,
        alphas=alphas,
        ridge_cv_folds=ridge_cv_folds,
        standardize=standardize,
        ridge_solver_params=ridge_solver_params,
        ridge_y_in_cpu=ridge_y_in_cpu,
        ridge_force_cpu=ridge_force_cpu,
        return_alphas=return_alphas,
        device=device,
        subspace_name="complement",
    )


def evaluate_arbitrary(
    X_arbitrary_train: np.ndarray,
    X_arbitrary_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    alphas: np.ndarray | None = None,
    ridge_cv_folds: int = 5,
    standardize: bool = True,
    ridge_solver_params: dict | None = None,
    ridge_y_in_cpu: bool | None = None,
    ridge_force_cpu: bool | None = None,
    return_alphas: bool = False,
    device: str = "cpu",
) -> dict:
    """Evaluate how well arbitrary (shuffled) embeddings predict neural data.

    Args:
        X_arbitrary_train: (n_train, n_embed) arbitrary embeddings (train split).
        X_arbitrary_test: (n_test, n_embed) arbitrary embeddings (test split).
        Y_train: (n_train, n_targets) training neural targets.
        Y_test: (n_test, n_targets) test neural targets.
        alphas: Ridge alpha grid; None → solver default.
        ridge_cv_folds: Number of inner CV folds used by RidgeCV alpha
            selection.
        device: Torch device string.

    Returns:
        Dict with keys arbitrary_corr, arbitrary_r2, arbitrary_corr_per_target,
        arbitrary_r2_per_target, arbitrary_Y_hat.
    """
    return evaluate_input_space(
        X_arbitrary_train,
        X_arbitrary_test,
        Y_train,
        Y_test,
        alphas=alphas,
        ridge_cv_folds=ridge_cv_folds,
        standardize=standardize,
        ridge_solver_params=ridge_solver_params,
        ridge_y_in_cpu=ridge_y_in_cpu,
        ridge_force_cpu=ridge_force_cpu,
        return_alphas=return_alphas,
        device=device,
        subspace_name="arbitrary",
    )


def compute_stacked_joint_metrics(
    all_Y_hat_A: list,
    all_Y_hat_B: list,
    all_Y_test: list,
    n_splits: int = 5,
) -> dict:
    """Compute joint OLS stacking metrics via a proper second cross-validation loop.

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
        Dict with keys:
            corr_per_target – (n_targets,) mean per-target joint correlation
            r2_per_target   – (n_targets,) mean per-target joint R²
    """
    from sklearn.model_selection import KFold

    Y_hat_A = np.concatenate(all_Y_hat_A, axis=0)
    Y_hat_B = np.concatenate(all_Y_hat_B, axis=0)
    Y = np.concatenate(all_Y_test, axis=0)

    n_targets = Y.shape[1]
    if n_splits < 2:
        return {"corr_per_target": np.zeros(n_targets), "r2_per_target": np.zeros(n_targets)}

    kf = KFold(n_splits=n_splits, shuffle=False)
    eye3 = np.eye(3) * 1e-8

    fold_corrs = []
    fold_r2s = []
    for train_idx, test_idx in kf.split(Y):
        A_tr = Y_hat_A[train_idx]
        B_tr = Y_hat_B[train_idx]
        Y_tr = Y[train_idx]
        n_train = len(train_idx)

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
        XtY = np.stack([AY, BY, Y_tr.sum(axis=0)], axis=-1)[:, :, np.newaxis]

        beta = np.linalg.solve(XtX, XtY).squeeze(-1)

        A_te = Y_hat_A[test_idx]
        B_te = Y_hat_B[test_idx]
        Y_te = Y[test_idx]
        Y_pred = A_te * beta[:, 0] + B_te * beta[:, 1] + beta[:, 2]

        fold_corrs.append(to_numpy(correlation_score(Y_te, Y_pred)))
        fold_r2s.append(to_numpy(r2_score(Y_te, Y_pred)))

    return {
        "corr_per_target": np.mean(fold_corrs, axis=0),
        "r2_per_target": np.mean(fold_r2s, axis=0),
    }


def _targetwise_standardize_pair(
    A: np.ndarray, B: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize target-specific A/B prediction pairs.

    Each target has its own two stacking features: ``A[:, t]`` and ``B[:, t]``.
    This mirrors ``StandardScaler`` for those per-target 2-feature designs while
    keeping the computation batched across targets.
    """
    means = np.stack([A.mean(axis=0), B.mean(axis=0)], axis=0)
    scales = np.stack([A.std(axis=0), B.std(axis=0)], axis=0)
    scales[scales == 0] = 1.0
    return (A - means[0]) / scales[0], (B - means[1]) / scales[1], means, scales


def _fit_targetwise_ridge_coefficients(
    A_train: np.ndarray,
    B_train: np.ndarray,
    Y_train: np.ndarray,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit target-specific 2-feature ridge stackers in one batched solve."""
    ZA, ZB, means, scales = _targetwise_standardize_pair(A_train, B_train)
    Y_mean = Y_train.mean(axis=0)
    Y_centered = Y_train - Y_mean

    n_targets = Y_train.shape[1]
    n_alphas = len(alphas)

    AA = np.einsum("it,it->t", ZA, ZA)
    AB = np.einsum("it,it->t", ZA, ZB)
    BB = np.einsum("it,it->t", ZB, ZB)
    AY = np.einsum("it,it->t", ZA, Y_centered)
    BY = np.einsum("it,it->t", ZB, Y_centered)

    XtX = np.empty((n_alphas, n_targets, 2, 2), dtype=np.float64)
    XtX[:, :, 0, 0] = AA[np.newaxis, :] + alphas[:, np.newaxis]
    XtX[:, :, 0, 1] = XtX[:, :, 1, 0] = AB[np.newaxis, :]
    XtX[:, :, 1, 1] = BB[np.newaxis, :] + alphas[:, np.newaxis]
    XtY = np.stack([AY, BY], axis=-1)[np.newaxis, :, :, np.newaxis]
    beta = np.linalg.solve(XtX, XtY).squeeze(-1)  # (n_alphas, n_targets, 2)
    return beta, means, scales, Y_mean


def _predict_targetwise_ridge_stack(
    A: np.ndarray,
    B: np.ndarray,
    beta: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    Y_mean: np.ndarray,
    alpha_idx: int | np.ndarray,
) -> np.ndarray:
    """Predict from batched target-wise ridge stacker coefficients."""
    ZA_test = (A - means[0]) / scales[0]
    ZB_test = (B - means[1]) / scales[1]
    if np.isscalar(alpha_idx):
        beta_alpha = beta[int(alpha_idx)]
    else:
        target_idx = np.arange(A.shape[1])
        beta_alpha = beta[alpha_idx, target_idx]
    return ZA_test * beta_alpha[:, 0] + ZB_test * beta_alpha[:, 1] + Y_mean


def _select_targetwise_ridge_alphas(
    A_train: np.ndarray,
    B_train: np.ndarray,
    Y_train: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Select per-target ridge alphas with the same fold structure as RidgeCV."""
    from sklearn.model_selection import KFold

    n_train = Y_train.shape[0]
    inner_splits = min(5, n_train)
    if inner_splits < 2:
        return np.zeros(Y_train.shape[1], dtype=int)

    fold_mse = []
    for inner_train_idx, inner_val_idx in KFold(
        n_splits=inner_splits, shuffle=False
    ).split(Y_train):
        beta, means, scales, Y_mean = _fit_targetwise_ridge_coefficients(
            A_train[inner_train_idx],
            B_train[inner_train_idx],
            Y_train[inner_train_idx],
            alphas,
        )
        alpha_mse = np.empty((len(alphas), Y_train.shape[1]), dtype=np.float64)
        for alpha_idx in range(len(alphas)):
            Y_val_pred = _predict_targetwise_ridge_stack(
                A_train[inner_val_idx],
                B_train[inner_val_idx],
                beta,
                means,
                scales,
                Y_mean,
                alpha_idx,
            )
            err = Y_val_pred - Y_train[inner_val_idx]
            alpha_mse[alpha_idx] = np.mean(err**2, axis=0)
        fold_mse.append(alpha_mse)

    return np.argmin(np.mean(fold_mse, axis=0), axis=0)


def compute_stacked_joint_ridge_means(
    all_Y_hat_A: list,
    all_Y_hat_B: list,
    all_Y_test: list,
    n_splits: int = 5,
    alphas: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """Compute scalar ridge-stacked joint metrics via second-level CV.

    This mirrors ``compute_stacked_joint_metrics`` but fits a ridge stacker on
    the two prediction streams instead of OLS. Each target still gets its own
    A/B stacking weights and alpha, but the ridge solves and alpha search are
    batched across targets instead of fitting one estimator per target.
    Only mean correlation/R² are returned because this is intended as an
    optional robustness check.
    """
    from sklearn.model_selection import KFold
    from src.utils.constants import RIDGE_ALPHAS

    Y_hat_A = np.concatenate(all_Y_hat_A, axis=0)
    Y_hat_B = np.concatenate(all_Y_hat_B, axis=0)
    Y = np.concatenate(all_Y_test, axis=0)

    if n_splits < 2:
        return {"corr": 0.0, "r2": 0.0}

    if alphas is None:
        alphas = RIDGE_ALPHAS
    alphas = np.asarray(alphas, dtype=np.float64)

    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_corrs = []
    fold_r2s = []

    for train_idx, test_idx in kf.split(Y):
        A_tr = Y_hat_A[train_idx]
        B_tr = Y_hat_B[train_idx]
        Y_tr = Y[train_idx]

        best_alpha_idx = _select_targetwise_ridge_alphas(A_tr, B_tr, Y_tr, alphas)
        beta, means, scales, Y_mean = _fit_targetwise_ridge_coefficients(
            A_tr,
            B_tr,
            Y_tr,
            alphas,
        )
        Y_pred = _predict_targetwise_ridge_stack(
            Y_hat_A[test_idx],
            Y_hat_B[test_idx],
            beta,
            means,
            scales,
            Y_mean,
            best_alpha_idx,
        )

        fold_corrs.append(to_numpy(correlation_score(Y[test_idx], Y_pred)))
        fold_r2s.append(to_numpy(r2_score(Y[test_idx], Y_pred)))

    return {
        "corr": float(np.mean(fold_corrs)),
        "r2": float(np.mean(fold_r2s)),
    }
