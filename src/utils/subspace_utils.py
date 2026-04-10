"""Shared utilities for subspace-based evaluation.

Used by both run_rrr.py (non-reduced-rank evaluation from RRR subspace) and
PCA analysis (evaluation from PCA subspace).  Extracting this logic ensures
changes to the evaluation pipeline are reflected in both cases.
"""

import numpy as np
import torch

from src.utils.analysis import partial_out_subspace
from src.utils.sae import solve_regression
from himalaya.scoring import r2_score
from himalaya.scoring import correlation_score


def evaluate_input_space(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    device: str = "cpu",
    alphas: np.ndarray | None = None,
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
        subspace_name: Prefix for all returned keys (e.g. "nr", "complement", "pca").

    Returns:
        Dict with keys (prefixed by ``{subspace_name}_`` when non-empty):
            corr             – float, mean Pearson r vs Y_test
            r2               – float, mean R² vs Y_test
            corr_per_target  – (n_targets,) per-target Pearson r
            r2_per_target    – (n_targets,) per-target R²
            Y_hat            – (n_test, n_targets) predictions
    """
    W, b = solve_regression(
        torch.as_tensor(X_train, dtype=torch.float32, device=device),
        torch.as_tensor(Y_train, dtype=torch.float32, device=device),
        "ridge",
        device,
        standardize=False,
        alphas=alphas,
    )
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    Y_hat = (X_test_t @ W + b).cpu().numpy()

    corr_arr = correlation_score(Y_test, Y_hat)
    r2_arr = r2_score(Y_test, Y_hat)

    prefix = f"{subspace_name}_" if subspace_name else ""
    return {
        f"{prefix}corr": float(corr_arr.mean()),
        f"{prefix}r2": float(r2_arr.mean()),
        f"{prefix}corr_per_target": corr_arr,
        f"{prefix}r2_per_target": r2_arr,
        f"{prefix}Y_hat": Y_hat,
    }


def evaluate_complement(
    A: np.ndarray,
    k: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    alphas: np.ndarray | None = None,
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
        device=device,
        subspace_name="complement",
    )


def evaluate_arbitrary(
    X_arbitrary_train: np.ndarray,
    X_arbitrary_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    alphas: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """Evaluate how well arbitrary (shuffled) embeddings predict neural data.

    Args:
        X_arbitrary_train: (n_train, n_embed) arbitrary embeddings (train split).
        X_arbitrary_test: (n_test, n_embed) arbitrary embeddings (test split).
        Y_train: (n_train, n_targets) training neural targets.
        Y_test: (n_test, n_targets) test neural targets.
        alphas: Ridge alpha grid; None → solver default.
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
    from src.utils.tensor_utils import to_numpy

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
