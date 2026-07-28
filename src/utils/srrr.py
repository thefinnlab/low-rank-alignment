"""Core functions for Sparse Reduced Rank Regression (SRRR) analysis.

Finds the minimal subset of SVD directions that explains all significant
variance captured by the full encoding model, via stability selection and
a statistical sufficiency test.
"""

import os
import warnings

import numpy as np
import torch
from tqdm import tqdm
from himalaya.scoring import correlation_score, r2_score

from src.utils.tensor_utils import to_numpy, to_torch
from src.utils.correlation import fisher_z_mean

# ---------------------------------------------------------------------------
# Tensor contract
# ---------------------------------------------------------------------------
# All *public* functions in this module accept and return np.ndarray.
# Internally, torch tensors are used for GPU-accelerated computation.
# Use to_torch() to enter torch space and to_numpy() to leave it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A. Data utilities
# ---------------------------------------------------------------------------


def load_svd_decomposition(svd_dir: str, n_folds: int) -> np.ndarray:
    """Load per-fold Vt arrays from an svd_decomposition.npz file.

    Args:
        svd_dir: Directory containing svd_decomposition.npz.
        n_folds: Number of folds to load.

    Returns:
        Vt_arr: (n_folds, n_dirs, n_embed) stacked right singular vectors.

    Raises:
        FileNotFoundError: If svd_decomposition.npz is missing.
    """
    path = os.path.join(svd_dir, "svd_decomposition.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"svd_decomposition.npz not found at {path}. "
            "Run run_svd_analysis.py first."
        )
    data = np.load(path)
    Vt_list = [data[f"Vt_fold{f}"] for f in range(n_folds)]
    return np.stack(Vt_list)  # (n_folds, n_dirs, n_embed)


# ---------------------------------------------------------------------------
# B. ProxGD optimizer
# ---------------------------------------------------------------------------


def prox_l21(A: torch.Tensor, threshold: float) -> torch.Tensor:
    """Row-wise group soft-threshold (L2,1 proximal operator).

    Shrinks each row of A towards zero; rows with L2-norm below threshold
    are zeroed entirely.

    Args:
        A: (n_rows, k) tensor.
        threshold: Non-negative soft-threshold value.

    Returns:
        Row-sparse A with shape (n_rows, k).
    """
    row_norms = torch.norm(A, dim=1, keepdim=True)  # (n_rows, 1)
    scale = torch.clamp(1.0 - threshold / (row_norms + 1e-12), min=0.0)
    return A * scale


def solve_B_closed_form(
    X_proj: torch.Tensor,
    Y: torch.Tensor,
    A: torch.Tensor,
    ridge_alpha: float = 1e-6,
) -> torch.Tensor:
    """Closed-form solution for B given fixed A.

    Minimises ||X_proj @ A @ B^T - Y||_F^2.

    B = ((XA)^T (XA) + alpha I)^{-1} (XA)^T Y

    Args:
        X_proj: (n, n_dirs) projected inputs.
        Y: (n, n_targets) outputs.
        A: (n_dirs, k) direction-selection matrix.
        ridge_alpha: Tiny constant added to the diagonal of (XA)^T(XA) purely
            for numerical stabilisation of the linear solve; not a regularisation
            parameter. CV is not appropriate here — this function is called on
            every ProxGD iteration.

    Returns:
        B: (n_targets, k) output factor matrix.
    """
    XA = X_proj @ A  # (n, k)
    gram = XA.T @ XA  # (k, k)
    gram = gram + ridge_alpha * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )
    rhs = XA.T @ Y  # (k, n_targets)
    B_T = torch.linalg.solve(gram, rhs)  # (k, n_targets)
    return B_T.T  # (n_targets, k)


def fit_srrr_proxgd(
    X_proj_train: np.ndarray,
    Y_train: np.ndarray,
    k: int,
    lam: float,
    max_iter: int = 500,
    tol: float = 1e-5,
    lr: float = 1e-3,
    A_init: np.ndarray | None = None,
    B_init: np.ndarray | None = None,
    device: str | None = None,
) -> dict:
    """Fit SRRR via proximal gradient descent (alternating updates).

    Minimises (1/n) ||X_proj @ A @ B^T - Y||_F^2 + lam ||A||_{2,1}
    subject to A of shape (n_dirs, k), B of shape (n_targets, k).

    Update rules:
      - B: closed-form exact solution (given A fixed)
      - A: proxGD step: A ← prox_{lr*lam}(A - lr * grad_A)
        where grad_A = (2/n) X^T (X A B^T - Y) B

    Args:
        X_proj_train: (n, n_dirs) projected training inputs.
        Y_train: (n, n_targets) training outputs.
        k: Rank (number of latent components).
        lam: L2,1 sparsity penalty on A.
        max_iter: Maximum number of alternating iterations.
        tol: Convergence tolerance on relative objective change.
        lr: Learning rate for A gradient step.
        A_init: Optional warm-start for A.
        B_init: Optional warm-start for B.
        device: torch device string; None → auto (cuda if available, else cpu).

    Returns:
        dict with keys:
            A: (n_dirs, k) numpy array
            B: (n_targets, k) numpy array
            n_iter: int
            converged: bool
            obj_history: list of float
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n, n_dirs = X_proj_train.shape
    n_targets = Y_train.shape[1]

    X = to_torch(X_proj_train, dtype=torch.float32, device=device)
    Y = to_torch(Y_train, dtype=torch.float32, device=device)

    # Initialise A
    if A_init is not None:
        A = to_torch(A_init, dtype=torch.float32, device=device)
    else:
        rng = torch.Generator(device=device)
        rng.manual_seed(0)
        A = torch.randn(n_dirs, k, generator=rng, device=device) * 0.01

    # Initialise B
    if B_init is not None:
        B = to_torch(B_init, dtype=torch.float32, device=device)
    else:
        B = solve_B_closed_form(X, Y, A)

    obj_history = []
    prev_obj = None
    converged = False

    for it in range(max_iter):
        # Update B (closed form)
        B = solve_B_closed_form(X, Y, A)

        # Compute residual
        resid = X @ A @ B.T - Y  # (n, n_targets)

        # Objective
        obj = (resid**2).mean().item() + lam * torch.norm(A, p=2, dim=1).sum().item()
        obj_history.append(obj)

        # Convergence check
        if prev_obj is not None and abs(prev_obj - obj) / (abs(prev_obj) + 1e-10) < tol:
            converged = True
            break
        prev_obj = obj

        # Gradient w.r.t. A: (2/n) X^T (X A B^T - Y) B
        grad_A = (2.0 / n) * X.T @ resid @ B  # (n_dirs, k)

        # ProxGD step on A
        A = prox_l21(A - lr * grad_A, lr * lam)

    return {
        "A": to_numpy(A),
        "B": to_numpy(B),
        "n_iter": it + 1,
        "converged": converged,
        "obj_history": obj_history,
    }


# ---------------------------------------------------------------------------
# C. Stability selection
# ---------------------------------------------------------------------------


def run_stability_selection(
    X_proj_train: np.ndarray,
    Y_train: np.ndarray,
    k: int,
    lam: float,
    n_iters: int = 100,
    subsample_frac: float = 0.8,
    max_iter: int = 500,
    tol: float = 1e-5,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
) -> np.ndarray:
    """Estimate direction selection probabilities via stability selection.

    For each of n_iters subsamples (drawn without replacement at subsample_frac),
    fit SRRR and record which rows of A have non-negligible L2-norm.

    Args:
        X_proj_train: (n, n_dirs) projected training inputs.
        Y_train: (n, n_targets) training outputs.
        k: Rank.
        lam: L2,1 penalty.
        n_iters: Number of subsample iterations.
        subsample_frac: Fraction of training samples per subsample.
        max_iter: Max ProxGD iterations per subsample.
        tol: ProxGD convergence tolerance.
        lr: ProxGD learning rate.
        seed: Random seed for reproducibility.
        device: torch device string.

    Returns:
        selection_probs: (n_dirs,) array of floats in [0, 1].
    """
    rng = np.random.default_rng(seed)
    n, n_dirs = X_proj_train.shape
    n_sub = max(1, int(n * subsample_frac))

    counts = np.zeros(n_dirs, dtype=np.float64)

    for _ in range(n_iters):
        idx = rng.choice(n, size=n_sub, replace=False)
        X_sub = X_proj_train[idx]
        Y_sub = Y_train[idx]

        result = fit_srrr_proxgd(
            X_sub,
            Y_sub,
            k=k,
            lam=lam,
            max_iter=max_iter,
            tol=tol,
            lr=lr,
            A_init=None,
            B_init=None,
            device=device,
        )
        A = result["A"]  # (n_dirs, k)
        row_norms = np.linalg.norm(A, axis=1)  # (n_dirs,)
        counts += (row_norms > 1e-8).astype(np.float64)

    return counts / n_iters


def select_directions(
    selection_probs: np.ndarray,
    threshold: float = 0.8,
) -> np.ndarray:
    """Select SVD directions whose stability probability meets the threshold.

    Args:
        selection_probs: (n_dirs,) probabilities from run_stability_selection.
        threshold: Minimum probability to include a direction.

    Returns:
        selected_idx: Integer index array of selected directions.
    """
    return np.where(selection_probs >= threshold)[0].astype(int)


# ---------------------------------------------------------------------------
# D. Consensus model + evaluation
# ---------------------------------------------------------------------------


def fit_dense_rrr(
    X_proj_train: np.ndarray,
    Y_train: np.ndarray,
    k: int,
    selected_idx: np.ndarray,
    ridge_model=None,
    ridge_alpha: float = 1e-6,
    device: str | None = None,
    precomputed_Y_hat_ols: np.ndarray | None = None,
    precomputed_beta_full: np.ndarray | None = None,
    svd_on_cpu: bool = False,
    use_low_rank_svd: bool = False,
    low_rank_svd_seed: int = 0,
) -> dict:
    """Fit dense (non-sparse) Reduced Rank Regression on the selected direction subset.

    Approach:
      1. Obtain beta_ols (n_selected, n_targets): either from precomputed values
         (preferred when using saved encoding models), a pre-fit ridge_model, or
         by solving an unregularized ridge system with ridge_alpha.
      2. Compute Y_hat_ols = X @ beta_ols and take its rank-k SVD: U S Vh
      3. A_dense = beta_ols @ Vh_k^T   (n_selected, k)
         B_dense = Vh_k^T              (n_targets, k)
      4. Prediction: X_new @ A_dense @ B_dense^T
               = X_new @ beta_ols @ (Vh_k^T @ Vh_k)
               = Y_hat_ols_new projected onto top-k output directions

    Args:
        X_proj_train: (n, n_dirs) projected training inputs.
        Y_train: (n, n_targets) training outputs.
        k: Rank.
        selected_idx: Integer indices of selected SVD directions.
        ridge_model: Optional pre-fit sklearn Pipeline(StandardScaler, Ridge/RidgeCV)
            already fit on X_proj_train and Y_train. When provided, its predictions
            and coefficients are used for beta_ols, avoiding a redundant and
            potentially under-regularised solve. ridge_alpha is ignored.
        ridge_alpha: Regularisation constant for the fallback internal solve (used
            only when ridge_model is None and precomputed values are not supplied).
        device: torch device string.
        precomputed_Y_hat_ols: Optional (n_train, n_targets) array of OLS predictions
            on the training set from a saved encoding model. When provided together
            with precomputed_beta_full, the ridge_model path is skipped entirely.
        precomputed_beta_full: Optional (n_dirs, n_targets) array of unscaled
            coefficients in the input feature space. Rows are indexed by direction;
            selected_idx is applied to slice the relevant subset.
        svd_on_cpu: If True, run the SVD on CPU in float64 for improved numerical
            robustness. The resulting singular vectors are converted back to the
            coefficient tensor's device and dtype before constructing the factors.
        use_low_rank_svd: If True, compute only a randomized approximation to the
            leading singular vectors, with 32 oversampling dimensions and two power
            iterations. If False, retain the exact full-SVD behavior.
        low_rank_svd_seed: Random seed used only by the low-rank SVD. Its use is
            isolated so this function does not change the caller's Torch RNG state.

    Returns:
        dict with keys:
            A_dense: (n_selected, k)
            B_dense: (n_targets, k)
            selected_idx: the input selected_idx
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_targets = Y_train.shape[1]
    n_selected = len(selected_idx)
    if n_selected == 0:
        return {
            "A_dense": np.zeros((0, k), dtype=np.float32),
            "B_dense": np.zeros((n_targets, k), dtype=np.float32),
            "selected_idx": selected_idx,
        }

    if precomputed_Y_hat_ols is not None and precomputed_beta_full is not None:
        # Use pre-computed values from a saved encoding model (avoids re-training).
        # to_torch() calls to_numpy() first, so this is safe even if the inputs
        # are torch tensors or himalaya arrays.
        Y_hat_ols = to_torch(precomputed_Y_hat_ols, dtype=torch.float32, device=device)
        beta_ols = to_torch(
            precomputed_beta_full[selected_idx], dtype=torch.float32, device=device
        )
    elif ridge_model is not None:
        # Use the already-fit ridge pipeline (StandardScaler + Ridge/RidgeCV).
        # Y_hat_ols comes from predict(), which correctly applies the scaler and
        # intercept — no separate OLS solve needed.
        Y_hat_ols_np = to_numpy(ridge_model.predict(X_proj_train))  # (n, n_targets)

        # Recover beta in X_proj space by undoing the StandardScaler transform.
        # pipeline[0] = StandardScaler, pipeline[-1] = Ridge/RidgeCV
        # sklearn coef_ shape: (n_targets, n_features); himalaya: (n_features, n_targets)
        # Normalise to (n_features, n_targets) then divide by scale_.
        coef = to_numpy(ridge_model[-1].coef_)
        n_features_in = X_proj_train.shape[1]
        if coef.shape[0] != n_features_in:
            coef = coef.T  # → (n_features, n_targets)
        beta_full = (
            coef / ridge_model[0].scale_[:, np.newaxis]
        )  # (n_features, n_targets)
        beta_ols_np = beta_full[selected_idx]  # (n_selected, n_targets)

        Y_hat_ols = to_torch(Y_hat_ols_np, dtype=torch.float32, device=device)
        beta_ols = to_torch(beta_ols_np, dtype=torch.float32, device=device)
    else:
        X_sel = X_proj_train[:, selected_idx]  # (n, n_selected)
        X = to_torch(X_sel, dtype=torch.float32, device=device)
        Y = to_torch(Y_train, dtype=torch.float32, device=device)

        # Step 1: Ridge OLS coefficients beta_ols: (n_selected, n_targets)
        gram = X.T @ X + ridge_alpha * torch.eye(
            n_selected, dtype=torch.float32, device=device
        )
        beta_ols = torch.linalg.solve(gram, X.T @ Y)  # (n_selected, n_targets)
        Y_hat_ols = X @ beta_ols

    # Step 2: Training-set predictions and their rank-k SVD. CPU mode promotes
    # only the decomposition input to float64; the returned factors retain the
    # existing coefficient dtype/device contract.
    svd_input = (
        Y_hat_ols.to(device="cpu")
        if svd_on_cpu
        else Y_hat_ols
    )
    print("SVD Input:", svd_input.shape)
    if use_low_rank_svd:
        q = min(k + 32, min(svd_input.shape))
        fork_devices = [svd_input.device] if svd_input.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(low_rank_svd_seed)
            _, _, V = torch.svd_lowrank(svd_input, q=q, niter=2)
        Vh = V.T
    else:
        _, _, Vh = torch.linalg.svd(svd_input, full_matrices=False)

    print("Vh:", Vh.shape)
    actual_k = min(k, Vh.shape[0])
    Vh_k = Vh[:actual_k, :].to(device=beta_ols.device, dtype=beta_ols.dtype)

    # Step 3: Factor matrices
    # A = beta_ols @ Vh_k^T  (n_selected, actual_k)
    # B = Vh_k^T             (n_targets, actual_k)
    A_dense = beta_ols @ Vh_k.T  # (n_selected, actual_k)
    B_dense = Vh_k.T  # (n_targets, actual_k)

    # Pad to k if actual_k < k
    if actual_k < k:
        pad = k - actual_k
        A_dense = torch.cat(
            [A_dense, torch.zeros(n_selected, pad, device=device)], dim=1
        )
        B_dense = torch.cat(
            [B_dense, torch.zeros(n_targets, pad, device=device)], dim=1
        )

    return {
        "A_dense": to_numpy(A_dense),
        "B_dense": to_numpy(B_dense),
        "selected_idx": selected_idx,
    }


def predict_from_factors(
    X_proj: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    selected_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Compute predictions Y_hat = X_proj[:, selected_idx] @ A @ B^T.

    Args:
        X_proj: (n_words, n_dirs) projected inputs.
        A: (n_selected, k) or (n_dirs, k) direction factor matrix.
        B: (n_targets, k) output factor matrix.
        selected_idx: Optional integer index array; if given, slices X_proj columns.

    Returns:
        Y_hat: (n_words, n_targets) predictions.
    """
    if selected_idx is not None:
        X_use = X_proj[:, selected_idx]
    else:
        X_use = X_proj
    return X_use @ A @ B.T


# ---------------------------------------------------------------------------
# E. Sufficiency test
# ---------------------------------------------------------------------------


def run_sufficiency_test(
    X_proj_train: np.ndarray,
    Y_resid_train: np.ndarray,
    X_proj_test: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_full: np.ndarray,
    Y_hat_min: np.ndarray,
    null_r2: np.ndarray,
    p_value_threshold: float = 0.05,
    regression_type: str = "ridge",
    alphas: np.ndarray | None = None,
    device: str | None = None,
) -> dict:
    """Test whether the minimal model's predictions are sufficient.

    Tests whether the residuals left by the minimal model contain any signal
    that a full-dim model can recover out-of-sample.

    Procedure:
      1. Fit ridge (or OLS) on (X_proj_train, Y_resid_train)
      2. Predict on X_proj_test; evaluate mean recovery R^2 against Y_test - Y_hat_min
      3. Compare recovery_r2 to null distribution noise floor

    Args:
        X_proj_train: (n_train, n_dirs) full-dimension projections (train set).
        Y_resid_train: (n_train, n_targets) residuals on the training set
            (Y_train - Y_hat_min_train), precomputed by the caller.
        X_proj_test: (n_test, n_dirs) full-dimension projections (test set).
        Y_test: (n_test, n_targets) true outputs (test set).
        Y_hat_full: (n_test, n_targets) full reference model predictions.
        Y_hat_min: (n_test, n_targets) minimal model predictions.
        null_r2: (n_permutations,) null mean R^2 values.
        p_value_threshold: Alpha level for noise floor percentile.
        regression_type: ``"ridge"`` (default) or ``"ols"``.
        alphas: Ridge alpha grid for RidgeCV. When None, uses RIDGE_ALPHAS
            from utils.constants. Ignored when regression_type is ``"ols"``.
        device: torch device string.

    Returns:
        dict with keys:
            recovery_r2: float — mean R^2 of model(X_train → Y_resid_train) on test
            noise_floor: float — null percentile at (1 - p_value_threshold)
            sufficient: bool
            p_value: float — fraction of null >= recovery_r2
            full_r2: float — mean R^2 of full model on Y_test
            minimal_r2: float — mean R^2 of minimal model on Y_test
            pred_r2: float — mean R^2 of minimal model predictions vs full model
                predictions (Y_hat_full as target). Unaffected by ridge shrinkage;
                equals 1 when both models make identical predictions.
            full_corr: float — mean Pearson correlation of full model with Y_test
            minimal_corr: float — mean Pearson correlation of minimal model with Y_test
            recovery_corr: float — mean Pearson correlation of residual model with
                Y_test - Y_hat_min
    """
    from src.utils.sae import solve_regression

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    full_r2 = float(to_numpy(r2_score(Y_test, Y_hat_full)).mean())
    minimal_r2 = float(to_numpy(r2_score(Y_test, Y_hat_min)).mean())
    pred_r2 = float(to_numpy(r2_score(Y_hat_full, Y_hat_min)).mean())

    full_corr_arr = to_numpy(correlation_score(Y_test, Y_hat_full))
    minimal_corr_arr = to_numpy(correlation_score(Y_test, Y_hat_min))
    full_corr = float(fisher_z_mean(full_corr_arr))
    minimal_corr = float(fisher_z_mean(minimal_corr_arr))

    Y_resid_test = Y_test - Y_hat_min

    W, b = solve_regression(
        to_torch(X_proj_train, dtype=torch.float32, device=device),
        to_torch(Y_resid_train, dtype=torch.float32, device=device),
        regression_type,
        device,
        standardize=True,
        alphas=alphas,
    )
    X_test_t = to_torch(X_proj_test, dtype=torch.float32, device=device)
    Y_resid_hat = to_numpy(X_test_t @ W + b)

    recovery_r2 = float(to_numpy(r2_score(Y_resid_test, Y_resid_hat)).mean())
    recovery_corr_arr = to_numpy(correlation_score(Y_resid_test, Y_resid_hat))
    recovery_corr = float(fisher_z_mean(recovery_corr_arr))

    if null_r2 is not None:
        noise_floor = float(np.percentile(null_r2, 100.0 * (1.0 - p_value_threshold)))
        p_value = float(np.mean(null_r2 >= recovery_r2))
        sufficient = recovery_r2 < noise_floor
    else:
        noise_floor = None
        p_value = None
        sufficient = None

    return {
        "recovery_r2": recovery_r2,
        "noise_floor": noise_floor,
        "sufficient": sufficient,
        "p_value": p_value,
        "full_r2": full_r2,
        "minimal_r2": minimal_r2,
        "pred_r2": pred_r2,
        "full_corr": full_corr,
        "minimal_corr": minimal_corr,
        "recovery_corr": recovery_corr,
        "full_corr_per_target": full_corr_arr,
        "minimal_corr_per_target": minimal_corr_arr,
        "recovery_corr_per_target": recovery_corr_arr,
    }


# ---------------------------------------------------------------------------
# F. Alignment metrics
# ---------------------------------------------------------------------------


def compute_cca_similarity(
    Y_hat_full: np.ndarray,
    Y_hat_min: np.ndarray,
    n_components: int | None = None,
) -> float:
    """Mean canonical correlation between full and minimal model predictions.

    Args:
        Y_hat_full: (n, n_targets).
        Y_hat_min: (n, n_targets).
        n_components: Number of CCA components; defaults to min(n_targets, 10).

    Returns:
        Mean canonical correlation in [0, 1].
    """
    from sklearn.cross_decomposition import CCA

    n_targets = Y_hat_full.shape[1]
    if n_components is None:
        n_components = min(n_targets, 10)
    n_components = min(n_components, Y_hat_full.shape[0] - 1, n_targets)

    if n_components < 1:
        return 0.0

    try:
        cca = CCA(n_components=n_components)
        cca.fit(Y_hat_full, Y_hat_min)
        X_c, Y_c = cca.transform(Y_hat_full, Y_hat_min)
        corrs = [
            float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]) for i in range(n_components)
        ]
        return float(fisher_z_mean(corrs))
    except Exception:
        return 0.0


def compute_procrustes_similarity(
    Y_hat_full: np.ndarray,
    Y_hat_min: np.ndarray,
) -> float:
    """Procrustes similarity between full and minimal model predictions.

    Args:
        Y_hat_full: (n, n_targets).
        Y_hat_min: (n, n_targets).

    Returns:
        1 - disparity (in [0, 1]).
    """
    from scipy.spatial import procrustes

    try:
        _, _, disparity = procrustes(Y_hat_full, Y_hat_min)
        return float(1.0 - disparity)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# G. I/O + grid utilities
# ---------------------------------------------------------------------------


def save_srrr_result(
    output_dir: str,
    k: int,
    lam: float,
    fold: int,
    selection_probs: np.ndarray,
    selected_idx: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    metrics: dict,
) -> None:
    """Save SRRR results for one (k, lam, fold) combination.

    Writes to {output_dir}/k{k}_lam{lam:.2e}/fold_{fold}/:
        selection_probs.npy, selected_idx.npy, A.npy, B.npy, metrics.npz

    Args:
        output_dir: Root SRRR output directory.
        k: Rank.
        lam: Lambda value.
        fold: Fold index.
        selection_probs: (n_dirs,) stability probabilities.
        selected_idx: Selected direction indices.
        A: (n_selected, k) factor matrix.
        B: (n_targets, k) factor matrix.
        metrics: Dict of scalar metrics (e.g. from run_sufficiency_test).
    """
    fold_dir = os.path.join(output_dir, f"k{k}_lam{lam:.2e}", f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    np.save(os.path.join(fold_dir, "selection_probs.npy"), selection_probs)
    np.save(os.path.join(fold_dir, "selected_idx.npy"), selected_idx)
    np.save(os.path.join(fold_dir, "A.npy"), A)
    np.save(os.path.join(fold_dir, "B.npy"), B)
    np.savez(
        os.path.join(fold_dir, "metrics.npz"),
        **{k_: np.array(v) for k_, v in metrics.items()},
    )


def load_srrr_result(output_dir: str, k: int, lam: float, fold: int) -> dict:
    """Load SRRR results for one (k, lam, fold) combination.

    Args:
        output_dir: Root SRRR output directory.
        k: Rank.
        lam: Lambda value.
        fold: Fold index.

    Returns:
        dict with keys: selection_probs, selected_idx, A, B, metrics (dict).
    """
    fold_dir = os.path.join(output_dir, f"k{k}_lam{lam:.2e}", f"fold_{fold}")
    metrics_npz = np.load(os.path.join(fold_dir, "metrics.npz"))
    return {
        "selection_probs": np.load(os.path.join(fold_dir, "selection_probs.npy")),
        "selected_idx": np.load(os.path.join(fold_dir, "selected_idx.npy")),
        "A": np.load(os.path.join(fold_dir, "A.npy")),
        "B": np.load(os.path.join(fold_dir, "B.npy")),
        "metrics": dict(metrics_npz),
    }


def save_rrr_factors(
    output_dir: str,
    fold: int,
    A: np.ndarray,
    B: "np.ndarray | None" = None,
) -> None:
    """Save full-rank RRR factor matrices for one fold.

    Writes to {output_dir}/fold_{fold}/:
        A.npy  (n_embed, max_k)
        B.npy  (n_targets, max_k; only when B is provided)

    Args:
        output_dir: Root RRR output directory.
        fold: Fold index.
        A: (n_embed, max_k) full-rank direction factor matrix.
        B: Optional (n_targets, max_k) full-rank output factor matrix.
    """
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    np.save(os.path.join(fold_dir, "A.npy"), A)
    if B is not None:
        np.save(os.path.join(fold_dir, "B.npy"), B)


def save_rrr_result(
    output_dir: str,
    k: int,
    fold: int,
    metrics: dict,
) -> None:
    """Save RRR metrics for one (k, fold) combination.

    Writes to {output_dir}/k{k}/fold_{fold}/:
        metrics.npz

    Args:
        output_dir: Root RRR output directory.
        k: Rank.
        fold: Fold index.
        metrics: Dict of scalar metrics.
    """
    from src.utils.tensor_utils import to_numpy

    fold_dir = os.path.join(output_dir, f"k{k}", f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    np.savez(
        os.path.join(fold_dir, "metrics.npz"),
        **{k_: to_numpy(v) for k_, v in metrics.items() if v is not None},
    )


def load_rrr_result(output_dir: str, k: int, fold: int) -> dict:
    """Load RRR results for one (k, fold) combination.

    Loads full-rank A/B from {output_dir}/fold_{fold}/ and truncates to rank k.
    Loads metrics from {output_dir}/k{k}/fold_{fold}/.

    Args:
        output_dir: Root RRR output directory.
        k: Rank.
        fold: Fold index.

    Returns:
        dict with keys: A (n_embed, k), B (n_targets, k), metrics (dict).
    """
    factors_dir = os.path.join(output_dir, f"fold_{fold}")
    metrics_dir = os.path.join(output_dir, f"k{k}", f"fold_{fold}")
    metrics_npz = np.load(os.path.join(metrics_dir, "metrics.npz"))
    A_full = np.load(os.path.join(factors_dir, "A.npy"))
    B_full = np.load(os.path.join(factors_dir, "B.npy"))
    return {
        "A": A_full[:, :k],
        "B": B_full[:, :k],
        "metrics": dict(metrics_npz),
    }


def find_knee_lambda(
    lambda_grid: list[float],
    r2_values: np.ndarray,
    sensitivity: float = 1.0,
) -> float:
    """Find the knee point in the lambda vs R^2 curve.

    Uses the Kneedle algorithm if available; falls back to argmax.

    Args:
        lambda_grid: Ordered list of lambda values.
        r2_values: Corresponding R^2 values.
        sensitivity: Kneedle sensitivity parameter.

    Returns:
        The lambda value at the knee (or at max R^2 if kneed unavailable).
    """
    lambdas = np.array(lambda_grid)
    r2 = np.array(r2_values)

    try:
        from kneed import KneeLocator

        kl = KneeLocator(
            lambdas,
            r2,
            curve="convex",
            direction="decreasing",
            S=sensitivity,
            interp_method="interp1d",
        )
        if kl.knee is not None:
            # Find closest lambda in grid
            idx = int(np.argmin(np.abs(lambdas - kl.knee)))
            return float(lambdas[idx])
    except ImportError:
        warnings.warn(
            "kneed package not available; falling back to argmax(r2_values) for knee selection.",
            RuntimeWarning,
            stacklevel=2,
        )

    return float(lambdas[int(np.argmax(r2))])
