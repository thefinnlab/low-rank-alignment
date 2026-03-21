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
    gram = gram + ridge_alpha * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
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

    X = torch.tensor(X_proj_train, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_train, dtype=torch.float32, device=device)

    # Initialise A
    if A_init is not None:
        A = torch.tensor(A_init, dtype=torch.float32, device=device)
    else:
        rng = torch.Generator(device=device)
        rng.manual_seed(0)
        A = torch.randn(n_dirs, k, generator=rng, device=device) * 0.01

    # Initialise B
    if B_init is not None:
        B = torch.tensor(B_init, dtype=torch.float32, device=device)
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
        obj = (resid ** 2).mean().item() + lam * torch.norm(A, p=2, dim=1).sum().item()
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
        "A": A.cpu().numpy(),
        "B": B.cpu().numpy(),
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
            X_sub, Y_sub, k=k, lam=lam,
            max_iter=max_iter, tol=tol, lr=lr,
            A_init=None, B_init=None,
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
        Y_hat_ols_np = precomputed_Y_hat_ols  # (n_train, n_targets)
        beta_ols_np = precomputed_beta_full[selected_idx]  # (n_selected, n_targets)
        Y_hat_ols = torch.tensor(Y_hat_ols_np, dtype=torch.float32, device=device)
        beta_ols = torch.tensor(beta_ols_np, dtype=torch.float32, device=device)
    elif ridge_model is not None:
        # Use the already-fit ridge pipeline (StandardScaler + Ridge/RidgeCV).
        # Y_hat_ols comes from predict(), which correctly applies the scaler and
        # intercept — no separate OLS solve needed.
        Y_hat_ols_np = ridge_model.predict(X_proj_train)  # (n, n_targets)

        # Recover beta in X_proj space by undoing the StandardScaler transform.
        # pipeline[0] = StandardScaler, pipeline[-1] = Ridge/RidgeCV
        # sklearn coef_ shape: (n_targets, n_features); himalaya: (n_features, n_targets)
        # Normalise to (n_features, n_targets) then divide by scale_.
        coef = np.asarray(ridge_model[-1].coef_)           # handles GPU tensors
        n_features_in = X_proj_train.shape[1]
        if coef.shape[0] != n_features_in:
            coef = coef.T                                  # → (n_features, n_targets)
        beta_full = coef / ridge_model[0].scale_[:, np.newaxis]    # (n_features, n_targets)
        beta_ols_np = beta_full[selected_idx]              # (n_selected, n_targets)

        Y_hat_ols = torch.tensor(Y_hat_ols_np, dtype=torch.float32, device=device)
        beta_ols = torch.tensor(beta_ols_np, dtype=torch.float32, device=device)
    else:
        X_sel = X_proj_train[:, selected_idx]  # (n, n_selected)
        X = torch.tensor(X_sel, dtype=torch.float32, device=device)
        Y = torch.tensor(Y_train, dtype=torch.float32, device=device)

        # Step 1: Ridge OLS coefficients beta_ols: (n_selected, n_targets)
        gram = X.T @ X + ridge_alpha * torch.eye(n_selected, dtype=torch.float32, device=device)
        beta_ols = torch.linalg.solve(gram, X.T @ Y)  # (n_selected, n_targets)
        Y_hat_ols = X @ beta_ols

    # Step 2: Training-set predictions and their rank-k SVD
    try:
        _, _, Vh = torch.linalg.svd(Y_hat_ols, full_matrices=False)
    except Exception:
        return {
            "A_dense": np.zeros((n_selected, k), dtype=np.float32),
            "B_dense": np.zeros((n_targets, k), dtype=np.float32),
            "selected_idx": selected_idx,
        }

    actual_k = min(k, Vh.shape[0])
    Vh_k = Vh[:actual_k, :]  # (actual_k, n_targets)

    # Step 3: Factor matrices
    # A = beta_ols @ Vh_k^T  (n_selected, actual_k)
    # B = Vh_k^T             (n_targets, actual_k)
    A_dense = beta_ols @ Vh_k.T   # (n_selected, actual_k)
    B_dense = Vh_k.T              # (n_targets, actual_k)

    # Pad to k if actual_k < k
    if actual_k < k:
        pad = k - actual_k
        A_dense = torch.cat([A_dense, torch.zeros(n_selected, pad, device=device)], dim=1)
        B_dense = torch.cat([B_dense, torch.zeros(n_targets, pad, device=device)], dim=1)

    return {
        "A_dense": A_dense.cpu().numpy(),
        "B_dense": B_dense.cpu().numpy(),
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


def compute_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Compute per-column R^2 (coefficient of determination).

    Constant columns return 0.0 instead of NaN.

    Args:
        Y_true: (n, n_targets).
        Y_pred: (n, n_targets).

    Returns:
        r2: (n_targets,) array.
    """
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    # Where ss_tot == 0 the column is constant; R^2 is defined as 0
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot < 1e-12, 0.0, 1.0 - ss_res / ss_tot)
    return r2


# ---------------------------------------------------------------------------
# E. Sufficiency test
# ---------------------------------------------------------------------------

def build_null_distribution(
    X_proj_train: np.ndarray,
    Y_train: np.ndarray,
    X_proj_test: np.ndarray,
    Y_test: np.ndarray,
    model_constructor_fn=None,
    k: int = 2,
    n_permutations: int = 5000,
    cache_path: str | None = None,
    seed: int = 42,
    device: str | None = None,
) -> np.ndarray:
    """Build a null distribution for mean R^2 via label permutation.

    For each permutation: shuffle Y_train rows, fit a model on the shuffled
    labels, evaluate on un-shuffled Y_test, and record mean R^2.

    Args:
        X_proj_train: (n_train, n_dirs).
        Y_train: (n_train, n_targets).
        X_proj_test: (n_test, n_dirs).
        Y_test: (n_test, n_targets).
        model_constructor_fn: Callable returning a fresh sklearn-compatible model
            on each call (e.g. ``get_ridge_model_constructor()``). When None,
            falls back to full-dim dense RRR of rank ``k`` (used for testing).
        k: Rank for the RRR fallback; ignored when model_constructor_fn is given.
        n_permutations: Number of permutations.
        cache_path: If given, load from file if it exists; save after computing.
        seed: Random seed.
        device: torch device string (only used by the RRR fallback path).

    Returns:
        null_r2: (n_permutations,) array of mean R^2 values under the null.
    """
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)

    rng = np.random.default_rng(seed)
    n_dirs = X_proj_train.shape[1]
    all_idx = np.arange(X_proj_train.shape[0])
    null_r2 = np.zeros(n_permutations)

    for i in tqdm(range(n_permutations), desc="Null distribution", unit="perm"):
        perm_idx = rng.permutation(all_idx)
        Y_perm = Y_train[perm_idx]

        if model_constructor_fn is not None:
            model = model_constructor_fn()(i)
            model.fit(X_proj_train, Y_perm)
            Y_hat = model.predict(X_proj_test)
        else:
            result = fit_dense_rrr(
                X_proj_train, Y_perm, k=k,
                selected_idx=np.arange(n_dirs),
                device=device,
            )
            Y_hat = predict_from_factors(
                X_proj_test,
                result["A_dense"],
                result["B_dense"],
                selected_idx=np.arange(n_dirs),
            )

        r2_vals = compute_r2(Y_test, Y_hat)
        null_r2[i] = float(r2_vals.mean())

    if cache_path is not None:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, null_r2)

    return null_r2


def run_sufficiency_test(
    X_proj_train: np.ndarray,
    Y_resid_train: np.ndarray,
    X_proj_test: np.ndarray,
    Y_test: np.ndarray,
    Y_hat_full: np.ndarray,
    Y_hat_min: np.ndarray,
    null_r2: np.ndarray,
    p_value_threshold: float = 0.05,
    model_constructor_fn=None,
) -> dict:
    """Test whether the minimal model's predictions are sufficient.

    Tests whether the residuals left by the minimal model contain any signal
    that a full-dim model can recover out-of-sample.

    Procedure:
      1. Fit model_constructor_fn on (X_proj_train, Y_resid_train)
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
        model_constructor_fn: Callable returning a factory function that creates
            a fresh sklearn-compatible model (e.g. ``get_ridge_model_constructor``).
            When None, falls back to OLS for backward-compatible testing.

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
    """
    full_r2 = float(compute_r2(Y_test, Y_hat_full).mean())
    minimal_r2 = float(compute_r2(Y_test, Y_hat_min).mean())
    pred_r2 = float(compute_r2(Y_hat_full, Y_hat_min).mean())

    Y_resid_test = Y_test - Y_hat_min

    if model_constructor_fn is not None:
        model = model_constructor_fn()(0)
        model.fit(X_proj_train, Y_resid_train)
        Y_resid_hat = model.predict(X_proj_test)
    else:
        # OLS fallback (no external model dependency; used for testing)
        n_dirs = X_proj_test.shape[1]
        X = torch.tensor(X_proj_test, dtype=torch.float64)
        Yr = torch.tensor(Y_resid_test, dtype=torch.float64)
        gram = X.T @ X + 1e-8 * torch.eye(n_dirs, dtype=torch.float64)
        coef = torch.linalg.solve(gram, X.T @ Yr)
        Y_resid_hat = (X @ coef).numpy()

    recovery_r2 = float(compute_r2(Y_resid_test, Y_resid_hat).mean())

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
            float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1])
            for i in range(n_components)
        ]
        return float(np.mean(corrs))
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
    np.savez(os.path.join(fold_dir, "metrics.npz"), **{k_: np.array(v) for k_, v in metrics.items()})


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


def save_rrr_result(
    output_dir: str,
    k: int,
    fold: int,
    A: np.ndarray,
    B: np.ndarray,
    metrics: dict,
) -> None:
    """Save RRR results for one (k, fold) combination.

    Writes to {output_dir}/k{k}/fold_{fold}/:
        A.npy, B.npy, metrics.npz

    Args:
        output_dir: Root RRR output directory.
        k: Rank.
        fold: Fold index.
        A: (n_embed, k) direction factor matrix.
        B: (n_targets, k) output factor matrix.
        metrics: Dict of scalar metrics.
    """
    fold_dir = os.path.join(output_dir, f"k{k}", f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    np.save(os.path.join(fold_dir, "A.npy"), A)
    np.save(os.path.join(fold_dir, "B.npy"), B)
    np.savez(os.path.join(fold_dir, "metrics.npz"), **{k_: np.array(v) for k_, v in metrics.items() if v is not None})


def load_rrr_result(output_dir: str, k: int, fold: int) -> dict:
    """Load RRR results for one (k, fold) combination.

    Args:
        output_dir: Root RRR output directory.
        k: Rank.
        fold: Fold index.

    Returns:
        dict with keys: A, B, metrics (dict).
    """
    fold_dir = os.path.join(output_dir, f"k{k}", f"fold_{fold}")
    metrics_npz = np.load(os.path.join(fold_dir, "metrics.npz"))
    return {
        "A": np.load(os.path.join(fold_dir, "A.npy")),
        "B": np.load(os.path.join(fold_dir, "B.npy")),
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
            lambdas, r2,
            curve="convex", direction="decreasing",
            S=sensitivity, interp_method="interp1d",
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
