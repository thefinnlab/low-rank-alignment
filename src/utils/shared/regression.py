"""General regression solver shared by retained workflows."""

import numpy as np
import torch

from src.utils.shared.tensors import to_numpy, to_torch


def solve_regression(
    H: torch.Tensor,
    Y: torch.Tensor,
    regression_type: str,
    device: str,
    standardize: bool = True,
    alphas: np.ndarray | None = None,
    ridge_cv_folds: int = 5,
    solver_params: dict | None = None,
    y_in_cpu: bool | None = None,
    force_cpu: bool | None = None,
    return_alphas: bool = False,
    absorb_scaler: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Solve regression (OLS or Ridge) and return (W, b) tensors.

    Args:
        standardize: When True (default), apply StandardScaler before ridge.
            Set to False when features already have a meaningful variance
            structure (e.g. SVD-derived projections) that should not be
            equalised.
        alphas: Optional array of ridge alpha values for RidgeCV. When
            provided, overrides the default grid from get_ridge_model_constructor.
        ridge_cv_folds: Number of inner CV folds used by RidgeCV alpha
            selection.
        solver_params: Optional Himalaya RidgeCV solver_params. Useful memory
            controls include n_alphas_batch, n_targets_batch, and
            n_targets_batch_refit.
        y_in_cpu: Optional Himalaya RidgeCV Y_in_cpu override.
        force_cpu: Optional Himalaya RidgeCV force_cpu override.
        return_alphas: When True and regression_type is ``"ridge"``, return a
            3-tuple ``(W, b, best_alphas)`` where *best_alphas* is a 1-D
            numpy array of per-target selected alphas. Default False preserves
            the original 2-tuple return.
        absorb_scaler: When standardizing ridge inputs, absorb the fitted scaler
            into W and b so they operate on raw inputs. When False, W and b
            operate on standardized inputs.

    Returns:
        reg_W: (hidden_dim, n_targets) weight tensor.
        reg_b: (n_targets,) bias tensor.
        best_alphas: (n_targets,) numpy array (only when return_alphas=True
            and regression_type="ridge").
    """
    H_np = to_numpy(H)
    Y_np = to_numpy(Y)

    if regression_type == "ridge":
        from src.utils.shared.encoding import get_ridge_model_constructor

        if standardize:
            model = get_ridge_model_constructor(
                alphas=alphas,
                ridge_cv_folds=ridge_cv_folds,
                solver_params=solver_params,
                y_in_cpu=y_in_cpu,
                force_cpu=force_cpu,
            )(0)
            model.fit(H_np, Y_np)
            # Absorb StandardScaler into W, b so that H @ W + b ≈ Y.
            # model.predict(H) = ((H - mean) / scale) @ coef_raw.T + intercept
            # = H @ diag(1/scale) @ coef_raw.T + intercept - (mean/scale) @ coef_raw.T
            coef_raw = to_numpy(
                model[-1].coef_
            )  # may be (n_targets, n_features) or transposed
            n_features = H_np.shape[1]
            if coef_raw.shape != (Y_np.shape[1], n_features):
                coef_raw = coef_raw.T  # → (n_targets, n_features)
            sc_mean = model[0].mean_  # (n_features,)
            sc_scale = model[0].scale_  # (n_features,)
            intercept = to_numpy(model[-1].intercept_)  # (n_targets,)
            if absorb_scaler:
                W = (coef_raw / sc_scale[np.newaxis, :]).T
                b = intercept - (sc_mean / sc_scale) @ coef_raw.T
            else:
                W = coef_raw.T
                b = intercept
            best_alphas_out = (
                to_numpy(model[-1].best_alphas_) if return_alphas else None
            )
        else:
            # Ridge without StandardScaler — appropriate when feature variance
            # carries signal (e.g. SVD-ordered projections).
            ridge_model = get_ridge_model_constructor(
                alphas=alphas,
                ridge_cv_folds=ridge_cv_folds,
                solver_params=solver_params,
                y_in_cpu=y_in_cpu,
                force_cpu=force_cpu,
            )(0)[-1]  # RidgeCV only
            ridge_model.fit(H_np, Y_np)
            coef_raw = to_numpy(ridge_model.coef_)
            n_features = H_np.shape[1]
            if coef_raw.shape != (Y_np.shape[1], n_features):
                coef_raw = coef_raw.T  # → (n_targets, n_features)
            intercept = to_numpy(ridge_model.intercept_)
            W = coef_raw.T  # (n_features, n_targets)
            b = intercept  # (n_targets,)
            best_alphas_out = (
                to_numpy(ridge_model.best_alphas_) if return_alphas else None
            )
        result = (
            to_torch(W.astype(np.float32), device=device),
            to_torch(b.astype(np.float32), device=device),
        )
        if return_alphas:
            return result + (best_alphas_out,)
        return result
    else:
        # OLS: H @ W + b ≈ Y  →  [H, 1] @ [W; b] ≈ Y
        H_t = to_torch(H_np, dtype=torch.float32, device=device)
        Y_t = to_torch(Y_np, dtype=torch.float32, device=device)
        n = H_t.shape[0]
        d = H_t.shape[1]
        H_aug = torch.cat(
            [H_t, torch.ones(n, 1, dtype=torch.float32, device=device)], dim=1
        )
        gram = H_aug.T @ H_aug + 1e-6 * torch.eye(
            d + 1, dtype=torch.float32, device=device
        )
        sol = torch.linalg.solve(gram, H_aug.T @ Y_t)  # (d+1, n_targets)
        return sol[:d], sol[d]
