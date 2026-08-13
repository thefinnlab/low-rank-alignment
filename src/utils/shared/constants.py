"""Project-wide constants and alpha-grid utilities."""

import numpy as np

# Default ridge regression alpha grid for RidgeCV.
# Covers light regularisation (1e-2) through heavy regularisation (1e10).
RIDGE_ALPHAS = np.logspace(-2, 10, 13)
LARGE_RIDGE_ALPHAS = np.logspace(1, 10, 13)


def refine_alpha_grid(
    best_alphas: np.ndarray,
    n_points: int = 20,
    margin_decades: float = 1.0,
    floor: float = 1e-4,
    ceiling: float = 1e12,
) -> np.ndarray:
    """Build a refined alpha grid centred on observed best alphas.

    Takes an array of per-target best_alphas_ from a coarse RidgeCV sweep
    and returns a denser logspace grid spanning the observed range plus
    *margin_decades* on each side.

    Args:
        best_alphas: 1-D array of selected alphas from a coarse sweep.
        n_points: Number of points in the refined grid.
        margin_decades: Extra decades to extend beyond the observed range.
        floor: Minimum alpha value (clamp).
        ceiling: Maximum alpha value (clamp).

    Returns:
        1-D array of *n_points* alpha values in logspace.
    """
    log_alphas = np.log10(best_alphas[best_alphas > 0])
    lo = max(log_alphas.min() - margin_decades, np.log10(floor))
    hi = min(log_alphas.max() + margin_decades, np.log10(ceiling))
    return np.logspace(lo, hi, n_points)


def check_alpha_grid_boundaries(
    best_alphas: np.ndarray,
    alpha_grid: np.ndarray,
    boundary_fraction_threshold: float = 0.05,
) -> dict:
    """Check whether selected alphas cluster at grid boundaries.

    Args:
        best_alphas: 1-D array of per-target selected alphas.
        alpha_grid: The alpha grid that was used for the sweep.
        boundary_fraction_threshold: Fraction above which a boundary is flagged.

    Returns:
        Dict with keys: hit_lower (bool), hit_upper (bool),
        frac_lower (float), frac_upper (float), n_targets (int).
    """
    lo, hi = alpha_grid.min(), alpha_grid.max()
    frac_lower = float((best_alphas <= lo).mean())
    frac_upper = float((best_alphas >= hi).mean())
    return {
        "hit_lower": frac_lower > boundary_fraction_threshold,
        "hit_upper": frac_upper > boundary_fraction_threshold,
        "frac_lower": frac_lower,
        "frac_upper": frac_upper,
        "n_targets": len(best_alphas),
    }
