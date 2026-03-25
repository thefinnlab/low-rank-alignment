"""Tests for alpha diagnostic utilities and two-pass training."""

import numpy as np
import pytest
import torch

from utils.constants import (
    RIDGE_ALPHAS,
    refine_alpha_grid,
    check_alpha_grid_boundaries,
)
from utils.encoding import extract_best_alphas, get_ridge_model_constructor
from utils.sae import solve_regression


# ---------------------------------------------------------------------------
# refine_alpha_grid
# ---------------------------------------------------------------------------


class TestRefineAlphaGrid:
    def test_output_length(self):
        best = np.array([1.0, 10.0, 100.0])
        grid = refine_alpha_grid(best, n_points=15)
        assert len(grid) == 15

    def test_spans_observed_range_with_margin(self):
        best = np.array([1.0, 100.0])  # log range: [0, 2]
        grid = refine_alpha_grid(best, n_points=10, margin_decades=1.0)
        assert grid[0] <= 0.1  # 10^(0 - 1)
        assert grid[-1] >= 1000.0  # 10^(2 + 1)

    def test_respects_floor_and_ceiling(self):
        best = np.array([1e-10, 1e15])
        grid = refine_alpha_grid(best, floor=1e-4, ceiling=1e12)
        assert grid[0] >= 1e-4
        assert grid[-1] <= 1e12

    def test_ignores_zero_alphas(self):
        best = np.array([0.0, 0.0, 10.0, 100.0])
        grid = refine_alpha_grid(best, n_points=5)
        assert len(grid) == 5
        assert all(g > 0 for g in grid)


# ---------------------------------------------------------------------------
# check_alpha_grid_boundaries
# ---------------------------------------------------------------------------


class TestCheckAlphaGridBoundaries:
    def test_no_boundary_hit(self):
        grid = np.logspace(0, 4, 5)  # [1, 10, 100, 1000, 10000]
        best = np.array([10.0, 100.0, 1000.0])  # all interior
        result = check_alpha_grid_boundaries(best, grid)
        assert result["hit_lower"] is False
        assert result["hit_upper"] is False

    def test_lower_boundary_hit(self):
        grid = np.logspace(0, 4, 5)
        # 60% at lower boundary
        best = np.array([1.0, 1.0, 1.0, 100.0, 1000.0])
        result = check_alpha_grid_boundaries(best, grid)
        assert result["hit_lower"] is True
        assert result["frac_lower"] == pytest.approx(0.6)

    def test_upper_boundary_hit(self):
        grid = np.logspace(0, 4, 5)
        best = np.array([100.0, 10000.0, 10000.0])
        result = check_alpha_grid_boundaries(best, grid)
        assert result["hit_upper"] is True

    def test_custom_threshold(self):
        grid = np.logspace(0, 4, 5)
        best = np.array([1.0, 100.0, 1000.0])  # 33% at lower
        result = check_alpha_grid_boundaries(best, grid, boundary_fraction_threshold=0.5)
        assert result["hit_lower"] is False  # 33% < 50%


# ---------------------------------------------------------------------------
# extract_best_alphas
# ---------------------------------------------------------------------------


class TestExtractBestAlphas:
    def test_shape_matches_targets(self):
        rng = np.random.default_rng(42)
        n_samples, n_features, n_targets = 100, 5, 3
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        Y = rng.standard_normal((n_samples, n_targets)).astype(np.float32)

        model = get_ridge_model_constructor(alphas=np.logspace(-1, 3, 5))(0)
        model.fit(X, Y)

        best = extract_best_alphas(model)
        assert best.shape == (n_targets,)

    def test_alphas_within_grid(self):
        rng = np.random.default_rng(123)
        X = rng.standard_normal((80, 4)).astype(np.float32)
        Y = rng.standard_normal((80, 2)).astype(np.float32)
        grid = np.logspace(-1, 3, 5)

        model = get_ridge_model_constructor(alphas=grid)(0)
        model.fit(X, Y)

        best = extract_best_alphas(model)
        for a in best:
            assert np.any(np.isclose(a, grid, rtol=1e-3))


# ---------------------------------------------------------------------------
# solve_regression return_alphas
# ---------------------------------------------------------------------------


class TestSolveRegressionReturnAlphas:
    def test_default_returns_two_tuple(self):
        rng = np.random.default_rng(7)
        H = torch.as_tensor(rng.standard_normal((50, 4)), dtype=torch.float32)
        Y = torch.as_tensor(rng.standard_normal((50, 3)), dtype=torch.float32)

        result = solve_regression(H, Y, "ridge", "cpu", standardize=True)
        assert len(result) == 2

    def test_return_alphas_true_gives_three_tuple(self):
        rng = np.random.default_rng(7)
        H = torch.as_tensor(rng.standard_normal((50, 4)), dtype=torch.float32)
        Y = torch.as_tensor(rng.standard_normal((50, 3)), dtype=torch.float32)

        result = solve_regression(
            H, Y, "ridge", "cpu", standardize=True, return_alphas=True
        )
        assert len(result) == 3
        W, b, best_alphas = result
        assert best_alphas.shape == (3,)

    def test_return_alphas_no_standardize(self):
        rng = np.random.default_rng(7)
        H = torch.as_tensor(rng.standard_normal((50, 4)), dtype=torch.float32)
        Y = torch.as_tensor(rng.standard_normal((50, 3)), dtype=torch.float32)

        result = solve_regression(
            H, Y, "ridge", "cpu", standardize=False, return_alphas=True
        )
        assert len(result) == 3
        _, _, best_alphas = result
        assert best_alphas.shape == (3,)

    def test_ols_ignores_return_alphas(self):
        rng = np.random.default_rng(7)
        H = torch.as_tensor(rng.standard_normal((50, 4)), dtype=torch.float32)
        Y = torch.as_tensor(rng.standard_normal((50, 3)), dtype=torch.float32)

        result = solve_regression(
            H, Y, "ols", "cpu", return_alphas=True
        )
        # OLS always returns 2-tuple
        assert len(result) == 2
