"""Unit tests for utils/srrr.py."""

import os
import warnings

import numpy as np
import pytest
import torch

from src.utils.srrr import (
    build_null_distribution,
    compute_cca_similarity,
    compute_procrustes_similarity,
    correlation_score,
    find_knee_lambda,
    fit_dense_rrr,
    fit_srrr_proxgd,
    predict_from_factors,
    prox_l21,
    run_stability_selection,
    run_sufficiency_test,
    select_directions,
    solve_B_closed_form,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_data(rng):
    """Small synthetic (X_proj, Y) pair for testing."""
    n, n_dirs, n_targets = 60, 10, 4
    X = rng.standard_normal((n, n_dirs)).astype(np.float32)
    Y = rng.standard_normal((n, n_targets)).astype(np.float32)
    return X, Y


# ---------------------------------------------------------------------------
# TestProxL21
# ---------------------------------------------------------------------------


class TestProxL21:
    def test_zero_rows_below_threshold(self):
        A = torch.tensor([[0.5, 0.5], [0.1, 0.0]], dtype=torch.float32)
        result = prox_l21(A, threshold=1.0)
        # Row 0: norm ≈ 0.707 < 1.0 → should be zeroed
        assert result[0].norm().item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_shrinkage_above_threshold(self):
        # Row with norm 1.0, threshold 0.5 → norm should become 0.5
        A = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        result = prox_l21(A, threshold=0.5)
        assert result[0, 0].item() == pytest.approx(0.5, abs=1e-5)

    def test_shape_preserved(self):
        A = torch.randn(8, 3)
        result = prox_l21(A, threshold=0.01)
        assert result.shape == A.shape

    def test_threshold_zero_is_identity(self):
        A = torch.randn(5, 4)
        result = prox_l21(A, threshold=0.0)
        assert torch.allclose(result, A, atol=1e-6)


# ---------------------------------------------------------------------------
# TestSolveBClosedForm
# ---------------------------------------------------------------------------


class TestSolveBClosedForm:
    def test_output_shape(self, small_data):
        X, Y = small_data
        k = 3
        Xt = torch.tensor(X)
        Yt = torch.tensor(Y)
        A = torch.randn(X.shape[1], k)
        B = solve_B_closed_form(Xt, Yt, A)
        assert B.shape == (Y.shape[1], k)

    def test_recovers_B_on_linear_system(self):
        rng = np.random.default_rng(0)
        n, n_dirs, k, n_targets = 80, 6, 2, 3
        A_true = rng.standard_normal((n_dirs, k)).astype(np.float32)
        B_true = rng.standard_normal((n_targets, k)).astype(np.float32)
        X = rng.standard_normal((n, n_dirs)).astype(np.float32)
        Y = (X @ A_true @ B_true.T).astype(np.float32)

        Xt = torch.tensor(X)
        Yt = torch.tensor(Y)
        At = torch.tensor(A_true)
        B_hat = solve_B_closed_form(Xt, Yt, At, ridge_alpha=1e-8)
        Y_hat = (Xt @ At @ B_hat.T).numpy()
        # Should reconstruct Y well
        assert np.mean((Y - Y_hat) ** 2) < 1e-3

    def test_no_error_on_near_singular_A(self):
        n, n_dirs, k = 20, 4, 4
        X = torch.randn(n, n_dirs)
        Y = torch.randn(n, 3)
        A = torch.zeros(n_dirs, k)  # Completely degenerate
        # Should not raise
        B = solve_B_closed_form(X, Y, A, ridge_alpha=1e-6)
        assert B.shape == (3, k)


# ---------------------------------------------------------------------------
# TestFitSRRRProxGD
# ---------------------------------------------------------------------------


class TestFitSRRRProxGD:
    def test_output_keys_and_shapes(self, small_data):
        X, Y = small_data
        k = 3
        result = fit_srrr_proxgd(X, Y, k=k, lam=0.01, max_iter=20, device="cpu")
        assert set(result.keys()) == {"A", "B", "n_iter", "converged", "obj_history"}
        assert result["A"].shape == (X.shape[1], k)
        assert result["B"].shape == (Y.shape[1], k)

    def test_high_lambda_most_rows_zero(self, small_data):
        X, Y = small_data
        k = 3
        result = fit_srrr_proxgd(X, Y, k=k, lam=100.0, max_iter=100, device="cpu")
        A = result["A"]
        row_norms = np.linalg.norm(A, axis=1)
        n_zero = np.sum(row_norms < 1e-6)
        # With very high lambda, most rows should be zero
        assert n_zero >= X.shape[1] // 2

    def test_low_lambda_few_rows_zero(self, small_data):
        X, Y = small_data
        k = 3
        result = fit_srrr_proxgd(X, Y, k=k, lam=1e-6, max_iter=100, device="cpu")
        A = result["A"]
        row_norms = np.linalg.norm(A, axis=1)
        n_nonzero = np.sum(row_norms > 1e-8)
        assert n_nonzero >= 1

    def test_warm_start_uses_fewer_iters(self, small_data):
        X, Y = small_data
        k = 3
        # First run to convergence
        r1 = fit_srrr_proxgd(X, Y, k=k, lam=0.01, max_iter=200, tol=1e-4, device="cpu")
        # Warm start from converged solution
        r2 = fit_srrr_proxgd(
            X,
            Y,
            k=k,
            lam=0.01,
            max_iter=200,
            tol=1e-4,
            A_init=r1["A"],
            B_init=r1["B"],
            device="cpu",
        )
        assert r2["n_iter"] <= r1["n_iter"]

    def test_obj_history_non_increasing(self, small_data):
        X, Y = small_data
        k = 3
        result = fit_srrr_proxgd(X, Y, k=k, lam=0.01, max_iter=50, device="cpu")
        obj = result["obj_history"]
        # Allow small numerical noise
        diffs = np.diff(obj)
        assert np.all(
            diffs <= 1e-4
        ), f"Objective increased by more than 1e-4: {diffs[diffs > 1e-4]}"


# ---------------------------------------------------------------------------
# TestRunStabilitySelection
# ---------------------------------------------------------------------------


class TestRunStabilitySelection:
    def test_output_shape_and_range(self, small_data):
        X, Y = small_data
        probs = run_stability_selection(
            X, Y, k=2, lam=0.1, n_iters=5, subsample_frac=0.8, max_iter=20, device="cpu"
        )
        assert probs.shape == (X.shape[1],)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_high_lambda_all_probs_near_zero(self, small_data):
        X, Y = small_data
        probs = run_stability_selection(
            X,
            Y,
            k=2,
            lam=1000.0,
            n_iters=5,
            subsample_frac=0.8,
            max_iter=30,
            device="cpu",
        )
        assert np.all(probs < 0.5)

    def test_reproducible_with_seed(self, small_data):
        X, Y = small_data
        p1 = run_stability_selection(
            X,
            Y,
            k=2,
            lam=0.1,
            n_iters=5,
            subsample_frac=0.8,
            max_iter=20,
            seed=7,
            device="cpu",
        )
        p2 = run_stability_selection(
            X,
            Y,
            k=2,
            lam=0.1,
            n_iters=5,
            subsample_frac=0.8,
            max_iter=20,
            seed=7,
            device="cpu",
        )
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# TestSelectDirections
# ---------------------------------------------------------------------------


class TestSelectDirections:
    def test_empty_when_all_below_threshold(self):
        probs = np.array([0.1, 0.2, 0.3])
        idx = select_directions(probs, threshold=0.8)
        assert len(idx) == 0

    def test_correct_subset(self):
        probs = np.array([0.9, 0.3, 0.85, 0.1])
        idx = select_directions(probs, threshold=0.8)
        np.testing.assert_array_equal(idx, [0, 2])

    def test_int_dtype(self):
        probs = np.array([0.9, 0.1])
        idx = select_directions(probs, threshold=0.8)
        assert idx.dtype in (np.int32, np.int64, int)


# ---------------------------------------------------------------------------
# TestComputeCorrelation
# ---------------------------------------------------------------------------


class TestComputeCorrelation:
    def test_perfect_prediction_returns_one(self, rng):
        Y = rng.standard_normal((30, 4)).astype(np.float32)
        corr = correlation_score(Y, Y)
        np.testing.assert_allclose(corr, 1.0, atol=1e-5)

    def test_uncorrelated_returns_near_zero(self, rng):
        Y_true = rng.standard_normal((1000, 4)).astype(np.float32)
        Y_pred = rng.standard_normal((1000, 4)).astype(np.float32)
        corr = correlation_score(Y_true, Y_pred)
        assert np.all(np.abs(corr) < 0.15)

    def test_output_shape(self, rng):
        Y = rng.standard_normal((20, 5)).astype(np.float32)
        corr = correlation_score(Y, Y * 0.9 + 1.0)
        assert corr.shape == (5,)

    def test_negative_correlation(self, rng):
        Y = rng.standard_normal((30, 2)).astype(np.float32)
        corr = correlation_score(Y, -Y)
        np.testing.assert_allclose(corr, -1.0, atol=1e-5)

    def test_constant_column_returns_zero_not_nan(self):
        Y = np.ones((20, 1), dtype=np.float32)
        Y_pred = np.zeros((20, 1), dtype=np.float32)
        corr = correlation_score(Y, Y_pred)
        assert np.isfinite(corr[0])
        assert corr[0] == 0.0

    def test_scale_invariant(self, rng):
        Y = rng.standard_normal((30, 3)).astype(np.float32)
        corr1 = correlation_score(Y, Y * 0.5)
        corr2 = correlation_score(Y, Y * 100.0)
        np.testing.assert_allclose(corr1, corr2, atol=1e-5)


# ---------------------------------------------------------------------------
# TestPredictFromFactors
# ---------------------------------------------------------------------------


class TestPredictFromFactors:
    def test_shape_without_selected_idx(self, rng):
        n, n_dirs, k, n_targets = 20, 8, 3, 5
        X = rng.standard_normal((n, n_dirs)).astype(np.float32)
        A = rng.standard_normal((n_dirs, k)).astype(np.float32)
        B = rng.standard_normal((n_targets, k)).astype(np.float32)
        Y_hat = predict_from_factors(X, A, B)
        assert Y_hat.shape == (n, n_targets)

    def test_shape_with_selected_idx(self, rng):
        n, n_dirs, k, n_targets = 20, 10, 3, 4
        X = rng.standard_normal((n, n_dirs)).astype(np.float32)
        selected = np.array([0, 2, 5])
        A = rng.standard_normal((3, k)).astype(np.float32)
        B = rng.standard_normal((n_targets, k)).astype(np.float32)
        Y_hat = predict_from_factors(X, A, B, selected_idx=selected)
        assert Y_hat.shape == (n, n_targets)

    def test_correct_values(self, rng):
        n, n_dirs, k, n_targets = 10, 4, 2, 3
        X = rng.standard_normal((n, n_dirs)).astype(np.float64)
        A = rng.standard_normal((n_dirs, k)).astype(np.float64)
        B = rng.standard_normal((n_targets, k)).astype(np.float64)
        Y_hat = predict_from_factors(X, A, B)
        expected = X @ A @ B.T
        np.testing.assert_allclose(Y_hat, expected, rtol=1e-5)

    def test_correct_values_with_selected_idx(self, rng):
        n, n_dirs, k, n_targets = 10, 8, 2, 3
        X = rng.standard_normal((n, n_dirs)).astype(np.float64)
        selected = np.array([1, 3, 6])
        A = rng.standard_normal((3, k)).astype(np.float64)
        B = rng.standard_normal((n_targets, k)).astype(np.float64)
        Y_hat = predict_from_factors(X, A, B, selected_idx=selected)
        expected = X[:, selected] @ A @ B.T
        np.testing.assert_allclose(Y_hat, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# TestBuildNullDistribution
# ---------------------------------------------------------------------------


class TestBuildNullDistribution:
    def test_shape_ols(self, small_data):
        X, Y = small_data
        n_train = 40
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        n_perms = 8
        null = build_null_distribution(
            X_train,
            Y_train,
            X_test,
            Y_test,
            regression_type="ols",
            k=2,
            n_permutations=n_perms,
            device="cpu",
            seed=0,
        )
        assert null.shape == (n_perms,)

    def test_shape_ridge(self, small_data):
        X, Y = small_data
        n_train = 40
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        n_perms = 8
        null = build_null_distribution(
            X_train,
            Y_train,
            X_test,
            Y_test,
            regression_type="ridge",
            n_permutations=n_perms,
            device="cpu",
            seed=0,
        )
        assert null.shape == (n_perms,)

    def test_near_zero_r2_on_random_data(self, rng):
        # Completely random data → null R^2 should be near 0
        n, n_dirs, n_targets = 60, 8, 3
        X = rng.standard_normal((n, n_dirs)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)
        X_train, X_test = X[:40], X[40:]
        Y_train, Y_test = Y[:40], Y[40:]
        null = build_null_distribution(
            X_train,
            Y_train,
            X_test,
            Y_test,
            regression_type="ols",
            k=2,
            n_permutations=20,
            device="cpu",
            seed=1,
        )
        assert np.abs(null.mean()) < 0.5  # Should be small under null

    def test_cache_roundtrip(self, tmp_path, small_data):
        X, Y = small_data
        n_train = 40
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        cache_file = str(tmp_path / "null_dist.npy")

        # First call — computes and caches
        null1 = build_null_distribution(
            X_train,
            Y_train,
            X_test,
            Y_test,
            regression_type="ols",
            k=2,
            n_permutations=5,
            device="cpu",
            seed=42,
            cache_path=cache_file,
        )
        assert os.path.exists(cache_file)

        # Second call — loads from cache (should be identical)
        null2 = build_null_distribution(
            X_train,
            Y_train,
            X_test,
            Y_test,
            regression_type="ols",
            k=2,
            n_permutations=5,
            device="cpu",
            seed=99,  # different seed shouldn't matter
            cache_path=cache_file,
        )
        np.testing.assert_array_equal(null1, null2)


# ---------------------------------------------------------------------------
# TestRunSufficiencyTest
# ---------------------------------------------------------------------------


class TestRunSufficiencyTest:
    def _make_null(self, rng, n=200):
        return rng.standard_normal(n).astype(np.float64) * 0.01

    def _split(self, rng, n_train, n_test, n_dirs, n_targets):
        X_train = rng.standard_normal((n_train, n_dirs)).astype(np.float64)
        Y_train = rng.standard_normal((n_train, n_targets)).astype(np.float64)
        X_test = rng.standard_normal((n_test, n_dirs)).astype(np.float64)
        Y_test = rng.standard_normal((n_test, n_targets)).astype(np.float64)
        return X_train, Y_train, X_test, Y_test

    def test_sufficient_when_min_equals_full_ridge(self, rng):
        X_train, Y_train, X_test, Y_test = self._split(rng, 30, 20, 8, 4)
        null = self._make_null(rng)
        # When Y_hat_min == Y_test, residuals are zero → recovery_r2 ≈ 0
        Y_resid_train = np.zeros_like(Y_train)
        result = run_sufficiency_test(
            X_train,
            Y_resid_train,
            X_test,
            Y_test,
            Y_hat_full=Y_test,
            Y_hat_min=Y_test,
            null_r2=null,
            regression_type="ridge",
        )
        assert result["sufficient"]

    def test_not_sufficient_when_large_residual_ridge(self, rng):
        n_dirs, n_targets = 6, 3
        X_train = rng.standard_normal((40, n_dirs)).astype(np.float32)
        X_test = rng.standard_normal((20, n_dirs)).astype(np.float32)
        # Y linearly recoverable from X
        Y_train = X_train[:, :n_targets]
        Y_test = X_test[:, :n_targets]
        Y_hat_min = np.zeros_like(Y_test)  # terrible minimal model
        Y_hat_full = Y_test  # perfect full model
        # Large residuals on train: the full-dim model can recover them
        Y_resid_train = Y_train  # Y_train - 0
        null = self._make_null(rng) - 0.5  # push null well below 0

        result = run_sufficiency_test(
            X_train,
            Y_resid_train,
            X_test,
            Y_test,
            Y_hat_full=Y_hat_full,
            Y_hat_min=Y_hat_min,
            null_r2=null,
            regression_type="ridge",
        )
        assert not result["sufficient"]

    def test_all_output_keys_present(self, rng):
        X_train, Y_train, X_test, Y_test = self._split(rng, 20, 10, 4, 2)
        null = self._make_null(rng)
        Y_resid_train = np.zeros_like(Y_train)
        result = run_sufficiency_test(
            X_train,
            Y_resid_train,
            X_test,
            Y_test,
            Y_hat_full=Y_test,
            Y_hat_min=Y_test,
            null_r2=null,
            regression_type="ols",
        )
        expected_keys = {
            "recovery_r2",
            "noise_floor",
            "sufficient",
            "p_value",
            "full_r2",
            "minimal_r2",
            "pred_r2",
            "full_corr",
            "minimal_corr",
            "recovery_corr",
            "full_corr_per_target",
            "minimal_corr_per_target",
            "recovery_corr_per_target",
        }
        assert expected_keys == set(result.keys())

    def test_p_value_in_range(self, rng):
        X_train, Y_train, X_test, Y_test = self._split(rng, 20, 10, 4, 2)
        null = self._make_null(rng)
        Y_resid_train = np.zeros_like(Y_train)
        result = run_sufficiency_test(
            X_train,
            Y_resid_train,
            X_test,
            Y_test,
            Y_hat_full=Y_test,
            Y_hat_min=Y_test,
            null_r2=null,
            regression_type="ols",
        )
        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# TestFindKneeLambda
# ---------------------------------------------------------------------------


class TestFindKneeLambda:
    def test_returns_value_in_lambda_grid(self):
        lam_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
        r2_vals = np.array([0.1, 0.4, 0.7, 0.75, 0.76, 0.77])
        knee = find_knee_lambda(lam_grid, r2_vals)
        assert knee in lam_grid

    def test_fallback_when_kneed_unavailable(self, monkeypatch):
        # Simulate kneed not installed
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "kneed":
                raise ImportError("kneed not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        lam_grid = [0.01, 0.1, 1.0, 10.0]
        r2_vals = np.array([0.1, 0.9, 0.5, 0.2])  # max at index 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            knee = find_knee_lambda(lam_grid, r2_vals)
        assert knee == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# TestFitDenseRRRPrecomputed
# ---------------------------------------------------------------------------


class TestFitDenseRRRPrecomputed:
    """Tests for fit_dense_rrr's precomputed_Y_hat_ols / precomputed_beta_full path."""

    @pytest.fixture
    def fitted_model_and_data(self):
        """Fit a small sklearn pipeline and return it along with train/test data."""
        from himalaya.backend import set_backend
        from himalaya.ridge import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        set_backend("numpy")

        rng = np.random.default_rng(0)
        n_train, n_dirs, n_targets = 60, 8, 4

        X_train = rng.standard_normal((n_train, n_dirs)).astype(np.float32)
        Y_train = rng.standard_normal((n_train, n_targets)).astype(np.float32)

        model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0]))
        model.fit(X_train, Y_train)

        coef = np.asarray(model[-1].coef_)
        if coef.shape[0] != n_dirs:
            coef = coef.T  # → (n_dirs, n_targets)
        beta_full = (coef / model[0].scale_[:, np.newaxis]).astype(np.float32)
        Y_hat_ols = np.asarray(model.predict(X_train)).astype(np.float32)

        return model, X_train, Y_train, beta_full, Y_hat_ols, n_dirs, n_targets

    def test_precomputed_matches_ridge_model_path(self, fitted_model_and_data):
        """Precomputed path should give identical A/B to the ridge_model path."""
        model, X_train, Y_train, beta_full, Y_hat_ols, n_dirs, n_targets = (
            fitted_model_and_data
        )
        selected_idx = np.arange(n_dirs)
        k = 2

        result_model = fit_dense_rrr(
            X_train,
            Y_train,
            k=k,
            selected_idx=selected_idx,
            ridge_model=model,
        )
        result_pre = fit_dense_rrr(
            X_train,
            Y_train,
            k=k,
            selected_idx=selected_idx,
            precomputed_Y_hat_ols=Y_hat_ols,
            precomputed_beta_full=beta_full,
        )

        np.testing.assert_allclose(
            result_model["A_dense"], result_pre["A_dense"], atol=1e-4
        )
        np.testing.assert_allclose(
            result_model["B_dense"], result_pre["B_dense"], atol=1e-4
        )

    def test_precomputed_with_selected_subset(self, fitted_model_and_data):
        """Precomputed path should work correctly when selected_idx is a subset."""
        model, X_train, Y_train, beta_full, Y_hat_ols, n_dirs, n_targets = (
            fitted_model_and_data
        )
        selected_idx = np.array([0, 2, 4])
        k = 2

        result = fit_dense_rrr(
            X_train,
            Y_train,
            k=k,
            selected_idx=selected_idx,
            precomputed_Y_hat_ols=Y_hat_ols,
            precomputed_beta_full=beta_full,
        )

        assert result["A_dense"].shape == (len(selected_idx), k)
        assert result["B_dense"].shape == (n_targets, k)

    def test_fallback_to_internal_solve_when_no_precomputed(self, small_data):
        """When neither ridge_model nor precomputed are given, internal solve is used."""
        X, Y = small_data
        selected_idx = np.arange(X.shape[1])
        k = 2

        result = fit_dense_rrr(X, Y, k=k, selected_idx=selected_idx)

        assert result["A_dense"].shape == (X.shape[1], k)
        assert result["B_dense"].shape == (Y.shape[1], k)
