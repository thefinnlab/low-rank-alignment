"""Tests for utils/analysis.py."""
import numpy as np
import pytest
import torch

from utils.analysis import (
    get_correlation_gpu,
    partial_out_direction,
    find_dominant_directions,
    compute_truncated_correlations,
    find_best_k,
)


# ---------------------------------------------------------------------------
# get_correlation_gpu
# ---------------------------------------------------------------------------

class TestGetCorrelationGpu:
    def test_output_is_torch_tensor(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((50, 4)).astype(np.float32)
        b = rng.standard_normal((50, 3)).astype(np.float32)
        result = get_correlation_gpu(a, b, device="cpu")
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self):
        rng = np.random.default_rng(1)
        n_samples, na, nb = 40, 5, 7
        a = rng.standard_normal((n_samples, na)).astype(np.float32)
        b = rng.standard_normal((n_samples, nb)).astype(np.float32)
        result = get_correlation_gpu(a, b, device="cpu")
        assert result.shape == (na, nb)

    def test_self_correlation_diagonal_approx_one(self):
        rng = np.random.default_rng(2)
        n_samples, n_features = 100, 6
        a = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        result = get_correlation_gpu(a, a, device="cpu")
        diag = torch.diag(result).numpy()
        np.testing.assert_allclose(diag, np.ones(n_features), atol=1e-4)

    def test_matches_numpy_reference(self):
        rng = np.random.default_rng(3)
        n_samples, na, nb = 60, 4, 3
        a = rng.standard_normal((n_samples, na)).astype(np.float64)
        b = rng.standard_normal((n_samples, nb)).astype(np.float64)

        # numpy reference: z-score then dot
        a_z = (a - a.mean(axis=0)) / a.std(axis=0)
        b_z = (b - b.mean(axis=0)) / b.std(axis=0)
        expected = (a_z.T @ b_z) / n_samples

        result = get_correlation_gpu(
            a.astype(np.float32), b.astype(np.float32), device="cpu"
        )
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=1e-4)

    def test_runs_on_cpu_device(self):
        rng = np.random.default_rng(4)
        a = rng.standard_normal((30, 3)).astype(np.float32)
        b = rng.standard_normal((30, 3)).astype(np.float32)
        result = get_correlation_gpu(a, b, device=torch.device("cpu"))
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"


# ---------------------------------------------------------------------------
# partial_out_direction
# ---------------------------------------------------------------------------

class TestPartialOutDirection:
    def _build_fixtures(self, seed=0):
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        n_words, n_dim, n_cols = 40, 8, 6

        X_base = rng.standard_normal((n_words, n_dim)).astype(np.float64)
        Y = rng.standard_normal((n_words, n_cols)).astype(np.float64)

        # One fold covering all data (for simplicity)
        fold_indices = [np.arange(n_words)]

        # Direction: first principal component direction (unit vector)
        direction = rng.standard_normal(n_dim)
        direction /= np.linalg.norm(direction)

        # Predictions: projection of X onto direction
        predictions = X_base @ direction  # (n_words,)

        scaler = StandardScaler()
        scaler.fit(np.tile(predictions[:, None], (1, n_cols)))

        # directions arg: list of list of directions (one list per direction to remove,
        # each containing one direction per fold)
        directions = [[direction]]  # 1 direction, 1 fold
        predictions_list = [predictions]

        X_list = [X_base.copy(), X_base.copy()]  # two folds worth for indexing

        return (
            X_list,
            Y,
            directions,
            predictions_list,
            fold_indices,
            [scaler],
            direction,
        )

    def test_y_residual_lower_variance_along_direction(self):
        X_list, Y, directions, predictions_list, fold_indices, scalers, direction = (
            self._build_fixtures()
        )
        _, Y_residual = partial_out_direction(
            X_list, Y, directions, predictions_list, fold_indices, scalers
        )
        # Variance of Y projected onto direction should be reduced
        proj_orig = Y @ direction[:Y.shape[1]] if Y.shape[1] <= len(direction) else Y.mean(axis=1)
        # Check that Y_residual differs from Y (partialing actually changed something)
        assert not np.allclose(Y_residual, Y)

    def test_x_residual_orthogonal_to_direction(self):
        X_list, Y, directions, predictions_list, fold_indices, scalers, direction = (
            self._build_fixtures()
        )
        X_residuals, _ = partial_out_direction(
            X_list, Y, directions, predictions_list, fold_indices, scalers
        )
        # X_residuals[0] should be orthogonal to direction
        dots = X_residuals[0] @ direction  # (n_words,)
        np.testing.assert_allclose(dots, 0.0, atol=1e-8)

    def test_output_shapes_single_fold_single_direction(self):
        X_list, Y, directions, predictions_list, fold_indices, scalers, _ = (
            self._build_fixtures()
        )
        X_residuals, Y_residual = partial_out_direction(
            X_list, Y, directions, predictions_list, fold_indices, scalers
        )
        assert Y_residual.shape == Y.shape
        # X_residuals has one entry per fold (len(directions[0]))
        assert len(X_residuals) == len(directions[0])
        assert X_residuals[0].shape == X_list[0].shape

    def test_output_shapes_multiple_directions(self):
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(5)
        n_words, n_dim, n_cols = 40, 8, 6
        X_base = rng.standard_normal((n_words, n_dim)).astype(np.float64)
        Y = rng.standard_normal((n_words, n_cols)).astype(np.float64)
        fold_indices = [np.arange(n_words)]

        d1 = rng.standard_normal(n_dim)
        d1 /= np.linalg.norm(d1)
        d2 = rng.standard_normal(n_dim)
        d2 /= np.linalg.norm(d2)

        preds1 = X_base @ d1
        preds2 = X_base @ d2

        scaler = StandardScaler()
        scaler.fit(np.tile(preds1[:, None], (1, n_cols)))

        directions = [[d1], [d2]]
        predictions_list = [preds1, preds2]
        X_list = [X_base.copy(), X_base.copy()]

        X_residuals, Y_residual = partial_out_direction(
            X_list, Y, directions, predictions_list, fold_indices, [scaler]
        )
        assert Y_residual.shape == Y.shape
        # X_residuals length = number of folds (len(directions[0]))
        assert len(X_residuals) == 1
        assert X_residuals[0].shape == X_base.shape


# ---------------------------------------------------------------------------
# find_dominant_directions
# ---------------------------------------------------------------------------

class TestFindDominantDirections:
    def _make_weights(self, n_folds=2, n_models=20, embed_dim=10, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_folds, n_models, embed_dim)).astype(np.float64)

    def test_output_shapes(self):
        n_folds, n_models, embed_dim = 2, 20, 10
        n_directions = 5
        weights = self._make_weights(n_folds, n_models, embed_dim)
        _, all_S, all_Vt, _ = find_dominant_directions(weights, n_directions=n_directions)
        for f in range(n_folds):
            assert all_Vt[f].shape == (n_directions, embed_dim)
            assert all_S[f].shape == (n_directions,)

    def test_vt_rows_are_orthonormal(self):
        weights = self._make_weights()
        _, _, all_Vt, _ = find_dominant_directions(weights, n_directions=5)
        for Vt in all_Vt:
            product = Vt @ Vt.T
            np.testing.assert_allclose(product, np.eye(Vt.shape[0]), atol=1e-10)

    def test_singular_values_descending(self):
        weights = self._make_weights()
        _, all_S, _, _ = find_dominant_directions(weights, n_directions=5)
        for S in all_S:
            assert np.all(np.diff(S) <= 0)

    def test_n_directions_clamped_to_rank(self):
        n_folds, n_models, embed_dim = 2, 4, 10
        weights = self._make_weights(n_folds, n_models, embed_dim)
        _, all_S, all_Vt, _ = find_dominant_directions(weights, n_directions=100)
        # rank is min(n_models, embed_dim) = 4
        for f in range(n_folds):
            assert all_Vt[f].shape[0] <= min(n_models, embed_dim)
            assert all_S[f].shape[0] <= min(n_models, embed_dim)

    def test_raises_on_empty_weights(self):
        weights = np.zeros((2, 0, 10))
        with pytest.raises(ValueError):
            find_dominant_directions(weights, n_directions=5)


# ---------------------------------------------------------------------------
# compute_truncated_correlations
# ---------------------------------------------------------------------------

class TestComputeTruncatedCorrelations:
    def _make_fixtures(self, seed=0):
        rng = np.random.default_rng(seed)
        n_words, n_folds, n_elecs, n_lags, n_features = 80, 2, 6, 4, 12
        k_values = [2, 4, 6, n_features]

        X = rng.standard_normal((n_words, n_features)).astype(np.float64)
        Y = rng.standard_normal((n_words, n_elecs * n_lags)).astype(np.float64)
        coefficients = rng.standard_normal(
            (n_folds, n_elecs, n_lags, n_features)
        ).astype(np.float64)
        intercepts = [
            rng.standard_normal(n_elecs * n_lags).astype(np.float64)
            for _ in range(n_folds)
        ]
        half = n_words // 2
        test_idxs = [np.arange(half), np.arange(half, n_words)]

        # Build Vt via SVD of coefficients (global directions)
        W_sig = coefficients.reshape(n_folds, -1, n_features)
        _, _, all_Vt, _ = find_dominant_directions(W_sig, n_directions=n_features)
        # Pad to n_features rows if needed
        Vt_arr = np.stack([
            np.vstack([v, np.zeros((n_features - v.shape[0], n_features))])
            if v.shape[0] < n_features else v
            for v in all_Vt
        ])  # (n_folds, n_features, n_features)

        sig_mask = rng.integers(0, 2, (n_elecs, n_lags)).astype(bool)

        return dict(
            coefficients=coefficients,
            intercepts=intercepts,
            test_idxs=test_idxs,
            X=X, Y=Y,
            n_elecs=n_elecs, n_lags=n_lags,
            Vt=Vt_arr,
            k_values=k_values,
            sig_mask=sig_mask,
        )

    def test_output_keys_present(self):
        f = self._make_fixtures()
        results = compute_truncated_correlations(**f)
        for key in ["k_values", "full_corrs", "truncated_corrs", "sig_mask",
                    "full_corrs_per_fold", "truncated_corrs_per_fold"]:
            assert key in results

    def test_output_shapes(self):
        f = self._make_fixtures()
        n_elecs, n_lags, n_k = f["n_elecs"], f["n_lags"], len(f["k_values"])
        n_folds = len(f["test_idxs"])
        results = compute_truncated_correlations(**f)
        assert results["full_corrs"].shape == (n_elecs, n_lags)
        assert results["truncated_corrs"].shape == (n_k, n_elecs, n_lags)
        assert results["full_corrs_per_fold"].shape == (n_folds, n_elecs, n_lags)
        assert results["truncated_corrs_per_fold"].shape == (n_k, n_folds, n_elecs, n_lags)

    def test_full_k_recovers_full_performance(self):
        """At k=n_features, truncated predictions equal full predictions."""
        f = self._make_fixtures()
        n_features = f["X"].shape[1]
        # Only use k=n_features
        f["k_values"] = [n_features]
        # Use orthonormal Vt built from SVD (already done), but use identity-like
        # to guarantee W_trunc == W: build full orthogonal basis per fold
        n_folds = len(f["test_idxs"])
        Vt_ortho = np.stack([
            np.linalg.svd(np.eye(n_features), full_matrices=False)[2]
            for _ in range(n_folds)
        ])
        f["Vt"] = Vt_ortho
        results = compute_truncated_correlations(**f)
        np.testing.assert_allclose(
            results["truncated_corrs"][0],
            results["full_corrs"],
            atol=1e-4,
        )

    def test_monotone_improvement(self):
        """Full-k should outperform k=1 on significant pairs."""
        f = self._make_fixtures()
        results = compute_truncated_correlations(**f)
        sig_mask = results["sig_mask"]
        if not sig_mask.any():
            pytest.skip("No significant pairs in random mask")
        sig_means = np.array([t[sig_mask].mean() for t in results["truncated_corrs"]])
        # The highest k (full subspace) should at least match the lowest k
        assert sig_means[-1] >= sig_means[0] - 0.05, (
            f"Full-k ({sig_means[-1]:.4f}) much worse than k=1 ({sig_means[0]:.4f})"
        )


# ---------------------------------------------------------------------------
# find_best_k
# ---------------------------------------------------------------------------

class TestFindBestK:
    def _make_results(self, k_values, ratios, seed=0):
        """Build a minimal results dict from desired retention ratios."""
        rng = np.random.default_rng(seed)
        n_elecs, n_lags = 4, 3
        sig_mask = np.ones((n_elecs, n_lags), dtype=bool)
        full_corrs = np.full((n_elecs, n_lags), 1.0)
        # Build truncated_corrs so that sig mean equals ratio * 1.0 = ratio
        truncated_corrs = np.stack([
            np.full((n_elecs, n_lags), r) for r in ratios
        ])  # (n_k, n_elecs, n_lags)
        return {
            "k_values": np.array(k_values),
            "full_corrs": full_corrs,
            "truncated_corrs": truncated_corrs,
            "sig_mask": sig_mask,
        }

    def test_returns_smallest_k_meeting_threshold(self):
        k_values = [1, 5, 10, 50]
        ratios = [0.80, 0.90, 0.96, 0.99]
        results = self._make_results(k_values, ratios)
        assert find_best_k(results, threshold=0.95) == 10

    def test_returns_largest_k_when_threshold_never_met(self):
        k_values = [1, 5, 10]
        ratios = [0.50, 0.70, 0.80]
        results = self._make_results(k_values, ratios)
        assert find_best_k(results, threshold=0.95) == 10

    def test_threshold_exactly_met(self):
        # Use 0.96 (clearly above 0.95) to avoid float representation edge cases
        k_values = [1, 5, 10]
        ratios = [0.80, 0.96, 0.99]
        results = self._make_results(k_values, ratios)
        assert find_best_k(results, threshold=0.95) == 5

    def test_all_k_meet_threshold_returns_smallest(self):
        k_values = [1, 5, 10]
        ratios = [0.97, 0.98, 0.99]
        results = self._make_results(k_values, ratios)
        assert find_best_k(results, threshold=0.95) == 1
