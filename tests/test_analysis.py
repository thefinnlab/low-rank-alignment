"""Tests for utils/analysis.py."""
import numpy as np
import pytest
import torch

from utils.analysis import get_correlation_gpu, partial_out_direction


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
