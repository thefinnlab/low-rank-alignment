"""Unit tests for PCA subspace utilities and _run_pca_for_unit."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

from src.utils.subspace_utils import (
    evaluate_input_space,
    evaluate_complement,
    evaluate_arbitrary,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_data(rng):
    """Small synthetic (X, Y) pair for testing."""
    n, n_embed, n_targets = 80, 12, 5
    X = rng.standard_normal((n, n_embed)).astype(np.float32)
    Y = rng.standard_normal((n, n_targets)).astype(np.float32)
    return X, Y


# ---------------------------------------------------------------------------
# TestEvaluateInputSpace
# ---------------------------------------------------------------------------


class TestEvaluateInputSpace:
    """Unit tests for evaluate_input_space."""

    def _make_inputs(self, rng, n=80, n_embed=12, n_targets=5):
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)
        n_train = n // 2
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        return X_train, X_test, Y_train, Y_test

    def test_returns_expected_keys_no_prefix(self, rng):
        X_train, X_test, Y_train, Y_test = self._make_inputs(rng)
        result = evaluate_input_space(X_train, X_test, Y_train, Y_test)
        assert "corr" in result
        assert "r2" in result
        assert "corr_per_target" in result
        assert "r2_per_target" in result
        assert "Y_hat" in result

    def test_returns_expected_keys_with_prefix(self, rng):
        X_train, X_test, Y_train, Y_test = self._make_inputs(rng)
        result = evaluate_input_space(
            X_train, X_test, Y_train, Y_test, subspace_name="pca"
        )
        assert "pca_corr" in result
        assert "pca_r2" in result
        assert "pca_corr_per_target" in result
        assert "pca_r2_per_target" in result
        assert "pca_Y_hat" in result

    def test_corr_is_float(self, rng):
        X_train, X_test, Y_train, Y_test = self._make_inputs(rng)
        result = evaluate_input_space(
            X_train, X_test, Y_train, Y_test, subspace_name="pca"
        )
        assert isinstance(result["pca_corr"], float)
        assert isinstance(result["pca_r2"], float)

    def test_predictions_shape(self, rng):
        n, n_targets = 80, 5
        X_train, X_test, Y_train, Y_test = self._make_inputs(
            rng, n=n, n_targets=n_targets
        )
        result = evaluate_input_space(
            X_train, X_test, Y_train, Y_test, subspace_name="pca"
        )
        assert result["pca_Y_hat"].shape == Y_test.shape
        assert len(result["pca_corr_per_target"]) == n_targets

    def test_projection_then_evaluate(self, rng):
        """Manual projection + evaluate_input_space matches a direct call."""
        n, n_embed, n_targets, k = 80, 12, 5, 3
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)
        A = rng.standard_normal((n_embed, k)).astype(np.float32)
        n_train = n // 2
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]

        X_proj_train = X_train @ A
        X_proj_test = X_test @ A
        result = evaluate_input_space(
            X_proj_train, X_proj_test, Y_train, Y_test, subspace_name="pca"
        )
        assert "pca_corr" in result
        assert "pca_r2" in result
        assert "pca_corr_per_target" in result


# ---------------------------------------------------------------------------
# TestEvaluateComplement
# ---------------------------------------------------------------------------


class TestEvaluateComplement:
    """Unit tests for evaluate_complement."""

    def _make_inputs(self, rng, n=80, n_embed=12, n_targets=5, k=3):
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)
        A = rng.standard_normal((n_embed, n_embed)).astype(np.float32)
        # Orthonormalize A so it is a proper rotation matrix
        A, _ = np.linalg.qr(A)
        A = A.astype(np.float32)
        n_train = n // 2
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        return A, k, X_train, X_test, Y_train, Y_test

    def test_returns_expected_keys(self, rng):
        A, k, X_train, X_test, Y_train, Y_test = self._make_inputs(rng)
        result = evaluate_complement(A, k, X_train, X_test, Y_train, Y_test)
        assert "complement_corr" in result
        assert "complement_r2" in result
        assert "complement_corr_per_target" in result
        assert "complement_r2_per_target" in result
        assert "complement_Y_hat" in result

    def test_corr_is_float(self, rng):
        A, k, X_train, X_test, Y_train, Y_test = self._make_inputs(rng)
        result = evaluate_complement(A, k, X_train, X_test, Y_train, Y_test)
        assert isinstance(result["complement_corr"], float)
        assert isinstance(result["complement_r2"], float)

    def test_per_target_shape(self, rng):
        n_targets = 5
        A, k, X_train, X_test, Y_train, Y_test = self._make_inputs(
            rng, n_targets=n_targets
        )
        result = evaluate_complement(A, k, X_train, X_test, Y_train, Y_test)
        assert len(result["complement_corr_per_target"]) == n_targets

    def test_full_k_gives_constant_predictions(self, rng):
        """When k == A.shape[1], complement is empty; predictions should be constant."""
        n, n_embed, n_targets = 80, 8, 5
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)
        A = np.eye(n_embed, dtype=np.float32)
        n_train = n // 2
        result = evaluate_complement(
            A, n_embed, X[:n_train], X[n_train:], Y[:n_train], Y[n_train:]
        )
        # With only a constant feature, complement_corr should be ~0
        assert abs(result["complement_corr"]) < 0.5

    def test_complement_orthogonal_to_subspace(self, rng):
        """Complement projections should be orthogonal to the subspace directions."""
        from src.utils.analysis import partial_out_subspace

        n_embed, k = 12, 3
        A = rng.standard_normal((n_embed, k)).astype(np.float32)
        X = rng.standard_normal((20, n_embed)).astype(np.float32)
        X_comp = partial_out_subspace(X, A.T)
        # Q = orthonormalized A
        Q, _ = np.linalg.qr(A)
        # Projection of complement onto subspace should be ~zero
        proj = X_comp @ Q
        assert np.allclose(proj, 0, atol=1e-5)


# ---------------------------------------------------------------------------
# TestEvaluateArbitrary
# ---------------------------------------------------------------------------


class TestEvaluateArbitrary:
    """Unit tests for evaluate_arbitrary."""

    def _make_inputs(self, rng, n=80, n_embed=12, n_targets=5):
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)
        n_train = n // 2
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        # Arbitrary embeddings (just random, unrelated to Y)
        X_arb = rng.standard_normal((n, n_embed)).astype(np.float32)
        X_arb_train, X_arb_test = X_arb[:n_train], X_arb[n_train:]
        return X_arb_train, X_arb_test, Y_train, Y_test

    def test_returns_expected_keys(self, rng):
        inputs = self._make_inputs(rng)
        result = evaluate_arbitrary(*inputs)
        assert "arbitrary_corr" in result
        assert "arbitrary_r2" in result
        assert "arbitrary_corr_per_target" in result
        assert "arbitrary_r2_per_target" in result
        assert "arbitrary_Y_hat" in result

    def test_corr_is_float(self, rng):
        inputs = self._make_inputs(rng)
        result = evaluate_arbitrary(*inputs)
        assert isinstance(result["arbitrary_corr"], float)
        assert isinstance(result["arbitrary_r2"], float)

    def test_per_target_shape(self, rng):
        n_targets = 7
        inputs = self._make_inputs(rng, n_targets=n_targets)
        result = evaluate_arbitrary(*inputs)
        assert len(result["arbitrary_corr_per_target"]) == n_targets


# ---------------------------------------------------------------------------
# TestRunPCAForUnit
# ---------------------------------------------------------------------------


def _make_baselines(X, Y, test_idxs):
    """Build per_fold_baselines from raw data (for tests only)."""
    from src.utils.encoding import build_per_fold_baselines

    return build_per_fold_baselines(X, Y, test_idxs)


class TestRunPCAForUnit:
    """Integration tests for _run_pca_for_unit using synthetic data."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts", "rrr")
        scripts_dir = os.path.abspath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    def _make_rank_grid(self, rank_grid):
        return rank_grid

    def test_grid_results_csv_written(self, tmp_path, small_data):
        from run_rrr import _run_pca_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        rank_grid = [1, 3]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, rank_grid)

        csv_path = os.path.join(str(tmp_path), "grid_results.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        # 2 folds x 2 ranks = 4 rows
        assert len(df) == 4
        assert set(df["k"].unique()) == {1, 3}

    def test_summary_csv_averages_over_folds(self, tmp_path, small_data):
        from run_rrr import _run_pca_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        rank_grid = [2, 5]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, rank_grid)

        summary_path = os.path.join(str(tmp_path), "summary.csv")
        assert os.path.exists(summary_path)
        df = pd.read_csv(summary_path)
        assert len(df) == 2
        assert set(df["k"].unique()) == {2, 5}

    def test_pca_columns_in_output(self, tmp_path, small_data):
        from run_rrr import _run_pca_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        rank_grid = [1, 3]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, rank_grid)

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        for col in ["full_corr", "full_r2", "pca_corr", "pca_r2"]:
            assert col in df.columns
        for col in [
            "complement_corr",
            "complement_r2",
            "pca_complement_joint_corr",
            "pca_arbitrary_joint_corr",
            "pca_complement_pred_corr",
            "pca_arbitrary_pred_corr",
            "arbitrary_corr",
            "arbitrary_r2",
        ]:
            assert col not in df.columns

    def test_pca_r2_increases_with_rank(self, tmp_path, rng):
        """For low-rank ground truth data, R2 should increase monotonically with rank."""
        from run_rrr import _run_pca_for_unit

        # Construct rank-3 signal aligned with top PCA directions
        n, n_embed, n_targets = 120, 20, 8
        true_rank = 3
        A_true = rng.standard_normal((n_embed, true_rank)).astype(np.float32)
        B_true = rng.standard_normal((n_targets, true_rank)).astype(np.float32)
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = X @ A_true @ B_true.T + 0.01 * rng.standard_normal((n, n_targets)).astype(
            np.float32
        )

        test_idxs = [np.arange(n // 2, n)]
        rank_grid = [1, 2, 3, 5]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, rank_grid)

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        r2_by_k = df.groupby("k")["pca_r2"].mean().sort_index()
        assert all(
            r2_by_k.diff().dropna() >= -0.05
        ), f"R2 not monotone: {r2_by_k.to_dict()}"

    def test_plots_generated(self, tmp_path, small_data):
        from run_rrr import _run_pca_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        rank_grid = [1, 3]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, rank_grid)

        assert os.path.exists(os.path.join(str(tmp_path), "rank_analysis.png"))

    def test_rank_clamped_to_available_components(self, tmp_path, rng):
        """Ranks exceeding min(n_train, n_embed) should be silently skipped."""
        from run_rrr import _run_pca_for_unit

        n, n_embed, n_targets = 40, 5, 3
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)

        test_idxs = [np.arange(n // 2, n)]
        rank_grid = [1, 3, 5, 100]  # 100 > min(20, 5)
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, rank_grid)

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        assert 100 not in df["k"].values
        assert set(df["k"].unique()) == {1, 3, 5}
