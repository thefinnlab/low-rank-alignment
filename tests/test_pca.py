"""Unit tests for PCA subspace utilities and _run_pca_for_unit."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

from src.utils.subspace_utils import (
    evaluate_subspace_ols,
    evaluate_complement,
    evaluate_arbitrary,
    build_nr_metrics_dict,
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
# TestEvaluateSubspaceOLS
# ---------------------------------------------------------------------------


class TestEvaluateSubspaceOLS:
    """Unit tests for the shared subspace OLS evaluation utility."""

    def _make_inputs(self, rng, n=80, n_embed=12, n_targets=5, k=3):
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)
        # Random projection matrix
        A = rng.standard_normal((n_embed, k)).astype(np.float32)
        n_train = n // 2
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        Y_hat_full = rng.standard_normal((n - n_train, n_targets)).astype(np.float32)
        return A, X_train, X_test, Y_train, Y_test, Y_hat_full

    def test_returns_expected_keys(self, rng):
        A, X_train, X_test, Y_train, Y_test, Y_hat_full = self._make_inputs(rng)
        result = evaluate_subspace_ols(A, X_train, X_test, Y_train, Y_test, Y_hat_full)
        assert "suf" in result
        assert "cca_sim" in result
        assert "proc_sim" in result
        assert "Y_hat" in result
        assert "Y_hat_train" in result
        # OLS should NOT have best_alphas
        assert "best_alphas" not in result

    def test_predictions_shape(self, rng):
        n, n_embed, n_targets, k = 80, 12, 5, 3
        A, X_train, X_test, Y_train, Y_test, Y_hat_full = self._make_inputs(
            rng, n=n, n_embed=n_embed, n_targets=n_targets, k=k
        )
        result = evaluate_subspace_ols(A, X_train, X_test, Y_train, Y_test, Y_hat_full)
        assert result["Y_hat"].shape == Y_test.shape
        assert result["Y_hat_train"].shape == Y_train.shape

    def test_cca_sim_none_by_default(self, rng):
        A, X_train, X_test, Y_train, Y_test, Y_hat_full = self._make_inputs(rng)
        result = evaluate_subspace_ols(A, X_train, X_test, Y_train, Y_test, Y_hat_full)
        assert result["cca_sim"] is None
        assert result["proc_sim"] is None

    def test_suf_has_expected_keys(self, rng):
        A, X_train, X_test, Y_train, Y_test, Y_hat_full = self._make_inputs(rng)
        result = evaluate_subspace_ols(A, X_train, X_test, Y_train, Y_test, Y_hat_full)
        suf = result["suf"]
        assert "minimal_r2" in suf
        assert "recovery_r2" in suf
        assert "full_corr" in suf
        assert "minimal_corr" in suf


# ---------------------------------------------------------------------------
# TestEvaluateComplement
# ---------------------------------------------------------------------------


class TestEvaluateComplement:
    """Unit tests for evaluate_complement."""

    def _make_inputs(self, rng, n=80, n_embed=12, n_targets=5, k=3):
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)
        A = rng.standard_normal((n_embed, k)).astype(np.float32)
        n_train = n // 2
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        Y_hat_train = rng.standard_normal((n_train, n_targets)).astype(np.float32)
        Y_hat_test = rng.standard_normal((n - n_train, n_targets)).astype(np.float32)
        return A, k, X_train, X_test, Y_train, Y_test, Y_hat_train, Y_hat_test

    def test_returns_expected_keys(self, rng):
        A, k, X_train, X_test, Y_train, Y_test, Y_hat_train, Y_hat_test = (
            self._make_inputs(rng)
        )
        result = evaluate_complement(
            A, k, X_train, X_test, Y_train, Y_test, Y_hat_train, Y_hat_test
        )
        assert "complement_corr" in result
        assert "complement_recovery_corr" in result
        assert "complement_corr_per_target" in result
        assert "complement_recovery_corr_per_target" in result

    def test_corr_is_float(self, rng):
        A, k, X_train, X_test, Y_train, Y_test, Y_hat_train, Y_hat_test = (
            self._make_inputs(rng)
        )
        result = evaluate_complement(
            A, k, X_train, X_test, Y_train, Y_test, Y_hat_train, Y_hat_test
        )
        assert isinstance(result["complement_corr"], float)
        assert isinstance(result["complement_recovery_corr"], float)

    def test_per_target_shape(self, rng):
        n_targets = 5
        A, k, X_train, X_test, Y_train, Y_test, Y_hat_train, Y_hat_test = (
            self._make_inputs(rng, n_targets=n_targets)
        )
        result = evaluate_complement(
            A, k, X_train, X_test, Y_train, Y_test, Y_hat_train, Y_hat_test
        )
        assert len(result["complement_corr_per_target"]) == n_targets

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
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)
        n_train = n // 2
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        Y_hat_train = rng.standard_normal((n_train, n_targets)).astype(np.float32)
        Y_hat_test = rng.standard_normal((n - n_train, n_targets)).astype(np.float32)
        # Arbitrary embeddings (just random)
        X_arb = rng.standard_normal((n, n_embed)).astype(np.float32)
        X_arb_train, X_arb_test = X_arb[:n_train], X_arb[n_train:]
        return X_arb_train, X_arb_test, Y_train, Y_test, Y_hat_train, Y_hat_test

    def test_returns_expected_keys(self, rng):
        inputs = self._make_inputs(rng)
        result = evaluate_arbitrary(*inputs)
        assert "arbitrary_corr" in result
        assert "arbitrary_recovery_corr" in result
        assert "arbitrary_corr_per_target" in result
        assert "arbitrary_recovery_corr_per_target" in result

    def test_corr_is_float(self, rng):
        inputs = self._make_inputs(rng)
        result = evaluate_arbitrary(*inputs)
        assert isinstance(result["arbitrary_corr"], float)
        assert isinstance(result["arbitrary_recovery_corr"], float)


# ---------------------------------------------------------------------------
# TestBuildNrMetricsDict
# ---------------------------------------------------------------------------


class TestBuildNrMetricsDict:
    """Unit tests for build_nr_metrics_dict."""

    def test_has_nr_prefix_keys(self, rng):
        nr_result = {
            "suf": {
                "minimal_r2": 0.5,
                "minimal_corr": 0.3,
                "recovery_r2": 0.1,
                "recovery_corr": 0.05,
                "pred_r2": 0.8,
                "minimal_corr_per_target": np.array([0.3, 0.4]),
                "recovery_corr_per_target": np.array([0.05, 0.06]),
            },
            "cca_sim": None,
            "proc_sim": None,
        }
        out = build_nr_metrics_dict(nr_result)
        assert "nr_r2" in out
        assert "nr_corr" in out
        assert "nr_recovery_r2" in out
        assert "nr_recovery_corr" in out
        assert "nr_pred_r2" in out

    def test_complement_and_arbitrary_included(self, rng):
        nr_result = {
            "suf": {
                "minimal_r2": 0.5,
                "minimal_corr": 0.3,
                "recovery_r2": 0.1,
                "recovery_corr": 0.05,
                "pred_r2": 0.8,
                "minimal_corr_per_target": np.array([0.3, 0.4]),
                "recovery_corr_per_target": np.array([0.05, 0.06]),
            },
            "cca_sim": None,
            "proc_sim": None,
        }
        complement_result = {
            "complement_corr": 0.2,
            "complement_recovery_corr": 0.1,
            "complement_corr_per_target": np.array([0.2, 0.3]),
            "complement_recovery_corr_per_target": np.array([0.1, 0.15]),
        }
        arbitrary_result = {
            "arbitrary_corr": 0.05,
            "arbitrary_recovery_corr": 0.02,
            "arbitrary_corr_per_target": np.array([0.05, 0.06]),
            "arbitrary_recovery_corr_per_target": np.array([0.02, 0.03]),
        }
        out = build_nr_metrics_dict(
            nr_result,
            complement_result=complement_result,
            arbitrary_result=arbitrary_result,
        )
        assert out["nr_complement_corr"] == 0.2
        assert out["nr_complement_recovery_corr"] == 0.1
        assert out["nr_arbitrary_corr"] == 0.05
        assert out["nr_arbitrary_recovery_corr"] == 0.02


# ---------------------------------------------------------------------------
# TestRunPCAForUnit
# ---------------------------------------------------------------------------


def _make_baselines(X, Y, test_idxs):
    """Build per_fold_baselines from raw data using a fresh ridge model (for tests only)."""
    from sklearn.preprocessing import StandardScaler
    from src.utils.encoding import get_ridge_model_constructor

    n_words, n_embed = X.shape
    baselines = []
    for test_idx in test_idxs:
        train_mask = np.ones(n_words, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        scaler = StandardScaler()
        Y_train = scaler.fit_transform(Y[train_idx])
        Y_test = scaler.transform(Y[test_idx])
        model = get_ridge_model_constructor()(0)
        model.fit(X[train_idx], Y_train)
        coef = np.asarray(model[-1].coef_)
        if coef.shape[0] != n_embed:
            coef = coef.T
        beta_full_raw = (coef / model[0].scale_[:, np.newaxis]).astype(np.float32)
        baselines.append(
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "Y_train": Y_train,
                "Y_test": Y_test,
                "Y_hat_full_train": np.asarray(model.predict(X[train_idx])),
                "Y_hat_full_test": np.asarray(model.predict(X[test_idx])),
                "beta_full_raw": beta_full_raw,
            }
        )
    return baselines


class TestRunPCAForUnit:
    """Integration tests for _run_pca_for_unit using synthetic data."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts", "rrr")
        scripts_dir = os.path.abspath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    def _make_config(self, rank_grid):
        from src.core.config import PCAConfig

        config = PCAConfig()
        config.rank_grid = rank_grid
        return config

    def test_grid_results_csv_written(self, tmp_path, small_data):
        from run_pca import _run_pca_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1, 3])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        csv_path = os.path.join(str(tmp_path), "grid_results.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        # 2 folds x 2 ranks = 4 rows
        assert len(df) == 4
        assert set(df["k"].unique()) == {1, 3}

    def test_summary_csv_averages_over_folds(self, tmp_path, small_data):
        from run_pca import _run_pca_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[2, 5])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        summary_path = os.path.join(str(tmp_path), "summary.csv")
        assert os.path.exists(summary_path)
        df = pd.read_csv(summary_path)
        assert len(df) == 2
        assert set(df["k"].unique()) == {2, 5}

    def test_complement_columns_in_output(self, tmp_path, small_data):
        from run_pca import _run_pca_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1, 3])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        assert "complement_corr" in df.columns
        assert "complement_recovery_corr" in df.columns

    def test_minimal_r2_increases_with_rank(self, tmp_path, rng):
        """For low-rank ground truth data, R2 should increase monotonically with rank."""
        from run_pca import _run_pca_for_unit

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
        config = self._make_config(rank_grid=[1, 2, 3, 5])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        r2_by_k = df.groupby("k")["minimal_r2"].mean().sort_index()
        assert all(
            r2_by_k.diff().dropna() >= -0.05
        ), f"R2 not monotone: {r2_by_k.to_dict()}"

    def test_plots_generated(self, tmp_path, small_data):
        from run_pca import _run_pca_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1, 3])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        assert os.path.exists(os.path.join(str(tmp_path), "rank_analysis.png"))
        # rank_analysis_rrr_raw.png is only generated when RRR overlay exists
        assert not os.path.exists(
            os.path.join(str(tmp_path), "rank_analysis_rrr_raw.png")
        )

    def test_rank_clamped_to_available_components(self, tmp_path, rng):
        """Ranks exceeding min(n_train, n_embed) should be silently skipped."""
        from run_pca import _run_pca_for_unit

        n, n_embed, n_targets = 40, 5, 3
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n, n_targets)).astype(np.float32)

        test_idxs = [np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1, 3, 5, 100])  # 100 > min(20, 5)
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_pca_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        assert 100 not in df["k"].values
        assert set(df["k"].unique()) == {1, 3, 5}
