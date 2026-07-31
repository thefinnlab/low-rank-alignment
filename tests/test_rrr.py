"""Unit tests for RRR utilities: save_rrr_result/load_rrr_result and _run_rrr_for_unit."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

from src.utils.srrr import save_rrr_factors, save_rrr_result, load_rrr_result
from scripts.rrr.run_rrr import _load_cross_dataset_A_by_rank

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
# TestSaveLoadRRRResult
# ---------------------------------------------------------------------------


class TestSaveLoadRRRResult:
    def test_roundtrip(self, tmp_path, rng):
        max_k, k, fold = 5, 3, 0
        A_full = rng.standard_normal((10, max_k)).astype(np.float32)
        B_full = rng.standard_normal((5, max_k)).astype(np.float32)
        metrics = {
            "full_r2": 0.5,
            "nr_r2": 0.4,
            "nr_corr": 0.02,
            "full_corr": 0.03,
            "fold": fold,
            "k": k,
        }

        save_rrr_factors(str(tmp_path), fold=fold, A=A_full, B=B_full)
        save_rrr_result(str(tmp_path), k=k, fold=fold, metrics=metrics)
        result = load_rrr_result(str(tmp_path), k=k, fold=fold)

        np.testing.assert_array_almost_equal(result["A"], A_full[:, :k])
        np.testing.assert_array_almost_equal(result["B"], B_full[:, :k])
        assert abs(float(result["metrics"]["full_r2"]) - 0.5) < 1e-5
        assert abs(float(result["metrics"]["nr_r2"]) - 0.4) < 1e-5

    def test_directory_created(self, tmp_path):
        A = np.zeros((4, 5), dtype=np.float32)
        B = np.zeros((3, 5), dtype=np.float32)
        save_rrr_factors(str(tmp_path), fold=1, A=A, B=B)
        save_rrr_result(str(tmp_path), k=5, fold=1, metrics={"full_r2": 0.1})
        factors_dir = os.path.join(str(tmp_path), "fold_1")
        metrics_dir = os.path.join(str(tmp_path), "k5", "fold_1")
        assert os.path.isdir(factors_dir)
        assert os.path.exists(os.path.join(factors_dir, "A.npy"))
        assert os.path.exists(os.path.join(factors_dir, "B.npy"))
        assert os.path.isdir(metrics_dir)
        assert os.path.exists(os.path.join(metrics_dir, "metrics.npz"))


class TestCrossDatasetRankBasisLoading:
    def test_loads_single_saved_basis_per_rank(self, tmp_path, rng):
        expected = rng.standard_normal((12, 4)).astype(np.float32)
        rank_dir = tmp_path / "rrr" / "k4"
        rank_dir.mkdir(parents=True)
        np.save(rank_dir / "A.npy", expected)

        loaded = _load_cross_dataset_A_by_rank(
            str(tmp_path), "rrr", [4], source_format="rank_bases"
        )

        np.testing.assert_array_equal(loaded[4], expected)

    def test_missing_rank_basis_reports_expected_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"rrr/k4/A.npy"):
            _load_cross_dataset_A_by_rank(
                str(tmp_path), "rrr", [4], source_format="rank_bases"
            )

    def test_multiple_folds_independent(self, tmp_path, rng):
        max_k, k = 4, 2
        for fold in range(3):
            A = rng.standard_normal((6, max_k)).astype(np.float32)
            B = rng.standard_normal((4, max_k)).astype(np.float32)
            save_rrr_factors(str(tmp_path), fold=fold, A=A, B=B)
            save_rrr_result(
                str(tmp_path), k=k, fold=fold, metrics={"full_r2": float(fold)}
            )

        for fold in range(3):
            result = load_rrr_result(str(tmp_path), k=k, fold=fold)
            assert abs(float(result["metrics"]["full_r2"]) - fold) < 1e-5

    def test_truncation_across_k_values(self, tmp_path, rng):
        """Loading different k values from the same full-rank factors truncates correctly."""
        max_k, fold = 5, 0
        A_full = rng.standard_normal((10, max_k)).astype(np.float32)
        B_full = rng.standard_normal((8, max_k)).astype(np.float32)
        save_rrr_factors(str(tmp_path), fold=fold, A=A_full, B=B_full)

        for k in [1, 3, 5]:
            save_rrr_result(str(tmp_path), k=k, fold=fold, metrics={"k": k})
            result = load_rrr_result(str(tmp_path), k=k, fold=fold)
            assert result["A"].shape == (10, k)
            assert result["B"].shape == (8, k)
            np.testing.assert_array_almost_equal(result["A"], A_full[:, :k])
            np.testing.assert_array_almost_equal(result["B"], B_full[:, :k])


# ---------------------------------------------------------------------------
# TestRunRRRForUnit
# ---------------------------------------------------------------------------


def _make_baselines(X, Y, test_idxs):
    """Build per_fold_baselines from raw data using a fresh ridge model."""
    from src.utils.encoding import build_per_fold_baselines

    return build_per_fold_baselines(X, Y, test_idxs)


class TestRunRRRForUnit:
    """Integration tests for _run_rrr_for_unit using synthetic data."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts", "rrr")
        scripts_dir = os.path.abspath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    def _make_config(self, rank_grid, refresh=False):
        from src.core.config import RRRConfig

        config = RRRConfig()
        config.rank_grid = rank_grid
        config.refresh = refresh
        config.save_b_matrices = True
        return config

    def test_rank_specific_a_is_orthonormalized_before_ridge(
        self, tmp_path, small_data, monkeypatch
    ):
        import run_rrr

        X, Y = small_data
        A_full = np.array(
            [[2.0, 1.0], [0.0, 3.0]] + [[0.0, 0.0]] * (X.shape[1] - 2),
            dtype=np.float32,
        )
        captured = []

        def capture_evaluation(X_train, X_test, *args, **kwargs):
            captured.append((X_train.copy(), X_test.copy()))
            n_targets = args[1].shape[1]
            return {
                "nr_corr": 0.0,
                "nr_r2": 0.0,
                "nr_corr_per_target": np.zeros(n_targets),
                "nr_r2_per_target": np.zeros(n_targets),
                "nr_best_alphas": np.ones(n_targets),
            }

        monkeypatch.setattr(run_rrr, "evaluate_input_space", capture_evaluation)
        monkeypatch.setattr(
            run_rrr, "resolve_ridge_alphas", lambda *a, **k: np.array([1.0])
        )
        monkeypatch.setattr(run_rrr, "save_alpha_diagnostics", lambda *a, **k: None)

        train_idx = np.arange(0, X.shape[0] // 2)
        test_idx = np.arange(X.shape[0] // 2, X.shape[0])
        run_rrr._run_common_subspace_fold_k(
            A_full,
            2,
            X[train_idx],
            X[test_idx],
            Y[train_idx],
            Y[test_idx],
            None,
            0,
            np.array([1.0]),
            None,
            False,
            2,
            None,
            None,
            None,
            "cpu",
            str(tmp_path),
            "nr",
            orthonormalize_a_per_rank=True,
        )

        Q = np.linalg.qr(A_full[:, :2], mode="reduced")[0]
        np.testing.assert_allclose(captured[0][0], X[train_idx] @ Q)
        np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-6)

    def test_b_matrices_and_raw_rrr_metrics_disabled_by_default(
        self, tmp_path, small_data
    ):
        from run_rrr import _run_rrr_for_unit
        from src.core.config import RRRConfig

        X, Y = small_data
        test_idxs = [np.arange(X.shape[0] // 2, X.shape[0])]
        config = RRRConfig(rank_grid=[1])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        assert (tmp_path / "fold_0" / "A.npy").exists()
        assert not (tmp_path / "fold_0" / "B.npy").exists()
        metrics = np.load(tmp_path / "k1" / "fold_0" / "metrics.npz")
        assert not any(key.startswith("rrr_") for key in metrics.files)
        df = pd.read_csv(tmp_path / "grid_results.csv")
        assert "rrr_corr" not in df.columns
        assert "rrr_r2" not in df.columns

    def test_null_shift_sampler_enforces_minimum_and_seed(self):
        from run_rrr import _sample_valid_circular_shift

        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        shifts_a = [_sample_valid_circular_shift(20, 4, rng_a) for _ in range(5)]
        shifts_b = [_sample_valid_circular_shift(20, 4, rng_b) for _ in range(5)]

        assert shifts_a == shifts_b
        assert all(4 <= shift < 16 for shift in shifts_a)

    def test_null_shift_sampler_rejects_short_splits(self):
        from run_rrr import _sample_valid_circular_shift

        with pytest.raises(ValueError, match="too short"):
            _sample_valid_circular_shift(8, 4, np.random.default_rng(0))

    def test_null_augmented_metrics_saved_per_fold_and_rank(self, tmp_path, small_data):
        from run_rrr import _run_rrr_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[2, 4])
        config.run_null_augmented = True
        config.null_augmented_repeats = 2
        config.null_augmented_min_shift = 2
        config.encoding.ridge_alphas = [1.0]
        config.encoding.ridge_cv_folds = 2
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        metrics = np.load(os.path.join(str(tmp_path), "k2", "fold_0", "metrics.npz"))
        assert metrics["null_aug_corr_per_target"].shape == (2, Y.shape[1])
        assert metrics["null_aug_r2_per_target"].shape == (2, Y.shape[1])
        assert metrics["null_aug_shifts_train"].shape == (2,)
        assert metrics["null_aug_shifts_test"].shape == (2,)
        assert np.all(metrics["null_aug_shifts_train"] >= 2)
        assert np.all(metrics["null_aug_shifts_test"] >= 2)

        full_rank_metrics = np.load(
            os.path.join(str(tmp_path), "k4", "fold_0", "metrics.npz")
        )
        assert full_rank_metrics["null_aug_corr_per_target"].shape == (0, Y.shape[1])
        assert full_rank_metrics["null_aug_r2_per_target"].shape == (0, Y.shape[1])
        assert full_rank_metrics["null_aug_shifts_train"].shape == (0,)
        assert full_rank_metrics["null_aug_shifts_test"].shape == (0,)
        assert np.isnan(full_rank_metrics["null_aug_corr"])

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        for col in [
            "null_aug_corr",
            "null_aug_r2",
            "null_aug_corr_q05",
            "null_aug_corr_q50",
            "null_aug_corr_q95",
            "null_aug_full_advantage_within_q95",
        ]:
            assert col in df.columns

    def test_null_augmented_resume_recomputes_legacy_metrics(
        self, tmp_path, small_data
    ):
        import run_rrr

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        config = self._make_config(rank_grid=[1, 2])
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )

        config = self._make_config(rank_grid=[1, 2], refresh=False)
        config.run_null_augmented = True
        config.null_augmented_repeats = 1
        config.null_augmented_min_shift = 2
        config.encoding.ridge_alphas = [1.0]
        config.encoding.ridge_cv_folds = 2
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )

        metrics = np.load(os.path.join(str(tmp_path), "k1", "fold_0", "metrics.npz"))
        assert "null_aug_corr_per_target" in metrics.files

    def test_grid_results_csv_written(self, tmp_path, small_data):
        from run_rrr import _run_rrr_for_unit

        X, Y = small_data
        n = X.shape[0]
        # Two-fold split
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1, 3])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        csv_path = os.path.join(str(tmp_path), "grid_results.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        # 2 folds × 2 ranks = 4 rows
        assert len(df) == 4
        assert set(df["k"].unique()) == {1, 3}

    def test_pca_baseline_saves_fold_a_factors(self, tmp_path, small_data):
        from run_rrr import _run_rrr_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1])
        config.run_pca_baseline = True
        config.encoding.ridge_alphas = [1.0]
        config.encoding.ridge_cv_folds = 2
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        rrr_dir = tmp_path / "rrr"
        _run_rrr_for_unit(str(rrr_dir), X, test_idxs, per_fold_baselines, config)

        pca_a_path = tmp_path / "pca" / "fold_0" / "A.npy"
        assert pca_a_path.exists()
        A_pca = np.load(pca_a_path)
        n_train = n - len(test_idxs[0])
        assert A_pca.shape[0] == X.shape[1]
        assert A_pca.shape[1] <= min(n_train, X.shape[1])
        assert not (tmp_path / "pca" / "fold_0" / "B.npy").exists()

    def test_cross_dataset_subspace_writes_nr_metrics_without_factors(
        self, tmp_path, small_data
    ):
        from run_rrr import _run_cross_dataset_subspace_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1, 2])
        config.encoding.ridge_alphas = [1.0]
        config.encoding.ridge_cv_folds = 2
        per_fold_baselines = _make_baselines(X, Y, test_idxs)
        A_by_rank = {
            1: np.eye(X.shape[1], 1, dtype=np.float32),
            2: np.eye(X.shape[1], 2, dtype=np.float32),
        }

        _run_cross_dataset_subspace_for_unit(
            str(tmp_path),
            A_by_rank,
            X,
            test_idxs,
            per_fold_baselines,
            config,
            primary_label="nr",
        )

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        assert len(df) == 4
        assert {"full_corr", "full_r2", "nr_corr", "nr_r2", "effective_rank"}.issubset(
            df.columns
        )
        assert os.path.exists(os.path.join(str(tmp_path), "summary.csv"))
        assert os.path.exists(os.path.join(str(tmp_path), "rank_analysis.png"))
        assert not os.path.exists(os.path.join(str(tmp_path), "fold_0", "A.npy"))
        assert os.path.exists(
            os.path.join(str(tmp_path), "fold_0", "full_model_metrics.npz")
        )

    def test_full_model_metrics_refresh_overwrites_existing_file(self, tmp_path):
        from run_rrr import _save_full_model_metrics_once

        y_test = np.arange(24, dtype=float).reshape(8, 3)
        first_prediction = y_test.copy()
        refreshed_prediction = -y_test

        _save_full_model_metrics_once(
            str(tmp_path), 0, y_test, first_prediction
        )
        path = tmp_path / "fold_0" / "full_model_metrics.npz"
        with np.load(path) as saved:
            first = saved["full_corr_per_target"].copy()

        _save_full_model_metrics_once(
            str(tmp_path), 0, y_test, refreshed_prediction
        )
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["full_corr_per_target"], first)

        _save_full_model_metrics_once(
            str(tmp_path), 0, y_test, refreshed_prediction, refresh=True
        )
        with np.load(path) as saved:
            assert not np.array_equal(saved["full_corr_per_target"], first)

    def test_cross_dataset_subspace_uses_rank_specific_source_average(
        self, tmp_path, small_data, monkeypatch
    ):
        import run_rrr

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1, 2])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)
        A_rank1 = np.full((X.shape[1], 1), 1.0, dtype=np.float32)
        A_rank2 = np.full((X.shape[1], 2), 2.0, dtype=np.float32)
        seen = []

        def capture_rank_loop(**kwargs):
            rank = kwargs["ranks_to_run"][0]
            seen.append((rank, kwargs["A_full"].copy()))
            return [
                {
                    "fold": kwargs["fold_f"],
                    "k": rank,
                    "effective_rank": rank,
                    "full_corr": 0.0,
                    "full_r2": 0.0,
                    "nr_corr": 0.0,
                    "nr_r2": 0.0,
                }
            ]

        monkeypatch.setattr(
            run_rrr, "_run_subspace_rank_loop_for_fold", capture_rank_loop
        )
        monkeypatch.setattr(run_rrr, "_build_results", lambda *args, **kwargs: None)

        run_rrr._run_cross_dataset_subspace_for_unit(
            str(tmp_path),
            {1: A_rank1, 2: A_rank2},
            X,
            test_idxs,
            per_fold_baselines,
            config,
            primary_label="nr",
        )

        by_rank = {rank: matrix for rank, matrix in seen[:2]}
        np.testing.assert_array_equal(by_rank[1], A_rank1)
        np.testing.assert_array_equal(by_rank[2], A_rank2)

    def test_cross_dataset_pca_sibling_dir_and_metrics(self, tmp_path, small_data):
        import run_rrr

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1])
        config.encoding.ridge_alphas = [1.0]
        config.encoding.ridge_cv_folds = 2
        per_fold_baselines = _make_baselines(X, Y, test_idxs)
        A_by_rank = {1: np.eye(X.shape[1], 1, dtype=np.float32)}

        rrr_dir = tmp_path / "cross_dataset_rrr" / "movie-cars-2"
        pca_dir = run_rrr._resolve_pca_sibling_dir(str(rrr_dir))
        assert pca_dir.endswith(os.path.join("cross_dataset_pca", "movie-cars-2"))

        run_rrr._run_cross_dataset_subspace_for_unit(
            pca_dir,
            A_by_rank,
            X,
            test_idxs,
            per_fold_baselines,
            config,
            primary_label="pca",
        )

        df = pd.read_csv(os.path.join(pca_dir, "grid_results.csv"))
        assert len(df) == 2
        assert {"pca_corr", "pca_r2"}.issubset(df.columns)
        assert not os.path.exists(os.path.join(pca_dir, "fold_0", "A.npy"))
        assert not os.path.exists(os.path.join(pca_dir, "fold_0", "B.npy"))

    def test_summary_csv_averages_over_folds(self, tmp_path, small_data):
        from run_rrr import _run_rrr_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[2, 5])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        summary_path = os.path.join(str(tmp_path), "summary.csv")
        assert os.path.exists(summary_path)
        df = pd.read_csv(summary_path)
        # One row per rank
        assert len(df) == 2
        assert set(df["k"].unique()) == {2, 5}
        expected = [
            "full_corr",
            "full_r2",
            "rrr_corr",
            "rrr_r2",
            "nr_corr",
            "nr_r2",
            "effective_rank",
        ]
        for col in expected:
            assert col in df.columns
        grid_df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        for col in expected:
            assert col in grid_df.columns
        removed = [
            "complement_corr",
            "complement_r2",
            "nr_complement_joint_corr",
            "nr_arbitrary_joint_corr",
            "nr_complement_pred_corr",
            "nr_arbitrary_pred_corr",
            "arbitrary_corr",
            "arbitrary_r2",
        ]
        for col in removed:
            assert col not in grid_df.columns
            assert col not in df.columns

        metrics = np.load(os.path.join(str(tmp_path), "k2", "fold_0", "metrics.npz"))
        assert "nr_corr_per_target" in metrics.files
        assert "rrr_corr_per_target" not in metrics.files
        assert "nr_best_alphas" in metrics.files
        assert "complement_corr" not in metrics.files

    def test_per_target_corr_save_flags(self, tmp_path, small_data):
        from run_rrr import _run_rrr_for_unit

        X, Y = small_data
        test_idxs = [np.arange(X.shape[0] // 2, X.shape[0])]
        config = self._make_config(rank_grid=[1])
        config.save_nr_corr_per_target = False
        config.save_rrr_corr_per_target = True
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        metrics = np.load(os.path.join(str(tmp_path), "k1", "fold_0", "metrics.npz"))
        assert "nr_corr_per_target" not in metrics.files
        assert "rrr_corr_per_target" in metrics.files

    def test_refresh_without_b_removes_stale_b(self, tmp_path, small_data):
        from run_rrr import _run_rrr_for_unit

        X, Y = small_data
        test_idxs = [np.arange(X.shape[0] // 2, X.shape[0])]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)
        config = self._make_config(rank_grid=[1])
        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)
        assert (tmp_path / "fold_0" / "B.npy").exists()

        config.save_b_matrices = False
        config.save_rrr_corr_per_target = False
        config.refresh = True
        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        assert not (tmp_path / "fold_0" / "B.npy").exists()
        metrics = np.load(tmp_path / "k1" / "fold_0" / "metrics.npz")
        assert not any(key.startswith("rrr_") for key in metrics.files)

    def test_nr_r2_increases_with_rank(self, tmp_path, rng):
        """For low-rank ground truth data, NR R² should increase monotonically with rank."""
        from run_rrr import _run_rrr_for_unit

        # Construct rank-3 signal
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

        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        r2_by_k = df.groupby("k")["nr_r2"].mean().sort_index()
        # R² should be non-decreasing with rank
        assert all(
            r2_by_k.diff().dropna() >= -0.05
        ), f"R² not monotone: {r2_by_k.to_dict()}"

    def test_result_files_saved_per_fold_and_rank(self, tmp_path, small_data):
        from run_rrr import _run_rrr_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[2])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        for fold in range(2):
            # Full-rank factors saved at fold level
            factors_dir = os.path.join(str(tmp_path), f"fold_{fold}")
            assert os.path.exists(os.path.join(factors_dir, "A.npy"))
            assert os.path.exists(os.path.join(factors_dir, "B.npy"))

            # load_rrr_result truncates to requested k
            result = load_rrr_result(str(tmp_path), k=2, fold=fold)
            assert result["A"].shape[1] == 2
            assert result["B"].shape[1] == 2
            assert "full_r2" in result["metrics"]
            assert "nr_r2" in result["metrics"]

    def test_resume_adds_only_missing_rank_and_preserves_existing_rows(
        self, tmp_path, small_data, monkeypatch
    ):
        import run_rrr

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        config = self._make_config(rank_grid=[1])
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )
        first_df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        rank1_before = (
            first_df[first_df["k"] == 1].sort_values("fold").reset_index(drop=True)
        )

        calls = {"n": 0}
        original_fit = run_rrr.fit_dense_rrr

        def counting_fit(*args, **kwargs):
            calls["n"] += 1
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(run_rrr, "fit_dense_rrr", counting_fit)
        config = self._make_config(rank_grid=[1, 3], refresh=False)
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )

        assert calls["n"] == 0
        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        assert set(df["k"].unique()) == {1, 3}
        rank1_after = df[df["k"] == 1].sort_values("fold").reset_index(drop=True)
        pd.testing.assert_frame_equal(rank1_before, rank1_after, check_dtype=False)
        assert os.path.exists(os.path.join(str(tmp_path), "summary.csv"))
        assert os.path.exists(os.path.join(str(tmp_path), "rank_analysis.png"))

    def test_resume_rebuilds_missing_metrics_file_from_saved_factors(
        self, tmp_path, small_data, monkeypatch
    ):
        import run_rrr

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        config = self._make_config(rank_grid=[1, 3])
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )
        os.remove(os.path.join(str(tmp_path), "k1", "fold_0", "metrics.npz"))

        calls = {"n": 0}
        original_fit = run_rrr.fit_dense_rrr

        def counting_fit(*args, **kwargs):
            calls["n"] += 1
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(run_rrr, "fit_dense_rrr", counting_fit)
        config = self._make_config(rank_grid=[1, 3], refresh=False)
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )

        assert calls["n"] == 0
        assert os.path.exists(
            os.path.join(str(tmp_path), "k1", "fold_0", "metrics.npz")
        )
        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        assert len(df) == 4
        assert set(df["k"].unique()) == {1, 3}

    def test_resume_skips_fitting_when_all_ranks_complete(
        self, tmp_path, small_data, monkeypatch
    ):
        import run_rrr

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        config = self._make_config(rank_grid=[1])
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )

        def fail_fit(*args, **kwargs):
            raise AssertionError(
                "fit_dense_rrr should not run when all ranks are complete"
            )

        monkeypatch.setattr(run_rrr, "fit_dense_rrr", fail_fit)
        config = self._make_config(rank_grid=[1], refresh=False)
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )

    def test_resume_rebuilds_grid_from_artifacts_without_grid_csv(
        self, tmp_path, small_data, monkeypatch
    ):
        import run_rrr

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        config = self._make_config(rank_grid=[1])
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )
        os.remove(os.path.join(str(tmp_path), "grid_results.csv"))

        def fail_fit(*args, **kwargs):
            raise AssertionError(
                "fit_dense_rrr should not run when artifacts are complete"
            )

        monkeypatch.setattr(run_rrr, "fit_dense_rrr", fail_fit)
        config = self._make_config(rank_grid=[1], refresh=False)
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        assert len(df) == 2
        assert set(df["fold"].unique()) == {0, 1}
        assert set(df["k"].unique()) == {1}
        assert "rrr_corr" in df.columns
        assert "nr_corr" in df.columns
        assert "nr_complement_joint_corr" not in df.columns

    def test_uses_supplied_resume_schedule_without_recomputing_it(
        self, tmp_path, small_data, monkeypatch
    ):
        import run_rrr

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)
        config = self._make_config(rank_grid=[1])

        def fail_schedule(*args, **kwargs):
            raise AssertionError("_run_rrr_for_unit should use the supplied schedule")

        monkeypatch.setattr(run_rrr, "_ranks_by_fold_to_run", fail_schedule)
        run_rrr._run_rrr_for_unit(
            str(tmp_path),
            X,
            test_idxs,
            per_fold_baselines,
            config,
            ranks_by_fold={0: [1], 1: [1]},
            completed_pairs=set(),
        )

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        assert len(df) == 2

    def test_refresh_recomputes_all_ranks(self, tmp_path, small_data, monkeypatch):
        import run_rrr

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        config = self._make_config(rank_grid=[1])
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )

        calls = {"n": 0}
        original_fit = run_rrr.fit_dense_rrr

        def counting_fit(*args, **kwargs):
            calls["n"] += 1
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(run_rrr, "fit_dense_rrr", counting_fit)
        config = self._make_config(rank_grid=[1], refresh=True)
        run_rrr._run_rrr_for_unit(
            str(tmp_path), X, test_idxs, per_fold_baselines, config
        )

        assert calls["n"] == len(test_idxs)

    def test_decomposition_inputs_accept_multiple_embeddings_and_slim_results(
        self, tmp_path, small_data, monkeypatch
    ):
        import run_rrr
        from src.core.config import EncodingConfig
        from src.utils.encoding import load_encoding_results, save_encoding_results

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        output_dir = tmp_path / "encoding_run"
        sub_dir = output_dir / "subjects" / "sub_1"
        sub_dir.mkdir(parents=True)
        save_encoding_results(
            {
                "corrs": np.zeros((2, 1, 5), dtype=np.float32),
                "train_idxs": [
                    np.arange(n // 2, n),
                    np.arange(0, n // 2),
                ],
                "test_idxs": test_idxs,
                "scalers": [None, None],
                "elec_names": ["E1"],
                "subject_id": 1,
                "offset": 0,
                "lags": np.linspace(-2.0, 2.0, 5, endpoint=False),
                "per_model_config": {"layer": 24},
            },
            str(sub_dir / "encoding_results.joblib"),
        )

        loaded = load_encoding_results(str(sub_dir / "encoding_results.joblib"))
        assert "models" not in loaded
        assert "coefficients" not in loaded
        assert "intercepts" not in loaded

        def fake_build_xy(
            subject_ids, encoding_config, sig_electrodes, embeddings, **kwargs
        ):
            assert subject_ids == [1]
            assert sig_electrodes == {1: ["E1"]}
            return X, Y, np.arange(n), [["E1"]]

        monkeypatch.setattr("src.utils.decomposition.build_XY_from_subjects", fake_build_xy)

        enc = EncodingConfig(ridge_alphas=[1.0, 10.0], ridge_cv_folds=2)
        result = run_rrr._get_decomposition_inputs(
            [1],
            str(output_dir),
            {1: ["E1"]},
            enc,
            embeddings={"first": X, "second": X + 1},
        )

        np.testing.assert_array_equal(result.embeddings["first"], X)
        np.testing.assert_array_equal(result.embeddings["second"], X + 1)
        np.testing.assert_array_equal(result.targets, Y)
        assert result.sig_mask_flat is None
        assert result.target_manifest[["subject", "electrode"]].drop_duplicates().values.tolist() == [[1, "E1"]]
        for got, expected in zip(result.test_idxs, test_idxs):
            np.testing.assert_array_equal(got, expected)

    def test_shared_selection_metadata_requires_matching_subjects(self, tmp_path):
        import run_rrr
        from src.utils.encoding import save_encoding_results

        output_dir = tmp_path / "encoding_run"
        for sub_id in [1, 2]:
            sub_dir = output_dir / "subjects" / f"sub_{sub_id}"
            sub_dir.mkdir(parents=True)
            save_encoding_results(
                {
                    "test_idxs": [np.array([0])],
                    "word_selection": {
                        "mode": "shared_intersection",
                        "subjects": [1, 3],
                        "epoch_selection": np.array([0, 2]),
                        "analysis_selection": np.array([0, 2]),
                        "offset": 0,
                        "valid_rows_applied": False,
                    },
                },
                str(sub_dir / "encoding_results.joblib"),
            )

        with pytest.raises(ValueError, match="subjects do not match"):
            run_rrr._load_shared_word_selection_metadata(str(output_dir), [1, 2])

    def test_shared_selection_metadata_loads_when_all_subjects_agree(self, tmp_path):
        import run_rrr
        from src.utils.encoding import save_encoding_results

        output_dir = tmp_path / "encoding_run"
        for sub_id in [1, 2]:
            sub_dir = output_dir / "subjects" / f"sub_{sub_id}"
            sub_dir.mkdir(parents=True)
            save_encoding_results(
                {
                    "test_idxs": [np.array([0])],
                    "word_selection": {
                        "mode": "shared_intersection",
                        "subjects": [1, 2],
                        "epoch_selection": np.array([0, 2]),
                        "analysis_selection": np.array([0, 2]),
                        "offset": 0,
                        "valid_rows_applied": False,
                    },
                },
                str(sub_dir / "encoding_results.joblib"),
            )

        metadata = run_rrr._load_shared_word_selection_metadata(str(output_dir), [1, 2])
        np.testing.assert_array_equal(metadata["epoch_selection"], np.array([0, 2]))

# ---------------------------------------------------------------------------
# TestBuildXYFromSubjects
# ---------------------------------------------------------------------------


class TestBuildXYFromSubjects:
    """Tests for build_XY_from_subjects using mocked data-loading functions."""

    def _make_enc(self):
        from src.core.config import EncodingConfig

        enc = EncodingConfig()
        enc.data_root = "data/"
        enc.offset = 0
        return enc

    def test_concatenates_Y_across_subjects(self, rng, monkeypatch):
        """Y should be column-concatenation of per-subject neural responses."""
        from src.utils.data_loaders import build_XY_from_subjects

        n_words, n_embed, n_elecs_per_sub, n_lags = 50, 8, 3, 4

        X_emb = rng.standard_normal((n_words, n_embed)).astype(np.float32)
        df_word = pd.DataFrame({"start": np.linspace(0, 10, n_words)})

        def _fake_load_precomputed_epochs(sub_id, data_root):
            return {
                "raw_data": np.zeros(
                    (n_elecs_per_sub, n_words * n_lags), dtype=np.float32
                ),
                "selection": np.arange(n_words),
                "epoch_starts": np.arange(n_words) * n_lags,
                "n_lags": n_lags,
                "ch_names": [f"ch{sub_id}{i}" for i in range(n_elecs_per_sub)],
            }

        def _fake_epochs_from_cache(cache, elec_names=None):
            n_elecs = len(elec_names) if elec_names is not None else n_elecs_per_sub
            return np.zeros((n_words, n_elecs, n_lags), dtype=np.float32)

        def _fake_apply_offset(X_sel, data, offset, valid_rows=None):
            n = data.shape[0]
            return X_sel, data.reshape(n, -1).astype(np.float32)

        import src.utils.data_loaders as dl

        monkeypatch.setattr(
            dl, "load_precomputed_epochs", _fake_load_precomputed_epochs
        )
        monkeypatch.setattr(dl, "epochs_from_cache", _fake_epochs_from_cache)

        import src.utils.encoding as enc_mod

        monkeypatch.setattr(enc_mod, "apply_offset", _fake_apply_offset)

        enc = self._make_enc()
        sig_electrodes = {1: ["ch1", "ch2", "ch3"], 2: ["ch4", "ch5", "ch6"]}

        X, Y, _, _ = build_XY_from_subjects([1, 2], enc, sig_electrodes, X_emb)

        assert X.shape == (n_words, n_embed)
        # Two subjects × (n_elecs_per_sub × n_lags) columns
        assert Y.shape[0] == n_words
        assert Y.shape[1] == 2 * n_elecs_per_sub * 4
        assert Y.dtype == np.float32

    def test_raises_when_no_sig_electrodes(self, rng):
        from src.utils.data_loaders import build_XY_from_subjects

        n_words, n_embed = 50, 8
        X_emb = rng.standard_normal((n_words, n_embed)).astype(np.float32)
        df_word = pd.DataFrame({"start": np.linspace(0, 10, n_words)})
        enc = self._make_enc()

        with pytest.raises(ValueError, match="No subjects with significant electrodes"):
            build_XY_from_subjects([1, 2], enc, {}, X_emb, df_word)
