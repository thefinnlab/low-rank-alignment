"""Unit tests for RRR utilities: save_rrr_result/load_rrr_result and _run_rrr_for_unit."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

from utils.srrr import save_rrr_factors, save_rrr_result, load_rrr_result


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
            "minimal_r2": 0.4,
            "recovery_r2": 0.05,
            "full_corr": 0.03,
            "minimal_corr": 0.02,
            "recovery_corr": 0.01,
            "cca_similarity": 0.9,
            "procrustes_similarity": 0.85,
            "fold": fold,
            "k": k,
        }

        save_rrr_factors(str(tmp_path), fold=fold, A=A_full, B=B_full)
        save_rrr_result(str(tmp_path), k=k, fold=fold, metrics=metrics)
        result = load_rrr_result(str(tmp_path), k=k, fold=fold)

        np.testing.assert_array_almost_equal(result["A"], A_full[:, :k])
        np.testing.assert_array_almost_equal(result["B"], B_full[:, :k])
        assert abs(float(result["metrics"]["full_r2"]) - 0.5) < 1e-5
        assert abs(float(result["metrics"]["recovery_r2"]) - 0.05) < 1e-5

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

    def test_multiple_folds_independent(self, tmp_path, rng):
        max_k, k = 4, 2
        for fold in range(3):
            A = rng.standard_normal((6, max_k)).astype(np.float32)
            B = rng.standard_normal((4, max_k)).astype(np.float32)
            save_rrr_factors(str(tmp_path), fold=fold, A=A, B=B)
            save_rrr_result(str(tmp_path), k=k, fold=fold, metrics={"full_r2": float(fold)})

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
    """Build per_fold_baselines from raw data using a fresh ridge model (for tests only).

    Replicates the old per-fold scaler + ridge fit so that TestRunRRRForUnit
    can call _run_rrr_for_unit without needing saved encoding_results.joblib files.
    """
    from sklearn.preprocessing import StandardScaler
    from utils.encoding import get_ridge_model_constructor
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
        baselines.append({
            "train_idx": train_idx,
            "test_idx": test_idx,
            "Y_train": Y_train,
            "Y_test": Y_test,
            "Y_hat_full_train": np.asarray(model.predict(X[train_idx])),
            "Y_hat_full_test": np.asarray(model.predict(X[test_idx])),
            "beta_full_raw": beta_full_raw,
        })
    return baselines


class TestRunRRRForUnit:
    """Integration tests for _run_rrr_for_unit using synthetic data."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
        scripts_dir = os.path.abspath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    def _make_config(self, rank_grid):
        from utils.config import RRRConfig
        config = RRRConfig()
        config.rank_grid = rank_grid
        return config

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

    def test_minimal_r2_increases_with_rank(self, tmp_path, rng):
        """For low-rank ground truth data, R² should increase monotonically with rank."""
        from run_rrr import _run_rrr_for_unit

        # Construct rank-3 signal
        n, n_embed, n_targets = 120, 20, 8
        true_rank = 3
        A_true = rng.standard_normal((n_embed, true_rank)).astype(np.float32)
        B_true = rng.standard_normal((n_targets, true_rank)).astype(np.float32)
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = X @ A_true @ B_true.T + 0.01 * rng.standard_normal((n, n_targets)).astype(np.float32)

        test_idxs = [np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1, 2, 3, 5])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        r2_by_k = df.groupby("k")["minimal_r2"].mean().sort_index()
        # R² should be non-decreasing with rank
        assert all(r2_by_k.diff().dropna() >= -0.05), f"R² not monotone: {r2_by_k.to_dict()}"

    def test_recovery_r2_decreases_with_rank(self, tmp_path, rng):
        """Residual recovery R² should decrease as rank increases (less residual signal)."""
        from run_rrr import _run_rrr_for_unit

        n, n_embed, n_targets = 120, 20, 8
        true_rank = 3
        A_true = rng.standard_normal((n_embed, true_rank)).astype(np.float32)
        B_true = rng.standard_normal((n_targets, true_rank)).astype(np.float32)
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = X @ A_true @ B_true.T + 0.01 * rng.standard_normal((n, n_targets)).astype(np.float32)

        test_idxs = [np.arange(n // 2, n)]
        config = self._make_config(rank_grid=[1, 2, 3, 5])
        per_fold_baselines = _make_baselines(X, Y, test_idxs)

        _run_rrr_for_unit(str(tmp_path), X, test_idxs, per_fold_baselines, config)

        df = pd.read_csv(os.path.join(str(tmp_path), "grid_results.csv"))
        recovery_by_k = df.groupby("k")["recovery_r2"].mean().sort_index()
        # Recovery R² should be non-increasing
        assert all(recovery_by_k.diff().dropna() <= 0.05), f"Recovery R² not monotone: {recovery_by_k.to_dict()}"

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
            assert "recovery_r2" in result["metrics"]


# ---------------------------------------------------------------------------
# TestBuildXYFromSubjects
# ---------------------------------------------------------------------------

class TestBuildXYFromSubjects:
    """Tests for build_XY_from_subjects using mocked data-loading functions."""

    def _make_enc(self):
        from utils.config import EncodingConfig
        enc = EncodingConfig()
        enc.data_root = "data/"
        enc.offset = 0
        return enc

    def test_concatenates_Y_across_subjects(self, rng, monkeypatch):
        """Y should be column-concatenation of per-subject neural responses."""
        from utils.data_loaders import build_XY_from_subjects

        n_words, n_embed, n_elecs_per_sub, n_lags = 50, 8, 3, 4

        X_emb = rng.standard_normal((n_words, n_embed)).astype(np.float32)
        df_word = pd.DataFrame({"start": np.linspace(0, 10, n_words)})

        def _fake_load_precomputed_epochs(sub_id, data_root):
            return {
                "raw_data": np.zeros((n_elecs_per_sub, n_words * n_lags), dtype=np.float32),
                "selection": np.arange(n_words),
                "epoch_starts": np.arange(n_words) * n_lags,
                "n_lags": n_lags,
                "ch_names": [f"ch{sub_id}{i}" for i in range(n_elecs_per_sub)],
            }

        def _fake_epochs_from_cache(cache, elec_names=None):
            n_elecs = len(elec_names) if elec_names is not None else n_elecs_per_sub
            return np.zeros((n_words, n_elecs, n_lags), dtype=np.float32)

        def _fake_apply_offset(X_sel, data, offset):
            n = data.shape[0]
            return X_sel, data.reshape(n, -1).astype(np.float32)

        import utils.data_loaders as dl
        monkeypatch.setattr(dl, "load_precomputed_epochs", _fake_load_precomputed_epochs)
        monkeypatch.setattr(dl, "epochs_from_cache", _fake_epochs_from_cache)

        import utils.encoding as enc_mod
        monkeypatch.setattr(enc_mod, "apply_offset", _fake_apply_offset)

        enc = self._make_enc()
        sig_electrodes = {1: ["ch1", "ch2", "ch3"], 2: ["ch4", "ch5", "ch6"]}

        X, Y, _ = build_XY_from_subjects([1, 2], enc, sig_electrodes, X_emb, df_word)

        assert X.shape == (n_words, n_embed)
        # Two subjects × (n_elecs_per_sub × n_lags) columns
        assert Y.shape[0] == n_words
        assert Y.shape[1] == 2 * n_elecs_per_sub * 4
        assert Y.dtype == np.float32

    def test_raises_when_no_sig_electrodes(self, rng):
        from utils.data_loaders import build_XY_from_subjects

        n_words, n_embed = 50, 8
        X_emb = rng.standard_normal((n_words, n_embed)).astype(np.float32)
        df_word = pd.DataFrame({"start": np.linspace(0, 10, n_words)})
        enc = self._make_enc()

        with pytest.raises(ValueError, match="No subjects with significant electrodes"):
            build_XY_from_subjects([1, 2], enc, {}, X_emb, df_word)
