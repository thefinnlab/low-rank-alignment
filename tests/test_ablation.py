"""Unit tests for RRR latent space ablation: masking functions and _run_ablation_for_unit."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

from src.utils.srrr import fit_dense_rrr, save_rrr_factors


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
# Helper: build baselines (same pattern as test_rrr.py)
# ---------------------------------------------------------------------------


def _make_baselines(X, Y, test_idxs):
    """Build per_fold_baselines from raw data using a fresh ridge model."""
    from sklearn.preprocessing import StandardScaler
    from src.utils.encoding import get_ridge_model_constructor

    n_words, n_embed = X.shape
    baselines = []
    for test_idx in test_idxs:
        train_mask = np.ones(n_words, dtype=bool)
        train_mask[test_idx] = False
        train_idx_arr = np.where(train_mask)[0]
        scaler = StandardScaler()
        Y_train = scaler.fit_transform(Y[train_idx_arr])
        Y_test = scaler.transform(Y[test_idx])
        model = get_ridge_model_constructor()(0)
        model.fit(X[train_idx_arr], Y_train)
        coef = np.asarray(model[-1].coef_)
        if coef.shape[0] != n_embed:
            coef = coef.T
        beta_full_raw = (coef / model[0].scale_[:, np.newaxis]).astype(np.float32)
        baselines.append(
            {
                "train_idx": train_idx_arr,
                "test_idx": test_idx,
                "Y_train": Y_train,
                "Y_test": Y_test,
                "Y_hat_full_train": np.asarray(model.predict(X[train_idx_arr])),
                "Y_hat_full_test": np.asarray(model.predict(X[test_idx])),
                "beta_full_raw": beta_full_raw,
            }
        )
    return baselines


def _setup_rrr_output(tmp_path, X, Y, test_idxs):
    """Run a minimal RRR to produce A/B matrices and grid_results.csv."""
    rrr_dir = str(tmp_path / "rrr")
    os.makedirs(rrr_dir, exist_ok=True)
    baselines = _make_baselines(X, Y, test_idxs)
    n_embed = X.shape[1]
    max_k = min(5, n_embed)

    # Build grid_results rows for the baseline
    grid_rows = []
    for fold_f, baseline in enumerate(baselines):
        result = fit_dense_rrr(
            X[baseline["train_idx"]],
            baseline["Y_train"],
            k=max_k,
            selected_idx=np.arange(n_embed),
            precomputed_Y_hat_ols=baseline["Y_hat_full_train"],
            precomputed_beta_full=baseline["beta_full_raw"],
        )
        save_rrr_factors(rrr_dir, fold=fold_f, A=result["A_dense"], B=result["B_dense"])
        for k in [1, 2, 3, 5]:
            if k > max_k:
                continue
            grid_rows.append(
                {
                    "fold": fold_f,
                    "k": k,
                    "full_r2": 0.5,
                    "full_corr": 0.1,
                    "nr_r2": 0.28,
                    "nr_corr": 0.065,
                    "complement_r2": 0.15,
                    "complement_corr": 0.04,
                }
            )

    pd.DataFrame(grid_rows).to_csv(
        os.path.join(rrr_dir, "grid_results.csv"), index=False
    )
    return rrr_dir, baselines


# ---------------------------------------------------------------------------
# TestMaskingFunctions
# ---------------------------------------------------------------------------


class TestMaskingFunctions:
    """Unit tests for quantile-based masking."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts", "rrr")
        scripts_dir = os.path.abspath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    def test_center_mask_keeps_correct_fraction(self, rng):
        from run_ablation import compute_quantile_thresholds, apply_quantile_mask

        X_proj = rng.standard_normal((1000, 5)).astype(np.float32)
        center_pct = 50
        lo, hi = compute_quantile_thresholds(X_proj, center_pct)
        masked = apply_quantile_mask(X_proj, lo, hi, "center")
        frac_kept = np.mean(masked != 0)
        assert abs(frac_kept - 0.50) < 0.05

    def test_tail_mask_zeros_center(self, rng):
        from run_ablation import compute_quantile_thresholds, apply_quantile_mask

        X_proj = rng.standard_normal((1000, 5)).astype(np.float32)
        center_pct = 30
        lo, hi = compute_quantile_thresholds(X_proj, center_pct)
        masked = apply_quantile_mask(X_proj, lo, hi, "tail")
        frac_zeroed = np.mean(masked == 0)
        assert abs(frac_zeroed - 0.30) < 0.05

    def test_tail_and_center_are_complementary(self, rng):
        from run_ablation import compute_quantile_thresholds, apply_quantile_mask

        X_proj = rng.standard_normal((200, 3)).astype(np.float32)
        lo, hi = compute_quantile_thresholds(X_proj, 60)
        tail = apply_quantile_mask(X_proj, lo, hi, "tail")
        center = apply_quantile_mask(X_proj, lo, hi, "center")
        reconstructed = tail + center
        np.testing.assert_array_almost_equal(reconstructed, X_proj)

    def test_random_mask_zeros_correct_fraction(self, rng):
        from run_ablation import apply_random_mask

        X_proj = rng.standard_normal((1000, 5)).astype(np.float32)
        masked = apply_random_mask(X_proj, 0.4, rng)
        frac_zeroed = np.mean(masked == 0)
        assert abs(frac_zeroed - 0.4) < 0.05

    def test_masks_preserve_nonzero_values(self, rng):
        from run_ablation import compute_quantile_thresholds, apply_quantile_mask

        X_proj = rng.standard_normal((100, 3)).astype(np.float32)
        lo, hi = compute_quantile_thresholds(X_proj, 50)
        for mask_type in ["tail", "center"]:
            masked = apply_quantile_mask(X_proj, lo, hi, mask_type)
            nonzero = masked != 0
            np.testing.assert_array_equal(masked[nonzero], X_proj[nonzero])

    def test_thresholds_centered_on_median(self, rng):
        from run_ablation import compute_quantile_thresholds

        X_proj = rng.standard_normal((10000, 3)).astype(np.float32)
        lo, hi = compute_quantile_thresholds(X_proj, 50)
        medians = np.median(X_proj, axis=0)
        # lo and hi should be roughly symmetric around the median
        for i in range(3):
            assert abs((hi[i] - medians[i]) - (medians[i] - lo[i])) < 0.1


# ---------------------------------------------------------------------------
# TestRunAblationForUnit
# ---------------------------------------------------------------------------


class TestRunAblationForUnit:
    """Integration tests for _run_ablation_for_unit."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts", "rrr")
        scripts_dir = os.path.abspath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    def _make_config(self, **overrides):
        from src.core.config import AblationConfig

        config = AblationConfig()
        for k, v in overrides.items():
            setattr(config, k, v)
        return config

    def test_grid_results_csv_written(self, tmp_path, small_data):
        from run_ablation import _run_ablation_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2), np.arange(n // 2, n)]
        rrr_dir, baselines = _setup_rrr_output(tmp_path, X, Y, test_idxs)

        config = self._make_config(
            rank_grid=[1, 3],
            center_pcts=[30, 70],
            mask_types=["tail", "center"],
            n_random_seeds=2,
        )

        ablation_dir = str(tmp_path / "ablation")
        _run_ablation_for_unit(ablation_dir, rrr_dir, X, test_idxs, baselines, config)

        csv_path = os.path.join(ablation_dir, "grid_results.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)

        # Per fold per k: 1 baseline + 2 pcts * 2 masks = 5 rows
        # 2 folds * 2 ranks * 5 = 20 rows
        n_expected = 2 * 2 * (1 + 2 * 2)
        assert len(df) == n_expected

    def test_summary_csv_written(self, tmp_path, small_data):
        from run_ablation import _run_ablation_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2)]
        rrr_dir, baselines = _setup_rrr_output(tmp_path, X, Y, test_idxs)

        config = self._make_config(
            rank_grid=[2],
            center_pcts=[50],
            mask_types=["tail"],
        )

        ablation_dir = str(tmp_path / "ablation")
        _run_ablation_for_unit(ablation_dir, rrr_dir, X, test_idxs, baselines, config)

        summary_path = os.path.join(ablation_dir, "summary.csv")
        assert os.path.exists(summary_path)
        df = pd.read_csv(summary_path)
        assert "mask_type" in df.columns
        assert "nr_corr" in df.columns

    def test_unmasked_baseline_included(self, tmp_path, small_data):
        from run_ablation import _run_ablation_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2)]
        rrr_dir, baselines = _setup_rrr_output(tmp_path, X, Y, test_idxs)

        config = self._make_config(
            rank_grid=[3],
            center_pcts=[50],
            mask_types=["tail"],
        )

        ablation_dir = str(tmp_path / "ablation")
        _run_ablation_for_unit(ablation_dir, rrr_dir, X, test_idxs, baselines, config)

        df = pd.read_csv(os.path.join(ablation_dir, "grid_results.csv"))
        assert "none" in df["mask_type"].values

    def test_per_k_plots_created(self, tmp_path, small_data):
        from run_ablation import _run_ablation_for_unit

        X, Y = small_data
        n = X.shape[0]
        test_idxs = [np.arange(0, n // 2)]
        rrr_dir, baselines = _setup_rrr_output(tmp_path, X, Y, test_idxs)

        config = self._make_config(
            rank_grid=[2, 3],
            center_pcts=[30, 70],
            mask_types=["tail", "center"],
        )

        ablation_dir = str(tmp_path / "ablation")
        _run_ablation_for_unit(ablation_dir, rrr_dir, X, test_idxs, baselines, config)

        for k in [2, 3]:
            plot_path = os.path.join(ablation_dir, f"k{k}", "ablation_pct_zeroed.png")
            assert os.path.exists(plot_path), f"Missing plot for k={k}"

        rank_plot_path = os.path.join(ablation_dir, "ablation_vs_rank.png")
        assert os.path.exists(rank_plot_path)

    def test_more_zeroing_reduces_performance(self, tmp_path, rng):
        """With strong low-rank signal, zeroing more data should reduce nr_corr."""
        from run_ablation import _run_ablation_for_unit

        n, n_embed, n_targets = 200, 20, 8
        true_rank = 3
        A_true = rng.standard_normal((n_embed, true_rank)).astype(np.float32)
        B_true = rng.standard_normal((n_targets, true_rank)).astype(np.float32)
        X = rng.standard_normal((n, n_embed)).astype(np.float32)
        Y = X @ A_true @ B_true.T + 0.01 * rng.standard_normal((n, n_targets)).astype(
            np.float32
        )

        test_idxs = [np.arange(n // 2, n)]
        rrr_dir, baselines = _setup_rrr_output(tmp_path, X, Y, test_idxs)

        config = self._make_config(
            rank_grid=[3],
            center_pcts=[10, 50, 90],
            mask_types=["random"],
            n_random_seeds=3,
        )

        ablation_dir = str(tmp_path / "ablation")
        _run_ablation_for_unit(ablation_dir, rrr_dir, X, test_idxs, baselines, config)

        df = pd.read_csv(os.path.join(ablation_dir, "summary.csv"))
        random_df = df[df["mask_type"] == "random"].sort_values("pct_zeroed")
        corrs = random_df["nr_corr"].values
        # More zeroing should reduce correlation (with tolerance)
        assert corrs[0] >= corrs[-1] - 0.1, f"Corrs not decreasing: {corrs}"
