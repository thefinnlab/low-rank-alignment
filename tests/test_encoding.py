"""Tests for utils/encoding.py."""
import numpy as np
import pytest
import torch

from utils.encoding import absorb_scaler, get_all_targets_test_set


# ---------------------------------------------------------------------------
# get_all_targets_test_set
# ---------------------------------------------------------------------------

class TestGetAllTargetsTestSet:
    def test_2fold_output_shape(self):
        n_words, n_electrodes, n_lags = 20, 3, 4
        rng = np.random.default_rng(0)
        Y = rng.standard_normal((n_words, n_electrodes * n_lags)).astype(np.float32)

        # 2-fold split: first 10 / last 10
        test_idxs = [np.arange(0, 10), np.arange(10, 20)]
        result = get_all_targets_test_set(test_idxs, Y, n_electrodes, n_lags)

        assert result.shape == (n_words, n_electrodes, n_lags)

    def test_values_from_correct_fold(self):
        n_words, n_electrodes, n_lags = 10, 2, 3
        rng = np.random.default_rng(1)
        Y = rng.standard_normal((n_words, n_electrodes * n_lags)).astype(np.float32)

        test_idxs = [np.arange(0, 5), np.arange(5, 10)]
        result = get_all_targets_test_set(test_idxs, Y, n_electrodes, n_lags)

        # The first 5 rows of result should come from Y[0:5]
        expected_first5 = Y[0:5].reshape(5, n_electrodes, n_lags)
        np.testing.assert_array_equal(result[:5], expected_first5)

        # The last 5 rows of result should come from Y[5:10]
        expected_last5 = Y[5:10].reshape(5, n_electrodes, n_lags)
        np.testing.assert_array_equal(result[5:], expected_last5)

    def test_non_contiguous_folds_reassemble_correctly(self):
        """Even-odd split: fold 0 = even indices, fold 1 = odd indices."""
        n_words, n_electrodes, n_lags = 12, 2, 2
        rng = np.random.default_rng(2)
        Y = rng.standard_normal((n_words, n_electrodes * n_lags)).astype(np.float32)

        fold0 = np.arange(0, n_words, 2)  # 0, 2, 4, 6, 8, 10
        fold1 = np.arange(1, n_words, 2)  # 1, 3, 5, 7, 9, 11

        result = get_all_targets_test_set([fold0, fold1], Y, n_electrodes, n_lags)

        # vstack([Y[fold0], Y[fold1]]) => rows 0,2,4,6,8,10,1,3,5,7,9,11
        expected_vstack = np.vstack([Y[fold0], Y[fold1]])
        expected = expected_vstack.reshape((n_words, n_electrodes, n_lags))
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# absorb_scaler
# ---------------------------------------------------------------------------

class TestAbsorbScaler:
    """
    Build a small himalaya pipeline on a known linear system and verify that
    absorb_scaler produces raw-space coefficients that reproduce predictions.
    """

    @pytest.fixture(autouse=True)
    def build_pipeline(self):
        """Fit a StandardScaler + RidgeCV pipeline on small synthetic data."""
        from himalaya.backend import set_backend
        from himalaya.ridge import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        set_backend("numpy")

        rng = np.random.default_rng(42)
        n_samples, n_features = 40, 5
        self.n_electrodes = 3
        self.n_lags = 2
        n_targets = self.n_electrodes * self.n_lags

        # Known linear relationship: Y = X @ W + small noise
        W = rng.standard_normal((n_features, n_targets)).astype(np.float64)
        X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
        noise = 0.01 * rng.standard_normal((n_samples, n_targets)).astype(np.float64)
        Y = X @ W + noise

        self.X_train = X[:30]
        self.Y_train = Y[:30]
        self.X_test = X[30:]

        pipeline = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=[0.1, 1.0, 10.0]),
        )
        pipeline.fit(self.X_train, self.Y_train)
        self.pipeline = pipeline

    def test_raw_coefs_shape(self):
        raw_coefs, raw_intercept = absorb_scaler(
            self.pipeline, self.n_electrodes, self.n_lags
        )
        assert raw_coefs.shape == (self.n_electrodes, self.n_lags, self.X_train.shape[1])

    def test_raw_coefs_reproduce_predictions(self):
        """X @ raw_coefs[e, t] + raw_intercept[e, t] ≈ pipeline.predict(X)[:, e*n_lags + t]."""
        raw_coefs, raw_intercept = absorb_scaler(
            self.pipeline, self.n_electrodes, self.n_lags
        )
        pipeline_preds = self.pipeline.predict(self.X_test)  # (n_test, n_targets)

        for e in range(self.n_electrodes):
            for t in range(self.n_lags):
                manual = self.X_test @ raw_coefs[e, t] + raw_intercept[e * self.n_lags + t]
                pipeline_col = pipeline_preds[:, e * self.n_lags + t]
                np.testing.assert_allclose(manual, pipeline_col, rtol=1e-4, atol=1e-4)

    def test_raw_intercept_is_scalar_per_target(self):
        _, raw_intercept = absorb_scaler(
            self.pipeline, self.n_electrodes, self.n_lags
        )
        assert raw_intercept.shape == (self.n_electrodes * self.n_lags,)
