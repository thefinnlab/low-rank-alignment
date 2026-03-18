"""Tests for utils/encoding.py."""
import numpy as np
import pytest
import torch

from utils.encoding import (
    absorb_scaler,
    apply_offset,
    get_all_targets_test_set,
    get_ridge_model_constructor,
    load_encoding_results,
    save_encoding_results,
    save_subject_results,
)


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


# ---------------------------------------------------------------------------
# get_ridge_model_constructor
# ---------------------------------------------------------------------------

class TestGetRidgeModelConstructor:
    def test_returns_callable(self):
        constructor = get_ridge_model_constructor()
        assert callable(constructor)

    def test_callable_produces_pipeline(self):
        from sklearn.pipeline import Pipeline
        constructor = get_ridge_model_constructor()
        model = constructor(0)
        assert isinstance(model, Pipeline)

    def test_has_standardscaler_step(self):
        constructor = get_ridge_model_constructor()
        model = constructor(0)
        assert "standardscaler" in model.named_steps

    def test_has_ridgecv_step(self):
        constructor = get_ridge_model_constructor()
        model = constructor(0)
        assert "ridgecv" in model.named_steps
        ridge = model.named_steps["ridgecv"]
        assert len(ridge.alphas) == 10

    def test_each_call_returns_new_model(self):
        constructor = get_ridge_model_constructor()
        assert constructor(0) is not constructor(1)


# ---------------------------------------------------------------------------
# save_encoding_results / load_encoding_results
# ---------------------------------------------------------------------------

class TestSaveLoadEncodingResults:
    def test_round_trip_simple_dict(self, tmp_path):
        results = {"corrs": np.array([1.0, 2.0]), "subject_id": 3, "offset": -1}
        filepath = str(tmp_path / "results.joblib")
        save_encoding_results(results, filepath)
        loaded = load_encoding_results(filepath)
        np.testing.assert_array_equal(loaded["corrs"], results["corrs"])
        assert loaded["subject_id"] == results["subject_id"]
        assert loaded["offset"] == results["offset"]

    def test_round_trip_nested_arrays(self, tmp_path):
        rng = np.random.default_rng(42)
        results = {
            "coefficients": [rng.standard_normal((3, 4, 5)), rng.standard_normal((3, 4, 5))],
            "intercepts": [rng.standard_normal((12,)), rng.standard_normal((12,))],
        }
        filepath = str(tmp_path / "nested.joblib")
        save_encoding_results(results, filepath)
        loaded = load_encoding_results(filepath)
        for orig, restored in zip(results["coefficients"], loaded["coefficients"]):
            np.testing.assert_array_equal(orig, restored)

    def test_creates_file_on_disk(self, tmp_path):
        import os
        filepath = str(tmp_path / "output.joblib")
        save_encoding_results({"x": 1}, filepath)
        assert os.path.exists(filepath)


# ---------------------------------------------------------------------------
# apply_offset
# ---------------------------------------------------------------------------

class TestApplyOffset:
    def test_zero_offset_unchanged_shape(self):
        rng = np.random.default_rng(0)
        embs = rng.standard_normal((20, 10)).astype(np.float32)
        data = rng.standard_normal((20, 3, 4)).astype(np.float32)
        X, Y = apply_offset(embs, data, offset=0)
        assert X.shape == (20, 10)
        assert Y.shape == (20, 12)

    def test_positive_offset_trims_correctly(self):
        rng = np.random.default_rng(1)
        embs = rng.standard_normal((20, 10))
        data = rng.standard_normal((20, 3, 4))
        X, Y = apply_offset(embs, data, offset=2)
        assert X.shape[0] == 18  # embs[2:]
        assert Y.shape[0] == 18  # data[:-2]
        assert X.shape[0] == Y.shape[0]
        # X rows align to embs[2:], Y rows align to data[:-2]
        np.testing.assert_array_equal(X, embs[2:].astype(np.float32))
        np.testing.assert_array_equal(Y, data[:-2].reshape(18, -1).astype(np.float32))

    def test_negative_offset_trims_correctly(self):
        rng = np.random.default_rng(2)
        embs = rng.standard_normal((20, 10))
        data = rng.standard_normal((20, 3, 4))
        X, Y = apply_offset(embs, data, offset=-2)
        assert X.shape[0] == 18  # embs[:-2]
        assert Y.shape[0] == 18  # data[2:]
        assert X.shape[0] == Y.shape[0]
        # X rows align to embs[:-2], Y rows align to data[2:]
        np.testing.assert_array_equal(X, embs[:-2].astype(np.float32))
        np.testing.assert_array_equal(Y, data[2:].reshape(18, -1).astype(np.float32))

    def test_output_is_float32(self):
        rng = np.random.default_rng(3)
        embs = rng.standard_normal((20, 10)).astype(np.float64)
        data = rng.standard_normal((20, 3, 4)).astype(np.float64)
        X, Y = apply_offset(embs, data, offset=1)
        assert X.dtype == np.float32
        assert Y.dtype == np.float32


# ---------------------------------------------------------------------------
# save_subject_results
# ---------------------------------------------------------------------------

class TestSaveSubjectResults:
    @pytest.fixture()
    def minimal_args(self):
        rng = np.random.default_rng(0)
        n_folds, n_elec, n_lags, n_feat = 2, 4, 3, 5
        corrs = rng.standard_normal((n_folds, n_elec, n_lags)).astype(np.float32)
        coefficients = [rng.standard_normal((n_elec, n_lags, n_feat)) for _ in range(n_folds)]
        intercepts = [rng.standard_normal((n_elec * n_lags,)) for _ in range(n_folds)]
        lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)
        return dict(
            corrs=corrs,
            coefficients=coefficients,
            intercepts=intercepts,
            models=[None, None],
            tr_idxs=[np.arange(10), np.arange(10, 20)],
            test_idxs=[np.arange(10, 20), np.arange(10)],
            scalers=[None, None],
            elec_names=[f"E{i}" for i in range(n_elec)],
            offset=0,
            layer_int=24,
            lags=lags,
        )

    def test_creates_expected_files_without_arb(self, tmp_path, minimal_args):
        import os
        save_subject_results(str(tmp_path), sub_id=1, **minimal_args)
        sub_dir = tmp_path / "subjects" / "sub_1"
        assert (sub_dir / "encoding_results.joblib").exists()
        assert (sub_dir / "corrs.npy").exists()
        assert not (sub_dir / "arbitrary_corrs.npy").exists()

    def test_creates_arb_files_when_provided(self, tmp_path, minimal_args):
        rng = np.random.default_rng(99)
        corrs_arb = rng.standard_normal(minimal_args["corrs"].shape).astype(np.float32)
        save_subject_results(str(tmp_path), sub_id=2, corrs_arb=corrs_arb, **minimal_args)
        sub_dir = tmp_path / "subjects" / "sub_2"
        assert (sub_dir / "arbitrary_corrs.npy").exists()
