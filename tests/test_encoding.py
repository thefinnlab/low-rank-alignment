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
    load_all_subject_correlations,
    load_all_subject_coefficients,
    save_encoding_results,
    save_subject_results,
    _make_col_idxs,
    build_enc_subjects_info,
    build_per_fold_baselines,
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
        assert len(ridge.alphas) == 13

    def test_default_alphas_match_constant(self):
        from utils.constants import RIDGE_ALPHAS
        constructor = get_ridge_model_constructor()
        model = constructor(0)
        ridge = model.named_steps["ridgecv"]
        np.testing.assert_array_equal(ridge.alphas, RIDGE_ALPHAS)

    def test_custom_alphas(self):
        custom = np.array([1.0, 10.0, 100.0])
        constructor = get_ridge_model_constructor(alphas=custom)
        model = constructor(0)
        ridge = model.named_steps["ridgecv"]
        np.testing.assert_array_equal(ridge.alphas, custom)

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


# ---------------------------------------------------------------------------
# Helpers for multi-subject tests
# ---------------------------------------------------------------------------

def _make_subject_args(rng, n_elec, n_folds=2, n_lags=3, n_feat=5):
    corrs = rng.standard_normal((n_folds, n_elec, n_lags)).astype(np.float32)
    coefficients = [rng.standard_normal((n_elec, n_lags, n_feat)) for _ in range(n_folds)]
    intercepts = [rng.standard_normal((n_elec * n_lags,)) for _ in range(n_folds)]
    lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)
    return dict(
        corrs=corrs,
        coefficients=coefficients,
        intercepts=intercepts,
        models=[None] * n_folds,
        tr_idxs=[np.arange(10)] * n_folds,
        test_idxs=[np.arange(10)] * n_folds,
        scalers=[None] * n_folds,
        elec_names=[f"S{n_elec}E{i}" for i in range(n_elec)],
        offset=0,
        layer_int=0,
        lags=lags,
    )


@pytest.fixture()
def two_subject_dir(tmp_path):
    """Write two subjects (3 and 2 electrodes) into tmp_path."""
    rng = np.random.default_rng(42)
    args1 = _make_subject_args(rng, n_elec=3)
    args2 = _make_subject_args(rng, n_elec=2)
    save_subject_results(str(tmp_path), sub_id=1, **args1)
    save_subject_results(str(tmp_path), sub_id=2, **args2)
    return tmp_path, args1, args2


# ---------------------------------------------------------------------------
# TestLoadAllSubjectCorrelations
# ---------------------------------------------------------------------------

class TestLoadAllSubjectCorrelations:
    def test_shape(self, two_subject_dir):
        output_dir, args1, args2 = two_subject_dir
        n_folds = args1["corrs"].shape[0]
        n_lags = args1["corrs"].shape[2]
        corrs_all, _ = load_all_subject_correlations(str(output_dir))
        assert corrs_all.shape == (n_folds, 5, n_lags)

    def test_metadata_order(self, two_subject_dir):
        output_dir, args1, args2 = two_subject_dir
        _, meta = load_all_subject_correlations(str(output_dir))
        assert list(meta["subject"]) == [1, 1, 1, 2, 2]
        assert list(meta["electrode"]) == args1["elec_names"] + args2["elec_names"]

    def test_values(self, two_subject_dir):
        output_dir, args1, _ = two_subject_dir
        corrs_all, _ = load_all_subject_correlations(str(output_dir))
        # First 3 electrodes should be sub_1's corrs: corrs_all[:, :3, :] == args1["corrs"]
        np.testing.assert_array_equal(corrs_all[:, :3, :], args1["corrs"])

    def test_no_subjects_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_subject_correlations(str(tmp_path))

    def test_subject_filter(self, two_subject_dir):
        output_dir, args1, _ = two_subject_dir
        corrs_all, meta = load_all_subject_correlations(str(output_dir), subjects=[1])
        assert corrs_all.shape[1] == 3
        assert list(meta["subject"]) == [1, 1, 1]

    def test_subject_filter_missing_raises(self, two_subject_dir):
        output_dir, _, _ = two_subject_dir
        with pytest.raises(FileNotFoundError):
            load_all_subject_correlations(str(output_dir), subjects=[99])


# ---------------------------------------------------------------------------
# TestLoadAllSubjectCoefficients
# ---------------------------------------------------------------------------

class TestLoadAllSubjectCoefficients:
    def test_shape(self, two_subject_dir):
        output_dir, args1, _ = two_subject_dir
        n_folds = len(args1["coefficients"])
        n_lags = args1["coefficients"][0].shape[1]
        n_feat = args1["coefficients"][0].shape[2]
        coefs_all, intercepts_all, _ = load_all_subject_coefficients(str(output_dir))
        assert coefs_all.shape == (n_folds, 5, n_lags, n_feat)
        assert len(intercepts_all) == n_folds

    def test_metadata_order(self, two_subject_dir):
        output_dir, args1, args2 = two_subject_dir
        _, _, meta = load_all_subject_coefficients(str(output_dir))
        assert list(meta["subject"]) == [1, 1, 1, 2, 2]
        assert list(meta["electrode"]) == args1["elec_names"] + args2["elec_names"]

    def test_values(self, two_subject_dir):
        output_dir, args1, args2 = two_subject_dir
        coefs_all, intercepts_all, _ = load_all_subject_coefficients(str(output_dir))
        # First 3 electrodes of each fold should match sub_1's coefficients
        for fold_idx in range(coefs_all.shape[0]):
            np.testing.assert_array_equal(coefs_all[fold_idx, :3], args1["coefficients"][fold_idx])
        # First n_elec1*n_lags entries of intercepts should match sub_1's intercepts
        n_elec1_lags = len(args1["intercepts"][0])
        for fold_idx, fold_arr in enumerate(intercepts_all):
            np.testing.assert_array_equal(fold_arr[:n_elec1_lags], args1["intercepts"][fold_idx])

    def test_no_subjects_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_subject_coefficients(str(tmp_path))

    def test_subject_filter(self, two_subject_dir):
        output_dir, _, args2 = two_subject_dir
        coefs_all, intercepts_all, meta = load_all_subject_coefficients(str(output_dir), subjects=[2])
        assert len(meta) == 2
        assert list(meta["subject"]) == [2, 2]
        assert coefs_all.shape[1] == 2
        for fold_idx in range(coefs_all.shape[0]):
            np.testing.assert_array_equal(coefs_all[fold_idx], args2["coefficients"][fold_idx])
        for fold_idx, fold_arr in enumerate(intercepts_all):
            np.testing.assert_array_equal(fold_arr, args2["intercepts"][fold_idx])


# ---------------------------------------------------------------------------
# TestMakeColIdxs
# ---------------------------------------------------------------------------

class TestMakeColIdxs:
    def _make_enc_results(self, elec_names, n_lags=4):
        lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)
        return {"elec_names": elec_names, "lags": lags}

    def test_single_electrode_correct_indices(self):
        enc = self._make_enc_results(["A", "B", "C"], n_lags=3)
        idxs = _make_col_idxs(enc, ["B"])
        np.testing.assert_array_equal(idxs, [3, 4, 5])

    def test_multiple_electrodes_in_order(self):
        enc = self._make_enc_results(["A", "B", "C"], n_lags=2)
        idxs = _make_col_idxs(enc, ["C", "A"])
        np.testing.assert_array_equal(idxs, [4, 5, 0, 1])

    def test_all_electrodes_identity(self):
        enc = self._make_enc_results(["X", "Y"], n_lags=3)
        idxs = _make_col_idxs(enc, ["X", "Y"])
        np.testing.assert_array_equal(idxs, [0, 1, 2, 3, 4, 5])

    def test_missing_electrode_raises(self):
        enc = self._make_enc_results(["A", "B"], n_lags=2)
        with pytest.raises(ValueError, match="not found"):
            _make_col_idxs(enc, ["A", "Z"])


# ---------------------------------------------------------------------------
# TestBuildPerFoldBaselines
# ---------------------------------------------------------------------------

def _make_fitted_enc_results(rng, n_words, n_embed, elec_names, n_lags=3, n_folds=2):
    """Build a minimal enc_results dict with real fitted pipelines and scalers."""
    from himalaya.backend import set_backend
    from himalaya.ridge import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    set_backend("numpy")

    n_elec = len(elec_names)
    n_targets = n_elec * n_lags
    Y = rng.standard_normal((n_words, n_targets)).astype(np.float32)
    X = rng.standard_normal((n_words, n_embed)).astype(np.float32)
    lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

    fold_size = n_words // n_folds
    models, scalers, test_idxs, train_idxs = [], [], [], []
    for f in range(n_folds):
        test_idx = np.arange(f * fold_size, (f + 1) * fold_size)
        train_idx = np.concatenate([np.arange(0, f * fold_size), np.arange((f + 1) * fold_size, n_words)])
        scaler = StandardScaler().fit(Y[train_idx])
        Y_train_s = scaler.transform(Y[train_idx])
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1.0]))
        model.fit(X[train_idx], Y_train_s)
        models.append(model)
        scalers.append(scaler)
        test_idxs.append(test_idx)
        train_idxs.append(train_idx)

    return {
        "models": models,
        "scalers": scalers,
        "test_idxs": test_idxs,
        "train_idxs": train_idxs,
        "elec_names": elec_names,
        "lags": lags,
    }, X, Y


class TestBuildPerFoldBaselines:
    @pytest.fixture
    def synthetic_setup(self):
        rng = np.random.default_rng(77)
        n_words, n_embed = 60, 6
        elec_names = ["E0", "E1", "E2", "E3"]
        n_lags = 3
        enc_results, X, Y_full = _make_fitted_enc_results(
            rng, n_words, n_embed, elec_names, n_lags=n_lags, n_folds=2
        )
        return enc_results, X, Y_full, n_words, n_embed, n_lags, elec_names

    def _make_enc_subjects_info(self, enc_results, target_elecs):
        col_idxs = _make_col_idxs(enc_results, target_elecs)
        n_lags = len(enc_results["lags"])
        n_target_cols = len(target_elecs) * n_lags
        return [{
            "models": enc_results["models"],
            "scalers": enc_results["scalers"],
            "col_idxs": col_idxs,
            "y_col_start": 0,
            "y_col_end": n_target_cols,
        }]

    def test_output_length_matches_n_folds(self, synthetic_setup):
        enc_results, X, Y_full, n_words, n_embed, n_lags, elec_names = synthetic_setup
        target_elecs = elec_names[:2]
        n_targets = len(target_elecs) * n_lags
        Y_target = Y_full[:, :n_targets]
        enc_info = self._make_enc_subjects_info(enc_results, target_elecs)
        baselines = build_per_fold_baselines(X, Y_target, enc_results["test_idxs"], enc_info)
        assert len(baselines) == len(enc_results["test_idxs"])

    def test_shapes_are_correct(self, synthetic_setup):
        enc_results, X, Y_full, n_words, n_embed, n_lags, elec_names = synthetic_setup
        target_elecs = elec_names[:2]
        n_targets = len(target_elecs) * n_lags
        Y_target = Y_full[:, :n_targets]
        enc_info = self._make_enc_subjects_info(enc_results, target_elecs)
        baselines = build_per_fold_baselines(X, Y_target, enc_results["test_idxs"], enc_info)

        fold_size = n_words // 2
        b = baselines[0]
        assert b["Y_train"].shape == (n_words - fold_size, n_targets)
        assert b["Y_test"].shape == (fold_size, n_targets)
        assert b["Y_hat_full_train"].shape == (n_words - fold_size, n_targets)
        assert b["Y_hat_full_test"].shape == (fold_size, n_targets)
        assert b["beta_full_raw"].shape == (n_embed, n_targets)

    def test_train_test_idx_partition_all_words(self, synthetic_setup):
        enc_results, X, Y_full, n_words, n_embed, n_lags, elec_names = synthetic_setup
        target_elecs = elec_names
        n_targets = len(target_elecs) * n_lags
        enc_info = self._make_enc_subjects_info(enc_results, target_elecs)
        baselines = build_per_fold_baselines(X, Y_full, enc_results["test_idxs"], enc_info)

        for b in baselines:
            combined = np.sort(np.concatenate([b["train_idx"], b["test_idx"]]))
            np.testing.assert_array_equal(combined, np.arange(n_words))

    def test_build_enc_subjects_info_y_col_cursor(self, tmp_path):
        """build_enc_subjects_info should accumulate y_col_start/end across subjects."""
        rng = np.random.default_rng(5)
        n_words, n_embed = 40, 4
        n_lags = 2

        # Save two subjects' enc_results to tmp_path
        for sub_id, elec_names in [(1, ["A", "B"]), (2, ["C", "D", "E"])]:
            enc_results, _, _ = _make_fitted_enc_results(
                rng, n_words, n_embed, elec_names, n_lags=n_lags
            )
            import joblib, os
            sub_dir = os.path.join(str(tmp_path), "subjects", f"sub_{sub_id}")
            os.makedirs(sub_dir, exist_ok=True)
            joblib.dump(enc_results, os.path.join(sub_dir, "encoding_results.joblib"))

        target_elecs_per_sub = {1: ["A", "B"], 2: ["C", "D", "E"]}
        info_list = build_enc_subjects_info(str(tmp_path), [1, 2], target_elecs_per_sub)

        assert len(info_list) == 2
        assert info_list[0]["y_col_start"] == 0
        assert info_list[0]["y_col_end"] == 2 * n_lags   # 2 elecs × n_lags
        assert info_list[1]["y_col_start"] == 2 * n_lags
        assert info_list[1]["y_col_end"] == (2 + 3) * n_lags  # 5 elecs total
