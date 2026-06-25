"""Tests for utils/encoding.py."""

import numpy as np
import pytest
import torch

from src.utils.encoding import (
    apply_offset,
    get_all_targets_test_set,
    get_ridge_model_constructor,
    load_encoding_results,
    load_all_subject_correlations,
    save_encoding_results,
    save_subject_results,
    _make_col_idxs,
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
        from src.utils.constants import RIDGE_ALPHAS

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

    def test_custom_ridge_cv_folds(self):
        constructor = get_ridge_model_constructor(ridge_cv_folds=3)
        model = constructor(0)
        ridge = model.named_steps["ridgecv"]
        assert ridge.cv.n_splits == 3

    def test_himalaya_memory_options_are_forwarded(self):
        constructor = get_ridge_model_constructor(
            solver_params={"n_alphas_batch": 2, "n_targets_batch": 3},
            y_in_cpu=True,
            force_cpu=True,
        )
        model = constructor(0)
        ridge = model.named_steps["ridgecv"]
        if hasattr(ridge, "solver_params"):
            assert ridge.solver_params == {
                "n_alphas_batch": 2,
                "n_targets_batch": 3,
            }
            assert ridge.Y_in_cpu is True
            assert ridge.force_cpu is True

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
            "arrays": [
                rng.standard_normal((3, 4, 5)),
                rng.standard_normal((3, 4, 5)),
            ],
            "vectors": [rng.standard_normal((12,)), rng.standard_normal((12,))],
        }
        filepath = str(tmp_path / "nested.joblib")
        save_encoding_results(results, filepath)
        loaded = load_encoding_results(filepath)
        for orig, restored in zip(results["arrays"], loaded["arrays"]):
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

    def test_valid_rows_filters_zero_offset(self):
        rng = np.random.default_rng(4)
        embs = rng.standard_normal((10, 6)).astype(np.float32)
        data = rng.standard_normal((10, 3, 2)).astype(np.float32)
        # Keep only even-indexed rows
        mask = np.array([True, False] * 5)
        X, Y = apply_offset(embs, data, offset=0, valid_rows=mask)
        assert X.shape[0] == 5
        assert Y.shape[0] == 5
        np.testing.assert_array_equal(X, embs[mask])

    def test_valid_rows_filters_after_positive_offset(self):
        rng = np.random.default_rng(5)
        embs = rng.standard_normal((10, 6)).astype(np.float32)
        data = rng.standard_normal((10, 3, 2)).astype(np.float32)
        # offset=2 trims to 8 rows; mask is applied to those 8
        mask = np.array([True, False] * 5)  # 10 elements; offset slices to [2:]→mask[2:]
        X, Y = apply_offset(embs, data, offset=2, valid_rows=mask)
        expected_mask = mask[2:]  # valid_rows[offset:]
        assert X.shape[0] == expected_mask.sum()
        assert Y.shape[0] == expected_mask.sum()
        np.testing.assert_array_equal(X, embs[2:][expected_mask])

    def test_valid_rows_filters_after_negative_offset(self):
        rng = np.random.default_rng(6)
        embs = rng.standard_normal((10, 6)).astype(np.float32)
        data = rng.standard_normal((10, 3, 2)).astype(np.float32)
        # offset=-2 trims to 8 rows; mask is applied to those 8
        mask = np.array([True, False] * 5)  # valid_rows[:offset] → mask[:-2]
        X, Y = apply_offset(embs, data, offset=-2, valid_rows=mask)
        expected_mask = mask[:-2]
        assert X.shape[0] == expected_mask.sum()
        assert Y.shape[0] == expected_mask.sum()
        np.testing.assert_array_equal(X, embs[:-2][expected_mask])

    def test_valid_rows_none_unchanged(self):
        """Passing valid_rows=None should behave identically to omitting it."""
        rng = np.random.default_rng(7)
        embs = rng.standard_normal((10, 6)).astype(np.float32)
        data = rng.standard_normal((10, 3, 2)).astype(np.float32)
        X_none, Y_none = apply_offset(embs, data, offset=1, valid_rows=None)
        X_omit, Y_omit = apply_offset(embs, data, offset=1)
        np.testing.assert_array_equal(X_none, X_omit)
        np.testing.assert_array_equal(Y_none, Y_omit)


# ---------------------------------------------------------------------------
# save_subject_results
# ---------------------------------------------------------------------------


class TestSaveSubjectResults:
    @pytest.fixture()
    def minimal_args(self):
        rng = np.random.default_rng(0)
        n_folds, n_elec, n_lags = 2, 4, 3
        corrs = rng.standard_normal((n_folds, n_elec, n_lags)).astype(np.float32)
        lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)
        return dict(
            corrs=corrs,
            tr_idxs=[np.arange(10), np.arange(10, 20)],
            test_idxs=[np.arange(10, 20), np.arange(10)],
            scalers=[None, None],
            elec_names=[f"E{i}" for i in range(n_elec)],
            offset=0,
            lags=lags,
        )

    def test_creates_expected_files_without_arb(self, tmp_path, minimal_args):
        import os

        save_subject_results(str(tmp_path), sub_id=1, **minimal_args)
        sub_dir = tmp_path / "subjects" / "sub_1"
        assert (sub_dir / "encoding_results.joblib").exists()
        assert (sub_dir / "corrs.npy").exists()
        assert not (sub_dir / "arbitrary_corrs.npy").exists()

    def test_joblib_is_slim(self, tmp_path, minimal_args):
        save_subject_results(
            str(tmp_path),
            sub_id=1,
            per_model_config={"layer": 24, "pooling": "mean"},
            **minimal_args,
        )
        loaded = load_encoding_results(
            str(tmp_path / "subjects" / "sub_1" / "encoding_results.joblib")
        )
        expected = {
            "corrs",
            "train_idxs",
            "test_idxs",
            "scalers",
            "elec_names",
            "subject_id",
            "offset",
            "lags",
            "per_model_config",
        }
        assert set(loaded) == expected
        assert "models" not in loaded
        assert "coefficients" not in loaded
        assert "intercepts" not in loaded
        assert loaded["per_model_config"] == {"layer": 24, "pooling": "mean"}

    def test_creates_arb_files_when_provided(self, tmp_path, minimal_args):
        rng = np.random.default_rng(99)
        corrs_arb = rng.standard_normal(minimal_args["corrs"].shape).astype(np.float32)
        save_subject_results(
            str(tmp_path), sub_id=2, corrs_arb=corrs_arb, **minimal_args
        )
        sub_dir = tmp_path / "subjects" / "sub_2"
        assert (sub_dir / "arbitrary_corrs.npy").exists()
        loaded = load_encoding_results(str(sub_dir / "encoding_results.joblib"))
        assert "arb_corrs" in loaded

    def test_persists_word_selection_metadata(self, tmp_path, minimal_args):
        metadata = {
            "mode": "shared_intersection",
            "subjects": [1, 2],
            "epoch_selection": np.array([0, 2, 4]),
            "analysis_selection": np.array([2, 4]),
            "offset": 1,
            "valid_rows_applied": False,
        }
        save_subject_results(
            str(tmp_path),
            sub_id=1,
            word_selection_metadata=metadata,
            **minimal_args,
        )
        loaded = load_encoding_results(
            str(tmp_path / "subjects" / "sub_1" / "encoding_results.joblib")
        )
        assert loaded["word_selection"]["mode"] == "shared_intersection"
        assert loaded["word_selection"]["subjects"] == [1, 2]
        np.testing.assert_array_equal(
            loaded["word_selection"]["epoch_selection"], np.array([0, 2, 4])
        )


# ---------------------------------------------------------------------------
# Helpers for multi-subject tests
# ---------------------------------------------------------------------------


def _make_subject_args(rng, n_elec, n_folds=2, n_lags=3):
    corrs = rng.standard_normal((n_folds, n_elec, n_lags)).astype(np.float32)
    lags = np.linspace(-2.0, 2.0, n_lags, endpoint=False)
    return dict(
        corrs=corrs,
        tr_idxs=[np.arange(10)] * n_folds,
        test_idxs=[np.arange(10)] * n_folds,
        scalers=[None] * n_folds,
        elec_names=[f"S{n_elec}E{i}" for i in range(n_elec)],
        offset=0,
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


class TestBuildPerFoldBaselines:
    """Tests for the fresh-training build_per_fold_baselines."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(77)
        n_words, n_embed, n_targets = 60, 6, 12
        X = rng.standard_normal((n_words, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n_words, n_targets)).astype(np.float32)
        from sklearn.model_selection import KFold

        test_idxs = [t for _, t in KFold(2, shuffle=False).split(X)]
        return X, Y, test_idxs, n_words, n_embed, n_targets

    def test_output_length_matches_n_folds(self, synthetic_data):
        X, Y, test_idxs, *_ = synthetic_data
        baselines = build_per_fold_baselines(X, Y, test_idxs)
        assert len(baselines) == len(test_idxs)

    def test_folds_to_build_leaves_other_entries_empty(self, synthetic_data):
        X, Y, test_idxs, *_ = synthetic_data
        baselines = build_per_fold_baselines(X, Y, test_idxs, folds_to_build={1})
        assert len(baselines) == len(test_idxs)
        assert baselines[0] is None
        assert baselines[1] is not None

    def test_shapes_are_correct(self, synthetic_data):
        X, Y, test_idxs, n_words, n_embed, n_targets = synthetic_data
        baselines = build_per_fold_baselines(X, Y, test_idxs)
        fold_size = n_words // 2
        b = baselines[0]
        assert b["Y_train"].shape == (n_words - fold_size, n_targets)
        assert b["Y_test"].shape == (fold_size, n_targets)
        assert b["Y_hat_full_train"].shape == (n_words - fold_size, n_targets)
        assert b["Y_hat_full_test"].shape == (fold_size, n_targets)
        assert b["beta_full_raw"].shape == (n_embed, n_targets)

    def test_train_test_idx_partition_all_words(self, synthetic_data):
        X, Y, test_idxs, n_words, *_ = synthetic_data
        baselines = build_per_fold_baselines(X, Y, test_idxs)
        for b in baselines:
            combined = np.sort(np.concatenate([b["train_idx"], b["test_idx"]]))
            np.testing.assert_array_equal(combined, np.arange(n_words))
