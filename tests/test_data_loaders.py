"""Tests for utils/data_loaders.py."""

import io
import types

import numpy as np
import pandas as pd
import pytest

from src.utils.data_loaders import (
    build_aligned_epoch_data,
    build_word_selection_metadata,
    build_word_epochs,
    compute_epoch_cache_indices,
    compute_shared_word_selection,
    filter_epoch_cache_to_selection,
    get_precomputed_epochs_path,
    get_arbitrary_embeddings,
    load_data_from_cache,
    load_precomputed_epochs,
    load_sig_electrodes,
    read_electrode_file,
    read_subject_mapping,
    save_precomputed_epochs,
    selection_after_offset,
)
from src.core.registry import DatasetContext


# ---------------------------------------------------------------------------
# compute_epoch_cache_indices
# ---------------------------------------------------------------------------


class TestComputeEpochCacheIndices:
    class FakeRaw:
        def __init__(self, sfreq=32.0, n_times=320):
            self.info = {"sfreq": sfreq}
            self.n_times = n_times

        def time_as_index(self, times):
            return np.floor(np.asarray(times) * self.info["sfreq"]).astype(int)

    def test_drops_duplicate_onset_samples_keep_first(self):
        raw = self.FakeRaw(sfreq=32.0, n_times=320)
        df_word = pd.DataFrame(
            {
                "start": [2.0, 2.0, 3.0, 3.01],
                "word": ["first", "duplicate", "third", "fourth"],
            }
        )

        selection, epoch_starts, n_lags = compute_epoch_cache_indices(
            raw, df_word, tmin=-1.0, tmax=1.0, target_sfreq=32.0
        )

        np.testing.assert_array_equal(selection, np.array([0, 2]))
        np.testing.assert_array_equal(epoch_starts, np.array([32, 64]))
        assert n_lags == 64

    def test_filters_nonfinite_and_out_of_bounds_windows(self):
        raw = self.FakeRaw(sfreq=32.0, n_times=160)
        df_word = pd.DataFrame(
            {
                "start": [0.25, 1.5, np.nan, 4.75],
                "word": ["too_early", "kept", "nan", "too_late"],
            }
        )

        selection, epoch_starts, n_lags = compute_epoch_cache_indices(
            raw, df_word, tmin=-1.0, tmax=1.0, target_sfreq=32.0
        )

        np.testing.assert_array_equal(selection, np.array([1]))
        np.testing.assert_array_equal(epoch_starts, np.array([16]))
        assert n_lags == 64

    def test_uses_canonical_start_when_estimated_indices_are_present(self):
        raw = self.FakeRaw(sfreq=32.0, n_times=320)
        df_word = pd.DataFrame(
            {
                "word_idx": [10, 11, 12],
                "start": [1.0, 2.0, 2.001],
                "est_idx": [2048, 2048, 2048],
                "est_idx_sfreq": [2048.0, 2048.0, 2048.0],
                "word": ["first", "second", "duplicate_at_32hz"],
            }
        )

        selection, epoch_starts, n_lags = compute_epoch_cache_indices(
            raw, df_word, tmin=-1.0, tmax=1.0, target_sfreq=32.0
        )

        np.testing.assert_array_equal(selection, np.array([10, 11]))
        np.testing.assert_array_equal(epoch_starts, np.array([0, 32]))
        assert n_lags == 64


# ---------------------------------------------------------------------------
# read_subject_mapping
# ---------------------------------------------------------------------------


class TestReadSubjectMapping:
    def test_basic_happy_path(self):
        f = io.StringIO("nyu_id\tparticipant_id\n661\tsub-05\n")
        result = read_subject_mapping(f)
        assert result == {661: 5}

    def test_multiple_rows(self):
        f = io.StringIO("nyu_id\tparticipant_id\n661\tsub-05\n717\tsub-12\n")
        result = read_subject_mapping(f)
        assert result == {661: 5, 717: 12}

    def test_custom_delimiter_comma(self):
        f = io.StringIO("nyu_id,participant_id\n100,sub-03\n200,sub-07\n")
        result = read_subject_mapping(f, delimiter=",")
        assert result == {100: 3, 200: 7}

    def test_invalid_participant_id_no_number(self):
        f = io.StringIO("nyu_id\tparticipant_id\n661\tsub-\n")
        with pytest.raises(ValueError):
            read_subject_mapping(f)

    def test_invalid_participant_id_no_prefix(self):
        f = io.StringIO("nyu_id\tparticipant_id\n661\tnosub\n")
        with pytest.raises(ValueError):
            read_subject_mapping(f)

    def test_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.tsv"
        with pytest.raises(FileNotFoundError):
            read_subject_mapping(str(missing))

    def test_leading_zero_stripped(self):
        # "sub-05" -> 5, not "05"
        f = io.StringIO("nyu_id\tparticipant_id\n1\tsub-05\n")
        result = read_subject_mapping(f)
        assert result[1] == 5
        assert isinstance(result[1], int)


# ---------------------------------------------------------------------------
# read_electrode_file
# ---------------------------------------------------------------------------


class TestReadElectrodeFile:
    def test_single_subject_multiple_electrodes(self):
        f = io.StringIO("subject,elec\n1,A1\n1,A2\n")
        result = read_electrode_file(f)
        assert result == {1: ["A1", "A2"]}

    def test_multiple_subjects(self):
        f = io.StringIO("subject,elec\n1,A1\n1,A2\n2,C1\n")
        result = read_electrode_file(f)
        assert result == {1: ["A1", "A2"], 2: ["C1"]}

    def test_with_subject_mapping(self):
        f = io.StringIO("subject,elec\n798,A1\n798,A2\n799,B1\n")
        mapping = {798: 9, 799: 10}
        result = read_electrode_file(f, subject_mapping=mapping)
        assert result == {9: ["A1", "A2"], 10: ["B1"]}

    def test_preserves_insertion_order(self):
        f = io.StringIO("subject,elec\n1,Z1\n1,A1\n1,M1\n")
        result = read_electrode_file(f)
        assert result[1] == ["Z1", "A1", "M1"]

    def test_missing_elec_column_raises(self):
        f = io.StringIO("subject,electrode\n1,A1\n")
        with pytest.raises((KeyError, AttributeError)):
            read_electrode_file(f)

    def test_missing_subject_column_raises(self):
        f = io.StringIO("sub,elec\n1,A1\n")
        with pytest.raises((KeyError, AttributeError)):
            read_electrode_file(f)


# ---------------------------------------------------------------------------
# get_arbitrary_embeddings
# ---------------------------------------------------------------------------


class TestGetArbitraryEmbeddings:
    """
    get_arbitrary_embeddings(X, epochs, df_word) where X is an ndarray of shape
    (n_words, n_features). The function randomly assigns one of the n_words rows
    of X (without replacement) to each unique word type, then replicates that row
    for all occurrences of that word.
    """

    def _make_epochs(self, selection):
        """Minimal mock epochs object with .selection attribute."""
        ep = types.SimpleNamespace(selection=np.array(selection))
        return ep

    def _make_df_word(self, words):
        return pd.DataFrame({"word": words})

    def test_output_shape_matches_input(self):
        rng = np.random.default_rng(42)
        n_words, n_features = 20, 8
        words = [f"word{i % 5}" for i in range(n_words)]  # 5 unique words
        X = rng.standard_normal((n_words, n_features)).astype(np.float32)
        epochs = self._make_epochs(list(range(n_words)))
        df_word = self._make_df_word(words)

        X_arb = get_arbitrary_embeddings(X, epochs, df_word)
        assert X_arb.shape == X.shape

    def test_same_word_same_embedding(self):
        rng = np.random.default_rng(0)
        n_words, n_features = 12, 6
        words = ["cat", "dog"] * (n_words // 2)  # 2 unique words
        X = rng.standard_normal((n_words, n_features)).astype(np.float32)
        epochs = self._make_epochs(list(range(n_words)))
        df_word = self._make_df_word(words)

        X_arb = get_arbitrary_embeddings(X, epochs, df_word)
        cat_rows = [i for i, w in enumerate(words) if w == "cat"]
        for r in cat_rows[1:]:
            np.testing.assert_array_equal(X_arb[cat_rows[0]], X_arb[r])

    def test_different_words_different_embeddings(self):
        rng = np.random.default_rng(1)
        n_words, n_features = 5, 8  # all 5 words are unique
        words = [f"word{i}" for i in range(n_words)]
        X = rng.standard_normal((n_words, n_features)).astype(np.float32)
        epochs = self._make_epochs(list(range(n_words)))
        df_word = self._make_df_word(words)

        X_arb = get_arbitrary_embeddings(X, epochs, df_word)
        # All rows should be distinct (sampled without replacement from X rows)
        for i in range(n_words):
            for j in range(i + 1, n_words):
                assert not np.array_equal(
                    X_arb[i], X_arb[j]
                ), f"Rows {i} and {j} are identical but words differ"

    def test_repeated_words_more_than_unique(self):
        rng = np.random.default_rng(7)
        n_words, n_features = 20, 5
        words = ["alpha", "beta", "gamma"] * 6 + ["alpha", "beta"]  # 3 unique
        X = rng.standard_normal((n_words, n_features)).astype(np.float32)
        epochs = self._make_epochs(list(range(n_words)))
        df_word = self._make_df_word(words)

        X_arb = get_arbitrary_embeddings(X, epochs, df_word)
        assert X_arb.shape == X.shape

        for word in ["alpha", "beta", "gamma"]:
            idxs = [i for i, w in enumerate(words) if w == word]
            for idx in idxs[1:]:
                np.testing.assert_array_equal(X_arb[idxs[0]], X_arb[idx])

    def test_seed_reproducibility(self):
        rng = np.random.default_rng(42)
        n_words, n_features = 20, 8
        words = [f"word{i % 5}" for i in range(n_words)]
        X = rng.standard_normal((n_words, n_features)).astype(np.float32)
        epochs = self._make_epochs(list(range(n_words)))
        df_word = self._make_df_word(words)

        X_arb1 = get_arbitrary_embeddings(X, epochs, df_word, seed=123)
        X_arb2 = get_arbitrary_embeddings(X, epochs, df_word, seed=123)
        np.testing.assert_array_equal(X_arb1, X_arb2)

    def test_different_seeds_different_results(self):
        rng = np.random.default_rng(42)
        n_words, n_features = 20, 8
        words = [f"word{i % 5}" for i in range(n_words)]
        X = rng.standard_normal((n_words, n_features)).astype(np.float32)
        epochs = self._make_epochs(list(range(n_words)))
        df_word = self._make_df_word(words)

        X_arb1 = get_arbitrary_embeddings(X, epochs, df_word, seed=123)
        X_arb2 = get_arbitrary_embeddings(X, epochs, df_word, seed=456)
        assert not np.array_equal(X_arb1, X_arb2)


# ---------------------------------------------------------------------------
# build_word_epochs
# ---------------------------------------------------------------------------


class TestBuildWordEpochs:
    @pytest.fixture
    def synthetic_raw(self):
        mne = pytest.importorskip("mne")
        sfreq = 512.0
        n_seconds = 10
        data = np.random.default_rng(0).standard_normal((1, int(sfreq * n_seconds)))
        info = mne.create_info(["CH1"], sfreq=sfreq, ch_types=["ecog"])
        return mne.io.RawArray(data, info, verbose=False)

    @pytest.fixture
    def df_word_simple(self):
        return pd.DataFrame(
            {"word": ["a", "b", "c"], "start": [1.0, 3.0, 5.0], "end": [1.1, 3.1, 5.1]}
        )

    def test_returns_tuple_of_two(self, synthetic_raw, df_word_simple):
        result = build_word_epochs(synthetic_raw, df_word_simple)
        assert len(result) == 2

    def test_data_shape(self, synthetic_raw, df_word_simple):
        epochs, data = build_word_epochs(synthetic_raw, df_word_simple)
        n_words_kept, n_elec, n_lags = data.shape
        assert n_elec == 1
        assert n_lags > 0

    def test_resampled_sfreq(self, synthetic_raw, df_word_simple):
        epochs, _ = build_word_epochs(synthetic_raw, df_word_simple, target_sfreq=32.0)
        assert int(epochs.info["sfreq"]) == 32


class TestBuildAlignedEpochData:
    def test_preserves_word_relative_alignment_across_global_grid_phases(self):
        mne = pytest.importorskip("mne")
        sfreq = 512.0
        data = np.zeros((1, int(8 * sfreq)))
        onset_samples = np.array([1536, 2562])
        data[0, onset_samples] = 1.0
        raw = mne.io.RawArray(
            data,
            mne.create_info(["A1"], sfreq=sfreq, ch_types=["seeg"]),
            verbose=False,
        )
        df_word = pd.DataFrame(
            {
                "word_idx": [10, 11],
                "start": onset_samples / sfreq,
                "word": ["first", "second"],
            }
        )

        epoch_data, selection = build_aligned_epoch_data(raw, df_word)

        assert epoch_data.shape == (2, 1, 128)
        np.testing.assert_array_equal(selection, np.array([10, 11]))
        np.testing.assert_array_equal(
            np.argmax(np.abs(epoch_data[:, 0]), axis=-1), np.array([64, 64])
        )

    def test_uses_canonical_start_and_filters_duplicates_and_bounds(self):
        mne = pytest.importorskip("mne")
        raw = mne.io.RawArray(
            np.zeros((1, 512 * 6)),
            mne.create_info(["A1"], sfreq=512.0, ch_types=["seeg"]),
            verbose=False,
        )
        df_word = pd.DataFrame(
            {
                "word_idx": [4, 5, 6, 7],
                "start": [0.25, 2.0, 2.0005, np.nan],
                # Dataset-specific provenance must not override canonical seconds.
                "est_idx": [4096, 512, 8192, 4096],
                "est_idx_sfreq": [2048.0] * 4,
            }
        )

        epoch_data, selection = build_aligned_epoch_data(
            raw, df_word, tmin=-1.0, tmax=1.0
        )

        assert epoch_data.shape == (1, 1, 64)
        np.testing.assert_array_equal(selection, np.array([5]))


class TestDatasetAwareLoaders:
    def test_aligned_epoch_cache_round_trip_and_electrode_selection(self, tmp_path):
        epoch_data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        save_precomputed_epochs(
            1,
            str(tmp_path),
            epoch_data,
            np.array([3, 7]),
            ["A1", "A2", "A3"],
            sfreq=32.0,
            tmin=-1.0,
            tmax=-0.875,
        )

        cache = load_precomputed_epochs(1, str(tmp_path))
        path = get_precomputed_epochs_path(1, str(tmp_path))

        assert path.endswith(".h5")
        assert hasattr(cache["epoch_data"], "read")
        np.testing.assert_array_equal(np.asarray(cache["epoch_data"]), epoch_data)
        assert cache["sfreq"] == 32.0
        assert cache["tmin"] == -1.0
        assert cache["tmax"] == -0.875
        from src.utils.data_loaders import epochs_from_cache

        np.testing.assert_array_equal(
            epochs_from_cache(cache, ["A3", "A1"]), epoch_data[:, [2, 0], :]
        )

    def test_legacy_npz_cache_is_ignored(self, tmp_path):
        path = get_precomputed_epochs_path(1, str(tmp_path))[:-3] + ".npz"
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            raw_data=np.zeros((1, 32), dtype=np.float32),
            selection=np.array([0]),
            epoch_starts=np.array([0]),
            n_lags=np.array(4),
            ch_names=np.array(["A1"]),
        )

        with pytest.raises(FileNotFoundError, match="Precomputed HDF5 epochs not found"):
            load_precomputed_epochs(1, str(tmp_path))

    def test_hdf5_epoch_cache_filters_shared_selection_before_electrodes(self, tmp_path):
        epoch_data = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
        save_precomputed_epochs(
            1,
            str(tmp_path),
            epoch_data,
            np.array([4, 2, 8, 6]),
            ["A1", "A2", "A3"],
        )

        cache = load_precomputed_epochs(1, str(tmp_path))
        filtered = filter_epoch_cache_to_selection(cache, np.array([8, 4]))

        from src.utils.data_loaders import epochs_from_cache

        np.testing.assert_array_equal(
            epochs_from_cache(filtered, ["A3", "A1"]),
            epoch_data[[2, 0]][:, [2, 0], :],
        )

    def test_brain_treebank_cache_path_includes_movie(self):
        context = DatasetContext(
            dataset="brain_treebank",
            stimulus_id="coraline",
            stimulus_label="coraline",
        )
        path = get_precomputed_epochs_path(
            3, "data/", dataset="brain_treebank", dataset_context=context
        )
        assert path.startswith("data/precomputed_epochs")
        assert "movie-coraline" in path
        assert "sub_3" in path

    def test_load_data_from_cache_rejects_mixed_stimuli(self, monkeypatch):
        def _fake_load_precomputed_epochs(sub_id, data_root, **kwargs):
            stimulus_id = "coraline" if sub_id == 1 else "venom"
            return {
                "epoch_data": np.zeros((2, 1, 4), dtype=np.float32),
                "selection": np.arange(2),
                "n_lags": 4,
                "ch_names": ["A1"],
                "stimulus_id": stimulus_id,
            }

        def _fake_epochs_from_cache(cache, elec_names=None):
            return np.zeros((2, 1, 4), dtype=np.float32)

        def _fake_apply_offset(X_sel, data, offset, valid_rows=None):
            return X_sel, data.reshape(data.shape[0], -1).astype(np.float32)

        import src.utils.data_loaders as dl
        import src.utils.encoding as enc_mod

        monkeypatch.setattr(dl, "load_precomputed_epochs", _fake_load_precomputed_epochs)
        monkeypatch.setattr(dl, "epochs_from_cache", _fake_epochs_from_cache)
        monkeypatch.setattr(enc_mod, "apply_offset", _fake_apply_offset)

        with pytest.raises(ValueError, match="share one stimulus"):
            load_data_from_cache(
                [1, 2],
                "data/",
                0,
                np.zeros((2, 3), dtype=np.float32),
                dataset="brain_treebank",
                per_dataset_config={"movie": "coraline"},
                dataset_context=DatasetContext(
                    dataset="brain_treebank",
                    stimulus_id="coraline",
                    stimulus_label="coraline",
                ),
            )

    def test_load_data_from_cache_rejects_mismatched_word_selections(self, monkeypatch):
        def _fake_load_precomputed_epochs(sub_id, data_root, **kwargs):
            selection = np.array([0, 1]) if sub_id == 1 else np.array([0, 2])
            return {
                "epoch_data": np.zeros((2, 1, 4), dtype=np.float32),
                "selection": selection,
                "n_lags": 4,
                "ch_names": ["A1"],
                "stimulus_id": "coraline",
            }

        import src.utils.data_loaders as dl

        monkeypatch.setattr(dl, "load_precomputed_epochs", _fake_load_precomputed_epochs)

        with pytest.raises(ValueError, match="identical word selections"):
            load_data_from_cache(
                [1, 2],
                "data/",
                0,
                np.zeros((3, 3), dtype=np.float32),
                dataset="brain_treebank",
                per_dataset_config={"movie": "coraline"},
                dataset_context=DatasetContext(
                    dataset="brain_treebank",
                    stimulus_id="coraline",
                    stimulus_label="coraline",
                ),
            )

    def test_shared_word_selection_helpers_filter_and_track_rows(self):
        cache = {
            "epoch_data": np.array([[[10]], [[11]], [[12]]], dtype=np.float32),
            "selection": np.array([4, 2, 8]),
        }
        filtered = filter_epoch_cache_to_selection(cache, np.array([8, 4]))
        np.testing.assert_array_equal(filtered["selection"], np.array([8, 4]))
        np.testing.assert_array_equal(
            filtered["epoch_data"], np.array([[[12]], [[10]]], dtype=np.float32)
        )

        valid_rows = np.array([True, False, True, True])
        np.testing.assert_array_equal(
            selection_after_offset(np.array([1, 2, 3, 4]), 1, valid_rows),
            np.array([3, 4]),
        )
        metadata = build_word_selection_metadata(
            "shared_intersection",
            [1, 2],
            np.array([1, 2, 3, 4]),
            1,
            valid_rows=valid_rows,
        )
        assert metadata["subjects"] == [1, 2]
        np.testing.assert_array_equal(metadata["epoch_selection"], np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(metadata["analysis_selection"], np.array([3, 4]))

    def test_compute_shared_word_selection_intersects_cached_subjects(self, monkeypatch):
        def _fake_load_precomputed_epochs(sub_id, data_root, **kwargs):
            selection = {
                1: np.array([0, 2, 4, 6]),
                2: np.array([2, 3, 4, 5]),
                3: np.array([1, 2, 4, 8]),
            }[sub_id]
            return {
                "epoch_data": np.zeros((len(selection), 1, 1), dtype=np.float32),
                "selection": selection,
                "n_lags": 1,
                "ch_names": ["A1"],
                "stimulus_id": "podcast",
            }

        import src.utils.data_loaders as dl

        monkeypatch.setattr(dl, "load_precomputed_epochs", _fake_load_precomputed_epochs)
        shared = compute_shared_word_selection([1, 2, 3], "data/")
        np.testing.assert_array_equal(shared, np.array([2, 4]))

    def test_load_data_from_cache_accepts_mismatched_selections_with_shared_selection(
        self, monkeypatch
    ):
        def _fake_load_precomputed_epochs(sub_id, data_root, **kwargs):
            if sub_id == 1:
                selection = np.array([0, 1])
                values = np.array([10, 11], dtype=np.float32)
            else:
                selection = np.array([1, 2])
                values = np.array([21, 22], dtype=np.float32)
            return {
                "epoch_data": values.reshape(2, 1, 1),
                "selection": selection,
                "n_lags": 1,
                "ch_names": ["A1"],
                "stimulus_id": "coraline",
            }

        import src.utils.data_loaders as dl

        monkeypatch.setattr(dl, "load_precomputed_epochs", _fake_load_precomputed_epochs)
        X, Y, selection, elec_names = load_data_from_cache(
            [1, 2],
            "data/",
            0,
            np.arange(9, dtype=np.float32).reshape(3, 3),
            dataset="brain_treebank",
            per_dataset_config={"movie": "coraline"},
            dataset_context=DatasetContext(
                dataset="brain_treebank",
                stimulus_id="coraline",
                stimulus_label="coraline",
            ),
            shared_selection=np.array([1]),
        )
        np.testing.assert_array_equal(selection, np.array([1]))
        np.testing.assert_array_equal(X, np.array([[3, 4, 5]], dtype=np.float32))
        np.testing.assert_array_equal(Y, np.array([[11, 21]], dtype=np.float32))
        assert elec_names == [["A1"], ["A1"]]

    def test_significance_loading_uses_stimulus_scoped_dir(self, tmp_path):
        sig_dir = tmp_path / "results" / "gpt2xl_layer24_offset0" / "movie-coraline"
        sig_dir.mkdir(parents=True)
        pd.DataFrame([{"subject": 1, "elec": "A1"}]).to_csv(
            sig_dir / "all_subject_sig.csv", index=False
        )

        sig_electrodes, sig_df = load_sig_electrodes(str(sig_dir), [1])
        assert sig_electrodes == {1: ["A1"]}
        assert sig_df["elec"].tolist() == ["A1"]
