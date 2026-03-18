"""Tests for utils/data_loaders.py."""
import io
import types

import numpy as np
import pandas as pd
import pytest

from utils.data_loaders import (
    build_word_epochs,
    get_arbitrary_embeddings,
    read_electrode_file,
    read_subject_mapping,
)


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
                assert not np.array_equal(X_arb[i], X_arb[j]), (
                    f"Rows {i} and {j} are identical but words differ"
                )

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
        return pd.DataFrame({"word": ["a", "b", "c"], "start": [1.0, 3.0, 5.0], "end": [1.1, 3.1, 5.1]})

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
