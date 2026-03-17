"""Tests for utils/examples.py."""
import numpy as np
import pytest

from utils.examples import _make_context, _sample


# ---------------------------------------------------------------------------
# _make_context
# ---------------------------------------------------------------------------

class TestMakeContext:
    def test_target_in_center_of_window(self):
        words = ["a", "b", "c", "d", "e"]
        result = _make_context(words, idx=2, window=1)
        # idx=2 is "c"; window=1 => b c d
        assert "**c**" in result
        assert result == "b **c** d"

    def test_target_at_start_clamping(self):
        words = ["alpha", "beta", "gamma"]
        result = _make_context(words, idx=0, window=3)
        # idx=0, window=3 => lo=max(0,-3)=0, hi=min(3,4)=3
        assert result.startswith("**alpha**")
        assert "beta" in result
        assert "gamma" in result

    def test_target_at_end_clamping(self):
        words = ["one", "two", "three"]
        result = _make_context(words, idx=2, window=5)
        assert result.endswith("**three**")
        assert "one" in result
        assert "two" in result

    def test_window_zero_returns_only_bolded_word(self):
        words = ["x", "y", "z"]
        result = _make_context(words, idx=1, window=0)
        assert result == "**y**"

    def test_window_larger_than_array(self):
        words = ["hello"]
        result = _make_context(words, idx=0, window=100)
        assert result == "**hello**"

    def test_bold_markers_present(self):
        words = ["p", "q", "r", "s"]
        result = _make_context(words, idx=2, window=1)
        assert "**r**" in result

    def test_non_target_words_not_bolded(self):
        words = ["p", "q", "r"]
        result = _make_context(words, idx=1, window=1)
        assert "**p**" not in result
        assert "**r**" not in result


# ---------------------------------------------------------------------------
# _sample
# ---------------------------------------------------------------------------

class TestSample:
    def _basic_call(self, seed=0, num_bins=4, samples_per_bin=3):
        rng = np.random.default_rng(99)
        n = 60
        activations = rng.standard_normal(n)
        words = [f"word{i}" for i in range(n)]
        return _sample(activations, words, num_bins, samples_per_bin, context_window=2, seed=seed)

    def test_returns_only_nonempty_bins(self):
        df = self._basic_call(num_bins=4)
        # All returned bins should have at least one word
        assert len(df) > 0
        for _, g in df.groupby("bin"):
            assert len(g) >= 1

    def test_activation_within_declared_range(self):
        df = self._basic_call(num_bins=5, samples_per_bin=5)
        for _, row in df.iterrows():
            lo_str, hi_str = row["range"].strip("[)").split(", ")
            lo, hi = float(lo_str), float(hi_str)
            assert lo <= row["activation"] < hi, (
                f"Activation {row['activation']} outside range [{lo}, {hi})"
            )

    def test_samples_per_bin_cap_respected(self):
        samples_per_bin = 2
        df = self._basic_call(samples_per_bin=samples_per_bin)
        for _, g in df.groupby("bin"):
            assert len(g) <= samples_per_bin

    def test_reproducible_with_same_seed(self):
        df1 = self._basic_call(seed=42)
        df2 = self._basic_call(seed=42)
        # Same seed → same words chosen
        assert list(df1["word"]) == list(df2["word"])

    def test_different_with_different_seed(self):
        df1 = self._basic_call(seed=0)
        df2 = self._basic_call(seed=999)
        # Different seeds → (very likely) different sample
        # This can theoretically be equal but is astronomically unlikely
        assert list(df1["word"]) != list(df2["word"])

    def test_context_contains_bolded_target(self):
        df = self._basic_call()
        for _, row in df.iterrows():
            assert f"**{row['word']}**" in row["context"]

    def test_bins_with_fewer_items_return_all(self):
        """With many bins and few samples, bins with 1 item should return that item."""
        rng = np.random.default_rng(5)
        # Only 5 items but 10 bins → many bins will be empty or have 1 item
        activations = np.linspace(0, 1, 5)
        words = [f"w{i}" for i in range(5)]
        df = _sample(activations, words, num_bins=5, samples_per_bin=3, context_window=1, seed=0)
        # All non-empty bins should return all their items (capped at samples_per_bin)
        for _, g in df.groupby("bin"):
            assert len(g) <= g["count_in_bin"].iloc[0]

    def test_empty_input_returns_empty_dataframe(self):
        activations = np.array([1.0])  # single item
        words = ["only"]
        df = _sample(activations, words, num_bins=1, samples_per_bin=1, context_window=0, seed=0)
        assert len(df) >= 1
