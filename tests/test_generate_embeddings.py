"""Tests for scripts/utilities/generate_embeddings.py.

Covers the four logic blocks called out in PR review:
  - _tokenize_transcript: pre-split word tokenization + word_id alignment checks
  - _build_transcript_df: per-token transcript.tsv schema assembly
  - _causal_window_plan: sliding-window step logic for causal models
  - _build_bidir_batch: left-padded batch construction for bidirectional_strict
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts.utilities.generate_embeddings import (
    _build_bidir_batch,
    _build_transcript_df,
    _causal_window_plan,
    _tokenize_transcript,
)


# ---------------------------------------------------------------------------
# Fake tokenizer that mimics the slice of the HF tokenizer API we rely on.
# ---------------------------------------------------------------------------

class _FakeEncoding:
    def __init__(self, input_ids, word_ids):
        self._input_ids = input_ids
        self._word_ids = word_ids

    def __getitem__(self, key):
        if key == "input_ids":
            return self._input_ids
        raise KeyError(key)

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    """Minimal HF-tokenizer-compatible stub.

    ``word_to_tokens`` maps each word → list of ``(token_str, token_id)``. An
    empty list simulates a word that fails to tokenize (to exercise the assertion
    in ``_tokenize_transcript``).
    """

    def __init__(self, word_to_tokens: dict):
        self.word_to_tokens = word_to_tokens
        self._id_to_tok = {
            tid: tok for toks in word_to_tokens.values() for tok, tid in toks
        }

    def __call__(
        self,
        words,
        is_split_into_words=False,
        add_special_tokens=True,
        return_tensors=None,
    ):
        assert is_split_into_words, "only is_split_into_words=True is used"
        assert not add_special_tokens, "only add_special_tokens=False is used"
        assert return_tensors is None
        input_ids: list[int] = []
        word_ids: list[int] = []
        for w_idx, word in enumerate(words):
            for _, tid in self.word_to_tokens[word]:
                input_ids.append(tid)
                word_ids.append(w_idx)
        return _FakeEncoding(input_ids, word_ids)

    def convert_ids_to_tokens(self, ids):
        return [self._id_to_tok[i] for i in ids]


# ---------------------------------------------------------------------------
# _tokenize_transcript
# ---------------------------------------------------------------------------

class TestTokenizeTranscript:
    def test_single_token_words(self):
        tok = FakeTokenizer({"hello": [("hello", 1)], "world": [("world", 2)]})
        input_ids, word_ids = _tokenize_transcript(["hello", "world"], tok)
        assert input_ids.dtype == np.int64
        assert input_ids.tolist() == [1, 2]
        assert word_ids == [0, 1]

    def test_multi_token_words(self):
        tok = FakeTokenizer(
            {"pre": [("pre", 10)], "fix": [("fix", 11)], "ation": [("ation", 12)], "banana": [("ban", 20), ("ana", 21)]}
        )
        input_ids, word_ids = _tokenize_transcript(["pre", "banana", "fix"], tok)
        assert input_ids.tolist() == [10, 20, 21, 11]
        assert word_ids == [0, 1, 1, 2]

    def test_all_words_mapped_assertion(self):
        """Every word must produce at least one token — empty mapping raises."""
        tok = FakeTokenizer({"good": [("good", 1)], "bad": []})
        with pytest.raises(AssertionError, match="failed to tokenize"):
            _tokenize_transcript(["good", "bad"], tok)

    def test_input_ids_word_ids_length_match(self):
        tok = FakeTokenizer({"a": [("a", 1)], "b": [("b1", 2), ("b2", 3)]})
        input_ids, word_ids = _tokenize_transcript(["a", "b"], tok)
        assert len(input_ids) == len(word_ids) == 3


# ---------------------------------------------------------------------------
# _build_transcript_df
# ---------------------------------------------------------------------------

class TestBuildTranscriptDf:
    @pytest.fixture
    def tokenizer(self):
        return FakeTokenizer(
            {
                "hello": [("hello", 100)],
                "banana": [("ban", 200), ("ana", 201)],
                "world": [("world", 300)],
            }
        )

    @pytest.fixture
    def words_df(self):
        return pd.DataFrame(
            {
                "word": ["hello", "banana", "world"],
                "start": [0.0, 0.5, 1.2],
                "end": [0.4, 1.1, 1.8],
            }
        )

    def test_schema_columns(self, words_df, tokenizer):
        input_ids = np.array([100, 200, 201, 300], dtype=np.int64)
        word_ids = [0, 1, 1, 2]
        df = _build_transcript_df(words_df, word_ids, input_ids, tokenizer)
        assert list(df.columns) == ["word_idx", "word", "start", "end", "hftoken", "token_id"]

    def test_multi_token_word_replicates_source_row(self, words_df, tokenizer):
        input_ids = np.array([100, 200, 201, 300], dtype=np.int64)
        word_ids = [0, 1, 1, 2]
        df = _build_transcript_df(words_df, word_ids, input_ids, tokenizer)

        # 4 tokens total
        assert len(df) == 4
        # "banana" produced 2 subtokens — start/end and word should match src row
        banana = df[df["word_idx"] == 1]
        assert len(banana) == 2
        assert banana["word"].tolist() == ["banana", "banana"]
        assert banana["start"].tolist() == [0.5, 0.5]
        assert banana["end"].tolist() == [1.1, 1.1]
        assert banana["hftoken"].tolist() == ["ban", "ana"]
        assert banana["token_id"].tolist() == [200, 201]

    def test_dtype_coercion(self, words_df, tokenizer):
        input_ids = np.array([100, 200, 201, 300], dtype=np.int64)
        word_ids = [0, 1, 1, 2]
        df = _build_transcript_df(words_df, word_ids, input_ids, tokenizer)
        # word_idx and token_id must be native python ints / integer dtype; start/end floats
        assert pd.api.types.is_integer_dtype(df["word_idx"])
        assert pd.api.types.is_integer_dtype(df["token_id"])
        assert pd.api.types.is_float_dtype(df["start"])
        assert pd.api.types.is_float_dtype(df["end"])

    def test_matches_gpt2xl_transcript_schema(self, words_df, tokenizer):
        """Output columns + ordering must match the existing gpt2-xl transcript.tsv
        exactly so downstream loaders don't need to branch."""
        input_ids = np.array([100, 200, 201, 300], dtype=np.int64)
        word_ids = [0, 1, 1, 2]
        df = _build_transcript_df(words_df, word_ids, input_ids, tokenizer)
        expected = ["word_idx", "word", "start", "end", "hftoken", "token_id"]
        assert list(df.columns) == expected


# ---------------------------------------------------------------------------
# _causal_window_plan
# ---------------------------------------------------------------------------

def _assert_partitions_T(plan, T):
    """Assert the keep ranges partition [0, T) exactly (no gaps, no overlaps)."""
    covered = np.zeros(T, dtype=bool)
    for _, _, lo, hi in plan:
        assert hi > lo, f"empty keep range ({lo}, {hi})"
        assert not covered[lo:hi].any(), f"overlap at ({lo}, {hi})"
        covered[lo:hi] = True
    assert covered.all(), f"missing coverage: {np.where(~covered)[0].tolist()}"


class TestCausalWindowPlan:
    def test_short_transcript_single_window(self):
        """T <= N: single window covers the entire transcript."""
        plan = _causal_window_plan(N=8, stride=4, T=5)
        assert plan == [(0, 5, 0, 5)]
        _assert_partitions_T(plan, T=5)

    def test_half_stride_full_coverage(self):
        plan = _causal_window_plan(N=4, stride=2, T=10)
        # Windows: [0,4) [2,6) [4,8) [6,10); keeps: [0,4) [4,6) [6,8) [8,10)
        assert plan == [(0, 4, 0, 4), (2, 6, 4, 6), (4, 8, 6, 8), (6, 10, 8, 10)]
        _assert_partitions_T(plan, T=10)

    def test_stride_equals_N_no_overlap(self):
        plan = _causal_window_plan(N=4, stride=4, T=12)
        assert plan == [(0, 4, 0, 4), (4, 8, 4, 8), (8, 12, 8, 12)]
        _assert_partitions_T(plan, T=12)

    def test_boundary_when_stride_does_not_divide_remainder(self):
        """The fix: at the T boundary, keep_lo = previous_end (not end - stride)
        so the last window doesn't overwrite short-context positions."""
        # N=4 stride=4 T=10: last window ends at T=10, prev_end was 8.
        # Buggy version would set keep_lo = 10 - 4 = 6, overlapping [4, 8).
        plan = _causal_window_plan(N=4, stride=4, T=10)
        assert plan == [(0, 4, 0, 4), (4, 8, 4, 8), (6, 10, 8, 10)]
        _assert_partitions_T(plan, T=10)

    def test_stride_1(self):
        plan = _causal_window_plan(N=3, stride=1, T=5)
        # [0,3) keep [0,3); [1,4) keep [3,4); [2,5) keep [4,5)
        assert plan == [(0, 3, 0, 3), (1, 4, 3, 4), (2, 5, 4, 5)]
        _assert_partitions_T(plan, T=5)

    def test_window_index_arithmetic_valid(self):
        """keep_lo - start and keep_hi - start must index inside [0, window_len)."""
        plan = _causal_window_plan(N=4, stride=3, T=11)
        for start, end, lo, hi in plan:
            assert 0 <= lo - start
            assert hi - start <= end - start
            assert lo >= start and hi <= end

    def test_stride_too_large_rejected(self):
        with pytest.raises(AssertionError, match="stride"):
            _causal_window_plan(N=4, stride=5, T=20)

    def test_zero_stride_rejected(self):
        with pytest.raises(AssertionError, match="stride"):
            _causal_window_plan(N=4, stride=0, T=10)


# ---------------------------------------------------------------------------
# _build_bidir_batch
# ---------------------------------------------------------------------------

class TestBuildBidirBatch:
    def test_shapes_and_dtypes(self):
        input_ids = torch.arange(1, 11, dtype=torch.long)  # 10 tokens: [1..10]
        batch_ids, batch_mask = _build_bidir_batch(input_ids, targets=[0, 1, 2], N=5, pad_id=0)
        assert batch_ids.shape == (3, 5)
        assert batch_mask.shape == (3, 5)
        assert batch_ids.dtype == torch.long
        assert batch_mask.dtype == torch.long

    def test_short_context_is_left_padded(self):
        """Targets near the start produce shorter windows; padding goes on the left
        so the target remains at position N-1."""
        input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
        batch_ids, batch_mask = _build_bidir_batch(input_ids, targets=[0, 1, 2], N=4, pad_id=0)
        # Target 0: seq=[10] → row=[0,0,0,10], mask=[0,0,0,1]
        assert batch_ids[0].tolist() == [0, 0, 0, 10]
        assert batch_mask[0].tolist() == [0, 0, 0, 1]
        # Target 1: seq=[10,20] → row=[0,0,10,20]
        assert batch_ids[1].tolist() == [0, 0, 10, 20]
        assert batch_mask[1].tolist() == [0, 0, 1, 1]
        # Target 2: seq=[10,20,30] → row=[0,10,20,30]
        assert batch_ids[2].tolist() == [0, 10, 20, 30]
        assert batch_mask[2].tolist() == [0, 1, 1, 1]

    def test_full_context_no_padding(self):
        """Once target index >= N-1, windows fill the full row — no padding."""
        input_ids = torch.tensor([10, 20, 30, 40, 50, 60, 70], dtype=torch.long)
        batch_ids, batch_mask = _build_bidir_batch(input_ids, targets=[3, 4], N=4, pad_id=0)
        # Target 3: seq = input_ids[0:4] = [10,20,30,40]
        assert batch_ids[0].tolist() == [10, 20, 30, 40]
        assert batch_mask[0].tolist() == [1, 1, 1, 1]
        # Target 4: seq = input_ids[1:5] = [20,30,40,50]
        assert batch_ids[1].tolist() == [20, 30, 40, 50]
        assert batch_mask[1].tolist() == [1, 1, 1, 1]

    def test_target_always_at_last_position(self):
        """After the strict-left-only framing, the hidden state at index -1 is
        always the target's hidden state — invariant we rely on in the caller."""
        input_ids = torch.arange(100, 110, dtype=torch.long)  # [100..109]
        for tgt in [0, 3, 5, 9]:
            batch_ids, batch_mask = _build_bidir_batch(input_ids, targets=[tgt], N=4, pad_id=-1)
            # Last mask position is always 1; last input_id is always the target.
            assert batch_mask[0, -1].item() == 1
            assert batch_ids[0, -1].item() == input_ids[tgt].item()

    def test_no_future_tokens_leak(self):
        """No row should contain any token with index > target (strict left-only)."""
        input_ids = torch.arange(1, 11, dtype=torch.long)  # unique ids = index + 1
        batch_ids, batch_mask = _build_bidir_batch(input_ids, targets=[0, 3, 9], N=5, pad_id=0)
        for row, tgt in enumerate([0, 3, 9]):
            real_tokens = batch_ids[row][batch_mask[row].bool()]
            # All real tokens should be <= the target token id
            assert real_tokens.max().item() <= input_ids[tgt].item()

    def test_custom_pad_id(self):
        input_ids = torch.tensor([5, 6, 7], dtype=torch.long)
        batch_ids, batch_mask = _build_bidir_batch(input_ids, targets=[0], N=4, pad_id=999)
        assert batch_ids[0].tolist() == [999, 999, 999, 5]
        assert batch_mask[0].tolist() == [0, 0, 0, 1]

    def test_variable_length_windows_in_one_batch(self):
        """A batch mixing short and full windows should pad each row independently."""
        input_ids = torch.arange(1, 11, dtype=torch.long)
        batch_ids, batch_mask = _build_bidir_batch(input_ids, targets=[0, 9], N=5, pad_id=0)
        # Target 0: 4 pads + [1]
        assert batch_mask[0].tolist() == [0, 0, 0, 0, 1]
        # Target 9: full window [6,7,8,9,10]
        assert batch_mask[1].tolist() == [1, 1, 1, 1, 1]
        assert batch_ids[1].tolist() == [6, 7, 8, 9, 10]

    def test_empty_targets(self):
        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        batch_ids, batch_mask = _build_bidir_batch(input_ids, targets=[], N=4, pad_id=0)
        assert batch_ids.shape == (0, 4)
        assert batch_mask.shape == (0, 4)
