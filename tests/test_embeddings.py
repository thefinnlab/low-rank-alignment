"""Tests for utils/embeddings.py."""
import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

from utils.embeddings import load_embeddings


@pytest.fixture
def tmp_hdf5_and_transcript(tmp_path):
    """6 tokens, embed_dim=4, 3 words (word_idx 0→tokens 0,1 / 1→tokens 2,3 / 2→tokens 4,5)."""
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((6, 4)).astype(np.float32)
    path = tmp_path / "stimuli" / "gpt2-xl"
    path.mkdir(parents=True)
    with h5py.File(path / "features.hdf5", "w") as f:
        f.create_dataset("layer-24", data=embeddings)
    df = pd.DataFrame({
        "word_idx": [0, 0, 1, 1, 2, 2],
        "word": ["hello", "hello", "world", "world", "foo", "foo"],
        "start": [0.0, 0.05, 0.2, 0.25, 0.5, 0.55],
        "end":   [0.05, 0.2, 0.25, 0.5, 0.55, 0.7],
    })
    df.to_csv(path / "transcript.tsv", sep="\t")
    return str(tmp_path), embeddings


class TestLoadEmbeddings:
    def test_output_shapes(self, tmp_hdf5_and_transcript):
        data_root, _ = tmp_hdf5_and_transcript
        aligned_embeddings, df_word = load_embeddings(data_root, layer=24)
        assert aligned_embeddings.shape == (3, 4)
        assert len(df_word) == 3

    def test_token_averaging(self, tmp_hdf5_and_transcript):
        data_root, raw_embeddings = tmp_hdf5_and_transcript
        aligned_embeddings, _ = load_embeddings(data_root, layer=24)
        # word 0 = mean of tokens 0 and 1
        expected = (raw_embeddings[0] + raw_embeddings[1]) / 2
        np.testing.assert_allclose(aligned_embeddings[0], expected, rtol=1e-5)

    def test_df_word_columns(self, tmp_hdf5_and_transcript):
        data_root, _ = tmp_hdf5_and_transcript
        _, df_word = load_embeddings(data_root, layer=24)
        assert {"word", "start", "end"} <= set(df_word.columns)

    def test_single_token_words(self, tmp_path):
        """When each token is its own word, output equals raw embedding rows."""
        rng = np.random.default_rng(1)
        embeddings = rng.standard_normal((3, 4)).astype(np.float32)
        path = tmp_path / "stimuli" / "gpt2-xl"
        path.mkdir(parents=True)
        with h5py.File(path / "features.hdf5", "w") as f:
            f.create_dataset("layer-24", data=embeddings)
        df = pd.DataFrame({
            "word_idx": [0, 1, 2],
            "word": ["a", "b", "c"],
            "start": [0.0, 0.2, 0.4],
            "end":   [0.1, 0.3, 0.5],
        })
        df.to_csv(path / "transcript.tsv", sep="\t")
        aligned, _ = load_embeddings(str(tmp_path), layer=24)
        np.testing.assert_allclose(aligned, embeddings, rtol=1e-5)

    def test_missing_layer_raises(self, tmp_hdf5_and_transcript):
        data_root, _ = tmp_hdf5_and_transcript
        with pytest.raises(KeyError):
            load_embeddings(data_root, layer=99)
