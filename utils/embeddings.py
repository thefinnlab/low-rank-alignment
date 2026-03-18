import os

import h5py
import numpy as np
import pandas as pd


def load_embeddings(data_root: str, layer: int) -> tuple:
    """
    Load GPT-2-XL token embeddings from HDF5, average per word.

    Args:
        data_root: BIDS dataset root (e.g. "data/").
        layer: Integer layer index (24 for mid, 48 for final).

    Returns:
        aligned_embeddings: (n_words, embed_dim) float32.
        df_word: DataFrame indexed by word_idx with columns ['word', 'start', 'end'].
    """
    embedding_path = os.path.join(data_root, "stimuli/gpt2-xl/features.hdf5")
    transcript_path = os.path.join(data_root, "stimuli/gpt2-xl/transcript.tsv")

    with h5py.File(embedding_path, "r") as f:
        contextual_embeddings = f[f"layer-{layer}"][...]

    df_contextual = pd.read_csv(transcript_path, sep="\t", index_col=0)

    # Average token embeddings per word
    aligned_embeddings = []
    for _, group in df_contextual.groupby("word_idx"):
        indices = group.index.to_numpy()
        aligned_embeddings.append(contextual_embeddings[indices].mean(0))
    aligned_embeddings = np.stack(aligned_embeddings).astype(np.float32)

    # Build df_word (one row per word)
    df_word = df_contextual.groupby("word_idx").agg(
        dict(word="first", start="first", end="last")
    )

    return aligned_embeddings, df_word
