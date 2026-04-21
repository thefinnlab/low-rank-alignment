import os
import re

from gensim.models import KeyedVectors
import h5py
import numpy as np
import pandas as pd

from src.core.registry import embedding_getter_registry, register_embedding_getter
from src.utils.data_loaders import get_arbitrary_embeddings


def load_embeddings(
    data_root: str, embedding_getter_name: str = "gpt2-xl", **kwargs
) -> tuple:
    # Which on-disk transcript.tsv to load depends on which model provided the
    # token alignment. The generic "transformer" getter passes its own model_dir;
    # gpt2xl/glove/arbitrary all rely on the canonical gpt2-xl transcript today.
    model_dir = kwargs.get("model_dir", "gpt2-xl")
    transcript_path = os.path.join(data_root, "stimuli", model_dir, "transcript.tsv")
    transcript_df = pd.read_csv(transcript_path, sep="\t", index_col=0)

    return embedding_getter_registry[embedding_getter_name](
        data_root, transcript_df, **kwargs
    )


@register_embedding_getter("transformer")
def get_transformer_embeddings(
    data_root, transcript_df, model_dir, layer, pooling="mean"
):
    """Load per-token hidden states for a HuggingFace model and pool per word.

    Reads from ``data/stimuli/{model_dir}/features.hdf5`` (written by
    scripts/utilities/generate_embeddings.py) and aggregates subtoken vectors into
    per-word vectors via the ``pooling`` strategy. Matches the layout used by the
    original GPT-2-XL embeddings so downstream code is unchanged.

    Args:
        data_root: BIDS dataset root (e.g. "data/").
        transcript_df: per-token DataFrame (loaded by ``load_embeddings``); must contain
            a ``word_idx`` column and have one row per subtoken.
        model_dir: subdirectory of ``{data_root}/stimuli/`` holding features.hdf5.
        layer: integer layer index (i.e. the ``L`` in ``layer-{L}``).
        pooling: "mean" | "last" | "first" — how to combine subtokens of a multi-token word.

    Returns:
        aligned_embeddings: (n_words, embed_dim) float32.
        word_df: DataFrame indexed by word_idx with columns ['word', 'start', 'end'].
    """
    embedding_path = os.path.join(data_root, "stimuli", model_dir, "features.hdf5")
    with h5py.File(embedding_path, "r") as f:
        contextual_embeddings = f[f"layer-{layer}"][...]

    if pooling == "mean":
        reduce = lambda x: x.mean(0)
    elif pooling == "last":
        reduce = lambda x: x[-1]
    elif pooling == "first":
        reduce = lambda x: x[0]
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")

    aligned_embeddings = []
    for _, group in transcript_df.groupby("word_idx"):
        indices = group.index.to_numpy()
        aligned_embeddings.append(reduce(contextual_embeddings[indices]))
    aligned_embeddings = np.stack(aligned_embeddings).astype(np.float32)

    word_df = transcript_df.groupby("word_idx").agg(
        dict(word="first", start="first", end="last")
    )

    return aligned_embeddings, word_df


@register_embedding_getter("gpt2xl")
def get_gpt2_xl_embeddings(data_root, transcript_df, layer):
    """
    Load GPT-2-XL token embeddings from HDF5, average per word.

    Args:
        data_root: BIDS dataset root (e.g. "data/").
        transcript_df: Dataframe for
        layer: Integer layer index (24 for mid, 48 for final).

    Returns:
        aligned_embeddings: (n_words, embed_dim) float32.
        word_df: DataFrame indexed by word_idx with columns ['word', 'start', 'end'].
    """
    embedding_path = os.path.join(data_root, "stimuli/gpt2-xl/features.hdf5")
    with h5py.File(embedding_path, "r") as f:
        contextual_embeddings = f[f"layer-{layer}"][...]

    # Average token embeddings per word
    aligned_embeddings = []
    for _, group in transcript_df.groupby("word_idx"):
        indices = group.index.to_numpy()
        aligned_embeddings.append(contextual_embeddings[indices].mean(0))
    aligned_embeddings = np.stack(aligned_embeddings).astype(np.float32)

    # Build word_df (one row per word)
    word_df = transcript_df.groupby("word_idx").agg(
        dict(word="first", start="first", end="last")
    )

    return aligned_embeddings, word_df


@register_embedding_getter("glove")
def get_glove_embeddings(data_root, transcript_df, dim):
    """
    Retrieves GloVe embeddings for each word in the input DataFrame.

    This function:
    1. Loads GloVe word vectors from a specified text file.
    2. Preprocesses each word by lowercasing and removing surrounding punctuation (while preserving internal characters like apostrophes).
    3. Checks whether each preprocessed word exists in the GloVe vocabulary.
    4. Collects embeddings for words found in GloVe and flags presence for each word.
    5. Adds a new column `in_glove` to the DataFrame indicating if a word was successfully matched.

    Args:
        df_word (pd.DataFrame): DataFrame with a 'word' column containing word-level text entries.
        config (WordEmbeddingConfig): Configuration object.
        data_root: path to where data is stored

    Returns:

    """
    glove_file = os.path.join(data_root, f"glove/glove.6B.{dim}d.txt")
    glove_vectors = KeyedVectors.load_word2vec_format(
        glove_file, binary=False, no_header=True
    )

    def preprocess_word(word):
        # Convert to lowercase
        word = word.lower()
        # Remove punctuation outside of words. i.e. keep apostrophes
        word = re.sub(r"\b[^\w\s]+|[^\w\s]+\b", "", word)
        return word

    word_df = transcript_df.groupby("word_idx").agg(
        dict(word="first", start="first", end="last")
    )
    words = word_df.word.tolist()
    preprocessed_words = [preprocess_word(word) for word in words]
    in_glove = []
    glove_embeddings = []

    for i, word in enumerate(preprocessed_words):
        if word in glove_vectors:
            glove_embeddings.append(glove_vectors[word])
            in_glove.append(True)
        else:
            glove_embeddings.append(
                np.zeros(glove_vectors.vector_size, dtype=np.float32)
            )
            in_glove.append(False)

    glove_embeddings = np.stack(glove_embeddings)
    word_df["in_glove"] = in_glove

    return glove_embeddings, word_df


@register_embedding_getter("arbitrary")
def get_and_write_arbitrary_embeddings(
    data_root, transcript_df, results_dir, dim, seed=42
):
    word_df = transcript_df.groupby("word_idx").agg(
        dict(word="first", start="first", end="last")
    )
    arbitrary_embedding_path = os.path.join(
        results_dir, f"arbitrary_embeddings_dim{dim}_seed{seed}.npy"
    )
    if os.path.exists(arbitrary_embedding_path):
        print(
            f"  Found existing arbitrary embeddings for dim={dim}, seed={seed}. Loading from disk."
        )
        arbitrary_embeddings = np.load(arbitrary_embedding_path)
        return arbitrary_embeddings, word_df
    correct_X_shape = (transcript_df.shape[0], dim)
    arbitrary_embeddings = get_arbitrary_embeddings(correct_X_shape, dim, seed)
    np.save(
        arbitrary_embedding_path,
        arbitrary_embeddings,
    )
    return arbitrary_embeddings, word_df
