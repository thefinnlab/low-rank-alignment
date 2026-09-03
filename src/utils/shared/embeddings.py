"""Embedding loaders shared by retained workflows."""

import os
import re
from gensim.models import KeyedVectors
import h5py
import numpy as np
import pandas as pd
from src.core.registry import (
    embedding_getter_registry,
    get_dataset_adapter,
    register_embedding_getter,
    resolve_dataset_context,
)
from src.utils.shared.ieeg import get_arbitrary_embeddings


def load_embeddings(
    data_root: str,
    embedding_getter_name: str = "gpt2xl",
    dataset: str = "podcast",
    per_dataset_config: dict | None = None,
    dataset_context=None,
    limit_to_glove: bool = False,
    glove_dim: int = 300,
    **kwargs,
) -> tuple:
    """Load model embeddings and optionally annotate shared GloVe coverage.

    The embedding matrix is deliberately left at its original row count so its
    rows remain compatible with cached word selections. When ``limit_to_glove``
    is true, the returned word table contains an ``in_glove`` boolean column
    for downstream alignment to apply.
    """
    if isinstance(embedding_getter_name, int):
        kwargs.setdefault("layer", embedding_getter_name)
        embedding_getter_name = "gpt2xl"

    model_dir = kwargs.get("model_dir", "gpt2-xl")
    drop_untimed_words = kwargs.pop("drop_untimed_words", True)
    context = dataset_context or resolve_dataset_context(
        dataset=dataset,
        subject_ids=kwargs.get("subject_ids"),
        per_dataset_config=per_dataset_config,
        data_root=data_root,
    )
    adapter = get_dataset_adapter(context.dataset)
    embedding_dir = adapter.get_embedding_dir(data_root, context, model_dir)
    if embedding_getter_name in {"transformer", "gpt2xl"}:
        kwargs.setdefault("embedding_dir", embedding_dir)
        if embedding_getter_name == "transformer":
            kwargs.setdefault("model_dir", model_dir)
        try:
            transcript_df = adapter.load_generated_transcript(
                data_root,
                context,
                model_dir,
                drop_untimed_words=drop_untimed_words,
            )
        except FileNotFoundError:
            transcript_df = adapter.load_word_table(
                data_root,
                context,
                drop_untimed_words=drop_untimed_words,
                **kwargs,
            )
    else:
        transcript_df = adapter.load_word_table(
            data_root,
            context,
            drop_untimed_words=drop_untimed_words,
            **kwargs,
        )

    embeddings, word_df = embedding_getter_registry[embedding_getter_name](
        data_root, transcript_df, **kwargs
    )
    if limit_to_glove and "in_glove" not in word_df.columns:
        _, glove_word_df = get_glove_embeddings(
            data_root, word_df, dim=int(glove_dim)
        )
        if len(glove_word_df) != len(word_df):
            raise ValueError(
                "GloVe coverage rows do not align with the loaded embeddings"
            )
        word_df = word_df.copy()
        word_df["in_glove"] = glove_word_df["in_glove"].to_numpy(dtype=bool)
    return embeddings, word_df


def _to_word_level_df(transcript_df: pd.DataFrame) -> pd.DataFrame:
    if "word_idx" in transcript_df.columns:
        return transcript_df.groupby("word_idx").agg(
            dict(word="first", start="first", end="last")
        )
    if transcript_df.index.name == "word_idx":
        return transcript_df
    if {"word", "start", "end"} <= set(transcript_df.columns):
        df = transcript_df.copy()
        df.index = pd.Index(range(len(df)), name="word_idx")
        return df
    raise ValueError("Transcript DataFrame must contain word/start/end information.")


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

    word_df = _to_word_level_df(transcript_df)
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

@register_embedding_getter("transformer")
def get_transformer_embeddings(
    data_root, transcript_df, model_dir, layer, pooling="mean", embedding_dir=None
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
    embedding_dir = embedding_dir or os.path.join(data_root, "stimuli", model_dir)
    embedding_path = os.path.join(embedding_dir, "features.hdf5")
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

    word_df = _to_word_level_df(transcript_df)

    return aligned_embeddings, word_df

@register_embedding_getter("gpt2xl")
def get_gpt2_xl_embeddings(
    data_root, transcript_df, layer, embedding_dir=None, model_dir=None
):
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
    embedding_dir = embedding_dir or os.path.join(data_root, "stimuli", "gpt2-xl")
    embedding_path = os.path.join(embedding_dir, "features.hdf5")
    with h5py.File(embedding_path, "r") as f:
        contextual_embeddings = f[f"layer-{layer}"][...]

    # Average token embeddings per word
    aligned_embeddings = []
    for _, group in transcript_df.groupby("word_idx"):
        indices = group.index.to_numpy()
        aligned_embeddings.append(contextual_embeddings[indices].mean(0))
    aligned_embeddings = np.stack(aligned_embeddings).astype(np.float32)

    # Build word_df (one row per word)
    word_df = _to_word_level_df(transcript_df)

    return aligned_embeddings, word_df
