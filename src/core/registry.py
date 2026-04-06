# Registry of functions for creating decoding models.
# Structure: {name: fn}
embedding_getter_registry = {}


def register_embedding_getter(name=None):
    """
    Decorator to register a function to gather embeddings.

    The decorated function must follow the signature:
        model_constructor(data_root, transcript_df, **kwargs) -> output_tensor (num_words, embed_dim), word_df

    Where:
        - data_root: path to the root of the data directory. Where embeddings are stored.
        - transcript_df: Dataframe containing all transcript information. Multiple rows per word, one per GPT2-XL token.
        - **kwargs are arbitrary kwargs set in the embedding_params part of EncodingConfig.
        - output_tensor provides the word embeddings for each word in the desired model.
        - word_df - the final transcript after any groupings made when generating the embeddings.

    Args:
        name (str, optional): Optional name to register the constructor under.
                              Defaults to the function's __name__.

    Returns:
        function: A decorator that registers the model constructor in
                  embedding_getter_registry with structure:
                  {name: fn}
    """

    def decorator(fn):
        model_name = name or fn.__name__
        embedding_getter_registry[model_name] = fn
        return fn

    return decorator
