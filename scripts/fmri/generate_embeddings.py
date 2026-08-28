"""Generate contextual embeddings for stories used by deep-fMRI-dataset analyses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from scripts.utilities.generate_embeddings import (
    _auto_output_name,
    _build_transcript_df,
    _load_model_for_generation,
    _resolve_defaults,
    _resolve_layers,
    _resolve_np_dtype,
    _run_bidirectional_strict,
    _run_causal_sliding,
    _run_podcast_per_token,
    _select_generation_model_class,
    _tokenize_transcript_per_word,
    _tokenizer_name_or_path,
    _write_features_hdf5,
)
from src.core.config import FMRIGenerateEmbeddingsConfig
from src.utils.shared.config import load_config, parse_args, save_effective_config
from src.utils.shared.fmri import (
    load_story_word_data,
    resolve_fmri_stories,
    TEMPORARILY_HELD_OUT_FMRI_STORIES,
)


def run_config(config: FMRIGenerateEmbeddingsConfig) -> None:
    cfg = config.generation
    output_name = cfg.output_name or _auto_output_name(cfg.hf_model_id)
    tokenizer_name = _tokenizer_name_or_path(cfg)
    kwargs = {
        "cache_dir": cfg.hf_cache_dir,
        "local_files_only": cfg.local_files_only,
        "token": cfg.hf_token,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, add_prefix_space=cfg.add_prefix_space, **kwargs
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)

    model = _load_model_for_generation(
        cfg, _select_generation_model_class(cfg.attention_mode)
    )
    attention_mode, stride = _resolve_defaults(cfg, model.config.model_type)
    layers = _resolve_layers(cfg.layers, model.config.num_hidden_layers + 1)
    train_stories, test_stories, _ = resolve_fmri_stories(
        config.fmri.fmri_repo_root, config.fmri.sessions
    )
    stories = train_stories + [story for story in test_stories if story not in train_stories]
    frames, _ = load_story_word_data(config.fmri.fmri_repo_root, stories)
    output_root = Path(config.fmri.embedding_root) / output_name
    save_effective_config(config, output_root)

    for story in stories:
        words_df = frames[story]
        input_ids, word_ids, hftokens = _tokenize_transcript_per_word(
            words_df["word"].tolist(), tokenizer
        )
        output = {
            layer: np.zeros(
                (len(input_ids), model.config.hidden_size),
                dtype=_resolve_np_dtype(cfg.output_dtype),
            )
            for layer in layers
        }
        performance = None
        if attention_mode == "causal_native":
            _run_causal_sliding(model, input_ids, layers, cfg, stride, output)
        elif attention_mode == "podcast_per_token":
            performance = _run_podcast_per_token(
                model, input_ids, layers, cfg, tokenizer, output
            )
        elif attention_mode == "bidirectional_strict":
            _run_bidirectional_strict(
                model, input_ids, layers, cfg, tokenizer, output
            )
        else:
            raise NotImplementedError(f"Unsupported attention mode: {attention_mode}")

        transcript = _build_transcript_df(
            words_df, word_ids, input_ids, tokenizer, hftokens=hftokens
        )
        if performance is not None:
            transcript = pd.concat([transcript, performance], axis=1)
        story_dir = output_root / story
        story_dir.mkdir(parents=True, exist_ok=True)
        _write_features_hdf5(str(story_dir / "features.hdf5"), layers, output)
        transcript.to_csv(story_dir / "transcript.tsv", sep="\t", index=False)
        metadata = {
            "story": story,
            "hf_model_id": cfg.hf_model_id,
            "output_name": output_name,
            "model_type": model.config.model_type,
            "hidden_dim": model.config.hidden_size,
            "layers_saved": layers,
            "context_window": cfg.context_window,
            "stride": stride,
            "attention_mode": attention_mode,
            "pooling": cfg.pooling,
            "n_tokens": int(len(input_ids)),
            "n_words": int(len(words_df)),
            "sessions": list(config.fmri.sessions),
            "temporarily_held_out_stories": sorted(TEMPORARILY_HELD_OUT_FMRI_STORIES),
        }
        (story_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"Wrote {story_dir}")


def main() -> None:
    args, overrides = parse_args(__doc__)
    run_config(load_config(args.config, overrides, FMRIGenerateEmbeddingsConfig))


if __name__ == "__main__":
    main()
