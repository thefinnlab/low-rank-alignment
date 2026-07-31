#!/usr/bin/env python
"""Generate model-specific word surprisal artifacts for language decoding."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.utilities.generate_embeddings import (  # noqa: E402
    _load_model_for_generation,
    _tokenize_transcript_per_word,
    _tokenizer_name_or_path,
)
from scripts.utilities.generate_language_features import expand_stimulus_configs  # noqa: E402
from src.core.config import GenerateEmbeddingsConfig, LanguageFeatureGenerationConfig  # noqa: E402
from src.utils.config_utils import load_config  # noqa: E402
from src.utils.language_features import DOCUMENT_LOADERS  # noqa: E402
from src.utils.language_feature_providers import _aggregate_token_surprisal  # noqa: E402


def _score_causal_words(
    words: list[str], cfg: GenerateEmbeddingsConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reproduce the original Podcast word surprisal and entropy procedure."""
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
    model = _load_model_for_generation(cfg, AutoModelForCausalLM)
    token_ids, token_words, _ = _tokenize_transcript_per_word(words, tokenizer)
    token_words_array = np.asarray(token_words, dtype=np.int64)
    token_surprisal = np.full(len(token_ids), np.nan, dtype=float)
    token_entropy = np.full(len(token_ids), np.nan, dtype=float)
    from tqdm.auto import tqdm

    context_len = int(cfg.context_window)
    width = context_len + 1
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_size = int(cfg.batch_size)
    for batch_start in tqdm(
        range(0, len(token_ids), batch_size), desc=cfg.output_name
    ):
        targets = np.arange(
            batch_start, min(batch_start + batch_size, len(token_ids)), dtype=int
        )
        batch = torch.full((len(targets), width), int(pad_id), dtype=torch.long)
        for row, target in enumerate(targets):
            start = max(0, int(target) - context_len)
            sequence = torch.from_numpy(token_ids[start : target + 1])
            batch[row, -len(sequence) :] = sequence
        batch = batch.to(cfg.device)
        with torch.inference_mode():
            logits = model(batch).logits[:, -2, :].float()
            probabilities = torch.softmax(logits, dim=-1)
            target_ids = torch.from_numpy(token_ids[targets]).to(cfg.device)
            true_probabilities = probabilities.gather(
                1, target_ids[:, None]
            ).squeeze(1)
            surprisal_bits = -torch.log2(true_probabilities)
            entropy = torch.distributions.Categorical(probs=probabilities).entropy()
        token_surprisal[targets] = surprisal_bits.cpu().numpy()
        token_entropy[targets] = entropy.cpu().numpy()

    word_surprisal = _aggregate_token_surprisal(
        token_surprisal, token_words_array, len(words)
    )
    word_entropy = np.full(len(words), np.nan, dtype=float)
    for word_index in np.unique(token_words_array):
        word_entropy[int(word_index)] = token_entropy[
            np.flatnonzero(token_words_array == word_index)[0]
        ]
    metadata = {
        "source": "model_lm_head",
        "hf_model_id": cfg.hf_model_id,
        "tokenizer_name_or_path": tokenizer_name,
        "random_init_model": cfg.random_init_model,
        "random_init_seed": cfg.random_init_seed,
        "context_window": context_len,
        "context_policy": "original_podcast_rolling_window",
        "padding_policy": "tokenizer_pad_id_or_zero_without_attention_mask",
        "surprisal_unit": "bits",
        "surprisal_word_aggregation": "sum_subtokens",
        "entropy_unit": "nats",
        "entropy_word_aggregation": "first_subtoken",
        "first_token": "scored_from_left_padding",
    }
    del model
    if str(cfg.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return word_surprisal, word_entropy, metadata


def _write_artifact(
    *,
    model_name: str,
    model_spec: dict,
    feature_cfg: LanguageFeatureGenerationConfig,
    output_root: Path,
    refresh: bool,
) -> Path:
    document = DOCUMENT_LOADERS[feature_cfg.dataset](feature_cfg)
    destination = output_root / model_name / document.dataset / document.stimulus_id
    token_path = destination / "tokens.tsv"
    if token_path.is_file() and not refresh:
        manifest_path = destination / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        columns = set(pd.read_csv(token_path, sep="\t", nrows=1).columns)
        if {"self_surprisal", "self_entropy"} <= columns and manifest.get(
            "context_policy"
        ) == "original_podcast_rolling_window":
            print(
                f"Skipping completed {model_name}: "
                f"{document.dataset}/{document.stimulus_id}"
            )
            return destination

    embedding_cfg = load_config(
        model_spec["embedding_config"], {}, GenerateEmbeddingsConfig
    )
    surprisal, entropy, metadata = _score_causal_words(
        document.tokens["word"].fillna("").astype(str).tolist(),
        embedding_cfg,
    )
    if model_spec.get("fallback_for"):
        metadata["source"] = "gpt2_small_causal_fallback"
        metadata["fallback_for"] = str(model_spec["fallback_for"])

    table = document.tokens[["token_id", "word", "start", "end"]].copy()
    table["self_surprisal"] = surprisal
    table["self_entropy"] = entropy
    destination.mkdir(parents=True, exist_ok=True)
    temporary = token_path.with_suffix(".tsv.tmp")
    table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(token_path)
    manifest = {
        "schema_version": "1",
        "dataset": document.dataset,
        "stimulus_id": document.stimulus_id,
        "model": model_name,
        "n_tokens": len(table),
        "surprisal_available": int(np.isfinite(surprisal).sum()),
        "entropy_available": int(np.isfinite(entropy).sum()),
        **metadata,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/language_features/self_surprisal.yml")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text()) or {}
    models = args.models or list(payload["models"])
    datasets = args.datasets or list(payload["datasets"])
    unknown_models = sorted(set(models) - set(payload["models"]))
    unknown_datasets = sorted(set(datasets) - set(payload["datasets"]))
    if unknown_models or unknown_datasets:
        raise ValueError(
            f"Unknown models={unknown_models}, datasets={unknown_datasets}"
        )
    output_root = Path(payload.get(
        "output_root", "processed_data/language_features/self_surprisal"
    ))
    for model_name in models:
        for dataset_name in datasets:
            feature_cfg = load_config(
                payload["datasets"][dataset_name], {}, LanguageFeatureGenerationConfig
            )
            for expanded in expand_stimulus_configs(feature_cfg):
                _write_artifact(
                    model_name=model_name,
                    model_spec=payload["models"][model_name],
                    feature_cfg=expanded,
                    output_root=output_root,
                    refresh=args.refresh,
                )


if __name__ == "__main__":
    main()
