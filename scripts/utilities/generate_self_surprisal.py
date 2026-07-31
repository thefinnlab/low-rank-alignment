#!/usr/bin/env python
"""Generate model-specific word surprisal artifacts for language decoding."""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.utilities.generate_embeddings import (  # noqa: E402
    _causal_window_plan,
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
) -> tuple[np.ndarray, dict]:
    """Score words using the same causal sliding windows as their embeddings."""
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
    scores = np.full(len(token_ids), np.nan, dtype=float)
    from tqdm.auto import tqdm

    stride = cfg.stride if cfg.stride is not None else int(cfg.context_window) // 2
    plan = _causal_window_plan(int(cfg.context_window), int(stride), len(token_ids))
    for window_start, window_end, keep_lo, keep_hi in tqdm(
        plan, desc=cfg.output_name
    ):
        window_ids = torch.from_numpy(token_ids[window_start:window_end]).to(cfg.device)
        with torch.inference_mode():
            logits = model(window_ids[None, :]).logits[0].float()
        target_lo = max(keep_lo, window_start + 1)
        targets = np.arange(target_lo, keep_hi)
        if len(targets):
            local_targets = torch.from_numpy(targets - window_start).to(cfg.device)
            target_ids = torch.from_numpy(token_ids[targets]).to(cfg.device)
            nll = torch.nn.functional.cross_entropy(
                logits[local_targets - 1], target_ids, reduction="none"
            )
            scores[targets] = nll.cpu().numpy() / log(2.0)

    word_scores = _aggregate_token_surprisal(scores, token_words_array, len(words))
    metadata = {
        "source": "model_lm_head",
        "hf_model_id": cfg.hf_model_id,
        "tokenizer_name_or_path": tokenizer_name,
        "random_init_model": cfg.random_init_model,
        "random_init_seed": cfg.random_init_seed,
        "context_window": int(cfg.context_window),
        "stride": int(stride),
        "context_policy": "embedding_causal_sliding_windows",
        "unit": "bits",
        "word_aggregation": "sum_subtokens",
        "first_token": "missing",
    }
    del model
    if str(cfg.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return word_scores, metadata


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
        print(f"Skipping completed {model_name}: {document.dataset}/{document.stimulus_id}")
        return destination

    fallback = model_spec.get("fallback")
    if fallback == "gpt2_surprisal":
        source = Path(feature_cfg.output_dir) / document.dataset / document.stimulus_id / "tokens.tsv"
        table = pd.read_csv(source, sep="\t", keep_default_na=False, na_values=[""])
        if "surprisal" not in table:
            raise ValueError(f"Standardized GPT-2 surprisal is missing from {source}")
        if len(table) != len(document.tokens) or not np.array_equal(
            table["word"].fillna("").astype(str).to_numpy(),
            document.tokens["word"].fillna("").astype(str).to_numpy(),
        ):
            raise ValueError(f"Standardized GPT-2 surprisal is misaligned at {source}")
        values = pd.to_numeric(table["surprisal"], errors="coerce").to_numpy(float)
        metadata = {
            "source": "standardized_gpt2_surprisal_fallback",
            "fallback_for": model_name,
            "unit": "bits",
        }
    else:
        embedding_cfg = load_config(
            model_spec["embedding_config"], {}, GenerateEmbeddingsConfig
        )
        values, metadata = _score_causal_words(
            document.tokens["word"].fillna("").astype(str).tolist(),
            embedding_cfg,
        )

    table = document.tokens[["token_id", "word", "start", "end"]].copy()
    table["self_surprisal"] = values
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
        "available": int(np.isfinite(values).sum()),
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
