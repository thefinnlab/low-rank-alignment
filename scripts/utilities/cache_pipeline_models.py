#!/usr/bin/env python
"""Cache every Hugging Face repository required by ``PIPELINE_MODELS``.

Run this script on an internet-connected host before moving the repository (or
``model_cache``) to the offline GPU host.  GloVe has no Hugging Face dependency,
and random-gpt2-xl reuses the cached GPT-2 XL config and tokenizer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


# Local directory names intentionally match the existing embedding configs.
PIPELINE_MODEL_REPOSITORIES = {
    "gpt-j-6b": ("EleutherAI/gpt-j-6b", "gpt-j-6b"),
    "gpt2-xl": ("openai-community/gpt2-xl", "gpt2-xl"),
    "llama3-8b": ("meta-llama/Llama-3.1-8B", "Llama-3.1-8B"),
    "roberta-large": ("FacebookAI/roberta-large", "roberta-large"),
    "gpt2": ("openai-community/gpt2", "gpt2"),
}

# Files needed by AutoConfig and AutoTokenizer. Weight files are selected from
# the live repository listing below so only one framework/serialization is kept.
CONFIG_AND_TOKENIZER_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "chat_template.jinja",
]


def required_snapshot_files(repo_files: list[str]) -> list[str]:
    """Select one Transformers-compatible weight format plus tokenizer/config."""
    safetensors = sorted(
        path
        for path in repo_files
        if Path(path).parent == Path(".") and path.endswith(".safetensors")
    )
    if safetensors:
        weights = safetensors
        index_suffix = ".safetensors.index.json"
    else:
        weights = sorted(
            path
            for path in repo_files
            if Path(path).parent == Path(".")
            and (
                Path(path).name == "pytorch_model.bin"
                or (
                    Path(path).name.startswith("pytorch_model-")
                    and path.endswith(".bin")
                )
            )
        )
        index_suffix = "pytorch_model.bin.index.json"
    if not weights:
        raise RuntimeError("Repository has neither safetensors nor PyTorch model weights")

    index_files = sorted(
        path
        for path in repo_files
        if Path(path).parent == Path(".") and path.endswith(index_suffix)
    )
    support_files = [path for path in CONFIG_AND_TOKENIZER_PATTERNS if path in repo_files]
    return support_files + index_files + weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("model_cache/models"),
        help="Root directory for the model snapshots.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face download cache directory.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token; defaults to HF_TOKEN (needed for Llama 3.1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    args.output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)

    for model_name, (repo_id, directory_name) in PIPELINE_MODEL_REPOSITORIES.items():
        destination = args.output_root / directory_name
        print(f"Caching {model_name}: {repo_id} -> {destination}", flush=True)
        allow_patterns = required_snapshot_files(api.list_repo_files(repo_id=repo_id))
        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            cache_dir=args.cache_dir,
            token=token,
            allow_patterns=allow_patterns,
        )

    print(f"Cached {len(PIPELINE_MODEL_REPOSITORIES)} repositories in {args.output_root}")


if __name__ == "__main__":
    main()
