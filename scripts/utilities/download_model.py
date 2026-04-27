#!/usr/bin/env python
"""Download a HuggingFace model repository to a local directory.

Example:
    python scripts/utilities/download_model.py \
        --model-id openai-community/gpt2-xl \
        --output-dir model_cache/models/gpt2-xl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_PATTERNS = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
]

TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model and write it locally."
    )
    parser.add_argument(
        "--model-id",
        default="openai-community/gpt2-xl",
        help="HuggingFace model id to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_cache/models/gpt2-xl",
        help="Directory where model files will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HuggingFace download cache directory.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HuggingFace token for private or gated repositories.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit SHA.",
    )
    parser.add_argument(
        "--include-tokenizer",
        action="store_true",
        help="Also download tokenizer files into the same directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = list(MODEL_PATTERNS)
    if args.include_tokenizer:
        allow_patterns.extend(TOKENIZER_PATTERNS)

    local_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=output_dir,
        token=args.token,
        allow_patterns=allow_patterns,
    )
    print(f"Saved model for {args.model_id} to {local_path}")


if __name__ == "__main__":
    main()
