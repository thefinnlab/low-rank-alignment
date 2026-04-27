#!/usr/bin/env python
"""Download and save a HuggingFace tokenizer to a local directory.

Example:
    python scripts/utilities/download_tokenizer.py \
        --model-id openai-community/gpt2-xl \
        --output-dir model_cache/tokenizers/gpt2-xl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace tokenizer and write it locally."
    )
    parser.add_argument(
        "--model-id",
        default="openai-community/gpt2-xl",
        help="HuggingFace model/tokenizer id to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_cache/tokenizers/gpt2-xl",
        help="Directory where tokenizer files will be written.",
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
        "--add-prefix-space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPT-2/RoBERTa-compatible prefix-space tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir,
            token=args.token,
            add_prefix_space=args.add_prefix_space,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir,
            token=args.token,
        )

    tokenizer.save_pretrained(output_dir)
    print(f"Saved tokenizer for {args.model_id} to {output_dir}")


if __name__ == "__main__":
    main()
