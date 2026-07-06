#!/usr/bin/env python
"""Export a dataset adapter's normalized transcript word table.

Examples:
    python scripts/utilities/export_transcript.py \
        --dataset brain_treebank \
        --data-root data/brain-treebank \
        --movie cars-2 \
        --output /tmp/cars-2-transcript.tsv

    python scripts/utilities/export_transcript.py \
        --dataset brain_treebank \
        --data-root data/brain-treebank \
        --movie cars-2 \
        --drop-untimed-words \
        --output /tmp/cars-2-timed-transcript.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.registry import get_dataset_adapter, resolve_dataset_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a normalized transcript word table for manual review."
    )
    parser.add_argument(
        "--dataset",
        default="brain_treebank",
        help="Dataset adapter name, e.g. brain_treebank or podcast.",
    )
    parser.add_argument(
        "--data-root",
        default="data/brain-treebank",
        help="Dataset root passed to the adapter.",
    )
    parser.add_argument(
        "--movie",
        default=None,
        help="Brain Treebank movie id, e.g. cars-2. Sets per_dataset_config.movie.",
    )
    parser.add_argument(
        "--per-dataset-config",
        default=None,
        help="Optional JSON object merged into per_dataset_config.",
    )
    parser.add_argument(
        "--model-dir",
        default="gpt2-xl",
        help="Model transcript directory for adapters that use model_dir.",
    )
    parser.add_argument(
        "--drop-untimed-words",
        action="store_true",
        help="Drop rows without numeric start/end after transcript normalization.",
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Write the full adapter DataFrame instead of the compact review table.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output transcript path. Use .tsv for tab-separated output; otherwise CSV.",
    )
    return parser.parse_args()


def _per_dataset_config(args: argparse.Namespace) -> dict:
    config = {}
    if args.per_dataset_config:
        config.update(json.loads(args.per_dataset_config))
    if args.movie:
        config["movie"] = args.movie
    return config


def _review_table(df):
    review = df.copy()

    preferred = [
        "word_idx",
        "word",
        "start",
        "end",
        "sentence_idx",
        "idx_in_sentence",
        "sentence",
        "source_word_idx",
        "source_word_idxs",
    ]
    return review[[col for col in preferred if col in review.columns]]


def main() -> None:
    args = parse_args()
    per_dataset_config = _per_dataset_config(args)
    context = resolve_dataset_context(
        dataset=args.dataset,
        subject_ids=[],
        per_dataset_config=per_dataset_config,
        data_root=args.data_root,
    )
    adapter = get_dataset_adapter(context.dataset)
    df = adapter.load_word_table(
        args.data_root,
        context,
        model_dir=args.model_dir,
        drop_untimed_words=args.drop_untimed_words,
    )
    if not args.all_columns:
        df = _review_table(df)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if output.suffix == ".tsv" else ","
    df.to_csv(output, sep=sep, index=False)
    print(f"Wrote {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
