#!/usr/bin/env python
"""Generate fMRI story embeddings from the flat PIPELINE_MODELS configs.

The ordinary embedding configs remain the single source of model, tokenizer,
layer, attention, dtype, and output-name settings.  The fMRI template supplies
only the story/session and output-root settings.
"""

from __future__ import annotations

import argparse

from scripts.fmri.generate_embeddings import run_config
from src.core.config import FMRIGenerateEmbeddingsConfig, GenerateEmbeddingsConfig
from src.utils.config_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Flat GenerateEmbeddingsConfig path; repeat for multiple models.",
    )
    parser.add_argument(
        "--fmri-template",
        default="configs/fmri/generate_gpt2_xl.yml",
        help="Nested fMRI config providing fmri paths, sessions, and subjects.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = load_config(args.fmri_template, {}, FMRIGenerateEmbeddingsConfig)
    for path in args.config:
        generation = load_config(path, {}, GenerateEmbeddingsConfig)
        print(f"\n===== Generating fMRI embeddings from {path} =====", flush=True)
        run_config(
            FMRIGenerateEmbeddingsConfig(
                generation=generation,
                fmri=template.fmri,
                refresh=template.refresh,
            )
        )


if __name__ == "__main__":
    main()
