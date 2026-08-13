"""
generate_random_embeddings.py — Generate contextual embeddings from a randomly
initialized HuggingFace architecture.

This is a thin wrapper around generate_embeddings.py. It uses the same config
schema, but always sets random_init_model=True before generation. Use
output_name to keep random-init outputs separate from pretrained outputs.
"""

import argparse

from src.core.config import GenerateEmbeddingsConfig
from src.utils.shared.config import load_config, parse_override_args
from scripts.utilities.generate_embeddings import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to YAML config file. Repeat to run multiple configs sequentially.",
    )
    args, unknown = parser.parse_known_args()
    overrides = parse_override_args(unknown)

    config_paths = args.config if args.config else [None]
    n = len(config_paths)
    for i, path in enumerate(config_paths, start=1):
        if n > 1:
            print(f"\n===== [{i}/{n}] Generating random embeddings from {path or '<defaults>'} =====")
        cfg = load_config(path, overrides, GenerateEmbeddingsConfig)
        cfg.random_init_model = True
        generate(cfg)


if __name__ == "__main__":
    main()
