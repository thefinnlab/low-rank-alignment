#!/usr/bin/env python
"""Generate or stage reusable language-feature artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.config import LanguageFeatureGenerationConfig
from src.utils.config_utils import load_config, parse_override_args
from src.utils.config_utils import save_effective_config
from src.utils.language_features import (
    generate_language_features,
    stage_language_resources,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage-resources",
        action="store_true",
        help="Download/validate configured resources and exit.",
    )
    args, unknown = parser.parse_known_args()
    config = load_config(
        args.config,
        parse_override_args(unknown),
        LanguageFeatureGenerationConfig,
    )
    if args.stage_resources:
        save_effective_config(config, Path(config.resource_dir) / "language_features")
        staged = stage_language_resources(config, download=True)
        for name, spec in staged.items():
            print(f"{name}: {spec['path']} ({spec['sha256']})")
        return
    output = generate_language_features(config)
    print(f"Language features written to {output}")


if __name__ == "__main__":
    main()
