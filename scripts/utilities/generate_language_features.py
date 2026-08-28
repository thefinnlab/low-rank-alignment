#!/usr/bin/env python
"""Generate or stage reusable language-feature artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.config import LanguageFeatureGenerationConfig
from src.utils.shared.config import load_config, parse_override_args
from src.utils.shared.config import save_effective_config
from src.utils.language_features.core import (
    generate_language_features,
    hydrate_provider_resources,
    stage_language_resources,
)


def expand_stimulus_configs(config):
    """Expand an fMRI story list into independent artifact configurations."""
    stories = config.per_dataset_config.get("stories")
    if config.dataset != "fmri" or stories is None:
        return [config]
    if stories == "all":
        grid_dir = (
            Path(config.data_root) / "data" / "ds003020" / "derivatives" / "TextGrids"
        )
        stories = sorted(path.stem for path in grid_dir.glob("*.TextGrid"))
    if not isinstance(stories, list) or not stories:
        raise ValueError("per_dataset_config.stories must be 'all' or a non-empty list")
    expanded = []
    for story in stories:
        item = deepcopy(config)
        item.per_dataset_config.pop("stories", None)
        item.per_dataset_config["story"] = str(story)
        expanded.append(item)
    return expanded


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
        hydrate_provider_resources(config)
        save_effective_config(config, Path(config.resource_dir) / "language_features")
        staged = stage_language_resources(config, download=True)
        for name, spec in staged.items():
            print(f"{name}: {spec['path']} ({spec['sha256']})")
        return
    for item in expand_stimulus_configs(config):
        output = generate_language_features(item)
        print(f"Language features written to {output}")


if __name__ == "__main__":
    main()
