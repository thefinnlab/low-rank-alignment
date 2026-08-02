#!/usr/bin/env python
"""Populate only Cars-2 concreteness labels in an existing token artifact."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.utilities.generate_surprisal import _load_existing_tokens
from src.core.config import LanguageFeatureGenerationConfig
from src.utils.config_utils import load_config
from src.utils.language_features import (
    DOCUMENT_LOADERS,
    PROVIDERS,
    hydrate_provider_resources,
    stage_language_resources,
)


OUTPUT_COLUMNS = (
    "concreteness",
    "concreteness_available",
    "concreteness_match_level",
    "concreteness_match_key",
)


def _write_concreteness(config: LanguageFeatureGenerationConfig) -> Path:
    hydrate_provider_resources(config)
    from src.utils import language_feature_providers as _builtins  # noqa: F401

    if config.dataset != "brain_treebank":
        raise ValueError("The focused concreteness generator requires brain_treebank")
    if config.per_dataset_config.get("movie") != "cars-2":
        raise ValueError("The focused concreteness generator requires movie=cars-2")

    document = DOCUMENT_LOADERS[config.dataset](config)
    resource_config = deepcopy(config)
    ud_resource = config.provider_options.get("ud", {}).get(
        "model_resource", "stanza_ewt"
    )
    resource_config.resources = {
        "concreteness": config.resources["concreteness"],
        ud_resource: config.resources[ud_resource],
    }
    resources = stage_language_resources(resource_config, download=False)

    working = document.tokens.copy()
    ud_result = PROVIDERS["ud"].runner(
        document,
        working.copy(),
        config.provider_options.get("ud", {}),
        resources,
    )
    working = pd.concat(
        [working, ud_result.token_features.drop(columns="token_id", errors="ignore")],
        axis=1,
    )
    result = PROVIDERS["lexical_norms"].runner(
        document,
        working,
        {"resources": ["concreteness"]},
        resources,
    )

    output = Path(config.output_dir) / document.dataset / document.stimulus_id
    token_table = _load_existing_tokens(output, document.tokens)
    token_table = token_table.drop(columns=list(OUTPUT_COLUMNS), errors="ignore")
    for column in OUTPUT_COLUMNS:
        token_table[column] = result.token_features[column].to_numpy()

    temporary_tokens = output / "tokens.tsv.tmp"
    token_table.to_csv(temporary_tokens, sep="\t", index=False)
    temporary_tokens.replace(output / "tokens.tsv")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("providers", {})["lexical_norms"] = {
        "version": PROVIDERS["lexical_norms"].version,
        **result.metadata,
    }
    for column in OUTPUT_COLUMNS:
        manifest.setdefault("feature_providers", {})[column] = "lexical_norms"
        manifest.setdefault("feature_coverage", {})[column] = {
            "available": int(token_table[column].notna().sum()),
            "total": int(len(token_table)),
        }
    manifest.setdefault("focused_updates", {})["cars2_concreteness"] = {
        "provider": "lexical_norms",
        "resource": "concreteness",
        "features": list(OUTPUT_COLUMNS),
    }
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, default=str))
    temporary_manifest.replace(manifest_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/language_features/brain_treebank.yml",
    )
    args = parser.parse_args()
    config = load_config(args.config, {}, LanguageFeatureGenerationConfig)
    output = _write_concreteness(config)
    print(f"Wrote Cars-2 concreteness labels to {output}")


if __name__ == "__main__":
    main()
