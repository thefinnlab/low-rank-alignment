#!/usr/bin/env python
"""Populate the common interpretable fMRI decoding features without heavy providers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.utilities.generate_language_features import expand_stimulus_configs
from scripts.utilities.generate_surprisal import _load_existing_tokens
from src.core.config import LanguageFeatureGenerationConfig
from src.utils.config_utils import load_config
from src.utils.language_features import DOCUMENT_LOADERS, PROVIDERS, hydrate_provider_resources, stage_language_resources


NATIVE_COLUMNS = (
    "word_zipf_frequency", "word_length", "sentence_position", "sentence_onset",
    "sentence_end", "word_duration", "pre_word_pause", "post_word_pause",
)
UD_COLUMNS = (
    "content_word", "part_of_speech", "dependency_depth", "dependency_length",
)
LEXICAL_COLUMNS = ("concreteness", "valence", "arousal")
OUTPUT_COLUMNS = NATIVE_COLUMNS + UD_COLUMNS + LEXICAL_COLUMNS


def _resources_for(config: LanguageFeatureGenerationConfig) -> dict:
    names = {config.provider_options["ud"].get("model_resource", "stanza_ewt")}
    names.update(config.provider_options["lexical_norms"].get("resources", []))
    subset = deepcopy(config)
    subset.resources = {name: config.resources[name] for name in names}
    return stage_language_resources(subset, download=False)


def _write_features(config: LanguageFeatureGenerationConfig) -> Path:
    hydrate_provider_resources(config)
    from src.utils import language_feature_providers as _builtins  # noqa: F401

    document = DOCUMENT_LOADERS[config.dataset](config)
    resources = _resources_for(config)
    working = document.tokens.copy()
    results = {}
    for name in ("native", "ud", "lexical_norms"):
        result = PROVIDERS[name].runner(
            document, working.copy(), config.provider_options.get(name, {}), resources
        )
        results[name] = result
        additions = result.token_features.drop(columns="token_id", errors="ignore")
        working = pd.concat([working, additions], axis=1)
    missing = set(OUTPUT_COLUMNS) - set(working)
    if missing:
        raise ValueError(f"{document.stimulus_id} is missing generated columns {sorted(missing)}")

    output = Path(config.output_dir) / document.dataset / document.stimulus_id
    token_table = _load_existing_tokens(output, document.tokens)
    token_table = token_table.drop(columns=list(OUTPUT_COLUMNS), errors="ignore")
    for column in OUTPUT_COLUMNS:
        token_table[column] = working[column].to_numpy()
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "tokens.tsv.tmp"
    token_table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(output / "tokens.tsv")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    owners = {
        **{column: "native" for column in NATIVE_COLUMNS},
        **{column: "ud" for column in UD_COLUMNS},
        **{column: "lexical_norms" for column in LEXICAL_COLUMNS},
    }
    for provider, result in results.items():
        manifest.setdefault("providers", {})[provider] = {
            "version": PROVIDERS[provider].version, **result.metadata
        }
    for column, provider in owners.items():
        manifest.setdefault("feature_providers", {})[column] = provider
        manifest.setdefault("feature_coverage", {})[column] = {
            "available": int(token_table[column].notna().sum()),
            "total": int(len(token_table)),
        }
    manifest.setdefault("focused_updates", {})["fmri_interpretable_features"] = {
        "providers": ["native", "ud", "lexical_norms"],
        "features": list(OUTPUT_COLUMNS),
    }
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, default=str))
    temporary_manifest.replace(manifest_path)
    print(f"Wrote interpretable features: {document.stimulus_id}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/language_features/fmri.yml")
    args = parser.parse_args()
    config = load_config(args.config, {}, LanguageFeatureGenerationConfig)
    if config.dataset != "fmri":
        raise ValueError("This focused generator is scoped to fMRI")
    for item in expand_stimulus_configs(config):
        _write_features(item)


if __name__ == "__main__":
    main()
