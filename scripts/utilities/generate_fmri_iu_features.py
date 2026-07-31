#!/usr/bin/env python
"""Populate fMRI word-level IU structure from existing aligned boundaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.utilities.generate_language_features import expand_stimulus_configs
from scripts.utilities.generate_surprisal import _load_existing_tokens
from src.core.config import LanguageFeatureGenerationConfig
from src.utils.config_utils import load_config
from src.utils.language_features import DOCUMENT_LOADERS, PROVIDERS, hydrate_provider_resources


IU_COLUMNS = ("iu_position", "iu_onset", "iu_end", "iu_length")


def _write_iu_features(config: LanguageFeatureGenerationConfig) -> Path:
    hydrate_provider_resources(config)
    from src.utils import language_feature_providers as _builtins  # noqa: F401

    document = DOCUMENT_LOADERS[config.dataset](config)
    if "iu_id" not in document.native_tokens:
        raise ValueError(f"No IU alignment is available for {document.stimulus_id}")
    result = PROVIDERS["native"].runner(document, document.tokens.copy(), {}, {})
    missing = set(IU_COLUMNS) - set(result.token_features)
    if missing:
        raise ValueError(
            f"Native provider did not produce IU features for {document.stimulus_id}: "
            f"{sorted(missing)}"
        )

    output = Path(config.output_dir) / document.dataset / document.stimulus_id
    token_table = _load_existing_tokens(output, document.tokens)
    token_table = token_table.drop(columns=list(IU_COLUMNS), errors="ignore")
    for column in IU_COLUMNS:
        token_table[column] = result.token_features[column].to_numpy()

    output.mkdir(parents=True, exist_ok=True)
    temporary_tokens = output / "tokens.tsv.tmp"
    token_table.to_csv(temporary_tokens, sep="\t", index=False)
    temporary_tokens.replace(output / "tokens.tsv")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("providers", {}).setdefault("native", {"version": "1"})
    feature_providers = manifest.setdefault("feature_providers", {})
    coverage = manifest.setdefault("feature_coverage", {})
    for column in IU_COLUMNS:
        feature_providers[column] = "native"
        coverage[column] = {
            "available": int(token_table[column].notna().sum()),
            "total": int(len(token_table)),
        }
    manifest.setdefault("focused_updates", {})["fmri_iu_features"] = {
        "source": "aligned_psst_boundaries",
        "features": list(IU_COLUMNS),
        "iu_count": int(token_table["iu_onset"].sum()),
    }
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, default=str))
    temporary_manifest.replace(manifest_path)
    print(
        f"Wrote IU features: {document.stimulus_id} "
        f"({int(token_table['iu_onset'].sum())} IUs)"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/language_features/fmri.yml")
    args = parser.parse_args()
    config = load_config(args.config, {}, LanguageFeatureGenerationConfig)
    if config.dataset != "fmri":
        raise ValueError("IU feature generation is scoped to fMRI")
    for item in expand_stimulus_configs(config):
        _write_iu_features(item)


if __name__ == "__main__":
    main()
