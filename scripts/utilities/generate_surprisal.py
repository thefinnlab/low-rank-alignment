#!/usr/bin/env python
"""Generate only standardized GPT-2 surprisal without rerunning other providers."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.utilities.generate_language_features import expand_stimulus_configs
from src.core.config import LanguageFeatureGenerationConfig
from src.utils.shared.config import load_config, parse_override_args
from src.utils.language_features.core import (
    DOCUMENT_LOADERS,
    EMPTY_SIDECAR_COLUMNS,
    PROVIDERS,
    SCHEMA_VERSION,
    hydrate_provider_resources,
    stage_language_resources,
)


def _load_existing_tokens(output: Path, canonical: pd.DataFrame) -> pd.DataFrame:
    path = output / "tokens.tsv"
    if not path.exists():
        return canonical.copy()
    existing = pd.read_csv(path, sep="\t", keep_default_na=False)
    if len(existing) != len(canonical):
        raise ValueError(
            f"Existing token count differs at {output}: {len(existing)} vs {len(canonical)}"
        )
    if not np.array_equal(
        existing["token_id"].to_numpy(dtype=int),
        canonical["token_id"].to_numpy(dtype=int),
    ) or not np.array_equal(
        existing["word"].astype(str).to_numpy(),
        canonical["word"].astype(str).to_numpy(),
    ):
        raise ValueError(f"Existing tokens do not match the canonical document at {output}")
    return existing


def _write_surprisal(config: LanguageFeatureGenerationConfig) -> Path:
    hydrate_provider_resources(config)
    # Register built-in providers.
    from src.utils.language_features import providers as _builtins  # noqa: F401

    document = DOCUMENT_LOADERS[config.dataset](config)
    resource_name = config.provider_options["gpt2_surprisal"].get(
        "model_resource", "gpt2_small"
    )
    resource_config = deepcopy(config)
    resource_config.resources = {resource_name: config.resources[resource_name]}
    resources = stage_language_resources(resource_config, download=False)
    resource = resources[resource_name]
    output = Path(config.output_dir) / document.dataset / document.stimulus_id
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    update = manifest.get("focused_updates", {}).get("gpt2_surprisal", {})
    existing_path = output / "tokens.tsv"
    if (
        existing_path.exists()
        and update.get("resource_sha256") == resource["sha256"]
        and {"surprisal", "standardized_entropy"}
        <= set(pd.read_csv(existing_path, sep="\t", nrows=1).columns)
    ):
        print(f"Skipping completed surprisal: {document.dataset}/{document.stimulus_id}")
        return output

    spec = PROVIDERS["gpt2_surprisal"]
    result = spec.runner(
        document,
        document.tokens.copy(),
        config.provider_options["gpt2_surprisal"],
        resources,
    )
    features = result.token_features
    if len(features) != len(document.tokens):
        raise ValueError("GPT-2 surprisal returned a misaligned token count")
    token_table = _load_existing_tokens(output, document.tokens)
    token_table = token_table.drop(
        columns=["surprisal", "standardized_entropy"], errors="ignore"
    )
    token_table["surprisal"] = features["surprisal"].to_numpy(dtype=float)
    token_table["standardized_entropy"] = features[
        "standardized_entropy"
    ].to_numpy(dtype=float)

    output.mkdir(parents=True, exist_ok=True)
    temporary_tokens = output / "tokens.tsv.tmp"
    token_table.to_csv(temporary_tokens, sep="\t", index=False)
    temporary_tokens.replace(output / "tokens.tsv")
    for filename, columns in EMPTY_SIDECAR_COLUMNS.items():
        sidecar = output / filename
        if not sidecar.exists():
            pd.DataFrame(columns=columns).to_csv(sidecar, sep="\t", index=False)

    manifest.setdefault("schema_version", SCHEMA_VERSION)
    manifest.update(
        {
            "dataset": document.dataset,
            "stimulus_id": document.stimulus_id,
            "n_tokens": len(token_table),
        }
    )
    manifest.setdefault("providers", {})["gpt2_surprisal"] = {
        "version": spec.version,
        **result.metadata,
    }
    manifest.setdefault("feature_providers", {})["surprisal"] = "gpt2_surprisal"
    manifest.setdefault("feature_providers", {})[
        "standardized_entropy"
    ] = "gpt2_surprisal"
    manifest.setdefault("feature_coverage", {})["surprisal"] = {
        "available": int(token_table["surprisal"].notna().sum()),
        "total": int(len(token_table)),
    }
    manifest.setdefault("feature_coverage", {})["standardized_entropy"] = {
        "available": int(token_table["standardized_entropy"].notna().sum()),
        "total": int(len(token_table)),
    }
    manifest.setdefault("focused_updates", {})["gpt2_surprisal"] = {
        "resource_sha256": resource["sha256"],
        **result.metadata,
    }
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, default=str))
    temporary_manifest.replace(manifest_path)
    print(f"Wrote surprisal: {document.dataset}/{document.stimulus_id}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    args, unknown = parser.parse_known_args()
    overrides = parse_override_args(unknown)
    for config_path in args.config:
        config = load_config(config_path, overrides, LanguageFeatureGenerationConfig)
        if "gpt2_surprisal" not in config.providers:
            raise ValueError(f"{config_path} does not configure gpt2_surprisal")
        for item in expand_stimulus_configs(config):
            _write_surprisal(item)


if __name__ == "__main__":
    main()
