#!/usr/bin/env python
"""Update recurrence labels without rerunning unrelated language providers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utilities.generate_language_features import expand_stimulus_configs
from scripts.utilities.generate_surprisal import _load_existing_tokens
from src.core.config import LanguageFeatureGenerationConfig
from src.utils.shared.config import load_config
from src.utils.language_features.core import DOCUMENT_LOADERS, PROVIDERS, hydrate_provider_resources

RECURRENCE_COLUMNS = (
    "immediate_word_repetition", "word_seen_before", "prior_word_count",
    "distance_from_previous_occurrence", "word_in_previous_5",
)


def update_recurrence(config: LanguageFeatureGenerationConfig) -> Path:
    hydrate_provider_resources(config)
    from src.utils.language_features import providers as _builtins  # noqa: F401

    document = DOCUMENT_LOADERS[config.dataset](config)
    output = Path(config.output_dir) / document.dataset / document.stimulus_id
    table = _load_existing_tokens(output, document.tokens)
    result = PROVIDERS["disfluency"].runner(document, table.reset_index(drop=True), {}, {})
    for column in RECURRENCE_COLUMNS:
        table[column] = result.token_features[column].to_numpy()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for column in RECURRENCE_COLUMNS:
        manifest.setdefault("feature_providers", {})[column] = "disfluency"
        manifest.setdefault("feature_coverage", {})[column] = {
            "available": int(table[column].notna().sum()), "total": len(table),
        }
    manifest.setdefault("focused_updates", {})["recurrence"] = {
        "provider": "disfluency", "version": PROVIDERS["disfluency"].version,
        "features": list(RECURRENCE_COLUMNS),
    }
    temporary = output / "tokens.tsv.tmp"
    table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(output / "tokens.tsv")
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, default=str))
    temporary.replace(manifest_path)
    print(f"Updated recurrence: {document.dataset}/{document.stimulus_id} ({len(table)} words)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()
    for path in args.config:
        config = load_config(path, {}, LanguageFeatureGenerationConfig)
        for item in expand_stimulus_configs(config):
            update_recurrence(item)


if __name__ == "__main__":
    main()
