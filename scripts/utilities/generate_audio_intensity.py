#!/usr/bin/env python
"""Add Brain Treebank-style word intensity to Podcast and fMRI artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utilities.generate_language_features import expand_stimulus_configs
from scripts.utilities.generate_surprisal import _load_existing_tokens
from src.core.config import LanguageFeatureGenerationConfig
from src.utils.language_features.audio import mean_rms_for_words
from src.utils.language_features.core import DOCUMENT_LOADERS
from src.utils.shared.config import load_config


FEATURE = "intensity_rms"
SAMPLE_RATE = 48_000


def generate_intensity(config: LanguageFeatureGenerationConfig) -> Path:
    if config.dataset not in {"podcast", "fmri"}:
        raise ValueError("Audio intensity supports Podcast and fMRI")
    document = DOCUMENT_LOADERS[config.dataset](config)
    audio_path = document.media.get("audio")
    if not audio_path:
        raise FileNotFoundError(f"No audio for {document.dataset}/{document.stimulus_id}")

    output = Path(config.output_dir) / document.dataset / document.stimulus_id
    if not (output / "tokens.tsv").is_file() or not (output / "manifest.json").is_file():
        raise FileNotFoundError(f"Generate the language-feature artifact first: {output}")
    table = _load_existing_tokens(output, document.tokens)
    waveform, rate = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    table[FEATURE] = mean_rms_for_words(waveform, rate, table)
    del waveform

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("feature_providers", {})[FEATURE] = "audio_intensity"
    manifest.setdefault("feature_coverage", {})[FEATURE] = {
        "available": int(table[FEATURE].notna().sum()), "total": len(table),
    }
    manifest.setdefault("focused_updates", {})[FEATURE] = {
        "method": "mean_librosa_rms_frames",
        "audio": str(audio_path),
        "sample_rate": rate,
        "window_seconds_after_onset": 0.5,
        "frame_length": 2048,
        "hop_length": 512,
    }
    temporary = output / "tokens.tsv.tmp"
    table.to_csv(temporary, sep="\t", index=False)
    temporary.replace(output / "tokens.tsv")
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, default=str))
    temporary.replace(manifest_path)
    print(f"Wrote {FEATURE}: {document.dataset}/{document.stimulus_id} "
          f"({int(table[FEATURE].notna().sum())}/{len(table)} words)", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()
    for path in args.config:
        config = load_config(path, {}, LanguageFeatureGenerationConfig)
        for item in expand_stimulus_configs(config):
            generate_intensity(item)


if __name__ == "__main__":
    main()
