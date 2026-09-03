#!/usr/bin/env python
"""Verify that every significance Artifact needed by final RRR jobs is complete."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_significance import Artifact
from src.core.config import SignificanceSelectionConfig
from src.utils.rrr.significance import load_significance_selection


MODEL_LAYERS = {
    "gpt2": ("gpt2", [0, 3, 6, 9, 12]),
    "gpt2-xl": ("gpt2-xl-batch", [0, 12, 24, 36, 48]),
    "gpt-j-6b": ("gpt-j-6b", [0, 7, 14, 21, 28]),
    "roberta-large": ("roberta-large", [0, 6, 12, 18, 24]),
    "llama3-8b": ("llama3-8b", [0, 8, 16, 24, 32]),
    "random-gpt2": ("random-gpt2", [0, 3, 6, 9, 12]),
    "glove": ("glove-300", [None]),
}

DATASETS = {
    "podcast": ("sig_lags", None, list(range(1, 10)), "ieeg"),
    "brain-treebank-cars-2": (
        "sig_lags", "rereference-laplacian", [3, 7, 10], "ieeg",
    ),
    "deep-fmri": (
        "held_out_session", None,
        [f"UTS{number:02d}" for number in range(1, 10)], "fmri",
    ),
}


def missing_artifacts(results_root: str | Path, models: list[str]):
    """Return ``(dataset, model, layer)`` records for incomplete artifacts."""
    missing = []
    for model_key in models:
        if model_key not in MODEL_LAYERS:
            raise ValueError(f"unsupported model key: {model_key}")
        model, layers = MODEL_LAYERS[model_key]
        for dataset, (mode, variant, _, _) in DATASETS.items():
            for layer in layers:
                artifact = Artifact(
                    mode=mode, dataset=dataset, offset=0, model=model,
                    layer=layer, variant=variant, results_root=results_root,
                )
                try:
                    complete = artifact.manifest().get("complete", False)
                except (FileNotFoundError, RuntimeError, ValueError):
                    complete = False
                if not complete:
                    missing.append((dataset, model_key, layer))
    return missing


def incomplete_artifacts(results_root: str | Path, models: list[str]) -> list[str]:
    """Return readable identities for missing, ambiguous, or incomplete Artifacts."""
    failures = []
    for model_key in models:
        if model_key not in MODEL_LAYERS:
            failures.append(f"unsupported model key: {model_key}")
            continue
        model, layers = MODEL_LAYERS[model_key]
        for dataset, (mode, variant, _, _) in DATASETS.items():
            for layer in layers:
                artifact = Artifact(
                    mode=mode,
                    dataset=dataset,
                    offset=0,
                    model=model,
                    layer=layer,
                    variant=variant,
                    results_root=results_root,
                )
                label = (
                    f"{dataset}/{model}/"
                    f"layer-{'none' if layer is None else layer}"
                )
                try:
                    manifest = artifact.manifest()
                except (FileNotFoundError, RuntimeError, ValueError) as error:
                    failures.append(f"{label}: {error}")
                    continue
                if not manifest.get("complete", False):
                    failures.append(f"{label}: Artifact is incomplete")
    if failures:
        return failures

    for dataset, (mode, variant, subjects, modality) in DATASETS.items():
        artifacts = []
        for model_key in models:
            model, layers = MODEL_LAYERS[model_key]
            artifacts.append({
                "mode": mode,
                "dataset": dataset,
                "offset": 0,
                "model": model,
                "layer": layers,
                **({"variant": variant} if variant else {}),
            })
        selection = SignificanceSelectionConfig(
            mode="cross-model-layer",
            reference_artifact={
                "mode": mode,
                "dataset": dataset,
                "offset": 0,
                "model": "gpt2",
                "layer": 0,
                **({"variant": variant} if variant else {}),
            },
            artifacts=artifacts,
        )
        try:
            load_significance_selection(
                selection,
                results_root=str(results_root),
                subjects=subjects,
                modality=modality,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            failures.append(f"{dataset} significance union is incompatible: {error}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--missing", action="store_true",
        help="Print incomplete dataset/model/layer records as tab-separated rows.",
    )
    args = parser.parse_args()
    if args.missing:
        try:
            rows = missing_artifacts(args.results_root, args.models)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        for dataset, model, layer in rows:
            print(f"{dataset}\t{model}\t{'none' if layer is None else layer}")
        return
    failures = incomplete_artifacts(args.results_root, args.models)
    if failures:
        details = "\n".join(f"  - {item}" for item in failures)
        raise SystemExit(
            "Required significance Artifacts are not ready:\n" + details
        )
    print("All required significance Artifacts are complete.")


if __name__ == "__main__":
    main()
