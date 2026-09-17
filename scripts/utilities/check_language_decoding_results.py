#!/usr/bin/env python
"""Report missing or incomplete directed cross-dataset decoding artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.run_language_decoding import Artifact
from scripts.run_language_decoding_batch import build_configs
from src.core.artifacts import normalize_dataset, normalize_model, normalized_config


MODELS = (
    "gpt2",
    "gpt2-xl",
    "llama3-8b",
    "gpt-j-6b",
    "roberta-large",
    "random-gpt2",
    "glove",
)
DATASETS = ("podcast", "cars-2", "fmri")


@dataclass(frozen=True)
class ExpectedArtifact:
    config_path: Path
    source_dataset: str
    target_dataset: str
    model: str
    layer: int | None
    label_offset: int
    leaf: Path
    configuration: dict

    @property
    def label(self) -> str:
        layer = "none" if self.layer is None else str(self.layer)
        return (
            f"{self.source_dataset} -> {self.target_dataset}\t{self.model}"
            f"\tlayer-{layer}\tlabel-offset-{self.label_offset}"
        )


@dataclass(frozen=True)
class Failure:
    status: str
    expected: ExpectedArtifact
    detail: str


def _manifest_path(repo_root: Path, source: str, target: str) -> Path:
    return (
        repo_root
        / "configs"
        / "language_decoding"
        / "cross_dataset"
        / target
        / f"from_{source}.yml"
    )


def _layer(config) -> int | None:
    return config.encoding.per_model_config.get("layer")


def _artifact(config: object, results_root: Path) -> Artifact:
    encoding = config.encoding
    model_dir = encoding.per_model_config.get("model_dir", encoding.model)
    identity = {
        "dataset": normalize_dataset(
            encoding.dataset, encoding.per_dataset_config.get("movie")
        ),
        "offset": encoding.offset,
        "model": normalize_model(
            model_dir, glove_dim=encoding.per_model_config.get("dim")
        ),
        "layer": _layer(config),
    }
    # fMRI identities come from the target RRR Artifact; the other modes derive
    # the same identity directly from their EncodingConfig.
    endpoint = SimpleNamespace(identity=identity)
    artifact = Artifact.from_config(config, source=endpoint, target=endpoint)
    artifact.results_root = results_root
    return artifact


def expected_artifacts(
    repo_root: Path,
    results_root: Path,
    *,
    models: list[str],
    datasets: list[str],
) -> list[ExpectedArtifact]:
    expected = []
    for target in datasets:
        for source in datasets:
            if source == target:
                continue
            path = _manifest_path(repo_root, source, target)
            if not path.is_file():
                raise FileNotFoundError(f"Missing language-decoding config: {path}")
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            offsets = list(payload.get("label_offsets", [0]))
            for model in models:
                for base_config in build_configs(str(path), model):
                    for offset in offsets:
                        config = deepcopy(base_config)
                        config.label_offset = int(offset)
                        config.validate()
                        artifact = _artifact(config, results_root)
                        expected.append(
                            ExpectedArtifact(
                                config_path=path,
                                source_dataset=source,
                                target_dataset=target,
                                model=model,
                                layer=_layer(config),
                                label_offset=int(offset),
                                leaf=artifact._leaf(),
                                configuration=normalized_config(config),
                            )
                        )
    return expected


def find_failures(expected: list[ExpectedArtifact]) -> list[Failure]:
    failures = []
    for item in expected:
        index_path = item.leaf / "manifest.json"
        if not index_path.is_file():
            failures.append(Failure("MISSING", item, "artifact identity does not exist"))
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(Failure("INCOMPLETE", item, f"unreadable index: {error}"))
            continue

        matches = []
        unreadable = []
        for variant in index.get("variants", {}):
            manifest_path = item.leaf / variant / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                unreadable.append(f"{variant}: {error}")
                continue
            if normalized_config(
                manifest.get("configuration", {}), drop_paths=False
            ) == item.configuration:
                matches.append((variant, manifest))

        if not matches:
            detail = "no artifact matches the current config"
            if unreadable:
                detail += "; unreadable unrelated variant manifest(s): " + "; ".join(
                    unreadable
                )
            failures.append(Failure("MISSING", item, detail))
            continue
        incomplete = [
            variant
            for variant, manifest in matches
            if not manifest.get("complete", False)
        ]
        if incomplete:
            failures.append(
                Failure(
                    "INCOMPLETE",
                    item,
                    "variant(s) not marked complete: " + ", ".join(incomplete),
                )
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    args = parser.parse_args()

    try:
        expected = expected_artifacts(
            args.repo_root.resolve(),
            args.results_root.resolve(),
            models=args.models,
            datasets=args.datasets,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    failures = find_failures(expected)
    for failure in failures:
        print(f"{failure.status}\t{failure.expected.label}\t{failure.detail}")
    if failures:
        raise SystemExit(
            f"Found {len(failures)} missing or incomplete cross-dataset decoding "
            f"artifacts out of {len(expected)} expected."
        )
    print(
        f"All {len(expected)} expected cross-dataset decoding artifacts are complete."
    )


if __name__ == "__main__":
    main()
