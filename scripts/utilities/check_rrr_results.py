#!/usr/bin/env python
"""Report missing or incomplete RRR artifacts in the configured result matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from scripts.run_rrr import Artifact, Config
from src.core.artifacts import normalize_dataset, normalized_config
from src.utils.shared.config import (
    expand_fmri_layer_configs,
    expand_per_model_layer_configs,
    load_config,
)


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
MODES = ("internal", "leave_one_subject_out", "cross_dataset")


@dataclass(frozen=True)
class ExpectedArtifact:
    config_path: Path
    mode: str
    dataset: str
    model: str
    layer: int | None
    source_dataset: str | None
    leaf: Path
    configuration: dict

    @property
    def label(self) -> str:
        layer = "none" if self.layer is None else str(self.layer)
        source = f" from {self.source_dataset}" if self.source_dataset else ""
        return f"{self.mode}\t{self.dataset}{source}\t{self.model}\tlayer-{layer}"


@dataclass(frozen=True)
class Failure:
    status: str
    expected: ExpectedArtifact
    detail: str


def _configuration_differences(expected: dict, actual: dict) -> list[str]:
    """Return differing config paths, including keys present on only one side."""
    differences = []

    def compare(expected_value, actual_value, path: str) -> None:
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            for key in sorted(expected_value.keys() | actual_value.keys()):
                child_path = f"{path}.{key}" if path else key
                if key not in expected_value:
                    differences.append(child_path)
                elif key not in actual_value:
                    differences.append(child_path)
                else:
                    compare(expected_value[key], actual_value[key], child_path)
        elif expected_value != actual_value:
            differences.append(path)

    compare(expected, actual, "")
    return differences


def _config_paths(
    repo_root: Path, mode: str, dataset: str, model: str
) -> list[Path]:
    if mode == "internal":
        relative = {
            "podcast": f"configs/rrr/{model}/cross_model_layer.yml",
            "cars-2": (
                f"configs/brain_treebank/cars_2/rrr/{model}/cross_model_layer.yml"
            ),
            "fmri": f"configs/fmri/within_fmri_cross_model_layer/{model}.yml",
        }[dataset]
        return [repo_root / relative]
    if mode == "leave_one_subject_out":
        relative = {
            "podcast": f"configs/rrr/{model}/leave_one_subject_out.yml",
            "cars-2": (
                f"configs/brain_treebank/cars_2/rrr/{model}/"
                "leave_one_subject_out.yml"
            ),
            "fmri": f"configs/fmri/leave_one_subject_out/{model}.yml",
        }[dataset]
        return [repo_root / relative]
    return [
        repo_root / f"configs/cross_dataset/{dataset}/from_{source}/{model}.yml"
        for source in DATASETS
        if source != dataset
    ]


def _expanded_configs(path: Path):
    facade = load_config(str(path), {}, Config)
    selected = facade.selected_config()
    if facade.modality == "fmri":
        return expand_fmri_layer_configs(selected)
    return expand_per_model_layer_configs(selected)


def _expected_artifact(
    config, config_path: Path, results_root: Path
) -> ExpectedArtifact:
    if hasattr(config, "encoding"):
        encoding = config.encoding
        stimulus = encoding.per_dataset_config.get("movie")
        context = SimpleNamespace(
            stimulus_id=stimulus,
            rereference=encoding.per_dataset_config.get("rereference"),
        )
        artifact = Artifact.from_ieeg_config(config, context)
        source = config.ieeg_rrr_source or config.fmri_rrr_source
    else:
        artifact = Artifact.from_fmri_config(config)
        source = config.ieeg_rrr_source
    artifact.results_root = results_root
    source_dataset = normalize_dataset(source["dataset"]) if source else None
    return ExpectedArtifact(
        config_path=config_path,
        mode=artifact.mode,
        dataset=artifact.dataset,
        model=artifact.model,
        layer=artifact.layer,
        source_dataset=source_dataset,
        leaf=artifact._leaf(),
        configuration=normalized_config(config),
    )


def expected_artifacts(
    repo_root: Path,
    results_root: Path,
    *,
    models: list[str],
    modes: list[str],
    datasets: list[str],
) -> list[ExpectedArtifact]:
    expected = []
    for mode in modes:
        for dataset in datasets:
            for model in models:
                for path in _config_paths(repo_root, mode, dataset, model):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing RRR config: {path}")
                    for config in _expanded_configs(path):
                        expected.append(_expected_artifact(config, path, results_root))
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
        config_differences = []
        unreadable = []
        for variant in index.get("variants", {}):
            manifest_path = item.leaf / variant / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                unreadable.append(f"{variant}: {error}")
                continue
            saved_configuration = normalized_config(
                manifest.get("configuration", {}), drop_paths=False
            )
            if saved_configuration == item.configuration:
                matches.append((variant, manifest))
            else:
                config_differences.extend(
                    (variant, difference)
                    for difference in _configuration_differences(
                        item.configuration, saved_configuration
                    )
                )

        if not matches:
            unique_differences = sorted(
                {difference for _, difference in config_differences}
            )
            difference_detail = (
                "; mismatched config fields: " + ", ".join(unique_differences)
                if unique_differences
                else ""
            )
            if unreadable:
                detail = (
                    "no artifact matches the current config; unreadable unrelated "
                    "variant manifest(s): " + "; ".join(unreadable)
                    + difference_detail
                )
            else:
                detail = "no artifact matches the current config" + difference_detail
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
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--missing-or-incomplete",
        action="store_true",
        help=(
            "print only runs with no artifact identity or a matched variant "
            "marked incomplete, ignoring config mismatches"
        ),
    )
    args = parser.parse_args()

    try:
        expected = expected_artifacts(
            args.repo_root.resolve(),
            args.results_root.resolve(),
            models=args.models,
            modes=args.modes,
            datasets=args.datasets,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    failures = find_failures(expected)
    if args.missing_or_incomplete:
        failures = [
            failure
            for failure in failures
            if failure.detail == "artifact identity does not exist"
            or failure.detail.startswith("variant(s) not marked complete:")
        ]
    for failure in failures:
        if args.missing_or_incomplete:
            print(f"{failure.status}\t{failure.expected.label}")
        else:
            print(f"{failure.status}\t{failure.expected.label}\t{failure.detail}")
    if failures:
        if args.missing_or_incomplete:
            raise SystemExit(1)
        raise SystemExit(
            f"Found {len(failures)} missing or incomplete RRR artifacts "
            f"out of {len(expected)} expected."
        )
    print(f"All {len(expected)} expected RRR artifacts are complete.")


if __name__ == "__main__":
    main()
