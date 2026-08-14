#!/usr/bin/env python3
"""Copy legacy analysis results into script-specific Artifact trees.

Dry-run is the default. Pass ``--apply`` only after reviewing the emitted mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

from src.core.artifacts import normalize_dataset, normalize_model


CANONICAL_SCRIPTS = {"run_significance", "run_rrr", "run_cross_dataset_rrr"}
SIGNIFICANCE_MARKERS = {"all_subject_sig.csv", "significance_detailed.csv",
                        "cluster_null_distribution.npz"}
RRR_MARKERS = {"grid_results.csv", "summary.csv"}


@dataclass(frozen=True)
class Mapping:
    source: str
    producer: str
    mode: str
    dataset: str
    offset: int
    model: str
    layer: int | None
    variant: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_leaf(root: Path, item: Mapping) -> Path:
    return (root / item.producer / item.mode / item.dataset
            / f"offset-{item.offset}" / item.model
            / f"layer-{'none' if item.layer is None else item.layer}"
            / "variants" / item.variant)


def is_canonical(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return bool(relative.parts and relative.parts[0] in CANONICAL_SCRIPTS)


def discover_ambiguous(legacy_root: Path, results_root: Path) -> list[dict]:
    """Return ready-to-edit entries for roots that cannot be safely identified."""
    entries = []
    for path in sorted(item for item in legacy_root.rglob("*") if item.is_dir()):
        if is_canonical(path, results_root):
            continue
        names = {item.name for item in path.iterdir() if item.is_file()}
        producers = []
        if names & SIGNIFICANCE_MARKERS or any(path.glob("subjects/*/significance.csv")):
            producers.append("run_significance")
        if names & RRR_MARKERS or any(path.glob("fold_*/A.npy")):
            producers.append("run_rrr")
        if len(producers) != 1:
            if producers:
                entries.append({"source": str(path), "producer": None, "mode": None,
                                "dataset": None, "offset": None, "model": None,
                                "layer": None, "variant": None,
                                "reason": "shared or conflicting producer markers"})
            continue
        entries.append({"source": str(path), "producer": producers[0], "mode": None,
                        "dataset": None, "offset": None, "model": None,
                        "layer": None, "variant": None,
                        "reason": "identity requires confirmation"})
    return entries


def _nearest_config(path: Path, boundary: Path) -> dict | None:
    current = path
    while True:
        for name in ("config.yml", "effective_config.yml", "config.yaml"):
            candidate = current / name
            if candidate.is_file():
                value = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                if isinstance(value, dict):
                    return value
        if current == boundary or current.parent == current:
            return None
        current = current.parent


def _mapping_from_config(path: Path, producer: str, config: dict) -> Mapping | None:
    """Infer only identities explicitly represented by a saved effective config."""
    if producer == "run_cross_dataset_rrr" or "fmri" in config:
        model = config.get("model_dir")
        layer = config.get("layer")
        if model is None or isinstance(layer, list):
            return None
        mode = ("held_out_session" if producer == "run_significance"
                else config.get("source_mode"))
        dataset, offset = "deep-fmri", 0
    else:
        encoding = config.get("encoding", config)
        per_model = encoding.get("per_model_config", {})
        raw_model = per_model.get("model_dir", encoding.get("model"))
        if raw_model is None:
            return None
        try:
            model = normalize_model(raw_model, glove_dim=per_model.get("dim"))
        except (TypeError, ValueError):
            return None
        layer = None if encoding.get("model") == "glove" else per_model.get("layer")
        if isinstance(layer, list):
            return None
        raw_dataset = encoding.get("dataset", "podcast")
        stimulus = encoding.get("per_dataset_config", {}).get("movie")
        try:
            dataset = normalize_dataset(raw_dataset, stimulus)
        except ValueError:
            return None
        offset = int(encoding.get("offset", 0))
        mode = "sig_lags" if producer == "run_significance" else config.get("mode", "cross_subject")
        if hasattr(mode, "value"):
            mode = mode.value
    if not mode or layer is None and not str(model).startswith("glove-"):
        return None
    identity = {"source": str(path), "producer": producer, "mode": str(mode),
                "dataset": dataset, "offset": offset, "model": str(model),
                "layer": layer}
    variant = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Mapping(**identity, variant=variant)


def discover_mappings(legacy_root: Path, results_root: Path) -> tuple[list[Mapping], list[dict]]:
    mappings, ambiguous = [], []
    for path in sorted(item for item in legacy_root.rglob("*") if item.is_dir()):
        if is_canonical(path, results_root):
            continue
        names = {item.name for item in path.iterdir() if item.is_file()}
        producers = []
        if names & SIGNIFICANCE_MARKERS or any(path.glob("subjects/*/significance.csv")):
            producers.append("run_significance")
        if names & RRR_MARKERS or any(path.glob("fold_*/A.npy")):
            producers.append("run_rrr")
        if not producers:
            continue
        if len(producers) != 1:
            ambiguous.append({"source": str(path), "producer": None, "mode": None,
                              "dataset": None, "offset": None, "model": None,
                              "layer": None, "variant": None,
                              "reason": "shared or conflicting producer markers"})
            continue
        config = _nearest_config(path, legacy_root)
        producer = producers[0]
        if producer == "run_rrr" and config and "fmri" in config:
            producer = "run_cross_dataset_rrr"
        inferred = _mapping_from_config(path, producer, config or {})
        if inferred is None:
            ambiguous.append({"source": str(path), "producer": producer,
                              "mode": None, "dataset": None, "offset": None,
                              "model": None, "layer": None, "variant": None,
                              "reason": "identity requires confirmation"})
        else:
            mappings.append(inferred)
    return mappings, ambiguous


def load_mappings(path: Path | None) -> list[Mapping]:
    if path is None:
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Mapping(**item) for item in raw.get("mappings", [])]


def copy_mapping(item: Mapping, results_root: Path, *, apply: bool) -> dict:
    source = Path(item.source)
    destination = canonical_leaf(results_root, item) / "legacy"
    operations = []
    conflicts = []
    for src in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = src.relative_to(source)
        if item.producer == "run_significance" and (
            relative.parts[0] in {"rrr", "pca", "per_subject_rrr", "per_region_rrr",
                                  "temporal_window_rrr", "loso_rrr", "multi_view_rrr"}
            or src.name in RRR_MARKERS
        ):
            continue
        if item.producer in {"run_rrr", "run_cross_dataset_rrr"} and (
            src.name in SIGNIFICANCE_MARKERS
            or (relative.parts and relative.parts[0] == "subjects"
                and src.name == "significance.csv")
        ):
            continue
        dst = destination / relative
        if dst.exists():
            if sha256(src) == sha256(dst):
                operations.append({"action": "skip-identical", "source": str(src),
                                   "destination": str(dst), "sha256": sha256(src)})
            else:
                conflicts.append({"source": str(src), "destination": str(dst)})
        else:
            operations.append({"action": "copy", "source": str(src),
                               "destination": str(dst), "sha256": sha256(src)})
    if conflicts:
        raise FileExistsError(f"Differing-content migration conflicts: {conflicts}")
    if apply:
        for operation in operations:
            if operation["action"] != "copy":
                continue
            src, dst = Path(operation["source"]), Path(operation["destination"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as temporary:
                temp_path = Path(temporary.name)
            try:
                shutil.copy2(src, temp_path)
                if sha256(temp_path) != operation["sha256"]:
                    raise IOError(f"Checksum verification failed while copying {src}")
                temp_path.replace(dst)
            finally:
                temp_path.unlink(missing_ok=True)
        migration = canonical_leaf(results_root, item) / "migration.json"
        migration.parent.mkdir(parents=True, exist_ok=True)
        migration.write_text(json.dumps({"mapping": asdict(item), "operations": operations,
                                         "verified": True}, indent=2), encoding="utf-8")
    return {"mapping": asdict(item), "operations": operations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", default="results")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--mapping")
    parser.add_argument("--mapping-template")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    legacy_root, results_root = Path(args.legacy_root), Path(args.results_root)
    inferred, ambiguous = discover_mappings(legacy_root, results_root)
    template = {"mappings": ambiguous}
    if args.mapping_template:
        Path(args.mapping_template).write_text(
            yaml.safe_dump(template, sort_keys=False), encoding="utf-8"
        )
    print(yaml.safe_dump(template, sort_keys=False))
    explicit = load_mappings(Path(args.mapping) if args.mapping else None)
    reports = [copy_mapping(item, results_root, apply=args.apply)
               for item in [*inferred, *explicit]]
    print(json.dumps({"applied": args.apply, "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
