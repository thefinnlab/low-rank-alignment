#!/usr/bin/env python3
"""Copy legacy analysis results into script-specific Artifact trees.

Dry-run is the default. Pass ``--apply`` only after reviewing the emitted mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml
import joblib
import numpy as np
import pandas as pd

from src.core.artifacts import normalize_dataset, normalize_model
from scripts.run_significance import Artifact as SignificanceArtifact
from scripts.run_rrr import Artifact as RRRArtifact, UnitSelector


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
    producer = "run_rrr" if item.producer == "run_cross_dataset_rrr" else item.producer
    mode = "internal" if item.mode in {"cross_subject", "within_fmri"} else item.mode
    return (root / producer / mode / item.dataset
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
    """Convert one recognized legacy producer tree into a loadable Artifact."""
    source = Path(item.source)
    files = sorted(path for path in source.rglob("*") if path.is_file())
    inventory = {str(path.relative_to(source)): sha256(path) for path in files}
    operations = [{"action": "convert", "source": str(path), "sha256": inventory[
        str(path.relative_to(source))
    ]} for path in files]
    if not apply:
        return {"mapping": asdict(item), "operations": operations}

    mode = "internal" if item.mode in {"cross_subject", "within_fmri"} else item.mode
    cls = SignificanceArtifact if item.producer == "run_significance" else RRRArtifact
    artifact = cls(mode=mode, dataset=item.dataset, offset=item.offset,
                   model=item.model, layer=item.layer, results_root=results_root)
    config = _nearest_config(source, source) or {
        "legacy_source": str(source), "legacy_identity": asdict(item)
    }
    upstreams = _upstreams_from_config(config)
    artifact.begin(config=config, upstreams=upstreams, refresh=False)
    migration_selector = {"name": "legacy_migration"}
    if artifact.has_field("diagnostics", selector=migration_selector):
        prior = artifact.load_field("diagnostics", selector=migration_selector)
        if prior.get("inventory") != inventory:
            raise FileExistsError("Differing-content migration conflict for " + str(source))
        return {"mapping": asdict(item), "operations": [
            {**operation, "action": "skip-identical"} for operation in operations
        ], "artifact": artifact.identity_record()}

    artifact.save_field("effective_config", config)
    if isinstance(artifact, SignificanceArtifact):
        _convert_significance(source, artifact)
    else:
        artifact.save_field("upstream_identities", upstreams)
        _convert_rrr(source, artifact)
    artifact.save_field("diagnostics", {"source": str(source), "inventory": inventory},
                        selector=migration_selector)
    manifest = artifact.manifest()
    if isinstance(artifact, SignificanceArtifact):
        target_entries = manifest.get("entries", {}).get("target_significance", {}).values()
        selectors = [entry["selector"] for entry in target_entries]
        if selectors:
            artifact.mark_complete({"target_significance": selectors})
    else:
        required = {}
        for name in ("target_manifest", "full_metrics", "rank_metrics", "mean_results"):
            entries = manifest.get("entries", {}).get(name, {}).values()
            selectors = [entry["selector"] for entry in entries]
            if not selectors:
                break
            required[name] = selectors
        else:
            artifact.mark_complete(required)
    return {"mapping": asdict(item), "operations": operations,
            "artifact": artifact.identity_record()}


def _convert_significance(source: Path, artifact: SignificanceArtifact) -> None:
    metadata_path = source / "metadata.json"
    metadata = (json.loads(metadata_path.read_text()) if metadata_path.is_file()
                else {"method": artifact.mode, "migrated": True})
    artifact.save_field("run_metadata", metadata)
    detailed_path = source / "significance_detailed.csv"
    detailed = pd.read_csv(detailed_path) if detailed_path.is_file() else None
    null_path = source / "cluster_null_distribution.npz"
    null_payload = None
    if null_path.is_file():
        with np.load(null_path, allow_pickle=True) as loaded:
            null_payload = {key: loaded[key] for key in loaded.files}
        artifact.save_subject_field("null_distribution", "all", null_payload)

    subjects = set()
    for path in sorted(source.glob("subjects/*/significance.csv")):
        subject = path.parent.name
        frame = pd.read_csv(path).rename(columns={"voxel_index": "target_id", "sig": "significant"})
        frame["subject"] = str(subject)
        frame["target_kind"] = "voxel"
        artifact.save_subject_field("target_significance", subject, frame)
        details = {}
        for name in ("null_metadata.json", "voxel_filter.json"):
            candidate = path.parent / name
            if candidate.is_file():
                details[name.removesuffix(".json")] = json.loads(candidate.read_text())
        alpha = path.parent / "ridge_alphas.npz"
        if alpha.is_file():
            with np.load(alpha, allow_pickle=True) as loaded:
                details["ridge_alphas"] = {key: loaded[key] for key in loaded.files}
        artifact.save_subject_field("significance_details", subject, details)
        subjects.add(str(subject))

    if not subjects and null_payload is not None and {
        "subjects", "electrodes"
    }.issubset(null_payload):
        null_subjects = np.asarray(null_payload["subjects"])
        electrodes = np.asarray(null_payload["electrodes"]).astype(str)
        sig_path = source / "all_subject_sig.csv"
        selected = pd.read_csv(sig_path) if sig_path.is_file() else pd.DataFrame(
            columns=["subject", "elec"]
        )
        for subject in sorted(set(map(int, null_subjects))):
            ids = electrodes[null_subjects.astype(int) == subject]
            chosen = set(selected.loc[selected.subject == subject, "elec"].astype(str))
            frame = pd.DataFrame({
                "subject": str(subject), "target_id": ids, "target_kind": "electrode",
                "significant": [item in chosen for item in ids],
            })
            artifact.save_subject_field("target_significance", subject, frame)
            if detailed is not None:
                artifact.save_subject_field(
                    "significance_details", subject,
                    detailed[detailed.subject == subject].reset_index(drop=True),
                )
            subjects.add(str(subject))

    for path in sorted(source.glob("subjects/sub_*/encoding_results.joblib")):
        subject = str(int(path.parent.name.removeprefix("sub_")))
        artifact.save_subject_field("encoding_results", subject, joblib.load(path))


def _upstreams_from_config(config: dict) -> list[dict]:
    upstreams = []
    significance = config.get("significance_artifact")
    if isinstance(significance, dict):
        upstreams.append({"script": "run_significance", **significance})
    sources = []
    for name in ("rrr_source", "ieeg_rrr_source", "fmri_rrr_source"):
        if isinstance(config.get(name), dict):
            sources.append(config[name])
    for view in config.get("views", []) if isinstance(config.get("views"), list) else []:
        for name in ("source", "ieeg_rrr_source", "fmri_rrr_source"):
            if isinstance(view.get(name), dict):
                sources.append(view[name])
    for source in sources:
        identity = dict(source)
        if identity.get("mode") in {"cross_subject", "within_fmri"}:
            identity["mode"] = "internal"
        upstreams.append({"script": "run_rrr", **identity})
    return upstreams


def _unit_from_path(path: Path, source: Path) -> tuple[UnitSelector, int | None, int | None]:
    parts = path.relative_to(source).parts
    method = "pca" if any("pca" in part.lower() for part in parts) else "rrr"
    values = {"method": method}
    fold = rank = None
    for part in parts:
        if part.startswith("fold_") and part.removeprefix("fold_").isdigit():
            fold = int(part.removeprefix("fold_"))
        if part.startswith("k") and part.removeprefix("k").isdigit():
            rank = int(part.removeprefix("k"))
        if part.startswith("sub_") and part.removeprefix("sub_").isdigit():
            values["subject"] = int(part.removeprefix("sub_"))
        if part.startswith("held_out_sub_"):
            values["held_out_subject"] = int(part.removeprefix("held_out_sub_"))
    return UnitSelector(**values), fold, rank


def _npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _convert_rrr(source: Path, artifact: RRRArtifact) -> None:
    for path in sorted(source.rglob("A.npy")):
        unit, fold, rank = _unit_from_path(path, source)
        artifact.save_factors(unit=unit, fold=fold, rank=rank,
                              input_factors=np.load(path))
        b_path = path.with_name("B.npy")
        if b_path.is_file():
            artifact.save_field("target_factors", np.load(b_path),
                                selector=artifact._selector(unit, fold=fold, rank=rank))
    for path in sorted(source.rglob("full_model_metrics.npz")):
        unit, fold, _ = _unit_from_path(path, source)
        artifact.save_full_metrics(unit=unit, fold=fold, metrics=_npz(path))
    for path in sorted(source.rglob("metrics.npz")):
        if path.name == "full_model_metrics.npz":
            continue
        unit, fold, rank = _unit_from_path(path, source)
        if path.parent.name == "full":
            artifact.save_field("full_metrics", _npz(path))
        elif rank is not None:
            artifact.save_rank_metrics(unit=unit, fold=fold, rank=rank,
                                       metrics=_npz(path))
    for path in sorted(source.rglob("target_manifest.csv")):
        unit, _, _ = _unit_from_path(path, source)
        artifact.save_target_manifest(unit=unit, target_manifest=pd.read_csv(path))
    summary_paths = sorted(source.rglob("summary.csv")) or sorted(source.rglob("grid_results.csv"))
    for path in summary_paths:
        unit, _, _ = _unit_from_path(path, source)
        artifact.save_summary(unit=unit, summary=pd.read_csv(path))
    metadata = source / "metadata.json"
    if metadata.is_file():
        artifact.save_field("metadata", json.loads(metadata.read_text()))


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
