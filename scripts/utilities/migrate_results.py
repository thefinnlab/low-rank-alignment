#!/usr/bin/env python3
"""Copy legacy analysis results into script-specific Artifact trees.

Dry-run is the default. Pass ``--apply`` only after reviewing the emitted mapping.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml
import joblib
import numpy as np
import pandas as pd

from src.core.artifacts import normalize_dataset, normalize_model, readable_variant
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
    variant: str | None = None


IGNORED_LEGACY_ROOTS = {"rank_saturation", "subspace_comparisons"}


def canonical_leaf(root: Path, item: Mapping) -> Path:
    producer = "run_rrr" if item.producer == "run_cross_dataset_rrr" else item.producer
    mode = "internal" if item.mode in {"cross_subject", "within_fmri"} else item.mode
    return (root / producer / mode / item.dataset
            / f"offset-{item.offset}" / item.model
            / f"layer-{'none' if item.layer is None else item.layer}"
            / (item.variant or "default"))


def _migration_source_id(source: Path) -> str:
    """Return a readable diagnostic selector that is stable across root moves."""
    parts = source.parts[-3:]
    return "--".join(re.sub(r"[^A-Za-z0-9._-]", "-", part) for part in parts)


def is_canonical(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return bool(relative.parts and relative.parts[0] in CANONICAL_SCRIPTS)


def is_ignored(path: Path, root: Path) -> bool:
    """Return whether a legacy path belongs to a deliberately deferred workflow."""
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return False
    return bool(parts and (
        parts[0] in IGNORED_LEGACY_ROOTS or "multi_view_rrr" in parts
    ))


def _is_configless_cross_dataset(path: Path, producer: str, boundary: Path) -> bool:
    return (
        producer == "run_rrr"
        and any(part.startswith("cross_dataset_") for part in path.parts)
        and _config_for_source(path, producer, boundary) is None
    )


def discover_ambiguous(legacy_root: Path, results_root: Path) -> list[dict]:
    """Return ready-to-edit entries for roots that cannot be safely identified."""
    entries = []
    for path in sorted(item for item in legacy_root.rglob("*") if item.is_dir()):
        if is_canonical(path, results_root) or is_ignored(path, legacy_root):
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
        if _is_configless_cross_dataset(path, producers[0], legacy_root):
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
                    return _abstract_legacy_dependencies(value)
        if current == boundary or current.parent == current:
            return None
        current = current.parent


def _config_for_source(path: Path, producer: str, boundary: Path) -> dict | None:
    """Find the config belonging to a producer source, including split sig trees."""
    if producer == "run_significance":
        for name in ("config.yml", "effective_config.yml", "config.yaml"):
            candidate = path / "significance" / name
            if candidate.is_file():
                value = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                if isinstance(value, dict):
                    return _abstract_legacy_dependencies(value)
    return _nearest_config(path, boundary)


def _rrr_source_from_legacy_path(value: str | Path) -> dict:
    """Translate a legacy RRR directory into a path-free Artifact identity."""
    parts = Path(value).parts
    if "fmri" in parts:
        index = parts.index("fmri")
        if index + 1 >= len(parts):
            raise ValueError(f"Cannot infer RRR source identity from {value!r}")
        model_component = parts[index + 1]
        dataset, offset = "deep-fmri", 0
    else:
        model_index = next(
            (index for index, part in enumerate(parts)
             if re.fullmatch(r".+_layer\d+_offset-?\d+", part)
             or re.fullmatch(r"glove_dim\d+_offset-?\d+", part)),
            None,
        )
        if model_index is None:
            raise ValueError(f"Cannot infer RRR source identity from {value!r}")
        model_component = parts[model_index]
        dataset = (
            "brain-treebank-cars-2"
            if model_index + 1 < len(parts) and parts[model_index + 1] == "movie-cars-2"
            else "podcast"
        )
        offset_match = re.search(r"_offset(-?\d+)$", model_component)
        offset = int(offset_match.group(1)) if offset_match else 0

    glove = re.fullmatch(r"glove_dim(\d+)_offset-?\d+", model_component)
    if glove:
        model, layer = f"glove-{int(glove.group(1))}", None
    else:
        match = re.fullmatch(r"(.+)_layer(\d+)(?:_offset-?\d+)?", model_component)
        if match is None:
            raise ValueError(f"Cannot infer model and layer from {value!r}")
        model, layer = match.group(1), int(match.group(2))
    return {
        "mode": "internal", "dataset": dataset, "offset": offset,
        "model": model, "layer": layer,
    }


def _abstract_legacy_dependencies(config: dict) -> dict:
    """Replace legacy dependency directories with canonical identity blocks."""
    result = deepcopy(config)
    sources = []
    for key in ("source_rrr_unit_dir", "cross_result_dir"):
        value = result.pop(key, None)
        if value:
            sources.append(_rrr_source_from_legacy_path(value))
    if sources:
        if any(source != sources[0] for source in sources[1:]):
            raise ValueError("Legacy RRR dependency paths resolve to different identities")
        existing = result.get("rrr_source")
        if existing is not None and existing != sources[0]:
            raise ValueError("Legacy RRR dependency conflicts with configured rrr_source")
        result["rrr_source"] = sources[0]
    return result


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
    variant = readable_variant(config, _upstreams_from_config(config))
    return Mapping(**identity, variant=variant)


def _mapping_from_glove_path(path: Path, producer: str) -> Mapping | None:
    """Infer the one legacy identity encoded completely by a GloVe path."""
    model_root = next((part for part in path.parts if part.startswith("glove_dim")), None)
    if model_root is None:
        return None
    stem = model_root.removeprefix("glove_dim")
    dimension, separator, offset = stem.partition("_offset")
    if not separator or not dimension.isdigit() or not offset.lstrip("-").isdigit():
        return None
    mode = "sig_lags" if producer == "run_significance" else "cross_subject"
    identity = {
        "source": str(path), "producer": producer, "mode": mode,
        "dataset": "podcast", "offset": int(offset),
        "model": f"glove-{int(dimension)}", "layer": None,
    }
    return Mapping(**identity)


def discover_mappings(legacy_root: Path, results_root: Path) -> tuple[list[Mapping], list[dict]]:
    mappings, ambiguous = [], []
    for path in sorted(item for item in legacy_root.rglob("*") if item.is_dir()):
        if is_canonical(path, results_root) or is_ignored(path, legacy_root):
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
        producer = producers[0]
        config = _config_for_source(path, producer, legacy_root)
        if config is None and any(
            part.startswith("cross_dataset_") for part in path.parts
        ):
            # Config-less nested cross-dataset outputs have unknown upstream identity.
            continue
        if producer == "run_rrr" and config and "fmri" in config:
            producer = "run_cross_dataset_rrr"
        inferred = _mapping_from_config(path, producer, config or {})
        if inferred is None and config is None:
            inferred = _mapping_from_glove_path(path, producer)
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


def _source_files(source: Path, producer: str) -> list[Path]:
    """Inventory only files owned by the selected producer source."""
    if producer != "run_significance":
        return sorted(path for path in source.rglob("*") if path.is_file())
    files = []
    for name in (*SIGNIFICANCE_MARKERS, "metadata.json"):
        path = source / name
        if path.is_file():
            files.append(path)
    files.extend(path for path in source.glob("subjects/*/*") if path.is_file())
    for name in ("config.yml", "effective_config.yml", "config.yaml"):
        path = source / "significance" / name
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def _unsupported_files(source: Path, producer: str, files: list[Path]) -> list[Path]:
    if producer == "run_significance":
        allowed = {
            "all_subject_sig.csv", "significance_detailed.csv",
            "cluster_null_distribution.npz", "metadata.json", "config.yml",
            "effective_config.yml", "config.yaml", "significance.csv",
            "null_metadata.json", "voxel_filter.json", "ridge_alphas.npz",
            "encoding_results.joblib", "corrs.npy", "arbitrary_corrs.npy",
            "acf_diagnostic.png", "corrs_plot.png", "alpha_diagnostics.png",
            "alpha_diagnostics.json", "alpha_diagnostics.npz",
        }
    else:
        allowed = {
            "A.npy", "B.npy", "full_fir_A.npy", "pre_fir_pca_A.npy",
            "full_model_metrics.npz", "metrics.npz", "target_manifest.csv",
            "grid_results.csv", "summary.csv", "config.yml",
            "effective_config.yml", "config.yaml", "metadata.json",
            "subspace_aggregation.json", "alpha_diagnostics.json",
            "alpha_diagnostics.npz", "alpha_grid_cache.npz",
            "banded_ridge_diagnostics.npz", "rank_analysis.png",
            "rank_analysis_sig.png", "rank_analysis_nonsig.png",
        }
    return [path for path in files if path.name not in allowed]


def _values_equal(left, right) -> bool:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(
                left, right, check_dtype=False, check_exact=False,
                rtol=1e-15, atol=0.0,
            )
            return True
        except AssertionError:
            return False
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.dtype == object:
            return all(
                _values_equal(a, b)
                for a, b in zip(left.flat, right.flat)
            )
        try:
            np.testing.assert_equal(left, right)
            return True
        except AssertionError:
            return False
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    if type(left) is type(right) and hasattr(left, "__dict__") and hasattr(right, "__dict__"):
        return _values_equal(vars(left), vars(right))
    try:
        equal = left == right
        return bool(np.all(equal)) if isinstance(equal, np.ndarray) else bool(equal)
    except (TypeError, ValueError):
        return False


def _save_verified(artifact, field: str, value, *, selector: dict | None = None) -> None:
    """Save one migrated value, refusing collisions and checking its decoded value."""
    selector = selector or {}
    if artifact.has_field(field, selector=selector):
        existing = artifact.load_field(field, selector=selector)
        if not _values_equal(existing, value):
            raise FileExistsError(
                f"Differing migrated values for {field} selector {selector}"
            )
        return
    artifact.save_field(field, value, selector=selector)
    decoded = artifact.load_field(field, selector=selector)
    if not _values_equal(decoded, value):
        raise ValueError(f"Migration verification failed for {field} selector {selector}")


def copy_mapping(item: Mapping, results_root: Path, *, apply: bool) -> dict:
    """Convert one recognized legacy producer tree into a loadable Artifact."""
    source = Path(item.source)
    files = _source_files(source, item.producer)
    unsupported = _unsupported_files(source, item.producer, files)
    if unsupported:
        examples = ", ".join(str(path) for path in unsupported[:10])
        raise ValueError(
            f"Unsupported legacy files under {source}; refusing partial migration: {examples}"
        )
    inventory = {str(path.relative_to(source)): path.stat().st_size for path in files}
    operations = [{"action": "convert", "source": str(path), "size": inventory[
        str(path.relative_to(source))
    ]} for path in files]
    if not apply:
        return {"mapping": asdict(item), "operations": operations}

    mode = "internal" if item.mode in {"cross_subject", "within_fmri"} else item.mode
    cls = SignificanceArtifact if item.producer == "run_significance" else RRRArtifact
    config = _config_for_source(source, item.producer, source)
    if config is None:
        identity = asdict(item)
        for key in ("source", "producer", "variant"):
            identity.pop(key)
        config = {"legacy_identity": identity, "legacy_unknown_config": True}
    upstreams = _upstreams_from_config(config)
    artifact = cls(mode=mode, dataset=item.dataset, offset=item.offset,
                   model=item.model, layer=item.layer, variant=item.variant,
                   results_root=results_root)
    artifact.begin(config=config, upstreams=upstreams, refresh=False)
    migration_selector = {
        "name": "legacy_migration",
        "source_id": _migration_source_id(source),
    }
    identical_rerun = False
    if artifact.has_field("diagnostics", selector=migration_selector):
        prior = artifact.load_field("diagnostics", selector=migration_selector)
        if prior.get("inventory") != inventory:
            raise FileExistsError("Differing-content migration conflict for " + str(source))
        identical_rerun = True

    _save_verified(artifact, "effective_config", config)
    if isinstance(artifact, SignificanceArtifact):
        _convert_significance(source, artifact)
    else:
        _save_verified(artifact, "upstream_identities", upstreams)
        _convert_rrr(source, artifact)
    _save_verified(
        artifact, "diagnostics", {"source": str(source), "inventory": inventory},
        selector=migration_selector,
    )
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
    if identical_rerun:
        operations = [{**operation, "action": "skip-identical"}
                      for operation in operations]
    return {"mapping": asdict(item), "operations": operations,
            "artifact": artifact.identity_record()}


def _convert_significance(source: Path, artifact: SignificanceArtifact) -> None:
    metadata_path = source / "metadata.json"
    metadata = (json.loads(metadata_path.read_text()) if metadata_path.is_file()
                else {"method": artifact.mode, "migrated": True})
    _save_verified(artifact, "run_metadata", metadata)
    detailed_path = source / "significance_detailed.csv"
    detailed = pd.read_csv(detailed_path) if detailed_path.is_file() else None
    null_path = source / "cluster_null_distribution.npz"
    null_payload = None
    if null_path.is_file():
        with np.load(null_path, allow_pickle=True) as loaded:
            null_payload = {key: loaded[key] for key in loaded.files}
        _save_verified(artifact, "null_distribution", null_payload,
                       selector={"subject": "all"})

    subjects = set()
    for path in sorted(source.glob("subjects/*/significance.csv")):
        subject = path.parent.name
        frame = pd.read_csv(path).rename(columns={"voxel_index": "target_id", "sig": "significant"})
        frame["subject"] = str(subject)
        frame["target_kind"] = "voxel"
        _save_verified(artifact, "target_significance", frame,
                       selector={"subject": str(subject)})
        details = {}
        for name in ("null_metadata.json", "voxel_filter.json"):
            candidate = path.parent / name
            if candidate.is_file():
                details[name.removesuffix(".json")] = json.loads(candidate.read_text())
        alpha = path.parent / "ridge_alphas.npz"
        if alpha.is_file():
            with np.load(alpha, allow_pickle=True) as loaded:
                details["ridge_alphas"] = {key: loaded[key] for key in loaded.files}
        _save_verified(artifact, "significance_details", details,
                       selector={"subject": str(subject)})
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
            _save_verified(artifact, "target_significance", frame,
                           selector={"subject": str(subject)})
            if detailed is not None:
                _save_verified(
                    artifact, "significance_details",
                    detailed[detailed.subject == subject].reset_index(drop=True),
                    selector={"subject": str(subject)},
                )
            subjects.add(str(subject))

    for path in sorted(source.glob("subjects/sub_*/encoding_results.joblib")):
        subject = str(int(path.parent.name.removeprefix("sub_")))
        _save_verified(artifact, "encoding_results", joblib.load(path),
                       selector={"subject": subject})

    for path in sorted(source.glob("subjects/*/corrs.npy")):
        subject = path.parent.name.removeprefix("sub_")
        _save_verified(artifact, "correlations", np.load(path),
                       selector={"subject": str(subject)})
    for path in sorted(source.glob("subjects/*/arbitrary_corrs.npy")):
        subject = path.parent.name.removeprefix("sub_")
        _save_verified(
            artifact, "diagnostics", np.load(path),
            selector={"subject": str(subject), "name": "arbitrary_correlations"},
        )
    for filename, name, loader in (
        ("alpha_diagnostics.json", "alpha_diagnostics_metadata",
         lambda path: json.loads(path.read_text(encoding="utf-8"))),
        ("alpha_diagnostics.npz", "alpha_diagnostics_arrays", _npz),
    ):
        for path in sorted(source.glob(f"subjects/*/{filename}")):
            subject = path.parent.name.removeprefix("sub_")
            _save_verified(
                artifact, "diagnostics", loader(path),
                selector={"subject": str(subject), "name": name},
            )
    for filename, name in (
        ("acf_diagnostic.png", "acf_diagnostic"),
        ("corrs_plot.png", "correlations"),
        ("alpha_diagnostics.png", "alpha_diagnostics"),
    ):
        for path in sorted(source.glob(f"subjects/*/{filename}")):
            subject = path.parent.name.removeprefix("sub_")
            _save_verified(
                artifact, "plots", path.read_bytes(),
                selector={"subject": str(subject), "name": name},
            )


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
    method_parts = (*source.parts, *parts)
    method = "pca" if any(
        part.lower() == "pca" or part.lower().endswith("_pca")
        for part in method_parts
    ) else "rrr"
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


def _diagnostic_name(path: Path, source: Path) -> str:
    relative = path.relative_to(source)
    parents = [part for part in relative.parts[:-1]
               if not part.startswith("k") and not part.startswith("fold_")
               and part not in {"pca", "rrr"}]
    suffix = "json" if path.suffix == ".json" else "arrays"
    return ".".join([*parents, path.stem, suffix])


def _convert_rrr(source: Path, artifact: RRRArtifact) -> None:
    for path in sorted(item for item in source.rglob("A.npy")
                       if "source_factors" not in item.relative_to(source).parts):
        unit, fold, rank = _unit_from_path(path, source)
        selector = artifact._selector(unit, fold=fold, rank=rank)
        _save_verified(artifact, "input_factors", np.load(path), selector=selector)
        b_path = path.with_name("B.npy")
        if b_path.is_file():
            _save_verified(artifact, "target_factors", np.load(b_path),
                           selector=selector)
    source_factors = source / "source_factors"
    for path in sorted(source_factors.glob("delay_*/A.npy")):
        delay = int(path.parent.name.removeprefix("delay_"))
        _save_verified(
            artifact, "input_factors", np.load(path),
            selector={"method": "rrr", "delay": delay, "basis": "delayed"},
        )
    for filename, basis in (("full_fir_A.npy", "full_fir"),
                            ("pre_fir_pca_A.npy", "pre_fir_pca")):
        path = source_factors / filename
        if path.is_file():
            _save_verified(artifact, "input_factors", np.load(path),
                           selector={"method": "rrr", "basis": basis})
    for path in sorted(source.rglob("full_model_metrics.npz")):
        unit, fold, _ = _unit_from_path(path, source)
        _save_verified(artifact, "full_metrics", _npz(path),
                       selector=artifact._selector(unit, fold=fold))
    for path in sorted(source.rglob("metrics.npz")):
        if path.name == "full_model_metrics.npz":
            continue
        unit, fold, rank = _unit_from_path(path, source)
        if path.parent.name == "full":
            _save_verified(artifact, "full_metrics", _npz(path))
        elif rank is not None:
            _save_verified(
                artifact, "rank_metrics", _npz(path),
                selector=artifact._selector(unit, fold=fold, rank=rank),
            )
    for path in sorted(source.rglob("target_manifest.csv")):
        unit, _, _ = _unit_from_path(path, source)
        _save_verified(artifact, "target_manifest", pd.read_csv(path),
                       selector=unit.mapping())
    for path in sorted(source.rglob("grid_results.csv")):
        unit, _, _ = _unit_from_path(path, source)
        frame = pd.read_csv(path)
        if "fold" in frame:
            _save_verified(artifact, "fold_results", frame,
                           selector=unit.mapping())
        else:
            _save_verified(artifact, "mean_results", frame)
    for path in sorted(source.rglob("summary.csv")):
        unit, _, _ = _unit_from_path(path, source)
        summary = pd.read_csv(path)
        normalized = summary.copy()
        if "k" in normalized and "rank" not in normalized:
            normalized = normalized.rename(columns={"k": "rank"})
        if "method" not in normalized:
            normalized["method"] = unit.method
        primary = "pca" if unit.method == "pca" else "nr"
        if "corr" not in normalized and f"{primary}_corr" in normalized:
            normalized["corr"] = normalized[f"{primary}_corr"]
        if "r2" not in normalized and f"{primary}_r2" in normalized:
            normalized["r2"] = normalized[f"{primary}_r2"]
        _save_verified(artifact, "mean_results", normalized,
                       selector=unit.mapping())
    for path in sorted(source.rglob("*.json")):
        if path.name in {"config.yml", "metadata.json", "subspace_aggregation.json"}:
            continue
        unit, fold, rank = _unit_from_path(path, source)
        _save_verified(
            artifact, "diagnostics", json.loads(path.read_text(encoding="utf-8")),
            selector=artifact._selector(
                unit, fold=fold, rank=rank, name=_diagnostic_name(path, source)
            ),
        )
    diagnostic_npz = {"alpha_diagnostics.npz", "alpha_grid_cache.npz",
                      "banded_ridge_diagnostics.npz"}
    for path in sorted(item for item in source.rglob("*.npz")
                       if item.name in diagnostic_npz):
        unit, fold, rank = _unit_from_path(path, source)
        _save_verified(
            artifact, "diagnostics", _npz(path),
            selector=artifact._selector(
                unit, fold=fold, rank=rank, name=_diagnostic_name(path, source)
            ),
        )
    for path in sorted(source.glob("*.png")):
        unit, _, _ = _unit_from_path(path, source)
        _save_verified(artifact, "plots", path.read_bytes(),
                       selector=artifact._selector(unit, name=path.stem))
    for path in sorted(source.rglob("subspace_aggregation.json")):
        unit, _, _ = _unit_from_path(path, source)
        _save_verified(artifact, "unit_metadata", json.loads(path.read_text()),
                       selector=unit.mapping())
    metadata = source / "metadata.json"
    if metadata.is_file():
        _save_verified(artifact, "metadata", json.loads(metadata.read_text()))


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
