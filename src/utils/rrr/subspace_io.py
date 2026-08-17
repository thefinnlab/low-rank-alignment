"""Compatibility I/O helpers for saved RRR subspaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.subspace_comparison.metrics import orthonormal_basis


@dataclass(frozen=True)
class ResolvedSubspaceRun:
    name: str
    kind: str
    artifact: object
    config: object
    dataset_key: tuple[str, str]
    subject_ids: list


@dataclass
class BaseComparisonData:
    X_base: np.ndarray
    train_idxs: list[np.ndarray]
    test_idxs: list[np.ndarray]
    selection: np.ndarray
    valid_rows_on_selection: np.ndarray | None
    dataset_key: tuple[str, str]
    word_count: int


def sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in name)
    safe = safe.strip("-._")
    if not safe:
        raise ValueError(f"Cannot sanitize empty run name from {name!r}")
    return safe


def load_fold_bases(unit_dir: Path, rank: int) -> dict[int, np.ndarray]:
    bases = {}
    for path in sorted(Path(unit_dir).glob("fold_*/A.npy")):
        try:
            fold = int(path.parent.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        array = np.load(path, mmap_mode="r")
        if int(rank) <= array.shape[1]:
            bases[fold] = orthonormal_basis(array, int(rank))
    return bases


def max_available_rank(unit_dir: Path) -> int:
    widths = [int(np.load(path, mmap_mode="r").shape[1])
              for path in sorted(Path(unit_dir).glob("fold_*/A.npy"))]
    if not widths:
        raise FileNotFoundError(f"No fold_*/A.npy files found under {unit_dir}")
    return min(widths)


def ambient_dim(unit_dir: Path) -> int:
    dims = [int(np.load(path, mmap_mode="r").shape[0])
            for path in sorted(Path(unit_dir).glob("fold_*/A.npy"))]
    if not dims:
        raise FileNotFoundError(f"No fold_*/A.npy files found under {unit_dir}")
    unique = sorted(set(dims))
    if len(unique) != 1:
        raise ValueError(f"Inconsistent A.npy ambient dimensions under {unit_dir}: {unique}")
    return unique[0]


def load_base_comparison_data(run: ResolvedSubspaceRun) -> BaseComparisonData:
    """Load row and fold provenance recorded by a resolved Artifact run."""
    artifact = run.artifact
    upstreams = artifact.load_field("upstream_identities")
    significance_id = next(item for item in upstreams
                           if item.get("script") == "run_significance")
    from scripts.run_significance import Artifact as SignificanceArtifact
    kwargs = {key: significance_id[key] for key in
              ("mode", "dataset", "offset", "model", "layer", "variant")}
    kwargs["results_root"] = artifact.results_root
    significance = SignificanceArtifact(**kwargs).load_artifacts(
        ["encoding_results"], selectors={"subjects": run.subject_ids})
    reference = significance.encoding_results[str(run.subject_ids[0])]
    selection = np.asarray(reference.get("selection", []), dtype=int)
    tests = [np.asarray(value, dtype=int) for value in reference["test_idxs"]]
    trains = [np.asarray(value, dtype=int) for value in reference.get("train_idxs", [])]
    return BaseComparisonData(np.empty((len(selection), 0)), trains, tests, selection,
                              None, run.dataset_key, len(selection))
