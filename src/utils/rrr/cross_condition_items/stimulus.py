"""Brain Treebank stimulus items for cross-condition RRR."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.rrr.cross_condition import ComparisonGroup, ComparisonItem
from src.utils.rrr.inputs import build_target_manifest
from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.encoding import apply_offset
from src.utils.shared.ieeg import load_precomputed_epochs


class StimulusItemConfig(Protocol):
    """Configuration fields consumed by the stimulus item provider."""

    data_root: str
    model: str
    per_model_config: dict
    offset: int
    rereference: str


@dataclass
class StimulusData:
    """One subject/movie's aligned embeddings and common-contact targets."""

    subject: int
    stimulus: str
    X: np.ndarray
    Y: np.ndarray
    selection: np.ndarray
    electrodes: list[str]
    lags: np.ndarray


def discover_subject_movies(
    data_root: str, subjects: list[int] | None = None, *, min_movies: int = 2,
) -> dict[int, list[str]]:
    """Discover Brain Treebank movie identifiers from subject metadata."""
    adapter = get_dataset_adapter("brain_treebank")
    if subjects is None:
        found = set()
        for path in Path(data_root, "subject_metadata").glob(
            "sub_*_trial*_metadata.json"
        ):
            match = re.match(r"sub_(\d+)_", path.name)
            if match:
                found.add(int(match.group(1)))
        subjects = sorted(found)
    result = {}
    for subject in subjects:
        movies = [
            record["movie"]
            for record in adapter._metadata_records(data_root, subject)
        ]
        movies = list(dict.fromkeys(movie for movie in movies if movie))
        if len(movies) >= int(min_movies):
            result[int(subject)] = movies
    return result


def cache_header(
    data_root: str, subject: int, stimulus: str, rereference: str,
) -> dict:
    """Load only the portable metadata for one epoch cache."""
    context = resolve_dataset_context(
        "brain_treebank", [subject],
        {"movie": stimulus, "rereference": rereference}, data_root,
    )
    cache = load_precomputed_epochs(
        subject, data_root, dataset="brain_treebank",
        per_dataset_config={"movie": stimulus, "rereference": rereference},
        dataset_context=context,
    )
    return {
        "context": context,
        "electrodes": list(cache["ch_names"]),
        "n_lags": int(cache["n_lags"]),
        "sfreq": float(cache["sfreq"]),
        "tmin": float(cache["tmin"]),
        "tmax": float(cache["tmax"]),
        "n_words": int(len(cache["selection"])),
    }


def ordered_electrode_intersection(headers: dict[str, dict]) -> list[str]:
    """Return first-movie electrode order restricted to contacts in every movie."""
    if not headers:
        return []
    first = next(iter(headers.values()))["electrodes"]
    shared = set(first)
    for header in headers.values():
        shared.intersection_update(header["electrodes"])
    return [electrode for electrode in first if electrode in shared]


def load_stimulus_data(
    *, data_root: str, subject: int, stimulus: str, rereference: str,
    model: str, per_model_config: dict, offset: int, electrodes: list[str],
) -> StimulusData:
    """Load one movie and align embeddings to its cached neural epochs."""
    per_dataset = {"movie": stimulus, "rereference": rereference}
    context = resolve_dataset_context(
        "brain_treebank", [subject], per_dataset, data_root
    )
    embeddings, _ = load_embeddings(
        data_root, model, dataset="brain_treebank",
        per_dataset_config=per_dataset, dataset_context=context,
        **per_model_config,
    )
    cache = load_precomputed_epochs(
        subject, data_root, dataset="brain_treebank",
        per_dataset_config=per_dataset, dataset_context=context,
    )
    missing = [name for name in electrodes if name not in cache["ch_names"]]
    if missing:
        raise ValueError(
            f"Stimulus {stimulus!r} is missing common electrodes: {missing}"
        )
    indices = [cache["ch_names"].index(name) for name in electrodes]
    epoch_data = cache["epoch_data"].read(electrode_indices=indices)
    selection = np.asarray(cache["selection"], dtype=int)
    X, Y = apply_offset(np.asarray(embeddings)[selection], epoch_data, offset)
    if len(X) < 2:
        raise ValueError(f"Stimulus {stimulus!r} has fewer than two aligned rows")
    lags = np.linspace(
        float(cache["tmin"]), float(cache["tmax"]),
        int(cache["n_lags"]), endpoint=False,
    )
    return StimulusData(
        subject=int(subject), stimulus=stimulus,
        X=np.asarray(X, dtype=np.float32), Y=np.asarray(Y, dtype=np.float32),
        selection=selection, electrodes=list(electrodes), lags=lags,
    )


def get_stimulus_items(
    config: StimulusItemConfig, subject: int, movies: list[str],
) -> ComparisonGroup:
    """Build one subject's comparison items from its viable movie stimuli."""
    exclusions = []
    headers = {}
    for movie in movies:
        try:
            headers[movie] = cache_header(
                config.data_root, subject, movie, config.rereference
            )
        except Exception as error:
            exclusions.append({
                "subject": subject, "item": movie, "stage": "cache_header",
                "reason": f"{type(error).__name__}: {error}",
            })
    while len(headers) >= 2:
        electrodes = ordered_electrode_intersection(headers)
        if not electrodes:
            exclusions.append({
                "subject": subject, "item": "*", "stage": "electrodes",
                "reason": "No usable Laplacian contact is common to all movies",
            })
            return ComparisonGroup([], {"subject": subject}, exclusions=exclusions)
        loaded, failures = {}, []
        for movie in headers:
            try:
                loaded[movie] = load_stimulus_data(
                    data_root=config.data_root, subject=subject, stimulus=movie,
                    rereference=config.rereference, model=config.model,
                    per_model_config=deepcopy(config.per_model_config),
                    offset=config.offset, electrodes=electrodes,
                )
            except Exception as error:
                failures.append(movie)
                exclusions.append({
                    "subject": subject, "item": movie, "stage": "data_loading",
                    "reason": f"{type(error).__name__}: {error}",
                })
        if not failures:
            break
        for movie in failures:
            headers.pop(movie, None)
    else:
        loaded, electrodes = {}, []

    if len(loaded) < 2:
        exclusions.append({
            "subject": subject, "item": "*", "stage": "eligibility",
            "reason": "Fewer than two fully loadable movies remain",
        })
        return ComparisonGroup([], {"subject": subject}, exclusions=exclusions)

    items = []
    target_manifests = {}
    eligibility = []
    for movie, stimulus in loaded.items():
        manifest = build_target_manifest(
            [subject], {subject: electrodes}, [electrodes], stimulus.lags
        )
        items.append(ComparisonItem(
            name=movie, X=stimulus.X, Y=stimulus.Y,
            target_set=f"subject-{subject}-targets",
        ))
        target_manifests[movie] = manifest
        eligibility.append({
            "subject": subject, "item": movie, "n_words": len(stimulus.X),
            "n_common_electrodes": len(electrodes),
            "n_lags": len(stimulus.lags), "eligible": True,
        })
    return ComparisonGroup(
        items=items, context={"subject": subject},
        target_manifests=target_manifests, eligibility=eligibility,
        exclusions=exclusions,
    )


__all__ = [
    "StimulusData",
    "cache_header",
    "discover_subject_movies",
    "get_stimulus_items",
    "load_stimulus_data",
    "ordered_electrode_intersection",
]
