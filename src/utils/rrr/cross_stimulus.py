"""Reusable mechanics for within-subject, cross-stimulus orthogonal RRR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.rrr.evaluation import evaluate_projected_input_space
from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.encoding import apply_offset
from src.utils.shared.ieeg import load_precomputed_epochs


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
        from pathlib import Path
        import re

        found = set()
        for path in Path(data_root, "subject_metadata").glob("sub_*_trial*_metadata.json"):
            match = re.match(r"sub_(\d+)_", path.name)
            if match:
                found.add(int(match.group(1)))
        subjects = sorted(found)
    result = {}
    for subject in subjects:
        movies = [record["movie"] for record in adapter._metadata_records(data_root, subject)]
        movies = list(dict.fromkeys(movie for movie in movies if movie))
        if len(movies) >= int(min_movies):
            result[int(subject)] = movies
    return result


def cache_header(data_root: str, subject: int, stimulus: str, rereference: str) -> dict:
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


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_projected_ridge(
    Q: np.ndarray, rank: int, X_train: np.ndarray, X_test: np.ndarray,
    Y_train: np.ndarray, Y_test: np.ndarray, *, alphas=None,
    alpha_policy=None, ridge_cv_folds=5, ridge_solver_params=None,
    ridge_y_in_cpu=None, ridge_force_cpu=None, use_low_rank_svd=True,
    low_rank_svd_seed=0,
) -> dict:
    """Fit a target readout on orthogonal projected coordinates."""
    result, _ = evaluate_projected_input_space(
        Q[:, :rank], X_train, X_test, Y_train, Y_test,
        alphas=(None if alphas is None else np.asarray(alphas, dtype=float)),
        alpha_policy=alpha_policy,
        alpha_context={
            "script": "run_cross_stimulus_rrr", "role": "orthogonal_projection",
            "rank": int(rank),
        }, standardize=False, ridge_cv_folds=ridge_cv_folds,
        ridge_solver_params=ridge_solver_params,
        ridge_y_in_cpu=ridge_y_in_cpu, ridge_force_cpu=ridge_force_cpu,
        device=_device(),
    )
    return result
