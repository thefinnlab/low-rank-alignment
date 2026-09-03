"""Shared Artifact-aware preparation and banded-ridge helpers for multiview RRR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.core.registry import resolve_dataset_context
from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.tensors import to_numpy
from src.utils.subspace_comparison.metrics import aggregate_subspace_basis
from src.utils.rrr.subspace_io import ResolvedSubspaceRun

@dataclass
class BandedRidgeBaseline:
    """One fold's fitted full-space banded encoding model."""

    Y_hat_train: np.ndarray
    Y_hat_test: np.ndarray
    beta_full_raw: np.ndarray
    intercept: np.ndarray
    best_alphas: np.ndarray
    deltas: np.ndarray



def resolve_rrr_views(config):
    from scripts.run_rrr import Artifact
    resolved = {}
    for view in config.views:
        source = view.ieeg_rrr_source or view.fmri_rrr_source
        kwargs = dict(source or {})
        kwargs.setdefault("results_root", config.results_root)
        artifact = Artifact(**kwargs).load_artifacts(["effective_config"])
        effective = artifact.effective_config
        enc = effective.get("encoding", {})
        dataset = artifact.dataset
        stimulus = dataset.removeprefix("brain-treebank-")
        resolved[view.name] = ResolvedSubspaceRun(
            view.name, view.kind or "rrr", artifact, SimpleNamespace(
                encoding=SimpleNamespace(**enc)), (dataset, stimulus),
            list(enc.get("subjects") or range(1, 10)),
        )
    return resolved


def target_dataset_key(config, subject_ids):
    enc = config.encoding
    context = resolve_dataset_context(enc.dataset, subject_ids,
                                      enc.per_dataset_config, enc.data_root)
    return (getattr(context, "dataset", enc.dataset),
            str(getattr(context, "stimulus_id", enc.dataset)))


def load_view_embeddings(config, resolved, subject_ids):
    target = config.encoding
    context = resolve_dataset_context(target.dataset, subject_ids,
                                      target.per_dataset_config, target.data_root)
    loaded, reference = {}, None
    for view in config.views:
        enc = resolved[view.name].config.encoding
        values, words = load_embeddings(
            target.data_root, enc.model, dataset=target.dataset,
            per_dataset_config=target.per_dataset_config,
            dataset_context=context,
            limit_to_glove=target.limit_to_glove or enc.model == "glove",
            glove_dim=enc.per_model_config.get("dim", 300),
            **enc.per_model_config,
        )
        if reference is not None and not np.array_equal(
                reference["word"].astype(str).to_numpy(),
                words["word"].astype(str).to_numpy()):
            raise ValueError(f"Embedding transcript differs for view {view.name}")
        reference = words
        loaded[view.name] = np.asarray(values)
    if reference is None:
        raise ValueError("No multiview embeddings were loaded")
    return loaded, reference


def load_view_bases(config, resolved, target_key, selection, test_idxs):
    bases, modes = {}, {}
    aggregation_method = config.subspace_aggregation or "extrinsic"
    for view in config.views:
        run = resolved[view.name]
        rank = int(view.rank)
        entries = run.artifact.load_collection("input_factors", {"method": "rrr"})
        fold_values = {int(dict(selector)["fold"]): np.asarray(value)[:, :rank]
                       for selector, value in entries.items()
                       if "fold" in dict(selector)}
        if run.dataset_key == target_key:
            if set(fold_values) != set(range(len(test_idxs))):
                raise ValueError(f"View {view.name} does not provide every target fold")
            bases[view.name] = fold_values
            modes[view.name] = "matching_target_fold_raw"
        else:
            mean = aggregate_subspace_basis(list(fold_values.values()), rank=rank,
                                            method=aggregation_method)
            bases[view.name] = {fold: mean for fold in range(len(test_idxs))}
            modes[view.name] = f"cross_dataset_fold_{aggregation_method}"
    return bases, modes

def project_view_bands(
    embeddings: dict[str, np.ndarray],
    bases: dict[str, dict[int, np.ndarray]],
    view_names: list[str],
    fold: int,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Project aligned embeddings and return bands plus concatenated boundaries."""
    bands = []
    boundaries = []
    start = 0
    for name in view_names:
        X = np.asarray(embeddings[name])
        A = np.asarray(bases[name][fold])
        if X.shape[1] != A.shape[0]:
            raise ValueError(
                f"View {name} embedding/A dimensions differ: {X.shape[1]} vs {A.shape[0]}"
            )
        band = X @ A
        bands.append(np.asarray(band, dtype=np.float32))
        boundaries.append((start, start + band.shape[1]))
        start += band.shape[1]
    return bands, boundaries


def fit_banded_ridge_baseline(
    train_bands: list[np.ndarray],
    test_bands: list[np.ndarray],
    Y_train: np.ndarray,
    *,
    alphas: np.ndarray,
    ridge_cv_folds: int,
    solver_params: dict,
    y_in_cpu: bool | None = None,
    force_cpu: bool | None = None,
) -> BandedRidgeBaseline:
    """Fit Himalaya banded ridge directly in the grouped primal feature space."""
    from himalaya.ridge import BandedRidgeCV

    if len(train_bands) != len(test_bands) or not train_bands:
        raise ValueError("Banded ridge requires matching non-empty train/test bands")
    n_train = Y_train.shape[0]
    if any(band.shape[0] != n_train for band in train_bands):
        raise ValueError("Every training band must share Y_train rows")
    resolved_solver_params = dict(solver_params)
    random_state = resolved_solver_params.pop("random_state", 42)
    resolved_solver_params["alphas"] = np.asarray(alphas, dtype=float)
    kwargs = {
        "groups": "input",
        "solver": "random_search",
        "cv": KFold(n_splits=ridge_cv_folds, shuffle=False),
        "random_state": random_state,
        "fit_intercept": True,
        "solver_params": resolved_solver_params,
    }
    if y_in_cpu is not None:
        kwargs["Y_in_cpu"] = y_in_cpu
    if force_cpu is not None:
        kwargs["force_cpu"] = force_cpu
    model = BandedRidgeCV(**kwargs)
    model.fit(train_bands, Y_train)
    Y_hat_train = to_numpy(model.predict(train_bands))
    Y_hat_test = to_numpy(model.predict(test_bands))
    beta = to_numpy(model.coef_)
    deltas = to_numpy(model.deltas_)
    intercept = to_numpy(model.intercept_)
    X_train = np.concatenate(train_bands, axis=1)
    X_test = np.concatenate(test_bands, axis=1)
    reconstructed_train = X_train @ beta + intercept
    reconstructed_test = X_test @ beta + intercept
    if not np.allclose(reconstructed_train, Y_hat_train, rtol=2e-4, atol=2e-4):
        raise RuntimeError("Banded ridge coefficients do not reproduce train predictions")
    if not np.allclose(reconstructed_test, Y_hat_test, rtol=2e-4, atol=2e-4):
        raise RuntimeError("Banded ridge coefficients do not reproduce test predictions")
    return BandedRidgeBaseline(
        Y_hat_train=np.asarray(Y_hat_train, dtype=np.float32),
        Y_hat_test=np.asarray(Y_hat_test, dtype=np.float32),
        beta_full_raw=np.asarray(beta, dtype=np.float32),
        intercept=np.asarray(intercept, dtype=np.float32),
        best_alphas=to_numpy(model.best_alphas_),
        deltas=np.asarray(deltas),
    )
