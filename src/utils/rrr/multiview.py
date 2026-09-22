"""Shared Artifact-aware preparation and banded-ridge helpers for multiview RRR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.core.artifacts import normalize_dataset
from src.core.registry import resolve_dataset_context
from src.utils.rrr.ieeg_rrr_mode.cross_dataset_helpers import (
    load_cross_dataset_A_by_rank,
    load_cross_dataset_scaler,
)
from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.tensors import to_numpy
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
        artifact = Artifact(**kwargs)
        if artifact.variant is None:
            artifact._resolve_variant()
        if artifact.mode not in {"internal", "full_fit"}:
            raise ValueError(
                f"Multiview source {view.name!r} must be an internal or full_fit "
                "RRR Artifact; "
                f"got mode={artifact.mode!r}"
            )
        manifest = artifact.manifest()
        if not manifest.get("complete"):
            raise ValueError(f"Multiview source {view.name!r} is incomplete")
        artifact.load_artifacts(["effective_config"])
        effective = artifact.effective_config
        enc = effective.get("encoding")
        if enc is None:
            is_glove = artifact.model.startswith("glove-")
            enc = {
                "model": "glove" if is_glove else "transformer",
                "offset": artifact.offset,
                "limit_to_glove": bool(
                    effective.get("fmri", {}).get("limit_to_glove", False)
                ),
                "per_model_config": {
                    "model_dir": artifact.model,
                    "layer": artifact.layer,
                    **(
                        {"dim": int(artifact.model.split("-", 1)[1])}
                        if is_glove
                        else {}
                    ),
                },
            }
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
    return (normalize_dataset(enc.dataset, context.stimulus_id),
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
    for view in config.views:
        run = resolved[view.name]
        rank = int(view.rank)
        if run.dataset_key == target_key:
            if run.artifact.mode != "internal":
                raise ValueError(
                    f"Matching-dataset multiview source {view.name!r} must use an "
                    "internal RRR Artifact"
                )
            entries = run.artifact.load_collection(
                "input_factors", {"method": "rrr"}
            )
            fold_values = {}
            for selector, value in entries.items():
                mapping = dict(selector)
                if "fold" not in mapping:
                    continue
                factor = _factor_in_raw_coordinates(
                    run.artifact, mapping, np.asarray(value)[:, :rank], view.name
                )
                fold_values[int(mapping["fold"])] = factor
            if set(fold_values) != set(range(len(test_idxs))):
                raise ValueError(f"View {view.name} does not provide every target fold")
            bases[view.name] = fold_values
            modes[view.name] = "matching_target_fold_raw"
        else:
            is_fmri = run.artifact.dataset == "deep-fmri"
            expected_mode = "internal" if is_fmri else "full_fit"
            if run.artifact.mode != expected_mode:
                raise ValueError(
                    f"Cross-dataset multiview source {view.name!r} must use a "
                    f"{expected_mode} RRR Artifact; got mode={run.artifact.mode!r}"
                )
            factor = load_cross_dataset_A_by_rank(
                run.artifact,
                kind="rrr",
                ranks=[rank],
                source_format="rank_bases",
                orthonormalize=True,
            )[rank]
            scaler = load_cross_dataset_scaler(
                run.artifact, "rrr", rank
            )
            factor = _factor_with_scaler_in_raw_coordinates(
                scaler, factor, view.name
            )
            bases[view.name] = {fold: factor for fold in range(len(test_idxs))}
            modes[view.name] = (
                "cross_dataset_fmri_rank_basis_raw"
                if is_fmri
                else "cross_dataset_full_fit_raw"
            )
    return bases, modes


def _factor_in_raw_coordinates(artifact, selector, factor, view_name):
    try:
        scaler = artifact.load_input_scaler(selector)
    except KeyError as error:
        raise ValueError(
            f"Multiview source {view_name!r} has no scaler for factor {selector}"
        ) from error
    return _factor_with_scaler_in_raw_coordinates(scaler, factor, view_name)


def _factor_with_scaler_in_raw_coordinates(scaler, factor, view_name):
    if scaler.get("coordinates") != "standardized-v1":
        raise ValueError(
            f"Multiview source {view_name!r} uses unsupported coordinates "
            f"{scaler.get('coordinates')!r}"
        )
    scale = np.asarray(scaler["scale"])
    value = np.asarray(factor)
    if value.ndim != 2 or value.shape[0] != scale.shape[0]:
        raise ValueError(
            f"Multiview source {view_name!r} factor/scaler dimensions differ: "
            f"{value.shape} vs {scale.shape}"
        )
    return value / scale[:, np.newaxis]

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
