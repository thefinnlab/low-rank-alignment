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
from src.utils.rrr.core import canonical_qr
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
        embedding_spec = (
            {
                "model_dir": effective.get("model_dir", artifact.model),
                "layer": effective.get("layer", artifact.layer),
                "pooling": effective.get("pooling", "mean"),
            }
            if artifact.dataset == "deep-fmri"
            else {
                "model_dir": enc.get("per_model_config", {}).get(
                    "model_dir", artifact.model
                ),
                "layer": enc.get("per_model_config", {}).get(
                    "layer", artifact.layer
                ),
                "pooling": enc.get("per_model_config", {}).get("pooling", "mean"),
            }
        )
        dataset = artifact.dataset
        stimulus = dataset.removeprefix("brain-treebank-")
        resolved[view.name] = ResolvedSubspaceRun(
            view.name, view.kind or "rrr", artifact, SimpleNamespace(
                encoding=SimpleNamespace(**enc),
                effective=effective,
                embedding_spec=SimpleNamespace(**embedding_spec),
            ), (dataset, stimulus),
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


def load_view_bases(
    config, resolved, target_key, selection, test_idxs, *,
    return_scalers=False, orthonormalize=True,
):
    bases, modes, scalers = {}, {}, {}
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
            fold_values, fold_scalers = {}, {}
            for selector, value in entries.items():
                mapping = dict(selector)
                if "fold" not in mapping:
                    continue
                scaler = run.artifact.load_input_scaler(mapping)
                factor = np.asarray(value)[:, :rank]
                if orthonormalize:
                    factor = canonical_qr(factor)[0]
                fold_values[int(mapping["fold"])] = _factor_with_scaler_in_raw_coordinates(
                    scaler, factor, view.name
                )
                fold_scalers[int(mapping["fold"])] = scaler
            if set(fold_values) != set(range(len(test_idxs))):
                raise ValueError(f"View {view.name} does not provide every target fold")
            bases[view.name] = fold_values
            scalers[view.name] = fold_scalers
            modes[view.name] = (
                "matching_target_fold_standardized_qr"
                if orthonormalize else "matching_target_fold_raw"
            )
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
                orthonormalize=orthonormalize,
            )[rank]
            scaler = load_cross_dataset_scaler(
                run.artifact, "rrr", rank
            )
            factor = _factor_with_scaler_in_raw_coordinates(
                scaler, factor, view.name
            )
            bases[view.name] = {fold: factor for fold in range(len(test_idxs))}
            scalers[view.name] = {fold: scaler for fold in range(len(test_idxs))}
            modes[view.name] = (
                ("cross_dataset_fmri_standardized_qr" if orthonormalize
                 else "cross_dataset_fmri_rank_basis_raw")
                if is_fmri
                else ("cross_dataset_full_fit_standardized_qr" if orthonormalize
                      else "cross_dataset_full_fit_raw")
            )
    return (bases, modes, scalers) if return_scalers else (bases, modes)


def load_global_view_transforms(config, resolved):
    """Load whole-dataset factors and scalers suitable for an fMRI target."""
    factors, scalers, modes = {}, {}, {}
    for view in config.views:
        run = resolved[view.name]
        is_fmri = run.artifact.dataset == "deep-fmri"
        expected_mode = "internal" if is_fmri else "full_fit"
        if run.artifact.mode != expected_mode:
            raise ValueError(
                f"fMRI multiview source {view.name!r} must use a {expected_mode} "
                f"Artifact; got mode={run.artifact.mode!r}"
            )
        rank = int(view.rank)
        kind = view.kind or "rrr"
        factors[view.name] = load_cross_dataset_A_by_rank(
            run.artifact,
            kind=kind,
            ranks=[rank],
            source_format="rank_bases",
            orthonormalize=True,
        )[rank]
        scalers[view.name] = load_cross_dataset_scaler(
            run.artifact, kind, rank
        )
        modes[view.name] = (
            "fmri_internal_whole_training" if is_fmri else "ieeg_full_fit"
        )
    return factors, scalers, modes


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
    scalers: dict[str, dict[int, dict]] | None = None,
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
        if scalers is None:
            band = X @ A
        else:
            scaler = scalers[name][fold]
            scale = np.asarray(scaler["scale"])
            mean = np.asarray(scaler["mean"])
            standardized = (X - mean) / scale
            band = standardized @ (scale.reshape(-1, 1) * A)
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
    solver_params: dict,
    ridge_cv_folds: int | None = None,
    cv=None,
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
    if cv is None:
        if ridge_cv_folds is None:
            raise ValueError("Banded ridge requires cv or ridge_cv_folds")
        cv = KFold(n_splits=ridge_cv_folds, shuffle=False)
    kwargs = {
        "groups": "input",
        "solver": "random_search",
        "cv": cv,
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
