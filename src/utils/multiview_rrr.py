"""Shared preparation and banded-ridge helpers for multiview RRR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.core.config import RRRConfig, SubspaceComparisonRun
from src.core.registry import resolve_dataset_context
from src.utils.data_loaders import load_sig_electrodes
from src.utils.embeddings import load_embeddings
from src.utils.subspace_compare import aggregate_subspace_from_unit_dir
from src.utils.subspace_io import (
    ResolvedSubspaceRun,
    load_base_comparison_data,
    resolve_subspace_run,
)
from src.utils.tensor_utils import to_numpy


@dataclass
class BandedRidgeBaseline:
    """One fold's fitted full-space banded encoding model."""

    Y_hat_train: np.ndarray
    Y_hat_test: np.ndarray
    beta_full_raw: np.ndarray
    intercept: np.ndarray
    best_alphas: np.ndarray
    deltas: np.ndarray


def resolve_rrr_views(config: RRRConfig) -> dict[str, ResolvedSubspaceRun]:
    """Resolve configured multiview entries using the shared subspace I/O path."""
    return {
        view.name: resolve_subspace_run(
            SubspaceComparisonRun(
                name=view.name,
                rrr_config=view.rrr_config,
                unit_dir=view.unit_dir,
                kind=view.kind,
            )
        )
        for view in config.views
    }


def target_dataset_key(config: RRRConfig, subject_ids: list[int]) -> tuple[str, str]:
    enc = config.encoding
    context = resolve_dataset_context(
        dataset=enc.dataset,
        subject_ids=subject_ids,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    return (
        getattr(context, "dataset", enc.dataset),
        str(getattr(context, "stimulus_id", enc.dataset)),
    )


def union_view_significance(
    resolved: dict[str, ResolvedSubspaceRun],
    target_key: tuple[str, str],
    subject_ids: list[int],
) -> tuple[dict[int, list[str]], list[pd.DataFrame], list[str]]:
    """Union electrode and lag significance across target-dataset views."""
    union = {int(subject): [] for subject in subject_ids}
    detailed_frames: list[pd.DataFrame] = []
    sources: list[str] = []
    matching = [run for run in resolved.values() if run.dataset_key == target_key]
    if not matching:
        raise ValueError(
            "multi_view requires at least one view from the target dataset/stimulus"
        )
    for run in matching:
        enc = run.config.encoding
        significant, _ = load_sig_electrodes(
            str(run.output_dir),
            subject_ids,
            sig_dir=str(run.sig_dir),
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            data_root=enc.data_root,
        )
        for subject in subject_ids:
            for electrode in significant.get(subject, []):
                if electrode not in union[int(subject)]:
                    union[int(subject)].append(electrode)
        detailed_path = run.sig_dir / "significance_detailed.csv"
        if detailed_path.exists():
            detailed_frames.append(pd.read_csv(detailed_path))
        sources.append(str(run.sig_dir / "all_subject_sig.csv"))
    union = {subject: electrodes for subject, electrodes in union.items() if electrodes}
    if not union:
        raise ValueError("No significant electrodes found in target-dataset views")
    return union, detailed_frames, sources


def union_sig_mask(
    target_manifest: pd.DataFrame,
    detailed_frames: list[pd.DataFrame],
) -> np.ndarray | None:
    """Return an OR-union lag significance mask aligned to a target manifest."""
    if not detailed_frames:
        return None
    significant: set[tuple[int, str, int]] = set()
    for frame in detailed_frames:
        required = {"subject", "electrode", "lag_idx", "sig"}
        if not required.issubset(frame.columns):
            raise ValueError(
                "significance_detailed.csv must contain subject, electrode, lag_idx, sig"
            )
        for row in frame.loc[frame["sig"].astype(bool)].itertuples(index=False):
            significant.add((int(row.subject), str(row.electrode), int(row.lag_idx)))
    return np.asarray(
        [
            (int(row.subject), str(row.electrode), int(row.lag_index)) in significant
            for row in target_manifest.itertuples(index=False)
        ],
        dtype=bool,
    )


def load_view_embeddings(
    config: RRRConfig,
    resolved: dict[str, ResolvedSubspaceRun],
    subject_ids: list[int],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load every view model over the configured target transcript."""
    target = config.encoding
    context = resolve_dataset_context(
        dataset=target.dataset,
        subject_ids=subject_ids,
        per_dataset_config=target.per_dataset_config,
        data_root=target.data_root,
    )
    loaded: dict[str, np.ndarray] = {}
    cache: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    reference_words = None
    for view in config.views:
        run_enc = resolved[view.name].config.encoding
        if int(run_enc.offset) != int(target.offset):
            raise ValueError(
                f"View {view.name} uses offset {run_enc.offset}; target uses {target.offset}"
            )
        identity = json.dumps(
            {"model": run_enc.model, "per_model_config": run_enc.per_model_config},
            sort_keys=True,
        )
        if identity not in cache:
            cache[identity] = load_embeddings(
                target.data_root,
                run_enc.model,
                dataset=target.dataset,
                per_dataset_config=target.per_dataset_config,
                dataset_context=context,
                **run_enc.per_model_config,
            )
        values, words = cache[identity]
        if reference_words is None:
            reference_words = words
        elif len(words) != len(reference_words) or not np.array_equal(
            words["word"].astype(str).to_numpy(),
            reference_words["word"].astype(str).to_numpy(),
        ):
            raise ValueError(f"Embedding transcript differs for view {view.name}")
        loaded[view.name] = np.asarray(values)
    if reference_words is None:
        raise ValueError("No multiview embeddings were loaded")
    return loaded, reference_words


def load_view_bases(
    config: RRRConfig,
    resolved: dict[str, ResolvedSubspaceRun],
    target_key: tuple[str, str],
    selection: np.ndarray,
    test_idxs: list[np.ndarray],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, str]]:
    """Load raw matching-fold A matrices or cross-dataset fold-mean bases."""
    bases: dict[str, dict[int, np.ndarray]] = {}
    modes: dict[str, str] = {}
    for view in config.views:
        run = resolved[view.name]
        rank = int(view.rank)
        if run.dataset_key == target_key:
            peer = load_base_comparison_data(run)
            if not np.array_equal(selection, peer.selection):
                raise ValueError(f"View {view.name} word selection differs from target")
            if len(peer.test_idxs) != len(test_idxs) or any(
                not np.array_equal(left, right)
                for left, right in zip(test_idxs, peer.test_idxs)
            ):
                raise ValueError(f"View {view.name} fold indices differ from target")
            fold_bases = {}
            for fold in range(len(test_idxs)):
                path = run.unit_dir / f"fold_{fold}" / "A.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Missing fold A matrix: {path}")
                matrix = np.load(path, mmap_mode="r")
                if matrix.ndim != 2 or matrix.shape[1] < rank:
                    raise ValueError(
                        f"View {view.name} fold {fold} cannot provide rank {rank}"
                    )
                fold_bases[fold] = np.asarray(matrix[:, :rank])
            bases[view.name] = fold_bases
            modes[view.name] = "matching_target_fold_raw"
        else:
            mean_basis = aggregate_subspace_from_unit_dir(
                run.unit_dir, rank=rank, method=config.subspace_aggregation
            )
            bases[view.name] = {
                fold: mean_basis for fold in range(len(test_idxs))
            }
            modes[view.name] = (
                f"cross_dataset_fold_{config.subspace_aggregation}"
            )
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
