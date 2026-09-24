"""Shared data, fitting, evaluation, and artifact helpers for fMRI RRR."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.run_rrr import Artifact, UnitSelector
from src.utils.rrr.core import canonical_qr, fit_dense_rrr
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.shared.encoding import extract_best_alphas, get_ridge_model_constructor
from src.utils.shared.fmri import (
    AlignedFMRIStimuli,
    PreparedFMRI,
    align_fmri_stimuli,
    build_fmri_stimulus,
    fit_pooled_embedding_scaler,
    columnwise_correlation,
    load_fmri_responses,
    load_story_embeddings,
    resolve_fmri_stories,
    session_cv_splits,
    target_manifest,
    TEMPORARILY_HELD_OUT_FMRI_STORIES,
)
from src.utils.rrr.significance import (
    load_significance_selection,
    validate_cross_dataset_significance_alignment,
)
from src.utils.shared.tensors import to_numpy
from src.utils.subspace_comparison.metrics import aggregate_subspace_basis


@dataclass
class FinalRidgeFit:
    """Predictions and fitted quantities needed by downstream fMRI RRR."""

    test_prediction: np.ndarray | None
    train_prediction: np.ndarray | None
    coefficients: np.ndarray | None
    best_alphas: np.ndarray
    alpha_grid: np.ndarray


@dataclass
class FMRIStimulusData:
    """Shared, response-free fMRI stimulus matrices for one model layer."""

    aligned: AlignedFMRIStimuli
    train: PreparedFMRI
    test: PreparedFMRI
    cv: list[tuple[np.ndarray, np.ndarray]]
    ambient_dim: int
    input_scaler: dict


@dataclass
class FMRIRunSetup:
    """Resolved Artifact, target selection, and story layout for one run."""

    config: object
    artifact: Artifact
    significance: object
    upstreams: list[dict]
    source: Artifact | None
    train_stories: list[str]
    test_stories: list[str]
    groups: dict[int, list[str]]
    masks: dict[str, np.ndarray]
    target_manifest: pd.DataFrame
    original_target_count: int

    def timed(self, label):
        return _timed(label)

    def base_metadata(self, *, data=None, target_pca_dim=None):
        config = self.config
        return {
            "subspace_aggregation": config.subspace_aggregation,
            "subspace_aggregation_weighting": (
                "prediction_singular_value_squared_energy_after_full_fir_component_normalization"
                if config.subspace_aggregation == "prediction_weighted"
                else None
            ),
            "rrr_svd_on_cpu": config.rrr_svd_on_cpu,
            "rrr_use_low_rank_svd": config.rrr_use_low_rank_svd,
            "rrr_low_rank_svd_seed": config.rrr_low_rank_svd_seed,
            "rrr_save_b_matrix": config.rrr_save_b_matrix,
            "pca_svd_device": config.pca_svd_device,
            "target_scaling": "stored_preprocessed_responses",
            "input_scaling": "pooled_training_story_standardization",
            "subjects": config.fmri.subjects,
            "sessions": config.fmri.sessions,
            "train_stories": self.train_stories,
            "test_story": self.test_stories[0],
            "temporarily_held_out_stories": sorted(TEMPORARILY_HELD_OUT_FMRI_STORIES),
            "trim": config.fmri.trim,
            "fixed_leading_trim": 5,
            "effective_start_trim": 5 + config.fmri.trim,
            "effective_end_trim": config.fmri.trim,
            "ndelays": config.fmri.ndelays,
            "limit_to_glove": config.fmri.limit_to_glove,
            "glove_dim": config.fmri.glove_dim,
            "glove_oov_handling": (
                "zero_before_lanczos_interpolation"
                if config.fmri.limit_to_glove
                else None
            ),
            "delay_order": list(range(1, int(config.fmri.ndelays) + 1)),
            "pre_fir_training_shape": (
                list(data.train.pre_fir_stimulus.shape) if data is not None else None
            ),
            "rank_grid": list(map(int, config.rank_grid)),
            "target_pca_dim": int(target_pca_dim) if target_pca_dim is not None else None,
            "original_target_count": int(self.original_target_count),
        }


@contextmanager
def _timed(label: str):
    start = time.perf_counter()
    print(f"[{label}] starting", flush=True)
    try:
        yield
    finally:
        print(f"[{label}] finished in {time.perf_counter() - start:.1f}s", flush=True)


def _array_summary(name: str, values: np.ndarray) -> str:
    size_gib = values.nbytes / (1024 ** 3)
    return f"{name}: shape={values.shape}, dtype={values.dtype}, size={size_gib:.2f} GiB"


def _fisher_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    clipped = np.clip(values[finite], -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(clipped))))


def _r2_per_target(true, predicted):
    numerator = np.sum((true - predicted) ** 2, axis=0)
    denominator = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    return np.nan_to_num(1.0 - numerator / denominator, nan=0.0, neginf=0.0)


def _fit_final_ridge(
    X_train,
    Y_train,
    X_test,
    cv,
    config,
    context,
    *,
    return_train_prediction=False,
    return_coefficients=False,
):
    """Fit one ridge model and return only the requested large fitted values."""
    explicit = np.asarray(config.ridge.alphas) if config.ridge.alphas else None
    alphas = resolve_ridge_alphas(
        explicit,
        X_train,
        standardize=False,
        context=context,
        cache_dir=None,
        policy=config.ridge.alpha_grid,
    )
    model = get_ridge_model_constructor(
        alphas=alphas,
        cv=cv,
        standardize=False,
        solver_params=config.ridge.solver_params,
        y_in_cpu=config.ridge.y_in_cpu,
        force_cpu=config.ridge.force_cpu,
    )(0)
    model.fit(X_train, Y_train)
    test_prediction = (
        to_numpy(model.predict(X_test)).astype(np.float32)
        if X_test is not None
        else None
    )
    train_prediction = None
    if return_train_prediction:
        train_prediction = to_numpy(model.predict(X_train)).astype(np.float32)
    coefficients = None
    if return_coefficients:
        coefficients = to_numpy(model.named_steps["ridgecv"].coef_)
        if coefficients.shape[0] != X_train.shape[1]:
            coefficients = coefficients.T
        expected = (X_train.shape[1], Y_train.shape[1])
        if coefficients.shape != expected:
            raise ValueError(
                "Unexpected full-ridge coefficient shape "
                f"{coefficients.shape}; expected {expected}"
            )
    return FinalRidgeFit(
        test_prediction=test_prediction,
        train_prediction=train_prediction,
        coefficients=(
            np.asarray(coefficients, dtype=np.float32)
            if coefficients is not None
            else None
        ),
        best_alphas=extract_best_alphas(model),
        alpha_grid=np.asarray(alphas),
    )


def _pca_right_singular_vectors(centered: np.ndarray, requested_device: str) -> np.ndarray:
    """Compute exact PCA right singular vectors on the requested device."""
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and torch.cuda.is_available()
    )
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("pca_svd_device='cuda' requested but CUDA is unavailable")
    if not use_cuda:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt
    try:
        tensor = torch.as_tensor(centered, device="cuda")
        _, _, vh = torch.linalg.svd(tensor, full_matrices=False)
        return vh.cpu().numpy()
    except RuntimeError:
        if requested_device != "auto":
            raise
        print("CUDA PCA SVD failed; retrying exact SVD on CPU", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt


def _fit_rrr_bases(full_train, ridge_fit, ambient_dim, config):
    """Fit full FIR RRR and aggregate its delay blocks into rank bases."""
    ranks = sorted(set(map(int, config.rank_grid)))
    max_rank = max(ranks)
    ndelays = int(config.fmri.ndelays)
    expected_width = ndelays * ambient_dim
    if full_train.stimulus.shape[1] != expected_width:
        raise ValueError(
            f"FIR design has width {full_train.stimulus.shape[1]}; expected "
            f"ndelays * embedding_dim = {expected_width}"
        )
    available_rank = min(
        full_train.stimulus.shape[0], full_train.responses.shape[1], ambient_dim
    )
    if max_rank > available_rank:
        raise ValueError(
            f"Requested fMRI RRR rank {max_rank} exceeds shared rank limit "
            f"{available_rank}"
        )
    if ridge_fit.train_prediction is None or ridge_fit.coefficients is None:
        raise ValueError("Within-fMRI factor fitting requires full-ridge fitted values")
    result = fit_dense_rrr(
        full_train.stimulus,
        full_train.responses,
        k=max_rank,
        selected_idx=np.arange(expected_width),
        precomputed_Y_hat_ols=ridge_fit.train_prediction,
        precomputed_beta_full=ridge_fit.coefficients,
        svd_on_cpu=config.rrr_svd_on_cpu,
        use_low_rank_svd=config.rrr_use_low_rank_svd,
        low_rank_svd_seed=config.rrr_low_rank_svd_seed,
        return_b=config.rrr_save_b_matrix,
    )
    full_fir = np.asarray(result["A_dense"], dtype=np.float32)
    singular_values = np.asarray(result["singular_values"], dtype=np.float32)
    delay_blocks = list(np.split(full_fir, ndelays, axis=0))
    bases = {
        rank: aggregate_subspace_basis(
            delay_blocks,
            rank=rank,
            method=config.subspace_aggregation,
            prediction_singular_values=singular_values,
        )
        for rank in ranks
    }
    factors = {
        "full_fir": full_fir,
        "target": result.get("B_dense") if config.rrr_save_b_matrix else None,
        "delay_blocks": delay_blocks,
        "prediction_singular_values": singular_values,
    }
    return bases, factors


def _fit_pca_bases(full_train, config):
    """Fit pre-FIR PCA bases for every configured rank."""
    ranks = sorted(set(map(int, config.rank_grid)))
    centered = full_train.pre_fir_stimulus - full_train.pre_fir_stimulus.mean(
        axis=0, keepdims=True
    )
    device = getattr(config, "pca_svd_device", "auto")
    with _timed(f"exact PCA SVD ({device})"):
        vt = _pca_right_singular_vectors(centered, device)
    pca_full = vt.T.astype(np.float32, copy=False)
    if max(ranks) > pca_full.shape[1]:
        raise ValueError(
            f"Requested fMRI PCA rank {max(ranks)} exceeds available rank "
            f"{pca_full.shape[1]}"
        )
    return {rank: pca_full[:, :rank] for rank in ranks}, pca_full


def _fit_within_fmri_bases(full_train, ridge_fit, ambient_dim, config):
    """Compatibility helper returning the combined RRR/PCA factor payload."""
    rrr_bases, rrr_factors = _fit_rrr_bases(
        full_train, ridge_fit, ambient_dim, config
    )
    pca_bases, pca_full = _fit_pca_bases(full_train, config)
    return {"rrr": rrr_bases, "pca": pca_bases}, {
        **rrr_factors,
        "pca_full": pca_full,
    }


def prepare_fmri_run(config, target_pca_dim=None):
    """Resolve dependencies, selections, Artifact identity, and story layout."""
    config.validate()
    if target_pca_dim is not None and config.source_mode != "within_fmri":
        raise ValueError("target_pca_dim is supported only for internal fMRI RRR")
    significance = load_significance_selection(
        config.significance_selection,
        results_root=config.results_root,
        subjects=config.fmri.subjects,
        modality="fmri",
    )
    significance.significant_voxels = {
        subject: frame.rename(columns={"target_id": "voxel_index", "significant": "sig"})
        for subject, frame in significance.target_significance.items()
    }
    upstreams = [significance.reference.identity_record()]
    for identity in significance.source_identities:
        if identity not in upstreams:
            upstreams.append(identity)
    source = None
    if config.source_mode == "cross_dataset":
        source_kwargs = dict(config.ieeg_rrr_source or {})
        source_kwargs.setdefault("results_root", config.results_root)
        source = Artifact(**source_kwargs)
        if source.variant is None:
            source._resolve_variant()
        source.manifest()
        upstreams.append(source.identity_record())
    if config.source_mode == "multi_view":
        from src.utils.rrr.multiview import resolve_rrr_views

        for run in resolve_rrr_views(config).values():
            identity = run.artifact.identity_record()
            if identity not in upstreams:
                upstreams.append(identity)

    artifact_cls = Artifact
    if target_pca_dim is not None:
        from scripts.run_target_dimensionality_rrr import Artifact as TargetDimArtifact

        artifact_cls = TargetDimArtifact
    artifact = artifact_cls.from_fmri_config(config)
    variant_extras = dict(getattr(artifact, "variant_extras", {}))
    if target_pca_dim is not None:
        artifact.target_pca_dim = int(target_pca_dim)
        variant_extras["target_pca_dim"] = int(target_pca_dim)
    if source is not None:
        validate_cross_dataset_significance_alignment(
            config.significance_selection,
            artifact.identity,
            source.requested_significance_selection,
            source.identity,
        )
    artifact.begin(
        config=config,
        upstreams=upstreams,
        refresh=config.refresh,
        variant_extras=variant_extras or None,
    )
    artifact.save_field("effective_config", asdict(config), overwrite=config.refresh)
    artifact.save_field("upstream_identities", upstreams, overwrite=config.refresh)

    train_stories, test_stories, groups = resolve_fmri_stories(
        config.fmri.fmri_repo_root, config.fmri.sessions
    )
    if len(test_stories) != 1:
        raise ValueError(f"Expected one shared test story, got {test_stories}")
    masks = {}
    for subject in config.fmri.subjects:
        frame = significance.significant_voxels[str(subject)].sort_values("voxel_index")
        expected = np.arange(len(frame))
        if not np.array_equal(frame["voxel_index"].to_numpy(), expected):
            raise ValueError(f"Non-contiguous voxel indices for {subject}")
        masks[subject] = frame["sig"].astype(bool).to_numpy()
        if not masks[subject].any():
            raise ValueError(f"No significant voxels found for {subject}")
    manifest = target_manifest(config.fmri.subjects, masks)
    original_target_count = len(manifest)
    if target_pca_dim is not None:
        manifest = pd.DataFrame(
            {
                "target_id": [f"pc-{index + 1}" for index in range(int(target_pca_dim))],
                "target_kind": "target_pca_component",
                "component": np.arange(1, int(target_pca_dim) + 1, dtype=int),
            }
        )
    if config.source_mode != "leave_one_subject_out":
        artifact.save_field("target_manifest", manifest, overwrite=config.refresh)
    return FMRIRunSetup(
        config=config,
        artifact=artifact,
        significance=significance,
        upstreams=upstreams,
        source=source,
        train_stories=train_stories,
        test_stories=test_stories,
        groups=groups,
        masks=masks,
        target_manifest=manifest,
        original_target_count=original_target_count,
    )


def load_stimulus_data(config, train_stories, test_stories, groups):
    """Load and align embeddings, then build shared response-free FIR designs."""
    all_stories = train_stories + test_stories
    with _timed("load pooled story embeddings"):
        embeddings = load_story_embeddings(
            config.fmri.embedding_root,
            config.model_dir,
            all_stories,
            config.layer,
            config.pooling,
            fmri_repo_root=config.fmri.fmri_repo_root,
            glove_data_root=config.fmri.glove_data_root,
            limit_to_glove=config.fmri.limit_to_glove,
            glove_dim=config.fmri.glove_dim,
        )
    ambient_dim = next(iter(embeddings.values())).shape[1]
    with _timed("Lanczos-align all stories"):
        aligned = align_fmri_stimuli(
            config.fmri.fmri_repo_root, all_stories, embeddings, trim=config.fmri.trim
        )
    input_scaler = fit_pooled_embedding_scaler(aligned, train_stories)
    train = build_fmri_stimulus(
        aligned, train_stories, ndelays=config.fmri.ndelays,
        input_scaler=input_scaler,
    )
    test = build_fmri_stimulus(
        aligned, test_stories, ndelays=config.fmri.ndelays,
        input_scaler=input_scaler,
    )
    print(_array_summary("full train design", train.stimulus), flush=True)
    return FMRIStimulusData(
        aligned=aligned,
        train=train,
        test=test,
        cv=session_cv_splits(train.story_slices, groups),
        ambient_dim=ambient_dim,
        input_scaler=input_scaler,
    )


def load_subject_responses(data, config, subjects, masks):
    """Attach one subject partition's response matrices to shared designs."""
    with _timed(f"load selected train responses ({len(subjects)} subject(s))"):
        train_y = load_fmri_responses(
            config.fmri.fmri_repo_root,
            list(data.train.story_slices),
            data.train.story_slices,
            subjects,
            masks,
        )
    with _timed(f"load selected test responses ({len(subjects)} subject(s))"):
        test_y = load_fmri_responses(
            config.fmri.fmri_repo_root,
            list(data.test.story_slices),
            data.test.story_slices,
            subjects,
            masks,
        )
    print(_array_summary("train responses", train_y), flush=True)
    train = PreparedFMRI(
        stimulus=data.train.stimulus,
        pre_fir_stimulus=data.train.pre_fir_stimulus,
        responses=train_y,
        story_slices=data.train.story_slices,
    )
    test = PreparedFMRI(
        stimulus=data.test.stimulus,
        pre_fir_stimulus=data.test.pre_fir_stimulus,
        responses=test_y,
        story_slices=data.test.story_slices,
    )
    return train, test


def _subject_summary(values, manifest):
    return {
        subject: _fisher_mean(values[manifest["subject"].to_numpy() == subject])
        for subject in manifest["subject"].unique()
    }


def _build_rank_designs(
    aligned, train_stories, test_stories, basis, ndelays, input_scaler,
    standardize_reduced,
):
    # The ambient source scaler is shared across ranks. This second scaler is
    # intentionally rank-specific: it is fit after projection on the target
    # training stories and therefore has one statistic per retained coordinate.
    reduced_scaler = (
        fit_pooled_embedding_scaler(
            aligned, train_stories, input_scaler=input_scaler, basis=basis
        )
        if standardize_reduced else None
    )
    train = build_fmri_stimulus(
        aligned, train_stories, ndelays=ndelays, basis=basis,
        input_scaler=input_scaler, projected_scaler=reduced_scaler,
    )
    test = build_fmri_stimulus(
        aligned, test_stories, ndelays=ndelays, basis=basis,
        input_scaler=input_scaler, projected_scaler=reduced_scaler,
    )
    return train.stimulus, test.stimulus


def _iter_rank_designs(
    jobs, aligned, train_stories, test_stories, ndelays, input_scaler,
    standardize_reduced,
):
    """Build and yield one rank design at a time to bound peak memory."""
    for method, rank, basis in jobs:
        yield method, rank, *_build_rank_designs(
            aligned, train_stories, test_stories, basis, ndelays, input_scaler,
            standardize_reduced,
        )


def fit_full_model(
    train,
    test,
    cv,
    config,
    *,
    context,
    need_factors,
):
    """Fit a full-embedding model for evaluation and/or RRR factor extraction."""
    return _fit_final_ridge(
        train.stimulus,
        train.responses,
        test.stimulus if test is not None else None,
        cv,
        config,
        context,
        return_train_prediction=need_factors,
        return_coefficients=need_factors,
    )


def full_metrics_from_fit(fit, test):
    if fit.test_prediction is None:
        raise ValueError("Full-model metric evaluation requires test predictions")
    corr = columnwise_correlation(test.responses, fit.test_prediction)
    r2 = _r2_per_target(test.responses, fit.test_prediction)
    return {"corr_per_target": corr, "r2_per_target": r2}


def save_full_fit(artifact, config, fit, metrics, *, selector=None):
    payload = dict(metrics)
    if config.ridge.log_alphas:
        payload["best_alphas"] = fit.best_alphas
    artifact.save_field(
        "full_metrics", payload, selector=selector, overwrite=config.refresh
    )
    if config.ridge.log_alphas:
        diagnostics_selector = dict(selector or {})
        diagnostics_selector.update({"method": "full"})
        artifact.save_field(
            "diagnostics",
            {"alpha_grid": fit.alpha_grid, "best_alphas": fit.best_alphas},
            selector=diagnostics_selector,
            overwrite=config.refresh,
        )


def save_rrr_factors(artifact, config, unit, bases, factors, *, root_compat=False):
    """Persist aggregated bases and fMRI-specific full FIR factor diagnostics."""
    for rank, basis in bases.items():
        artifact.save_factors(
            unit=unit,
            rank=rank,
            factors=basis,
            overwrite=config.refresh,
        )
    full_selector = (
        UnitSelector(method="rrr_full_fir") if root_compat else unit
    )
    full_basis = None if root_compat else "full_fir"
    artifact.save_factors(
        unit=full_selector, basis=full_basis, factors=factors["full_fir"],
        target_factors=factors["target"], overwrite=config.refresh,
    )
    artifact.save_field(
        "diagnostics",
        factors["prediction_singular_values"],
        selector=artifact._selector(
            full_selector, basis=full_basis, name="prediction_singular_values"
        ),
        overwrite=config.refresh,
    )
    for delay, block in enumerate(factors["delay_blocks"], start=1):
        delay_unit = UnitSelector(method="rrr_delay") if root_compat else unit
        artifact.save_factors(
            unit=delay_unit, basis=None if root_compat else "delay_block",
            delay=delay, factors=block, overwrite=config.refresh,
        )


def save_pca_factors(artifact, config, unit, bases, pca_full, *, root_compat=False):
    for rank, basis in bases.items():
        artifact.save_factors(
            unit=unit, rank=rank, factors=basis, overwrite=config.refresh
        )
    full_unit = UnitSelector(method="pca_full") if root_compat else unit
    artifact.save_factors(
        unit=full_unit, basis=None if root_compat else "full", factors=pca_full,
        overwrite=config.refresh,
    )


def evaluate_rank_jobs(
    *,
    artifact,
    config,
    data,
    train,
    test,
    train_stories,
    test_stories,
    jobs,
    unit_by_method=None,
    orthonormalize=False,
    input_scaler=None,
    standardize_reduced=False,
):
    """Fit rank-reduced ridge models and write each result to the Artifact."""
    if not jobs:
        return {}
    normalized_jobs = []
    for method, rank, basis in jobs:
        value = canonical_qr(basis)[0] if orthonormalize else basis
        expected = (data.ambient_dim, rank)
        if value.shape != expected:
            raise ValueError(
                f"{method.upper()} rank {rank} basis has shape {value.shape}; "
                f"expected {expected}"
            )
        normalized_jobs.append((method, rank, value))
    computed = {}
    for method, rank, train_design, test_design in _iter_rank_designs(
        normalized_jobs,
        data.aligned,
        train_stories,
        test_stories,
        int(config.fmri.ndelays),
        data.input_scaler if input_scaler is None else input_scaler,
        standardize_reduced,
    ):
        expected_width = int(config.fmri.ndelays) * rank
        if train_design.shape[1] != expected_width:
            raise ValueError(
                f"{method.upper()} rank {rank} FIR design has shape {train_design.shape}"
            )
        with _timed(f"fit {method.upper()} rank={rank}"):
            fitted = _fit_final_ridge(
                train_design,
                train.responses,
                test_design,
                data.cv,
                config,
                {
                    "script": "fmri_rrr",
                    "method": method,
                    "rank": rank,
                    "n_features": train_design.shape[1],
                    "standardize": False,
                },
            )
        corr = columnwise_correlation(test.responses, fitted.test_prediction)
        r2 = _r2_per_target(test.responses, fitted.test_prediction)
        metrics = {"corr_per_target": corr, "r2_per_target": r2}
        if config.ridge.log_alphas:
            metrics["best_alphas"] = fitted.best_alphas
        unit = (unit_by_method or {}).get(method)
        artifact.save_rank_metrics(
            unit=unit,
            method=None if unit is not None else method,
            rank=rank,
            metrics=metrics,
            overwrite=config.refresh,
        )
        if config.ridge.log_alphas:
            artifact.save_field(
                "diagnostics",
                {"alpha_grid": fitted.alpha_grid, "best_alphas": fitted.best_alphas},
                selector=artifact._selector(
                    unit, method=None if unit is not None else method, rank=rank
                ),
                overwrite=config.refresh,
            )
        computed[(method, rank)] = (corr, r2)
        print(
            f"{method.upper()} rank={rank}: "
            f"r={_fisher_mean(corr):.4f}, R2={float(np.mean(r2)):.4f}",
            flush=True,
        )
    return computed


def load_rank_metrics(artifact, method, rank, unit=None):
    loaded = artifact.load_rank_metrics(
        unit=unit, method=None if unit is not None else method, rank=rank
    )
    return np.asarray(loaded["corr_per_target"]), np.asarray(loaded["r2_per_target"])


def make_summary(
    artifact,
    methods,
    ranks,
    full_metrics,
    computed,
    *,
    manifest=None,
    unit_by_method=None,
):
    full_corr = _fisher_mean(full_metrics["corr_per_target"])
    full_r2 = float(np.mean(full_metrics["r2_per_target"]))
    rows = []
    for method in methods:
        unit = (unit_by_method or {}).get(method)
        for rank in ranks:
            if (method, rank) in computed:
                corr, r2 = computed[(method, rank)]
            else:
                corr, r2 = load_rank_metrics(artifact, method, rank, unit=unit)
            row = {
                "method": method,
                "rank": rank,
                "corr": _fisher_mean(corr),
                "r2": float(np.mean(r2)),
                "full_corr": full_corr,
                "full_r2": full_r2,
            }
            if manifest is not None and "subject" in manifest:
                row.update(
                    {
                        f"corr_{subject}": value
                        for subject, value in _subject_summary(corr, manifest).items()
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "rank"]).reset_index(drop=True)


def _plot_results(frame: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, group in frame.groupby("method"):
        ordered = group.sort_values("rank")
        axes[0].plot(ordered["rank"], ordered["corr"], marker="o", label=method.upper())
        axes[1].plot(ordered["rank"], ordered["r2"], marker="o", label=method.upper())
    if len(frame):
        axes[0].axhline(frame["full_corr"].iloc[0], color="black", ls="--", label="Full")
        axes[1].axhline(frame["full_r2"].iloc[0], color="black", ls="--", label="Full")
    for ax, ylabel in zip(axes, ["Fisher-z mean Pearson r", "Mean R²"]):
        ax.set_xscale("log", base=2)
        ax.set(xlabel="Rank", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_title("Primary encoding performance")
    axes[1].set_title("Variance explained")
    fig.tight_layout()
    return fig


def save_summary_and_plot(artifact, config, frame, *, unit=None):
    artifact.save_field(
        "mean_results",
        frame,
        selector=artifact._selector(unit),
        overwrite=True,
    )
    if config.plot:
        figure = _plot_results(frame)
        name = "rank_analysis"
        artifact.save_plot(name=name, figure=figure, unit=unit, overwrite=True)
        plt.close(figure)


def pending_ranks(artifact, config, unit, *, require_factors=False):
    pending = []
    for rank in map(int, config.rank_grid):
        metric_selector = artifact._selector(unit, rank=rank)
        factor_selector = artifact._selector(unit, rank=rank)
        complete = artifact.has_field("rank_metrics", selector=metric_selector)
        if require_factors:
            complete &= artifact.has_field("input_factors", selector=factor_selector)
        if config.refresh or not complete:
            pending.append(rank)
    return pending


def release_memory():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


__all__ = [
    "FMRIRunSetup",
    "FMRIStimulusData",
    "FinalRidgeFit",
    "UnitSelector",
    "_fisher_mean",
    "_fit_final_ridge",
    "_fit_pca_bases",
    "_fit_rrr_bases",
    "_fit_within_fmri_bases",
    "_pca_right_singular_vectors",
    "_plot_results",
    "evaluate_rank_jobs",
    "fit_full_model",
    "full_metrics_from_fit",
    "load_rank_metrics",
    "load_stimulus_data",
    "load_subject_responses",
    "make_summary",
    "pending_ranks",
    "prepare_fmri_run",
    "release_memory",
    "save_full_fit",
    "save_pca_factors",
    "save_rrr_factors",
    "save_summary_and_plot",
]
