"""Multi-view reduced-rank regression on the held-out-story fMRI dataset."""

from __future__ import annotations

import numpy as np

from scripts.run_rrr import UnitSelector
from src.utils.rrr.fmri_rrr_mode.common import (
    FMRIStimulusData,
    FinalRidgeFit,
    _fit_rrr_bases,
    evaluate_rank_jobs,
    full_metrics_from_fit,
    load_subject_responses,
    make_summary,
    save_full_fit,
    save_rrr_factors,
    save_summary_and_plot,
)
from src.utils.rrr.multiview import (
    fit_banded_ridge_baseline,
    load_global_view_transforms,
    resolve_rrr_views,
)
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.shared.fmri import (
    AlignedFMRIStimuli,
    apply_embedding_scaler,
    align_fmri_stimuli,
    build_fmri_stimulus,
    load_story_embeddings,
    session_cv_splits,
)


def _identity_scaler(width: int) -> dict:
    return {
        "mean": np.zeros(width, dtype=np.float32),
        "scale": np.ones(width, dtype=np.float32),
        "coordinates": "standardized-v1",
    }


def _load_projected_component_views(
    config, resolved, factors, scalers, stories, *, align_to_tr
):
    by_story = {story: [] for story in stories}
    reference_rows = None
    for view in config.views:
        run = resolved[view.name]
        spec = run.config.embedding_spec
        embeddings = load_story_embeddings(
            config.fmri.embedding_root,
            spec.model_dir,
            stories,
            spec.layer,
            spec.pooling,
            fmri_repo_root=config.fmri.fmri_repo_root,
            glove_data_root=config.fmri.glove_data_root,
            limit_to_glove=config.fmri.limit_to_glove,
            glove_dim=config.fmri.glove_dim,
        )
        values = (
            align_fmri_stimuli(
                config.fmri.fmri_repo_root,
                stories,
                embeddings,
                trim=config.fmri.trim,
            ).stories
            if align_to_tr
            else embeddings
        )
        factor = np.asarray(factors[view.name], dtype=np.float32)
        rows = {story: len(values[story]) for story in stories}
        if reference_rows is not None and rows != reference_rows:
            raise ValueError(
                f"{'TR' if align_to_tr else 'Word'} alignment differs for "
                f"multiview component {view.name!r}"
            )
        reference_rows = rows
        for story in stories:
            standardized = apply_embedding_scaler(
                values[story], scalers[view.name]
            )
            if standardized.shape[1] != factor.shape[0]:
                raise ValueError(
                    f"View {view.name!r} embedding/factor dimensions differ: "
                    f"{standardized.shape[1]} vs {factor.shape[0]}"
                )
            by_story[story].append(
                np.asarray(standardized @ factor, dtype=np.float32)
            )
    return {
        story: np.concatenate(parts, axis=1) for story, parts in by_story.items()
    }


def load_projected_fmri_views(config, resolved, factors, scalers, stories):
    """Load each ambient story space once and retain only projected TR rows."""
    return AlignedFMRIStimuli(
        stories=_load_projected_component_views(
            config,
            resolved,
            factors,
            scalers,
            stories,
            align_to_tr=True,
        )
    )


def load_projected_fmri_word_views(config, resolved, factors, scalers, stories):
    """Build the combined component space on the original story word rows."""
    return _load_projected_component_views(
        config,
        resolved,
        factors,
        scalers,
        stories,
        align_to_tr=False,
    )


def _view_delay_bands(design, view_widths, ndelays):
    """Split a delay-major FIR design into delay-major/view-major bands."""
    ambient = int(sum(view_widths))
    if design.shape[1] != int(ndelays) * ambient:
        raise ValueError(
            f"FIR design width {design.shape[1]} does not match "
            f"ndelays * multiview width = {int(ndelays) * ambient}"
        )
    offsets = np.cumsum([0, *view_widths])
    bands, boundaries, labels = [], [], []
    cursor = 0
    for delay in range(1, int(ndelays) + 1):
        delay_start = (delay - 1) * ambient
        for view_index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            band = design[:, delay_start + start : delay_start + stop]
            bands.append(band)
            boundaries.append((cursor, cursor + band.shape[1]))
            labels.append((delay, view_index))
            cursor += band.shape[1]
    return bands, boundaries, labels


def _load_multiview_data(setup, resolved, factors, scalers):
    config = setup.config
    all_stories = setup.train_stories + setup.test_stories
    aligned = load_projected_fmri_views(
        config, resolved, factors, scalers, all_stories
    )
    ambient_dim = sum(int(view.rank) for view in config.views)
    identity = _identity_scaler(ambient_dim)
    train = build_fmri_stimulus(
        aligned,
        setup.train_stories,
        ndelays=config.fmri.ndelays,
        input_scaler=identity,
    )
    test = build_fmri_stimulus(
        aligned,
        setup.test_stories,
        ndelays=config.fmri.ndelays,
        input_scaler=identity,
    )
    return FMRIStimulusData(
        aligned=aligned,
        train=train,
        test=test,
        cv=session_cv_splits(train.story_slices, setup.groups),
        ambient_dim=ambient_dim,
        input_scaler=identity,
    )


def run_multi_view(setup):
    """Fit a view-by-delay banded ceiling and its fMRI RRR rank sweep."""
    config, artifact = setup.config, setup.artifact
    ranks = list(map(int, config.rank_grid))
    unit = UnitSelector(method="rrr")
    plan = artifact.resume_method_plan(["rrr"], ranks, refresh=config.refresh)
    pending = plan["pending"]
    cached_full = (
        artifact.load_field("full_metrics")
        if not config.refresh and artifact.has_field("full_metrics")
        else None
    )
    factors_missing = (
        config.refresh
        or not artifact.has_field("input_scalers")
        or any(
            not artifact.has_field(
                "input_factors", selector=artifact._selector(method="rrr", rank=rank)
            )
            for rank in ranks
        )
    )

    resolved = resolve_rrr_views(config)
    component_factors, component_scalers, basis_modes = load_global_view_transforms(
        config, resolved
    )
    data = train = test = None
    fitted_banded = None
    if cached_full is None or factors_missing or pending:
        with setup.timed("load and project multiview story embeddings"):
            data = _load_multiview_data(
                setup, resolved, component_factors, component_scalers
            )
        train, test = load_subject_responses(
            data, config, config.fmri.subjects, setup.masks
        )
        if config.refresh or not artifact.has_field("input_scalers"):
            artifact.save_field(
                "input_scalers", data.input_scaler, overwrite=config.refresh
            )

    if cached_full is None or factors_missing:
        widths = [int(view.rank) for view in config.views]
        train_bands, boundaries, indexed_labels = _view_delay_bands(
            train.stimulus, widths, config.fmri.ndelays
        )
        test_bands, test_boundaries, test_labels = _view_delay_bands(
            test.stimulus, widths, config.fmri.ndelays
        )
        if boundaries != test_boundaries or indexed_labels != test_labels:
            raise ValueError("Training and test multiview FIR bands differ")
        explicit = (
            np.asarray(config.banded_ridge_alphas, dtype=float)
            if config.banded_ridge_alphas is not None
            else None
        )
        banded_alphas = resolve_ridge_alphas(
            explicit,
            train.stimulus,
            standardize=False,
            context={
                "script": "fmri_rrr",
                "mode": "multi_view",
                "method": "banded_full",
                "n_features": train.stimulus.shape[1],
                "bands": len(train_bands),
                "standardize": False,
            },
            cache_dir=None,
            policy=config.ridge.alpha_grid,
        )
        with setup.timed("fit view-by-delay banded ridge ceiling"):
            fitted_banded = fit_banded_ridge_baseline(
                train_bands,
                test_bands,
                train.responses,
                alphas=banded_alphas,
                solver_params=config.banded_ridge_solver_params,
                cv=data.cv,
                y_in_cpu=config.ridge.y_in_cpu,
                force_cpu=config.ridge.force_cpu,
            )
        full_fit = FinalRidgeFit(
            test_prediction=fitted_banded.Y_hat_test,
            train_prediction=fitted_banded.Y_hat_train,
            coefficients=fitted_banded.beta_full_raw,
            best_alphas=fitted_banded.best_alphas,
            alpha_grid=banded_alphas,
        )
        fresh_full = full_metrics_from_fit(full_fit, test)
        if cached_full is None:
            save_full_fit(artifact, config, full_fit, fresh_full)
            full_metrics = fresh_full
        else:
            full_metrics = cached_full
        if config.ridge.log_alphas:
            artifact.save_field(
                "diagnostics",
                {
                    "band_labels": [
                        {
                            "delay": delay,
                            "view": config.views[view_index].name,
                        }
                        for delay, view_index in indexed_labels
                    ],
                    "band_boundaries": boundaries,
                    "alphas": banded_alphas,
                    "deltas": fitted_banded.deltas,
                    "kernel_weights": np.exp(fitted_banded.deltas),
                    "best_alphas": fitted_banded.best_alphas,
                    "intercept": fitted_banded.intercept,
                },
                selector={"method": "banded_full"},
                overwrite=config.refresh,
            )
    else:
        full_metrics = cached_full

    if factors_missing:
        ridge_fit = FinalRidgeFit(
            test_prediction=fitted_banded.Y_hat_test,
            train_prediction=fitted_banded.Y_hat_train,
            coefficients=fitted_banded.beta_full_raw,
            best_alphas=fitted_banded.best_alphas,
            alpha_grid=np.asarray([]),
        )
        with setup.timed("fit and aggregate multiview full-FIR RRR factors"):
            bases, factors = _fit_rrr_bases(
                train, ridge_fit, data.ambient_dim, config
            )
        save_rrr_factors(
            artifact, config, unit, bases, factors, root_compat=True
        )
    else:
        bases = {
            rank: artifact.load_field(
                "input_factors",
                selector=artifact._selector(method="rrr", rank=rank),
            )
            for method, rank in pending
            if method == "rrr"
        }

    jobs = [("rrr", rank, bases[rank]) for method, rank in pending if method == "rrr"]
    computed = evaluate_rank_jobs(
        artifact=artifact,
        config=config,
        data=data,
        train=train,
        test=test,
        train_stories=setup.train_stories,
        test_stories=setup.test_stories,
        jobs=jobs,
    )
    frame = make_summary(
        artifact, ["rrr"], ranks, full_metrics, computed, manifest=setup.target_manifest
    )
    save_summary_and_plot(artifact, config, frame)
    metadata = setup.base_metadata(data=data)
    metadata.update(
        {
            "source_mode": "multi_view",
            "multi_view_name": config.multi_view_name,
            "views": [view.name for view in config.views],
            "view_ranks": {view.name: int(view.rank) for view in config.views},
            "basis_modes": basis_modes,
            "band_granularity": "view_by_delay",
            "band_order": "delay_major_then_view_major",
            "n_bands": len(config.views) * int(config.fmri.ndelays),
            "ceiling": "view_by_delay_banded_ridge",
            "pca_available": False,
        }
    )
    artifact.save_field("metadata", metadata, overwrite=True)
    artifact.save_field(
        "mode_metadata", metadata, selector={"kind": "multi_view"}, overwrite=True
    )
    artifact.mark_complete(
        {
            "target_manifest": [],
            "full_metrics": [],
            "mean_results": [],
            "rank_metrics": [
                artifact._selector(method="rrr", rank=rank) for rank in ranks
            ],
            "input_factors": [
                artifact._selector(method="rrr", rank=rank) for rank in ranks
            ],
            "input_scalers": [{}],
        }
    )
    return artifact


__all__ = [
    "_view_delay_bands",
    "load_projected_fmri_views",
    "load_projected_fmri_word_views",
    "run_multi_view",
]
