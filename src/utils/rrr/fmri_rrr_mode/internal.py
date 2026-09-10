"""Within-fMRI RRR mode."""

from __future__ import annotations

from scripts.run_rrr import UnitSelector
from src.utils.shared.encoding import fit_transform_target_pca

from .common import (
    _fit_within_fmri_bases,
    evaluate_rank_jobs,
    fit_full_model,
    full_metrics_from_fit,
    load_stimulus_data,
    load_subject_responses,
    make_summary,
    save_full_fit,
    save_pca_factors,
    save_rrr_factors,
    save_summary_and_plot,
)


def run_internal(setup, target_pca_dim=None):
    """Fit and evaluate within-dataset fMRI RRR and PCA spaces."""
    config, artifact = setup.config, setup.artifact
    ranks = list(map(int, config.rank_grid))
    methods = ["rrr", "pca"]
    plan = artifact.resume_method_plan(methods, ranks, refresh=config.refresh)
    pending = plan["pending"]
    cached_full = (
        artifact.load_field("full_metrics")
        if not config.refresh and artifact.has_field("full_metrics")
        else None
    )
    factors_missing = any(
        config.refresh
        or not artifact.has_field(
            "input_factors", selector=artifact._selector(method=method, rank=rank)
        )
        for method in methods
        for rank in ranks
    )
    need_data = cached_full is None or factors_missing or bool(pending)
    data = train = test = None
    full_fit = None
    if need_data:
        data = load_stimulus_data(
            config, setup.train_stories, setup.test_stories, setup.groups
        )
        train, test = load_subject_responses(
            data, config, config.fmri.subjects, setup.masks
        )
        if target_pca_dim is not None:
            train.responses, test.responses = fit_transform_target_pca(
                train.responses,
                test.responses,
                int(target_pca_dim),
                context="fMRI training targets",
            )

    if cached_full is None or factors_missing:
        with setup.timed("fit full-embedding ridge ceiling"):
            full_fit = fit_full_model(
                train,
                test,
                data.cv,
                config,
                context={
                    "script": "fmri_rrr",
                    "mode": "internal",
                    "method": "full",
                    "n_features": train.stimulus.shape[1],
                    "standardize": False,
                },
                need_factors=factors_missing,
            )
        fresh_full = full_metrics_from_fit(full_fit, test)
        if cached_full is None:
            save_full_fit(artifact, config, full_fit, fresh_full)
            full_metrics = fresh_full
        else:
            full_metrics = cached_full
    else:
        full_metrics = cached_full
        print("Reusing cached full-embedding ridge metrics", flush=True)

    if factors_missing:
        with setup.timed("fit within-fMRI RRR and PCA bases"):
            bases, factors = _fit_within_fmri_bases(
                train, full_fit, data.ambient_dim, config
            )
        save_rrr_factors(
            artifact,
            config,
            UnitSelector(method="rrr"),
            bases["rrr"],
            factors,
            root_compat=True,
        )
        save_pca_factors(
            artifact,
            config,
            UnitSelector(method="pca"),
            bases["pca"],
            factors["pca_full"],
            root_compat=True,
        )
    else:
        bases = {
            method: {
                rank: artifact.load_field(
                    "input_factors",
                    selector=artifact._selector(method=method, rank=rank),
                )
                for rank in ranks
                if (method, rank) in pending
            }
            for method in methods
        }

    jobs = [(method, rank, bases[method][rank]) for method, rank in pending]
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
    manifest = None if target_pca_dim is not None else setup.target_manifest
    frame = make_summary(
        artifact, methods, ranks, full_metrics, computed, manifest=manifest
    )
    save_summary_and_plot(artifact, config, frame)
    metadata = setup.base_metadata(data=data, target_pca_dim=target_pca_dim)
    metadata.update(
        {
            "source_artifact": None,
            "source_mode": "within_fmri",
            "pca_available": True,
            "pca_training_representation": "tr_aligned_trimmed_story_zscored_pre_fir",
            "ceiling": "fresh_full_embedding_ridge",
        }
    )
    artifact.save_field("metadata", metadata, overwrite=True)
    required = {
        "target_manifest": [],
        "full_metrics": [],
        "mean_results": [],
        "rank_metrics": [
            artifact._selector(method=method, rank=rank)
            for method in methods
            for rank in ranks
        ],
        "input_factors": [
            artifact._selector(method=method, rank=rank)
            for method in methods
            for rank in ranks
        ],
    }
    artifact.mark_complete(required)
    return artifact


__all__ = ["run_internal"]
