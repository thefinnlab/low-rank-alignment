"""iEEG-to-fMRI cross-dataset RRR mode."""

from __future__ import annotations

from .common import (
    evaluate_rank_jobs,
    fit_full_model,
    full_metrics_from_fit,
    load_stimulus_data,
    load_subject_responses,
    make_summary,
    save_full_fit,
    save_summary_and_plot,
)


def run_cross_dataset(setup):
    """Fit fMRI ridge models in input spaces learned from an iEEG source."""
    from src.utils.rrr.ieeg_rrr_mode.cross_dataset_helpers import (
        load_cross_dataset_A_by_rank,
    )
    from src.utils.rrr.ieeg_source_bases import fit_whole_stimulus_ieeg_bases

    config, artifact, source = setup.config, setup.artifact, setup.source
    ranks = list(map(int, config.rank_grid))
    methods = ["rrr"]
    if source.load_collection("input_factors", {"method": "pca"}):
        methods.append("pca")
    plan = artifact.resume_method_plan(methods, ranks, refresh=config.refresh)
    pending = plan["pending"]
    cached_full = (
        artifact.load_field("full_metrics")
        if not config.refresh and artifact.has_field("full_metrics")
        else None
    )
    need_data = cached_full is None or bool(pending)
    data = train = test = None
    if need_data:
        data = load_stimulus_data(
            config, setup.train_stories, setup.test_stories, setup.groups
        )
        train, test = load_subject_responses(
            data, config, config.fmri.subjects, setup.masks
        )

    if cached_full is None:
        with setup.timed("fit full-embedding ridge ceiling"):
            full_fit = fit_full_model(
                train,
                test,
                data.cv,
                config,
                context={
                    "script": "fmri_rrr",
                    "mode": "cross_dataset",
                    "method": "full",
                    "n_features": train.stimulus.shape[1],
                    "standardize": False,
                },
                need_factors=False,
            )
        full_metrics = full_metrics_from_fit(full_fit, test)
        save_full_fit(artifact, config, full_fit, full_metrics)
    else:
        full_metrics = cached_full
        print("Reusing cached full-embedding ridge metrics", flush=True)

    bases = {}
    if pending:
        requested_ranks = sorted({rank for _, rank in pending})
        if source.identity["mode"] == "full_fit":
            fitted = {
                method: load_cross_dataset_A_by_rank(
                    source,
                    method,
                    requested_ranks,
                    source_format="rank_bases",
                    orthonormalize=method == "rrr",
                )
                for method in methods
            }
        else:
            fitted = fit_whole_stimulus_ieeg_bases(
                source,
                requested_ranks,
                standardized_input_coordinates=True,
                standardized_rrr_coordinates=True,
            )
        bases = {
            method: {
                rank: fitted[method][rank]
                for rank in requested_ranks
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
    frame = make_summary(
        artifact,
        methods,
        ranks,
        full_metrics,
        computed,
        manifest=setup.target_manifest,
    )
    save_summary_and_plot(artifact, config, frame)
    metadata = setup.base_metadata(data=data)
    metadata.update(
        {
            "source_artifact": source.identity_record(),
            "source_mode": "cross_dataset",
            "pca_available": "pca" in methods,
            "pca_training_representation": "source_fold_factors",
            "ceiling": "fresh_full_embedding_ridge",
        }
    )
    artifact.save_field("metadata", metadata, overwrite=True)
    artifact.mark_complete(
        {
            "target_manifest": [],
            "full_metrics": [],
            "mean_results": [],
            "rank_metrics": [
                artifact._selector(method=method, rank=rank)
                for method in methods
                for rank in ranks
            ],
        }
    )
    return artifact


__all__ = ["run_cross_dataset"]
