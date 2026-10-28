"""Leave-one-subject-out fMRI RRR mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.run_rrr import UnitSelector
from src.utils.shared.fmri import target_manifest

from .common import (
    _fit_pca_bases,
    _fit_rrr_bases,
    evaluate_rank_jobs,
    fit_full_model,
    full_metrics_from_fit,
    load_stimulus_data,
    load_subject_responses,
    make_summary,
    pending_ranks,
    release_memory,
    save_full_fit,
    save_pca_factors,
    save_rrr_factors,
    save_summary_and_plot,
)


def _units(subject):
    common = {"held_out_subject": str(subject)}
    return {
        "train": UnitSelector(
            method="rrr", loso_phase="train_subjects", **common
        ),
        "transfer": UnitSelector(method="rrr", loso_phase="transfer", **common),
        "baseline": UnitSelector(method="rrr", loso_phase="baseline", **common),
        "pca": UnitSelector(method="pca", loso_phase="baseline", **common),
    }


def _subject_manifest(subject, masks):
    return target_manifest([subject], masks)


def _factor_missing(artifact, config, unit):
    return [
        rank
        for rank in map(int, config.rank_grid)
        if config.refresh
        or not artifact.has_field(
            "input_factors", selector=artifact._selector(unit, rank=rank)
        )
    ]


def _load_bases(artifact, unit, ranks):
    return {
        rank: artifact.load_field(
            "input_factors", selector=artifact._selector(unit, rank=rank)
        )
        for rank in ranks
    }


@dataclass(frozen=True)
class LOSOPartitionPlan:
    """Resume plan and Artifact selectors for one held-out subject."""

    held_out: object
    units: dict[str, UnitSelector]
    train_factor_ranks: list[int]
    transfer_ranks: list[int]
    baseline_ranks: list[int]
    baseline_factor_ranks: list[int]
    pca_ranks: list[int]
    pca_factor_ranks: list[int]
    full_missing: bool

    @property
    def needs_stimulus(self):
        return bool(
            self.train_factor_ranks
            or self.transfer_ranks
            or self.baseline_ranks
            or self.pca_ranks
            or self.full_missing
        )

    @property
    def needs_held_out_responses(self):
        return bool(
            self.transfer_ranks
            or self.baseline_ranks
            or self.pca_ranks
            or self.full_missing
        )


def _plan_partition(artifact, config, subject):
    units = _units(subject)
    return LOSOPartitionPlan(
        held_out=subject,
        units=units,
        train_factor_ranks=_factor_missing(artifact, config, units["train"]),
        transfer_ranks=pending_ranks(artifact, config, units["transfer"]),
        baseline_ranks=pending_ranks(
            artifact, config, units["baseline"], require_factors=True
        ),
        baseline_factor_ranks=_factor_missing(
            artifact, config, units["baseline"]
        ),
        pca_ranks=pending_ranks(
            artifact, config, units["pca"], require_factors=True
        ),
        pca_factor_ranks=_factor_missing(artifact, config, units["pca"]),
        full_missing=config.refresh
        or not artifact.has_field(
            "full_metrics", selector={"held_out_subject": str(subject)}
        ),
    )


def _plan_run(artifact, config):
    plans = [
        _plan_partition(artifact, config, subject)
        for subject in config.fmri.subjects
    ]
    return plans, any(plan.needs_stimulus for plan in plans)


def _empty_requirements():
    return {
        "target_manifest": [],
        "input_factors": [],
        "input_scalers": [{}],
        "full_metrics": [],
        "rank_metrics": [],
        "mean_results": [],
    }


def _training_subjects(config, held_out):
    subjects = [subject for subject in config.fmri.subjects if subject != held_out]
    if not subjects:
        raise ValueError(
            "fMRI leave_one_subject_out requires at least two configured subjects"
        )
    return subjects


def _save_target_manifests(artifact, config, units, manifest, required):
    for key in ("transfer", "baseline", "pca"):
        artifact.save_target_manifest(
            unit=units[key],
            target_manifest=manifest,
            overwrite=config.refresh,
        )
        required["target_manifest"].append(units[key].mapping())


def _register_input_factors(artifact, required, units, ranks):
    for key in ("train", "baseline", "pca"):
        required["input_factors"].extend(
            artifact._selector(units[key], rank=rank) for rank in ranks
        )


def _fit_source_bases(setup, data, plan, train_subjects):
    """Fit RRR bases on every subject except the held-out subject."""
    if not plan.train_factor_ranks:
        return None
    config, artifact = setup.config, setup.artifact
    source_train, _ = load_subject_responses(
        data, config, train_subjects, setup.masks
    )
    with setup.timed("fit non-held-out full ridge for RRR factors"):
        source_fit = fit_full_model(
            source_train,
            None,
            data.cv,
            config,
            context={
                "script": "fmri_rrr",
                "mode": "leave_one_subject_out",
                "phase": "train_subjects",
                "held_out_subject": str(plan.held_out),
                "n_features": source_train.stimulus.shape[1],
                "standardize": False,
            },
            need_factors=True,
        )
    with setup.timed("fit non-held-out fMRI RRR bases"):
        bases, factors = _fit_rrr_bases(
            source_train, source_fit, data.ambient_dim, config
        )
    save_rrr_factors(
        artifact, config, plan.units["train"], bases, factors,
    )
    del source_train, source_fit, factors
    release_memory()
    return bases


def _load_held_out_responses(setup, data, plan):
    if not plan.needs_held_out_responses:
        return None, None
    return load_subject_responses(
        data, setup.config, [plan.held_out], setup.masks
    )


def _fit_or_load_held_out_ceiling(setup, data, plan, held_train, held_test):
    """Fit once when needed, then return the shared held-out ceiling and fit."""
    config, artifact = setup.config, setup.artifact
    selector = {"held_out_subject": str(plan.held_out)}
    held_fit = None
    if plan.full_missing or plan.baseline_factor_ranks:
        with setup.timed("fit held-out full-embedding ridge ceiling"):
            held_fit = fit_full_model(
                held_train,
                held_test,
                data.cv,
                config,
                context={
                    "script": "fmri_rrr",
                    "mode": "leave_one_subject_out",
                    "phase": "held_out",
                    "held_out_subject": str(plan.held_out),
                    "n_features": held_train.stimulus.shape[1],
                    "standardize": False,
                },
                need_factors=bool(plan.baseline_factor_ranks),
            )
        fresh_metrics = full_metrics_from_fit(held_fit, held_test)
        if plan.full_missing:
            save_full_fit(
                artifact, config, held_fit, fresh_metrics, selector=selector
            )
            return fresh_metrics, held_fit
    return artifact.load_field("full_metrics", selector=selector), held_fit


def _fit_baseline_bases(setup, data, plan, held_train, held_fit):
    """Fit the within-held-out-subject RRR comparison bases when missing."""
    if not plan.baseline_factor_ranks:
        return None
    with setup.timed("fit held-out within-subject RRR bases"):
        bases, factors = _fit_rrr_bases(
            held_train, held_fit, data.ambient_dim, setup.config
        )
    save_rrr_factors(
        setup.artifact, setup.config, plan.units["baseline"], bases, factors,
    )
    del factors
    return bases


def _prepare_pca_bases(setup, data, plan, shared_pca):
    """Fit shared stimulus PCA once and save it for this subject if needed."""
    if not plan.pca_factor_ranks:
        return None, shared_pca
    if shared_pca is None:
        shared_pca = _fit_pca_bases(data.train, setup.config)
    bases, full_basis = shared_pca
    save_pca_factors(
        setup.artifact, setup.config, plan.units["pca"], bases, full_basis,
    )
    return bases, shared_pca


def _evaluate_partition_space(
    setup,
    data,
    plan,
    held_train,
    held_test,
    held_manifest,
    full_metrics,
    *,
    unit_key,
    method,
    ranks_to_run,
    bases,
    orthonormalize=False,
):
    """Evaluate one transfer/baseline space and publish its summary and plot."""
    unit = plan.units[unit_key]
    if ranks_to_run and bases is None:
        bases = _load_bases(setup.artifact, unit, ranks_to_run)
    jobs = [(method, rank, bases[rank]) for rank in ranks_to_run]
    computed = evaluate_rank_jobs(
        artifact=setup.artifact,
        config=setup.config,
        data=data,
        train=held_train,
        test=held_test,
        train_stories=setup.train_stories,
        test_stories=setup.test_stories,
        jobs=jobs,
        unit_by_method={method: unit},
        orthonormalize=orthonormalize,
    )
    frame = make_summary(
        setup.artifact,
        [method],
        list(map(int, setup.config.rank_grid)),
        full_metrics,
        computed,
        manifest=held_manifest,
        unit_by_method={method: unit},
    )
    save_summary_and_plot(setup.artifact, setup.config, frame, unit=unit)


def _register_partition_outputs(artifact, required, plan, ranks):
    required["full_metrics"].append({"held_out_subject": str(plan.held_out)})
    required["mean_results"].extend(
        plan.units[key].mapping() for key in ("transfer", "baseline", "pca")
    )
    for rank in ranks:
        required["rank_metrics"].extend(
            artifact._selector(plan.units[key], rank=rank)
            for key in ("transfer", "baseline", "pca")
        )


def _partition_metadata(setup, held_out, train_subjects):
    return {
        "held_out_subject": str(held_out),
        "training_subjects": list(map(str, train_subjects)),
        "training_target_count": int(
            sum(np.count_nonzero(setup.masks[s]) for s in train_subjects)
        ),
        "held_out_target_count": int(np.count_nonzero(setup.masks[held_out])),
    }


def _run_partition(setup, data, plan, ranks, shared_pca, required):
    """Run the source fit, held-out fits, and evaluations for one subject."""
    train_subjects = _training_subjects(setup.config, plan.held_out)
    print(f"\n{'=' * 60}\nHeld out: subject {plan.held_out}", flush=True)
    print(f"Training subjects: {train_subjects}", flush=True)

    held_manifest = _subject_manifest(plan.held_out, setup.masks)
    _save_target_manifests(
        setup.artifact, setup.config, plan.units, held_manifest, required
    )
    _register_input_factors(setup.artifact, required, plan.units, ranks)

    source_bases = _fit_source_bases(setup, data, plan, train_subjects)
    held_train, held_test = _load_held_out_responses(setup, data, plan)
    full_metrics, held_fit = _fit_or_load_held_out_ceiling(
        setup, data, plan, held_train, held_test
    )
    baseline_bases = _fit_baseline_bases(
        setup, data, plan, held_train, held_fit
    )
    pca_bases, shared_pca = _prepare_pca_bases(
        setup, data, plan, shared_pca
    )

    _evaluate_partition_space(
        setup, data, plan, held_train, held_test, held_manifest, full_metrics,
        unit_key="transfer", method="rrr", ranks_to_run=plan.transfer_ranks,
        bases=source_bases, orthonormalize=True,
    )
    _evaluate_partition_space(
        setup, data, plan, held_train, held_test, held_manifest, full_metrics,
        unit_key="baseline", method="rrr", ranks_to_run=plan.baseline_ranks,
        bases=baseline_bases, orthonormalize=True,
    )
    _evaluate_partition_space(
        setup, data, plan, held_train, held_test, held_manifest, full_metrics,
        unit_key="pca", method="pca", ranks_to_run=plan.pca_ranks,
        bases=pca_bases,
    )
    _register_partition_outputs(setup.artifact, required, plan, ranks)

    if plan.needs_held_out_responses:
        del held_train, held_test
    release_memory()
    return shared_pca, _partition_metadata(
        setup, plan.held_out, train_subjects
    )


def _finalize_run(setup, data, partition_rows, required):
    metadata = setup.base_metadata(data=data)
    metadata.update(
        {
            "source_artifact": None,
            "source_mode": "leave_one_subject_out",
            "pca_available": True,
            "pca_training_representation": (
                "shared_tr_aligned_trimmed_pooled_train_standardized_pre_fir"
            ),
            "ceiling": "one_held_out_subject_full_embedding_ridge_shared_by_phases",
            "transfer_basis": "reduced_qr_orthonormalized",
            "training_cohort_evaluated": False,
        }
    )
    setup.artifact.save_field("metadata", metadata, overwrite=True)
    setup.artifact.save_field(
        "mode_metadata",
        pd.DataFrame(partition_rows),
        selector={"kind": "leave_one_subject_out"},
        overwrite=True,
    )
    setup.artifact.mark_complete(required)
    return setup.artifact


def run_leave_one_subject_out(setup):
    """Plan LOSO partitions, run each incomplete phase, and finalize outputs."""
    config, artifact = setup.config, setup.artifact
    ranks = list(map(int, config.rank_grid))
    plans, need_stimulus = _plan_run(artifact, config)
    need_stimulus = (
        need_stimulus or config.refresh or not artifact.has_field("input_scalers")
    )
    data = (
        load_stimulus_data(config, setup.train_stories, setup.test_stories, setup.groups)
        if need_stimulus
        else None
    )
    if data is not None and (config.refresh or not artifact.has_field("input_scalers")):
        # All LOSO factors use the same stimulus rows; only neural targets vary by
        # held-out subject, so one ambient embedding scaler serves every phase/rank.
        artifact.save_field(
            "input_scalers", data.input_scaler, overwrite=config.refresh
        )
    shared_pca = None
    partition_rows = []
    required = _empty_requirements()
    for plan in plans:
        shared_pca, row = _run_partition(
            setup, data, plan, ranks, shared_pca, required
        )
        partition_rows.append(row)
    return _finalize_run(setup, data, partition_rows, required)


__all__ = ["run_leave_one_subject_out"]
