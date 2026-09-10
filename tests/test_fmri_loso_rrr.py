from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from scripts.run_rrr import Artifact
from src.core.config import FMRIDataConfig, FMRICrossDatasetRRRConfig
from src.utils.rrr.fmri_rrr_mode import leave_one_subject_out as loso
from src.utils.rrr.fmri_rrr_mode.common import (
    FMRIRunSetup,
    FMRIStimulusData,
    FinalRidgeFit,
)
from src.utils.shared.fmri import AlignedFMRIStimuli, PreparedFMRI


def test_fmri_loso_partitions_targets_writes_phases_and_resumes(tmp_path, monkeypatch):
    config = FMRICrossDatasetRRRConfig(
        fmri=FMRIDataConfig(
            subjects=["UTS01", "UTS02"], sessions=[1], ndelays=1
        ),
        source_mode="leave_one_subject_out",
        rank_grid=[1],
        results_root=str(tmp_path),
        plot=False,
    )
    artifact = Artifact.from_fmri_config(config)
    artifact.begin(config=config, variant_extras=artifact.variant_extras)
    artifact.save_field("effective_config", {})
    artifact.save_field("upstream_identities", [])
    masks = {
        "UTS01": np.array([True, False]),
        "UTS02": np.array([True, True, False]),
    }
    setup = FMRIRunSetup(
        config=config,
        artifact=artifact,
        significance=SimpleNamespace(),
        upstreams=[],
        source=None,
        train_stories=["train"],
        test_stories=["test"],
        groups={1: ["train"]},
        masks=masks,
        target_manifest=SimpleNamespace(),
        original_target_count=3,
    )
    setup.timed = lambda label: nullcontext()
    train_design = np.arange(12, dtype=np.float32).reshape(4, 3)
    test_design = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = FMRIStimulusData(
        aligned=AlignedFMRIStimuli(stories={}),
        train=PreparedFMRI(
            stimulus=train_design,
            pre_fir_stimulus=train_design,
            responses=None,
            story_slices={"train": slice(0, 4)},
        ),
        test=PreparedFMRI(
            stimulus=test_design,
            pre_fir_stimulus=test_design,
            responses=None,
            story_slices={"test": slice(0, 2)},
        ),
        cv=[],
        ambient_dim=3,
    )
    monkeypatch.setattr(loso, "load_stimulus_data", lambda *args, **kwargs: data)

    loaded_subject_groups = []

    def fake_load_responses(data_arg, config_arg, subjects, masks_arg):
        loaded_subject_groups.append(tuple(subjects))
        n_targets = sum(np.count_nonzero(masks_arg[subject]) for subject in subjects)
        return (
            PreparedFMRI(
                stimulus=train_design,
                pre_fir_stimulus=train_design,
                responses=np.ones((4, n_targets), dtype=np.float32),
                story_slices={"train": slice(0, 4)},
            ),
            PreparedFMRI(
                stimulus=test_design,
                pre_fir_stimulus=test_design,
                responses=np.ones((2, n_targets), dtype=np.float32),
                story_slices={"test": slice(0, 2)},
            ),
        )

    monkeypatch.setattr(loso, "load_subject_responses", fake_load_responses)
    fit_calls = []

    def fake_fit_full(train, test, cv, config, *, context, need_factors):
        fit_calls.append((context["phase"], context["held_out_subject"], train.responses.shape[1]))
        return FinalRidgeFit(
            test_prediction=(
                np.zeros((2, train.responses.shape[1]), dtype=np.float32)
                if test is not None
                else None
            ),
            train_prediction=np.zeros_like(train.responses) if need_factors else None,
            coefficients=(
                np.zeros((3, train.responses.shape[1]), dtype=np.float32)
                if need_factors
                else None
            ),
            best_alphas=np.ones(train.responses.shape[1]),
            alpha_grid=np.ones(1),
        )

    monkeypatch.setattr(loso, "fit_full_model", fake_fit_full)
    monkeypatch.setattr(
        loso,
        "full_metrics_from_fit",
        lambda fit, test: {
            "corr_per_target": np.zeros(test.responses.shape[1]),
            "r2_per_target": np.zeros(test.responses.shape[1]),
        },
    )

    factor_target_counts = []

    def fake_rrr_bases(train, fit, ambient_dim, config):
        factor_target_counts.append(train.responses.shape[1])
        return {1: np.eye(3, 1, dtype=np.float32)}, {
            "full_fir": np.eye(3, 1, dtype=np.float32),
            "target": None,
            "delay_blocks": [np.eye(3, 1, dtype=np.float32)],
            "prediction_singular_values": np.ones(1, dtype=np.float32),
        }

    monkeypatch.setattr(loso, "_fit_rrr_bases", fake_rrr_bases)
    pca_fits = []

    def fake_pca_bases(train, config):
        pca_fits.append(True)
        full = np.eye(3, dtype=np.float32)
        return {1: full[:, :1]}, full

    monkeypatch.setattr(loso, "_fit_pca_bases", fake_pca_bases)
    evaluations = []

    def fake_evaluate(
        *, artifact, jobs, test, unit_by_method=None, orthonormalize=False, **kwargs
    ):
        computed = {}
        for method, rank, _ in jobs:
            unit = unit_by_method[method]
            evaluations.append((unit.loso_phase, method, orthonormalize))
            corr = np.zeros(test.responses.shape[1], dtype=np.float32)
            r2 = np.zeros(test.responses.shape[1], dtype=np.float32)
            artifact.save_rank_metrics(
                unit=unit,
                rank=rank,
                metrics={"corr_per_target": corr, "r2_per_target": r2},
            )
            computed[(method, rank)] = (corr, r2)
        return computed

    monkeypatch.setattr(loso, "evaluate_rank_jobs", fake_evaluate)
    monkeypatch.setattr(loso, "release_memory", lambda: None)

    result = loso.run_leave_one_subject_out(setup)

    assert result.manifest()["complete"] is True
    assert loaded_subject_groups == [
        ("UTS02",),
        ("UTS01",),
        ("UTS01",),
        ("UTS02",),
    ]
    assert fit_calls == [
        ("train_subjects", "UTS01", 2),
        ("held_out", "UTS01", 1),
        ("train_subjects", "UTS02", 1),
        ("held_out", "UTS02", 2),
    ]
    assert factor_target_counts == [2, 1, 1, 2]
    assert pca_fits == [True]
    assert evaluations == [
        ("transfer", "rrr", True),
        ("baseline", "rrr", True),
        ("baseline", "pca", False),
        ("transfer", "rrr", True),
        ("baseline", "rrr", True),
        ("baseline", "pca", False),
    ]
    assert len(result.load_collection("full_metrics")) == 2
    assert not result.has_field(
        "full_metrics",
        selector={
            "method": "rrr",
            "held_out_subject": "UTS01",
            "loso_phase": "train_subjects",
        },
    )

    monkeypatch.setattr(
        loso,
        "load_stimulus_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete LOSO resume must not load stimuli")
        ),
    )
    loso.run_leave_one_subject_out(setup)
