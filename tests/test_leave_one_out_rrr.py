from types import SimpleNamespace

import pytest

from scripts.run_rrr import UnitSelector
from src.core.config import RRRConfig, ScriptMode
from src.utils.rrr.ieeg_rrr_mode.leave_one_out import (
    LeaveOneOutPartition,
    run_leave_one_out,
    validate_disjoint_electrodes,
)
from src.utils.rrr.ieeg_rrr_mode.leave_one_subject_out import _subject_partitions


def test_leave_one_region_out_mode_is_removed():
    with pytest.raises(ValueError):
        ScriptMode("leave_one_region_out")


def test_subject_partitions_remain_disjoint_by_subject_and_electrode():
    partitions = _subject_partitions(
        [1, 2], {1: ["A1", "SHARED_NAME"], 2: ["B1", "SHARED_NAME"]}
    )
    for partition in partitions:
        validate_disjoint_electrodes(partition)


def test_overlap_fails_before_artifact_or_data_access():
    partition = LeaveOneOutPartition(
        label="region 'overlap'",
        train_subjects=[1],
        train_electrodes={1: ["A1"]},
        held_subjects=[1],
        held_electrodes={1: ["A1"]},
        train_unit=UnitSelector(held_out_subject=1, loso_phase="train_subjects"),
        transfer_unit=UnitSelector(held_out_subject=1, loso_phase="transfer"),
        baseline_unit=UnitSelector(held_out_subject=1, loso_phase="baseline"),
    )

    class FailOnAccess:
        def __getattr__(self, name):
            raise AssertionError(f"artifact accessed before validation: {name}")

    with pytest.raises(ValueError, match=r"subject=1, electrode='A1'"):
        run_leave_one_out(
            config=RRRConfig(),
            enc=SimpleNamespace(),
            artifact=FailOnAccess(),
            significance=SimpleNamespace(),
            subject_ids=[1],
            sig_electrodes={1: ["A1"]},
            aligned_embeddings=None,
            word_df=None,
            partitions=[partition],
        )


def test_loso_fits_train_factors_and_evaluates_only_held_out_subject(monkeypatch):
    import src.utils.rrr.ieeg_rrr_mode.leave_one_out as leave_one_out

    partition = _subject_partitions([1, 2], {1: ["A1"], 2: ["B1"]})[1]
    plan = {
        "folds_to_build": {0},
        "ranks_by_fold": {0: [1]},
        "completed_pairs": set(),
    }

    class ArtifactStub:
        @staticmethod
        def _selector(unit, **extra):
            return {**unit.mapping(), **extra}

        def has_field(self, *args, **kwargs):
            return False

        def resume_plan(self, **kwargs):
            assert kwargs["unit"].loso_phase in {"transfer", "baseline"}
            return plan

        def save_target_manifest(self, **kwargs):
            assert kwargs["unit"].loso_phase in {"transfer", "baseline"}

    built_for = []

    def build_inputs(subjects, electrodes, *args, **kwargs):
        built_for.append((subjects, electrodes))
        return (
            [[0.0], [1.0]],
            [{"train_idx": [0], "test_idx": [1]}],
            [[1]],
            None,
            SimpleNamespace(),
        )

    factor_fits = []
    transfers = []
    evaluations = []
    releases = []
    monkeypatch.setattr(leave_one_out, "_build_rrr_decomposition_inputs", build_inputs)
    monkeypatch.setattr(
        leave_one_out,
        "_fit_rrr_input_factors_for_unit",
        lambda **kwargs: factor_fits.append(kwargs),
    )
    monkeypatch.setattr(
        leave_one_out,
        "_run_leave_one_out_transfer_for_unit",
        lambda **kwargs: transfers.append(kwargs),
    )
    monkeypatch.setattr(
        leave_one_out,
        "_run_rrr_for_unit",
        lambda **kwargs: evaluations.append(kwargs),
    )
    monkeypatch.setattr(
        leave_one_out,
        "_release_phase_memory",
        lambda: releases.append(None),
    )
    significance = SimpleNamespace(
        encoding_results={1: {"test_idxs": [[1]]}, 2: {"test_idxs": [[1]]}},
        detailed_significance=SimpleNamespace(),
    )

    units = run_leave_one_out(
        config=RRRConfig(rank_grid=[1]),
        enc=SimpleNamespace(),
        artifact=ArtifactStub(),
        significance=significance,
        subject_ids=[1, 2],
        sig_electrodes={1: ["A1"], 2: ["B1"]},
        aligned_embeddings=None,
        word_df=None,
        partitions=[partition],
    )

    assert built_for == [([1], {1: ["A1"]}), ([2], {2: ["B1"]})]
    assert len(factor_fits) == 1
    assert factor_fits[0]["unit"].loso_phase == "train_subjects"
    assert factor_fits[0]["folds_to_fit"] == {0}
    assert len(transfers) == 1
    assert transfers[0]["unit"].loso_phase == "transfer"
    assert len(evaluations) == 1
    assert evaluations[0]["unit"].loso_phase == "baseline"
    assert evaluations[0]["orthogonal_only"] is True
    assert len(releases) == 2
    assert [(item[0].loso_phase, item[1]) for item in units] == [
        ("transfer", 1), ("baseline", 1)
    ]
