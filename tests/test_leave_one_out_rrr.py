from types import SimpleNamespace

import pytest

from scripts.run_rrr import Config, UnitSelector
from src.core.config import RRRConfig, ScriptMode
from src.utils.rrr.ieeg_rrr_mode.leave_one_out import (
    LeaveOneOutPartition,
    run_leave_one_out,
    validate_disjoint_electrodes,
)
from src.utils.rrr.ieeg_rrr_mode.leave_one_region_out import (
    build_region_partitions,
)
from src.utils.rrr.ieeg_rrr_mode.leave_one_subject_out import _subject_partitions


def test_loro_config_requires_region_groups_and_per_region_is_removed():
    with pytest.raises(ValueError, match="requires non-empty atlas_region_groups"):
        RRRConfig(mode=ScriptMode.LEAVE_ONE_REGION_OUT)
    config = RRRConfig(
        mode=ScriptMode.LEAVE_ONE_REGION_OUT,
        atlas_region_groups={"temporal": ["L temporal"]},
    )
    assert config.mode == ScriptMode.LEAVE_ONE_REGION_OUT
    with pytest.raises(ValueError, match="per_region"):
        Config(mode="per_region").validate()


def test_region_partitions_use_other_mapped_significant_electrodes_only():
    region_map = {
        "temporal": {1: ["A1", "A2"], 2: ["B1"]},
        "frontal": {1: ["A3"], 2: ["B2", "B3"]},
    }
    significant = {
        1: ["A1", "A3", "UNMAPPED"],
        2: ["B1", "B2"],
    }

    partitions = build_region_partitions(region_map, significant)
    temporal = next(item for item in partitions
                    if item.train_unit.held_out_region == "temporal")

    assert temporal.held_electrodes == {1: ["A1"], 2: ["B1"]}
    assert temporal.train_electrodes == {1: ["A3"], 2: ["B2"]}
    assert "UNMAPPED" not in {
        electrode
        for electrodes in temporal.train_electrodes.values()
        for electrode in electrodes
    }
    assert temporal.train_unit.loro_phase == "train_regions"
    assert temporal.transfer_unit.loro_phase == "transfer"
    assert temporal.baseline_unit.loro_phase == "baseline"
    validate_disjoint_electrodes(temporal)


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
        train_unit=UnitSelector(
            held_out_region="overlap", loro_phase="train_regions"
        ),
        transfer_unit=UnitSelector(
            held_out_region="overlap", loro_phase="transfer"
        ),
        baseline_unit=UnitSelector(
            held_out_region="overlap", loro_phase="baseline"
        ),
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


def test_loro_partition_allows_baseline_without_training_electrodes():
    partitions = build_region_partitions(
        {"only": {1: ["A1"]}}, {1: ["A1"]}
    )
    assert len(partitions) == 1
    assert partitions[0].train_subjects == []
    assert partitions[0].train_electrodes == {}
    assert partitions[0].allow_baseline_without_training is True


def test_loro_without_training_runs_baseline_only(monkeypatch):
    import src.utils.rrr.ieeg_rrr_mode.leave_one_out as leave_one_out

    partition = build_region_partitions(
        {"only": {1: ["A1"]}}, {1: ["A1"]}
    )[0]
    plan = {
        "folds_to_build": {0},
        "ranks_by_fold": {0: [1]},
        "completed_pairs": set(),
    }

    class ArtifactStub:
        def resume_plan(self, **kwargs):
            assert kwargs["unit"].loro_phase == "baseline"
            return plan

        def save_target_manifest(self, **kwargs):
            assert kwargs["unit"].loro_phase == "baseline"

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

    fitted = []
    monkeypatch.setattr(leave_one_out, "_build_rrr_decomposition_inputs", build_inputs)
    monkeypatch.setattr(
        leave_one_out,
        "_run_rrr_for_unit",
        lambda **kwargs: fitted.append(kwargs["unit"]),
    )
    significance = SimpleNamespace(
        encoding_results={1: {"test_idxs": [[1]]}},
        detailed_significance=SimpleNamespace(),
    )

    units = run_leave_one_out(
        config=RRRConfig(rank_grid=[1]),
        enc=SimpleNamespace(),
        artifact=ArtifactStub(),
        significance=significance,
        subject_ids=[1],
        sig_electrodes={1: ["A1"]},
        aligned_embeddings=None,
        word_df=None,
        partitions=[partition],
    )

    assert built_for == [([1], {1: ["A1"]})]
    assert [unit.loro_phase for unit in fitted] == ["baseline"]
    assert [(unit.loro_phase, folds) for unit, folds in units] == [("baseline", 1)]
