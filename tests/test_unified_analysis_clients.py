import numpy as np
import pandas as pd
import pytest

from scripts.run_rrr import Artifact as RRRArtifact
from scripts.run_rrr import Config as RRRClientConfig
from scripts.run_rrr import UnitSelector
from scripts.run_significance import Artifact as SignificanceArtifact
from scripts.run_significance import Config as SignificanceClientConfig
from src.core.config import FMRICrossDatasetRRRConfig, FMRISignificanceConfig
from src.core.config import EncodingConfig, RRRConfig, SignificanceConfig
from src.core.registry import DatasetContext
from src.utils.shared.config import dict_to_dataclass


def test_significance_config_dispatches_dataset_and_rejects_legacy_shape():
    ieeg = SignificanceClientConfig(dataset="brain_treebank", ieeg={
        "encoding": {"per_dataset_config": {"movie": "cars-2"}}
    })
    assert isinstance(ieeg.selected_config(), SignificanceConfig)
    assert ieeg.selected_config().encoding.dataset == "brain_treebank"

    fmri = SignificanceClientConfig(
        dataset="deep-fmri", mode="held_out_session", ieeg=None,
        fmri={"data": {"subjects": ["UTS01"]}}, minimal_shift=10,
    )
    assert isinstance(fmri.selected_config(), FMRISignificanceConfig)
    with pytest.raises(ValueError, match="Unknown Config fields"):
        dict_to_dataclass({"encoding": {}}, SignificanceClientConfig)


def test_brain_treebank_laplacian_separates_downstream_artifact_variants(tmp_path):
    context = DatasetContext(
        dataset="brain_treebank",
        stimulus_id="cars-2",
        stimulus_label="cars-2",
        rereference="laplacian",
    )
    encoding = EncodingConfig(
        dataset="brain_treebank",
        per_dataset_config={"movie": "cars-2", "rereference": "laplacian"},
    )

    significance = SignificanceArtifact.from_ieeg_config(
        SignificanceConfig(encoding=encoding, results_root=str(tmp_path)), context
    )
    rrr = RRRArtifact.from_ieeg_config(
        RRRConfig(encoding=encoding, results_root=str(tmp_path)), context
    )

    assert significance.variant == "rereference-laplacian"
    assert rrr.variant_extras["rereference"] == "laplacian"


def test_rrr_internal_mode_and_generic_source_translation():
    ieeg = RRRClientConfig(
        significance_selection={
            "mode": "within-layer", "reference_artifact": {"dataset": "podcast"}
        }, ieeg={"encoding": {}},
    ).selected_config()
    assert isinstance(ieeg, RRRConfig)
    assert ieeg.mode.value == "cross_subject"
    assert ieeg.subspace_aggregation is None

    within_fmri = RRRClientConfig(
        dataset="deep-fmri", mode="internal", ieeg=None,
        significance_selection={
            "mode": "within-layer",
            "reference_artifact": {"dataset": "deep-fmri"},
        },
        fmri={"data": {"subjects": ["UTS01"]}},
    ).selected_config()
    assert within_fmri.subspace_aggregation == "prediction_weighted"

    fmri = RRRClientConfig(
        dataset="deep-fmri", mode="cross_dataset", ieeg=None,
        significance_selection={
            "mode": "within-layer",
            "reference_artifact": {"dataset": "deep-fmri"},
        },
        rrr_source={"dataset": "podcast", "mode": "internal"},
        fmri={"data": {"subjects": ["UTS01"]}},
    ).selected_config()
    assert isinstance(fmri, FMRICrossDatasetRRRConfig)
    assert fmri.source_mode == "cross_dataset"
    assert fmri.ieeg_rrr_source["dataset"] == "podcast"
    assert fmri.subspace_aggregation is None
    artifact = RRRArtifact.from_fmri_config(fmri)
    assert artifact.variant_extras == {
        "input_scaling": "pooled-training-stories-v1",
        "transfer_scaling": "source-then-target-reduced-pre-fir-v1",
    }


def test_rrr_fmri_leave_one_subject_out_translation():
    selected = RRRClientConfig(
        dataset="deep-fmri",
        mode="leave_one_subject_out",
        ieeg=None,
        significance_selection={
            "mode": "within-layer",
            "reference_artifact": {"dataset": "deep-fmri"},
        },
        fmri={"data": {"subjects": ["UTS01", "UTS02"]}},
    ).selected_config()

    assert isinstance(selected, FMRICrossDatasetRRRConfig)
    assert selected.source_mode == "leave_one_subject_out"
    assert selected.subspace_aggregation == "prediction_weighted"

    artifact = RRRArtifact.from_fmri_config(selected)
    assert artifact.identity["mode"] == "leave_one_subject_out"
    assert artifact.variant_extras == {
        "input_scaling": "pooled-training-stories-v1",
        "loso_contract": "orthogonal-held-out-fmri-v1"
    }


def test_rrr_fmri_rejects_removed_rank_design_prefetch_option():
    config = RRRClientConfig(
        dataset="deep-fmri",
        mode="internal",
        ieeg=None,
        fmri={
            "data": {"subjects": ["UTS01"]},
            "prefetch_rank_designs": True,
        },
    )

    with pytest.raises(ValueError, match="prefetch_rank_designs"):
        config.selected_config()


def test_rrr_generic_source_preserves_rereference_selector():
    selected = RRRClientConfig(
        mode="cross_dataset",
        significance_selection={
            "mode": "within-layer",
            "reference_artifact": {"dataset": "podcast"},
        },
        rrr_source={
            "dataset": "brain-treebank-cars-2",
            "mode": "internal",
            "rereference": "laplacian",
        },
        ieeg={"encoding": {}},
    ).selected_config()

    assert selected.ieeg_rrr_source["rereference"] == "laplacian"


def test_rrr_fmri_source_defaults_to_prediction_weighted():
    selected = RRRClientConfig(
        mode="cross_dataset",
        significance_selection={
            "mode": "within-layer",
            "reference_artifact": {"dataset": "podcast"},
        },
        rrr_source={"dataset": "deep-fmri", "mode": "internal"},
        ieeg={"encoding": {}},
    ).selected_config()

    assert selected.subspace_aggregation is None
    assert selected.fmri_rrr_source["subspace_aggregation"] == (
        "prediction_weighted"
    )


def test_rrr_cross_temporal_mode_translates_generic_source():
    selected = RRRClientConfig(
        mode="cross_temporal_windows",
        rrr_source={"dataset": "podcast", "mode": "temporal_windows"},
        significance_selection={
            "mode": "within-layer",
            "reference_artifact": {"dataset": "podcast"},
        },
        ieeg={
            "encoding": {},
            "temporal_windows": [
                {"name": "pre", "start_sec": -2.0, "end_sec": 0.0},
                {"name": "post", "start_sec": 0.0, "end_sec": 2.0},
            ],
        },
    ).selected_config()

    assert selected.mode.value == "cross_temporal_windows"
    assert selected.ieeg_rrr_source["mode"] == "temporal_windows"


def test_significance_targets_and_rrr_factor_sets_are_modality_neutral(tmp_path):
    significance = SignificanceArtifact(results_root=tmp_path).begin(config={"x": 1})
    frame = pd.DataFrame({
        "subject": ["1", "1"], "target_id": ["A", "B"],
        "target_kind": ["electrode", "electrode"],
        "significant": [True, False],
    })
    significance.save_subject_field("target_significance", 1, frame)
    assert significance.load_targets([1])["1"].equals(frame)

    rrr = RRRArtifact(results_root=tmp_path).begin(config={"x": 2})
    unit = UnitSelector(method="rrr")
    rrr.save_factors(unit=unit, fold=0, input_factors=np.eye(4, 3))
    rrr.save_factors(unit=unit, fold=1, input_factors=np.ones((4, 3)))
    factors = rrr.load_input_factors("rrr", rank=2)
    assert len(factors) == 2
    assert all(value.shape == (4, 2) for value in factors)
