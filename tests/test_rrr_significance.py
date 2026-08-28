from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.config import SignificanceSelectionConfig
from src.utils.rrr.significance import (
    expand_significance_artifact_specs,
    union_significance_artifacts,
    validate_ieeg_encoding_alignment,
)


def _artifact(targets, details=None):
    return SimpleNamespace(
        target_significance=targets,
        significance_details=details or {},
    )


def test_expands_one_artifact_per_configured_layer_in_order():
    specs = [
        {"model": "first", "layer": [0, 2]},
        {"model": "second", "layer": 4},
    ]

    assert expand_significance_artifact_specs(specs) == [
        {"model": "first", "layer": 0},
        {"model": "first", "layer": 2},
        {"model": "second", "layer": 4},
    ]


def test_expansion_rejects_empty_layers_and_duplicate_identities():
    with pytest.raises(ValueError, match="cannot be empty"):
        expand_significance_artifact_specs([{"model": "model", "layer": []}])
    with pytest.raises(ValueError, match="Duplicate"):
        expand_significance_artifact_specs([
            {"model": "model", "layer": [0, 1]},
            {"model": "model", "layer": 1},
        ])


def test_union_ors_disjoint_targets_and_ieeg_lags():
    first = _artifact(
        {"1": pd.DataFrame({
            "subject": ["1", "1"], "target_id": ["A", "B"],
            "target_kind": ["electrode", "electrode"],
            "significant": [True, False],
        })},
        {"1": pd.DataFrame({
            "subject": [1, 1], "electrode": ["A", "B"],
            "lag_idx": [0, 0], "sig": [True, False],
        })},
    )
    second = _artifact(
        {"1": pd.DataFrame({
            "subject": ["1", "1"], "target_id": ["B", "C"],
            "target_kind": ["electrode", "electrode"],
            "significant": [True, True],
        })},
        {"1": pd.DataFrame({
            "subject": [1, 1], "electrode": ["B", "C"],
            "lag_idx": [0, 1], "sig": [True, True],
        })},
    )

    targets, details = union_significance_artifacts(
        [first, second], modality="ieeg"
    )

    assert dict(zip(targets["1"].target_id, targets["1"].significant)) == {
        "A": True, "B": True, "C": True,
    }
    assert {
        (row.electrode, row.lag_idx): row.sig
        for row in details["1"].itertuples()
    } == {("A", 0): True, ("B", 0): True, ("C", 1): True}


def test_selection_config_limits_layer_lists_to_cross_sources():
    with pytest.raises(ValueError, match="reference_artifact.layer"):
        SignificanceSelectionConfig(
            reference_artifact={"layer": [0, 1]}
        )
    with pytest.raises(ValueError, match="requires artifacts"):
        SignificanceSelectionConfig(
            mode="cross-model-layer", reference_artifact={"layer": 0}
        )


def _encoding_artifact(model, selection, folds, word_selection=None):
    artifact = SimpleNamespace(
        encoding_results={
            "1": {
                "selection": np.asarray(selection),
                "test_idxs": [np.asarray(fold) for fold in folds],
                "word_selection": word_selection,
            }
        }
    )
    artifact.identity_record = lambda: {"model": model, "layer": 0}
    return artifact


def test_ieeg_union_sources_must_share_reference_rows_and_folds():
    words = {"mode": "shared_intersection", "epoch_selection": np.array([1, 3])}
    reference = _encoding_artifact("reference", [1, 3], [[0], [1]], words)
    matching = _encoding_artifact("matching", [1, 3], [[0], [1]], words)

    validate_ieeg_encoding_alignment(reference, [matching], [1])

    different_rows = _encoding_artifact("rows", [1, 4], [[0], [1]], words)
    with pytest.raises(ValueError, match="different selection"):
        validate_ieeg_encoding_alignment(reference, [different_rows], [1])

    different_folds = _encoding_artifact("folds", [1, 3], [[1], [0]], words)
    with pytest.raises(ValueError, match="different test_idxs"):
        validate_ieeg_encoding_alignment(reference, [different_folds], [1])

    different_words = _encoding_artifact(
        "words", [1, 3], [[0], [1]],
        {"mode": "shared_intersection", "epoch_selection": np.array([1, 4])},
    )
    with pytest.raises(ValueError, match="different word_selection"):
        validate_ieeg_encoding_alignment(reference, [different_words], [1])
