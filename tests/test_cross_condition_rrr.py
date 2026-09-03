"""Tests for the standalone cross-condition orthogonal RRR workflow."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.run_cross_condition_rrr import (
    Artifact, Config, _artifact_config, _metric_payload, run_config,
    run_item_comparisons,
)
from src.utils.rrr.cross_condition import (
    ComparisonGroup,
    ComparisonItem,
    validate_comparison_items,
)
from src.utils.rrr.cross_condition_items.stimulus import (
    discover_subject_movies,
    get_stimulus_items,
    ordered_electrode_intersection,
)
from src.utils.rrr.cross_condition_items.region import get_region_items
from src.utils.rrr.core import canonical_qr, fit_full_rrr, orthogonal_rank_factors
from src.utils.atlas.core import REGION_GROUPS
from src.utils.shared.encoding import contiguous_folds


def test_cross_condition_client_is_a_distinct_artifact(tmp_path):
    config = Config(results_root=str(tmp_path))
    artifact = Artifact.from_config(config)

    assert artifact.SCRIPT == "run_cross_condition_rrr"
    assert artifact.identity["mode"] == "cross_condition"
    assert artifact.identity["dataset"] == "brain-treebank"
    assert "significance" not in artifact.FIELD_CONTRACT


def test_config_locks_initial_study_controls():
    Config().validate()
    with pytest.raises(ValueError, match="laplacian"):
        Config(rereference="common_average").validate()
    with pytest.raises(ValueError, match="five target folds"):
        Config(num_folds=2).validate()
    with pytest.raises(ValueError, match="duplicate"):
        Config(rank_grid=[1, 1]).validate()


def test_region_config_uses_podcast_controls_instead_of_stimulus_controls():
    config = Config(
        dataset="podcast",
        item_getter="region",
        rereference="common_average",
        num_folds=2,
        atlas_region_groups={"temporal": ["L temporal"]},
    )
    config.validate()
    artifact = Artifact.from_config(config)

    assert artifact.identity["dataset"] == "podcast"
    default = Config(dataset="podcast", item_getter="region")
    default.validate()
    assert default.atlas_region_groups == REGION_GROUPS
    with pytest.raises(ValueError, match="requires dataset='podcast'"):
        Config(item_getter="region", atlas_region_groups={"x": ["L x"]}).validate()


def test_cross_condition_alpha_metrics_follow_log_alphas():
    result = {
        "corr": 0.2, "r2": 0.1,
        "corr_per_target": np.array([0.2]),
        "r2_per_target": np.array([0.1]),
        "best_alphas": np.array([10.0]),
    }
    assert "best_alphas" not in _metric_payload(result, log_alphas=False)
    assert "best_alphas" in _metric_payload(result, log_alphas=True)


def test_discovers_only_subjects_with_multiple_movies(tmp_path):
    root = tmp_path / "subject_metadata"
    root.mkdir()
    for subject, trial, movie in ((1, 0, "a"), (1, 1, "b"), (2, 0, "c")):
        (root / f"sub_{subject}_trial{trial:03d}_metadata.json").write_text(
            json.dumps({"filename": movie, "title": movie})
        )

    assert discover_subject_movies(str(tmp_path)) == {1: ["a", "b"]}


def test_electrode_intersection_preserves_first_movie_order():
    headers = {
        "a": {"electrodes": ["A2", "A1", "B1"]},
        "b": {"electrodes": ["B1", "A2", "C1"]},
        "c": {"electrodes": ["A2", "B1"]},
    }
    assert ordered_electrode_intersection(headers) == ["A2", "B1"]


def test_five_folds_are_contiguous_and_cover_rows_once():
    folds = contiguous_folds(23, 5)
    tests = [test for _, test in folds]
    assert np.array_equal(np.concatenate(tests), np.arange(23))
    assert all(np.all(np.diff(test) == 1) for test in tests)
    for train, test in folds:
        assert not set(train).intersection(test)


def test_canonical_qr_is_orthonormal_deterministic_and_mapping_preserving():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(12, 4))
    B = rng.normal(size=(9, 4))
    Q, R, adjusted = canonical_qr(A, B)
    Q_again, R_again, adjusted_again = canonical_qr(A, B)

    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-6)
    assert np.all(np.diag(R) >= 0)
    assert np.allclose(A @ B.T, Q @ adjusted.T, atol=1e-5)
    assert np.array_equal(Q, Q_again)
    assert np.array_equal(R, R_again)
    assert np.array_equal(adjusted, adjusted_again)


def test_fitted_orthogonal_frozen_mapping_matches_reduced_ridge_prediction():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 6)).astype(np.float32)
    Y = (X @ rng.normal(size=(6, 10)) + rng.normal(scale=.1, size=(60, 10))).astype(
        np.float32
    )
    fitted = fit_full_rrr(
        X, Y, 3, alphas=[1.0], ridge_cv_folds=2, alpha_policy={}
    )

    for rank in (1, 2, 3):
        factors = orthogonal_rank_factors(
            fitted["A_raw"], fitted["B_raw"], rank
        )
        assert np.allclose(
            factors["Q"].T @ factors["Q"], np.eye(rank), atol=1e-5
        )
        B = factors["B_raw"]
        B_orth = factors["B_orth"]
        intercept = fitted["ridge_intercept"] @ B @ B.T
        orth_prediction = X @ factors["Q"] @ B_orth.T + intercept
        reference = fitted["full_prediction"] @ B @ B.T
        assert np.allclose(orth_prediction, reference, atol=2e-4)


def test_rank_factors_truncate_a_before_calling_qr(monkeypatch):
    import src.utils.rrr.core as module

    calls = []
    original = module.canonical_qr

    def recording_qr(A, B):
        calls.append((A.shape, B.shape))
        return original(A, B)

    monkeypatch.setattr(module, "canonical_qr", recording_qr)
    A = np.arange(30, dtype=float).reshape(6, 5) + np.eye(6, 5)
    B = np.arange(35, dtype=float).reshape(7, 5) + np.eye(7, 5)
    factors = module.orthogonal_rank_factors(A, B, 2)

    assert calls == [((6, 2), (7, 2))]
    assert factors["Q"].shape == (6, 2)
    assert factors["R"].shape == (2, 2)


def test_source_factor_contract_saves_one_full_a_without_rank_selector(tmp_path):
    config = Config(results_root=str(tmp_path))
    artifact = Artifact.from_config(config).begin(config=_artifact_config(config))
    selector = {"subject": 2, "source_item": "a", "method": "rrr"}
    A = np.arange(20, dtype=np.float32).reshape(4, 5)
    artifact.save_field("source_factors", A, selector=selector)
    saved = artifact.load_field("source_factors", selector=selector)

    assert np.array_equal(saved, A)
    entries = artifact.manifest()["entries"]["source_factors"]
    assert len(entries) == 1
    assert entries[next(iter(entries))]["selector"] == selector


def test_completed_artifact_resumes_without_discovering_data(tmp_path, monkeypatch):
    config = Config(results_root=str(tmp_path))
    artifact = Artifact.from_config(config).begin(config=_artifact_config(config))
    artifact.save_field("effective_config", {"complete": True})
    artifact.save_field("eligibility", [])
    artifact.save_field("exclusions", [])
    artifact.save_field("fold_results", [])
    artifact.save_field("summary", [])
    artifact.mark_complete()
    monkeypatch.setattr(
        "scripts.run_cross_condition_rrr.discover_subject_movies",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete resume should not discover data")
        ),
    )

    assert run_config(config)[0].complete


def _comparison_items(*, matching_targets):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(20, 3)).astype(np.float32)
    first_target_set = "shared-targets"
    second_target_set = first_target_set if matching_targets else "other-targets"
    return [
        ComparisonItem(
            "first", X, (X @ rng.normal(size=(3, 2))).astype(np.float32),
            target_set=first_target_set,
        ),
        ComparisonItem(
            "second", X.copy(),
            (X @ rng.normal(size=(3, 2))).astype(np.float32),
            target_set=second_target_set,
        ),
    ]


def test_item_validation_requires_common_input_width_and_valid_target_set():
    items = _comparison_items(matching_targets=True)
    validate_comparison_items(items, [1], 5)

    items[1].X = np.ones((20, 4))
    with pytest.raises(ValueError, match="same X feature width"):
        validate_comparison_items(items, [1], 5)

    items = _comparison_items(matching_targets=True)
    items[1].target_set = ""
    with pytest.raises(ValueError, match="target_set"):
        validate_comparison_items(items, [1], 5)


def test_stimulus_provider_builds_named_items_with_subject_target_set(monkeypatch):
    monkeypatch.setattr(
        "src.utils.rrr.cross_condition_items.stimulus.cache_header",
        lambda *args: {"electrodes": ["E1", "E2"]},
    )

    def load_stimulus_data(**kwargs):
        return SimpleNamespace(
            X=np.ones((10, 3), dtype=np.float32),
            Y=np.ones((10, 4), dtype=np.float32),
            lags=np.array([-0.5, 0.0]),
        )

    monkeypatch.setattr(
        "src.utils.rrr.cross_condition_items.stimulus.load_stimulus_data",
        load_stimulus_data,
    )

    group = get_stimulus_items(Config(rank_grid=[1]), 7, ["movie-a", "movie-b"])

    assert group.context == {"subject": 7}
    assert [item.name for item in group.items] == ["movie-a", "movie-b"]
    assert {item.target_set for item in group.items} == {"subject-7-targets"}
    assert set(group.target_manifests) == {"movie-a", "movie-b"}


def test_region_provider_pools_all_mapped_electrodes_and_marks_distinct_targets(
    monkeypatch,
):
    config = Config(
        dataset="podcast",
        item_getter="region",
        data_root="data",
        subjects=[1, 2],
        rank_grid=[1],
        limit_to_glove=False,
        atlas_region_groups={
            "temporal": ["L temporal"],
            "frontal": ["L frontal"],
        },
    )
    region_map = {
        "temporal": {1: ["A1", "A2"], 2: ["B1"]},
        "frontal": {1: ["A3"], 2: ["B2", "B3"]},
    }
    monkeypatch.setattr(
        "src.utils.rrr.cross_condition_items.region.build_electrode_region_map",
        lambda *args: region_map,
    )
    monkeypatch.setattr(
        "src.utils.rrr.cross_condition_items.region.load_embeddings",
        lambda *args, **kwargs: (np.ones((12, 3), dtype=np.float32), None),
    )
    selections = []

    def load_data(subjects, _root, _offset, embeddings, electrodes, **kwargs):
        selections.append(electrodes)
        groups = [electrodes[subject] for subject in subjects]
        width = sum(len(group) for group in groups) * 2
        return (
            embeddings[:10],
            np.ones((10, width), dtype=np.float32),
            np.arange(10),
            groups,
        )

    monkeypatch.setattr(
        "src.utils.rrr.cross_condition_items.region.load_data_from_cache", load_data
    )
    monkeypatch.setattr(
        "src.utils.rrr.cross_condition_items.region.load_precomputed_epochs",
        lambda *args, **kwargs: {"tmin": -1.0, "tmax": 1.0, "n_lags": 2},
    )

    group = get_region_items(config)

    assert [item.name for item in group.items] == ["temporal", "frontal"]
    assert [item.target_set for item in group.items] == [
        "region-temporal-targets", "region-frontal-targets",
    ]
    assert selections == [region_map["temporal"], region_map["frontal"]]
    assert all(np.array_equal(group.items[0].X, item.X) for item in group.items)
    assert len(group.target_manifests["temporal"]) == 6
    assert group.eligibility[0]["n_electrodes"] == 3

    monkeypatch.setattr(
        "src.utils.rrr.cross_condition_items.region.load_data_from_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken cache")),
    )
    with pytest.raises(RuntimeError, match="broken cache"):
        get_region_items(config)


@pytest.mark.parametrize(
    ("matching_targets", "expected_full", "expected_diagnostics"),
    [(True, 10, 0), (False, 0, 2)],
)
def test_item_provider_runs_all_directed_basis_pairs_and_safe_full_maps(
    tmp_path, matching_targets, expected_full, expected_diagnostics,
):
    config = Config(
        results_root=str(tmp_path), rank_grid=[1], ridge_alphas=[1.0],
        ridge_cv_folds=2,
    )
    artifact = Artifact.from_config(config).begin(config=_artifact_config(config))
    calls = []

    def provider():
        calls.append(True)
        return ComparisonGroup(
            _comparison_items(matching_targets=matching_targets),
            context={"subject": 4},
        )

    rows, _, _, diagnostics = run_item_comparisons(config, artifact, provider)
    frame = pd.DataFrame(rows)

    assert calls == [True]
    assert len(frame[frame["method"] == "transfer_rrr_basis"]) == 10
    assert len(frame[frame["method"] == "transfer_pca_basis"]) == 10
    assert len(frame[frame["method"] == "within_rrr"]) == 10
    assert len(frame[frame["method"] == "within_pca"]) == 10
    assert len(frame[frame["method"] == "transfer_rrr_full"]) == expected_full
    assert len(diagnostics) == expected_diagnostics
    assert {"source_item", "target_item"}.issubset(frame.columns)
    assert "source_stimulus" not in frame.columns
