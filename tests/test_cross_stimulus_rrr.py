"""Tests for the standalone cross-stimulus orthogonal RRR workflow."""

import json

import numpy as np
import pytest

from scripts.run_cross_stimulus_rrr import (
    Artifact, Config, _metric_payload, run_config,
)
from src.utils.rrr.cross_stimulus import (
    discover_subject_movies,
    ordered_electrode_intersection,
)
from src.utils.rrr.core import canonical_qr, fit_full_rrr, orthogonal_rank_factors
from src.utils.shared.encoding import contiguous_folds


def test_cross_stimulus_client_is_a_distinct_artifact(tmp_path):
    config = Config(results_root=str(tmp_path))
    artifact = Artifact.from_config(config)

    assert artifact.SCRIPT == "run_cross_stimulus_rrr"
    assert artifact.identity["mode"] == "cross_stimulus"
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


def test_cross_stimulus_alpha_metrics_follow_log_alphas():
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
    artifact = Artifact.from_config(config).begin(config=config)
    selector = {"subject": 2, "source_stimulus": "a", "method": "rrr"}
    A = np.arange(20, dtype=np.float32).reshape(4, 5)
    artifact.save_field("source_factors", A, selector=selector)
    saved = artifact.load_field("source_factors", selector=selector)

    assert np.array_equal(saved, A)
    entries = artifact.manifest()["entries"]["source_factors"]
    assert len(entries) == 1
    assert entries[next(iter(entries))]["selector"] == selector


def test_completed_artifact_resumes_without_discovering_data(tmp_path, monkeypatch):
    config = Config(results_root=str(tmp_path))
    artifact = Artifact.from_config(config).begin(config=config)
    artifact.save_field("effective_config", {"complete": True})
    artifact.save_field("eligibility", [])
    artifact.save_field("exclusions", [])
    artifact.save_field("fold_results", [])
    artifact.save_field("summary", [])
    artifact.mark_complete()
    monkeypatch.setattr(
        "scripts.run_cross_stimulus_rrr.discover_subject_movies",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete resume should not discover data")
        ),
    )

    assert run_config(config)[0].complete
