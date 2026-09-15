import numpy as np
import pandas as pd
import pytest

from scripts.run_language_reconstruction import Artifact as ReconstructionArtifact
from scripts.run_rrr import Artifact as RRRArtifact
from src.core.config import LanguageDecodingMode, LanguageReconstructionConfig
from src.utils.language_interpret.reconstruction import (
    _load_artifact_factors,
    _run_fit,
    build_predictor_preprocessor,
    build_preprocessor,
    filter_selection_and_folds,
    orthonormal_factor,
    prepare_target_factors,
    reconstruction_metrics,
    run_language_reconstruction,
    summarize_comparisons,
    summarize_predictor_contrasts,
    summarize_results,
    weighted_mean_factor,
)


def _config(**overrides):
    values = {
        "rrr_source": {"mode": "internal", "dataset": "podcast"},
        "rank_grid": [1, 2],
        "tasks": ["surprisal", "part_of_speech"],
        "ridge_alphas": [0.1, 1.0],
    }
    values.update(overrides)
    return LanguageReconstructionConfig(**values)


def test_reconstruction_config_validates_choices():
    config = _config()
    assert config.target_views == ["weighted", "span"]
    assert config.predictor_sets == ["tasks"]
    assert config.glove_dim == 300
    with pytest.raises(ValueError, match="target_views"):
        _config(target_views=["raw"])
    with pytest.raises(ValueError, match="duplicate ranks"):
        _config(rank_grid=[2, 2])
    with pytest.raises(ValueError, match="predictor_sets"):
        _config(predictor_sets=["unknown"])
    with pytest.raises(ValueError, match="glove_dim"):
        _config(glove_dim=25)
    assert _config(predictor_sets=["glove"], tasks=[]).tasks == []


def test_orthonormal_factor_preserves_span_and_weighted_factor_keeps_spectrum():
    rng = np.random.default_rng(0)
    factor = rng.normal(size=(8, 3)) @ np.diag([4.0, 2.0, 0.5])
    basis = orthonormal_factor(factor, 3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-6)
    projected = basis @ basis.T @ factor
    np.testing.assert_allclose(projected, factor, atol=1e-6)

    weighted = weighted_mean_factor([factor, factor], 3)
    np.testing.assert_allclose(
        weighted @ weighted.T, factor @ factor.T, rtol=1e-6, atol=1e-6
    )


def test_weighted_mean_is_invariant_to_fold_coordinate_rotations():
    rng = np.random.default_rng(1)
    factor = rng.normal(size=(7, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    left = weighted_mean_factor([factor, factor @ rotation], 3)
    right = weighted_mean_factor([factor, factor], 3)
    np.testing.assert_allclose(
        left @ left.T, right @ right.T, rtol=1e-6, atol=1e-6
    )


def test_prepare_target_factors_uses_fold_specific_within_and_shared_cross():
    factors = {0: np.eye(4), 1: np.eye(4)[:, [1, 0, 2, 3]]}
    folds = [(np.array([2, 3]), np.array([0, 1]))] * 2
    within = prepare_target_factors(
        factors, folds, [1], LanguageDecodingMode.WITHIN_DATASET, "extrinsic"
    )
    np.testing.assert_allclose(within[("weighted", 0, 1)], [[1], [0], [0], [0]])
    np.testing.assert_allclose(within[("weighted", 1, 1)], [[0], [1], [0], [0]])
    cross = prepare_target_factors(
        factors, folds, [1], LanguageDecodingMode.CROSS_DATASET, "extrinsic"
    )
    np.testing.assert_allclose(
        cross[("weighted", 0, 1)], cross[("weighted", 1, 1)]
    )


def test_prepare_target_factors_uses_single_full_fit_factor_directly(monkeypatch):
    factor = np.array([[3.0], [1.0], [0.5]], dtype=np.float32)
    folds = [(np.array([0, 1]), np.array([2]))]
    monkeypatch.setattr(
        "src.utils.language_interpret.reconstruction.weighted_mean_factor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a single full-fit factor must not be averaged")
        ),
    )

    prepared = prepare_target_factors(
        {0: factor}, folds, [1], LanguageDecodingMode.CROSS_DATASET, "extrinsic"
    )

    np.testing.assert_array_equal(prepared[("weighted", 0, 1)], factor)
    span = prepared[("span", 0, 1)]
    np.testing.assert_allclose(span.T @ span, np.eye(1), atol=1e-6)


def test_preprocessor_retains_rows_and_encodes_missing_values():
    values = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [np.nan, 0.0],
            [3.0, np.nan],
            [4.0, 2.0],
            [5.0, 3.0],
        ]
    )
    transformer = build_preprocessor([0], [1])
    transformed = transformer.fit_transform(values[:5])
    held_out = transformer.transform(values[5:])
    assert transformed.shape[0] == 5
    assert held_out.shape == (1, transformed.shape[1])
    assert np.isfinite(transformed).all()
    assert np.isfinite(held_out).all()


def test_combined_preprocessor_scales_glove_and_encodes_tasks():
    values = np.array(
        [
            [0.0, 0.0, 2.0, 20.0],
            [1.0, 1.0, 4.0, 40.0],
            [2.0, 0.0, 6.0, 60.0],
        ]
    )
    transformer = build_predictor_preprocessor("tasks_glove", 2, [0], [1], 2)
    transformed = transformer.fit_transform(values[:2])
    assert transformed.shape == (2, 5)
    np.testing.assert_allclose(transformed[:, -2:].mean(axis=0), 0.0, atol=1e-7)


def test_glove_filter_preserves_and_remaps_source_folds():
    selection = np.array([2, 4, 6, 8, 10, 12])
    folds = [
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
    ]
    valid = np.zeros(13, dtype=bool)
    valid[[2, 6, 8, 12]] = True
    filtered, remapped, source_rows = filter_selection_and_folds(
        selection, folds, valid
    )
    np.testing.assert_array_equal(filtered, [2, 6, 8, 12])
    np.testing.assert_array_equal(source_rows, [0, 2, 3, 5])
    np.testing.assert_array_equal(remapped[0][0], [2, 3])
    np.testing.assert_array_equal(remapped[0][1], [0, 1])


def test_reconstruction_metrics_perfect_and_rotation_sensitive_outputs():
    rng = np.random.default_rng(2)
    target = rng.normal(size=(30, 4))
    metrics = reconstruction_metrics(target, target)
    assert metrics["multivariate_r2"] == pytest.approx(1.0)
    assert metrics["mean_coordinate_correlation"] == pytest.approx(1.0)
    assert metrics["linear_cka"] == pytest.approx(1.0)
    scalar_metrics = reconstruction_metrics(target[:, 0], target[:, 0])
    assert scalar_metrics["multivariate_r2"] == pytest.approx(1.0)


def test_run_fit_writes_linear_reconstruction(tmp_path):
    rng = np.random.default_rng(3)
    tasks = np.column_stack(
        [rng.normal(size=50), rng.integers(0, 3, size=50).astype(float)]
    )
    tasks[4, 0] = np.nan
    tasks[8, 1] = np.nan
    target = np.nan_to_num(tasks[:, [0]], nan=0.0) @ np.array([[1.0, -0.5]])
    train, test = np.arange(40), np.arange(40, 50)
    config = _config(rank_grid=[2])
    artifact = ReconstructionArtifact(
        dataset="podcast",
        model="gpt2",
        layer=0,
        results_root=tmp_path,
    ).begin(config={})
    row = _run_fit(
        predictor_set="tasks",
        subspace="rrr",
        target_view="weighted",
        rank=2,
        fold=0,
        predictor_matrix=tasks,
        n_tasks=2,
        n_glove=0,
        continuous=[0],
        categorical=[1],
        target=target.astype(np.float32),
        train_idx=train,
        test_idx=test,
        config=config,
        artifact=artifact,
    )
    assert row["subspace"] == "rrr"
    assert row["predictor_set"] == "tasks"
    assert len(artifact.load_collection("fit_metrics")) == 1


def test_artifact_factor_loading_preserves_rank_and_scaler(tmp_path):
    source = RRRArtifact(
        mode="full_fit",
        dataset="podcast",
        model="gpt2",
        layer=0,
        results_root=tmp_path,
    ).begin(config={})
    factor = np.arange(6, dtype=np.float32).reshape(3, 2)
    scaler = {
        "mean": np.array([1.0, 2.0, 3.0]),
        "scale": np.array([2.0, 2.0, 2.0]),
    }
    source.save_factors(
        method="rrr", rank=2, input_factors=factor, input_scaler=scaler
    )

    factors, scalers, ranked = _load_artifact_factors(
        source, "rrr", {}, embedding_dim=3
    )

    assert ranked
    np.testing.assert_array_equal(factors[2], factor)
    np.testing.assert_array_equal(scalers[2]["mean"], scaler["mean"])
    np.testing.assert_array_equal(scalers[2]["scale"], scaler["scale"])


def test_run_fit_caches_directly_in_reconstruction_artifact(tmp_path):
    rng = np.random.default_rng(4)
    predictors = rng.normal(size=(30, 2))
    target = predictors @ np.array([[1.0], [-0.5]])
    config = _config(rank_grid=[1])
    artifact = ReconstructionArtifact(
        dataset="podcast",
        model="gpt2",
        layer=0,
        results_root=tmp_path,
    ).begin(config={})
    arguments = dict(
        predictor_set="tasks",
        subspace="rrr",
        target_view="weighted",
        rank=1,
        fold=0,
        predictor_matrix=predictors,
        n_tasks=2,
        n_glove=0,
        continuous=[0, 1],
        categorical=[],
        target=target.astype(np.float32),
        train_idx=np.arange(20),
        test_idx=np.arange(20, 30),
        config=config,
        artifact=artifact,
    )

    first = _run_fit(**arguments)
    second = _run_fit(**arguments)
    selector = {
        "predictor_set": "tasks",
        "subspace": "rrr",
        "target_view": "weighted",
        "rank": 1,
        "fold": 0,
    }

    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], float):
            assert second[key] == pytest.approx(first[key])
        else:
            assert second[key] == first[key]
    saved = artifact.load_field("fit_metrics", selector=selector)
    assert saved.iloc[0]["multivariate_r2"] == pytest.approx(
        first["multivariate_r2"]
    )


def test_reconstruction_runs_natively_between_artifacts(tmp_path, monkeypatch):
    source = RRRArtifact(
        mode="full_fit",
        dataset="podcast",
        model="gpt2",
        layer=0,
        results_root=tmp_path,
    ).begin(config={})
    source.save_factors(
        method="rrr", rank=1, input_factors=np.array([[1.0], [0.0], [0.0]])
    )
    source.save_factors(
        method="pca", rank=1, input_factors=np.array([[0.0], [1.0], [0.0]])
    )
    config = _config(
        rrr_source=source.identity_record(),
        mode=LanguageDecodingMode.CROSS_DATASET,
        rank_grid=[1],
        tasks=["surprisal"],
        target_views=["weighted"],
    )
    output = ReconstructionArtifact(
        mode="cross_dataset",
        dataset="podcast",
        model="gpt2",
        layer=0,
        results_root=tmp_path,
    ).begin(config=config, upstreams=[source.identity_record()])
    rng = np.random.default_rng(5)
    embeddings = rng.normal(size=(20, 3)).astype(np.float32)
    words = pd.DataFrame({"word": [f"word-{index}" for index in range(20)]})
    tasks = embeddings[:, [0]].astype(np.float64)
    monkeypatch.setattr(
        "src.utils.language_interpret.reconstruction.load_embeddings",
        lambda *args, **kwargs: (embeddings, words),
    )
    monkeypatch.setattr(
        "src.utils.language_interpret.reconstruction.load_task_matrix",
        lambda *args, **kwargs: (tasks, [0], [], {"surprisal": {"kind": "continuous"}}),
    )

    result = run_language_reconstruction(
        config, source_artifact=source, output_artifact=output
    )

    assert result is output
    assert output.complete
    assert output.load_field("upstream_identities") == [source.identity_record()]
    assert len(output.load_collection("fit_metrics")) == 4


def test_summary_and_paired_rrr_pca_comparison():
    rows = []
    for subspace in ("rrr", "pca"):
        for fold in (0, 1):
            score = 0.1 + 0.05 * (subspace == "rrr")
            rows.append(
                {
                    "predictor_set": "tasks",
                    "subspace": subspace,
                    "target_view": "span",
                    "rank": 2,
                    "fold": fold,
                    "multivariate_r2": score,
                    "mean_coordinate_correlation": score,
                    "linear_cka": score,
                }
            )
    fold_results, summary = summarize_results(pd.DataFrame(rows))
    assert len(fold_results) == 4
    assert len(summary) == 2
    comparison_grid, comparisons = summarize_comparisons(fold_results)
    assert len(comparison_grid) == 6
    assert set(comparisons.metric) == {
        "multivariate_r2",
        "mean_coordinate_correlation",
        "linear_cka",
    }
    assert np.allclose(comparisons["mean"], 0.05)


def test_predictor_contrasts_are_fold_paired():
    rows = []
    scores = {"tasks": 0.1, "glove": 0.2, "tasks_glove": 0.35}
    for predictor_set, score in scores.items():
        for fold in (0, 1):
            rows.append(
                {
                    "predictor_set": predictor_set,
                    "subspace": "rrr",
                    "target_view": "span",
                    "rank": 2,
                    "fold": fold,
                    **{metric: score for metric in (
                        "multivariate_r2",
                        "mean_coordinate_correlation",
                        "linear_cka",
                    )},
                }
            )
    grid, summary = summarize_predictor_contrasts(pd.DataFrame(rows))
    assert len(grid) == 18
    assert set(summary.contrast) == {
        "glove_minus_tasks",
        "tasks_glove_minus_glove",
        "tasks_glove_minus_tasks",
    }
    values = summary.set_index(["contrast", "metric"])["mean"]
    assert values.loc[("glove_minus_tasks", "linear_cka")] == pytest.approx(0.1)
    assert values.loc[("tasks_glove_minus_glove", "linear_cka")] == pytest.approx(0.15)
