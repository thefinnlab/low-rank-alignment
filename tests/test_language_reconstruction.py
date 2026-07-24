import numpy as np
import pandas as pd
import pytest

from src.core.config import LanguageDecodingMode, LanguageReconstructionConfig
from src.utils.language_reconstruction import (
    _run_fit,
    build_preprocessor,
    orthonormal_factor,
    prepare_target_factors,
    reconstruction_metrics,
    resolve_pca_sibling,
    summarize_comparisons,
    summarize_results,
    weighted_mean_factor,
)


def _config(**overrides):
    values = {
        "rrr_run_path": "unused",
        "rank_grid": [1, 2],
        "tasks": ["surprisal", "part_of_speech"],
        "ridge_alphas": [0.1, 1.0],
    }
    values.update(overrides)
    return LanguageReconstructionConfig(**values)


def test_reconstruction_config_validates_choices():
    config = _config()
    assert config.target_views == ["weighted", "span"]
    with pytest.raises(ValueError, match="target_views"):
        _config(target_views=["raw"])
    with pytest.raises(ValueError, match="duplicate ranks"):
        _config(rank_grid=[2, 2])


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("results/model/rrr", "results/model/pca"),
        (
            "results/model/cross_dataset_rrr/source",
            "results/model/cross_dataset_pca/source",
        ),
    ],
)
def test_resolve_pca_sibling(source, expected):
    assert str(resolve_pca_sibling(source)) == expected


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
    row = _run_fit(
        output_dir=tmp_path,
        subspace="rrr",
        target_view="weighted",
        rank=2,
        fold=0,
        task_matrix=tasks,
        continuous=[0],
        categorical=[1],
        target=target.astype(np.float32),
        train_idx=train,
        test_idx=test,
        config=config,
    )
    assert row["subspace"] == "rrr"
    assert len(list(tmp_path.rglob("metrics.json"))) == 1


def test_summary_and_paired_rrr_pca_comparison():
    rows = []
    for subspace in ("rrr", "pca"):
        for fold in (0, 1):
            score = 0.1 + 0.05 * (subspace == "rrr")
            rows.append(
                {
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
