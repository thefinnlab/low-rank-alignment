import numpy as np
import pandas as pd
import pytest

from src.core.config import LanguageAxisEncodingConfig
from src.utils.language_axis_encoding import (
    align_decoder_folds,
    build_comparisons,
    covariance_transforms,
    decompose_task_span,
    orthonormal_span,
    random_span_weights,
)


def test_config_defaults_and_validation():
    config = LanguageAxisEncodingConfig(language_decoding_run_path="run")
    assert config.rank_grid == [32]
    assert config.geometries == ["raw", "whitened"]
    assert config.random_control_draws == 20
    with pytest.raises(ValueError, match="geometries"):
        LanguageAxisEncodingConfig(
            language_decoding_run_path="run", geometries=["invalid"]
        )
    assert LanguageAxisEncodingConfig(
        language_decoding_run_path="run", random_control_draws=0
    ).random_control_draws == 0
    with pytest.raises(ValueError, match="non-negative integer"):
        LanguageAxisEncodingConfig(
            language_decoding_run_path="run", random_control_draws=-1
        )


def test_orthonormal_span_ignores_axis_scale_and_dependence():
    weights = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    basis = orthonormal_span(weights)
    assert basis.shape == (3, 2)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
    projector = basis @ basis.T
    scaled = orthonormal_span(weights @ np.diag([5.0, 0.5, 2.0]))
    np.testing.assert_allclose(projector, scaled @ scaled.T, atol=1e-12)


@pytest.mark.parametrize("geometry", ["raw", "whitened"])
def test_task_and_residual_reconstruct_and_residual_has_zero_task_score(geometry):
    rng = np.random.default_rng(4)
    z_train = rng.normal(size=(200, 8)) @ np.diag(np.linspace(0.5, 3.0, 8))
    z_test = rng.normal(size=(40, 8)) @ np.diag(np.linspace(0.5, 3.0, 8))
    weights = rng.normal(size=(8, 3))
    split = decompose_task_span(z_train, z_test, weights, geometry)
    np.testing.assert_allclose(
        split["task_train"] + split["residual_train"],
        split["full_train"],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        split["residual_train"] @ weights, 0.0, atol=2e-5
    )


def test_whitening_is_fit_from_training_covariance():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(500, 5)) @ rng.normal(size=(5, 5))
    _, inverse = covariance_transforms(z)
    whitened = (z - z.mean(axis=0)) @ inverse
    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(5), atol=1e-9)
    with pytest.raises(ValueError, match="rank-deficient"):
        covariance_transforms(np.column_stack([z[:, 0], z[:, 0]]))


def test_random_spans_are_deterministic_and_have_requested_rank():
    left = random_span_weights(10, 4, 12)
    right = random_span_weights(10, 4, 12)
    np.testing.assert_allclose(left, right)
    assert left.shape == (10, 4)
    np.testing.assert_allclose(left.T @ left, np.eye(4), atol=1e-12)


def test_decoder_folds_map_onto_neural_subset():
    decoder_selection = np.array([10, 11, 12, 13, 14, 15])
    neural_selection = np.array([10, 12, 13, 15])
    folds = align_decoder_folds(
        decoder_selection,
        neural_selection,
        [np.array([0, 1, 2]), np.array([3, 4, 5])],
    )
    np.testing.assert_array_equal(folds[0], [0, 1])
    np.testing.assert_array_equal(folds[1], [2, 3])


def test_comparisons_are_fold_paired_and_report_rrr_minus_pca():
    rows = []
    for subspace, offset in (("rrr", 0.1), ("pca", 0.0)):
        for condition, value in (("full", 0.5), ("joint", 0.3), ("residual", 0.4)):
            rows.append(
                {
                    "subspace": subspace,
                    "rank": 32,
                    "geometry": "raw",
                    "fold": 0,
                    "condition": condition,
                    "task": None,
                    "draw": None,
                    "corr": value + offset,
                    "r2": value + offset,
                }
            )
    comparisons, gaps = build_comparisons(pd.DataFrame(rows))
    rrr = comparisons[comparisons.subspace == "rrr"].iloc[0]
    assert rrr.corr_sufficiency_gap == pytest.approx(0.2)
    assert rrr.corr_removal_loss == pytest.approx(0.1)
    assert gaps.iloc[0].corr_joint_rrr_minus_pca == pytest.approx(0.1)
