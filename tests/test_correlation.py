import numpy as np

from src.utils.correlation import (
    fisher_z_mean,
    fisher_z_mean_sem,
    targetwise_correlation,
)


def test_fisher_z_mean_differs_from_arithmetic_mean():
    corrs = np.array([0.1, 0.5, 0.9])

    assert not np.isclose(fisher_z_mean(corrs), corrs.mean())


def test_fisher_z_mean_handles_nan_and_clips_extreme_values():
    corrs = np.array([np.nan, -1.0, 1.0])

    result = fisher_z_mean(corrs)

    assert np.isfinite(result)
    assert -1.0 < result < 1.0


def test_fisher_z_sem_returns_asymmetric_r_space_bounds():
    corrs = np.array([0.2, 0.6, 0.9])

    mean_r, lower, upper = fisher_z_mean_sem(corrs)

    assert np.isfinite(mean_r)
    assert lower > 0
    assert upper > 0
    assert not np.isclose(lower, upper)


def test_fisher_z_sem_collapses_for_single_valid_sample():
    mean_r, lower, upper = fisher_z_mean_sem(np.array([np.nan, 0.4]))

    assert np.isclose(mean_r, 0.4)
    assert lower == 0.0
    assert upper == 0.0


def test_targetwise_correlation_matches_columns_and_handles_constants():
    A = np.array(
        [
            [1.0, 1.0, 5.0],
            [2.0, 2.0, 5.0],
            [3.0, 3.0, 5.0],
        ]
    )
    B = np.array(
        [
            [1.0, 3.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 1.0, 3.0],
        ]
    )

    corr = targetwise_correlation(A, B)

    np.testing.assert_allclose(corr[:2], [1.0, -1.0])
    assert np.isnan(corr[2])
