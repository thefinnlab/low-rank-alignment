import numpy as np

from src.utils.significance import ieeg
from src.utils.significance.significance_ieeg import (
    _best_lag_indices,
    _empirical_pvalues,
)


def test_per_lag_pvalues_use_electrode_specific_max_lag_null():
    observed = np.array([[0.2, 0.5], [0.4, np.nan]])
    null_maxima = np.array([[0.1, 0.5], [0.3, 0.2], [0.6, 0.4]])

    result = _empirical_pvalues(observed, null_maxima)

    np.testing.assert_allclose(result, [[0.75, 0.5], [0.75, 1.0]])


def test_best_lag_uses_first_maximum_and_marks_invalid_electrode():
    result = _best_lag_indices(
        np.array([[0.1, 0.4, 0.4], [np.nan, np.nan, np.nan]])
    )

    np.testing.assert_array_equal(result, [1, -1])


def test_max_lag_null_fisher_averages_folds_then_maximizes_lags(monkeypatch):
    predictions = [np.ones((8, 1, 2)), np.ones((8, 1, 2))]
    targets = [np.ones((8, 1, 2)), np.ones((8, 1, 2))]
    shifts = np.array([[2, 3], [3, 4]])
    shifted_fold_corrs = np.array(
        [
            [[[0.1, 0.4]], [[0.3, 0.2]]],
            [[[0.5, 0.1]], [[0.1, 0.3]]],
        ]
    )

    monkeypatch.setattr(
        ieeg,
        "_compute_shifted_fold_correlations_batch",
        lambda *_args, **_kwargs: shifted_fold_corrs,
    )
    maxima, metadata = ieeg.fold_local_max_lag_correlation_null_scores(
        predictions,
        targets,
        shift_matrix=shifts,
        null_batch_size=2,
    )

    expected_lags = np.tanh(np.mean(np.arctanh(shifted_fold_corrs), axis=1))
    np.testing.assert_allclose(maxima, np.max(expected_lags, axis=-1))
    np.testing.assert_array_equal(metadata["shift_matrix"], shifts)
