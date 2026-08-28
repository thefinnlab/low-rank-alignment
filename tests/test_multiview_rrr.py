"""Focused behavior tests for multiview RRR helpers."""

import numpy as np

from src.utils.rrr.multiview import (
    fit_banded_ridge_baseline,
    project_view_bands,
)


def test_project_view_bands_preserves_raw_A_and_config_order():
    embeddings = {
        "second": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "first": np.array([[2.0, 1.0], [0.0, 3.0]], dtype=np.float32),
    }
    raw_a_first = np.array([[2.0], [0.0]], dtype=np.float32)
    raw_a_second = np.array([[0.0], [3.0]], dtype=np.float32)
    bases = {"first": {0: raw_a_first}, "second": {0: raw_a_second}}

    bands, boundaries = project_view_bands(
        embeddings, bases, ["first", "second"], fold=0
    )

    np.testing.assert_array_equal(bands[0], embeddings["first"] @ raw_a_first)
    np.testing.assert_array_equal(bands[1], embeddings["second"] @ raw_a_second)
    assert boundaries == [(0, 1), (1, 2)]


def test_banded_ridge_recovers_concatenated_primal_weights():
    rng = np.random.default_rng(3)
    bands = [
        rng.normal(size=(48, 3)).astype(np.float32),
        rng.normal(size=(48, 2)).astype(np.float32),
    ]
    true_weights = [rng.normal(size=(3, 4)), rng.normal(size=(2, 4))]
    targets = (bands[0] @ true_weights[0] + bands[1] @ true_weights[1]).astype(
        np.float32
    )

    fitted = fit_banded_ridge_baseline(
        [band[:36] for band in bands],
        [band[36:] for band in bands],
        targets[:36],
        alphas=np.logspace(-2, 2, 5),
        ridge_cv_folds=3,
        solver_params={"n_iter": 5, "random_state": 7, "progress_bar": False},
        force_cpu=True,
    )

    assert fitted.beta_full_raw.shape == (5, 4)
    assert fitted.deltas.shape == (2, 4)
    np.testing.assert_allclose(
        np.concatenate([band[:36] for band in bands], axis=1)
        @ fitted.beta_full_raw
        + fitted.intercept,
        fitted.Y_hat_train,
        rtol=2e-4,
        atol=2e-4,
    )
