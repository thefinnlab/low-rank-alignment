import numpy as np
import pytest

from src.utils.subspace_comparison.metrics import aggregate_subspace_basis


def _projector(basis):
    return basis @ basis.T


def test_prediction_weighted_uses_singular_energy_not_raw_component_scale():
    blocks = [
        np.array([[100.0, 0.0], [0.0, 0.0], [0.0, 1000.0]]),
        np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 1000.0]]),
    ]

    basis = aggregate_subspace_basis(
        blocks,
        rank=2,
        method="prediction_weighted",
        prediction_singular_values=np.array([10.0, 1.0]),
    )

    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(
        _projector(basis), np.diag([1.0, 1.0, 0.0]), atol=1e-6
    )

    sign_flipped = [block * np.array([-1.0, 1.0]) for block in blocks]
    flipped_basis = aggregate_subspace_basis(
        sign_flipped,
        rank=2,
        method="prediction_weighted",
        prediction_singular_values=np.array([10.0, 1.0]),
    )
    np.testing.assert_allclose(
        _projector(flipped_basis), _projector(basis), atol=1e-6
    )


def test_prediction_weighted_requires_valid_weights_and_full_rank():
    blocks = [np.eye(2)]
    with pytest.raises(ValueError, match="requires prediction_singular_values"):
        aggregate_subspace_basis(blocks, rank=1, method="prediction_weighted")
    with pytest.raises(ValueError, match="must be finite"):
        aggregate_subspace_basis(
            blocks,
            rank=1,
            method="prediction_weighted",
            prediction_singular_values=np.array([np.nan]),
        )
    with pytest.raises(ValueError, match="rank deficient"):
        aggregate_subspace_basis(
            blocks,
            rank=2,
            method="prediction_weighted",
            prediction_singular_values=np.array([1.0, 0.0]),
        )
