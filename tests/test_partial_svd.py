"""Tests for partial SVD analysis utilities."""

import numpy as np
import pytest

from utils.analysis import partial_out_svd_directions
from utils.config import PartialSVDConfig
from utils.config_utils import dict_to_dataclass


# ---------------------------------------------------------------------------
# partial_out_svd_directions tests
# ---------------------------------------------------------------------------

@pytest.fixture
def orthogonal_basis(rng):
    """Create a random orthonormal basis (10 directions, 20 features)."""
    n_dirs, n_features = 10, 20
    A = rng.standard_normal((n_dirs, n_features))
    Q, _ = np.linalg.qr(A.T)
    Vt = Q.T[:n_dirs]  # (n_dirs, n_features), rows are orthonormal
    return Vt


@pytest.fixture
def sample_X(rng):
    """Sample embedding matrix (50 words, 20 features)."""
    return rng.standard_normal((50, 20)).astype(np.float64)


def test_partial_out_svd_directions_orthogonality(sample_X, orthogonal_basis):
    """After partialling, X_partial should have zero projection onto removed directions."""
    start, size = 2, 3
    X_partial = partial_out_svd_directions(sample_X, orthogonal_basis, start, size)

    V_block = orthogonal_basis[start:start + size]
    projection = X_partial @ V_block.T  # should be ~0
    np.testing.assert_allclose(projection, 0, atol=1e-10)


def test_partial_out_svd_directions_preserves_complement(sample_X, orthogonal_basis):
    """Projection onto non-removed directions should be unchanged."""
    start, size = 2, 3
    X_partial = partial_out_svd_directions(sample_X, orthogonal_basis, start, size)

    # Directions outside the removed block
    other_indices = [i for i in range(orthogonal_basis.shape[0]) if i < start or i >= start + size]
    V_other = orthogonal_basis[other_indices]

    proj_original = sample_X @ V_other.T
    proj_partial = X_partial @ V_other.T
    np.testing.assert_allclose(proj_partial, proj_original, atol=1e-10)


def test_partial_out_svd_directions_idempotent(sample_X, orthogonal_basis):
    """Applying the same partialling twice gives the same result."""
    start, size = 0, 4
    X_once = partial_out_svd_directions(sample_X, orthogonal_basis, start, size)
    X_twice = partial_out_svd_directions(X_once, orthogonal_basis, start, size)
    np.testing.assert_allclose(X_twice, X_once, atol=1e-10)


def test_partial_out_svd_directions_shape(sample_X, orthogonal_basis):
    """Output shape must match input shape."""
    X_partial = partial_out_svd_directions(sample_X, orthogonal_basis, 1, 5)
    assert X_partial.shape == sample_X.shape


def test_partial_out_svd_directions_full_removal(rng):
    """Removing all directions from a rank-k matrix should give zeros."""
    n_words, n_features, k = 30, 10, 5
    # Build X that lives in the span of Vt[:k]
    A = rng.standard_normal((n_features, n_features))
    Q, _ = np.linalg.qr(A)
    Vt = Q.T  # (n_features, n_features) full basis
    coeffs = rng.standard_normal((n_words, k))
    X = coeffs @ Vt[:k]  # X lives in span of first k directions

    X_partial = partial_out_svd_directions(X, Vt, 0, k)
    np.testing.assert_allclose(X_partial, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# Config round-trip test
# ---------------------------------------------------------------------------

def test_partial_svd_config_round_trip():
    """PartialSVDConfig can be serialized to dict and back."""
    config = PartialSVDConfig(
        partial_group_sizes=[1, 5, 20],
        max_partial_direction=[50, 50, 100],
        start_direction=[0, 0, 0],
        step_size=[1, 5, 20],
    )

    config_dict = {
        "encoding": {
            "data_root": config.encoding.data_root,
            "model": config.encoding.model,
            "layer": config.encoding.layer,
            "offset": config.encoding.offset,
        },
        "partial_group_sizes": config.partial_group_sizes,
        "max_partial_direction": config.max_partial_direction,
        "start_direction": config.start_direction,
        "step_size": config.step_size,
    }

    restored = dict_to_dataclass(config_dict, PartialSVDConfig)
    assert restored.partial_group_sizes == [1, 5, 20]
    assert restored.max_partial_direction == [50, 50, 100]
    assert restored.step_size == [1, 5, 20]
    assert restored.encoding.layer == 24


# ---------------------------------------------------------------------------
# Sweep window generation test
# ---------------------------------------------------------------------------

def test_sweep_window_generation():
    """Verify the sweep loop produces expected (group_size, window_start) pairs."""
    config = PartialSVDConfig(
        partial_group_sizes=[1, 3],
        max_partial_direction=[5, 9],
        start_direction=[0, 0],
        step_size=[1, 3],
    )

    n_dirs_available = 10
    windows = []
    for i in range(len(config.partial_group_sizes)):
        gs = config.partial_group_sizes[i]
        start = config.start_direction[i]
        max_dir = config.max_partial_direction[i]
        step = config.step_size[i]

        ws = start
        while ws + gs <= max_dir and ws + gs <= n_dirs_available:
            windows.append((gs, ws))
            ws += step

    # group_size=1: windows at 0,1,2,3,4 (5 windows, [0,5) range)
    expected_gs1 = [(1, w) for w in range(5)]
    # group_size=3: windows at 0,3,6 (3 windows, start+size <= 9)
    expected_gs3 = [(3, 0), (3, 3), (3, 6)]

    assert windows == expected_gs1 + expected_gs3
