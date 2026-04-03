"""Tests for partial RRR analysis: partial_out_subspace and sweep logic."""

import os
import sys

import numpy as np
import pytest

from src.utils.analysis import partial_out_subspace, partial_out_svd_directions
from src.core.config import PartialRRRConfig, SVDMode
from src.utils.config_utils import dict_to_dataclass
from src.utils.srrr import save_rrr_factors, save_rrr_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orthogonal_basis(rng):
    """Random orthonormal basis (10 directions, 20 features)."""
    n_dirs, n_features = 10, 20
    A = rng.standard_normal((n_dirs, n_features))
    Q, _ = np.linalg.qr(A.T)
    Vt = Q.T[:n_dirs]  # (n_dirs, n_features), rows are orthonormal
    return Vt


@pytest.fixture
def sample_X(rng):
    """Sample embedding matrix (50 words, 20 features)."""
    return rng.standard_normal((50, 20)).astype(np.float64)


# ---------------------------------------------------------------------------
# partial_out_subspace tests
# ---------------------------------------------------------------------------


def test_partial_out_subspace_orthogonality(sample_X, rng):
    """After partialling, X_partial has zero projection onto removed subspace."""
    n_features = sample_X.shape[1]
    k = 3
    A = rng.standard_normal((n_features, k))
    # Directions as rows: transpose A so rows are the k directions
    directions = A.T  # (k, n_features)
    X_partial = partial_out_subspace(sample_X, directions)
    Q, _ = np.linalg.qr(A)
    projection = X_partial @ Q
    np.testing.assert_allclose(projection, 0, atol=1e-10)


def test_partial_out_subspace_preserves_complement(rng):
    """Projection onto directions orthogonal to A is unchanged."""
    n_words, n_features, k = 50, 20, 3
    Q_full, _ = np.linalg.qr(rng.standard_normal((n_features, n_features)))
    directions = Q_full[:, :k].T  # (k, n_features) — first k as rows
    complement = Q_full[:, k:]
    X = rng.standard_normal((n_words, n_features))
    X_partial = partial_out_subspace(X, directions)
    np.testing.assert_allclose(X_partial @ complement, X @ complement, atol=1e-10)


def test_partial_out_subspace_idempotent(sample_X, rng):
    """Applying the same partialling twice gives the same result."""
    k = 4
    directions = rng.standard_normal((k, sample_X.shape[1]))
    X_once = partial_out_subspace(sample_X, directions)
    X_twice = partial_out_subspace(X_once, directions)
    np.testing.assert_allclose(X_twice, X_once, atol=1e-10)


def test_partial_out_subspace_shape(sample_X, rng):
    """Output shape must match input shape."""
    directions = rng.standard_normal((5, sample_X.shape[1]))
    assert partial_out_subspace(sample_X, directions).shape == sample_X.shape


def test_partial_out_subspace_non_orthonormal(rng):
    """Works correctly even when direction vectors are not orthonormal."""
    n_words, n_features, k = 50, 20, 3
    X = rng.standard_normal((n_words, n_features))
    # Non-orthonormal directions (scaled and skewed)
    A_raw = rng.standard_normal((k, n_features)) * np.arange(1, k + 1)[:, None]
    X_partial = partial_out_subspace(X, A_raw)

    # Verify: projection onto the QR-orthogonalized basis should be zero
    Q, _ = np.linalg.qr(A_raw.T)  # (n_features, k)
    proj = X_partial @ Q
    np.testing.assert_allclose(proj, 0, atol=1e-10)


def test_partial_out_subspace_matches_svd_version(sample_X, orthogonal_basis):
    """When directions are rows of Vt, result matches partial_out_svd_directions."""
    start, size = 2, 3
    # partial_out_subspace takes rows-as-directions, same as Vt
    X_subspace = partial_out_subspace(sample_X, orthogonal_basis[start : start + size])
    X_svd = partial_out_svd_directions(sample_X, orthogonal_basis, start, size)
    np.testing.assert_allclose(X_subspace, X_svd, atol=1e-10)


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_partial_rrr_config_round_trip():
    """PartialRRRConfig can be serialized to dict and back."""
    config_dict = {
        "encoding": {
            "data_root": "data/",
            "model": "gpt2xl",
            "layer": 24,
            "offset": 0,
        },
        "rank_grid": [1, 5, 10],
        "mode": "per_subject",
    }

    restored = dict_to_dataclass(config_dict, PartialRRRConfig)
    assert restored.rank_grid == [1, 5, 10]


def test_partial_rrr_config_default_rank_grid_is_none():
    """Default rank_grid should be None (auto-discover)."""
    config = PartialRRRConfig()
    assert config.rank_grid is None


# ---------------------------------------------------------------------------
# Rank discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverRanks:
    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
        scripts_dir = os.path.abspath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    def test_discovers_all_ranks(self, tmp_path):
        """Discovers all k* directories sorted."""
        from run_partial_rrr import _discover_ranks

        for k in [5, 1, 10]:
            os.makedirs(tmp_path / f"k{k}" / "fold_0")
        assert _discover_ranks(str(tmp_path)) == [1, 5, 10]

    def test_filters_by_rank_grid(self, tmp_path):
        """When rank_grid is provided, only matching ranks are returned."""
        from run_partial_rrr import _discover_ranks

        for k in [1, 2, 5, 10, 20]:
            os.makedirs(tmp_path / f"k{k}" / "fold_0")
        assert _discover_ranks(str(tmp_path), rank_grid=[2, 10]) == [2, 10]

    def test_raises_on_missing_dir(self, tmp_path):
        from run_partial_rrr import _discover_ranks

        with pytest.raises(FileNotFoundError):
            _discover_ranks(str(tmp_path / "nonexistent"))

    def test_raises_on_empty_dir(self, tmp_path):
        from run_partial_rrr import _discover_ranks

        with pytest.raises(FileNotFoundError, match="No k\\* rank directories"):
            _discover_ranks(str(tmp_path))

    def test_raises_when_no_matching_ranks(self, tmp_path):
        from run_partial_rrr import _discover_ranks

        os.makedirs(tmp_path / "k1" / "fold_0")
        with pytest.raises(ValueError, match="None of the requested ranks"):
            _discover_ranks(str(tmp_path), rank_grid=[99])


# ---------------------------------------------------------------------------
# Sweep logic test
# ---------------------------------------------------------------------------


class TestPartialRRRSweep:
    """Integration test for _run_partial_rrr_sweep with synthetic data."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
        scripts_dir = os.path.abspath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    def test_sweep_returns_expected_keys_and_shapes(self, tmp_path, rng):
        """Run sweep with synthetic A matrices, verify output dict."""
        from run_partial_rrr import _run_partial_rrr_sweep

        n_words, n_embed, n_elecs, n_lags = 80, 12, 3, 2
        n_targets = n_elecs * n_lags
        X = rng.standard_normal((n_words, n_embed)).astype(np.float32)
        Y = rng.standard_normal((n_words, n_targets)).astype(np.float32)

        rank_grid = [1, 3]
        n_folds = 2
        test_idxs = [np.arange(0, n_words // 2), np.arange(n_words // 2, n_words)]
        provided_idxs = []
        for ti in test_idxs:
            train_mask = np.ones(n_words, dtype=bool)
            train_mask[ti] = False
            provided_idxs.append((np.where(train_mask)[0], ti))

        # Create synthetic A/B matrices and save as RRR results
        rrr_dir = str(tmp_path / "rrr")
        max_k = max(rank_grid)
        for f in range(n_folds):
            A_full = rng.standard_normal((n_embed, max_k)).astype(np.float32)
            B_full = rng.standard_normal((n_targets, max_k)).astype(np.float32)
            save_rrr_factors(rrr_dir, fold=f, A=A_full, B=B_full)
            for k in rank_grid:
                save_rrr_result(rrr_dir, k=k, fold=f, metrics={"full_r2": 0.5})

        out_dir = str(tmp_path / "partial_rrr")
        os.makedirs(out_dir)

        results = _run_partial_rrr_sweep(
            out_dir,
            X,
            Y,
            rrr_dir,
            provided_idxs,
            n_elecs,
            n_lags,
            rank_grid,
        )

        assert set(results.keys()) == {1, 3}
        for k in rank_grid:
            assert results[k].shape == (n_folds, n_elecs, n_lags)

    def test_larger_subspace_removes_more_signal(self, tmp_path, rng):
        """Partialling a larger subspace should generally reduce correlation more."""
        from run_partial_rrr import _run_partial_rrr_sweep

        n_words, n_embed, n_elecs, n_lags = 100, 15, 4, 2
        n_targets = n_elecs * n_lags
        true_rank = 5
        A_true = rng.standard_normal((n_embed, true_rank)).astype(np.float32)
        B_true = rng.standard_normal((n_targets, true_rank)).astype(np.float32)
        X = rng.standard_normal((n_words, n_embed)).astype(np.float32)
        Y = (
            X @ A_true @ B_true.T + 0.5 * rng.standard_normal((n_words, n_targets))
        ).astype(np.float32)

        rank_grid = [1, 5]
        n_folds = 2
        test_idxs = [np.arange(0, n_words // 2), np.arange(n_words // 2, n_words)]
        provided_idxs = []
        for ti in test_idxs:
            train_mask = np.ones(n_words, dtype=bool)
            train_mask[ti] = False
            provided_idxs.append((np.where(train_mask)[0], ti))

        # Use the true A directions as the RRR result
        rrr_dir = str(tmp_path / "rrr")
        for f in range(n_folds):
            save_rrr_factors(rrr_dir, fold=f, A=A_true, B=B_true)
            for k in rank_grid:
                save_rrr_result(rrr_dir, k=k, fold=f, metrics={"full_r2": 0.5})

        out_dir = str(tmp_path / "partial_rrr")
        os.makedirs(out_dir)

        results = _run_partial_rrr_sweep(
            out_dir,
            X,
            Y,
            rrr_dir,
            provided_idxs,
            n_elecs,
            n_lags,
            rank_grid,
        )

        # Mean correlation after removing rank-5 subspace should be lower
        # than after removing rank-1 subspace
        mean_k1 = results[1].mean()
        mean_k5 = results[5].mean()
        assert mean_k5 < mean_k1, (
            f"Expected removing larger subspace to reduce correlation more: "
            f"k=1 mean={mean_k1:.4f}, k=5 mean={mean_k5:.4f}"
        )


# ---------------------------------------------------------------------------
# Save results test
# ---------------------------------------------------------------------------


class TestSavePartialRRRResults:
    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
        scripts_dir = os.path.abspath(scripts_dir)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

    def test_saves_npz_and_csv(self, tmp_path, rng):
        from run_partial_rrr import _save_partial_rrr_results

        n_folds, n_elecs, n_lags = 2, 4, 3
        baseline_corrs = rng.standard_normal((n_folds, n_elecs, n_lags))
        sig_mask = np.zeros((n_elecs, n_lags), dtype=bool)
        sig_mask[:2, :] = True

        all_results = {
            1: rng.standard_normal((n_folds, n_elecs, n_lags)),
            5: rng.standard_normal((n_folds, n_elecs, n_lags)),
        }

        out_dir = str(tmp_path)
        _save_partial_rrr_results(out_dir, all_results, baseline_corrs, sig_mask)

        assert os.path.exists(os.path.join(out_dir, "partial_rrr_results.npz"))
        assert os.path.exists(os.path.join(out_dir, "summary.csv"))

        import pandas as pd

        df = pd.read_csv(os.path.join(out_dir, "summary.csv"))
        assert len(df) == 2
        assert set(df["rank_k"]) == {1, 5}
        assert "mean_corr_sig" in df.columns
        assert "drop_sig" in df.columns
