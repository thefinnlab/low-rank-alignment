"""Tests for utils/significance.py."""

import numpy as np
import pytest
import torch

from src.utils.significance import batch_corrcoef, significant_lags_all_electrodes_gpu


# ---------------------------------------------------------------------------
# batch_corrcoef
# ---------------------------------------------------------------------------


class TestBatchCorrcoef:
    def _make_tensors(self, n_electrodes=4, n_lags=5, n_words=30, seed=0):
        rng = np.random.default_rng(seed)
        x = torch.tensor(
            rng.standard_normal((n_electrodes, n_lags, n_words)), dtype=torch.float32
        )
        return x

    def test_identical_inputs_corr_one(self):
        x = self._make_tensors()
        r = batch_corrcoef(x, x)
        assert r.shape == (x.shape[0], x.shape[1])
        np.testing.assert_allclose(r.numpy(), np.ones_like(r.numpy()), atol=1e-5)

    def test_negated_input_corr_minus_one(self):
        x = self._make_tensors()
        r = batch_corrcoef(x, -x)
        np.testing.assert_allclose(r.numpy(), -np.ones_like(r.numpy()), atol=1e-5)

    def test_orthogonal_inputs_corr_zero(self):
        """Construct x and y that are orthogonal along the words dimension."""
        n_electrodes, n_lags, n_words = 2, 3, 100
        rng = np.random.default_rng(7)
        # Build a signal and its residual after projection (guaranteed orthogonal)
        raw = rng.standard_normal((n_electrodes, n_lags, n_words))
        # Orthogonalize by centering then subtracting projection
        raw_c = raw - raw.mean(axis=-1, keepdims=True)
        # Create y by rotating 90 degrees via random mixing
        mix = rng.standard_normal((n_words, n_words))
        q, _ = np.linalg.qr(mix)
        # y = x @ q  (q is orthogonal), so corr(x, y) should be near 0 on average
        y = np.einsum("eln,nm->elm", raw_c, q)
        x_t = torch.tensor(raw_c, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        r = batch_corrcoef(x_t, y_t)
        # Correlations should be small (not necessarily exactly 0, but close)
        assert r.abs().max().item() < 0.3

    def test_output_shape(self):
        n_electrodes, n_lags, n_words = 7, 6, 50
        x = self._make_tensors(n_electrodes, n_lags, n_words)
        r = batch_corrcoef(x, x)
        assert r.shape == (n_electrodes, n_lags)

    def test_returns_tensor(self):
        x = self._make_tensors()
        r = batch_corrcoef(x, x)
        assert isinstance(r, torch.Tensor)


# ---------------------------------------------------------------------------
# significant_lags_all_electrodes_gpu (CPU, small n_permutations)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSignificantLagsAllElectrodes:
    @pytest.fixture
    def cpu_device(self):
        return torch.device("cpu")

    def test_output_shapes(self, cpu_device):
        rng = np.random.default_rng(0)
        n_electrodes, n_lags, n_words = 3, 4, 20
        pred = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)
        actual = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)

        sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(
            pred, actual, n_permutations=100, device=cpu_device
        )
        for arr in [sig, p_values, q_values, obs_corrs]:
            assert arr.shape == (n_electrodes, n_lags)

    def test_obs_corrs_approx_one_when_identical(self, cpu_device):
        rng = np.random.default_rng(1)
        n_electrodes, n_lags, n_words = 2, 3, 30
        pred = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)

        sig, p_values, q_values, obs_corrs = significant_lags_all_electrodes_gpu(
            pred, pred, n_permutations=100, device=cpu_device
        )
        np.testing.assert_allclose(
            obs_corrs, np.ones((n_electrodes, n_lags)), atol=1e-5
        )

    def test_sig_dtype_is_bool(self, cpu_device):
        rng = np.random.default_rng(2)
        n_electrodes, n_lags, n_words = 2, 2, 15
        pred = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)
        actual = rng.standard_normal((n_electrodes, n_lags, n_words)).astype(np.float32)

        sig, _, _, _ = significant_lags_all_electrodes_gpu(
            pred, actual, n_permutations=50, device=cpu_device
        )
        assert sig.dtype == bool
