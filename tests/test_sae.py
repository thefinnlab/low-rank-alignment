"""Unit tests for SAE model, training, metrics, and config."""

import os

import numpy as np
import pytest
import torch
import yaml

from utils.sae import (
    SAEEncoder,
    SAERegressorEndToEnd,
    compute_sae_metrics,
    train_sae_end_to_end,
    train_sae_alternating,
    _resample_dead_neurons,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_data(rng):
    """Small synthetic (X_proj, Y) pair for testing: 80 samples, k=4, 6 targets."""
    n, k, n_targets = 80, 4, 6
    X_proj = rng.standard_normal((n, k)).astype(np.float32)
    Y = rng.standard_normal((n, n_targets)).astype(np.float32)
    return X_proj, Y


@pytest.fixture
def structured_data(rng):
    """Synthetic data with learnable structure: Y = X @ W + noise."""
    n, k, hidden, n_targets = 120, 4, 8, 6
    X_proj = rng.standard_normal((n, k)).astype(np.float32)
    W = rng.standard_normal((k, n_targets)).astype(np.float32)
    Y = X_proj @ W + 0.1 * rng.standard_normal((n, n_targets)).astype(np.float32)
    return X_proj, Y


# ---------------------------------------------------------------------------
# TestSAEEncoder
# ---------------------------------------------------------------------------

class TestSAEEncoder:
    def test_output_shape(self):
        encoder = SAEEncoder(input_dim=4, hidden_dim=10)
        x = torch.randn(16, 4)
        out = encoder(x)
        assert out.shape == (16, 10)

    def test_relu_nonnegativity(self):
        encoder = SAEEncoder(input_dim=4, hidden_dim=10)
        x = torch.randn(32, 4)
        out = encoder(x)
        assert (out >= 0).all()

    def test_single_sample(self):
        encoder = SAEEncoder(input_dim=3, hidden_dim=7)
        x = torch.randn(1, 3)
        out = encoder(x)
        assert out.shape == (1, 7)


# ---------------------------------------------------------------------------
# TestSAERegressorEndToEnd
# ---------------------------------------------------------------------------

class TestSAERegressorEndToEnd:
    def test_output_shapes(self):
        model = SAERegressorEndToEnd(input_dim=4, hidden_dim=10, output_dim=6)
        x = torch.randn(16, 4)
        preds, acts = model(x)
        assert preds.shape == (16, 6)
        assert acts.shape == (16, 10)

    def test_activations_nonnegative(self):
        model = SAERegressorEndToEnd(input_dim=4, hidden_dim=10, output_dim=6)
        x = torch.randn(32, 4)
        _, acts = model(x)
        assert (acts >= 0).all()


# ---------------------------------------------------------------------------
# TestComputeSAEMetrics
# ---------------------------------------------------------------------------

class TestComputeSAEMetrics:
    def test_all_dead_neurons(self):
        acts = torch.zeros(20, 10)
        m = compute_sae_metrics(acts)
        assert m["dead_neurons"] == 1.0
        assert m["l0_sparsity"] == 0.0
        assert m["mean_activation"] == 0.0

    def test_all_active(self):
        acts = torch.ones(20, 10)
        m = compute_sae_metrics(acts)
        assert m["dead_neurons"] == 0.0
        assert m["l0_sparsity"] == 10.0
        assert abs(m["mean_activation"] - 1.0) < 1e-5

    def test_l0_sparsity_partial(self):
        acts = torch.zeros(4, 10)
        acts[0, :3] = 1.0  # 3 active
        acts[1, :5] = 1.0  # 5 active
        acts[2, :1] = 1.0  # 1 active
        acts[3, :7] = 1.0  # 7 active
        m = compute_sae_metrics(acts)
        assert abs(m["l0_sparsity"] - 4.0) < 1e-5  # mean(3, 5, 1, 7) = 4

    def test_activation_frequency_shape(self):
        acts = torch.randn(20, 10).relu()
        m = compute_sae_metrics(acts)
        assert m["activation_frequency"].shape == (10,)
        assert (m["activation_frequency"] >= 0).all()
        assert (m["activation_frequency"] <= 1).all()

    def test_l1_loss(self):
        acts = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
        m = compute_sae_metrics(acts)
        expected_l1 = (1.0 + 2.0 + 0.0 + 3.0 + 0.0 + 0.0) / 6.0
        assert abs(m["l1_loss"] - expected_l1) < 1e-5


# ---------------------------------------------------------------------------
# TestTrainSAEEndToEnd
# ---------------------------------------------------------------------------

class TestTrainSAEEndToEnd:
    def test_loss_decreases(self, structured_data):
        X, Y = structured_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=12, l1_penalty=1e-4, lr=1e-2,
            num_epochs=50, device="cpu", log_every=1,
        )
        log = result["training_log"]
        assert log[-1]["total_loss"] < log[0]["total_loss"]

    def test_output_keys(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, lr=1e-3,
            num_epochs=5, device="cpu", log_every=1,
        )
        assert "encoder_state_dict" in result
        assert "regressor_weight" in result
        assert "regressor_bias" in result
        assert "metrics" in result
        assert "training_log" in result
        assert "Y_hat_test" in result
        assert "Y_hat_train" in result

        m = result["metrics"]
        for key in ["r2", "correlation", "l0_sparsity", "dead_neurons",
                     "mean_activation", "l1_loss", "activation_frequency"]:
            assert key in m, f"Missing metric: {key}"

    def test_training_log_format(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, lr=1e-3,
            num_epochs=10, device="cpu", log_every=1,
        )
        log = result["training_log"]
        assert len(log) == 10
        for entry in log:
            for key in ["epoch", "mse_loss", "l1_loss", "total_loss", "test_r2", "test_corr"]:
                assert key in entry

    def test_log_every_skips_test_metrics(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, lr=1e-3,
            num_epochs=10, device="cpu", log_every=5,
        )
        log = result["training_log"]
        assert len(log) == 10
        # Epoch 0 and 5 should have metrics (log_every=5), epoch 9 (last) too
        assert log[0]["test_r2"] is not None
        assert log[1]["test_r2"] is None
        assert log[9]["test_r2"] is not None

    def test_l1_increases_sparsity(self, structured_data):
        X, Y = structured_data
        n = X.shape[0]
        split = n // 2

        result_low = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=12, l1_penalty=1e-5, lr=1e-2,
            num_epochs=100, device="cpu", log_every=100,
        )
        result_high = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=12, l1_penalty=1.0, lr=1e-2,
            num_epochs=100, device="cpu", log_every=100,
        )
        assert result_high["metrics"]["l0_sparsity"] <= result_low["metrics"]["l0_sparsity"]

    def test_mini_batch(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, lr=1e-3,
            num_epochs=5, batch_size=16, device="cpu",
        )
        assert len(result["training_log"]) == 5

    def test_prediction_shapes(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, lr=1e-3,
            num_epochs=5, device="cpu",
        )
        assert result["Y_hat_test"].shape == (n - split, Y.shape[1])
        assert result["Y_hat_train"].shape == (split, Y.shape[1])


# ---------------------------------------------------------------------------
# TestTrainSAEAlternating
# ---------------------------------------------------------------------------

class TestTrainSAEAlternating:
    def test_loss_decreases(self, structured_data):
        X, Y = structured_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_alternating(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=12, l1_penalty=1e-4, regression_type="ols",
            sae_lr=1e-2, sae_steps_per_cycle=20, num_cycles=10,
            device="cpu", log_every=1,
        )
        log = result["training_log"]
        assert log[-1]["total_loss"] < log[0]["total_loss"]

    def test_ols_regression(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_alternating(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, regression_type="ols",
            sae_lr=1e-3, sae_steps_per_cycle=5, num_cycles=3,
            device="cpu",
        )
        assert "regressor_weight" in result
        assert result["regressor_weight"].shape == (10, Y.shape[1])

    def test_ridge_regression(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_alternating(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, regression_type="ridge",
            sae_lr=1e-3, sae_steps_per_cycle=5, num_cycles=3,
            device="cpu",
        )
        assert "regressor_weight" in result
        assert result["regressor_weight"].shape == (10, Y.shape[1])

    def test_output_keys(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_alternating(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, regression_type="ols",
            sae_lr=1e-3, sae_steps_per_cycle=5, num_cycles=3,
            device="cpu",
        )
        for key in ["encoder_state_dict", "regressor_weight", "regressor_bias",
                     "metrics", "training_log", "Y_hat_test", "Y_hat_train"]:
            assert key in result

    def test_training_log_format(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_alternating(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, regression_type="ols",
            sae_lr=1e-3, sae_steps_per_cycle=5, num_cycles=4,
            device="cpu", log_every=1,
        )
        log = result["training_log"]
        assert len(log) == 4
        for entry in log:
            for key in ["cycle", "mse_loss", "l1_loss", "total_loss", "test_r2", "test_corr"]:
                assert key in entry

    def test_mini_batch(self, small_data):
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_alternating(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, regression_type="ols",
            sae_lr=1e-3, sae_steps_per_cycle=5, num_cycles=2,
            batch_size=16, device="cpu",
        )
        assert len(result["training_log"]) == 2


# ---------------------------------------------------------------------------
# TestSAEConfig
# ---------------------------------------------------------------------------

class TestSAEConfig:
    def test_roundtrip_yaml(self, tmp_path):
        from utils.config import SAEConfig
        from utils.config_utils import load_config

        config_dict = {
            "encoding": {"model": "gpt2xl", "layer": 24, "offset": 0},
            "k": 5,
            "hidden_dim": 20,
            "l1_penalty": 0.01,
            "training_mode": "alternating",
            "regression_type": "ridge",
            "mode": "cross_subject",
        }
        config_path = str(tmp_path / "sae_test.yml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_path, {}, SAEConfig)
        assert config.k == 5
        assert config.hidden_dim == 20
        assert config.l1_penalty == 0.01
        assert config.training_mode == "alternating"
        assert config.regression_type == "ridge"
        assert config.encoding.layer == 24

    def test_cli_overrides(self, tmp_path):
        from utils.config import SAEConfig
        from utils.config_utils import load_config

        config_dict = {"k": 5, "hidden_dim": 20}
        config_path = str(tmp_path / "sae_test.yml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        overrides = {"hidden_dim": 100, "encoding.layer": 48}
        config = load_config(config_path, overrides, SAEConfig)
        assert config.hidden_dim == 100
        assert config.encoding.layer == 48


# ---------------------------------------------------------------------------
# TestPlotSAETrainingCurves
# ---------------------------------------------------------------------------

class TestPlotSAETrainingCurves:
    def test_saves_png(self, tmp_path, structured_data):
        from utils.plotting import plot_sae_training_curves
        X, Y = structured_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=12, l1_penalty=1e-3, lr=1e-2,
            num_epochs=20, device="cpu", log_every=5,
        )
        save_path = str(tmp_path / "curves.png")
        plot_sae_training_curves(result["training_log"], save_path=save_path)
        assert os.path.exists(save_path)

    def test_handles_all_logged(self, tmp_path, small_data):
        from utils.plotting import plot_sae_training_curves
        X, Y = small_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=10, l1_penalty=1e-3, lr=1e-3,
            num_epochs=5, device="cpu", log_every=1,
        )
        save_path = str(tmp_path / "curves_all.png")
        plot_sae_training_curves(result["training_log"], save_path=save_path)
        assert os.path.exists(save_path)


# ---------------------------------------------------------------------------
# TestSAEEncoderInit
# ---------------------------------------------------------------------------

class TestSAEEncoderInit:
    def test_bias_initialized_to_zero(self):
        encoder = SAEEncoder(input_dim=4, hidden_dim=10)
        assert torch.allclose(encoder.linear.bias, torch.zeros(10))

    def test_weights_nonzero(self):
        encoder = SAEEncoder(input_dim=4, hidden_dim=10)
        assert not torch.allclose(encoder.linear.weight, torch.zeros(10, 4))


# ---------------------------------------------------------------------------
# TestResampleDeadNeurons
# ---------------------------------------------------------------------------

class TestResampleDeadNeurons:
    def test_resamples_dead_neurons(self):
        torch.manual_seed(42)
        encoder = SAEEncoder(input_dim=4, hidden_dim=10)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # Force 3 neurons to be dead by zeroing their weights and setting large negative bias
        with torch.no_grad():
            encoder.linear.weight[0:3] = 0.0
            encoder.linear.bias[0:3] = -100.0

        X = torch.randn(50, 4)
        losses = torch.rand(50)
        dead_mask = torch.zeros(10, dtype=torch.bool)
        dead_mask[0:3] = True

        n = _resample_dead_neurons(encoder, optimizer, X, losses, dead_mask)
        assert n == 3

        # Verify dead neurons now have nonzero weights and zero bias
        assert encoder.linear.weight[0:3].norm() > 0
        assert torch.allclose(encoder.linear.bias[0:3], torch.zeros(3))

    def test_no_dead_neurons_is_noop(self):
        encoder = SAEEncoder(input_dim=4, hidden_dim=10)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        X = torch.randn(50, 4)
        losses = torch.rand(50)
        dead_mask = torch.zeros(10, dtype=torch.bool)

        weight_before = encoder.linear.weight.clone()
        n = _resample_dead_neurons(encoder, optimizer, X, losses, dead_mask)
        assert n == 0
        assert torch.allclose(encoder.linear.weight, weight_before)

    def test_resampled_norm_is_scaled(self):
        torch.manual_seed(42)
        encoder = SAEEncoder(input_dim=4, hidden_dim=10)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # Make one neuron dead
        with torch.no_grad():
            encoder.linear.weight[0] = 0.0
            encoder.linear.bias[0] = -100.0

        X = torch.randn(50, 4)
        losses = torch.rand(50)
        dead_mask = torch.zeros(10, dtype=torch.bool)
        dead_mask[0] = True

        alive_avg_norm = encoder.linear.weight[1:].norm(dim=1).mean().item()
        _resample_dead_neurons(encoder, optimizer, X, losses, dead_mask)

        resampled_norm = encoder.linear.weight[0].norm().item()
        expected_norm = 0.2 * alive_avg_norm
        assert abs(resampled_norm - expected_norm) < 1e-5


# ---------------------------------------------------------------------------
# TestResamplingIntegration
# ---------------------------------------------------------------------------

class TestResamplingIntegration:
    def test_end_to_end_with_resampling(self, structured_data):
        X, Y = structured_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=12, l1_penalty=1e-4, lr=1e-2,
            num_epochs=20, device="cpu", log_every=1,
            resample_interval=5,
        )
        log = result["training_log"]
        assert len(log) == 20
        # n_resampled key should be present in every log entry
        assert all("n_resampled" in entry for entry in log)

    def test_alternating_with_resampling(self, structured_data):
        X, Y = structured_data
        n = X.shape[0]
        split = n // 2
        result = train_sae_alternating(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=12, l1_penalty=1e-4, regression_type="ols",
            sae_lr=1e-2, sae_steps_per_cycle=10, num_cycles=6,
            device="cpu", log_every=1,
            resample_interval=2,
        )
        log = result["training_log"]
        assert len(log) == 6
        assert all("n_resampled" in entry for entry in log)
