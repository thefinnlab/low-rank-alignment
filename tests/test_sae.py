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
            num_epochs=50, device="cpu",
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
            num_epochs=5, device="cpu",
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
            num_epochs=10, device="cpu",
        )
        log = result["training_log"]
        assert len(log) == 10
        for entry in log:
            for key in ["epoch", "mse_loss", "l1_loss", "total_loss", "test_r2", "test_corr"]:
                assert key in entry

    def test_l1_increases_sparsity(self, structured_data):
        X, Y = structured_data
        n = X.shape[0]
        split = n // 2

        result_low = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=12, l1_penalty=1e-5, lr=1e-2,
            num_epochs=100, device="cpu",
        )
        result_high = train_sae_end_to_end(
            X[:split], Y[:split], X[split:], Y[split:],
            hidden_dim=12, l1_penalty=1.0, lr=1e-2,
            num_epochs=100, device="cpu",
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
            device="cpu",
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
            device="cpu",
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
