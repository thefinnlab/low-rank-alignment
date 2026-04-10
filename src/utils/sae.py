"""Sparse autoencoder model and training utilities for SAE-based encoding analysis."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.tensor_utils import to_numpy, to_torch
from himalaya.scoring import correlation_score, r2_score


class SAEEncoder(nn.Module):
    """Single hidden-layer sparse encoder.

    Maps k-dimensional RRR-projected embeddings to a wider hidden representation
    via Linear + bias + ReLU.  Not a traditional autoencoder — there is no decoder;
    the downstream regression head serves that role.

    Args:
        input_dim: Dimensionality of the RRR-projected input (k).
        hidden_dim: Width of the overcomplete hidden layer (must be > input_dim).
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        nn.init.zeros_(self.linear.bias)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (batch, hidden_dim) non-negative activations."""
        return self.relu(self.linear(x))


class SAERegressorEndToEnd(nn.Module):
    """SAE encoder + linear regression head, trained jointly.

    Forward returns ``(predictions, activations)`` so the caller can compute
    both MSE loss on predictions and L1 penalty on activations.

    Args:
        input_dim: Dimensionality of the RRR-projected input (k).
        hidden_dim: Width of the SAE hidden layer.
        output_dim: Number of regression targets (n_targets).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = SAEEncoder(input_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.regressor(h), h


def compute_sae_metrics(activations: torch.Tensor) -> dict:
    """Compute SAE-specific metrics from hidden-layer activations.

    Args:
        activations: (n_samples, hidden_dim) non-negative tensor.

    Returns:
        Dict with scalar metrics and per-neuron activation_frequency array.
    """
    active = activations > 0
    l0_sparsity = float(active.float().sum(dim=1).mean())
    dead_neurons = float((activations.sum(dim=0) == 0).float().mean())
    nonzero_vals = activations[active]
    mean_activation = float(nonzero_vals.mean()) if nonzero_vals.numel() > 0 else 0.0
    l1_loss = float(activations.abs().mean())
    activation_frequency = to_numpy(active.float().mean(dim=0))

    return {
        "l0_sparsity": l0_sparsity,
        "dead_neurons": dead_neurons,
        "mean_activation": mean_activation,
        "l1_loss": l1_loss,
        "activation_frequency": activation_frequency,
    }


def _make_dataloader(X: torch.Tensor, Y: torch.Tensor, batch_size: int | None):
    """Return a DataLoader or None (full-batch) depending on batch_size."""
    if batch_size is None:
        return None
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)


def _compute_l1(
    acts: torch.Tensor, reg_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute L1 penalty on activations, optionally weighted by regression column norms.

    When reg_weights is provided, computes the decoder-norm-weighted L1 from
    Anthropic (2024): mean(|h_i| * ||W_reg[i, :]||). This prevents the model
    from shrinking activations and inflating regression weights to cheat the
    L1 penalty.

    Args:
        acts: (batch, hidden_dim) non-negative activations.
        reg_weights: Regression weight matrix. Shape (hidden_dim, n_targets) or
            (n_targets, hidden_dim) — the per-hidden-unit norm is computed along
            the targets axis. None for unweighted L1.

    Returns:
        Scalar L1 loss.
    """
    if reg_weights is None:
        return acts.abs().mean()

    # Compute per-hidden-unit norm of regression weights
    if reg_weights.shape[0] == acts.shape[1]:
        # (hidden_dim, n_targets) — norm along targets axis
        col_norms = reg_weights.norm(dim=1)
    else:
        # (n_targets, hidden_dim) — norm along targets axis
        col_norms = reg_weights.norm(dim=0)

    return (acts.abs() * col_norms.unsqueeze(0)).mean()


def _get_l1_coeff(
    l1_penalty: float, step: int, warmup_start: int, warmup_epochs: int
) -> float:
    """Compute the effective L1 coefficient at a given training step.

    Returns 0 before warmup_start, linearly ramps to l1_penalty over
    warmup_epochs steps, then stays at l1_penalty. When warmup_start=0
    and warmup_epochs=0 (defaults), returns l1_penalty immediately.
    """
    if warmup_epochs <= 0 or step >= warmup_start + warmup_epochs:
        return l1_penalty
    if step < warmup_start:
        return 0.0
    return l1_penalty * (step - warmup_start) / warmup_epochs


def _resample_dead_neurons(
    encoder: SAEEncoder,
    optimizer: torch.optim.Adam,
    X: torch.Tensor,
    losses_per_sample: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
) -> int:
    """Resample dead encoder neurons following Anthropic (2023).

    For each dead neuron:
      - Sample a training input weighted by squared per-sample loss.
      - Set decoder (encoder weight row) to that input, normalized to unit norm.
      - Scale encoder weight to 0.2x the average norm of alive neurons.
      - Set encoder bias to zero.
      - Reset Adam optimizer state for modified parameters.

    Args:
        encoder: The SAEEncoder module.
        optimizer: Adam optimizer for the encoder.
        X: (n_samples, input_dim) training inputs.
        losses_per_sample: (n_samples,) per-sample loss values.
        dead_neuron_mask: (hidden_dim,) boolean tensor, True for dead neurons.

    Returns:
        Number of neurons resampled.
    """
    dead_indices = dead_neuron_mask.nonzero(as_tuple=True)[0]
    n_dead = len(dead_indices)
    if n_dead == 0:
        return 0

    device = X.device
    weight = encoder.linear.weight  # (hidden_dim, input_dim)
    bias = encoder.linear.bias  # (hidden_dim,)

    # Compute alive neuron average norm for scaling
    alive_mask = ~dead_neuron_mask
    if alive_mask.any():
        alive_norms = weight[alive_mask].norm(dim=1)
        avg_alive_norm = float(alive_norms.mean())
    else:
        avg_alive_norm = float(weight.norm(dim=1).mean())

    # Sampling probabilities proportional to squared loss
    probs = losses_per_sample**2
    probs = probs / probs.sum()

    # Sample one input per dead neuron
    sample_indices = torch.multinomial(probs, n_dead, replacement=True)

    with torch.no_grad():
        for i, dead_idx in enumerate(dead_indices):
            sampled_input = X[sample_indices[i]]  # (input_dim,)
            norm = sampled_input.norm()
            if norm > 0:
                direction = sampled_input / norm
            else:
                direction = torch.randn_like(sampled_input)
                direction = direction / direction.norm()

            # Set encoder weight row to normalized input scaled to 0.2x alive avg
            weight[dead_idx] = direction * (0.2 * avg_alive_norm)
            bias[dead_idx] = 0.0

    # Reset Adam state for encoder parameters
    for param in [weight, bias]:
        state = optimizer.state.get(param, {})
        if state:
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            if "step" in state:
                state["step"] = (
                    torch.tensor(0.0) if isinstance(state["step"], torch.Tensor) else 0
                )

    return n_dead


def train_sae_end_to_end(
    X_proj_train: np.ndarray,
    Y_train: np.ndarray,
    X_proj_test: np.ndarray,
    Y_test: np.ndarray,
    hidden_dim: int,
    l1_penalty: float,
    lr: float,
    num_epochs: int,
    batch_size: int | None = None,
    device: str | None = None,
    log_every: int = 10,
    resample_interval: int = 0,
    weight_l1_by_decoder_norm: bool = False,
    l1_warmup_start: int = 0,
    l1_warmup_epochs: int = 0,
    log_callback=None,
) -> dict:
    """Train SAE + regression end-to-end via backpropagation.

    Loss = MSE(Y_pred, Y_true) + l1_penalty * L1(activations).

    When weight_l1_by_decoder_norm is True, the L1 term is weighted by
    regression column norms (Anthropic 2024): mean(|h_i| * ||W_reg[i, :]||).
    This prevents the model from shrinking activations while inflating
    regression weights to circumvent the sparsity penalty.

    X_proj inputs are standardised internally (fit on train, transform test).

    Args:
        X_proj_train: (n_train, k) RRR-projected embeddings.
        Y_train: (n_train, n_targets) scaled neural targets.
        X_proj_test: (n_test, k) RRR-projected embeddings.
        Y_test: (n_test, n_targets) scaled neural targets.
        hidden_dim: SAE hidden layer width.
        l1_penalty: L1 coefficient on activations.
        lr: Adam learning rate.
        num_epochs: Number of training epochs.
        batch_size: Mini-batch size; None for full-batch.
        device: Torch device string; auto-selects if None.
        log_every: Compute and display test metrics every N epochs.
        resample_interval: Resample dead neurons every N epochs (0 = disabled).
        weight_l1_by_decoder_norm: Weight L1 by regression column norms.
        l1_warmup_start: Epoch at which L1 penalty begins ramping (0 = immediate).
        l1_warmup_epochs: Epochs over which L1 linearly ramps to target (0 = no ramp).
        log_callback: Optional callable(Y_hat_train, Y_hat_test) -> dict of extra
            metrics to merge into the training log at each log step.

    Returns:
        Dict with keys:
            encoder_state_dict: SAE encoder weights.
            regressor_weight: (hidden_dim, n_targets) regression coefficients.
            regressor_bias: (n_targets,) regression bias.
            metrics: Dict of evaluation metrics on test set.
            training_log: List of per-epoch metric dicts.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_proj_train)
    X_test_s = scaler.transform(X_proj_test)

    input_dim = X_train_s.shape[1]
    n_targets = Y_train.shape[1]

    X_tr = to_torch(X_train_s, dtype=torch.float32, device=device)
    Y_tr = to_torch(Y_train, dtype=torch.float32, device=device)
    X_te = to_torch(X_test_s, dtype=torch.float32, device=device)
    Y_te = to_torch(Y_test, dtype=torch.float32, device=device)

    model = SAERegressorEndToEnd(input_dim, hidden_dim, n_targets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_fn = nn.MSELoss()

    dataloader = _make_dataloader(X_tr, Y_tr, batch_size)
    training_log = []

    pbar = tqdm(range(num_epochs), desc="  end_to_end", unit="ep", leave=True)
    for epoch in pbar:
        model.train()
        effective_l1 = _get_l1_coeff(
            l1_penalty, epoch, l1_warmup_start, l1_warmup_epochs
        )
        reg_w = model.regressor.weight.detach() if weight_l1_by_decoder_norm else None
        if dataloader is not None:
            epoch_mse, epoch_l1, n_seen = 0.0, 0.0, 0
            for X_batch, Y_batch in dataloader:
                optimizer.zero_grad()
                preds, acts = model(X_batch)
                mse = mse_fn(preds, Y_batch)
                l1 = _compute_l1(acts, reg_w)
                loss = mse + effective_l1 * l1
                loss.backward()
                optimizer.step()
                bs = X_batch.shape[0]
                epoch_mse += mse.item() * bs
                epoch_l1 += l1.item() * bs
                n_seen += bs
            epoch_mse /= n_seen
            epoch_l1 /= n_seen
        else:
            optimizer.zero_grad()
            preds, acts = model(X_tr)
            mse = mse_fn(preds, Y_tr)
            l1 = _compute_l1(acts, reg_w)
            loss = mse + effective_l1 * l1
            loss.backward()
            optimizer.step()
            epoch_mse = mse.item()
            epoch_l1 = l1.item()

        total_loss = epoch_mse + effective_l1 * epoch_l1

        # Dead feature resampling
        n_resampled = 0
        if (
            resample_interval > 0
            and (epoch + 1) % resample_interval == 0
            and epoch < num_epochs - 1
        ):
            model.eval()
            with torch.no_grad():
                preds_full, acts_full = model(X_tr)
                dead_mask = acts_full.sum(dim=0) == 0
                if dead_mask.any():
                    per_sample_loss = (preds_full - Y_tr).pow(2).mean(dim=1)
                    n_resampled = _resample_dead_neurons(
                        model.encoder,
                        optimizer,
                        X_tr,
                        per_sample_loss,
                        dead_mask,
                    )

        # Compute test metrics at log_every intervals and on the last epoch
        is_log_step = (epoch % log_every == 0) or (epoch == num_epochs - 1)
        if is_log_step:
            model.eval()
            with torch.no_grad():
                preds_te, acts_te = model(X_te)
                test_r2 = float(r2_score(to_numpy(Y_te), to_numpy(preds_te)).mean())
                test_corr = float(
                    correlation_score(to_numpy(Y_te), to_numpy(preds_te)).mean()
                )
                step_sae_metrics = compute_sae_metrics(acts_te)
                l0 = step_sae_metrics["l0_sparsity"]
            if log_callback is not None:
                with torch.no_grad():
                    preds_tr_cb, _ = model(X_tr)
                extra = log_callback(to_numpy(preds_tr_cb), to_numpy(preds_te))
            else:
                extra = {}
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                r2=f"{test_r2:.4f}",
                corr=f"{test_corr:.4f}",
                L0=f"{l0:.1f}",
            )
        else:
            test_r2 = None
            test_corr = None
            step_sae_metrics = None
            extra = {}
            pbar.set_postfix(loss=f"{total_loss:.4f}")

        log_entry = {
            "epoch": epoch,
            "mse_loss": epoch_mse,
            "l1_loss": epoch_l1,
            "effective_l1_coeff": effective_l1,
            "total_loss": total_loss,
            "test_r2": test_r2,
            "test_corr": test_corr,
            "n_resampled": n_resampled,
            "l0_sparsity": (
                step_sae_metrics["l0_sparsity"] if step_sae_metrics else None
            ),
            "dead_neurons": (
                step_sae_metrics["dead_neurons"] if step_sae_metrics else None
            ),
            "mean_activation": (
                step_sae_metrics["mean_activation"] if step_sae_metrics else None
            ),
            **extra,
        }
        training_log.append(log_entry)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        preds_te, acts_te = model(X_te)
        preds_tr, acts_tr = model(X_tr)

    sae_metrics = compute_sae_metrics(acts_te)
    test_r2 = float(r2_score(to_numpy(Y_te), to_numpy(preds_te)).mean())
    test_corr = float(correlation_score(to_numpy(Y_te), to_numpy(preds_te)).mean())

    metrics = {
        "r2": test_r2,
        "correlation": test_corr,
        **sae_metrics,
    }

    return {
        "encoder_state_dict": {
            k: v.cpu() for k, v in model.encoder.state_dict().items()
        },
        "regressor_weight": to_numpy(model.regressor.weight.data),
        "regressor_bias": to_numpy(model.regressor.bias.data),
        "metrics": metrics,
        "training_log": training_log,
        "Y_hat_test": to_numpy(preds_te),
        "Y_hat_train": to_numpy(preds_tr),
    }


def train_sae_alternating(
    X_proj_train: np.ndarray,
    Y_train: np.ndarray,
    X_proj_test: np.ndarray,
    Y_test: np.ndarray,
    hidden_dim: int,
    l1_penalty: float,
    regression_type: str,
    sae_lr: float,
    sae_steps_per_cycle: int,
    num_cycles: int,
    batch_size: int | None = None,
    device: str | None = None,
    log_every: int = 1,
    resample_interval: int = 0,
    weight_l1_by_decoder_norm: bool = False,
    l1_warmup_start: int = 0,
    l1_warmup_epochs: int = 0,
    regression_momentum: float = 1.0,
    input_jitter_std: float = 0.0,
    log_callback=None,
    ridge_alphas: np.ndarray | None = None,
) -> dict:
    """Train SAE with alternating optimisation.

    Each cycle: (1) train SAE encoder for ``sae_steps_per_cycle`` gradient steps
    with frozen regression weights, (2) re-solve regression optimally on the
    SAE's current activations (OLS or Ridge).

    Cycle 0 initialises regression by solving on the raw (pre-SAE) projected
    inputs to give meaningful gradients from the start.

    When weight_l1_by_decoder_norm is True, the L1 term is weighted by
    regression column norms (Anthropic 2024): mean(|h_i| * ||W_reg[i, :]||).

    X_proj inputs are standardised internally (fit on train, transform test).

    Args:
        X_proj_train: (n_train, k) RRR-projected embeddings.
        Y_train: (n_train, n_targets) scaled neural targets.
        X_proj_test: (n_test, k) RRR-projected embeddings.
        Y_test: (n_test, n_targets) scaled neural targets.
        hidden_dim: SAE hidden layer width.
        l1_penalty: L1 coefficient on activations.
        regression_type: "ols" or "ridge".
        sae_lr: Adam learning rate for the SAE encoder.
        sae_steps_per_cycle: Gradient steps per SAE training phase.
        num_cycles: Number of alternating cycles.
        batch_size: Mini-batch size; None for full-batch.
        device: Torch device string; auto-selects if None.
        log_every: Compute and display test metrics every N cycles.
        resample_interval: Resample dead neurons every N cycles (0 = disabled).
        weight_l1_by_decoder_norm: Weight L1 by regression column norms.
        l1_warmup_start: Cycle at which L1 penalty begins ramping (0 = immediate).
        l1_warmup_epochs: Cycles over which L1 linearly ramps to target (0 = no ramp).
        regression_momentum: Proximal damping for regression updates.
            W_next = (1 - η) * W_old + η * W_optimal. 1.0 = no damping (default).
        input_jitter_std: Std of Gaussian noise added to X_proj during SAE
            training. 0.0 = disabled (default).
        log_callback: Optional callable(Y_hat_train, Y_hat_test) -> dict of extra
            metrics to merge into the training log at each log step.
        ridge_alphas: Optional array of ridge alpha values for RidgeCV. Overrides
            the default grid (logspace(1, 10)) which may over-regularise when the
            number of features is small.

    Returns:
        Same structure as train_sae_end_to_end.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_proj_train)
    X_test_s = scaler.transform(X_proj_test)

    input_dim = X_train_s.shape[1]
    n_targets = Y_train.shape[1]

    X_tr = to_torch(X_train_s, dtype=torch.float32, device=device)
    Y_tr = to_torch(Y_train, dtype=torch.float32, device=device)
    X_te = to_torch(X_test_s, dtype=torch.float32, device=device)
    Y_te = to_torch(Y_test, dtype=torch.float32, device=device)

    encoder = SAEEncoder(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=sae_lr)
    mse_fn = nn.MSELoss()

    # Cycle 0 init: run X through the randomly-initialised encoder and solve
    # regression on those activations.  This gives every hidden unit non-zero
    # regression weights from the start so no unit is born dead.
    encoder.eval()
    with torch.no_grad():
        H_init = encoder(X_tr)
    reg_W, reg_b = solve_regression(
        H_init, Y_tr, regression_type, device, alphas=ridge_alphas
    )

    training_log = []

    pbar = tqdm(range(num_cycles), desc="  alternating", unit="cyc", leave=True)
    for cycle in pbar:
        # Phase 1: Train SAE encoder with frozen regression weights
        encoder.train()
        effective_l1 = _get_l1_coeff(
            l1_penalty, cycle, l1_warmup_start, l1_warmup_epochs
        )
        frozen_W = reg_W.detach()
        frozen_b = reg_b.detach()
        l1_weights = frozen_W if weight_l1_by_decoder_norm else None

        dataloader = _make_dataloader(X_tr, Y_tr, batch_size)

        for step in range(sae_steps_per_cycle):
            if dataloader is not None:
                for X_batch, Y_batch in dataloader:
                    optimizer.zero_grad()
                    X_in = (
                        (X_batch + torch.randn_like(X_batch) * input_jitter_std)
                        if input_jitter_std > 0
                        else X_batch
                    )
                    h = encoder(X_in)
                    preds = h @ frozen_W + frozen_b
                    mse = mse_fn(preds, Y_batch)
                    l1 = _compute_l1(h, l1_weights)
                    loss = mse + effective_l1 * l1
                    loss.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                X_in = (
                    (X_tr + torch.randn_like(X_tr) * input_jitter_std)
                    if input_jitter_std > 0
                    else X_tr
                )
                h = encoder(X_in)
                preds = h @ frozen_W + frozen_b
                mse = mse_fn(preds, Y_tr)
                l1 = _compute_l1(h, l1_weights)
                loss = mse + effective_l1 * l1
                loss.backward()
                optimizer.step()

        # Phase 2: Re-solve regression optimally on SAE activations
        encoder.eval()
        with torch.no_grad():
            H_train = encoder(X_tr)

        opt_W, opt_b = solve_regression(
            H_train, Y_tr, regression_type, device, alphas=ridge_alphas
        )
        if regression_momentum < 1.0:
            eta = regression_momentum
            reg_W = (1 - eta) * reg_W + eta * opt_W
            reg_b = (1 - eta) * reg_b + eta * opt_b
        else:
            reg_W, reg_b = opt_W, opt_b

        # Dead feature resampling
        n_resampled = 0
        if (
            resample_interval > 0
            and (cycle + 1) % resample_interval == 0
            and cycle < num_cycles - 1
        ):
            with torch.no_grad():
                dead_mask = H_train.sum(dim=0) == 0
                if dead_mask.any():
                    preds_tr_tmp = H_train @ reg_W + reg_b
                    per_sample_loss = (preds_tr_tmp - Y_tr).pow(2).mean(dim=1)
                    n_resampled = _resample_dead_neurons(
                        encoder,
                        optimizer,
                        X_tr,
                        per_sample_loss,
                        dead_mask,
                    )

        # Log cycle metrics
        is_log_step = (cycle % log_every == 0) or (cycle == num_cycles - 1)
        with torch.no_grad():
            preds_tr = H_train @ reg_W + reg_b
            train_mse = float(mse_fn(preds_tr, Y_tr))
            train_l1 = float(H_train.abs().mean())
            total_loss = train_mse + effective_l1 * train_l1

        if is_log_step:
            with torch.no_grad():
                H_test = encoder(X_te)
                preds_te = H_test @ reg_W + reg_b
                test_r2 = float(r2_score(to_numpy(Y_te), to_numpy(preds_te)).mean())
                test_corr = float(
                    correlation_score(to_numpy(Y_te), to_numpy(preds_te)).mean()
                )
                step_sae_metrics = compute_sae_metrics(H_test)
                l0 = step_sae_metrics["l0_sparsity"]
            if log_callback is not None:
                extra = log_callback(to_numpy(preds_tr), to_numpy(preds_te))
            else:
                extra = {}
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                r2=f"{test_r2:.4f}",
                corr=f"{test_corr:.4f}",
                L0=f"{l0:.1f}",
            )
        else:
            test_r2 = None
            test_corr = None
            step_sae_metrics = None
            extra = {}
            pbar.set_postfix(loss=f"{total_loss:.4f}")

        log_entry = {
            "cycle": cycle,
            "mse_loss": train_mse,
            "l1_loss": train_l1,
            "effective_l1_coeff": effective_l1,
            "total_loss": total_loss,
            "test_r2": test_r2,
            "test_corr": test_corr,
            "n_resampled": n_resampled,
            "l0_sparsity": (
                step_sae_metrics["l0_sparsity"] if step_sae_metrics else None
            ),
            "dead_neurons": (
                step_sae_metrics["dead_neurons"] if step_sae_metrics else None
            ),
            "mean_activation": (
                step_sae_metrics["mean_activation"] if step_sae_metrics else None
            ),
            **extra,
        }
        training_log.append(log_entry)

    # Final evaluation
    encoder.eval()
    with torch.no_grad():
        H_test = encoder(X_te)
        H_train = encoder(X_tr)
        preds_te = H_test @ reg_W + reg_b
        preds_tr = H_train @ reg_W + reg_b

    sae_metrics = compute_sae_metrics(H_test)
    test_r2 = float(r2_score(to_numpy(Y_te), to_numpy(preds_te)).mean())
    test_corr = float(correlation_score(to_numpy(Y_te), to_numpy(preds_te)).mean())

    metrics = {
        "r2": test_r2,
        "correlation": test_corr,
        **sae_metrics,
    }

    return {
        "encoder_state_dict": {k: v.cpu() for k, v in encoder.state_dict().items()},
        "regressor_weight": to_numpy(reg_W),
        "regressor_bias": to_numpy(reg_b),
        "metrics": metrics,
        "training_log": training_log,
        "Y_hat_test": to_numpy(preds_te),
        "Y_hat_train": to_numpy(preds_tr),
    }


def solve_regression(
    H: torch.Tensor,
    Y: torch.Tensor,
    regression_type: str,
    device: str,
    standardize: bool = True,
    alphas: np.ndarray | None = None,
    return_alphas: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Solve regression (OLS or Ridge) and return (W, b) tensors.

    Args:
        standardize: When True (default), apply StandardScaler before ridge.
            Set to False when features already have a meaningful variance
            structure (e.g. SVD-derived projections) that should not be
            equalised.
        alphas: Optional array of ridge alpha values for RidgeCV. When
            provided, overrides the default grid from get_ridge_model_constructor.
        return_alphas: When True and regression_type is ``"ridge"``, return a
            3-tuple ``(W, b, best_alphas)`` where *best_alphas* is a 1-D
            numpy array of per-target selected alphas. Default False preserves
            the original 2-tuple return.

    Returns:
        reg_W: (hidden_dim, n_targets) weight tensor.
        reg_b: (n_targets,) bias tensor.
        best_alphas: (n_targets,) numpy array (only when return_alphas=True
            and regression_type="ridge").
    """
    H_np = to_numpy(H)
    Y_np = to_numpy(Y)

    if regression_type == "ridge":
        from src.utils.encoding import get_ridge_model_constructor

        if standardize:
            model = get_ridge_model_constructor(alphas=alphas)(0)
            model.fit(H_np, Y_np)
            # Absorb StandardScaler into W, b so that H @ W + b ≈ Y.
            # model.predict(H) = ((H - mean) / scale) @ coef_raw.T + intercept
            # = H @ diag(1/scale) @ coef_raw.T + intercept - (mean/scale) @ coef_raw.T
            coef_raw = to_numpy(
                model[-1].coef_
            )  # may be (n_targets, n_features) or transposed
            n_features = H_np.shape[1]
            if coef_raw.shape != (Y_np.shape[1], n_features):
                coef_raw = coef_raw.T  # → (n_targets, n_features)
            sc_mean = model[0].mean_  # (n_features,)
            sc_scale = model[0].scale_  # (n_features,)
            intercept = to_numpy(model[-1].intercept_)  # (n_targets,)
            W = (coef_raw / sc_scale[np.newaxis, :]).T  # (n_features, n_targets)
            b = intercept - (sc_mean / sc_scale) @ coef_raw.T  # (n_targets,)
            best_alphas_out = (
                to_numpy(model[-1].best_alphas_) if return_alphas else None
            )
        else:
            # Ridge without StandardScaler — appropriate when feature variance
            # carries signal (e.g. SVD-ordered projections).
            ridge_model = get_ridge_model_constructor(alphas=alphas)(0)[
                -1
            ]  # RidgeCV only
            ridge_model.fit(H_np, Y_np)
            coef_raw = to_numpy(ridge_model.coef_)
            n_features = H_np.shape[1]
            if coef_raw.shape != (Y_np.shape[1], n_features):
                coef_raw = coef_raw.T  # → (n_targets, n_features)
            intercept = to_numpy(ridge_model.intercept_)
            W = coef_raw.T  # (n_features, n_targets)
            b = intercept  # (n_targets,)
            best_alphas_out = (
                to_numpy(ridge_model.best_alphas_) if return_alphas else None
            )
        result = (
            to_torch(W.astype(np.float32), device=device),
            to_torch(b.astype(np.float32), device=device),
        )
        if return_alphas:
            return result + (best_alphas_out,)
        return result
    else:
        # OLS: H @ W + b ≈ Y  →  [H, 1] @ [W; b] ≈ Y
        H_t = to_torch(H_np, dtype=torch.float32, device=device)
        Y_t = to_torch(Y_np, dtype=torch.float32, device=device)
        n = H_t.shape[0]
        d = H_t.shape[1]
        H_aug = torch.cat(
            [H_t, torch.ones(n, 1, dtype=torch.float32, device=device)], dim=1
        )
        gram = H_aug.T @ H_aug + 1e-6 * torch.eye(
            d + 1, dtype=torch.float32, device=device
        )
        sol = torch.linalg.solve(gram, H_aug.T @ Y_t)  # (d+1, n_targets)
        return sol[:d], sol[d]
