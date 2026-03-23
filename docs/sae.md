# SAE Pipeline

## What this does

We hypothesize that the RRR-learned k-dimensional subspace encodes more than k features via **superposition** — multiple sparse features packed into fewer dimensions. The SAE expands this subspace into a wider, overcomplete hidden layer and uses an L1 penalty to recover individual sparse features. Crucially, this is **not a reconstruction SAE** — there is no decoder. The SAE encoder feeds a linear regression head that predicts neural targets directly, so features are defined by task utility, not reconstruction fidelity.

## Architecture

```
X_raw (n, embed_dim)
  → @ A (embed_dim, k)        # RRR projection, loaded from prior stage
  → StandardScaler             # zero-center + unit-variance
  → Linear(k, hidden_dim) + ReLU  # SAEEncoder — the overcomplete expansion
  → Linear(hidden_dim, n_targets)  # regression head (end_to_end) or OLS/Ridge solve (alternating)
```

## Key files

| File | What to read for |
|---|---|
| `utils/sae.py` | Model classes, both training loops, resampling, weighted L1 |
| `scripts/run_sae.py` | Entry point, fold/unit orchestration, metric assembly |
| `utils/config.py` → `SAEConfig` | All tunable parameters and defaults |
| `configs/sae_default.yml` | Default config values |
| `tests/test_sae.py` | Synthetic-data tests for every component |

## Two training modes

**`end_to_end`** — SAE encoder + regression head trained jointly via Adam. Simpler, single optimizer. Good default.

**`alternating`** — Each cycle: (1) train encoder with frozen regression weights for N gradient steps, (2) re-solve regression optimally (OLS or Ridge) on current activations. Cycle 0 initializes regression on the random encoder's output so no unit starts with zero regression weights.

## Initialization & dead feature prevention

Three mechanisms work together:

1. **Zero-bias init** — Encoder bias is initialized to zero (`nn.init.zeros_`). With Kaiming-uniform weights on zero-centered inputs, roughly half of inputs activate each neuron at init — no neuron is born dead.

2. **Dead feature resampling** (`resample_interval` > 0) — At fixed intervals, neurons with zero activation on the full training set are reinitialized: encoder weight set to a high-loss input (normalized, scaled to 0.2x alive neuron average norm), bias reset to zero, Adam state cleared. Follows Anthropic (2023) "Towards Monosemanticity."

3. **Decoder-norm-weighted L1** (`weight_l1_by_decoder_norm: true`) — L1 becomes `mean(|h_i| * ||W_reg[i, :]||)`, preventing the model from shrinking activations and inflating regression weights to circumvent the sparsity penalty. Follows Anthropic (2024) "Scaling Monosemanticity." Requires re-tuning `l1_penalty` since the scale changes.

## Metrics to watch

- **L0 sparsity**: average number of active features per sample (lower = sparser)
- **dead_neurons**: fraction of features that never activate on test set
- **r2 / correlation**: prediction quality vs neural targets
- **recovery_r2**: after partialing out SAE predictions, how much can a full ridge model recover? Low = SAE captured most of the signal
