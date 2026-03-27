"""
generate_null_distribution.py — Batchable phase-randomization null distribution generator.

Each invocation runs ``n_iterations`` permutations per subject and writes
results to a uniquely-named file inside ``{null_dir}/sub_{id}/``.  Multiple
invocations can run in parallel (e.g. as separate SLURM jobs) to accumulate
the full null distribution needed by compute_significance.py.

For each iteration the procedure is:
    1. Phase-randomize each electrode's continuous signal (preserving
       autocorrelation, destroying word–brain alignment).
    2. Re-run the full encoding procedure (ridge regression).
    3. Compute test-set Pearson correlations per (electrode, lag).
    4. Retain the max correlation across all lags for each electrode.
    5. Take the max across electrodes → single scalar.

To amortise the cost of ridge regression, multiple permutations are batched
together: their Y matrices are concatenated column-wise so a single model
predicts all permutations at once (ridge is independent per target).  The
``batch_size`` config parameter controls how many permutations share a model
(limited by RAM).

Writes:
  {null_dir}/sub_{id}/{batch_uuid}.npy   — shape (n_iterations,)
"""

import gc
import math
import os
import uuid

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    load_embeddings,
    apply_offset,
    train_encoding,
    get_all_predictions_and_targets_test_set,
    load_precomputed_epochs,
)
from utils.config import NullDistConfig, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import get_ridge_model_constructor
from utils.significance import phase_randomize_data, compute_observed_max_per_electrode

from himalaya.backend import get_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score


TOTAL_TARGET_CAPACITY = 5702400 * 30


def print_gpu_utilization():
    if torch.cuda.is_available():
        info = torch.cuda.mem_get_info()  # Returns (free, total) in bytes
        used = (info[1] - info[0]) / 1e9
        print(f"Current VRAM Usage: {used:.2f} / {info[1]/1e9:.2f} GB")


def torch_phase_randomize(raw_data, batch_size, rng_seed=None):
    """
    GPU-accelerated phase randomization for a batch of permutations.
    raw_data: (n_elec, n_samples)
    Returns: (batch_size, n_elec, n_samples)
    """
    device = raw_data.device
    n_elec, n_samples = raw_data.shape
    print(f"Total memory required for batch: {n_elec * n_samples}")

    # 1. FFT
    freq_data = torch.fft.rfft(raw_data, dim=-1)
    n_freq = freq_data.shape[-1]

    # 2. Generate random phases for the batch
    # We want different phases per permutation, but shared across electrodes
    # to preserve spatial correlation if desired, or unique per elec.
    # Here we do unique per (batch, elec) for maximum noise.
    if rng_seed:
        torch.manual_seed(rng_seed)
    phases = torch.rand((batch_size, n_elec, n_freq), device=device) * 2 * np.pi
    exp_phases = torch.complex(torch.cos(phases), torch.sin(phases))

    # 3. Apply phases
    # freq_data is (n_elec, n_freq), we broadcast to (batch, n_elec, n_freq)
    randomized_freq = freq_data.unsqueeze(0) * exp_phases

    # 4. Inverse FFT
    return torch.fft.irfft(randomized_freq, n=n_samples, dim=-1).float()


def _run_batched_permutations(
    raw_data,
    selection,
    epoch_starts,
    n_lags,
    aligned_embeddings,
    enc,
    model_constructor,
    rng,
    batch_size,
):
    """Generate ``batch_size`` phase-randomised permutations and train one model.

    The Y matrices for each permutation are concatenated column-wise so that a
    single ridge model predicts all permutations simultaneously (ridge is
    independent per target, so this is equivalent to training separate models
    but much faster due to batched linear algebra).

    Args:
        raw_data: (n_elecs, n_samples) float32 array at 32 Hz (precomputed).
        selection: word indices kept by epoching (precomputed).
        epoch_starts: start sample in raw_data for each kept epoch (precomputed).
        n_lags: samples per epoch window.

    Returns:
        np.ndarray of shape (batch_size,) — max-across-electrodes scalar per
        permutation.
    """

    # Move static data to GPU
    raw_data_gpu = torch.as_tensor(raw_data, device="cuda")
    X_gpu = torch.as_tensor(aligned_embeddings, device="cuda")

    # 1. Batched Randomization (Batch, Elecs, Samples)
    # This replaces the 'for _ in range(batch_size)' loop
    raw_perm_batch = torch_phase_randomize(raw_data_gpu, batch_size)

    # 2. Vectorized Epoching (Advanced Indexing)
    # Create a grid of indices: (n_epochs, n_lags)
    starts = torch.tensor(epoch_starts, device="cuda").unsqueeze(1)
    lags = torch.arange(n_lags, device="cuda").unsqueeze(0)
    idx_grid = starts + lags  # Shape: (n_words, n_lags)

    # Slice the randomized data for all batches at once
    # Result: (batch, n_elec, n_words, n_lags)
    data_epoched = raw_perm_batch[:, :, idx_grid]

    # 3. Reshape for Ridge: (n_words, batch * n_elec * n_lags)
    # We permute to get words as first dim to match X
    Y_concat = data_epoched.permute(2, 0, 1, 3).reshape(len(epoch_starts), -1)

    # 4. 2-Fold CV inside the function for speed
    # (Simple split for demonstration; use KFold if non-contiguous required)
    mid = len(Y_concat) // 2
    train_idx, test_idx = torch.arange(0, mid), torch.arange(mid, len(Y_concat))

    # Scale Y
    Y_mean = Y_concat[train_idx].mean(0)
    Y_std = Y_concat[train_idx].std(0) + 1e-6
    Y_train = (Y_concat[train_idx] - Y_mean) / Y_std
    Y_test = (Y_concat[test_idx] - Y_mean) / Y_std

    X_train, X_test = X_gpu[train_idx], X_gpu[test_idx]

    # 5. Himalaya RidgeCV (Fits all permutations/elecs simultaneously)
    model = RidgeCV(alphas=torch.logspace(1, 10, 13))
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # 6. Scoring (GPU Vectorized)
    # corrs shape: (batch * n_elec * n_lags)
    corrs = correlation_score(Y_test, Y_pred)

    # 7. Max-across-lags and Max-across-electrodes
    # Reshape back to (batch, n_elec, n_lags)
    corrs_reshaped = corrs.view(batch_size, -1, n_lags)

    # Max over lags (dim 2), then Max over electrodes (dim 1)
    null_maxes = corrs_reshaped.max(dim=2)[0].max(dim=1)[0]

    print_gpu_utilization()

    return null_maxes.cpu().numpy()


def main():
    print_gpu_utilization()
    args, overrides = parse_args("Phase-randomization null distribution generation")
    config: NullDistConfig = load_config(args.config, overrides, NullDistConfig)
    enc = config.encoding

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)
    null_dir = config.null_dir or os.path.join(output_dir, "null_dist")
    batch_size = max(1, config.batch_size)

    # Use GPU backend if available
    if torch.cuda.is_available():
        from himalaya.backend import set_backend

        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    # Seed: use provided seed or generate a random one for this batch
    if config.seed is not None:
        rng = np.random.default_rng(config.seed)
    else:
        rng = np.random.default_rng()

    batch_id = uuid.uuid4().hex[:12]
    print(f"Batch ID: {batch_id}")
    print(f"Null dir: {null_dir}")
    print(f"Iterations per subject: {config.n_iterations}")
    print(f"Batch size (permutations per model): {batch_size}")

    # Load embeddings once — shared across all subjects
    aligned_embeddings = load_embeddings(enc.data_root, enc.layer)[0].astype(np.float32)

    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None
    model_constructor = get_ridge_model_constructor(alphas=alphas)

    for bids_sub_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Subject {bids_sub_id}")
        print(f"{'='*60}")

        # Load precomputed 32 Hz cache once per subject
        cache = load_precomputed_epochs(bids_sub_id, enc.data_root)
        raw_data = cache["raw_data"].astype(np.float32)
        selection = cache["selection"]
        epoch_starts = cache["epoch_starts"]
        n_lags_sub = cache["n_lags"]
        n_elec = raw_data.shape[0]
        print(f"  Electrodes: {n_elec}")
        print(f"  Raw Data Shape: {raw_data.shape}")

        recommended_batch = TOTAL_TARGET_CAPACITY // (n_elec * raw_data.shape[1])
        print("Recommended batch size based on memory capacity:", recommended_batch)
        current_batch_size = max(1, min(recommended_batch, config.n_iterations))
        print(
            f"Adjusting batch size to {current_batch_size} (Max capacity: {TOTAL_TARGET_CAPACITY})"
        )
        n_batches = math.ceil(config.n_iterations / current_batch_size)
        null_max_values = []

        for b_idx in tqdm(range(n_batches), desc=f"  Sub {bids_sub_id} batches"):
            remaining = config.n_iterations - len(null_max_values)
            cur_batch = min(current_batch_size, remaining)

            batch_maxes = _run_batched_permutations(
                raw_data,
                selection,
                epoch_starts,
                n_lags_sub,
                aligned_embeddings,
                enc,
                model_constructor,
                rng,
                batch_size=cur_batch,
            )
            null_max_values.extend(batch_maxes.tolist())

            # Free memory between batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        null_max_values = np.array(null_max_values)

        # Save null distribution shard for this subject
        sub_null_dir = os.path.join(null_dir, f"sub_{bids_sub_id}")
        os.makedirs(sub_null_dir, exist_ok=True)
        out_path = os.path.join(sub_null_dir, f"{batch_id}.npy")
        np.save(out_path, null_max_values)
        print(f"  Saved {len(null_max_values)} null values to {out_path}")

        # Free subject-level memory
        del cache, raw_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
