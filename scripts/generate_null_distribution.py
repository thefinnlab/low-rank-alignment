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
    load_raws,
    load_embeddings,
    build_word_epochs,
    apply_offset,
    train_encoding,
    get_all_predictions_and_targets_test_set,
)
from utils.config import NullDistConfig, get_results_dir
from utils.config_utils import parse_args, load_config
from utils.encoding import get_ridge_model_constructor
from utils.significance import phase_randomize_raw, compute_observed_max_per_electrode


def _run_batched_permutations(
    raw, df_word, aligned_embeddings, enc, model_constructor, rng, batch_size,
):
    """Generate ``batch_size`` phase-randomised permutations and train one model.

    The Y matrices for each permutation are concatenated column-wise so that a
    single ridge model predicts all permutations simultaneously (ridge is
    independent per target, so this is equivalent to training separate models
    but much faster due to batched linear algebra).

    Returns:
        np.ndarray of shape (batch_size,) — max-across-electrodes scalar per
        permutation.
    """
    Y_parts = []
    n_elec = None
    n_lags = None
    X_arr = None

    for _ in range(batch_size):
        raw_perm = phase_randomize_raw(raw, rng=rng)
        epochs_perm, data_perm = build_word_epochs(raw_perm, df_word)
        X_i, Y_i = apply_offset(
            aligned_embeddings[epochs_perm.selection], data_perm, enc.offset
        )
        if n_elec is None:
            n_elec = data_perm.shape[1]
            n_lags = data_perm.shape[2]
            X_arr = X_i
        Y_parts.append(Y_i)
        del raw_perm, epochs_perm, data_perm, X_i

    # Concatenate targets: (n_words, batch_size * n_elec * n_lags)
    Y_concat = np.concatenate(Y_parts, axis=1).astype(np.float32)
    del Y_parts

    total_elec = batch_size * n_elec
    X = [X_arr, X_arr]  # 2-fold CV

    # Train one model for all permutations at once
    corrs, models, tr_idxs, test_idxs, scalers = train_encoding(
        X, Y_concat, model_constructor,
        epochs_shape=(total_elec, n_lags),
    )

    # Get combined predictions: (n_words, total_elec, n_lags)
    all_predictions, all_targets = get_all_predictions_and_targets_test_set(
        models, test_idxs, X, Y_concat, total_elec, n_lags
    )

    # Split per permutation and compute max-across-electrodes scalar each
    null_maxes = np.zeros(batch_size)
    for b in range(batch_size):
        start = b * n_elec
        end = (b + 1) * n_elec
        max_per_elec, _ = compute_observed_max_per_electrode(
            all_predictions[:, start:end, :],
            all_targets[:, start:end, :],
        )
        null_maxes[b] = max_per_elec.max()

    return null_maxes


def main():
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
    aligned_embeddings, df_word = load_embeddings(enc.data_root, enc.layer)

    alphas = np.array(enc.ridge_alphas) if enc.ridge_alphas is not None else None
    model_constructor = get_ridge_model_constructor(alphas=alphas)

    for bids_sub_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Subject {bids_sub_id}")
        print(f"{'='*60}")

        # Load raw data once per subject
        raws = load_raws([bids_sub_id], enc.data_root, use_high_gamma=True)
        raw = raws[0]
        n_elec = len(raw.ch_names)
        print(f"  Electrodes: {n_elec}")

        n_batches = math.ceil(config.n_iterations / batch_size)
        null_max_values = []

        for b_idx in tqdm(range(n_batches), desc=f"  Sub {bids_sub_id} batches"):
            remaining = config.n_iterations - len(null_max_values)
            cur_batch = min(batch_size, remaining)

            batch_maxes = _run_batched_permutations(
                raw, df_word, aligned_embeddings, enc, model_constructor, rng,
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
        del raws, raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
