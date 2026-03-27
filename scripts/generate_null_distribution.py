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

The per-subject output file contains the array of max-across-electrode scalars
(one per iteration).

Writes:
  {null_dir}/sub_{id}/{batch_uuid}.npy   — shape (n_iterations,)
"""

import gc
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


def main():
    args, overrides = parse_args("Phase-randomization null distribution generation")
    config: NullDistConfig = load_config(args.config, overrides, NullDistConfig)
    enc = config.encoding

    subject_ids = enc.subjects or list(range(1, 10))
    output_dir = config.output_dir or get_results_dir(enc.model, enc.layer, enc.offset)
    null_dir = config.null_dir or os.path.join(output_dir, "null_dist")

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
        electrode_names = raw.ch_names
        n_elec = len(electrode_names)
        print(f"  Electrodes: {n_elec}")

        # Pre-compute epoch metadata from the real raw to know dimensions
        epochs_real, data_real = build_word_epochs(raw, df_word)
        n_words, _, n_lags = data_real.shape
        print(f"  Epoch data shape: {data_real.shape}")
        del data_real

        null_max_values = np.zeros(config.n_iterations)

        for i in tqdm(range(config.n_iterations), desc=f"  Sub {bids_sub_id} permutations"):
            # 1. Phase-randomize the raw signal
            raw_perm = phase_randomize_raw(raw, rng=rng)

            # 2. Build word epochs from the phase-randomized signal
            epochs_perm, data_perm = build_word_epochs(raw_perm, df_word)

            # 3. Align embeddings and neural data
            X_arr, Y = apply_offset(
                aligned_embeddings[epochs_perm.selection], data_perm, enc.offset
            )
            n_elec_perm = data_perm.shape[1]
            n_lags_perm = data_perm.shape[2]
            X = [X_arr, X_arr]  # one per fold (2-fold CV)

            # 4. Train encoding model
            corrs, models, tr_idxs, test_idxs, scalers = train_encoding(
                X, Y, model_constructor,
                epochs_shape=(n_elec_perm, n_lags_perm),
            )

            # 5. Get test-set predictions
            all_predictions, all_targets = get_all_predictions_and_targets_test_set(
                models, test_idxs, X, Y, n_elec_perm, n_lags_perm
            )

            # 6. Compute max correlation across lags per electrode, then max across electrodes
            max_per_elec, _ = compute_observed_max_per_electrode(
                all_predictions, all_targets
            )
            null_max_values[i] = max_per_elec.max()

            # Free memory
            del raw_perm, epochs_perm, data_perm, X_arr, Y, X
            del corrs, models, tr_idxs, test_idxs, scalers
            del all_predictions, all_targets, max_per_elec
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save null distribution shard for this subject
        sub_null_dir = os.path.join(null_dir, f"sub_{bids_sub_id}")
        os.makedirs(sub_null_dir, exist_ok=True)
        out_path = os.path.join(sub_null_dir, f"{batch_id}.npy")
        np.save(out_path, null_max_values)
        print(f"  Saved {config.n_iterations} null values to {out_path}")

        # Free subject-level memory
        del raws, raw, epochs_real
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
