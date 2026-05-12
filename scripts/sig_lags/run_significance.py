"""
run_significance.py — Batched per-subject encoding fold-correlation export.

Processes one subject at a time to avoid OOM while training, then writes one
correlation tensor and a parallel electrode metadata CSV.
"""

import gc
import os

import numpy as np
import pandas as pd
from himalaya.backend import set_backend
from src.utils.embeddings import get_glove_embeddings
import torch

from src.utils import (
    train_encoding,
    load_embeddings,
)
from src.core.config import SignificanceConfig, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import (
    parse_args,
    load_config,
    expand_per_model_layer_configs,
)
from src.utils.encoding import extract_best_alphas, get_ridge_model_constructor
from src.utils.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
from src.utils.data_loaders import load_data_from_cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_config(config: SignificanceConfig):
    enc = config.encoding

    subject_ids = enc.subjects or list(range(1, 10))
    dataset_context = resolve_dataset_context(
        dataset=enc.dataset,
        subject_ids=subject_ids,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    get_dataset_adapter(enc.dataset).validate_subjects_share_stimulus(
        subject_ids, enc.per_dataset_config, enc.data_root
    )
    output_dir = config.output_dir or get_results_dir(
        enc.model,
        enc.offset,
        enc.per_model_config,
        dataset_context=dataset_context,
    )

    # Use GPU backend if available
    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {output_dir}")

    # Load embeddings once — shared across all subjects
    aligned_embeddings, transcript_df = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        **enc.per_model_config,
    )

    word_in_glove = None
    if enc.limit_to_glove or enc.model == "glove":
        word_in_glove = get_glove_embeddings(
            enc.data_root, transcript_df, **enc.per_model_config
        )[1]["in_glove"]

    explicit_alphas = (
        np.array(enc.ridge_alphas, dtype=float) if enc.ridge_alphas is not None else None
    )

    diagnostics_dir = os.path.join(output_dir, "correlation_diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    all_corrs = []
    electrode_rows = []
    expected_shape = None
    electrode_offset = 0

    for bids_sub_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Processing subject {bids_sub_id}")
        print(f"{'='*60}")

        X_arr, Y, selection, elec_names = load_data_from_cache(
            [bids_sub_id],
            enc.data_root,
            enc.offset,
            aligned_embeddings,
            valid_rows=word_in_glove,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=dataset_context,
        )
        _, n_targets = Y.shape
        elec_counts = [len(x) for x in elec_names]
        n_lags = n_targets // sum(elec_counts)
        n_elec = sum(elec_counts)
        X = [X_arr] * enc.num_folds
        alpha_context = {
            "script": "run_significance",
            "role": "subject",
            "subject": int(bids_sub_id),
            "model": enc.model,
            "per_model_config": enc.per_model_config,
            "offset": enc.offset,
            "n_features": int(X_arr.shape[1]),
            "standardize": True,
        }
        alpha_dir = os.path.join(output_dir, "subjects", f"sub_{bids_sub_id}")
        alphas = resolve_ridge_alphas(
            explicit_alphas,
            X_arr,
            standardize=True,
            context=alpha_context,
            cache_dir=alpha_dir,
            policy=enc.alpha_grid,
        )
        model_constructor = get_ridge_model_constructor(
            alphas=alphas,
            ridge_cv_folds=enc.ridge_cv_folds,
            solver_params=enc.ridge_solver_params,
            y_in_cpu=enc.ridge_y_in_cpu,
            force_cpu=enc.ridge_force_cpu,
        )

        # Update lag_sec based on actual n_lags (in case resampling gives different count)
        actual_lag_sec = np.linspace(-2.0, 2.0, n_lags, endpoint=False)

        # ------------------------------------------------------------------
        # e. Train encoding model
        # ------------------------------------------------------------------
        print("  Training encoding model...")
        corrs, models, tr_idxs, test_idxs, scalers = train_encoding(
            X,
            Y,
            model_constructor,
            epochs_shape=(n_elec, n_lags),
            num_folds=enc.num_folds,
        )
        if enc.log_alphas:
            save_alpha_diagnostics(
                alpha_dir,
                alphas,
                best_alphas=[extract_best_alphas(m) for m in models],
                context=alpha_context,
                policy=enc.alpha_grid,
            )

        # ------------------------------------------------------------------
        # f. Collect fold-level target correlations and electrode metadata
        # ------------------------------------------------------------------
        if expected_shape is None:
            expected_shape = (corrs.shape[0], corrs.shape[2])
        elif expected_shape != (corrs.shape[0], corrs.shape[2]):
            raise ValueError(
                "All subjects must have the same fold and lag dimensions to "
                "write one correlation tensor. "
                f"Expected {expected_shape}, got {(corrs.shape[0], corrs.shape[2])} "
                f"for subject {bids_sub_id}."
            )

        all_corrs.append(corrs.astype(np.float32, copy=False))
        for elec_idx, elec_name in enumerate(elec_names[0]):
            electrode_rows.append(
                {
                    "electrode_index": electrode_offset + elec_idx,
                    "subject": bids_sub_id,
                    "subject_electrode_index": elec_idx,
                    "electrode": elec_name,
                    "n_lags": n_lags,
                    "lag_start_sec": float(actual_lag_sec[0]),
                    "lag_step_sec": float(actual_lag_sec[1] - actual_lag_sec[0])
                    if n_lags > 1
                    else np.nan,
                }
            )
        electrode_offset += n_elec
        print(
            f"  Collected {n_elec:,} electrodes "
            f"(tensor shape {corrs.shape})"
        )

        # ------------------------------------------------------------------
        # g. Free memory
        # ------------------------------------------------------------------
        del X_arr, X, Y, models, scalers
        del corrs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Write combined diagnostics
    # ------------------------------------------------------------------
    if not all_corrs:
        raise ValueError("No subject correlations were computed.")

    corr_data = np.concatenate(all_corrs, axis=1)
    corr_path = os.path.join(diagnostics_dir, "fold_correlations.npy")
    metadata_path = os.path.join(diagnostics_dir, "electrodes.csv")
    np.save(corr_path, corr_data)
    pd.DataFrame(electrode_rows).to_csv(metadata_path, index=False)
    print(f"\nWrote {corr_path}  shape={corr_data.shape}")
    print(f"Wrote {metadata_path}  ({len(electrode_rows):,} electrodes)")


def main():
    args, overrides = parse_args("Per-subject encoding fold-correlation export")
    config: SignificanceConfig = load_config(args.config, overrides, SignificanceConfig)
    configs = expand_per_model_layer_configs(config)
    for layer_config in configs:
        if len(configs) > 1:
            print(
                f"\nExporting fold correlations for layer "
                f"{layer_config.encoding.per_model_config['layer']}"
            )
        run_config(layer_config)


if __name__ == "__main__":
    main()
