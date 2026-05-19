"""
run_significance.py — Cluster-based lag significance with fold-local nulls.

Processes one subject at a time to avoid OOM while training, builds reliable
electrode-lag clusters from out-of-fold predictions, scores a fold-local
circular-shift null, and writes downstream-compatible significance CSVs.
"""

import gc
import os

import numpy as np
import pandas as pd
from himalaya.backend import set_backend
from src.utils.embeddings import get_glove_embeddings
import torch
from statsmodels.stats.multitest import multipletests

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
from src.utils.significance import (
    aggregate_global_null,
    fold_local_circular_shift_null_scores,
    reshape_fold_predictions_and_targets,
    score_fold_lag_clusters,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_config(config: SignificanceConfig):
    enc = config.encoding
    mean_corr_threshold = (
        float(np.tanh(config.mean_z_threshold))
        if config.mean_z_threshold is not None
        else float(config.mean_corr_threshold)
    )

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
        np.array(enc.ridge_alphas, dtype=float)
        if enc.ridge_alphas is not None
        else None
    )

    all_cluster_records = []
    all_lag_records = []
    subject_nulls = []
    electrode_rows = []
    subject_shift_metadata = {}

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
        corrs, models, tr_idxs, test_idxs, scalers, fold_predictions = train_encoding(
            X,
            Y,
            model_constructor,
            epochs_shape=(n_elec, n_lags),
            num_folds=enc.num_folds,
            return_preds=True,
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
        # f. Build observed clusters from out-of-fold predictions
        # ------------------------------------------------------------------
        fold_predictions_3d, fold_targets_3d = reshape_fold_predictions_and_targets(
            fold_predictions, test_idxs, Y, n_elec, n_lags
        )
        print("  Computing observed block correlations and lag clusters...")
        cluster_result = score_fold_lag_clusters(
            fold_predictions_3d,
            fold_targets_3d,
            config.blocks_per_fold,
            max_matching_distance=config.max_matching_distance,
            positive_fraction=config.positive_fraction,
            mean_corr_threshold=mean_corr_threshold,
            device=config.device,
        )
        block_rows = cluster_result["block_rows"]
        obs_corrs = cluster_result["corr"]
        positive_frac = cluster_result["positive_fraction"]
        mean_corr = cluster_result["mean_corr"]
        cluster_labels = cluster_result["cluster_labels"]
        scored_clusters = cluster_result["scored_clusters"]

        for cluster in scored_clusters:
            elec_idx = int(cluster["electrode_idx"])
            all_cluster_records.append(
                {
                    "subject": bids_sub_id,
                    "electrode": elec_names[0][elec_idx],
                    "subject_electrode_index": elec_idx,
                    "cluster_id": int(cluster["cluster_id"]),
                    "lag_indices": cluster["lag_indices"],
                    "cluster_score": float(cluster["cluster_score"]),
                    "cluster_size": int(cluster["cluster_size"]),
                    "peak_corr": float(cluster["peak_corr"]),
                    "peak_score": float(cluster["peak_target_score"]),
                }
            )

        # ------------------------------------------------------------------
        # g. Score fold-local circular-shift null for this subject
        # ------------------------------------------------------------------
        print("  Scoring fold-local circular-shift null...")
        max_per_electrode, shift_meta = fold_local_circular_shift_null_scores(
            fold_predictions_3d,
            fold_targets_3d,
            blocks_per_fold=config.blocks_per_fold,
            max_matching_distance=config.max_matching_distance,
            positive_fraction=config.positive_fraction,
            mean_corr_threshold=mean_corr_threshold,
            minimal_shift=config.minimal_shift,
            shift_step=config.shift_step,
            device=config.device,
            null_batch_size=config.null_batch_size,
        )
        subject_nulls.append(max_per_electrode)
        subject_shift_metadata[int(bids_sub_id)] = {
            **shift_meta,
            "block_rows": block_rows,
        }

        # ------------------------------------------------------------------
        # h. Collect per-lag rows and electrode metadata
        # ------------------------------------------------------------------
        for elec_idx, elec_name in enumerate(elec_names[0]):
            electrode_rows.append(
                {
                    "electrode_index": len(electrode_rows),
                    "subject": bids_sub_id,
                    "subject_electrode_index": elec_idx,
                    "electrode": elec_name,
                    "n_lags": n_lags,
                    "lag_start_sec": float(actual_lag_sec[0]),
                    "lag_step_sec": (
                        float(actual_lag_sec[1] - actual_lag_sec[0])
                        if n_lags > 1
                        else np.nan
                    ),
                }
            )
            for lag_idx in range(n_lags):
                label = int(cluster_labels[elec_idx, lag_idx])
                all_lag_records.append(
                    {
                        "subject": bids_sub_id,
                        "electrode": elec_name,
                        "lag_idx": lag_idx,
                        "lag_sec": float(actual_lag_sec[lag_idx]),
                        "sig": False,
                        "p_value": 1.0,
                        "q_value": 1.0,
                        "obs_corr": float(obs_corrs[elec_idx, lag_idx]),
                        "cluster_id": label,
                        "cluster_score": np.nan,
                        "cluster_p_value": 1.0,
                        "cluster_q_value": 1.0,
                        "positive_fraction": float(positive_frac[elec_idx, lag_idx]),
                        "mean_corr": float(mean_corr[elec_idx, lag_idx]),
                    }
                )
        print(
            f"  Found {len(scored_clusters):,} candidate clusters across "
            f"{n_elec:,} electrodes"
        )

        # ------------------------------------------------------------------
        # i. Free memory
        # ------------------------------------------------------------------
        del X_arr, X, Y, models, scalers
        del corrs, fold_predictions, fold_predictions_3d, fold_targets_3d
        del cluster_result, obs_corrs
        del positive_frac, mean_corr, cluster_labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Compute global max-statistic p-values and write outputs
    # ------------------------------------------------------------------
    if not subject_nulls:
        raise ValueError("No subject null distributions were computed.")

    # Compare nulls to actual data.
    global_max = aggregate_global_null(subject_nulls)
    if all_cluster_records:
        cluster_p = np.array(
            [
                (np.sum(global_max >= rec["cluster_score"]) + 1) / (len(global_max) + 1)
                for rec in all_cluster_records
            ],
            dtype=float,
        )
        reject, cluster_q, _, _ = multipletests(
            cluster_p, alpha=config.fdr_q, method="fdr_bh"
        )
    else:
        cluster_p = np.array([], dtype=float)
        cluster_q = np.array([], dtype=float)
        reject = np.array([], dtype=bool)

    # Write data to CSVs and NPZ, merging cluster-level results back to lag-level records for convenience.
    cluster_lookup = {}
    for rec, p_value, q_value, is_sig in zip(
        all_cluster_records, cluster_p, cluster_q, reject
    ):
        rec["cluster_p_value"] = float(p_value)
        rec["cluster_q_value"] = float(q_value)
        rec["sig"] = bool(is_sig)
        key = (rec["subject"], rec["electrode"], rec["cluster_id"])
        cluster_lookup[key] = rec

    for row in all_lag_records:
        key = (row["subject"], row["electrode"], row["cluster_id"])
        rec = cluster_lookup.get(key)
        if rec is None:
            continue
        row["cluster_score"] = rec["cluster_score"]
        row["cluster_p_value"] = rec["cluster_p_value"]
        row["cluster_q_value"] = rec["cluster_q_value"]
        row["p_value"] = rec["cluster_p_value"]
        row["q_value"] = rec["cluster_q_value"]
        row["sig"] = rec["sig"]

    detailed_path = os.path.join(output_dir, "significance_detailed.csv")
    df_detailed = pd.DataFrame(all_lag_records)
    df_detailed.to_csv(detailed_path, index=False)

    sig_pairs = (
        df_detailed[df_detailed["sig"]][["subject", "electrode"]]
        .drop_duplicates()
        .rename(columns={"electrode": "elec"})
    )
    sig_path = os.path.join(output_dir, "all_subject_sig.csv")
    sig_pairs.to_csv(sig_path, index=False, columns=["subject", "elec"])

    max_per_electrode = np.concatenate(subject_nulls, axis=1)
    null_path = os.path.join(output_dir, "cluster_null_distribution.npz")
    np.savez(
        null_path,
        max_per_electrode=max_per_electrode,
        global_max=global_max,
        subjects=np.array([row["subject"] for row in electrode_rows], dtype=int),
        electrodes=np.array([row["electrode"] for row in electrode_rows], dtype=object),
        subject_electrode_index=np.array(
            [row["subject_electrode_index"] for row in electrode_rows], dtype=int
        ),
        blocks_per_fold=int(config.blocks_per_fold),
        max_matching_distance=int(config.max_matching_distance),
        positive_fraction=float(config.positive_fraction),
        mean_corr_threshold=float(mean_corr_threshold),
        minimal_shift=int(config.minimal_shift),
        shift_step=int(config.shift_step),
        subject_shift_metadata=np.array(subject_shift_metadata, dtype=object),
    )

    print(f"\nWrote {detailed_path}  ({len(df_detailed):,} rows)")
    print(f"Wrote {sig_path}  ({len(sig_pairs):,} electrodes)")
    print(f"Wrote {null_path}  shape={max_per_electrode.shape}")


def main():
    args, overrides = parse_args("Cluster-based lag significance")
    config: SignificanceConfig = load_config(args.config, overrides, SignificanceConfig)
    configs = expand_per_model_layer_configs(config)
    for layer_config in configs:
        if len(configs) > 1:
            print(
                f"\nComputing cluster significance for layer "
                f"{layer_config.encoding.per_model_config['layer']}"
            )
        run_config(layer_config)


if __name__ == "__main__":
    main()
