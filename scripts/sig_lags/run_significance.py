"""
run_significance.py — Cluster-based lag significance with fold-local nulls.

Processes one subject at a time to avoid OOM while training, builds reliable
electrode-lag clusters from out-of-fold predictions, scores a fold-local
circular-shift null, and writes downstream-compatible significance CSVs.
"""

import gc
from dataclasses import asdict

import numpy as np
import pandas as pd
from himalaya.backend import set_backend
from src.utils.shared.embeddings import get_glove_embeddings
import torch
from src.utils.shared.encoding import train_encoding
from src.utils.shared.embeddings import load_embeddings
from src.core.config import SignificanceConfig
from src.core.artifacts import ArtifactStore, normalize_dataset, normalize_model
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.shared.config import (
    parse_args,
    load_config,
    expand_per_model_layer_configs,
)
from src.utils.shared.encoding import (
    extract_best_alphas,
    get_ridge_model_constructor,
)
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.shared.ieeg import (
    build_word_selection_metadata,
    compute_shared_word_selection,
    load_data_from_cache,
)
from src.utils.significance.ieeg import (
    apply_cluster_fdr,
    compute_per_electrode_cluster_p_values,
    derive_subject_null_seed,
    fold_local_circular_shift_null_scores,
    reshape_fold_predictions_and_targets,
    score_fold_lag_clusters,
)

# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


class Artifact(ArtifactStore):
    """Storage contract for iEEG lag-cluster significance results."""

    SCRIPT = "run_significance"
    FIELD_CONTRACT = {
        "effective_config": True,
        "significant_electrodes": True,
        "detailed_significance": True,
        "encoding_results": True,
        "correlations": True,
        "subject_checkpoint": True,
        "cluster_null_distribution": True,
        "alpha_diagnostics": False,
        "plots": False,
    }

    def __init__(self, *, mode="sig_lags", dataset="podcast", offset=0,
                 model="gpt2-xl", layer=24, variant=None, results_root="results"):
        self._init_store(mode=mode, dataset=dataset, offset=offset, model=model,
                         layer=layer, variant=variant, results_root=results_root)

    @classmethod
    def from_config(cls, config, dataset_context):
        enc = config.encoding
        model_dir = enc.per_model_config.get("model_dir", enc.model)
        glove_dim = enc.per_model_config.get("dim")
        return cls(
            dataset=normalize_dataset(enc.dataset, dataset_context.stimulus_id),
            offset=enc.offset,
            model=normalize_model(model_dir, glove_dim=glove_dim),
            layer=None if enc.model == "glove" else enc.per_model_config.get("layer"),
            results_root=getattr(config, "results_root", "results"),
        )

    def save_subject(self, subject, *, encoding_results, correlations=None,
                     alpha_diagnostics=None, checkpoint=None, overwrite=False):
        selector = {"subject": int(subject)}
        self.save_field("encoding_results", encoding_results, selector=selector,
                        overwrite=overwrite)
        if correlations is not None:
            self.save_field("correlations", correlations, selector=selector,
                            overwrite=overwrite)
        if alpha_diagnostics is not None:
            self.save_field("alpha_diagnostics", alpha_diagnostics,
                            selector=selector, overwrite=overwrite)
        if checkpoint is not None:
            self.save_field("subject_checkpoint", checkpoint, selector=selector,
                            overwrite=overwrite)

    def load_artifacts(self, fields, selectors=None):
        selectors = selectors or {}
        for field in fields:
            if field in {"encoding_results", "correlations", "alpha_diagnostics",
                         "subject_checkpoint"}:
                collection = self.load_collection(
                    field, {"subject": selectors.get("subjects", [])}
                ) if selectors.get("subjects") else self.load_collection(field)
                setattr(self, field, {
                    dict(selector)["subject"]: value
                    for selector, value in collection.items()
                })
            else:
                setattr(self, field, self.load_field(field))
        self.complete = bool(self.manifest().get("complete"))
        return self

    def save_artifacts(self, *, effective_config, significant_electrodes,
                       detailed_significance, cluster_null_distribution,
                       plot=True, overwrite=False):
        self.save_field("effective_config", effective_config, overwrite=overwrite)
        self.save_field("significant_electrodes", significant_electrodes,
                        overwrite=overwrite)
        self.save_field("detailed_significance", detailed_significance,
                        overwrite=overwrite)
        self.save_field("cluster_null_distribution", cluster_null_distribution,
                        overwrite=overwrite)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_config(config: SignificanceConfig):
    config.validate()
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
    artifact = Artifact.from_config(config, dataset_context).begin(
        config=config, refresh=config.refresh
    )

    # Use GPU backend if available
    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using torch_cuda backend.")
    else:
        print("CUDA not available, using CPU.")

    print(f"Artifact identity: {artifact.identity_record()}")

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
    shared_selection = None
    if enc.shared_word_selection:
        shared_selection = compute_shared_word_selection(
            subject_ids,
            enc.data_root,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=dataset_context,
        )
        print(
            "Using shared word-selection intersection: "
            f"{len(shared_selection):,} cached words across {len(subject_ids)} subjects"
        )

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

        checkpoint_selector = {"subject": int(bids_sub_id)}
        if (not config.refresh and artifact.has_field(
                "subject_checkpoint", selector=checkpoint_selector)):
            checkpoint = artifact.load_field(
                "subject_checkpoint", selector=checkpoint_selector
            )
            all_cluster_records.extend(checkpoint["cluster_records"])
            all_lag_records.extend(checkpoint["lag_records"])
            subject_nulls.append(checkpoint["null_distribution"])
            electrode_rows.extend(checkpoint["electrode_rows"])
            subject_shift_metadata[int(bids_sub_id)] = checkpoint["shift_metadata"]
            print("  Reusing complete subject checkpoint")
            continue

        X_arr, Y, selection, elec_names = load_data_from_cache(
            [bids_sub_id],
            enc.data_root,
            enc.offset,
            aligned_embeddings,
            valid_rows=word_in_glove,
            dataset=enc.dataset,
            per_dataset_config=enc.per_dataset_config,
            dataset_context=dataset_context,
            shared_selection=shared_selection,
        )
        word_selection_metadata = None
        if enc.shared_word_selection:
            word_selection_metadata = build_word_selection_metadata(
                "shared_intersection",
                subject_ids,
                selection,
                enc.offset,
                valid_rows=(
                    word_in_glove[selection] if word_in_glove is not None else None
                ),
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
        alpha_selector = {"subject": int(bids_sub_id)}
        if (not config.refresh and
                artifact.has_field("alpha_diagnostics", selector=alpha_selector)):
            alphas = np.asarray(artifact.load_field(
                "alpha_diagnostics", selector=alpha_selector
            )["alpha_grid"], dtype=float)
        else:
            alphas = resolve_ridge_alphas(
                explicit_alphas,
                X_arr,
                standardize=True,
                context=alpha_context,
                cache_dir=None,
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
        encoding_result = {
            "test_idxs": test_idxs, "train_idxs": tr_idxs,
            "lags": actual_lag_sec, "scalers": scalers,
            "elec_names": list(elec_names[0]), "subject_id": bids_sub_id,
            "offset": enc.offset, "per_model_config": dict(enc.per_model_config),
        }
        if word_selection_metadata is not None:
            encoding_result["word_selection"] = word_selection_metadata
        alpha_payload = None
        if enc.log_alphas:
            alpha_payload = {
                "alpha_grid": np.asarray(alphas),
                "best_alphas": [extract_best_alphas(m) for m in models],
                "context": alpha_context, "policy": enc.alpha_grid,
            }

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
        subject_cluster_start = len(all_cluster_records) - len(scored_clusters)

        # ------------------------------------------------------------------
        # g. Score fold-local circular-shift null for this subject
        # ------------------------------------------------------------------
        print("  Scoring fold-local circular-shift null...")
        subject_null_seed = derive_subject_null_seed(config.null_seed, bids_sub_id)
        max_per_electrode, shift_meta = fold_local_circular_shift_null_scores(
            fold_predictions_3d,
            fold_targets_3d,
            blocks_per_fold=config.blocks_per_fold,
            max_matching_distance=config.max_matching_distance,
            positive_fraction=config.positive_fraction,
            mean_corr_threshold=mean_corr_threshold,
            minimal_shift=config.minimal_shift,
            shift_step=config.shift_step,
            num_null_shifts=config.num_null_shifts,
            seed=subject_null_seed,
            device=config.device,
            null_batch_size=config.null_batch_size,
        )
        subject_nulls.append(max_per_electrode)
        subject_shift_metadata[int(bids_sub_id)] = {
            **shift_meta,
            "block_rows": block_rows,
        }
        subject_cluster_records = all_cluster_records[subject_cluster_start:]
        subject_cluster_p = compute_per_electrode_cluster_p_values(
            subject_cluster_records, max_per_electrode
        )
        for rec, p_value in zip(subject_cluster_records, subject_cluster_p):
            rec["cluster_p_value"] = float(p_value)

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
                        "electrode_p_value": 1.0,
                        "electrode_q_value": 1.0,
                        "is_electrode_best_cluster": False,
                        "positive_fraction": float(positive_frac[elec_idx, lag_idx]),
                        "mean_corr": float(mean_corr[elec_idx, lag_idx]),
                    }
                )
        print(
            f"  Found {len(scored_clusters):,} candidate clusters across "
            f"{n_elec:,} electrodes"
        )

        subject_lag_records = [
            row for row in all_lag_records if int(row["subject"]) == int(bids_sub_id)
        ]
        subject_electrode_rows = [
            row for row in electrode_rows if int(row["subject"]) == int(bids_sub_id)
        ]
        artifact.save_subject(
            bids_sub_id, encoding_results=encoding_result, correlations=corrs,
            alpha_diagnostics=alpha_payload,
            checkpoint={
                "cluster_records": subject_cluster_records,
                "lag_records": subject_lag_records,
                "null_distribution": max_per_electrode,
                "electrode_rows": subject_electrode_rows,
                "shift_metadata": subject_shift_metadata[int(bids_sub_id)],
            },
            overwrite=config.refresh,
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
    # 4. Compute per-electrode max-cluster p-values and write outputs
    # ------------------------------------------------------------------
    if not subject_nulls:
        raise ValueError("No subject null distributions were computed.")

    apply_cluster_fdr(all_cluster_records, fdr_q=config.fdr_q, fdr_mode=config.fdr_mode)

    # Write data to CSVs and NPZ, merging cluster-level results back to lag-level records for convenience.
    cluster_lookup = {}
    for rec in all_cluster_records:
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
        row["electrode_p_value"] = rec["electrode_p_value"]
        row["electrode_q_value"] = rec["electrode_q_value"]
        row["is_electrode_best_cluster"] = rec["is_electrode_best_cluster"]
        row["p_value"] = rec["p_value"]
        row["q_value"] = rec["q_value"]
        row["sig"] = rec["sig"]

    df_detailed = pd.DataFrame(all_lag_records)

    sig_pairs = (
        df_detailed[df_detailed["sig"]][["subject", "electrode"]]
        .drop_duplicates()
        .rename(columns={"electrode": "elec"})
    )

    n_null_max = max(null.shape[0] for null in subject_nulls)
    n_total_electrodes = sum(null.shape[1] for null in subject_nulls)
    max_per_electrode = np.full(
        (n_null_max, n_total_electrodes), np.nan, dtype=np.float32
    )
    null_sample_counts = np.empty(n_total_electrodes, dtype=int)
    col_start = 0
    for null in subject_nulls:
        col_stop = col_start + null.shape[1]
        max_per_electrode[: null.shape[0], col_start:col_stop] = null
        null_sample_counts[col_start:col_stop] = null.shape[0]
        col_start = col_stop
    null_payload = dict(
        max_per_electrode=max_per_electrode,
        null_sample_counts=null_sample_counts,
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
        num_null_shifts=int(config.num_null_shifts),
        null_seed=int(config.null_seed),
        null_sampling_mode="iid_with_replacement_per_fold",
        fdr_mode=config.fdr_mode,
        subject_shift_metadata=subject_shift_metadata,
    )
    artifact.save_artifacts(
        effective_config=asdict(config), significant_electrodes=sig_pairs,
        detailed_significance=df_detailed,
        cluster_null_distribution=null_payload,
        plot=getattr(config, "plot", True), overwrite=True,
    )
    required = {
        "encoding_results": [{"subject": int(s)} for s in subject_ids],
        "correlations": [{"subject": int(s)} for s in subject_ids],
        "subject_checkpoint": [{"subject": int(s)} for s in subject_ids],
    }
    artifact.mark_complete(required)
    print(f"Completed Artifact {artifact.identity_record()}")
    return artifact


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
