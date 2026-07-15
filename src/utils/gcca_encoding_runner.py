"""End-to-end downstream encoding evaluation for a saved GCCA directory."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.core.config import GCCAEncodingConfig, SubspaceComparisonRun
from src.utils.correlation import fisher_z_mean
from src.utils.data_loaders import build_XY_from_subjects, load_sig_electrodes
from src.utils.embeddings import load_embeddings
from src.utils.encoding import build_per_fold_baselines
from src.utils.gcca_encoding import (
    ENCODING_SCHEMA_VERSION, atomic_replace_directory, baseline_fingerprint,
    component_weights, deduplicate_embedding_runs, paired_improvements,
    reconstruct_view_scores,
)
from src.utils.subspace_io import load_base_comparison_data, load_peer_embeddings_on_base_rows, resolve_subspace_run, sanitize_name
from src.utils.subspace_utils import evaluate_input_space


def _json(path: Path):
    with path.open() as handle: return json.load(handle)


def _resolved_runs(metadata):
    return {name: resolve_subspace_run(SubspaceComparisonRun(
        name=name, rrr_config=run["rrr_config"], unit_dir=run["unit_dir"], kind=run.get("kind")
    )) for name, run in metadata["runs"].items()}


def _load_problem(gcca_dir: Path, metadata):
    resolved = _resolved_runs(metadata); anchor = resolved[metadata["anchor"]]
    base = load_base_comparison_data(anchor)
    anchor_key = tuple(anchor.dataset_key)
    union = {subject: [] for subject in anchor.subject_ids}; provenance = []
    for name, run in resolved.items():
        if tuple(run.dataset_key) != anchor_key: continue
        sig, frame = load_sig_electrodes(str(run.output_dir), run.subject_ids, sig_dir=str(run.sig_dir),
            dataset=run.config.encoding.dataset, per_dataset_config=run.config.encoding.per_dataset_config,
            data_root=run.config.encoding.data_root)
        provenance.append(str(Path(run.sig_dir) / "all_subject_sig.csv"))
        for subject, electrodes in sig.items():
            if subject in union:
                union[subject].extend(e for e in electrodes if e not in union[subject])
    union = {subject: values for subject, values in union.items() if values}
    enc = anchor.config.encoding
    raw_anchor, _ = load_embeddings(enc.data_root, enc.model, dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config, **enc.per_model_config)
    X, Y, selection, electrode_groups = build_XY_from_subjects(anchor.subject_ids, enc, union, raw_anchor)
    if not np.array_equal(np.asarray(selection), np.asarray(base.selection)) or X.shape[0] != base.word_count:
        raise ValueError("Target-union rows do not match saved GCCA anchor rows")
    n_per_electrode = Y.shape[1] // sum(map(len, electrode_groups))
    targets = []
    for subject, electrodes in zip([s for s in anchor.subject_ids if s in union], electrode_groups):
        for electrode in electrodes:
            for lag in range(n_per_electrode): targets.append({"subject": int(subject), "electrode": electrode, "lag_index": lag})
    return resolved, anchor, base, np.asarray(Y), pd.DataFrame(targets), provenance


def _policy(config):
    return {key: value for key, value in asdict(config).items() if key not in {
        "gcca_dir", "run_name", "refresh", "averaging_method", "calibrated_scores", "shared_dimensions"
    }}


def _baseline_cache_complete(path: Path, identities, n_folds):
    if not (path / "complete.json").exists(): return False
    return all((path / f"fold_{fold}" / "targets.npz").exists() and
               (path / f"fold_{fold}" / f"baseline_{identity}.npz").exists()
               for identity in identities for fold in range(n_folds))


def _build_baselines(path, *, unique, resolved, anchor, base, Y, targets, config, fingerprint, provenance):
    anchor_enc = anchor.config.encoding; rows = []
    for identity, run_meta in unique.items():
        view = next(name for name in resolved if deduplicate_embedding_runs({name: run_meta})[1][name] == identity)
        run = resolved[view]
        X = base.X_base if view == anchor.name else load_peer_embeddings_on_base_rows(anchor, run, base)
        folds = build_per_fold_baselines(X, Y, base.test_idxs,
            alphas=np.asarray(config.ridge_alphas) if config.ridge_alphas is not None else None,
            ridge_cv_folds=config.ridge_cv_folds, ridge_solver_params=config.ridge_solver_params,
            ridge_y_in_cpu=config.ridge_y_in_cpu, ridge_force_cpu=config.ridge_force_cpu,
            alpha_policy=config.alpha_grid)
        for fold, item in enumerate(folds):
            from himalaya.scoring import correlation_score, r2_score
            corr = np.asarray(correlation_score(item["Y_test"], item["Y_hat_full_test"])); r2 = np.asarray(r2_score(item["Y_test"], item["Y_hat_full_test"]))
            folder = path / f"fold_{fold}"; folder.mkdir(parents=True, exist_ok=True)
            target_path = folder / "targets.npz"
            if not target_path.exists():
                np.savez_compressed(target_path, Y_test=item["Y_test"], test_idx=item["test_idx"],
                    target_mean=item["target_mean"], target_scale=item["target_scale"])
            np.savez_compressed(folder / f"baseline_{identity}.npz", predictions=item["Y_hat_full_test"],
                corr_per_target=corr, r2_per_target=r2, selected_alphas=item["best_alphas"])
            rows.append({"identity": identity, "fold": fold, "corr": float(fisher_z_mean(corr)), "r2": float(np.nanmean(r2))})
    pd.DataFrame(rows).to_csv(path / "controlled_baselines.csv", index=False)
    targets.to_csv(path / "target_manifest.csv", index=False)
    np.savez_compressed(path / "folds.npz", **{f"test_{i}": x for i, x in enumerate(base.test_idxs)})
    (path / "provenance.json").write_text(json.dumps({"fingerprint": fingerprint, "sources": provenance, "policy": _policy(config)}, indent=2))
    (path / "complete.json").write_text(json.dumps({"schema": ENCODING_SCHEMA_VERSION, "completed_at": datetime.now(timezone.utc).isoformat()}))


def _reliability(gcca_dir, fold, dimension, views, method):
    if method in {"contribution", "equal"}: return None, None
    frame = pd.read_csv(gcca_dir / "inner_reliability.csv")
    frame = frame[(frame.outer_fold == fold) & (frame.shared_dimension == dimension)]
    agree = frame.groupby(["view", "final_component"]).loo_corr.mean().unstack().reindex(views).to_numpy()
    error = frame.groupby(["view", "final_component"]).validation_calibration_mse.mean().unstack().reindex(views).to_numpy()
    return agree, error


def _build_run(path, *, gcca_dir, metadata, base, cache, identities, view_map, config):
    dimensions = config.shared_dimensions or metadata["shared_dimensions"]
    unavailable = sorted(set(dimensions) - set(metadata["shared_dimensions"]))
    if unavailable: raise ValueError(f"Unavailable GCCA shared dimensions: {unavailable}")
    grid, weight_rows, paired = [], [], []
    for fold, (train, test) in enumerate(zip(base.train_idxs, base.test_idxs)):
        baseline_arrays = {identity: np.load(cache / f"fold_{fold}" / f"baseline_{identity}.npz") for identity in identities}
        Y_train = StandardScaler().fit_transform(np.zeros((1, 1))) if False else None
        # Cached Y_test uses the authoritative fold-local target scaling. Recreate train scaling below from raw targets supplied in closure.
        for k in dimensions:
            scores, contributions, views = [], [], list(metadata["views"])
            for view in views:
                with np.load(gcca_dir / f"fold_{fold}" / f"shared_dim_{k}" / f"view_{sanitize_name(view)}.npz") as artifact:
                    scores.append(reconstruct_view_scores(artifact, _build_run.embeddings[view], calibrated=config.calibrated_scores))
                    contributions.append(np.asarray(artifact["contribution_share"]))
            agreement, error = _reliability(gcca_dir, fold, k, views, config.averaging_method)
            weights = component_weights(config.averaging_method, np.stack(contributions), inner_agreement=agreement, calibration_error=error)
            combined = np.sum(np.stack(scores) * weights[:, None, :], axis=0)
            scaler = StandardScaler().fit(_build_run.Y[train]); Y_train = scaler.transform(_build_run.Y[train]).astype("float32"); Y_test = scaler.transform(_build_run.Y[test]).astype("float32")
            result = evaluate_input_space(combined[train], combined[test], Y_train, Y_test,
                device="cuda" if torch.cuda.is_available() else "cpu", alphas=np.asarray(config.ridge_alphas) if config.ridge_alphas is not None else None,
                ridge_cv_folds=config.ridge_cv_folds, standardize=config.standardize_inputs,
                ridge_solver_params=config.ridge_solver_params, ridge_y_in_cpu=config.ridge_y_in_cpu,
                ridge_force_cpu=config.ridge_force_cpu, return_alphas=True, subspace_name="nr")
            baseline_corrs = {identity: np.asarray(artifact["corr_per_target"]) for identity, artifact in baseline_arrays.items()}
            metrics = {key: value for key, value in result.items() if key != "nr_Y_hat"}
            for identity, corr in baseline_corrs.items(): metrics[f"delta_{identity}_corr_per_target"] = result["nr_corr_per_target"] - corr
            best = np.nanmax(np.stack(list(baseline_corrs.values())), axis=0); metrics["delta_best_baseline_corr_per_target"] = result["nr_corr_per_target"] - best
            folder = path / f"k{k}" / f"fold_{fold}"; folder.mkdir(parents=True, exist_ok=True); np.savez_compressed(folder / "metrics.npz", **metrics)
            row = {"fold": fold, "k": k, "effective_rank": int(np.linalg.matrix_rank(combined[train])), "nr_corr": result["nr_corr"], "nr_r2": result["nr_r2"]}
            for identity, artifact in baseline_arrays.items(): row[f"baseline_{identity}_corr"] = float(fisher_z_mean(artifact["corr_per_target"])); row[f"baseline_{identity}_r2"] = float(np.nanmean(artifact["r2_per_target"]))
            anchor_id = view_map[metadata["anchor"]]; row["full_corr"] = row[f"baseline_{anchor_id}_corr"]; row["full_r2"] = row[f"baseline_{anchor_id}_r2"]
            grid.append(row); summary = paired_improvements(result["nr_corr_per_target"], baseline_corrs); summary.insert(0, "fold", fold); summary.insert(1, "k", k); paired.append(summary)
            for vi, view in enumerate(views):
                for component in range(k): weight_rows.append({"fold": fold, "k": k, "component": component + 1, "view": view, "weight": weights[vi, component], "method": config.averaging_method})
    frame = pd.DataFrame(grid); frame.to_csv(path / "grid_results.csv", index=False); frame.groupby("k", as_index=False).mean(numeric_only=True).to_csv(path / "summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(path / "weights.csv", index=False); pd.concat(paired).to_csv(path / "paired_improvements.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); grouped = frame.groupby("k").nr_corr.mean(); ax.plot(grouped.index, grouped.values, marker="o"); ax.set(xlabel="Shared dimension", ylabel="Correlation", title="GCCA encoding"); fig.savefig(path / "rank_analysis.png", dpi=150); plt.close(fig)


def run_gcca_encoding(config: GCCAEncodingConfig) -> Path:
    config.validate(); gcca_dir = Path(config.gcca_dir); metadata = _json(gcca_dir / "metadata.json")
    if metadata.get("artifact_schema_version") != 1: raise ValueError("Incompatible GCCA artifact schema")
    resolved, anchor, base, Y, targets, provenance = _load_problem(gcca_dir, metadata)
    unique, view_map = deduplicate_embedding_runs(metadata["runs"])
    fingerprint = baseline_fingerprint(target_manifest=targets, selection=base.selection, folds=base.test_idxs,
        embeddings=unique, evaluation_policy=_policy(config), source_paths=[gcca_dir, *provenance])
    cache = gcca_dir / "encoding" / "baselines" / fingerprint; output = gcca_dir / "encoding" / "runs" / config.run_name
    if output.exists() and not config.refresh: raise FileExistsError(f"Encoding run exists: {output}; set refresh=true to replace it")
    if config.refresh:
        atomic_replace_directory(cache, lambda path: _build_baselines(path, unique=unique, resolved=resolved, anchor=anchor, base=base, Y=Y, targets=targets, config=config, fingerprint=fingerprint, provenance=provenance))
    elif not _baseline_cache_complete(cache, unique, len(base.test_idxs)):
        raise FileNotFoundError(f"Compatible baseline cache is missing or stale at {cache}. Re-run with refresh=true to create it.")
    embeddings = {name: (base.X_base if name == anchor.name else load_peer_embeddings_on_base_rows(anchor, run, base)) for name, run in resolved.items()}
    _build_run.embeddings, _build_run.Y = embeddings, Y
    def build(path):
        _build_run(path, gcca_dir=gcca_dir, metadata=metadata, base=base, cache=cache, identities=unique, view_map=view_map, config=config)
        (path / "metadata.json").write_text(json.dumps({"schema": ENCODING_SCHEMA_VERSION, "source_gcca_schema": metadata["artifact_schema_version"], "baseline_fingerprint": fingerprint, "baseline_path": str(cache), "target_union_sources": provenance, "config": asdict(config)}, indent=2))
    atomic_replace_directory(output, build)
    return output
