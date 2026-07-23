"""Five-stage downstream neural encoding evaluation for saved GCCA artifacts."""

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
from src.core.registry import resolve_dataset_context
from src.utils.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
from src.utils.correlation import fisher_z_mean
from src.utils.data_loaders import load_sig_electrodes
from src.utils.decomposition import (
    _get_decomposition_inputs,
    build_target_manifest,
)
from src.utils.embeddings import get_glove_embeddings, load_embeddings
from src.utils.encoding import load_encoding_results
from src.utils.gcca_encoding import (
    ENCODING_SCHEMA_VERSION,
    atomic_replace_directory,
    get_gcca_embeddings,
)
from src.utils.subspace_io import (
    load_shared_selection_metadata,
    resolve_subspace_run,
    sanitize_name,
)
from src.utils.subspace_utils import evaluate_input_space
from src.utils.srrr import fit_dense_rrr
from src.utils.multiview_rrr import union_view_significance
from src.utils.config_utils import save_effective_config


def _read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _resolved_runs(metadata):
    return {
        name: resolve_subspace_run(
            SubspaceComparisonRun(
                name=name,
                rrr_config=run["rrr_config"],
                unit_dir=run["unit_dir"],
                kind=run.get("kind"),
            )
        )
        for name, run in metadata["runs"].items()
    }


# Stage 1
def get_union_sig_elecs(metadata, resolved):
    """Union significant electrodes across anchor-dataset GCCA member runs."""
    anchor = resolved[metadata["anchor"]]
    anchor_key = tuple(anchor.dataset_key)
    union, _, sources = union_view_significance(
        {name: resolved[name] for name in metadata["views"]},
        anchor_key,
        anchor.subject_ids,
    )
    comparison_runs = {
        name: resolved[name]
        for name in metadata["views"]
        if tuple(resolved[name].dataset_key) == anchor_key
    }
    return anchor, union, comparison_runs, sources


# Stage 2
def load_gcca_model_embeddings(metadata, resolved, anchor):
    """Load every configured model over the anchor transcript and dataset context."""
    anchor_enc = anchor.config.encoding
    context = resolve_dataset_context(
        dataset=anchor_enc.dataset,
        subject_ids=anchor.subject_ids,
        per_dataset_config=anchor_enc.per_dataset_config,
        data_root=anchor_enc.data_root,
    )
    loaded = {}
    word_df = None
    cache = {}
    for name in metadata["views"]:
        run_enc = resolved[name].config.encoding
        if int(run_enc.offset) != int(anchor_enc.offset):
            raise ValueError(
                f"GCCA view {name} uses offset {run_enc.offset}; anchor uses {anchor_enc.offset}"
            )
        identity = json.dumps(
            {"model": run_enc.model, "per_model_config": run_enc.per_model_config},
            sort_keys=True,
        )
        if identity not in cache:
            cache[identity] = load_embeddings(
                anchor_enc.data_root,
                run_enc.model,
                dataset=anchor_enc.dataset,
                per_dataset_config=anchor_enc.per_dataset_config,
                dataset_context=context,
                **run_enc.per_model_config,
            )
        values, words = cache[identity]
        if word_df is None:
            word_df = words
        elif len(words) != len(word_df):
            raise ValueError(f"Embedding transcript length differs for GCCA view {name}")
        loaded[name] = np.asarray(values)
    return loaded, word_df


# Stage 3
def build_gcca_decomposition_inputs(anchor, union_sig_elecs, embeddings, word_df):
    """Build one aligned multi-model input/target problem using the RRR helper."""
    enc = anchor.config.encoding
    valid_rows = None
    if enc.limit_to_glove or enc.model == "glove":
        valid_rows = get_glove_embeddings(enc.data_root, word_df, **enc.per_model_config)[1][
            "in_glove"
        ]
    shared = (
        load_shared_selection_metadata(anchor.sig_dir, anchor.subject_ids)
        if enc.shared_word_selection
        else None
    )
    return _get_decomposition_inputs(
        anchor.subject_ids,
        str(anchor.output_dir),
        union_sig_elecs,
        enc,
        embeddings,
        word_in_glove=valid_rows,
        shared_selection_metadata=shared,
    )


def _load_artifacts(gcca_dir, metadata, fold, dimension):
    artifacts = {}
    for view in metadata["views"]:
        path = (
            gcca_dir
            / f"fold_{fold}"
            / f"shared_dim_{dimension}"
            / f"view_{sanitize_name(view)}.npz"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing GCCA view artifact: {path}")
        with np.load(path) as saved:
            artifacts[view] = {key: saved[key] for key in saved.files}
    return artifacts


def _reliability(gcca_dir, fold, dimension, views, method):
    if method in {"contribution", "equal"}:
        return None, None
    frame = pd.read_csv(gcca_dir / "inner_reliability.csv")
    frame = frame[
        (frame.outer_fold == fold) & (frame.shared_dimension == dimension)
    ]
    agreement = (
        frame.groupby(["view", "final_component"])
        .loo_corr.mean()
        .unstack()
        .reindex(views)
        .to_numpy()
    )
    error = (
        frame.groupby(["view", "final_component"])
        .validation_calibration_mse.mean()
        .unstack()
        .reindex(views)
        .to_numpy()
    )
    return agreement, error


def _summary(frame):
    rows = []
    for dimension, group in frame.groupby("k", sort=True):
        rows.append(
            {
                "k": int(dimension),
                "gcca_corr": float(fisher_z_mean(group.gcca_corr.to_numpy())),
                "gcca_r2": float(np.nanmean(group.gcca_r2)),
                "effective_rank": float(np.nanmean(group.effective_rank)),
            }
        )
    return pd.DataFrame(rows)


# Stage 4
def evaluate_gcca_encoders(path, gcca_dir, metadata, prepared, config):
    """Construct fold/rank GCCA coordinates and fit ridge neural encoders."""
    dimensions = config.shared_dimensions or metadata["shared_dimensions"]
    unavailable = sorted(set(dimensions) - set(metadata["shared_dimensions"]))
    if unavailable:
        raise ValueError(f"Unavailable GCCA shared dimensions: {unavailable}")
    rows = []
    weight_rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = np.arange(prepared.targets.shape[0], dtype=int)
    for fold, test_idx in enumerate(prepared.test_idxs):
        shared_path = gcca_dir / f"fold_{fold}" / f"shared_dim_{dimensions[0]}" / "shared.npz"
        with np.load(shared_path) as shared:
            train_idx = np.asarray(shared["outer_train_idxs"], dtype=int)
            saved_test = np.asarray(shared["outer_test_idxs"], dtype=int)
        if not np.array_equal(saved_test, test_idx):
            raise ValueError(f"GCCA fold {fold} test rows differ from anchor encoding folds")
        if not np.array_equal(np.sort(np.concatenate([train_idx, test_idx])), all_rows):
            raise ValueError(f"GCCA fold {fold} does not partition all analysis rows")
        target_scaler = StandardScaler().fit(prepared.targets[train_idx])
        Y_train = target_scaler.transform(prepared.targets[train_idx]).astype("float32")
        Y_test = target_scaler.transform(prepared.targets[test_idx]).astype("float32")
        for dimension in dimensions:
            artifacts = _load_artifacts(gcca_dir, metadata, fold, dimension)
            agreement, error = _reliability(
                gcca_dir, fold, dimension, list(metadata["views"]), config.averaging_method
            )
            gcca_embeddings, weights = get_gcca_embeddings(
                artifacts,
                prepared.embeddings,
                averaging_method=config.averaging_method,
                inner_agreement=agreement,
                calibration_error=error,
            )
            rank_dir = path / f"k{dimension}" / f"fold_{fold}"
            alpha_dir = rank_dir / (
                "gcca_initial_alpha" if config.run_rrr else "gcca_alpha"
            )
            context = {
                "script": "gcca_encoding",
                "role": "gcca",
                "fold": int(fold),
                "k": int(dimension),
                "n_features": int(gcca_embeddings.shape[1]),
                "standardize": bool(config.standardize_inputs),
            }
            alphas = resolve_ridge_alphas(
                config.ridge_alphas,
                gcca_embeddings[train_idx],
                standardize=config.standardize_inputs,
                context=context,
                cache_dir=str(alpha_dir),
                policy=config.alpha_grid,
            )
            result = evaluate_input_space(
                gcca_embeddings[train_idx],
                gcca_embeddings[test_idx],
                Y_train,
                Y_test,
                device=device,
                alphas=alphas,
                ridge_cv_folds=config.ridge_cv_folds,
                standardize=config.standardize_inputs,
                ridge_solver_params=config.ridge_solver_params,
                ridge_y_in_cpu=config.ridge_y_in_cpu,
                ridge_force_cpu=config.ridge_force_cpu,
                return_alphas=True,
                return_weights=config.run_rrr,
                subspace_name="gcca",
            )
            rank_dir.mkdir(parents=True, exist_ok=True)
            initial_best_alphas = result["gcca_best_alphas"]
            evaluated_train = gcca_embeddings[train_idx]
            if config.run_rrr:
                rrr_result = fit_dense_rrr(
                    gcca_embeddings[train_idx],
                    Y_train,
                    k=int(dimension),
                    selected_idx=np.arange(gcca_embeddings.shape[1], dtype=int),
                    precomputed_Y_hat_ols=(
                        gcca_embeddings[train_idx] @ result["gcca_weights"]
                        + result["gcca_intercept"]
                    ),
                    precomputed_beta_full=result["gcca_weights"],
                    device=device,
                )
                A = rrr_result["A_dense"]
                np.save(rank_dir / "A.npy", A)
                aligned_train = gcca_embeddings[train_idx] @ A
                aligned_test = gcca_embeddings[test_idx] @ A
                evaluated_train = aligned_train
                final_alpha_dir = rank_dir / "gcca_alpha"
                final_context = {
                    **context,
                    "role": "gcca_rrr_aligned",
                    "n_features": int(aligned_train.shape[1]),
                }
                final_alphas = resolve_ridge_alphas(
                    config.ridge_alphas,
                    aligned_train,
                    standardize=config.standardize_inputs,
                    context=final_context,
                    cache_dir=str(final_alpha_dir),
                    policy=config.alpha_grid,
                )
                result = evaluate_input_space(
                    aligned_train,
                    aligned_test,
                    Y_train,
                    Y_test,
                    device=device,
                    alphas=final_alphas,
                    ridge_cv_folds=config.ridge_cv_folds,
                    standardize=config.standardize_inputs,
                    ridge_solver_params=config.ridge_solver_params,
                    ridge_y_in_cpu=config.ridge_y_in_cpu,
                    ridge_force_cpu=config.ridge_force_cpu,
                    return_alphas=True,
                    subspace_name="gcca",
                )
                save_alpha_diagnostics(
                    str(final_alpha_dir),
                    final_alphas,
                    best_alphas=result["gcca_best_alphas"],
                    context=final_context,
                    policy=config.alpha_grid,
                )
            metrics = {key: value for key, value in result.items() if key != "gcca_Y_hat"}
            np.savez_compressed(rank_dir / "metrics.npz", **metrics)
            save_alpha_diagnostics(
                str(alpha_dir),
                alphas,
                best_alphas=initial_best_alphas,
                context=context,
                policy=config.alpha_grid,
            )
            rows.append(
                {
                    "fold": fold,
                    "k": dimension,
                    "effective_rank": int(np.linalg.matrix_rank(evaluated_train)),
                    "gcca_corr": result["gcca_corr"],
                    "gcca_r2": result["gcca_r2"],
                }
            )
            for view_index, view in enumerate(metadata["views"]):
                for component in range(dimension):
                    weight_rows.append(
                        {
                            "fold": fold,
                            "k": dimension,
                            "component": component + 1,
                            "view": view,
                            "weight": weights[view_index, component],
                            "method": config.averaging_method,
                        }
                    )
    grid = pd.DataFrame(rows)
    grid.to_csv(path / "grid_results.csv", index=False)
    _summary(grid).to_csv(path / "summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(path / "weights.csv", index=False)
    return grid


def _run_target_manifest(run):
    enc = run.config.encoding
    significant, _ = load_sig_electrodes(
        str(run.output_dir),
        run.subject_ids,
        sig_dir=str(run.sig_dir),
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    first_subject = next(subject for subject in run.subject_ids if significant.get(subject))
    saved = load_encoding_results(
        str(run.output_dir / "subjects" / f"sub_{first_subject}" / "encoding_results.joblib")
    )
    groups = [significant[subject] for subject in run.subject_ids if significant.get(subject)]
    return build_target_manifest(run.subject_ids, significant, groups, saved["lags"])


def _manifest_index(frame):
    keys = list(zip(frame.subject.astype(int), frame.electrode.astype(str), frame.lag_index.astype(int)))
    if len(keys) != len(set(keys)):
        raise ValueError("Target manifest contains duplicate (subject, electrode, lag) identities")
    return {key: index for index, key in enumerate(keys)}, keys


def _rank_from_dir(path):
    try:
        return int(path.name[1:])
    except ValueError:
        return None


def _comparison_for_run(path, prepared, run_name, run, dimensions):
    run_manifest = _run_target_manifest(run)
    union_index, _ = _manifest_index(prepared.target_manifest)
    _, run_keys = _manifest_index(run_manifest)
    missing = [key for key in run_keys if key not in union_index]
    if missing:
        raise ValueError(
            f"RRR run {run_name} has {len(missing)} targets absent from union_sig_elecs"
        )
    gcca_columns = np.asarray([union_index[key] for key in run_keys], dtype=int)
    rows = []
    for fold in range(len(prepared.test_idxs)):
        for dimension in dimensions:
            with np.load(path / f"k{dimension}" / f"fold_{fold}" / "metrics.npz") as saved:
                values = np.asarray(saved["gcca_corr_per_target"])[gcca_columns]
            rows.append(
                {"run": run_name, "source": "GCCA", "fold": fold, "k": dimension,
                 "corr": float(fisher_z_mean(values)), "n_targets": len(values)}
            )
        for rank_dir in sorted(run.unit_dir.glob("k*"), key=lambda item: _rank_from_dir(item) or -1):
            rank = _rank_from_dir(rank_dir)
            metrics_path = rank_dir / f"fold_{fold}" / "metrics.npz"
            if rank is None or not metrics_path.exists():
                continue
            with np.load(metrics_path) as saved:
                if "nr_corr_per_target" not in saved:
                    continue
                values = np.asarray(saved["nr_corr_per_target"])
            if len(values) != len(run_manifest):
                raise ValueError(
                    f"RRR run {run_name} rank {rank} fold {fold} has {len(values)} "
                    f"correlations for {len(run_manifest)} target identities"
                )
            rows.append(
                {"run": run_name, "source": "RRR NR", "fold": fold, "k": rank,
                 "corr": float(fisher_z_mean(values)), "n_targets": len(values)}
            )
    curves = pd.DataFrame(rows)
    summary_rows = []
    for (source, rank), group in curves.groupby(["source", "k"], sort=True):
        summary_rows.append(
            {"run": run_name, "source": source, "k": rank,
             "corr": float(fisher_z_mean(group["corr"].to_numpy())),
             "n_targets": int(group.n_targets.iloc[0])}
        )
    summary = pd.DataFrame(summary_rows)
    gcca = summary[summary.source == "GCCA"][["k", "corr"]].rename(columns={"corr": "gcca_corr"})
    rrr = summary[summary.source == "RRR NR"][["k", "corr"]].rename(columns={"corr": "rrr_nr_corr"})
    deltas = gcca.merge(rrr, on="k", how="inner")
    deltas.insert(0, "run", run_name)
    deltas["gcca_minus_rrr"] = deltas.gcca_corr - deltas.rrr_nr_corr

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"GCCA": "#7c3aed", "RRR NR": "#3b82f6"}
    for source in ("RRR NR", "GCCA"):
        subset = curves[curves.source == source]
        for _, fold_frame in subset.groupby("fold"):
            ax.semilogx(fold_frame.k, fold_frame["corr"], color=colors[source], alpha=.2, linewidth=.8)
        mean = summary[summary.source == source]
        ax.semilogx(mean.k, mean["corr"], marker="o", color=colors[source], label=source)
    ax.axhline(0, color="black", linewidth=.5, linestyle=":")
    ax.set(xlabel="Rank k", ylabel="Correlation", title=f"GCCA vs RRR NR: {run_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / f"comparison_{sanitize_name(run_name)}.png", dpi=150)
    plt.close(fig)
    return curves, summary, deltas


# Stage 5
def write_results_and_comparisons(
    path, gcca_dir, metadata, prepared, comparison_runs, config
):
    """Evaluate, serialize, and compare against existing RRR result artifacts."""
    evaluate_gcca_encoders(path, gcca_dir, metadata, prepared, config)
    prepared.target_manifest.to_csv(path / "target_manifest.csv", index=False)
    np.savez_compressed(
        path / "folds.npz",
        **{f"test_{fold}": indices for fold, indices in enumerate(prepared.test_idxs)},
    )
    dimensions = config.shared_dimensions or metadata["shared_dimensions"]
    curve_frames, summary_frames, delta_frames = [], [], []
    for run_name, run in comparison_runs.items():
        curves, summary, deltas = _comparison_for_run(
            path, prepared, run_name, run, dimensions
        )
        curve_frames.append(curves)
        summary_frames.append(summary)
        delta_frames.append(deltas)
    pd.concat(curve_frames, ignore_index=True).to_csv(path / "comparison_curves.csv", index=False)
    pd.concat(summary_frames, ignore_index=True).to_csv(path / "comparison_summary.csv", index=False)
    pd.concat(delta_frames, ignore_index=True).to_csv(path / "comparison_deltas.csv", index=False)


def run_gcca_encoding(config: GCCAEncodingConfig) -> Path:
    config.validate()
    gcca_dir = Path(config.gcca_dir)
    metadata = _read_json(gcca_dir / "metadata.json")
    if metadata.get("artifact_schema_version") != 1:
        raise ValueError("Incompatible GCCA artifact schema")
    output = gcca_dir / "encoding" / "runs" / config.run_name
    if output.exists() and not config.refresh:
        raise FileExistsError(
            f"GCCA encoding run exists: {output}; set refresh=true to replace it"
        )
    resolved = _resolved_runs(metadata)
    anchor, union, comparison_runs, sources = get_union_sig_elecs(metadata, resolved)
    embeddings, word_df = load_gcca_model_embeddings(metadata, resolved, anchor)
    prepared = build_gcca_decomposition_inputs(anchor, union, embeddings, word_df)
    if prepared is None:
        raise ValueError("The GCCA anchor dataset produced no neural targets")
    if prepared.targets.shape[0] != int(metadata["word_count"]):
        raise ValueError(
            f"Prepared {prepared.targets.shape[0]} words; GCCA metadata expects {metadata['word_count']}"
        )

    def build(path):
        save_effective_config(config, path)
        write_results_and_comparisons(
            path, gcca_dir, metadata, prepared, comparison_runs, config
        )
        (path / "metadata.json").write_text(
            json.dumps(
                {
                    "schema": ENCODING_SCHEMA_VERSION,
                    "source_gcca_schema": metadata["artifact_schema_version"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "gcca_dir": str(gcca_dir.resolve()),
                    "anchor": metadata["anchor"],
                    "comparison_runs": list(comparison_runs),
                    "target_union_sources": sources,
                    "config": asdict(config),
                },
                indent=2,
            )
        )

    atomic_replace_directory(output, build)
    return output
