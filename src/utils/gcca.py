"""Config-driven orchestration and artifact I/O for partial-support MAXVAR GCCA."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.core.config import MaxvarGCCAConfig, SubspaceComparisonRun
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.embeddings import load_embeddings
from src.utils.subspace_compare import (
    align_maxvar_folds,
    fit_regularized_maxvar_gcca,
    maxvar_ambient_map,
    maxvar_block_overlap,
    maxvar_component_blocks,
    score_maxvar_partial_support,
    select_maxvar_alpha,
    transform_maxvar_gcca_view,
)
from src.utils.subspace_io import (
    load_base_comparison_data,
    load_fold_bases,
    load_peer_embeddings_on_base_rows,
    resolve_subspace_run,
    run_metadata,
    sanitize_name,
)

SCHEMA_VERSION = 1


def resolve_gcca_output_dir(config: MaxvarGCCAConfig, anchor) -> Path:
    """Resolve an explicit output or the anchor stimulus scope below output_root."""
    if config.output_dir is not None:
        return Path(config.output_dir)
    enc = anchor.config.encoding
    context = resolve_dataset_context(
        dataset=enc.dataset,
        subject_ids=anchor.subject_ids,
        per_dataset_config=enc.per_dataset_config,
        data_root=enc.data_root,
    )
    suffix = get_dataset_adapter(enc.dataset).get_results_suffix(context)
    return Path(config.output_root) / (suffix or str(context.stimulus_id))


def _calibration(estimate: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Fit independent affine calibrations from view estimates to shared scores."""
    x, y = np.asarray(estimate, float), np.asarray(target, float)
    x_mean, y_mean = x.mean(0), y.mean(0)
    centered = x - x_mean
    denominator = np.sum(centered**2, axis=0)
    gain = np.divide(
        np.sum(centered * (y - y_mean), axis=0), denominator,
        out=np.zeros(x.shape[1]), where=denominator > np.finfo(float).eps,
    )
    intercept = y_mean - gain * x_mean
    residual = y - (x * gain + intercept)
    residual_variance = np.mean(residual**2, axis=0)
    target_variance = np.var(y, axis=0)
    r2 = 1.0 - np.divide(
        residual_variance, target_variance, out=np.full_like(target_variance, np.nan),
        where=target_variance > np.finfo(float).eps,
    )
    return {
        "gain": gain, "intercept": intercept,
        "residual_variance": residual_variance,
        "target_variance": target_variance, "r2": r2,
    }


def _match_components(reference: np.ndarray, candidate: np.ndarray):
    def normalize(values):
        centered = values - values.mean(0)
        norm = np.linalg.norm(centered, axis=0)
        return centered / np.where(norm > 0, norm, 1.0)
    correlations = normalize(reference).T @ normalize(candidate)
    rows, cols = linear_sum_assignment(-np.abs(correlations))
    order = np.empty(reference.shape[1], dtype=int)
    matches = np.empty(reference.shape[1], dtype=float)
    signs = np.empty(reference.shape[1], dtype=float)
    order[rows] = cols
    matches[rows] = np.abs(correlations[rows, cols])
    signs[rows] = np.where(correlations[rows, cols] < 0, -1.0, 1.0)
    return order, matches, signs


def _inner_splits(base_data, outer_fold: int, dimensions: list[int]):
    outer_train = base_data.train_idxs[outer_fold]
    splits = []
    for inner_fold, block in enumerate(base_data.test_idxs):
        if inner_fold == outer_fold:
            continue
        validation = np.intersect1d(outer_train, block)
        training = np.setdiff1d(outer_train, validation)
        if len(validation) >= 2 and len(training) > max(dimensions):
            splits.append((inner_fold, training, validation))
    return splits


def _row_manifest(anchor, base_data) -> pd.DataFrame:
    enc = anchor.config.encoding
    context = resolve_dataset_context(
        dataset=enc.dataset, subject_ids=anchor.subject_ids,
        per_dataset_config=enc.per_dataset_config, data_root=enc.data_root,
    )
    _, words = load_embeddings(
        enc.data_root, enc.model, dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config, dataset_context=context,
        **enc.per_model_config,
    )
    selection = np.asarray(base_data.selection, dtype=int)
    available = [name for name in ("word_idx", "word", "start", "end") if name in words]
    manifest = words.iloc[selection][available].reset_index(drop=True).copy()
    manifest.insert(0, "transcript_row", selection)
    manifest.insert(0, "analysis_row", np.arange(len(selection), dtype=int))
    fold_for_row = np.full(len(selection), -1, dtype=int)
    for fold, indices in enumerate(base_data.test_idxs):
        fold_for_row[np.asarray(indices, dtype=int)] = fold
    manifest["outer_test_fold"] = fold_for_row
    return manifest


def _save_plot(summary: pd.DataFrame, components: pd.DataFrame, stability: pd.DataFrame,
               path: Path) -> None:
    import matplotlib.pyplot as plt
    grouped = summary.groupby("shared_dimension", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for metric in ("mean_partial_support_corr", "mean_loo_corr", "mean_pairwise_corr"):
        axes[0].plot(grouped.shared_dimension, grouped[metric], marker="o", label=metric)
    axes[0].legend(fontsize=7); axes[0].set_title("Held-out agreement")
    shares = components.groupby(["shared_dimension", "view"]).contribution_share.mean().unstack()
    shares.plot(ax=axes[1], marker="o"); axes[1].set_title("Mean contribution share")
    counts = components.groupby("shared_dimension").support_view_count.mean()
    axes[2].plot(counts.index, counts.values, marker="o"); axes[2].set_title("Support views")
    unique = stability.drop_duplicates(["shared_dimension", "component"])
    stable = unique.groupby("shared_dimension").stable_axis.sum()
    axes[3].bar(stable.index.astype(str), stable.values); axes[3].set_title("Stable axes")
    for ax in axes: ax.grid(True, alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _artifact_path(output: Path, fold: int, dimension: int, name: str) -> Path:
    folder = output / f"fold_{fold}" / f"shared_dim_{dimension}"
    folder.mkdir(parents=True, exist_ok=True)
    filename = "shared.npz" if name == "__shared__" else f"view_{sanitize_name(name)}.npz"
    return folder / filename


def _update_npz(path: Path, **updates) -> None:
    with np.load(path) as saved:
        payload = {name: saved[name] for name in saved.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)


def run_gcca(config: MaxvarGCCAConfig) -> Path:
    """Run and atomically save one configured anchor-scoped GCCA analysis."""
    config.validate()
    resolved = {
        view.name: resolve_subspace_run(SubspaceComparisonRun(
            name=view.name, rrr_config=view.rrr_config,
            unit_dir=view.unit_dir, kind=view.kind,
        )) for view in config.views
    }
    anchor = resolved[config.anchor]
    output = resolve_gcca_output_dir(config, anchor)
    if output.exists() and not config.refresh:
        raise FileExistsError(f"GCCA output already exists: {output}; set refresh=true to replace it")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        _run_into(config, resolved, anchor, temporary)
        if output.exists():
            shutil.rmtree(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _run_into(config, resolved, anchor, output: Path) -> None:
    names = tuple(view.name for view in config.views)
    ranks = {view.name: int(view.rank) for view in config.views}
    dimensions = sorted(set(int(value) for value in config.shared_dimensions))
    base = load_base_comparison_data(anchor)
    embeddings, bases, modes = {}, {}, {}
    for name in names:
        run, rank = resolved[name], ranks[name]
        embeddings[name] = np.asarray(
            base.X_base if name == anchor.name else load_peer_embeddings_on_base_rows(anchor, run, base)
        )
        fold_bases = load_fold_bases(run.unit_dir, rank)
        if run.dataset_key == anchor.dataset_key:
            peer = load_base_comparison_data(run)
            if not np.array_equal(base.selection, peer.selection):
                raise ValueError(f"View {name} selection differs from anchor")
            if any(not np.array_equal(a, b) for a, b in zip(base.test_idxs, peer.test_idxs)):
                raise ValueError(f"View {name} fold indices differ from anchor")
            bases[name], modes[name] = fold_bases, "matching_anchor_fold"
        else:
            stacked = np.concatenate([fold_bases[key] for key in sorted(fold_bases)], axis=1)
            left, _, _ = np.linalg.svd(stacked, full_matrices=False)
            mean_basis = left[:, :rank]
            bases[name] = {fold: mean_basis for fold in range(len(base.test_idxs))}
            modes[name] = "cross_dataset_fold_mean"

    summary_rows, view_rows, component_rows, pair_rows = [], [], [], []
    inner_grid_rows, inner_rows, inner_alignment_rows = [], [], []
    fold_scores_cache, eigenvalues_cache = {}, {}
    for outer_fold, (outer_train, outer_test) in enumerate(zip(base.train_idxs, base.test_idxs)):
        projected = {name: embeddings[name] @ bases[name][outer_fold] for name in names}
        splits = _inner_splits(base, outer_fold, dimensions)
        selected, grid = select_maxvar_alpha(
            projected, [(train, validation) for _, train, validation in splits],
            dimensions, config.ridge_alphas,
            support_cumulative_mass=config.support_cumulative_mass,
        )
        inner_grid_rows.extend({"outer_fold": outer_fold, **row} for row in grid)
        train_views = {name: projected[name][outer_train] for name in names}
        test_views = {name: projected[name][outer_test] for name in names}
        for dimension in dimensions:
            alpha = selected[dimension]
            fit = fit_regularized_maxvar_gcca(train_views, dimension, alpha=alpha)
            score = score_maxvar_partial_support(
                fit, train_views, test_views, n_components=dimension,
                cumulative_mass=config.support_cumulative_mass,
            )
            eigenvalues_cache[(outer_fold, dimension)] = fit.eigenvalues.copy()
            summary_rows.append({
                "outer_fold": outer_fold, "shared_dimension": dimension,
                "selected_alpha": alpha, "n_train": len(outer_train), "n_test": len(outer_test),
                "mean_loo_corr": score["mean_loo_corr"],
                "mean_pairwise_corr": score["mean_pairwise_corr"],
                "mean_partial_support_corr": score["mean_partial_support_corr"],
                "mean_component_support": float(np.mean(fit.eigenvalues / len(names))),
            })
            for (left, right), values in score["pairwise_corrs"].items():
                pair_rows.extend({
                    "outer_fold": outer_fold, "shared_dimension": dimension,
                    "component": component + 1, "view_left": left, "view_right": right,
                    "held_out_correlation": float(value),
                } for component, value in enumerate(values))
            all_scores = {}
            calibrations = {}
            for name in names:
                ambient, intercept = maxvar_ambient_map(fit, name, bases[name][outer_fold])
                raw = embeddings[name] @ ambient + intercept
                train_raw = raw[outer_train]
                calibration = _calibration(train_raw, fit.shared_scores)
                calibrated = raw * calibration["gain"] + calibration["intercept"]
                all_scores[name], calibrations[name] = raw, calibration
                corrs = score["per_view_corrs"][name]
                view_rows.append({
                    "outer_fold": outer_fold, "shared_dimension": dimension, "view": name,
                    "rank": ranks[name], "basis_mode": modes[name],
                    "mean_loo_corr": float(np.nanmean(corrs)),
                })
                for component, corr in enumerate(corrs):
                    component_rows.append({
                        "outer_fold": outer_fold, "shared_dimension": dimension,
                        "component": component + 1, "view": name, "loo_corr": float(corr),
                        "selected_support": name in score["support_views"][component],
                        "support_view_count": int(score["support_counts"][component]),
                        "support_view_names": "|".join(score["support_views"][component]),
                        "selected_view_corr": float(score["selected_view_corrs"][name][component]),
                        "selected_consensus_corr": float(score["selected_consensus_corrs"][name][component]),
                        "maxvar_eigenvalue": float(fit.eigenvalues[component]),
                        "compound_singular_value": float(fit.singular_values[component]),
                        "view_support": float(fit.view_support[name][component]),
                        "objective_contribution": float(fit.objective_contributions[name][component]),
                        "contribution_share": float(fit.contribution_shares[name][component]),
                    })
                np.savez_compressed(_artifact_path(output, outer_fold, dimension, name), **{
                    "basis": bases[name][outer_fold], "rotation": fit.rotations[name],
                    "projected_mean": fit.means[name], "projected_scale": fit.scales[name],
                    "ridge_value": np.asarray(fit.ridge_values[name]),
                    "effective_rank": np.asarray(fit.effective_ranks[name]),
                    "ambient_map": ambient, "intercept": intercept,
                    "all_anchor_scores": raw, "calibrated_all_anchor_scores": calibrated,
                    **{f"calibration_{key}": value for key, value in calibration.items()},
                    "view_support": fit.view_support[name],
                    "objective_contribution": fit.objective_contributions[name],
                    "contribution_share": fit.contribution_shares[name],
                    "selected_support": np.asarray([name in group for group in score["support_views"]]),
                })
            consensus = np.mean(np.stack(list(all_scores.values())), axis=0)
            fold_scores_cache[(outer_fold, dimension)] = consensus
            np.savez_compressed(_artifact_path(output, outer_fold, dimension, "__shared__"), **{
                "train_shared_scores": fit.shared_scores, "eigenvalues": fit.eigenvalues,
                "singular_values": fit.singular_values, "outer_train_idxs": outer_train,
                "outer_test_idxs": outer_test, "all_anchor_consensus_scores": consensus,
                "alpha": np.asarray(alpha), "view_names": np.asarray(names),
                "support_counts": score["support_counts"],
                "support_membership": np.asarray([[name in group for name in names] for group in score["support_views"]]),
            })

            final_train_consensus = consensus[outer_train]
            for inner_fold, inner_train, validation in splits:
                inner_train_views = {name: projected[name][inner_train] for name in names}
                inner_validation_views = {name: projected[name][validation] for name in names}
                inner_fit = fit_regularized_maxvar_gcca(inner_train_views, dimension, alpha=alpha)
                inner_score = score_maxvar_partial_support(
                    inner_fit, inner_train_views, inner_validation_views,
                    cumulative_mass=config.support_cumulative_mass,
                )
                inner_all = {name: transform_maxvar_gcca_view(inner_fit, name, projected[name][outer_train]) for name in names}
                inner_consensus = np.mean(np.stack(list(inner_all.values())), axis=0)
                assignment, match, signs = _match_components(final_train_consensus, inner_consensus)
                for final_component, inner_component in enumerate(assignment):
                    inner_alignment_rows.append({
                        "outer_fold": outer_fold, "inner_fold": inner_fold,
                        "shared_dimension": dimension, "final_component": final_component + 1,
                        "inner_component": int(inner_component) + 1,
                        "match_correlation": float(match[final_component]), "sign": float(signs[final_component]),
                    })
                    group = inner_score["support_views"][inner_component]
                    calibrated_validation = {}
                    for name in names:
                        inner_training_raw = transform_maxvar_gcca_view(inner_fit, name, inner_train_views[name])
                        calibration = _calibration(inner_training_raw, inner_fit.shared_scores)
                        raw_validation = inner_score["test_scores"][name]
                        calibrated_validation[name] = raw_validation * calibration["gain"] + calibration["intercept"]
                    for name in names:
                        prediction = calibrated_validation[name][:, inner_component]
                        target_names = [peer for peer in group if peer != name]
                        if not target_names:
                            target_names = list(group)
                        target = np.mean(np.stack([
                            calibrated_validation[peer][:, inner_component]
                            for peer in target_names
                        ]), axis=0)
                        target_variance = float(np.var(target))
                        mse = float(np.mean((prediction - target) ** 2))
                        inner_rows.append({
                            "outer_fold": outer_fold, "inner_fold": inner_fold,
                            "shared_dimension": dimension, "selected_alpha": alpha,
                            "final_component": final_component + 1,
                            "inner_component": int(inner_component) + 1, "view": name,
                            "n_validation": len(validation), "assignment_correlation": float(match[final_component]),
                            "selected_support": name in group,
                            "view_support": float(inner_fit.view_support[name][inner_component]),
                            "contribution_share": float(inner_fit.contribution_shares[name][inner_component]),
                            "selected_view_corr": float(inner_score["selected_view_corrs"][name][inner_component]),
                            "selected_consensus_corr": float(inner_score["selected_consensus_corrs"][name][inner_component]),
                            "loo_corr": float(inner_score["per_view_corrs"][name][inner_component]),
                            "validation_calibration_mse": mse,
                            "validation_target_variance": target_variance,
                            "validation_calibration_r2": (1.0 - mse / target_variance if target_variance > np.finfo(float).eps else np.nan),
                        })

    components = pd.DataFrame(component_rows)
    stability_rows, block_rows = [], []
    for dimension in dimensions:
        fold_scores = {fold: fold_scores_cache[(fold, dimension)] for fold in range(len(base.test_idxs))}
        eigenvalues = {fold: eigenvalues_cache[(fold, dimension)] for fold in range(len(base.test_idxs))}
        alignment = align_maxvar_folds(
            fold_scores, eigenvalues,
            match_threshold=config.axis_match_correlation_threshold,
            eigengap_threshold=config.relative_neighboring_eigengap_threshold,
        )
        medoid = alignment["medoid_fold"]
        for fold in alignment["folds"]:
            _update_npz(
                _artifact_path(output, fold, dimension, "__shared__"),
                fold_alignment_assignment=alignment["assignments"][fold] + 1,
                fold_alignment_sign=alignment["signs"][fold],
                fold_alignment_correlation=alignment["match_correlations"][fold],
                stable_axis=alignment["stable"],
            )
            for component in range(dimension):
                stability_rows.append({
                    "shared_dimension": dimension, "component": component + 1,
                    "outer_fold": fold, "medoid_fold": medoid,
                    "matched_component": int(alignment["assignments"][fold][component]) + 1,
                    "match_correlation": float(alignment["match_correlations"][fold][component]),
                    "sign": float(alignment["signs"][fold][component]),
                    "median_absolute_match_correlation": float(alignment["median_match_correlation"][component]),
                    "relative_neighboring_eigengap": float(alignment["relative_neighboring_eigengap"][component]),
                    "stable_axis": bool(alignment["stable"][component]),
                })
        blocks = maxvar_component_blocks(
            alignment["stable"], alignment["relative_neighboring_eigengap"],
            eigengap_threshold=config.relative_neighboring_eigengap_threshold,
        )
        stable_indices = np.flatnonzero(alignment["stable"])
        if stable_indices.size:
            aligned = []
            for fold in alignment["folds"]:
                matched = alignment["assignments"][fold][stable_indices]
                aligned.append(
                    fold_scores[fold][:, matched] * alignment["signs"][fold][stable_indices]
                )
            _update_npz(
                _artifact_path(output, medoid, dimension, "__shared__"),
                stable_component_indices=stable_indices + 1,
                aggregate_stable_anchor_scores=np.mean(np.stack(aligned), axis=0),
                fold_subspace_similarity=alignment["subspace_similarity"],
            )
        for block_id, block in enumerate(blocks, 1):
            overlap = []
            reference = fold_scores[medoid][:, block]
            for fold in alignment["folds"]:
                matched = alignment["assignments"][fold][list(block)]
                overlap.extend(maxvar_block_overlap(reference, fold_scores[fold][:, matched]))
            block_components = [value + 1 for value in block]
            medoid_rows = components[(components.outer_fold == medoid) & (components.shared_dimension == dimension) & components.component.isin(block_components)]
            total = medoid_rows.objective_contribution.sum()
            for name in names:
                support = medoid_rows[medoid_rows.view == name].view_support.sum()
                block_rows.append({
                    "shared_dimension": dimension, "block": block_id,
                    "component_indices": "|".join(map(str, block_components)),
                    "block_size": len(block), "view": name,
                    "summed_view_support": float(support),
                    "block_contribution_share": float(support / max(total, np.finfo(float).eps)),
                    "mean_canonical_correlation": float(np.mean(overlap)),
                })
            _update_npz(
                _artifact_path(output, medoid, dimension, "__shared__"),
                **{f"block_{block_id}_components": np.asarray(block_components)},
            )

    tables = {
        "summary.csv": pd.DataFrame(summary_rows), "view_metrics.csv": pd.DataFrame(view_rows),
        "component_metrics.csv": components, "view_component_support.csv": components,
        "pairwise_component_correlations.csv": pd.DataFrame(pair_rows),
        "inner_ridge_grid.csv": pd.DataFrame(inner_grid_rows),
        "inner_reliability.csv": pd.DataFrame(inner_rows),
        "inner_component_alignment.csv": pd.DataFrame(inner_alignment_rows),
        "component_stability.csv": pd.DataFrame(stability_rows),
        "component_blocks.csv": pd.DataFrame(block_rows),
    }
    for filename, table in tables.items(): table.to_csv(output / filename, index=False)
    _row_manifest(anchor, base).to_csv(output / "anchor_rows.csv", index=False)
    fold_payload = {"selection": base.selection}
    for fold, (train, test) in enumerate(zip(base.train_idxs, base.test_idxs)):
        fold_payload[f"train_{fold}"] = train; fold_payload[f"test_{fold}"] = test
    np.savez_compressed(output / "folds.npz", **fold_payload)
    metadata = {
        "artifact_schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(), "anchor": anchor.name,
        "output_dir": str(resolve_gcca_output_dir(config, anchor)),
        "views": list(names), "view_ranks": ranks, "basis_modes": modes,
        "shared_dimensions": dimensions, "ridge_alphas": list(map(float, config.ridge_alphas)),
        "support_cumulative_mass": config.support_cumulative_mass,
        "axis_match_correlation_threshold": config.axis_match_correlation_threshold,
        "relative_neighboring_eigengap_threshold": config.relative_neighboring_eigengap_threshold,
        "reliability_weighting": "not_selected; raw ingredients and calibration saved",
        "anchor_run": run_metadata(anchor),
        "runs": {name: run_metadata(resolved[name]) for name in names},
        "n_folds": len(base.test_idxs), "word_count": base.word_count,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    _save_plot(tables["summary.csv"], components, tables["component_stability.csv"], output / "quicklook.png")
