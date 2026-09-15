"""Compare every ordered pair of configured saved RRR/PCA spaces."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent:
    if (_ROOT / "src").is_dir():
        break
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import SubspaceComparisonConfig
from src.core.artifacts import ArtifactStore, normalized_config
from src.utils.shared.config import load_config, parse_args
from src.utils.subspace_comparison.directed import (
    ALL_METRICS,
    METRIC_IMPLEMENTATION,
    SYMMETRIC_METRICS,
    align_views,
    configure_metric_backend,
    configured_models,
    expand_spaces,
    factor_for,
    fold_indices,
    load_evaluation_data,
    project_subspace,
    standardize_evaluation_values,
    score_ordered_pair,
    shared_ranks,
    summarize_fold_metrics,
)
from src.utils.rrr.subspace_io import sanitize_name


def _selection_keys(models):
    if not models:
        raise ValueError("models must contain at least one selection")
    keys = set()
    for item in models:
        missing = {"model", "layers", "datasets"} - set(item)
        if missing:
            raise ValueError(
                "Each model selection requires model, layers, and datasets; "
                f"missing {sorted(missing)}"
            )
        for dataset in item["datasets"]:
            for layer in item["layers"]:
                keys.add((str(dataset), str(item["model"]), int(layer)))
    return keys


class Artifact(ArtifactStore):
    """One source-owned comparison result or a filtered downstream collection."""

    SCRIPT = "compare_subspaces"
    SELECTOR_ORDER = (
        "target_dataset", "target_offset", "target_model", "target_layer",
        "target_kind", "comparison_mode", "geometry", "source_rank",
        "target_rank", "equal_capacity_rank", "fold", "metric", "name",
    )
    FIELD_CONTRACT = {
        "effective_config": True, "upstream_identities": True,
        "source_metadata": True, "validation": True, "cell_metrics": True,
        "fold_metrics": True, "summary": True, "matrices": False,
        "plots": False,
    }

    def __init__(self, *, analysis_name, evaluation_dataset,
                 source_dataset=None, source_offset=0, source_model=None,
                 source_layer=None, source_kind=None, models=None, variant=None,
                 results_root="results"):
        self.analysis_name = str(analysis_name)
        self.evaluation_dataset = str(evaluation_dataset)
        self.models = models
        self.source_kind = source_kind
        self._collection = models is not None and source_model is None
        if self._collection:
            _selection_keys(models)
            self.results_root = Path(results_root)
            self.complete = False
            for field in self.FIELD_CONTRACT:
                setattr(self, field, None)
            return
        if (source_model is None or source_layer is None or source_dataset is None
                or source_kind is None):
            raise ValueError(
                "A stored comparison Artifact requires source_dataset, "
                "source_model, source_layer, and source_kind"
            )
        self._init_store(
            mode=self.analysis_name, dataset=source_dataset,
            offset=source_offset, model=source_model, layer=source_layer,
            variant=variant, results_root=results_root,
        )

    @classmethod
    def from_space(cls, config, space):
        return cls(
            analysis_name=config.analysis_name,
            evaluation_dataset=config.evaluation_dataset,
            source_dataset=space.dataset, source_offset=space.offset,
            source_model=space.model, source_layer=space.layer,
            source_kind=space.kind,
            variant=(f"{config.variant_name}/source-{space.kind}"
                     if config.variant_name else f"source-{space.kind}"),
            results_root=config.output_root,
        )

    @staticmethod
    def _matches(frame, keys, prefix):
        return frame.apply(
            lambda row: (
                str(row[f"{prefix}_dataset"]), str(row[f"{prefix}_model"]),
                int(row[f"{prefix}_layer"]),
            ) in keys,
            axis=1,
        )

    def _discover(self):
        keys = _selection_keys(self.models)
        artifacts = []
        root = self.results_root / self.SCRIPT / self.analysis_name
        for dataset, model, layer in sorted(keys):
            model_root = root / dataset
            if not model_root.exists():
                continue
            for offset_dir in sorted(model_root.glob("offset-*")):
                leaf = offset_dir / model / f"layer-{layer}"
                index = self._read_json(leaf / "manifest.json", default={})
                for variant, status in sorted(index.get("variants", {}).items()):
                    if not status.get("complete"):
                        continue
                    artifacts.append(Artifact(
                        analysis_name=self.analysis_name,
                        evaluation_dataset=self.evaluation_dataset,
                        source_dataset=dataset,
                        source_offset=int(offset_dir.name.removeprefix("offset-")),
                        source_model=model, source_layer=layer,
                        source_kind=variant.rsplit("/", 1)[-1].removeprefix("source-"),
                        variant=variant,
                        results_root=self.results_root,
                    ))
        return artifacts, keys

    def load_artifacts(self, fields, selectors=None):
        if not self._collection:
            scalar = {"effective_config", "upstream_identities", "source_metadata", "validation"}
            for field in fields:
                setattr(self, field, self.load_field(field) if field in scalar
                        else self.load_collection(field, selectors))
            self.complete = bool(self.manifest().get("complete"))
            return self
        artifacts, keys = self._discover()
        table_fields = {"fold_metrics", "summary"}
        for field in fields:
            if field not in table_fields:
                raise ValueError(
                    "Collection Artifacts load fold_metrics or summary; load a "
                    "source Artifact for lower-level fields"
                )
            frames = []
            for artifact in artifacts:
                frame = artifact.load_field(field)
                selected = frame[
                    self._matches(frame, keys, "source")
                    & self._matches(frame, keys, "target")
                ]
                if len(selected):
                    frames.append(selected)
            setattr(self, field, pd.concat(frames, ignore_index=True) if frames
                    else pd.DataFrame())
        basis = getattr(self, "fold_metrics", None)
        if basis is None:
            basis = getattr(self, "summary", pd.DataFrame())
        columns = [
            "source_dataset", "source_offset", "source_model", "source_layer",
            "source_kind", "target_dataset", "target_offset", "target_model",
            "target_layer", "target_kind",
        ]
        self.available_comparisons = (
            basis[columns].drop_duplicates().reset_index(drop=True)
            if len(basis) else pd.DataFrame(columns=columns)
        )
        self.complete = bool(artifacts) and all(item.manifest().get("complete") for item in artifacts)
        return self

    def save_artifacts(self, *, config, upstreams, source_metadata, validation):
        self.save_field(
            "effective_config", normalized_config(config, drop_paths=False),
            overwrite=config.refresh,
        )
        self.save_field("upstream_identities", upstreams, overwrite=config.refresh)
        self.save_field("source_metadata", source_metadata, overwrite=config.refresh)
        self.save_field("validation", validation, overwrite=config.refresh)
        return self


@dataclass(frozen=True)
class ComparisonJob:
    """One ordered space comparison with explicit source and target capacity."""

    mode: str
    source_index: int
    target_index: int
    source_rank: int
    target_rank: int
    equal_capacity_rank: int | None = None


def _saturation_rank(config, space) -> int:
    models = dict(configured_models(config))
    by_layer = models[space.model]["saturation_ranks"][space.kind][space.dataset]
    return int(by_layer.get(space.layer, by_layer.get(str(space.layer))))


def generate_comparison_jobs(config, spaces, equal_ranks) -> list[ComparisonJob]:
    """Expand configured modes into ordered comparisons and their two ranks."""
    jobs = []
    for mode in config.comparison_modes:
        for source_index, source in enumerate(spaces):
            for target_index, target in enumerate(spaces):
                if source_index == target_index:
                    continue
                if mode == "equal_capacity":
                    for rank in equal_ranks:
                        jobs.append(
                            ComparisonJob(
                                mode, source_index, target_index, rank, rank, rank
                            )
                        )
                    continue
                source_saturation = _saturation_rank(config, source)
                target_saturation = _saturation_rank(config, target)
                if mode == "capacity_controlled":
                    rank = min(source_saturation, target_saturation)
                    jobs.append(
                        ComparisonJob(mode, source_index, target_index, rank, rank)
                    )
                elif mode == "full_neural_space":
                    jobs.append(
                        ComparisonJob(
                            mode,
                            source_index,
                            target_index,
                            source_saturation,
                            target_saturation,
                        )
                    )
                else:  # Config validation should make this unreachable.
                    raise ValueError(f"Unknown comparison mode {mode!r}")
    return jobs


def _space_metadata(space) -> dict:
    return {
        "label": space.label,
        "generating_dataset": space.dataset,
        "offset": space.offset,
        "model": space.model,
        "layer": space.layer,
        "kind": space.kind,
        "artifact": space.artifact.identity_record(),
        "factor_set": space.source_format,
        "available_ranks": sorted(space.ranks),
    }


def _cell_selector(target, geometry, job, fold):
    selector = {
        "target_dataset": target.dataset, "target_offset": target.offset,
        "target_model": target.model, "target_layer": target.layer,
        "target_kind": target.kind, "comparison_mode": job.mode,
        "geometry": geometry, "source_rank": job.source_rank,
        "target_rank": job.target_rank, "fold": fold,
    }
    if job.equal_capacity_rank is not None:
        selector["equal_capacity_rank"] = job.equal_capacity_rank
    return selector


def _cached_metrics(artifact, selector=None, refresh=None, calculate=None) -> dict:
    # Compatibility for callers of the former path cache helper.
    if isinstance(artifact, Path):
        path, old_refresh, old_calculate = artifact, selector, refresh
        if path.exists() and not old_refresh:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload.get("implementation") == METRIC_IMPLEMENTATION
                    and set(payload.get("metrics", {})) == set(ALL_METRICS)):
                return payload["metrics"]
        metrics = old_calculate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "implementation": METRIC_IMPLEMENTATION, "metrics": metrics,
        }), encoding="utf-8")
        return metrics
    if not refresh and artifact.has_field("cell_metrics", selector=selector):
        payload = artifact.load_field("cell_metrics", selector=selector)
        if payload.get("implementation") == METRIC_IMPLEMENTATION:
            metrics = payload.get("metrics", {})
            if set(metrics) == set(ALL_METRICS):
                return metrics
    metrics = calculate()
    if set(metrics) != set(ALL_METRICS):
        raise ValueError(
            f"Metric implementation returned {sorted(metrics)}, expected "
            f"{sorted(ALL_METRICS)}"
        )
    artifact.save_field(
        "cell_metrics",
        {"implementation": METRIC_IMPLEMENTATION, "metrics": metrics},
        selector=selector, overwrite=refresh,
    )
    return metrics


def _row_prefix(source, target, geometry, job, fold, n_train, n_test):
    return {
        "comparison_mode": job.mode,
        "source_dataset": source.dataset,
        "source_offset": source.offset,
        "source_model": source.model,
        "source_layer": source.layer,
        "source_kind": source.kind,
        "source_label": source.label,
        "target_dataset": target.dataset,
        "target_offset": target.offset,
        "target_model": target.model,
        "target_layer": target.layer,
        "target_kind": target.kind,
        "target_label": target.label,
        "geometry": geometry,
        "source_rank": job.source_rank,
        "target_rank": job.target_rank,
        "equal_capacity_rank": job.equal_capacity_rank,
        "fold": fold,
        "n_train": n_train,
        "n_test": n_test,
    }


def _canonicalize_symmetric_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Make declared symmetric diagnostics exactly equal in both matrix triangles."""
    result = frame.copy()
    source_first = result["source_label"] <= result["target_label"]
    result["unordered_label_0"] = result["source_label"].where(
        source_first, result["target_label"]
    )
    result["unordered_label_1"] = result["target_label"].where(
        source_first, result["source_label"]
    )
    result["unordered_rank_0"] = result["source_rank"].where(
        source_first, result["target_rank"]
    )
    result["unordered_rank_1"] = result["target_rank"].where(
        source_first, result["source_rank"]
    )
    keys = [
        "comparison_mode", "geometry", "fold",
        "unordered_label_0", "unordered_label_1",
        "unordered_rank_0", "unordered_rank_1",
    ]
    means = result.groupby(keys, sort=False)[list(SYMMETRIC_METRICS)].transform("mean")
    result.loc[:, list(SYMMETRIC_METRICS)] = means
    return result.drop(
        columns=[
            "unordered_label_0", "unordered_label_1",
            "unordered_rank_0", "unordered_rank_1",
        ]
    )


def _weighted_job_supported(config, source, target) -> bool:
    return all(
        space.dataset == config.evaluation_dataset
        and space.source_format == "fold_factors"
        for space in (source, target)
    )


def _run_comparisons(config, spaces, jobs, evaluation, artifacts) -> pd.DataFrame:
    rows = []
    factor_cache = {}

    def cached_factor(space_index, space, rank, fold, geometry):
        key = (space_index, rank, fold, geometry)
        if key not in factor_cache:
            factor_cache[key] = factor_for(
                space, rank, fold, config.evaluation_dataset, geometry
            )
        return factor_cache[key]

    views = {
        space.key: evaluation.load_view(space.model, space.layer, space.offset)
        for space in spaces
    }
    eligible = [
        (geometry, job)
        for geometry in config.geometries
        for job in jobs
        if geometry != "weighted"
        or _weighted_job_supported(
            config, spaces[job.source_index], spaces[job.target_index]
        )
    ]
    total = len(eligible) * len(evaluation.fold_test_ids)
    progress = tqdm(total=total, desc="ordered comparisons", unit="fold")
    for geometry, job in eligible:
        source_index, target_index = job.source_index, job.target_index
        source, target = spaces[source_index], spaces[target_index]
        common_ids, source_values, target_values = align_views(
            views[source.key], views[target.key]
        )
        for fold in range(len(evaluation.fold_test_ids)):
            train_idx, test_idx = fold_indices(
                common_ids, source, target, evaluation, fold
            )
            selector = _cell_selector(target, geometry, job, fold)
            def calculate_metrics():
                source_factor = cached_factor(
                    source_index, source, job.source_rank, fold, geometry
                )
                target_factor = cached_factor(
                    target_index, target, job.target_rank, fold, geometry
                )
                if source_factor is None or target_factor is None:
                    raise RuntimeError(
                        "Preflight accepted a comparison factor that is "
                        "unavailable during scoring"
                    )
                if source_values.shape[1] != source_factor.shape[0]:
                    raise ValueError(
                        f"Ambient mismatch for {source.label}: evaluation "
                        f"values have {source_values.shape[1]} columns but "
                        f"factor has {source_factor.shape[0]} rows"
                    )
                if target_values.shape[1] != target_factor.shape[0]:
                    raise ValueError(
                        f"Ambient mismatch for {target.label}: evaluation "
                        f"values have {target_values.shape[1]} columns but "
                        f"factor has {target_factor.shape[0]} rows"
                    )
                return score_ordered_pair(
                    project_subspace(
                        standardize_evaluation_values(
                            source,
                            source_values,
                            train_idx,
                            fold=fold,
                            evaluation_dataset=config.evaluation_dataset,
                        ),
                        source_factor, device="cuda",
                    ),
                    project_subspace(
                        standardize_evaluation_values(
                            target,
                            target_values,
                            train_idx,
                            fold=fold,
                            evaluation_dataset=config.evaluation_dataset,
                        ),
                        target_factor, device="cuda",
                    ),
                    train_idx,
                    test_idx,
                    ridge_alphas=config.ridge_alphas,
                    rsm_metric=config.rsm_metric,
                    device="cuda",
                )

            metrics = _cached_metrics(
                artifacts[source_index], selector,
                config.refresh,
                calculate_metrics,
            )
            rows.append(
                {
                    **_row_prefix(
                        source,
                        target,
                        geometry,
                        job,
                        fold,
                        len(train_idx),
                        len(test_idx),
                    ),
                    **metrics,
                }
            )
            progress.update(1)
    progress.close()
    if not rows:
        raise ValueError("No comparison cells were produced")
    return _canonicalize_symmetric_metrics(pd.DataFrame(rows))


def _validate_preflight(config, spaces, jobs, evaluation) -> dict:
    """Resolve every required row/factor/fold combination before scoring."""
    views = {
        space.key: evaluation.load_view(space.model, space.layer, space.offset)
        for space in spaces
    }
    required_ranks = {index: set() for index in range(len(spaces))}
    for job in jobs:
        required_ranks[job.source_index].add(job.source_rank)
        required_ranks[job.target_index].add(job.target_rank)
    for space_index, space in enumerate(spaces):
        view = views[space.key]
        # Self-alignment validates row counts, uniqueness, and finite embeddings.
        align_views(view, view)
        unavailable = required_ranks[space_index] - space.ranks
        if unavailable:
            raise ValueError(
                f"Configured ranks {sorted(unavailable)} are unavailable for "
                f"{space.label}; completed ranks are {sorted(space.ranks)}"
            )
        for rank in sorted(required_ranks[space_index]):
            for fold in range(len(evaluation.fold_test_ids)):
                factor = factor_for(
                    space, rank, fold, config.evaluation_dataset, "span"
                )
                if factor is None or factor.shape != (view.values.shape[1], rank):
                    raise ValueError(
                        f"Invalid rank-{rank} factor for {space.label}: "
                        f"expected {(view.values.shape[1], rank)}, got "
                        f"{None if factor is None else factor.shape}"
                    )
                if (
                    "weighted" in config.geometries
                    and space.dataset == config.evaluation_dataset
                    and space.source_format == "fold_factors"
                ):
                    weighted = factor_for(
                        space, rank, fold, config.evaluation_dataset, "weighted"
                    )
                    if weighted is None or weighted.shape != (view.values.shape[1], rank):
                        raise ValueError(
                            f"Invalid weighted rank-{rank} factor for {space.label}"
                        )
    minimum_train, minimum_test = None, None
    for source_index, source in enumerate(spaces):
        for target_index, target in enumerate(spaces):
            if source_index == target_index:
                continue
            common_ids, _, _ = align_views(views[source.key], views[target.key])
            for fold in range(len(evaluation.fold_test_ids)):
                train_idx, test_idx = fold_indices(
                    common_ids, source, target, evaluation, fold
                )
                minimum_train = (
                    len(train_idx)
                    if minimum_train is None
                    else min(minimum_train, len(train_idx))
                )
                minimum_test = (
                    len(test_idx)
                    if minimum_test is None
                    else min(minimum_test, len(test_idx))
                )
    return {"minimum_train_rows": minimum_train, "minimum_test_rows": minimum_test}


def _matrix(summary, labels, geometry, metric, equal_capacity_rank=None):
    selected = summary[summary.geometry == geometry]
    if equal_capacity_rank is not None:
        selected = selected[selected.equal_capacity_rank == equal_capacity_rank]
    means = np.full((len(labels), len(labels)), np.nan)
    sems = np.full_like(means, np.nan)
    counts = np.zeros_like(means, dtype=int)
    positions = {label: i for i, label in enumerate(labels)}
    for row in selected.itertuples():
        i, j = positions[row.source_label], positions[row.target_label]
        means[i, j] = getattr(row, f"{metric}_mean")
        sems[i, j] = getattr(row, f"{metric}_sem")
        counts[i, j] = getattr(row, f"{metric}_n")
    return means, sems, counts


def _rank_matrix(summary, labels, geometry, field, equal_capacity_rank=None):
    selected = summary[summary.geometry == geometry]
    if equal_capacity_rank is not None:
        selected = selected[selected.equal_capacity_rank == equal_capacity_rank]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    positions = {label: i for i, label in enumerate(labels)}
    for row in selected.itertuples():
        matrix[positions[row.source_label], positions[row.target_label]] = int(
            getattr(row, field)
        )
    return matrix


def _group_boundaries(spaces):
    boundaries = []
    for index in range(1, len(spaces)):
        previous, current = spaces[index - 1], spaces[index]
        if previous.dataset != current.dataset:
            boundaries.append((index - 0.5, 2.8, "black"))
        elif previous.offset != current.offset:
            boundaries.append((index - 0.5, 2.0, "#333333"))
        elif previous.model != current.model:
            boundaries.append((index - 0.5, 1.2, "white"))
    return boundaries


def _plot_matrix(matrix, spaces, metric, geometry, mode, rank):
    labels = [space.label for space in spaces]
    size = max(9.0, min(28.0, 0.34 * len(labels) + 4.0))
    fig, ax = plt.subplots(figsize=(size, size))
    finite = matrix[np.isfinite(matrix)]
    if finite.size:
        lo, hi = np.nanpercentile(finite, [2, 98])
        if lo < 0 < hi:
            bound = max(abs(lo), abs(hi))
            vmin, vmax, cmap = -bound, bound, "coolwarm"
        else:
            vmin, vmax, cmap = lo, hi if hi > lo else lo + 1.0, "viridis"
    else:
        vmin, vmax, cmap = 0.0, 1.0, "viridis"
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#d9d9d9")
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap_obj, aspect="equal")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    for boundary, linewidth, color in _group_boundaries(spaces):
        ax.axhline(boundary, color=color, linewidth=linewidth)
        ax.axvline(boundary, color=color, linewidth=linewidth)
    ax.set_xlabel("Target / reconstructed space")
    ax.set_ylabel("Source space")
    capacity = f", rank {rank}" if rank is not None else ""
    ax.set_title(f"{metric} — {mode}, {geometry}{capacity}")
    fig.colorbar(image, ax=ax, shrink=0.72)
    fig.tight_layout()
    return fig


def _save_matrices_and_plots(artifacts, summary, spaces, mode, config):
    labels = [space.label for space in spaces]
    ranks = (
        sorted(summary.equal_capacity_rank.dropna().astype(int).unique())
        if mode == "equal_capacity"
        else [None]
    )
    for geometry in config.geometries:
        for rank in ranks:
            for metric in ALL_METRICS:
                mean, sem_values, counts = _matrix(
                    summary, labels, geometry, metric, rank
                )
                source_ranks = _rank_matrix(
                    summary, labels, geometry, "source_rank", rank
                )
                target_ranks = _rank_matrix(
                    summary, labels, geometry, "target_rank", rank
                )
                rank_suffix = f"__k{rank}" if rank is not None else ""
                stem = f"{sanitize_name(metric)}__{geometry}{rank_suffix}"
                for source_index, artifact in artifacts.items():
                    selector = {
                        "comparison_mode": mode, "geometry": geometry,
                        "metric": metric,
                    }
                    if rank is not None:
                        selector["equal_capacity_rank"] = int(rank)
                    payload = {
                        "target_labels": np.asarray(labels),
                        "mean": mean[source_index], "sem": sem_values[source_index],
                        "n_folds": counts[source_index],
                        "source_ranks": source_ranks[source_index],
                        "target_ranks": target_ranks[source_index],
                        "symmetric": metric in SYMMETRIC_METRICS,
                    }
                    artifact.save_field(
                        "matrices", payload, selector=selector,
                        overwrite=config.refresh,
                    )
                    width = max(9.0, min(28.0, 0.34 * len(labels) + 4.0))
                    figure, axis = plt.subplots(figsize=(width, 2.8))
                    image = axis.imshow(mean[[source_index]], aspect="auto", cmap="viridis")
                    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
                    axis.set_yticks([0], [spaces[source_index].label], fontsize=7)
                    axis.set_title(f"{metric} — {mode}, {geometry}")
                    figure.colorbar(image, ax=axis, shrink=0.72)
                    figure.tight_layout()
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
                    plt.close(figure)
                    plot_selector = {**selector, "name": stem}
                    artifact.save_field(
                        "plots", buffer.getvalue(), selector=plot_selector,
                        overwrite=config.refresh,
                    )


def _write_matrices_and_plots(summary, spaces, mode, config, output_dir):
    """Compatibility cleanup for legacy CCA outputs; new runs use Artifact fields."""
    for directory_name in ("matrices", "figures"):
        directory = Path(output_dir) / directory_name
        if not directory.exists():
            continue
        for path in directory.glob("cca_*"):
            if path.is_file() and path.suffix in {".csv", ".npz", ".png", ".svg"}:
                path.unlink()


def run_config(config: SubspaceComparisonConfig) -> dict[int, Artifact]:
    configure_metric_backend("cuda")
    spaces = expand_spaces(config)
    equal_ranks = (
        shared_ranks(spaces, config.max_rank)
        if "equal_capacity" in config.comparison_modes
        else []
    )
    jobs = generate_comparison_jobs(config, spaces, equal_ranks)
    evaluation = load_evaluation_data(config, spaces)
    fold_counts = {
        len(space.heldout_ids)
        for space in spaces
        if space.heldout_ids is not None
    }
    if fold_counts and fold_counts != {len(evaluation.fold_test_ids)}:
        raise ValueError(
            "Evaluation-dataset spaces expose inconsistent fold counts: "
            f"{sorted(fold_counts)} versus reference {len(evaluation.fold_test_ids)}"
        )
    validation = _validate_preflight(config, spaces, jobs, evaluation)
    upstreams = []
    for space in spaces:
        identity = space.artifact.identity_record()
        if identity not in upstreams:
            upstreams.append(identity)
    artifacts = {}
    for index, space in enumerate(spaces):
        artifact = Artifact.from_space(config, space).begin(
            config=config, upstreams=upstreams, refresh=config.refresh
        )
        artifact.save_artifacts(
            config=config, upstreams=upstreams,
            source_metadata=_space_metadata(space),
            validation={
                **validation, "evaluation": evaluation.metadata,
                "equal_capacity_ranks": equal_ranks,
                "n_comparison_jobs": len(jobs), "n_spaces": len(spaces),
                "metric_implementation": METRIC_IMPLEMENTATION,
                "metric_device": "cuda",
            },
        )
        artifacts[index] = artifact
    frame = _run_comparisons(config, spaces, jobs, evaluation, artifacts)
    for mode in config.comparison_modes:
        mode_frame = frame[frame.comparison_mode == mode].copy()
        summary = summarize_fold_metrics(mode_frame)
        _save_matrices_and_plots(artifacts, summary, spaces, mode, config)
    summary = summarize_fold_metrics(frame)
    for index, artifact in artifacts.items():
        source = spaces[index]
        source_rows = frame[
            (frame.source_dataset == source.dataset)
            & (frame.source_offset == source.offset)
            & (frame.source_model == source.model)
            & (frame.source_layer == source.layer)
            & (frame.source_kind == source.kind)
        ].reset_index(drop=True)
        source_summary = summary[
            (summary.source_dataset == source.dataset)
            & (summary.source_offset == source.offset)
            & (summary.source_model == source.model)
            & (summary.source_layer == source.layer)
            & (summary.source_kind == source.kind)
        ].reset_index(drop=True)
        artifact.save_field("fold_metrics", source_rows, overwrite=config.refresh)
        artifact.save_field("summary", source_summary, overwrite=config.refresh)
        selectors = [entry["selector"] for entry in
                     artifact.manifest()["entries"]["cell_metrics"].values()]
        artifact.mark_complete({"cell_metrics": selectors})
    return artifacts


def main() -> None:
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, SubspaceComparisonConfig)
    artifacts = run_config(config)
    print(f"Saved {len(artifacts)} source-organized subspace Artifacts")


if __name__ == "__main__":
    main()
