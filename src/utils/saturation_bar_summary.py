"""Discovery and plotting helpers for cross-model saturation summaries."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd
import yaml


DATASET_LABELS = {
    "podcast": "Podcast",
    "brain_treebank": "Brain Treebank",
    "fmri": "fMRI",
}
DATASET_ABBREVIATIONS = {
    "podcast": "P",
    "brain_treebank": "BT",
    "fmri": "fMRI",
}
MODEL_LABELS = {
    "glove": "GloVe",
    "gpt2": "GPT-2",
    "gpt2-xl": "GPT-2 XL",
    "gpt-j-6b": "GPT-J 6B",
    "llama3-8b": "Llama 3 8B",
    "roberta-large": "RoBERTa Large",
    "random-gpt2-xl": "Random GPT-2 XL",
}
MODEL_ORDER = list(MODEL_LABELS)
MODEL_COLORS = {
    "glove": "#7f7f7f",
    "gpt2": "#4c78a8",
    "gpt2-xl": "#2f5597",
    "gpt-j-6b": "#59a14f",
    "llama3-8b": "#f28e2b",
    "roberta-large": "#b07aa1",
    "random-gpt2-xl": "#e15759",
}
KNOWN_DIMS = {
    "glove": 300,
    "gpt2": 768,
    "gpt2-xl": 1600,
    "gpt-j-6b": 4096,
    "llama3-8b": 4096,
    "roberta-large": 1024,
    "random-gpt2-xl": 1600,
}


def normalize_model(name: str) -> str:
    value = str(name).lower().replace("gpt2-xl-batch", "gpt2-xl")
    if value.startswith("glove"):
        return "glove"
    return value


def dataset_from_path(path: str | Path) -> str:
    text = str(path).lower()
    if "/fmri/" in f"/{text.strip('/')}/" or text.startswith("fmri/"):
        return "fmri"
    if "movie-cars-2" in text or "brain-treebank" in text or "cars-2" in text:
        return "brain_treebank"
    return "podcast"


def _rrr_config_path(run: Path, method: str) -> Path | None:
    candidates = [run / "config.yml"]
    if method == "pca":
        parts = list(run.parts)
        candidates.append(run.parent / "rrr" / "config.yml")
        for index, part in enumerate(parts):
            if part == "cross_dataset_pca":
                replaced = parts.copy()
                replaced[index] = "cross_dataset_rrr"
                candidates.append(Path(*replaced) / "config.yml")
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_config(run: Path, method: str) -> tuple[dict, Path | None]:
    path = _rrr_config_path(run, method)
    if path is None:
        return {}, None
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}, path


def _scalar(value):
    if isinstance(value, list):
        return value[0] if len(value) == 1 else None
    return value


def infer_model_layer(run: Path, config: dict) -> tuple[str, int, str]:
    if "fmri" in config:
        model = config.get("model_dir")
        layer = _scalar(config.get("layer"))
    else:
        per_model = config.get("encoding", {}).get("per_model_config", {})
        model = per_model.get("model_dir")
        layer = _scalar(per_model.get("layer"))
    if model is not None and layer is not None:
        return normalize_model(model), int(layer), "config"

    for part in run.parts:
        match = re.match(r"(.+)_layer(\d+)(?:_offset-?\d+)?$", part)
        if match:
            return normalize_model(match.group(1)), int(match.group(2)), "path"
        match = re.match(r"glove_dim(\d+)_offset-?\d+$", part)
        if match:
            return "glove", 0, "path"
    raise ValueError(f"Could not infer model/layer from {run}")


def infer_source_target(run: Path, config: dict) -> tuple[str, str, str]:
    if "fmri" in config:
        target = "fmri"
        if config.get("source_mode") == "cross_dataset":
            source_path = config.get("source_rrr_unit_dir")
            if not source_path:
                raise ValueError(f"Missing source_rrr_unit_dir in fMRI config for {run}")
            source = dataset_from_path(source_path)
        else:
            source = target
        return source, target, "config"

    encoding = config.get("encoding", {})
    if encoding:
        target = str(encoding.get("dataset", "podcast"))
        if target == "brain_treebank":
            target = "brain_treebank"
        if str(config.get("mode")) == "cross_dataset":
            source_path = config.get("cross_result_dir")
            if not source_path:
                raise ValueError(f"Missing cross_result_dir in config for {run}")
            source = dataset_from_path(source_path)
        else:
            source = target
        return source, target, "config"

    text = str(run)
    if "/fmri/" in f"/{text.strip('/')}/":
        target = "fmri"
        name = run.name.lower()
        if "from_podcast" in name or name == "cross_dataset_rrr":
            return "podcast", target, "path"
        if "from_cars" in name or "cars2" in name:
            return "brain_treebank", target, "path"
        return target, target, "path"
    cross_index = next(
        (i for i, part in enumerate(run.parts) if part in {"cross_dataset_rrr", "cross_dataset_pca"}),
        None,
    )
    if cross_index is None:
        target = dataset_from_path(run)
        return target, target, "path"
    target_root = Path(*run.parts[:cross_index])
    source_name = Path(*run.parts[cross_index + 1 :])
    return dataset_from_path(source_name), dataset_from_path(target_root), "path"


def factor_dimension(run: Path, method: str, layout: str) -> int | None:
    if layout == "ieeg":
        paths = sorted(run.glob("fold_*/A.npy"))
    else:
        paths = sorted((run / method).glob("k*/A.npy"))
    if not paths:
        return None
    return int(np.load(paths[0], mmap_mode="r").shape[0])


def build_saturation_bar_table(
    summary: pd.DataFrame,
    *,
    saturation_column: str = "saturation_rank",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the rank summary into one auditable row per plotted bar."""
    rows, skipped = [], []
    for saved in summary.to_dict("records"):
        run = Path(saved["run"])
        method = str(saved["method"])
        try:
            if run.name == "within_fmri_rrr_unnormalized" or "multi_view" in run.name:
                raise ValueError("diagnostic run excluded from primary summary")
            config, config_path = _load_config(run, method)
            model, layer, model_source = infer_model_layer(run, config)
            source, target, context_source = infer_source_target(run, config)
            dimension = factor_dimension(run, method, str(saved["layout"]))
            dimension_source = "factor"
            if dimension is None:
                dimension = KNOWN_DIMS.get(model)
                dimension_source = "known_model_dimension"
            if dimension is None:
                raise ValueError(f"Could not infer ambient dimension for {model}")
            rank = float(saved[saturation_column])
            if not np.isfinite(rank) or rank <= 0:
                raise ValueError(f"Invalid saturation rank {rank}")
            rows.append({
                "run": str(run),
                "method": method,
                "source_dataset": source,
                "target_dataset": target,
                "mode": "within" if source == target else f"from_{source}",
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "layer": layer,
                "saturation_rank": int(rank),
                "ambient_dimension": int(dimension),
                "saturation_percent": 100.0 * rank / dimension,
                "config_path": str(config_path) if config_path else "",
                "model_layer_source": model_source,
                "dataset_context_source": context_source,
                "dimension_source": dimension_source,
            })
        except Exception as error:
            skipped.append({"run": str(run), "method": method, "reason": str(error)})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        duplicates = frame.duplicated(
            ["source_dataset", "target_dataset", "model", "layer", "method"],
            keep=False,
        )
        if duplicates.any():
            duplicate_rows = frame.loc[duplicates]
            raise ValueError(
                "Multiple runs map to the same summary cell:\n"
                + duplicate_rows.to_string(index=False)
            )
    return frame, pd.DataFrame(skipped)


def build_auc_comparison_table(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair cross-dataset AUCs with matching within-target reference runs."""
    required = {"run", "method", "layout", "normalized_log_auc"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"AUC summary is missing columns {sorted(missing)}")

    metadata, skipped = [], []
    for saved in summary.to_dict("records"):
        run = Path(saved["run"])
        method = str(saved["method"])
        try:
            if run.name == "within_fmri_rrr_unnormalized" or "multi_view" in run.name:
                raise ValueError("diagnostic run excluded from AUC comparison")
            config, config_path = _load_config(run, method)
            model, layer, model_source = infer_model_layer(run, config)
            source, target, context_source = infer_source_target(run, config)
            metadata.append({
                **saved,
                "model": model,
                "layer": layer,
                "source_dataset": source,
                "target_dataset": target,
                "config_path": str(config_path) if config_path else "",
                "model_layer_source": model_source,
                "dataset_context_source": context_source,
            })
        except Exception as error:
            skipped.append({"run": str(run), "method": method, "reason": str(error)})

    enriched = pd.DataFrame(metadata)
    comparisons = []
    if enriched.empty:
        return pd.DataFrame(comparisons), pd.DataFrame(skipped)
    references = enriched.loc[enriched.source_dataset == enriched.target_dataset]
    transfers = enriched.loc[enriched.source_dataset != enriched.target_dataset]
    keys = ["target_dataset", "model", "layer", "method"]
    for transfer in transfers.to_dict("records"):
        matches = references
        for key in keys:
            matches = matches.loc[matches[key] == transfer[key]]
        if len(matches) != 1:
            skipped.append({
                "run": transfer["run"],
                "method": transfer["method"],
                "reason": f"expected one matching within-target reference, found {len(matches)}",
            })
            continue
        reference = matches.iloc[0]
        reference_auc = float(reference["normalized_log_auc"])
        if not np.isfinite(reference_auc) or reference_auc <= 0:
            skipped.append({
                "run": transfer["run"],
                "method": transfer["method"],
                "reason": f"reference AUC must be positive, got {reference_auc}",
            })
            continue
        comparisons.append({
            "cross_run": transfer["run"],
            "reference_run": reference["run"],
            "method": transfer["method"],
            "source_dataset": transfer["source_dataset"],
            "target_dataset": transfer["target_dataset"],
            "model": transfer["model"],
            "layer": int(transfer["layer"]),
            "cross_normalized_log_auc": float(transfer["normalized_log_auc"]),
            "reference_normalized_log_auc": reference_auc,
            "cross_dataset_effectiveness": (
                float(transfer["normalized_log_auc"]) / reference_auc
            ),
        })
    return pd.DataFrame(comparisons), pd.DataFrame(skipped)


def summarize_auc_distances(
    comparisons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize clipped ``1 - effectiveness`` across model/layer runs."""
    required = {
        "method", "source_dataset", "target_dataset", "model", "layer",
        "cross_dataset_effectiveness",
    }
    missing = required.difference(comparisons.columns)
    if missing:
        raise ValueError(f"AUC comparisons are missing columns {sorted(missing)}")
    values = comparisons.copy()
    effectiveness = pd.to_numeric(
        values["cross_dataset_effectiveness"], errors="coerce"
    )
    if effectiveness.isna().any():
        raise ValueError("cross_dataset_effectiveness must be finite")
    values["distance"] = np.maximum(0.0, 1.0 - effectiveness)
    group_columns = ["method", "source_dataset", "target_dataset"]
    summary = (
        values.groupby(group_columns, sort=True)["distance"]
        .agg(
            median_distance="median",
            q25_distance=lambda series: series.quantile(0.25),
            q75_distance=lambda series: series.quantile(0.75),
            n_model_layers="size",
        )
        .reset_index()
    )
    return values, summary


def plot_asymmetric_auc_distances(
    values: pd.DataFrame, summary: pd.DataFrame, method: str, path: Path
) -> None:
    """Plot both directed distances in diagonally split lower-triangle cells."""
    values = values.loc[values.method == method]
    summary = summary.loc[summary.method == method]
    if summary.empty:
        raise ValueError(f"No {method} AUC distances to plot")
    preferred = ["podcast", "brain_treebank", "fmri"]
    present = set(values.source_dataset) | set(values.target_dataset)
    datasets = [item for item in preferred if item in present]
    datasets.extend(sorted(present.difference(datasets)))
    positions = {dataset: index for index, dataset in enumerate(datasets)}
    vmax = max(1.0, float(values["distance"].max()))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("magma_r")
    fig, ax = plt.subplots(
        figsize=(max(6.8, 2.25 * len(datasets)), max(5.8, 1.9 * len(datasets)))
    )

    def directed(source: str, target: str):
        match = summary.loc[
            (summary.source_dataset == source) & (summary.target_dataset == target)
        ]
        return None if match.empty else match.iloc[0]

    def draw_half(vertices, row, center, direction, source, target):
        if row is None:
            ax.add_patch(Polygon(vertices, facecolor="#eeeeee", edgecolor="white", hatch="///"))
            ax.text(*center, f"{direction}\nmissing", ha="center", va="center", fontsize=7)
            return
        median = float(row.median_distance)
        ax.add_patch(Polygon(vertices, facecolor=cmap(norm(median)), edgecolor="white"))
        ax.text(
            *center,
            f"{direction}\n{median:.2f} [{row.q25_distance:.2f}, {row.q75_distance:.2f}]\nn={int(row.n_model_layers)}",
            ha="center", va="center", fontsize=7,
            color="white" if norm(median) > 0.48 else "black",
        )
        points = values.loc[
            (values.source_dataset == source) & (values.target_dataset == target)
        ].sort_values(["model", "layer"])
        offsets = np.linspace(-0.13, 0.13, len(points)) if len(points) > 1 else [0.0]
        for offset, distance in zip(offsets, points.distance):
            ax.scatter(
                center[0] + offset, center[1] + 0.27,
                s=13, c=[cmap(norm(float(distance)))], edgecolors="white",
                linewidths=0.45, zorder=5,
            )

    for row_index in range(1, len(datasets)):
        for column_index in range(row_index):
            row_dataset = datasets[row_index]
            column_dataset = datasets[column_index]
            row_short = DATASET_ABBREVIATIONS.get(row_dataset, row_dataset)
            column_short = DATASET_ABBREVIATIONS.get(column_dataset, column_dataset)
            x, y = column_index, row_index
            # With rows interpreted as targets and columns as sources, the
            # lower-right half is column -> row and the upper-left is row -> column.
            draw_half(
                [(x - .5, y - .5), (x + .5, y - .5), (x + .5, y + .5)],
                directed(column_dataset, row_dataset),
                (x + .17, y - .13),
                f"{column_short}→{row_short}",
                column_dataset, row_dataset,
            )
            draw_half(
                [(x - .5, y - .5), (x - .5, y + .5), (x + .5, y + .5)],
                directed(row_dataset, column_dataset),
                (x - .17, y + .13),
                f"{row_short}→{column_short}",
                row_dataset, column_dataset,
            )

    labels = [DATASET_LABELS.get(dataset, dataset) for dataset in datasets]
    ax.set_xticks(range(len(datasets)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(datasets)), labels)
    ax.set_xlim(-.55, len(datasets) - .45)
    ax.set_ylim(len(datasets) - .45, -.55)
    ax.set_xlabel("Dataset pair (split cells show both directions)")
    ax.set_ylabel("Dataset pair")
    ax.set_title(
        f"Asymmetric cross-dataset distance — {method.upper()}\n"
        "median [IQR] across model/layer runs; dots are individual runs"
    )
    for index in range(len(datasets)):
        ax.text(index, index, "within\ndataset", ha="center", va="center", color="#777777", fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=.72, pad=.03)
    colorbar.set_label("Distance = max(0, 1 − effectiveness)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_saturation_bars(frame: pd.DataFrame, path: Path) -> None:
    """Plot PCA as a wide cross-hatched background bar behind solid RRR."""
    models = sorted(
        frame["model"].unique(),
        key=lambda value: (
            MODEL_ORDER.index(value) if value in MODEL_ORDER else len(MODEL_ORDER),
            value,
        ),
    )
    positions, ticks, tick_labels, model_centers, cursor = {}, [], [], {}, 0.0
    for model in models:
        layers = sorted(frame.loc[frame.model == model, "layer"].unique())
        group_positions = []
        for layer in layers:
            positions[(model, int(layer))] = cursor
            ticks.append(cursor)
            tick_labels.append("emb." if model == "glove" else str(int(layer)))
            group_positions.append(cursor)
            cursor += 1.0
        model_centers[model] = float(np.mean(group_positions))
        cursor += 0.8

    fig_width = max(7.0, 0.42 * len(ticks) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    positive = frame.loc[frame.saturation_percent > 0, "saturation_percent"]
    for method, hatch, zorder in (("pca", "xxxx", 1), ("rrr", None, 2)):
        for row in frame.loc[frame.method == method].itertuples(index=False):
            x = positions[(row.model, int(row.layer))]
            color = MODEL_COLORS.get(row.model, "#4c78a8")
            if method == "pca":
                ax.bar(
                    x, row.saturation_percent,
                    width=0.7, facecolor="white", edgecolor=color,
                    hatch=hatch, linewidth=1.0, zorder=zorder,
                )
            else:
                ax.bar(
                    x, row.saturation_percent,
                    width=0.7, color=color, edgecolor="white",
                    linewidth=0.5, zorder=zorder,
                )

    ax.set_ylim(0, max(105, positive.max() * 1.1))
    percentage_ticks = [0, 20, 40, 60, 80, 100]
    ax.set_yticks(percentage_ticks)
    ax.set_yticklabels([f"{value}%" for value in percentage_ticks])
    ax.set_xticks(ticks, tick_labels, rotation=45, ha="right")
    ax.set_xlabel("Layer (earliest to latest within model)")
    ax.set_ylabel("Dimensionality at saturation (%)")
    ax.grid(axis="y", which="both", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for model, center in model_centers.items():
        ax.text(
            center, -0.26, MODEL_LABELS.get(model, model),
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9,
        )
    source = frame["source_dataset"].iloc[0]
    target = frame["target_dataset"].iloc[0]
    ax.set_title(
        f"{DATASET_LABELS.get(target, target)} — "
        + ("within dataset" if source == target else f"from {DATASET_LABELS.get(source, source)}")
    )
    ax.legend(
        handles=[
            Patch(facecolor="#666666", edgecolor="white", label="RRR"),
            Patch(facecolor="white", edgecolor="#666666", hatch="xxxx", label="PCA"),
        ],
        frameon=False, loc="upper right",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(bottom=0.28, left=0.11, right=0.98, top=0.9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
