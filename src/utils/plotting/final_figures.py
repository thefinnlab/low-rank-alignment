"""Compositions for the curated paper figures."""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.utils.plotting.final_axes import (
    COLORS, add_decoding_gradient, correlation_scatter_axis, decoding_relative_axis,
    distribution_axis, finish_axis,
    format_p_value,
    normalized_curve_axis, pairwise_test_brackets, paper_style, quantile_axis, rank_curve_axis,
    target_bar_axis,
)
from src.utils.plotting.final_data import DATASETS, LABELS
from src.utils.rrr.plot_metrics import (
    PEAK_FLOOR, QUANTILES, average_normalized_quantiles, fisher_mean,
    normalized_log_rank_auc, paired_sign_flip, quantile_saturation_rank,
)


def _layer(value):
    return None if pd.isna(value) else int(value)


def _pair_rows(frame, model, layer):
    selected = frame[frame["model"].eq(model)]
    return selected[selected["layer"].isna() if layer is None else selected["layer"].eq(layer)]


def _curves(frame):
    result = {}
    for method, group in frame.groupby("method"):
        grouped = group.groupby("rank")["correlation"].agg(fisher_mean).sort_index()
        result[method] = (grouped.index.to_numpy(), grouped.to_numpy())
    return result


def _members(frame):
    members = defaultdict(list)
    for (_, _, method), group in frame.groupby(["model", "layer", "method"], dropna=False):
        if "subject" in group:
            group = group.groupby(["method", "rank"], as_index=False)["correlation"].agg(fisher_mean)
        members[method].append(_curves(group)[method])
    return members


def _saturation_values(quantiles, dataset, methods, fraction, *, min_peak=PEAK_FLOOR):
    values = defaultdict(list)
    for key, (ranks, matrix) in quantiles.items():
        if key[0] != dataset or key[-1] not in methods:
            continue
        rank = quantile_saturation_rank(ranks, matrix, fraction, min_peak)
        if np.isfinite(rank):
            values[key[-1]].append((key[1:3], rank))
    return values


def _saturation_comparison_axis(ax, ranks, methods, labels, tests, *,
                                show_mean_value=False):
    """Draw paired model-layer ranks with the aggregate comparison style."""
    rank_by_pair = {method: dict(ranks.get(method, [])) for method in methods}
    common = set.intersection(*(set(values) for values in rank_by_pair.values()))
    paired = [
        {label: rank_by_pair[method][pair]
         for method, label in zip(methods, labels)}
        for pair in sorted(common, key=str)
    ]
    distribution_axis(ax, {
        label: [row[label] for row in paired] for label in labels
    }, log2_rank=True, paired=paired, show_mean_value=show_mean_value)
    if not paired:
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_xlim(.45, len(labels) + .55)
        return
    p_values = np.asarray([
        paired_sign_flip(
            [np.log2(row[first]) for row in paired],
            [np.log2(row[second]) for row in paired],
        )
        for first, second in tests
    ])
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    adjusted[order] = np.maximum.accumulate(np.minimum(
        1.0, (len(p_values) - np.arange(len(p_values))) * p_values[order]
    ))
    pairwise_test_brackets(ax, labels, [
        (first, second, p) for (first, second), p in zip(tests, adjusted)
    ])


def _loso_saturation_values(quantiles, dataset, fraction):
    by_pair = defaultdict(list)
    for key, (ranks, matrix) in quantiles.items():
        if key[0] != dataset:
            continue
        rank = quantile_saturation_rank(ranks, matrix, fraction)
        if np.isfinite(rank):
            by_pair[(key[1], key[2], key[4])].append(rank)
    values = defaultdict(list)
    for (model, layer, method), ranks in by_pair.items():
        values[method].append(((model, layer), np.mean(ranks)))
    return values


def rrr_aggregate(internal, loso, quantiles, loso_quantiles, full_rows,
                  selected_pair, *, mode="internal", fraction=.99):
    """Two stacked curve columns beside three full-height rank distributions."""
    with paper_style():
        figure = plt.figure(figsize=(19, 7.4))
        grid = figure.add_gridspec(
            3, 3, width_ratios=[1.25, 1.25, 3],
            hspace=.52, wspace=.35,
        )
        saturation_grid = grid[:, 2].subgridspec(1, 3, wspace=.10)
        source = internal if mode == "internal" else loso
        methods = ["rrr", "pca"] if mode == "internal" else [
            "RRR baseline", "RRR transfer", "PCA baseline"]
        baseline_map = {(d, m, l): float(fisher_mean(v)) for d, m, l, v in full_rows}
        first_saturation_ax = None
        for row, dataset in enumerate(DATASETS):
            frame = source[source["dataset"] == dataset]
            sample = _pair_rows(frame, *selected_pair)
            sample_curves = _curves(sample)
            if mode == "internal":
                ranks = _saturation_values(quantiles, dataset, methods, fraction)
                markers = {
                    method: quantile_saturation_rank(*quantiles[(dataset, *selected_pair, method)], fraction)
                    for method in methods if (dataset, *selected_pair, method) in quantiles
                }
            else:
                ranks = _loso_saturation_values(loso_quantiles, dataset, fraction)
                markers = {}
                for method in methods:
                    subject_ranks = [quantile_saturation_rank(*value, fraction)
                                     for key, value in loso_quantiles.items()
                                     if key[:3] == (dataset, *selected_pair) and key[-1] == method]
                    finite = [value for value in subject_ranks if np.isfinite(value)]
                    if finite:
                        markers[method] = float(np.mean(finite))
            sample_ax = figure.add_subplot(grid[row, 0])
            rank_curve_axis(sample_ax, sample_curves, saturation=markers,
                            baseline=baseline_map.get((dataset, *selected_pair)))
            sample_ax.set_title(LABELS[dataset])
            if row == 0:
                sample_ax.legend(frameon=False, ncol=2, fontsize=9,
                                 handlelength=1.2, columnspacing=.7)
            if row != 2:
                sample_ax.set_xlabel("")
            if row != 1:
                sample_ax.set_ylabel("")
            mean_ax = figure.add_subplot(grid[row, 1])
            normalized_curve_axis(mean_ax, _members(frame))
            mean_ax.set_title(LABELS[dataset])
            if row == 0:
                mean_ax.legend(frameon=False, ncol=2, fontsize=9,
                               handlelength=1.2, columnspacing=.7)
            mean_ax.set_xticks([0, .01, .1, 1], ["0", "1%", "10%", "100%"])
            if row != 2:
                mean_ax.set_xlabel("")
            if row != 1:
                mean_ax.set_ylabel("")
            distribution_ax = figure.add_subplot(
                saturation_grid[0, row], sharey=first_saturation_ax,
            )
            if first_saturation_ax is None:
                first_saturation_ax = distribution_ax
            labels = [method.upper() if mode == "internal" else method for method in methods]
            tests = [("RRR", "PCA")] if mode == "internal" else [
                ("RRR transfer", "PCA baseline"),
                ("RRR baseline", "RRR transfer"),
            ]
            _saturation_comparison_axis(distribution_ax, ranks, methods, labels, tests,
                                        show_mean_value=True)
            if mode == "loso":
                distribution_ax.set_xticklabels(
                    ["RRR\nbaseline", "RRR\ntransfer", "PCA\nbaseline"],
                    fontsize=10,
                )
            if row != 0:
                distribution_ax.set_ylabel("")
                distribution_ax.tick_params(axis="y", labelleft=False)
            distribution_ax.set_title(LABELS[dataset])
        figure.subplots_adjust(left=.055, right=.985, top=.89, bottom=.10)
        sample_box = sample_ax.get_position()
        mean_box = mean_ax.get_position()
        saturation_start = first_saturation_ax.get_position()
        saturation_end = distribution_ax.get_position()
        figure.text((sample_box.x0 + sample_box.x1) / 2, .965,
                    "A. Sample Rank Curves", ha="center",
                    va="bottom", fontsize=15, weight="bold")
        figure.text((mean_box.x0 + mean_box.x1) / 2, .965,
                    "B. Mean Convergence", ha="center",
                    va="bottom", fontsize=15, weight="bold")
        figure.text((saturation_start.x0 + saturation_end.x1) / 2, .965,
                    f"C. Saturation Distributions ({fraction * 100:g}% Peak Performance)",
                    ha="center", va="bottom", fontsize=15, weight="bold")
        return figure


def _cross_auc(frame):
    aucs = defaultdict(dict)
    for (model, layer, method), group in frame.groupby(
        ["model", "layer", "method"], dropna=False
    ):
        value = normalized_log_rank_auc(group["rank"], group["correlation"])
        if np.isfinite(value):
            aucs[(model, _layer(layer))][method] = value
    return aucs


def cross_aggregate(cross):
    """Six directed panels and the three current normalized-AUC comparisons."""
    with paper_style():
        figure = plt.figure(figsize=(21, 10.5))
        grid = figure.add_gridspec(3, 6)
        left = grid[:, :3].subgridspec(2, 3, hspace=.28, wspace=.10)
        right = grid[:, 3:].subgridspec(3, 1, hspace=.30)
        comparisons = [
            ("Within RRR − Transfer RRR", "Within RRR", "Transfer RRR"),
            ("Transfer RRR − Transfer PCA", "Transfer RRR", "Transfer PCA"),
            ("Within PCA − Transfer PCA", "Within PCA", "Transfer PCA"),
        ]
        groups = [defaultdict(list) for _ in comparisons]
        first_directed_ax = None
        for index, (target, source) in enumerate(
            (t, s) for t in DATASETS for s in DATASETS if s != t
        ):
            panel = cross[(cross["target_dataset"] == target)
                          & (cross["source_dataset"] == source)]
            ax = figure.add_subplot(
                left[index % 2, index // 2], sharey=first_directed_ax,
            )
            if first_directed_ax is None:
                first_directed_ax = ax
            normalized_curve_axis(ax, _members(panel))
            ax.set_title(f"{LABELS[source]} → {LABELS[target]}", fontsize=13)
            ax.set_xlabel("")
            if index // 2 != 0:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xticks([0, .01, .1, 1], ["0", "1%", "10%", "100%"])
            if index % 2 == 1 and index // 2 > 0:
                ax.set_xticks([0, .01, .1, 1], ["", "1%", "10%", "100%"])
            ax.tick_params(axis="both", labelsize=10)
            ax.tick_params(axis="x", labelbottom=index % 2 == 1)
            if index == 0:
                ax.legend(frameon=False, fontsize=8, ncol=2,
                          handlelength=1.2, columnspacing=.7)
            label = f"{LABELS[source]} → {LABELS[target]}"
            for values in _cross_auc(panel).values():
                for output, (_, first, second) in zip(groups, comparisons):
                    if first in values and second in values:
                        output[label].append(values[first] - values[second])
        for row, (output, (title, _, _)) in enumerate(zip(groups, comparisons)):
            ax = figure.add_subplot(right[row, 0])
            distribution_axis(ax, output, show_points=True)
            ax.axhline(0, color="#111827", linestyle="--", linewidth=1)
            ax.set_title(title, fontsize=14)
            ax.set_ylabel("Normalized AUC difference" if row == 1 else "")
            ax.tick_params(axis="x", labelbottom=row == 2, labelsize=10)
            if row == 2:
                plt.setp(ax.get_xticklabels(), rotation=28, ha="right", fontsize=10)
                ax.set_xlabel("Directed source → target comparison", fontsize=13)
            for position, values in enumerate(output.values(), start=1):
                if values:
                    p = paired_sign_flip(values, np.zeros(len(values)), alternative="greater")
                    label = "*" if p < .05 else "n.s."
                    ax.text(position, .97, f"{label} {format_p_value(p)}",
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=10)
        figure.subplots_adjust(left=.045, right=.99, top=.88, bottom=.15,
                               hspace=.48, wspace=.42)
        left_start, left_end = (grid[0, column].get_position(figure) for column in (0, 2))
        right_start, right_end = (grid[0, column].get_position(figure) for column in (3, 5))
        figure.text((left_start.x0 + left_end.x1) / 2, .965,
                    "A. Directed Transfer Curves", ha="center", va="bottom",
                    fontsize=15, weight="bold")
        figure.text((right_start.x0 + right_end.x1) / 2, .965,
                    "B. Normalized AUC Differences", ha="center", va="bottom",
                    fontsize=15, weight="bold")
        figure.text((left_start.x0 + left_end.x1) / 2, .075,
                    "Relative rank", ha="center", fontsize=13)
        return figure


def cross_gap_scatter(cross):
    """Correlate matched internal-minus-transfer PCA and RRR AUC gaps."""
    with paper_style():
        figure, ax = plt.subplots(figsize=(10.5, 7))
        all_points = []
        colors = plt.get_cmap("tab10").colors
        for index, (target, source) in enumerate(
            (t, s) for t in DATASETS for s in DATASETS if s != t
        ):
            panel = cross[(cross["target_dataset"] == target)
                          & (cross["source_dataset"] == source)]
            points = np.asarray([
                (item["Within PCA"] - item["Transfer PCA"],
                 item["Within RRR"] - item["Transfer RRR"])
                for item in _cross_auc(panel).values()
                if {"Within PCA", "Transfer PCA", "Within RRR", "Transfer RRR"} <= item.keys()
            ]).reshape(-1, 2)
            if len(points):
                all_points.append(points)
                ax.scatter(points[:, 0], points[:, 1], s=35, alpha=.72,
                           color=colors[index], edgecolors="white", linewidths=.4,
                           label=f"{LABELS[source]} → {LABELS[target]} (n={len(points)})")
        if all_points:
            points = np.vstack(all_points)
            lo, hi = min(0, points.min()), max(0, points.max())
            pad = max((hi - lo) * .06, .01)
            limits = (lo - pad, hi + pad)
            ax.plot(limits, limits, color="#94a3b8", linestyle="--", label="Equal gaps")
            if len(points) >= 2 and np.unique(points[:, 0]).size > 1:
                slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
                ax.plot(limits, slope * np.asarray(limits) + intercept,
                        color="#111827", linewidth=2, label="Pooled linear fit")
            if len(points) >= 2 and np.unique(points[:, 0]).size > 1 and np.unique(points[:, 1]).size > 1:
                from scipy.stats import pearsonr

                correlation = pearsonr(points[:, 0], points[:, 1])
                r, p = correlation.statistic, correlation.pvalue
                p_text = format_p_value(p)
            else:
                r, p_text = np.nan, "p=NA"
            ax.set(xlim=limits, ylim=limits)
            ax.set_title(f"Internal − transfer AUC gaps\nPearson r = {r:.2f}, {p_text}, n = {len(points)}")
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, .5))
        ax.axhline(0, color="#64748b", linewidth=.8)
        ax.axvline(0, color="#64748b", linewidth=.8)
        ax.set_xlabel("Within PCA − transfer PCA normalized AUC")
        ax.set_ylabel("Within RRR − transfer RRR normalized AUC")
        finish_axis(ax)
        figure.subplots_adjust(left=.13, right=.72, top=.88, bottom=.14)
        return figure


def _model_layout(items, order):
    positions, labels, colors, centers = [], [], [], {}
    color_map = {model: plt.get_cmap("tab20")(i / max(len(order) - 1, 1))
                 for i, model in enumerate(order)}
    x = 0.0
    for model in order:
        layers = sorted([layer for name, layer in items if name == model],
                        key=lambda value: (value is not None, value or -1))
        if not layers:
            continue
        start = x
        for layer in layers:
            positions.append(x)
            labels.append("none" if layer is None else str(layer))
            colors.append(color_map[model])
            x += 1
        centers[model] = (start + x - 1) / 2
        x += .8
    return positions, labels, colors, centers, color_map


def model_order(full_rows):
    scores = defaultdict(list)
    for _, model, _, values in full_rows:
        finite = np.asarray(values)[np.isfinite(values)]
        if len(finite):
            scores[model].append(float(np.mean(finite)))
    return sorted(scores, key=lambda model: np.mean(scores[model]))


def _model_grid_style(ax, positions, labels, centers):
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    for model, center in centers.items():
        ax.text(center, -.23, model, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=12)
    finish_axis(ax)


def encoding_grid(full_rows, order):
    with paper_style():
        figure, axes = plt.subplots(3, 1, figsize=(20, 13), sharex=False)
        for ax, dataset in zip(axes, DATASETS):
            entries = {(model, layer): values for d, model, layer, values in full_rows
                       if d == dataset}
            positions, labels, colors, centers, _ = _model_layout(entries, order)
            target_bar_axis(ax, positions,
                            [entries[key] for key in entries_order(entries, order)],
                            colors, error="sd")
            _model_grid_style(ax, positions, labels, centers)
            ax.set_ylabel("Mean correlation ± SD")
            ax.set_title(LABELS[dataset])
        figure.suptitle("Full encoding-model performance")
        figure.subplots_adjust(left=.06, right=.99, top=.93, bottom=.18, hspace=.55)
        return figure


def entries_order(entries, order):
    return [(model, layer) for model in order
            for layer in sorted((layer for name, layer in entries if name == model),
                                key=lambda value: (value is not None, value or -1))]


def significance_grid(significant, order):
    with paper_style():
        order = list(order) + sorted({model for _, model, _ in significant} - set(order))
        figure, axes = plt.subplots(3, 1, figsize=(20, 13))
        for ax, dataset in zip(axes, DATASETS):
            entries = {(model, layer): len(targets) for (d, model, layer), targets
                       in significant.items() if d == dataset}
            positions, labels, colors, centers, _ = _model_layout(entries, order)
            values = [entries[key] for key in entries_order(entries, order)]
            ax.bar(positions, values, color=colors, width=.8)
            _model_grid_style(ax, positions, labels, centers)
            ax.set(ylabel="Significant targets", title=LABELS[dataset])
        figure.suptitle("Significant electrodes or voxels by model layer")
        figure.subplots_adjust(left=.06, right=.99, top=.93, bottom=.18, hspace=.55)
        return figure


def significance_latex(significant, order):
    """Return one table with model-layer counts and dataset union totals."""
    order = list(order) + sorted({model for _, model, _ in significant} - set(order))
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Dataset & Model & Layer & Significant targets \\", r"\midrule"]
    for dataset in DATASETS:
        entries = {(model, layer): targets for (d, model, layer), targets
                   in significant.items() if d == dataset}
        union = set().union(*entries.values()) if entries else set()
        for model, layer in entries_order(entries, order):
            name = model.replace("_", r"\_")
            layer_name = "none" if layer is None else str(layer)
            lines.append(f"{LABELS[dataset]} & {name} & {layer_name} & {len(entries[(model, layer)]):,} " + r"\\")
        lines.extend([r"\midrule", f"{LABELS[dataset]} & Union & -- & {len(union):,} " + r"\\", r"\midrule"])
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def quantile_grid(quantiles, selected_pair, *, fraction=.99):
    with paper_style():
        figure, axes = plt.subplots(3, 2, figsize=(13, 12))
        for row, dataset in enumerate(DATASETS):
            for col, method in enumerate(("rrr", "pca")):
                ax = axes[row, col]
                value = quantiles.get((dataset, *selected_pair, method))
                if value is not None:
                    ranks, matrix = value
                    quantile_axis(ax, ranks, matrix, QUANTILES)
                    sat = quantile_saturation_rank(ranks, matrix, fraction)
                    if np.isfinite(sat):
                        ax.axvline(sat, color=COLORS[method], linestyle="--", linewidth=1.4)
                ax.set_title(f"{LABELS[dataset]} · {method.upper()}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, title="Target quantile", frameon=False,
                      loc="upper center", bbox_to_anchor=(.5, .945), ncol=len(labels))
        figure.suptitle(f"Target correlation quantiles · {selected_pair[0]} / {selected_pair[1]}")
        figure.tight_layout(rect=(0, 0, 1, .89))
        return figure


def average_quantile_grid(quantiles):
    with paper_style():
        figure, axes = plt.subplots(3, 2, figsize=(13, 12))
        palette = plt.get_cmap("viridis")(np.linspace(.08, .92, len(QUANTILES)))
        for row, dataset in enumerate(DATASETS):
            for col, method in enumerate(("rrr", "pca")):
                ax = axes[row, col]
                curves = [value for key, value in quantiles.items()
                          if key[0] == dataset and key[-1] == method]
                grid, means, counts = average_normalized_quantiles(curves)
                for index, (level, color) in enumerate(zip(QUANTILES, palette)):
                    if index < means.shape[1] and counts[index]:
                        ax.plot(grid, means[:, index], color=color, linewidth=2,
                                label=f"{level:.0%} (n={counts[index]})")
                ax.set_xscale("symlog", linthresh=.01)
                ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Relative rank",
                       ylabel="Mean normalized quantile performance",
                       title=f"{LABELS[dataset]} · {method.upper()}")
                finish_axis(ax)
        axes[0, 0].legend(frameon=False, ncol=2)
        figure.suptitle("Mean quantile convergence across model layers · peak > 0.005")
        figure.tight_layout(rect=(0, 0, 1, .96))
        return figure


def sensitivity_grid(quantiles, values, *, variable):
    """Compare paired internal RRR/PCA saturation ranks across settings."""
    values = tuple(values)
    if not values:
        raise ValueError("values must contain at least one sensitivity setting")
    if variable not in {"floor", "fraction"}:
        raise ValueError("variable must be 'floor' or 'fraction'")
    with paper_style():
        figure, axes = plt.subplots(
            len(values), len(DATASETS),
            figsize=(16, 3.15 * len(values) + 1.1), sharey=True,
            squeeze=False,
        )
        for row, value in enumerate(values):
            for col, dataset in enumerate(DATASETS):
                ranks = _saturation_values(
                    quantiles, dataset, ("rrr", "pca"),
                    .99 if variable == "floor" else value,
                    min_peak=value if variable == "floor" else PEAK_FLOOR,
                )
                ax = axes[row, col]
                _saturation_comparison_axis(
                    ax, ranks, ("rrr", "pca"), ("RRR", "PCA"),
                    (("RRR", "PCA"),), show_mean_value=True,
                )
                if row == 0:
                    ax.set_title(LABELS[dataset])
                if col == 0:
                    ax.set_ylabel(f"{value:g}", labelpad=8)
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)
        figure.suptitle("Internal RRR saturation sensitivity: "
                         + ("minimum quantile correlation" if variable == "floor"
                            else "required peak fraction"))
        figure.text(.015, .5, "Saturation rank $k$", rotation=90,
                    va="center", ha="center", fontsize=14)
        figure.subplots_adjust(left=.13, right=.99, top=.94, bottom=.06,
                               hspace=.54, wspace=.10)
        return figure


def _decoding_relative(decoding):
    required = {"space", "subspace", "k", "kind", "task", "primary_score"}
    if decoding.empty or not required <= set(decoding):
        return pd.DataFrame()
    frame = decoding[(decoding["generation"] == "within_dataset")
                     & (decoding["space"] == "rank")
                     & (decoding["subspace"].isin(["rrr", "pca"]))
                     & (decoding["label_offset"] == 0)].copy()
    volume_tasks = {"podcast": "intensity_rms", "cars-2": "native_intensity_rms",
                    "fmri": "intensity_rms"}
    for dataset, task in volume_tasks.items():
        frame.loc[frame["dataset"].eq(dataset) & frame["task"].eq(task), "task"] = "volume"
    frame["layer"] = frame["layer"].fillna(-1).astype(int)
    keys = ["dataset", "model", "layer", "task", "kind", "k", "variant"]
    wide = frame.pivot_table(index=keys, columns="subspace", values="primary_score",
                             aggfunc="mean").reset_index()
    if not {"rrr", "pca"} <= set(wide):
        return pd.DataFrame()
    rrr, pca = wide["rrr"].clip(lower=0), wide["pca"].clip(lower=0)
    wide["relative"] = (rrr - pca) / ((rrr + pca) / 2 + .05)
    return wide


def decoding_aggregate(decoding):
    """Four example decoding curves above three paired dataset AUC scatters."""
    with paper_style():
        figure = plt.figure(figsize=(20, 10.5))
        grid = figure.add_gridspec(2, 12, height_ratios=[1, 1.6],
                                  hspace=.67, wspace=.46)
        frame = _decoding_relative(decoding)
        features = [("concreteness", "Concreteness"),
                    ("sentence_position", "Sentence position"),
                    ("standardized_entropy", "Entropy"),
                    ("volume", "Volume")]
        pairs = [("cars-2", "podcast"), ("podcast", "fmri"),
                 ("cars-2", "fmri")]
        auc_rows = []
        if not frame.empty:
            for (dataset, model, layer, task), group in frame.groupby(
                ["dataset", "model", "layer", "task"], dropna=False
            ):
                mean = group.groupby("k")["relative"].mean().sort_index()
                ranks = mean.index.to_numpy(dtype=float)
                if len(ranks) < 2:
                    continue
                x = np.log(ranks)
                span = x[-1] - x[0]
                if span > 0:
                    auc_rows.append((dataset, model, layer, task,
                                     float(np.trapezoid(mean, x=(x - x[0]) / span))))
        auc = pd.DataFrame(auc_rows, columns=["dataset", "model", "layer", "task", "auc"])
        consensus = (auc.pivot_table(
            index=["model", "layer", "task"], columns="dataset", values="auc"
        ).dropna(subset=list(DATASETS)).reset_index()
                     if not auc.empty else pd.DataFrame())
        bound = (max(float(np.nanmax(np.abs(consensus[list(DATASETS)].to_numpy()))), .05)
                 if not consensus.empty else .05) * 1.05
        ticks = MaxNLocator(nbins=5, symmetric=True).tick_values(-bound, bound)
        limits = (float(ticks[0]), float(ticks[-1]))
        curve_axes = []
        for col, (task, title) in enumerate(features):
            ax = figure.add_subplot(grid[0, 3 * col:3 * (col + 1)],
                                    sharey=curve_axes[0] if curve_axes else None)
            curve_axes.append(ax)
            for dataset, linestyle in zip(DATASETS, ("-", "--", ":")):
                selected = frame[(frame["task"] == task) & (frame["dataset"] == dataset)] if not frame.empty else frame
                if len(selected):
                    mean = selected.groupby("k")["relative"].mean().sort_index()
                    decoding_relative_axis(ax, mean.index, mean, LABELS[dataset],
                                           linestyle=linestyle)
            ax.set_title(title)
            if col:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
        for ax in curve_axes:
            add_decoding_gradient(ax)
        handles, labels = curve_axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, frameon=False, ncol=3,
                      loc="upper center", bbox_to_anchor=(.5, .948))
        scatter_axes = []
        for col, (first, second) in enumerate(pairs):
            scatter = figure.add_subplot(grid[1, 4 * col:4 * (col + 1)])
            scatter_axes.append(scatter)
            matched = (consensus[[first, second]].dropna() if not consensus.empty
                       else pd.DataFrame(columns=[first, second]))
            correlation_scatter_axis(scatter, matched[first], matched[second], limits, ticks)
            if len(matched):
                if (len(matched) >= 2 and matched[first].nunique() > 1
                        and matched[second].nunique() > 1):
                    from scipy.stats import pearsonr

                    result = pearsonr(matched[first], matched[second])
                    r, p_value = result.statistic, result.pvalue
                else:
                    r, p_value = np.nan, np.nan
                scatter.set_title(
                    f"{LABELS[first]} vs {LABELS[second]}\n"
                    f"Pearson r = {r:.2f}, {format_p_value(p_value)}\n"
                    f"n = {len(matched)}",
                    fontsize=13,
                )
            if col == 0:
                scatter.legend(frameon=False, fontsize=10, loc="upper left")
            scatter.set_xlabel(f"{LABELS[first]} relative AUC")
            scatter.set_ylabel(f"{LABELS[second]} relative AUC")
        figure.subplots_adjust(left=.065, right=.985, top=.84, bottom=.10)
        curve_left = curve_axes[0].get_position()
        curve_right = curve_axes[-1].get_position()
        scatter_left = scatter_axes[0].get_position()
        scatter_right = scatter_axes[-1].get_position()
        figure.text((curve_left.x0 + curve_right.x1) / 2, .975,
                    "A. Relative Decoding Curves", ha="center", va="bottom",
                    fontsize=15, weight="bold")
        figure.text((scatter_left.x0 + scatter_right.x1) / 2,
                    scatter_left.y1 + .085,
                    "B. Cross-Dataset AUC Correlations", ha="center", va="bottom",
                    fontsize=15, weight="bold")
        return figure
