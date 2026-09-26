"""Reusable Matplotlib axis types for final neuroscience result figures."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.utils.rrr.plot_metrics import monotonic_normalized_curve


COLORS = {
    "rrr": "#6ab2e2", "pca": "#63bc51",
    "RRR": "#6ab2e2", "PCA": "#63bc51",
    "RRR baseline": "#6ab2e2", "RRR transfer": "#4b98cc",
    "PCA baseline": "#63bc51", "Within RRR": "#6ab2e2",
    "Within PCA": "#63bc51", "Transfer RRR": "#4b98cc",
    "Transfer PCA": "#45963a",
}


def paper_style():
    """Use type sizes that remain readable when figures are placed in a paper."""
    return plt.rc_context({
        "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "figure.titlesize": 19,
        "savefig.dpi": 300,
    })


def finish_axis(ax):
    ax.grid(axis="y", alpha=.20)
    ax.spines[["top", "right"]].set_visible(False)


def format_p_value(value):
    """Use an inequality when a small p-value would round to zero."""
    if not np.isfinite(value):
        return "p = NA"
    return "p < 0.001" if value < .001 else f"p = {value:.3f}"


def rank_curve_axis(ax, curves, *, saturation=None, baseline=None):
    """Draw a selected model layer's mean rank curves on an existing axis."""
    for method, (ranks, values) in curves.items():
        color = COLORS.get(method, "#64748b")
        ax.plot(ranks, values, marker="o", color=color, linewidth=1.6,
                markersize=3, linestyle="--" if "transfer" in method.lower() else "-",
                label=method.upper() if method in {"rrr", "pca"} else method)
        if saturation is not None and np.isfinite(saturation.get(method, np.nan)):
            ax.axvline(saturation[method], color=color, linestyle="--", linewidth=1.4,
                       alpha=.9)
    if baseline is not None and np.isfinite(baseline):
        ax.axhline(baseline, color="#dc2626", linestyle="--", linewidth=1.6,
                   label="Full model baseline")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("Correlation")
    finish_axis(ax)


def normalized_curve_axis(ax, curves, *, grid=None):
    """Draw individual and equal-weighted mean monotonic model-layer curves."""
    if grid is None:
        grid = np.r_[0.0, np.geomspace(1e-4, 1.0, 199)]
    for method, members in curves.items():
        aligned = []
        color = COLORS.get(method, "#64748b")
        for ranks, values in members:
            x, y = monotonic_normalized_curve(ranks, values)
            if len(x):
                ax.plot(x, y, color=color, linewidth=.7, alpha=.1,
                        linestyle="--" if "transfer" in method.lower() else "-")
                aligned.append(np.interp(grid, x, y))
        if aligned:
            ax.plot(grid, np.mean(aligned, axis=0), color=color, linewidth=2,
                    linestyle="--" if "transfer" in method.lower() else "-",
                    label=method.upper() if method in {"rrr", "pca"} else method)
    ax.set_xscale("symlog", linthresh=.01)
    ax.set_xticks([0, .001, .01, .1, 1], ["0", "0.1%", "1%", "10%", "100%"])
    ax.set(xlim=(0, 1), ylim=(-.02, 1.02), xlabel="Relative rank",
           ylabel="Normalized performance")
    finish_axis(ax)


def distribution_axis(ax, groups, *, log2_rank=False, show_points=True, paired=None,
                      show_mean_value=False):
    """Draw the established violin, box, points, and black mean diamond."""
    labels = list(groups)
    values = [np.asarray(groups[label], dtype=float) for label in labels]
    values = [item[np.isfinite(item)] for item in values]
    valid = [(label, item) for label, item in zip(labels, values) if len(item)]
    if not valid:
        ax.text(.5, .5, "No finite results", ha="center", va="center",
                transform=ax.transAxes)
        finish_axis(ax)
        return
    labels, values = zip(*valid)
    transformed = [np.log2(item) if log2_rank else item for item in values]
    positions = np.arange(1, len(labels) + 1)
    if paired:
        for row in paired:
            available = [(position, np.log2(row[label]) if log2_rank else row[label])
                         for position, label in zip(positions, labels)
                         if label in row and np.isfinite(row[label]) and row[label] > 0]
            for (x0, y0), (x1, y1) in zip(available, available[1:]):
                ax.plot([x0, x1], [y0, y1], color="#9ca3af",
                        linewidth=.65, alpha=.5, zorder=1)
    violins = ax.violinplot(transformed, positions=positions, widths=.72,
                            showextrema=False)
    colors = [COLORS.get(label, "#6ab2e2") for label in labels]
    for body, color in zip(violins["bodies"], colors):
        body.set(facecolor=color, edgecolor=color, alpha=.22, linewidth=1)
    boxes = ax.boxplot(transformed, positions=positions, widths=.20,
                       patch_artist=True, showfliers=False,
                       medianprops={"color": "black", "linewidth": 1.2})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(.18)
    for position, item, raw, color in zip(positions, transformed, values, colors):
        if show_points:
            offsets = np.linspace(-.14, .14, len(item)) if len(item) > 1 else [0]
            ax.scatter(position + np.asarray(offsets), item, s=19, color=color,
                       edgecolors="white", linewidths=.35, zorder=3)
        mean = float(np.mean(raw))
        mean_height = np.log2(mean) if log2_rank else mean
        ax.scatter(position, mean_height,
                   marker="D", s=34, color="black", zorder=4)
        if show_mean_value:
            ax.annotate(f"mean = {mean:.1f}", (position, mean_height),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, zorder=5,
                        bbox={"facecolor": "white", "edgecolor": "none",
                              "alpha": .75, "pad": 1})
    ax.set_xticks(positions, labels)
    ax.set_xlim(.45, len(labels) + .55)
    if log2_rank:
        maximum = int(np.ceil(max(max(item) for item in transformed)))
        ticks = np.arange(maximum + 1)
        ax.set_yticks(ticks, [str(2 ** int(tick)) for tick in ticks])
        ax.set_ylabel("Saturation rank $k$")
    finish_axis(ax)


def pairwise_test_brackets(ax, labels, comparisons):
    """Draw compact comparison brackets with adjusted p-values above violins."""
    positions = {label: index for index, label in enumerate(labels, start=1)}
    transform = ax.get_xaxis_transform()
    for offset, (first, second, p_value) in enumerate(comparisons):
        left, right = sorted((positions[first], positions[second]))
        y, tick = .90 - offset * .12, .025
        ax.plot([left, left, right, right], [y, y + tick, y + tick, y],
                transform=transform, color="#374151", linewidth=1,
                clip_on=False, zorder=6)
        mark = "*" if p_value < .05 else "n.s."
        p_text = format_p_value(p_value)
        ax.text((left + right) / 2, y + tick, f"{mark} {p_text}",
                transform=transform, ha="center", va="bottom", fontsize=10,
                color="#374151", clip_on=False, zorder=7)


def target_box_axis(ax, positions, target_values, colors):
    """Draw per-target distributions with the same black mean mark."""
    finite = [(position, np.asarray(values)[np.isfinite(values)], color)
              for position, values, color in zip(positions, target_values, colors)]
    finite = [(position, values, color) for position, values, color in finite if len(values)]
    if not finite:
        finish_axis(ax)
        return
    positions, values, colors = zip(*finite)
    boxes = ax.boxplot(values, positions=positions, widths=.68, patch_artist=True,
                       showfliers=False, medianprops={"color": "black", "linewidth": 1})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(.45)
    for position, item in zip(positions, values):
        ax.scatter(position, np.mean(item), marker="D", s=29, color="black", zorder=4)
    finish_axis(ax)


def target_bar_axis(ax, positions, target_values, colors, *, error):
    """Draw mean target correlations with a selected target-level error measure."""
    finite = [(position, np.asarray(values)[np.isfinite(values)], color)
              for position, values, color in zip(positions, target_values, colors)]
    finite = [(position, values, color) for position, values, color in finite if len(values)]
    if not finite:
        finish_axis(ax)
        return
    if error not in {"sd", "sem", "ci95"}:
        raise ValueError("error must be 'sd', 'sem', or 'ci95'")
    positions, values, colors = zip(*finite)
    means = np.asarray([np.mean(item) for item in values])
    std = np.asarray([np.std(item, ddof=1) if len(item) > 1 else 0
                      for item in values])
    counts = np.asarray([len(item) for item in values])
    errors = std if error == "sd" else std / np.sqrt(counts)
    if error == "ci95":
        errors *= 1.96
    ax.bar(positions, means, width=.8, color=colors,
           yerr=errors, capsize=3,
           error_kw={"ecolor": "#374151", "elinewidth": 1.1, "capthick": 1.1})
    finish_axis(ax)


def quantile_axis(ax, ranks, matrix, levels, *, normalized=False, floor=.005):
    """Draw target quantiles across rank, optionally normalized by their peaks."""
    from src.utils.rrr.plot_metrics import normalized_quantile_curve

    palette = plt.get_cmap("viridis")(np.linspace(.08, .92, len(levels)))
    for level, column, color in zip(levels, np.asarray(matrix).T, palette):
        values = normalized_quantile_curve(column, floor) if normalized else column
        if np.isfinite(values).any():
            ax.plot(ranks, values, "o-", color=color, linewidth=1.8,
                    markersize=3.2, label=f"{level:.0%}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Rank $k$")
    ax.set_ylabel("Normalized quantile performance" if normalized else "Correlation")
    if normalized:
        ax.set_ylim(0, 1.05)
    finish_axis(ax)


def decoding_relative_axis(ax, ranks, values, label=None, *, linestyle="-"):
    """Draw one decoding advantage curve on an existing axis."""
    ax.plot(ranks, values, marker="o", color="#4b5563", linestyle=linestyle,
            linewidth=2.1, markersize=3.5, label=label)
    ax.axhline(0, color="#111827", linewidth=.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Decoding rank")
    ax.set_ylabel("Mean RRR–PCA difference")
    finish_axis(ax)


def correlation_scatter_axis(ax, x_values, y_values, limits, ticks):
    """Draw a square paired-AUC scatter with identity and fitted lines."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[finite], y_values[finite]
    if len(x_values):
        ax.scatter(x_values, y_values, color="#2563eb", s=18, alpha=.4,
                   edgecolors="white", linewidths=.3)
    ax.plot(limits, limits, color="#94a3b8", linestyle="--", label="Equal AUC")
    if len(x_values) >= 2 and np.unique(x_values).size >= 2:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        line_x = np.linspace(x_values.min(), x_values.max(), 100)
        ax.plot(line_x, slope * line_x + intercept, color="#111827",
                linewidth=2, label="Linear fit")
    ax.set(xlim=limits, ylim=limits)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=11)
    finish_axis(ax)


def add_decoding_gradient(ax):
    """Add the established blue/green background after all lines are drawn."""
    ylo, yhi = ax.get_ylim()
    bound = max(abs(ylo), abs(yhi), .01)
    ax.set_ylim(-bound, bound)
    xlo, xhi = ax.get_xlim()
    vertical = np.linspace(-1, 1, 256)
    white = np.ones(3)
    green = np.array([99, 188, 81]) / 255
    blue = np.array([106, 178, 226]) / 255
    rgb = np.where(vertical[:, None] < 0,
                   white - vertical[:, None] * (green - white),
                   white + vertical[:, None] * (blue - white))
    ax.imshow(np.repeat(rgb[:, None, :], 2, axis=1),
              extent=(xlo, xhi, -bound, bound), origin="lower",
              aspect="auto", alpha=.25, interpolation="bilinear", zorder=-10)
    ax.set(xlim=(xlo, xhi), ylim=(-bound, bound))
