"""Exploratory per-target saturation, separate from aggregate figures."""

from collections import defaultdict
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from src.utils.plotting.final_axes import finish_axis, paper_style
from src.utils.plotting.final_data import (
    DATASETS, LABELS, _collapse, _manifest_for, _metric_name, _rrr,
)
from src.utils.rrr.plot_metrics import (
    fisher_mean, normalized_quantile_curve, quantile_saturation_rank,
)


@lru_cache(maxsize=12)
def load_target_curves(results_root, dataset, model, layer):
    """Load one model layer; preserve target identities across evaluated ranks."""
    artifact = _rrr(results_root, "internal", dataset, model, layer).load_artifacts(
        ["rank_metrics", "target_manifest"]
    )
    values = defaultdict(list)
    for encoded, metrics in artifact.rank_metrics.items():
        selector = dict(encoded)
        method, rank = selector.get("method"), selector.get("rank")
        if method not in {"rrr", "pca"} or rank is None:
            continue
        name = _metric_name(dataset, method)
        if name in metrics:
            values[(method, int(rank))].append(
                _collapse(metrics[name], _manifest_for(artifact, {"method": method}))
            )
    curves = {}
    for method in ("rrr", "pca"):
        ranks = sorted(rank for item_method, rank in values if item_method == method)
        if ranks:
            curves[method] = (np.asarray(ranks, dtype=float), np.stack([
                fisher_mean(np.stack(values[(method, rank)]), axis=0) for rank in ranks
            ]))
    return curves


def target_progress(ranks, matrix, performance=.99, coverage=.90):
    """Positive-peak target curves and their interpolated first crossings."""
    normalized, crossings = [], []
    for values in np.asarray(matrix).T:
        curve = normalized_quantile_curve(values, min_peak=0)
        if np.isfinite(curve).any():
            normalized.append(curve)
            crossings.append(quantile_saturation_rank(
                ranks, values[:, None], fraction=performance, min_peak=0,
            ))
    crossings = np.sort(crossings)
    hit_rank = crossings[int(np.ceil(coverage * len(crossings))) - 1] if len(crossings) else np.nan
    return np.asarray(normalized), crossings, float(hit_rank)


def target_saturation_grid(results_root, selected_pair):
    """Show every eligible internal target and the fraction reaching 99% of peak."""
    rows = []
    with paper_style():
        figure, axes = plt.subplots(3, 2, figsize=(14, 12))
        for row, dataset in enumerate(DATASETS):
            curves = load_target_curves(results_root, dataset, *selected_pair)
            for col, method in enumerate(("rrr", "pca")):
                ax = axes[row, col]
                value = curves.get(method)
                if value is None:
                    ax.text(.5, .5, "No target curves", transform=ax.transAxes, ha="center")
                    continue
                ranks, matrix = value
                normalized, crossings, hit = target_progress(ranks, matrix)
                segments = [np.column_stack((ranks[np.isfinite(curve)], curve[np.isfinite(curve)]))
                            for curve in normalized]
                ax.add_collection(LineCollection(segments, colors="#64748b", alpha=.08,
                                                 linewidths=.5, rasterized=True))
                if len(crossings):
                    x = np.r_[ranks[0], crossings, ranks[-1]]
                    y = np.r_[0, np.arange(1, len(crossings) + 1) / len(crossings), 1]
                    ax.step(x, y, where="post", color="#ea580c", linewidth=2,
                            label="Fraction of targets reaching 99%")
                    ax.axvline(hit, color="#ea580c", linestyle="--",
                               label=f"90% reached at rank {hit:.2f}")
                ax.axhline(.99, color="#64748b", linestyle=":", label="99% of target peak")
                ax.axhline(.90, color="#ea580c", linestyle=":", label="90% of targets")
                ax.set(xlim=(ranks[0], ranks[-1]), ylim=(0, 1.03),
                       xlabel="Rank", ylabel="Normalized performance / target fraction",
                       title=f"{LABELS[dataset]} · {method.upper()} · {len(crossings):,} positive-peak targets")
                ax.set_xscale("log")
                ax.legend(frameon=False, fontsize=8, loc="lower right")
                finish_axis(ax)
                rows.append({"Dataset": LABELS[dataset], "Method": method.upper(),
                             "Eligible targets": len(crossings), "90% at 99% rank": hit})
        figure.suptitle(f"Per-target saturation · {selected_pair[0]} / {selected_pair[1]}")
        figure.tight_layout(rect=(0, 0, 1, .96))
    return figure, pd.DataFrame(rows)
