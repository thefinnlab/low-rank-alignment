"""Shared plotting helpers for cross-k and lag-specific result figures.

These helpers are designed to be called from both ``plots.py`` (cross-subject
RRR analysis) and ``loso-results.py`` (per-subject LOSO analysis).
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Shared color constants
RED_BASELINE = "#d62728"
GRAY = "#6b7280"


def make_viridis_cmap():
    """Return a truncated viridis colormap (range 0.08–0.92)."""
    _raw = cm.get_cmap("viridis", 256)
    return mcolors.LinearSegmentedColormap.from_list(
        "viridis_trunc", _raw(np.linspace(0.08, 0.92, 256))
    )


def get_star_label(p):
    """Convert a p-value to a significance star string, or ``None``."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return None


def apply_publication_style():
    """Apply standard publication ``rcParams`` to matplotlib."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 300,
    })


def plot_cross_k_panel(
    ax,
    ks,
    mean_corrs,
    fold_corrs,
    p_values=None,
    label="RRR Space",
    color=None,
    base_mean=None,
    base_label="Full Model (Baseline)",
):
    """Draw a semilog cross-k panel onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ks : array-like, shape (n_ks,)
        Subspace rank values (plotted on a log x-axis).
    mean_corrs : array-like, shape (n_ks,)
        Mean correlation per rank (averaged over folds, electrodes, and lags).
    fold_corrs : array-like, shape (n_ks, n_folds)
        Per-fold correlations used to draw faint background fold lines.
    p_values : array-like, shape (n_ks,) or None
        Global p-values for significance annotation. Stars are omitted when
        *p_values* is ``None``.
    label : str
        Legend label for the mean line.
    color : color-spec or None
        Explicit line colour.  Defaults to the viridis colormap midpoint.
    base_mean : float or None
        If provided, a red dashed horizontal reference line is drawn at this
        value and added to the legend under *base_label*.
    base_label : str
        Legend label for the baseline reference hline.
    """
    cmap = make_viridis_cmap()
    line_color = color if color is not None else cmap(0.45)
    rgba = mcolors.to_rgba(line_color)
    fold_color = (*rgba[:3], 0.3)

    fold_corrs = np.asarray(fold_corrs)
    for fold in range(fold_corrs.shape[1]):
        ax.semilogx(ks, fold_corrs[:, fold], color=fold_color, linewidth=0.7)

    ax.semilogx(
        ks, mean_corrs,
        color=line_color, marker="o", markersize=3.5,
        linewidth=1.4, label=label, zorder=5,
    )

    if base_mean is not None:
        ax.axhline(
            base_mean, color=RED_BASELINE, ls="--", linewidth=1.0,
            label=base_label,
        )

    if p_values is not None:
        y_lo, y_hi = ax.get_ylim()
        offset = (y_hi - y_lo) * 0.03
        for i, (k, val) in enumerate(zip(ks, mean_corrs)):
            stars = get_star_label(p_values[i])
            if stars:
                ax.text(k, val + offset, stars,
                        ha="center", va="bottom", fontsize=8, color="k")

    ax.set_xlabel("Rank $k$ (log scale)")
    ax.set_ylabel("Correlation (r)")
    ax.legend(loc="lower right", fontsize=7)


def plot_lag_panel(
    ax,
    lags,
    data,
    ks,
    base_corrs_mean=None,
    sig_mask=None,
    saturation_k=20,
    cmap=None,
    fig=None,
    base_label="Full Model (Baseline)",
):
    """Draw a lag-specific correlation panel onto *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    lags : array-like, shape (n_lags,)
        Lag values in seconds.
    data : ndarray, shape (n_ks, n_folds, n_elecs, n_lags) or (n_ks, n_lags)
        Correlation data. Four-dimensional arrays are averaged over the fold
        and electrode axes internally before plotting.
    ks : array-like, shape (n_ks,)
        Subspace rank values corresponding to the first axis of *data*.
    base_corrs_mean : array-like, shape (n_lags,) or None
        Mean baseline correlation per lag.  Drawn as a red reference line when
        provided.
    sig_mask : ndarray, shape (n_ks, n_lags) or None
        Boolean significance mask.  Significance scatter markers are omitted
        when ``None``.
    saturation_k : int
        Only k values up to this cutoff are plotted (default 20).
    cmap : colormap or None
        Colormap for per-rank lines.  Defaults to ``make_viridis_cmap()``.
    fig : matplotlib.figure.Figure or None
        Parent figure used to attach a colorbar.  No colorbar is added when
        ``None``.
    base_label : str
        Legend label for the baseline reference line.
    """
    if cmap is None:
        cmap = make_viridis_cmap()

    ks = np.asarray(ks)
    data = np.asarray(data)
    lags = np.asarray(lags)

    # Restrict to k values up to the saturation point.
    mask = ks <= saturation_k
    saturated_ks = ks[mask]
    saturated_data = data[mask]

    norm = mcolors.LogNorm(vmin=int(min(saturated_ks)), vmax=int(max(saturated_ks)))

    # Average over folds (and electrodes if present).
    if saturated_data.ndim == 4:
        mean_data = saturated_data.mean((1, 2))   # (n_ks, n_folds, n_elecs, n_lags) → (n_ks, n_lags)
    elif saturated_data.ndim == 3:
        mean_data = saturated_data.mean(1)         # (n_ks, n_folds, n_lags) → (n_ks, n_lags)
    else:
        mean_data = saturated_data                 # already (n_ks, n_lags)

    for i in range(len(saturated_ks)):
        ax.plot(
            lags, mean_data[i],
            color=cmap(norm(saturated_ks[i])), alpha=0.75, linewidth=0.9,
        )

    if base_corrs_mean is not None:
        ax.plot(lags, base_corrs_mean, color=RED_BASELINE, linewidth=1.3,
                label=base_label)

    if sig_mask is not None:
        saturated_sig = sig_mask[mask]
        y_min, y_max = ax.get_ylim()
        marker_base = y_max
        spacing = (y_max - y_min) * 0.025
        for rank, i in enumerate(np.argsort(saturated_ks)):
            curr_sig = saturated_sig[i]
            if np.any(curr_sig):
                y_pos = marker_base + rank * spacing
                ax.scatter(
                    lags[curr_sig],
                    np.full(curr_sig.sum(), y_pos),
                    marker="*", s=14,
                    color=cmap(norm(saturated_ks[i])),
                    linewidths=0,
                )
        ax.set_ylim(y_min, marker_base + len(saturated_ks) * spacing)

    ax.axvline(0, color=GRAY, alpha=0.7, linewidth=0.5, ymax=0.9, ls="--")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Correlation (r)")
    ax.set_title("Correlation by Lag and Subspace Rank", fontsize=9, pad=8)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.85), fontsize=7)

    if fig is not None:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = ax.inset_axes([1.03, 0.0, 0.025, 1.0])
        cbar = fig.colorbar(sm, cax=cax, extend="both")
        cbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
        cbar.set_label("Rank $k$", fontsize=8, labelpad=1)
        cbar.outline.set_linewidth(0.6)
