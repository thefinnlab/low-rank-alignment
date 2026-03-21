import numpy as np
from utils.analysis import find_best_k


def plot_svd_truncation_results(results, threshold=0.95, save_path=None):
    """4-panel figure summarizing SVD truncation performance.

    Panel 1 (top-left): Mean correlation vs k, split by sig/non-sig, with best_k marker.
    Panel 2 (top-right): Retention ratio (trunc_sig / full_sig) with threshold line + best_k.
    Panel 3 (bottom-left): Per-electrode fractional drop at best_k, colored by frac sig lags.
    Panel 4 (bottom-right): Per-lag drop at best_k with sig-fraction overlay.

    Args:
        results: dict from compute_truncated_correlations.
        threshold: performance retention threshold (default 0.95).
        save_path: if given, save figure here; otherwise call plt.show().
    """
    import matplotlib.pyplot as plt

    k_values = results["k_values"]
    full_corrs = results["full_corrs"]          # (n_elecs, n_lags)
    trunc_corrs = results["truncated_corrs"]    # (n_k, n_elecs, n_lags)
    sig_mask = results["sig_mask"]              # (n_elecs, n_lags)

    best_k = find_best_k(results, threshold)
    best_ki = int(np.where(k_values == best_k)[0][0])

    sig_mean = np.array([t[sig_mask].mean() for t in trunc_corrs])
    nonsig_mean = np.array([t[~sig_mask].mean() for t in trunc_corrs])
    full_sig = full_corrs[sig_mask].mean()
    full_nonsig = full_corrs[~sig_mask].mean()
    ratios = sig_mean / full_sig

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: mean correlation vs k
    ax = axes[0, 0]
    ax.semilogx(k_values, sig_mean, "o-", color="steelblue", label="Sig")
    ax.semilogx(k_values, nonsig_mean, "o-", color="gray", label="Non-sig")
    ax.axhline(full_sig, color="steelblue", linestyle="--", alpha=0.5, label="Full sig")
    ax.axhline(full_nonsig, color="gray", linestyle="--", alpha=0.5, label="Full non-sig")
    ax.axvline(best_k, color="red", linestyle=":", label=f"best_k={best_k}")
    ax.set_xlabel("k (SVD components)")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("Mean correlation vs k")
    ax.legend(fontsize=8)

    # Panel 2: retention ratio
    ax = axes[0, 1]
    ax.semilogx(k_values, ratios, "o-", color="steelblue")
    ax.axhline(threshold, color="orange", linestyle="--", label=f"Threshold {threshold:.0%}")
    ax.axvline(best_k, color="red", linestyle=":", label=f"best_k={best_k}")
    ax.set_xlabel("k (SVD components)")
    ax.set_ylabel("Retention ratio")
    ax.set_title("Sig performance retention vs k")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Panel 3: per-electrode fractional drop at best_k
    ax = axes[1, 0]
    full_elec = full_corrs.mean(axis=1)   # (n_elecs,)
    trunc_elec = trunc_corrs[best_ki].mean(axis=1)  # (n_elecs,)
    drop_elec = (full_elec - trunc_elec) / np.maximum(np.abs(full_elec), 1e-12)
    frac_sig_elec = sig_mask.mean(axis=1)  # (n_elecs,)
    sc = ax.scatter(np.arange(len(drop_elec)), drop_elec, c=frac_sig_elec,
                    cmap="viridis", s=10, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Frac sig lags")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Electrode index")
    ax.set_ylabel("Fractional drop at best_k")
    ax.set_title(f"Per-electrode drop at k={best_k}")

    # Panel 4: per-lag drop at best_k with sig-fraction overlay
    ax = axes[1, 1]
    full_lag = full_corrs.mean(axis=0)   # (n_lags,)
    trunc_lag = trunc_corrs[best_ki].mean(axis=0)  # (n_lags,)
    drop_lag = (full_lag - trunc_lag) / np.maximum(np.abs(full_lag), 1e-12)
    lags = np.arange(len(drop_lag))
    ax.plot(lags, drop_lag, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag index")
    ax.set_ylabel("Fractional drop at best_k")
    ax.set_title(f"Per-lag drop at k={best_k}")

    ax2 = ax.twinx()
    frac_sig_lag = sig_mask.mean(axis=0)  # (n_lags,)
    ax2.bar(lags, frac_sig_lag, alpha=0.2, color="coral", label="Frac sig elecs")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Frac sig electrodes", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def add_masked_corrs(corrs, mask, lags, ax=None, color="blue", label=None, linestyle=None):
    """Plot mean ± SEM of correlations, averaging only over electrodes selected by mask at each lag.

    At each lag position l, only electrodes where mask[:, l] is True contribute to the mean.
    Lags where no electrode passes the mask are plotted as NaN (gap in line).

    Args:
        corrs: (n_folds, n_electrodes, n_lags).
        mask: (n_electrodes, n_lags) boolean array — True where an electrode/lag pair is included.
        lags: 1-D array of lag values (seconds).
        ax: matplotlib Axes; uses plt.gca() if None.
        color: line color.
        label: legend label.
        linestyle: line style string.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    # Apply mask: set excluded positions to NaN, then nanmean over electrodes per fold
    masked = np.where(mask[np.newaxis], corrs, np.nan)          # (n_folds, n_elecs, n_lags)
    per_fold = np.nanmean(masked, axis=1)                        # (n_folds, n_lags)

    mean = np.nanmean(per_fold, axis=0)
    n = np.sum(~np.isnan(per_fold), axis=0)
    sem = np.nanstd(per_fold, axis=0) / np.sqrt(np.maximum(n, 1))

    kwargs = dict(color=color, label=label)
    if linestyle is not None:
        kwargs["linestyle"] = linestyle

    ax.plot(lags, mean, **kwargs)
    ax.fill_between(lags, mean - sem, mean + sem, alpha=0.2, color=color)


def add_sig_proportion_histogram(sig_mask, lags, ax=None, color="coral", alpha=0.2,
                                  label="Frac sig electrodes"):
    """Add a background histogram of the fraction of significant electrodes at each lag.

    Draws bars on a twin y-axis so the histogram does not distort the correlation scale.
    Returns the twin Axes so the caller can adjust its y-label / tick color if desired.

    Args:
        sig_mask: (n_electrodes, n_lags) boolean significance mask.
        lags: 1-D array of lag values (seconds).
        ax: primary matplotlib Axes to twin off; uses plt.gca() if None.
        color: bar fill color.
        alpha: bar opacity.
        label: legend label for the histogram series.

    Returns:
        ax2: the twin Axes carrying the histogram.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    sig_frac = sig_mask.mean(axis=0)           # (n_lags,)
    lag_width = np.diff(lags).mean() if len(lags) > 1 else 1.0

    ax2 = ax.twinx()
    ax2.bar(lags, sig_frac, width=lag_width * 0.9, color=color, alpha=alpha,
            label=label, zorder=0)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Frac sig electrodes", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)                # let background bars show through
    return ax2


def plot_corrs_by_significance(corrs, sig_mask, lags, save_path=None):
    """Two-panel figure splitting correlations by significance.

    Left panel: mean ± SEM over only the significant (electrode, lag) pairs.
    Right panel: mean ± SEM over only the insignificant (electrode, lag) pairs.
    Both panels show the fraction of significant electrodes per lag as a
    background histogram on a twin y-axis.

    Args:
        corrs: (n_folds, n_electrodes, n_lags).
        sig_mask: (n_electrodes, n_lags) boolean array — True where significant.
        lags: 1-D array of lag values (seconds).
        save_path: if given, save figure here; otherwise call plt.show().
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

    for ax, mask, title, color in [
        (axes[0], sig_mask,  "Significant lags",   "steelblue"),
        (axes[1], ~sig_mask, "Insignificant lags", "gray"),
    ]:
        ax2 = add_sig_proportion_histogram(sig_mask, lags, ax=ax)
        add_masked_corrs(corrs, mask, lags, ax=ax, color=color, label="Mean corr")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Correlation")
        ax.set_title(title)
        ax.set_zorder(ax2.get_zorder() + 1)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def add_corrs(corrs, lags, electrode=None, label=None, color="black", ax=None, linestyle=None):
    """
    Plot mean ± SEM of correlation timecourses.

    Args:
        corrs: (n_folds, n_electrodes, n_lags).
        lags: 1-D array of lag values (seconds).
        electrode: if given, plot single electrode; else average over folds + electrodes.
        label: legend label.
        color: line color.
        ax: matplotlib Axes object; uses plt.gca() if None.
        linestyle: line style string.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if electrode is not None:
        data = corrs[:, electrode, :]  # (n_folds, n_lags)
    else:
        data = corrs.mean(axis=1)  # average over electrodes → (n_folds, n_lags)

    mean = data.mean(axis=0)
    sem = data.std(axis=0) / np.sqrt(data.shape[0])

    kwargs = dict(color=color, label=label)
    if linestyle is not None:
        kwargs["linestyle"] = linestyle

    ax.plot(lags, mean, **kwargs)
    ax.fill_between(lags, mean - sem, mean + sem, alpha=0.2, color=color)


def plot_svd_corrs(corrs_trunc, full_corrs, lags, title, save_path, best_k):
    """Plot SVD-truncated vs full-model correlations and save.

    Args:
        corrs_trunc: (n_folds, n_electrodes, n_lags) truncated-model correlations.
        full_corrs: (n_folds, n_electrodes, n_lags) full-model correlations.
        lags: 1-D array of lag values (seconds).
        title: figure title string.
        save_path: if given, save figure here; otherwise call plt.show().
        best_k: number of SVD components used (for legend label).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    add_corrs(full_corrs, lags, label="Full model", color="blue", ax=ax)
    add_corrs(corrs_trunc, lags, label=f"SVD k={best_k}", color="steelblue",
              linestyle="--", ax=ax)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Correlation")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_rrr_rank_results(summary_df, grid_df=None, title="RRR: R² vs rank", save_path=None):
    """Two-panel figure summarising RRR rank analysis.

    Panel 1 (left): full_r2, minimal_r2, recovery_r2 vs rank on a log-x axis.
                    Per-fold traces shown as light lines when grid_df is provided.
    Panel 2 (right): Fold × rank heatmap of recovery_r2 (lower = more sufficient).
                     Omitted when grid_df is None.

    Args:
        summary_df: DataFrame with columns [k, full_r2, minimal_r2, recovery_r2]
            averaged over folds (i.e. summary.csv).
        grid_df: Optional per-fold DataFrame with columns [k, fold, full_r2,
            minimal_r2, recovery_r2] (i.e. grid_results.csv).
        title: Figure title.
        save_path: If given, save figure here and close it; otherwise return the Figure.

    Returns:
        matplotlib Figure when save_path is None, else None.
    """
    import matplotlib.pyplot as plt

    has_folds = grid_df is not None
    n_panels = 2 if has_folds else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    colors = {"full_r2": "#9ca3af", "minimal_r2": "#3b82f6", "recovery_r2": "#ef4444", "pred_r2": "#10b981"}
    labels = {
        "full_r2": "full_r2 (ceiling)",
        "minimal_r2": "minimal_r2 (RRR)",
        "recovery_r2": "recovery_r2 (residual)",
        "pred_r2": "pred_r2 (vs full model)",
    }
    styles = {"full_r2": "--", "minimal_r2": "-", "recovery_r2": "-", "pred_r2": "-"}
    metrics = [m for m in ["full_r2", "minimal_r2", "recovery_r2", "pred_r2"] if m in summary_df.columns]

    ax = axes[0]
    ks = summary_df["k"].values

    if has_folds:
        for metric in metrics:
            if metric not in grid_df.columns:
                continue
            for _, fold_df in grid_df.groupby("fold"):
                ax.semilogx(fold_df["k"], fold_df[metric],
                            color=colors[metric], alpha=0.2, linewidth=0.8)

    for metric in metrics:
        ax.semilogx(ks, summary_df[metric], color=colors[metric],
                    linestyle=styles[metric], marker="o", markersize=4,
                    label=labels[metric], linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.legend(fontsize=8)

    if has_folds:
        folds = sorted(grid_df["fold"].unique())
        ks_ord = sorted(grid_df["k"].unique())
        k_idx = {k: i for i, k in enumerate(ks_ord)}
        f_idx = {f: i for i, f in enumerate(folds)}
        hm = np.full((len(folds), len(ks_ord)), np.nan)
        for _, row in grid_df.iterrows():
            hm[f_idx[row["fold"]], k_idx[row["k"]]] = row["recovery_r2"]

        ax2 = axes[1]
        im = ax2.imshow(hm, aspect="auto", cmap="Reds_r")
        ax2.set_xticks(range(len(ks_ord)))
        ax2.set_xticklabels([str(k) for k in ks_ord], rotation=45, ha="right")
        ax2.set_yticks(range(len(folds)))
        ax2.set_yticklabels([f"fold {f}" for f in folds])
        ax2.set_xlabel("Rank k")
        ax2.set_title("Recovery R² per fold (lower = more sufficient)")
        plt.colorbar(im, ax=ax2, label="recovery_r2")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_srrr_rank_results(
    summary_df,
    knee_df=None,
    active_df=None,
    title="SRRR: Rank analysis",
    save_path=None,
):
    """Three-panel figure summarising SRRR rank analysis.

    Uses *active_df* if provided (already lambda-resolved), otherwise resolves
    the knee lambda per rank from *knee_df*, or falls back to the lowest lambda
    in *summary_df*.

    Panel 1 (left):   sufficient_frac vs rank (bar chart).
    Panel 2 (centre): full_r2, minimal_r2, recovery_r2 vs rank.
    Panel 3 (right):  avg. n_selected SVD directions vs rank.

    Args:
        summary_df: DataFrame from summary.csv with columns
            [k, lam, full_r2, minimal_r2, recovery_r2, sufficient_frac, n_selected].
        knee_df: Optional DataFrame from knee_selection.csv with columns [k, knee_lam].
            Used when active_df is None to resolve the knee lambda per rank.
        active_df: Optional pre-filtered single-lambda-per-k DataFrame. When
            provided, summary_df/knee_df are ignored for the plot data.
        title: Figure suptitle.
        save_path: If given, save figure here and close it; otherwise return the Figure.

    Returns:
        matplotlib Figure when save_path is None, else None.
    """
    import matplotlib.pyplot as plt

    if active_df is not None:
        active = active_df.copy()
    elif knee_df is not None:
        lam_per_k = dict(zip(knee_df["k"], knee_df["knee_lam"]))
        active = summary_df[
            summary_df.apply(
                lambda r: abs(r["lam"] - lam_per_k.get(r["k"], -1)) < 1e-12, axis=1
            )
        ].copy()
    else:
        min_lam = summary_df["lam"].min()
        active = summary_df[
            summary_df["lam"].apply(lambda l: abs(l - min_lam) < 1e-12)
        ].copy()

    if active.empty:
        return None

    active = active.sort_values("k")
    ks = active["k"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: sufficient_frac
    ax = axes[0]
    if "sufficient_frac" in active.columns:
        ax.bar(range(len(ks)), active["sufficient_frac"].values, color="#60a5fa")
        ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1)
        suf_mask = active["sufficient_frac"] >= 1.0
        if suf_mask.any():
            first_k = int(active[suf_mask]["k"].min())
            first_i = list(ks).index(first_k)
            ax.axvline(first_i, color="#16a34a", linestyle=":", linewidth=1.5,
                       label=f"first sufficient k={first_k}")
            ax.legend(fontsize=8)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Sufficient fraction")
    ax.set_title("Sufficient fraction vs rank")
    ax.set_ylim(0, 1.05)

    # Panel 2: R² metrics
    ax = axes[1]
    colors = {"full_r2": "#9ca3af", "minimal_r2": "#3b82f6", "recovery_r2": "#ef4444", "pred_r2": "#10b981"}
    labels = {"full_r2": "full_r2", "minimal_r2": "minimal_r2", "recovery_r2": "recovery_r2", "pred_r2": "pred_r2 (vs full model)"}
    styles = {"full_r2": "--", "minimal_r2": "-", "recovery_r2": "-", "pred_r2": "-"}
    for metric in ["full_r2", "minimal_r2", "recovery_r2", "pred_r2"]:
        if metric not in active.columns:
            continue
        ax.semilogx(ks, active[metric].values, color=colors[metric],
                    linestyle=styles[metric], marker="o", markersize=4,
                    label=labels[metric], linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("R²")
    ax.set_title("R² metrics vs rank")
    ax.legend(fontsize=8)

    # Panel 3: n_selected
    ax = axes[2]
    if "n_selected" in active.columns:
        ax.semilogx(ks, active["n_selected"].values, color="#8b5cf6",
                    marker="o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Avg. selected SVD directions")
    ax.set_title("Selected directions vs rank")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_corrs_vs_arbitrary(corrs, corrs_arb, lags, save_path=None):
    """
    Plot real encoding correlations (blue) vs arbitrary null (red).

    Args:
        corrs: (n_folds, n_electrodes, n_lags).
        corrs_arb: (n_folds, n_electrodes, n_lags).
        lags: 1-D array of lag values (seconds).
        save_path: if given, save figure here; otherwise call plt.show().
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    add_corrs(corrs, lags, label="Real", color="blue", ax=ax)
    add_corrs(corrs_arb, lags, label="Arbitrary", color="red", ax=ax)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Correlation")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
