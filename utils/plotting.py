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
    full_corrs = results["full_corrs"]  # (n_elecs, n_lags)
    trunc_corrs = results["truncated_corrs"]  # (n_k, n_elecs, n_lags)
    sig_mask = results["sig_mask"]  # (n_elecs, n_lags)

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
    ax.axhline(
        full_nonsig, color="gray", linestyle="--", alpha=0.5, label="Full non-sig"
    )
    ax.axvline(best_k, color="red", linestyle=":", label=f"best_k={best_k}")
    ax.set_xlabel("k (SVD components)")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("Mean correlation vs k")
    ax.legend(fontsize=8)

    # Panel 2: retention ratio
    ax = axes[0, 1]
    ax.semilogx(k_values, ratios, "o-", color="steelblue")
    ax.axhline(
        threshold, color="orange", linestyle="--", label=f"Threshold {threshold:.0%}"
    )
    ax.axvline(best_k, color="red", linestyle=":", label=f"best_k={best_k}")
    ax.set_xlabel("k (SVD components)")
    ax.set_ylabel("Retention ratio")
    ax.set_title("Sig performance retention vs k")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Panel 3: per-electrode fractional drop at best_k
    ax = axes[1, 0]
    full_elec = full_corrs.mean(axis=1)  # (n_elecs,)
    trunc_elec = trunc_corrs[best_ki].mean(axis=1)  # (n_elecs,)
    drop_elec = (full_elec - trunc_elec) / np.maximum(np.abs(full_elec), 1e-12)
    frac_sig_elec = sig_mask.mean(axis=1)  # (n_elecs,)
    sc = ax.scatter(
        np.arange(len(drop_elec)),
        drop_elec,
        c=frac_sig_elec,
        cmap="viridis",
        s=10,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(sc, ax=ax, label="Frac sig lags")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Electrode index")
    ax.set_ylabel("Fractional drop at best_k")
    ax.set_title(f"Per-electrode drop at k={best_k}")

    # Panel 4: per-lag drop at best_k with sig-fraction overlay
    ax = axes[1, 1]
    full_lag = full_corrs.mean(axis=0)  # (n_lags,)
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


def add_masked_corrs(
    corrs, mask, lags, ax=None, color="blue", label=None, linestyle=None
):
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
    masked = np.where(mask[np.newaxis], corrs, np.nan)  # (n_folds, n_elecs, n_lags)
    per_fold = np.nanmean(masked, axis=1)  # (n_folds, n_lags)

    mean = np.nanmean(per_fold, axis=0)
    n = np.sum(~np.isnan(per_fold), axis=0)
    sem = np.nanstd(per_fold, axis=0) / np.sqrt(np.maximum(n, 1))

    kwargs = dict(color=color, label=label)
    if linestyle is not None:
        kwargs["linestyle"] = linestyle

    ax.plot(lags, mean, **kwargs)
    ax.fill_between(lags, mean - sem, mean + sem, alpha=0.2, color=color)


def add_sig_proportion_histogram(
    sig_mask, lags, ax=None, color="coral", alpha=0.2, label="Frac sig electrodes"
):
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

    sig_frac = sig_mask.mean(axis=0)  # (n_lags,)
    lag_width = np.diff(lags).mean() if len(lags) > 1 else 1.0

    ax2 = ax.twinx()
    ax2.bar(
        lags,
        sig_frac,
        width=lag_width * 0.9,
        color=color,
        alpha=alpha,
        label=label,
        zorder=0,
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Frac sig electrodes", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)  # let background bars show through
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
        (axes[0], sig_mask, "Significant lags", "steelblue"),
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


def add_corrs(
    corrs, lags, electrode=None, label=None, color="black", ax=None, linestyle=None
):
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
        data = np.nanmean(corrs, axis=1)  # average over electrodes → (n_folds, n_lags)

    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])

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
    add_corrs(
        corrs_trunc,
        lags,
        label=f"SVD k={best_k}",
        color="steelblue",
        linestyle="--",
        ax=ax,
    )
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


def plot_rrr_rank_results(
    summary_df, grid_df=None, title="RRR: perf vs rank", save_path=None
):
    """Multi-panel figure summarising RRR rank analysis.

    Row 1 (R² metrics — always shown):
        Panel 1: full_r2 - minimal_r2 (gap) and recovery_r2 vs rank.
        Panel 2: Absolute R² (full_r2 dashed, minimal_r2 solid).
        Panel 3: minimal_r2 / full_r2 ratio.
        Panel 4: Fold × rank heatmap of recovery_r2 (omitted when grid_df is None).

    Row 2 (correlation metrics — shown when full_corr column exists):
        Panel 5: Absolute correlations (full_corr, minimal_corr, recovery_corr).
        Panel 6: minimal_corr / full_corr ratio.
        Panel 7: pred_r2 vs rank (R² of rank-k vs full predictions).
        Panel 8: Fold × rank heatmap of recovery_corr.

    Args:
        summary_df: DataFrame with columns [k, full_r2, minimal_r2, recovery_r2]
            averaged over folds (i.e. summary.csv). May also contain
            full_corr, minimal_corr, recovery_corr, pred_r2.
        grid_df: Optional per-fold DataFrame with columns [k, fold, ...]
            (i.e. grid_results.csv).
        title: Figure title.
        save_path: If given, save figure here and close it; otherwise return the Figure.

    Returns:
        matplotlib Figure when save_path is None, else None.
    """
    import matplotlib.pyplot as plt

    has_folds = grid_df is not None
    has_corr = (
        "full_corr" in summary_df.columns and "minimal_corr" in summary_df.columns
    )
    has_sig_split = "minimal_corr_sig" in summary_df.columns
    has_effective_rank = "effective_rank" in summary_df.columns
    has_row3 = has_sig_split or has_effective_rank

    n_cols = 4 if has_folds else 3
    n_rows = 1 + int(has_corr) + int(has_row3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # uniform 2D indexing

    ks = summary_df["k"].values
    gap = summary_df["full_r2"].values - summary_df["minimal_r2"].values
    ratio = summary_df["minimal_r2"].values / np.where(
        summary_df["full_r2"].values > 0, summary_df["full_r2"].values, np.nan
    )

    # ===== Row 1: R² panels =====

    # --- Panel 1: gap + recovery_r2 ---
    ax1 = axes[0, 0]
    if has_folds:
        for _, fold_df in grid_df.groupby("fold"):
            fold_gap = fold_df["full_r2"].values - fold_df["minimal_r2"].values
            ax1.semilogx(
                fold_df["k"], fold_gap, color="#f59e0b", alpha=0.2, linewidth=0.8
            )
        if "recovery_r2" in grid_df.columns:
            for _, fold_df in grid_df.groupby("fold"):
                ax1.semilogx(
                    fold_df["k"],
                    fold_df["recovery_r2"],
                    color="#ef4444",
                    alpha=0.2,
                    linewidth=0.8,
                )

    ax1.semilogx(
        ks,
        gap,
        color="#f59e0b",
        linestyle="-",
        marker="o",
        markersize=4,
        label="full_r2 − minimal_r2 (gap)",
        linewidth=1.5,
    )
    if "recovery_r2" in summary_df.columns:
        ax1.semilogx(
            ks,
            summary_df["recovery_r2"],
            color="#ef4444",
            linestyle="-",
            marker="o",
            markersize=4,
            label="recovery_r2 (residual)",
            linewidth=1.5,
        )
    ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_xlabel("Rank k")
    ax1.set_ylabel("R²")
    ax1.set_title(f"{title}\n(gap + residual)")
    ax1.legend(fontsize=8)

    # --- Panel 2: absolute R² ---
    ax2 = axes[0, 1]
    if has_folds:
        for _, fold_df in grid_df.groupby("fold"):
            ax2.semilogx(
                fold_df["k"],
                fold_df["full_r2"],
                color="#9ca3af",
                alpha=0.2,
                linewidth=0.8,
            )
            ax2.semilogx(
                fold_df["k"],
                fold_df["minimal_r2"],
                color="#3b82f6",
                alpha=0.2,
                linewidth=0.8,
            )

    ax2.semilogx(
        ks,
        summary_df["full_r2"],
        color="#9ca3af",
        linestyle="--",
        marker="o",
        markersize=4,
        label="full_r2 (ceiling)",
        linewidth=1.5,
    )
    ax2.semilogx(
        ks,
        summary_df["minimal_r2"],
        color="#3b82f6",
        linestyle="-",
        marker="o",
        markersize=4,
        label="minimal_r2 (RRR)",
        linewidth=1.5,
    )
    ax2.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax2.set_xlabel("Rank k")
    ax2.set_ylabel("R²")
    ax2.set_title(f"{title}\n(absolute R²)")
    ax2.legend(fontsize=8)

    # --- Panel 3: ratio minimal_r2 / full_r2 ---
    ax3 = axes[0, 2]
    if has_folds:
        for _, fold_df in grid_df.groupby("fold"):
            fold_ratio = fold_df["minimal_r2"].values / np.where(
                fold_df["full_r2"].values > 0, fold_df["full_r2"].values, np.nan
            )
            ax3.semilogx(
                fold_df["k"], fold_ratio, color="#8b5cf6", alpha=0.2, linewidth=0.8
            )

    ax3.semilogx(
        ks,
        ratio,
        color="#8b5cf6",
        linestyle="-",
        marker="o",
        markersize=4,
        label="minimal_r2 / full_r2",
        linewidth=1.5,
    )
    ax3.axhline(1, color="black", linewidth=0.5, linestyle=":")
    ax3.set_xlabel("Rank k")
    ax3.set_ylabel("Fraction of ceiling")
    ax3.set_title(f"{title}\n(normalised ratio)")
    ax3.legend(fontsize=8)

    # --- Panel 4: heatmap recovery_r2 per fold ---
    if has_folds:
        folds = sorted(grid_df["fold"].unique())
        ks_ord = sorted(grid_df["k"].unique())
        k_idx = {k: i for i, k in enumerate(ks_ord)}
        f_idx = {f: i for i, f in enumerate(folds)}
        hm = np.full((len(folds), len(ks_ord)), np.nan)
        for _, row in grid_df.iterrows():
            hm[f_idx[row["fold"]], k_idx[row["k"]]] = row["recovery_r2"]

        ax4 = axes[0, 3]
        im = ax4.imshow(hm, aspect="auto", cmap="Reds_r")
        ax4.set_xticks(range(len(ks_ord)))
        ax4.set_xticklabels([str(k) for k in ks_ord], rotation=45, ha="right")
        ax4.set_yticks(range(len(folds)))
        ax4.set_yticklabels([f"fold {f}" for f in folds])
        ax4.set_xlabel("Rank k")
        ax4.set_title("Recovery R² per fold\n(lower = more sufficient)")
        plt.colorbar(im, ax=ax4, label="recovery_r2")

    # ===== Row 2: Correlation panels (only when correlation data available) =====

    if has_corr:
        # --- Panel 5: absolute correlations + recovery_corr ---
        ax5 = axes[1, 0]
        if has_folds:
            for _, fold_df in grid_df.groupby("fold"):
                ax5.semilogx(
                    fold_df["k"],
                    fold_df["full_corr"],
                    color="#9ca3af",
                    alpha=0.2,
                    linewidth=0.8,
                )
                ax5.semilogx(
                    fold_df["k"],
                    fold_df["minimal_corr"],
                    color="#10b981",
                    alpha=0.2,
                    linewidth=0.8,
                )
                if "recovery_corr" in fold_df.columns:
                    ax5.semilogx(
                        fold_df["k"],
                        fold_df["recovery_corr"],
                        color="#ef4444",
                        alpha=0.2,
                        linewidth=0.8,
                    )

        ax5.semilogx(
            ks,
            summary_df["full_corr"],
            color="#9ca3af",
            linestyle="--",
            marker="o",
            markersize=4,
            label="full_corr (ceiling)",
            linewidth=1.5,
        )
        ax5.semilogx(
            ks,
            summary_df["minimal_corr"],
            color="#10b981",
            linestyle="-",
            marker="o",
            markersize=4,
            label="minimal_corr (RRR)",
            linewidth=1.5,
        )
        if "recovery_corr" in summary_df.columns:
            ax5.semilogx(
                ks,
                summary_df["recovery_corr"],
                color="#ef4444",
                linestyle="-",
                marker="o",
                markersize=4,
                label="recovery_corr (residual)",
                linewidth=1.5,
            )
        ax5.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax5.set_xlabel("Rank k")
        ax5.set_ylabel("Correlation")
        ax5.set_title(f"{title}\n(absolute correlation)")
        ax5.legend(fontsize=8)

        # --- Panel 6: ratio minimal_corr / full_corr ---
        ax6 = axes[1, 1]
        corr_ratio = summary_df["minimal_corr"].values / np.where(
            summary_df["full_corr"].values > 0, summary_df["full_corr"].values, np.nan
        )
        if has_folds:
            for _, fold_df in grid_df.groupby("fold"):
                fold_corr_ratio = fold_df["minimal_corr"].values / np.where(
                    fold_df["full_corr"].values > 0, fold_df["full_corr"].values, np.nan
                )
                ax6.semilogx(
                    fold_df["k"],
                    fold_corr_ratio,
                    color="#8b5cf6",
                    alpha=0.2,
                    linewidth=0.8,
                )

        ax6.semilogx(
            ks,
            corr_ratio,
            color="#8b5cf6",
            linestyle="-",
            marker="o",
            markersize=4,
            label="minimal_corr / full_corr",
            linewidth=1.5,
        )
        ax6.axhline(1, color="black", linewidth=0.5, linestyle=":")
        ax6.set_xlabel("Rank k")
        ax6.set_ylabel("Fraction of ceiling")
        ax6.set_title(f"{title}\n(correlation ratio)")
        ax6.legend(fontsize=8)

        # --- Panel 7: pred_r2 vs rank ---
        ax7 = axes[1, 2]
        if "pred_r2" in summary_df.columns:
            if has_folds and "pred_r2" in grid_df.columns:
                for _, fold_df in grid_df.groupby("fold"):
                    ax7.semilogx(
                        fold_df["k"],
                        fold_df["pred_r2"],
                        color="#f97316",
                        alpha=0.2,
                        linewidth=0.8,
                    )

            ax7.semilogx(
                ks,
                summary_df["pred_r2"],
                color="#f97316",
                linestyle="-",
                marker="o",
                markersize=4,
                label="pred_r2",
                linewidth=1.5,
            )
            ax7.axhline(1, color="black", linewidth=0.5, linestyle=":")
            ax7.set_xlabel("Rank k")
            ax7.set_ylabel("R² (Y_hat_full as target)")
            ax7.set_title(f"{title}\n(pred R²: rank-k vs full)")
            ax7.legend(fontsize=8)

        # --- Panel 8: heatmap recovery_corr per fold ---
        if has_folds and "recovery_corr" in grid_df.columns:
            ax8 = axes[1, 3]
            hm_corr = np.full((len(folds), len(ks_ord)), np.nan)
            for _, row in grid_df.iterrows():
                hm_corr[f_idx[row["fold"]], k_idx[row["k"]]] = row["recovery_corr"]

            im2 = ax8.imshow(hm_corr, aspect="auto", cmap="Reds_r")
            ax8.set_xticks(range(len(ks_ord)))
            ax8.set_xticklabels([str(k) for k in ks_ord], rotation=45, ha="right")
            ax8.set_yticks(range(len(folds)))
            ax8.set_yticklabels([f"fold {f}" for f in folds])
            ax8.set_xlabel("Rank k")
            ax8.set_title("Recovery corr per fold\n(lower = more sufficient)")
            plt.colorbar(im2, ax=ax8, label="recovery_corr")

    # ===== Row 3: Sig/non-sig split + effective rank (when available) =====

    row3_idx = 1 + int(has_corr)  # row index for row 3

    if has_row3 and has_sig_split:
        # --- Panel: Sig correlations vs k ---
        ax_sig = axes[row3_idx, 0]
        if has_folds:
            for _, fold_df in grid_df.groupby("fold"):
                ax_sig.semilogx(
                    fold_df["k"],
                    fold_df["minimal_corr_sig"],
                    color="#10b981",
                    alpha=0.2,
                    linewidth=0.8,
                )
        ax_sig.semilogx(
            ks,
            summary_df["full_corr_sig"],
            color="#9ca3af",
            linestyle="--",
            marker="o",
            markersize=4,
            label="full_corr_sig (ceiling)",
            linewidth=1.5,
        )
        ax_sig.semilogx(
            ks,
            summary_df["minimal_corr_sig"],
            color="#10b981",
            linestyle="-",
            marker="o",
            markersize=4,
            label="minimal_corr_sig (RRR)",
            linewidth=1.5,
        )
        if "recovery_corr_sig" in summary_df.columns:
            ax_sig.semilogx(
                ks,
                summary_df["recovery_corr_sig"],
                color="#ef4444",
                linestyle="-",
                marker="o",
                markersize=4,
                label="recovery_corr_sig",
                linewidth=1.5,
            )
        ax_sig.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax_sig.set_xlabel("Rank k")
        ax_sig.set_ylabel("Correlation")
        ax_sig.set_title(f"{title}\n(significant lags)")
        ax_sig.legend(fontsize=8)

        # --- Panel: Non-sig correlations vs k ---
        ax_nonsig = axes[row3_idx, 1]
        if has_folds:
            for _, fold_df in grid_df.groupby("fold"):
                ax_nonsig.semilogx(
                    fold_df["k"],
                    fold_df["minimal_corr_nonsig"],
                    color="#f97316",
                    alpha=0.2,
                    linewidth=0.8,
                )
        ax_nonsig.semilogx(
            ks,
            summary_df["full_corr_nonsig"],
            color="#9ca3af",
            linestyle="--",
            marker="o",
            markersize=4,
            label="full_corr_nonsig (ceiling)",
            linewidth=1.5,
        )
        ax_nonsig.semilogx(
            ks,
            summary_df["minimal_corr_nonsig"],
            color="#f97316",
            linestyle="-",
            marker="o",
            markersize=4,
            label="minimal_corr_nonsig (RRR)",
            linewidth=1.5,
        )
        if "recovery_corr_nonsig" in summary_df.columns:
            ax_nonsig.semilogx(
                ks,
                summary_df["recovery_corr_nonsig"],
                color="#ef4444",
                linestyle="-",
                marker="o",
                markersize=4,
                label="recovery_corr_nonsig",
                linewidth=1.5,
            )
        ax_nonsig.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax_nonsig.set_xlabel("Rank k")
        ax_nonsig.set_ylabel("Correlation")
        ax_nonsig.set_title(f"{title}\n(non-significant lags)")
        ax_nonsig.legend(fontsize=8)

        # --- Panel: Sig vs non-sig overlay ---
        ax_overlay = axes[row3_idx, 2]
        if has_folds:
            for _, fold_df in grid_df.groupby("fold"):
                ax_overlay.semilogx(
                    fold_df["k"],
                    fold_df["minimal_corr_sig"],
                    color="#10b981",
                    alpha=0.2,
                    linewidth=0.8,
                )
                ax_overlay.semilogx(
                    fold_df["k"],
                    fold_df["minimal_corr_nonsig"],
                    color="#f97316",
                    alpha=0.2,
                    linewidth=0.8,
                )
        ax_overlay.semilogx(
            ks,
            summary_df["minimal_corr_sig"],
            color="#10b981",
            linestyle="-",
            marker="o",
            markersize=4,
            label="sig",
            linewidth=1.5,
        )
        ax_overlay.semilogx(
            ks,
            summary_df["minimal_corr_nonsig"],
            color="#f97316",
            linestyle="-",
            marker="o",
            markersize=4,
            label="non-sig",
            linewidth=1.5,
        )
        ax_overlay.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax_overlay.set_xlabel("Rank k")
        ax_overlay.set_ylabel("Correlation")
        ax_overlay.set_title(f"{title}\n(sig vs non-sig)")
        ax_overlay.legend(fontsize=8)

    if has_row3 and has_effective_rank:
        # --- Panel: Effective rank vs k ---
        col_idx = 3 if has_sig_split else 0
        ax_rank = axes[row3_idx, col_idx]
        if has_folds:
            for _, fold_df in grid_df.groupby("fold"):
                ax_rank.plot(
                    fold_df["k"],
                    fold_df["effective_rank"],
                    color="#6366f1",
                    alpha=0.2,
                    linewidth=0.8,
                    marker=".",
                    markersize=3,
                )
        ax_rank.plot(
            ks,
            summary_df["effective_rank"],
            color="#6366f1",
            linestyle="-",
            marker="o",
            markersize=4,
            label="effective rank",
            linewidth=1.5,
        )
        # y = x reference line
        k_range = np.array([ks.min(), ks.max()])
        ax_rank.plot(
            k_range, k_range, color="black", linewidth=0.8, linestyle=":", label="y = k"
        )
        ax_rank.set_xlabel("Configured rank k")
        ax_rank.set_ylabel("Effective rank")
        ax_rank.set_title(f"{title}\n(effective rank of A)")
        ax_rank.legend(fontsize=8)

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
            ax.axvline(
                first_i,
                color="#16a34a",
                linestyle=":",
                linewidth=1.5,
                label=f"first sufficient k={first_k}",
            )
            ax.legend(fontsize=8)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Sufficient fraction")
    ax.set_title("Sufficient fraction vs rank")
    ax.set_ylim(0, 1.05)

    # Panel 2: R² metrics
    ax = axes[1]
    colors = {
        "full_r2": "#9ca3af",
        "minimal_r2": "#3b82f6",
        "recovery_r2": "#ef4444",
        "pred_r2": "#10b981",
    }
    labels = {
        "full_r2": "full_r2",
        "minimal_r2": "minimal_r2",
        "recovery_r2": "recovery_r2",
        "pred_r2": "pred_r2 (vs full model)",
    }
    styles = {"full_r2": "--", "minimal_r2": "-", "recovery_r2": "-", "pred_r2": "-"}
    for metric in ["full_r2", "minimal_r2", "recovery_r2", "pred_r2"]:
        if metric not in active.columns:
            continue
        ax.semilogx(
            ks,
            active[metric].values,
            color=colors[metric],
            linestyle=styles[metric],
            marker="o",
            markersize=4,
            label=labels[metric],
            linewidth=1.5,
        )
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("R²")
    ax.set_title("R² metrics vs rank")
    ax.legend(fontsize=8)

    # Panel 3: n_selected
    ax = axes[2]
    if "n_selected" in active.columns:
        ax.semilogx(
            ks,
            active["n_selected"].values,
            color="#8b5cf6",
            marker="o",
            markersize=4,
            linewidth=1.5,
        )
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


def plot_partial_svd_sweep(
    all_corrs,
    baseline_corrs,
    sig_mask,
    group_sizes,
    title="Partial SVD sweep",
    save_path=None,
):
    """Two-panel figure showing correlation drop as SVD directions are partialled out.

    Panel 1: Mean correlation (sig electrode-lag pairs) vs window start index.
    Panel 2: Same for non-significant electrode-lag pairs.
    One line per group size, with a dashed baseline.

    Args:
        all_corrs: dict mapping (group_size, window_start) → (n_folds, n_elecs, n_lags).
        baseline_corrs: (n_folds, n_elecs, n_lags) from the full (un-partialled) model.
        sig_mask: (n_elecs, n_lags) boolean significance mask.
        group_sizes: list of group sizes to plot (one line each).
        title: figure title.
        save_path: if given, save figure here; otherwise call plt.show().
    """
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(group_sizes), 1)))

    baseline_sig = baseline_corrs.mean(axis=0)[sig_mask].mean()
    baseline_nonsig = baseline_corrs.mean(axis=0)[~sig_mask].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for panel_idx, (ax, mask, mask_label, bl_val) in enumerate(
        [
            (axes[0], sig_mask, "Significant", baseline_sig),
            (axes[1], ~sig_mask, "Non-significant", baseline_nonsig),
        ]
    ):
        ax.axhline(bl_val, color="black", linestyle="--", alpha=0.5, label="Baseline")

        for gi, gs in enumerate(group_sizes):
            window_starts = sorted(ws for (g, ws) in all_corrs.keys() if g == gs)
            if not window_starts:
                continue
            mean_corrs = []
            for ws in window_starts:
                corrs = all_corrs[(gs, ws)]  # (n_folds, n_elecs, n_lags)
                mean_corrs.append(corrs.mean(axis=0)[mask].mean())
            ax.plot(
                window_starts,
                mean_corrs,
                "o-",
                color=colors[gi],
                label=f"group_size={gs}",
                markersize=3,
            )

        ax.set_xlabel("Window start (SVD direction index)")
        ax.set_ylabel("Mean Pearson r")
        ax.set_title(f"{mask_label} electrodes")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_partial_svd_lag_heatmap(
    partialled_corrs,
    baseline_corrs,
    sig_mask,
    lags,
    title="Correlation drop per electrode-lag",
    save_path=None,
):
    """Heatmap of correlation drop (baseline - partialled) for significant pairs.

    Args:
        partialled_corrs: (n_folds, n_elecs, n_lags) from one partialling window.
        baseline_corrs: (n_folds, n_elecs, n_lags) full-model correlations.
        sig_mask: (n_elecs, n_lags) boolean significance mask.
        lags: 1-D array of lag values (seconds).
        title: figure title.
        save_path: if given, save figure here; otherwise call plt.show().
    """
    import matplotlib.pyplot as plt

    drop = baseline_corrs.mean(axis=0) - partialled_corrs.mean(
        axis=0
    )  # (n_elecs, n_lags)
    drop_masked = np.where(sig_mask, drop, np.nan)

    # Keep only electrodes with at least one significant lag
    has_sig = sig_mask.any(axis=1)
    drop_filtered = drop_masked[has_sig]

    fig, ax = plt.subplots(figsize=(12, max(4, drop_filtered.shape[0] * 0.15)))
    im = ax.imshow(
        drop_filtered,
        aspect="auto",
        cmap="RdBu_r",
        extent=[lags[0], lags[-1], drop_filtered.shape[0] - 0.5, -0.5],
    )
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Electrode index (sig only)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Correlation drop (baseline - partialled)")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_partial_rrr_sweep(
    all_corrs,
    baseline_corrs,
    sig_mask,
    title="Partial RRR sweep",
    save_path=None,
):
    """Two-panel figure showing correlation drop as RRR subspaces are partialled out.

    Panel 1: Mean correlation (sig electrode-lag pairs) vs rank k.
    Panel 2: Same for non-significant electrode-lag pairs.

    Args:
        all_corrs: dict mapping rank_k (int) → (n_folds, n_elecs, n_lags).
        baseline_corrs: (n_folds, n_elecs, n_lags) from the full (un-partialled) model.
        sig_mask: (n_elecs, n_lags) boolean significance mask.
        title: figure title.
        save_path: if given, save figure here; otherwise call plt.show().
    """
    import matplotlib.pyplot as plt

    baseline_sig = baseline_corrs.mean(axis=0)[sig_mask].mean()
    baseline_nonsig = baseline_corrs.mean(axis=0)[~sig_mask].mean()

    rank_ks = sorted(all_corrs.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mask, mask_label, bl_val in [
        (axes[0], sig_mask, "Significant", baseline_sig),
        (axes[1], ~sig_mask, "Non-significant", baseline_nonsig),
    ]:
        ax.axhline(bl_val, color="black", linestyle="--", alpha=0.5, label="Baseline")

        mean_corrs = []
        for k in rank_ks:
            corrs = all_corrs[k]  # (n_folds, n_elecs, n_lags)
            mean_corrs.append(corrs.mean(axis=0)[mask].mean())
        ax.plot(rank_ks, mean_corrs, "o-", color="steelblue", markersize=4)

        ax.set_xlabel("Rank k (RRR subspace dimension)")
        ax.set_ylabel("Mean Pearson r")
        ax.set_title(f"{mask_label} electrodes")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


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


def plot_sae_corrs_vs_full(corrs_sae, corrs_full, lags, corrs_rrr=None,
                           title=None, save_path=None, sig_mask=None):
    """Plot SAE correlation timecourse vs full encoding model baseline.

    Args:
        corrs_sae: (n_folds, n_electrodes, n_lags) SAE per-target correlations.
        corrs_full: (n_folds, n_electrodes, n_lags) full model per-target correlations.
        lags: 1-D array of lag values (seconds).
        corrs_rrr: Optional (n_folds, n_electrodes, n_lags) RRR-direct regression
            per-target correlations (linear regression from RRR subspace, no SAE).
        title: Optional plot title.
        save_path: If given, save figure here; otherwise call plt.show().
        sig_mask: Optional (n_electrodes, n_lags) boolean mask of significant pairs.
            When provided, two additional panels are added for sig-only and nonsig-only.
    """
    import matplotlib.pyplot as plt

    ncols = 3 if sig_mask is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 4), squeeze=False)

    def _plot_panel(ax, corrs_sae_p, corrs_full_p, corrs_rrr_p, panel_title):
        add_corrs(corrs_full_p, lags, label="Full model", color="blue", ax=ax)
        if corrs_rrr_p is not None:
            add_corrs(corrs_rrr_p, lags, label="RRR direct", color="green", ax=ax)
        add_corrs(corrs_sae_p, lags, label="SAE", color="orange", ax=ax)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Correlation")
        ax.set_title(panel_title)
        ax.legend()

    # Panel 1: all electrode-lag pairs
    panel_title = title or "All"
    _plot_panel(axes[0, 0], corrs_sae, corrs_full, corrs_rrr, panel_title)

    if sig_mask is not None:
        mask = sig_mask[None, :, :]  # broadcast to (1, n_electrodes, n_lags)

        # Panel 2: significant pairs only
        corrs_sae_sig = np.where(mask, corrs_sae, np.nan)
        corrs_full_sig = np.where(mask, corrs_full, np.nan)
        corrs_rrr_sig = np.where(mask, corrs_rrr, np.nan) if corrs_rrr is not None else None
        _plot_panel(axes[0, 1], corrs_sae_sig, corrs_full_sig, corrs_rrr_sig,
                    "Significant only")

        # Panel 3: non-significant pairs only
        nonsig = ~mask
        corrs_sae_nonsig = np.where(nonsig, corrs_sae, np.nan)
        corrs_full_nonsig = np.where(nonsig, corrs_full, np.nan)
        corrs_rrr_nonsig = np.where(nonsig, corrs_rrr, np.nan) if corrs_rrr is not None else None
        _plot_panel(axes[0, 2], corrs_sae_nonsig, corrs_full_nonsig, corrs_rrr_nonsig,
                    "Non-significant only")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_sae_training_curves(training_log, title="SAE training", save_path=None,
                             rrr_recovery_corr=None, rrr_recovery_corr_sig=None,
                             rrr_recovery_corr_nonsig=None):
    """Multi-panel figure of SAE training metrics vs epoch/cycle.

    Panels (up to 3 rows x 3 columns):
        Row 1: Loss components, Test R², Test correlation.
        Row 2: L0 sparsity, Dead neuron fraction, Mean activation.
        Row 3: Recovery correlation (sufficiency test during training).
            Only shown when training log contains recovery_corr.

    Rows with None test/SAE metrics (skipped by log_every) are omitted from
    panels that use those columns.

    Args:
        training_log: List of dicts from train_sae_end_to_end or
            train_sae_alternating.  Each dict has keys like epoch/cycle,
            mse_loss, l1_loss, total_loss, test_r2, test_corr, l0_sparsity,
            dead_neurons, mean_activation.  Optionally recovery_corr,
            recovery_r2, recovery_corr_sig, recovery_corr_nonsig when a
            log_callback was provided during training.
        title: Figure title.
        save_path: If given, save figure here and close; otherwise return Figure.
        rrr_recovery_corr: If given, draw a horizontal dashed line at this value
            in the recovery correlation panel (RRR baseline for comparison).
        rrr_recovery_corr_sig: RRR recovery_corr for significant targets only.
        rrr_recovery_corr_nonsig: RRR recovery_corr for non-significant targets only.

    Returns:
        matplotlib Figure when save_path is None, else None.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(training_log)
    step_col = "epoch" if "epoch" in df.columns else "cycle"
    steps = df[step_col]

    # Determine whether row 3 (recovery) is needed
    has_recovery = "recovery_corr" in df.columns
    df_recovery = df.dropna(subset=["recovery_corr"]) if has_recovery else pd.DataFrame()
    show_row3 = len(df_recovery) > 0

    n_rows = 3 if show_row3 else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    fig.suptitle(title, fontsize=12)

    # Row 1, Panel 1: Loss components
    ax = axes[0, 0]
    ax.plot(steps, df["total_loss"], label="total", linewidth=1.5)
    ax.plot(steps, df["mse_loss"], label="MSE", linewidth=1, alpha=0.8)
    ax.plot(steps, df["l1_loss"], label="L1", linewidth=1, alpha=0.8)
    ax.set_xlabel(step_col)
    ax.set_ylabel("loss")
    ax.set_title("Loss components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Filter to rows with test metrics computed
    df_logged = df.dropna(subset=["test_r2"])
    if len(df_logged) > 0:
        log_steps = df_logged[step_col]

        # Row 1, Panel 2: Test R²
        ax = axes[0, 1]
        ax.plot(log_steps, df_logged["test_r2"], marker=".", markersize=3, linewidth=1)
        ax.set_xlabel(step_col)
        ax.set_ylabel("R²")
        ax.set_title("Test R²")
        ax.grid(True, alpha=0.3)

        # Row 1, Panel 3: Test correlation
        ax = axes[0, 2]
        ax.plot(log_steps, df_logged["test_corr"], marker=".", markersize=3, linewidth=1, label="all")
        # Sig/nonsig correlation split (from log callback)
        has_corr_sig = "test_corr_sig" in df_logged.columns
        df_corr_sig = df_logged.dropna(subset=["test_corr_sig"]) if has_corr_sig else pd.DataFrame()
        if len(df_corr_sig) > 0:
            ax.plot(df_corr_sig[step_col], df_corr_sig["test_corr_sig"],
                    marker=".", markersize=2, linewidth=0.8, alpha=0.7,
                    color="tab:blue", label="sig")
            ax.plot(df_corr_sig[step_col], df_corr_sig["test_corr_nonsig"],
                    marker=".", markersize=2, linewidth=0.8, alpha=0.7,
                    color="tab:red", label="nonsig")
            ax.legend(fontsize=7)
        ax.set_xlabel(step_col)
        ax.set_ylabel("correlation")
        ax.set_title("Test correlation")
        ax.grid(True, alpha=0.3)
    else:
        for ax in axes[0, 1:]:
            ax.set_visible(False)

    # Row 2: SAE-specific metrics (only at logged steps)
    has_sae_cols = "l0_sparsity" in df.columns
    df_sae = df.dropna(subset=["l0_sparsity"]) if has_sae_cols else pd.DataFrame()

    if len(df_sae) > 0:
        sae_steps = df_sae[step_col]

        # Row 2, Panel 1: L0 sparsity
        ax = axes[1, 0]
        ax.plot(sae_steps, df_sae["l0_sparsity"], marker=".", markersize=3, linewidth=1, color="tab:orange")
        ax.set_xlabel(step_col)
        ax.set_ylabel("L0 (avg active features)")
        ax.set_title("L0 sparsity")
        ax.grid(True, alpha=0.3)

        # Row 2, Panel 2: Dead neuron fraction
        ax = axes[1, 1]
        ax.plot(sae_steps, df_sae["dead_neurons"], marker=".", markersize=3, linewidth=1, color="tab:red")
        ax.set_xlabel(step_col)
        ax.set_ylabel("fraction dead")
        ax.set_title("Dead neuron fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Row 2, Panel 3: Mean activation
        ax = axes[1, 2]
        ax.plot(sae_steps, df_sae["mean_activation"], marker=".", markersize=3, linewidth=1, color="tab:green")
        ax.set_xlabel(step_col)
        ax.set_ylabel("mean activation")
        ax.set_title("Mean activation (non-zero)")
        ax.grid(True, alpha=0.3)
    else:
        for ax in axes[1, :]:
            ax.set_visible(False)

    # Row 3: Recovery correlation (sufficiency test during training)
    if show_row3:
        recovery_steps = df_recovery[step_col]

        def _is_valid(val):
            return val is not None and not np.isnan(val)

        # Panel 1: Overall recovery correlation
        ax = axes[2, 0]
        ax.plot(recovery_steps, df_recovery["recovery_corr"],
                marker=".", markersize=3, linewidth=1, color="tab:purple",
                label="SAE")
        if _is_valid(rrr_recovery_corr):
            ax.axhline(rrr_recovery_corr, color="tab:gray", linestyle="--",
                        linewidth=1.5, label="RRR")
        ax.set_xlabel(step_col)
        ax.set_ylabel("recovery corr")
        ax.set_title("Sufficiency: recovery corr (all)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: Sig recovery correlation
        has_sig = "recovery_corr_sig" in df_recovery.columns
        df_sig = df_recovery.dropna(subset=["recovery_corr_sig"]) if has_sig else pd.DataFrame()
        if len(df_sig) > 0:
            ax = axes[2, 1]
            sig_steps = df_sig[step_col]
            ax.plot(sig_steps, df_sig["recovery_corr_sig"],
                    marker=".", markersize=3, linewidth=1, color="tab:blue",
                    label="SAE (sig)")
            if _is_valid(rrr_recovery_corr_sig):
                ax.axhline(rrr_recovery_corr_sig, color="tab:blue", linestyle="--",
                            linewidth=1.5, alpha=0.5, label="RRR (sig)")
            ax.set_xlabel(step_col)
            ax.set_ylabel("recovery corr")
            ax.set_title("Sufficiency: recovery corr (sig)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            axes[2, 1].set_visible(False)

        # Panel 3: Non-sig recovery correlation
        has_nonsig = "recovery_corr_nonsig" in df_recovery.columns
        df_nonsig = df_recovery.dropna(subset=["recovery_corr_nonsig"]) if has_nonsig else pd.DataFrame()
        if len(df_nonsig) > 0:
            ax = axes[2, 2]
            nonsig_steps = df_nonsig[step_col]
            ax.plot(nonsig_steps, df_nonsig["recovery_corr_nonsig"],
                    marker=".", markersize=3, linewidth=1, color="tab:red",
                    label="SAE (nonsig)")
            if _is_valid(rrr_recovery_corr_nonsig):
                ax.axhline(rrr_recovery_corr_nonsig, color="tab:red", linestyle="--",
                            linewidth=1.5, alpha=0.5, label="RRR (nonsig)")
            ax.set_xlabel(step_col)
            ax.set_ylabel("recovery corr")
            ax.set_title("Sufficiency: recovery corr (nonsig)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            axes[2, 2].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig


def plot_sae_recovery_summary(grid_results_df, title="SAE vs RRR recovery",
                              save_path=None):
    """Summary figure comparing SAE vs RRR recovery and prediction quality.

    Layout (up to 2x2):
        (0,0): Overall recovery correlation (lower = better).
        (0,1): Prediction correlation (higher = better).
        (1,0): Sig-only recovery correlation (if available).
        (1,1): Nonsig-only recovery correlation (if available).

    Args:
        grid_results_df: DataFrame with columns fold, recovery_corr,
            rrr_recovery_corr, correlation, rrr_baseline_corr.  Optionally
            recovery_corr_sig, recovery_corr_nonsig, rrr_recovery_corr_sig,
            rrr_recovery_corr_nonsig for significance-stratified panels.
        title: Figure title.
        save_path: If given, save figure here and close; otherwise return Figure.

    Returns:
        matplotlib Figure when save_path is None, else None.
    """
    import matplotlib.pyplot as plt

    df = grid_results_df.sort_values("fold")
    folds = df["fold"].values
    x = np.arange(len(folds))
    width = 0.35

    has_sig_split = (
        "recovery_corr_sig" in df.columns
        and "rrr_recovery_corr_sig" in df.columns
        and df["recovery_corr_sig"].notna().any()
    )

    n_rows = 2 if has_sig_split else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=12)

    def _recovery_bar_panel(ax, rrr_col, sae_col, panel_title, color="tab:purple"):
        ax.bar(x - width / 2, df[rrr_col], width,
               label="RRR", color="tab:gray", alpha=0.7)
        ax.bar(x + width / 2, df[sae_col], width,
               label="SAE", color=color, alpha=0.7)
        rrr_m = df[rrr_col].mean()
        sae_m = df[sae_col].mean()
        ax.axhline(rrr_m, color="tab:gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(sae_m, color=color, linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Recovery correlation")
        ax.set_title(f"{panel_title}\n"
                     f"RRR mean: {rrr_m:.4f}, SAE mean: {sae_m:.4f}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(f) for f in folds])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # (0,0): Overall recovery correlation (lower is better)
    _recovery_bar_panel(axes[0, 0], "rrr_recovery_corr", "recovery_corr",
                        "Sufficiency — all targets (lower = better)")

    # (0,1): Prediction correlation (higher is better)
    if "rrr_baseline_corr" in df.columns and "correlation" in df.columns:
        ax = axes[0, 1]
        ax.bar(x - width / 2, df["rrr_baseline_corr"], width,
               label="RRR", color="tab:gray", alpha=0.7)
        ax.bar(x + width / 2, df["correlation"], width,
               label="SAE", color="tab:orange", alpha=0.7)
        rrr_corr_mean = df["rrr_baseline_corr"].mean()
        sae_corr_mean = df["correlation"].mean()
        ax.axhline(rrr_corr_mean, color="tab:gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(sae_corr_mean, color="tab:orange", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Prediction correlation")
        ax.set_title(f"Prediction quality (higher = better)\n"
                     f"RRR mean: {rrr_corr_mean:.4f}, SAE mean: {sae_corr_mean:.4f}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(f) for f in folds])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        axes[0, 1].set_visible(False)

    # Row 2: Sig/nonsig split
    if has_sig_split:
        _recovery_bar_panel(axes[1, 0], "rrr_recovery_corr_sig", "recovery_corr_sig",
                            "Sufficiency — sig targets (lower = better)",
                            color="tab:blue")
        _recovery_bar_panel(axes[1, 1], "rrr_recovery_corr_nonsig", "recovery_corr_nonsig",
                            "Sufficiency — nonsig targets (lower = better)",
                            color="tab:red")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig


# ---------------------------------------------------------------------------
# Sweep summary plots
# ---------------------------------------------------------------------------

# Metrics to include in sweep summary, with direction (True = higher is better)
_SWEEP_METRICS = [
    ("correlation", True),
    ("correlation_sig", True),
    ("correlation_nonsig", True),
    ("r2", True),
    ("recovery_corr", False),
    ("recovery_corr_sig", False),
    ("recovery_corr_nonsig", False),
    ("l0_sparsity", False),
    ("dead_neurons", False),
    ("mean_activation", None),  # None = no direction, informational
]

# Map final metric names → training log column names for start/end bars
_METRIC_TO_TRAINING_COL = {
    "correlation": "test_corr",
    "correlation_sig": "test_corr_sig",
    "correlation_nonsig": "test_corr_nonsig",
    "r2": "test_r2",
    "l0_sparsity": "l0_sparsity",
    "dead_neurons": "dead_frac",
    "recovery_corr": "recovery_corr",
    "recovery_corr_sig": "recovery_corr_sig",
    "recovery_corr_nonsig": "recovery_corr_nonsig",
}


def plot_sae_sweep_summary(combined_df, sweep_param, save_path=None,
                           title_prefix=""):
    """Summary figure showing how metrics vary with a swept hyperparameter.

    For each metric subplot:
    - X-axis: values of sweep_param.
    - Best point (opaque, large): best fold-averaged metric across all other
      swept param combinations.
    - Other points (translucent, small): remaining combinations.
    - Start-to-end bar: vertical line from training start to end value for the
      best combination, showing training improvement.

    Args:
        combined_df: DataFrame with columns for sweep params, metrics, and
            optionally {training_col}_start / {training_col}_end columns.
        sweep_param: Name of the swept parameter to plot on x-axis.
        save_path: If given, save figure here and close.
        title_prefix: Prefix for figure title (e.g. "k=5").

    Returns:
        matplotlib Figure when save_path is None, else None.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Determine which metrics are actually present
    available = [(m, d) for m, d in _SWEEP_METRICS if m in combined_df.columns]
    if not available:
        return None

    n_metrics = len(available)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    # Identify other swept params (columns that have multiple unique values
    # and aren't the current sweep_param or standard metric/fold columns)
    metric_cols = {m for m, _ in _SWEEP_METRICS}
    meta_cols = {"fold", "k", sweep_param}
    # Add training start/end columns to non-grouping set
    for tc in _METRIC_TO_TRAINING_COL.values():
        meta_cols.add(f"{tc}_start")
        meta_cols.add(f"{tc}_end")
    other_sweep_params = [
        c for c in combined_df.columns
        if c not in metric_cols and c not in meta_cols
        and combined_df[c].nunique() > 1
        and not c.endswith("_start") and not c.endswith("_end")
    ]

    # Group by sweep_param + other sweep params, average over folds
    group_cols = [sweep_param] + other_sweep_params
    # Include start/end columns in aggregation
    agg_cols = [m for m, _ in available]
    for m, _ in available:
        tc = _METRIC_TO_TRAINING_COL.get(m)
        if tc:
            for suffix in ("_start", "_end"):
                col = f"{tc}{suffix}"
                if col in combined_df.columns:
                    agg_cols.append(col)
    agg_cols = list(dict.fromkeys(agg_cols))  # deduplicate preserving order

    grouped = combined_df.groupby(group_cols, as_index=False)[agg_cols].mean()

    param_values = sorted(grouped[sweep_param].unique())

    for idx, (metric_name, higher_is_better) in enumerate(available):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        for pval in param_values:
            subset = grouped[grouped[sweep_param] == pval]
            values = subset[metric_name].dropna().values
            if len(values) == 0:
                continue

            x = pval

            # Determine best value
            if higher_is_better is None:
                best_idx = 0  # arbitrary for informational metrics
            elif higher_is_better:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)

            best_val = values[best_idx]

            # Plot non-best points (translucent)
            other_vals = np.delete(values, best_idx)
            if len(other_vals) > 0:
                ax.scatter([x] * len(other_vals), other_vals,
                           color="tab:blue", alpha=0.25, s=20, zorder=2)

            # Plot best point (opaque)
            ax.scatter([x], [best_val], color="tab:blue", alpha=1.0, s=60,
                       zorder=3, edgecolors="black", linewidth=0.5)

            # Start-to-end training bar for best combo
            tc = _METRIC_TO_TRAINING_COL.get(metric_name)
            if tc:
                start_col = f"{tc}_start"
                end_col = f"{tc}_end"
                if start_col in subset.columns and end_col in subset.columns:
                    best_row = subset.iloc[best_idx]
                    start_v = best_row.get(start_col)
                    end_v = best_row.get(end_col)
                    if pd.notna(start_v) and pd.notna(end_v):
                        ax.plot([x, x], [start_v, end_v],
                                color="tab:gray", linewidth=2, alpha=0.5,
                                zorder=1, solid_capstyle="round")
                        # Small marker at start
                        ax.scatter([x], [start_v], color="tab:gray",
                                   marker="_", s=40, zorder=1, alpha=0.7)

        ax.set_xlabel(sweep_param)
        ax.set_ylabel(metric_name)
        direction = ""
        if higher_is_better is True:
            direction = " (↑)"
        elif higher_is_better is False:
            direction = " (↓)"
        ax.set_title(f"{metric_name}{direction}")
        ax.grid(True, alpha=0.3)

        # Use numeric x-axis ticks at actual param values
        ax.set_xticks(param_values)
        ax.set_xticklabels([str(v) for v in param_values], fontsize=8)

    # Hide unused axes
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    title = f"Sweep summary: {sweep_param}"
    if title_prefix:
        title = f"{title_prefix} — {title}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig


# ---------------------------------------------------------------------------
# RRR Latent Space Ablation plots
# ---------------------------------------------------------------------------


def plot_ablation_pct_zeroed(
    summary_df,
    title="Ablation: perf vs % data zeroed",
    save_path=None,
):
    """Plot ablation performance vs percentage of data zeroed for a single k.

    Three panels: minimal_corr, minimal_r2, recovery_corr vs pct_zeroed.
    Each mask type (tail, center, random) gets its own color gradient.
    The unmasked RRR baseline (mask_type='none') is a horizontal dashed line.

    Args:
        summary_df: DataFrame for a single k value with columns
            [center_pct, mask_type, pct_zeroed, minimal_corr, minimal_r2, recovery_corr].
        title: Figure title.
        save_path: If given, save figure here; otherwise return the Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    metrics = ["minimal_corr", "minimal_r2", "recovery_corr"]
    labels = ["Minimal Correlation", "Minimal R\u00b2", "Recovery Correlation"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    cmap_map = {"tail": "Reds", "center": "Blues", "random": "Greys"}

    baseline = summary_df[summary_df["mask_type"] == "none"]
    ablated = summary_df[summary_df["mask_type"] != "none"]

    for ax, metric, label in zip(axes, metrics, labels):
        if len(baseline) > 0 and metric in baseline.columns:
            bl_val = baseline[metric].mean()
            ax.axhline(bl_val, color="black", linestyle="--", linewidth=1.5,
                        label="unmasked RRR", zorder=5)

        for mask_type in ["tail", "center", "random"]:
            mask_df = ablated[ablated["mask_type"] == mask_type]
            if mask_df.empty or metric not in mask_df.columns:
                continue

            cmap = matplotlib.colormaps.get_cmap(cmap_map.get(mask_type, "viridis"))
            sorted_df = mask_df.sort_values("pct_zeroed")
            pct_vals = sorted_df["pct_zeroed"].values
            metric_vals = sorted_df[metric].values
            center_pcts = sorted_df["center_pct"].values
            unique_pcts = np.unique(center_pcts)
            n_pcts = len(unique_pcts)

            color_start, color_end = 0.3, 0.85
            colors = [cmap(color_start + (color_end - color_start) * i / max(n_pcts - 1, 1))
                      for i in range(n_pcts)]

            ax.plot(pct_vals, metric_vals, color=cmap(0.6), linewidth=1.5,
                    alpha=0.7, label=mask_type)
            for j in range(len(pct_vals)):
                cpct = center_pcts[j]
                pct_idx = int(np.where(unique_pcts == cpct)[0][0])
                ax.scatter(pct_vals[j], metric_vals[j], color=colors[pct_idx],
                           s=40, zorder=4, edgecolors="white", linewidths=0.5)

        ax.set_xlabel("% Data Zeroed")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    
    
def plot_alpha_diagnostics(
    per_fold_alphas: list[np.ndarray],
    alpha_grid: np.ndarray,
    title: str = "Ridge alpha diagnostics",
    save_path: str | None = None,
):
    """3-panel figure summarising which alphas RidgeCV selected.

    Panel 1: Histogram of log10(best_alphas) across all targets/folds.
    Panel 2: Per-fold boxplot of log10(best_alphas).
    Panel 3: Text summary (median, IQR, boundary warnings, suggested refined grid).

    Args:
        per_fold_alphas: List of 1-D arrays, one per fold, each shape (n_targets,).
        alpha_grid: The alpha grid that was swept.
        title: Figure title.
        save_path: If given, save figure here; otherwise return fig.
    """
    import matplotlib.pyplot as plt
    from utils.constants import check_alpha_grid_boundaries, refine_alpha_grid

    all_alphas = np.concatenate(per_fold_alphas)
    log_all = np.log10(all_alphas[all_alphas > 0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: histogram
    ax = axes[0]
    ax.hist(log_all, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    lo_line = np.log10(alpha_grid.min())
    hi_line = np.log10(alpha_grid.max())
    ax.axvline(lo_line, color="red", linestyle="--", label=f"grid min ({alpha_grid.min():.1e})")
    ax.axvline(hi_line, color="red", linestyle="--", label=f"grid max ({alpha_grid.max():.1e})")
    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel("Count")
    ax.set_title("Selected alpha distribution")
    ax.legend(fontsize=7)

    # Panel 2: per-fold violin plot (scales to 100K+ targets without outlier clutter)
    ax = axes[1]
    log_per_fold = [np.log10(a[a > 0]) for a in per_fold_alphas]
    parts = ax.violinplot(log_per_fold, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(per_fold_alphas) + 1))
    ax.set_xticklabels([f"Fold {i}" for i in range(len(per_fold_alphas))])
    ax.set_ylabel("log10(alpha)")
    ax.set_title("Per-fold alpha distribution")

    # Panel 3: text summary
    ax = axes[2]
    ax.axis("off")
    boundary = check_alpha_grid_boundaries(all_alphas, alpha_grid)
    median_a = float(np.median(all_alphas))
    q25, q75 = float(np.percentile(all_alphas, 25)), float(np.percentile(all_alphas, 75))
    refined = refine_alpha_grid(all_alphas)

    lines = [
        f"Targets: {len(all_alphas)}",
        f"Median alpha: {median_a:.2e}",
        f"IQR: [{q25:.2e}, {q75:.2e}]",
        "",
        f"Hit lower boundary: {'YES' if boundary['hit_lower'] else 'no'} ({boundary['frac_lower']:.1%})",
        f"Hit upper boundary: {'YES' if boundary['hit_upper'] else 'no'} ({boundary['frac_upper']:.1%})",
        "",
        f"Suggested refined grid ({len(refined)} pts):",
        f"  [{refined[0]:.2e} .. {refined[-1]:.2e}]",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            verticalalignment="top", fontsize=10, fontfamily="monospace")
    ax.set_title("Summary")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig


def plot_ablation_vs_rank(
    summary_df,
    title="Ablation vs rank",
    save_path=None,
):
    """Plot ablation performance vs rank k with separate panels per condition.

    3x3 panel grid: rows = (minimal_corr, minimal_r2, recovery_corr),
    cols = (tail, center, random).  Within each panel, one line per
    center_pct value colored along a gradient.  Black dashed line =
    unmasked RRR performance vs k.

    Args:
        summary_df: DataFrame with columns [k, center_pct, mask_type, ...].
        title: Figure title.
        save_path: If given, save figure here; otherwise return the Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    metrics = ["minimal_corr", "minimal_r2", "recovery_corr"]
    metric_labels = ["Minimal Correlation", "Minimal R\u00b2", "Recovery Correlation"]
    mask_types = ["tail", "center", "random"]
    cmap_map = {"tail": "Reds", "center": "Blues", "random": "Greys"}

    n_rows = len(metrics)
    n_cols = len(mask_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    baseline = summary_df[summary_df["mask_type"] == "none"]
    ablated = summary_df[summary_df["mask_type"] != "none"]

    for row, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        for col, mask_type in enumerate(mask_types):
            ax = axes[row, col]

            if len(baseline) > 0 and metric in baseline.columns:
                bl = baseline.groupby("k")[metric].mean().sort_index()
                ax.semilogx(bl.index, bl.values, "k--", linewidth=1.5,
                            label="unmasked RRR", zorder=5)

            mask_df = ablated[ablated["mask_type"] == mask_type]
            if mask_df.empty or metric not in mask_df.columns:
                if row == 0:
                    ax.set_title(mask_type, fontsize=11)
                if col == 0:
                    ax.set_ylabel(metric_label)
                ax.set_xlabel("k (rank)")
                ax.grid(True, alpha=0.3)
                continue

            cmap = matplotlib.colormaps.get_cmap(cmap_map.get(mask_type, "viridis"))
            unique_pcts = sorted(mask_df["center_pct"].unique())
            n_pcts = len(unique_pcts)
            color_start, color_end = 0.3, 0.85

            for i, cpct in enumerate(unique_pcts):
                color = cmap(color_start + (color_end - color_start) * i / max(n_pcts - 1, 1))
                pct_df = mask_df[mask_df["center_pct"] == cpct].sort_values("k")
                ax.semilogx(pct_df["k"], pct_df[metric], "o-", color=color,
                            markersize=4, linewidth=1.2,
                            label=f"center_pct={cpct}")

            if row == 0:
                ax.set_title(mask_type, fontsize=11)
            if col == 0:
                ax.set_ylabel(metric_label)
            ax.set_xlabel("k (rank)")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            
            
def plot_alpha_vs_rank(
    rank_alpha_dict: dict[int, np.ndarray],
    alpha_grid: np.ndarray,
    title: str = "Ridge alpha distribution vs RRR rank",
    save_path: str | None = None,
):
    """2-panel figure showing how selected alphas shift with RRR rank.

    Panel 1: Boxplot of log10(best_alphas) per rank k.
    Panel 2: Median alpha vs rank (line) with IQR band.

    Args:
        rank_alpha_dict: Dict mapping rank k → 1-D array of best_alphas_.
        alpha_grid: The alpha grid that was swept.
        title: Figure title.
        save_path: If given, save figure here; otherwise return fig.
    """
    import matplotlib.pyplot as plt

    ranks = sorted(rank_alpha_dict.keys())
    log_data = [np.log10(rank_alpha_dict[k][rank_alpha_dict[k] > 0]) for k in ranks]
    medians = [float(np.median(d)) for d in log_data]
    q25s = [float(np.percentile(d, 25)) for d in log_data]
    q75s = [float(np.percentile(d, 75)) for d in log_data]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: violin plot per rank (scales to 100K+ targets without outlier clutter)
    ax = axes[0]
    parts = ax.violinplot(log_data, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(ranks) + 1))
    ax.set_xticklabels([str(k) for k in ranks])
    ax.axhline(np.log10(alpha_grid.min()), color="red", linestyle="--", alpha=0.4)
    ax.axhline(np.log10(alpha_grid.max()), color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Rank k")
    ax.set_ylabel("log10(alpha)")
    ax.set_title("Alpha distribution per rank")

    # Panel 2: median + IQR band
    ax = axes[1]
    ax.plot(ranks, medians, "o-", color="steelblue", label="Median")
    ax.fill_between(ranks, q25s, q75s, alpha=0.2, color="steelblue", label="IQR")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("log10(alpha)")
    ax.set_title("Median alpha vs rank")
    ax.legend()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig
