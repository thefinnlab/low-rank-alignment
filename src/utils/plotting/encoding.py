"""Plotting used by retained encoding and RRR scripts."""

import numpy as np
from src.utils.rrr.analysis import find_best_k


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
    summary_df,
    grid_df=None,
    title="RRR: perf vs rank",
    save_path=None,
    primary_label="rrr",
):
    """Multi-panel figure summarising RRR/PCA rank analysis.

    Layout (rows × 3 columns):
        Row 1 — Correlation vs rank:
            Panel (0,0): Mean correlations for all models (primary, complement, joint,
                         arbitrary, full, and raw RRR if present).
            Panel (0,1): Per-fold background traces for the primary model correlation.
            Panel (0,2): Fold × rank heatmap of primary_corr [when grid_df provided].

        Row 2 — R² vs rank (same structure as Row 1).

        Row 3 — Effective rank [shown only when effective_rank column exists]:
            Panel (2,0): Effective rank of A vs configured rank k.
            Panels (2,1-2): hidden.

    Args:
        summary_df: DataFrame with columns [k, {primary_label}_corr, nr_corr, ...]
            averaged over folds (i.e. summary.csv).
        grid_df: Optional per-fold DataFrame (i.e. grid_results.csv).
        title: Figure title.
        save_path: If given, save figure here and close it; otherwise return the Figure.
        primary_label: Label prefix for the primary model (e.g. "rrr", "nr", "pca").

    Returns:
        matplotlib Figure when save_path is None, else None.
    """
    import matplotlib.pyplot as plt

    has_folds = grid_df is not None
    has_effective_rank = "effective_rank" in summary_df.columns
    pred_corr_cols = [
        (
            f"{primary_label}_complement_pred_corr",
            f"{primary_label} vs complement",
            "#7c3aed",
        ),
        (
            f"{primary_label}_arbitrary_pred_corr",
            f"{primary_label} vs arbitrary",
            "#d97706",
        ),
        ("complement_arbitrary_pred_corr", "complement vs arbitrary", "#0f766e"),
    ]
    has_pred_corr = any(col in summary_df.columns for col, _, _ in pred_corr_cols)

    n_rows = 2 + int(has_pred_corr) + int(has_effective_rank)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))

    ks = summary_df["k"].values

    # Colour scheme
    _colors = {
        "full": "#9ca3af",  # light gray — ceiling
        "rrr": "#1e40af",  # dark blue — raw RRR
        "nr": "#3b82f6",  # blue — NR ridge (primary in RRR run)
        "pca": "#3b82f6",  # blue — PCA (primary in PCA run)
        "orth": "#ea580c",  # orange — ridge on QR-orthonormalized A
        "complement": "#8b5cf6",  # purple
        "arbitrary": "#78716c",  # gray
        "comp_joint": "#10b981",  # green — NR+complement joint
        "arb_joint": "#f59e0b",  # amber — NR+arbitrary joint
        "comp_joint_ridge": "#047857",  # dark green — ridge-stacked joint
        "arb_joint_ridge": "#b45309",  # dark amber — ridge-stacked joint
    }

    primary_color = _colors.get(primary_label, "#3b82f6")
    orth_color = _colors["orth"]
    comp_joint_col = f"{primary_label}_complement_joint_corr"
    comp_joint_ridge_col = f"{primary_label}_complement_joint_ridge_corr"
    arb_joint_col = f"{primary_label}_arbitrary_joint_corr"
    arb_joint_ridge_col = f"{primary_label}_arbitrary_joint_ridge_corr"

    # ---- Local helpers ----

    def _fold_traces(ax, col, color):
        if has_folds and col in grid_df.columns:
            for _, fd in grid_df.groupby("fold"):
                ax.semilogx(fd["k"], fd[col], color=color, alpha=0.25, linewidth=0.8)

    def _mean_line(ax, col, df, color, label, ls="-", mk="o", ms=4):
        if col in df.columns:
            ax.semilogx(
                df["k"],
                df[col],
                color=color,
                linestyle=ls,
                marker=mk,
                markersize=ms,
                label=label,
                linewidth=1.5,
            )

    def _panel_finish(ax, xlabel, ylabel, ptitle, hline=None):
        if hline is not None:
            ax.axhline(hline, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ptitle)
        ax.legend(fontsize=8)

    def _heatmap(ax, col, cmap, clabel):
        if not has_folds or col not in grid_df.columns:
            ax.set_visible(False)
            return
        folds = sorted(grid_df["fold"].unique())
        ks_ord = sorted(grid_df["k"].unique())
        k_idx = {k: i for i, k in enumerate(ks_ord)}
        f_idx = {f: i for i, f in enumerate(folds)}
        hm = np.full((len(folds), len(ks_ord)), np.nan)
        for _, row in grid_df.iterrows():
            if col in row:
                hm[f_idx[row["fold"]], k_idx[row["k"]]] = row[col]
        im = ax.imshow(hm, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(ks_ord)))
        ax.set_xticklabels([str(k) for k in ks_ord], rotation=45, ha="right")
        ax.set_yticks(range(len(folds)))
        ax.set_yticklabels([f"fold {f}" for f in folds])
        ax.set_xlabel("Rank k")
        ax.set_title(f"{clabel} per fold")
        plt.colorbar(im, ax=ax, label=clabel)

    # ===== Row 0: Correlation vs rank =====

    ax_corr = axes[0, 0]
    primary_corr_col = f"{primary_label}_corr"
    orth_corr_col = f"{primary_label}_orth_corr"
    has_orth = orth_corr_col in summary_df.columns
    _fold_traces(ax_corr, primary_corr_col, primary_color)
    _fold_traces(ax_corr, orth_corr_col, orth_color)
    _mean_line(
        ax_corr, "full_corr", summary_df, _colors["full"], "full (ceiling)", ls="--"
    )
    _mean_line(ax_corr, "rrr_corr", summary_df, _colors["rrr"], "rrr (raw)", mk="^")
    _mean_line(
        ax_corr, primary_corr_col, summary_df, primary_color,
        f"{primary_label} (raw A)" if has_orth else primary_label,
    )
    _mean_line(
        ax_corr, orth_corr_col, summary_df, orth_color,
        f"{primary_label} (QR A)", ls="--", mk="s",
    )
    _mean_line(
        ax_corr,
        "complement_corr",
        summary_df,
        _colors["complement"],
        "complement",
        ls="--",
    )
    _mean_line(
        ax_corr,
        comp_joint_col,
        summary_df,
        _colors["comp_joint"],
        f"{primary_label}+complement joint",
        ls="-.",
        mk="s",
    )
    _mean_line(
        ax_corr,
        comp_joint_ridge_col,
        summary_df,
        _colors["comp_joint_ridge"],
        f"{primary_label}+complement ridge joint",
        ls=":",
        mk="s",
        ms=3,
    )
    _mean_line(
        ax_corr, "arbitrary_corr", summary_df, _colors["arbitrary"], "arbitrary", ls=":"
    )
    _mean_line(
        ax_corr,
        arb_joint_col,
        summary_df,
        _colors["arb_joint"],
        f"{primary_label}+arbitrary joint",
        ls="-.",
        mk="D",
        ms=3,
    )
    _mean_line(
        ax_corr,
        arb_joint_ridge_col,
        summary_df,
        _colors["arb_joint_ridge"],
        f"{primary_label}+arbitrary ridge joint",
        ls=":",
        mk="D",
        ms=3,
    )
    _panel_finish(
        ax_corr, "Rank k", "Correlation", f"{title}\n(correlation vs rank)", hline=0
    )

    ax_corr_folds = axes[0, 1]
    _fold_traces(ax_corr_folds, primary_corr_col, primary_color)
    _fold_traces(ax_corr_folds, orth_corr_col, orth_color)
    _mean_line(
        ax_corr_folds, primary_corr_col, summary_df, primary_color,
        f"{primary_label} (raw A)" if has_orth else primary_label,
    )
    _mean_line(
        ax_corr_folds, orth_corr_col, summary_df, orth_color,
        f"{primary_label} (QR A)", ls="--", mk="s",
    )
    _panel_finish(
        ax_corr_folds,
        "Rank k",
        "Correlation",
        f"{title}\n(per-fold: {primary_label}_corr)",
        hline=0,
    )

    _heatmap(axes[0, 2], primary_corr_col, "Blues", f"{primary_label}_corr")

    # ===== Row 1: R² vs rank =====

    primary_r2_col = f"{primary_label}_r2"
    orth_r2_col = f"{primary_label}_orth_r2"
    comp_joint_r2_col = f"{primary_label}_complement_joint_r2"
    comp_joint_ridge_r2_col = f"{primary_label}_complement_joint_ridge_r2"
    arb_joint_r2_col = f"{primary_label}_arbitrary_joint_r2"
    arb_joint_ridge_r2_col = f"{primary_label}_arbitrary_joint_ridge_r2"

    ax_r2 = axes[1, 0]
    _fold_traces(ax_r2, primary_r2_col, primary_color)
    _fold_traces(ax_r2, orth_r2_col, orth_color)
    _mean_line(ax_r2, "full_r2", summary_df, _colors["full"], "full (ceiling)", ls="--")
    _mean_line(ax_r2, "rrr_r2", summary_df, _colors["rrr"], "rrr (raw)", mk="^")
    _mean_line(
        ax_r2, primary_r2_col, summary_df, primary_color,
        f"{primary_label} (raw A)" if has_orth else primary_label,
    )
    _mean_line(
        ax_r2, orth_r2_col, summary_df, orth_color,
        f"{primary_label} (QR A)", ls="--", mk="s",
    )
    _mean_line(
        ax_r2, "complement_r2", summary_df, _colors["complement"], "complement", ls="--"
    )
    _mean_line(
        ax_r2,
        comp_joint_r2_col,
        summary_df,
        _colors["comp_joint"],
        f"{primary_label}+complement joint",
        ls="-.",
        mk="s",
    )
    _mean_line(
        ax_r2,
        comp_joint_ridge_r2_col,
        summary_df,
        _colors["comp_joint_ridge"],
        f"{primary_label}+complement ridge joint",
        ls=":",
        mk="s",
        ms=3,
    )
    _mean_line(
        ax_r2, "arbitrary_r2", summary_df, _colors["arbitrary"], "arbitrary", ls=":"
    )
    _mean_line(
        ax_r2,
        arb_joint_r2_col,
        summary_df,
        _colors["arb_joint"],
        f"{primary_label}+arbitrary joint",
        ls="-.",
        mk="D",
        ms=3,
    )
    _mean_line(
        ax_r2,
        arb_joint_ridge_r2_col,
        summary_df,
        _colors["arb_joint_ridge"],
        f"{primary_label}+arbitrary ridge joint",
        ls=":",
        mk="D",
        ms=3,
    )
    _panel_finish(ax_r2, "Rank k", "R²", f"{title}\n(R² vs rank)", hline=0)

    ax_r2_folds = axes[1, 1]
    _fold_traces(ax_r2_folds, primary_r2_col, primary_color)
    _fold_traces(ax_r2_folds, orth_r2_col, orth_color)
    _mean_line(
        ax_r2_folds, primary_r2_col, summary_df, primary_color,
        f"{primary_label} (raw A)" if has_orth else primary_label,
    )
    _mean_line(
        ax_r2_folds, orth_r2_col, summary_df, orth_color,
        f"{primary_label} (QR A)", ls="--", mk="s",
    )
    _panel_finish(
        ax_r2_folds, "Rank k", "R²", f"{title}\n(per-fold: {primary_label}_r2)", hline=0
    )

    _heatmap(axes[1, 2], primary_r2_col, "Blues", f"{primary_label}_r2")

    next_row = 2

    # ===== Row 2: Prediction correlations (optional) =====

    if has_pred_corr:
        ax_pred = axes[next_row, 0]
        for col, label, color in pred_corr_cols:
            if col not in summary_df.columns:
                continue
            lower_col = f"{col}_sem_lower"
            upper_col = f"{col}_sem_upper"
            sem_col = f"{col}_sem"
            yerr = None
            if lower_col in summary_df.columns and upper_col in summary_df.columns:
                yerr = np.vstack(
                    [
                        summary_df[lower_col].to_numpy(),
                        summary_df[upper_col].to_numpy(),
                    ]
                )
            elif sem_col in summary_df.columns:
                yerr = summary_df[sem_col].to_numpy()
            ax_pred.errorbar(
                ks,
                summary_df[col].to_numpy(),
                yerr=yerr,
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=3,
                label=label,
            )
        ax_pred.set_xscale("log")
        _panel_finish(
            ax_pred,
            "Rank k",
            "Prediction correlation",
            f"{title}\n(pairwise prediction correlations)",
            hline=0,
        )
        for ci in range(1, n_cols):
            axes[next_row, ci].set_visible(False)
        next_row += 1

    # ===== Effective rank (optional) =====

    if has_effective_rank:
        ax_rank = axes[next_row, 0]
        if has_folds and "effective_rank" in grid_df.columns:
            for _, fd in grid_df.groupby("fold"):
                ax_rank.plot(
                    fd["k"],
                    fd["effective_rank"],
                    color="#6366f1",
                    alpha=0.25,
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
        k_range = np.array([ks.min(), ks.max()])
        ax_rank.plot(
            k_range, k_range, color="black", linewidth=0.8, linestyle=":", label="y = k"
        )
        ax_rank.set_xlabel("Configured rank k")
        ax_rank.set_ylabel("Effective rank")
        ax_rank.set_title(f"{title}\n(effective rank of A)")
        ax_rank.legend(fontsize=8)
        for ci in range(1, n_cols):
            axes[next_row, ci].set_visible(False)

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
