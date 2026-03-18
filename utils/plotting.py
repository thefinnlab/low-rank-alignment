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
