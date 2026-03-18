import numpy as np


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
