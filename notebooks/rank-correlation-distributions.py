import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pandas as pd

    return Path, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Rank Correlation Distribution Explorer

    Compare low-rank RRR/PCA per-target correlations against a max-rank
    reference. The two comparison values are:

    - `small_rank_corr / max_rank_corr`
    - `small_rank_corr - max_rank_corr`
    - `small_rank_corr ** 2 / max_rank_corr ** 2`

    Ratio plots exclude targets whose max-rank reference correlation is too
    close to zero, because those ratios are dominated by numerical instability
    rather than rank-comparison structure.

    Set `FILTER_TO_SIGNIFICANT_TARGETS = True` to restrict targets to
    electrode-lag pairs marked significant in `significance_detailed.csv`.
    """)
    return


@app.cell
def _(Path):
    # Set these values when switching between result folders or methods.
    RESULT_SOURCES = {
        "RRR": Path("rrr"),
        "PCA": Path("pca"),
    }
    MAX_RANK = 1600
    SMALL_RANKS = None  # None means all ranks below MAX_RANK shared by every source.
    PLOT_RANKS = [1, 4, 16, 64, 256, 1024]  # Keeps faceted plots readable.
    EPSILON = 0.05  # Minimum abs(max-rank corr) required for ratio plots.
    FILTER_TO_SIGNIFICANT_TARGETS = False
    SIGNIFICANCE_DETAILED_CSV = Path("significance_detailed.csv")
    SIGNIFICANT_ELECTRODES_CSV = None  # None means sibling all_subject_sig.csv.
    return (
        EPSILON,
        FILTER_TO_SIGNIFICANT_TARGETS,
        MAX_RANK,
        PLOT_RANKS,
        RESULT_SOURCES,
        SIGNIFICANCE_DETAILED_CSV,
        SIGNIFICANT_ELECTRODES_CSV,
        SMALL_RANKS,
    )


@app.cell
def _(Path, np, pd):
    def discover_rank_dirs(results_dir: Path) -> dict[int, Path]:
        """Return {rank: k<rank> directory} for a results folder."""
        results_dir = Path(results_dir)
        rank_dirs = {}
        for path in sorted(results_dir.glob("k*")):
            if not path.is_dir():
                continue
            try:
                rank = int(path.name.removeprefix("k"))
            except ValueError:
                continue
            rank_dirs[rank] = path
        return rank_dirs


    def discover_folds(rank_dir: Path) -> list[int]:
        """Return folds with metrics files under one rank directory."""
        folds = []
        for path in sorted(Path(rank_dir).glob("fold_*/metrics.npz")):
            try:
                folds.append(int(path.parent.name.removeprefix("fold_")))
            except ValueError:
                continue
        return folds


    def detect_corr_key(metrics_path: Path) -> str:
        """Detect the per-target correlation array in an RRR or PCA metrics file."""
        with np.load(metrics_path, allow_pickle=True) as metrics:
            for key in ("nr_corr_per_target", "pca_corr_per_target", "corr_per_target"):
                if key in metrics.files:
                    return key
            candidates = [
                key
                for key in metrics.files
                if key.endswith("_corr_per_target") or key == "corr_per_target"
            ]
        if not candidates:
            raise KeyError(f"No per-target correlation key found in {metrics_path}")
        return candidates[0]


    def load_fold_correlations(metrics_path: Path, corr_key: str | None = None) -> np.ndarray:
        """Load one fold's per-target correlations."""
        metrics_path = Path(metrics_path)
        corr_key = corr_key or detect_corr_key(metrics_path)
        with np.load(metrics_path, allow_pickle=True) as metrics:
            return np.asarray(metrics[corr_key], dtype=float)


    def load_rank_correlations(
        results_dir: Path,
        rank: int,
        corr_key: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Load per-target correlations for all folds at one rank."""
        rank_dir = Path(results_dir) / f"k{rank}"
        rows = []
        for fold in discover_folds(rank_dir):
            metrics_path = rank_dir / f"fold_{fold}" / "metrics.npz"
            corrs = load_fold_correlations(metrics_path, corr_key)
            rows.append(
                pd.DataFrame(
                    {
                        "source": source or Path(results_dir).name,
                        "rank": rank,
                        "fold": fold,
                        "target": np.arange(corrs.shape[0]),
                        "corr": corrs,
                    }
                )
            )
        if not rows:
            raise FileNotFoundError(f"No fold metrics found for rank {rank} in {rank_dir}")
        return pd.concat(rows, ignore_index=True)


    def load_all_rank_correlations(
        results_dir: Path,
        ranks: list[int],
        corr_key: str | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        """Load per-target correlations for multiple ranks."""
        frames = [
            load_rank_correlations(results_dir, rank, corr_key, source=source)
            for rank in ranks
        ]
        return pd.concat(frames, ignore_index=True)


    def load_result_sources(
        result_sources: dict[str, Path],
        ranks_by_source: dict[str, list[int]],
        corr_keys_by_source: dict[str, str],
    ) -> pd.DataFrame:
        """Load per-target correlations for named result folders."""
        frames = []
        for source, results_dir in result_sources.items():
            frames.append(
                load_all_rank_correlations(
                    results_dir,
                    ranks_by_source[source],
                    corr_keys_by_source[source],
                    source=source,
                )
            )
        return pd.concat(frames, ignore_index=True)


    def _coerce_sig_bool(series: pd.Series) -> pd.Series:
        """Handle bool or string significance columns from CSV files."""
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )


    def load_significant_target_mask(
        significance_detailed_csv: Path,
        n_targets: int,
        significant_electrodes_csv: Path | None = None,
    ) -> pd.DataFrame:
        """Build a target-index mask from significance_detailed.csv.

        Target order is assumed to be electrode-major then lag-major:
        target = electrode_index * n_lags + lag_idx. When a sibling
        all_subject_sig.csv exists, its row order is used as the electrode order,
        matching how model targets are loaded.
        """
        significance_detailed_csv = Path(significance_detailed_csv)
        sig_df = pd.read_csv(significance_detailed_csv)
        required = {"subject", "electrode", "lag_idx", "sig"}
        missing = required - set(sig_df.columns)
        if missing:
            raise ValueError(
                f"{significance_detailed_csv} is missing columns: {sorted(missing)}"
            )

        sig_df = sig_df.copy()
        sig_df["subject"] = pd.to_numeric(sig_df["subject"]).astype(int)
        sig_df["electrode"] = sig_df["electrode"].astype(str)
        sig_df["lag_idx"] = pd.to_numeric(sig_df["lag_idx"]).astype(int)
        sig_df["sig"] = _coerce_sig_bool(sig_df["sig"])
        n_lags = int(sig_df["lag_idx"].max()) + 1
        if n_targets % n_lags != 0:
            raise ValueError(
                f"n_targets={n_targets} is not divisible by n_lags={n_lags} "
                f"from {significance_detailed_csv}."
            )

        if significant_electrodes_csv is None:
            candidate = significance_detailed_csv.parent / "all_subject_sig.csv"
            significant_electrodes_csv = candidate if candidate.exists() else None

        if significant_electrodes_csv is not None:
            elec_df = pd.read_csv(significant_electrodes_csv)
            elec_col = "elec" if "elec" in elec_df.columns else "electrode"
            if not {"subject", elec_col} <= set(elec_df.columns):
                raise ValueError(
                    f"{significant_electrodes_csv} must contain subject and "
                    "elec/electrode columns."
                )
            metadata = (
                elec_df[["subject", elec_col]]
                .rename(columns={elec_col: "electrode"})
                .copy()
            )
            metadata["subject"] = pd.to_numeric(metadata["subject"]).astype(int)
            metadata["electrode"] = metadata["electrode"].astype(str)
        else:
            metadata = (
                sig_df.loc[sig_df["sig"], ["subject", "electrode"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

        n_electrodes = n_targets // n_lags
        if len(metadata) != n_electrodes:
            raise ValueError(
                f"Significance metadata has {len(metadata)} electrodes, but metrics "
                f"imply {n_electrodes} electrodes ({n_targets} targets / {n_lags} lags). "
                "Pass SIGNIFICANT_ELECTRODES_CSV if the electrode order differs."
            )

        sig_lookup = {
            (int(row.subject), str(row.electrode), int(row.lag_idx)): bool(row.sig)
            for row in sig_df.itertuples(index=False)
        }
        rows = []
        for elec_idx, row in enumerate(metadata.itertuples(index=False)):
            subject = int(row.subject)
            electrode = str(row.electrode)
            for lag_idx in range(n_lags):
                target = elec_idx * n_lags + lag_idx
                rows.append(
                    {
                        "target": target,
                        "subject": subject,
                        "electrode": electrode,
                        "lag_idx": lag_idx,
                        "target_sig": sig_lookup.get(
                            (subject, electrode, lag_idx), False
                        ),
                    }
                )
        return pd.DataFrame(rows)


    def apply_target_filter(
        all_corrs: pd.DataFrame,
        target_mask_df: pd.DataFrame | None,
        require_significant: bool,
    ) -> pd.DataFrame:
        """Optionally restrict loaded correlations to significant targets."""
        if not require_significant:
            return all_corrs
        if target_mask_df is None:
            raise ValueError("target_mask_df is required when filtering targets.")
        keep_targets = target_mask_df.loc[target_mask_df["target_sig"], "target"]
        return all_corrs[all_corrs["target"].isin(keep_targets)].copy()

    return (
        apply_target_filter,
        detect_corr_key,
        discover_rank_dirs,
        load_result_sources,
        load_significant_target_mask,
    )


@app.cell
def _(EPSILON, np, pd):
    def compare_rank_to_reference(
        rank_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        epsilon: float = EPSILON,
    ) -> pd.DataFrame:
        """Join rank/reference correlations and compute ratio and delta."""
        reference = reference_df.rename(
            columns={"rank": "reference_rank", "corr": "reference_corr"}
        )
        merged = rank_df.merge(
            reference[
                ["source", "fold", "target", "reference_rank", "reference_corr"]
            ],
            on=["source", "fold", "target"],
            how="inner",
            validate="one_to_one",
        )
        denom = merged["reference_corr"].to_numpy()
        numer = merged["corr"].to_numpy()
        valid_ratio = np.abs(denom) > epsilon
        merged["corr_ratio"] = np.where(valid_ratio, numer / denom, np.nan)
        merged["corr_squared_ratio"] = np.where(
            valid_ratio,
            np.square(numer) / np.square(denom),
            np.nan,
        )
        merged["corr_delta"] = merged["corr"] - merged["reference_corr"]
        merged["abs_reference_corr"] = np.abs(merged["reference_corr"])
        merged["valid_ratio"] = valid_ratio
        return merged


    def build_rank_comparisons(
        all_corrs: pd.DataFrame,
        max_rank: int,
        epsilon: float = EPSILON,
    ) -> pd.DataFrame:
        """Compute low-rank/reference comparisons for every non-reference rank."""
        reference_df = all_corrs[all_corrs["rank"] == max_rank]
        if reference_df.empty:
            raise ValueError(f"Reference rank {max_rank} is not present.")

        frames = []
        for source in sorted(all_corrs["source"].unique()):
            source_corrs = all_corrs[all_corrs["source"] == source]
            source_reference = reference_df[reference_df["source"] == source]
            if source_reference.empty:
                raise ValueError(f"Reference rank {max_rank} is missing for {source}.")
            for rank in sorted(r for r in source_corrs["rank"].unique() if r != max_rank):
                rank_df = source_corrs[source_corrs["rank"] == rank]
                frames.append(compare_rank_to_reference(rank_df, source_reference, epsilon))
        if not frames:
            raise ValueError("No non-reference ranks are available to compare.")
        return pd.concat(frames, ignore_index=True)


    def summarize_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
        """Summarize ratio and delta distributions by rank."""
        return (
            comparisons.groupby(["source", "rank"], as_index=False)
            .agg(
                n=("corr_delta", "size"),
                n_valid_ratio=("valid_ratio", "sum"),
                mean_ratio=("corr_ratio", "mean"),
                median_ratio=("corr_ratio", "median"),
                q05_ratio=("corr_ratio", lambda x: x.quantile(0.05)),
                q95_ratio=("corr_ratio", lambda x: x.quantile(0.95)),
                mean_squared_ratio=("corr_squared_ratio", "mean"),
                median_squared_ratio=("corr_squared_ratio", "median"),
                q05_squared_ratio=("corr_squared_ratio", lambda x: x.quantile(0.05)),
                q95_squared_ratio=("corr_squared_ratio", lambda x: x.quantile(0.95)),
                mean_delta=("corr_delta", "mean"),
                median_delta=("corr_delta", "median"),
                q05_delta=("corr_delta", lambda x: x.quantile(0.05)),
                q95_delta=("corr_delta", lambda x: x.quantile(0.95)),
                mean_rank_corr=("corr", "mean"),
                mean_reference_corr=("reference_corr", "mean"),
            )
            .sort_values(["source", "rank"])
        )

    return build_rank_comparisons, summarize_comparisons


@app.cell
def _(np, plt):
    def _hist_percent_weights(values):
        """Return histogram weights so bar heights sum to 100%."""
        if len(values) == 0:
            return None
        return np.ones(len(values)) * 100.0 / len(values)


    def _values_for_plot(values, xlim=None):
        """Return finite values, restricted to the displayed range if supplied."""
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if xlim is not None:
            values = values[(values >= xlim[0]) & (values <= xlim[1])]
        return values


    def plot_distribution_grid(
        comparisons,
        value_col: str,
        xlabel: str,
        title: str,
        bins: int = 80,
        xlim: tuple[float, float] | None = None,
    ):
        """Plot per-rank histograms for one comparison value."""
        ranks = sorted(comparisons["rank"].unique())
        sources = sorted(comparisons["source"].unique())
        ncols = min(3, len(ranks))
        nrows = int(np.ceil(len(ranks) / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.2 * ncols, 3.4 * nrows),
            squeeze=False,
            sharex=xlim is not None,
        )
        for ax, rank in zip(axes.flat, ranks):
            hist_range = xlim if xlim is not None else None
            visible_count = 0
            for source in sources:
                values = _values_for_plot(
                    comparisons.loc[
                        (comparisons["rank"] == rank)
                        & (comparisons["source"] == source),
                        value_col,
                    ],
                    xlim=xlim,
                )
                visible_count += len(values)
                ax.hist(
                    values,
                    bins=bins,
                    range=hist_range,
                    weights=_hist_percent_weights(values),
                    histtype="step",
                    linewidth=1.5,
                    label=source,
                )
            ax.axvline(0, color="#333333", linewidth=1, alpha=0.75)
            if value_col == "corr_ratio":
                ax.axvline(1, color="#b54708", linewidth=1.2, alpha=0.9)
            if value_col == "corr_squared_ratio":
                ax.axvline(1, color="#b54708", linewidth=1.2, alpha=0.9)
            ax.set_title(f"k={rank} (visible n={visible_count:,})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Percent of visible targets")
            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.legend(frameon=False)
        for ax in axes.flat[len(ranks) :]:
            ax.axis("off")
        fig.suptitle(title, y=1.01, fontsize=14)
        fig.tight_layout()
        return fig


    def plot_overlay_histograms(
        comparisons,
        value_col: str,
        xlabel: str,
        title: str,
        bins: int = 100,
        xlim: tuple[float, float] | None = None,
    ):
        """Overlay rank distributions in one histogram."""
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for source in sorted(comparisons["source"].unique()):
            source_df = comparisons[comparisons["source"] == source]
            for rank in sorted(source_df["rank"].unique()):
                values = _values_for_plot(
                    source_df.loc[source_df["rank"] == rank, value_col],
                    xlim=xlim,
                )
                ax.hist(
                    values,
                    bins=bins,
                    range=xlim if xlim is not None else None,
                    weights=_hist_percent_weights(values),
                    histtype="step",
                    linewidth=1.7,
                    label=f"{source} k={rank}",
                )

        ax.axvline(0, color="#333333", linewidth=1, alpha=0.75)
        if value_col == "corr_ratio":
            ax.axvline(1, color="#b54708", linewidth=1.2, alpha=0.9)
        if value_col == "corr_squared_ratio":
            ax.axvline(1, color="#b54708", linewidth=1.2, alpha=0.9)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Percent of visible targets")
        ax.set_title(title)
        ax.legend(ncols=2, frameon=False)
        fig.tight_layout()
        return fig


    def plot_rank_pair_histograms(
        comparisons,
        value_col: str,
        xlabel: str,
        title: str,
        ranks: list[int] | None = None,
        bins: int = 80,
        xlim: tuple[float, float] | None = None,
    ):
        """Plot one source-comparison panel per rank."""
        if ranks is None:
            ranks = sorted(comparisons["rank"].unique())
        sources = sorted(comparisons["source"].unique())
        ncols = min(3, len(ranks))
        nrows = int(np.ceil(len(ranks) / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.2 * ncols, 3.4 * nrows),
            squeeze=False,
            sharex=xlim is not None,
            sharey=True,
        )
        for ax, rank in zip(axes.flat, ranks):
            for source in sources:
                values = _values_for_plot(
                    comparisons.loc[
                        (comparisons["rank"] == rank)
                        & (comparisons["source"] == source),
                        value_col,
                    ],
                    xlim=xlim,
                )
                ax.hist(
                    values,
                    bins=bins,
                    range=xlim if xlim is not None else None,
                    weights=_hist_percent_weights(values),
                    histtype="step",
                    linewidth=1.7,
                    label=source,
                )
            ax.axvline(0, color="#333333", linewidth=1, alpha=0.75)
            if value_col == "corr_ratio":
                ax.axvline(1, color="#b54708", linewidth=1.2, alpha=0.9)
            if value_col == "corr_squared_ratio":
                ax.axvline(1, color="#b54708", linewidth=1.2, alpha=0.9)
            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.set_title(f"k={rank}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Percent of visible targets")
            ax.legend(frameon=False)
        for ax in axes.flat[len(ranks) :]:
            ax.axis("off")
        fig.suptitle(title, y=1.01, fontsize=14)
        fig.tight_layout()
        return fig


    def plot_summary_lines(
        summary,
        value_col: str,
        ylabel: str,
        title: str,
    ):
        """Plot a summary statistic across ranks for each source."""
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for source in sorted(summary["source"].unique()):
            source_summary = summary[summary["source"] == source].sort_values("rank")
            ax.plot(
                source_summary["rank"],
                source_summary[value_col],
                marker="o",
                linewidth=1.8,
                label=source,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Rank")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    return (
        plot_overlay_histograms,
        plot_rank_pair_histograms,
        plot_summary_lines,
    )


@app.cell
def _(
    MAX_RANK,
    RESULT_SOURCES,
    SMALL_RANKS,
    detect_corr_key,
    discover_rank_dirs,
    mo,
):
    rank_dirs_by_source = {
        source: discover_rank_dirs(results_dir)
        for source, results_dir in RESULT_SOURCES.items()
    }
    missing_reference = [
        source
        for source, rank_dirs in rank_dirs_by_source.items()
        if MAX_RANK not in rank_dirs
    ]
    mo.stop(
        len(missing_reference) > 0,
        mo.callout(
            mo.md(
                f"Reference rank `k{MAX_RANK}` was not found for: "
                f"`{missing_reference}`."
            ),
            kind="warn",
        ),
    )

    if SMALL_RANKS is None:
        shared_ranks = set.intersection(
            *[
                {rank for rank in rank_dirs if rank <= MAX_RANK}
                for rank_dirs in rank_dirs_by_source.values()
            ]
        )
        selected_ranks = sorted(shared_ranks)
    else:
        selected_ranks = sorted(set(SMALL_RANKS) | {MAX_RANK})

    missing_ranks = {
        source: [
            rank for rank in selected_ranks if rank not in rank_dirs_by_source[source]
        ]
        for source in RESULT_SOURCES
    }
    missing_ranks = {
        source: ranks for source, ranks in missing_ranks.items() if len(ranks) > 0
    }
    mo.stop(
        len(missing_ranks) > 0,
        mo.callout(
            mo.md(f"Some selected ranks are missing: `{missing_ranks}`."),
            kind="warn",
        ),
    )

    corr_keys_by_source = {}
    for source, rank_dirs in rank_dirs_by_source.items():
        first_metrics = rank_dirs[MAX_RANK] / "fold_0" / "metrics.npz"
        if not first_metrics.exists():
            first_metrics = next(rank_dirs[MAX_RANK].glob("fold_*/metrics.npz"))
        corr_keys_by_source[source] = detect_corr_key(first_metrics)

    ranks_by_source = {source: selected_ranks for source in RESULT_SOURCES}

    mo.md(
        f"""
        **Loading:** `{RESULT_SOURCES}`  
        **Ranks:** `{selected_ranks}`  
        **Reference:** `k{MAX_RANK}`  
        **Correlation keys:** `{corr_keys_by_source}`
        """
    )
    return corr_keys_by_source, ranks_by_source


@app.cell
def _(
    FILTER_TO_SIGNIFICANT_TARGETS,
    RESULT_SOURCES,
    SIGNIFICANCE_DETAILED_CSV,
    SIGNIFICANT_ELECTRODES_CSV,
    apply_target_filter,
    corr_keys_by_source,
    load_result_sources,
    load_significant_target_mask,
    mo,
    ranks_by_source,
):
    all_corrs = load_result_sources(
        RESULT_SOURCES,
        ranks_by_source,
        corr_keys_by_source,
    )
    target_mask_df = None
    if FILTER_TO_SIGNIFICANT_TARGETS:
        mo.stop(
            not SIGNIFICANCE_DETAILED_CSV.exists(),
            mo.callout(
                mo.md(
                    f"`FILTER_TO_SIGNIFICANT_TARGETS` is true, but "
                    f"`{SIGNIFICANCE_DETAILED_CSV}` does not exist."
                ),
                kind="warn",
            ),
        )
        n_targets = int(all_corrs["target"].max()) + 1
        target_mask_df = load_significant_target_mask(
            SIGNIFICANCE_DETAILED_CSV,
            n_targets,
            SIGNIFICANT_ELECTRODES_CSV,
        )
        all_corrs = apply_target_filter(
            all_corrs,
            target_mask_df,
            require_significant=FILTER_TO_SIGNIFICANT_TARGETS,
        )
    all_corrs.head()
    return all_corrs, target_mask_df


@app.cell
def _(all_corrs):
    all_corrs
    return


@app.cell
def _(FILTER_TO_SIGNIFICANT_TARGETS, all_corrs, mo, target_mask_df):
    if FILTER_TO_SIGNIFICANT_TARGETS and target_mask_df is not None:
        _n_sig = int(target_mask_df["target_sig"].sum())
        _n_total = len(target_mask_df)
        mo.md(
            f"""
            **Target filter:** significant electrode-lag pairs only  
            **Significant targets:** `{_n_sig:,} / {_n_total:,}`  
            **Rows after filter:** `{len(all_corrs):,}`
            """
        )
    else:
        mo.md(
            f"""
            **Target filter:** all model targets  
            **Rows loaded:** `{len(all_corrs):,}`
            """
        )
    return


@app.cell
def _(MAX_RANK, all_corrs, plt):
    plt.hist(all_corrs[(all_corrs["rank"] == MAX_RANK) & (all_corrs["source"] == "RRR")]["corr"], bins=100)
    plt.title(f"Max-rank correlation distribution for RRR")
    plt.ylabel("Frequency")
    plt.xlabel("Correlation by Target")
    return


@app.cell
def _(MAX_RANK, all_corrs, build_rank_comparisons, summarize_comparisons):
    comparisons = build_rank_comparisons(all_corrs, MAX_RANK)
    comparison_summary = summarize_comparisons(comparisons)
    comparison_summary
    return comparison_summary, comparisons


@app.cell
def _(comparison_summary, plot_summary_lines):
    plot_summary_lines(
        comparison_summary,
        value_col="median_ratio",
        ylabel="Median small-rank / max-rank correlation",
        title="Median relative correlation ratio by source",
    )
    return


@app.cell
def _(comparison_summary, plot_summary_lines):
    plot_summary_lines(
        comparison_summary,
        value_col="mean_ratio",
        ylabel="Mean small-rank / max-rank correlation",
        title="Mean relative correlation ratio by source",
    )
    return


@app.cell
def _(comparison_summary, plot_summary_lines):
    plot_summary_lines(
        comparison_summary,
        value_col="median_delta",
        ylabel="Median small-rank - max-rank correlation",
        title="Median correlation delta by source",
    )
    return


@app.cell
def _(comparison_summary, plot_summary_lines):
    plot_summary_lines(
        comparison_summary,
        value_col="mean_delta",
        ylabel="Mean small-rank - max-rank correlation",
        title="Mean correlation delta by source",
    )
    return


@app.cell
def _(comparison_summary, plot_summary_lines):
    plot_summary_lines(
        comparison_summary,
        value_col="median_squared_ratio",
        ylabel="Median squared correlation ratio",
        title="Median relative squared correlation by source",
    )
    return


@app.cell
def _(PLOT_RANKS, comparisons, plot_rank_pair_histograms):
    plot_rank_pair_histograms(
        comparisons,
        value_col="corr_ratio",
        xlabel="small-rank correlation / max-rank correlation",
        title="RRR vs PCA relative correlation ratio by rank",
        ranks=PLOT_RANKS,
        xlim=(-5, 5),
    )
    return


@app.cell
def _(PLOT_RANKS, comparisons, plot_rank_pair_histograms):
    plot_rank_pair_histograms(
        comparisons,
        value_col="corr_squared_ratio",
        xlabel="small-rank correlation^2 / max-rank correlation^2",
        title="RRR vs PCA relative squared correlation by rank",
        ranks=PLOT_RANKS,
        xlim=(0, 2),
    )
    return


@app.cell
def _(PLOT_RANKS, comparisons, plot_rank_pair_histograms):
    plot_rank_pair_histograms(
        comparisons,
        value_col="corr_delta",
        xlabel="small-rank correlation - max-rank correlation",
        title="RRR vs PCA correlation delta by rank",
        ranks=PLOT_RANKS,
        xlim=(-0.2, 0.2),
    )
    return


@app.cell
def _(comparisons, plot_overlay_histograms):
    plot_overlay_histograms(
        comparisons,
        value_col="corr_squared_ratio",
        xlabel="small-rank correlation^2 / max-rank correlation^2",
        title="All source/rank relative squared correlation distributions",
        xlim=(0, 2),
    )
    return


@app.cell
def _(comparisons, plot_overlay_histograms):
    plot_overlay_histograms(
        comparisons,
        value_col="corr_ratio",
        xlabel="small-rank correlation / max-rank correlation",
        title="All source/rank relative correlation ratio distributions",
        xlim=(-5, 5),
    )
    return


@app.cell
def _(comparisons, plot_overlay_histograms):
    plot_overlay_histograms(
        comparisons,
        value_col="corr_delta",
        xlabel="small-rank correlation - max-rank correlation",
        title="All source/rank correlation delta distributions",
        xlim=(-0.2, 0.2),
    )
    return


@app.cell
def _(comparison_summary):
    comparison_summary
    return


if __name__ == "__main__":
    app.run()
