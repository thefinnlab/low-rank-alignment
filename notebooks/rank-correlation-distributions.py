import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, "/Users/zachparis/Documents/projects/alignment-attribution")

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from src.utils.correlation import fisher_z_mean
    from src.utils.data_loaders import load_raws

    alt.data_transformers.disable_max_rows()
    return Path, alt, mo, np, pd


@app.cell
def _(mo):
    mo.Html(
        """
        <h1>Rank Correlation Distributions</h1>
        <p>
          Descriptive comparison of low-rank subspace predictions against the
          max-rank reference. The recovery check is
          <code>mean_diff_z &gt; -epsilon_z</code>, where
          <code>epsilon_z = (1 - recovery_threshold) * abs(reference_mean_z)</code>.
        </p>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    rrr_dir_input = mo.ui.text(value="rrr", label="RRR results dir", full_width=True)
    include_pca_input = mo.ui.checkbox(value=False, label="Include PCA")
    pca_dir_input = mo.ui.text(value="pca", label="PCA results dir", full_width=True)
    data_root_input = mo.ui.text(value="data/", label="Data root", full_width=True)
    recovery_threshold_input = mo.ui.slider(
        start=0.5,
        stop=1.0,
        step=0.01,
        value=0.95,
        label="Recovery threshold",
    )
    lag_count_input = mo.ui.number(value=128, start=1, label="Lag count")
    mo.vstack(
        [
            mo.hstack([rrr_dir_input, pca_dir_input, data_root_input]),
            mo.hstack([include_pca_input, recovery_threshold_input, lag_count_input]),
        ]
    )
    return (
        data_root_input,
        include_pca_input,
        lag_count_input,
        pca_dir_input,
        recovery_threshold_input,
        rrr_dir_input,
    )


@app.cell(hide_code=True)
def _(Path, np, pd):
    METHOD_METRIC_KEYS = {
        "RRR": "nr_corr_per_target",
        "PCA": "pca_corr_per_target",
    }

    def fisher_z(values) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        eps = np.finfo(np.float64).eps
        return np.arctanh(np.clip(arr, -1.0 + eps, 1.0 - eps))

    def discover_ranks(result_dir: Path) -> list[int]:
        if not result_dir.exists():
            return []
        ranks = []
        for path in result_dir.glob("k*"):
            if path.is_dir() and path.name[1:].isdigit():
                ranks.append(int(path.name[1:]))
        return sorted(ranks)

    def discover_folds(rank_dir: Path) -> list[int]:
        folds = []
        for path in rank_dir.glob("fold_*"):
            suffix = path.name.split("_", 1)[-1]
            if path.is_dir() and suffix.isdigit() and (path / "metrics.npz").exists():
                folds.append(int(suffix))
        return sorted(folds)

    def load_rank_corrs(
        result_dir: Path,
        rank: int,
        metric_key: str,
        lag_count: int,
    ) -> np.ndarray:
        rank_dir = result_dir / f"k{rank}"
        fold_corrs = []
        for fold in discover_folds(rank_dir):
            metrics_path = rank_dir / f"fold_{fold}" / "metrics.npz"
            with np.load(metrics_path) as metrics:
                if metric_key not in metrics:
                    raise KeyError(f"{metrics_path} does not contain {metric_key!r}")
                corrs = np.asarray(metrics[metric_key], dtype=np.float32)
            if corrs.size % lag_count != 0:
                raise ValueError(
                    f"{metrics_path} has {corrs.size} targets, not divisible by "
                    f"lag_count={lag_count}"
                )
            fold_corrs.append(corrs.reshape(corrs.size // lag_count, lag_count))
        if not fold_corrs:
            raise FileNotFoundError(f"No fold metrics found under {rank_dir}")
        return np.stack(fold_corrs, axis=0)

    def load_rank_null_corrs(result_dir: Path, rank: int, lag_count: int):
        rank_dir = result_dir / f"k{rank}"
        fold_corrs = []
        for fold in discover_folds(rank_dir):
            metrics_path = rank_dir / f"fold_{fold}" / "metrics.npz"
            with np.load(metrics_path) as metrics:
                if "null_aug_corr_per_target" not in metrics:
                    return None
                corrs = np.asarray(metrics["null_aug_corr_per_target"], dtype=np.float32)
            if corrs.ndim != 2:
                raise ValueError(
                    f"{metrics_path} null_aug_corr_per_target must be "
                    f"(n_repeats, n_targets), got {corrs.shape}"
                )
            if corrs.shape[1] % lag_count != 0:
                raise ValueError(
                    f"{metrics_path} has {corrs.shape[1]} targets, not divisible by "
                    f"lag_count={lag_count}"
                )
            fold_corrs.append(
                corrs.reshape(corrs.shape[0], corrs.shape[1] // lag_count, lag_count)
            )
        if not fold_corrs:
            return None
        return np.stack(fold_corrs, axis=0)

    def load_source(label: str, result_dir: Path, lag_count: int) -> dict:
        ranks = discover_ranks(result_dir)
        if not ranks:
            raise FileNotFoundError(f"No rank directories found under {result_dir}")
        metric_key = METHOD_METRIC_KEYS[label]
        corrs_by_rank = {
            rank: load_rank_corrs(result_dir, rank, metric_key, lag_count)
            for rank in ranks
        }
        null_corrs_by_rank = {}
        for rank in ranks:
            null_corrs = load_rank_null_corrs(result_dir, rank, lag_count)
            if null_corrs is not None:
                null_corrs_by_rank[rank] = null_corrs
        shape_by_rank = {rank: corrs.shape for rank, corrs in corrs_by_rank.items()}
        ref_rank = max(ranks)
        ref_shape = shape_by_rank[ref_rank]
        compatible_ranks = [
            rank for rank in ranks if shape_by_rank[rank] == ref_shape
        ]
        grid_path = result_dir / "grid_results.csv"
        summary_path = result_dir / "summary.csv"
        return {
            "label": label,
            "dir": result_dir,
            "ranks": ranks,
            "compatible_ranks": compatible_ranks,
            "ref_rank": ref_rank,
            "corrs_by_rank": corrs_by_rank,
            "null_corrs_by_rank": null_corrs_by_rank,
            "shape_by_rank": shape_by_rank,
            "grid_df": pd.read_csv(grid_path) if grid_path.exists() else None,
            "summary_df": pd.read_csv(summary_path) if summary_path.exists() else None,
        }

    return fisher_z, load_source


@app.cell(hide_code=True)
def _(
    Path,
    include_pca_input,
    lag_count_input,
    load_source,
    mo,
    pca_dir_input,
    rrr_dir_input,
):
    sources = {}
    load_errors = []

    try:
        sources["RRR"] = load_source(
            "RRR", Path(rrr_dir_input.value), int(lag_count_input.value)
        )
    except Exception as exc:
        load_errors.append(f"RRR: {exc}")

    if include_pca_input.value:
        try:
            sources["PCA"] = load_source(
                "PCA", Path(pca_dir_input.value), int(lag_count_input.value)
            )
        except Exception as exc:
            load_errors.append(f"PCA: {exc}")

    mo.stop(
        not sources,
        mo.callout(
            mo.md("No result sources could be loaded:\n\n" + "\n".join(load_errors)),
            kind="warn",
        ),
    )
    return load_errors, sources


@app.cell(hide_code=True)
def _(load_errors, mo, sources):
    _rows = []
    for _label, _source in sources.items():
        _shape = _source["shape_by_rank"][_source["ref_rank"]]
        _rows.append(
            {
                "source": _label,
                "dir": str(_source["dir"]),
                "n_ranks": len(_source["ranks"]),
                "reference_rank": _source["ref_rank"],
                "n_folds": _shape[0],
                "n_electrodes": _shape[1],
                "n_lags": _shape[2],
            }
        )
    _messages = [mo.ui.table(_rows)]
    if load_errors:
        _messages.append(
            mo.callout(
                mo.md("Some sources were skipped:\n\n" + "\n".join(load_errors)),
                kind="warn",
            )
        )
    if "PCA" in sources:
        _messages.append(
            mo.callout(
                mo.md("PCA is included for comparison, but the local PCA data may be stale."),
                kind="warn",
            )
        )
    mo.vstack(_messages)
    return


@app.cell(hide_code=True)
def _(fisher_z, np, pd):
    def build_group_corrs(
        corrs: np.ndarray,
        grouping: str,
        cluster_idxs: list[list[tuple[int, int]]] | None = None,
    ) -> tuple[list[int], list[np.ndarray]]:
        if corrs.ndim == 3:
            electrode_axis = 1
            lag_axis = 2
            cluster_stack_axis = 1
        elif corrs.ndim == 4:
            electrode_axis = 2
            lag_axis = 3
            cluster_stack_axis = 2
        else:
            raise ValueError(
                "Grouped correlations must be fold x electrode x lag or "
                "fold x repeat x electrode x lag, got shape "
                f"{corrs.shape}"
            )

        if grouping == "electrode":
            labels = list(range(corrs.shape[electrode_axis]))
            if corrs.ndim == 3:
                groups = [corrs[:, elec, :] for elec in labels]
            else:
                groups = [corrs[:, :, elec, :] for elec in labels]
        elif grouping == "lag":
            labels = list(range(corrs.shape[lag_axis]))
            if corrs.ndim == 3:
                groups = [corrs[:, :, lag] for lag in labels]
            else:
                groups = [corrs[:, :, :, lag] for lag in labels]
        elif grouping == "cluster":
            if not cluster_idxs:
                return [], []
            labels = []
            groups = []
            for idx, cluster in enumerate(cluster_idxs):
                if not cluster:
                    continue
                if corrs.ndim == 3:
                    values = [corrs[:, elec, lag] for elec, lag in cluster]
                else:
                    values = [corrs[:, :, elec, lag] for elec, lag in cluster]
                labels.append(idx)
                groups.append(np.stack(values, axis=cluster_stack_axis))
        else:
            raise ValueError(f"Unknown grouping: {grouping}")
        return labels, groups

    def fisher_z_group_mean(corrs: np.ndarray) -> float:
        return float(np.tanh(np.nanmean(fisher_z(corrs))))

    def fisher_z_group_samples(corrs: np.ndarray) -> np.ndarray:
        return np.tanh(np.nanmean(fisher_z(corrs), axis=1))

    def fisher_z_null_repeat_group_samples(corrs: np.ndarray) -> np.ndarray:
        return np.tanh(np.nanmean(fisher_z(corrs), axis=(0, 2)))

    def summarize_group_diffs(
        cmp_groups: list[np.ndarray],
        ref_groups: list[np.ndarray],
        recovery_threshold: float,
    ) -> dict:
        mean_diffs = []
        sd_diffs = []
        se_diffs = []
        epsilons = []
        passes = []
        diffs_by_fold = []

        for cmp_corrs, ref_corrs in zip(cmp_groups, ref_groups):
            cmp_z = fisher_z(cmp_corrs)
            ref_z = fisher_z(ref_corrs)
            cmp_fold_mean = np.nanmean(cmp_z, axis=1)
            ref_fold_mean = np.nanmean(ref_z, axis=1)
            diff_z = cmp_fold_mean - ref_fold_mean
            n_folds = int(np.sum(np.isfinite(diff_z)))

            mean_diff = float(np.nanmean(diff_z))
            sd_diff = float(np.nanstd(diff_z, ddof=1)) if n_folds > 1 else 0.0
            se_diff = sd_diff / np.sqrt(n_folds) if n_folds > 0 else np.nan
            epsilon = float(
                (1.0 - recovery_threshold) * abs(np.nanmean(ref_fold_mean))
            )

            mean_diffs.append(mean_diff)
            sd_diffs.append(sd_diff)
            se_diffs.append(se_diff)
            epsilons.append(epsilon)
            passes.append(mean_diff > -epsilon)
            diffs_by_fold.append(diff_z)

        return {
            "mean_diff_z": np.asarray(mean_diffs),
            "sd_diff_z": np.asarray(sd_diffs),
            "se_diff_z": np.asarray(se_diffs),
            "epsilon_z": np.asarray(epsilons),
            "passes_recovery": np.asarray(passes, dtype=bool),
            "diff_by_fold_z": np.stack(diffs_by_fold, axis=0),
        }

    def summarize_rank_totals(source: dict, recovery_threshold: float) -> pd.DataFrame:
        ref = source["corrs_by_rank"][source["ref_rank"]]
        ref_flat = ref.reshape(ref.shape[0], -1)
        rows = []
        for rank in source["compatible_ranks"]:
            cmp_corrs = source["corrs_by_rank"][rank]
            cmp_flat = cmp_corrs.reshape(cmp_corrs.shape[0], -1)
            result = summarize_group_diffs(
                [cmp_flat],
                [ref_flat],
                recovery_threshold=recovery_threshold,
            )
            cmp_mean = float(np.tanh(np.nanmean(fisher_z(cmp_flat))))
            ref_mean = float(np.tanh(np.nanmean(fisher_z(ref_flat))))
            rows.append(
                {
                    "source": source["label"],
                    "rank": rank,
                    "reference_rank": source["ref_rank"],
                    "mean_corr": cmp_mean,
                    "reference_mean_corr": ref_mean,
                    "corr_delta": cmp_mean - ref_mean,
                    "corr_ratio": cmp_mean / ref_mean if ref_mean != 0 else np.nan,
                    "mean_diff_z": result["mean_diff_z"][0],
                    "epsilon_z": result["epsilon_z"][0],
                    "sd_diff_z": result["sd_diff_z"][0],
                    "se_diff_z": result["se_diff_z"][0],
                    "passes_recovery": result["passes_recovery"][0],
                }
            )
        return pd.DataFrame(rows)

    def make_group_summary(
        source: dict,
        rank: int,
        grouping: str,
        recovery_threshold: float,
        cluster_idxs: list[list[tuple[int, int]]] | None = None,
    ) -> pd.DataFrame:
        cmp_corrs = source["corrs_by_rank"][rank]
        ref_corrs = source["corrs_by_rank"][source["ref_rank"]]
        labels, cmp_groups = build_group_corrs(cmp_corrs, grouping, cluster_idxs)
        _, ref_groups = build_group_corrs(ref_corrs, grouping, cluster_idxs)
        if not labels:
            return pd.DataFrame()

        results = summarize_group_diffs(
            cmp_groups,
            ref_groups,
            recovery_threshold=recovery_threshold,
        )
        return pd.DataFrame(
            {
                "source": source["label"],
                "rank": rank,
                grouping: labels,
                "selected_corr": [
                    fisher_z_group_mean(group) for group in cmp_groups
                ],
                "reference_corr": [
                    fisher_z_group_mean(group) for group in ref_groups
                ],
                "mean_diff_z": results["mean_diff_z"],
                "epsilon_z": results["epsilon_z"],
                "sd_diff_z": results["sd_diff_z"],
                "se_diff_z": results["se_diff_z"],
                "passes_recovery": results["passes_recovery"],
            }
        ).assign(
            delta=lambda df: df["selected_corr"] - df["reference_corr"],
            ratio=lambda df: np.divide(
                df["selected_corr"],
                df["reference_corr"],
                out=np.full(len(df), np.nan, dtype=np.float64),
                where=np.abs(df["reference_corr"]) > 1e-12,
            ),
        )

    def make_group_null_summary(
        source: dict,
        rank: int,
        grouping: str,
        cluster_idxs: list[list[tuple[int, int]]] | None = None,
    ) -> pd.DataFrame:
        null_corrs = source.get("null_corrs_by_rank", {}).get(rank)
        if null_corrs is None or null_corrs.size == 0:
            return pd.DataFrame()

        selected_corrs = source["corrs_by_rank"][rank]
        reference_corrs = source["corrs_by_rank"][source["ref_rank"]]
        labels, selected_groups = build_group_corrs(
            selected_corrs, grouping, cluster_idxs
        )
        _, reference_groups = build_group_corrs(reference_corrs, grouping, cluster_idxs)
        _, null_groups = build_group_corrs(null_corrs, grouping, cluster_idxs)
        if not labels:
            return pd.DataFrame()

        rows = []
        for label, selected_group, reference_group, null_group in zip(
            labels, selected_groups, reference_groups, null_groups
        ):
            selected_corr = fisher_z_group_mean(selected_group)
            reference_corr = fisher_z_group_mean(reference_group)
            null_values = fisher_z_null_repeat_group_samples(null_group)
            finite_null_values = null_values[np.isfinite(null_values)]
            if finite_null_values.size == 0:
                null_mean = np.nan
                null_q05 = np.nan
                null_q50 = np.nan
                null_q95 = np.nan
                null_advantage_q95 = np.nan
            else:
                null_mean = float(np.nanmean(finite_null_values))
                null_q05 = float(np.nanpercentile(finite_null_values, 5))
                null_q50 = float(np.nanpercentile(finite_null_values, 50))
                null_q95 = float(np.nanpercentile(finite_null_values, 95))
                null_advantage_q95 = float(
                    np.nanpercentile(reference_corr - finite_null_values, 95)
                )
            reference_advantage = reference_corr - selected_corr
            rows.append(
                {
                    "source": source["label"],
                    "rank": rank,
                    "reference_rank": source["ref_rank"],
                    "grouping": grouping,
                    grouping: label,
                    "selected_corr": selected_corr,
                    "reference_corr": reference_corr,
                    "delta": selected_corr - reference_corr,
                    "ratio": (
                        selected_corr / reference_corr
                        if abs(reference_corr) > 1e-12
                        else np.nan
                    ),
                    "null_aug_corr_mean": null_mean,
                    "null_aug_corr_q05": null_q05,
                    "null_aug_corr_q50": null_q50,
                    "null_aug_corr_q95": null_q95,
                    "null_sample_count": int(finite_null_values.size),
                    "reference_advantage": reference_advantage,
                    "null_advantage_q95": null_advantage_q95,
                    "under_recovered_vs_null": reference_advantage
                    > null_advantage_q95,
                }
            )
        return pd.DataFrame(rows)

    def make_group_null_samples(
        source: dict,
        rank: int,
        grouping: str,
        group_label: int,
        cluster_idxs: list[list[tuple[int, int]]] | None = None,
    ) -> pd.DataFrame:
        null_corrs = source.get("null_corrs_by_rank", {}).get(rank)
        if null_corrs is None or null_corrs.size == 0:
            return pd.DataFrame()

        selected_corrs = source["corrs_by_rank"][rank]
        reference_corrs = source["corrs_by_rank"][source["ref_rank"]]
        labels, selected_groups = build_group_corrs(
            selected_corrs, grouping, cluster_idxs
        )
        _, reference_groups = build_group_corrs(reference_corrs, grouping, cluster_idxs)
        _, null_groups = build_group_corrs(null_corrs, grouping, cluster_idxs)
        if group_label not in labels:
            return pd.DataFrame()

        group_pos = labels.index(group_label)
        null_values = fisher_z_null_repeat_group_samples(null_groups[group_pos])
        return pd.DataFrame(
            {
                "source": source["label"],
                "rank": rank,
                "reference_rank": source["ref_rank"],
                "grouping": grouping,
                grouping: group_label,
                "sample_idx": np.arange(null_values.size),
                "null_corr": null_values,
                "selected_corr": fisher_z_group_mean(selected_groups[group_pos]),
                "reference_corr": fisher_z_group_mean(reference_groups[group_pos]),
            }
        )

    def pct_passing_by_recovery(
        source: dict,
        rank: int,
        recovery_points: np.ndarray,
    ) -> pd.DataFrame:
        cmp_corrs = source["corrs_by_rank"][rank]
        ref_corrs = source["corrs_by_rank"][source["ref_rank"]]
        cmp_groups = [cmp_corrs[:, elec, :] for elec in range(cmp_corrs.shape[1])]
        ref_groups = [ref_corrs[:, elec, :] for elec in range(ref_corrs.shape[1])]
        rows = []
        for recovery in recovery_points:
            results = summarize_group_diffs(
                cmp_groups,
                ref_groups,
                recovery_threshold=float(recovery),
            )
            rows.append(
                {
                    "source": source["label"],
                    "rank": rank,
                    "recovery_threshold": float(recovery),
                    "pct_electrodes_passing": float(
                        np.nanmean(results["passes_recovery"])
                    ),
                    "median_mean_diff_z": float(np.nanmedian(results["mean_diff_z"])),
                    "median_epsilon_z": float(np.nanmedian(results["epsilon_z"])),
                }
            )
        return pd.DataFrame(rows)

    return (
        make_group_null_samples,
        make_group_null_summary,
        make_group_summary,
        pct_passing_by_recovery,
        summarize_rank_totals,
    )


@app.cell(hide_code=True)
def _(pd):
    def load_significance_tables():
        detailed_path = "significance_detailed.csv"
        sig_elecs_path = "all_subject_sig.csv"
        try:
            return pd.read_csv(detailed_path), pd.read_csv(sig_elecs_path)
        except FileNotFoundError:
            return None, None

    def build_cluster_indices(source: dict, sig_detailed, sig_elecs):
        if sig_detailed is None or sig_elecs is None:
            return []
        n_elecs = source["shape_by_rank"][source["ref_rank"]][1]
        if n_elecs == len(sig_elecs):
            elec_order = sig_elecs.rename(columns={"elec": "electrode"})[
                ["subject", "electrode"]
            ]
        else:
            elec_order = sig_detailed[["subject", "electrode"]].drop_duplicates()
            if n_elecs != len(elec_order):
                return []

        sub_elec_to_idx = {
            (row["subject"], row["electrode"]): idx
            for idx, row in elec_order.reset_index(drop=True).iterrows()
        }
        cluster_ids = sorted(
            int(cluster_id)
            for cluster_id in sig_detailed["cluster_id"].dropna().unique()
            if int(cluster_id) >= 0
        )
        cluster_idxs = [[] for _ in cluster_ids]
        cluster_id_to_pos = {cluster_id: pos for pos, cluster_id in enumerate(cluster_ids)}
        for _, row in sig_detailed.iterrows():
            cluster_id = int(row["cluster_id"])
            if cluster_id < 0 or not bool(row["sig"]):
                continue
            elec_idx = sub_elec_to_idx.get((row["subject"], row["electrode"]))
            if elec_idx is None:
                continue
            cluster_idxs[cluster_id_to_pos[cluster_id]].append(
                (int(elec_idx), int(row["lag_idx"]))
            )
        return cluster_idxs

    def electrode_order_for_source(source: dict, sig_detailed, sig_elecs):
        n_elecs = source["shape_by_rank"][source["ref_rank"]][1]
        if sig_elecs is not None and n_elecs == len(sig_elecs):
            return (
                sig_elecs.rename(columns={"elec": "electrode"})[
                    ["subject", "electrode"]
                ]
                .reset_index(drop=True)
                .copy()
            )
        if sig_detailed is None:
            return pd.DataFrame(columns=["subject", "electrode"])
        order = sig_detailed[["subject", "electrode"]].drop_duplicates()
        if n_elecs != len(order):
            return pd.DataFrame(columns=["subject", "electrode"])
        return order.reset_index(drop=True).copy()

    return (
        build_cluster_indices,
        electrode_order_for_source,
        load_significance_tables,
    )


@app.cell(hide_code=True)
def _(load_significance_tables):
    sig_detailed, sig_elecs = load_significance_tables()
    return sig_detailed, sig_elecs


@app.cell(hide_code=True)
def _(pd, recovery_threshold_input, sources, summarize_rank_totals):
    rank_summary_df = pd.concat(
        [
            summarize_rank_totals(source, float(recovery_threshold_input.value))
            for source in sources.values()
        ],
        ignore_index=True,
    )
    rank_summary_df
    return (rank_summary_df,)


@app.cell(hide_code=True)
def _(alt, mo, rank_summary_df):
    _base = alt.Chart(rank_summary_df).encode(
        x=alt.X("rank:Q", scale=alt.Scale(type="log"), title="Rank k"),
        color=alt.Color("source:N", title="Source"),
        tooltip=[
            "source",
            "rank",
            alt.Tooltip("mean_corr:Q", format=".4f"),
            alt.Tooltip("reference_mean_corr:Q", format=".4f"),
            alt.Tooltip("mean_diff_z:Q", format=".4f"),
            alt.Tooltip("epsilon_z:Q", format=".4f"),
            alt.Tooltip("sd_diff_z:Q", format=".4f"),
            alt.Tooltip("se_diff_z:Q", format=".4f"),
            "passes_recovery",
        ],
    )
    _corr = _base.mark_line(point=True).encode(
        y=alt.Y("mean_corr:Q", title="Mean target correlation")
    )
    _diff = _base.mark_line(point=True).encode(
        y=alt.Y("mean_diff_z:Q", title="Mean diff z")
    )
    _epsilon = _base.mark_line(strokeDash=[4, 4], point=True).encode(
        y=alt.Y("-epsilon_z:Q", title="Mean diff z"),
    )
    _spread = _base.transform_fold(
        ["sd_diff_z", "se_diff_z"],
        as_=["spread_metric", "spread_value"],
    ).mark_line(point=True).encode(
        y=alt.Y("spread_value:Q", title="Fold-diff spread"),
        strokeDash=alt.StrokeDash("spread_metric:N", title="Metric"),
    )
    mo.ui.tabs(
        {
            "Correlation": mo.ui.altair_chart(
                _corr.properties(width=720, height=320)
            ),
            "Mean diff vs epsilon": mo.ui.altair_chart(
                (_diff + _epsilon).properties(width=720, height=320)
            ),
            "SD / SE diff": mo.ui.altair_chart(
                _spread.properties(width=720, height=320)
            ),
            "Table": mo.ui.table(rank_summary_df),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo, sources):
    source_selector = mo.ui.dropdown(
        options=list(sources.keys()),
        value="RRR" if "RRR" in sources else next(iter(sources)),
        label="Source",
    )
    source_selector
    return (source_selector,)


@app.cell(hide_code=True)
def _(mo, source_selector, sources):
    selected_source = sources[source_selector.value]
    rank_options = [
        rank
        for rank in selected_source["compatible_ranks"]
        if rank != selected_source["ref_rank"]
    ]
    default_rank = 16 if 16 in rank_options else rank_options[min(4, len(rank_options) - 1)]
    rank_selector = mo.ui.dropdown(
        options=rank_options,
        value=default_rank,
        label="Rank",
    )
    grouping_selector = mo.ui.dropdown(
        options=["electrode", "lag", "cluster"],
        value="electrode",
        label="Grouping",
    )
    mo.hstack([rank_selector, grouping_selector])
    return grouping_selector, rank_selector, selected_source


@app.cell(hide_code=True)
def _(
    build_cluster_indices,
    electrode_order_for_source,
    grouping_selector,
    make_group_null_summary,
    make_group_summary,
    rank_selector,
    recovery_threshold_input,
    selected_source,
    sig_detailed,
    sig_elecs,
):
    cluster_idxs = build_cluster_indices(selected_source, sig_detailed, sig_elecs)
    electrode_order_df = electrode_order_for_source(
        selected_source, sig_detailed, sig_elecs
    )
    electrode_summary_df = make_group_summary(
        selected_source,
        int(rank_selector.value),
        "electrode",
        float(recovery_threshold_input.value),
    ).rename(columns={"electrode": "electrode_idx"})
    if not electrode_order_df.empty and len(electrode_order_df) == len(
        electrode_summary_df
    ):
        electrode_summary_df = electrode_summary_df.join(electrode_order_df)
    group_summary_df = make_group_summary(
        selected_source,
        int(rank_selector.value),
        grouping_selector.value,
        float(recovery_threshold_input.value),
        cluster_idxs=cluster_idxs,
    )
    grouped_null_summary_df = make_group_null_summary(
        selected_source,
        int(rank_selector.value),
        grouping_selector.value,
        cluster_idxs=cluster_idxs,
    )
    group_summary_df
    return (
        cluster_idxs,
        electrode_summary_df,
        group_summary_df,
        grouped_null_summary_df,
    )


@app.cell(hide_code=True)
def _(
    alt,
    group_summary_df,
    grouping_selector,
    mo,
    rank_selector,
    selected_source,
):
    if group_summary_df.empty:
        _output = mo.callout(
            mo.md(
                f"No {grouping_selector.value} summary is available for "
                f"{selected_source['label']} k={rank_selector.value}."
            ),
            kind="warn",
        )
    else:
        _group_col = grouping_selector.value
        _chart = alt.Chart(group_summary_df).encode(
            x=alt.X("mean_diff_z:Q", title="Mean diff z"),
            y=alt.Y("epsilon_z:Q", title="Epsilon z"),
            color=alt.Color("passes_recovery:N", title="Passes recovery"),
            tooltip=[
                "source",
                "rank",
                _group_col,
                alt.Tooltip("mean_diff_z:Q", format=".4f"),
                alt.Tooltip("epsilon_z:Q", format=".4f"),
                alt.Tooltip("sd_diff_z:Q", format=".4f"),
                alt.Tooltip("se_diff_z:Q", format=".4f"),
                "passes_recovery",
            ],
        ).mark_circle(size=45, opacity=0.75)
        _hist = alt.Chart(group_summary_df).mark_bar(opacity=0.75).encode(
            x=alt.X("mean_diff_z:Q", bin=alt.Bin(maxbins=60), title="Mean diff z"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color("passes_recovery:N", title="Passes recovery"),
            tooltip=["count()"],
        )
        _output = mo.ui.tabs(
            {
                "Mean diff vs epsilon": mo.ui.altair_chart(
                    _chart.properties(width=720, height=360)
                ),
                "Distribution": mo.ui.altair_chart(
                    _hist.properties(width=720, height=320)
                ),
                "Table": mo.ui.table(group_summary_df),
            }
        )
    _output
    return


@app.cell(hide_code=True)
def _(data_root_input, electrode_summary_df, np, pd):
    def add_electrode_coordinates(electrode_df: pd.DataFrame, data_root: str):
        if not {"subject", "electrode"}.issubset(electrode_df.columns):
            return pd.DataFrame()

        def _load_subject_coord_map(subject: int):
            import os

            import json
            file_path = os.path.join(data_root, f"sub-0{subject}/ieeg/sub-0{subject}_space-MNI152NLin2009aSym_electrodes.tsv")
            loc_df = pd.read_csv(file_path, sep="\t")  
            return {ch_name: np.asarray([x, y, z], dtype=float) for ch_name, x, y, z in zip(loc_df["name"], loc_df["x"], loc_df["y"], loc_df["z"])}

        coord_rows = []
        for subject, sub_df in electrode_df.groupby("subject", sort=True):
            coord_by_elec = _load_subject_coord_map(int(subject))

            for _, row in sub_df.iterrows():
                coord = np.asarray(
                    coord_by_elec.get(row["electrode"], [np.nan, np.nan, np.nan]),
                    dtype=float,
                )
                coord_rows.append(
                    {
                        "mni_x": coord[0],
                        "mni_y": coord[1],
                        "mni_z": coord[2],
                    }
                )

        coords_df = pd.DataFrame(coord_rows, index=electrode_df.index)
        out = electrode_df.join(coords_df)
        out["valid_coord"] = np.isfinite(out[["mni_x", "mni_y", "mni_z"]]).all(axis=1)
        out["valid_coord"] &= (
            np.abs(out[["mni_x", "mni_y", "mni_z"]]).sum(axis=1) > 0
        )
        return out

    brain_map_df = add_electrode_coordinates(
        electrode_summary_df, data_root_input.value
    )
    brain_map_df
    return (brain_map_df,)


@app.cell(hide_code=True)
def _(brain_map_df, mo, np, rank_selector, selected_source):
    mo.stop(
        brain_map_df.empty,
        mo.callout(
            mo.md(
                "No electrode coordinate metadata is available for this source. "
                "The brain map needs `subject`/`electrode` labels from the "
                "significance CSVs."
            ),
            kind="warn",
        ),
    )

    _valid_df = brain_map_df[brain_map_df["valid_coord"]].copy()
    mo.stop(
        _valid_df.empty,
        mo.callout(mo.md("No valid MNI coordinates found for electrodes."), kind="warn"),
    )

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from nilearn.plotting import plot_markers

    _coords = _valid_df[["mni_x", "mni_y", "mni_z"]].to_numpy(float)
    _values = _valid_df["passes_recovery"].astype(int).to_numpy()
    _order = np.argsort(_values)

    plt.figure(figsize=(14, 5))
    plot_markers(
        _values[_order],
        _coords[_order],
        node_size=45,
        display_mode="lzr",
        node_cmap=ListedColormap(["#dc2626", "#2563eb"]),
        node_vmin=0,
        node_vmax=1,
        colorbar=True,
    )
    _passed = int(_valid_df["passes_recovery"].sum())
    _total = len(_valid_df)
    plt.suptitle(
        (
            f"{selected_source['label']} k={rank_selector.value}: "
            f"electrodes passing recovery ({_passed}/{_total})"
        ),
        fontsize=12,
        y=1.02,
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(brain_map_df, mo):
    if brain_map_df.empty:
        _output = mo.md("")
    else:
        _cols = [
            col
            for col in [
                "source",
                "rank",
                "subject",
                "electrode",
                "passes_recovery",
                "mean_diff_z",
                "epsilon_z",
                "sd_diff_z",
                "se_diff_z",
                "mni_x",
                "mni_y",
                "mni_z",
                "valid_coord",
            ]
            if col in brain_map_df.columns
        ]
        _output = mo.ui.table(brain_map_df[_cols])
    _output
    return


@app.cell
def _(group_summary_df, grouped_null_summary_df, grouping_selector):
    grouped_distribution_df = group_summary_df.copy()
    if not grouped_distribution_df.empty and not grouped_null_summary_df.empty:
        _group_col = grouping_selector.value
        _null_cols = [
            "source",
            "rank",
            _group_col,
            "null_aug_corr_mean",
            "null_aug_corr_q05",
            "null_aug_corr_q50",
            "null_aug_corr_q95",
            "null_sample_count",
            "reference_advantage",
            "null_advantage_q95",
            "under_recovered_vs_null",
        ]
        grouped_distribution_df = grouped_distribution_df.merge(
            grouped_null_summary_df[_null_cols],
            on=["source", "rank", _group_col],
            how="left",
        )
    grouped_distribution_df
    return (grouped_distribution_df,)


@app.cell(hide_code=True)
def _(alt, grouped_distribution_df, grouping_selector, mo):
    if grouped_distribution_df.empty:
        _output = mo.callout(
            mo.md(f"No {grouping_selector.value} distribution is available."),
            kind="warn",
        )
    else:
        _group_col = grouping_selector.value
        _scatter = alt.Chart(grouped_distribution_df).mark_circle(
            size=35,
            opacity=0.65,
        ).encode(
            x=alt.X("reference_corr:Q", title="Reference corr"),
            y=alt.Y("selected_corr:Q", title="Selected rank corr"),
            tooltip=[
                _group_col,
                alt.Tooltip("reference_corr:Q", format=".4f"),
                alt.Tooltip("selected_corr:Q", format=".4f"),
                alt.Tooltip("delta:Q", format=".4f"),
                alt.Tooltip("ratio:Q", format=".4f"),
            ],
        )
        _delta_hist = alt.Chart(grouped_distribution_df).mark_bar(opacity=0.75).encode(
            x=alt.X("delta:Q", bin=alt.Bin(maxbins=80), title="Selected - reference"),
            y=alt.Y("count():Q", title=f"{grouping_selector.value.title()} groups"),
        )
        _ratio_hist = alt.Chart(grouped_distribution_df).mark_bar(opacity=0.75).encode(
            x=alt.X("ratio:Q", bin=alt.Bin(maxbins=80), title="Selected / reference"),
            y=alt.Y("count():Q", title=f"{grouping_selector.value.title()} groups"),
        )
        _output = mo.ui.tabs(
            {
                "Group scatter": mo.ui.altair_chart(
                    _scatter.properties(width=720, height=420)
                ),
                "Delta distribution": mo.ui.altair_chart(
                    _delta_hist.properties(width=720, height=320)
                ),
                "Ratio distribution": mo.ui.altair_chart(
                    _ratio_hist.properties(width=720, height=320)
                ),
                "Table": mo.ui.table(grouped_distribution_df),
            }
        )
    _output
    return


@app.cell(hide_code=True)
def _(grouped_null_summary_df, grouping_selector, mo):
    if grouped_null_summary_df.empty:
        focused_group_selector = None
        _output = mo.callout(
            mo.md("No null-augmented metrics are available for this selected rank."),
            kind="warn",
        )
    else:
        _group_col = grouping_selector.value
        _options = grouped_null_summary_df[_group_col].dropna().astype(int).tolist()
        if not _options:
            focused_group_selector = None
            _output = mo.callout(
                mo.md("No valid null-augmented groups are available."),
                kind="warn",
            )
        else:
            focused_group_selector = mo.ui.dropdown(
                options=_options,
                value=_options[0],
                label=f"{grouping_selector.value.title()} focus",
            )
            _output = focused_group_selector
    _output
    return (focused_group_selector,)


@app.cell
def _(
    cluster_idxs,
    focused_group_selector,
    grouping_selector,
    make_group_null_samples,
    pd,
    rank_selector,
    selected_source,
):
    if focused_group_selector is None:
        focused_group_null_samples_df = pd.DataFrame()
    else:
        focused_group_null_samples_df = make_group_null_samples(
            selected_source,
            int(rank_selector.value),
            grouping_selector.value,
            int(focused_group_selector.value),
            cluster_idxs=cluster_idxs,
        )
    focused_group_null_samples_df
    return (focused_group_null_samples_df,)


@app.cell(hide_code=True)
def _(
    alt,
    focused_group_null_samples_df,
    grouped_null_summary_df,
    grouping_selector,
    mo,
    pd,
):
    if grouped_null_summary_df.empty:
        _output = mo.md("")
    else:
        _group_col = grouping_selector.value
        _scatter = alt.Chart(grouped_null_summary_df).mark_circle(
            size=35,
            opacity=0.65,
        ).encode(
            x=alt.X("null_aug_corr_mean:Q", title="Null-augmented mean corr"),
            y=alt.Y("reference_corr:Q", title="Reference corr"),
            color=alt.Color(
                "under_recovered_vs_null:N",
                title="Under-recovered vs null",
            ),
            tooltip=[
                _group_col,
                alt.Tooltip("selected_corr:Q", format=".4f"),
                alt.Tooltip("reference_corr:Q", format=".4f"),
                alt.Tooltip("null_aug_corr_mean:Q", format=".4f"),
                alt.Tooltip("null_aug_corr_q05:Q", format=".4f"),
                alt.Tooltip("null_aug_corr_q95:Q", format=".4f"),
                "null_sample_count",
                alt.Tooltip("reference_advantage:Q", format=".4f"),
                alt.Tooltip("null_advantage_q95:Q", format=".4f"),
                "under_recovered_vs_null",
            ],
        )
        _advantage = alt.Chart(grouped_null_summary_df).mark_bar(opacity=0.75).encode(
            x=alt.X(
                "reference_advantage:Q",
                bin=alt.Bin(maxbins=80),
                title="Reference - selected",
            ),
            y=alt.Y("count():Q", title=f"{grouping_selector.value.title()} groups"),
            color=alt.Color(
                "under_recovered_vs_null:N",
                title="Under-recovered vs null",
            ),
        )
        if focused_group_null_samples_df.empty:
            _focused = alt.Chart(pd.DataFrame({"null_corr": []})).mark_bar()
        else:
            _first = focused_group_null_samples_df.iloc[0]
            _line_df = pd.DataFrame(
                {
                    "label": ["Selected rank", "Reference"],
                    "corr": [_first["selected_corr"], _first["reference_corr"]],
                }
            )
            _hist = alt.Chart(focused_group_null_samples_df).mark_bar(
                opacity=0.45,
                color="#94a3b8",
            ).encode(
                x=alt.X(
                    "null_corr:Q",
                    bin=alt.Bin(maxbins=50),
                    title="Grouped null correlation",
                ),
                y=alt.Y("count():Q", title="Null samples"),
                tooltip=["count()"],
            )
            _lines = alt.Chart(_line_df).mark_rule(size=2).encode(
                x=alt.X("corr:Q"),
                color=alt.Color("label:N", title="Observed value"),
                tooltip=[
                    "label",
                    alt.Tooltip("corr:Q", format=".4f"),
                ],
            )
            _focused = _hist + _lines
        _output = mo.ui.tabs(
            {
                "Null scatter": mo.ui.altair_chart(
                    _scatter.properties(width=720, height=420)
                ),
                "Advantage distribution": mo.ui.altair_chart(
                    _advantage.properties(width=720, height=320)
                ),
                "Focused null": mo.ui.altair_chart(
                    _focused.properties(width=720, height=320)
                ),
                "Table": mo.ui.table(grouped_null_summary_df),
            }
        )
    _output
    return


@app.cell
def _(np, pct_passing_by_recovery, rank_selector, selected_source):
    recovery_curve_df = pct_passing_by_recovery(
        selected_source,
        int(rank_selector.value),
        np.linspace(0.5, 1.0, 51),
    )
    recovery_curve_df
    return (recovery_curve_df,)


@app.cell(hide_code=True)
def _(alt, mo, recovery_curve_df):
    _chart = alt.Chart(recovery_curve_df).mark_line(point=True).encode(
        x=alt.X("recovery_threshold:Q", title="Recovery threshold"),
        y=alt.Y("pct_electrodes_passing:Q", title="Fraction of electrodes passing"),
        tooltip=[
            alt.Tooltip("recovery_threshold:Q", format=".2f"),
            alt.Tooltip("pct_electrodes_passing:Q", format=".2%"),
            alt.Tooltip("median_mean_diff_z:Q", format=".4f"),
            alt.Tooltip("median_epsilon_z:Q", format=".4f"),
        ],
    )
    mo.ui.altair_chart(_chart.properties(width=720, height=320))
    return


@app.cell
def _():
    num_examples = 500
    num_targets = 500

    num_linear_features = 10
    num_interaction_features = 10
    num_random_features = 100

    prob_linear_inclusion = 0.5
    prob_linear_interaction_inclusion = 0.1
    prob_interaction_inclusion = 0.2

    linear_weight_strength = 10
    interaction_weight_strength = 0
    noise_strength = 30
    return (
        interaction_weight_strength,
        linear_weight_strength,
        noise_strength,
        num_examples,
        num_interaction_features,
        num_linear_features,
        num_targets,
        prob_interaction_inclusion,
        prob_linear_inclusion,
        prob_linear_interaction_inclusion,
    )


@app.cell
def _(
    interaction_weight_strength,
    linear_weight_strength,
    noise_strength,
    np,
    num_examples,
    num_interaction_features,
    num_linear_features,
    num_targets,
    prob_interaction_inclusion,
    prob_linear_inclusion,
    prob_linear_interaction_inclusion,
):
    features = np.random.randn(num_examples, num_linear_features)
    interaction_term_combos = np.random.randn(num_linear_features, num_interaction_features) * (np.random.uniform(0, 1, (num_linear_features, num_interaction_features)) > prob_linear_interaction_inclusion).astype(int)

    interaction_term_weights = np.random.randn(num_targets, num_interaction_features)
    interaction_features = (features - features.mean(0)) @ interaction_term_combos
    all_features = np.concatenate((features, interaction_features), axis=1)

    linear_target_weights = np.random.randn(num_targets, num_linear_features) * linear_weight_strength * (np.random.uniform(0, 1, (num_targets, 10)) > prob_linear_inclusion).astype(int)
    interaction_target_weights = np.random.randn(num_targets, num_interaction_features) * interaction_weight_strength * (np.random.uniform(0, 1, (num_targets, num_interaction_features)) > prob_interaction_inclusion).astype(int)
    target_values = features @ linear_target_weights.T + interaction_features @ interaction_target_weights.T
    # Add noise
    target_values = target_values + np.random.randn(*target_values.shape) * noise_strength
    return


@app.cell
def _():
    from src.utils.encoding import train_encoding, get_ridge_model_constructor

    return


@app.cell
def _():
    # from scipy.stats import ortho_group


    # all_corrs = []
    # for _ in range(100):
    #     # Generate model input feature space. Add random noise features and then rotate. Do not include interaction terms.
    #     random_features = np.random.randn(num_examples, num_random_features)
    #     input_features = np.concatenate((features, random_features), axis=1)

    #     rotation_matrix = ortho_group.rvs(dim=input_features.shape[-1])

    #     if np.linalg.det(rotation_matrix) < 0:
    #         rotation_matrix[0, :] *= -1

    #     rotated_features = input_features @ rotation_matrix

    #     corrs, _, _, _, _ = train_encoding(rotated_features, target_values, model_constructor_fn=get_ridge_model_constructor(), num_folds=5)
    #     all_corrs.append(corrs)
    return


@app.cell
def _():
    # plt.hist(np.stack(all_corrs).mean(1)[:, 0])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
