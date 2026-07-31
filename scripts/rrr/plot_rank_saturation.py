"""Plot combined RRR/PCA response-level rank saturation with full-model baselines."""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent:
    if (_ROOT / "src").is_dir():
        break
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils.plot_helpers import RED_BASELINE, RRR_PCA_COLORS, apply_publication_style
from src.utils.rank_saturation import (
    SaturationSummary,
    _cluster_bootstrap_means,
    electrode_units,
    fisher_r,
    fmri_spatial_blocks,
    mean_folds,
    summarize_equivalence_curve,
    summarize_rank_curve,
)


METHOD_COLORS = RRR_PCA_COLORS


@dataclass(frozen=True)
class Analysis:
    root: Path
    layout: str
    units: dict[str, Path]


def _rank(path: Path) -> int:
    return int(path.name[1:])


def discover_analyses(root: Path) -> list[Analysis]:
    """Find analysis roots that contain complete RRR and PCA result units."""
    grouped: dict[Path, dict[str, tuple[Path, str]]] = {}
    candidates = {path.parents[2] for path in root.rglob("k*/fold_*/metrics.npz")}
    candidates.update(path.parents[1] for path in root.rglob("k*/metrics.npz"))
    for unit in sorted(candidates):
        if unit.name.lower() not in {"rrr", "pca"}:
            continue
        method = unit.name.lower()
        if list(unit.glob("k*/fold_*/metrics.npz")):
            layout = "ieeg"
        elif list(unit.glob("k*/metrics.npz")) and (unit.parent / "target_manifest.csv").exists():
            layout = "fmri"
        else:
            continue
        grouped.setdefault(unit.parent, {})[method] = (unit, layout)

    analyses = []
    for analysis_root, found in grouped.items():
        if set(found) != {"rrr", "pca"}:
            continue
        layouts = {layout for _, layout in found.values()}
        if len(layouts) == 1:
            analyses.append(Analysis(analysis_root, layouts.pop(), {
                method: unit for method, (unit, _) in found.items()
            }))
    return analyses


def _ancestor_with(path: Path, relative: str) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / relative).exists():
            return candidate
    raise ValueError(f"could not find {relative} above {path}")


def _ieeg_manifest(unit: Path) -> tuple[pd.DataFrame, np.ndarray]:
    artifact_root = _ancestor_with(unit, "all_subject_sig.csv")
    sig = pd.read_csv(artifact_root / "all_subject_sig.csv").rename(columns={"elec": "electrode"})
    subject_root = _ancestor_with(unit, "subjects")
    subject_file = next((subject_root / "subjects").glob("sub_*/encoding_results.joblib"))
    lags = np.asarray(joblib.load(subject_file)["lags"])
    manifest = sig.loc[sig.index.repeat(len(lags)), ["subject", "electrode"]].reset_index(drop=True)
    manifest["lag"] = np.tile(lags, len(sig))
    return manifest, lags


def _load_ieeg(unit: Path, method: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ranks, per_rank = [], []
    key = "nr_corr_per_target" if method == "rrr" else "pca_corr_per_target"
    for rank_dir in sorted((p for p in unit.glob("k*") if p.is_dir()), key=_rank):
        paths = sorted(rank_dir.glob("fold_*/metrics.npz"))
        fold_values = []
        for path in paths:
            with np.load(path) as saved:
                if key not in saved.files:
                    break
                fold_values.append(np.asarray(saved[key]))
        if paths and len(fold_values) == len(paths):
            ranks.append(_rank(rank_dir))
            per_rank.append(fold_values)
    if not ranks:
        raise ValueError(f"no complete per-target {method.upper()} metrics under {unit}")
    target_z = mean_folds(np.asarray(per_rank).transpose(1, 0, 2))
    manifest, _ = _ieeg_manifest(unit)
    if len(manifest) != target_z.shape[1]:
        raise ValueError(f"manifest has {len(manifest)} rows, metrics have {target_z.shape[1]} targets")
    units, labels = electrode_units(target_z, manifest)
    return np.asarray(ranks), units, pd.DataFrame({"cluster": labels})


def _load_ieeg_full(unit: Path) -> float:
    paths = sorted(unit.glob("fold_*/full_model_metrics.npz"))
    values = []
    for path in paths:
        with np.load(path) as saved:
            values.append(np.asarray(saved["full_corr_per_target"]))
    if not values:
        raise ValueError(f"no full-model per-target metrics under {unit}")
    target_z = mean_folds(np.asarray(values)[:, None, :])[0]
    manifest, _ = _ieeg_manifest(unit)
    units, _ = electrode_units(target_z[None, :], manifest)
    return float(fisher_r(np.nanmean(units)))


def _load_fmri(
    unit: Path,
    method: str,
    repo_root: Path,
    block_mm: float,
    bootstrap_unit: str = "spatial_block",
):
    manifest = pd.read_csv(unit.parent / "target_manifest.csv")
    ranks, values = [], []
    for rank_dir in sorted(unit.glob("k*"), key=_rank):
        path = rank_dir / "metrics.npz"
        if not path.exists():
            continue
        with np.load(path) as saved:
            values.append(np.asarray(saved["corr_per_target"]))
        ranks.append(_rank(rank_dir))
    target_z = np.arctanh(np.clip(np.asarray(values), -1 + 1e-7, 1 - 1e-7))
    if target_z.shape[1] != len(manifest):
        raise ValueError("fMRI manifest/metric target count mismatch")
    if bootstrap_unit == "subject":
        clusters = manifest["subject"].astype(str).to_numpy()
    elif bootstrap_unit == "spatial_block":
        clusters = fmri_spatial_blocks(manifest, repo_root, block_mm)
    else:
        raise ValueError(f"unknown fMRI bootstrap unit: {bootstrap_unit}")
    return np.asarray(ranks), target_z, pd.DataFrame({"cluster": clusters})


def _load_fmri_full(root: Path) -> float:
    with np.load(root / "full" / "metrics.npz") as saved:
        values = np.asarray(saved["corr_per_target"])
    return float(fisher_r(np.nanmean(np.arctanh(np.clip(values, -1 + 1e-7, 1 - 1e-7)))))


def output_parts(analysis: Analysis, results_root: Path) -> tuple[str, str, str, str]:
    """Return the canonical dataset/condition/model/layer output hierarchy."""
    relative = analysis.root.relative_to(results_root)
    parts = relative.parts
    dataset = "fmri" if parts and parts[0] == "fmri" else "podcast"
    movie = next((part.removeprefix("movie-") for part in parts if part.startswith("movie-")), None)
    if movie:
        dataset = movie
    if dataset == "fmri":
        condition = (
            "internal_rrr"
            if analysis.root.name.startswith("within_fmri_rrr")
            else "cross_dataset"
        )
    else:
        condition = (
            "cross_dataset"
            if any("cross_dataset" in part for part in parts)
            else "internal_rrr"
        )
    match = next(
        (item for part in parts if (item := re.fullmatch(r"(.+)_layer(\d+)(?:_offset-?\d+)?", part))),
        None,
    )
    if match is None:
        raise ValueError(f"could not infer model and layer from {relative}")
    return dataset, condition, match.group(1), f"layer{match.group(2)}"


def _artifact_stem(analysis: Analysis, results_root: Path) -> str:
    """Name an analysis within its grouped leaf without losing nested run identity."""
    ignored = {"fmri"}
    remaining = [
        part for part in analysis.root.relative_to(results_root).parts
        if part not in ignored
        and not part.startswith("movie-")
        and re.fullmatch(r".+_layer\d+(?:_offset-?\d+)?", part) is None
    ]
    label = "__".join(remaining) if remaining else "rank_saturation"
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", label)


def _plot(
    summaries: dict[str, SaturationSummary],
    full_mean: float,
    title: str,
    response_name: str,
    path: Path,
) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for method, summary in summaries.items():
        color = METHOD_COLORS[method]
        label = method.upper()
        curve = summary.curve
        saturation = curve["saturation_rank"].iloc[0]
        axes[0].fill_between(curve["rank"], curve["lower_r"], curve["upper_r"], color=color, alpha=.18)
        axes[0].semilogx(curve["rank"], curve["mean_r"], color=color, marker="o", ms=3.5,
                        linewidth=1.4, label=label, zorder=4)
        if pd.notna(saturation):
            saturation = int(saturation)
            axes[0].axvline(saturation, color=color, linestyle="--", linewidth=1.4, alpha=.55,
                            label=f"{label} saturation ($k={saturation}$)")

        retention = summary.retention
        axes[1].fill_between(retention["rank"], retention["q25_loss_z"], retention["q75_loss_z"],
                             color=color, alpha=.18)
        axes[1].semilogx(retention["rank"], retention["median_loss_z"], color=color,
                        marker="o", ms=3.5, linewidth=1.4, label=label, zorder=4)
        if pd.notna(saturation):
            axes[1].axvline(saturation, color=color, linestyle="--", linewidth=1.4, alpha=.55,
                            label=f"{label} saturation ($k={saturation}$)")

    axes[0].axhline(full_mean, color=RED_BASELINE, linestyle="--", linewidth=1.0,
                    label="Full Model (Baseline)")
    axes[0].set(xlabel="Rank $k$ (log scale)", ylabel="Mean correlation (r)", title="Encoding Performance")
    axes[1].set(xlabel="Rank $k$ (log scale)", ylabel="Loss from condition peak (Fisher z)",
                title=f"Per-{response_name} Gap to Best Rank")
    for ax in axes:
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.05)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def _family_name(model: str) -> str:
    return "random-gpt2-xl" if model.startswith("random-gpt2-xl") else model


def _exact_family_signflip(effects: np.ndarray) -> float:
    effects = np.asarray(effects, dtype=float)
    if not len(effects):
        return np.nan
    observed = float(np.mean(effects))
    null = np.asarray([
        np.mean(effects * np.asarray(signs))
        for signs in itertools.product((-1, 1), repeat=len(effects))
    ])
    return float(np.mean(null >= observed))


def _pvalue_stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n/a"
    if pvalue < .001:
        return "***"
    if pvalue < .01:
        return "**"
    if pvalue < .05:
        return "*"
    return "n.s."


def _shared_saturation_max_exponent(frame: pd.DataFrame) -> int:
    """Return the log2 exponent of the shared upper saturation-rank tick."""
    values = frame["saturation_rank"].to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        raise ValueError("no finite positive saturation ranks")
    return max(0, int(np.ceil(np.log2(np.max(values)))))


def _plot_cross_model_summary(
    frame: pd.DataFrame,
    title: str,
    path: Path,
    *,
    max_tick_exponent: int,
) -> dict:
    """Draw paired saturation violins with explicit distribution means."""
    paired = frame.pivot_table(
        index=["run", "model", "layer"], columns="method",
        values="saturation_rank", aggfunc="first",
    ).dropna(subset=["rrr", "pca"]).reset_index()
    if paired.empty:
        raise ValueError("no paired finite RRR/PCA saturation ranks")
    paired["family"] = paired["model"].map(_family_name)
    paired["log2_pca_over_rrr"] = np.log2(paired["pca"] / paired["rrr"])
    family = (
        paired.groupby("family", sort=False)
        .agg(
            n_layers=("log2_pca_over_rrr", "size"),
            mean_log2_pca_over_rrr=("log2_pca_over_rrr", "mean"),
            median_rrr_rank=("rrr", "median"),
            median_pca_rank=("pca", "median"),
        ).reset_index()
    )
    pvalue = _exact_family_signflip(family["mean_log2_pca_over_rrr"].to_numpy())
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(3.8, 4.0))
    methods = ["rrr", "pca"]
    positions = [1, 2]
    log_values = [np.log2(paired[method].to_numpy(dtype=float)) for method in methods]
    violin = ax.violinplot(log_values, positions=positions, widths=.72, showextrema=False)
    for body, method in zip(violin["bodies"], methods):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_edgecolor(METHOD_COLORS[method])
        body.set_alpha(.22)
        body.set_linewidth(1.0)
    boxes = ax.boxplot(
        log_values, positions=positions, widths=.20, patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "#4b5563", "linewidth": .8},
        capprops={"color": "#4b5563", "linewidth": .8},
        boxprops={"color": "#4b5563", "linewidth": .8},
    )
    for patch, method in zip(boxes["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(.18)
    offsets = np.linspace(-.12, .12, len(paired)) if len(paired) > 1 else np.zeros(1)
    for offset, (_, row) in zip(offsets, paired.iterrows()):
        ys = [np.log2(row["rrr"]), np.log2(row["pca"])]
        paired_positions = np.asarray(positions) + offset
        ax.plot(paired_positions, ys, color="#9ca3af", linewidth=.65, alpha=.5, zorder=2)
        ax.scatter(paired_positions, ys, c=[METHOD_COLORS[m] for m in methods], s=19,
                   edgecolors="white", linewidths=.35, zorder=3)

    means = {method: float(paired[method].mean()) for method in methods}
    for position, method, values in zip(positions, methods, log_values):
        mean_log_position = np.log2(means[method])
        ax.scatter(position, mean_log_position, marker="D", s=28, color="black", zorder=5)
        ax.annotate(
            f"mean = {means[method]:.1f}",
            (position, mean_log_position), xytext=(0, 8), textcoords="offset points",
            ha="center", va="bottom", fontsize=7,
        )
    bracket_y = max_tick_exponent + .55
    ax.plot([1, 1, 2, 2], [bracket_y - .10, bracket_y, bracket_y, bracket_y - .10],
            color="black", linewidth=.8)
    ax.text(1.5, bracket_y + .06, _pvalue_stars(pvalue),
            ha="center", va="bottom", fontsize=9)
    integer_ticks = np.arange(0, max_tick_exponent + 1)
    ax.set_yticks(integer_ticks, [str(int(2 ** value)) for value in integer_ticks])
    ax.set_xticks(positions, ["RRR", "PCA"])
    ax.set(
        ylabel="Saturation rank $k$",
        title=title,
        ylim=(-.35, bracket_y + .45),
        xlim=(.45, 2.55),
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.05)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    paired.to_csv(path.with_name(path.name + "__paired_runs.csv"), index=False)
    family.to_csv(path.with_name(path.name + "__family_effects.csv"), index=False)
    return {
        "n_runs": len(paired),
        "n_unpaired_or_nonequivalent": int(
            frame[["run", "model", "layer"]].drop_duplicates().shape[0] - len(paired)
        ),
        "n_families": len(family),
        "median_rrr_rank": float(paired["rrr"].median()),
        "median_pca_rank": float(paired["pca"].median()),
        "geometric_mean_pca_over_rrr": float(2 ** family["mean_log2_pca_over_rrr"].mean()),
        "mean_rrr_rank": means["rrr"],
        "mean_pca_rank": means["pca"],
        "pvalue": pvalue,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rank_saturation"))
    parser.add_argument("--fmri-repo-root", type=Path, default=Path("data/deep-fMRI-dataset"))
    parser.add_argument("--fmri-block-mm", type=float, default=10.0)
    parser.add_argument(
        "--fmri-bootstrap-unit",
        choices=["spatial_block", "subject"],
        default="spatial_block",
        help=(
            "Resample 10-mm spatial blocks (default) or equal-weighted subjects "
            "for fMRI performance uncertainty."
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Independent analyses to process concurrently (default: 4; use 1 for serial).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--performance-fraction",
        type=float,
        default=0.985,
        help="Fraction of peak Fisher-z performance defining iEEG saturation (default: 0.985).",
    )
    parser.add_argument(
        "--fmri-equivalence-fraction",
        type=float,
        default=0.985,
        help="fMRI practical-equivalence fraction of best performance (default: 0.985).",
    )
    parser.add_argument(
        "--condition",
        choices=["internal_rrr", "cross_dataset"],
        help="Only plot analyses from this condition (default: plot both).",
    )
    args = parser.parse_args()
    if not 0 < args.performance_fraction <= 1:
        parser.error("--performance-fraction must lie in (0, 1]")
    if not 0 < args.fmri_equivalence_fraction < 1:
        parser.error("--fmri-equivalence-fraction must lie between zero and one")
    if args.jobs < 1:
        parser.error("--jobs must be at least one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyses = []
    for analysis in discover_analyses(args.results_root):
        try:
            parts = output_parts(analysis, args.results_root)
        except Exception:
            continue
        if args.condition is None or parts[1] == args.condition:
            analyses.append((analysis, parts))

    def process_analysis(item):
        analysis, parts = item
        analysis_rows = []
        try:
            summaries = {}
            equivalence = {}
            fraction_saturation = {}
            loaded = {}
            for method, unit in analysis.units.items():
                if analysis.layout == "ieeg":
                    ranks, target_z, groups = _load_ieeg(unit, method)
                else:
                    ranks, target_z, groups = _load_fmri(
                        unit,
                        method,
                        args.fmri_repo_root,
                        args.fmri_block_mm,
                        args.fmri_bootstrap_unit,
                    )
                loaded[method] = (target_z, groups)
                equal_cluster_weight = (
                    analysis.layout == "fmri"
                    and args.fmri_bootstrap_unit == "subject"
                )
                bootstrap_means = None
                if analysis.layout == "fmri":
                    fraction_saturation[method] = int(
                        summaries[method].curve["saturation_rank"].iloc[0]
                    )
                    bootstrap_means = _cluster_bootstrap_means(
                        target_z,
                        groups["cluster"].to_numpy(),
                        args.bootstrap_repeats,
                        args.seed,
                        equal_cluster_weight=equal_cluster_weight,
                    )
                summaries[method] = summarize_rank_curve(
                    ranks, target_z, groups["cluster"].to_numpy(),
                    bootstrap_repeats=args.bootstrap_repeats,
                    seed=args.seed,
                    equal_cluster_weight=equal_cluster_weight,
                    bootstrap_means=bootstrap_means,
                    performance_fraction=args.performance_fraction,
                )
                if analysis.layout == "fmri":
                    best_z = float(summaries[method].curve["mean_z"].max())
                    delta_z = (1 - args.fmri_equivalence_fraction) * abs(best_z)
                    equivalence[method] = summarize_equivalence_curve(
                        ranks,
                        target_z,
                        groups["cluster"].to_numpy(),
                        delta_z,
                        bootstrap_repeats=args.bootstrap_repeats,
                        seed=args.seed,
                        equal_cluster_weight=equal_cluster_weight,
                        bootstrap_means=bootstrap_means,
                    )
                    equivalence[method].curve["delta_source"] = (
                        f"{args.fmri_equivalence_fraction:.4f}_of_best"
                    )
                    summaries[method].curve["saturation_rank"] = (
                        equivalence[method].saturation_rank
                    )
            full_mean = (_load_ieeg_full(analysis.units["rrr"]) if analysis.layout == "ieeg"
                         else _load_fmri_full(analysis.root))
            destination = args.output_dir.joinpath(*parts)
            destination.mkdir(parents=True, exist_ok=True)
            stem = _artifact_stem(analysis, args.results_root)
            if analysis.layout == "fmri" and args.fmri_bootstrap_unit == "subject":
                stem += "__subject_bootstrap"
            for method, summary in summaries.items():
                summary.curve.to_csv(destination / f"{stem}__{method}_curve.csv", index=False)
                summary.retention.to_csv(destination / f"{stem}__{method}_retention.csv", index=False)
                if analysis.layout == "fmri":
                    equivalence[method].curve.to_csv(
                        destination / f"{stem}__{method}_equivalence.csv", index=False
                    )
                    pd.DataFrame({
                        "fraction_of_peak_saturation_rank": summary.bootstrap_saturation
                    }).to_csv(
                        destination / f"{stem}__{method}_fraction_saturation_bootstrap.csv",
                        index=False,
                    )
                else:
                    pd.DataFrame({"saturation_rank": summary.bootstrap_saturation}).to_csv(
                        destination / f"{stem}__{method}_saturation_bootstrap.csv", index=False
                    )
                row = summary.curve.iloc[0]
                sat_q = np.quantile(summary.bootstrap_saturation, [0.025, 0.5, 0.975])
                output_row = {"run": str(analysis.root), "dataset": parts[0],
                             "condition": parts[1], "model": parts[2], "layer": parts[3],
                             "method": method, "layout": analysis.layout,
                             "output_dir": str(destination), "full_model_mean_r": full_mean,
                             "n_targets": loaded[method][0].shape[1],
                             "n_clusters": loaded[method][1]["cluster"].nunique(),
                             "peak_rank": int(row["peak_rank"]),
                             "saturation_rank": row["saturation_rank"],
                             "saturation_method": (
                                 "simultaneous_equivalence" if analysis.layout == "fmri"
                                 else "fraction_of_peak"
                             )}
                if analysis.layout == "fmri":
                    output_row.update({
                        "fraction_of_peak_rank": fraction_saturation[method],
                        "equivalence_delta_z": float(
                            equivalence[method].curve["delta_z"].iloc[0]
                        ),
                        "equivalence_delta_source": equivalence[method].curve[
                            "delta_source"
                        ].iloc[0],
                        "minimum_adjusted_pvalue": equivalence[method].minimum_pvalue,
                    })
                else:
                    output_row.update({
                        "performance_fraction": float(row["performance_fraction"]),
                        "threshold_z": float(row["threshold_z"]),
                        "bootstrap_saturation_q025": sat_q[0],
                        "bootstrap_saturation_median": sat_q[1],
                        "bootstrap_saturation_q975": sat_q[2],
                    })
                analysis_rows.append(output_row)
            title = " / ".join(parts)
            response_name = "Voxel" if parts[0] == "fmri" else "Electrode"
            _plot(summaries, full_mean, title, response_name, destination / stem)
            return analysis_rows, None
        except Exception as error:
            return [], {"run": str(analysis.root), "reason": str(error)}

    processed = joblib.Parallel(n_jobs=min(args.jobs, len(analyses) or 1))(
        joblib.delayed(process_analysis)(item) for item in analyses
    )
    rows, skipped = [], []
    for analysis_rows, error in processed:
        rows.extend(analysis_rows)
        if error is not None:
            skipped.append(error)
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    cross_model_rows = []
    results_frame = pd.DataFrame(rows)
    if not results_frame.empty:
        max_tick_exponent = _shared_saturation_max_exponent(results_frame)
        for (dataset, condition), group in results_frame.groupby(["dataset", "condition"]):
            try:
                directory = args.output_dir / dataset / condition
                directory.mkdir(parents=True, exist_ok=True)
                metadata = _plot_cross_model_summary(
                    group,
                    f"{dataset} / {condition}",
                    directory / "cross_model_saturation",
                    max_tick_exponent=max_tick_exponent,
                )
                cross_model_rows.append({"dataset": dataset, "condition": condition, **metadata})
            except Exception as error:
                skipped.append({"run": f"{dataset}/{condition}/cross_model", "reason": str(error)})
    pd.DataFrame(cross_model_rows).to_csv(
        args.output_dir / "cross_model_summary.csv", index=False
    )
    pd.DataFrame(skipped).to_csv(args.output_dir / "skipped.csv", index=False)
    print(f"Wrote {len(rows) // 2} combined analyses to {args.output_dir}; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
