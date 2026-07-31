"""Plot combined RRR/PCA response-level rank saturation with full-model baselines."""

from __future__ import annotations

import argparse
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
from src.utils.plot_helpers import RED_BASELINE, apply_publication_style, make_viridis_cmap
from src.utils.rank_saturation import (
    SaturationSummary,
    electrode_units,
    fisher_r,
    fmri_spatial_blocks,
    mean_folds,
    summarize_rank_curve,
)


METHOD_COLORS = {"rrr": make_viridis_cmap()(0.45), "pca": "#d95f02"}


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
    condition = "cross_dataset" if any("cross_dataset" in part for part in parts) else "internal_rrr"
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
        peak = int(curve["peak_rank"].iloc[0])
        axes[0].fill_between(curve["rank"], curve["lower_r"], curve["upper_r"], color=color, alpha=.18)
        axes[0].semilogx(curve["rank"], curve["mean_r"], color=color, marker="o", ms=3.5,
                        linewidth=1.4, label=label, zorder=4)
        axes[0].axvline(peak, color=color, linestyle="--", linewidth=1.4, alpha=.55,
                        label=f"{label} peak ($k={peak}$)")

        retention = summary.retention
        axes[1].fill_between(retention["rank"], retention["q25_loss_z"], retention["q75_loss_z"],
                             color=color, alpha=.18)
        axes[1].semilogx(retention["rank"], retention["median_loss_z"], color=color,
                        marker="o", ms=3.5, linewidth=1.4, label=label, zorder=4)
        axes[1].axvline(peak, color=color, linestyle="--", linewidth=1.4, alpha=.55,
                        label=f"{label} peak ($k={peak}$)")

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--condition",
        choices=["internal_rrr", "cross_dataset"],
        help="Only plot analyses from this condition (default: plot both).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for analysis in discover_analyses(args.results_root):
        try:
            parts = output_parts(analysis, args.results_root)
            if args.condition is not None and parts[1] != args.condition:
                continue
            summaries = {}
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
                summaries[method] = summarize_rank_curve(
                    ranks, target_z, groups["cluster"].to_numpy(),
                    bootstrap_repeats=args.bootstrap_repeats,
                    seed=args.seed,
                    equal_cluster_weight=(
                        analysis.layout == "fmri"
                        and args.fmri_bootstrap_unit == "subject"
                    ),
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
                pd.DataFrame({"saturation_rank": summary.bootstrap_saturation}).to_csv(
                    destination / f"{stem}__{method}_saturation_bootstrap.csv", index=False)
                row = summary.curve.iloc[0]
                sat_q = np.quantile(summary.bootstrap_saturation, [0.025, 0.5, 0.975])
                rows.append({"run": str(analysis.root), "method": method, "layout": analysis.layout,
                             "output_dir": str(destination), "full_model_mean_r": full_mean,
                             "n_targets": loaded[method][0].shape[1],
                             "n_clusters": loaded[method][1]["cluster"].nunique(),
                             "peak_rank": int(row["peak_rank"]),
                             "saturation_rank": int(row["saturation_rank"]),
                             "bootstrap_saturation_q025": sat_q[0],
                             "bootstrap_saturation_median": sat_q[1],
                             "bootstrap_saturation_q975": sat_q[2]})
            title = " / ".join(parts)
            response_name = "Voxel" if parts[0] == "fmri" else "Electrode"
            _plot(summaries, full_mean, title, response_name, destination / stem)
        except Exception as error:
            skipped.append({"run": str(analysis.root), "reason": str(error)})
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.output_dir / "skipped.csv", index=False)
    print(f"Wrote {len(rows) // 2} combined analyses to {args.output_dir}; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
