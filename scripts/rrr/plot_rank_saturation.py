"""Discover completed RRR runs and plot target-level rank saturation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.utils.rank_saturation import (
    electrode_units,
    fmri_spatial_blocks,
    mean_folds,
    summarize_rank_curve,
)


def _rank(path: Path) -> int:
    return int(path.name[1:])


def discover_units(root: Path, methods: list[str]) -> list[tuple[Path, str, str]]:
    """Return ``(analysis_dir, method, layout)`` tuples without duplicate nesting."""
    found = []
    candidates = {path.parents[2] for path in root.rglob("k*/fold_*/metrics.npz")}
    candidates.update(path.parents[1] for path in root.rglob("k*/metrics.npz"))
    for directory in sorted(candidates):
        ancestry = [directory.name.lower(), *(parent.name.lower() for parent in directory.parents)]
        method = next(
            (item for item in methods if any(name == item or name.endswith(f"_{item}") for name in ancestry)),
            None,
        )
        if method is None:
            continue
        if list(directory.glob("k*/fold_*/metrics.npz")):
            found.append((directory, method, "ieeg"))
        elif list(directory.glob("k*/metrics.npz")) and (directory.parent / "target_manifest.csv").exists():
            found.append((directory, method, "fmri"))
    return found


def _ancestor_with(path: Path, relative: str) -> Path:
    for candidate in [path, *path.parents]:
        if (candidate / relative).exists():
            return candidate
    raise ValueError(f"could not find {relative} above {path}")


def _load_ieeg(unit: Path, method: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rank_dirs = sorted((p for p in unit.glob("k*") if p.is_dir()), key=_rank)
    ranks, per_rank = [], []
    key = "nr_corr_per_target" if method == "rrr" else "pca_corr_per_target"
    for rank_dir in rank_dirs:
        fold_values = []
        for path in sorted(rank_dir.glob("fold_*/metrics.npz")):
            with np.load(path) as saved:
                if key not in saved.files:
                    break
                fold_values.append(np.asarray(saved[key]))
        if fold_values and len(fold_values) == len(list(rank_dir.glob("fold_*/metrics.npz"))):
            ranks.append(_rank(rank_dir))
            per_rank.append(fold_values)
    if not ranks:
        raise ValueError(f"no complete per-target metrics under {unit}")
    target_z = mean_folds(np.asarray(per_rank).transpose(1, 0, 2))
    artifact_root = _ancestor_with(unit, "all_subject_sig.csv")
    sig_path = artifact_root / "all_subject_sig.csv"
    sig = pd.read_csv(sig_path).rename(columns={"elec": "electrode"})
    subject_root = _ancestor_with(unit, "subjects")
    subject_file = next((subject_root / "subjects").glob("sub_*/encoding_results.joblib"))
    lags = np.asarray(joblib.load(subject_file)["lags"])
    manifest = sig.loc[sig.index.repeat(len(lags)), ["subject", "electrode"]].reset_index(drop=True)
    manifest["lag"] = np.tile(lags, len(sig))
    if len(manifest) != target_z.shape[1]:
        raise ValueError(f"manifest has {len(manifest)} rows, metrics have {target_z.shape[1]} targets")
    units, labels = electrode_units(target_z, manifest)
    return np.asarray(ranks), units, pd.DataFrame({"cluster": labels})


def _load_fmri(unit: Path, method: str, repo_root: Path, block_mm: float):
    manifest = pd.read_csv(unit / "target_manifest.csv")
    ranks, values = [], []
    for rank_dir in sorted((unit / method).glob("k*"), key=_rank):
        path = rank_dir / "metrics.npz"
        if not path.exists():
            continue
        with np.load(path) as saved:
            values.append(np.asarray(saved["corr_per_target"]))
        ranks.append(_rank(rank_dir))
    target_z = np.arctanh(np.clip(np.asarray(values), -1 + 1e-7, 1 - 1e-7))
    if target_z.shape[1] != len(manifest):
        raise ValueError("fMRI manifest/metric target count mismatch")
    clusters = fmri_spatial_blocks(manifest, repo_root, block_mm)
    return np.asarray(ranks), target_z, pd.DataFrame({"cluster": clusters})


def _safe_name(path: Path, root: Path, method: str) -> str:
    relative = path.relative_to(root)
    label = root.name if str(relative) == "." else str(relative)
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", label) + f"__{method}"


def _plot(summary, title: str, path: Path) -> None:
    curve, retention = summary.curve, summary.retention
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    ax = axes[0]
    ax.fill_between(curve["rank"], curve["lower_r"], curve["upper_r"], alpha=.2)
    ax.plot(curve["rank"], curve["mean_r"], marker="o", ms=3)
    sat = curve.iloc[0]["saturation_rank"]
    point = curve.loc[curve["rank"] == sat].iloc[0]
    ax.scatter([sat], [point["mean_r"]], marker="*", s=90, zorder=5, label=f"one-SE rank {int(sat)}")
    ax.set(xscale="log", xlabel="Rank", ylabel="Mean correlation (r)", title="Performance")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.fill_between(retention["rank"], retention["q25_loss_z"], retention["q75_loss_z"], alpha=.2)
    ax.plot(retention["rank"], retention["median_loss_z"], marker="o", ms=3, label="median loss")
    ax2 = ax.twinx()
    ax2.plot(retention["rank"], retention["fraction_within_peak_sem"], color="#d95f02", label="within peak SEM")
    ax.set(xscale="log", xlabel="Rank", ylabel="Loss from target peak (Fisher z)", title="Target retention")
    ax2.set(ylabel="Fraction within peak SEM", ylim=(0, 1.02))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rank_saturation"))
    parser.add_argument(
        "--methods", nargs="+", choices=["rrr", "pca"], default=["rrr", "pca"]
    )
    parser.add_argument("--fmri-repo-root", type=Path, default=Path("data/deep-fMRI-dataset"))
    parser.add_argument("--fmri-block-mm", type=float, default=10.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for unit, method, layout in discover_units(args.results_root, args.methods):
        try:
            if layout == "ieeg":
                ranks, target_z, groups = _load_ieeg(unit, method)
                analysis_dir = unit
            else:
                ranks, target_z, groups = _load_fmri(unit.parent, method, args.fmri_repo_root, args.fmri_block_mm)
                analysis_dir = unit.parent
            summary = summarize_rank_curve(
                ranks, target_z, groups["cluster"].to_numpy(),
                bootstrap_repeats=args.bootstrap_repeats, seed=args.seed,
            )
            name = _safe_name(analysis_dir, args.results_root, method)
            summary.curve.to_csv(args.output_dir / f"{name}__curve.csv", index=False)
            summary.retention.to_csv(args.output_dir / f"{name}__retention.csv", index=False)
            pd.DataFrame({"saturation_rank": summary.bootstrap_saturation}).to_csv(
                args.output_dir / f"{name}__saturation_bootstrap.csv", index=False
            )
            _plot(summary, f"{analysis_dir.relative_to(args.results_root)} · {method.upper()}", args.output_dir / name)
            row = summary.curve.iloc[0]
            sat_q = np.quantile(summary.bootstrap_saturation, [0.025, 0.5, 0.975])
            rows.append({"run": str(analysis_dir), "method": method, "layout": layout,
                         "n_targets": target_z.shape[1], "n_clusters": groups["cluster"].nunique(),
                         "peak_rank": int(row["peak_rank"]), "saturation_rank": int(row["saturation_rank"]),
                         "bootstrap_saturation_q025": sat_q[0],
                         "bootstrap_saturation_median": sat_q[1],
                         "bootstrap_saturation_q975": sat_q[2]})
        except Exception as error:
            skipped.append({"run": str(unit), "method": method, "reason": str(error)})
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.output_dir / "skipped.csv", index=False)
    print(f"Wrote {len(rows)} analyses to {args.output_dir}; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
