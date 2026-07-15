"""Summarize RRR small-GPU sweep outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _summarize_variant(name: str, run_dir: Path) -> dict:
    summary_path = run_dir / "summary.csv"
    grid_path = run_dir / "grid_results.csv"
    row = {
        "variant": name,
        "run_dir": str(run_dir),
        "status": "missing",
        "best_k": None,
        "best_nr_corr": None,
        "best_rrr_corr": None,
        "full_corr": None,
        "n_grid_rows": 0,
    }
    if not summary_path.exists() and not grid_path.exists():
        return row

    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        row["n_grid_rows"] = int(len(grid))
    else:
        grid = pd.DataFrame()

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        source = summary
    else:
        source = grid.groupby("k", as_index=False).mean(numeric_only=True)

    if source.empty or "nr_corr" not in source:
        row["status"] = "incomplete"
        return row

    best_idx = source["nr_corr"].idxmax()
    best = source.loc[best_idx]
    row.update(
        {
            "status": "complete" if summary_path.exists() else "partial",
            "best_k": int(best["k"]),
            "best_nr_corr": float(best["nr_corr"]),
            "best_rrr_corr": (
                float(best["rrr_corr"]) if "rrr_corr" in best else None
            ),
            "full_corr": float(best["full_corr"]) if "full_corr" in best else None,
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="results/gpt2-xl-batch_layer24_offset0/rrr_small_gpu",
        help="Directory containing one subdirectory per sweep variant.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["conservative", "balanced", "aggressive"],
        help="Variant subdirectories to summarize.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = [_summarize_variant(name, root / name) for name in args.variants]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["best_nr_corr", "n_grid_rows"],
            ascending=[False, False],
            na_position="last",
        )
    print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
