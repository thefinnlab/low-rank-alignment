"""Generate model/layer saturation bar plots for every dataset transfer mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.saturation_bar_summary import (
    build_saturation_bar_table,
    plot_saturation_bars,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank-summary", type=Path,
        default=Path("results/rank_saturation/summary.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/rank_saturation/model_summaries"),
    )
    parser.add_argument("--saturation-column", default="saturation_rank")
    args = parser.parse_args()

    summary = pd.read_csv(args.rank_summary)
    if args.saturation_column not in summary:
        raise ValueError(f"Missing {args.saturation_column!r} in {args.rank_summary}")
    table, skipped = build_saturation_bar_table(
        summary, saturation_column=args.saturation_column
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "saturation_bars.csv", index=False)
    skipped.to_csv(args.output_dir / "skipped.csv", index=False)

    count = 0
    for (target, source), group in table.groupby(
        ["target_dataset", "source_dataset"], sort=True
    ):
        filename = f"{target}__within" if target == source else f"{target}__from_{source}"
        plot_saturation_bars(group, args.output_dir / filename)
        count += 1
    print(
        f"Wrote {count} dataset/mode figure(s) from {len(table)} bars to "
        f"{args.output_dir}; skipped {len(skipped)} row(s)"
    )


if __name__ == "__main__":
    main()
