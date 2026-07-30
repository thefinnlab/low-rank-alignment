"""Plot linguistic RRR-minus-PCA decoding at each model/layer RRR saturation rank."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.utils.language_decoding_saturation_summary import (
    build_decoding_saturation_table,
    plot_decoding_saturation_bars,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=Path("results/language_decoding_batch"))
    parser.add_argument("--rank-summary", type=Path, default=Path("results/rank_saturation/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or args.batch_dir / "saturation_summary"

    paired_paths = sorted(args.batch_dir.rglob("paired_rrr_pca_gaps.csv"))
    if not paired_paths:
        raise FileNotFoundError(f"No paired_rrr_pca_gaps.csv files under {args.batch_dir}")
    paired_frames, manifest_frames = [], []
    for paired_path in paired_paths:
        manifest_path = paired_path.with_name("source_manifest.csv")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing source manifest beside {paired_path}")
        paired_frames.append(pd.read_csv(paired_path))
        manifest_frames.append(pd.read_csv(manifest_path))
    table, skipped = build_decoding_saturation_table(
        pd.concat(paired_frames, ignore_index=True),
        pd.concat(manifest_frames, ignore_index=True),
        pd.read_csv(args.rank_summary),
    )
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "decoding_at_rrr_saturation.csv", index=False)
    skipped.to_csv(output / "skipped.csv", index=False)
    for (dataset, task), group in table.groupby(["source_dataset", "task"], sort=True):
        plot_decoding_saturation_bars(group, output / f"{_slug(dataset)}__{_slug(task)}")
    print(f"Wrote {table.task.nunique()} target(s) across {table.source_dataset.nunique()} dataset(s) to {output}; skipped {len(skipped)} cells")


if __name__ == "__main__":
    main()
