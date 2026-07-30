"""Summarize linguistic decoding gaps at each source's RRR saturation rank."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.saturation_bar_summary import MODEL_COLORS, MODEL_LABELS, normalize_model


MODEL_DEPTHS = {
    "gpt2": 12,
    "gpt2-xl": 48,
    "gpt-j-6b": 28,
    "llama3-8b": 32,
    "roberta-large": 24,
    "random-gpt2-xl": 48,
}


def _resolved(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def build_decoding_saturation_table(
    paired: pd.DataFrame,
    manifest: pd.DataFrame,
    rank_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join scalar decoding gaps to the exact RRR neural saturation rank."""
    sources = manifest[["source_id", "rrr_dir"]].drop_duplicates().copy()
    saturation = rank_summary.loc[
        rank_summary.method.eq("rrr"), ["run", "saturation_rank"]
    ].copy()
    saturation["run_key"] = saturation.run.map(_resolved)
    if saturation.run_key.duplicated().any():
        raise ValueError("Rank summary has duplicate RRR rows for the same run")
    rank_by_run = saturation.set_index("run_key").saturation_rank.to_dict()

    def saturation_for_source(value: str) -> float:
        unit = Path(value).expanduser().resolve()
        # iEEG summaries name the factor unit itself; rank-specific fMRI factors
        # live in an ``rrr`` child while rank saturation names its parent run.
        for candidate in (unit, unit.parent if unit.name == "rrr" else unit):
            if str(candidate) in rank_by_run:
                return float(rank_by_run[str(candidate)])
        return np.nan

    sources["saturation_rank"] = sources.rrr_dir.map(saturation_for_source)

    scalar = paired.loc[paired.task.ne("glove")].copy()
    joined = scalar.merge(
        sources[["source_id", "saturation_rank"]], on="source_id", how="left", validate="many_to_one"
    )
    matched = joined.loc[joined.k.eq(joined.saturation_rank)].copy()
    keys = [
        "source_id", "source_dataset", "architecture", "model_dir", "layer",
        "control", "task", "kind", "saturation_rank",
    ]
    table = matched.groupby(keys, dropna=False, as_index=False).agg(
        rrr_decoding=("rrr", "mean"),
        pca_decoding=("pca", "mean"),
        rrr_minus_pca=("rrr_minus_pca", "mean"),
        sem=("rrr_minus_pca", "sem"),
        n_folds=("fold", "nunique"),
    )
    if not table.empty:
        table["model"] = table.model_dir.map(normalize_model)
        unknown = sorted(set(table.model).difference(MODEL_DEPTHS))
        if unknown:
            raise ValueError(f"No full layer depth registered for models {unknown}")
        table["depth_percent"] = table.apply(
            lambda row: 100.0 * row.layer / MODEL_DEPTHS[row.model], axis=1
        )

    available = scalar[["source_id", "source_dataset", "model_dir", "layer", "task"]].drop_duplicates()
    requested = available.merge(
        sources[["source_id", "saturation_rank"]], on="source_id", how="left", validate="many_to_one"
    )
    present = scalar[["source_id", "task", "k"]].drop_duplicates()
    audit = requested.merge(
        present,
        left_on=["source_id", "task", "saturation_rank"],
        right_on=["source_id", "task", "k"],
        how="left",
        indicator=True,
    )
    skipped = audit.loc[audit._merge.ne("both")].drop(columns=["k", "_merge"])
    skipped["reason"] = np.where(
        skipped.saturation_rank.isna(),
        "no matching RRR run in rank summary",
        "RRR saturation rank was not decoded",
    )
    return table, skipped


def plot_decoding_saturation_bars(frame: pd.DataFrame, output: Path) -> None:
    """Plot depth-grouped model bars for one dataset and linguistic target."""
    frame = frame.sort_values(["depth_percent", "model"])
    depths = sorted(frame.depth_percent.unique())
    models = sorted(frame.model.unique())
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(depths), dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * len(depths)), 4.8))
    for index, model in enumerate(models):
        part = frame.loc[frame.model.eq(model)].set_index("depth_percent")
        values = [part.rrr_minus_pca.get(depth, np.nan) for depth in depths]
        errors = [part["sem"].get(depth, np.nan) for depth in depths]
        positions = x + (index - (len(models) - 1) / 2) * width
        ax.bar(
            positions, values, width=width, yerr=errors,
            color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model),
            capsize=2,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xticks=x,
        xticklabels=[f"{depth:g}%" for depth in depths],
        xlabel="Layer depth (% of full model depth)",
        ylabel="RRR decoding − PCA decoding",
        title=str(frame.task.iloc[0]),
    )
    ax.legend(title="Model", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)
