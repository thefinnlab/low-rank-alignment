"""Batch pairwise CCA/RSA comparisons over saved RRR/PCA subspaces."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent:
    if (_ROOT / "src").is_dir():
        break
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import SubspaceComparisonConfig
from src.utils.config_utils import load_config, parse_args
from src.utils.subspace_compare import (
    score_projected_cca_alignment,
    score_projected_rsa_alignment,
    summarize_cca_fold_results,
    summarize_rsa_fold_results,
)
from src.utils.subspace_io import (
    ambient_dim,
    load_base_comparison_data,
    load_fold_bases,
    load_peer_embeddings_on_base_rows,
    max_available_rank,
    resolve_subspace_run,
    run_metadata,
    sanitize_name,
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)


def _available_ranks(requested: list[int], base_dir: Path, peer_dir: Path) -> tuple[list[int], list[int]]:
    max_rank = min(max_available_rank(base_dir), max_available_rank(peer_dir))
    available = [int(rank) for rank in requested if int(rank) <= max_rank]
    skipped = [int(rank) for rank in requested if int(rank) > max_rank]
    return available, skipped


def _component_path(rank: int, fold: int) -> str:
    return f"cca_components_rank{rank}_fold{fold}.npz"


def _rsa_vector_path(rank: int, fold: int) -> str:
    return f"rsa_vectors_rank{rank}_fold{fold}.npz"


def _plot_cca_rank_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].semilogx(summary["rank"], summary["mean_top10_test_corr"], marker="o", label="top 10")
    axes[0].semilogx(summary["rank"], summary["mean_all_test_corr"], marker="o", label="all")
    axes[0].semilogx(summary["rank"], summary["mean_first_test_corr"], marker="o", label="first")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Held-out corr")
    axes[0].set_title("CCA correlations")
    axes[0].legend()

    axes[1].semilogx(summary["rank"], summary["mean_test_shared_dim_pr"], marker="o")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Participation ratio")
    axes[1].set_title("Effective shared dimensionality")

    axes[2].semilogx(summary["rank"], summary["mean_n_test_corr_ge_threshold"], marker="o")
    axes[2].set_xlabel("Rank")
    axes[2].set_ylabel("Components")
    axes[2].set_title("Components above threshold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_rsa_rank_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(summary["rank"], summary["mean_pearson"], marker="o", label="Pearson")
    ax.semilogx(summary["rank"], summary["mean_spearman"], marker="o", label="Spearman")
    ax.set_xlabel("Rank")
    ax.set_ylabel("RSA correlation")
    ax.set_title("RSA over projected word embeddings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_cca_spectrum(per_fold: pd.DataFrame, output_dir: Path) -> None:
    component_paths = per_fold.get("component_path")
    if component_paths is None or per_fold.empty:
        return
    import matplotlib.pyplot as plt

    ranks = sorted(per_fold["rank"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for rank in ranks:
        spectra = []
        for row in per_fold[per_fold["rank"] == rank].itertuples():
            with np.load(output_dir / row.component_path) as data:
                spectra.append(np.asarray(data["test_corrs"], dtype=float))
        if not spectra:
            continue
        max_len = max(len(spec) for spec in spectra)
        padded = np.full((len(spectra), max_len), np.nan)
        for i, spec in enumerate(spectra):
            padded[i, : len(spec)] = spec
        mean = np.nanmean(padded, axis=0)
        ax.plot(np.arange(1, len(mean) + 1), mean, marker="o", linewidth=1.2, label=f"k={rank}")
    ax.set_xlabel("CCA component")
    ax.set_ylabel("Mean held-out corr")
    ax.set_title("Held-out CCA spectrum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "cca_spectrum.png", dpi=150)
    plt.close(fig)


def _plot_base_heatmaps(base_name: str, rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    peers = sorted(df["peer"].unique())
    ranks = sorted(df["rank"].unique())
    metrics = [
        ("mean_all_test_corr", "CCA all components"),
        ("mean_test_shared_dim_pr", "CCA shared PR"),
        ("mean_spearman", "RSA Spearman"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metrics):
        matrix = np.full((len(peers), len(ranks)), np.nan)
        for row in df.itertuples():
            if metric in df.columns:
                matrix[peers.index(row.peer), ranks.index(row.rank)] = getattr(row, metric)
        im = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([str(rank) for rank in ranks], rotation=45, ha="right")
        ax.set_yticks(range(len(peers)))
        ax.set_yticklabels(peers)
        ax.set_xlabel("Rank")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Pairwise subspace comparisons for {base_name}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "pairwise_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_pair(base, peer, config: SubspaceComparisonConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    peer_name = sanitize_name(peer.name)
    output_dir = base.unit_dir / config.output_name / f"vs-{peer_name}"
    cca_summary_path = output_dir / "cca_summary.csv"
    rsa_summary_path = output_dir / "rsa_summary.csv"
    if not config.refresh and cca_summary_path.exists() and rsa_summary_path.exists():
        print(f"Skipping existing comparison {base.name} -> {peer.name}: {output_dir}")
        return pd.read_csv(cca_summary_path), pd.read_csv(rsa_summary_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{base.name} -> {peer.name}]")
    base_data = load_base_comparison_data(base)
    X_peer = load_peer_embeddings_on_base_rows(base, peer, base_data)
    base_ambient = ambient_dim(base.unit_dir)
    peer_ambient = ambient_dim(peer.unit_dir)
    if base_data.X_base.shape[1] != base_ambient:
        raise ValueError(
            f"Base embeddings for {base.name} have {base_data.X_base.shape[1]} columns, "
            f"but {base.unit_dir} A factors have ambient dimension {base_ambient}."
        )
    if X_peer.shape[1] != peer_ambient:
        raise ValueError(
            f"Peer embeddings for {peer.name} on base rows have {X_peer.shape[1]} columns, "
            f"but {peer.unit_dir} A factors have ambient dimension {peer_ambient}."
        )

    cca_ranks, skipped_cca = _available_ranks(config.cca_ranks, base.unit_dir, peer.unit_dir)
    rsa_ranks, skipped_rsa = _available_ranks(config.rsa_ranks, base.unit_dir, peer.unit_dir)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": run_metadata(base),
        "peer": run_metadata(peer),
        "output_dir": str(output_dir),
        "output_name": config.output_name,
        "shared_corr_threshold": float(config.shared_corr_threshold),
        "rsm_metric": config.rsm_metric,
        "requested_cca_ranks": [int(rank) for rank in config.cca_ranks],
        "requested_rsa_ranks": [int(rank) for rank in config.rsa_ranks],
        "available_cca_ranks": cca_ranks,
        "available_rsa_ranks": rsa_ranks,
        "skipped_cca_ranks": skipped_cca,
        "skipped_rsa_ranks": skipped_rsa,
        "n_folds": len(base_data.test_idxs),
        "word_count": base_data.word_count,
        "save_ambient_axes": bool(config.save_ambient_axes),
        "save_rsm_vectors": bool(config.save_rsm_vectors),
    }
    _save_json(output_dir / "comparison_metadata.json", metadata)

    cca_rows = []
    cca_summary_rows = []
    for rank in tqdm(cca_ranks, desc="CCA ranks", unit="rank"):
        base_bases = load_fold_bases(base.unit_dir, rank)
        peer_bases = load_fold_bases(peer.unit_dir, rank)
        fold_results = []
        for fold, test_idx in enumerate(base_data.test_idxs):
            if fold not in base_bases or fold not in peer_bases:
                raise FileNotFoundError(
                    f"Missing fold {fold} A.npy basis for rank {rank} in "
                    f"{base.unit_dir} or {peer.unit_dir}"
                )
            metrics, artifact = score_projected_cca_alignment(
                base_data.X_base,
                X_peer,
                base_data.train_idxs[fold],
                test_idx,
                base_bases[fold],
                peer_bases[fold],
                shared_corr_threshold=float(config.shared_corr_threshold),
                save_ambient_axes=bool(config.save_ambient_axes),
            )
            artifact_rel = _component_path(rank, fold)
            np.savez(output_dir / artifact_rel, **artifact)
            row = {
                "base": base.name,
                "peer": peer.name,
                "rank": int(rank),
                "fold": int(fold),
                "component_path": artifact_rel,
                "base_a_path": str(base.unit_dir / f"fold_{fold}" / "A.npy"),
                "peer_a_path": str(peer.unit_dir / f"fold_{fold}" / "A.npy"),
                **metrics,
            }
            cca_rows.append(row)
            fold_results.append(row)
        cca_summary_rows.append(
            {
                "base": base.name,
                "peer": peer.name,
                "rank": int(rank),
                **summarize_cca_fold_results(fold_results),
            }
        )

    rsa_rows = []
    rsa_summary_rows = []
    for rank in tqdm(rsa_ranks, desc="RSA ranks", unit="rank"):
        base_bases = load_fold_bases(base.unit_dir, rank)
        peer_bases = load_fold_bases(peer.unit_dir, rank)
        fold_results = []
        for fold, test_idx in enumerate(base_data.test_idxs):
            if fold not in base_bases or fold not in peer_bases:
                raise FileNotFoundError(
                    f"Missing fold {fold} A.npy basis for rank {rank} in "
                    f"{base.unit_dir} or {peer.unit_dir}"
                )
            result = score_projected_rsa_alignment(
                base_data.X_base,
                X_peer,
                test_idx,
                base_bases[fold],
                peer_bases[fold],
                metric=config.rsm_metric,
                return_vectors=bool(config.save_rsm_vectors),
            )
            vector_rel = None
            if config.save_rsm_vectors:
                vector_rel = _rsa_vector_path(rank, fold)
                np.savez(
                    output_dir / vector_rel,
                    rsm_base_vector=result.pop("rsm_a_vector"),
                    rsm_peer_vector=result.pop("rsm_b_vector"),
                )
            row = {
                "base": base.name,
                "peer": peer.name,
                "rank": int(rank),
                "fold": int(fold),
                "vector_path": vector_rel,
                "base_a_path": str(base.unit_dir / f"fold_{fold}" / "A.npy"),
                "peer_a_path": str(peer.unit_dir / f"fold_{fold}" / "A.npy"),
                **result,
            }
            rsa_rows.append(row)
            fold_results.append(row)
        rsa_summary_rows.append(
            {
                "base": base.name,
                "peer": peer.name,
                "rank": int(rank),
                **summarize_rsa_fold_results(fold_results),
            }
        )

    cca_per_fold = pd.DataFrame(cca_rows)
    cca_summary = pd.DataFrame(cca_summary_rows)
    rsa_per_fold = pd.DataFrame(rsa_rows)
    rsa_summary = pd.DataFrame(rsa_summary_rows)
    cca_per_fold.to_csv(output_dir / "cca_per_fold.csv", index=False)
    cca_summary.to_csv(cca_summary_path, index=False)
    rsa_per_fold.to_csv(output_dir / "rsa_per_fold.csv", index=False)
    rsa_summary.to_csv(rsa_summary_path, index=False)
    _plot_cca_spectrum(cca_per_fold, output_dir)
    _plot_cca_rank_summary(cca_summary, output_dir / "cca_rank_summary.png")
    _plot_rsa_rank_summary(rsa_summary, output_dir / "rsa_rank_summary.png")
    return cca_summary, rsa_summary


def run_config(config: SubspaceComparisonConfig) -> None:
    runs = [resolve_subspace_run(run) for run in config.runs]
    heatmap_rows_by_base = {run.name: [] for run in runs}
    for base in runs:
        for peer in runs:
            if base.name == peer.name and not config.include_self:
                continue
            cca_summary, rsa_summary = _run_pair(base, peer, config)
            merged = pd.merge(
                cca_summary,
                rsa_summary,
                on=["base", "peer", "rank"],
                how="outer",
            )
            heatmap_rows_by_base[base.name].extend(merged.to_dict("records"))
        base_output_dir = base.unit_dir / config.output_name
        _plot_base_heatmaps(base.name, heatmap_rows_by_base[base.name], base_output_dir)


def main() -> None:
    args, overrides = parse_args("Compare saved RRR/PCA subspaces with CCA and RSA")
    config = load_config(args.config, overrides, SubspaceComparisonConfig)
    run_config(config)


if __name__ == "__main__":
    main()
