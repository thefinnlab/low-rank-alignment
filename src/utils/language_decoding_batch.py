"""Batch orchestration for Podcast probes of saved neural subspaces."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.core.config import LanguageDecodingConfig, LanguageDecodingMode
from src.utils.config_utils import save_effective_config
from src.utils.embeddings import load_embeddings
from src.utils.language_decoding import run_language_decoding, source_slug


@dataclass(frozen=True)
class DecodingSource:
    configured_path: str
    root: Path
    rrr_dir: Path
    pca_dir: Path
    factor_format: str
    source_dataset: str
    model_dir: str
    layer: int
    architecture: str
    control: bool

    @property
    def source_id(self) -> str:
        return source_slug(self.rrr_dir, f"{self.source_dataset}_{self.model_dir}_layer{self.layer}")


def _factor_format(path: Path) -> str | None:
    if list(path.glob("fold_*/A.npy")):
        return "folds"
    if list(path.glob("k*/A.npy")):
        return "rank_bases"
    return None


def _mirror_pca(rrr: Path) -> Path:
    value = str(rrr)
    for left, right in (
        ("multi_view_rrr", "multi_view_pca"),
        ("cross_dataset_rrr", "cross_dataset_pca"),
        ("per_subject_rrr", "per_subject_pca"),
        ("per_region_rrr", "per_region_pca"),
        ("/rrr", "/pca"),
    ):
        if left in value:
            return Path(value.replace(left, right, 1))
    return Path(value + "_pca")


def _candidate_rrr_units(root: Path) -> list[Path]:
    candidates = []
    for candidate in (root, root / "rrr"):
        if _factor_format(candidate):
            candidates.append(candidate)
    if not candidates and root.exists():
        candidates.extend(
            sorted(
                {
                    path.parent.parent
                    for path in root.rglob("A.npy")
                    if path.parent.name.startswith("fold_")
                    and "pca" not in str(path.parent.parent).lower()
                }
            )
        )
        candidates.extend(
            sorted(
                {
                    path.parent.parent
                    for path in root.rglob("A.npy")
                    if path.parent.name.startswith("k")
                    and path.parent.name[1:].isdigit()
                    and path.parent.parent.name == "rrr"
                }
            )
        )
    return sorted(set(candidates))


def _load_source_config(root: Path, rrr: Path) -> dict:
    candidates = [rrr / "config.yml", root / "config.yml"]
    for ancestor in root.parents:
        candidates.extend([ancestor / "rrr" / "config.yml", ancestor / "config.yml"])
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            value = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            if isinstance(value, dict):
                return value
    raise FileNotFoundError(f"No saved config.yml found for source {root}")


def discover_source(path: str, ranks: list[int]) -> DecodingSource:
    """Resolve and validate one path-only batch source."""
    configured = Path(path).expanduser().resolve()
    if not configured.exists():
        raise FileNotFoundError(f"Result path does not exist: {configured}")
    units = _candidate_rrr_units(configured)
    if not units:
        raise FileNotFoundError(f"No RRR factor unit found under {configured}")
    if len(units) != 1:
        raise ValueError(
            f"Expected exactly one RRR unit under {configured}; found {units}"
        )
    rrr = units[0]
    root = rrr.parent if rrr.name == "rrr" else configured
    pca = _mirror_pca(rrr)
    rrr_format = _factor_format(rrr)
    if not pca.exists() or _factor_format(pca) != rrr_format:
        raise FileNotFoundError(
            f"Missing matching {rrr_format} PCA factors for {rrr}; expected {pca}"
        )
    saved = _load_source_config(root, rrr)
    encoding = saved.get("encoding", {})
    per_model = encoding.get("per_model_config", {})
    model_dir = per_model.get("model_dir", saved.get("model_dir"))
    layer = per_model.get("layer", saved.get("layer"))
    path_text = str(configured)
    layer_match = re.search(r"_layer(\d+)(?:_|/|$)", path_text)
    if isinstance(layer, list):
        inferred_layer = int(layer_match.group(1)) if layer_match else None
        layer = inferred_layer if inferred_layer in set(map(int, layer)) else None
    elif layer is None and layer_match:
        layer = int(layer_match.group(1))
    if not model_dir:
        model_match = re.search(r"/([^/]+)_layer\d+(?:_|/)", path_text)
        model_dir = model_match.group(1) if model_match else None
    if not model_dir or layer is None:
        raise ValueError(
            f"Saved config for {configured} must identify one model_dir and scalar layer"
        )
    if "fmri" in saved or "fmri" in {part.lower() for part in configured.parts}:
        dataset = "fmri"
    elif "movie-cars-2" in configured.parts or encoding.get("dataset") == "brain_treebank":
        movie = (
            "cars-2"
            if "movie-cars-2" in configured.parts
            else encoding.get("per_dataset_config", {}).get("movie")
        )
        if movie != "cars-2":
            raise ValueError(f"Only Cars 2 Brain Treebank sources are supported, got {movie!r}")
        dataset = "cars-2"
    else:
        dataset = "podcast"
    if dataset == "fmri" and rrr_format != "rank_bases":
        raise ValueError("Within-fMRI sources must contain rank-specific k*/A.npy bases")
    if dataset != "fmri" and rrr_format != "folds":
        raise ValueError(f"{dataset} sources must contain fold_*/A.npy factors")
    for kind, unit in (("RRR", rrr), ("PCA", pca)):
        if rrr_format == "rank_bases":
            missing = [rank for rank in ranks if not (unit / f"k{rank}" / "A.npy").exists()]
            if missing:
                raise ValueError(f"{kind} source {unit} is missing ranks {missing}")
        else:
            paths = sorted(unit.glob("fold_*/A.npy"))
            widths = [np.load(item, mmap_mode="r").shape[1] for item in paths]
            if not paths or min(widths) < max(ranks):
                raise ValueError(f"{kind} source {unit} cannot supply ranks {ranks}")
    control = bool(saved.get("random_init_model", False)) or "random" in str(model_dir).lower()
    return DecodingSource(
        configured_path=path,
        root=root,
        rrr_dir=rrr,
        pca_dir=pca,
        factor_format=str(rrr_format),
        source_dataset=dataset,
        model_dir=str(model_dir),
        layer=int(layer),
        architecture=str(model_dir),
        control=control,
    )


def _discover_sources_with_failures(
    config: LanguageDecodingConfig,
) -> tuple[list[DecodingSource], list[dict]]:
    sources = []
    failures = []
    for path in config.result_paths:
        try:
            source = discover_source(path, config.rank_grid)
        except FileNotFoundError as exc:
            if not config.skip_missing_sources:
                raise
            failure = {"configured_path": path, "error": str(exc)}
            failures.append(failure)
            print(f"[missing source] {path}: {exc}", flush=True)
            continue
        if source.rrr_dir.resolve() in {item.rrr_dir.resolve() for item in sources}:
            raise ValueError(f"Duplicate RRR unit resolved from {path}: {source.rrr_dir}")
        sources.append(source)
    embedding_dims = {}
    valid_sources = []
    for source in sources:
        model_key = (source.model_dir, source.layer)
        try:
            if model_key not in embedding_dims:
                embeddings, _ = load_embeddings(
                    config.encoding.data_root,
                    "transformer",
                    dataset="podcast",
                    per_dataset_config={},
                    model_dir=source.model_dir,
                    layer=source.layer,
                )
                embedding_dims[model_key] = int(embeddings.shape[1])
            expected = embedding_dims[model_key]
            for kind, unit in (("RRR", source.rrr_dir), ("PCA", source.pca_dir)):
                paths = sorted(unit.glob("fold_*/A.npy")) or sorted(unit.glob("k*/A.npy"))
                ambient = {int(np.load(path, mmap_mode="r").shape[0]) for path in paths}
                if ambient != {expected}:
                    raise ValueError(
                        f"{kind} ambient dimensions {sorted(ambient)} for {unit} do not "
                        f"match Podcast {source.model_dir} layer {source.layer} dimension {expected}"
                    )
        except FileNotFoundError as exc:
            if not config.skip_missing_sources:
                raise
            failure = {"configured_path": source.configured_path, "error": str(exc)}
            failures.append(failure)
            print(f"[missing source] {source.configured_path}: {exc}", flush=True)
            continue
        valid_sources.append(source)
    return valid_sources, failures


def discover_sources(config: LanguageDecodingConfig) -> list[DecodingSource]:
    """Return valid sources, printing configured missing sources when allowed."""
    return _discover_sources_with_failures(config)[0]


def _paired_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_id", "source_dataset", "architecture", "model_dir", "layer",
        "control", "task", "kind", "offset", "k", "fold",
    ]
    ranked = frame[frame.subspace.isin(["rrr", "pca"])].pivot_table(
        index=keys, columns="subspace", values="primary_score"
    ).dropna().reset_index()
    full = frame[frame.subspace == "full"][
        ["source_id", "task", "offset", "fold", "primary_score"]
    ].rename(columns={"primary_score": "full_score"})
    paired = ranked.merge(
        full, on=["source_id", "task", "offset", "fold"], validate="many_to_one"
    )
    paired["rrr_minus_pca"] = paired.rrr - paired.pca
    denominator = paired.full_score
    paired["normalized_retention_gap"] = np.where(
        paired.kind.isin(["continuous", "glove"]) & (np.abs(denominator) > 1e-12),
        paired.rrr_minus_pca / denominator,
        np.nan,
    )
    return paired


def _hierarchical_draw(group: pd.DataFrame, rng: np.random.Generator) -> float:
    architectures = group.architecture.unique()
    values = []
    for architecture in rng.choice(architectures, len(architectures), replace=True):
        architecture_rows = group[group.architecture == architecture]
        paths = architecture_rows.source_id.unique()
        path_values = []
        for source_id in rng.choice(paths, len(paths), replace=True):
            fold_values = architecture_rows[
                architecture_rows.source_id == source_id
            ].rrr_minus_pca.to_numpy(dtype=float)
            path_values.append(float(rng.choice(fold_values, len(fold_values), replace=True).mean()))
        values.append(float(np.mean(path_values)))
    return float(np.mean(values))


def hierarchical_bootstrap(
    paired: pd.DataFrame, *, draws: int, seed: int, confidence_level: float
) -> pd.DataFrame:
    """Bootstrap architecture -> result path/layer -> fold within each dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    group_keys = ["source_dataset", "control", "task", "kind", "offset", "k"]
    alpha = (1.0 - confidence_level) / 2.0
    for labels, group in paired.groupby(group_keys, dropna=False, sort=False):
        by_path = group.groupby(["architecture", "source_id"]).rrr_minus_pca.mean()
        point = float(by_path.groupby(level=0).mean().mean())
        samples = np.asarray([_hierarchical_draw(group, rng) for _ in range(draws)])
        rows.append(
            {
                **dict(zip(group_keys, labels)),
                "mean_rrr_minus_pca": point,
                "ci_lower": float(np.quantile(samples, alpha)),
                "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
                "fraction_above_zero": float(np.mean(samples > 0)),
                "n_architectures": int(group.architecture.nunique()),
                "n_paths": int(group.source_id.nunique()),
                "n_folds": int(group.fold.nunique()),
                "bootstrap_draws": int(draws),
            }
        )
    return pd.DataFrame(rows)


def _plot_batch_summary(summary: pd.DataFrame, output: Path) -> None:
    """Write compact dataset-separated scalar curves and GloVe heatmaps."""
    if summary.empty:
        return
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    trained = summary[~summary.control.astype(bool)]
    for task in trained.loc[trained.task != "glove", "task"].unique():
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for dataset, part in trained[trained.task == task].groupby("source_dataset"):
            part = part.sort_values("k")
            ax.plot(part.k, part.mean_rrr_minus_pca, marker="o", label=dataset)
            ax.fill_between(part.k, part.ci_lower, part.ci_upper, alpha=0.18)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xscale="log", xlabel="Subspace rank", ylabel="RRR − PCA", title=task)
        ax.legend(title="Source dataset")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task}_dataset_comparison.png", dpi=180)
        plt.close(fig)
    glove = trained[trained.task == "glove"]
    for dataset, part in glove.groupby("source_dataset"):
        table = part.pivot(index="offset", columns="k", values="mean_rrr_minus_pca").sort_index()
        if table.empty:
            continue
        limit = float(np.nanmax(np.abs(table.to_numpy()))) or 1.0
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        image = ax.imshow(table, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(
            xticks=np.arange(len(table.columns)),
            xticklabels=[str(int(rank)) for rank in table.columns],
            yticks=np.arange(len(table.index)),
            yticklabels=[f"{int(offset):+d}" for offset in table.index],
            xlabel="Subspace rank",
            ylabel="Target-word offset",
            title=f"{dataset}: GloVe RRR − PCA",
        )
        fig.colorbar(image, ax=ax, label="Mean cosine gap")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{dataset}_glove_offsets.png", dpi=180)
        plt.close(fig)


def run_language_decoding_batch(config: LanguageDecodingConfig) -> Path:
    """Preflight every source, run independent probes, and aggregate their results."""
    output = Path(config.batch_output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, output)
    sources, missing = _discover_sources_with_failures(config)
    pd.DataFrame(missing, columns=["configured_path", "error"]).to_csv(
        output / "missing_sources.csv", index=False
    )
    if not sources:
        (output / "REPORT.md").write_text(
            "# Batch RRR/PCA language decoding\n\n"
            "No valid sources were available; see `missing_sources.csv`.\n",
            encoding="utf-8",
        )
        return output
    manifest = pd.DataFrame(
        [{**asdict(source), "root": str(source.root), "rrr_dir": str(source.rrr_dir),
          "pca_dir": str(source.pca_dir), "source_id": source.source_id}
         for source in sources]
    )
    manifest.to_csv(output / "source_manifest.csv", index=False)
    all_results = []
    for source in sources:
        child = deepcopy(config)
        child.result_paths = []
        child.batch_output_dir = None
        child.rrr_run_path = str(source.rrr_dir)
        child.pca_run_path = str(source.pca_dir)
        child.source_name = source.source_id
        child.output_dir = str(output / "runs" / source.source_id)
        child.mode = (
            LanguageDecodingMode.WITHIN_DATASET
            if source.source_dataset == "podcast"
            else LanguageDecodingMode.CROSS_DATASET
        )
        child.encoding.dataset = "podcast"
        child.encoding.subjects = None
        child.encoding.per_dataset_config = {}
        child.encoding.model = "transformer"
        child.encoding.per_model_config = {
            "model_dir": source.model_dir, "layer": source.layer
        }
        child.validate()
        run_dir = run_language_decoding(child)
        scalar = pd.read_csv(run_dir / "grid_results.csv").assign(offset=0)
        frames = [scalar]
        glove_path = run_dir / "glove_grid_results.csv"
        if glove_path.exists():
            frames.append(pd.read_csv(glove_path))
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.assign(
            source_id=source.source_id,
            source_dataset=source.source_dataset,
            architecture=source.architecture,
            model_dir=source.model_dir,
            layer=source.layer,
            control=source.control,
            run_output_dir=str(run_dir),
        )
        all_results.append(frame)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output / "combined_fold_results.csv", index=False)
    paired = _paired_gaps(combined)
    paired.to_csv(output / "paired_rrr_pca_gaps.csv", index=False)
    descriptive = paired.groupby(
        ["source_dataset", "control", "architecture", "task", "kind", "offset", "k"],
        dropna=False,
    ).rrr_minus_pca.agg(["mean", "sem", "count"]).reset_index()
    descriptive.to_csv(output / "descriptive_summary.csv", index=False)
    bootstrap = pd.DataFrame()
    if config.bootstrap.enabled:
        bootstrap = hierarchical_bootstrap(
            paired,
            draws=config.bootstrap.draws,
            seed=config.bootstrap.seed,
            confidence_level=config.bootstrap.confidence_level,
        )
        bootstrap.to_csv(output / "hierarchical_bootstrap.csv", index=False)
        _plot_batch_summary(bootstrap, output)
    report = [
        "# Batch RRR/PCA language decoding", "",
        f"Valid sources: {len(sources)}", f"Skipped missing sources: {len(missing)}",
        f"Successful fold cells: {len(combined)}", "",
        "Cars 2 and fMRI rows are cross-stimulus Podcast probes; intervals do not "
        "provide subject-level neural-population inference.", "",
    ]
    if not bootstrap.empty:
        report.extend(["## Dataset-level RRR−PCA estimates", ""])
        for row in bootstrap.itertuples():
            offset = "" if row.task != "glove" else f", offset {int(row.offset):+d}"
            population = "control" if bool(row.control) else "trained"
            report.append(
                f"- {row.source_dataset} ({population}), {row.task}{offset}, "
                f"k={int(row.k)}: "
                f"{row.mean_rrr_minus_pca:.4f} "
                f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}]"
            )
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "run_metadata.json").write_text(
        json.dumps({"probe_dataset": "podcast", "source_count": len(sources)}, indent=2)
    )
    return output
