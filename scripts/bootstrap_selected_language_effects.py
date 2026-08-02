#!/usr/bin/env python
"""Bootstrap the preregistered, selected RRR language-decoding effects."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fmri.bootstrap_language_decoding import (  # noqa: E402
    _block_lookup,
    _discover_fmri,
    _discover_generic,
    _score,
)
from src.utils.plot_helpers import apply_publication_style  # noqa: E402


DATASETS = ("podcast", "cars-2", "fmri")
CAUSAL_MODELS = ("gpt2", "gpt2-xl-batch", "gpt-j-6b", "llama3-8b")
ROBERTA_MODELS = ("roberta-large",)
STANDARD_MODELS = CAUSAL_MODELS + ROBERTA_MODELS
EXCLUDED_MODELS = ("random-gpt2-xl",)
TASKS = ("sentence_position", "self_entropy", "concreteness")
DATASET_COLORS = {"podcast": "#0072B2", "cars-2": "#D55E00", "fmri": "#009E73"}


def smooth_preference(rrr, pca, tau: float = 0.05):
    """Stable signed RRR preference, applied after hierarchical averaging."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    rrr, pca = np.broadcast_arrays(np.asarray(rrr, float), np.asarray(pca, float))
    result = 2 * (rrr - pca) / (
        np.sqrt(rrr**2 + tau**2) + np.sqrt(pca**2 + tau**2)
    )
    return float(result) if result.ndim == 0 else result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--normalization-tau", type=float, default=0.05)
    parser.add_argument("--output-dir", default="results/language_decoding_selected_bootstrap")
    parser.add_argument("--standard-run-name", default=None,
                        help="Optional run name for sentence position and concreteness")
    parser.add_argument("--entropy-run-name", default="predictability-context-quick-noiu")
    parser.add_argument("--config-dir", default="configs/language_decoding/internal")
    parser.add_argument(
        "--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Worker processes (default: SLURM_CPUS_PER_TASK, otherwise 1)",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _records_by_cell(records: list[dict], task: str, allowed: tuple[str, ...]) -> dict[int, dict[str, list[dict]]]:
    cells: dict[int, dict[str, list[dict]]] = {}
    for record in records:
        if record["task"] == task and record["model"] in allowed:
            cells.setdefault(int(record["rank"]), {}).setdefault(record["model"], []).append(record)
    return cells


def _common_blocks(groups: list[dict[str, list[dict]]]) -> np.ndarray:
    pairs = [record["pair"] for group in groups for records in group.values() for record in records]
    if not pairs:
        return np.asarray([], dtype=str)
    reference = _block_lookup(pairs[0]["block_id"])[0].astype(str)
    reference_set = set(reference)
    for pair in pairs[1:]:
        blocks = set(_block_lookup(pair["block_id"])[0].astype(str))
        if blocks != reference_set:
            raise ValueError("Model/layer prediction artifacts do not share sentence blocks")
    return reference


def _method_score(pair: dict, method: str, sampled_blocks: np.ndarray | None) -> float:
    if sampled_blocks is None:
        rows = np.arange(len(pair["y_true"]))
    else:
        _, lookup = _block_lookup(pair["block_id"])
        rows = np.concatenate([lookup[str(block)] for block in sampled_blocks])
    return _score("continuous", pair["y_true"][rows], pair[method][rows])


def _observed_group(group: dict[str, list[dict]]) -> tuple[float, float]:
    models = sorted(group)
    scores = {
        method: float(np.mean([
            np.mean([_method_score(record["pair"], method, None) for record in group[model]])
            for model in models
        ]))
        for method in ("rrr", "pca")
    }
    return scores["rrr"], scores["pca"]


def _draw_group(
    group: dict[str, list[dict]], sampled_blocks: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    models = np.asarray(sorted(group))
    sampled_models = rng.choice(models, size=len(models), replace=True)
    sampled_units = []
    for model in sampled_models:
        records = group[str(model)]
        sampled_layers = rng.choice(len(records), size=len(records), replace=True)
        sampled_units.append((records, sampled_layers))
    result = {}
    for method in ("rrr", "pca"):
        model_scores = []
        for records, sampled_layers in sampled_units:
            model_scores.append(float(np.mean([
                _method_score(records[int(layer)]["pair"], method, sampled_blocks)
                for layer in sampled_layers
            ])))
        result[method] = float(np.mean(model_scores))
    return result["rrr"], result["pca"]


def bootstrap_groups(
    groups: dict[str, dict[str, list[dict]]], repeats: int,
    rng: np.random.Generator, tau: float = 0.05,
) -> tuple[dict[str, dict], np.ndarray | None]:
    """Bootstrap groups jointly, sharing sentence draws across every group."""
    blocks = _common_blocks(list(groups.values()))
    if not len(blocks):
        raise ValueError("Cannot bootstrap an empty group")
    observed = {}
    boots = {name: np.full(repeats, np.nan) for name in groups}
    for name, group in groups.items():
        rrr, pca = _observed_group(group)
        observed[name] = {"rrr_score": rrr, "pca_score": pca,
                          "smooth_effect": smooth_preference(rrr, pca, tau)}
    for repeat in range(repeats):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        for name, group in groups.items():
            rrr, pca = _draw_group(group, sampled_blocks, rng)
            boots[name][repeat] = smooth_preference(rrr, pca, tau)
    return {name: observed[name] | {"bootstrap": boots[name]} for name in groups}, blocks


def _inference(effect: float, bootstrap: np.ndarray, alpha: float) -> dict:
    finite = bootstrap[np.isfinite(bootstrap)]
    if not len(finite) or not np.isfinite(effect):
        return {"ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    return {
        "ci_low": float(np.quantile(finite, alpha / 2)),
        "ci_high": float(np.quantile(finite, 1 - alpha / 2)),
        "p_value": float((1 + np.count_nonzero(finite <= 0)) / (len(finite) + 1)),
    }


def apply_dataset_rank_fdr(frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """BH-correct all available requested hypotheses within dataset and rank."""
    result = frame.copy()
    result["q_value"] = np.nan
    result["significant"] = False
    for _, indices in result.groupby(["dataset", "rank"]).groups.items():
        indices = list(indices)
        selected = result.loc[indices, "p_value"].dropna().index
        if not len(selected):
            continue
        reject, q_values, _, _ = multipletests(
            result.loc[selected, "p_value"], alpha=alpha, method="fdr_bh"
        )
        result.loc[selected, "q_value"] = q_values
        result.loc[selected, "significant"] = reject & (
            result.loc[selected, "smooth_effect"].to_numpy() > 0
        )
    return result


def _row(dataset: str, rank: int, task: str, hypothesis: str, group: str,
         result: dict, units: dict[str, list[dict]], alpha: float,
         run_name: str | None, config: str) -> dict:
    return {
        "dataset": dataset, "rank": rank, "task": task,
        "hypothesis": hypothesis, "model_group": group,
        "rrr_score": result["rrr_score"], "pca_score": result["pca_score"],
        "smooth_effect": result["smooth_effect"],
        **_inference(result["smooth_effect"], result["bootstrap"], alpha),
        "n_models": len(units), "n_layers": sum(len(value) for value in units.values()),
        "models": ";".join(sorted(units)), "run_name": run_name or "default",
        "config": config,
    }


def _discover(dataset: str, config: str, run_name: str | None) -> list[dict]:
    records, _ = (_discover_fmri(config, run_name) if dataset == "fmri"
                  else _discover_generic(config, run_name))
    return records


def _coverage_entry(dataset: str, task: str, run_name: str | None,
                    cells: dict[int, dict[str, list[dict]]], required_models) -> dict:
    return {
        "dataset": dataset, "task": task, "run_name": run_name or "default",
        "available_ranks": sorted(cells),
        "available_models_by_rank": {str(rank): sorted(group) for rank, group in cells.items()},
        "required_model_group": list(required_models),
        "status": "available" if cells else "missing_heldout_predictions",
    }


def _standard_job(payload: tuple) -> list[dict]:
    dataset, rank, task, group, repeats, tau, alpha, seed, run_name, config = payload
    result, _ = bootstrap_groups(
        {"all": group}, repeats, np.random.default_rng(seed), tau,
    )
    return [_row(
        dataset, rank, task, f"{task}:rrr_gt_pca", "all_real_models",
        result["all"], group, alpha, run_name, config,
    )]


def _entropy_job(payload: tuple) -> list[dict]:
    dataset, rank, all_models, repeats, tau, alpha, seed, run_name, config = payload
    causal = {model: records for model, records in all_models.items() if model in CAUSAL_MODELS}
    roberta = {model: records for model, records in all_models.items() if model in ROBERTA_MODELS}
    result, _ = bootstrap_groups(
        {"causal": causal, "roberta": roberta}, repeats,
        np.random.default_rng(seed), tau,
    )
    rows = [
        _row(
            dataset, rank, "self_entropy", f"self_entropy:{name}:rrr_gt_pca", name,
            result[name], group, alpha, run_name, config,
        )
        for name, group in (("causal", causal), ("roberta", roberta))
    ]
    contrast_boot = result["causal"]["bootstrap"] - result["roberta"]["bootstrap"]
    contrast = result["causal"]["smooth_effect"] - result["roberta"]["smooth_effect"]
    rows.append({
        "dataset": dataset, "rank": rank, "task": "self_entropy",
        "hypothesis": "self_entropy:causal_gt_roberta", "model_group": "causal_minus_roberta",
        "rrr_score": np.nan, "pca_score": np.nan, "smooth_effect": contrast,
        **_inference(contrast, contrast_boot, alpha),
        "n_models": len(causal) + len(roberta),
        "n_layers": sum(map(len, causal.values())) + sum(map(len, roberta.values())),
        "models": ";".join(sorted(causal | roberta)),
        "run_name": run_name, "config": config,
    })
    return rows


def _run_jobs(payloads: list[tuple], jobs: int) -> list[dict]:
    """Run independent rank cells and print elapsed-time/ETA progress."""
    total = len(payloads)
    if not total:
        return []
    started = time.monotonic()
    print(f"[bootstrap] starting {total} dataset/task/rank cells with {jobs} worker(s)",
          file=sys.stderr, flush=True)
    results = []

    def consume(iterator) -> None:
        for completed, cell_rows in enumerate(iterator, start=1):
            results.extend(cell_rows)
            elapsed = time.monotonic() - started
            eta = elapsed / completed * (total - completed)
            print(
                f"[bootstrap] {completed}/{total} cells ({completed / total:.0%}); "
                f"elapsed {elapsed:.1f}s; ETA {eta:.1f}s",
                file=sys.stderr, flush=True,
            )

    if jobs == 1:
        consume(function(payload) for function, payload in payloads)
    else:
        # executor.map preserves payload order, so results remain deterministic.
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(function, payload) for function, payload in payloads]
            consume(future.result() for future in futures)
    return results


def run_analysis(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    coverage = []
    seed_sequence = np.random.SeedSequence(args.seed)
    seeds = iter(seed_sequence.spawn(10000))
    discovered = {}
    for dataset in DATASETS:
        config = str(Path(args.config_dir) / f"{dataset}.yml")
        standard = _discover(dataset, config, args.standard_run_name)
        entropy = _discover(dataset, config, args.entropy_run_name)
        task_cells = {
            "sentence_position": _records_by_cell(standard, "sentence_position", STANDARD_MODELS),
            "self_entropy": _records_by_cell(entropy, "self_entropy", STANDARD_MODELS),
            "concreteness": (_records_by_cell(standard, "concreteness", STANDARD_MODELS)
                             if dataset != "cars-2" else {}),
        }
        entropy_testable = {
            rank: group for rank, group in task_cells["self_entropy"].items()
            if any(model in CAUSAL_MODELS for model in group)
            and any(model in ROBERTA_MODELS for model in group)
        }
        task_cells["self_entropy"] = entropy_testable
        discovered[dataset] = (config, task_cells)
        for task in TASKS:
            required = STANDARD_MODELS if task != "self_entropy" else CAUSAL_MODELS + ROBERTA_MODELS
            entry = _coverage_entry(
                dataset, task, args.entropy_run_name if task == "self_entropy" else args.standard_run_name,
                task_cells[task], required,
            )
            if dataset == "cars-2" and task == "concreteness":
                entry["status"] = "unavailable_by_design"
            coverage.append(entry)
    missing = [item for item in coverage if item["status"] == "missing_heldout_predictions"]
    if missing:
        report = {"coverage": coverage, "missing_required": missing}
        raise RuntimeError("Required held-out prediction artifacts are absent. Coverage report:\n" +
                           json.dumps(report, indent=2))

    payloads = []
    for dataset in DATASETS:
        config, task_cells = discovered[dataset]
        for task in ("sentence_position", "concreteness"):
            for rank, group in sorted(task_cells[task].items()):
                seed = int(next(seeds).generate_state(1)[0])
                payloads.append((_standard_job, (
                    dataset, rank, task, group, args.bootstrap_repeats,
                    args.normalization_tau, args.alpha, seed,
                    args.standard_run_name, config,
                )))

        for rank, all_models in sorted(task_cells["self_entropy"].items()):
            seed = int(next(seeds).generate_state(1)[0])
            payloads.append((_entropy_job, (
                dataset, rank, all_models, args.bootstrap_repeats,
                args.normalization_tau, args.alpha, seed,
                args.entropy_run_name, config,
            )))
    rows = _run_jobs(payloads, args.jobs)
    return apply_dataset_rank_fdr(pd.DataFrame(rows), args.alpha), {"coverage": coverage}


def _save_figure(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.05)
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight", pad_inches=.05)
    plt.close(fig)


def _plot(frame: pd.DataFrame, hypotheses: list[str], title: str, destination: Path,
          group_styles: dict[str, str] | None = None) -> None:
    apply_publication_style()
    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    selected = frame.loc[frame["hypothesis"].isin(hypotheses)]
    for (dataset, group), values in selected.groupby(["dataset", "model_group"]):
        values = values.sort_values("rank")
        ranks = values["rank"].to_numpy(float)
        effects = values["smooth_effect"].to_numpy(float)
        axis.semilogx(ranks, effects, color=DATASET_COLORS[dataset],
                      linestyle=(group_styles or {}).get(group, "-"), marker="o",
                      markersize=3.2, linewidth=1.6,
                      label=f"{dataset} · {group}" if len(set(selected.model_group)) > 1 else dataset)
        axis.fill_between(ranks, values["ci_low"].to_numpy(float),
                          values["ci_high"].to_numpy(float), color=DATASET_COLORS[dataset],
                          alpha=.16, linewidth=0)
        significant = values["significant"].astype(bool).to_numpy()
        axis.scatter(ranks[significant], effects[significant], marker="*", s=45,
                     color=DATASET_COLORS[dataset], edgecolor="black", linewidth=.3, zorder=8)
    axis.axhline(0, color="black", linewidth=.8, alpha=.65)
    axis.set_xlabel("Rank $k$ (log scale)")
    axis.set_ylabel("Smooth RRR preference")
    axis.set_title(title)
    axis.legend(loc="best", fontsize=7)
    _save_figure(fig, destination)


def write_plots(frame: pd.DataFrame, output: Path) -> None:
    _plot(frame, ["sentence_position:rrr_gt_pca"], "Sentence position", output / "sentence_position")
    _plot(frame, ["self_entropy:causal:rrr_gt_pca", "self_entropy:roberta:rrr_gt_pca"],
          "Self-entropy", output / "self_entropy_groups", {"causal": "-", "roberta": "--"})
    _plot(frame, ["self_entropy:causal_gt_roberta"], "Self-entropy: causal minus RoBERTa",
          output / "self_entropy_causal_minus_roberta")
    _plot(frame, ["concreteness:rrr_gt_pca"], "Concreteness", output / "concreteness")


def main() -> None:
    args = _args()
    for value in (args.standard_run_name, args.entropy_run_name):
        if value and not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError("Run names may contain only letters, numbers, '.', '_', and '-'")
    if (args.bootstrap_repeats < 1 or args.jobs < 1
            or not 0 < args.alpha < 1 or args.normalization_tau <= 0):
        raise ValueError("repeats and tau must be positive, and alpha must be between zero and one")
    frame, report = run_analysis(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "selected_language_effects_by_rank.csv", index=False)
    if not args.no_plots:
        write_plots(frame, output / "plots")
    metadata = report | {
        "task_run_names": {"sentence_position": args.standard_run_name or "default",
                           "concreteness": args.standard_run_name or "default",
                           "self_entropy": args.entropy_run_name},
        "model_groups": {"standard": list(STANDARD_MODELS), "causal": list(CAUSAL_MODELS),
                         "roberta": list(ROBERTA_MODELS), "excluded": list(EXCLUDED_MODELS)},
        "bootstrap_repeats": args.bootstrap_repeats, "seed": args.seed, "alpha": args.alpha,
        "jobs": args.jobs,
        "bootstrap": "paired sentence blocks; equal model weights; layers nested within model",
        "smooth_formula": "2*(RRR-PCA)/(sqrt(RRR^2+tau^2)+sqrt(PCA^2+tau^2))",
        "normalization_tau": args.normalization_tau,
        "alternative": "one-sided positive; entropy contrast causal greater than RoBERTa",
        "fdr_family": "BH across all available requested hypotheses within dataset/rank",
        "available_model_policy": "retain available models and layers independently at each rank",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(frame)} rank-level tests to {output}")


if __name__ == "__main__":
    main()
