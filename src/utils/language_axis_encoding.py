"""Neural encoding from interpretable language-decoder axes and residuals."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.core.config import (
    LanguageAxisEncodingConfig,
    LanguageDecodingConfig,
    LanguageDecodingMode,
    get_results_dir,
)
from src.core.registry import resolve_dataset_context
from src.utils.alpha_grids import resolve_ridge_alphas
from src.utils.config_utils import load_config, save_effective_config
from src.utils.constants import RIDGE_ALPHAS
from src.utils.data_loaders import load_sig_electrodes, selection_after_offset
from src.utils.decomposition import _get_decomposition_inputs
from src.utils.embeddings import load_embeddings
from src.utils.language_decoding import resolve_subspace_unit_dir
from src.utils.language_reconstruction import build_preprocessor, load_task_matrix
from src.utils.language_tasks import PodcastTaskContext
from src.utils.subspace_compare import aggregate_subspace_basis
from src.utils.subspace_utils import evaluate_input_space


def orthonormal_span(weights: np.ndarray) -> np.ndarray:
    """Return a deterministic orthonormal basis for decoder weight columns."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.ndim != 2 or weights.shape[1] == 0:
        raise ValueError("Decoder weights must be a non-empty matrix")
    left, singular, _ = np.linalg.svd(weights, full_matrices=False)
    tolerance = np.finfo(float).eps * max(weights.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank == 0:
        raise ValueError("Decoder axes have numerical rank zero")
    basis = left[:, :rank]
    for column in range(rank):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return basis


def covariance_transforms(z_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return symmetric train-covariance square root and inverse square root."""
    centered = np.asarray(z_train, dtype=np.float64) - np.mean(z_train, axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    values, vectors = np.linalg.eigh(covariance)
    tolerance = np.finfo(float).eps * max(covariance.shape) * max(values[-1], 1.0)
    if np.any(values <= tolerance):
        raise ValueError(
            "Training rank representation has rank-deficient covariance; "
            "cannot construct an invertible whitening transform"
        )
    square_root = (vectors * np.sqrt(values)) @ vectors.T
    inverse_square_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    return square_root, inverse_square_root


def decompose_task_span(
    z_train: np.ndarray,
    z_test: np.ndarray,
    weights: np.ndarray,
    geometry: str,
) -> dict[str, np.ndarray | int | float]:
    """Split a rank representation into task and residual components.

    Whitening determines the projection metric only. Both returned components are
    mapped back into the original rank coordinates and therefore sum exactly to the
    train-centered full representation.
    """
    z_train = np.asarray(z_train, dtype=np.float64)
    z_test = np.asarray(z_test, dtype=np.float64)
    mean = np.mean(z_train, axis=0)
    train_centered = z_train - mean
    test_centered = z_test - mean
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.shape[0] != z_train.shape[1]:
        raise ValueError("Decoder weight dimension does not match rank representation")

    if geometry == "raw":
        basis = orthonormal_span(weights)
        projector = basis @ basis.T
        task_train = train_centered @ projector
        task_test = test_centered @ projector
    elif geometry == "whitened":
        square_root, inverse_square_root = covariance_transforms(z_train)
        basis = orthonormal_span(square_root @ weights)
        projector = basis @ basis.T
        task_train = train_centered @ inverse_square_root @ projector @ square_root
        task_test = test_centered @ inverse_square_root @ projector @ square_root
    else:
        raise ValueError("geometry must be 'raw' or 'whitened'")

    residual_train = train_centered - task_train
    residual_test = test_centered - task_test
    variance_fraction = float(
        np.sum(task_train**2) / np.sum(train_centered**2)
    )
    return {
        "full_train": train_centered.astype(np.float32),
        "full_test": test_centered.astype(np.float32),
        "task_train": task_train.astype(np.float32),
        "task_test": task_test.astype(np.float32),
        "residual_train": residual_train.astype(np.float32),
        "residual_test": residual_test.astype(np.float32),
        "basis": basis.astype(np.float32),
        "effective_rank": int(basis.shape[1]),
        "variance_fraction": variance_fraction,
    }


def random_span_weights(rank: int, span_rank: int, seed: int) -> np.ndarray:
    """Draw one deterministic Haar-oriented span in rank coordinates."""
    if span_rank < 1 or span_rank > rank:
        raise ValueError("span_rank must lie between one and rank")
    rng = np.random.default_rng(seed)
    return orthonormal_span(rng.normal(size=(rank, span_rank)))


def task_axis_overlaps(
    z_train: np.ndarray, task_weights: dict[str, np.ndarray], geometry: str
) -> list[dict]:
    """Return pairwise average squared canonical cosine between task spans."""
    transform = np.eye(z_train.shape[1])
    if geometry == "whitened":
        transform, _ = covariance_transforms(z_train)
    bases = {
        task: orthonormal_span(transform @ weights)
        for task, weights in task_weights.items()
    }
    rows = []
    names = list(bases)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            singular = np.linalg.svd(bases[left].T @ bases[right], compute_uv=False)
            rows.append(
                {
                    "task_left": left,
                    "task_right": right,
                    "mean_squared_canonical_cosine": float(np.mean(singular**2)),
                    "maximum_canonical_cosine": float(np.max(singular)),
                }
            )
    return rows


def _configure_device(requested: str) -> str:
    import torch
    from himalaya.backend import set_backend

    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable")
    use_cuda = requested == "cuda" or (requested == "auto" and available)
    set_backend("torch_cuda" if use_cuda else "numpy")
    return "cuda" if use_cuda else "cpu"


def _decoder_folds(
    saved: LanguageDecodingConfig, n_rows: int, within_test_idxs: list[np.ndarray]
) -> list[np.ndarray]:
    if saved.mode == LanguageDecodingMode.WITHIN_DATASET:
        return [np.asarray(index, dtype=int) for index in within_test_idxs]
    return [
        np.asarray(test, dtype=int)
        for _, test in KFold(
            n_splits=saved.encoding.num_folds, shuffle=False
        ).split(np.arange(n_rows))
    ]


def align_decoder_folds(
    decoder_selection: np.ndarray,
    neural_selection: np.ndarray,
    decoder_test_idxs: list[np.ndarray],
) -> list[np.ndarray]:
    """Map decoder-fold transcript rows onto a neural-analysis word selection."""
    decoder_selection = np.asarray(decoder_selection, dtype=int)
    neural_selection = np.asarray(neural_selection, dtype=int)
    if len(np.unique(decoder_selection)) != len(decoder_selection):
        raise ValueError("Decoder word selection contains duplicate transcript indices")
    lookup = {int(word): row for row, word in enumerate(decoder_selection)}
    missing = [int(word) for word in neural_selection if int(word) not in lookup]
    if missing:
        raise ValueError(
            f"Neural selection has {len(missing)} words absent from decoder selection"
        )
    decoder_row_to_fold = np.full(len(decoder_selection), -1, dtype=int)
    for fold, indices in enumerate(decoder_test_idxs):
        decoder_row_to_fold[np.asarray(indices, dtype=int)] = fold
    if np.any(decoder_row_to_fold < 0):
        raise ValueError("Decoder test folds do not partition its word selection")
    assignments = np.asarray(
        [decoder_row_to_fold[lookup[int(word)]] for word in neural_selection], dtype=int
    )
    return [np.flatnonzero(assignments == fold) for fold in range(len(decoder_test_idxs))]


def _load_factor(
    unit: Path,
    fold: int,
    rank: int,
    mode: LanguageDecodingMode,
    aggregation: str,
) -> np.ndarray:
    paths = sorted(
        unit.glob("fold_*/A.npy"),
        key=lambda path: int(path.parent.name.removeprefix("fold_")),
    )
    if not paths:
        raise FileNotFoundError(f"No fold factors found under {unit}")
    matrices = [np.asarray(np.load(path)) for path in paths]
    if any(matrix.shape[1] < rank for matrix in matrices):
        raise ValueError(f"A saved factor under {unit} has fewer than {rank} columns")
    if mode == LanguageDecodingMode.WITHIN_DATASET:
        if fold >= len(matrices):
            raise ValueError(f"Missing factor fold {fold} under {unit}")
        return matrices[fold][:, :rank]
    return aggregate_subspace_basis(matrices, rank=rank, method=aggregation)


def _load_task_weights(
    decoding_run: Path, task: str, subspace: str, rank: int, fold: int
) -> np.ndarray:
    path = (
        decoding_run
        / "tasks"
        / task
        / subspace
        / f"k{rank}"
        / f"fold_{fold}"
        / "weights.npz"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing saved decoder weights: {path}")
    with np.load(path, allow_pickle=False) as payload:
        weights = np.asarray(payload["reduced_weights"], dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    return weights


def _evaluate(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    saved: LanguageDecodingConfig,
    device: str,
    context: dict,
) -> dict:
    alphas = resolve_ridge_alphas(
        np.asarray(saved.encoding.ridge_alphas or RIDGE_ALPHAS, dtype=float),
        x_train,
        standardize=False,
        context=context,
        policy=saved.encoding.alpha_grid,
    )
    return evaluate_input_space(
        x_train,
        x_test,
        y_train,
        y_test,
        alphas=alphas,
        ridge_cv_folds=saved.encoding.ridge_cv_folds,
        standardize=False,
        ridge_solver_params=saved.encoding.ridge_solver_params,
        ridge_y_in_cpu=saved.encoding.ridge_y_in_cpu,
        ridge_force_cpu=saved.encoding.ridge_force_cpu,
        return_alphas=True,
        device=device,
    )


def _save_result(
    output_dir: Path,
    identifiers: dict,
    result: dict,
    target_manifest: pd.DataFrame,
) -> dict:
    parts = [
        identifiers["subspace"],
        f"k{identifiers['rank']}",
        identifiers["geometry"],
        identifiers["condition"],
        f"fold_{identifiers['fold']}",
    ]
    if identifiers.get("task"):
        parts.insert(-1, str(identifiers["task"]))
    if identifiers.get("draw") is not None:
        parts.insert(-1, f"draw_{identifiers['draw']}")
    artifact = output_dir.joinpath("artifacts", *parts)
    artifact.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact / "metrics.npz",
        corr_per_target=result["corr_per_target"],
        r2_per_target=result["r2_per_target"],
        best_alphas=result["best_alphas"],
    )
    row = {
        **identifiers,
        "corr": result["corr"],
        "r2": result["r2"],
        "n_targets": int(len(target_manifest)),
    }
    (artifact / "metrics.json").write_text(json.dumps(row, indent=2))
    return row


def _summarize(grid: pd.DataFrame) -> pd.DataFrame:
    keys = ["subspace", "rank", "geometry", "condition", "task"]
    observed = grid[grid["draw"].isna()]
    summary = observed.groupby(keys, dropna=False, sort=False)[["corr", "r2"]].agg(
        ["mean", "sem"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def _random_summary(grid: pd.DataFrame) -> pd.DataFrame:
    random = grid[grid["draw"].notna()]
    if random.empty:
        return pd.DataFrame()
    by_draw = (
        random.groupby(
            ["subspace", "rank", "geometry", "condition", "draw"], sort=False
        )[["corr", "r2"]]
        .mean()
        .reset_index()
    )
    rows = []
    for keys, group in by_draw.groupby(
        ["subspace", "rank", "geometry", "condition"], sort=False
    ):
        row = dict(zip(["subspace", "rank", "geometry", "condition"], keys))
        for metric in ("corr", "r2"):
            values = group[metric].to_numpy()
            row.update(
                {
                    f"{metric}_mean": float(np.mean(values)),
                    f"{metric}_p05": float(np.quantile(values, 0.05)),
                    f"{metric}_p95": float(np.quantile(values, 0.95)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_comparisons(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fold-paired sufficiency/removal and RRR-minus-PCA comparisons."""
    observed = grid[grid["draw"].isna() & grid["subspace"].isin(["rrr", "pca"])]
    index = ["subspace", "rank", "geometry", "fold"]
    wide = observed[observed["condition"].isin(["full", "joint", "residual"])].pivot(
        index=index, columns="condition", values=["corr", "r2"]
    )
    rows = []
    for keys, values in wide.iterrows():
        row = dict(zip(index, keys))
        for metric in ("corr", "r2"):
            row[f"{metric}_full"] = float(values[(metric, "full")])
            row[f"{metric}_joint"] = float(values[(metric, "joint")])
            row[f"{metric}_residual"] = float(values[(metric, "residual")])
            row[f"{metric}_sufficiency_gap"] = float(
                values[(metric, "full")] - values[(metric, "joint")]
            )
            row[f"{metric}_removal_loss"] = float(
                values[(metric, "full")] - values[(metric, "residual")]
            )
        rows.append(row)
    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        return comparisons, pd.DataFrame()
    paired = comparisons.pivot(
        index=["rank", "geometry", "fold"], columns="subspace"
    )
    gap_rows = []
    for keys, values in paired.iterrows():
        row = dict(zip(["rank", "geometry", "fold"], keys))
        for column in comparisons.columns:
            if column.startswith(("corr_", "r2_")):
                row[f"{column}_rrr_minus_pca"] = float(
                    values[(column, "rrr")] - values[(column, "pca")]
                )
        gap_rows.append(row)
    return comparisons, pd.DataFrame(gap_rows)


def empirical_random_tests(grid: pd.DataFrame) -> pd.DataFrame:
    """Compare observed joint sufficiency/removal loss with matched random spans."""
    if not grid["draw"].notna().any():
        return pd.DataFrame()
    rows = []
    for keys, group in grid[grid["subspace"].isin(["rrr", "pca"])].groupby(
        ["subspace", "rank", "geometry"], sort=False
    ):
        full = group[group.condition == "full"].set_index("fold")
        joint = group[group.condition == "joint"].set_index("fold")
        residual = group[group.condition == "residual"].set_index("fold")
        random_joint = group[group.condition == "random_joint"]
        random_residual = group[group.condition == "random_residual"]
        for metric in ("corr", "r2"):
            observed_joint = float(joint[metric].mean())
            observed_loss = float((full[metric] - residual[metric]).mean())
            null_joint = random_joint.groupby("draw")[metric].mean().to_numpy()
            random_loss_rows = random_residual.copy()
            random_loss_rows["loss"] = random_loss_rows.apply(
                lambda row: full.loc[int(row["fold"]), metric] - row[metric], axis=1
            )
            null_loss = random_loss_rows.groupby("draw")["loss"].mean().to_numpy()
            rows.append({
                "subspace": keys[0], "rank": keys[1], "geometry": keys[2],
                "metric": metric,
                "observed_joint": observed_joint,
                "random_joint_mean": float(np.mean(null_joint)),
                "joint_empirical_p": float((1 + np.count_nonzero(null_joint >= observed_joint)) / (1 + len(null_joint))),
                "observed_removal_loss": observed_loss,
                "random_removal_loss_mean": float(np.mean(null_loss)),
                "removal_empirical_p": float((1 + np.count_nonzero(null_loss >= observed_loss)) / (1 + len(null_loss))),
            })
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, random_summary: pd.DataFrame, output_dir: Path) -> None:
    for metric in ("corr", "r2"):
        subset = summary[summary["condition"].isin(["full", "joint", "residual", "labels_joint"])]
        if subset.empty:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for axis, geometry in zip(axes, ("raw", "whitened")):
            current = pd.concat(
                [
                    subset[subset.geometry == geometry],
                    subset[
                        (subset.geometry == "labels")
                        & (subset.condition == "labels_joint")
                    ],
                ],
                ignore_index=True,
            )
            labels = []
            values = []
            errors = []
            colors = []
            for _, row in current.iterrows():
                labels.append(f"{row['subspace']}\n{row['condition']}")
                values.append(row[f"{metric}_mean"])
                errors.append(row[f"{metric}_sem"])
                colors.append("#3b82f6" if row["subspace"] == "rrr" else "#f59e0b")
            axis.bar(np.arange(len(values)), values, yerr=errors, color=colors, alpha=0.85)
            axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
            axis.set_title(geometry.capitalize())
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(metric.upper())
        figure.tight_layout()
        figure.savefig(output_dir / f"encoding_{metric}.png", dpi=180)
        plt.close(figure)

    if not random_summary.empty:
        random_summary.to_csv(output_dir / "random_control_summary.csv", index=False)
        for metric in ("corr", "r2"):
            figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
            for axis, observed_condition, random_condition in zip(
                axes,
                ("joint", "residual"),
                ("random_joint", "random_residual"),
            ):
                labels = []
                positions = []
                observed_values = []
                null_means = []
                lower = []
                upper = []
                position = 0
                for subspace in ("rrr", "pca"):
                    for geometry in ("raw", "whitened"):
                        observed = summary[
                            (summary.subspace == subspace)
                            & (summary.geometry == geometry)
                            & (summary.condition == observed_condition)
                        ]
                        null = random_summary[
                            (random_summary.subspace == subspace)
                            & (random_summary.geometry == geometry)
                            & (random_summary.condition == random_condition)
                        ]
                        if observed.empty or null.empty:
                            continue
                        labels.append(f"{subspace}\n{geometry}")
                        positions.append(position)
                        observed_values.append(float(observed.iloc[0][f"{metric}_mean"]))
                        null_means.append(float(null.iloc[0][f"{metric}_mean"]))
                        lower.append(float(null.iloc[0][f"{metric}_p05"]))
                        upper.append(float(null.iloc[0][f"{metric}_p95"]))
                        position += 1
                positions = np.asarray(positions)
                null_means = np.asarray(null_means)
                axis.errorbar(
                    positions,
                    null_means,
                    yerr=np.vstack([null_means - lower, np.asarray(upper) - null_means]),
                    fmt="o",
                    color="#64748b",
                    capsize=4,
                    label="random 5–95%",
                )
                axis.scatter(positions, observed_values, marker="D", color="#dc2626", label="task axes")
                axis.set_xticks(positions, labels)
                axis.set_title(observed_condition.capitalize())
                axis.grid(axis="y", alpha=0.25)
            axes[0].set_ylabel(metric.upper())
            axes[0].legend()
            figure.tight_layout()
            figure.savefig(output_dir / f"random_controls_{metric}.png", dpi=180)
            plt.close(figure)


def _plot_comparisons(
    comparisons: pd.DataFrame, gaps: pd.DataFrame, output_dir: Path
) -> None:
    if comparisons.empty:
        return
    for metric in ("corr", "r2"):
        aggregate = comparisons.groupby(
            ["subspace", "geometry"], sort=False
        )[[f"{metric}_sufficiency_gap", f"{metric}_removal_loss"]].mean()
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
        for axis, column, title in zip(
            axes,
            (f"{metric}_sufficiency_gap", f"{metric}_removal_loss"),
            ("Full minus joint task component", "Full minus residual"),
        ):
            values = aggregate[column]
            labels = [f"{space}\n{geometry}" for space, geometry in values.index]
            colors = ["#3b82f6" if space == "rrr" else "#f59e0b" for space, _ in values.index]
            axis.bar(np.arange(len(values)), values.to_numpy(), color=colors)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_xticks(np.arange(len(labels)), labels)
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle(metric.upper())
        figure.tight_layout()
        figure.savefig(output_dir / f"sufficiency_removal_{metric}.png", dpi=180)
        plt.close(figure)

        if not gaps.empty:
            gap_columns = [
                f"{metric}_joint_rrr_minus_pca",
                f"{metric}_residual_rrr_minus_pca",
                f"{metric}_removal_loss_rrr_minus_pca",
            ]
            gap_means = gaps.groupby("geometry", sort=False)[gap_columns].mean()
            axis = gap_means.plot(kind="bar", figsize=(8, 4.5))
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_ylabel("RRR minus PCA")
            axis.set_title(f"Task-axis encoding gaps ({metric.upper()})")
            axis.grid(axis="y", alpha=0.25)
            axis.figure.tight_layout()
            axis.figure.savefig(output_dir / f"rrr_pca_gaps_{metric}.png", dpi=180)
            plt.close(axis.figure)


def run_language_axis_encoding(config: LanguageAxisEncodingConfig) -> Path:
    """Run task-axis sufficiency/removal encoding and return its output directory."""
    decoding_run = Path(config.language_decoding_run_path).expanduser().resolve()
    output_dir = (
        Path(config.output_dir).expanduser()
        if config.output_dir
        else decoding_run / "axis_encoding"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, output_dir)
    if (output_dir / "grid_results.csv").exists() and not config.refresh:
        return output_dir
    saved_config_path = decoding_run / "config.yml"
    if not saved_config_path.exists():
        raise FileNotFoundError(f"Missing language-decoding config: {saved_config_path}")
    saved = load_config(str(saved_config_path), {}, LanguageDecodingConfig)
    missing_ranks = set(map(int, config.rank_grid)) - set(map(int, saved.rank_grid))
    missing_tasks = set(config.tasks) - set(saved.tasks)
    if missing_ranks or missing_tasks:
        raise ValueError(
            f"Requested artifacts were not in the decoding run: "
            f"ranks={sorted(missing_ranks)}, tasks={sorted(missing_tasks)}"
        )
    if not saved.pca_run_path:
        raise ValueError("The saved decoding run has no PCA comparison path")

    device = _configure_device(config.device)
    enc = saved.encoding
    subjects = enc.subjects or list(range(1, 10))
    dataset_context = resolve_dataset_context(
        enc.dataset, subjects, enc.per_dataset_config, enc.data_root
    )
    result_root = get_results_dir(
        enc.model,
        enc.offset,
        enc.per_model_config,
        dataset_context=dataset_context,
    )
    embeddings, word_df = load_embeddings(
        enc.data_root,
        enc.model,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        **enc.per_model_config,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    sig_root = enc.sig_dir or result_root
    sig_electrodes, _ = load_sig_electrodes(
        result_root,
        subjects,
        sig_dir=sig_root,
        dataset=enc.dataset,
        per_dataset_config=enc.per_dataset_config,
        dataset_context=dataset_context,
        data_root=enc.data_root,
    )
    prepared = _get_decomposition_inputs(
        subjects,
        result_root,
        sig_electrodes,
        enc,
        embeddings,
    )
    if prepared is None:
        raise ValueError("No neural targets were available")
    x_raw = next(iter(prepared.embeddings.values())).astype(np.float32)
    if len(x_raw) != len(prepared.targets):
        raise ValueError("Aligned embeddings and neural targets have different rows")

    word_selection = pd.read_csv(decoding_run / "word_selection.csv")
    decoder_selection = word_selection["transcript_word_idx"].to_numpy(dtype=int)
    neural_selection = selection_after_offset(
        np.asarray(prepared.selection, dtype=int), int(enc.offset)
    )
    if len(neural_selection) != len(x_raw):
        raise ValueError(
            "Could not reconstruct the transcript indices of neural analysis rows: "
            f"selection={len(neural_selection)}, rows={len(x_raw)}"
        )
    if not np.allclose(
        x_raw, embeddings[neural_selection], rtol=1e-5, atol=1e-5
    ):
        raise ValueError("Neural embeddings do not match reconstructed word selection")
    decoder_test_idxs = _decoder_folds(
        saved, len(decoder_selection), prepared.test_idxs
    )
    test_idxs = align_decoder_folds(
        decoder_selection, neural_selection, decoder_test_idxs
    )
    if saved.mode == LanguageDecodingMode.WITHIN_DATASET and any(
        not np.array_equal(left, right)
        for left, right in zip(test_idxs, prepared.test_idxs)
    ):
        raise ValueError("Mapped decoder folds do not match saved neural folds")

    task_context = PodcastTaskContext(word_df, enc.data_root, saved.task_data)
    task_matrix, continuous, categorical, task_schema = load_task_matrix(
        task_context, config.tasks
    )
    task_matrix = task_matrix[neural_selection]

    prepared.target_manifest.to_csv(output_dir / "target_manifest.csv", index=False)
    metadata = {
        "config": asdict(config),
        "source_decoding_run": str(decoding_run),
        "source_decoding_config": asdict(saved),
        "effective_device": device,
        "task_schema": task_schema,
        "n_rows": int(len(x_raw)),
        "n_targets": int(prepared.targets.shape[1]),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    units = {
        "rrr": resolve_subspace_unit_dir(saved.rrr_run_path, "rrr"),
        "pca": resolve_subspace_unit_dir(saved.pca_run_path, "pca"),
    }
    rows = []
    diagnostics = []
    overlap_rows = []
    for fold, test_idx in enumerate(test_idxs):
        train_mask = np.ones(len(x_raw), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.flatnonzero(train_mask)
        scaler = StandardScaler()
        y_train = scaler.fit_transform(prepared.targets[train_idx]).astype(np.float32)
        y_test = scaler.transform(prepared.targets[test_idx]).astype(np.float32)

        # Actual-language-label controls are independent of RRR/PCA geometry.
        for task_index, task in enumerate(config.tasks):
            prep = build_preprocessor(
                [0] if task_index in continuous else [],
                [0] if task_index in categorical else [],
            )
            label_train = np.asarray(
                prep.fit_transform(task_matrix[train_idx][:, [task_index]]), dtype=np.float32
            )
            label_test = np.asarray(
                prep.transform(task_matrix[test_idx][:, [task_index]]), dtype=np.float32
            )
            result = _evaluate(
                label_train, label_test, y_train, y_test,
                saved=saved, device=device,
                context={"script": "language_axis_encoding", "condition": "labels", "task": task, "fold": fold},
            )
            rows.append(_save_result(output_dir, {
                "subspace": "labels", "rank": 0, "geometry": "labels",
                "condition": "labels_individual", "task": task, "fold": fold, "draw": None,
            }, result, prepared.target_manifest))
        joint_preprocessor = build_preprocessor(continuous, categorical)
        label_train = np.asarray(joint_preprocessor.fit_transform(task_matrix[train_idx]), dtype=np.float32)
        label_test = np.asarray(joint_preprocessor.transform(task_matrix[test_idx]), dtype=np.float32)
        result = _evaluate(
            label_train, label_test, y_train, y_test,
            saved=saved, device=device,
            context={"script": "language_axis_encoding", "condition": "labels_joint", "fold": fold},
        )
        rows.append(_save_result(output_dir, {
            "subspace": "labels", "rank": 0, "geometry": "labels",
            "condition": "labels_joint", "task": None, "fold": fold, "draw": None,
        }, result, prepared.target_manifest))

        for subspace, unit in units.items():
            for rank in map(int, config.rank_grid):
                factor = _load_factor(
                    unit, fold, rank, saved.mode, saved.subspace_aggregation
                )
                if factor.shape[0] != x_raw.shape[1]:
                    raise ValueError(
                        "Language-axis encoding currently requires a single-view factor "
                        f"with ambient dimension {x_raw.shape[1]}; got {factor.shape}"
                    )
                z_train = x_raw[train_idx] @ factor
                z_test = x_raw[test_idx] @ factor
                task_weights = {
                    task: _load_task_weights(decoding_run, task, subspace, rank, fold)
                    for task in config.tasks
                }
                joint_weights = np.concatenate(list(task_weights.values()), axis=1)
                for geometry in config.geometries:
                    joint = decompose_task_span(z_train, z_test, joint_weights, geometry)
                    overlap_rows.extend(
                        {
                            "subspace": subspace,
                            "rank": rank,
                            "geometry": geometry,
                            "fold": fold,
                            **overlap,
                        }
                        for overlap in task_axis_overlaps(
                            z_train, task_weights, geometry
                        )
                    )
                    diagnostics.append({
                        "subspace": subspace, "rank": rank, "geometry": geometry,
                        "fold": fold, "effective_rank": joint["effective_rank"],
                        "variance_fraction": joint["variance_fraction"],
                        "reconstruction_error": float(np.max(np.abs(
                            joint["task_train"] + joint["residual_train"] - joint["full_train"]
                        ))),
                    })
                    for condition, train_key, test_key in (
                        ("full", "full_train", "full_test"),
                        ("joint", "task_train", "task_test"),
                        ("residual", "residual_train", "residual_test"),
                    ):
                        result = _evaluate(
                            joint[train_key], joint[test_key], y_train, y_test,
                            saved=saved, device=device,
                            context={"script": "language_axis_encoding", "subspace": subspace, "rank": rank, "geometry": geometry, "condition": condition, "fold": fold},
                        )
                        rows.append(_save_result(output_dir, {
                            "subspace": subspace, "rank": rank, "geometry": geometry,
                            "condition": condition, "task": None, "fold": fold, "draw": None,
                        }, result, prepared.target_manifest))
                    for task, weights in task_weights.items():
                        one = decompose_task_span(z_train, z_test, weights, geometry)
                        result = _evaluate(
                            one["task_train"], one["task_test"], y_train, y_test,
                            saved=saved, device=device,
                            context={"script": "language_axis_encoding", "subspace": subspace, "rank": rank, "geometry": geometry, "condition": "individual", "task": task, "fold": fold},
                        )
                        rows.append(_save_result(output_dir, {
                            "subspace": subspace, "rank": rank, "geometry": geometry,
                            "condition": "individual", "task": task, "fold": fold, "draw": None,
                        }, result, prepared.target_manifest))
                    for draw in range(config.random_control_draws):
                        seed = config.random_seed + fold * 100000 + rank * 1000 + draw
                        random_weights = random_span_weights(
                            rank, int(joint["effective_rank"]), seed
                        )
                        if geometry == "whitened":
                            _, inverse_square_root = covariance_transforms(z_train)
                            # Draw Haar-uniformly in the whitened metric, then express
                            # that covector span in the saved raw rank coordinates.
                            random_weights = inverse_square_root @ random_weights
                        random_split = decompose_task_span(
                            z_train, z_test, random_weights, geometry
                        )
                        for condition, train_key, test_key in (
                            ("random_joint", "task_train", "task_test"),
                            ("random_residual", "residual_train", "residual_test"),
                        ):
                            result = _evaluate(
                                random_split[train_key], random_split[test_key], y_train, y_test,
                                saved=saved, device=device,
                                context={"script": "language_axis_encoding", "subspace": subspace, "rank": rank, "geometry": geometry, "condition": condition, "fold": fold, "draw": draw},
                            )
                            rows.append(_save_result(output_dir, {
                                "subspace": subspace, "rank": rank, "geometry": geometry,
                                "condition": condition, "task": None, "fold": fold, "draw": draw,
                            }, result, prepared.target_manifest))

    grid = pd.DataFrame(rows)
    grid.to_csv(output_dir / "grid_results.csv", index=False)
    summary = _summarize(grid)
    summary.to_csv(output_dir / "summary.csv", index=False)
    random_summary = _random_summary(grid)
    comparisons, rrr_pca_gaps = build_comparisons(grid)
    comparisons.to_csv(output_dir / "sufficiency_removal.csv", index=False)
    rrr_pca_gaps.to_csv(output_dir / "rrr_pca_gaps.csv", index=False)
    random_tests = empirical_random_tests(grid)
    if not random_tests.empty:
        random_tests.to_csv(output_dir / "random_control_tests.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "task_span_diagnostics.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output_dir / "task_axis_overlaps.csv", index=False)
    _plot(summary, random_summary, output_dir)
    _plot_comparisons(comparisons, rrr_pca_gaps, output_dir)
    return output_dir
