"""Evaluate saved iEEG RRR and optional PCA axes on deep-fMRI-dataset."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend

from src.core.config import FMRICrossDatasetRRRConfig
from src.utils.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
from src.utils.config_utils import load_config, parse_args
from src.utils.encoding import extract_best_alphas, get_ridge_model_constructor
from src.utils.fmri import (
    columnwise_correlation,
    load_story_embeddings,
    prepare_fmri_data,
    resolve_fmri_stories,
    resolve_pca_sibling,
    session_cv_splits,
    target_manifest,
    TEMPORARILY_HELD_OUT_FMRI_STORIES,
)
from src.utils.subspace_compare import extrinsic_mean_from_unit_dir
from src.utils.tensor_utils import to_numpy


def _fold_factor_paths(unit_dir: str) -> list[Path]:
    return sorted(Path(unit_dir).glob("fold_*/A.npy"))


def _validate_optional_pca(rrr_dir: str) -> str | None:
    pca_dir = resolve_pca_sibling(rrr_dir)
    pca_paths = _fold_factor_paths(pca_dir)
    if not pca_paths:
        return None
    rrr_folds = {path.parent.name for path in _fold_factor_paths(rrr_dir)}
    pca_folds = {path.parent.name for path in pca_paths}
    if pca_folds != rrr_folds:
        raise ValueError(
            f"PCA factors in {pca_dir} are partial: folds={sorted(pca_folds)}, "
            f"RRR folds={sorted(rrr_folds)}"
        )
    return pca_dir


def _load_masks(config) -> dict[str, np.ndarray]:
    masks = {}
    for subject in config.fmri.subjects:
        path = Path(config.significance_dir) / "subjects" / subject / "significance.csv"
        frame = pd.read_csv(path).sort_values("voxel_index")
        expected = np.arange(len(frame))
        if not np.array_equal(frame["voxel_index"].to_numpy(), expected):
            raise ValueError(f"Non-contiguous voxel indices in {path}")
        masks[subject] = frame["sig"].astype(bool).to_numpy()
        if not masks[subject].any():
            raise ValueError(f"No significant voxels found for {subject}")
    return masks


def _fisher_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.tanh(np.mean(np.arctanh(np.clip(values[finite], -0.999999, 0.999999)))))


def _r2_per_target(true, predicted):
    numerator = np.sum((true - predicted) ** 2, axis=0)
    denominator = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    return np.nan_to_num(1.0 - numerator / denominator, nan=0.0, neginf=0.0)


def _fit_final_ridge(X_train, Y_train, X_test, cv, config, cache_dir, context):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    explicit = np.asarray(config.ridge.alphas) if config.ridge.alphas else None
    alphas = resolve_ridge_alphas(
        explicit,
        X_train,
        standardize=False,
        context=context,
        cache_dir=str(cache_dir),
        policy=config.ridge.alpha_grid,
    )
    model = get_ridge_model_constructor(
        alphas=alphas,
        cv=cv,
        standardize=False,
        solver_params=config.ridge.solver_params,
        y_in_cpu=config.ridge.y_in_cpu,
        force_cpu=config.ridge.force_cpu,
    )(0)
    model.fit(X_train, Y_train)
    prediction = to_numpy(model.predict(X_test)).astype(np.float32)
    selected = extract_best_alphas(model)
    if config.ridge.log_alphas:
        save_alpha_diagnostics(
            str(cache_dir), alphas, best_alphas=[selected], context=context,
            policy=config.ridge.alpha_grid,
        )
    return prediction, selected


def _project_story_embeddings(embeddings, basis):
    return {
        story: np.asarray(values @ basis, dtype=np.float32)
        for story, values in embeddings.items()
    }


def _subject_summary(values, manifest):
    return {
        subject: _fisher_mean(values[manifest["subject"].to_numpy() == subject])
        for subject in manifest["subject"].unique()
    }


def _plot_results(frame: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for method, group in frame.groupby("method"):
        ordered = group.sort_values("rank")
        axes[0].plot(ordered["rank"], ordered["corr"], marker="o", label=method.upper())
        axes[1].plot(ordered["rank"], ordered["r2"], marker="o", label=method.upper())
    if len(frame):
        axes[0].axhline(frame["full_corr"].iloc[0], color="black", ls="--", label="Full")
        axes[1].axhline(frame["full_r2"].iloc[0], color="black", ls="--", label="Full")
    for ax, ylabel in zip(axes, ["Fisher-z mean Pearson r", "Mean R²"]):
        ax.set_xscale("log", base=2)
        ax.set(xlabel="Rank", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_title("Primary encoding performance")
    axes[1].set_title("Variance explained")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_config(config: FMRICrossDatasetRRRConfig) -> None:
    config.validate()
    if torch.cuda.is_available():
        set_backend("torch_cuda")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rrr_paths = _fold_factor_paths(config.source_rrr_unit_dir)
    if not rrr_paths:
        raise FileNotFoundError(
            f"No fold_*/A.npy factors found in {config.source_rrr_unit_dir}"
        )
    pca_dir = _validate_optional_pca(config.source_rrr_unit_dir)
    methods = {"rrr": config.source_rrr_unit_dir}
    if pca_dir is not None:
        methods["pca"] = pca_dir

    train_stories, test_stories, groups = resolve_fmri_stories(
        config.fmri.fmri_repo_root, config.fmri.sessions
    )
    if len(test_stories) != 1:
        raise ValueError(f"Expected one shared test story, got {test_stories}")
    all_stories = train_stories + test_stories
    embeddings = load_story_embeddings(
        config.fmri.embedding_root,
        config.model_dir,
        all_stories,
        config.layer,
        config.pooling,
    )
    ambient_dim = next(iter(embeddings.values())).shape[1]
    masks = _load_masks(config)
    manifest = target_manifest(config.fmri.subjects, masks)
    manifest.to_csv(output_dir / "target_manifest.csv", index=False)

    full_train = prepare_fmri_data(
        config.fmri.fmri_repo_root,
        train_stories,
        embeddings,
        trim=config.fmri.trim,
        ndelays=config.fmri.ndelays,
        subjects=config.fmri.subjects,
        voxel_masks=masks,
    )
    full_test = prepare_fmri_data(
        config.fmri.fmri_repo_root,
        test_stories,
        embeddings,
        trim=config.fmri.trim,
        ndelays=config.fmri.ndelays,
        subjects=config.fmri.subjects,
        voxel_masks=masks,
    )
    cv = session_cv_splits(full_train.story_slices, groups)
    full_prediction, full_alphas = _fit_final_ridge(
        full_train.stimulus,
        full_train.responses,
        full_test.stimulus,
        cv,
        config,
        output_dir / "full",
        {"script": "fmri_cross_dataset_rrr", "method": "full", "n_features": full_train.stimulus.shape[1], "standardize": False},
    )
    full_corr_arr = columnwise_correlation(full_test.responses, full_prediction)
    full_r2_arr = _r2_per_target(full_test.responses, full_prediction)
    np.savez_compressed(
        output_dir / "full" / "metrics.npz",
        corr_per_target=full_corr_arr,
        r2_per_target=full_r2_arr,
        best_alphas=full_alphas,
    )
    full_corr = _fisher_mean(full_corr_arr)
    full_r2 = float(np.mean(full_r2_arr))

    rows = []
    factor_metadata = {}
    for method, unit_dir in methods.items():
        factor_metadata[method] = unit_dir
        for rank in map(int, config.rank_grid):
            metrics_path = output_dir / method / f"k{rank}" / "metrics.npz"
            if metrics_path.exists() and not config.refresh:
                with np.load(metrics_path) as saved:
                    corr_arr = saved["corr_per_target"]
                    r2_arr = saved["r2_per_target"]
            else:
                basis = extrinsic_mean_from_unit_dir(unit_dir, rank=rank)
                if basis.shape != (ambient_dim, rank):
                    raise ValueError(
                        f"{method.upper()} rank {rank} basis has shape {basis.shape}; "
                        f"expected ({ambient_dim}, {rank})"
                    )
                projected = _project_story_embeddings(embeddings, basis)
                train = prepare_fmri_data(
                    config.fmri.fmri_repo_root,
                    train_stories,
                    projected,
                    trim=config.fmri.trim,
                    ndelays=config.fmri.ndelays,
                    subjects=config.fmri.subjects,
                    voxel_masks=masks,
                )
                test = prepare_fmri_data(
                    config.fmri.fmri_repo_root,
                    test_stories,
                    projected,
                    trim=config.fmri.trim,
                    ndelays=config.fmri.ndelays,
                    subjects=config.fmri.subjects,
                    voxel_masks=masks,
                )
                prediction, selected = _fit_final_ridge(
                    train.stimulus,
                    train.responses,
                    test.stimulus,
                    cv,
                    config,
                    metrics_path.parent,
                    {"script": "fmri_cross_dataset_rrr", "method": method, "rank": rank, "n_features": train.stimulus.shape[1], "standardize": False},
                )
                corr_arr = columnwise_correlation(test.responses, prediction)
                r2_arr = _r2_per_target(test.responses, prediction)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    metrics_path,
                    corr_per_target=corr_arr,
                    r2_per_target=r2_arr,
                    best_alphas=selected,
                )
            row = {
                "method": method,
                "rank": rank,
                "corr": _fisher_mean(corr_arr),
                "r2": float(np.mean(r2_arr)),
                "full_corr": full_corr,
                "full_r2": full_r2,
            }
            row.update({f"corr_{subject}": value for subject, value in _subject_summary(corr_arr, manifest).items()})
            rows.append(row)
            print(f"{method.upper()} rank={rank}: r={row['corr']:.4f}, R2={row['r2']:.4f}")

    frame = pd.DataFrame(rows).sort_values(["method", "rank"])
    frame.to_csv(output_dir / "grid_results.csv", index=False)
    _plot_results(frame, output_dir / "rank_analysis.png")
    metadata = {
        "source_factors": factor_metadata,
        "pca_available": pca_dir is not None,
        "subjects": config.fmri.subjects,
        "sessions": config.fmri.sessions,
        "train_stories": train_stories,
        "test_story": test_stories[0],
        "temporarily_held_out_stories": sorted(TEMPORARILY_HELD_OUT_FMRI_STORIES),
        "trim": config.fmri.trim,
        "fixed_leading_trim": 5,
        "effective_start_trim": 5 + config.fmri.trim,
        "effective_end_trim": config.fmri.trim,
        "ndelays": config.fmri.ndelays,
        "rank_grid": list(map(int, config.rank_grid)),
        "ceiling": "fresh_full_embedding_ridge",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args, overrides = parse_args(__doc__)
    run_config(load_config(args.config, overrides, FMRICrossDatasetRRRConfig))


if __name__ == "__main__":
    main()
