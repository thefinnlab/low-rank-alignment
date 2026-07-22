"""Evaluate cross-dataset or fMRI-derived RRR/PCA axes on deep-fMRI-dataset."""

import json
from dataclasses import dataclass
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
from src.utils.srrr import fit_dense_rrr
from src.utils.subspace_compare import (
    aggregate_subspace_basis,
    aggregate_subspace_from_unit_dir,
)
from src.utils.tensor_utils import to_numpy


@dataclass
class FinalRidgeFit:
    """Predictions and fitted quantities needed by downstream fMRI RRR."""

    test_prediction: np.ndarray
    train_prediction: np.ndarray
    coefficients: np.ndarray
    best_alphas: np.ndarray


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
    test_prediction = to_numpy(model.predict(X_test)).astype(np.float32)
    train_prediction = to_numpy(model.predict(X_train)).astype(np.float32)
    coefficients = to_numpy(model.named_steps["ridgecv"].coef_)
    if coefficients.shape[0] != X_train.shape[1]:
        coefficients = coefficients.T
    if coefficients.shape != (X_train.shape[1], Y_train.shape[1]):
        raise ValueError(
            "Unexpected full-ridge coefficient shape "
            f"{coefficients.shape}; expected {(X_train.shape[1], Y_train.shape[1])}"
        )
    selected = extract_best_alphas(model)
    if config.ridge.log_alphas:
        save_alpha_diagnostics(
            str(cache_dir), alphas, best_alphas=[selected], context=context,
            policy=config.ridge.alpha_grid,
        )
    return FinalRidgeFit(
        test_prediction=test_prediction,
        train_prediction=train_prediction,
        coefficients=np.asarray(coefficients, dtype=np.float32),
        best_alphas=selected,
    )


def _validate_cache_contract(output_dir: Path, config) -> None:
    """Prevent reuse of metrics produced by another source/aggregation mode."""
    path = output_dir / "metadata.json"
    if config.refresh:
        return
    if not path.exists():
        if config.subspace_aggregation != "extrinsic" and any(output_dir.iterdir()):
            raise ValueError(
                f"Existing legacy results in {output_dir} are assumed to use "
                "'extrinsic'; set refresh=true or use a different output_dir."
            )
        return
    with path.open() as handle:
        metadata = json.load(handle)
    existing_mode = metadata.get("source_mode", "cross_dataset")
    existing_aggregation = metadata.get("subspace_aggregation", "extrinsic")
    expected = (config.source_mode, config.subspace_aggregation)
    existing = (existing_mode, existing_aggregation)
    if existing != expected:
        raise ValueError(
            f"Existing results in {output_dir} use source/aggregation {existing}, "
            f"but the configuration requests {expected}. Set refresh=true or use "
            "a different output_dir."
        )


def _fit_within_fmri_bases(full_train, ridge_fit, ambient_dim, config, output_dir):
    """Derive and persist shared FIR-RRR and pre-FIR PCA bases."""
    ranks = sorted(set(map(int, config.rank_grid)))
    max_rank = max(ranks)
    ndelays = int(config.fmri.ndelays)
    expected_width = ndelays * ambient_dim
    if full_train.stimulus.shape[1] != expected_width:
        raise ValueError(
            f"FIR design has width {full_train.stimulus.shape[1]}; expected "
            f"ndelays * embedding_dim = {expected_width}"
        )
    available_rrr_rank = min(
        full_train.stimulus.shape[0],
        full_train.responses.shape[1],
        ambient_dim,
    )
    if max_rank > available_rrr_rank:
        raise ValueError(
            f"Requested fMRI RRR rank {max_rank} exceeds shared rank limit "
            f"{available_rrr_rank}"
        )

    rrr_result = fit_dense_rrr(
        full_train.stimulus,
        full_train.responses,
        k=max_rank,
        selected_idx=np.arange(expected_width),
        precomputed_Y_hat_ols=ridge_fit.train_prediction,
        precomputed_beta_full=ridge_fit.coefficients,
    )
    fir_a = np.asarray(rrr_result["A_dense"], dtype=np.float32)
    delay_blocks = list(np.split(fir_a, ndelays, axis=0))
    source_dir = output_dir / "source_factors"
    source_dir.mkdir(parents=True, exist_ok=True)
    np.save(source_dir / "full_fir_A.npy", fir_a)
    for delay, block in enumerate(delay_blocks, start=1):
        delay_dir = source_dir / f"delay_{delay}"
        delay_dir.mkdir(parents=True, exist_ok=True)
        np.save(delay_dir / "A.npy", block)

    rrr_bases = {}
    for rank in ranks:
        basis = aggregate_subspace_basis(
            delay_blocks, rank=rank, method=config.subspace_aggregation
        )
        rank_dir = output_dir / "rrr" / f"k{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        np.save(rank_dir / "A.npy", basis)
        rrr_bases[rank] = basis

    centered = full_train.pre_fir_stimulus - full_train.pre_fir_stimulus.mean(
        axis=0, keepdims=True
    )
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pca_a = vt.T.astype(np.float32, copy=False)
    if max_rank > pca_a.shape[1]:
        raise ValueError(
            f"Requested fMRI PCA rank {max_rank} exceeds available rank {pca_a.shape[1]}"
        )
    np.save(source_dir / "pre_fir_pca_A.npy", pca_a)
    pca_bases = {}
    for rank in ranks:
        basis = pca_a[:, :rank]
        rank_dir = output_dir / "pca" / f"k{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        np.save(rank_dir / "A.npy", basis)
        pca_bases[rank] = basis

    return {"rrr": rrr_bases, "pca": pca_bases}, {
        "rrr": str(source_dir / "full_fir_A.npy"),
        "rrr_delay_blocks": [
            str(source_dir / f"delay_{delay}" / "A.npy")
            for delay in range(1, ndelays + 1)
        ],
        "pca": str(source_dir / "pre_fir_pca_A.npy"),
        "rrr_rank_bases": {
            str(rank): str(output_dir / "rrr" / f"k{rank}" / "A.npy")
            for rank in ranks
        },
        "pca_rank_bases": {
            str(rank): str(output_dir / "pca" / f"k{rank}" / "A.npy")
            for rank in ranks
        },
    }


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
    _validate_cache_contract(output_dir, config)
    source_unit_dirs = {}
    pca_dir = None
    if config.source_mode == "cross_dataset":
        rrr_paths = _fold_factor_paths(config.source_rrr_unit_dir)
        if not rrr_paths:
            raise FileNotFoundError(
                f"No fold_*/A.npy factors found in {config.source_rrr_unit_dir}"
            )
        pca_dir = _validate_optional_pca(config.source_rrr_unit_dir)
        source_unit_dirs["rrr"] = config.source_rrr_unit_dir
        if pca_dir is not None:
            source_unit_dirs["pca"] = pca_dir

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
    full_fit = _fit_final_ridge(
        full_train.stimulus,
        full_train.responses,
        full_test.stimulus,
        cv,
        config,
        output_dir / "full",
        {
            "script": "fmri_cross_dataset_rrr",
            "method": "full",
            "n_features": full_train.stimulus.shape[1],
            "standardize": False,
        },
    )
    full_corr_arr = columnwise_correlation(
        full_test.responses, full_fit.test_prediction
    )
    full_r2_arr = _r2_per_target(full_test.responses, full_fit.test_prediction)
    np.savez_compressed(
        output_dir / "full" / "metrics.npz",
        corr_per_target=full_corr_arr,
        r2_per_target=full_r2_arr,
        best_alphas=full_fit.best_alphas,
    )
    full_corr = _fisher_mean(full_corr_arr)
    full_r2 = float(np.mean(full_r2_arr))

    if config.source_mode == "within_fmri":
        bases_by_method, factor_metadata = _fit_within_fmri_bases(
            full_train, full_fit, ambient_dim, config, output_dir
        )
    else:
        bases_by_method = {
            method: {
                rank: aggregate_subspace_from_unit_dir(
                    unit_dir,
                    rank=rank,
                    method=config.subspace_aggregation,
                )
                for rank in map(int, config.rank_grid)
            }
            for method, unit_dir in source_unit_dirs.items()
        }
        factor_metadata = dict(source_unit_dirs)

    rows = []
    for method, rank_bases in bases_by_method.items():
        for rank in map(int, config.rank_grid):
            metrics_path = output_dir / method / f"k{rank}" / "metrics.npz"
            if metrics_path.exists() and not config.refresh:
                with np.load(metrics_path) as saved:
                    corr_arr = saved["corr_per_target"]
                    r2_arr = saved["r2_per_target"]
            else:
                basis = rank_bases[rank]
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
                fitted = _fit_final_ridge(
                    train.stimulus,
                    train.responses,
                    test.stimulus,
                    cv,
                    config,
                    metrics_path.parent,
                    {
                        "script": "fmri_cross_dataset_rrr",
                        "method": method,
                        "rank": rank,
                        "n_features": train.stimulus.shape[1],
                        "standardize": False,
                    },
                )
                corr_arr = columnwise_correlation(
                    test.responses, fitted.test_prediction
                )
                r2_arr = _r2_per_target(test.responses, fitted.test_prediction)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    metrics_path,
                    corr_per_target=corr_arr,
                    r2_per_target=r2_arr,
                    best_alphas=fitted.best_alphas,
                )
            row = {
                "method": method,
                "rank": rank,
                "corr": _fisher_mean(corr_arr),
                "r2": float(np.mean(r2_arr)),
                "full_corr": full_corr,
                "full_r2": full_r2,
            }
            row.update(
                {
                    f"corr_{subject}": value
                    for subject, value in _subject_summary(corr_arr, manifest).items()
                }
            )
            rows.append(row)
            print(
                f"{method.upper()} rank={rank}: "
                f"r={row['corr']:.4f}, R2={row['r2']:.4f}"
            )

    frame = pd.DataFrame(rows).sort_values(["method", "rank"])
    frame.to_csv(output_dir / "grid_results.csv", index=False)
    _plot_results(frame, output_dir / "rank_analysis.png")
    metadata = {
        "source_factors": factor_metadata,
        "source_mode": config.source_mode,
        "subspace_aggregation": config.subspace_aggregation,
        "pca_available": "pca" in bases_by_method,
        "pca_training_representation": (
            "tr_aligned_trimmed_story_zscored_pre_fir"
            if config.source_mode == "within_fmri"
            else "source_fold_factors"
        ),
        "target_scaling": "stored_preprocessed_responses",
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
        "delay_order": list(range(1, int(config.fmri.ndelays) + 1)),
        "pre_fir_training_shape": list(full_train.pre_fir_stimulus.shape),
        "rank_grid": list(map(int, config.rank_grid)),
        "ceiling": "fresh_full_embedding_ridge",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args, overrides = parse_args(__doc__)
    run_config(load_config(args.config, overrides, FMRICrossDatasetRRRConfig))


if __name__ == "__main__":
    main()
