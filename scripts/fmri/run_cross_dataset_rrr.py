"""Evaluate cross-dataset or fMRI-derived RRR/PCA axes on deep-fMRI-dataset."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend

from src.core.config import FMRICrossDatasetRRRConfig
from src.utils.shared.alpha_grids import resolve_ridge_alphas, save_alpha_diagnostics
from src.utils.shared.config import (
    expand_fmri_layer_configs,
    load_config,
    parse_args,
    save_effective_config,
)
from src.utils.shared.encoding import extract_best_alphas, get_ridge_model_constructor
from src.utils.shared.fmri import (
    align_fmri_stimuli,
    build_fmri_stimulus,
    columnwise_correlation,
    load_fmri_responses,
    load_story_embeddings,
    resolve_fmri_stories,
    resolve_pca_sibling,
    session_cv_splits,
    target_manifest,
    TEMPORARILY_HELD_OUT_FMRI_STORIES,
)
from src.utils.rrr.core import fit_dense_rrr
from src.utils.subspace_comparison.metrics import (
    aggregate_subspace_basis,
    aggregate_subspace_from_unit_dir,
)
from src.utils.shared.tensors import to_numpy


@dataclass
class FinalRidgeFit:
    """Predictions and fitted quantities needed by downstream fMRI RRR."""

    test_prediction: np.ndarray
    train_prediction: np.ndarray | None
    coefficients: np.ndarray | None
    best_alphas: np.ndarray


@contextmanager
def _timed(label: str):
    start = time.perf_counter()
    print(f"[{label}] starting", flush=True)
    try:
        yield
    finally:
        print(f"[{label}] finished in {time.perf_counter() - start:.1f}s", flush=True)


def _array_summary(name: str, values: np.ndarray) -> str:
    size_gib = values.nbytes / (1024 ** 3)
    return f"{name}: shape={values.shape}, dtype={values.dtype}, size={size_gib:.2f} GiB"


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


def _fit_final_ridge(
    X_train,
    Y_train,
    X_test,
    cv,
    config,
    cache_dir,
    context,
    *,
    return_train_prediction=False,
    return_coefficients=False,
):
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
    train_prediction = None
    if return_train_prediction:
        train_prediction = to_numpy(model.predict(X_train)).astype(np.float32)
    coefficients = None
    if return_coefficients:
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
        coefficients=(
            np.asarray(coefficients, dtype=np.float32)
            if coefficients is not None
            else None
        ),
        best_alphas=selected,
    )


def _validate_cache_contract(output_dir: Path, config) -> None:
    """Prevent reuse of metrics produced by another source/aggregation mode."""
    path = output_dir / "metadata.json"
    if config.refresh:
        return
    if not path.exists():
        if config.subspace_aggregation != "extrinsic" and any(
            item.name != "config.yml" for item in output_dir.iterdir()
        ):
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


def _pca_right_singular_vectors(centered: np.ndarray, requested_device: str) -> np.ndarray:
    """Compute exact PCA right singular vectors on the requested device."""
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and torch.cuda.is_available()
    )
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("pca_svd_device='cuda' requested but CUDA is unavailable")
    if not use_cuda:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt

    try:
        tensor = torch.as_tensor(centered, device="cuda")
        _, _, vh = torch.linalg.svd(tensor, full_matrices=False)
        return vh.cpu().numpy()
    except RuntimeError:
        if requested_device != "auto":
            raise
        print("CUDA PCA SVD failed; retrying exact SVD on CPU", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt


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

    if ridge_fit.train_prediction is None or ridge_fit.coefficients is None:
        raise ValueError("Within-fMRI factor fitting requires full-ridge fitted values")
    rrr_result = fit_dense_rrr(
        full_train.stimulus,
        full_train.responses,
        k=max_rank,
        selected_idx=np.arange(expected_width),
        precomputed_Y_hat_ols=ridge_fit.train_prediction,
        precomputed_beta_full=ridge_fit.coefficients,
        svd_on_cpu=config.rrr_svd_on_cpu,
        use_low_rank_svd=config.rrr_use_low_rank_svd,
        low_rank_svd_seed=config.rrr_low_rank_svd_seed,
        return_b=config.rrr_save_b_matrix,
    )
    fir_a = np.asarray(rrr_result["A_dense"], dtype=np.float32)
    delay_blocks = list(np.split(fir_a, ndelays, axis=0))
    source_dir = output_dir / "source_factors"
    source_dir.mkdir(parents=True, exist_ok=True)
    np.save(source_dir / "full_fir_A.npy", fir_a)
    if config.rrr_save_b_matrix:
        np.save(source_dir / "full_B.npy", rrr_result["B_dense"])
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
    pca_device = getattr(config, "pca_svd_device", "auto")
    with _timed(f"exact PCA SVD ({pca_device})"):
        vt = _pca_right_singular_vectors(centered, pca_device)
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
        "rrr_b": str(source_dir / "full_B.npy") if config.rrr_save_b_matrix else None,
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


def _subject_summary(values, manifest):
    return {
        subject: _fisher_mean(values[manifest["subject"].to_numpy() == subject])
        for subject in manifest["subject"].unique()
    }


def _load_metric_arrays(path: Path, n_targets: int):
    if not path.exists():
        return None
    with np.load(path) as saved:
        corr = np.asarray(saved["corr_per_target"])
        r2 = np.asarray(saved["r2_per_target"])
        best_alphas = (
            np.asarray(saved["best_alphas"])
            if "best_alphas" in saved.files
            else None
        )
    if corr.shape != (n_targets,) or r2.shape != (n_targets,):
        raise ValueError(
            f"Cached metrics in {path} have target shapes {corr.shape}/{r2.shape}; "
            f"expected {(n_targets,)}"
        )
    return corr, r2, best_alphas


def _within_factor_metadata(
    output_dir: Path, ranks: list[int], ndelays: int, save_b: bool = False
) -> dict:
    source_dir = output_dir / "source_factors"
    return {
        "rrr": str(source_dir / "full_fir_A.npy"),
        "rrr_b": str(source_dir / "full_B.npy") if save_b else None,
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


def _load_complete_within_bases(
    output_dir: Path,
    ranks: list[int],
    ndelays: int,
    save_b: bool = False,
    n_targets: int | None = None,
):
    metadata = _within_factor_metadata(output_dir, ranks, ndelays, save_b)
    required = [metadata["rrr"], metadata["pca"]]
    if metadata["rrr_b"] is not None:
        required.append(metadata["rrr_b"])
    required.extend(metadata["rrr_delay_blocks"])
    required.extend(metadata["rrr_rank_bases"].values())
    required.extend(metadata["pca_rank_bases"].values())
    if not all(Path(path).exists() for path in required):
        return None, None
    max_rank = max(ranks)
    pca_full = np.load(metadata["pca"], mmap_mode="r")
    ambient_dim = pca_full.shape[0]
    if pca_full.ndim != 2 or pca_full.shape[1] < max_rank:
        raise ValueError(f"Invalid cached PCA factor shape {pca_full.shape}")
    fir_full = np.load(metadata["rrr"], mmap_mode="r")
    if fir_full.shape != (ndelays * ambient_dim, max_rank):
        raise ValueError(f"Invalid cached FIR-RRR factor shape {fir_full.shape}")
    for path in metadata["rrr_delay_blocks"]:
        shape = np.load(path, mmap_mode="r").shape
        if shape != (ambient_dim, max_rank):
            raise ValueError(f"Invalid cached delay-factor shape {shape} in {path}")
    if metadata["rrr_b"] is not None:
        b_shape = np.load(metadata["rrr_b"], mmap_mode="r").shape
        expected_b = (n_targets, max_rank) if n_targets is not None else None
        if len(b_shape) != 2 or b_shape[1] != max_rank or (
            expected_b is not None and b_shape != expected_b
        ):
            raise ValueError(f"Invalid cached target-factor shape {b_shape}")
    bases = {
        "rrr": {
            rank: np.load(metadata["rrr_rank_bases"][str(rank)]) for rank in ranks
        },
        "pca": {
            rank: np.load(metadata["pca_rank_bases"][str(rank)]) for rank in ranks
        },
    }
    for method, rank_bases in bases.items():
        for rank, basis in rank_bases.items():
            if basis.shape != (ambient_dim, rank):
                raise ValueError(
                    f"Invalid cached {method.upper()} rank-{rank} basis shape "
                    f"{basis.shape}"
                )
    return bases, metadata


def _build_rank_designs(aligned, train_stories, test_stories, basis, ndelays):
    train = build_fmri_stimulus(
        aligned, train_stories, ndelays=ndelays, basis=basis
    )
    test = build_fmri_stimulus(
        aligned, test_stories, ndelays=ndelays, basis=basis
    )
    return train.stimulus, test.stimulus


def _iter_rank_designs(jobs, aligned, train_stories, test_stories, ndelays, prefetch):
    """Yield rank designs, optionally preparing one design ahead."""
    if not prefetch:
        for method, rank, basis in jobs:
            yield method, rank, *_build_rank_designs(
                aligned, train_stories, test_stories, basis, ndelays
            )
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for index, (method, rank, basis) in enumerate(jobs):
            if future is None:
                future = executor.submit(
                    _build_rank_designs,
                    aligned,
                    train_stories,
                    test_stories,
                    basis,
                    ndelays,
                )
            train, test = future.result()
            if index + 1 < len(jobs):
                _, _, next_basis = jobs[index + 1]
                future = executor.submit(
                    _build_rank_designs,
                    aligned,
                    train_stories,
                    test_stories,
                    next_basis,
                    ndelays,
                )
            else:
                future = None
            yield method, rank, train, test


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
    backend_name = "torch_cuda" if torch.cuda.is_available() else "numpy"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Backend: {backend_name} ({device_name})", flush=True)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, output_dir)
    _validate_cache_contract(output_dir, config)
    previous_metadata = {}
    if (output_dir / "metadata.json").exists():
        with (output_dir / "metadata.json").open() as handle:
            previous_metadata = json.load(handle)
    ranks = list(map(int, config.rank_grid))
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
    masks = _load_masks(config)
    manifest = target_manifest(config.fmri.subjects, masks)
    manifest.to_csv(output_dir / "target_manifest.csv", index=False)
    n_targets = len(manifest)

    method_names = ["rrr", "pca"] if config.source_mode == "within_fmri" else list(source_unit_dirs)
    missing_jobs = [
        (method, rank)
        for method in method_names
        for rank in ranks
        if config.refresh or not (output_dir / method / f"k{rank}" / "metrics.npz").exists()
    ]
    full_metrics_path = output_dir / "full" / "metrics.npz"
    cached_full = None if config.refresh else _load_metric_arrays(full_metrics_path, n_targets)

    bases_by_method = None
    factor_metadata = None
    factors_missing = False
    if config.source_mode == "within_fmri" and not config.refresh:
        bases_by_method, factor_metadata = _load_complete_within_bases(
            output_dir,
            ranks,
            int(config.fmri.ndelays),
            config.rrr_save_b_matrix,
            n_targets,
        )
    if config.source_mode == "within_fmri" and bases_by_method is None:
        factors_missing = True

    need_full_fit = cached_full is None or factors_missing
    need_data = need_full_fit or bool(missing_jobs)
    aligned = full_train = full_test = cv = None
    full_fit = None
    ambient_dim = None
    if need_data:
        all_stories = train_stories + test_stories
        with _timed("load pooled story embeddings"):
            embeddings = load_story_embeddings(
                config.fmri.embedding_root,
                config.model_dir,
                all_stories,
                config.layer,
                config.pooling,
            )
        ambient_dim = next(iter(embeddings.values())).shape[1]
        with _timed("Lanczos-align all stories"):
            aligned = align_fmri_stimuli(
                config.fmri.fmri_repo_root,
                all_stories,
                embeddings,
                trim=config.fmri.trim,
            )
        del embeddings
        full_train = build_fmri_stimulus(
            aligned, train_stories, ndelays=config.fmri.ndelays
        )
        full_test = build_fmri_stimulus(
            aligned, test_stories, ndelays=config.fmri.ndelays
        )
        print(_array_summary("full train design", full_train.stimulus), flush=True)
        with _timed("load selected train responses"):
            full_train.responses = load_fmri_responses(
                config.fmri.fmri_repo_root,
                train_stories,
                full_train.story_slices,
                config.fmri.subjects,
                masks,
            )
        with _timed("load selected test responses"):
            full_test.responses = load_fmri_responses(
                config.fmri.fmri_repo_root,
                test_stories,
                full_test.story_slices,
                config.fmri.subjects,
                masks,
            )
        print(_array_summary("train responses", full_train.responses), flush=True)
        cv = session_cv_splits(full_train.story_slices, groups)

    if need_full_fit:
        with _timed("fit full-embedding ridge ceiling"):
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
                return_train_prediction=config.source_mode == "within_fmri",
                return_coefficients=config.source_mode == "within_fmri",
            )
        full_corr_arr = columnwise_correlation(
            full_test.responses, full_fit.test_prediction
        )
        full_r2_arr = _r2_per_target(full_test.responses, full_fit.test_prediction)
        full_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            full_metrics_path,
            corr_per_target=full_corr_arr,
            r2_per_target=full_r2_arr,
            best_alphas=full_fit.best_alphas,
        )
    else:
        full_corr_arr, full_r2_arr, _ = cached_full
        print("Reusing cached full-embedding ridge metrics", flush=True)
    full_corr = _fisher_mean(full_corr_arr)
    full_r2 = float(np.mean(full_r2_arr))

    if factors_missing:
        with _timed("fit within-fMRI RRR and PCA bases"):
            bases_by_method, factor_metadata = _fit_within_fmri_bases(
                full_train, full_fit, ambient_dim, config, output_dir
            )
    elif config.source_mode == "cross_dataset" and missing_jobs:
        bases_by_method = {
            method: {
                rank: aggregate_subspace_from_unit_dir(
                    unit_dir,
                    rank=rank,
                    method=config.subspace_aggregation,
                )
                for rank in ranks
                if (method, rank) in missing_jobs
            }
            for method, unit_dir in source_unit_dirs.items()
        }
        factor_metadata = dict(source_unit_dirs)
    elif config.source_mode == "cross_dataset":
        bases_by_method = {}
        factor_metadata = dict(source_unit_dirs)

    jobs = [
        (method, rank, bases_by_method[method][rank])
        for method, rank in missing_jobs
    ]
    for method, rank, basis in jobs:
        if basis.shape != (ambient_dim, rank):
            raise ValueError(
                f"{method.upper()} rank {rank} basis has shape {basis.shape}; "
                f"expected ({ambient_dim}, {rank})"
            )
    computed_metrics = {}
    for method, rank, train_design, test_design in _iter_rank_designs(
        jobs,
        aligned,
        train_stories,
        test_stories,
        int(config.fmri.ndelays),
        bool(config.prefetch_rank_designs),
    ):
        metrics_path = output_dir / method / f"k{rank}" / "metrics.npz"
        if train_design.shape[1] != int(config.fmri.ndelays) * rank:
            raise ValueError(
                f"{method.upper()} rank {rank} FIR design has shape "
                f"{train_design.shape}"
            )
        with _timed(f"fit {method.upper()} rank={rank}"):
            fitted = _fit_final_ridge(
                train_design,
                full_train.responses,
                test_design,
                cv,
                config,
                metrics_path.parent,
                {
                    "script": "fmri_cross_dataset_rrr",
                    "method": method,
                    "rank": rank,
                    "n_features": train_design.shape[1],
                    "standardize": False,
                },
            )
        corr_arr = columnwise_correlation(
            full_test.responses, fitted.test_prediction
        )
        r2_arr = _r2_per_target(full_test.responses, fitted.test_prediction)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            metrics_path,
            corr_per_target=corr_arr,
            r2_per_target=r2_arr,
            best_alphas=fitted.best_alphas,
        )
        computed_metrics[(method, rank)] = (corr_arr, r2_arr)
        print(
            f"{method.upper()} rank={rank}: "
            f"r={_fisher_mean(corr_arr):.4f}, R2={float(np.mean(r2_arr)):.4f}",
            flush=True,
        )

    rows = []
    for method in method_names:
        for rank in ranks:
            metrics_path = output_dir / method / f"k{rank}" / "metrics.npz"
            if (method, rank) in computed_metrics:
                corr_arr, r2_arr = computed_metrics[(method, rank)]
            else:
                loaded = _load_metric_arrays(metrics_path, n_targets)
                if loaded is None:
                    raise FileNotFoundError(f"Missing expected rank metrics: {metrics_path}")
                corr_arr, r2_arr, _ = loaded
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
            if (method, rank) not in computed_metrics:
                print(
                    f"{method.upper()} rank={rank} (cached): "
                    f"r={row['corr']:.4f}, R2={row['r2']:.4f}",
                    flush=True,
                )

    frame = pd.DataFrame(rows).sort_values(["method", "rank"])
    frame.to_csv(output_dir / "grid_results.csv", index=False)
    _plot_results(frame, output_dir / "rank_analysis.png")
    metadata = {
        "source_factors": factor_metadata,
        "source_mode": config.source_mode,
        "subspace_aggregation": config.subspace_aggregation,
        "rrr_svd_on_cpu": config.rrr_svd_on_cpu,
        "rrr_use_low_rank_svd": config.rrr_use_low_rank_svd,
        "rrr_low_rank_svd_seed": config.rrr_low_rank_svd_seed,
        "rrr_save_b_matrix": config.rrr_save_b_matrix,
        "pca_svd_device": config.pca_svd_device,
        "prefetch_rank_designs": config.prefetch_rank_designs,
        "pca_available": "pca" in method_names,
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
        "pre_fir_training_shape": (
            list(full_train.pre_fir_stimulus.shape)
            if full_train is not None
            else previous_metadata.get("pre_fir_training_shape")
        ),
        "rank_grid": ranks,
        "ceiling": "fresh_full_embedding_ridge",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args, overrides = parse_args(__doc__)
    config = load_config(args.config, overrides, FMRICrossDatasetRRRConfig)
    for layer_config in expand_fmri_layer_configs(config):
        run_config(layer_config)


if __name__ == "__main__":
    main()
