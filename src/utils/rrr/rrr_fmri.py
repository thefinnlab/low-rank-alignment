"""fMRI reduced-rank regression implementation for the unified client."""

import io
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from himalaya.backend import set_backend

from src.core.config import FMRICrossDatasetRRRConfig
from src.utils.shared.alpha_grids import resolve_ridge_alphas
from src.utils.shared.config import (
    expand_fmri_layer_configs,
    load_config,
    parse_args,
)
from src.utils.shared.encoding import extract_best_alphas, get_ridge_model_constructor
from src.utils.shared.fmri import (
    align_fmri_stimuli,
    build_fmri_stimulus,
    columnwise_correlation,
    load_fmri_responses,
    load_story_embeddings,
    resolve_fmri_stories,
    session_cv_splits,
    target_manifest,
    TEMPORARILY_HELD_OUT_FMRI_STORIES,
)
from src.utils.rrr.core import fit_dense_rrr
from src.utils.rrr.significance import load_significance_selection
from src.utils.subspace_comparison.metrics import aggregate_subspace_basis
from src.utils.shared.tensors import to_numpy


from scripts.run_rrr import Artifact
from scripts.run_rrr import Artifact as IEEGRRRArtifact


@dataclass
class FinalRidgeFit:
    """Predictions and fitted quantities needed by downstream fMRI RRR."""

    test_prediction: np.ndarray
    train_prediction: np.ndarray | None
    coefficients: np.ndarray | None
    best_alphas: np.ndarray
    alpha_grid: np.ndarray


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
    context,
    *,
    return_train_prediction=False,
    return_coefficients=False,
):
    explicit = np.asarray(config.ridge.alphas) if config.ridge.alphas else None
    alphas = resolve_ridge_alphas(
        explicit,
        X_train,
        standardize=False,
        context=context,
        cache_dir=None,
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
    return FinalRidgeFit(
        test_prediction=test_prediction,
        train_prediction=train_prediction,
        coefficients=(
            np.asarray(coefficients, dtype=np.float32)
            if coefficients is not None
            else None
        ),
        best_alphas=selected,
        alpha_grid=np.asarray(alphas),
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


def _fit_within_fmri_bases(full_train, ridge_fit, ambient_dim, config):
    """Derive shared FIR-RRR and pre-FIR PCA bases as scientific values."""
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
    rrr_bases = {}
    for rank in ranks:
        basis = aggregate_subspace_basis(
            delay_blocks, rank=rank, method=config.subspace_aggregation
        )
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
    pca_bases = {}
    for rank in ranks:
        basis = pca_a[:, :rank]
        pca_bases[rank] = basis
    factors = {
        "full_fir": fir_a,
        "target": (rrr_result.get("B_dense") if config.rrr_save_b_matrix else None),
        "delay_blocks": delay_blocks,
        "pca_full": pca_a,
    }
    return {"rrr": rrr_bases, "pca": pca_bases}, factors


def _subject_summary(values, manifest):
    return {
        subject: _fisher_mean(values[manifest["subject"].to_numpy() == subject])
        for subject in manifest["subject"].unique()
    }


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


def _plot_results(frame: pd.DataFrame):
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
    return fig


def run_fmri(config: FMRICrossDatasetRRRConfig) -> Artifact:
    config.validate()
    if torch.cuda.is_available():
        set_backend("torch_cuda")
    backend_name = "torch_cuda" if torch.cuda.is_available() else "numpy"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Backend: {backend_name} ({device_name})", flush=True)
    ranks = list(map(int, config.rank_grid))
    significance = load_significance_selection(
        config.significance_selection,
        results_root=config.results_root,
        subjects=config.fmri.subjects,
        modality="fmri",
    )
    significance.significant_voxels = {
        subject: frame.rename(columns={"target_id": "voxel_index",
                                       "significant": "sig"})
        for subject, frame in significance.target_significance.items()
    }
    upstreams = [significance.reference.identity_record()]
    for identity in significance.source_identities:
        if identity not in upstreams:
            upstreams.append(identity)
    source = None
    if config.source_mode == "cross_dataset":
        source_kwargs = dict(config.ieeg_rrr_source or {})
        source_kwargs.setdefault("results_root", config.results_root)
        source = IEEGRRRArtifact(**source_kwargs)
        if source.variant is None:
            source._resolve_variant()
        source.manifest()
        upstreams.append(source.identity_record())
    artifact = Artifact.from_fmri_config(config)
    artifact.begin(config=config, upstreams=upstreams, refresh=config.refresh)
    artifact.save_field("effective_config", asdict(config), overwrite=config.refresh)
    artifact.save_field("upstream_identities", upstreams, overwrite=config.refresh)

    train_stories, test_stories, groups = resolve_fmri_stories(
        config.fmri.fmri_repo_root, config.fmri.sessions
    )
    if len(test_stories) != 1:
        raise ValueError(f"Expected one shared test story, got {test_stories}")
    masks = {}
    for subject in config.fmri.subjects:
        frame = significance.significant_voxels[str(subject)].sort_values("voxel_index")
        expected = np.arange(len(frame))
        if not np.array_equal(frame["voxel_index"].to_numpy(), expected):
            raise ValueError(f"Non-contiguous voxel indices for {subject}")
        masks[subject] = frame["sig"].astype(bool).to_numpy()
        if not masks[subject].any():
            raise ValueError(f"No significant voxels found for {subject}")
    manifest = target_manifest(config.fmri.subjects, masks)
    artifact.save_field("target_manifest", manifest, overwrite=config.refresh)
    n_targets = len(manifest)

    method_names = ["rrr", "pca"] if config.source_mode == "within_fmri" else ["rrr"]
    if source is not None and source.load_collection("input_factors", {"method": "pca"}):
        method_names.append("pca")
    plan = artifact.resume_method_plan(method_names, ranks, refresh=config.refresh)
    missing_jobs = plan["pending"]
    cached_full = None
    if not config.refresh and artifact.has_field("full_metrics"):
        cached_full = artifact.load_field("full_metrics")

    bases_by_method = None
    factors_missing = config.source_mode == "within_fmri" and any(
        config.refresh or not artifact.has_field(
            "input_factors", selector=artifact._selector(method=method, rank=rank)
        ) for method in method_names for rank in ranks
    )

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
        artifact.save_field("full_metrics", {
            "corr_per_target": full_corr_arr, "r2_per_target": full_r2_arr,
            "best_alphas": full_fit.best_alphas,
        }, overwrite=config.refresh)
        if config.ridge.log_alphas:
            artifact.save_field("diagnostics", {
                "alpha_grid": full_fit.alpha_grid,
                "best_alphas": full_fit.best_alphas,
            }, selector={"method": "full"}, overwrite=config.refresh)
    else:
        full_corr_arr = np.asarray(cached_full["corr_per_target"])
        full_r2_arr = np.asarray(cached_full["r2_per_target"])
        print("Reusing cached full-embedding ridge metrics", flush=True)
    full_corr = _fisher_mean(full_corr_arr)
    full_r2 = float(np.mean(full_r2_arr))

    if factors_missing:
        with _timed("fit within-fMRI RRR and PCA bases"):
            bases_by_method, factor_values = _fit_within_fmri_bases(
                full_train, full_fit, ambient_dim, config
            )
        for method, rank_bases in bases_by_method.items():
            for rank, basis in rank_bases.items():
                artifact.save_factors(method=method, rank=rank, factors=basis,
                                      overwrite=config.refresh)
        artifact.save_factors(method="rrr_full_fir", factors=factor_values["full_fir"],
                              target_factors=factor_values["target"], overwrite=config.refresh)
        artifact.save_factors(method="pca_full", factors=factor_values["pca_full"],
                              overwrite=config.refresh)
        for delay, block in enumerate(factor_values["delay_blocks"], start=1):
            artifact.save_factors(method="rrr_delay", delay=delay, factors=block,
                                  overwrite=config.refresh)
    elif config.source_mode == "cross_dataset" and missing_jobs:
        bases_by_method = {
            method: {
                rank: aggregate_subspace_basis([
                    value for selector, value in source.load_collection(
                        "input_factors", {"method": method}
                    ).items() if "fold" in dict(selector)
                ], rank=rank, method=config.subspace_aggregation)
                for rank in ranks
                if (method, rank) in missing_jobs
            }
            for method in method_names
        }
    elif config.source_mode == "cross_dataset":
        bases_by_method = {}
    elif not factors_missing:
        bases_by_method = {
            method: {rank: artifact.load_field(
                "input_factors", selector=artifact._selector(method=method, rank=rank)
            ) for rank in ranks if (method, rank) in missing_jobs}
            for method in method_names
        }

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
        artifact.save_rank_metrics(method=method, rank=rank, metrics={
            "corr_per_target": corr_arr, "r2_per_target": r2_arr,
            "best_alphas": fitted.best_alphas,
        }, overwrite=config.refresh)
        if config.ridge.log_alphas:
            artifact.save_field("diagnostics", {
                "alpha_grid": fitted.alpha_grid,
                "best_alphas": fitted.best_alphas,
            }, selector={"method": method, "rank": rank},
                overwrite=config.refresh)
        computed_metrics[(method, rank)] = (corr_arr, r2_arr)
        print(
            f"{method.upper()} rank={rank}: "
            f"r={_fisher_mean(corr_arr):.4f}, R2={float(np.mean(r2_arr)):.4f}",
            flush=True,
        )

    rows = []
    for method in method_names:
        for rank in ranks:
            if (method, rank) in computed_metrics:
                corr_arr, r2_arr = computed_metrics[(method, rank)]
            else:
                loaded = artifact.load_rank_metrics(method=method, rank=rank)
                corr_arr = np.asarray(loaded["corr_per_target"])
                r2_arr = np.asarray(loaded["r2_per_target"])
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
    artifact.save_field("mean_results", frame, overwrite=True)
    if config.plot:
        figure = _plot_results(frame)
        artifact.save_plot(name="rank_analysis", figure=figure,
                           overwrite=True)
        plt.close(figure)
    metadata = {
        "source_artifact": source.identity_record() if source is not None else None,
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
            else None
        ),
        "rank_grid": ranks,
        "ceiling": "fresh_full_embedding_ridge",
    }
    artifact.save_field("metadata", metadata, overwrite=True)
    required = {"rank_metrics": [artifact._selector(method=method, rank=rank)
                                  for method in method_names for rank in ranks]}
    artifact.mark_complete(required)
    return artifact
