"""Ordered leakage-aware comparisons of saved embedding-space factors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from himalaya.backend import get_backend, set_backend
from himalaya.ridge import RidgeCV as HimalayaRidgeCV
from scipy.stats import sem

from src.core.config import SubspaceComparisonConfig, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.shared.constants import RIDGE_ALPHAS
from src.utils.shared.ieeg import selection_after_offset
from src.utils.shared.embeddings import load_embeddings
from src.utils.shared.encoding import apply_offset, load_encoding_results
from src.utils.shared.fmri import load_story_embeddings, resolve_fmri_stories
from src.utils.language_interpret.reconstruction import orthonormal_factor
from src.utils.subspace_comparison.metrics import aggregate_subspace_basis
from src.core.artifacts import normalize_dataset
from scripts.run_rrr import Artifact as RRRArtifact
from scripts.run_significance import Artifact as SignificanceArtifact


DIRECTED_METRICS = (
    "ridge_r2",
    "ridge_coordinate_correlation",
    "ridge_prediction_cka",
    "ridge_normalized_error",
    "ridge_alpha",
    "procrustes_r2",
    "procrustes_normalized_mse",
    "procrustes_rmse",
    "procrustes_scale",
    "procrustes_mean_row_cosine",
)
SYMMETRIC_METRICS = (
    "direct_cka",
    "rsa_pearson",
    "rsa_spearman",
)
ALL_METRICS = DIRECTED_METRICS + SYMMETRIC_METRICS
METRIC_IMPLEMENTATION = "torch_cuda_himalaya_shared_alpha_5fold_v2"


def configured_models(config):
    """Return ordered ``(model, specification)`` pairs for either config syntax."""
    if isinstance(config.models, dict):
        return list(config.models.items())
    return [(item["model"], {"offsets": [0], **item}) for item in config.models]

@dataclass(frozen=True)
class EvaluationView:
    """One model/layer/offset representation on the evaluation stimulus."""

    ids: np.ndarray
    values: np.ndarray


@dataclass
class EvaluationData:
    """Dataset-specific representation loader and leakage-safe comparison folds."""

    alias: str
    fold_test_ids: list[np.ndarray]
    load_view: Callable[[str, int, int], EvaluationView]
    metadata: dict


@dataclass
class ComparisonSpace:
    """One configured saved subspace and its evaluation-row provenance."""

    dataset: str
    offset: int
    model: str
    layer: int
    kind: str
    result_root: Path
    unit_dir: Path
    source_format: str
    ranks: set[int]
    heldout_ids: list[np.ndarray] | None = None
    artifact: RRRArtifact | None = None

    @property
    def label(self) -> str:
        return (
            f"{self.dataset} | o{self.offset} | {self.model} | "
            f"l{self.layer} | {self.kind.upper()}"
        )

    @property
    def key(self) -> tuple:
        return (self.dataset, self.offset, self.model, self.layer, self.kind)


def _dataset_context(spec: dict):
    subjects = spec.get("subjects") or list(range(1, 10))
    dataset = spec.get("dataset", "podcast")
    context = resolve_dataset_context(
        dataset=dataset,
        subject_ids=subjects,
        per_dataset_config=spec.get("per_dataset_config", {}),
        data_root=spec.get("data_root", "data/"),
    )
    return subjects, dataset, context


def resolve_result_root(
    dataset_alias: str,
    dataset_spec: dict,
    model: str,
    layer: int,
    offset: int,
) -> Path:
    """Resolve a configured iEEG fold run or fMRI ambient-basis run."""
    results_root = dataset_spec.get("results_root", "results")
    loader = dataset_spec.get("loader", "ieeg_words")
    if loader == "fmri_words":
        if int(offset) != 0:
            raise ValueError(
                f"fMRI-generated spaces currently require offset 0, got {offset} "
                f"for {model} layer {layer}"
            )
        return Path(results_root) / "fmri" / f"{model}_layer{layer}" / "within_fmri_rrr"
    _, _, context = _dataset_context(dataset_spec)
    return Path(
        get_results_dir(
            "transformer",
            int(offset),
            {"model_dir": model, "layer": int(layer)},
            base_dir=str(Path(results_root)) + "/",
            dataset_context=context,
        )
    )


def _completed_fold_ranks(unit_dir: Path) -> set[int]:
    factor_paths = sorted(unit_dir.glob("fold_*/A.npy"))
    if not factor_paths:
        raise FileNotFoundError(f"No fold_*/A.npy factors found under {unit_dir}")
    fold_ids = {int(path.parent.name.split("_", 1)[1]) for path in factor_paths}
    grid_path = unit_dir / "grid_results.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing completed-rank table: {grid_path}")
    grid = pd.read_csv(grid_path)
    if not {"fold", "k"}.issubset(grid.columns):
        raise ValueError(f"{grid_path} must contain fold and k columns")
    completed = {
        int(rank)
        for rank, rows in grid.groupby("k")
        if set(map(int, rows["fold"].unique())) == fold_ids
    }
    if not completed:
        raise ValueError(f"No ranks are complete across folds under {unit_dir}")
    return completed


def _completed_rank_basis_ranks(unit_dir: Path) -> set[int]:
    ranks = set()
    for path in unit_dir.glob("k*/A.npy"):
        suffix = path.parent.name.removeprefix("k")
        if suffix.isdigit():
            ranks.add(int(suffix))
    if not ranks:
        raise FileNotFoundError(f"No k*/A.npy rank bases found under {unit_dir}")
    return ranks


def _encoding_result_path(result_root: Path, subjects: list) -> Path:
    if not subjects:
        raise ValueError("At least one subject is required for iEEG fold provenance")
    return result_root / "subjects" / f"sub_{subjects[0]}" / "encoding_results.joblib"


def load_ieeg_heldout_ids(
    result_root: Path,
    subjects: list,
) -> list[np.ndarray]:
    """Load fold test IDs in canonical transcript coordinates."""
    path = _encoding_result_path(result_root, subjects)
    result = load_encoding_results(str(path))
    metadata = result.get("word_selection")
    if metadata is None or "analysis_selection" not in metadata:
        raise ValueError(f"Missing analysis_selection word provenance in {path}")
    ids = np.asarray(metadata["analysis_selection"], dtype=int)
    test_idxs = [np.asarray(test, dtype=int) for test in result["test_idxs"]]
    if not test_idxs:
        raise ValueError(f"No saved test folds found in {path}")
    if any(len(test) and int(np.max(test)) >= len(ids) for test in test_idxs):
        raise ValueError(
            f"Saved fold indices in {path} exceed the {len(ids)} aligned evaluation rows"
        )
    concatenated = np.concatenate(test_idxs)
    if len(concatenated) != len(ids) or not np.array_equal(
        np.sort(concatenated), np.arange(len(ids), dtype=int)
    ):
        raise ValueError(
            f"Saved test folds in {path} must partition all {len(ids)} evaluation "
            "rows exactly once"
        )
    return [ids[test] for test in test_idxs]


def expand_spaces(config: SubspaceComparisonConfig) -> list[ComparisonSpace]:
    """Expand model configuration into deterministically ordered space nodes."""
    spaces = []
    missing = []
    kinds = ["rrr", "pca"] if config.include_pca else ["rrr"]
    dataset_order = {name: i for i, name in enumerate(config.datasets)}
    model_specs = configured_models(config)
    model_order = {name: i for i, (name, _) in enumerate(model_specs)}
    for dataset_alias in config.datasets:
        dataset_spec = config.datasets[dataset_alias]
        for model, model_spec in model_specs:
            if dataset_alias not in model_spec["datasets"]:
                continue
            for offset in sorted(map(int, model_spec["offsets"])):
                for layer in sorted(map(int, model_spec["layers"])):
                    try:
                        if dataset_spec.get("loader", "ieeg_words") == "fmri_words":
                            artifact_dataset = "deep-fmri"
                        else:
                            _, raw_dataset, context = _dataset_context(dataset_spec)
                            artifact_dataset = normalize_dataset(raw_dataset, context.stimulus_id)
                        artifact = RRRArtifact(
                            mode="internal", dataset=artifact_dataset, offset=int(offset),
                            model=model, layer=int(layer), results_root=dataset_spec.get(
                                "results_root", "results"
                            ),
                        )
                        artifact._resolve_variant()
                        factors = artifact.load_collection("input_factors")
                        source_format = (
                            "fold_factors"
                            if any("fold" in dict(selector) for selector in factors)
                            else "rank_bases"
                        )
                        root = artifact._variant_dir()
                    except Exception as exc:
                        missing.append(f"{dataset_alias}/{model}/l{layer}/o{offset}: {exc}")
                        continue
                    for kind in kinds:
                        try:
                            rank_entries = artifact.load_collection(
                                "rank_metrics", {"method": kind}
                            )
                            ranks = {int(dict(selector)["rank"])
                                     for selector in rank_entries}
                            if not ranks:
                                raise ValueError("no completed rank metrics")
                        except Exception as exc:
                            missing.append(
                                f"{dataset_alias}/{model}/l{layer}/o{offset}/{kind}: {exc}"
                            )
                            continue
                        spaces.append(
                            ComparisonSpace(
                                dataset=dataset_alias,
                                offset=offset,
                                model=model,
                                layer=layer,
                                kind=kind,
                                result_root=root,
                                unit_dir=root,
                                source_format=source_format,
                                ranks=ranks,
                                heldout_ids=None,
                                artifact=artifact,
                            )
                        )
    if missing:
        raise FileNotFoundError(
            "Missing or invalid configured subspace results:\n- " + "\n- ".join(missing)
        )
    if len(spaces) < 2:
        raise ValueError("At least two resolved spaces are required")
    spaces.sort(
        key=lambda s: (
            dataset_order[s.dataset],
            s.offset,
            model_order[s.model],
            s.layer,
            0 if s.kind == "rrr" else 1,
        )
    )
    return spaces


def shared_ranks(spaces: list[ComparisonSpace], max_rank: int) -> list[int]:
    """Return the complete global rank intersection through an inclusive cap."""
    ranks = set.intersection(*(space.ranks for space in spaces))
    selected = sorted(rank for rank in ranks if rank <= int(max_rank))
    if not selected:
        raise ValueError(f"No globally shared completed ranks are <= max_rank={max_rank}")
    return selected


def _encoding_result(space: ComparisonSpace, subject):
    upstreams = space.artifact.load_field("upstream_identities")
    identity = next((item for item in upstreams
                     if item.get("script") == "run_significance"), None)
    if identity is None:
        raise ValueError(f"{space.label} has no significance upstream identity")
    kwargs = {key: value for key, value in identity.items() if key != "script"}
    kwargs["results_root"] = space.artifact.results_root
    artifact = SignificanceArtifact(**kwargs)
    return artifact.load_field("encoding_results", selector={"subject": str(subject)})


def _load_reference_ieeg(config: SubspaceComparisonConfig, spaces) -> EvaluationData:
    alias = config.evaluation_dataset
    spec = config.datasets[alias]
    reference = next(
        (space for space in spaces if space.dataset == alias and space.kind == "rrr"),
        None,
    )
    if reference is None:
        raise ValueError(
            "iEEG evaluation requires at least one RRR space generated on the "
            f"evaluation dataset {alias!r}"
        )
    subjects, dataset, context = _dataset_context(spec)
    get_dataset_adapter(dataset).validate_subjects_share_stimulus(
        subjects,
        spec.get("per_dataset_config", {}),
        spec.get("data_root", "data/"),
    )
    reference_result_path = reference.label
    reference_result = _encoding_result(reference, subjects[0])
    reference_provenance = reference_result.get("word_selection")
    if reference_provenance is None or "epoch_selection" not in reference_provenance:
        raise ValueError(
            f"Missing epoch_selection word provenance in {reference_result_path}"
        )
    epoch_selection = np.asarray(reference_provenance["epoch_selection"], dtype=int)
    if epoch_selection.ndim != 1 or len(np.unique(epoch_selection)) != len(epoch_selection):
        raise ValueError(
            f"epoch_selection in {reference_result_path} must contain unique 1D IDs"
        )

    heldout_by_run = {}
    for space in spaces:
        if space.dataset != alias or space.source_format != "fold_factors":
            continue
        run_key = (space.artifact.variant, space.offset)
        if run_key in heldout_by_run:
            space.heldout_ids = heldout_by_run[run_key]
            continue
        path = space.label
        result = _encoding_result(space, subjects[0])
        provenance = result.get("word_selection")
        required = {"epoch_selection", "analysis_selection"}
        if provenance is None or not required.issubset(provenance):
            raise ValueError(
                f"Missing required {sorted(required)} word provenance in {path}"
            )
        own_epoch_selection = np.asarray(provenance["epoch_selection"], dtype=int)
        if not np.array_equal(own_epoch_selection, epoch_selection):
            raise ValueError(
                f"Evaluation-dataset epoch_selection differs in {path}; all runs "
                "must use the same pre-offset stimulus rows"
            )
        expected_analysis = selection_after_offset(epoch_selection, int(space.offset))
        if not np.array_equal(
            np.asarray(provenance["analysis_selection"], dtype=int),
            expected_analysis,
        ):
            raise ValueError(
                f"analysis_selection in {path} is inconsistent with offset {space.offset}"
            )
        ids = np.asarray(provenance["analysis_selection"], dtype=int)
        space.heldout_ids = [ids[np.asarray(test, dtype=int)]
                             for test in result["test_idxs"]]
        heldout_by_run[run_key] = space.heldout_ids

    @lru_cache(maxsize=None)
    def load_view(model: str, layer: int, offset: int) -> EvaluationView:
        values, _ = load_embeddings(
            spec.get("data_root", "data/"),
            "transformer",
            dataset=dataset,
            per_dataset_config=spec.get("per_dataset_config", {}),
            dataset_context=context,
            model_dir=model,
            layer=int(layer),
        )
        dummy = np.zeros((len(epoch_selection), 1), dtype=np.float32)
        aligned, _ = apply_offset(values[epoch_selection], dummy, int(offset))
        ids = selection_after_offset(epoch_selection, int(offset))
        return EvaluationView(ids=ids, values=np.asarray(aligned, dtype=np.float32))

    return EvaluationData(
        alias=alias,
        fold_test_ids=[np.asarray(ids, dtype=int) for ids in reference.heldout_ids],
        load_view=load_view,
        metadata={
            "loader": "ieeg_words",
            "reference_space": reference.label,
        },
    )


def _offset_story(values: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(len(values), dtype=int)
    if offset < 0:
        return values[:offset], ids[:offset]
    if offset > 0:
        return values[offset:], ids[offset:]
    return values, ids


def _load_fmri_words(config: SubspaceComparisonConfig) -> EvaluationData:
    alias = config.evaluation_dataset
    spec = config.datasets[alias]
    train_stories, test_stories, _ = resolve_fmri_stories(
        spec.get("fmri_repo_root", "data/deep-fMRI-dataset"),
        spec.get("sessions", [1, 2, 3, 4, 5]),
    )
    stories = list(spec.get("evaluation_stories") or test_stories)
    if not stories:
        raise ValueError("fMRI word evaluation resolved no held-out stories")
    unsafe = sorted(set(stories) - set(test_stories))
    if unsafe:
        raise ValueError(
            "fMRI word evaluation may use only stories held out from factor fitting; "
            f"not held out: {unsafe}"
        )

    @lru_cache(maxsize=None)
    def load_view(model: str, layer: int, offset: int) -> EvaluationView:
        by_story = load_story_embeddings(
            spec.get("embedding_root", "data/fmri/stimuli"),
            model,
            stories,
            int(layer),
            spec.get("pooling", "mean"),
        )
        ids, blocks = [], []
        for story in stories:
            block, word_ids = _offset_story(np.asarray(by_story[story]), int(offset))
            blocks.append(block)
            ids.extend(f"{story}:{word_id}" for word_id in word_ids)
        return EvaluationView(
            ids=np.asarray(ids, dtype=str),
            values=np.vstack(blocks).astype(np.float32, copy=False),
        )

    # Fold identities do not depend on the model. A model configured for fMRI
    # supplies the reference rows so missing evaluation embeddings fail early.
    reference_item = next(
        (
            (model, model_spec)
            for model, model_spec in configured_models(config)
            if alias in model_spec["datasets"]
        ),
        None,
    )
    if reference_item is None:
        raise ValueError(
            f"fMRI evaluation requires at least one model generated on {alias!r}"
        )
    model, model_spec = reference_item
    reference = load_view(model, int(model_spec["layers"][0]), int(model_spec["offsets"][0]))
    if len(stories) > 1:
        folds = [reference.ids[np.char.startswith(reference.ids, f"{story}:")] for story in stories]
        fold_mode = "leave_one_heldout_story_out"
    else:
        n_folds = min(int(spec.get("num_folds", 5)), len(reference.ids))
        if n_folds < 2:
            raise ValueError("fMRI word evaluation requires at least two comparison folds")
        folds = [np.asarray(block) for block in np.array_split(reference.ids, n_folds)]
        fold_mode = "contiguous_blocks_within_heldout_story"
    return EvaluationData(
        alias=alias,
        fold_test_ids=folds,
        load_view=load_view,
        metadata={
            "loader": "fmri_words",
            "factor_training_stories": train_stories,
            "evaluation_stories": stories,
            "fold_mode": fold_mode,
        },
    )


def load_evaluation_data(
    config: SubspaceComparisonConfig, spaces: list[ComparisonSpace]
) -> EvaluationData:
    """Dispatch evaluation loading without coupling comparison code to a dataset."""
    loader = config.datasets[config.evaluation_dataset].get("loader", "ieeg_words")
    if loader == "ieeg_words":
        return _load_reference_ieeg(config, spaces)
    if loader == "fmri_words":
        return _load_fmri_words(config)
    raise ValueError(f"Unsupported evaluation loader: {loader}")


def _load_raw_factor(space: ComparisonSpace, rank: int, fold: int | None):
    if space.artifact is None:
        root = Path(space.unit_dir)
        if space.source_format == "rank_bases":
            return np.asarray(np.load(root / f"k{rank}" / "A.npy"), dtype=np.float32)
        if fold is not None:
            return np.asarray(np.load(root / f"fold_{fold}" / "A.npy"), dtype=np.float32)
        matrices = [np.load(path) for path in sorted(root.glob("fold_*/A.npy"))]
        return aggregate_subspace_basis(matrices, rank=rank, method="extrinsic")
    factors = space.artifact.load_input_factors(space.kind, rank)
    if not factors:
        raise FileNotFoundError(f"No {space.kind} factors in {space.label}")
    if fold is not None:
        if fold >= len(factors):
            raise IndexError(f"{space.label} has no factor for fold {fold}")
        return np.asarray(factors[fold], dtype=np.float32)
    if len(factors) == 1:
        return np.asarray(factors[0], dtype=np.float32)
    return aggregate_subspace_basis(factors, rank=rank, method="extrinsic")


def factor_for(
    space: ComparisonSpace,
    rank: int,
    fold: int,
    evaluation_dataset: str,
    geometry: str,
) -> np.ndarray | None:
    """Return a fold-matched or cross-dataset-aggregated rank-k factor."""
    fold_matched = space.dataset == evaluation_dataset and space.source_format == "fold_factors"
    if geometry == "weighted" and not fold_matched:
        return None
    raw = _load_raw_factor(space, int(rank), int(fold) if fold_matched else None)
    if raw.ndim != 2:
        raise ValueError(f"{space.unit_dir} factor must be 2D, got shape {raw.shape}")
    if raw.shape[1] < rank:
        raise ValueError(f"{space.unit_dir} factor has width {raw.shape[1]} below rank {rank}")
    if not np.isfinite(raw[:, :rank]).all():
        raise ValueError(f"{space.unit_dir} factor contains non-finite values through rank {rank}")
    return (
        orthonormal_factor(raw, rank)
        if geometry == "span"
        else np.asarray(raw[:, :rank], dtype=np.float32)
    )


def align_views(left: EvaluationView, right: EvaluationView):
    """Align two representations by canonical stimulus IDs."""
    for name, view in (("Source", left), ("Target", right)):
        if view.values.ndim != 2 or view.values.shape[0] != len(view.ids):
            raise ValueError(
                f"{name} evaluation view rows/IDs are inconsistent: "
                f"{view.values.shape} versus {len(view.ids)} IDs"
            )
        if not np.isfinite(view.values).all():
            raise ValueError(f"{name} evaluation view contains non-finite values")
    if len(np.unique(left.ids)) != len(left.ids):
        raise ValueError("Source evaluation view contains duplicate canonical IDs")
    if len(np.unique(right.ids)) != len(right.ids):
        raise ValueError("Target evaluation view contains duplicate canonical IDs")
    if np.array_equal(left.ids.astype(str), right.ids.astype(str)):
        if len(left.ids) < 4:
            raise ValueError("Fewer than four common evaluation rows remain after alignment")
        return left.ids.astype(str), left.values, right.values
    left_rows = {str(identifier): i for i, identifier in enumerate(left.ids)}
    right_rows = {str(identifier): i for i, identifier in enumerate(right.ids)}
    common = [str(identifier) for identifier in left.ids if str(identifier) in right_rows]
    if len(common) < 4:
        raise ValueError("Fewer than four common evaluation rows remain after alignment")
    return (
        np.asarray(common),
        left.values[[left_rows[identifier] for identifier in common]],
        right.values[[right_rows[identifier] for identifier in common]],
    )


def configure_metric_backend(device: str = "cuda") -> torch.device:
    """Configure the Himalaya/Torch backend used by pairwise metric scoring."""
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Metric device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Subspace comparison metrics require CUDA, but torch.cuda.is_available() "
            "is false. Run this analysis on a CUDA worker."
        )
    backend_name = "torch_cuda" if device == "cuda" else "torch"
    if get_backend().name != backend_name:
        set_backend(backend_name)
    return torch.device(device)


def _metric_tensor(values, device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float64)
    return torch.as_tensor(values, dtype=torch.float64, device=device)


def project_subspace(values, factor, *, device: str = "cuda") -> torch.Tensor:
    """Project one evaluation view through a saved factor on the metric device."""
    metric_device = configure_metric_backend(device)
    return _metric_tensor(values, metric_device) @ _metric_tensor(factor, metric_device)


def _pearson(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_centered = left - torch.mean(left)
    right_centered = right - torch.mean(right)
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(
        right_centered
    )
    if float(denominator) == 0.0:
        return torch.full((), torch.nan, dtype=left.dtype, device=left.device)
    return torch.sum(left_centered * right_centered) / denominator


def _linear_cka(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left - torch.mean(left, dim=0, keepdim=True)
    right = right - torch.mean(right, dim=0, keepdim=True)
    numerator = torch.linalg.matrix_norm(left.T @ right, ord="fro") ** 2
    denominator = torch.linalg.matrix_norm(
        left.T @ left, ord="fro"
    ) * torch.linalg.matrix_norm(right.T @ right, ord="fro")
    if float(denominator) == 0.0:
        return torch.full((), torch.nan, dtype=left.dtype, device=left.device)
    return numerator / denominator


def _reconstruction_metrics(
    target: torch.Tensor, prediction: torch.Tensor
) -> dict[str, float]:
    residual = torch.sum((target - prediction) ** 2)
    centered_target = target - torch.mean(target, dim=0, keepdim=True)
    denominator = torch.sum(centered_target**2)
    r2 = 1.0 - residual / denominator if float(denominator) > 0 else torch.nan

    target_centered = target - torch.mean(target, dim=0, keepdim=True)
    prediction_centered = prediction - torch.mean(prediction, dim=0, keepdim=True)
    coordinate_denominators = torch.linalg.vector_norm(
        target_centered, dim=0
    ) * torch.linalg.vector_norm(prediction_centered, dim=0)
    valid = coordinate_denominators > 0
    if bool(torch.any(valid)):
        correlations = torch.sum(target_centered * prediction_centered, dim=0)[valid]
        correlations = correlations / coordinate_denominators[valid]
        mean_correlation = torch.mean(correlations)
    else:
        mean_correlation = torch.full(
            (), torch.nan, dtype=target.dtype, device=target.device
        )
    return {
        "multivariate_r2": float(r2),
        "mean_coordinate_correlation": float(mean_correlation),
        "linear_cka": float(_linear_cka(target, prediction)),
    }


def _score_procrustes(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
    source_test: torch.Tensor,
    target_test: torch.Tensor,
) -> dict[str, float]:
    source_mean = torch.mean(source_train, dim=0)
    target_mean = torch.mean(target_train, dim=0)
    source_centered = source_train - source_mean
    target_centered = target_train - target_mean
    source_norm_sq = torch.sum(source_centered**2)
    if float(source_norm_sq) <= torch.finfo(source_train.dtype).eps:
        raise ValueError("Procrustes source training rows have zero variance")
    u, singular_values, vh = torch.linalg.svd(
        source_centered.T @ target_centered,
        full_matrices=False,
    )
    rotation = u @ vh
    scale = torch.sum(singular_values) / source_norm_sq
    aligned_test = scale * ((source_test - source_mean) @ rotation)
    target_centered_test = target_test - target_mean
    residual = aligned_test - target_centered_test
    sse = torch.sum(residual**2)
    target_sse = torch.sum(target_centered_test**2)
    normalized_mse = sse / target_sse if float(target_sse) > 0 else torch.nan
    row_denominators = torch.linalg.vector_norm(
        aligned_test, dim=1
    ) * torch.linalg.vector_norm(target_centered_test, dim=1)
    valid = row_denominators > 0
    if bool(torch.any(valid)):
        row_cosine = torch.mean(
            torch.sum(aligned_test * target_centered_test, dim=1)[valid]
            / row_denominators[valid]
        )
    else:
        row_cosine = torch.full(
            (), torch.nan, dtype=source_train.dtype, device=source_train.device
        )
    return {
        "test_r2": float(1.0 - normalized_mse),
        "test_normalized_mse": float(normalized_mse),
        "test_rmse": float(torch.sqrt(torch.mean(residual**2))),
        "scale": float(scale),
        "test_mean_row_cosine": float(row_cosine),
    }


def _projected_rsm(vectors: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "cosine":
        norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
        safe = torch.where(norms > 0, vectors / torch.clamp_min(norms, 1e-30), 0.0)
        rsm = 1.0 - torch.clamp(safe @ safe.T, -1.0, 1.0)
    elif metric == "euclidean":
        rsm = torch.cdist(vectors, vectors, p=2.0)
    else:
        raise ValueError(f"Unknown RSM metric: {metric!r}")
    rsm.fill_diagonal_(0.0)
    return rsm


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return SciPy-style average ranks without overflowing large tie groups."""
    if values.ndim != 1:
        raise ValueError(f"Rank input must be one-dimensional, got {values.shape}")
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    starts_group = torch.ones(len(values), dtype=torch.bool, device=values.device)
    starts_group[1:] = sorted_values[1:] != sorted_values[:-1]
    group_ids = torch.cumsum(starts_group.to(torch.int64), dim=0) - 1
    counts = torch.bincount(group_ids)
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    average_by_group = (
        starts.to(torch.float64) + 1.0 + ends.to(torch.float64)
    ) / 2.0
    sorted_ranks = average_by_group[group_ids]
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = sorted_ranks
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _score_rsa(
    source_test: torch.Tensor, target_test: torch.Tensor, metric: str
) -> dict[str, float]:
    source_rsm = _projected_rsm(source_test, metric)
    target_rsm = _projected_rsm(target_test, metric)
    triangle = torch.tril_indices(
        source_rsm.shape[0], source_rsm.shape[1], offset=-1, device=source_rsm.device
    )
    source_values = source_rsm[triangle[0], triangle[1]]
    target_values = target_rsm[triangle[0], triangle[1]]
    if source_values.numel() < 2:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson = _pearson(source_values, target_values)
    if float(torch.std(source_values)) == 0.0 or float(torch.std(target_values)) == 0.0:
        spearman = torch.full(
            (), torch.nan, dtype=source_values.dtype, device=source_values.device
        )
    else:
        spearman = _spearman(source_values, target_values)
    return {"pearson": float(pearson), "spearman": float(spearman)}


def score_ordered_pair(
    source,
    target,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    ridge_alphas: list[float] | None,
    rsm_metric: str,
    device: str = "cpu",
) -> dict[str, float]:
    """Fit and score one ordered held-out comparison with Torch/Himalaya."""
    metric_device = configure_metric_backend(device)
    source = _metric_tensor(source, metric_device)
    target = _metric_tensor(target, metric_device)
    train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=metric_device)
    test_idx = torch.as_tensor(test_idx, dtype=torch.long, device=metric_device)
    source_train, source_test = source[train_idx], source[test_idx]
    target_train, target_test = target[train_idx], target[test_idx]

    alphas = np.asarray(ridge_alphas or RIDGE_ALPHAS, dtype=float)
    if np.any(alphas <= 0):
        raise ValueError("GPU Himalaya RidgeCV requires strictly positive ridge_alphas")
    ridge = HimalayaRidgeCV(
        alphas=alphas,
        cv=5,
        fit_intercept=True,
        solver_params={"local_alpha": False},
    )
    ridge.fit(source_train, target_train)
    coefficients = _metric_tensor(ridge.coef_, metric_device)
    intercept = _metric_tensor(ridge.intercept_, metric_device)
    prediction = source_test @ coefficients + intercept
    best_alphas = _metric_tensor(ridge.best_alphas_, metric_device)
    if not bool(torch.allclose(best_alphas, best_alphas[0].expand_as(best_alphas))):
        raise RuntimeError("Shared-alpha Himalaya RidgeCV returned inconsistent alphas")

    reconstruction = _reconstruction_metrics(target_test, prediction)
    normalized_error = (
        float(1.0 - reconstruction["multivariate_r2"])
        if np.isfinite(reconstruction["multivariate_r2"])
        else float("nan")
    )
    procrustes = _score_procrustes(
        source_train, target_train, source_test, target_test
    )
    rsa = _score_rsa(source_test, target_test, rsm_metric)
    return {
        "ridge_r2": reconstruction["multivariate_r2"],
        "ridge_coordinate_correlation": reconstruction["mean_coordinate_correlation"],
        "ridge_prediction_cka": reconstruction["linear_cka"],
        "ridge_normalized_error": normalized_error,
        "ridge_alpha": float(best_alphas[0]),
        "procrustes_r2": procrustes["test_r2"],
        "procrustes_normalized_mse": procrustes["test_normalized_mse"],
        "procrustes_rmse": procrustes["test_rmse"],
        "procrustes_scale": procrustes["scale"],
        "procrustes_mean_row_cosine": procrustes["test_mean_row_cosine"],
        "direct_cka": float(_linear_cka(source_test, target_test)),
        "rsa_pearson": rsa["pearson"],
        "rsa_spearman": rsa["spearman"],
    }


def fold_indices(
    common_ids: np.ndarray,
    source: ComparisonSpace,
    target: ComparisonSpace,
    evaluation: EvaluationData,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build comparison indices from the intersection of leakage-safe held-out IDs."""
    default = {str(value) for value in evaluation.fold_test_ids[fold]}
    source_safe = (
        {str(value) for value in source.heldout_ids[fold]}
        if source.heldout_ids is not None
        else default
    )
    target_safe = (
        {str(value) for value in target.heldout_ids[fold]}
        if target.heldout_ids is not None
        else default
    )
    test_ids = source_safe & target_safe & default
    test_mask = np.fromiter(
        (str(value) in test_ids for value in common_ids),
        dtype=bool,
        count=len(common_ids),
    )
    test = np.flatnonzero(test_mask)
    train = np.flatnonzero(~test_mask)
    if len(test) < 2 or len(train) < 2:
        raise ValueError(
            f"Fold {fold} has {len(train)} train and {len(test)} test rows after alignment"
        )
    return train, test


def summarize_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize metric folds without hiding missing or failed cells."""
    keys = [
        "comparison_mode",
        "source_dataset", "source_offset", "source_model", "source_layer", "source_kind",
        "source_label", "target_dataset", "target_offset", "target_model", "target_layer",
        "target_kind", "target_label", "geometry", "source_rank", "target_rank",
        "equal_capacity_rank",
    ]
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, values))
        row["n_folds"] = int(group["fold"].nunique())
        for metric in ALL_METRICS:
            valid = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(valid)) if len(valid) else np.nan
            row[f"{metric}_sem"] = float(sem(valid)) if len(valid) > 1 else np.nan
            row[f"{metric}_n"] = int(len(valid))
        rows.append(row)
    return pd.DataFrame(rows)
