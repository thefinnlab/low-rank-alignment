"""Ordered, leakage-aware comparisons of saved embedding-space factors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import sem
from sklearn.linear_model import RidgeCV

from src.core.config import SubspaceComparisonConfig, get_results_dir
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.constants import RIDGE_ALPHAS
from src.utils.data_loaders import selection_after_offset
from src.utils.embeddings import load_embeddings
from src.utils.encoding import apply_offset, load_encoding_results
from src.utils.fmri import load_story_embeddings, resolve_fmri_stories
from src.utils.language_reconstruction import orthonormal_factor, reconstruction_metrics
from src.utils.subspace_compare import (
    aggregate_subspace_basis,
    projected_rsm,
    rsa_lower_triangle_correlations,
    score_cca_alignment,
    score_procrustes_alignment,
)


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
    "cca_first_test_corr",
    "cca_median_test_corr",
    "cca_mean_all_test_corr",
    "cca_mean_top10_test_corr",
    "cca_test_shared_dim_pr",
    "cca_n_test_corr_ge_threshold",
)
ALL_METRICS = DIRECTED_METRICS + SYMMETRIC_METRICS


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
    model_order = {name: i for i, name in enumerate(config.models)}
    for dataset_alias in config.datasets:
        dataset_spec = config.datasets[dataset_alias]
        for model, model_spec in config.models.items():
            if dataset_alias not in model_spec["datasets"]:
                continue
            for offset in sorted(map(int, model_spec["offsets"])):
                for layer in sorted(map(int, model_spec["layers"])):
                    try:
                        root = resolve_result_root(
                            dataset_alias, dataset_spec, model, layer, offset
                        )
                    except Exception as exc:
                        missing.append(f"{dataset_alias}/{model}/l{layer}/o{offset}: {exc}")
                        continue
                    source_format = (
                        "rank_bases"
                        if dataset_spec.get("loader", "ieeg_words") == "fmri_words"
                        else "fold_factors"
                    )
                    for kind in kinds:
                        unit = root / kind
                        try:
                            ranks = (
                                _completed_rank_basis_ranks(unit)
                                if source_format == "rank_bases"
                                else _completed_fold_ranks(unit)
                            )
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
                                unit_dir=unit,
                                source_format=source_format,
                                ranks=ranks,
                                heldout_ids=None,
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
    reference_result_path = _encoding_result_path(reference.result_root, subjects)
    reference_result = load_encoding_results(str(reference_result_path))
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

    heldout_by_run: dict[tuple[Path, int], list[np.ndarray]] = {}
    for space in spaces:
        if space.dataset != alias or space.source_format != "fold_factors":
            continue
        run_key = (space.result_root, space.offset)
        if run_key in heldout_by_run:
            space.heldout_ids = heldout_by_run[run_key]
            continue
        path = _encoding_result_path(space.result_root, subjects)
        result = load_encoding_results(str(path))
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
        space.heldout_ids = load_ieeg_heldout_ids(
            space.result_root,
            subjects,
        )
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
            for model, model_spec in config.models.items()
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


@lru_cache(maxsize=None)
def _load_raw_factor(unit_dir: str, source_format: str, rank: int, fold: int | None):
    root = Path(unit_dir)
    if source_format == "rank_bases":
        return np.asarray(np.load(root / f"k{rank}" / "A.npy"), dtype=np.float32)
    if fold is not None:
        return np.asarray(np.load(root / f"fold_{fold}" / "A.npy"), dtype=np.float32)
    matrices = [np.load(path) for path in sorted(root.glob("fold_*/A.npy"))]
    return aggregate_subspace_basis(matrices, rank=rank, method="extrinsic")


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
    raw = _load_raw_factor(
        str(space.unit_dir), space.source_format, int(rank), int(fold) if fold_matched else None
    )
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


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) - np.mean(left, axis=0, keepdims=True)
    right = np.asarray(right, dtype=np.float64) - np.mean(right, axis=0, keepdims=True)
    numerator = np.linalg.norm(left.T @ right, ord="fro") ** 2
    denominator = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(
        right.T @ right, ord="fro"
    )
    return float(numerator / denominator) if denominator > 0 else float("nan")


def score_ordered_pair(
    source: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    ridge_alphas: list[float] | None,
    rsm_metric: str,
    shared_corr_threshold: float,
) -> dict[str, float]:
    """Fit all train-only mappings and score one ordered held-out comparison."""
    alphas = np.asarray(ridge_alphas or RIDGE_ALPHAS, dtype=float)
    ridge = RidgeCV(alphas=alphas, cv=None, fit_intercept=True, scoring="neg_mean_squared_error")
    ridge.fit(source[train_idx], target[train_idx])
    prediction = ridge.predict(source[test_idx])
    reconstruction = reconstruction_metrics(target[test_idx], prediction)
    normalized_error = (
        float(1.0 - reconstruction["multivariate_r2"])
        if np.isfinite(reconstruction["multivariate_r2"])
        else float("nan")
    )
    procrustes = score_procrustes_alignment(
        source[train_idx], target[train_idx], source[test_idx], target[test_idx]
    )
    cca = score_cca_alignment(
        source[train_idx],
        target[train_idx],
        source[test_idx],
        target[test_idx],
        shared_corr_threshold=shared_corr_threshold,
    )
    rsa = rsa_lower_triangle_correlations(
        projected_rsm(source[test_idx], metric=rsm_metric),
        projected_rsm(target[test_idx], metric=rsm_metric),
    )
    return {
        "ridge_r2": reconstruction["multivariate_r2"],
        "ridge_coordinate_correlation": reconstruction["mean_coordinate_correlation"],
        "ridge_prediction_cka": reconstruction["linear_cka"],
        "ridge_normalized_error": normalized_error,
        "ridge_alpha": float(ridge.alpha_),
        "procrustes_r2": procrustes["test_r2"],
        "procrustes_normalized_mse": procrustes["test_normalized_mse"],
        "procrustes_rmse": procrustes["test_rmse"],
        "procrustes_scale": procrustes["scale"],
        "procrustes_mean_row_cosine": procrustes["test_mean_row_cosine"],
        "direct_cka": _linear_cka(source[test_idx], target[test_idx]),
        "rsa_pearson": rsa["pearson"],
        "rsa_spearman": rsa["spearman"],
        "cca_first_test_corr": cca["first_test_corr"],
        "cca_median_test_corr": cca["median_test_corr"],
        "cca_mean_all_test_corr": cca["mean_all_test_corr"],
        "cca_mean_top10_test_corr": cca["mean_top10_test_corr"],
        "cca_test_shared_dim_pr": cca["test_shared_dim_pr"],
        "cca_n_test_corr_ge_threshold": float(cca["n_test_corr_ge_threshold"]),
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
    test = np.asarray([i for i, value in enumerate(common_ids) if str(value) in test_ids])
    train = np.asarray([i for i in range(len(common_ids)) if i not in set(test)], dtype=int)
    if len(test) < 2 or len(train) < 2:
        raise ValueError(
            f"Fold {fold} has {len(train)} train and {len(test)} test rows after alignment"
        )
    return train, test


def summarize_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize metric folds without hiding missing or failed cells."""
    keys = [
        "source_dataset", "source_offset", "source_model", "source_layer", "source_kind",
        "source_label", "target_dataset", "target_offset", "target_model", "target_layer",
        "target_kind", "target_label", "geometry", "rank",
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
