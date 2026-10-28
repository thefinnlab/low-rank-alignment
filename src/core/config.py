"""Dataclass configs for encoding model training and significance testing."""


import math


import os


from dataclasses import dataclass, field


from enum import Enum


from typing import Any


class ScriptMode(str, Enum):
    """Which Script branch to run in analysis"""

    CROSS_SUBJECT = "cross_subject"
    CROSS_DATASET = "cross_dataset"
    PER_SUBJECT = "per_subject"
    LEAVE_ONE_SUBJECT_OUT = "leave_one_subject_out"
    MULTI_VIEW = "multi_view"
    TEMPORAL_WINDOWS = "temporal_windows"
    CROSS_TEMPORAL_WINDOWS = "cross_temporal_windows"
    INTERNAL = "internal"
    FULL_FIT = "full_fit"


class LanguageDecodingMode(str, Enum):
    """Relationship between the decoding dataset and saved RRR factors."""

    WITHIN_DATASET = "within_dataset"
    CROSS_DATASET = "cross_dataset"


SUBSPACE_AGGREGATION_METHODS = {"extrinsic", "factor_weighted"}
FMRI_FIR_SUBSPACE_AGGREGATION_METHODS = (
    SUBSPACE_AGGREGATION_METHODS | {"prediction_weighted"}
)


def _validate_subspace_aggregation(value: str) -> None:
    if value not in SUBSPACE_AGGREGATION_METHODS:
        raise ValueError(
            "subspace_aggregation must be one of "
            f"{sorted(SUBSPACE_AGGREGATION_METHODS)}, got {value!r}"
        )


def gpt2_xl_results_dir(base_dir, offset, layer):
    return os.path.join(base_dir, f"gpt2xl_layer{layer}_offset{offset}")


def glove_results_dir(base_dir, offset, dim):
    return os.path.join(base_dir, f"glove_dim{dim}_offset{offset}")


def transformer_results_dir(base_dir, offset, model_dir, layer, pooling="mean"):
    suffix = "" if pooling == "mean" else f"_{pooling}"
    return os.path.join(base_dir, f"{model_dir}_layer{layer}_offset{offset}{suffix}")


def get_results_dir(
    model: str,
    offset: int,
    per_model_config: dict,
    base_dir: str = "results/",
    dataset_context: Any = None,
) -> str:
    """Return a canonical results directory path derived from run parameters.

    Example: get_results_dir("gpt2xl", 24, 0) → "results/gpt2xl_layer24_offset0"
    """
    if model == "gpt2xl":
        root = gpt2_xl_results_dir(base_dir, offset, **per_model_config)
    elif model == "glove":
        root = glove_results_dir(base_dir, offset, **per_model_config)
    elif model == "transformer":
        root = transformer_results_dir(base_dir, offset, **per_model_config)
    else:
        raise ValueError(f"Model: {model} not implemented yet.")
    if (
        dataset_context is None
        or getattr(dataset_context, "dataset", "podcast") == "podcast"
    ):
        return root
    stimulus_id = getattr(dataset_context, "stimulus_id", None)
    if not stimulus_id:
        raise ValueError("dataset_context must provide a non-empty stimulus_id.")
    return os.path.join(root, f"movie-{stimulus_id}")


@dataclass
class EncodingConfig:
    """Parameters for encoding model training (used by train_encoding_models.py)."""

    data_root: str = "data/"
    model: str = (
        "gpt2xl"  # Embedding model name; used in results dir generation. Should be registered in src.core.registry.embedding_getter_registry
    )
    offset: int = 0  # Word offset for aligning embeddings to neural data
    dataset: str = "podcast"
    subjects: list[int] | None = None  # None → defaults to 1–9 at runtime
    batch: bool = False  # Process subjects one at a time (lower peak RAM)
    ridge_alphas: list[float] | None = None  # None → use RIDGE_ALPHAS from constants.py
    ridge_cv_folds: int = 5  # Inner CV folds used by RidgeCV alpha selection.
    ridge_solver_params: dict | None = None  # Optional Himalaya RidgeCV solver_params.
    ridge_y_in_cpu: bool | None = None  # Optional Himalaya RidgeCV Y_in_cpu override.
    ridge_force_cpu: bool | None = None  # Optional Himalaya RidgeCV force_cpu override.
    alpha_grid: dict = field(default_factory=dict)  # Dimension-aware alpha policy.
    log_alphas: bool = False  # Save alpha diagnostics (.npz + .png) per subject
    refine_alphas: bool = False  # Two-pass training: coarse sweep → refined grid
    arbitrary_seed: int = 42  # Seed used for generating arbitrary embedding baseline
    num_folds: int = 2  # Number of cross-validation folds
    limit_to_glove: bool = (
        True  # Keep model comparisons on the shared GloVe-covered vocabulary.
    )
    shared_word_selection: bool = (
        False  # If true, use the intersection of cached word selections across subjects.
    )
    # Per model type config. Used when generating embeddings in load_embeddings.
    per_model_config: dict = field(default_factory=lambda: {"layer": 24})
    per_dataset_config: dict = field(default_factory=dict)
    refresh: bool = False


@dataclass
class SignificanceConfig:
    """Parameters for significance testing (superset of encoding)."""

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    results_root: str = "results"
    plot: bool = True
    blocks_per_fold: int = 5
    max_matching_distance: int = 1
    positive_fraction: float = 0.8
    mean_corr_threshold: float = 0.0
    mean_z_threshold: float | None = None  # Deprecated: converted to r-space if set.
    minimal_shift: int = 500
    shift_step: int = 1
    num_null_shifts: int = 10000
    null_seed: int = 42
    null_batch_size: int = 16
    device: str | None = None
    fdr_q: float = 0.01
    fdr_mode: str = "electrode"
    save_weights: bool = False
    min_cluster_size: int = 1  # Heuristic significance: minimum generated cluster size.
    refresh: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if isinstance(self.num_null_shifts, bool) or not isinstance(
            self.num_null_shifts, int
        ):
            raise ValueError("num_null_shifts must be a positive integer")
        if self.num_null_shifts <= 0:
            raise ValueError("num_null_shifts must be a positive integer")
        if isinstance(self.null_seed, bool) or not isinstance(self.null_seed, int):
            raise ValueError("null_seed must be an integer")
        if isinstance(self.null_batch_size, bool) or not isinstance(
            self.null_batch_size, int
        ):
            raise ValueError("null_batch_size must be a positive integer")
        if self.null_batch_size <= 0:
            raise ValueError("null_batch_size must be a positive integer")
        if self.fdr_mode not in {"electrode", "cluster"}:
            raise ValueError(
                "fdr_mode must be either 'electrode' or 'cluster', "
                f"got {self.fdr_mode!r}"
            )


@dataclass
class RRRViewConfig:
    """One saved RRR subspace contributing a band to multiview RRR."""

    name: str
    rank: int
    kind: str | None = None
    ieeg_rrr_source: dict | None = None
    fmri_rrr_source: dict | None = None


@dataclass
class RRRTemporalWindowConfig:
    """One half-open neural-target lag window, in seconds."""

    name: str
    start_sec: float
    end_sec: float


@dataclass
class RRRCrossTemporalWindowPairConfig:
    """One directed transfer from a source window's A to target-window responses."""

    source: str
    target: str


@dataclass
class SignificanceSelectionConfig:
    """Select neural targets from one significance run or an explicit union."""

    mode: str = "within-layer"
    reference_artifact: dict = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)

    def validate(self):
        # Preserve constructible defaults; executable runners require a populated
        # selection before resolving artifacts.
        if not self.reference_artifact and not self.artifacts:
            return
        if self.mode not in {"within-layer", "cross-model-layer"}:
            raise ValueError(
                "significance_selection.mode must be 'within-layer' or "
                f"'cross-model-layer', got {self.mode!r}"
            )
        if not self.reference_artifact:
            raise ValueError("significance_selection.reference_artifact is required")
        if isinstance(self.reference_artifact.get("layer"), list):
            raise ValueError("reference_artifact.layer must be a scalar")
        if self.mode == "within-layer" and self.artifacts:
            raise ValueError("within-layer significance selection cannot define artifacts")
        if self.mode == "cross-model-layer" and not self.artifacts:
            raise ValueError("cross-model-layer significance selection requires artifacts")

    def __post_init__(self):
        self.validate()


@dataclass
class RRRConfig:
    """Parameters for raw-input Reduced Rank Regression analysis (run_rrr.py).

    Performs RRR directly from raw embeddings to neural targets — no SVD projection.
    Sweeps over a rank grid to find the minimal rank that captures all variance
    explained by an unconstrained ridge baseline.

    Mode → output directory mapping (default, overridable via rrr_output_dir):
        cross_subject : {output_dir}/rrr/
        temporal_windows: {output_dir}/temporal_window_rrr/<window>/{rrr,pca}/
        cross_temporal_windows: directed, fold-matched transfer between saved windows
        per_subject   : {output_dir}/per_subject_rrr/
        leave_one_subject_out: factor transfer and baseline per held-out subject
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    results_root: str = "results"
    plot: bool = True
    significance_selection: SignificanceSelectionConfig = field(
        default_factory=SignificanceSelectionConfig
    )
    mode: ScriptMode = ScriptMode.CROSS_SUBJECT
    ieeg_rrr_source: dict | None = None
    fmri_rrr_source: dict | None = None
    cross_source_format: str = "fold_factors"
    views: list[RRRViewConfig] = field(default_factory=list)
    multi_view_name: str | None = None
    temporal_windows: list[RRRTemporalWindowConfig] = field(default_factory=list)
    temporal_window_pairs: list[RRRCrossTemporalWindowPairConfig] = field(
        default_factory=list
    )
    subspace_aggregation: str | None = None

    # Full-model banded ridge used by mode=multi_view.
    banded_ridge_alphas: list[float] | None = None
    banded_ridge_solver_params: dict = field(
        default_factory=lambda: {"n_iter": 100, "random_state": 42}
    )

    # Rank sweep
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])
    refresh: bool = False  # If true, recompute all ranks even when outputs exist.

    # Optional metrics (expensive; disabled by default to speed up rank sweeps)
    compute_cca: bool = False
    compute_procrustes: bool = False
    compute_sig_nonsig: bool = True
    compute_joint_ridge: bool = False  # Deprecated; ignored by RRR/PCA rank sweeps.

    # Per-target correlation artifacts saved in each rank/fold metrics.npz.
    save_nr_corr_per_target: bool = True
    save_rrr_corr_per_target: bool = False
    save_b_matrices: bool = False

    # Alpha diagnostics
    log_alphas: bool = False  # Save per-rank alpha diagnostics in RRR output
    standardize_subspace_inputs: bool = (
        False  # Preserve RRR/PCA subspace coordinate scales by default.
    )
    # Arbitrary embedding seed for null baseline
    arbitrary_seed: int = 42

    # PCA baseline: run PCA subspace analysis alongside RRR using the same rank_grid
    run_pca_baseline: bool = False

    # Null-augmented rank check: append circularly shifted omitted RRR coordinates
    # to each rank-k subspace to estimate high-dimensional nuisance effects.
    run_null_augmented: bool = False
    null_augmented_repeats: int = 20
    null_augmented_reference_rank: int | None = None
    null_augmented_min_shift: int = 500
    null_augmented_seed: int = 42

    def __post_init__(self):
        self.validate()

    def validate(self):
        self.significance_selection.validate()
        if self.subspace_aggregation is not None:
            _validate_subspace_aggregation(self.subspace_aggregation)
        if self.cross_source_format not in {"fold_factors", "rank_bases"}:
            raise ValueError(
                "cross_source_format must be 'fold_factors' or 'rank_bases', "
                f"got {self.cross_source_format!r}"
            )
        if self.save_rrr_corr_per_target and not self.save_b_matrices:
            raise ValueError(
                "save_rrr_corr_per_target=True requires save_b_matrices=True"
            )
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")
        if self.banded_ridge_alphas is not None and (
            not self.banded_ridge_alphas
            or any(float(alpha) < 0 for alpha in self.banded_ridge_alphas)
        ):
            raise ValueError("banded_ridge_alphas must contain non-negative values")
        if self.mode == ScriptMode.MULTI_VIEW and not self.views:
            raise ValueError("RRRConfig.views must contain at least one view")
        if self.mode == ScriptMode.MULTI_VIEW and not self.multi_view_name:
            raise ValueError("mode=multi_view requires multi_view_name")
        names = [view.name for view in self.views]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate RRR view name(s): {duplicates}")
        if self.mode in {
            ScriptMode.TEMPORAL_WINDOWS, ScriptMode.CROSS_TEMPORAL_WINDOWS
        } and not self.temporal_windows:
            raise ValueError(
                "RRRConfig.temporal_windows must contain at least one window"
            )
        window_names = [window.name for window in self.temporal_windows]
        if any(not name or not name.strip() for name in window_names):
            raise ValueError("Temporal window names must be non-empty")
        duplicate_windows = sorted(
            {name for name in window_names if window_names.count(name) > 1}
        )
        if duplicate_windows:
            raise ValueError(
                f"Duplicate temporal window name(s): {duplicate_windows}"
            )
        for window in self.temporal_windows:
            bounds = (float(window.start_sec), float(window.end_sec))
            if not all(math.isfinite(value) for value in bounds):
                raise ValueError(
                    f"Temporal window {window.name!r} bounds must be finite"
                )
            if bounds[0] >= bounds[1]:
                raise ValueError(
                    f"Temporal window {window.name!r} must have start_sec < end_sec"
                )
        if self.mode == ScriptMode.CROSS_TEMPORAL_WINDOWS:
            if not self.ieeg_rrr_source or self.fmri_rrr_source:
                raise ValueError(
                    "mode=cross_temporal_windows requires an ieeg_rrr_source"
                )
            if len(window_names) < 2:
                raise ValueError(
                    "mode=cross_temporal_windows requires at least two windows"
                )
            known_windows = set(window_names)
            pairs = [(pair.source, pair.target) for pair in self.temporal_window_pairs]
            if len(pairs) != len(set(pairs)):
                raise ValueError("Duplicate temporal window transfer pair")
            for source, target in pairs:
                if source == target:
                    raise ValueError(
                        "Temporal window transfer source and target must differ"
                    )
                unknown = sorted({source, target} - known_windows)
                if unknown:
                    raise ValueError(
                        f"Temporal window transfer references unknown window(s): {unknown}"
                    )
        if any(not view.name for view in self.views):
            raise ValueError("Every RRR view name must be non-empty")
        if any(int(view.rank) < 1 for view in self.views):
            raise ValueError("Every RRR view rank must be positive")
        for view in self.views:
            if bool(view.ieeg_rrr_source) == bool(view.fmri_rrr_source):
                raise ValueError(
                    f"RRR view {view.name!r} must define exactly one of "
                    "ieeg_rrr_source or fmri_rrr_source"
                )
        if self.mode == ScriptMode.CROSS_DATASET:
            if bool(self.ieeg_rrr_source) == bool(self.fmri_rrr_source):
                raise ValueError(
                    "mode=cross_dataset requires exactly one of ieeg_rrr_source "
                    "or fmri_rrr_source"
                )


@dataclass
class LanguageDecodingConfig:
    """Decode word-aligned linguistic targets from saved RRR/PCA subspaces."""

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    rrr_source: dict | None = None
    # For cross-dataset decoding onto fMRI, this internal fMRI artifact supplies
    # the target story/data configuration while rrr_source supplies the basis.
    fmri_target_artifact: dict | None = None
    rrr_selectors: dict = field(default_factory=dict)
    results_root: str = "results"
    variant_name: str | None = None
    # Deprecated compatibility inputs; executable clients require rrr_source.
    rrr_run_path: str | None = None
    pca_run_path: str | None = None
    mode: LanguageDecodingMode = LanguageDecodingMode.WITHIN_DATASET
    fmri_tr_pre_fir: bool = False
    include_full_embeddings: bool = True
    # Decode label at transcript word t + label_offset from embedding at t.
    label_offset: int = 0
    source_name: str | None = None
    fold_results_dir: str | None = None
    output_dir: str | None = None
    device: str = "auto"
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])
    subspace_aggregation: str = "extrinsic"
    tasks: list[str] = field(
        default_factory=lambda: [
            "part_of_speech",
            "content_word",
            "surprisal",
            "entropy",
            "sentence_onset",
            "iu_onset",
            "sentence_position",
            "iu_position",
            "word_duration",
            "word_length",
            "word_zipf_frequency",
        ]
    )
    task_data: dict = field(
        default_factory=lambda: {
            "language_features": (
                "processed_data/language_features/podcast/podcast/tokens.tsv"
            ),
            "word_annotations": "processed_data/df_word_onset_with_pos_class.csv",
        }
    )
    logistic_cs: list[float] = field(
        default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    )
    refresh: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if isinstance(self.mode, str):
            self.mode = LanguageDecodingMode(self.mode)
        if isinstance(self.label_offset, bool) or not isinstance(self.label_offset, int):
            raise ValueError("label_offset must be an integer")
        if not isinstance(self.fmri_tr_pre_fir, bool):
            raise ValueError("fmri_tr_pre_fir must be a boolean")
        if not isinstance(self.include_full_embeddings, bool):
            raise ValueError("include_full_embeddings must be a boolean")
        if self.fmri_tr_pre_fir and self.label_offset != 0:
            raise ValueError("label_offset is only supported for word decoding")
        _validate_subspace_aggregation(self.subspace_aggregation)
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of ['auto', 'cpu', 'cuda']")
        if not self.rrr_source and not self.rrr_run_path:
            raise ValueError("LanguageDecodingConfig.rrr_source is required")
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")
        if len(set(int(rank) for rank in self.rank_grid)) != len(self.rank_grid):
            raise ValueError("rank_grid must not contain duplicate ranks")
        if not self.tasks or any(not str(task).strip() for task in self.tasks):
            raise ValueError("tasks must contain at least one non-empty task name")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must not contain duplicate names")
        if not self.logistic_cs or any(float(value) <= 0 for value in self.logistic_cs):
            raise ValueError("logistic_cs must contain positive values")
        if self.encoding.dataset in {"fmri", "deep_fmri", "deep-fmri"}:
            source_dataset = str((self.rrr_source or {}).get("dataset", ""))
            target_dataset = str((self.fmri_target_artifact or {}).get("dataset", ""))
            fmri_names = {"fmri", "deep_fmri", "deep-fmri"}
            if source_dataset not in fmri_names and target_dataset not in fmri_names:
                raise ValueError(
                    "fMRI decoding requires a deep-fmri rrr_source or "
                    "fmri_target_artifact"
                )
            if self.mode == LanguageDecodingMode.CROSS_DATASET and target_dataset not in fmri_names:
                raise ValueError("Cross-dataset fMRI decoding requires fmri_target_artifact")
            if self.mode == LanguageDecodingMode.WITHIN_DATASET and self.fmri_target_artifact is not None:
                raise ValueError("Within-dataset fMRI decoding cannot use fmri_target_artifact")
        elif self.fmri_tr_pre_fir:
            raise ValueError("fmri_tr_pre_fir is only available for fMRI targets")


@dataclass
class LanguageReconstructionConfig:
    """Linearly reconstruct saved language-model subspaces from fixed predictors."""

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    rrr_source: dict | None = None
    rrr_selectors: dict = field(default_factory=dict)
    results_root: str = "results"
    variant_name: str | None = None
    mode: LanguageDecodingMode = LanguageDecodingMode.WITHIN_DATASET
    subspace_aggregation: str = "extrinsic"
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    tasks: list[str] = field(default_factory=lambda: [
        "part_of_speech",
        "content_word",
        "surprisal",
        "entropy",
        "sentence_position",
        "iu_position",
    ])
    task_data: dict = field(default_factory=dict)
    predictor_sets: list[str] = field(default_factory=lambda: ["tasks"])
    glove_dim: int = 300
    target_views: list[str] = field(default_factory=lambda: ["weighted", "span"])
    ridge_alphas: list[float] | None = None
    save_models: bool = False
    refresh: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if isinstance(self.mode, str):
            self.mode = LanguageDecodingMode(self.mode)
        _validate_subspace_aggregation(self.subspace_aggregation)
        if self.mode not in {
            LanguageDecodingMode.WITHIN_DATASET,
            LanguageDecodingMode.CROSS_DATASET,
        }:
            raise ValueError(
                f"Language reconstruction does not support mode={self.mode.value}"
            )
        if not self.rrr_source:
            raise ValueError("LanguageReconstructionConfig.rrr_source is required")
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")
        if len(set(map(int, self.rank_grid))) != len(self.rank_grid):
            raise ValueError("rank_grid must not contain duplicate ranks")
        allowed_predictors = {"tasks", "glove", "tasks_glove"}
        if not self.predictor_sets or set(self.predictor_sets) - allowed_predictors:
            raise ValueError(
                f"predictor_sets must be drawn from {sorted(allowed_predictors)}"
            )
        if len(set(self.predictor_sets)) != len(self.predictor_sets):
            raise ValueError("predictor_sets must not contain duplicate names")
        uses_tasks = bool({"tasks", "tasks_glove"} & set(self.predictor_sets))
        if uses_tasks and (
            not self.tasks or any(not str(task).strip() for task in self.tasks)
        ):
            raise ValueError(
                "tasks must contain at least one non-empty task name when a "
                "task predictor set is enabled"
            )
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must not contain duplicate names")
        if self.glove_dim not in {50, 100, 200, 300}:
            raise ValueError("glove_dim must be one of [50, 100, 200, 300]")
        allowed_views = {"weighted", "span"}
        if not self.target_views or set(self.target_views) - allowed_views:
            raise ValueError(
                f"target_views must be drawn from {sorted(allowed_views)}"
            )
        if len(set(self.target_views)) != len(self.target_views):
            raise ValueError("target_views must not contain duplicate names")
        if self.ridge_alphas is not None and (
            not self.ridge_alphas or any(float(value) <= 0 for value in self.ridge_alphas)
        ):
            raise ValueError("ridge_alphas must contain positive values")


@dataclass
class LanguageFeatureGenerationConfig:
    """Generate reusable, stimulus-aligned language annotation artifacts."""

    dataset: str = "podcast"
    data_root: str = "data/"
    per_dataset_config: dict = field(default_factory=dict)
    model_dir: str = "gpt2-xl"
    output_dir: str = "processed_data/language_features"
    resource_dir: str = "data/language_resources"
    provider_resource_config: str | None = None
    providers: list[str] = field(
        default_factory=lambda: ["native", "ud", "lexical_norms", "swow", "wordnet"]
    )
    dataset_options: dict = field(default_factory=dict)
    provider_options: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    refresh: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not self.dataset.strip():
            raise ValueError("dataset must be non-empty")
        if not self.providers or any(not str(name).strip() for name in self.providers):
            raise ValueError("providers must contain at least one non-empty name")
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("providers must not contain duplicate names")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")
        if not self.resource_dir:
            raise ValueError("resource_dir must be non-empty")


@dataclass
class SubspaceComparisonRun:
    """One saved subspace run to include in pairwise comparisons."""

    name: str
    rrr_config: str
    unit_dir: str
    kind: str | None = None


@dataclass
class SubspaceComparisonConfig:
    """Parameters for ordered comparisons of saved RRR/PCA spaces."""

    analysis_name: str = "subspace_comparison"
    variant_name: str | None = None
    output_root: str = "results/subspace_comparisons"
    evaluation_dataset: str = "podcast"
    datasets: dict[str, dict] = field(default_factory=dict)
    models: list[dict] | dict[str, dict] = field(default_factory=list)
    comparison_modes: list[str] = field(default_factory=lambda: ["equal_capacity"])
    max_rank: int = 128
    include_pca: bool = False
    geometries: list[str] = field(default_factory=lambda: ["span", "weighted"])
    ridge_alphas: list[float] | None = None
    refresh: bool = False
    rsm_metric: str = "cosine"

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not self.analysis_name.strip():
            raise ValueError("analysis_name must be non-empty")
        if not self.output_root:
            raise ValueError("output_root must be non-empty")
        if not self.datasets:
            raise ValueError("datasets must contain at least one dataset")
        if self.evaluation_dataset not in self.datasets:
            raise ValueError(
                f"evaluation_dataset {self.evaluation_dataset!r} is absent from datasets"
            )
        if not self.models:
            raise ValueError("models must contain at least one model")
        model_items = (
            list(self.models.items())
            if isinstance(self.models, dict)
            else [
                (item.get("model"), {"offsets": [0], **item})
                for item in self.models
            ]
        )
        names = [name for name, _ in model_items]
        if any(not name for name in names):
            raise ValueError("Each models list entry must define model")
        if len(set(names)) != len(names):
            raise ValueError("models must not contain duplicate model names")
        if isinstance(self.max_rank, bool) or int(self.max_rank) < 1:
            raise ValueError("max_rank must be a positive integer")
        allowed_loaders = {"ieeg_words", "fmri_words"}
        for name, spec in self.datasets.items():
            if not str(name).strip() or not isinstance(spec, dict):
                raise ValueError("dataset names must be non-empty and map to objects")
            loader = spec.get("loader", "ieeg_words")
            if loader not in allowed_loaders:
                raise ValueError(
                    f"Dataset {name!r} loader must be one of {sorted(allowed_loaders)}"
                )
            required = (
                {"loader", "dataset", "data_root", "results_root", "subjects"}
                if loader == "ieeg_words"
                else {
                    "loader",
                    "results_root",
                    "fmri_repo_root",
                    "embedding_root",
                    "sessions",
                }
            )
            missing = sorted(required - set(spec))
            if missing:
                raise ValueError(f"Dataset {name!r} is missing required fields {missing}")
            if loader == "ieeg_words" and not spec["subjects"]:
                raise ValueError(f"Dataset {name!r} subjects must be non-empty")
            if loader == "fmri_words" and not spec["sessions"]:
                raise ValueError(f"Dataset {name!r} sessions must be non-empty")
        allowed_modes = {
            "equal_capacity",
            "capacity_controlled",
            "full_neural_space",
        }
        if not self.comparison_modes or set(self.comparison_modes) - allowed_modes:
            raise ValueError(
                "comparison_modes must be drawn from "
                f"{sorted(allowed_modes)}"
            )
        if len(set(self.comparison_modes)) != len(self.comparison_modes):
            raise ValueError("comparison_modes must not contain duplicates")
        needs_saturation = bool(
            {"capacity_controlled", "full_neural_space"}
            & set(self.comparison_modes)
        )
        for name, spec in model_items:
            if not str(name).strip() or not isinstance(spec, dict):
                raise ValueError("model names must be non-empty and map to objects")
            required = {"offsets", "layers", "datasets"}
            allowed = required | {"model", "saturation_ranks"}
            if not required.issubset(spec) or set(spec) - allowed:
                raise ValueError(
                    f"Model {name!r} must contain exactly required fields "
                    f"{sorted(required)} and only optional field 'saturation_ranks', "
                    f"got {sorted(spec)}"
                )
            if not spec["offsets"] or any(isinstance(x, bool) for x in spec["offsets"]):
                raise ValueError(f"Model {name!r} offsets must contain integers")
            if not all(isinstance(x, int) for x in spec["offsets"]):
                raise ValueError(f"Model {name!r} offsets must contain integers")
            if not spec["layers"] or not all(
                isinstance(x, int) and not isinstance(x, bool) for x in spec["layers"]
            ):
                raise ValueError(f"Model {name!r} layers must contain integers")
            if not spec["datasets"] or any(
                dataset not in self.datasets for dataset in spec["datasets"]
            ):
                raise ValueError(
                    f"Model {name!r} datasets must reference configured datasets"
                )
            for field_name in required:
                if len(set(spec[field_name])) != len(spec[field_name]):
                    raise ValueError(
                        f"Model {name!r} {field_name} must not contain duplicates"
                    )
            saturation = spec.get("saturation_ranks")
            if needs_saturation and not isinstance(saturation, dict):
                raise ValueError(
                    f"Model {name!r} requires saturation_ranks for selected "
                    "comparison_modes"
                )
            if saturation is not None:
                required_kinds = {"rrr", "pca"} if self.include_pca else {"rrr"}
                known_kinds = {"rrr", "pca"}
                missing_kinds = required_kinds - set(saturation)
                unknown_kinds = set(saturation) - known_kinds
                if missing_kinds:
                    raise ValueError(
                        f"Model {name!r} saturation_ranks is missing required kinds "
                        f"{sorted(missing_kinds)}"
                    )
                if unknown_kinds:
                    raise ValueError(
                        f"Model {name!r} saturation_ranks contains unknown kinds "
                        f"{sorted(unknown_kinds)}; expected only {sorted(known_kinds)}"
                    )
                expected_datasets = set(spec["datasets"])
                expected_layers = set(spec["layers"])
                for kind, by_dataset in saturation.items():
                    if (
                        not isinstance(by_dataset, dict)
                        or set(by_dataset) != expected_datasets
                    ):
                        raise ValueError(
                            f"Model {name!r} saturation_ranks.{kind} must contain "
                            f"exactly datasets {sorted(expected_datasets)}"
                        )
                    for dataset, by_layer in by_dataset.items():
                        if not isinstance(by_layer, dict):
                            raise ValueError(
                                f"Model {name!r} saturation ranks for {kind}/{dataset} "
                                "must map layers to ranks"
                            )
                        normalized = {
                            int(layer): rank for layer, rank in by_layer.items()
                        }
                        if set(normalized) != expected_layers:
                            raise ValueError(
                                f"Model {name!r} saturation ranks for {kind}/{dataset} "
                                f"must contain exactly layers {sorted(expected_layers)}"
                            )
                        if any(
                            isinstance(rank, bool)
                            or not isinstance(rank, int)
                            or rank < 1
                            for rank in normalized.values()
                        ):
                            raise ValueError(
                                f"Model {name!r} saturation ranks for {kind}/{dataset} "
                                "must be positive integers"
                            )
        if not self.geometries or set(self.geometries) - {"span", "weighted"}:
            raise ValueError("geometries must be drawn from ['span', 'weighted']")
        if len(set(self.geometries)) != len(self.geometries):
            raise ValueError("geometries must not contain duplicates")
        if self.ridge_alphas is not None and (
            not self.ridge_alphas or any(float(alpha) <= 0 for alpha in self.ridge_alphas)
        ):
            raise ValueError("ridge_alphas must contain positive values")
        if self.rsm_metric not in {"cosine", "euclidean"}:
            raise ValueError("rsm_metric must be either 'cosine' or 'euclidean'")


@dataclass
class GenerateEmbeddingsConfig:
    """Parameters for generating contextual embeddings from a HuggingFace model
    over a dataset transcript (used by scripts/utilities/generate_embeddings.py).

    Output layout under ``{data_root}/stimuli/{output_name}/``:
      - features.hdf5  : one dataset per retained layer, ``layer-{L}`` → (n_tokens, hidden_dim)
      - transcript.tsv : per-token rows (word_idx, word, start, end, hftoken, token_id)
      - metadata.json  : generation parameters for reproducibility

    Generic sliding window: each forward pass processes ``context_window`` tokens and
    retains the last ``stride`` token positions (except the first window, which retains
    all of its positions). All model types tokenize each word independently with the
    space-prefixed convention from ``data/code/llm.py``. Causal models enforce left-only
    context natively; for bidirectional models use ``attention_mode='bidirectional_strict'``
    with ``stride=1`` to ensure each target position sees only preceding tokens.
    Podcast-compatible causal LM generation uses ``attention_mode='podcast_per_token'``
    and builds one ``context_window + 1`` row per target token.
    """

    hf_model_id: str = "openai-community/gpt2-xl"
    output_name: str | None = None  # None → derived from hf_model_id
    data_root: str = "data/"
    dataset: str = "podcast"
    per_dataset_config: dict = field(default_factory=dict)
    transcript_path: str = "data/stimuli/podcast_transcript.csv"

    # Sliding window
    context_window: int = 1024
    stride: int | None = (
        None  # None → auto (1 if bidirectional_strict, else context_window // 2)
    )

    # Layer selection
    layers: list[int] | str = "all"  # "all" or explicit list of hidden-state indices

    # Pooling strategy for multi-token words (applied downstream in the loader)
    pooling: str = "mean"  # "mean" | "last" | "first"

    # Attention mode
    # auto                  : causal_native for decoders, bidirectional_strict for encoders
    # causal_native         : standard causal LM forward pass
    # podcast_per_token     : legacy podcast causal LM contexting, logits, and tokenization
    # bidirectional_strict  : bidirectional attention, but stride=1 so each target sees only preceding tokens
    # bidirectional_masked  : bidirectional model with a causal attention mask (out-of-distribution; use with care)
    attention_mode: str = "auto"

    # Inference
    batch_size: int = 1
    device: str = "cuda"
    dtype: str = "float16"  # compute dtype: float32 | float16 | bfloat16
    output_dtype: str = "float16"  # stored dtype in HDF5

    # HuggingFace cache / offline
    hf_cache_dir: str | None = None  # None → honor HF_HOME / default
    local_files_only: bool = (
        True  # True for offline GPU hosts; operator must pre-populate cache
    )
    hf_token: str | None = (
        None  # for gated models (e.g. Llama-3); prefer HF_TOKEN env var
    )
    random_init_model: bool = (
        False  # If true, use hf_model_id only for architecture config
    )
    random_init_seed: int | None = (
        42  # None → intentionally unseeded random initialization
    )

    # Tokenizer
    tokenizer_name_or_path: str | None = (
        None  # None → use hf_model_id; can point to a local tokenizer dir
    )
    add_prefix_space: bool = True
    refresh: bool = False


@dataclass
class FMRIDataConfig:
    """Paths and preprocessing shared by the deep-fMRI-dataset analyses."""

    fmri_repo_root: str = "data/deep-fMRI-dataset"
    embedding_root: str = "data/fmri/stimuli"
    glove_data_root: str = "data"
    limit_to_glove: bool = False
    glove_dim: int = 300
    sessions: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    subjects: list[str] = field(default_factory=lambda: ["UTS03"])
    trim: int = 5
    ndelays: int = 4

    def validate(self):
        if self.glove_dim not in {50, 100, 200, 300}:
            raise ValueError("fmri.glove_dim must be one of [50, 100, 200, 300]")


@dataclass
class FMRIGenerateEmbeddingsConfig:
    """Generate one set of contextual embeddings per fMRI story."""

    generation: GenerateEmbeddingsConfig = field(default_factory=GenerateEmbeddingsConfig)
    fmri: FMRIDataConfig = field(default_factory=FMRIDataConfig)
    refresh: bool = False


@dataclass
class FMRIRidgeConfig:
    """Himalaya ridge settings shared by fMRI significance and rank sweeps."""

    alphas: list[float] | None = None
    alpha_grid: dict = field(default_factory=dict)
    solver_params: dict | None = None
    y_in_cpu: bool | None = None
    force_cpu: bool | None = None
    log_alphas: bool = False


@dataclass
class FMRISignificanceConfig:
    """Session-held-out voxel significance for deep-fMRI-dataset."""

    fmri: FMRIDataConfig = field(default_factory=FMRIDataConfig)
    ridge: FMRIRidgeConfig = field(default_factory=FMRIRidgeConfig)
    model_dir: str = "gpt2-xl"
    layer: int | list[int] | None = 24
    pooling: str = "mean"
    results_root: str = "results"
    plot: bool = True
    batch_subjects: bool = False
    num_null_shifts: int = 10000
    null_seed: int = 42
    null_batch_size: int = 1024
    minimal_shift: int = 10
    fdr_q: float = 0.05
    refresh: bool = False

    def validate(self):
        self.fmri.validate()
        is_glove = self.model_dir == "glove" or self.model_dir.startswith("glove-")
        if is_glove and self.layer is not None:
            raise ValueError("fMRI GloVe configurations require layer=null")
        if len(self.fmri.sessions) < 2:
            raise ValueError("fMRI significance requires at least two sessions")
        if not self.fmri.subjects:
            raise ValueError("At least one fMRI subject is required")
        if self.num_null_shifts < 1 or self.null_batch_size < 1:
            raise ValueError("Null counts and batch size must be positive")
        if self.minimal_shift < 1:
            raise ValueError("minimal_shift must be positive")
        if not 0 < self.fdr_q < 1:
            raise ValueError("fdr_q must lie in (0, 1)")

    def __post_init__(self):
        self.validate()


@dataclass
class FMRICrossDatasetRRRConfig:
    """Configuration for internal, cross-dataset, and LOSO fMRI RRR."""

    fmri: FMRIDataConfig = field(default_factory=FMRIDataConfig)
    ridge: FMRIRidgeConfig = field(default_factory=FMRIRidgeConfig)
    model_dir: str = "gpt2-xl"
    layer: int | list[int] | None = 24
    pooling: str = "mean"
    results_root: str = "results"
    plot: bool = True
    significance_selection: SignificanceSelectionConfig = field(
        default_factory=SignificanceSelectionConfig
    )
    ieeg_rrr_source: dict | None = None
    views: list[RRRViewConfig] = field(default_factory=list)
    multi_view_name: str | None = None
    banded_ridge_alphas: list[float] | None = None
    banded_ridge_solver_params: dict = field(
        default_factory=lambda: {"n_iter": 100, "random_state": 42}
    )
    source_mode: str = "cross_dataset"
    subspace_aggregation: str | None = None
    rrr_svd_on_cpu: bool = False
    rrr_use_low_rank_svd: bool = False
    rrr_low_rank_svd_seed: int = 0
    rrr_save_b_matrix: bool = False
    pca_svd_device: str = "auto"
    rank_grid: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128]
    )
    refresh: bool = False

    def validate(self):
        self.fmri.validate()
        self.significance_selection.validate()
        is_glove = self.model_dir == "glove" or self.model_dir.startswith("glove-")
        if is_glove and self.layer is not None:
            raise ValueError("fMRI GloVe configurations require layer=null")
        if self.source_mode not in {
            "cross_dataset", "within_fmri", "leave_one_subject_out", "multi_view"
        }:
            raise ValueError(
                "source_mode must be 'cross_dataset', 'within_fmri', "
                "'leave_one_subject_out', or 'multi_view', "
                f"got {self.source_mode!r}"
            )
        if self.subspace_aggregation is None and self.source_mode in {
            "within_fmri", "leave_one_subject_out", "multi_view"
        }:
            self.subspace_aggregation = "prediction_weighted"
        allowed_aggregation = (
            FMRI_FIR_SUBSPACE_AGGREGATION_METHODS
            if self.source_mode in {"within_fmri", "leave_one_subject_out", "multi_view"}
            else SUBSPACE_AGGREGATION_METHODS
        )
        if (
            self.subspace_aggregation is not None
            and self.subspace_aggregation not in allowed_aggregation
        ):
            raise ValueError(
                "subspace_aggregation must be one of "
                f"{sorted(allowed_aggregation)} for source_mode={self.source_mode!r}, "
                f"got {self.subspace_aggregation!r}"
            )
        if not self.fmri.subjects:
            raise ValueError("At least one fMRI subject is required")
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")
        if len(set(map(int, self.rank_grid))) != len(self.rank_grid):
            raise ValueError("rank_grid must not contain duplicates")
        if self.pca_svd_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("pca_svd_device must be one of ['auto', 'cpu', 'cuda']")
        if self.source_mode == "cross_dataset" and not self.ieeg_rrr_source:
            raise ValueError("ieeg_rrr_source is required")
        if self.source_mode == "leave_one_subject_out" and len(self.fmri.subjects) < 2:
            raise ValueError(
                "fMRI leave_one_subject_out requires at least two subjects"
            )
        if self.banded_ridge_alphas is not None and (
            not self.banded_ridge_alphas
            or any(float(alpha) < 0 for alpha in self.banded_ridge_alphas)
        ):
            raise ValueError("banded_ridge_alphas must contain non-negative values")
        if self.source_mode == "multi_view":
            if not self.multi_view_name:
                raise ValueError("source_mode=multi_view requires multi_view_name")
            if not self.views:
                raise ValueError("source_mode=multi_view requires at least one view")
            names = [view.name for view in self.views]
            if any(not name for name in names):
                raise ValueError("Every RRR view name must be non-empty")
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"Duplicate RRR view name(s): {duplicates}")
            for view in self.views:
                if int(view.rank) < 1:
                    raise ValueError("Every RRR view rank must be positive")
                if bool(view.ieeg_rrr_source) == bool(view.fmri_rrr_source):
                    raise ValueError(
                        f"RRR view {view.name!r} must define exactly one of "
                        "ieeg_rrr_source or fmri_rrr_source"
                    )

    def __post_init__(self):
        self.validate()
