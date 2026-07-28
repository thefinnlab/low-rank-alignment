"""Dataclass configs for encoding model training and significance testing."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScriptMode(str, Enum):
    """Which Script branch to run in analysis"""

    CROSS_SUBJECT = "cross_subject"
    CROSS_DATASET = "cross_dataset"
    PER_SUBJECT = "per_subject"
    PER_REGION = "per_region"
    LEAVE_ONE_SUBJECT_OUT = "leave_one_subject_out"
    MULTI_VIEW = "multi_view"


class LanguageDecodingMode(str, Enum):
    """Relationship between the decoding dataset and saved RRR factors."""

    WITHIN_DATASET = "within_dataset"
    CROSS_DATASET = "cross_dataset"


SUBSPACE_AGGREGATION_METHODS = {"extrinsic", "unnormalized"}


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
    run_name: str | None = (
        None  # Output dir; auto-generated from model/layer/offset if None
    )
    sig_dir: str | None = (
        None  # Override dir for significance CSVs; defaults to run_name/output_dir
    )
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
        False  # If true, only run over words which have a glove embedding.
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
    output_dir: str | None = (
        None  # Results dir; auto-generated from model/layer/offset if None
    )
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
    rrr_config: str
    unit_dir: str
    rank: int
    kind: str | None = None


@dataclass
class RRRConfig:
    """Parameters for raw-input Reduced Rank Regression analysis (run_rrr.py).

    Performs RRR directly from raw embeddings to neural targets — no SVD projection.
    Sweeps over a rank grid to find the minimal rank that captures all variance
    explained by an unconstrained ridge baseline.

    Mode → output directory mapping (default, overridable via rrr_output_dir):
        cross_subject : {output_dir}/rrr/
        per_subject   : {output_dir}/per_subject_rrr/
        per_region    : {output_dir}/per_region_rrr/
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = (
        None  # Root results dir; auto from model/layer/offset if None
    )
    sig_dir: str | None = (
        None  # Override dir for significance CSVs; defaults to output_dir
    )
    rrr_output_dir: str | None = (
        None  # Override for RRR root subdir; default depends on mode
    )
    mode: ScriptMode = ScriptMode.CROSS_SUBJECT
    cross_result_dir: str | None = (
        None  # Source results run root for mode=cross_dataset.
    )
    views: list[RRRViewConfig] = field(default_factory=list)
    subspace_aggregation: str = "extrinsic"

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
    orthonormalize_a_per_rank: bool = (
        False  # QR-orthonormalize A[:, :k] before each projected ridge fit.
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

    # Per-region mode
    atlas_path: str | None = (
        None  # Path to Destrieux 2009 .nii.gz; None = download via nilearn
    )
    atlas_region_groups: dict | None = None  # Required when mode=per_region

    def __post_init__(self):
        self.validate()

    def validate(self):
        _validate_subspace_aggregation(self.subspace_aggregation)
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
        names = [view.name for view in self.views]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate RRR view name(s): {duplicates}")
        if any(not view.name for view in self.views):
            raise ValueError("Every RRR view name must be non-empty")
        if any(int(view.rank) < 1 for view in self.views):
            raise ValueError("Every RRR view rank must be positive")


@dataclass
class LanguageDecodingConfig:
    """Decode word-aligned linguistic targets from saved RRR/PCA subspaces."""

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    rrr_run_path: str | None = None
    pca_run_path: str | None = None
    mode: LanguageDecodingMode = LanguageDecodingMode.WITHIN_DATASET
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
        _validate_subspace_aggregation(self.subspace_aggregation)
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of ['auto', 'cpu', 'cuda']")
        if not self.rrr_run_path:
            raise ValueError("LanguageDecodingConfig.rrr_run_path is required")
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


@dataclass
class LanguageReconstructionConfig:
    """Linearly reconstruct saved language-model subspaces from fixed predictors."""

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    rrr_run_path: str | None = None
    mode: LanguageDecodingMode = LanguageDecodingMode.WITHIN_DATASET
    source_name: str | None = None
    fold_results_dir: str | None = None
    output_dir: str | None = None
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
        _validate_subspace_aggregation(self.subspace_aggregation)
        if not self.rrr_run_path:
            raise ValueError("LanguageReconstructionConfig.rrr_run_path is required")
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
class LanguageAxisEncodingConfig:
    """Encode neural responses from saved language-decoder task axes."""

    language_decoding_run_path: str | None = None
    output_dir: str | None = None
    rank_grid: list[int] = field(default_factory=lambda: [32])
    tasks: list[str] = field(
        default_factory=lambda: [
            "surprisal",
            "entropy",
            "iu_position",
            "iu_onset",
            "sentence_position",
        ]
    )
    geometries: list[str] = field(default_factory=lambda: ["raw", "whitened"])
    random_control_draws: int = 20
    random_seed: int = 42
    device: str = "auto"
    refresh: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not self.language_decoding_run_path:
            raise ValueError("language_decoding_run_path is required")
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")
        if len(set(map(int, self.rank_grid))) != len(self.rank_grid):
            raise ValueError("rank_grid must not contain duplicate ranks")
        if not self.tasks or any(not str(task).strip() for task in self.tasks):
            raise ValueError("tasks must contain non-empty names")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must not contain duplicate names")
        allowed = {"raw", "whitened"}
        if not self.geometries or set(self.geometries) - allowed:
            raise ValueError(f"geometries must be drawn from {sorted(allowed)}")
        if len(set(self.geometries)) != len(self.geometries):
            raise ValueError("geometries must not contain duplicate names")
        if isinstance(self.random_control_draws, bool) or self.random_control_draws < 0:
            raise ValueError("random_control_draws must be a non-negative integer")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of ['auto', 'cpu', 'cuda']")


@dataclass
class LanguageFeatureGenerationConfig:
    """Generate reusable, stimulus-aligned language annotation artifacts."""

    dataset: str = "podcast"
    data_root: str = "data/"
    per_dataset_config: dict = field(default_factory=dict)
    model_dir: str = "gpt2-xl"
    output_dir: str = "processed_data/language_features"
    resource_dir: str = "data/language_resources"
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
    """Parameters for pairwise RRR/PCA subspace reproducibility analysis."""

    runs: list[SubspaceComparisonRun] = field(default_factory=list)
    cca_ranks: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    rsa_ranks: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    output_name: str = "subspace_reproducibility"
    refresh: bool = False
    shared_corr_threshold: float = 0.2
    rsm_metric: str = "cosine"
    save_ambient_axes: bool = False
    save_rsm_vectors: bool = False
    include_self: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not self.runs:
            raise ValueError("SubspaceComparisonConfig.runs must contain at least one run")
        names = [run.name for run in self.runs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate subspace comparison run name(s): {duplicates}")
        for field_name in ("cca_ranks", "rsa_ranks"):
            ranks = getattr(self, field_name)
            if any(int(rank) < 1 for rank in ranks):
                raise ValueError(f"{field_name} must contain positive integer ranks")
        if not self.output_name:
            raise ValueError("output_name must be non-empty")
        if self.rsm_metric not in {"cosine", "euclidean"}:
            raise ValueError("rsm_metric must be either 'cosine' or 'euclidean'")


@dataclass
class MaxvarGCCAViewConfig:
    """One brain-aligned subspace view in a MAXVAR GCCA analysis."""

    name: str
    rrr_config: str
    unit_dir: str
    rank: int
    kind: str | None = None


@dataclass
class MaxvarGCCAConfig:
    """Configuration for anchor-scoped partial-support MAXVAR GCCA."""

    anchor: str = ""
    views: list[MaxvarGCCAViewConfig] = field(default_factory=list)
    shared_dimensions: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    ridge_alphas: list[float] = field(
        default_factory=lambda: [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    )
    support_cumulative_mass: float = 0.8
    axis_match_correlation_threshold: float = 0.8
    relative_neighboring_eigengap_threshold: float = 0.05
    output_root: str = "results/maxvar_gcca"
    output_dir: str | None = None
    subspace_aggregation: str = "extrinsic"
    refresh: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        _validate_subspace_aggregation(self.subspace_aggregation)
        if len(self.views) < 2:
            raise ValueError("MaxvarGCCAConfig.views must contain at least two views")
        names = [view.name for view in self.views]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate GCCA view name(s): {duplicates}")
        if self.anchor not in names:
            raise ValueError("MaxvarGCCAConfig.anchor must name one configured view")
        if any(int(view.rank) < 1 for view in self.views):
            raise ValueError("Every GCCA view rank must be positive")
        dimensions = [int(value) for value in self.shared_dimensions]
        if not dimensions or any(value < 1 for value in dimensions):
            raise ValueError("shared_dimensions must contain positive integers")
        if max(dimensions) > sum(int(view.rank) for view in self.views):
            raise ValueError("A shared dimension exceeds the configured union-rank upper bound")
        if not self.ridge_alphas or any(float(value) < 0 for value in self.ridge_alphas):
            raise ValueError("ridge_alphas must contain non-negative values")
        if not 0 < float(self.support_cumulative_mass) <= 1:
            raise ValueError("support_cumulative_mass must be in (0, 1]")
        for name, value in (
            ("axis_match_correlation_threshold", self.axis_match_correlation_threshold),
            ("relative_neighboring_eigengap_threshold", self.relative_neighboring_eigengap_threshold),
        ):
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not self.output_root or self.output_dir == "":
            raise ValueError("output_root must be non-empty and output_dir cannot be empty")


@dataclass
class GCCAEncodingConfig:
    """Downstream neural encoding from saved fold-specific GCCA scores."""

    gcca_dir: str = ""
    run_name: str = ""
    shared_dimensions: list[int] = field(default_factory=list)
    refresh: bool = False
    averaging_method: str = "contribution"
    ridge_alphas: list[float] | None = None
    ridge_cv_folds: int = 5
    ridge_solver_params: dict | None = None
    ridge_y_in_cpu: bool | None = None
    ridge_force_cpu: bool | None = None
    alpha_grid: dict = field(default_factory=dict)
    standardize_inputs: bool = False
    run_rrr: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not self.gcca_dir:
            raise ValueError("gcca_dir must be non-empty")
        if not self.run_name or any(ch in self.run_name for ch in "/\\"):
            raise ValueError("run_name must be a non-empty directory name")
        allowed = {"contribution", "equal", "inner_agreement", "inverse_error", "hybrid"}
        if self.averaging_method not in allowed:
            raise ValueError(f"averaging_method must be one of {sorted(allowed)}")
        if self.shared_dimensions and any(int(k) < 1 for k in self.shared_dimensions):
            raise ValueError("shared_dimensions must contain positive integers")
        if self.ridge_alphas is not None and (
            not self.ridge_alphas or any(float(a) < 0 for a in self.ridge_alphas)
        ):
            raise ValueError("ridge_alphas must contain non-negative values")


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
    sessions: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    subjects: list[str] = field(default_factory=lambda: ["UTS03"])
    trim: int = 5
    ndelays: int = 4


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
    layer: int = 24
    pooling: str = "mean"
    output_dir: str = "results/fmri/gpt2-xl_layer24/significance"
    batch_subjects: bool = False
    num_null_shifts: int = 10000
    null_seed: int = 42
    null_batch_size: int = 1024
    minimal_shift: int = 10
    fdr_q: float = 0.05
    refresh: bool = False

    def validate(self):
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
    """Evaluate saved iEEG RRR/PCA input spaces on fMRI responses."""

    fmri: FMRIDataConfig = field(default_factory=FMRIDataConfig)
    ridge: FMRIRidgeConfig = field(default_factory=FMRIRidgeConfig)
    model_dir: str = "gpt2-xl"
    layer: int = 24
    pooling: str = "mean"
    significance_dir: str = "results/fmri/gpt2-xl_layer24/significance"
    source_rrr_unit_dir: str = "results/gpt2-xl_layer24_offset0/rrr"
    output_dir: str = "results/fmri/gpt2-xl_layer24/cross_dataset_rrr"
    source_mode: str = "cross_dataset"
    subspace_aggregation: str = "extrinsic"
    rrr_svd_on_cpu: bool = False
    rrr_use_low_rank_svd: bool = False
    rrr_low_rank_svd_seed: int = 0
    rrr_save_b_matrix: bool = False
    rank_grid: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128]
    )
    refresh: bool = False

    def validate(self):
        _validate_subspace_aggregation(self.subspace_aggregation)
        if self.source_mode not in {"cross_dataset", "within_fmri"}:
            raise ValueError(
                "source_mode must be 'cross_dataset' or 'within_fmri', "
                f"got {self.source_mode!r}"
            )
        if not self.fmri.subjects:
            raise ValueError("At least one fMRI subject is required")
        if not self.rank_grid or any(int(rank) < 1 for rank in self.rank_grid):
            raise ValueError("rank_grid must contain positive integers")
        if len(set(map(int, self.rank_grid))) != len(self.rank_grid):
            raise ValueError("rank_grid must not contain duplicates")
        if self.source_mode == "cross_dataset" and not self.source_rrr_unit_dir:
            raise ValueError("source_rrr_unit_dir is required")

    def __post_init__(self):
        self.validate()
