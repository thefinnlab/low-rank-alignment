"""Dataclass configs for encoding model training and significance testing."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScriptMode(str, Enum):
    """Which Script branch to run in analysis"""

    CROSS_SUBJECT = "cross_subject"
    PER_SUBJECT = "per_subject"
    PER_REGION = "per_region"
    LEAVE_ONE_SUBJECT_OUT = "leave_one_subject_out"


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
    # Per model type config. Used when generating embeddings in load_embeddings.
    per_model_config: dict = field(default_factory=lambda: {"layer": 24})
    per_dataset_config: dict = field(default_factory=dict)


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
    null_batch_size: int = 16
    device: str | None = None
    fdr_q: float = 0.01
    save_weights: bool = False
    min_cluster_size: int = (
        1  # Heuristic significance: minimum generated cluster size.
    )


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

    # Rank sweep
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])
    refresh: bool = False  # If true, recompute all ranks even when outputs exist.

    # Optional metrics (expensive; disabled by default to speed up rank sweeps)
    compute_cca: bool = False
    compute_procrustes: bool = False
    compute_sig_nonsig: bool = True
    compute_joint_ridge: bool = False  # Deprecated; ignored by RRR/PCA rank sweeps.

    # Alpha diagnostics
    log_alphas: bool = False  # Save per-rank alpha diagnostics in RRR output
    standardize_subspace_inputs: bool = (
        False  # Preserve RRR/PCA subspace coordinate scales by default.
    )

    # Arbitrary embedding seed for null baseline
    arbitrary_seed: int = 42

    # PCA baseline: run PCA subspace analysis alongside RRR using the same rank_grid
    run_pca_baseline: bool = False

    # Per-region mode
    atlas_path: str | None = (
        None  # Path to Destrieux 2009 .nii.gz; None = download via nilearn
    )
    atlas_region_groups: dict | None = None  # Required when mode=per_region


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
    all of its positions). Causal models enforce left-only context natively; for
    bidirectional models use ``attention_mode='bidirectional_strict'`` with ``stride=1``
    to ensure each target position sees only preceding tokens. Podcast-compatible
    causal LM generation uses ``attention_mode='podcast_per_token'`` and builds one
    ``context_window + 1`` row per target token.
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
    add_prefix_space: bool = (
        True  # required for GPT-2 / RoBERTa BPE with is_split_into_words
    )
