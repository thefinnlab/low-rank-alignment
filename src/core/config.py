"""Dataclass configs for encoding model training and significance testing."""

import os
from dataclasses import dataclass, field
from enum import Enum


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
    model: str, offset: int, per_model_config: dict, base_dir: str = "results/"
) -> str:
    """Return a canonical results directory path derived from run parameters.

    Example: get_results_dir("gpt2xl", 24, 0) → "results/gpt2xl_layer24_offset0"
    """
    if model == "gpt2xl":
        return gpt2_xl_results_dir(base_dir, offset, **per_model_config)
    elif model == "glove":
        return glove_results_dir(base_dir, offset, **per_model_config)
    elif model == "transformer":
        return transformer_results_dir(base_dir, offset, **per_model_config)
    else:
        raise ValueError(f"Model: {model} not implemented yet.")


@dataclass
class EncodingConfig:
    """Parameters for encoding model training (used by train_encoding_models.py)."""

    data_root: str = "data/"
    model: str = (
        "gpt2xl"  # Embedding model name; used in results dir generation. Should be registered in src.core.registry.embedding_getter_registry
    )
    offset: int = 0  # Word offset for aligning embeddings to neural data
    subjects: list[int] | None = None  # None → defaults to 1–9 at runtime
    run_name: str | None = (
        None  # Output dir; auto-generated from model/layer/offset if None
    )
    sig_dir: str | None = (
        None  # Override dir for significance CSVs; defaults to run_name/output_dir
    )
    batch: bool = False  # Process subjects one at a time (lower peak RAM)
    ridge_alphas: list[float] | None = None  # None → use RIDGE_ALPHAS from constants.py
    log_alphas: bool = False  # Save alpha diagnostics (.npz + .png) per subject
    refine_alphas: bool = False  # Two-pass training: coarse sweep → refined grid
    arbitrary_seed: int = 42  # Seed used for generating arbitrary embedding baseline
    num_folds: int = 2  # Number of cross-validation folds
    limit_to_glove: bool = (
        False  # If true, only run over words which have a glove embedding.
    )
    # Per model type config. Used when generating embeddings in load_embeddings.
    per_model_config: dict = field(default_factory=lambda: {"layer": 24})


@dataclass
class SignificanceConfig:
    """Parameters for significance testing (superset of encoding)."""

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = (
        None  # Results dir; auto-generated from model/layer/offset if None
    )
    minimal_shift: int = 500
    fdr_q: float = 0.01
    save_weights: bool = False


@dataclass
class NullDistConfig:
    """Parameters for generating phase-randomization null distributions.

    Each invocation of generate_null_distribution.py runs ``n_iterations``
    permutations and writes results to a uniquely-named file inside
    ``{null_dir}/sub_{id}/``.  Multiple invocations can run in parallel
    to accumulate the full null distribution.
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = None  # Root results dir (for lag computation, etc.)
    null_dir: str | None = (
        None  # Where to write null distribution shards; defaults to {output_dir}/null_dist
    )
    n_iterations: int = 50  # Number of permutations per invocation
    batch_size: int = 5  # Permutations to batch into a single model (limited by RAM)
    seed: int | None = None  # None = random seed per invocation (for parallel batching)


@dataclass
class NewSignificanceConfig:
    """Parameters for computing significance from pre-built null distributions.

    Reads null distribution shards produced by generate_null_distribution.py,
    re-runs encoding on real data to get observed correlations, and applies
    the max-statistic significance procedure with FDR correction.
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = None  # Where to write significance CSVs
    null_dir: str | None = None  # Where null distribution shards live
    fdr_q: float = 0.01  # FDR threshold for Benjamini-Hochberg correction
    min_permutations: int = 5000  # Warn if fewer total permutations found


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

    # Optional metrics (expensive; disabled by default to speed up rank sweeps)
    compute_cca: bool = False
    compute_procrustes: bool = False
    compute_sig_nonsig: bool = True

    # Alpha diagnostics
    log_alphas: bool = False  # Save per-rank alpha diagnostics in RRR output

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
class PCAConfig:
    """Parameters for PCA-based subspace analysis (run_pca.py).

    Computes PCA on word embeddings to find a low-dimensional input subspace,
    then trains ridge regression from the projected embeddings to neural targets.
    Sweeps over a rank grid analogous to RRR.

    Mode → output directory mapping (default, overridable via pca_output_dir):
        cross_subject : {output_dir}/pca/
        per_subject   : {output_dir}/per_subject_pca/
        per_region    : {output_dir}/per_region_pca/
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = (
        None  # Root results dir; auto from model/layer/offset if None
    )
    sig_dir: str | None = (
        None  # Override dir for significance CSVs; defaults to output_dir
    )
    pca_output_dir: str | None = (
        None  # Override for PCA root subdir; default depends on mode
    )
    mode: ScriptMode = ScriptMode.CROSS_SUBJECT

    # Rank sweep
    rank_grid: list[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 40, 100, 200, 400, 800, 1600]
    )

    # Optional metrics (expensive; disabled by default)
    compute_cca: bool = False
    compute_procrustes: bool = False
    compute_sig_nonsig: bool = True

    # Alpha diagnostics
    log_alphas: bool = False

    # Arbitrary embedding seed for null baseline
    arbitrary_seed: int = 42

    # Per-region mode
    atlas_path: str | None = None
    atlas_region_groups: dict | None = None


@dataclass
class PartialRRRConfig:
    """Parameters for partial RRR analysis (run_partial_rrr.py).

    Loads RRR-learned subspaces (A matrices) for each rank k, partials them
    out of the word embeddings, and retrains ridge regression to measure how
    much each rank-k subspace contributes to encoding performance.

    Mode → RRR source / output directory mapping (defaults, overridable):
        cross_subject : RRR in {output_dir}/rrr/,                  out {output_dir}/partial_rrr/
        per_subject   : RRR in {output_dir}/per_subject_rrr/,      out {output_dir}/per_subject_partial_rrr/
        per_region    : RRR in {output_dir}/per_region_rrr/,       out {output_dir}/per_region_partial_rrr/
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = (
        None  # Root results dir; auto from model/layer/offset if None
    )
    sig_dir: str | None = (
        None  # Override dir for significance CSVs; defaults to output_dir
    )
    partial_rrr_output_dir: str | None = (
        None  # Override for output subdir; default depends on mode
    )
    rrr_source_dir: str | None = (
        None  # Where to find RRR A matrices; auto-resolved per mode
    )
    mode: ScriptMode = ScriptMode.CROSS_SUBJECT

    # Rank sweep — None discovers all ranks from rrr_source_dir; set to filter
    rank_grid: list[int] | None = None

    # Per-region mode
    atlas_path: str | None = (
        None  # Path to Destrieux 2009 .nii.gz; None = download via nilearn
    )
    atlas_region_groups: dict | None = None  # Required when mode=per_region


@dataclass
class SAEConfig:
    """Parameters for SAE-based encoding analysis (run_sae.py).

    Loads an RRR-learned subspace (A matrix) at a given rank k, projects word
    embeddings into that k-dimensional space, applies a sparse autoencoder
    (overcomplete hidden layer with ReLU + L1 penalty), and regresses onto
    neural targets.

    Mode → SAE output directory mapping (default, overridable via sae_output_dir):
        cross_subject : {output_dir}/sae/
        per_subject   : {output_dir}/per_subject_sae/
        per_region    : {output_dir}/per_region_sae/
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = (
        None  # Root results dir; auto from model/layer/offset if None
    )
    sig_dir: str | None = (
        None  # Override dir for significance CSVs; defaults to output_dir
    )
    sae_output_dir: str | None = (
        None  # Override for SAE root subdir; default depends on mode
    )
    rrr_source_dir: str | None = (
        None  # Where to find RRR A matrices; auto-resolved per mode
    )
    mode: ScriptMode = ScriptMode.CROSS_SUBJECT

    # RRR rank to use
    k: int = 10

    # SAE architecture
    hidden_dim: int = 50  # Must be > k
    l1_penalty: float = 1e-3
    l1_warmup_start: int = 0  # Epoch/cycle to begin L1 ramp (0 = immediate)
    l1_warmup_epochs: int = (
        0  # Epochs/cycles to linearly ramp L1 to target (0 = no ramp)
    )

    # Training mode
    training_mode: str = "end_to_end"  # "end_to_end" or "alternating"
    regression_type: str = "ols"  # "ols" or "ridge"

    # End-to-end params
    learning_rate: float = 1e-3
    num_epochs: int = 500
    batch_size: int | None = None  # None = full batch
    log_every: int = 10  # Compute & display test metrics every N epochs/cycles

    # Alternating params
    sae_learning_rate: float = 1e-3
    sae_steps_per_cycle: int = 50
    num_cycles: int = 20
    regression_momentum: float = (
        1.0  # Proximal damping: W = (1-η)W_old + η*W_optimal (1.0 = no damping)
    )

    # Input jitter (Gaussian noise on X_proj during SAE training, 0 = disabled)
    input_jitter_std: float = 0.0

    # Dead feature resampling (0 = disabled)
    resample_interval: int = 0  # Resample dead neurons every N epochs/cycles

    # Norm-weighted L1 (Anthropic 2024): weight L1 by regression column norms
    weight_l1_by_decoder_norm: bool = False

    # Track recovery correlation during training (expensive — runs sufficiency test at each log step)
    track_recovery_during_training: bool = False

    # Subject/region limiting
    max_subjects: int | None = None  # Limit number of subjects to run
    max_regions: int | None = None  # Limit number of regions to run

    # Per-region mode
    atlas_path: str | None = None
    atlas_region_groups: dict | None = None


@dataclass
class AblationConfig:
    """Parameters for RRR latent space ablation study (run_ablation.py).

    Loads pre-computed RRR A and B matrices, projects embeddings into the
    latent space, applies quantile-based masks (tail-only, center-only,
    random), and evaluates neural reconstruction under each mask by training
    a fresh ridge regression from the masked projections to neural targets.

    Mode -> output directory mapping (default, overridable via ablation_output_dir):
        cross_subject : {output_dir}/ablation/
        per_subject   : {output_dir}/per_subject_ablation/
        per_region    : {output_dir}/per_region_ablation/
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = (
        None  # Root results dir; auto from model/layer/offset if None
    )
    sig_dir: str | None = (
        None  # Override dir for significance CSVs; defaults to output_dir
    )
    ablation_output_dir: str | None = None  # Override for ablation root subdir
    rrr_source_dir: str | None = (
        None  # Where to find RRR A/B matrices; auto-resolved per mode
    )
    mode: ScriptMode = ScriptMode.CROSS_SUBJECT

    # Rank sweep — which top-k dimensions to evaluate
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])

    # Percentile sweep — center zone sizes (percentage of distribution)
    center_pcts: list[int] = field(default_factory=lambda: [10, 30, 50, 70, 90])

    # Mask types to evaluate
    mask_types: list[str] = field(default_factory=lambda: ["tail", "center", "random"])

    # Number of random seeds for random mask (averaged for stability)
    n_random_seeds: int = 5

    # Sig/nonsig split
    compute_sig_nonsig: bool = True

    # Per-region mode
    atlas_path: str | None = None
    atlas_region_groups: dict | None = None


@dataclass
class GenerateEmbeddingsConfig:
    """Parameters for generating contextual embeddings from a HuggingFace model
    over the podcast transcript (used by scripts/utilities/generate_embeddings.py).

    Output layout under ``{data_root}/stimuli/{output_name}/``:
      - features.hdf5  : one dataset per retained layer, ``layer-{L}`` → (n_tokens, hidden_dim)
      - transcript.tsv : per-token rows (word_idx, word, start, end, hftoken, token_id)
      - metadata.json  : generation parameters for reproducibility

    Sliding window: each forward pass processes ``context_window`` tokens and retains
    the last ``stride`` token positions (except the first window, which retains all of
    its positions). Causal models enforce left-only context natively; for bidirectional
    models use ``attention_mode='bidirectional_strict'`` with ``stride=1`` to ensure each
    target position sees only preceding tokens.
    """

    hf_model_id: str = "openai-community/gpt2-xl"
    output_name: str | None = None  # None → derived from hf_model_id
    data_root: str = "data/"
    transcript_path: str = "data/stimuli/podcast_transcript.csv"

    # Sliding window
    context_window: int = 1024
    stride: int | None = None  # None → auto (1 if bidirectional_strict, else context_window // 2)

    # Layer selection
    layers: list[int] | str = "all"  # "all" or explicit list of hidden-state indices

    # Pooling strategy for multi-token words (applied downstream in the loader)
    pooling: str = "mean"  # "mean" | "last" | "first"

    # Attention mode
    # auto                  : causal_native for decoders, bidirectional_strict for encoders
    # causal_native         : standard causal LM forward pass
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
    local_files_only: bool = True  # True for offline GPU hosts; operator must pre-populate cache
    hf_token: str | None = None  # for gated models (e.g. Llama-3); prefer HF_TOKEN env var

    # Tokenizer
    add_prefix_space: bool = True  # required for GPT-2 / RoBERTa BPE with is_split_into_words
