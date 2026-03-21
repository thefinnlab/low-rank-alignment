"""Dataclass configs for encoding model training and significance testing."""

import os
from dataclasses import dataclass, field
from enum import Enum


class SVDMode(str, Enum):
    """Which SVD branch to run in run_svd_analysis.py."""
    CROSS_SUBJECT = "cross_subject"
    PER_SUBJECT = "per_subject"
    PER_REGION = "per_region"


def get_results_dir(model: str, layer: int, offset: int, base_dir: str = "results/") -> str:
    """Return a canonical results directory path derived from run parameters.

    Example: get_results_dir("gpt2xl", 24, 0) → "results/gpt2xl_layer24_offset0"
    """
    return os.path.join(base_dir, f"{model}_layer{layer}_offset{offset}")


@dataclass
class EncodingConfig:
    """Parameters for encoding model training (used by train_encoding_models.py)."""

    data_root: str = "data/"
    model: str = "gpt2xl"    # Embedding model name; used in results dir generation
    layer: int = 24          # GPT-2-XL layer index (24=mid, 48=final)
    offset: int = 0          # Word offset for aligning embeddings to neural data
    subjects: list[int] | None = None   # None → defaults to 1–9 at runtime
    run_name: str | None = None  # Output dir; auto-generated from model/layer/offset if None
    batch: bool = False      # Process subjects one at a time (lower peak RAM)


@dataclass
class SignificanceConfig:
    """Parameters for significance testing (superset of encoding)."""

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = None  # Results dir; auto-generated from model/layer/offset if None
    n_permutations: int = 5000
    fdr_q: float = 0.01
    save_weights: bool = False


@dataclass
class SVDConfig:
    """Parameters for SVD decomposition analysis (run_svd_analysis.py)."""

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = None        # Root results dir; auto-generated if None
    svd_output_dir: str | None = None    # Override for SVD subdir; defaults to {output_dir}/svd
    n_directions: int | None = None      # Max singular vectors to compute; None → full SVD
    k_values: list[int] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1600]
    )
    threshold: float = 0.95              # Performance retention threshold for best_k
    mode: SVDMode = SVDMode.CROSS_SUBJECT  # Which SVD branch to run
    atlas_path: str | None = None        # Path to Destrieux 2009 .nii.gz; None = download via nilearn
    atlas_region_groups: dict | None = None  # Required when mode=per_region


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
    output_dir: str | None = None         # Root results dir; auto from model/layer/offset if None
    rrr_output_dir: str | None = None     # Override for RRR root subdir; default depends on mode
    mode: SVDMode = SVDMode.CROSS_SUBJECT

    # Rank sweep
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])

    # Optional metrics (expensive; disabled by default to speed up rank sweeps)
    compute_cca: bool = False
    compute_procrustes: bool = False


@dataclass
class SRRRConfig:
    """Parameters for Sparse Reduced Rank Regression analysis (run_srrr.py).

    Reads SVD decompositions written by run_svd_analysis.py and finds the
    minimal subset of SVD directions that captures all significant encoding
    variance, via stability selection and a statistical sufficiency test.

    Mode → SVD input / SRRR output directory mapping (default, overridable via srrr_output_dir):
        cross_subject : SVD in {output_dir}/svd/,               SRRR in {output_dir}/srrr/
        per_subject   : SVD in {output_dir}/per_subject_svd/,   SRRR in {output_dir}/per_subject_srrr/
        per_region    : SVD in {output_dir}/per_region_svd/,    SRRR in {output_dir}/per_region_srrr/
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    output_dir: str | None = None         # Root results dir; auto from model/layer/offset if None
    srrr_output_dir: str | None = None    # Override for SRRR root subdir; default depends on mode
    mode: SVDMode = SVDMode.CROSS_SUBJECT # Which SVD outputs to read (mirrors SVDConfig.mode)

    # Model grid
    rank_grid: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])
    lambda_grid: list[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])

    # ProxGD
    max_iter: int = 500
    tol: float = 1e-5
    learning_rate: float = 1e-3

    # Stability selection
    n_stability_iters: int = 100
    subsample_fraction: float = 0.8
    selection_threshold: float = 0.8     # Min P_i to include direction in minimal model

    # Sufficiency test
    use_null_dist: bool = False          # Set True to enable null distribution and sufficiency test
    n_permutations: int = 5000
    p_value_threshold: float = 0.05      # 95th percentile noise floor
    cache_null_dist: bool = True

    # Analysis
    knee_sensitivity: float = 1.0        # Kneedle sensitivity parameter

    # Optional metrics (expensive; disabled by default to speed up grid sweeps)
    compute_cca: bool = False
    compute_procrustes: bool = False
