"""Dataclass configs for encoding model training and significance testing."""

import os
from dataclasses import dataclass, field


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
