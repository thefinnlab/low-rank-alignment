"""Artifact compatibility for target-PCA variants in the shared RRR runners."""

import pandas as pd

from scripts.run_rrr import Artifact as RRRArtifact
from src.core.artifacts import readable_variant


class Artifact(RRRArtifact):
    """RRR-compatible outputs for one target-PCA dimensionality variant."""

    SCRIPT = "run_target_dimensionality_rrr"
    COMPACT_STORAGE = False
    FIELD_CONTRACT = {**RRRArtifact.FIELD_CONTRACT, "sweep_results": False}

    def __init__(self, *, target_pca_dim=None, **kwargs):
        significance_selection = kwargs.get("significance_selection")
        variant = kwargs.get("variant")
        if target_pca_dim is None or variant is not None:
            super().__init__(**kwargs)
        else:
            kwargs = dict(kwargs)
            kwargs.pop("significance_selection", None)
            kwargs["variant"] = readable_variant(
                {"significance_selection": significance_selection}
                if significance_selection is not None else {},
                extras={"target_pca_dim": int(target_pca_dim)},
            )
            super().__init__(**kwargs)
            self.requested_significance_selection = significance_selection
        self.target_pca_dim = (
            None if target_pca_dim is None else int(target_pca_dim)
        )

    def save_sweep_results(self, frame: pd.DataFrame, *, overwrite=False):
        self.save_field("sweep_results", frame, overwrite=overwrite)

    def load_sweep_results(self) -> pd.DataFrame:
        return self.load_field("sweep_results")

