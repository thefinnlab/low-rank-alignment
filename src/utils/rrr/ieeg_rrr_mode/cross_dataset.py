"""Cross-dataset iEEG RRR mode."""

from .common import (
    _load_cross_dataset_A_by_rank,
    _run_cross_dataset_rrr,
    _run_cross_dataset_subspace_for_unit,
)

def run_cross_dataset(*args, **kwargs):
    return _run_cross_dataset_rrr(*args, **kwargs)

__all__ = [
    "run_cross_dataset",
    "_load_cross_dataset_A_by_rank",
    "_run_cross_dataset_subspace_for_unit",
]
