"""Per-subject iEEG RRR mode."""

from .common import _run_per_subject_rrr


def run_per_subject(*args, **kwargs):
    return _run_per_subject_rrr(*args, **kwargs)

__all__ = ["run_per_subject"]
