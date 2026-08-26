"""Cross-subject/default iEEG RRR mode."""

from .common import _run_cross_subject_rrr


def run_cross_subject(*args, **kwargs):
    return _run_cross_subject_rrr(*args, **kwargs)

__all__ = ["run_cross_subject"]
