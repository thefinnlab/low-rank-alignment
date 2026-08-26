"""Leave-one-subject-out iEEG RRR mode."""

from .common import _run_leave_one_subject_out_rrr


def run_leave_one_subject_out(*args, **kwargs):
    return _run_leave_one_subject_out_rrr(*args, **kwargs)

__all__ = ["run_leave_one_subject_out"]
