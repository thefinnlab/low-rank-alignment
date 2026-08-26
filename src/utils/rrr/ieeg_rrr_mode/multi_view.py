"""Multiview iEEG RRR mode."""

from .common import _run_multi_view_rrr


def run_multi_view(*args, **kwargs):
    return _run_multi_view_rrr(*args, **kwargs)

__all__ = ["run_multi_view"]
