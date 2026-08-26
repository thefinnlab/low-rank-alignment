"""Temporal-window iEEG RRR mode."""

from .common import _run_temporal_windows_rrr


def run_temporal_windows(*args, **kwargs):
    return _run_temporal_windows_rrr(*args, **kwargs)

__all__ = ["run_temporal_windows"]
