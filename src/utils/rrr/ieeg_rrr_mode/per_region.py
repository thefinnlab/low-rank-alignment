"""Per-region iEEG RRR mode."""

from .common import _run_per_region_rrr


def run_per_region(*args, **kwargs):
    return _run_per_region_rrr(*args, **kwargs)

__all__ = ["run_per_region"]
