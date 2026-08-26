"""Compatibility facade for the iEEG reduced-rank regression runner.

The implementation is organized under :mod:`src.utils.rrr.ieeg_rrr_mode`.
Legacy helper imports remain available here so downstream callers do not need
to change as part of this organizational refactor.
"""

from src.utils.rrr.ieeg_rrr_mode import common as _common
from src.utils.rrr.ieeg_rrr_mode.common import *  # noqa: F401,F403
from src.utils.rrr.ieeg_rrr_mode.common import run_ieeg
from src.utils.rrr.significance import load_significance_selection


def __getattr__(name):
    """Resolve legacy private helper imports from the implementation module."""
    return getattr(_common, name)
