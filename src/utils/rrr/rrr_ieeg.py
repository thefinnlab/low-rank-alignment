"""Compatibility facade for the iEEG reduced-rank regression runner.

The implementation is organized under :mod:`src.utils.rrr.ieeg_rrr_mode`.
Legacy helper imports remain available here so downstream callers do not need
to change as part of this organizational refactor.
"""

from src.utils.rrr.ieeg_rrr_mode import common as _common
from src.utils.rrr.ieeg_rrr_mode.common import *  # noqa: F401,F403
from src.utils.rrr.ieeg_rrr_mode.common import run_ieeg
from src.utils.rrr.ieeg_rrr_mode.cross_dataset import (
    run_cross_dataset as _run_cross_dataset_rrr,
)
from src.utils.rrr.ieeg_rrr_mode.cross_dataset_helpers import (
    _load_cross_dataset_A_by_rank,
    _load_cross_temporal_A_by_fold,
    _run_cross_dataset_subspace_for_unit,
)
from src.utils.rrr.ieeg_rrr_mode.cross_subject import (
    run_cross_subject as _run_cross_subject_rrr,
)
from src.utils.rrr.ieeg_rrr_mode.leave_one_subject_out import (
    _run_loso_transfer_for_unit,
    run_leave_one_subject_out as _run_leave_one_subject_out_rrr,
)
from src.utils.rrr.ieeg_rrr_mode.multi_view import run_multi_view as _run_multi_view_rrr
from src.utils.rrr.ieeg_rrr_mode.leave_one_region_out import (
    run_leave_one_region_out as _run_leave_one_region_out_rrr,
)
from src.utils.rrr.ieeg_rrr_mode.per_subject import (
    run_per_subject as _run_per_subject_rrr,
)
from src.utils.rrr.ieeg_rrr_mode.temporal_windows import (
    run_temporal_windows as _run_temporal_windows_rrr,
)
from src.utils.rrr.significance import load_significance_selection


def __getattr__(name):
    """Resolve legacy private helper imports from the implementation module."""
    return getattr(_common, name)
