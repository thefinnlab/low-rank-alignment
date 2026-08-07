"""Backfill strict word-row provenance into existing Podcast encoding results.

This utility does not retrain any model or load neural epoch arrays. It reconstructs
the row identities from each subject's existing HDF5 epoch cache, verifies them
against the saved cross-validation folds, and atomically adds ``word_selection`` to
each ``encoding_results.joblib``.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import yaml

from src.utils.data_loaders import (
    build_word_selection_metadata,
    load_precomputed_epochs,
)
from src.utils.encoding import load_encoding_results


SUBJECT_DIR_RE = re.compile(r"sub_(\d+)$")


@dataclass(frozen=True)
class BackfillPlan:
    path: Path
    result: dict
    metadata: dict
    already_complete: bool


def _load_run_config(result_root: Path) -> dict:
    config_path = result_root / "significance" / "config.yml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing significance config for result directory {result_root}: "
            f"expected {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("encoding"), dict):
        raise ValueError(f"Missing encoding configuration in {config_path}")
    return config["encoding"]


def discover_podcast_result_roots(results_root: Path) -> list[tuple[Path, dict]]:
    """Find configured top-level Podcast roots, excluding all GloVe runs."""
    results_root = results_root.resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    candidates = sorted(path.parent for path in results_root.glob("*/subjects"))
    if not candidates:
        raise FileNotFoundError(
            f"No top-level result directories with subjects found in {results_root}"
        )

    selected = []
    for result_root in candidates:
        if result_root.name.startswith("glove_"):
            continue
        enc = _load_run_config(result_root)
        if enc.get("dataset") != "podcast":
            continue
        if enc.get("model") == "glove":
            continue
        if bool(enc.get("limit_to_glove", False)):
            raise ValueError(
                f"Non-GloVe Podcast run unexpectedly applies a GloVe row filter: "
                f"{result_root}"
            )
        selected.append((result_root, enc))

    if not selected:
        raise ValueError(f"No non-GloVe Podcast result directories found in {results_root}")
    return selected


def _subject_id(subject_dir: Path) -> int:
    match = SUBJECT_DIR_RE.fullmatch(subject_dir.name)
    if match is None:
        raise ValueError(f"Invalid subject directory name: {subject_dir}")
    return int(match.group(1))


def _validate_folds(result: dict, n_rows: int, path: Path) -> None:
    test_idxs = result.get("test_idxs")
    if not isinstance(test_idxs, (list, tuple)) or not test_idxs:
        raise ValueError(f"Missing non-empty test_idxs in {path}")

    normalized_tests = []
    for fold, values in enumerate(test_idxs):
        indices = np.asarray(values)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"test_idxs fold {fold} in {path} must be a 1D integer array")
        indices = indices.astype(int, copy=False)
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"test_idxs fold {fold} in {path} contains duplicate rows")
        if np.any(indices < 0) or np.any(indices >= n_rows):
            raise ValueError(
                f"test_idxs fold {fold} in {path} contains rows outside [0, {n_rows})"
            )
        normalized_tests.append(indices)

    concatenated = np.concatenate(normalized_tests)
    if len(concatenated) != n_rows or not np.array_equal(
        np.sort(concatenated), np.arange(n_rows, dtype=int)
    ):
        raise ValueError(
            f"Saved test folds in {path} must partition all {n_rows} analysis rows "
            "exactly once"
        )

    train_idxs = result.get("train_idxs")
    if train_idxs is None:
        return
    if not isinstance(train_idxs, (list, tuple)) or len(train_idxs) != len(test_idxs):
        raise ValueError(f"train_idxs in {path} must match the number of test folds")
    all_rows = np.arange(n_rows, dtype=int)
    for fold, (train_values, test_values) in enumerate(zip(train_idxs, normalized_tests)):
        train = np.asarray(train_values)
        if train.ndim != 1 or not np.issubdtype(train.dtype, np.integer):
            raise ValueError(f"train_idxs fold {fold} in {path} must be a 1D integer array")
        train = train.astype(int, copy=False)
        expected = np.setdiff1d(all_rows, test_values, assume_unique=True)
        if len(np.unique(train)) != len(train) or not np.array_equal(np.sort(train), expected):
            raise ValueError(
                f"train_idxs fold {fold} in {path} is not the exact complement of test_idxs"
            )


def _metadata_equal(left: dict, right: dict) -> bool:
    if not isinstance(left, dict) or set(left) != set(right):
        return False
    for key, expected in right.items():
        actual = left[key]
        if isinstance(expected, np.ndarray):
            if not np.array_equal(np.asarray(actual), expected):
                return False
        elif actual != expected:
            return False
    return True


def build_backfill_plans(
    results_root: Path,
    data_root_override: Path | None = None,
) -> list[BackfillPlan]:
    """Preflight every selected artifact before allowing any file to be changed."""
    plans = []
    for result_root, enc in discover_podcast_result_roots(results_root):
        if "offset" not in enc:
            raise ValueError(f"Missing encoding.offset for {result_root}")
        offset = int(enc["offset"])
        configured_data_root = Path(enc.get("data_root", "data/"))
        data_root = data_root_override or configured_data_root

        subject_dirs = sorted(
            path for path in (result_root / "subjects").iterdir() if path.is_dir()
        )
        if not subject_dirs:
            raise FileNotFoundError(f"No subject directories found in {result_root}")
        for subject_dir in subject_dirs:
            sub_id = _subject_id(subject_dir)
            path = subject_dir / "encoding_results.joblib"
            if not path.is_file():
                raise FileNotFoundError(f"Missing encoding result: {path}")
            result = load_encoding_results(str(path), location="cpu")
            if "offset" in result and int(result["offset"]) != offset:
                raise ValueError(
                    f"Offset mismatch in {path}: result has {result['offset']}, "
                    f"config has {offset}"
                )
            if "subject_id" in result and int(result["subject_id"]) != sub_id:
                raise ValueError(f"Inconsistent subject_id in {path}")

            cache = load_precomputed_epochs(sub_id, str(data_root), dataset="podcast")
            epoch_selection = np.asarray(cache["selection"], dtype=int)
            if epoch_selection.ndim != 1 or len(np.unique(epoch_selection)) != len(
                epoch_selection
            ):
                raise ValueError(
                    f"Podcast epoch selection for subject {sub_id} must contain unique 1D IDs"
                )
            metadata = build_word_selection_metadata(
                "per_subject",
                [sub_id],
                epoch_selection,
                offset,
            )
            _validate_folds(result, len(metadata["analysis_selection"]), path)

            existing = result.get("word_selection")
            if existing is not None and not _metadata_equal(existing, metadata):
                raise ValueError(f"Conflicting existing word_selection metadata in {path}")
            plans.append(
                BackfillPlan(
                    path=path,
                    result=result,
                    metadata=metadata,
                    already_complete=existing is not None,
                )
            )
    return plans


def _atomic_save(result: dict, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(result, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def apply_backfill(plans: list[BackfillPlan]) -> int:
    changed = 0
    for plan in plans:
        if plan.already_complete:
            continue
        updated = dict(plan.result)
        updated["word_selection"] = plan.metadata
        _atomic_save(updated, plan.path)
        changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the data_root saved in each significance config.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write validated metadata. Without this flag, only run the preflight.",
    )
    args = parser.parse_args()

    plans = build_backfill_plans(args.results_root, args.data_root)
    roots = {plan.path.parents[2] for plan in plans}
    pending = sum(not plan.already_complete for plan in plans)
    complete = len(plans) - pending
    print(
        f"Validated {len(plans)} subject artifacts across {len(roots)} non-GloVe "
        f"Podcast result directories ({pending} pending, {complete} already complete)."
    )
    if not args.apply:
        print("Dry run complete; no files changed. Pass --apply to write the metadata.")
        return
    changed = apply_backfill(plans)
    print(f"Backfilled word_selection metadata in {changed} artifacts.")


if __name__ == "__main__":
    main()
