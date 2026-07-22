"""Shared deep-fMRI-dataset story alignment, preprocessing, and statistics."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


BAD_WORDS = frozenset({"sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"})

# TEMPORARY DATA HOLDOUT: these response files are absent for at least one configured
# subject. Keep this explicit and centralized so the stories can be restored as soon
# as the missing data are recovered; do not silently add further stories here.
TEMPORARILY_HELD_OUT_FMRI_STORIES = frozenset(
    {"life", "fromboyhoodtofatherhood"}
)


@dataclass
class PreparedFMRI:
    """Concatenated story matrices plus row boundaries."""

    stimulus: np.ndarray
    pre_fir_stimulus: np.ndarray
    responses: np.ndarray | None
    story_slices: dict[str, slice]


def _toolbox_modules(fmri_repo_root: str):
    encoding_dir = str((Path(fmri_repo_root) / "encoding").resolve())
    if encoding_dir not in sys.path:
        sys.path.insert(0, encoding_dir)
    from ridge_utils.interpdata import lanczosinterp2D
    from ridge_utils.stimulus_utils import load_simulated_trfiles
    from ridge_utils.textgrid import TextGrid

    return lanczosinterp2D, load_simulated_trfiles, TextGrid


def resolve_fmri_stories(
    fmri_repo_root: str, sessions: list[int]
) -> tuple[list[str], list[str], dict[int, list[str]]]:
    """Resolve unique training stories, shared tests, and session groups."""
    session_path = Path(fmri_repo_root) / "em_data" / "sess_to_story.json"
    with session_path.open() as handle:
        mapping = json.load(handle)
    groups: dict[int, list[str]] = {}
    train: list[str] = []
    tests: list[str] = []
    for session in sessions:
        key = str(int(session))
        if key not in mapping:
            raise ValueError(f"Session {session} is absent from {session_path}")
        stories, test_story = mapping[key]
        retained_stories = [
            story for story in stories
            if story not in TEMPORARILY_HELD_OUT_FMRI_STORIES
        ]
        if not retained_stories:
            raise ValueError(
                f"Session {session} has no training stories after temporary holdouts"
            )
        groups[int(session)] = retained_stories
        train.extend(story for story in retained_stories if story not in train)
        if test_story not in tests:
            tests.append(test_story)
    overlap = set(train) & set(tests)
    if overlap:
        raise ValueError(f"Training and test stories overlap: {sorted(overlap)}")
    return train, tests, groups


def load_story_word_data(
    fmri_repo_root: str, stories: list[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Load the exact filtered word tier and upstream-compatible TR sequences."""
    _, load_simulated_trfiles, TextGrid = _toolbox_modules(fmri_repo_root)
    derivative = Path(fmri_repo_root) / "data" / "ds003020" / "derivatives"
    with (derivative / "respdict.json").open() as handle:
        respdict = json.load(handle)
    trfiles = load_simulated_trfiles(respdict)
    frames: dict[str, pd.DataFrame] = {}
    sequences: dict[str, object] = {}
    for story in stories:
        grid = TextGrid((derivative / "TextGrids" / f"{story}.TextGrid").read_text())
        transcript = grid.tiers[1].make_simple_transcript()
        retained = [
            (float(start), float(end), str(word).lower())
            for start, end, word in transcript
            if str(word).lower().strip("{}").strip() not in BAD_WORDS
        ]
        if not retained:
            raise ValueError(f"No retained words found for story {story}")
        starts, ends, words = zip(*retained)
        trfile = trfiles[story][0]
        tr = trfile.avgtr
        tr_times = trfile.get_reltriggertimes() + tr / 2.0
        avg_times = (np.asarray(starts) + np.asarray(ends)) / 2.0
        frames[story] = pd.DataFrame(
            {
                "word_idx": np.arange(len(words), dtype=int),
                "word": words,
                "start": starts,
                "end": ends,
            }
        )
        sequences[story] = type(
            "StorySequence",
            (),
            {"data": list(words), "data_times": avg_times, "tr_times": tr_times},
        )()
    return frames, sequences


def load_story_embeddings(
    embedding_root: str,
    model_dir: str,
    stories: list[str],
    layer: int,
    pooling: str = "mean",
) -> dict[str, np.ndarray]:
    """Load token HDF5 files and pool them into one vector per fMRI word."""
    reducers = {
        "mean": lambda values: values.mean(axis=0),
        "first": lambda values: values[0],
        "last": lambda values: values[-1],
    }
    if pooling not in reducers:
        raise ValueError(f"Unknown pooling strategy: {pooling}")
    result = {}
    for story in stories:
        story_dir = Path(embedding_root) / model_dir / story
        transcript = pd.read_csv(story_dir / "transcript.tsv", sep="\t")
        transcript = transcript.loc[:, ~transcript.columns.astype(str).str.startswith("Unnamed:")]
        with h5py.File(story_dir / "features.hdf5", "r") as handle:
            token_values = handle[f"layer-{int(layer)}"][...]
        pooled = []
        for _, group in transcript.groupby("word_idx", sort=True):
            pooled.append(reducers[pooling](token_values[group.index.to_numpy()]))
        result[story] = np.asarray(pooled, dtype=np.float32)
    return result


def _zscore_columns(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return centered / scale


def make_delayed(stimulus: np.ndarray, ndelays: int) -> np.ndarray:
    """Concatenate zero-padded delays 1..ndelays, matching encoding.py."""
    delayed = []
    for delay in range(1, int(ndelays) + 1):
        block = np.zeros_like(stimulus)
        block[delay:] = stimulus[:-delay]
        delayed.append(block)
    return np.hstack(delayed)


def response_path(fmri_repo_root: str, subject: str, story: str) -> Path:
    return (
        Path(fmri_repo_root)
        / "data"
        / "ds003020"
        / "derivatives"
        / "preprocessed_data"
        / subject
        / f"{story}.hf5"
    )


def finite_response_voxel_masks(
    fmri_repo_root: str,
    subjects: list[str],
    stories: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Keep voxels with finite responses at every TR in every selected story."""
    masks = {}
    excluded_by_story = {}
    for subject in subjects:
        mask = None
        story_counts = {}
        expected_voxels = None
        for story in stories:
            path = response_path(fmri_repo_root, subject, story)
            with h5py.File(path, "r") as handle:
                data = handle["data"]
                if expected_voxels is None:
                    expected_voxels = int(data.shape[1])
                    mask = np.ones(expected_voxels, dtype=bool)
                elif data.shape[1] != expected_voxels:
                    raise ValueError(
                        f"Subject {subject} has {data.shape[1]} voxels in {story}, "
                        f"expected {expected_voxels}"
                    )
                story_mask = np.ones(expected_voxels, dtype=bool)
                for start in range(0, data.shape[0], 32):
                    story_mask &= np.isfinite(data[start : start + 32]).all(axis=0)
            story_counts[story] = int((~story_mask).sum())
            mask &= story_mask
        masks[subject] = mask
        excluded_by_story[subject] = story_counts
    return masks, excluded_by_story


def prepare_fmri_data(
    fmri_repo_root: str,
    stories: list[str],
    word_embeddings: dict[str, np.ndarray],
    *,
    trim: int = 5,
    ndelays: int = 4,
    subjects: list[str] | None = None,
    voxel_masks: dict[str, np.ndarray] | None = None,
) -> PreparedFMRI:
    """Lanczos-resample and apply the exact upstream trim/z-score/FIR contract."""
    lanczosinterp2D, _, _ = _toolbox_modules(fmri_repo_root)
    frames, sequences = load_story_word_data(fmri_repo_root, stories)
    stimulus_parts = []
    pre_fir_parts = []
    response_parts = []
    story_slices = {}
    cursor = 0
    for story in stories:
        vectors = np.asarray(word_embeddings[story])
        if vectors.shape[0] != len(frames[story]):
            raise ValueError(
                f"Story {story} has {vectors.shape[0]} embeddings for "
                f"{len(frames[story])} transcript words"
            )
        sequence = sequences[story]
        downsampled = lanczosinterp2D(
            vectors, sequence.data_times, sequence.tr_times, window=3
        )
        stop = -int(trim) if trim else None
        trimmed = downsampled[5 + int(trim) : stop]
        pre_fir = _zscore_columns(trimmed).astype(np.float32)
        delayed = make_delayed(pre_fir, ndelays).astype(np.float32)
        pre_fir_parts.append(pre_fir)
        stimulus_parts.append(delayed)
        story_slices[story] = slice(cursor, cursor + len(delayed))
        cursor += len(delayed)

        if subjects:
            subject_parts = []
            for subject in subjects:
                with h5py.File(response_path(fmri_repo_root, subject, story), "r") as handle:
                    response = handle["data"][...]
                if voxel_masks is not None:
                    response = response[:, np.asarray(voxel_masks[subject], dtype=bool)]
                subject_parts.append(np.asarray(response, dtype=np.float32))
            combined = np.hstack(subject_parts)
            if combined.shape[0] != delayed.shape[0]:
                raise ValueError(
                    f"Story {story}: stimulus has {delayed.shape[0]} rows but "
                    f"response has {combined.shape[0]} rows"
                )
            response_parts.append(combined)
    return PreparedFMRI(
        stimulus=np.vstack(stimulus_parts),
        pre_fir_stimulus=np.vstack(pre_fir_parts),
        responses=np.vstack(response_parts) if response_parts else None,
        story_slices=story_slices,
    )


def session_cv_splits(
    story_slices: dict[str, slice], session_groups: dict[int, list[str]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create row-index folds that hold out complete sessions."""
    n_rows = max(part.stop for part in story_slices.values())
    all_rows = np.arange(n_rows)
    folds = []
    for stories in session_groups.values():
        test = np.concatenate(
            [np.arange(story_slices[story].start, story_slices[story].stop) for story in stories]
        )
        folds.append((np.setdiff1d(all_rows, test, assume_unique=True), test))
    return folds


def remap_inner_splits(
    outer_train: np.ndarray,
    story_slices: dict[str, slice],
    session_groups: dict[int, list[str]],
    held_out_session: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return session-grouped inner folds indexed relative to an outer train matrix."""
    location = {int(row): index for index, row in enumerate(outer_train)}
    relative_all = np.arange(len(outer_train))
    splits = []
    for session, stories in session_groups.items():
        if session == held_out_session:
            continue
        absolute = np.concatenate(
            [np.arange(story_slices[story].start, story_slices[story].stop) for story in stories]
        )
        validation = np.asarray([location[int(row)] for row in absolute], dtype=int)
        splits.append((np.setdiff1d(relative_all, validation, assume_unique=True), validation))
    return splits


def columnwise_correlation(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    true_z = _zscore_columns(np.asarray(true, dtype=np.float32))
    pred_z = _zscore_columns(np.asarray(predicted, dtype=np.float32))
    return np.nan_to_num(np.mean(true_z * pred_z, axis=0))


def circular_shift_pvalues(
    true: np.ndarray,
    predicted: np.ndarray,
    story_slices: dict[str, slice],
    *,
    num_shifts: int,
    minimal_shift: int,
    seed: int,
    batch_size: int = 1024,
) -> tuple[np.ndarray, dict]:
    """One-sided empirical p-values using FFT cross-correlations in target batches."""
    true_z = _zscore_columns(np.asarray(true, dtype=np.float64))
    pred_z = _zscore_columns(np.asarray(predicted, dtype=np.float64))
    observed = np.nan_to_num(np.mean(true_z * pred_z, axis=0))
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(observed.shape, dtype=np.int64)
    story_items = list(story_slices.items())
    sampled_shifts = np.empty((num_shifts, len(story_items)), dtype=np.int32)
    for story_index, (story, part) in enumerate(story_items):
        length = part.stop - part.start
        allowed = np.arange(minimal_shift, length - minimal_shift + 1)
        if not len(allowed):
            raise ValueError(
                f"Story {story} has {length} rows, too short for minimal_shift={minimal_shift}"
            )
        sampled_shifts[:, story_index] = rng.choice(
            allowed, size=num_shifts, replace=True
        )

    n_rows, n_targets = true_z.shape
    for target_start in range(0, n_targets, batch_size):
        target_stop = min(target_start + batch_size, n_targets)
        null_dot = np.zeros((num_shifts, target_stop - target_start), dtype=np.float64)
        for story_index, (_, part) in enumerate(story_items):
            story_true = true_z[part, target_start:target_stop]
            story_pred = pred_z[part, target_start:target_stop]
            cross = np.fft.irfft(
                np.conj(np.fft.rfft(story_true, axis=0))
                * np.fft.rfft(story_pred, axis=0),
                n=len(story_true),
                axis=0,
            )
            lookup = (-sampled_shifts[:, story_index]) % len(story_true)
            null_dot += cross[lookup]
        null_corr = null_dot / n_rows
        exceedances[target_start:target_stop] = np.sum(
            null_corr >= observed[target_start:target_stop], axis=0
        )
    pvalues = (exceedances + 1.0) / (num_shifts + 1.0)
    shifts_used = [
        {
            story: int(sampled_shifts[draw, story_index])
            for story_index, (story, _) in enumerate(story_items)
        }
        for draw in range(min(10, num_shifts))
    ]
    return pvalues, {"observed": observed, "representative_shifts": shifts_used}


def bh_fdr(pvalues: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    rejected, adjusted, _, _ = multipletests(pvalues, alpha=q, method="fdr_bh")
    return rejected.astype(bool), adjusted


def target_manifest(subjects: list[str], masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        for voxel in np.flatnonzero(np.asarray(masks[subject], dtype=bool)):
            rows.append({"subject": subject, "voxel_index": int(voxel)})
    return pd.DataFrame(rows)


def resolve_pca_sibling(source_rrr_unit_dir: str) -> str:
    path = os.path.normpath(source_rrr_unit_dir)
    parts = path.split(os.sep)
    if parts[-1] == "rrr":
        parts[-1] = "pca"
        return os.sep.join(parts)
    replacements = {
        "cross_dataset_rrr": "cross_dataset_pca",
        "per_subject_rrr": "per_subject_pca",
        "per_region_rrr": "per_region_pca",
        "multi_view_rrr": "multi_view_pca",
    }
    for index, part in enumerate(parts):
        if part in replacements:
            parts[index] = replacements[part]
            return os.sep.join(parts)
    return path + "_pca"
