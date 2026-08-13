"""Shared deep-fMRI dataset loading and alignment."""

from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
BAD_WORDS = frozenset({"sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"})
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


@dataclass
class AlignedFMRIStimuli:
    """Trimmed, TR-aligned story stimuli before z-scoring and FIR expansion."""

    stories: dict[str, np.ndarray]


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


def align_fmri_stimuli(
    fmri_repo_root: str,
    stories: list[str],
    word_embeddings: dict[str, np.ndarray],
    *,
    trim: int = 5,
) -> AlignedFMRIStimuli:
    """Lanczos-resample stories once and retain their unscaled TR representation."""
    lanczosinterp2D, _, _ = _toolbox_modules(fmri_repo_root)
    frames, sequences = load_story_word_data(fmri_repo_root, stories)
    aligned = {}
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
        aligned[story] = np.asarray(downsampled[5 + int(trim) : stop])
    return AlignedFMRIStimuli(stories=aligned)


def build_fmri_stimulus(
    aligned: AlignedFMRIStimuli,
    stories: list[str],
    *,
    ndelays: int = 4,
    basis: np.ndarray | None = None,
) -> PreparedFMRI:
    """Build a story-z-scored FIR design, optionally in a projected space."""
    stimulus_parts = []
    pre_fir_parts = []
    story_slices = {}
    cursor = 0
    for story in stories:
        values = aligned.stories[story]
        if basis is not None:
            values = np.asarray(values @ basis, dtype=np.float32)
        pre_fir = _zscore_columns(values).astype(np.float32)
        delayed = make_delayed(pre_fir, ndelays).astype(np.float32)
        pre_fir_parts.append(pre_fir)
        stimulus_parts.append(delayed)
        story_slices[story] = slice(cursor, cursor + len(delayed))
        cursor += len(delayed)
    return PreparedFMRI(
        stimulus=np.vstack(stimulus_parts),
        pre_fir_stimulus=np.vstack(pre_fir_parts),
        responses=None,
        story_slices=story_slices,
    )


def _read_response_columns(
    dataset: h5py.Dataset,
    indices: np.ndarray,
    *,
    column_batch_size: int = 1024,
    row_batch_size: int = 64,
) -> np.ndarray:
    """Read sorted response columns without materializing every voxel."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(np.diff(indices) <= 0):
        raise ValueError("Response column indices must be strictly increasing")
    result = np.empty((dataset.shape[0], len(indices)), dtype=np.float32)
    # Dense masks are much faster as a small number of bounded row reads than as
    # thousands of HDF5 fancy-index selections. Sparse significance masks retain
    # the selected-column path and avoid reading most voxel values.
    if len(indices) >= dataset.shape[1] / 2:
        for start in range(0, dataset.shape[0], int(row_batch_size)):
            stop = min(start + int(row_batch_size), dataset.shape[0])
            result[start:stop] = dataset[start:stop, :][:, indices]
        return result
    for start in range(0, len(indices), int(column_batch_size)):
        stop = min(start + int(column_batch_size), len(indices))
        result[:, start:stop] = dataset[:, indices[start:stop]]
    return result


def load_fmri_responses(
    fmri_repo_root: str,
    stories: list[str],
    story_slices: dict[str, slice],
    subjects: list[str],
    voxel_masks: dict[str, np.ndarray] | None = None,
    *,
    column_batch_size: int = 1024,
) -> np.ndarray:
    """Load selected responses once into a story-by-subject ordered matrix."""
    subject_indices = {}
    subject_slices = {}
    target_cursor = 0
    for subject in subjects:
        if voxel_masks is None:
            first_path = response_path(fmri_repo_root, subject, stories[0])
            with h5py.File(first_path, "r") as handle:
                indices = np.arange(handle["data"].shape[1], dtype=np.int64)
        else:
            indices = np.flatnonzero(
                np.asarray(voxel_masks[subject], dtype=bool)
            ).astype(np.int64, copy=False)
        subject_indices[subject] = indices
        subject_slices[subject] = slice(target_cursor, target_cursor + len(indices))
        target_cursor += len(indices)

    n_rows = max(part.stop for part in story_slices.values())
    responses = np.empty((n_rows, target_cursor), dtype=np.float32)
    for story in stories:
        row_slice = story_slices[story]
        for subject in subjects:
            path = response_path(fmri_repo_root, subject, story)
            with h5py.File(path, "r") as handle:
                dataset = handle["data"]
                if dataset.shape[0] != row_slice.stop - row_slice.start:
                    raise ValueError(
                        f"Story {story}: stimulus has "
                        f"{row_slice.stop - row_slice.start} rows but response "
                        f"has {dataset.shape[0]} rows"
                    )
                indices = subject_indices[subject]
                if len(indices) and indices[-1] >= dataset.shape[1]:
                    raise ValueError(
                        f"Subject {subject} response for {story} has "
                        f"{dataset.shape[1]} voxels, but the mask selects "
                        f"index {indices[-1]}"
                    )
                responses[row_slice, subject_slices[subject]] = _read_response_columns(
                    dataset, indices, column_batch_size=column_batch_size
                )
    return responses


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
    """Compatibility wrapper for the complete fMRI preparation contract."""
    aligned = align_fmri_stimuli(
        fmri_repo_root, stories, word_embeddings, trim=trim
    )
    prepared = build_fmri_stimulus(aligned, stories, ndelays=ndelays)
    if subjects:
        prepared.responses = load_fmri_responses(
            fmri_repo_root,
            stories,
            prepared.story_slices,
            subjects,
            voxel_masks,
        )
    return prepared


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
