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
    layer: int | None,
    pooling: str = "mean",
    *,
    fmri_repo_root: str | None = None,
    glove_data_root: str = "data",
    limit_to_glove: bool = False,
    glove_dim: int = 300,
) -> dict[str, np.ndarray]:
    """Load story embeddings and optionally zero words outside GloVe coverage.

    Zero masking preserves every word timestamp for fMRI interpolation while making
    the represented word set identical across embedding models.
    """
    frames = None
    if model_dir == "glove" or model_dir.startswith("glove-"):
        if layer is not None:
            raise ValueError("fMRI GloVe embeddings require layer=None")
        if fmri_repo_root is None:
            raise ValueError("fmri_repo_root is required for fMRI GloVe embeddings")
        dim = int(model_dir.split("-", 1)[1]) if "-" in model_dir else 300
        from src.utils.shared.embeddings import get_glove_embeddings

        frames, _ = load_story_word_data(fmri_repo_root, stories)
        result = {
            story: get_glove_embeddings(glove_data_root, frames[story], dim)[0]
            .astype(np.float32, copy=False)
            for story in stories
        }
    else:
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
                token_values = handle[f"layer-{int(layer)}"]
                pooled = []
                for _, group in transcript.groupby("word_idx", sort=True):
                    indices = group.index.to_numpy()
                    # Pool one word at a time so long stories do not require a
                    # second in-memory copy of the complete token-level layer.
                    values = token_values[indices]
                    pooled.append(reducers[pooling](values))
            result[story] = np.asarray(pooled, dtype=np.float32)

    if limit_to_glove:
        if fmri_repo_root is None:
            raise ValueError("fmri_repo_root is required when limit_to_glove=True")
        if frames is None:
            frames, _ = load_story_word_data(fmri_repo_root, stories)
        from src.utils.shared.embeddings import get_glove_embeddings

        for story in stories:
            _, coverage = get_glove_embeddings(
                glove_data_root, frames[story], int(glove_dim)
            )
            mask = coverage["in_glove"].to_numpy(dtype=bool)
            if len(result[story]) != len(mask):
                raise ValueError(
                    f"Story {story} has {len(result[story])} embeddings for "
                    f"{len(mask)} GloVe coverage rows"
                )
            result[story] = result[story].copy()
            result[story][~mask] = 0.0
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
    input_scaler: dict | None = None,
    projected_scaler: dict | None = None,
) -> PreparedFMRI:
    """Build an FIR design, optionally in explicitly standardized coordinates.

    With no scaler, the compatibility behavior z-scores each story independently.
    RRR and decoding callers pass a scaler fitted on source training stories so
    that projection occurs in the factor's saved coordinate system.  An optional
    projected scaler is applied before FIR expansion.
    """
    stimulus_parts = []
    pre_fir_parts = []
    story_slices = {}
    cursor = 0
    for story in stories:
        values = np.asarray(aligned.stories[story], dtype=np.float32)
        values = (
            _zscore_columns(values).astype(np.float32)
            if input_scaler is None
            else apply_embedding_scaler(values, input_scaler)
        )
        if basis is not None:
            values = np.asarray(values @ basis, dtype=np.float32)
        if projected_scaler is not None:
            values = apply_embedding_scaler(values, projected_scaler)
        pre_fir = values
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


def fit_pooled_embedding_scaler(
    aligned: AlignedFMRIStimuli,
    stories: list[str],
    *,
    input_scaler: dict | None = None,
    basis: np.ndarray | None = None,
) -> dict:
    """Fit population moments over every row of the selected stories."""
    if not stories:
        raise ValueError("Cannot fit an embedding scaler without stories")

    def transformed_story(story):
        values = np.asarray(aligned.stories[story], dtype=np.float64)
        if input_scaler is not None:
            values = apply_embedding_scaler(values, input_scaler).astype(np.float64)
        if basis is not None:
            values = values @ np.asarray(basis, dtype=np.float64)
        return values

    # Use two passes instead of E[X^2] - E[X]^2. The latter can lose precision
    # when a feature's mean is large relative to its variance.
    count = 0
    total = None
    for story in stories:
        values = transformed_story(story)
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(axis=0, dtype=np.float64)
        count += len(values)
    if count == 0:
        raise ValueError("Cannot fit an embedding scaler on zero rows")
    mean = total / count
    squared_deviation_sum = np.zeros_like(mean)
    for story in stories:
        deviations = transformed_story(story) - mean
        squared_deviation_sum += np.square(deviations).sum(
            axis=0, dtype=np.float64
        )
    variance = squared_deviation_sum / count
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "coordinates": "standardized-v1",
    }


def apply_embedding_scaler(values: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply a persisted embedding scaler and validate its coordinate contract."""
    coordinates = scaler.get("coordinates", "standardized-v1")
    if coordinates != "standardized-v1":
        raise ValueError(f"Unsupported embedding scaler coordinates: {coordinates!r}")
    mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, -1)
    scale = np.asarray(scaler["scale"], dtype=np.float32).reshape(1, -1)
    values = np.asarray(values, dtype=np.float32)
    if mean.shape[1] != values.shape[1] or scale.shape != mean.shape:
        raise ValueError(
            "Embedding scaler width does not match values: "
            f"mean={mean.shape}, scale={scale.shape}, values={values.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Embedding scaler contains invalid mean or scale values")
    return ((values - mean) / scale).astype(np.float32)


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
