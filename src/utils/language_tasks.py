"""Extensible word-aligned linguistic targets for RRR subspace decoding."""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd


TaskKind = Literal["continuous", "categorical"]


@dataclass(frozen=True)
class TaskSpec:
    """Metadata and target getter for one decoding task."""

    name: str
    display_name: str
    kind: TaskKind
    getter: Callable[["PodcastTaskContext"], np.ndarray]
    chance: float | None = None


@dataclass
class PodcastTaskContext:
    """Lazy podcast annotations aligned to a model's word-level transcript."""

    word_df: pd.DataFrame
    data_root: str = "data/"
    paths: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def _resolve_path(self, key: str, default: str) -> str:
        configured = str(self.paths.get(key, default))
        if os.path.isabs(configured) or os.path.exists(configured):
            return configured
        return os.path.join(self.data_root, configured)

    @property
    def words(self) -> pd.DataFrame:
        if "words" not in self._cache:
            path = self._resolve_path(
                "word_annotations",
                "processed_data/df_word_onset_with_pos_class.csv",
            )
            annotations = pd.read_csv(path, index_col=0).reset_index(drop=True)
            model_words = self.word_df.reset_index()
            if "word_idx" not in model_words:
                model_words["word_idx"] = np.arange(len(model_words))
            required = {"word_idx", "word", "onset", "offset"}
            missing = required - set(annotations)
            if missing:
                raise ValueError(f"Word annotations are missing columns: {sorted(missing)}")
            if len(annotations) != len(model_words):
                raise ValueError(
                    "Word annotation and embedding transcript lengths differ: "
                    f"{len(annotations)} vs {len(model_words)}"
                )
            if not np.array_equal(
                annotations["word_idx"].to_numpy(), model_words["word_idx"].to_numpy()
            ):
                raise ValueError("Word annotation indices do not match embedding word_idx")
            if not np.array_equal(
                annotations["word"].astype(str).to_numpy(),
                model_words["word"].astype(str).to_numpy(),
            ):
                raise ValueError("Word annotation text does not match embedding transcript")
            model_onsets = pd.to_numeric(model_words["start"], errors="coerce").to_numpy()
            annotation_onsets = pd.to_numeric(
                annotations["onset"], errors="coerce"
            ).to_numpy()
            onset_error = np.abs(model_onsets - annotation_onsets)
            mismatched = np.isfinite(onset_error) & (onset_error > 1e-6)
            if mismatched.mean() > 0.01 or np.nanmax(onset_error) > 0.25:
                raise ValueError(
                    "Word annotation onsets are not sufficiently aligned with the "
                    "embedding transcript"
                )
            if mismatched.any():
                warnings.warn(
                    f"Accepted {int(mismatched.sum())} minor onset differences after "
                    "exact word_idx/text alignment",
                    RuntimeWarning,
                )
            self._cache["words"] = annotations
        return self._cache["words"]

    @property
    def sentences(self) -> pd.DataFrame:
        if "sentences" not in self._cache:
            path = self._resolve_path(
                "sentence_annotations", "processed_data/all_sentences_podcast.csv"
            )
            sentences = pd.read_csv(path)
            if "sentence_onset" not in sentences:
                raise ValueError("Sentence annotations require 'sentence_onset'")
            self._cache["sentences"] = sentences
        return self._cache["sentences"]

    @property
    def iu_boundaries(self) -> pd.DataFrame:
        if "iu_boundaries" not in self._cache:
            path = self._resolve_path(
                "iu_annotations", "processed_data/iu_boundaries.csv"
            )
            boundaries = pd.read_csv(path).sort_values("time").reset_index(drop=True)
            if not {"time", "prev_word", "next_word"} <= set(boundaries):
                raise ValueError(
                    "IU annotations require time, prev_word, and next_word columns"
                )
            self._cache["iu_boundaries"] = boundaries
        return self._cache["iu_boundaries"]

    def iu_alignment(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Return IU-onset labels, normalized positions, and alignment provenance."""
        if "iu_alignment" in self._cache:
            return self._cache["iu_alignment"]

        words = self.words
        normalized = [_normalize_surface(word) for word in words["word"]]
        onsets = words["onset"].to_numpy(dtype=float)
        starts = {0}
        previous_start = 0
        rows = []
        fallback_count = 0

        for boundary_i, boundary in self.iu_boundaries.iterrows():
            previous = _normalize_surface(boundary["prev_word"])
            following = _normalize_surface(boundary["next_word"])
            candidates = []
            if previous and following:
                candidates = [
                    index + 1
                    for index in range(previous_start, len(normalized) - 1)
                    if normalized[index] == previous
                    and normalized[index + 1] == following
                ]
            if candidates:
                word_index = min(
                    candidates,
                    key=lambda index: abs(onsets[index] - float(boundary["time"])),
                )
                method = "exact_pair"
            else:
                word_index = int(
                    np.clip(
                        np.searchsorted(onsets, float(boundary["time"]), side="left"),
                        0,
                        len(onsets) - 1,
                    )
                )
                method = "time_fallback"
                fallback_count += 1
            starts.add(word_index)
            previous_start = max(previous_start, word_index)
            rows.append(
                {
                    "boundary_index": int(boundary_i),
                    "boundary_time": float(boundary["time"]),
                    "word_idx": int(words.iloc[word_index]["word_idx"]),
                    "word": str(words.iloc[word_index]["word"]),
                    "method": method,
                    "expected_prev_word": boundary["prev_word"],
                    "expected_next_word": boundary["next_word"],
                }
            )

        if fallback_count:
            warnings.warn(
                f"Used time fallback for {fallback_count} of "
                f"{len(self.iu_boundaries)} IU boundaries",
                RuntimeWarning,
            )
        ordered_starts = np.array(sorted(starts), dtype=int)
        onset_target = np.zeros(len(words), dtype=np.int64)
        onset_target[ordered_starts] = 1
        position_target = np.zeros(len(words), dtype=np.float64)
        stops = np.r_[ordered_starts[1:], len(words)]
        for start, stop in zip(ordered_starts, stops):
            length = int(stop - start)
            if length > 1:
                position_target[start:stop] = np.arange(length) / (length - 1)
        provenance = pd.DataFrame(rows)
        result = onset_target, position_target, provenance
        self._cache["iu_alignment"] = result
        return result


TASK_REGISTRY: dict[str, TaskSpec] = {}


def register_language_task(
    name: str,
    display_name: str,
    kind: TaskKind,
    chance: float | None = None,
):
    """Register a target getter; adding a task requires only a function and metadata."""

    def decorator(getter):
        if name in TASK_REGISTRY:
            raise ValueError(f"Duplicate language task: {name}")
        TASK_REGISTRY[name] = TaskSpec(name, display_name, kind, getter, chance)
        return getter

    return decorator


def get_language_task(name: str) -> TaskSpec:
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown language task {name!r}; available={sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[name]


def _normalize_surface(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"^[^\w]+|[^\w]+$", "", str(value).lower())


def _sentence_groups(context: PodcastTaskContext) -> list[np.ndarray]:
    sentence_ids = context.words["sentence_idx"].to_numpy(dtype=int)
    return [
        np.flatnonzero(sentence_ids == sentence_id)
        for sentence_id in np.unique(sentence_ids[sentence_ids >= 0])
    ]


def _iu_groups(context: PodcastTaskContext) -> list[np.ndarray]:
    starts = np.flatnonzero(context.iu_alignment()[0])
    stops = np.r_[starts[1:], len(context.words)]
    return [np.arange(start, stop) for start, stop in zip(starts, stops)]


def _unit_end_target(
    n_words: int, groups: list[np.ndarray], *, unlabeled_value: float = 0.0
) -> np.ndarray:
    target = np.full(n_words, unlabeled_value, dtype=np.float64)
    for indices in groups:
        target[indices[:-1]] = 0.0
        target[indices[-1]] = 1.0
    return target


def _unit_length_target(
    n_words: int, groups: list[np.ndarray], *, unlabeled_value: float = np.nan
) -> np.ndarray:
    target = np.full(n_words, unlabeled_value, dtype=np.float64)
    for indices in groups:
        target[indices] = len(indices)
    return target


@register_language_task("part_of_speech", "Part of speech", "categorical", 0.2)
def part_of_speech_target(context: PodcastTaskContext) -> np.ndarray:
    return context.words["pos_class"].to_numpy(dtype=np.int64)


@register_language_task("content_word", "Content vs function", "categorical", 0.5)
def content_word_target(context: PodcastTaskContext) -> np.ndarray:
    return context.words["is_content"].to_numpy(dtype=np.int64)


@register_language_task("surprisal", "Surprisal", "continuous", 0.0)
def surprisal_target(context: PodcastTaskContext) -> np.ndarray:
    return context.words["surprise"].to_numpy(dtype=np.float64)


@register_language_task("entropy", "Entropy", "continuous", 0.0)
def entropy_target(context: PodcastTaskContext) -> np.ndarray:
    return context.words["entropy"].to_numpy(dtype=np.float64)


@register_language_task("story_position", "Position in story (words)", "continuous", 0.0)
def story_position_target(context: PodcastTaskContext) -> np.ndarray:
    n_words = len(context.words)
    if n_words <= 1:
        return np.zeros(n_words, dtype=np.float64)
    return np.linspace(0.0, 1.0, n_words, dtype=np.float64)


@register_language_task(
    "story_time_position", "Position in story (time)", "continuous", 0.0
)
def story_time_position_target(context: PodcastTaskContext) -> np.ndarray:
    onsets = context.words["onset"].to_numpy(dtype=np.float64)
    if len(onsets) <= 1:
        return np.zeros(len(onsets), dtype=np.float64)
    duration = onsets[-1] - onsets[0]
    if duration <= 0:
        raise ValueError("Story time position requires the final onset after the first")
    return (onsets - onsets[0]) / duration


@register_language_task(
    "fine_part_of_speech", "Fine-grained part of speech", "categorical"
)
def fine_part_of_speech_target(context: PodcastTaskContext) -> np.ndarray:
    tags = context.words["pos"]
    if tags.isna().any():
        raise ValueError("Fine-grained part-of-speech annotations contain missing tags")
    string_tags = tags.astype(str).to_numpy()
    tag_ids = {tag: index for index, tag in enumerate(sorted(set(string_tags)))}
    return np.array([tag_ids[tag] for tag in string_tags], dtype=np.int64)


@register_language_task(
    "surprisal_class", "Surprisal class", "categorical", 1.0 / 3.0
)
def surprisal_class_target(context: PodcastTaskContext) -> np.ndarray:
    return context.words["surprise_class"].to_numpy(dtype=np.int64)


@register_language_task("sentence_onset", "Sentence onset", "categorical", 0.5)
def sentence_onset_target(context: PodcastTaskContext) -> np.ndarray:
    target = np.zeros(len(context.words), dtype=np.int64)
    word_onsets = context.words["onset"].to_numpy(dtype=float)
    for onset in context.sentences["sentence_onset"].to_numpy(dtype=float):
        matches = np.flatnonzero(np.isclose(word_onsets, onset, atol=1e-6, rtol=0))
        if len(matches) != 1:
            raise ValueError(
                f"Sentence onset {onset} matched {len(matches)} transcript words"
            )
        target[matches[0]] = 1
    return target


@register_language_task("iu_onset", "IU onset", "categorical", 0.5)
def iu_onset_target(context: PodcastTaskContext) -> np.ndarray:
    return context.iu_alignment()[0].copy()


@register_language_task("sentence_position", "Position in sentence", "continuous", 0.0)
def sentence_position_target(context: PodcastTaskContext) -> np.ndarray:
    target = np.full(len(context.words), np.nan, dtype=np.float64)
    for indices in _sentence_groups(context):
        if len(indices) == 1:
            target[indices] = 0.0
        else:
            target[indices] = np.arange(len(indices)) / (len(indices) - 1)
    return target


@register_language_task("iu_position", "Position in IU", "continuous", 0.0)
def iu_position_target(context: PodcastTaskContext) -> np.ndarray:
    return context.iu_alignment()[1].copy()


@register_language_task("sentence_end", "Sentence end", "categorical", 0.5)
def sentence_end_target(context: PodcastTaskContext) -> np.ndarray:
    return _unit_end_target(
        len(context.words), _sentence_groups(context), unlabeled_value=np.nan
    )


@register_language_task("iu_end", "IU end", "categorical", 0.5)
def iu_end_target(context: PodcastTaskContext) -> np.ndarray:
    return _unit_end_target(len(context.words), _iu_groups(context))


@register_language_task("sentence_length", "Sentence length", "continuous", 0.0)
def sentence_length_target(context: PodcastTaskContext) -> np.ndarray:
    return _unit_length_target(len(context.words), _sentence_groups(context))


@register_language_task("iu_length", "IU length", "continuous", 0.0)
def iu_length_target(context: PodcastTaskContext) -> np.ndarray:
    return _unit_length_target(len(context.words), _iu_groups(context))


@register_language_task("pre_word_pause", "Pause before word", "continuous", 0.0)
def pre_word_pause_target(context: PodcastTaskContext) -> np.ndarray:
    onsets = context.words["onset"].to_numpy(dtype=np.float64)
    offsets = context.words["offset"].to_numpy(dtype=np.float64)
    target = np.full(len(context.words), np.nan, dtype=np.float64)
    target[1:] = np.maximum(0.0, onsets[1:] - offsets[:-1])
    return target


@register_language_task("word_duration", "Word duration", "continuous", 0.0)
def word_duration_target(context: PodcastTaskContext) -> np.ndarray:
    return (
        context.words["offset"].to_numpy(dtype=float)
        - context.words["onset"].to_numpy(dtype=float)
    )


@register_language_task("word_length", "Word length", "continuous", 0.0)
def word_length_target(context: PodcastTaskContext) -> np.ndarray:
    return np.array(
        [len(_normalize_surface(word)) for word in context.words["word"]], dtype=float
    )


@register_language_task(
    "word_zipf_frequency", "Word Zipf frequency", "continuous", 0.0
)
def word_zipf_frequency_target(context: PodcastTaskContext) -> np.ndarray:
    from wordfreq import zipf_frequency

    unique_words = {_normalize_surface(word) for word in context.words["word"]}
    frequency = {
        word: zipf_frequency(word, "en", minimum=1.0) for word in unique_words
    }
    return np.array(
        [frequency[_normalize_surface(word)] for word in context.words["word"]],
        dtype=float,
    )
