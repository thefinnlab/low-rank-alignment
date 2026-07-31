"""Built-in providers for :mod:`src.utils.language_features`."""

from __future__ import annotations

import re
import wave
from collections import defaultdict
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.language_features import (
    AnnotationResult,
    register_annotation_provider,
)


def _iter_diarization_turns(annotation):
    """Yield ``(start, end, label)`` across supported pyannote APIs."""
    if hasattr(annotation, "itertracks"):
        for turn, _, label in annotation.itertracks(yield_label=True):
            yield float(turn.start), float(turn.end), str(label)
        return
    for item in annotation:
        if len(item) == 2:
            turn, label = item
        elif len(item) == 3:
            turn, _, label = item
        else:
            raise ValueError(f"Unexpected diarization item: {item!r}")
        yield float(turn.start), float(turn.end), str(label)


def _canonicalize_speaker_turns(annotation) -> pd.DataFrame:
    """Sort turns and name speaker clusters by their first appearance."""
    turns = sorted(
        _iter_diarization_turns(annotation),
        key=lambda row: (row[0], row[1], row[2]),
    )
    speaker_names: dict[str, str] = {}
    rows = []
    for start, end, raw_label in turns:
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(
                f"Speaker diarization returned invalid turn [{start}, {end}]"
            )
        label = speaker_names.setdefault(
            raw_label, f"SPEAKER_{len(speaker_names):02d}"
        )
        rows.append(
            {
                "segment_id": f"speaker_turn_{len(rows)}",
                "start": start,
                "end": end,
                "label": label,
                "segment_type": "speaker_turn",
            }
        )
    if not rows:
        raise ValueError("Speaker diarization returned no turns")
    return pd.DataFrame(rows)


def _assign_speakers_to_words(
    token_table: pd.DataFrame, turns: pd.DataFrame
) -> pd.Series:
    """Assign the greatest-overlap exclusive speaker turn to each timed word."""
    turn_starts = turns["start"].to_numpy(dtype=float)
    turn_ends = turns["end"].to_numpy(dtype=float)
    turn_labels = turns["label"].astype(str).to_numpy()
    labels = pd.Series(pd.NA, index=token_table.index, dtype="string")
    for index, token in token_table.iterrows():
        start = pd.to_numeric(pd.Series([token.get("start")]), errors="coerce").iloc[0]
        end = pd.to_numeric(pd.Series([token.get("end")]), errors="coerce").iloc[0]
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        overlaps = np.minimum(float(end), turn_ends) - np.maximum(float(start), turn_starts)
        candidates = np.flatnonzero(overlaps > 0)
        if len(candidates):
            # Turns are time-sorted, so np.argmax provides the specified earliest-onset
            # tie break when two turns have exactly the same overlap.
            winner = candidates[int(np.argmax(overlaps[candidates]))]
            labels.loc[index] = turn_labels[winner]
    return labels


def _load_pcm_wav(path: str, torch) -> dict:
    """Load a PCM WAV in memory so pyannote does not require FFmpeg/TorchCodec."""
    with wave.open(path, "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        frame_count = stream.getnframes()
        dtype_and_scale = {
            1: (np.uint8, 128.0),
            2: (np.dtype("<i2"), 32768.0),
            4: (np.dtype("<i4"), 2147483648.0),
        }
        if sample_width not in dtype_and_scale:
            raise ValueError(
                f"Unsupported {sample_width * 8}-bit WAV for speaker diarization"
            )
        dtype, scale = dtype_and_scale[sample_width]
        # Downmix in chunks so the 30-minute stereo Podcast WAV does not require
        # several full-size intermediate arrays.
        mono = np.empty(frame_count, dtype=np.float32)
        position = 0
        while position < frame_count:
            requested = min(1_000_000, frame_count - position)
            raw = stream.readframes(requested)
            samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if len(samples) != requested * channels:
                raise ValueError(f"Malformed PCM WAV at {path}")
            if sample_width == 1:
                samples -= 128.0
            block = samples.reshape(-1, channels).mean(axis=1) / scale
            mono[position : position + requested] = block
            position += requested
    waveform = mono[None, :]
    return {
        "waveform": torch.from_numpy(np.ascontiguousarray(waveform)),
        "sample_rate": sample_rate,
    }


@register_annotation_provider("speaker_diarization", version="1")
def speaker_diarization_provider(
    document, token_table, options, resources
) -> AnnotationResult:
    """Run local pyannote diarization and align anonymous speakers to words."""
    audio_path = document.media.get("audio")
    if not audio_path or not Path(audio_path).exists():
        raise FileNotFoundError(
            "The speaker_diarization provider requires document.media['audio']"
        )
    resource_name = options.get("model_resource", "pyannote_community_1")
    if resource_name not in resources:
        raise ValueError(
            f"Unknown speaker diarization model resource {resource_name!r}"
        )
    model_path = Path(resources[resource_name]["path"])
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError(
            "The speaker_diarization provider requires 'pyannote.audio'"
        ) from exc

    device = str(options.get("device", "auto"))
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("speaker_diarization device must be auto, cpu, or cuda")
    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Speaker diarization requested CUDA, but CUDA is unavailable")

    count_options = {
        key: options.get(key)
        for key in ("num_speakers", "min_speakers", "max_speakers")
        if options.get(key) is not None
    }
    if "num_speakers" in count_options and len(count_options) > 1:
        raise ValueError("num_speakers cannot be combined with min_speakers/max_speakers")
    for key, value in count_options.items():
        if int(value) < 1:
            raise ValueError(f"{key} must be positive")
        count_options[key] = int(value)
    if (
        "min_speakers" in count_options
        and "max_speakers" in count_options
        and count_options["min_speakers"] > count_options["max_speakers"]
    ):
        raise ValueError("min_speakers cannot exceed max_speakers")

    pipeline = Pipeline.from_pretrained(str(model_path))
    effective_device = "cuda" if use_cuda else "cpu"
    pipeline.to(torch.device(effective_device))
    audio = _load_pcm_wav(str(audio_path), torch)
    diarization = pipeline(audio, **count_options)
    exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
    if exclusive is None:
        raise ValueError(
            "The configured pyannote pipeline did not return exclusive diarization"
        )
    turns = _canonicalize_speaker_turns(exclusive)
    speaker_identity = _assign_speakers_to_words(token_table, turns)
    word_counts = speaker_identity.value_counts().sort_index()
    durations = (turns["end"] - turns["start"]).groupby(turns["label"]).sum()
    output = pd.DataFrame(
        {"token_id": token_table["token_id"], "speaker_identity": speaker_identity}
    )
    return AnnotationResult(
        output,
        segments=turns,
        metadata={
            "model_resource": resource_name,
            "device": effective_device,
            "speaker_count_options": count_options,
            "detected_speaker_count": int(turns["label"].nunique()),
            "labeled_word_count": int(speaker_identity.notna().sum()),
            "unmatched_word_count": int(speaker_identity.isna().sum()),
            "labeled_word_fraction": float(speaker_identity.notna().mean()),
            "speaker_word_counts": {
                str(label): int(count) for label, count in word_counts.items()
            },
            "speaker_durations_seconds": {
                str(label): float(duration) for label, duration in durations.items()
            },
        },
    )


LANCASTER_COLUMNS = (
    "visual_strength",
    "auditory_strength",
    "haptic_strength",
    "olfactory_strength",
    "gustatory_strength",
    "interoceptive_strength",
    "hand_arm_action_strength",
    "mouth_throat_action_strength",
    "foot_leg_action_strength",
    "head_action_strength",
    "torso_action_strength",
)
CONTENT_POS = {"ADJ", "ADV", "NOUN", "PROPN", "VERB"}
FILLED_PAUSES = {"uh", "um", "er", "erm", "hm", "hmm"}
MULTIWORD_DISCOURSE_MARKERS = {("you", "know"), ("i", "mean")}
CONTEXTUAL_DISCOURSE_MARKERS = {"well", "so", "anyway"}


def _surface(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"^[^\w]+|[^\w]+$", "", str(value)).lower()


def _empty_features(token_table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"token_id": token_table["token_id"].to_numpy()})


def _encode_gpt2_words(tokenizer, words: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode space-delimited transcript words and retain their word ownership."""
    token_ids: list[int] = []
    token_words: list[int] = []
    for word_index, word in enumerate(words):
        if not word:
            continue
        pieces = tokenizer.encode(f" {word}", add_special_tokens=False)
        if not pieces:
            continue
        token_ids.extend(int(piece) for piece in pieces)
        token_words.extend([word_index] * len(pieces))
    return np.asarray(token_ids, dtype=np.int64), np.asarray(token_words, dtype=np.int64)


def _aggregate_token_surprisal(
    token_surprisal: np.ndarray, token_words: np.ndarray, n_words: int
) -> np.ndarray:
    """Sum BPE-token surprisal within each transcript word."""
    result = np.full(n_words, np.nan, dtype=float)
    for word_index in np.unique(token_words):
        values = token_surprisal[token_words == word_index]
        if np.isfinite(values).all():
            result[int(word_index)] = float(values.sum())
    return result


@register_annotation_provider("gpt2_surprisal", version="1")
def gpt2_surprisal_provider(
    document, token_table, options, resources
) -> AnnotationResult:
    """Compute word surprisal from GPT-2 small using preceding story context."""
    del document
    resource_name = str(options.get("model_resource", "gpt2_small"))
    if resource_name not in resources:
        raise ValueError(f"Unknown GPT-2 model resource {resource_name!r}")
    model_path = Path(resources[resource_name]["path"])
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "The gpt2_surprisal provider requires torch and transformers"
        ) from exc

    requested_device = str(options.get("device", "auto"))
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("gpt2_surprisal device must be auto, cpu, or cuda")
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and torch.cuda.is_available()
    )
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPT-2 surprisal requested CUDA, but CUDA is unavailable")
    device = "cuda" if use_cuda else "cpu"
    batch_size = int(options.get("batch_size", 16))
    context_tokens = int(options.get("context_tokens", 1024))
    if batch_size < 1:
        raise ValueError("gpt2_surprisal batch_size must be positive")
    if context_tokens != 1024:
        raise ValueError("Standardized GPT-2 surprisal requires context_tokens=1024")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True
    ).eval().to(device)
    words = token_table["word"].fillna("").astype(str).tolist()
    token_ids, token_words = _encode_gpt2_words(tokenizer, words)
    token_surprisal = np.full(len(token_ids), np.nan, dtype=float)
    # Token zero has no preceding linguistic context. Every later target is scored
    # from up to the previous 1,024 BPE tokens, with no cross-story state.
    batch_starts = range(1, len(token_ids), batch_size)
    for batch_start in tqdm(
        batch_starts,
        desc=f"GPT-2 surprisal ({len(token_table)} words)",
        unit="batch",
    ):
        targets = np.arange(
            batch_start, min(batch_start + batch_size, len(token_ids)), dtype=int
        )
        lengths = np.minimum(targets, context_tokens)
        width = int(lengths.max())
        input_ids = torch.full(
            (len(targets), width),
            int(tokenizer.eos_token_id),
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(targets), width), dtype=torch.long)
        position_ids = torch.zeros((len(targets), width), dtype=torch.long)
        for row, (target, length) in enumerate(zip(targets, lengths)):
            context = torch.from_numpy(token_ids[target - length : target])
            input_ids[row, width - length :] = context
            attention_mask[row, width - length :] = 1
            position_ids[row, width - length :] = torch.arange(length)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
        with torch.inference_mode():
            logits = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits[:, -1, :].float()
            target_ids = torch.from_numpy(token_ids[targets]).to(device)
            nll_nats = torch.nn.functional.cross_entropy(
                logits, target_ids, reduction="none"
            )
        token_surprisal[targets] = nll_nats.cpu().numpy() / log(2.0)
    word_surprisal = _aggregate_token_surprisal(
        token_surprisal, token_words, len(token_table)
    )
    del model
    if use_cuda:
        torch.cuda.empty_cache()
    return AnnotationResult(
        pd.DataFrame(
            {
                "token_id": token_table["token_id"].to_numpy(),
                "surprisal": word_surprisal,
            }
        ),
        metadata={
            "model": "gpt2",
            "model_resource": resource_name,
            "device": device,
            "context_reset": "stimulus",
            "context_tokens": context_tokens,
            "unit": "bits",
            "word_aggregation": "sum_bpe_tokens",
            "first_token": "missing",
        },
    )


@register_annotation_provider("disfluency", version="1", requires=("ud",))
def disfluency_provider(document, token_table, options, resources) -> AnnotationResult:
    """Generate conservative transcript-based disfluency and recurrence labels."""
    del document, resources
    normalized = token_table["word"].map(_surface).tolist()
    n_tokens = len(token_table)
    output = _empty_features(token_table)
    output["filled_pause"] = [int(word in FILLED_PAUSES) for word in normalized]

    immediate = np.zeros(n_tokens, dtype=int)
    seen_before = np.zeros(n_tokens, dtype=int)
    prior_count = np.zeros(n_tokens, dtype=float)
    previous_distance = np.full(n_tokens, np.nan)
    in_previous_five = np.zeros(n_tokens, dtype=int)
    occurrences: dict[str, list[int]] = defaultdict(list)
    for index, word in enumerate(normalized):
        if not word:
            continue
        prior = occurrences[word]
        if prior:
            seen_before[index] = 1
            prior_count[index] = len(prior)
            previous_distance[index] = index - prior[-1]
            immediate[index] = int(prior[-1] == index - 1)
            in_previous_five[index] = int(prior[-1] >= index - 5)
        occurrences[word].append(index)
    output["immediate_word_repetition"] = immediate
    output["word_seen_before"] = seen_before
    output["prior_word_count"] = prior_count
    output["distance_from_previous_occurrence"] = previous_distance
    output["word_in_previous_5"] = in_previous_five

    repeated_phrase = np.zeros(n_tokens, dtype=int)
    marker = np.zeros(n_tokens, dtype=int)
    marker_type = np.full(n_tokens, pd.NA, dtype=object)
    spans = []
    span_number = 0
    max_phrase_words = int(options.get("max_repeated_phrase_words", 5))
    for _, group in token_table.groupby("sentence_id", sort=False):
        indices = group.index.to_numpy(dtype=int)
        sentence_words = [normalized[index] for index in indices]
        position = 0
        while position < len(indices):
            best = 0
            for width in range(2, min(max_phrase_words, position, len(indices) - position) + 1):
                if (
                    all(sentence_words[position : position + width])
                    and sentence_words[position - width : position]
                    == sentence_words[position : position + width]
                ):
                    best = width
            if best:
                selected = indices[position : position + best]
                repeated_phrase[selected] = 1
                spans.append(
                    {
                        "span_id": f"repeated_phrase_{span_number}",
                        "start_token_id": int(selected[0]),
                        "end_token_id": int(selected[-1]),
                        "label": "repeated_phrase",
                        "span_type": "disfluency",
                    }
                )
                span_number += 1
                position += best
            else:
                position += 1

        for local, index in enumerate(indices):
            discourse_label = None
            for phrase in MULTIWORD_DISCOURSE_MARKERS:
                if tuple(sentence_words[local : local + len(phrase)]) == phrase:
                    discourse_label = "_".join(phrase)
                    selected = indices[local : local + len(phrase)]
                    marker[selected] = 1
                    marker_type[selected] = discourse_label
                    spans.append(
                        {
                            "span_id": f"discourse_marker_{span_number}",
                            "start_token_id": int(selected[0]),
                            "end_token_id": int(selected[-1]),
                            "label": discourse_label,
                            "span_type": "discourse_marker",
                        }
                    )
                    span_number += 1
                    break
            word = sentence_words[local]
            is_interjection = str(token_table.loc[index].get("part_of_speech", "")) == "INTJ"
            if discourse_label is None and word in CONTEXTUAL_DISCOURSE_MARKERS and (
                local == 0 or is_interjection
            ):
                marker[index] = 1
                marker_type[index] = word
                spans.append(
                    {
                        "span_id": f"discourse_marker_{span_number}",
                        "start_token_id": int(index),
                        "end_token_id": int(index),
                        "label": word,
                        "span_type": "discourse_marker",
                    }
                )
                span_number += 1
    output["repeated_phrase"] = repeated_phrase
    output["discourse_marker"] = marker
    output["discourse_marker_type"] = marker_type
    return AnnotationResult(
        output,
        spans=pd.DataFrame(spans),
        metadata={
            "normalization": "lowercase_strip_surrounding_punctuation",
            "filled_pause_inventory": sorted(FILLED_PAUSES),
            "max_repeated_phrase_words": max_phrase_words,
        },
    )


@register_annotation_provider("native", version="1")
def native_provider(document, token_table, options, resources) -> AnnotationResult:
    """Normalize controls and retain source annotations under native names."""
    del options, resources
    native = document.native_tokens.reset_index(drop=True)
    output = _empty_features(token_table)
    mapping = {
        "lemma": "native_lemma",
        "pos": "native_part_of_speech",
        "deprel": "native_dependency_relation",
        "head": "native_dependency_head",
        "gpt2_surprisal": "native_surprisal",
        "surprise": "native_surprisal",
        "entropy": "native_entropy",
        "speaker": "native_speaker",
        "pitch": "native_pitch",
        "rms": "native_intensity_rms",
        "syllable": "native_syllable_count",
        "phoneme_num": "native_phoneme_count",
    }
    for source, target in mapping.items():
        if source in native and target not in output:
            output[target] = native[source].replace("", pd.NA).to_numpy()
    if "native_entropy" in output:
        output["entropy"] = output["native_entropy"]

    words = token_table["word"].astype(str)
    starts = pd.to_numeric(token_table["start"], errors="coerce").to_numpy(float)
    ends = pd.to_numeric(token_table["end"], errors="coerce").to_numpy(float)
    output["word_length"] = np.array([len(_surface(word)) for word in words], float)
    from wordfreq import zipf_frequency

    unique_words = {_surface(word) for word in words}
    frequencies = {
        word: zipf_frequency(word, "en", minimum=1.0) for word in unique_words
    }
    output["word_zipf_frequency"] = [frequencies[_surface(word)] for word in words]
    output["word_duration"] = ends - starts
    pre_pause = np.full(len(output), np.nan)
    post_pause = np.full(len(output), np.nan)
    if len(output) > 1:
        gaps = np.maximum(0.0, starts[1:] - ends[:-1])
        pre_pause[1:] = gaps
        post_pause[:-1] = gaps
    output["pre_word_pause"] = pre_pause
    output["post_word_pause"] = post_pause
    sentence_position = np.full(len(output), np.nan)
    sentence_onset = np.zeros(len(output), dtype=int)
    sentence_end = np.zeros(len(output), dtype=int)
    sentence_length = np.full(len(output), np.nan)
    for _, group in token_table.groupby("sentence_id", sort=False):
        indices = group.index.to_numpy()
        sentence_position[indices] = (
            0.0 if len(indices) == 1 else np.arange(len(indices)) / (len(indices) - 1)
        )
        sentence_onset[indices[0]] = 1
        sentence_end[indices[-1]] = 1
        sentence_length[indices] = len(indices)
    output["sentence_position"] = sentence_position
    output["sentence_onset"] = sentence_onset
    output["sentence_end"] = sentence_end
    output["sentence_length"] = sentence_length
    if "iu_id" in native:
        iu_position = np.full(len(output), np.nan)
        iu_onset = np.zeros(len(output), dtype=int)
        iu_end = np.zeros(len(output), dtype=int)
        iu_length = np.full(len(output), np.nan)
        for _, group in native.groupby("iu_id", sort=False):
            indices = group.index.to_numpy()
            iu_position[indices] = (
                0.0 if len(indices) == 1 else np.arange(len(indices)) / (len(indices) - 1)
            )
            iu_onset[indices[0]] = 1
            iu_end[indices[-1]] = 1
            iu_length[indices] = len(indices)
        output["iu_position"] = iu_position
        output["iu_onset"] = iu_onset
        output["iu_end"] = iu_end
        output["iu_length"] = iu_length
    output["story_position"] = (
        np.zeros(len(output))
        if len(output) <= 1
        else np.linspace(0.0, 1.0, len(output))
    )
    if len(output) <= 1 or not np.isfinite(starts[[0, -1]]).all() or starts[-1] <= starts[0]:
        output["story_time_position"] = np.nan
    else:
        output["story_time_position"] = (starts - starts[0]) / (starts[-1] - starts[0])
    metadata = {
        "native_columns": sorted(set(output) - {"token_id"}),
        "media": document.media,
    }
    return AnnotationResult(output, metadata=metadata)


def _read_table(spec: dict) -> pd.DataFrame:
    path = Path(spec["path"])
    separator = spec.get("separator")
    if separator is None:
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator, encoding=spec.get("encoding", "utf-8"))


def _resource_lookup(
    table: pd.DataFrame,
    spec: dict,
    token_table: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    key_column = spec.get("key_column", "word")
    if key_column not in table:
        raise ValueError(f"Lexical resource is missing key column {key_column!r}")
    columns = spec.get("columns", {})
    missing = set(columns.values()) - set(table)
    if missing:
        raise ValueError(f"Lexical resource is missing value columns {sorted(missing)}")
    resource_pos = spec.get("pos_column")
    table = table.copy()
    table = table.loc[table[key_column].notna()].copy()
    table["__key_exact"] = table[key_column].astype(str)
    table = table.loc[table["__key_exact"].str.strip().ne("")].copy()
    table["__key_lower"] = table["__key_exact"].str.lower()
    if resource_pos and resource_pos in table:
        table["__pos"] = table[resource_pos].astype(str).str.lower()

    exact = {}
    lower = {}
    exact_pos = {}
    lower_pos = {}
    lemma_pos = {}
    for _, row in table.iterrows():
        exact.setdefault(row["__key_exact"], row)
        lower.setdefault(row["__key_lower"], row)
        if "__pos" in row:
            exact_pos.setdefault((row["__key_exact"], row["__pos"]), row)
            lower_pos.setdefault((row["__key_lower"], row["__pos"]), row)
            lemma_pos.setdefault((row["__key_lower"], row["__pos"]), row)

    lemma_column = spec.get("lemma_feature", "lemma")
    pos_column = spec.get("pos_feature", "part_of_speech")
    lemmas = (
        token_table[lemma_column].fillna("").astype(str).str.lower().to_numpy()
        if lemma_column in token_table
        else np.array([_surface(word) for word in token_table["word"]])
    )
    poses = (
        token_table[pos_column].fillna("").astype(str).str.lower().to_numpy()
        if pos_column in token_table
        else np.repeat("", len(token_table))
    )
    rows: list[pd.Series | None] = [None] * len(token_table)
    levels = np.repeat("missing", len(token_table)).astype(object)
    keys = np.repeat("", len(token_table)).astype(object)
    for index, surface in enumerate(token_table["word"].astype(str)):
        if resource_pos:
            candidates = (
                ("exact_surface_pos", exact_pos.get((surface, poses[index]))),
                ("lower_surface_pos", lower_pos.get((surface.lower(), poses[index]))),
                ("lemma_pos", lemma_pos.get((lemmas[index], poses[index]))),
            )
        else:
            candidates = (
                ("exact_surface", exact.get(surface)),
                ("lower_surface", lower.get(surface.lower())),
                ("lower_lemma", lower.get(lemmas[index])),
            )
        for level, row in candidates:
            if row is not None:
                rows[index] = row
                levels[index] = level
                keys[index] = str(row["__key_exact"])
                break

    spans = []
    if spec.get("multiword", True):
        phrases = {
            tuple(key.split()): row
            for key, row in lower.items()
            if len(key.split()) > 1
        }
        normalized = [_surface(word) for word in token_table["word"]]
        occupied: set[int] = set()
        candidates = []
        for start in range(len(normalized)):
            for phrase, row in phrases.items():
                stop = start + len(phrase)
                if stop > len(normalized):
                    continue
                if tuple(normalized[start:stop]) != phrase:
                    continue
                if token_table.loc[start:stop - 1, "sentence_id"].nunique() != 1:
                    continue
                candidates.append((len(phrase), start, stop, row))
        for _, start, stop, row in sorted(candidates, key=lambda item: (-item[0], item[1])):
            indices = set(range(start, stop))
            if indices & occupied:
                continue
            occupied |= indices
            for index in indices:
                rows[index] = row
                levels[index] = "multiword"
                keys[index] = str(row["__key_exact"])
            spans.append(
                {
                    "span_id": f"phrase_{start}_{stop}",
                    "start_token_id": start,
                    "end_token_id": stop - 1,
                    "label": str(row["__key_exact"]),
                    "span_type": "lexical_multiword",
                }
            )

    values = {}
    for output_name, source_name in columns.items():
        values[output_name] = np.array(
            [np.nan if row is None else row[source_name] for row in rows]
        )
    return values, pd.DataFrame(spans), {"levels": levels, "keys": keys}


@register_annotation_provider("lexical_norms", version="1", requires=("native", "ud"))
def lexical_norms_provider(document, token_table, options, resources) -> AnnotationResult:
    del document
    output = _empty_features(token_table)
    spans = []
    coverage = {}
    resource_names = options.get("resources") or [
        name for name, spec in resources.items() if spec.get("type") == "lexical_norm"
    ]
    for resource_name in resource_names:
        if resource_name not in resources:
            raise ValueError(f"No staged lexical resource named {resource_name!r}")
        spec = resources[resource_name]
        values, resource_spans, provenance = _resource_lookup(
            _read_table(spec), spec, token_table
        )
        for feature, feature_values in values.items():
            if feature in output:
                raise ValueError(f"Lexical resources both emit {feature!r}")
            numeric = pd.to_numeric(pd.Series(feature_values), errors="coerce")
            output[feature] = numeric.to_numpy()
            output[f"{feature}_available"] = numeric.notna().to_numpy()
            coverage[feature] = int(numeric.notna().sum())
        output[f"{resource_name}_match_level"] = provenance["levels"]
        output[f"{resource_name}_match_key"] = provenance["keys"]
        if not resource_spans.empty:
            resource_spans.insert(0, "resource", resource_name)
            resource_spans["span_id"] = resource_spans["span_id"].map(
                lambda value: f"{resource_name}_{value}"
            )
            spans.append(resource_spans)

    if set(LANCASTER_COLUMNS) <= set(output):
        strengths = output[list(LANCASTER_COLUMNS)].to_numpy(float)
        complete = np.isfinite(strengths).all(axis=1)
        magnitude = np.full(len(output), np.nan)
        exclusivity = np.full(len(output), np.nan)
        magnitude[complete] = np.power(np.sum(np.power(strengths[complete], 3), axis=1), 1 / 3)
        totals = np.sum(strengths[complete], axis=1)
        valid_total = totals > 0
        complete_indices = np.flatnonzero(complete)
        exclusivity[complete_indices[valid_total]] = (
            strengths[complete][valid_total].max(axis=1) / totals[valid_total]
        )
        output["sensorimotor_strength"] = magnitude
        output["sensorimotor_strength_available"] = np.isfinite(magnitude)
        output["sensorimotor_exclusivity"] = exclusivity
        output["sensorimotor_exclusivity_available"] = np.isfinite(exclusivity)
    if "valence" in output:
        valence = pd.to_numeric(output["valence"], errors="coerce")
        output["emotional_intensity"] = np.abs(valence)
        output["emotional_intensity_available"] = np.isfinite(valence)
    return AnnotationResult(
        output,
        spans=pd.concat(spans, ignore_index=True) if spans else pd.DataFrame(),
        metadata={"coverage": coverage, "resources": resource_names},
    )


@register_annotation_provider("swow", version="1", requires=("native", "ud"))
def swow_provider(document, token_table, options, resources) -> AnnotationResult:
    del document
    output = _empty_features(token_table)
    resource_name = options.get("resource", "swow")
    if resource_name not in resources:
        raise ValueError(f"No staged SWOW resource named {resource_name!r}")
    spec = resources[resource_name]
    table = _read_table(spec)
    cue_col = spec.get("cue_column", "cue")
    response_col = spec.get("response_column", "response")
    weight_col = spec.get("weight_column")
    required = {cue_col, response_col} | ({weight_col} if weight_col else set())
    if not required <= set(table):
        raise ValueError(f"SWOW resource is missing columns {sorted(required - set(table))}")
    statistics = {}
    for cue, group in table.groupby(table[cue_col].astype(str).str.lower()):
        if weight_col:
            weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0.0)
            response_weights = weights.groupby(group[response_col].astype(str)).sum()
        else:
            response_weights = group[response_col].astype(str).value_counts().astype(float)
        probabilities = response_weights.to_numpy(dtype=float, copy=True)
        total = probabilities.sum()
        if total <= 0:
            entropy = np.nan
        else:
            probabilities /= total
            entropy = float(-(probabilities * np.log(probabilities)).sum())
        statistics[str(cue)] = (float(len(response_weights)), entropy)
    lemmas = token_table.get("lemma", token_table["word"]).fillna("").astype(str)
    matched = [statistics.get(_surface(lemma), (np.nan, np.nan)) for lemma in lemmas]
    output["association_degree"] = [item[0] for item in matched]
    output["association_degree_available"] = np.isfinite(output["association_degree"])
    output["association_entropy"] = [item[1] for item in matched]
    output["association_entropy_available"] = np.isfinite(output["association_entropy"])
    return AnnotationResult(output, metadata={"resource": resource_name})


def _wordnet_pos(value: object):
    mapping = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
    return mapping.get(str(value).upper())


@register_annotation_provider("wordnet", version="1", requires=("native", "ud"))
def wordnet_provider(document, token_table, options, resources) -> AnnotationResult:
    del document
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError as exc:
        raise ImportError("The wordnet provider requires the 'nltk' package") from exc
    nltk_data = options.get("nltk_data")
    if options.get("model_resource"):
        nltk_data = resources[options["model_resource"]]["path"]
    if nltk_data:
        nltk.data.path.insert(0, str(nltk_data))
    output = _empty_features(token_table)
    names = [
        "sense_count",
        "wordnet_lexname_mfs",
        "person_relatedness_mfs",
        "animacy_mfs",
        "motion_concept_mfs",
        "cognition_concept_mfs",
        "communication_concept_mfs",
        "perception_concept_mfs",
        "number_of_hypernyms_mfs",
        "ontology_depth_mfs",
    ]
    values = {name: [] for name in names}
    person = wn.synset("person.n.01")
    animal = wn.synset("animal.n.01")
    living = wn.synset("living_thing.n.01")
    for _, row in token_table.iterrows():
        lemma = _surface(row.get("lemma", row["word"]))
        pos = _wordnet_pos(row.get("part_of_speech", ""))
        synsets = wn.synsets(lemma, pos=pos) if lemma and pos else []
        values["sense_count"].append(len(synsets))
        if not synsets:
            for name in names[1:]:
                values[name].append(np.nan if name.endswith("depth_mfs") or name.startswith("number_") else pd.NA)
            continue
        synset = synsets[0]
        ancestors = set(
            synset.closure(lambda item: item.hypernyms() + item.instance_hypernyms())
        )
        ancestors.add(synset)
        lexname = synset.lexname()
        is_person = person in ancestors
        if pos != "n":
            animacy = pd.NA
        elif is_person:
            animacy = "human"
        elif animal in ancestors:
            animacy = "animal"
        elif living in ancestors:
            animacy = "other_animate"
        else:
            animacy = "inanimate"
        values["wordnet_lexname_mfs"].append(lexname)
        values["person_relatedness_mfs"].append(bool(is_person))
        values["animacy_mfs"].append(animacy)
        values["motion_concept_mfs"].append(lexname == "verb.motion")
        values["cognition_concept_mfs"].append(lexname in {"verb.cognition", "noun.cognition"})
        values["communication_concept_mfs"].append(lexname in {"verb.communication", "noun.communication"})
        values["perception_concept_mfs"].append(lexname == "verb.perception")
        values["number_of_hypernyms_mfs"].append(len(ancestors) - 1)
        values["ontology_depth_mfs"].append(synset.min_depth())
    for name, feature_values in values.items():
        output[name] = feature_values
        output[f"{name}_available"] = pd.Series(feature_values).notna().to_numpy()
    return AnnotationResult(output, metadata={"sense_policy": "most_frequent_sense"})


def _align_parser_word(start: int, end: int, canonical_spans: list[tuple[int, int]]) -> int | None:
    overlaps = [
        (min(end, canonical_end) - max(start, canonical_start), index)
        for index, (canonical_start, canonical_end) in enumerate(canonical_spans)
        if min(end, canonical_end) > max(start, canonical_start)
    ]
    return max(overlaps)[1] if overlaps else None


def _sentence_parse_text(words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    pieces = []
    spans = []
    cursor = 0
    for word in words:
        if pieces:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(word)
        cursor += len(word)
        spans.append((start, cursor))
    return "".join(pieces), spans


def _dependency_derivatives(heads: np.ndarray) -> dict[str, np.ndarray]:
    n = len(heads)
    children = [[] for _ in range(n)]
    for dependent, head in enumerate(heads):
        if head >= 0:
            children[head].append(dependent)
    depth = np.full(n, np.nan)
    subtree = np.zeros(n, dtype=float)

    def get_depth(index, trail=()):
        if np.isfinite(depth[index]):
            return depth[index]
        if index in trail:
            raise ValueError("Collapsed dependency graph contains a cycle")
        depth[index] = 0 if heads[index] < 0 else get_depth(heads[index], (*trail, index)) + 1
        return depth[index]

    def get_subtree(index, trail=()):
        if index in trail:
            raise ValueError("Collapsed dependency graph contains a cycle")
        subtree[index] = 1 + sum(get_subtree(child, (*trail, index)) for child in children[index])
        return subtree[index]

    for index in range(n):
        get_depth(index)
    roots = np.flatnonzero(heads < 0)
    for root in roots:
        get_subtree(root)
    for index in range(n):
        if not subtree[index]:
            get_subtree(index)
    length = np.where(heads < 0, np.nan, np.abs(np.arange(n) - heads))
    direction = np.where(heads < 0, "root", np.where(heads < np.arange(n), "left", "right"))
    open_dependencies = np.zeros(n, dtype=float)
    locality = np.zeros(n, dtype=float)
    for dependent, head in enumerate(heads):
        if head < 0:
            continue
        left, right = sorted((dependent, int(head)))
        open_dependencies[left:right] += 1
        locality[right] += right - left
    return {
        "dependency_depth": depth,
        "dependency_length": length,
        "head_direction": direction,
        "number_of_dependents": np.array([len(items) for items in children], float),
        "subtree_size": subtree,
        "left_dependents": np.array([sum(child < index for child in items) for index, items in enumerate(children)], float),
        "right_dependents": np.array([sum(child > index for child in items) for index, items in enumerate(children)], float),
        "open_dependencies": open_dependencies,
        "dependency_locality_cost": locality,
    }


def _choose_parser_component(parser_words, candidates, parser_depths) -> int:
    """Choose a canonical token's lexical syntactic head from parser components."""
    lexical_candidates = [
        index for index in candidates if parser_words[index].upos != "PUNCT"
    ]
    eligible = lexical_candidates or candidates
    return min(eligible, key=lambda index: (parser_depths[index], index))


@register_annotation_provider("ud", version="2", requires=("native",))
def stanza_ud_provider(document, token_table, options, resources) -> AnnotationResult:
    del document
    try:
        import stanza
    except ImportError as exc:
        raise ImportError("The UD provider requires the 'stanza' package") from exc
    model_dir = options.get("model_dir")
    if options.get("model_resource"):
        model_dir = resources[options["model_resource"]]["path"]
    pipeline_kwargs = dict(
        lang=options.get("language", "en"),
        package=options.get("package", "ewt"),
        processors="tokenize,pos,lemma,depparse",
        tokenize_no_ssplit=True,
        download_method=None,
        use_gpu=bool(options.get("use_gpu", False)),
        verbose=False,
    )
    if model_dir:
        pipeline_kwargs["dir"] = model_dir
    pipeline = stanza.Pipeline(**pipeline_kwargs)
    output = _empty_features(token_table)
    scalar_columns = {
        "lemma": pd.NA,
        "part_of_speech": pd.NA,
        "fine_part_of_speech": pd.NA,
        "morphological_feature": pd.NA,
        "dependency_relation": pd.NA,
        "dependency_head_token_id": -1,
    }
    for name, default in scalar_columns.items():
        output[name] = [default] * len(output)
    provider_rows = []
    sentence_derivatives = defaultdict(lambda: np.full(len(output), np.nan, dtype=object))
    for sentence_id, group in token_table.groupby("sentence_id", sort=False):
        canonical_indices = group.index.to_numpy()
        text, spans = _sentence_parse_text(group["word"].astype(str).tolist())
        parsed = pipeline(text)
        parser_words = [word for sentence in parsed.sentences for word in sentence.words]
        parser_to_canonical = {}
        for parser_index, word in enumerate(parser_words):
            start = getattr(word, "start_char", None)
            end = getattr(word, "end_char", None)
            if start is None or end is None:
                # Stanza Word offsets can be absent; its parent Token carries them.
                token = next(
                    token
                    for sentence in parsed.sentences
                    for token in sentence.tokens
                    if word in token.words
                )
                start, end = token.start_char, token.end_char
            local = _align_parser_word(int(start), int(end), spans)
            if local is None:
                if str(word.text).strip():
                    raise ValueError(
                        f"Could not align UD token {word.text!r} in sentence {sentence_id}"
                    )
                continue
            canonical = int(canonical_indices[local])
            parser_to_canonical[parser_index] = canonical
            provider_rows.append(
                {
                    "provider_token_id": parser_index,
                    "sentence_id": sentence_id,
                    "token_id": canonical,
                    "text": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats,
                    "head": word.head,
                    "deprel": word.deprel,
                    "start_char": start,
                    "end_char": end,
                }
            )
        by_canonical = defaultdict(list)
        for parser_index, canonical in parser_to_canonical.items():
            by_canonical[canonical].append(parser_index)
        parser_depths = np.full(len(parser_words), np.nan)

        def parser_depth(index, trail=()):
            if np.isfinite(parser_depths[index]):
                return parser_depths[index]
            if index in trail:
                raise ValueError(f"Stanza returned a dependency cycle in sentence {sentence_id}")
            head = parser_words[index].head - 1
            parser_depths[index] = (
                0 if head < 0 else parser_depth(head, (*trail, index)) + 1
            )
            return parser_depths[index]

        for parser_index in range(len(parser_words)):
            parser_depth(parser_index)
        # Stanza head IDs are one-based within each parsed sentence. Parsing one
        # reconstructed source sentence at a time normally yields one sentence.
        for canonical, candidates in by_canonical.items():
            chosen = _choose_parser_component(
                parser_words,
                candidates,
                parser_depths,
            )
            word = parser_words[chosen]
            output.loc[canonical, "lemma"] = word.lemma
            output.loc[canonical, "part_of_speech"] = word.upos
            output.loc[canonical, "fine_part_of_speech"] = word.xpos
            output.loc[canonical, "morphological_feature"] = word.feats or ""
            output.loc[canonical, "dependency_relation"] = word.deprel
            head = word.head - 1
            seen = set()
            while head >= 0 and parser_to_canonical.get(head) == canonical:
                if head in seen:
                    raise ValueError("Cycle while collapsing UD token components")
                seen.add(head)
                head = parser_words[head].head - 1
            output.loc[canonical, "dependency_head_token_id"] = (
                -1 if head < 0 else parser_to_canonical.get(head, -1)
            )
        local_heads = output.loc[canonical_indices, "dependency_head_token_id"].to_numpy(int)
        global_to_local = {global_id: local for local, global_id in enumerate(canonical_indices)}
        local_heads = np.array([global_to_local.get(head, -1) for head in local_heads], int)
        derivatives = _dependency_derivatives(local_heads)
        for name, values in derivatives.items():
            sentence_derivatives[name][canonical_indices] = values
    output["content_word"] = output["part_of_speech"].isin(CONTENT_POS).astype(int)
    for feature in ("Tense", "Aspect", "Mood", "Voice", "Person", "Number", "Definite"):
        output_name = {
            "Person": "grammatical_person",
            "Number": "number",
            "Definite": "definiteness",
        }.get(feature, feature.lower())
        output[output_name] = output["morphological_feature"].map(
            lambda value: next(
                (
                    item.split("=", 1)[1]
                    for item in str(value).split("|")
                    if item.startswith(feature + "=")
                ),
                pd.NA,
            )
        )
    for name, values in sentence_derivatives.items():
        output[name] = pd.Series(values).infer_objects()
    return AnnotationResult(
        output,
        provider_tokens=pd.DataFrame(provider_rows),
        metadata={
            "backend": "stanza",
            "language": options.get("language", "en"),
            "package": options.get("package", "ewt"),
            "tokenization": "parser_native_with_character_alignment",
        },
    )


def _ner_spans_from_tags(tags: list[str], canonical_ids: np.ndarray) -> list[tuple[int, int, str]]:
    """Convert Stanza BIOES tags into inclusive canonical-token spans."""
    spans = []
    start = None
    label = None
    for position, raw_tag in enumerate([*tags, "O"]):
        tag = raw_tag or "O"
        prefix, _, current_label = tag.partition("-")
        if start is not None and (prefix in {"O", "B", "S"} or current_label != label):
            spans.append((int(canonical_ids[start]), int(canonical_ids[position - 1]), str(label)))
            start = None
            label = None
        if prefix == "S":
            spans.append((int(canonical_ids[position]), int(canonical_ids[position]), current_label))
        elif prefix == "B":
            start, label = position, current_label
        elif prefix in {"I", "E"} and start is None:
            start, label = position, current_label
        if prefix == "E" and start is not None:
            spans.append((int(canonical_ids[start]), int(canonical_ids[position]), str(label)))
            start = None
            label = None
    return spans


def _coreference_mention_form(token_table: pd.DataFrame, start: int, end: int) -> str:
    parts = set(token_table.loc[start:end, "part_of_speech"].dropna().astype(str))
    if "PROPN" in parts:
        return "proper"
    if "PRON" in parts:
        return "pronominal"
    return "nominal"


@register_annotation_provider("entity_coreference", version="1", requires=("ud",))
def entity_coreference_provider(document, token_table, options, resources) -> AnnotationResult:
    """Generate Stanza NER and document-level coreference annotations."""
    del document
    try:
        import stanza
    except ImportError as exc:
        raise ImportError("The entity/coreference provider requires 'stanza'") from exc

    model_dir = options.get("model_dir")
    if options.get("model_resource"):
        model_dir = resources[options["model_resource"]]["path"]
    packages = {
        "ner": options.get("ner_package", "ontonotes-ww-multi_charlm"),
        "coref": options.get("coref_package", "gum-nospeakers_roberta-large-lora"),
    }
    pipeline_kwargs = dict(
        lang=options.get("language", "en"),
        processors="tokenize,ner,coref",
        package=packages,
        tokenize_pretokenized=True,
        download_method=None,
        use_gpu=bool(options.get("use_gpu", False)),
        verbose=False,
    )
    if model_dir:
        pipeline_kwargs["dir"] = model_dir
    foundation_resource = options.get("foundation_resource")
    temporary_coref_path = None
    if foundation_resource:
        import tempfile
        import torch

        foundation_path = resources[foundation_resource]["path"]
        coref_path = (
            Path(model_dir)
            / options.get("language", "en")
            / "coref"
            / f"{packages['coref']}.pt"
        )
        state = torch.load(coref_path, map_location="cpu", weights_only=True)
        state["config"] = dict(state["config"])
        state["config"]["bert_model"] = foundation_path
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temporary:
            temporary_coref_path = temporary.name
        torch.save(state, temporary_coref_path)
        pipeline_kwargs["coref_model_path"] = temporary_coref_path
    try:
        pipeline = stanza.Pipeline(**pipeline_kwargs)
    finally:
        if temporary_coref_path:
            Path(temporary_coref_path).unlink(missing_ok=True)

    sentence_groups = [
        group.index.to_numpy(dtype=int)
        for _, group in token_table.groupby("sentence_id", sort=False)
    ]
    parsed = pipeline(
        [token_table.loc[indices, "word"].astype(str).tolist() for indices in sentence_groups]
    )
    if len(parsed.sentences) != len(sentence_groups):
        raise ValueError(
            "Stanza changed the configured sentence segmentation: "
            f"{len(parsed.sentences)} vs {len(sentence_groups)}"
        )

    output = _empty_features(token_table)
    output["is_named_entity"] = 0
    output["named_entity_type"] = pd.NA
    spans = []
    for sentence_index, (sentence, canonical_ids) in enumerate(
        zip(parsed.sentences, sentence_groups)
    ):
        if len(sentence.words) != len(canonical_ids) or len(sentence.tokens) != len(canonical_ids):
            raise ValueError(
                f"Stanza word alignment changed in sentence {sentence_index}: "
                f"{len(sentence.words)} vs {len(canonical_ids)}"
            )
        tags = [getattr(token, "ner", "O") or "O" for token in sentence.tokens]
        for entity_index, (start, end, label) in enumerate(
            _ner_spans_from_tags(tags, canonical_ids)
        ):
            output.loc[start:end, "is_named_entity"] = 1
            output.loc[start:end, "named_entity_type"] = label
            spans.append(
                {
                    "span_id": f"entity_{sentence_index}_{entity_index}",
                    "start_token_id": start,
                    "end_token_id": end,
                    "label": label,
                    "span_type": "named_entity",
                }
            )

    output["is_coreference_mention"] = 0
    output["coreference_newness"] = pd.NA
    output["coreference_mention_form"] = pd.NA
    output["coreference_mention_order"] = np.nan
    output["coreference_chain_size"] = np.nan
    output["coreference_antecedent_distance_words"] = np.nan
    output["coreference_chain_id"] = pd.NA
    output["coreference_mention_id"] = pd.NA
    mention_records = []
    relations = []
    for chain in parsed.coref:
        previous_start = None
        previous_span_id = None
        surface_mentions = []
        for mention in chain.mentions:
            if not isinstance(mention.start_word, int) or not isinstance(mention.end_word, int):
                continue
            canonical_ids = sentence_groups[mention.sentence]
            if mention.start_word < 0 or mention.end_word > len(canonical_ids):
                raise ValueError(f"Stanza returned invalid coreference mention {mention}")
            surface_mentions.append(
                (
                    int(canonical_ids[mention.start_word]),
                    int(canonical_ids[mention.end_word - 1]),
                )
            )
        surface_mentions.sort()
        chain_size = len(surface_mentions)
        for mention_order, (start, end) in enumerate(surface_mentions, start=1):
            span_id = f"coreference_{chain.index}_{mention_order}"
            record = {
                "chain_id": int(chain.index),
                "mention_id": span_id,
                "mention_order": mention_order,
                "chain_size": chain_size,
                "start": start,
                "end": end,
                "newness": "first" if mention_order == 1 else "subsequent",
                "form": _coreference_mention_form(token_table, start, end),
                "antecedent_distance": (
                    np.nan if previous_start is None else start - previous_start
                ),
            }
            mention_records.append(record)
            spans.append(
                {
                    "span_id": span_id,
                    "start_token_id": start,
                    "end_token_id": end,
                    "label": record["newness"],
                    "span_type": "coreference_mention",
                }
            )
            if previous_span_id is not None:
                relations.append(
                    {
                        "relation_id": f"antecedent_{chain.index}_{mention_order}",
                        "source_span_id": previous_span_id,
                        "target_span_id": span_id,
                        "label": "precedes_in_chain",
                    }
                )
            previous_start = start
            previous_span_id = span_id

    # Sidecars preserve every overlapping mention. Token columns use the shortest
    # mention, then the earliest chain/order, for one deterministic scalar label.
    for record in sorted(
        mention_records,
        key=lambda item: (
            item["end"] - item["start"],
            item["chain_id"],
            item["mention_order"],
        ),
        reverse=True,
    ):
        selected = slice(record["start"], record["end"])
        output.loc[selected, "is_coreference_mention"] = 1
        output.loc[selected, "coreference_newness"] = record["newness"]
        output.loc[selected, "coreference_mention_form"] = record["form"]
        output.loc[selected, "coreference_mention_order"] = record["mention_order"]
        output.loc[selected, "coreference_chain_size"] = record["chain_size"]
        output.loc[selected, "coreference_antecedent_distance_words"] = record[
            "antecedent_distance"
        ]
        output.loc[selected, "coreference_chain_id"] = record["chain_id"]
        output.loc[selected, "coreference_mention_id"] = record["mention_id"]

    return AnnotationResult(
        output,
        spans=pd.DataFrame(spans),
        relations=pd.DataFrame(relations),
        metadata={
            "backend": "stanza",
            "language": options.get("language", "en"),
            "ner_package": packages["ner"],
            "coref_package": packages["coref"],
            "tokenization": "canonical_pretokenized_sentences",
            "overlap_policy": "shortest_then_earliest_chain_and_mention",
        },
    )
