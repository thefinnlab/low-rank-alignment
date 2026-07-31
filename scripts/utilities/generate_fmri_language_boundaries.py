#!/usr/bin/env python
"""Generate SaT sentence and PSST intonation-unit boundaries for fMRI stories."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.io import wavfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from wtpsplit import SaT

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.fmri import load_story_word_data


DEFAULT_PSST = Path(
    "foundation_model_weights/models--NathanRoll--psst-medium-en/snapshots/"
    "a1112ed5890f654737cfe71eb27bffd8e1ffe363"
)


def normalize_word(word: object) -> str:
    """Lowercase and strip punctuation except apostrophes."""
    return re.sub(r"[^a-z0-9']", "", str(word).lower())


def parse_psst_output(text: str) -> list[tuple[str, str | None]]:
    """Retain words and explicit IU markers from decoded PSST text."""
    result: list[tuple[str, str | None]] = []
    parts = re.split(r"<\|IU_Boundary\|>", text.replace("!!!!!", "<|IU_Boundary|>"))
    for index, part in enumerate(parts):
        result.extend(("word", word) for word in part.strip().split() if word)
        if index < len(parts) - 1:
            result.append(("boundary", None))
    return result


def align_psst_to_transcript(
    psst_tokens: list[tuple[str, str | None]], transcript: pd.DataFrame
) -> list[dict]:
    """DP-align PSST words and map its IU markers onto reference timestamps."""
    psst_words: list[str] = []
    boundary_after: list[int] = []
    for kind, value in psst_tokens:
        if kind == "word":
            psst_words.append(normalize_word(value))
        else:
            boundary_after.append(len(psst_words) - 1)
    if not psst_words:
        return []
    reference = [normalize_word(word) for word in transcript["word"]]
    n_psst, n_ref = len(psst_words), len(reference)
    scores = np.full((n_psst + 1, n_ref + 1), -np.inf, dtype=np.float32)
    back = np.zeros((n_psst + 1, n_ref + 1), dtype=np.int8)
    scores[0, :] = 0.0
    for i in range(1, n_psst + 1):
        for j in range(1, n_ref + 1):
            candidates = (
                scores[i - 1, j - 1]
                + (2.0 if psst_words[i - 1] == reference[j - 1] else -1.0),
                scores[i, j - 1] - 0.3,
                scores[i - 1, j] - 2.0,
            )
            back[i, j] = int(np.argmax(candidates))
            scores[i, j] = candidates[back[i, j]]
    i, j = n_psst, int(np.argmax(scores[n_psst]))
    mapping: dict[int, int] = {}
    while i > 0 and j > 0:
        direction = back[i, j]
        if direction == 0:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif direction == 1:
            j -= 1
        else:
            i -= 1
    boundaries = []
    for position in boundary_after:
        previous = mapping.get(position)
        following = mapping.get(position + 1)
        if previous is None and following is None:
            continue
        if previous is None:
            time = float(transcript.iloc[following]["start"])
            previous_word, following_word = "", transcript.iloc[following]["word"]
        elif following is None:
            time = float(transcript.iloc[previous]["end"])
            previous_word, following_word = transcript.iloc[previous]["word"], ""
        else:
            previous_end = float(transcript.iloc[previous]["end"])
            following_start = float(transcript.iloc[following]["start"])
            time = (
                (previous_end + following_start) / 2
                if following_start > previous_end
                else previous_end
            )
            previous_word = transcript.iloc[previous]["word"]
            following_word = transcript.iloc[following]["word"]
        boundaries.append(
            {"time": time, "prev_word": previous_word, "next_word": following_word}
        )
    return boundaries


def sentence_ids(words: list[str], sat: SaT) -> pd.DataFrame:
    """Run TED-adapted SaT and snap character boundaries to reference words."""
    text = " ".join(words)
    segments = list(sat.split(text, strip_whitespace=False))
    word_stops = np.cumsum([len(word) + 1 for word in words]) - 1
    boundary_stops = []
    cursor = 0
    for segment in segments[:-1]:
        stripped = segment.strip()
        start = text.find(stripped, cursor)
        if start < 0:
            raise ValueError("Could not align a SaT segment to the fMRI word stream")
        stop = start + len(stripped)
        # SaT predicts character boundaries and may split inside a TextGrid word.
        # Snap to the nearest word end while preserving monotonically ordered cuts.
        word_index = int(np.argmin(np.abs(word_stops - stop)))
        boundary_stops.append(word_index)
        cursor = stop
    boundary_stops = np.unique(boundary_stops)
    ids = np.searchsorted(boundary_stops, np.arange(len(words)), side="left")
    return pd.DataFrame(
        {"token_id": np.arange(len(words)), "word": words, "sentence_id": ids}
    )


def transcribe_ius(
    audio_path: Path,
    transcript: pd.DataFrame,
    model,
    processor,
    *,
    chunk_duration: float,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Run PSST in non-overlapping chunks and align IUs to timed reference words."""
    sample_rate, samples = wavfile.read(audio_path)
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples.astype(np.float32).mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        scale = float(max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max))
        samples = samples.astype(np.float32) / scale
    else:
        samples = samples.astype(np.float32)
    waveform = torch.from_numpy(np.ascontiguousarray(samples))
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    chunk_samples = int(chunk_duration * 16_000)
    chunks = []
    for start_sample in range(0, waveform.numel(), chunk_samples):
        stop_sample = min(start_sample + chunk_samples, waveform.numel())
        start_time, stop_time = start_sample / 16_000, stop_sample / 16_000
        reference = transcript.loc[
            (transcript["end"] >= start_time - 0.5)
            & (transcript["start"] <= stop_time + 0.5)
        ].reset_index(drop=True)
        if reference.empty:
            continue
        chunks.append((waveform[start_sample:stop_sample].numpy(), reference))
    boundaries: list[dict] = []
    seen: set[float] = set()
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        inputs = processor(
            [audio for audio, _ in batch],
            return_tensors="pt",
            sampling_rate=16_000,
            padding=True,
        )
        features = inputs.input_features.to(device=device, dtype=model.dtype)
        with torch.inference_mode():
            generated = model.generate(input_features=features, max_length=448)
        decoded_batch = processor.batch_decode(generated, skip_special_tokens=False)
        for raw, (_, reference) in zip(decoded_batch, batch):
            decoded = re.sub(r"<\|(?!IU_Boundary)[^|]+\|>", "", raw).strip()
            for boundary in align_psst_to_transcript(
                parse_psst_output(decoded), reference
            ):
                rounded = round(float(boundary["time"]), 4)
                if rounded not in seen:
                    seen.add(rounded)
                    boundaries.append(boundary)
    return pd.DataFrame(
        boundaries, columns=["time", "prev_word", "next_word"]
    ).sort_values("time").reset_index(drop=True)


def selected_stories(repo_root: Path, requested: list[str] | None) -> list[str]:
    grids = repo_root / "data" / "ds003020" / "derivatives" / "TextGrids"
    available = {path.stem for path in grids.glob("*.TextGrid")}
    if requested:
        missing = set(requested) - available
        if missing:
            raise ValueError(f"Missing fMRI TextGrids: {sorted(missing)}")
        return requested
    return sorted(available)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmri-repo-root", default="data/deep-fMRI-dataset")
    parser.add_argument("--output-root", default="processed_data/language_boundaries/fmri")
    parser.add_argument("--stories", nargs="+")
    parser.add_argument("--psst-model", default=str(DEFAULT_PSST))
    parser.add_argument("--sat-model", default="sat-3l")
    parser.add_argument("--sat-style", default="ted2020-corrupted")
    parser.add_argument("--chunk-duration", type=float, default=25.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    repo_root, output_root = Path(args.fmri_repo_root), Path(args.output_root)
    stories = selected_stories(repo_root, args.stories)
    frames, _ = load_story_word_data(str(repo_root), stories)
    sat = SaT(args.sat_model, style_or_domain=args.sat_style, language="en")
    if args.device == "cuda":
        sat.half().to("cuda")
    psst_path = Path(args.psst_model)
    processor = AutoProcessor.from_pretrained(psst_path, local_files_only=True)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        psst_path, local_files_only=True, torch_dtype=dtype
    ).to(args.device).eval()

    for number, story in enumerate(stories, start=1):
        output = output_root / story
        sentence_path, iu_path = output / "sentence_ids.tsv", output / "iu_boundaries.csv"
        if sentence_path.exists() and iu_path.exists() and not args.refresh:
            print(f"[{number}/{len(stories)}] {story}: reuse")
            continue
        print(f"[{number}/{len(stories)}] {story}", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        transcript = frames[story]
        sentence_ids(transcript["word"].astype(str).tolist(), sat).to_csv(
            sentence_path, sep="\t", index=False
        )
        audio = repo_root / "data" / "ds003020" / "stimuli" / f"{story}.wav"
        if not audio.exists():
            raise FileNotFoundError(audio)
        transcribe_ius(
            audio,
            transcript,
            model,
            processor,
            chunk_duration=args.chunk_duration,
            device=args.device,
            batch_size=args.batch_size,
        ).to_csv(iu_path, index=False)


if __name__ == "__main__":
    main()
