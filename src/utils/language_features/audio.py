"""Word-aligned audio measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_rms_for_words(
    audio: np.ndarray,
    sample_rate: int,
    words: pd.DataFrame,
    *,
    window_seconds: float = 0.5,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Mean Librosa frame RMS in a fixed window after each word onset.

    ``audio`` is mono floating-point PCM. Word start times are in seconds.
    """
    import librosa

    samples = np.asarray(audio)
    if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.floating):
        raise ValueError("audio must be a one-dimensional floating-point waveform")
    if sample_rate <= 0 or window_seconds <= 0:
        raise ValueError("sample_rate and window_seconds must be positive")
    if "start" not in words:
        raise ValueError("words must contain a start column")
    starts = pd.to_numeric(words["start"], errors="coerce").to_numpy(float)
    values = np.full(len(words), np.nan, dtype=float)
    window_samples = round(window_seconds * sample_rate)
    for index, start in enumerate(starts):
        if not (np.isfinite(start) and start >= 0):
            continue
        first = round(start * sample_rate)
        last = first + window_samples
        if first < 0 or last > len(samples):
            continue
        frames = librosa.feature.rms(
            y=samples[first:last], frame_length=frame_length,
            hop_length=hop_length, center=True,
        )
        values[index] = float(frames.mean())
    return values
