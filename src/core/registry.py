"""Registries for embedding getters and dataset adapters."""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Embedding getter registry
# ---------------------------------------------------------------------------


embedding_getter_registry = {}


def register_embedding_getter(name=None):
    """Decorator to register an embedding getter."""

    def decorator(fn):
        model_name = name or fn.__name__
        embedding_getter_registry[model_name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetContext:
    """Normalized stimulus context shared across dataset-aware loaders."""

    dataset: str
    stimulus_id: str
    stimulus_label: str
    subject_metadata: dict[int, dict] = field(default_factory=dict)


dataset_registry = {}


def register_dataset(name=None):
    """Decorator to register a dataset adapter instance."""

    def decorator(cls):
        adapter = cls()
        dataset_name = name or getattr(adapter, "name", cls.__name__)
        dataset_registry[dataset_name] = adapter
        return cls

    return decorator


def get_dataset_adapter(name: str):
    if name not in dataset_registry:
        raise ValueError(f"Dataset '{name}' is not registered.")
    return dataset_registry[name]


def resolve_dataset_context(
    dataset: str = "podcast",
    subject_ids: list[int] | None = None,
    per_dataset_config: dict | None = None,
    data_root: str = "data/",
) -> DatasetContext:
    adapter = get_dataset_adapter(dataset)
    return adapter.resolve_analysis_context(
        subject_ids=subject_ids or [],
        per_dataset_config=per_dataset_config or {},
        data_root=data_root,
    )


def _load_tsv_or_csv(path: str, sep: str):
    df = pd.read_csv(path, sep=sep)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _ensure_word_idx(df: pd.DataFrame) -> pd.DataFrame:
    if "word_idx" in df.columns:
        return df
    df = df.copy()
    df["word_idx"] = df.index.to_numpy()
    return df


def _drop_untimed_word_rows(df: pd.DataFrame, reset_index: bool = True) -> pd.DataFrame:
    start = pd.to_numeric(df["start"], errors="coerce")
    end = pd.to_numeric(df["end"], errors="coerce")
    dropped = df.loc[start.notna() & end.notna()].copy()
    if reset_index:
        dropped = dropped.reset_index(drop=True)
    return dropped


def _drop_untimed_generated_transcript(df: pd.DataFrame) -> pd.DataFrame:
    dropped = _drop_untimed_word_rows(df, reset_index=False)
    if "word_idx" not in dropped.columns:
        return dropped
    old_word_ids = dropped["word_idx"].drop_duplicates().tolist()
    remap = {old: new for new, old in enumerate(old_word_ids)}
    dropped = dropped.copy()
    dropped["source_word_idx"] = dropped["word_idx"]
    dropped["word_idx"] = dropped["word_idx"].map(remap).astype(int)
    return dropped


@register_dataset("podcast")
class PodcastDatasetAdapter:
    name = "podcast"

    def resolve_analysis_context(
        self, subject_ids: list[int], per_dataset_config: dict, data_root: str
    ) -> DatasetContext:
        return DatasetContext(
            dataset=self.name,
            stimulus_id="podcast",
            stimulus_label="podcast",
        )

    def load_word_table(
        self,
        data_root: str,
        context: DatasetContext,
        model_dir: str = "gpt2-xl",
        drop_untimed_words: bool = False,
        **_: dict,
    ) -> pd.DataFrame:
        transcript_path = os.path.join(data_root, "stimuli", model_dir, "transcript.tsv")
        df = _load_tsv_or_csv(transcript_path, sep="\t")
        df = _ensure_word_idx(df)
        if drop_untimed_words and {"start", "end"} <= set(df.columns):
            df = _drop_untimed_word_rows(df, reset_index=True)
        return df

    def get_embedding_dir(
        self, data_root: str, context: DatasetContext, model_dir: str
    ) -> str:
        return os.path.join(data_root, "stimuli", model_dir)

    def load_generated_transcript(
        self,
        data_root: str,
        context: DatasetContext,
        model_dir: str,
        drop_untimed_words: bool = False,
    ) -> pd.DataFrame:
        transcript_path = os.path.join(
            self.get_embedding_dir(data_root, context, model_dir), "transcript.tsv"
        )
        df = _load_tsv_or_csv(transcript_path, sep="\t")
        df = _ensure_word_idx(df)
        if drop_untimed_words and {"start", "end"} <= set(df.columns):
            df = _drop_untimed_generated_transcript(df)
        return df

    def load_raw_recording(
        self,
        subject_id: int,
        data_root: str,
        context: DatasetContext,
        load_options: dict,
    ):
        import mne
        from mne_bids import BIDSPath

        use_high_gamma = load_options.get("use_high_gamma", False)
        no_preload = load_options.get("no_preload", False)
        do_drop_bads = load_options.get("do_drop_bads", False)
        signal_unit = load_options.get("signal_unit")
        per_subject_electrodes = load_options.get("per_subject_electrodes")
        channel_reg_ex = load_options.get("channel_reg_ex")

        root_path = os.path.join(data_root, "derivatives/ecogprep")
        description = "highgamma" if use_high_gamma else None
        task = load_options.get("task", "podcast")
        file_path = BIDSPath(
            root=root_path,
            subject=f"{subject_id:02}",
            task=task,
            datatype="ieeg",
            description=description,
            suffix="ieeg",
            extension=".fif",
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        raw = mne.io.read_raw_fif(file_path, preload=not no_preload, verbose=False)

        if do_drop_bads:
            try:
                ch_path = file_path.copy()
                ch_path.update(suffix="channels", extension=".tsv")
                tsv_path = str(ch_path)
                if not os.path.exists(tsv_path) and getattr(file_path, "description", None):
                    ch_path2 = file_path.copy()
                    ch_path2.update(description=None, suffix="channels", extension=".tsv")
                    tsv_path = str(ch_path2)
                if not os.path.exists(tsv_path) and "derivatives" in str(file_path):
                    raw_bids_path = BIDSPath(
                        root=data_root,
                        subject=f"{subject_id:02}",
                        task=task,
                        datatype="ieeg",
                        suffix="channels",
                        extension=".tsv",
                    )
                    tsv_path = str(raw_bids_path)
                if os.path.exists(tsv_path):
                    df_ch = pd.read_csv(tsv_path, sep="\t")
                    if "status" in df_ch.columns and "name" in df_ch.columns:
                        bad_channels = df_ch[df_ch["status"] == "bad"]["name"].tolist()
                        bad_channels = [ch for ch in bad_channels if ch in raw.ch_names]
                        if bad_channels:
                            raw.info["bads"].extend(bad_channels)
                            print(f"  - [Config] Dropping {len(bad_channels)} bad channels")
                            raw.drop_channels(bad_channels)
            except Exception as exc:
                print(f"Warning: Failed to process bad channels: {exc}")

        if signal_unit == "uV":
            print("  - [Config] Scaling data: Volt -> MicroVolt (x 1e6)")
            raw.apply_function(lambda x: x * 1e6, channel_wise=False)

        if per_subject_electrodes:
            picks = [ch for ch in per_subject_electrodes[subject_id] if ch in raw.ch_names]
            raw = raw.pick(picks)
        elif channel_reg_ex:
            picks = mne.pick_channels_regexp(raw.ch_names, channel_reg_ex)
            raw = raw.pick(picks)

        return raw

    def get_precomputed_epochs_path(
        self, subject_id: int, data_root: str, context: DatasetContext
    ) -> str:
        return os.path.join(
            data_root,
            "derivatives",
            "ecogprep",
            f"sub-{subject_id:02d}",
            "ieeg",
            f"sub-{subject_id:02d}_task-podcast_desc-epochs32hz_ieeg.h5",
        )

    def load_significance(
        self,
        output_dir: str,
        subject_ids: list[int],
        context: DatasetContext,
        sig_dir: str | None = None,
    ) -> tuple[dict[int, list[str]], pd.DataFrame]:
        csv_dir = sig_dir if sig_dir is not None else output_dir
        sig_csv = os.path.join(csv_dir, "all_subject_sig.csv")
        sig_df = pd.read_csv(sig_csv)
        mapping = {
            sub_id: sig_df[sig_df["subject"] == sub_id]["elec"].tolist()
            for sub_id in subject_ids
        }
        return {s: e for s, e in mapping.items() if e}, sig_df

    def get_results_suffix(self, context: DatasetContext) -> str:
        return ""

    def validate_subjects_share_stimulus(
        self, subject_ids: list[int], per_dataset_config: dict, data_root: str
    ) -> None:
        return None


@register_dataset("brain_treebank")
class BrainTreebankDatasetAdapter:
    name = "brain_treebank"
    sfreq = 2048.0
    hdf5_unit_scale_to_volts = 1e-6
    preprocessing_sfreq = 512.0

    def _metadata_records(self, data_root: str, subject_id: int) -> list[dict]:
        pattern = os.path.join(
            data_root,
            "subject_metadata",
            f"sub_{subject_id}_trial*_metadata.json",
        )
        records = []
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                metadata = json.load(f)
            trial = os.path.basename(path).split("_trial", 1)[1].split("_", 1)[0]
            movie = metadata.get("filename")
            records.append(
                {
                    "subject_id": subject_id,
                    "trial_id": int(trial),
                    "movie": movie,
                    "metadata_path": path,
                    "recording_path": os.path.join(
                        data_root,
                        "all_subject_data",
                        f"sub_{subject_id}_trial{int(trial):03d}.h5",
                    ),
                    "timings_path": os.path.join(
                        data_root,
                        "subject_timings",
                        f"sub_{subject_id}_trial{int(trial):03d}_timings.csv",
                    ),
                    "electrode_labels_path": os.path.join(
                        data_root,
                        "electrode_labels",
                        f"sub_{subject_id}",
                        "electrode_labels.json",
                    ),
                    "title": metadata.get("title", movie),
                }
            )
        return records

    def _select_subject_movie_record(
        self, data_root: str, subject_id: int, movie: str
    ) -> dict:
        records = [
            r for r in self._metadata_records(data_root, subject_id) if r["movie"] == movie
        ]
        if not records:
            raise ValueError(
                f"Subject {subject_id} does not have Brain Treebank stimulus '{movie}'."
            )
        return records[0]

    def resolve_analysis_context(
        self, subject_ids: list[int], per_dataset_config: dict, data_root: str
    ) -> DatasetContext:
        movie = per_dataset_config.get("movie")
        if not movie:
            raise ValueError(
                "Brain Treebank runs require per_dataset_config.movie to identify the movie stimulus."
            )

        subject_metadata = {}
        for subject_id in subject_ids:
            subject_metadata[subject_id] = self._select_subject_movie_record(
                data_root, subject_id, movie
            )

        return DatasetContext(
            dataset=self.name,
            stimulus_id=movie,
            stimulus_label=movie,
            subject_metadata=subject_metadata,
        )

    def load_word_table(
        self,
        data_root: str,
        context: DatasetContext,
        drop_untimed_words: bool = False,
        **_: dict,
    ) -> pd.DataFrame:
        path = os.path.join(
            data_root,
            "transcripts",
            context.stimulus_id,
            "features.csv",
        )
        df = pd.read_csv(path, keep_default_na=False)
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)
        if "text" not in df.columns:
            raise ValueError(f"Brain Treebank transcript missing 'text' column: {path}")
        if "sentence" in df.columns:
            df = df.loc[df["sentence"].map(lambda value: str(value).strip() != "")].copy()
            df = df.reset_index(drop=True)
            df = self._normalize_transcript_words_from_sentences(df)
        else:
            valid_text = (
                df["text"].notna()
                & df["text"].map(lambda value: isinstance(value, str))
                & df["text"].map(lambda value: isinstance(value, str) and value.strip() != "")
            )
            df = df.loc[valid_text].copy().reset_index(drop=True)
            if "word_idx" in df.columns:
                df["source_word_idx"] = df["word_idx"]
                df["source_word_idxs"] = df["word_idx"].map(lambda value: str([int(value)]))
            df = df.rename(columns={"text": "word"})
            df["start"] = pd.to_numeric(df["start"], errors="coerce")
            df["end"] = pd.to_numeric(df["end"], errors="coerce")
        if drop_untimed_words:
            df = _drop_untimed_word_rows(df)
        df["word_idx"] = range(len(df))
        return df

    def load_subject_word_table(
        self,
        data_root: str,
        context: DatasetContext,
        subject_id: int,
        word_table: pd.DataFrame | None = None,
        drop_untimed_words: bool = False,
    ) -> pd.DataFrame:
        """Return a word table aligned from movie time to subject neural time."""
        if word_table is None:
            word_table = self.load_word_table(
                data_root,
                context,
                drop_untimed_words=drop_untimed_words,
            )
        elif drop_untimed_words:
            word_table = _drop_untimed_word_rows(word_table, reset_index=False)
        metadata = context.subject_metadata.get(subject_id)
        if metadata is None:
            metadata = self._select_subject_movie_record(
                data_root, subject_id, context.stimulus_id
            )
        return self._align_word_table_to_subject(word_table, metadata)

    @staticmethod
    def _canonical_transcript_text(value: object) -> str:
        return str(value).replace("’", "'").replace("‘", "'").strip()

    @classmethod
    def _treebank_tokenizer(cls):
        from nltk.tokenize import TreebankWordTokenizer

        return TreebankWordTokenizer()

    @classmethod
    def _compact_token(cls, value: object) -> str:
        return re.sub(r"[^\w]+", "", cls._canonical_transcript_text(value))

    @classmethod
    def _tokens_match(cls, source_token: str, sentence_token: str) -> bool:
        source = cls._canonical_transcript_text(source_token)
        sentence = cls._canonical_transcript_text(sentence_token)
        if source == sentence:
            return True
        source_compact = cls._compact_token(source)
        sentence_compact = cls._compact_token(sentence)
        return bool(source_compact) and source_compact == sentence_compact

    @classmethod
    def _is_punctuation_token(cls, value: object) -> bool:
        text = cls._canonical_transcript_text(value)
        return bool(text) and cls._compact_token(text) == ""

    @classmethod
    def _sentence_word_spans(cls, sentence: object) -> list[dict]:
        sentence = str(sentence)
        spans = []
        for match in re.finditer(r"[^ ]+", sentence):
            spans.append(
                {
                    "word": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        return spans

    @classmethod
    def _sentence_treebank_tokens(cls, sentence: object) -> list[dict]:
        sentence = str(sentence)
        word_spans = cls._sentence_word_spans(sentence)
        tokens = []
        for start, end in cls._treebank_tokenizer().span_tokenize(sentence):
            token = sentence[start:end]
            word_idx = None
            for idx, word_span in enumerate(word_spans):
                if start >= word_span["start"] and end <= word_span["end"]:
                    word_idx = idx
                    break
            if word_idx is not None:
                tokens.append(
                    {
                        "token": token,
                        "word_idx": word_idx,
                        "word": word_spans[word_idx]["word"],
                    }
                )
        return tokens

    def _normalize_transcript_words_from_sentences(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        if "sentence_idx" in df.columns:
            group_keys = ["sentence_idx", "sentence"]
        else:
            group_keys = ["sentence"]

        for key, group in df.groupby(group_keys, sort=False, dropna=False):
            sentence = group["sentence"].dropna().iloc[0] if group["sentence"].notna().any() else ""
            word_spans = self._sentence_word_spans(sentence)
            sentence_tokens = self._sentence_treebank_tokens(sentence)
            source_rows = list(group.iterrows())
            token_pos = 0
            partial_token_pos = None
            partial_compact = ""
            rows_by_word: dict[int, list[pd.Series]] = {}

            for _, source in source_rows:
                source_text = self._canonical_transcript_text(source["text"])
                source_tokens = (
                    self._treebank_tokenizer().tokenize(source_text)
                    if source_text
                    else []
                )
                # Some rows have timing but blank text while the sentence has
                # the actual word, e.g. sentence word="None". Use the next
                # non-punctuation Treebank token so the timed row is retained.
                if not source_tokens:
                    while (
                        token_pos < len(sentence_tokens)
                        and self._is_punctuation_token(sentence_tokens[token_pos]["token"])
                    ):
                        token_pos += 1
                    if token_pos < len(sentence_tokens):
                        source_tokens = [sentence_tokens[token_pos]["token"]]

                matched_word_idxs = []
                for source_token in source_tokens:
                    if self._is_punctuation_token(source_token):
                        if matched_word_idxs:
                            matched_word_idxs.append(matched_word_idxs[-1])
                        elif rows_by_word:
                            matched_word_idxs.append(max(rows_by_word))
                        continue

                    source_compact = self._compact_token(source_token)
                    if partial_token_pos is not None:
                        sentence_token = sentence_tokens[partial_token_pos]
                        sentence_compact = self._compact_token(sentence_token["token"])
                        combined = partial_compact + source_compact
                        if not sentence_compact.startswith(combined):
                            sentence_idx = source.get("sentence_idx", key)
                            raise ValueError(
                                "Brain Treebank split source token does not match "
                                f"sentence text at sentence_idx={sentence_idx!r}: "
                                f"source token {source_token!r}, sentence token "
                                f"{sentence_token['token']!r}."
                            )
                        matched_word_idxs.append(sentence_token["word_idx"])
                        if combined == sentence_compact:
                            token_pos = partial_token_pos + 1
                            partial_token_pos = None
                            partial_compact = ""
                        else:
                            partial_compact = combined
                        continue

                    while (
                        token_pos < len(sentence_tokens)
                        and not self._tokens_match(
                            source_token,
                            sentence_tokens[token_pos]["token"],
                        )
                    ):
                        # Sentence punctuation often has no source row
                        # (e.g. source text="Turbo", sentence word="Turbo.").
                        if self._is_punctuation_token(sentence_tokens[token_pos]["token"]):
                            token_pos += 1
                            continue
                        if sentence_tokens[token_pos]["word_idx"] in rows_by_word:
                            # The sentence may contain Treebank suffix tokens
                            # that are omitted from source rows, e.g.
                            # sentence="Isildur's bane." with source rows
                            # "Isildur" and "bane". The final word text still
                            # comes from the sentence word ("Isildur's"), so skip
                            # the unmatched suffix once that word has a source row.
                            token_pos += 1
                            continue
                        sentence_compact = self._compact_token(
                            sentence_tokens[token_pos]["token"]
                        )
                        if (
                            source_compact
                            and sentence_compact.startswith(source_compact)
                        ):
                            matched_word_idxs.append(
                                sentence_tokens[token_pos]["word_idx"]
                            )
                            partial_token_pos = token_pos
                            partial_compact = source_compact
                            break
                        sentence_idx = source.get("sentence_idx", key)
                        raise ValueError(
                            "Brain Treebank transcript token does not match sentence "
                            f"text at sentence_idx={sentence_idx!r}: source token "
                            f"{source_token!r}, sentence token "
                            f"{sentence_tokens[token_pos]['token']!r}."
                        )
                    if partial_token_pos is not None:
                        continue
                    if token_pos >= len(sentence_tokens):
                        sentence_idx = source.get("sentence_idx", key)
                        raise ValueError(
                            "Brain Treebank transcript ended before matching source "
                            f"token at sentence_idx={sentence_idx!r}: {source_token!r}."
                        )
                    matched_word_idxs.append(sentence_tokens[token_pos]["word_idx"])
                    token_pos += 1

                for word_idx in sorted(set(matched_word_idxs)):
                    rows_by_word.setdefault(word_idx, []).append(source)

            for word_idx, piece_rows in sorted(rows_by_word.items()):
                timed_piece_rows = [
                    r
                    for r in piece_rows
                    if pd.notna(pd.to_numeric(r.get("start"), errors="coerce"))
                    and pd.notna(pd.to_numeric(r.get("end"), errors="coerce"))
                ]
                if len(timed_piece_rows) > 1:
                    # If multiple timed rows map to one whitespace sentence word,
                    # the source has more timing detail than the sentence split.
                    # Example: sentence word="Baskin-Robbins." with timed rows
                    # text="Baskin" and text="Robbins". Emit one word row per
                    # timed source row instead of merging both timings.
                    for source in timed_piece_rows:
                        row = source.copy()
                        if "word_idx" in source and pd.notna(source["word_idx"]):
                            row["source_word_idx"] = int(source["word_idx"])
                            row["source_word_idxs"] = str([int(source["word_idx"])])
                        row["word"] = self._canonical_transcript_text(source["text"])
                        row["text"] = row["word"]
                        row["start"] = pd.to_numeric(source.get("start"), errors="coerce")
                        row["end"] = pd.to_numeric(source.get("end"), errors="coerce")
                        rows.append(row.to_dict())
                    continue

                row = piece_rows[0].copy()
                source_word_idxs = [
                    int(r["word_idx"])
                    for r in piece_rows
                    if "word_idx" in r and pd.notna(r["word_idx"])
                ]
                starts = pd.to_numeric(
                    pd.Series([r.get("start") for r in piece_rows]), errors="coerce"
                ).dropna()
                ends = pd.to_numeric(
                    pd.Series([r.get("end") for r in piece_rows]), errors="coerce"
                ).dropna()
                row["word"] = word_spans[word_idx]["word"]
                row["text"] = word_spans[word_idx]["word"]
                row["start"] = float(starts.iloc[0]) if not starts.empty else np.nan
                row["end"] = float(ends.iloc[-1]) if not ends.empty else np.nan
                if source_word_idxs:
                    row["source_word_idx"] = source_word_idxs[0]
                    row["source_word_idxs"] = str(source_word_idxs)
                rows.append(row.to_dict())

        normalized = pd.DataFrame(rows)
        if "text" in normalized.columns:
            normalized = normalized.drop(columns=["text"])
        return normalized.reset_index(drop=True)

    def _align_word_table_to_subject(
        self, word_table: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        timings_path = metadata.get("timings_path")
        if not timings_path or not os.path.exists(timings_path):
            raise FileNotFoundError(
                f"Brain Treebank timing file not found: {timings_path}"
            )

        timings = pd.read_csv(timings_path)
        required_cols = {"movie_time", "index"}
        missing = required_cols - set(timings.columns)
        if missing:
            raise ValueError(
                f"Brain Treebank timing file missing columns {sorted(missing)}: "
                f"{timings_path}"
            )
        timings = timings.copy()
        timings["movie_time"] = pd.to_numeric(timings["movie_time"], errors="coerce")
        timings["index"] = pd.to_numeric(timings["index"], errors="coerce")
        timings = timings.dropna(subset=["movie_time", "index"]).reset_index(drop=True)
        if timings.empty:
            raise ValueError(
                f"Brain Treebank timing file has no valid rows: {timings_path}"
            )

        df = word_table.copy().reset_index(drop=False)
        if "word_idx" not in df.columns:
            idx_col = df.columns[0]
            if idx_col != "index":
                df["word_idx"] = df[idx_col]
            else:
                df["word_idx"] = np.arange(len(df), dtype=int)
        df["movie_start"] = pd.to_numeric(df["start"], errors="coerce")
        df["movie_end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna(subset=["movie_start", "movie_end"]).copy()

        trig_times = timings["movie_time"].to_numpy(dtype=float)
        trig_indices = timings["index"].to_numpy(dtype=float)
        last_trigger_time = trig_times[-1]
        df = df[df["movie_start"] <= last_trigger_time].copy()
        if df.empty:
            return df

        def estimate_sample(movie_time: float) -> int:
            nearest = int(np.abs(trig_times - movie_time).argmin())
            return int(
                round(
                    trig_indices[nearest]
                    + (movie_time - trig_times[nearest]) * self.sfreq
                )
            )

        df["est_idx"] = [
            estimate_sample(t) for t in df["movie_start"].to_numpy(dtype=float)
        ]
        df["est_end_idx"] = [
            estimate_sample(t) for t in df["movie_end"].to_numpy(dtype=float)
        ]
        df["est_idx_sfreq"] = self.sfreq
        df["start"] = df["est_idx"] / self.sfreq
        df["end"] = df["est_end_idx"] / self.sfreq
        return df.reset_index(drop=True)

    def get_embedding_dir(
        self, data_root: str, context: DatasetContext, model_dir: str
    ) -> str:
        return os.path.join(
            data_root,
            "stimuli",
            model_dir,
            f"movie-{context.stimulus_id}",
        )

    def load_generated_transcript(
        self,
        data_root: str,
        context: DatasetContext,
        model_dir: str,
        drop_untimed_words: bool = False,
    ) -> pd.DataFrame:
        transcript_path = os.path.join(
            self.get_embedding_dir(data_root, context, model_dir), "transcript.tsv"
        )
        df = _load_tsv_or_csv(transcript_path, sep="\t")
        df = _ensure_word_idx(df)
        if drop_untimed_words and {"start", "end"} <= set(df.columns):
            df = _drop_untimed_generated_transcript(df)
        return df

    def _load_electrode_labels(self, path: str) -> dict[int, str]:
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            labels = json.load(f)
        if isinstance(labels, list):
            return {idx: str(label) for idx, label in enumerate(labels)}
        if isinstance(labels, dict):
            parsed = {}
            for key, label in labels.items():
                try:
                    parsed[int(key)] = str(label)
                except (TypeError, ValueError):
                    match = re.search(r"(\d+)$", str(key))
                    if match:
                        parsed[int(match.group(1))] = str(label)
            return parsed
        return {}

    def _load_hdf5_raw_array(
        self,
        recording_path: str,
        electrode_labels_path: str,
    ):
        import h5py
        import mne

        if not os.path.exists(recording_path):
            raise FileNotFoundError(
                f"Brain Treebank HDF5 recording not found: {recording_path}"
            )

        labels = self._load_electrode_labels(electrode_labels_path)
        with h5py.File(recording_path, "r") as h5f:
            if "data" not in h5f:
                raise ValueError(
                    f"Brain Treebank recording missing 'data' group: {recording_path}"
                )
            data_group = h5f["data"]
            electrode_keys = []
            for key in data_group.keys():
                match = re.fullmatch(r"electrode_(\d+)", key)
                if match:
                    electrode_keys.append((int(match.group(1)), key))
            if not electrode_keys:
                raise ValueError(
                    f"Brain Treebank recording contains no electrode_<n> datasets: {recording_path}"
                )

            arrays = []
            ch_names = []
            for electrode_num, key in sorted(electrode_keys):
                arr = np.asarray(data_group[key], dtype=np.float32).squeeze()
                if arr.ndim != 1:
                    raise ValueError(
                        f"Expected 1D data for {key} in {recording_path}, got shape {arr.shape}."
                    )
                arrays.append(arr)
                ch_names.append(labels.get(electrode_num, key))

        data = np.stack(arrays, axis=0) * self.hdf5_unit_scale_to_volts
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types="seeg")
        return mne.io.RawArray(data, info, verbose=False)

    def _drop_corrupted_electrodes(self, raw, subject_id: int, data_root: str):
        corrupted_path = os.path.join(data_root, "corrupted_elec.json")
        if not os.path.exists(corrupted_path):
            raise FileNotFoundError(
                "Brain Treebank corrupted-electrode metadata not found: "
                f"{corrupted_path}"
            )
        with open(corrupted_path) as f:
            corrupted_by_subject = json.load(f)

        subject_key = f"sub_{subject_id}"
        if subject_key not in corrupted_by_subject:
            raise KeyError(
                f"Brain Treebank corrupted-electrode metadata has no '{subject_key}' entry: "
                f"{corrupted_path}"
            )

        listed = [str(name) for name in corrupted_by_subject[subject_key]]
        to_drop = [name for name in listed if name in raw.ch_names]
        missing = [name for name in listed if name not in raw.ch_names]
        if missing:
            print(
                f"  - [Config] Warning: {len(missing)} listed corrupted electrodes "
                "were absent from this recording"
            )
        if to_drop:
            print(f"  - [Config] Dropping {len(to_drop)} corrupted electrodes")
            raw.drop_channels(to_drop)
        if len(raw.ch_names) < 2:
            raise ValueError(
                "Brain Treebank common-average reference requires at least two "
                "non-corrupted electrodes."
            )
        return raw

    def _attach_electrode_coordinates(self, raw, subject_id: int, data_root: str):
        """Attach available Brain Treebank MNI coordinates to an MNE Raw."""
        import warnings

        import mne

        localization_path = os.path.join(
            data_root, "localization", "elec_coords_full.csv"
        )
        if not os.path.exists(localization_path):
            print(
                "  - [Config] Warning: Brain Treebank electrode localization "
                f"not found: {localization_path}"
            )
            return raw

        localization = pd.read_csv(localization_path)
        required = {"Subject", "Electrode", "X", "Y", "Z"}
        missing_columns = required - set(localization.columns)
        if missing_columns:
            raise ValueError(
                "Brain Treebank localization file missing columns "
                f"{sorted(missing_columns)}: {localization_path}"
            )

        subject_rows = localization[
            localization["Subject"].astype(str) == f"sub_{subject_id}"
        ].copy()
        coordinate_lookup = {}
        for row in subject_rows.to_dict("records"):
            coords_mm = np.asarray([row["X"], row["Y"], row["Z"]], dtype=float)
            if not np.all(np.isfinite(coords_mm)):
                continue
            normalized_name = re.sub(r"[#*]", "", str(row["Electrode"]))
            if normalized_name in coordinate_lookup:
                raise ValueError(
                    "Duplicate Brain Treebank localization entry for "
                    f"subject {subject_id}, electrode {normalized_name}."
                )
            coordinate_lookup[normalized_name] = coords_mm / 1000.0

        ch_pos = {}
        for ch_name in raw.ch_names:
            normalized_name = re.sub(r"[#*]", "", ch_name)
            if normalized_name in coordinate_lookup:
                ch_pos[ch_name] = coordinate_lookup[normalized_name]

        if not ch_pos:
            print(
                f"  - [Config] Warning: No electrode coordinates matched subject {subject_id}"
            )
            return raw

        montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame="mni_tal",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Fiducial point nasion not found.*",
                category=RuntimeWarning,
            )
            raw.set_montage(montage, on_missing="ignore", verbose=False)
        missing_count = len(raw.ch_names) - len(ch_pos)
        print(
            f"  - [Config] Attached MNI coordinates for {len(ch_pos)} electrodes"
            + (f" ({missing_count} unavailable)" if missing_count else "")
        )
        return raw

    def _despike(self, raw, iqr_mult: float = 4.0, dil_len: float = 0.10):
        """Match the Podcast derivative's per-channel spike interpolation."""
        from scipy import interpolate, signal, stats

        raw.load_data()
        data = raw._data
        medians = np.median(data, axis=-1, keepdims=True)
        iqrs = stats.iqr(data, axis=-1)
        spike_mask = np.abs(data - medians) > (iqrs[:, np.newaxis] * iqr_mult)

        if dil_len > 0:
            dilation_filter = np.ones(int(dil_len * raw.info["sfreq"]))
            spike_mask = signal.filtfilt(dilation_filter, 1, spike_mask) > 0

        for channel_idx, spikes in enumerate(spike_mask):
            if not np.any(spikes):
                continue
            good_indices = np.flatnonzero(~spikes)
            if good_indices.size < 2:
                raise ValueError(
                    f"Cannot interpolate spikes for {raw.ch_names[channel_idx]}: "
                    "fewer than two clean samples remain."
                )
            spike_indices = np.flatnonzero(spikes)
            raw._data[channel_idx, spikes] = interpolate.pchip_interpolate(
                good_indices,
                raw._data[channel_idx, ~spikes],
                spike_indices,
            )
        return raw

    def _apply_high_gamma(self, raw):
        raw = raw.copy()
        if int(raw.info["sfreq"]) != int(self.preprocessing_sfreq):
            raw.resample(sfreq=self.preprocessing_sfreq)
        raw = self._despike(raw)
        raw.set_eeg_reference(ref_channels="average", ch_type="seeg", verbose=False)
        raw.notch_filter(
            freqs=[60, 120, 180],
            notch_widths=2,
            picks="data",
            n_jobs=1,
            verbose=False,
        )
        raw.filter(
            l_freq=70.0,
            h_freq=200.0,
            method="iir",
            iir_params={"order": 4, "ftype": "butter"},
            picks="data",
            verbose=False,
        )
        raw.apply_hilbert(picks="data", envelope=True, verbose=False)
        return raw

    def load_raw_recording(
        self,
        subject_id: int,
        data_root: str,
        context: DatasetContext,
        load_options: dict,
    ):
        import mne

        metadata = context.subject_metadata.get(subject_id, {})
        recording_path = load_options.get("recording_path") or metadata.get(
            "recording_path",
            os.path.join(
                data_root,
                "all_subject_data",
                f"sub_{subject_id}_trial{metadata.get('trial_id', 0):03d}.h5",
            ),
        )
        electrode_labels_path = load_options.get(
            "electrode_labels_path"
        ) or metadata.get(
            "electrode_labels_path",
            os.path.join(
                data_root,
                "electrode_labels",
                f"sub_{subject_id}",
                "electrode_labels.json",
            ),
        )
        raw = self._load_hdf5_raw_array(recording_path, electrode_labels_path)
        raw = self._attach_electrode_coordinates(raw, subject_id, data_root)

        if load_options.get("do_drop_bads", False):
            raw = self._drop_corrupted_electrodes(raw, subject_id, data_root)

        if load_options.get("use_high_gamma", False):
            print("  - [Config] Computing high-gamma envelope (70-200 Hz)")
            raw = self._apply_high_gamma(raw)

        signal_unit = load_options.get("signal_unit")
        if signal_unit == "uV":
            print("  - [Config] Scaling data: Volt -> MicroVolt (x 1e6)")
            raw.apply_function(lambda x: x * 1e6, channel_wise=False)

        per_subject_electrodes = load_options.get("per_subject_electrodes")
        channel_reg_ex = load_options.get("channel_reg_ex")
        if per_subject_electrodes:
            picks = [
                ch for ch in per_subject_electrodes[subject_id] if ch in raw.ch_names
            ]
            raw = raw.pick(picks)
        elif channel_reg_ex:
            picks = mne.pick_channels_regexp(raw.ch_names, channel_reg_ex)
            raw = raw.pick(picks)

        return raw

    def get_precomputed_epochs_path(
        self, subject_id: int, data_root: str, context: DatasetContext
    ) -> str:
        return os.path.join(
            data_root,
            "precomputed_epochs",
            f"sub_{subject_id}",
            f"sub_{subject_id}_movie-{context.stimulus_id}_desc-epochs32hz_ieeg.h5",
        )

    def load_significance(
        self,
        output_dir: str,
        subject_ids: list[int],
        context: DatasetContext,
        sig_dir: str | None = None,
    ) -> tuple[dict[int, list[str]], pd.DataFrame]:
        csv_dir = sig_dir if sig_dir is not None else output_dir
        sig_csv = os.path.join(csv_dir, "all_subject_sig.csv")
        sig_df = pd.read_csv(sig_csv)
        mapping = {
            sub_id: sig_df[sig_df["subject"] == sub_id]["elec"].tolist()
            for sub_id in subject_ids
        }
        return {s: e for s, e in mapping.items() if e}, sig_df

    def get_results_suffix(self, context: DatasetContext) -> str:
        return f"movie-{context.stimulus_id}"

    def validate_subjects_share_stimulus(
        self, subject_ids: list[int], per_dataset_config: dict, data_root: str
    ) -> None:
        movie = per_dataset_config.get("movie")
        if not movie:
            raise ValueError(
                "Brain Treebank runs require per_dataset_config.movie before loading subjects."
            )
        mismatches = []
        for subject_id in subject_ids:
            try:
                self._select_subject_movie_record(data_root, subject_id, movie)
            except ValueError:
                available = sorted(
                    {
                        r["movie"]
                        for r in self._metadata_records(data_root, subject_id)
                        if r["movie"] is not None
                    }
                )
                mismatches.append((subject_id, available))
        if mismatches:
            details = ", ".join(
                f"sub_{subject}: {movies if movies else 'no matching movie'}"
                for subject, movies in mismatches
            )
            raise ValueError(
                f"Brain Treebank runs require all selected subjects to share movie '{movie}'. "
                f"Mismatched subjects: {details}"
            )
