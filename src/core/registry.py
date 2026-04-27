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
        **_: dict,
    ) -> pd.DataFrame:
        transcript_path = os.path.join(data_root, "stimuli", model_dir, "transcript.tsv")
        df = _load_tsv_or_csv(transcript_path, sep="\t")
        return _ensure_word_idx(df)

    def get_embedding_dir(
        self, data_root: str, context: DatasetContext, model_dir: str
    ) -> str:
        return os.path.join(data_root, "stimuli", model_dir)

    def load_generated_transcript(
        self, data_root: str, context: DatasetContext, model_dir: str
    ) -> pd.DataFrame:
        transcript_path = os.path.join(
            self.get_embedding_dir(data_root, context, model_dir), "transcript.tsv"
        )
        df = _load_tsv_or_csv(transcript_path, sep="\t")
        return _ensure_word_idx(df)

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
        target_sr = load_options.get("target_sr")
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

        if target_sr is not None and int(raw.info["sfreq"]) != int(target_sr):
            print(f"  - [Config] Resampling {raw.info['sfreq']}Hz -> {target_sr}Hz")
            raw.resample(target_sr, npad="auto", method="fft", window="hamming")

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
            f"sub-{subject_id:02d}_task-podcast_desc-epochs32hz_ieeg.npz",
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
        **_: dict,
    ) -> pd.DataFrame:
        path = os.path.join(
            data_root,
            "transcripts",
            context.stimulus_id,
            "features.csv",
        )
        df = pd.read_csv(path)
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)
        if "text" not in df.columns:
            raise ValueError(f"Brain Treebank transcript missing 'text' column: {path}")
        valid_text = (
            df["text"].notna()
            & df["text"].map(lambda value: isinstance(value, str))
            & df["text"].map(lambda value: isinstance(value, str) and value.strip() != "")
        )
        keep = (
            valid_text
            & df["start"].notna()
            & df["end"].notna()
            & pd.to_numeric(df["start"], errors="coerce").notna()
            & pd.to_numeric(df["end"], errors="coerce").notna()
        )
        df = df.loc[keep].copy().reset_index(drop=True)
        if "word_idx" in df.columns:
            df["source_word_idx"] = df["word_idx"]
        df = df.rename(columns={"text": "word"})
        df["word_idx"] = range(len(df))
        return df

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
        self, data_root: str, context: DatasetContext, model_dir: str
    ) -> pd.DataFrame:
        transcript_path = os.path.join(
            self.get_embedding_dir(data_root, context, model_dir), "transcript.tsv"
        )
        df = _load_tsv_or_csv(transcript_path, sep="\t")
        return _ensure_word_idx(df)

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

    def _apply_high_gamma(self, raw):
        raw = raw.copy()
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

        target_sr = load_options.get("target_sr")
        if target_sr is not None and int(raw.info["sfreq"]) != int(target_sr):
            print(f"  - [Config] Resampling {raw.info['sfreq']}Hz -> {target_sr}Hz")
            raw.resample(target_sr, npad="auto", method="fft", window="hamming")

        return raw

    def get_precomputed_epochs_path(
        self, subject_id: int, data_root: str, context: DatasetContext
    ) -> str:
        return os.path.join(
            data_root,
            "precomputed_epochs",
            f"sub_{subject_id}",
            f"sub_{subject_id}_movie-{context.stimulus_id}_desc-epochs32hz_ieeg.npz",
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
