import json

import numpy as np
import pandas as pd
import pytest

from src.core.registry import get_dataset_adapter, resolve_dataset_context

h5py = pytest.importorskip("h5py")


def _write_btb_metadata(root, subject_id, trial_id, movie, title=None):
    path = (
        root
        / "subject_metadata"
        / f"sub_{subject_id}_trial{trial_id:03d}_metadata.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stimulus-type": "movie",
                "title": title or movie.title(),
                "filename": movie,
                "language": "en",
            }
        )
    )


def _write_btb_transcript(root, movie, rows):
    path = root / "transcripts" / movie / "features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_btb_timings(root, subject_id, trial_id, rows):
    path = (
        root
        / "subject_timings"
        / f"sub_{subject_id}_trial{trial_id:03d}_timings.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_btb_h5(root, subject_id, trial_id, data_by_electrode):
    path = root / "all_subject_data" / f"sub_{subject_id}_trial{trial_id:03d}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("data")
        for electrode_num, data in data_by_electrode.items():
            group.create_dataset(f"electrode_{electrode_num}", data=np.asarray(data))
    return path


def _write_btb_electrode_labels(root, subject_id, labels):
    path = root / "electrode_labels" / f"sub_{subject_id}" / "electrode_labels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels))
    return path


def _write_btb_corrupted_electrodes(root, mapping):
    path = root / "corrupted_elec.json"
    path.write_text(json.dumps(mapping))
    return path


def _write_btb_surface_coordinates(root, rows):
    path = root / "localization" / "elec_coords_full.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestDatasetRegistry:
    def test_default_datasets_are_registered(self):
        assert get_dataset_adapter("podcast").name == "podcast"
        assert get_dataset_adapter("brain_treebank").name == "brain_treebank"

    def test_podcast_embedding_dir_is_flat_model_dir(self, tmp_path):
        context = resolve_dataset_context(dataset="podcast", data_root=str(tmp_path))

        path = get_dataset_adapter("podcast").get_embedding_dir(
            str(tmp_path), context, "roberta-base"
        )

        assert path == str(tmp_path / "stimuli" / "roberta-base")

    def test_brain_treebank_embedding_dir_is_movie_scoped(self, tmp_path):
        context = resolve_dataset_context(
            dataset="brain_treebank",
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )

        path = get_dataset_adapter("brain_treebank").get_embedding_dir(
            str(tmp_path), context, "roberta-base"
        )

        assert path == str(tmp_path / "stimuli" / "roberta-base" / "movie-coraline")

    def test_brain_treebank_transcript_is_normalized(self, tmp_path):
        _write_btb_transcript(
            tmp_path,
            "coraline",
            [
                {"text": "Hello", "start": 1.0, "end": 1.2, "word_idx": 10},
                {"text": "skip", "start": None, "end": 1.4, "word_idx": 11},
                {"text": np.nan, "start": 1.5, "end": 1.6, "word_idx": 12},
                {"text": "   ", "start": 1.7, "end": 1.8, "word_idx": 13},
                {"text": "world", "start": 2.0, "end": 2.2, "word_idx": 14},
            ],
        )

        context = resolve_dataset_context(
            dataset="brain_treebank",
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )
        df = get_dataset_adapter("brain_treebank").load_word_table(
            str(tmp_path),
            context,
            drop_untimed_words=True,
        )

        assert df["word"].tolist() == ["Hello", "world"]
        assert df["word_idx"].tolist() == [0, 1]
        assert df["source_word_idx"].tolist() == [10, 14]
        assert {"start", "end"} <= set(df.columns)

    def test_brain_treebank_sentence_guided_transcript_merges_suffixes(
        self, tmp_path
    ):
        _write_btb_transcript(
            tmp_path,
            "coraline",
            [
                {
                    "text": "cover",
                    "start": 1.0,
                    "end": 1.2,
                    "sentence": "cover's here.",
                    "sentence_idx": 0,
                    "word_idx": 10,
                },
                {
                    "text": "'s",
                    "start": None,
                    "end": None,
                    "sentence": "cover's here.",
                    "sentence_idx": 0,
                    "word_idx": 11,
                },
                {
                    "text": "here",
                    "start": 1.3,
                    "end": 1.5,
                    "sentence": "cover's here.",
                    "sentence_idx": 0,
                    "word_idx": 12,
                },
                {
                    "text": "wo",
                    "start": 2.0,
                    "end": 2.2,
                    "sentence": "won't stop",
                    "sentence_idx": 1,
                    "word_idx": 13,
                },
                {
                    "text": "n't",
                    "start": None,
                    "end": None,
                    "sentence": "won't stop",
                    "sentence_idx": 1,
                    "word_idx": 14,
                },
                {
                    "text": "stop",
                    "start": None,
                    "end": None,
                    "sentence": "won't stop",
                    "sentence_idx": 1,
                    "word_idx": 15,
                },
            ],
        )

        context = resolve_dataset_context(
            dataset="brain_treebank",
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )
        adapter = get_dataset_adapter("brain_treebank")

        df = adapter.load_word_table(str(tmp_path), context)

        assert df["word"].tolist() == ["cover's", "here.", "won't", "stop"]
        assert df["source_word_idxs"].tolist() == ["[10, 11]", "[12]", "[13, 14]", "[15]"]
        assert df["start"].tolist()[:3] == [1.0, 1.3, 2.0]
        assert pd.isna(df.loc[3, "start"])
        assert df["word_idx"].tolist() == [0, 1, 2, 3]

        timed = adapter.load_word_table(
            str(tmp_path),
            context,
            drop_untimed_words=True,
        )

        assert timed["word"].tolist() == ["cover's", "here.", "won't"]
        assert timed["word_idx"].tolist() == [0, 1, 2]

    def test_brain_treebank_splits_sentence_word_when_multiple_timed_rows_map(
        self, tmp_path
    ):
        _write_btb_transcript(
            tmp_path,
            "coraline",
            [
                {
                    "text": "Baskin",
                    "start": 1.0,
                    "end": 1.4,
                    "sentence": "Baskin-Robbins.",
                    "sentence_idx": 0,
                    "word_idx": 10,
                },
                {
                    "text": "Robbins",
                    "start": 1.5,
                    "end": 1.9,
                    "sentence": "Baskin-Robbins.",
                    "sentence_idx": 0,
                    "word_idx": 11,
                },
            ],
        )

        context = resolve_dataset_context(
            dataset="brain_treebank",
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )

        df = get_dataset_adapter("brain_treebank").load_word_table(
            str(tmp_path), context
        )

        assert df["word"].tolist() == ["Baskin", "Robbins"]
        assert df["source_word_idx"].tolist() == [10, 11]
        assert df["start"].tolist() == [1.0, 1.5]
        assert df["end"].tolist() == [1.4, 1.9]

    def test_brain_treebank_keeps_sentence_word_when_source_omits_suffix(
        self, tmp_path
    ):
        _write_btb_transcript(
            tmp_path,
            "coraline",
            [
                {
                    "text": "Isildur",
                    "start": 1.0,
                    "end": 1.4,
                    "sentence": "Isildur's bane.",
                    "sentence_idx": 0,
                    "word_idx": 10,
                },
                {
                    "text": "bane",
                    "start": 1.5,
                    "end": 1.9,
                    "sentence": "Isildur's bane.",
                    "sentence_idx": 0,
                    "word_idx": 11,
                },
            ],
        )

        context = resolve_dataset_context(
            dataset="brain_treebank",
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )

        df = get_dataset_adapter("brain_treebank").load_word_table(
            str(tmp_path), context
        )

        assert df["word"].tolist() == ["Isildur's", "bane."]
        assert df["source_word_idx"].tolist() == [10, 11]
        assert df["start"].tolist() == [1.0, 1.5]
        assert df["end"].tolist() == [1.4, 1.9]

    def test_brain_treebank_subject_word_table_aligns_movie_time_to_neural_samples(
        self, tmp_path
    ):
        _write_btb_metadata(tmp_path, 3, 0, "coraline")
        _write_btb_transcript(
            tmp_path,
            "coraline",
            [
                {"text": "a", "start": 1.0, "end": 1.25},
                {"text": "b", "start": 2.0, "end": 2.25},
                {"text": "late", "start": 7.0, "end": 7.25},
            ],
        )
        _write_btb_timings(
            tmp_path,
            3,
            0,
            [
                {
                    "type": "movie_start",
                    "movie_time": 0.0,
                    "start_time": 10.0,
                    "index": 1000,
                },
                {
                    "type": "movie_tick",
                    "movie_time": 4.0,
                    "start_time": 14.0,
                    "index": 10000,
                },
            ],
        )
        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=[3],
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )
        adapter = get_dataset_adapter("brain_treebank")
        word_table = adapter.load_word_table(str(tmp_path), context)

        aligned = adapter.load_subject_word_table(str(tmp_path), context, 3, word_table)

        assert aligned["word"].tolist() == ["a", "b"]
        assert aligned["movie_start"].tolist() == [1.0, 2.0]
        assert aligned["movie_end"].tolist() == [1.25, 2.25]
        assert aligned["est_idx"].tolist() == [3048, 5096]
        assert aligned["est_idx_sfreq"].tolist() == [2048.0, 2048.0]
        np.testing.assert_allclose(
            aligned["start"], np.array([3048, 5096]) / 2048.0
        )

    def test_brain_treebank_rejects_mixed_movie_subjects(self, tmp_path):
        _write_btb_metadata(tmp_path, 1, 0, "coraline")
        _write_btb_metadata(tmp_path, 2, 0, "venom")

        adapter = get_dataset_adapter("brain_treebank")
        with pytest.raises(ValueError, match="share movie 'coraline'"):
            adapter.validate_subjects_share_stimulus(
                [1, 2],
                {"movie": "coraline"},
                str(tmp_path),
            )

    def test_brain_treebank_context_contains_subject_trial_metadata(self, tmp_path):
        _write_btb_metadata(tmp_path, 1, 2, "thor-ragnarok", title="Thor Ragnarok")

        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=[1],
            per_dataset_config={"movie": "thor-ragnarok"},
            data_root=str(tmp_path),
        )

        assert context.stimulus_id == "thor-ragnarok"
        assert context.subject_metadata[1]["trial_id"] == 2
        assert context.subject_metadata[1]["movie"] == "thor-ragnarok"
        assert context.subject_metadata[1]["recording_path"].endswith(
            "all_subject_data/sub_1_trial002.h5"
        )

    def test_brain_treebank_loads_hdf5_as_seeg_raw_array(self, tmp_path):
        mne = pytest.importorskip("mne")
        _ = mne
        _write_btb_metadata(tmp_path, 3, 0, "coraline")
        _write_btb_h5(
            tmp_path,
            3,
            0,
            {
                2: np.full(8, 2.0, dtype=np.float32),
                0: np.zeros(8, dtype=np.float32),
            },
        )
        _write_btb_electrode_labels(tmp_path, 3, ["A0", "A1", "A2"])
        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=[3],
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )

        raw = get_dataset_adapter("brain_treebank").load_raw_recording(
            3,
            str(tmp_path),
            context,
            {"use_high_gamma": False},
        )

        assert raw.info["sfreq"] == 2048.0
        assert raw.ch_names == ["A0", "A2"]
        assert raw.get_channel_types() == ["seeg", "seeg"]
        np.testing.assert_allclose(raw.get_data(), [[0.0] * 8, [2e-6] * 8])

    def test_brain_treebank_attaches_mni_electrode_coordinates(self, tmp_path):
        _write_btb_metadata(tmp_path, 3, 0, "coraline")
        _write_btb_h5(
            tmp_path,
            3,
            0,
            {0: np.zeros(8, dtype=np.float32), 1: np.ones(8, dtype=np.float32)},
        )
        _write_btb_electrode_labels(tmp_path, 3, ["A*1", "A2"])
        _write_btb_surface_coordinates(
            tmp_path,
            [
                {
                    "Subject": "sub_3",
                    "Electrode": "A1",
                    "X": 12.0,
                    "Y": -34.0,
                    "Z": 56.0,
                }
            ],
        )
        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=[3],
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )

        raw = get_dataset_adapter("brain_treebank").load_raw_recording(
            3, str(tmp_path), context, {"use_high_gamma": False}
        )

        np.testing.assert_allclose(raw.info["chs"][0]["loc"][:3], [0.012, -0.034, 0.056])
        assert np.isnan(raw.info["chs"][1]["loc"][:3]).all()
        assert raw.get_montage().get_positions()["coord_frame"] == "mni_tal"

    def test_brain_treebank_signal_unit_uv_returns_hdf5_scale(self, tmp_path):
        _write_btb_metadata(tmp_path, 3, 0, "coraline")
        _write_btb_h5(tmp_path, 3, 0, {0: np.full(8, 2.0, dtype=np.float32)})
        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=[3],
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )

        raw = get_dataset_adapter("brain_treebank").load_raw_recording(
            3,
            str(tmp_path),
            context,
            {"signal_unit": "uV"},
        )

        np.testing.assert_allclose(raw.get_data(), [[2.0] * 8])

    def test_brain_treebank_high_gamma_option_uses_adapter_path(
        self, tmp_path, monkeypatch
    ):
        _write_btb_metadata(tmp_path, 3, 0, "coraline")
        _write_btb_h5(tmp_path, 3, 0, {0: np.arange(16, dtype=np.float32)})
        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=[3],
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )
        adapter = get_dataset_adapter("brain_treebank")
        called = {"value": False}

        def _fake_high_gamma(raw):
            called["value"] = True
            return raw

        monkeypatch.setattr(adapter, "_apply_high_gamma", _fake_high_gamma)

        raw = adapter.load_raw_recording(
            3,
            str(tmp_path),
            context,
            {"use_high_gamma": True},
        )

        assert called["value"] is True
        assert raw.get_channel_types() == ["seeg"]

    def test_brain_treebank_high_gamma_preprocessing_order(self, monkeypatch):
        adapter = get_dataset_adapter("brain_treebank")

        class FakeRaw:
            def __init__(self):
                self.info = {"sfreq": 2048.0}
                self.calls = []

            def copy(self):
                return self

            def resample(self, sfreq, **kwargs):
                self.calls.append(("resample", sfreq))
                self.info["sfreq"] = sfreq
                return self

            def set_eeg_reference(self, ref_channels, ch_type, verbose=False):
                self.calls.append(("reference", ref_channels, ch_type))
                return self

            def notch_filter(self, freqs, **kwargs):
                self.calls.append(("notch", list(freqs)))
                return self

            def filter(self, l_freq, h_freq, **kwargs):
                self.calls.append(("filter", l_freq, h_freq))
                return self

            def apply_hilbert(self, **kwargs):
                self.calls.append(("hilbert",))
                return self

        raw = FakeRaw()
        monkeypatch.setattr(
            adapter,
            "_despike",
            lambda inst: inst.calls.append(("despike",)) or inst,
        )
        processed = adapter._apply_high_gamma(raw)

        assert processed is raw
        assert raw.calls == [
            ("resample", 512.0),
            ("despike",),
            ("reference", "average", "seeg"),
            ("notch", [60, 120, 180]),
            ("filter", 70.0, 200.0),
            ("hilbert",),
        ]

    def test_brain_treebank_despiking_interpolates_large_transient(self):
        mne = pytest.importorskip("mne")
        rng = np.random.default_rng(4)
        data = rng.normal(scale=1e-6, size=(2, 2048))
        data[0, 1024] = 1e-3
        raw = mne.io.RawArray(
            data,
            mne.create_info(["A1", "A2"], sfreq=512.0, ch_types=["seeg"] * 2),
            verbose=False,
        )

        processed = get_dataset_adapter("brain_treebank")._despike(raw)

        assert abs(processed.get_data()[0, 1024]) < 1e-5

    def test_brain_treebank_does_not_apply_analysis_rate_resampling(self, tmp_path):
        _write_btb_metadata(tmp_path, 3, 0, "coraline")
        _write_btb_h5(
            tmp_path,
            3,
            0,
            {0: np.arange(2048, dtype=np.float32)},
        )
        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=[3],
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )

        raw = get_dataset_adapter("brain_treebank").load_raw_recording(
            3,
            str(tmp_path),
            context,
            {"target_sr": 32.0},
        )

        assert raw.info["sfreq"] == 2048.0
        assert raw.n_times == 2048

    def test_brain_treebank_drops_corrupted_contacts_before_preprocessing(
        self, tmp_path
    ):
        _write_btb_metadata(tmp_path, 3, 0, "coraline")
        _write_btb_h5(
            tmp_path,
            3,
            0,
            {
                0: np.zeros(16, dtype=np.float32),
                1: np.ones(16, dtype=np.float32),
                2: np.full(16, 2.0, dtype=np.float32),
            },
        )
        _write_btb_electrode_labels(tmp_path, 3, ["A0", "BAD", "A2"])
        _write_btb_corrupted_electrodes(tmp_path, {"sub_3": ["BAD"]})
        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=[3],
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )

        raw = get_dataset_adapter("brain_treebank").load_raw_recording(
            3,
            str(tmp_path),
            context,
            {"use_high_gamma": False, "do_drop_bads": True},
        )

        assert raw.ch_names == ["A0", "A2"]

    def test_brain_treebank_drop_bads_requires_corruption_metadata(self, tmp_path):
        _write_btb_metadata(tmp_path, 3, 0, "coraline")
        _write_btb_h5(
            tmp_path,
            3,
            0,
            {0: np.zeros(16, dtype=np.float32), 1: np.ones(16, dtype=np.float32)},
        )
        context = resolve_dataset_context(
            dataset="brain_treebank",
            subject_ids=[3],
            per_dataset_config={"movie": "coraline"},
            data_root=str(tmp_path),
        )

        with pytest.raises(FileNotFoundError, match="corrupted-electrode metadata"):
            get_dataset_adapter("brain_treebank").load_raw_recording(
                3,
                str(tmp_path),
                context,
                {"use_high_gamma": False, "do_drop_bads": True},
            )
