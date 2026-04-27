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
            str(tmp_path), context
        )

        assert df["word"].tolist() == ["Hello", "world"]
        assert df["word_idx"].tolist() == [0, 1]
        assert df["source_word_idx"].tolist() == [10, 14]
        assert {"start", "end"} <= set(df.columns)

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

    def test_brain_treebank_explicit_target_resampling(self, tmp_path):
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

        assert raw.info["sfreq"] == 32.0
        assert raw.n_times == 32
