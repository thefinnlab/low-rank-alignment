import json

import h5py
import numpy as np

from scripts.generate_brain_treebank_embeddings import embedding_is_complete
from scripts.utilities.precompute_epochs import (
    _cache_is_complete,
    _trial_record,
    cross_stimulus_trials,
)
from src.core.config import GenerateEmbeddingsConfig


def test_trial_record_resolves_exact_trial(tmp_path):
    metadata_dir = tmp_path / "subject_metadata"
    metadata_dir.mkdir()
    (metadata_dir / "sub_2_trial004_metadata.json").write_text(
        json.dumps({"filename": "black-panther"})
    )

    record = _trial_record(str(tmp_path), 2, 4)

    assert record["movie"] == "black-panther"
    assert record["recording_path"].endswith("sub_2_trial004.h5")


def test_cross_stimulus_trial_plan_includes_all_trials_for_eligible_subjects(tmp_path):
    metadata_dir = tmp_path / "subject_metadata"
    metadata_dir.mkdir()
    records = (
        (1, 0, "cars-2"),
        (1, 1, "megamind"),
        (2, 0, "cars-2"),
    )
    for subject, trial, movie in records:
        (metadata_dir / f"sub_{subject}_trial{trial:03d}_metadata.json").write_text(
            json.dumps({"filename": movie})
        )

    assert cross_stimulus_trials(str(tmp_path)) == [
        (1, 0, "cars-2"),
        (1, 1, "megamind"),
    ]


def test_epoch_cache_completeness_requires_matching_context(tmp_path):
    path = tmp_path / "epochs.h5"
    with h5py.File(path, "w") as cache:
        cache.create_dataset("epoch_data", data=np.zeros((3, 2, 128)))
        cache.create_dataset("selection", data=np.arange(3))
        cache.create_dataset("ch_names", data=np.asarray(["A1", "A2"], dtype="S2"))
        cache.attrs.update(
            n_lags=128,
            sfreq=32.0,
            tmin=-2.0,
            tmax=2.0,
            dataset="brain_treebank",
            stimulus_id="cars-2",
            rereference="laplacian",
        )

    assert _cache_is_complete(
        str(path),
        dataset="brain_treebank",
        stimulus="cars-2",
        rereference="laplacian",
    )
    assert not _cache_is_complete(
        str(path),
        dataset="brain_treebank",
        stimulus="megamind",
        rereference="laplacian",
    )


def test_embedding_completeness_checks_requested_layers_and_movie(tmp_path):
    config = GenerateEmbeddingsConfig(
        data_root=str(tmp_path),
        dataset="brain_treebank",
        hf_model_id="gpt2",
        output_name="gpt2",
        layers=[0, 12],
    )
    output_dir = tmp_path / "stimuli" / "gpt2" / "movie-cars-2"
    output_dir.mkdir(parents=True)
    with h5py.File(output_dir / "features.hdf5", "w") as features:
        features.create_dataset("layer-0", data=np.zeros((2, 3)))
        features.create_dataset("layer-12", data=np.zeros((2, 3)))
    (output_dir / "transcript.tsv").write_text("word_idx\tword\n0\tHello\n")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "brain_treebank",
                "stimulus_id": "cars-2",
                "layers_saved": [0, 12],
            }
        )
    )

    assert embedding_is_complete(config, "cars-2")
    assert not embedding_is_complete(config, "megamind")
